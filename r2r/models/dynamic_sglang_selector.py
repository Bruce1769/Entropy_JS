import uuid
import time
import torch
from tqdm import tqdm
from typing import Optional, Tuple, Union, List
import multiprocessing as mp
import torch.distributed as dist

from r2r.models.recorder import GenerationRecord, GenerationRecorder
from r2r.utils.config import (
    QUICK_COLOR,
    REFERENCE_COLOR,
    RESET,
)
from r2r.utils.switching import (
    append_entropy_lookahead_score_log,
    create_switching_strategy,
    EntropyLookaheadSwitching,
)
from r2r.utils.token_manager import SGLangTokenManager
from r2r.utils.dataclass import ModelOutputs
from r2r.utils.sampling import sample_token
from r2r.utils.metrics import compute_entropy, log_prob_of_token
from r2r.models.sglang_patch.schedule_req import EntropyLookaheadRpc, EntropyLookaheadResp

from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.managers.schedule_batch import (
    Req,
    ScheduleBatch,
    ForwardMode,
    SamplingBatchInfo,
    get_last_loc,
    write_req_to_token_pool_triton,
)
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.io_struct import AbortReq
from sglang.srt.server_args import PortArgs, ServerArgs

_ds_el_rpc_fail_last_log = 0.0
_ds_el_rpc_fail_pending = 0


def _ds_log_entropy_lookahead_rpc_fail_throttled(err: object, seq_i: int) -> None:
    global _ds_el_rpc_fail_last_log, _ds_el_rpc_fail_pending
    now = time.monotonic()
    _ds_el_rpc_fail_pending += 1
    elapsed = now - _ds_el_rpc_fail_last_log
    if _ds_el_rpc_fail_last_log != 0.0 and elapsed < 15.0:
        return
    es = str(err).replace("\n", " ")[:180]
    print(
        f"[DynamicSimpleSGLangSelector] entropy lookahead LLM logprob RPC failed "
        f"x{_ds_el_rpc_fail_pending} in ~{max(elapsed, 0.001):.1f}s, seq={seq_i}: {es}; falling back"
    )
    _ds_el_rpc_fail_pending = 0
    _ds_el_rpc_fail_last_log = now


class DynamicSimpleSGLangSelector:
    """Dynamic model selection using SGLang models"""

    def __init__(
        self,
        model_config: dict,
        device: Optional[str] = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        switching_strategy: str = "neural",
        strategy_kwargs: Optional[dict] = None,
        is_record: bool = False,
        sglang_kwargs: Optional[dict] = None,
        is_logits_processor: bool = True,
    ):
        self.device = device
        self.dtype = dtype
        self.model_config = model_config
        self.strategy_kwargs = strategy_kwargs or {}
        self.switching_strategy_name = switching_strategy
        self.is_record = is_record

        # Combine default with provided kwargs
        quick_sglang_kwargs = {**(sglang_kwargs or {})}
        reference_sglang_kwargs = {**(sglang_kwargs or {})}

        # Reference TP degree (not "total GPU count"). Quick model is always tp_size=1.
        ref_tp = int(reference_sglang_kwargs.get("tp_size", 1))
        if ref_tp < 1:
            raise ValueError(f"tp_size for reference must be >= 1, got {ref_tp}")
        self.world_size = ref_tp
        quick_sglang_kwargs["tp_size"] = 1
        reference_sglang_kwargs["tp_size"] = ref_tp

        # Layout:
        # - ref_tp==1: quick on cuda:0, reference on cuda:1 (fits Qwen3-8B bf16 on 24GB + small SLM).
        # - ref_tp>=2: reference ranks on cuda:0..ref_tp-1, quick colocated on cuda:ref_tp-1 (legacy).
        if ref_tp == 1:
            self.ref_base_gpu = 1
            quick_gpu_id = 0
        else:
            self.ref_base_gpu = 0
            quick_gpu_id = ref_tp - 1

        n_visible = torch.cuda.device_count()
        max_idx = max(quick_gpu_id, self.ref_base_gpu + ref_tp - 1)
        assert n_visible > max_idx, (
            f"Need at least {max_idx + 1} visible CUDA device(s) (have {n_visible}). "
            f"reference tp={ref_tp}: quick on cuda:{quick_gpu_id}, reference ranks start at cuda:{self.ref_base_gpu}. "
            f"Try CUDA_VISIBLE_DEVICES with enough free GPUs, or lower --tp-size for the reference model."
        )

        if switching_strategy == "entropy_lookahead" and self.world_size != 1:
            raise ValueError(
                "entropy_lookahead with DynamicSimpleSGLangSelector requires reference tp_size=1 "
                f"(got {self.world_size})"
            )
        self._entropy_lookahead_reply_queue: Optional[mp.Queue] = (
            mp.Queue() if switching_strategy == "entropy_lookahead" else None
        )

        # Create dictionary to store recorders
        self.generation_records = {}

        print(
            f"Using {n_visible} visible GPU(s): quick on cuda:{quick_gpu_id}, "
            f"reference tp={ref_tp} on cuda:{self.ref_base_gpu}..cuda:{self.ref_base_gpu + ref_tp - 1}"
        )

        # Initialize SGLang models
        print(f"Loading quick model {self.model_config['quick']['model_path']}...")

        self.quick_server_args = ServerArgs(
            model_path=self.model_config["quick"]["model_path"],
            disable_cuda_graph=False,
            disable_overlap_schedule=True,
            disable_radix_cache=False,
            mem_fraction_static=self.model_config["quick"].get("mem_fraction_static", 0.9),
            **quick_sglang_kwargs,
        )
        quick_port_args = PortArgs.init_new(self.quick_server_args)
        self.quick_scheduler = Scheduler(
            server_args=self.quick_server_args,
            port_args=quick_port_args,
            gpu_id=quick_gpu_id,
            tp_rank=0,
            dp_rank=0,
            moe_ep_rank=0,
            pp_rank=0,
        )
        # Load tokenizer
        self.tokenizer = self.quick_scheduler.tokenizer
        # # warm up the quick model
        self.warm_up_quick_model()

        print(f"Loading reference model {self.model_config['reference']['model_path']}...")
        self.reference_server_args = ServerArgs(
            model_path=self.model_config["reference"]["model_path"],
            disable_cuda_graph=True,
            disable_overlap_schedule=True,
            disable_radix_cache=False,
            mem_fraction_static=self.model_config["reference"].get("mem_fraction_static", 0.9),
            **reference_sglang_kwargs,
        )

        self.reference_model_input_queues = [mp.Queue() for _ in range(self.world_size)]
        self.reference_model_ack_queues = [mp.Queue() for _ in range(self.world_size)]
        self.reference_model_output_queue = mp.Queue()

        self.reference_model_procs = []
        for rank in range(self.world_size):
            proc = mp.Process(
                target=self.reference_model_worker,
                args=(
                    rank,
                    self.world_size,
                    self.ref_base_gpu,
                    self.reference_server_args,
                    self.reference_model_input_queues,
                    self.reference_model_output_queue,
                    self.reference_model_ack_queues[rank],
                    self._entropy_lookahead_reply_queue,
                ),
            )
            proc.start()
            self.reference_model_procs.append(proc)

        # Initialize prefix indices list for reference model
        self.reference_prefix_indices_list = []

        # warm up the reference model
        self.warm_up_reference_model()

        # Initialize switching strategy
        # Get override_init_args from router config
        router_config = self.model_config.get('router', {})
        override_init_args = router_config.get('override_init_args', {})
        self.strategy_kwargs["override_init_args"] = override_init_args
        self.switching_strategy = create_switching_strategy(
            switching_strategy, **self.strategy_kwargs
        )
    
    def warm_up_reference_model(self):
        # dummy call to warm up the reference model
        if not self.reference_prefix_indices_list:
            self.reference_prefix_indices_list.append([])
        
        test_input = [self.tokenizer.encode("Hi")]
        
        self.extend_step(
            input_ids=test_input,
            input_indices=[0],  
            sampling_params=SamplingParams(temperature=0.0, top_p=1.0, top_k=-1, max_new_tokens=1, stop=[])
        )

    def warm_up_quick_model(self):
        # dummy call to warm up the quick model
        warmup_iter = 5
        req = Req(
            rid=str(uuid.uuid4()),
            origin_input_text="Hi",
            origin_input_ids=self.quick_scheduler.tokenizer.encode("Hi"),
            sampling_params=SamplingParams(
                temperature=0.0,
                top_p=1.0,
                top_k=-1,
                max_new_tokens=warmup_iter,
                stop=[]
            ),
            eos_token_ids=self.quick_scheduler.model_config.hf_eos_token_id,
            return_hidden_states=True,
            vocab_size=self.quick_scheduler.model_config.vocab_size
        )
        req.sampling_params.normalize(None)
        self.quick_scheduler.waiting_queue.append(req)
        for _ in range(warmup_iter):
            batch = self.quick_scheduler.get_next_batch_to_run()
            if batch is None:
                break
            result = self.quick_scheduler.run_batch(batch)
            next_token_ids = result.next_token_ids
            self.quick_scheduler.last_batch = batch
        for req in batch.reqs:
            self.quick_scheduler.abort_request(AbortReq(req.rid))
            req.check_finished()
            if req.finished():
                self.quick_scheduler.tree_cache.cache_finished_req(req)
        self.quick_scheduler.last_batch = batch

    @staticmethod
    def reference_model_worker(
        rank,
        world_size: int,
        ref_base_gpu: int,
        server_args: ServerArgs,
        input_queues: List[mp.Queue],
        output_queue: mp.Queue,
        ack_queue: mp.Queue,
        el_reply_queue: Optional[mp.Queue] = None,
    ):
        # Spawned workers must see the same TMPDIR as the parent so FlashInfer JIT / nvcc
        # do not fall back to a full or broken host /tmp (see cuda_build_env).
        from r2r.utils.cuda_build_env import ensure_cuda_jit_environment

        ensure_cuda_jit_environment()

        # initialize the process group
        dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
        gpu_id = ref_base_gpu + rank
        torch.cuda.set_device(gpu_id)

        global end_of_cache_loc
        end_of_cache_loc = 0

        input_queue = input_queues[rank]
        port_args = PortArgs.init_new(server_args)
        scheduler = Scheduler(
            server_args=server_args,
            port_args=port_args,
            gpu_id=gpu_id,
            tp_rank=rank,
            dp_rank=0,
            moe_ep_rank=0,
            pp_rank=0,
        )

        while True:
            reqs: Union[List[Req], int] = input_queue.get()
            if isinstance(reqs, int):
                # terminate the process
                break
            elif isinstance(reqs, str):
                if reqs == "RESET_CACHE":
                    # reset the cache
                    end_of_cache_loc = 0
                    ack_queue.put(end_of_cache_loc)
                    continue
            elif isinstance(reqs, tuple) and reqs[0] == "ENTROPY_LA":
                from r2r.models.sglang_patch.llm_server import LLMServer

                rpc = reqs[1]
                resp = LLMServer._entropy_lookahead_llm_logprobs(scheduler, rpc)
                if rank == 0 and el_reply_queue is not None:
                    el_reply_queue.put(resp)
                continue
            else:
                new_batch = ScheduleBatch.init_new(
                    reqs,
                    scheduler.req_to_token_pool,
                    scheduler.token_to_kv_pool_allocator,
                    scheduler.tree_cache,
                    scheduler.model_config,
                    scheduler.enable_overlap,
                    scheduler.spec_algorithm,
                    scheduler.server_args.enable_custom_logit_processor,
                )
                DynamicSimpleSGLangSelector.simple_prepare_for_extend(new_batch)
                batch = new_batch.get_model_worker_batch()
                kv_locs = new_batch.out_cache_loc
                try:
                    result = scheduler.tp_worker.forward_batch_generation(batch)
                    # sglang 0.5.1 returns (logits_output, next_token_ids, ...) - extract next_token_ids
                    next_token_ids = result[1] if isinstance(result, tuple) else result
                    next_token_ids_list = next_token_ids.tolist()

                    if rank == 0:
                        output_queue.put(next_token_ids_list)
                finally:
                    try:
                        if kv_locs is not None and kv_locs.numel() > 0:
                            scheduler.token_to_kv_pool_allocator.free(kv_locs)
                    except Exception:
                        pass
                    for r in new_batch.reqs:
                        idx = getattr(r, "req_pool_idx", None)
                        if idx is not None:
                            try:
                                scheduler.req_to_token_pool.free(idx)
                            except Exception:
                                pass

    def init_model_switching_strategy(self):
        """Initialize or reinitialize the model switching strategy with stored parameters"""
        self.switching_strategy = create_switching_strategy(
            self.switching_strategy_name, **self.strategy_kwargs
        )

    def _llm_entropy_lookahead_rpc(self, rpc: EntropyLookaheadRpc) -> EntropyLookaheadResp:
        r = self.switching_strategy
        assert isinstance(r, EntropyLookaheadSwitching)
        assert self._entropy_lookahead_reply_queue is not None
        msg = ("ENTROPY_LA", rpc)
        for q in self.reference_model_input_queues:
            q.put_nowait(msg)
        return self._entropy_lookahead_reply_queue.get(timeout=r.rpc_timeout_s)

    def _ds_entropy_lookahead_choices(
        self,
        router: EntropyLookaheadSwitching,
        batch: ScheduleBatch,
        logits: torch.Tensor,
        next_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        from r2r.models.sglang_patch.slm_server import SLMServer

        device = logits.device
        batch_size = logits.shape[0]
        choices = torch.zeros(batch_size, dtype=torch.int, device=device)
        n = router.lookahead_steps
        for i in range(batch_size):
            req = batch.reqs[i]
            L0 = logits[i, 0, :]
            H = compute_entropy(L0.unsqueeze(0))
            if float(H) < router.entropy_threshold:
                choices[i] = 0
                continue
            t0 = int(next_token_ids[i].item())
            base_ctx = list(req.origin_input_ids) + list(req.output_ids)
            slm_lps = [log_prob_of_token(L0, t0)]
            draft = [t0]
            entropy_path = [float(H)]
            temp = float(req.sampling_params.temperature)
            top_p = float(req.sampling_params.top_p)
            top_k = int(req.sampling_params.top_k)
            for _ in range(n):
                ctx_k = base_ctx + draft
                Lk = SLMServer._slm_one_off_forward_logits(self.quick_scheduler, ctx_k)
                entropy_path.append(float(compute_entropy(Lk.unsqueeze(0))))
                tk = sample_token(Lk, temperature=temp, top_p=top_p, top_k=top_k)
                tid = int(tk.item()) if isinstance(tk, torch.Tensor) else int(tk)
                slm_lps.append(log_prob_of_token(Lk, tid))
                draft.append(tid)
            entropy_path_sum = float(sum(entropy_path))
            contexts = [base_ctx + draft[:k] for k in range(len(draft))]
            qid = int(time.time_ns() % (1 << 62))
            rpc = EntropyLookaheadRpc(query_id=qid, contexts=contexts, tokens=draft)
            resp = self._llm_entropy_lookahead_rpc(rpc)
            if not isinstance(resp, EntropyLookaheadResp) or not resp.ok or len(resp.logprobs) != len(draft):
                err = getattr(resp, "error", "unknown")
                _ds_log_entropy_lookahead_rpc_fail_throttled(err, i)
                choices[i] = 1
                append_entropy_lookahead_score_log(
                    router.score_log_path,
                    {
                        "event": "entropy_lookahead_triggered",
                        "rid": getattr(req, "rid", None),
                        "seq_in_batch": i,
                        "entropy": float(H),
                        "entropy_path": entropy_path,
                        "entropy_path_sum": entropy_path_sum,
                        "entropy_threshold": float(router.entropy_threshold),
                        "S": None,
                        "score_threshold": float(router.score_threshold),
                        "lookahead_steps": n,
                        "num_scored_tokens": len(draft),
                        "routed_to_llm": True,
                        "rpc_ok": False,
                        "error": str(err),
                    },
                )
                continue
            S = sum(slm_lps[j] - resp.logprobs[j] for j in range(len(draft)))
            choices[i] = 1 if S > router.score_threshold else 0
            append_entropy_lookahead_score_log(
                router.score_log_path,
                {
                    "event": "entropy_lookahead_triggered",
                    "rid": getattr(req, "rid", None),
                    "seq_in_batch": i,
                    "entropy": float(H),
                    "entropy_path": entropy_path,
                    "entropy_path_sum": entropy_path_sum,
                    "entropy_threshold": float(router.entropy_threshold),
                    "S": float(S),
                    "score_threshold": float(router.score_threshold),
                    "lookahead_steps": n,
                    "num_scored_tokens": len(draft),
                    "routed_to_llm": bool(int(choices[i].item()) == 1),
                    "rpc_ok": True,
                    "slm_logprobs": [float(x) for x in slm_lps],
                    "llm_logprobs": [float(x) for x in resp.logprobs],
                },
            )
        router.state.last_model = "reference" if choices.any().item() else "quick"
        return choices

    @staticmethod
    def simple_prepare_for_extend(batch: ScheduleBatch):
        batch.forward_mode = ForwardMode.EXTEND

        # Allocate req slots (must be pool indices from allocator, not req.rid — rids can be str)
        bs = len(batch.reqs)
        req_pool_indices = batch.alloc_req_slots(bs)

        # Init tensors
        reqs = batch.reqs
        input_ids = [r.fill_ids[len(r.prefix_indices) :] for r in reqs]
        extend_num_tokens = sum(len(ids) for ids in input_ids)
        seq_lens = [len(r.fill_ids) for r in reqs]
        prefix_lens = [len(r.prefix_indices) for r in reqs]
        extend_lens = [r.extend_input_len for r in reqs]
        req_pool_indices_tensor = torch.tensor(req_pool_indices, dtype=torch.int64).to(
            batch.device, non_blocking=True
        )
        input_ids_tensor = torch.tensor(sum(input_ids, []), dtype=torch.int64).to(
            batch.device, non_blocking=True
        )
        seq_lens_tensor = torch.tensor(seq_lens, dtype=torch.int64).to(
            batch.device, non_blocking=True
        )
        prefix_lens_tensor = torch.tensor(prefix_lens, dtype=torch.int64).to(
            batch.device, non_blocking=True
        )
        extend_lens_tensor = seq_lens_tensor - prefix_lens_tensor
        for i, (req, seq_len, pre_len) in enumerate(zip(reqs, seq_lens, prefix_lens)):
            req.req_pool_idx = req_pool_indices[i]
            assert seq_len - pre_len == req.extend_input_len
            req.cached_tokens += pre_len - req.already_computed
            req.already_computed = seq_len
            req.is_retracted = False
        if extend_num_tokens == 0:
            out_cache_loc = None
        elif batch.token_to_kv_pool_allocator.page_size == 1:
            out_cache_loc = batch.alloc_token_slots(extend_num_tokens)
        else:
            last_loc = get_last_loc(
                batch.req_to_token_pool.req_to_token,
                req_pool_indices_tensor,
                prefix_lens_tensor,
            )
            out_cache_loc = batch.alloc_paged_token_slots_extend(
                prefix_lens_tensor,
                seq_lens_tensor,
                last_loc,
                extend_num_tokens,
            )

        # Set fields
        batch.input_ids = input_ids_tensor
        batch.req_pool_indices = req_pool_indices_tensor
        batch.seq_lens = seq_lens_tensor
        batch.out_cache_loc = out_cache_loc
        batch.input_embeds = None
        batch.seq_lens_sum = sum(seq_lens)
        if batch.return_logprob:
            batch.top_logprobs_nums = [r.top_logprobs_num for r in reqs]
            batch.token_ids_logprobs = [r.token_ids_logprob for r in reqs]
        batch.extend_logprob_start_lens = [r.extend_logprob_start_len for r in reqs]
        batch.extend_num_tokens = extend_num_tokens
        batch.prefix_lens = prefix_lens
        batch.extend_lens = extend_lens
        if out_cache_loc is not None:
            write_req_to_token_pool_triton[(bs,)](
                batch.req_to_token_pool.req_to_token,
                req_pool_indices_tensor,
                prefix_lens_tensor,
                seq_lens_tensor,
                extend_lens_tensor,
                out_cache_loc,
                batch.req_to_token_pool.req_to_token.shape[1],
            )
        # Build sampling info
        batch.sampling_info = SamplingBatchInfo.from_schedule_batch(
            batch,
            batch.model_config.vocab_size,
        )

    def extend_step(self, input_ids: List[List[int]], input_indices: List[int], sampling_params: SamplingParams) -> List[int]:
        """
        Extend the input ids using the reference model
        """
        subset_batch_size = len(input_ids)
        input_texts = self.tokenizer.batch_decode(input_ids)
        reqs = []
        for i, (input_text, input_id) in enumerate(zip(input_texts, input_ids)):
            req = Req(
                rid=input_indices[i],
                origin_input_text=input_text,
                origin_input_ids=input_id,
                sampling_params=sampling_params,
                eos_token_ids=self.quick_scheduler.model_config.hf_eos_token_id, # noqa
                return_hidden_states=False,
                vocab_size=self.quick_scheduler.model_config.vocab_size
            )
            req.sampling_params.normalize(None) # disable str-based stop token
            req.prefix_indices = self.reference_prefix_indices_list[input_indices[i]]
            req.fill_ids = input_id
            req.extend_input_len = len(input_id) - len(self.reference_prefix_indices_list[input_indices[i]])
            reqs.append(req)

        for q in self.reference_model_input_queues:
            q.put_nowait(reqs)
        next_token_ids = self.reference_model_output_queue.get()

        # Update prefix indices for each prompt
        for i in range(subset_batch_size):
            self.reference_prefix_indices_list[input_indices[i]]=list(range(len(input_ids[i])))

        return next_token_ids

    def decode_step(self, scheduler: Scheduler, temperature: float = 0.0, top_p: float = 1.0, top_k: int = -1):
        """
        Decode one step using the quick model
        
        Args:
            scheduler: The scheduler to use
            
        Returns:
            batch: The batch to use
            hidden_states: The hidden states from the quick model, shape (batch_size, seq_len, hidden_size)
            logits: The logits from the quick model, shape (batch_size, 1, vocab_size)
            next_token_ids: The next token ids from the quick model, shape (batch_size)
        """
        batch = scheduler.get_next_batch_to_run()
        result = scheduler.run_batch(batch)

        device = batch.seq_lens.device
        extend_lens = torch.tensor(batch.extend_lens, device=device)
        batch_size = batch.batch_size()
        is_prefill = (result.logits_output.hidden_states.shape[0] != batch_size)

        if is_prefill:
            # For prefill, use cumsum of extend_lens to get correct indices
            hidden_indices = torch.cumsum(extend_lens, dim=0) - 1
        else:
            # For decode, use sequential indices
            hidden_indices = torch.arange(batch_size, device=device)

        # Get hidden states for the relevant positions
        hidden_states = result.logits_output.hidden_states[hidden_indices, :][:, None, :] # batch_size, 1, hidden_size
        logits = result.logits_output.next_token_logits # batch_size, vocab_size
        next_token_ids = sample_token(logits, temperature=temperature, top_p=top_p, top_k=top_k)

        return batch, hidden_states, logits[:, None, :], next_token_ids

    def update_output_ids(self, batch: ScheduleBatch, scheduler: Scheduler, next_token_ids: List[int]):
        """Update the output ids for the batch"""
        batch.output_ids = next_token_ids

        for req, next_token_id in zip(batch.reqs, next_token_ids):
            if next_token_id in self.quick_scheduler.model_config.hf_eos_token_id:
                scheduler.abort_request(AbortReq(req.rid))
            req.output_ids.append(next_token_id.item())
            req.check_finished()
            if req.finished():
                scheduler.tree_cache.cache_finished_req(req)

        scheduler.last_batch = batch

    def generate(
        self,
        input_ids: List[List[int]],
        max_new_tokens: int = 2048,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 100,
        record_generation: bool = False,
        print_tokens: bool = False,
    ) -> Union[
        List[str],
        Tuple[List[str], List[GenerationRecorder]]
    ]:
        """
        Generate text using dynamic model selection with SGLang models.

        Args:
            input_ids: A list of lists of token IDs for batch processing
            max_new_tokens: Maximum number of tokens to generate
            temperature: Temperature for sampling
            top_p: Top-p probability threshold for nucleus sampling
            top_k: Top-k for sampling
            record_generation: If True, return both generated text and generation records
            print_tokens: Whether to print tokens during generation

        Returns:
            If record_generation is False: list of generated texts
            If record_generation is True: tuple of (list of generated texts, list of GenerationRecorders)
        """

        self.reset_cache_simple()
        batch_input_ids = input_ids
        batch_size = len(batch_input_ids)
        self.reference_prefix_indices_list = [[] for _ in range(batch_size)]

        # Setup recorders if recording is enabled
        recorders = (
            [GenerationRecorder() for _ in range(batch_size)]
            if record_generation
            else None
        )

        # Prepare sampling parameters for SGLang
        reference_sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_new_tokens=1,
            stop=[],
        )

        # sglang will revise the output logits in-place if we set temperature > 0.0
        # so we set temperature to 0.0 here and sample in the decode_step
        quick_sampling_params = SamplingParams(
            temperature=0.0,
            top_p=top_p,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
            stop=[],
        )

        # Use uid to generate unique ids for each request
        rids = [str(uuid.uuid4()) for _ in range(batch_size)]
        for i, input_id in enumerate(batch_input_ids):
            req = Req(
                rid=rids[i],
                origin_input_text=self.tokenizer.decode(input_id),
                origin_input_ids=input_id,
                sampling_params=quick_sampling_params,
                eos_token_ids=self.quick_scheduler.model_config.hf_eos_token_id,
                return_hidden_states=True,
                vocab_size=self.quick_scheduler.model_config.vocab_size
            )
            self.quick_scheduler.waiting_queue.append(req)

        # Initialize token manager with tokenized inputs
        token_manager = SGLangTokenManager(
            batch_input_ids, self.tokenizer, max_new_tokens=max_new_tokens
        )

        # Generate tokens one by one until all prompts reach EOS or max limit
        position = 0

        if not print_tokens:
            # Create tqdm progress bar for token generation
            pbar = tqdm(total=max_new_tokens, desc="Dynamic SGLang: Generating tokens", leave=True)
        while not token_manager.is_generation_complete() and position < max_new_tokens:
            if not print_tokens:
                pbar.update(1)

            active_count = token_manager.get_active_count()

            if active_count < 1:
                break

            # Generate with quick model to get hidden states
            batch, hidden_states, logits, next_token_ids = self.decode_step(self.quick_scheduler, temperature=temperature, top_p=top_p, top_k=top_k)

            # Create a ModelOutputs object for switching strategy
            model_outputs = ModelOutputs(
                logits=logits,
                hidden_states=[hidden_states],  # dummy layer dimension
                token=next_token_ids[:, None],
            )

            if isinstance(self.switching_strategy, EntropyLookaheadSwitching):
                model_choices = self._ds_entropy_lookahead_choices(
                    self.switching_strategy, batch, logits, next_token_ids
                )
            else:
                model_choices = self.switching_strategy.route(model_outputs)

            # Check if reference model is needed for any prompt
            reference_needed = torch.any(model_choices).item()

            if reference_needed:
                # Get indices of inputs that need reference model as a list
                reference_indices = torch.where(model_choices == 1)[0].tolist()
                active_to_original = token_manager.get_active_index()
                reference_original_indices = [active_to_original[i] for i in reference_indices]
                reference_input_ids = token_manager.fetch_active_input_ids(reference_indices)

                # Generate with reference model for inputs that need it
                reference_outputs = self.extend_step(
                    input_ids=reference_input_ids,
                    input_indices=reference_original_indices,
                    sampling_params=reference_sampling_params,
                )
                for i, reference_output_token in enumerate(reference_outputs):
                    next_token_ids[reference_indices[i]] = reference_output_token

                # Combine outputs based on model choices
                # Record if needed
                if record_generation and recorders:
                    for i in range(active_count):
                        if model_choices[i].item() == 1:  # Use reference model
                            # update next token ids
                            source_model = "reference"
                            param_size = float(self.model_config["reference"]["param"])
                        else:  # Use quick model
                            source_model = "quick"
                            param_size = float(self.model_config["quick"]["param"])

                        token = next_token_ids[i].item()
                        token_str = self.tokenizer.decode(token)

                        # Add record
                        active_indicies = token_manager.get_active_index()
                        seq_idx = active_indicies[i]
                        recorders[seq_idx].add_record(
                            GenerationRecord(
                                token_id=token,
                                token_str=token_str,
                                source_model=source_model,
                                position=position,
                                batch_id=seq_idx,
                                param_size=param_size,
                            )
                        )

                        # Print tokens if requested
                        if (
                            print_tokens and seq_idx == 0
                        ):  # Only print for the first batch
                            color = (
                                REFERENCE_COLOR
                                if source_model == "reference"
                                else QUICK_COLOR
                            )
                            print(f"{color}{token_str}{RESET}", end="", flush=True)

            else:
                # Use quick model for all outputs
                # Record if needed
                if record_generation and recorders:
                    for i in range(active_count):
                        token = next_token_ids[i].item()
                        token_str = self.tokenizer.decode(token)

                        # Find original batch index
                        seq_idx = token_manager.get_active_index()[i]

                        # Add record
                        recorders[seq_idx].add_record(
                            GenerationRecord(
                                token_id=token,
                                token_str=token_str,
                                source_model="quick",
                                position=position,
                                batch_id=seq_idx,
                                param_size=float(self.model_config["quick"]["param"]),
                            )
                        )

                        # Print tokens if requested
                        if (
                            print_tokens and seq_idx == 0
                        ):  # Only print for the first batch
                            print(f"{QUICK_COLOR}{token_str}{RESET}", end="", flush=True)

            # Update token manager with final outputs
            self.update_output_ids(batch, self.quick_scheduler, next_token_ids)
            token_manager.update_sequences_direct([token_id.item() for token_id in next_token_ids])
            position += 1

        # Get final outputs from token manager
        final_results = token_manager.get_final_outputs()

        # Prepare return values
        generated_texts = []
        for result in final_results:
            # Combine prompt and output
            output_text = self.tokenizer.decode(result["output_ids"])
            # generated_text = result["prompt"] + output_text
            generated_text = output_text
            generated_texts.append(generated_text)

        if record_generation:
            return generated_texts, recorders
        return generated_texts, None

    def reset_cache_simple(self):
        """Reset the cache for the quick model"""
        for q in self.reference_model_input_queues:
            q.put_nowait("RESET_CACHE")
        # Wait for acknowledgment from the reference model
        for q in self.reference_model_ack_queues:
            ack = q.get()    
            # print(f"cache location reset to {ack}")

    def shutdown(self):
        """Shut down the SGLang engines to free resources"""
        for q in self.reference_model_input_queues:
            q.put_nowait(-1)  # Termination signal

    def __del__(self):
        if hasattr(self, "reference_model_procs"):
            for proc in self.reference_model_procs:
                if proc.is_alive():
                    proc.terminate()
                    proc.join()
