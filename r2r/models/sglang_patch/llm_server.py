import uuid
import torch
import time
import zmq
import pickle
from tqdm import tqdm
from typing import Optional, Tuple, Union, List
import atexit
import os
import signal
import multiprocessing as mp
import socket
import torch.distributed as dist
import threading
import queue
import sys
import traceback
from transformers import AutoTokenizer
from multiprocessing import Value
import nvtx

from r2r.models.recorder import GenerationRecord, GenerationRecorder
from r2r.utils.config import (
    QUICK_COLOR,
    REFERENCE_COLOR,
    RESET,
)
from r2r.utils.switching import create_switching_strategy
from r2r.utils.token_manager import SGLangTokenManager
from r2r.utils.dataclass import ModelOutputs
from r2r.utils.sampling import sample_token
from r2r.utils.metrics import log_prob_of_token
from r2r.models.sglang_patch.schedule_req import (
    WaitingReq,
    SimpleSamplingParams,
    EntropyLookaheadRpc,
    EntropyLookaheadResp,
    SlidingWindowJsRpc,
    SlidingWindowJsResp,
    NextTokenJsRpc,
    NextTokenJsAbortRpc,
    NextTokenJsResp,
)
from r2r.models.sglang_patch.llm_scheduler import LLMScheduler

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
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.managers.io_struct import AbortReq
from sglang.srt.utils import broadcast_pyobj

class LLMServer:
    """LLM Server launched by SGLang"""
    def __init__(
        self,
        model_config: dict,
        reference_sglang_kwargs: dict,
        quick_num_gpus: int,
        reference_num_gpus: int,
        reference_master_port: int | None = None,
        ready_queue: Optional[mp.Queue] = None,
        overlap_tp_schedule: bool = False,
        mem_fraction_static: Optional[float] = None,
        llm_kvcache_size: Optional[Value] = None,
        min_batch_size: Union[int, list[int]] = 1,
        entropy_lookahead_query_queue: Optional[mp.Queue] = None,
        entropy_lookahead_reply_queue: Optional[mp.Queue] = None,
    ):
        self.is_reset_cache = False
        self.shutdown_loop = False
        self.batch = None
        self.new_reqs = []
        self.model_config = model_config
        self.reference_sglang_kwargs = reference_sglang_kwargs
        self.quick_num_gpus = quick_num_gpus if overlap_tp_schedule is False else 0
        self.reference_num_gpus = reference_num_gpus
        # Queue for signaling worker readiness to outside controller (e.g., SLDisaggregationSystem)
        self.ready_queue = ready_queue
        # Inter-server queues (outbound to SLM / inbound from SLM)
        self.queue_to_slm = mp.Queue()
        self._inbound_queues = mp.Queue()

        # ================ Inter-server PUB (LLM -> SLM) BEFORE workers =====================
        # (Notify PUB for finished reqs remains separate above; this PUB is for generic inter-server msgs)
        try:
            self._pub_slm = zmq.Context.instance().socket(zmq.PUB)
            self._pub_slm.setsockopt(zmq.LINGER, 0)
            self._pub_slm.setsockopt(zmq.SNDHWM, 100000)
            self.slm_recv_port = self._pub_slm.bind_to_random_port("tcp://127.0.0.1")
        except Exception as e:
            print(f"[LLMServer] Failed to bind PUB to SLM (early): {e}")
            self._pub_slm = None
            self.slm_recv_port = None

        self._send_slm_stop = threading.Event()
        def _send_slm_loop():
            time.sleep(0.05)
            while not self._send_slm_stop.is_set():
                try:
                    item = self.queue_to_slm.get(timeout=0.1)
                except Exception:
                    continue
                if item is None:
                    break
                if self._pub_slm is None:
                    continue
                try:
                    self._pub_slm.send_pyobj(item)
                except Exception:
                    pass
        self._send_slm_thread = threading.Thread(target=_send_slm_loop, daemon=True)
        self._send_slm_thread.start()

        self._sub_from_slm = None
        self._recv_from_slm_stop = threading.Event()
        self._recv_from_slm_thread = None

        # Pick a dedicated master port for reference model's process group to avoid clashing with quick model (default 29500)
        if reference_master_port is None:
            # find a free port
            def _find_free_port():
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(("127.0.0.1", 0))
                    return s.getsockname()[1]
            reference_master_port = _find_free_port()
        self.reference_master_port = reference_master_port

        # Ensure worker cleanup on interpreter exit and on signals
        atexit.register(self.shutdown)
        def _sig_handler(sig, frame):
            try:
                self.shutdown()
            finally:
                # Hard-exit to avoid hanging CUDA/NCCL threads
                os._exit(0)
        try:
            signal.signal(signal.SIGINT, _sig_handler)
            signal.signal(signal.SIGTERM, _sig_handler)
        except Exception:
            # Some environments may disallow setting signals (e.g., Jupyter)
            pass

        print(f"Loading reference model {self.model_config['reference']['model_name']}...")

        if reference_sglang_kwargs.get("attention_backend", None) != "flashinfer":
            print(f"Only support flashinfer attention backend for reference model.")
            reference_sglang_kwargs["attention_backend"] = "flashinfer"
        
        # Entropy lookahead issues many long one-off prefills on the reference model.
        # - disable_cuda_graph: frees graph-capture / private-pool memory that contributed to CUDA OOM.
        # - radix cache on: shared prefix across lookahead contexts (base+draft[:k]) cuts redundant KV work.
        # - smaller chunked_prefill_size: lowers peak activation memory on long prefills (override via env).
        _chunk_prefill = int(
            os.environ.get("R2R_REFERENCE_CHUNKED_PREFILL", "1024")
        )
        reference_server_args = ServerArgs(
            model_path=self.model_config["reference"]["model_path"],
            disable_cuda_graph=True,
            disable_overlap_schedule=True,
            disable_radix_cache=False,
            chunked_prefill_size=_chunk_prefill,
            mem_fraction_static=mem_fraction_static,
            **reference_sglang_kwargs,
        )
        reference_server_args.tp_size = reference_num_gpus
        # Rank queues kept for future forwarded request injection (e.g., from SLM via inbound queue processing)
        self._stop_event = threading.Event()
        self._recv_thread = None  # Deprecated broadcast SUB thread removed.
        self.reference_model_procs = []
        self.llm_kvcache_size = llm_kvcache_size
        self.entropy_lookahead_query_queue = entropy_lookahead_query_queue
        self.entropy_lookahead_reply_queue = entropy_lookahead_reply_queue
        for rank in range(reference_num_gpus):
            proc = mp.Process(
                target=self.reference_model_worker,
                args=(
                    rank, 
                    self.quick_num_gpus, 
                    self.reference_num_gpus, 
                    reference_server_args, 
                    self.reference_master_port, 
                    self.ready_queue,
                    self._inbound_queues, 
                    self.queue_to_slm,
                    self.llm_kvcache_size if rank == 0 else None,
                    min_batch_size,
                    self.entropy_lookahead_query_queue,
                    self.entropy_lookahead_reply_queue,
                ),
            )
            # Mark as daemon so that workers die when parent exits unexpectedly
            proc.daemon = True
            proc.start()
            self.reference_model_procs.append(proc)

    @staticmethod
    def _free_evjs_stash_entry(scheduler: Scheduler, rid: str) -> None:
        """Release KV for a fused NextTokenJsRpc prefill that will not be continued."""
        st = getattr(scheduler, "_r2r_evjs_stash", None)
        if not st:
            return
        entry = st.pop(rid, None)
        if not entry:
            return
        req = entry["req"]
        kv_locs = entry.get("kv_locs")
        try:
            if kv_locs is not None and kv_locs.numel() > 0:
                scheduler.token_to_kv_pool_allocator.free(kv_locs)
        except Exception:
            pass
        try:
            if getattr(req, "req_pool_idx", None) is not None:
                scheduler.req_to_token_pool.free(req.req_pool_idx)
        except Exception:
            pass
        try:
            scheduler.abort_request(AbortReq(rid))
        except Exception:
            pass

    @staticmethod
    def _llm_one_off_forward_logits_persist(scheduler: Scheduler, context_ids: List[int], rid: str) -> torch.Tensor:
        """Keeps KV for EVJS_CONTINUE. Reuses stashed KV when possible (incremental extend)."""
        rid = str(rid)
        if not hasattr(scheduler, "_r2r_evjs_stash"):
            scheduler._r2r_evjs_stash = {}

        stash = scheduler._r2r_evjs_stash.get(rid)
        if stash is not None:
            try:
                row = LLMServer._llm_incremental_forward(
                    scheduler, stash, context_ids, rid
                )
                if row is not None:
                    return row
            except Exception as e:
                print(f"[evjs-incr-fallback] rid={rid} error={e}, falling back to full prefill", flush=True)
            LLMServer._free_evjs_stash_entry(scheduler, rid)
        else:
            LLMServer._free_evjs_stash_entry(scheduler, rid)

        return LLMServer._llm_full_prefill_persist(scheduler, context_ids, rid)

    @staticmethod
    def _llm_incremental_forward(
        scheduler: Scheduler, stash: dict, context_ids: List[int], rid: str,
    ) -> Optional[torch.Tensor]:
        """Extend stashed KV with new tokens only. Returns None if extension not possible."""
        old_req = stash["req"]
        old_len = stash.get("context_len", len(list(old_req.origin_input_ids)))
        new_len = len(context_ids)
        if new_len <= old_len:
            return None
        old_prefix = list(old_req.origin_input_ids)
        if context_ids[:old_len] != old_prefix[:old_len]:
            return None
        old_pool_idx = getattr(old_req, "req_pool_idx", None)
        if old_pool_idx is None:
            return None

        new_tokens = context_ids[old_len:]
        n_new = len(new_tokens)

        sp = SamplingParams(temperature=0.0, top_p=1.0, top_k=-1, max_new_tokens=1, stop=[])
        sp.normalize(None)
        req = Req(
            rid=rid,
            origin_input_text="",
            origin_input_ids=context_ids,
            sampling_params=sp,
            eos_token_ids=scheduler.model_config.hf_eos_token_id,
            return_hidden_states=False,
            vocab_size=scheduler.model_config.vocab_size,
            status="need",
        )
        req.req_pool_idx = old_pool_idx
        req.output_ids = []
        req.fill_ids = list(context_ids)
        req.extend_input_len = n_new
        req.already_computed = old_len
        req.cached_tokens = old_len
        req.is_retracted = False
        req.prefix_indices = [0] * old_len

        batch = ScheduleBatch.init_new(
            [req],
            scheduler.req_to_token_pool,
            scheduler.token_to_kv_pool_allocator,
            scheduler.tree_cache,
            scheduler.model_config,
            scheduler.enable_overlap,
            scheduler.spec_algorithm,
            scheduler.server_args.enable_custom_logit_processor,
        )
        batch.forward_mode = ForwardMode.EXTEND
        if not hasattr(req, "device"):
            req.device = batch.device
        device = batch.device

        if batch.token_to_kv_pool_allocator.page_size == 1:
            out_cache_loc = batch.alloc_token_slots(n_new)
        else:
            prefix_t = torch.tensor([old_len], dtype=torch.int64, device=device)
            seq_t = torch.tensor([new_len], dtype=torch.int64, device=device)
            pool_t = torch.tensor([old_pool_idx], dtype=torch.int64, device=device)
            last_loc = get_last_loc(
                batch.req_to_token_pool.req_to_token, pool_t, prefix_t,
            )
            out_cache_loc = batch.alloc_paged_token_slots_extend(
                prefix_t, seq_t, last_loc, n_new,
            )

        input_ids_t = torch.tensor(new_tokens, dtype=torch.int64, device=device)
        req_pool_t = torch.tensor([old_pool_idx], dtype=torch.int64, device=device)
        seq_lens_t = torch.tensor([new_len], dtype=torch.int64, device=device)
        prefix_lens_t = torch.tensor([old_len], dtype=torch.int64, device=device)
        extend_lens_t = seq_lens_t - prefix_lens_t

        write_req_to_token_pool_triton[(1,)](
            batch.req_to_token_pool.req_to_token,
            req_pool_t,
            prefix_lens_t,
            seq_lens_t,
            extend_lens_t,
            out_cache_loc,
            batch.req_to_token_pool.req_to_token.shape[1],
        )

        batch.input_ids = input_ids_t
        batch.req_pool_indices = req_pool_t
        batch.seq_lens = seq_lens_t
        batch.out_cache_loc = out_cache_loc
        batch.input_embeds = None
        batch.seq_lens_sum = new_len
        batch.extend_logprob_start_lens = [0]
        batch.extend_num_tokens = n_new
        batch.prefix_lens = [old_len]
        batch.extend_lens = [n_new]
        batch.return_logprob = False
        batch.sampling_info = SamplingBatchInfo.from_schedule_batch(
            batch, batch.model_config.vocab_size,
        )

        model_batch = batch.get_model_worker_batch()
        result = scheduler.tp_worker.forward_batch_generation(model_batch)
        logits_output = result[0] if isinstance(result, tuple) else result.logits_output
        row = logits_output.next_token_logits[0].float()

        all_kv = torch.cat([stash.get("kv_locs", torch.tensor([], device=device)), out_cache_loc])
        scheduler._r2r_evjs_stash[rid] = {
            "req": req,
            "kv_locs": all_kv,
            "context_len": new_len,
        }
        if os.environ.get("R2R_LOG_EVJS_ALL", "").strip().lower() in ("1", "true", "yes", "on"):
            print(
                f"[evjs-incr-ok] rid={rid} old_len={old_len} new_len={new_len} "
                f"delta={n_new} kv_total={all_kv.numel()}",
                flush=True,
            )
        return row

    @staticmethod
    def _llm_full_prefill_persist(scheduler: Scheduler, context_ids: List[int], rid: str) -> torch.Tensor:
        """Full-context prefill, stashing KV for potential EVJS_CONTINUE reuse."""
        sp = SamplingParams(temperature=0.0, top_p=1.0, top_k=-1, max_new_tokens=1, stop=[])
        sp.normalize(None)
        req = Req(
            rid=rid,
            origin_input_text=scheduler.tokenizer.decode(context_ids),
            origin_input_ids=context_ids,
            sampling_params=sp,
            eos_token_ids=scheduler.model_config.hf_eos_token_id,
            return_hidden_states=False,
            vocab_size=scheduler.model_config.vocab_size,
            status="need",
        )
        req.prefix_indices = []
        req.output_ids = []
        req.fill_ids = list(context_ids)
        req.extend_input_len = len(req.fill_ids)
        req.init_next_round_input(scheduler.tree_cache)
        new_batch = ScheduleBatch.init_new(
            [req],
            scheduler.req_to_token_pool,
            scheduler.token_to_kv_pool_allocator,
            scheduler.tree_cache,
            scheduler.model_config,
            scheduler.enable_overlap,
            scheduler.spec_algorithm,
            scheduler.server_args.enable_custom_logit_processor,
        )
        if not hasattr(req, "device"):
            req.device = new_batch.device
        LLMServer.simple_prepare_for_extend(new_batch)
        model_batch = new_batch.get_model_worker_batch()
        kv_locs = new_batch.out_cache_loc
        try:
            result = scheduler.tp_worker.forward_batch_generation(model_batch)
            logits_output = result[0] if isinstance(result, tuple) else result.logits_output
            row = logits_output.next_token_logits[0].float()
            scheduler._r2r_evjs_stash[rid] = {
                "req": req,
                "kv_locs": kv_locs,
                "context_len": len(context_ids),
            }
            return row
        except Exception:
            try:
                if kv_locs is not None and kv_locs.numel() > 0:
                    scheduler.token_to_kv_pool_allocator.free(kv_locs)
            except Exception:
                pass
            try:
                if getattr(req, "req_pool_idx", None) is not None:
                    scheduler.req_to_token_pool.free(req.req_pool_idx)
            except Exception:
                pass
            try:
                scheduler.abort_request(AbortReq(rid))
            except Exception:
                pass
            raise

    @staticmethod
    def _llm_one_off_forward_logits(scheduler: Scheduler, context_ids: List[int]) -> torch.Tensor:
        rid = f"_elb_{uuid.uuid4()}"
        sp = SamplingParams(temperature=0.0, top_p=1.0, top_k=-1, max_new_tokens=1, stop=[])
        sp.normalize(None)
        req = Req(
            rid=rid,
            origin_input_text=scheduler.tokenizer.decode(context_ids),
            origin_input_ids=context_ids,
            sampling_params=sp,
            eos_token_ids=scheduler.model_config.hf_eos_token_id,
            return_hidden_states=False,
            vocab_size=scheduler.model_config.vocab_size,
            status="need",
        )
        req.prefix_indices = []
        req.output_ids = []
        req.fill_ids = list(context_ids)
        req.extend_input_len = len(req.fill_ids)
        req.init_next_round_input(scheduler.tree_cache)
        new_batch = ScheduleBatch.init_new(
            [req],
            scheduler.req_to_token_pool,
            scheduler.token_to_kv_pool_allocator,
            scheduler.tree_cache,
            scheduler.model_config,
            scheduler.enable_overlap,
            scheduler.spec_algorithm,
            scheduler.server_args.enable_custom_logit_processor,
        )
        if not hasattr(req, "device"):
            req.device = new_batch.device
        LLMServer.simple_prepare_for_extend(new_batch)
        model_batch = new_batch.get_model_worker_batch()
        kv_locs = new_batch.out_cache_loc
        try:
            result = scheduler.tp_worker.forward_batch_generation(model_batch)
            logits_output = result[0] if isinstance(result, tuple) else result.logits_output
            # Single-token extend: one row
            return logits_output.next_token_logits[0].float()
        finally:
            try:
                if kv_locs is not None and kv_locs.numel() > 0:
                    scheduler.token_to_kv_pool_allocator.free(kv_locs)
            except Exception:
                pass
            try:
                if getattr(req, "req_pool_idx", None) is not None:
                    scheduler.req_to_token_pool.free(req.req_pool_idx)
            except Exception:
                pass
            try:
                scheduler.abort_request(AbortReq(rid))
            except Exception:
                pass

    @staticmethod
    def _entropy_lookahead_llm_logprobs(scheduler: Scheduler, rpc: EntropyLookaheadRpc) -> EntropyLookaheadResp:
        try:
            lps = []
            for ctx, tid in zip(rpc.contexts, rpc.tokens):
                L = LLMServer._llm_one_off_forward_logits(scheduler, ctx)
                lps.append(log_prob_of_token(L, int(tid)))
            return EntropyLookaheadResp(query_id=rpc.query_id, logprobs=lps, ok=True)
        except Exception as e:
            return EntropyLookaheadResp(
                query_id=rpc.query_id, logprobs=[], ok=False, error=str(e)
            )

    @staticmethod
    def _next_token_js_llm_logits(scheduler: Scheduler, rpc: NextTokenJsRpc) -> NextTokenJsResp:
        try:
            if getattr(rpc, "rid", None):
                row = LLMServer._llm_one_off_forward_logits_persist(
                    scheduler, list(rpc.context_ids), str(rpc.rid)
                )
            else:
                row = LLMServer._llm_one_off_forward_logits(scheduler, list(rpc.context_ids))
            k = max(1, int(os.environ.get("R2R_EVJS_TOPK", "16")))
            probs = torch.softmax(row.float(), dim=-1)
            k = min(k, int(probs.numel()))
            vals, idx = torch.topk(probs, k=k, dim=-1)
            tail = float(torch.clamp(1.0 - vals.sum(), min=0.0).item())
            return NextTokenJsResp(
                query_id=rpc.query_id,
                llm_topk_indices=[int(x) for x in idx.detach().cpu().tolist()],
                llm_topk_probs=[float(x) for x in vals.detach().cpu().tolist()],
                llm_tail_mass=tail,
                topk=int(k),
                ok=True,
            )
        except Exception as e:
            return NextTokenJsResp(
                query_id=rpc.query_id,
                llm_logits=None,
                ok=False,
                error=str(e),
            )

    @staticmethod
    def _sliding_window_js_llm_logits(scheduler: Scheduler, rpc: SlidingWindowJsRpc) -> SlidingWindowJsResp:
        try:
            full_ids = list(rpc.full_ids)
            B = int(rpc.base_len)
            n = int(rpc.window_size)
            if B < 0 or n < 1 or B > len(full_ids):
                raise ValueError(
                    f"invalid SlidingWindowJsRpc: base_len={B}, window_size={n}, len(full_ids)={len(full_ids)}"
                )
            L = len(full_ids) - B
            if L < n:
                raise ValueError(f"draft segment too short: L={L} < n={n}")
            # SGLang prefill only exposes next_token_logits at the *last* position per request
            # (see LogitsProcessor: extend without input logprobs uses last_index only).
            # So we cannot get a [seq_len, vocab] matrix from one forward; compute each
            # window row with the same prefix as SLM: context_j = full_ids[: B+L-n+j].
            outs = []
            for j in range(n):
                ctx = full_ids[: B + L - n + j]
                row = LLMServer._llm_one_off_forward_logits(scheduler, ctx)
                outs.append(row.detach().cpu().numpy())
            return SlidingWindowJsResp(query_id=rpc.query_id, llm_logits=outs, ok=True)
        except Exception as e:
            return SlidingWindowJsResp(
                query_id=rpc.query_id, llm_logits=[], ok=False, error=str(e)
            )

    @staticmethod
    def reference_model_worker(rank, quick_num_gpus: int, world_size: int, server_args: ServerArgs, master_port: int = 29500, ready_queue: Optional[mp.Queue] = None, inbound_queue: Optional[mp.Queue] = None, outbound_queue: Optional[mp.Queue] = None, llm_kvcache_size: Optional[Value] = None, min_batch_size: Union[int, list[int]] = 1, entropy_lookahead_query_queue: Optional[mp.Queue] = None, entropy_lookahead_reply_queue: Optional[mp.Queue] = None):
        # Register signal handler to ensure finally block execution on terminate
        def _worker_sig_handler(signum, frame):
            sys.exit(0)
        signal.signal(signal.SIGTERM, _worker_sig_handler)
        # Use a dedicated tcp init_method to avoid port collision with quick model's default 29500 store
        init_method = f"tcp://127.0.0.1:{master_port}"
        dist.init_process_group(backend='nccl', init_method=init_method, rank=rank, world_size=world_size)
        torch.cuda.set_device(rank + quick_num_gpus)

        port_args = PortArgs.init_new(server_args)
        scheduler = LLMScheduler(
            server_args=server_args,
            port_args=port_args,
            gpu_id=rank+quick_num_gpus,
            tp_rank=rank,
            dp_rank=0,
            moe_ep_rank=0,
            pp_rank=0, # Pipeline parallelism is not Supported
            llm_kvcache_size=llm_kvcache_size,
            min_batch_size=min_batch_size,
        )
        print(f"[reference rank {rank}] attn_tp_rank: {scheduler.attn_tp_rank}")
        
        # Signal readiness
        if ready_queue is not None:
            try:
                ready_queue.put(("READY", rank, scheduler.tokenizer if rank == 0 else None))
            except Exception as e:
                print(f"[rank {rank}] failed to put READY: {e}")
        
        LLMServer.init_batch_not_need(scheduler)
        print(f"Reference model worker {rank} started, waiting for requests...")
        
        # event_loop
        try:
            while True:
                if (
                    rank == 0
                    and entropy_lookahead_query_queue is not None
                    and entropy_lookahead_reply_queue is not None
                ):
                    while True:
                        try:
                            rpc = entropy_lookahead_query_queue.get_nowait()
                        except queue.Empty:
                            break
                        if isinstance(rpc, EntropyLookaheadRpc):
                            resp = LLMServer._entropy_lookahead_llm_logprobs(scheduler, rpc)
                            entropy_lookahead_reply_queue.put(resp)
                        elif isinstance(rpc, SlidingWindowJsRpc):
                            resp = LLMServer._sliding_window_js_llm_logits(scheduler, rpc)
                            entropy_lookahead_reply_queue.put(resp)
                        elif isinstance(rpc, NextTokenJsRpc):
                            resp = LLMServer._next_token_js_llm_logits(scheduler, rpc)
                            entropy_lookahead_reply_queue.put(resp)
                        elif isinstance(rpc, NextTokenJsAbortRpc):
                            pass

                if inbound_queue is not None: # Process message from LLM
                    slm_reqs = LLMServer.recv_reqs_from_slm(
                        inbound_queue=inbound_queue,
                        scheduler=scheduler,
                    )
                    is_shutdown = [msg for msg in slm_reqs if getattr(msg, "status", "") == "SHUTDOWN"]
                    is_reset_cache = [msg for msg in slm_reqs if getattr(msg, "status", "") == "RESET_CACHE"]
                    slm_reqs = [msg for msg in slm_reqs if getattr(msg, "status", "") not in ("SHUTDOWN", "RESET_CACHE")]
                    if is_shutdown:
                        print(f"[reference rank{rank}] SHUTDOWN received (queue), exiting...")
                        break
                    elif is_reset_cache:
                        ok = scheduler.flush_cache()
                        print(f"[reference rank{rank}] Cache reset (queue): {ok}")
                    if slm_reqs:
                        LLMServer.process_result_from_slm(scheduler, slm_reqs)

                # For LLM, there is no need to process new reqs from rank_queue
                if scheduler.waiting_queue:
                    nvtx.push_range("LLM")
                batch = scheduler.get_next_batch_to_run()
                if batch:
                    result = scheduler.run_batch(batch)
                    LLMServer.process_batch_results(rank, batch, result, scheduler, outbound_queue)
                    scheduler.last_batch = batch
                    nvtx.pop_range()
                try:
                    if os.getppid() == 1:
                        print(f"[rank {rank}] parent process disappeared, exiting worker.")
                        break
                except Exception:
                    pass
        except (SystemExit, KeyboardInterrupt):
            pass 
        except BaseException as e:
            # Any unexpected error -> exit loop to avoid orphaned NCCL workers
            print(f"[rank {rank}] reference worker fatal error: {e}. Exiting loop.")
            import traceback
            traceback.print_exc()
        finally:
            try:
                if dist.is_initialized():
                    dist.destroy_process_group()
            except Exception as e:
                print(f"[rank {rank}] destroy_process_group/close socket error: {e}")
    
    @staticmethod
    def init_batch_not_need(scheduler: Scheduler):
        scheduler.batch_not_need = ScheduleBatch.init_new(
            [],
            scheduler.req_to_token_pool,
            scheduler.token_to_kv_pool_allocator,
            scheduler.tree_cache,
            scheduler.model_config,
            scheduler.enable_overlap,
            scheduler.spec_algorithm,
            scheduler.server_args.enable_custom_logit_processor,
        )
        LLMServer.simple_prepare_for_extend(scheduler.batch_not_need)
        scheduler.batch_not_need.multimodal_inputs = []
        scheduler.batch_not_need.output_ids = torch.tensor([], dtype=torch.int64).to(
            scheduler.batch_not_need.device, non_blocking=True
        )
        scheduler.batch_not_need.orig_seq_lens = torch.tensor([], dtype=torch.int64).to(
            scheduler.batch_not_need.device, non_blocking=True
        )

    @staticmethod
    def _retract_rid_from_batch_not_need(scheduler: Scheduler, rid) -> None:
        """Drop cached LLM state for ``rid`` so the next message can full-prefill from ``new_token_ids``."""
        if scheduler.batch_not_need is None or not scheduler.batch_not_need.reqs:
            return
        not_keep = [
            i
            for i, r in enumerate(scheduler.batch_not_need.reqs)
            if r.rid == rid
        ]
        if not not_keep:
            return
        for i in not_keep:
            try:
                scheduler.tree_cache.cache_finished_req(scheduler.batch_not_need.reqs[i])
            except Exception:
                pass
        keep_indices = [
            i for i in range(len(scheduler.batch_not_need.reqs)) if i not in not_keep
        ]
        scheduler.batch_not_need.filter_batch(keep_indices=keep_indices)
        scheduler.n_active_reqs = max(0, int(getattr(scheduler, "n_active_reqs", 0)) - len(not_keep))

    @staticmethod
    def _enqueue_full_prefill_waiting_req(scheduler: Scheduler, waiting_req: WaitingReq) -> None:
        """First-time LLM prefill: ``new_token_ids`` is the full prompt + generated prefix."""
        if scheduler.batch_not_need is None:
            LLMServer.init_batch_not_need(scheduler)
        scheduler.n_active_reqs += 1
        origin_input_ids = waiting_req.new_token_ids
        origin_input_text = scheduler.tokenizer.decode(origin_input_ids)
        new_req = Req(
            rid=waiting_req.rid,
            origin_input_text=origin_input_text,
            origin_input_ids=origin_input_ids,
            sampling_params=waiting_req.sampling_params.derive_sampling_params(),
            eos_token_ids=scheduler.model_config.hf_eos_token_id,
            return_hidden_states=False,
            vocab_size=scheduler.model_config.vocab_size,
            status="need",
            last_cached_loc=[],
        )
        if not hasattr(new_req, "device"):
            new_req.device = scheduler.batch_not_need.device
        scheduler.waiting_queue.append(new_req)

    @staticmethod
    def _evjs_handle_continue(scheduler: Scheduler, waiting_req: WaitingReq) -> None:
        """Continue after fused JS prefill: reuse KV when stash matches, else full prefill fallback."""
        rid = str(waiting_req.rid)
        full_ids = list(waiting_req.new_token_ids)
        st = getattr(scheduler, "_r2r_evjs_stash", None)
        entry = st.pop(rid, None) if st else None
        if scheduler.batch_not_need is None:
            LLMServer.init_batch_not_need(scheduler)
        if entry is None:
            LLMServer._enqueue_full_prefill_waiting_req(scheduler, waiting_req)
            return
        req_old = entry["req"]
        prefix = list(req_old.origin_input_ids)
        if len(full_ids) < len(prefix) or full_ids[: len(prefix)] != prefix:
            LLMServer._enqueue_full_prefill_waiting_req(scheduler, waiting_req)
            return
        origin_input_ids = full_ids
        origin_input_text = scheduler.tokenizer.decode(origin_input_ids)
        new_req = Req(
            rid=req_old.rid,
            origin_input_text=origin_input_text,
            origin_input_ids=origin_input_ids,
            sampling_params=waiting_req.sampling_params.derive_sampling_params(),
            eos_token_ids=scheduler.model_config.hf_eos_token_id,
            return_hidden_states=False,
            vocab_size=scheduler.model_config.vocab_size,
            status="need",
            last_cached_loc=req_old.last_cached_loc,
        )
        new_req.req_pool_idx = req_old.req_pool_idx
        if not hasattr(new_req, "device"):
            new_req.device = scheduler.batch_not_need.device
        scheduler.waiting_queue.append(new_req)

    @staticmethod
    def process_result_from_slm(scheduler: Scheduler, commit_msgs):
        new_token_ids = {}
        returned_rid_list = []
        finished_rid_list = set()
        retract_prefill_rids = set()
        if scheduler.batch_not_need is not None:
            req_already_prefilled = [req.rid for req in scheduler.batch_not_need.reqs]
        else:
            req_already_prefilled = []
        for waiting_req in commit_msgs:
            if getattr(waiting_req, "status", "need") == "EVJS_CONTINUE":
                LLMServer._evjs_handle_continue(scheduler, waiting_req)
                continue
            if getattr(waiting_req, "status", "need") == "RETRACT_AND_PREFILL":
                retract_prefill_rids.add(waiting_req.rid)
                LLMServer._retract_rid_from_batch_not_need(scheduler, waiting_req.rid)
                if scheduler.batch_not_need is not None:
                    req_already_prefilled = [req.rid for req in scheduler.batch_not_need.reqs]
                else:
                    req_already_prefilled = []
            new_token_ids[waiting_req.rid] = waiting_req.new_token_ids
            returned_rid_list.append(waiting_req.rid)
            if waiting_req.status == "finished":
                finished_rid_list.add(waiting_req.rid)
                scheduler.n_active_reqs -= 1
                continue
            if waiting_req.rid not in req_already_prefilled:
                scheduler.n_active_reqs += 1
                origin_input_ids = waiting_req.new_token_ids
                origin_input_text = scheduler.tokenizer.decode(origin_input_ids)
                new_req = Req(
                    rid=waiting_req.rid,
                    origin_input_text=origin_input_text,
                    origin_input_ids=origin_input_ids,
                    sampling_params=waiting_req.sampling_params.derive_sampling_params(),
                    eos_token_ids=scheduler.model_config.hf_eos_token_id,
                    return_hidden_states=False,
                    vocab_size=scheduler.model_config.vocab_size,
                    status="need",
                    last_cached_loc=[],
                )
                if not hasattr(new_req, 'device'):
                    new_req.device = scheduler.batch_not_need.device
                scheduler.waiting_queue.append(new_req)

        if scheduler.batch_not_need is not None:
            not_keep_indices = []
            for i, req in enumerate(scheduler.batch_not_need.reqs):
                if req.rid in finished_rid_list:
                    scheduler.tree_cache.cache_finished_req(req)
                    continue
                if req.rid in retract_prefill_rids:
                    try:
                        scheduler.tree_cache.cache_finished_req(req)
                    except Exception:
                        pass
                    # Same as an in-place update: drop this slot from batch_not_need (do not append i to keep list).
                    continue
                if req.rid in returned_rid_list:
                    origin_input_ids = req.origin_input_ids+new_token_ids[req.rid]
                    origin_input_text = scheduler.tokenizer.decode(origin_input_ids)
                    new_req = Req(
                        rid=req.rid,
                        origin_input_text=origin_input_text,
                        origin_input_ids=origin_input_ids,
                        sampling_params=req.sampling_params,
                        return_hidden_states=False,
                        status="need",
                        last_cached_loc=req.last_cached_loc,
                    )
                    new_req.req_pool_idx = req.req_pool_idx
                    if not hasattr(new_req, 'device'):
                        new_req.device = scheduler.batch_not_need.device
                    scheduler.waiting_queue.append(new_req)
                else:
                    not_keep_indices.append(i)
            scheduler.batch_not_need.filter_batch(keep_indices=not_keep_indices)
    
    @staticmethod
    def recv_reqs_from_slm(inbound_queue: Optional[mp.Queue], scheduler: Scheduler):

        if scheduler.pp_rank == 0:
            if scheduler.attn_tp_rank == 0:
                recv_reqs = []

                while True:
                    try:
                        item = inbound_queue.get_nowait()
                    except queue.Empty:
                        break
                    except Exception:
                        break
                    recv_reqs.extend(item)
            else:
                recv_reqs = None
        else:
            raise RuntimeError("Pipeline parallelism is not supported.")

        if scheduler.server_args.enable_dp_attention:
            if scheduler.attn_tp_rank == 0:
                work_reqs = [
                    req
                    for req in recv_reqs
                ]
                control_reqs = [
                    req
                    for req in recv_reqs
                ]
            else:
                work_reqs = None
                control_reqs = None

            if scheduler.attn_tp_size != 1:
                work_reqs = broadcast_pyobj(
                    work_reqs,
                    scheduler.attn_tp_group.rank,
                    scheduler.attn_tp_cpu_group,
                    src=scheduler.attn_tp_group.ranks[0],
                )
            if scheduler.tp_size != 1:
                control_reqs = broadcast_pyobj(
                    control_reqs,
                    scheduler.tp_group.rank,
                    scheduler.tp_cpu_group,
                    src=scheduler.tp_group.ranks[0],
                )
            recv_reqs = work_reqs + control_reqs
        elif scheduler.tp_size != 1:
            recv_reqs = broadcast_pyobj(
                recv_reqs,
                scheduler.tp_group.rank,
                scheduler.tp_cpu_group,
                src=scheduler.tp_group.ranks[0],
            )
        return recv_reqs

    def shutdown(self):
        """Terminate reference model worker processes and close notify PUB.
        This is a best-effort cleanup for broken NCCL/TCPStore scenarios.
        """
        # stop recv thread
        try:
            if hasattr(self, "_stop_event"):
                self._stop_event.set()
            if hasattr(self, "_recv_thread") and self._recv_thread.is_alive():
                self._recv_thread.join(timeout=2)
        except Exception:
            pass
        # stop forwarder thread
        try:
            if hasattr(self, "_notify_queue"):
                try:
                    self._notify_queue.put_nowait(None)
                except Exception:
                    pass
        except Exception:
            pass
        # terminate worker processes
        if hasattr(self, "reference_model_procs"):
            for p in self.reference_model_procs:
                try:
                    if p.is_alive():
                        p.terminate()
                        p.join(timeout=5)
                        if p.is_alive():
                            # Force kill if it still refuses to exit
                            try:
                                p.kill()
                            except Exception:
                                pass
                except Exception:
                    pass
        # close PUB socket
        try:
            if hasattr(self, "notify_pub"):
                self.notify_pub.setsockopt(zmq.LINGER, 0)
                self.notify_pub.close(0)
        except Exception:
            pass
        # stop inter-server send thread
        try:
            if hasattr(self, "_send_slm_stop"):
                self._send_slm_stop.set()
            if hasattr(self, "queue_to_slm"):
                try:
                    self.queue_to_slm.put_nowait(None)
                except Exception:
                    pass
            if hasattr(self, "_send_slm_thread") and self._send_slm_thread.is_alive():
                self._send_slm_thread.join(timeout=2)
        except Exception:
            pass
        # stop inter-server recv thread
        try:
            if hasattr(self, "_recv_from_slm_stop"):
                self._recv_from_slm_stop.set()
            if hasattr(self, "_recv_from_slm_thread") and self._recv_from_slm_thread and self._recv_from_slm_thread.is_alive():
                self._recv_from_slm_thread.join(timeout=2)
        except Exception:
            pass
        # close inter-server sockets
        try:
            if hasattr(self, "_pub_slm") and self._pub_slm is not None:
                self._pub_slm.close(0)
        except Exception:
            pass
        try:
            if hasattr(self, "_sub_from_slm") and self._sub_from_slm is not None:
                self._sub_from_slm.close(0)
        except Exception:
            pass

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass
    # NOTE: original recv_requests removed; using central SUB thread.

    # Removed _sub_recv_loop: LLM no longer directly SUBscribes to broadcast req_port.

    # Controller-triggered: start SUB to receive from SLM (SLM -> LLM)
    def start_llm_sub(self, port: int):
        if port is None:
            print('[LLMServer] start_llm_sub called with None port')
            return
        if self._sub_from_slm is not None:
            return
        ctx = zmq.Context.instance()
        try:
            self._sub_from_slm = ctx.socket(zmq.SUB)
            self._sub_from_slm.setsockopt(zmq.LINGER, 0)
            self._sub_from_slm.connect(f"tcp://127.0.0.1:{port}")
            self._sub_from_slm.setsockopt(zmq.SUBSCRIBE, b"")
        except Exception as e:
            print(f"[LLMServer] Failed to connect SUB from SLM: {e}")
            self._sub_from_slm = None
            return
        def _recv_loop():
            poller = zmq.Poller()
            poller.register(self._sub_from_slm, zmq.POLLIN)
            while not self._recv_from_slm_stop.is_set():
                try:
                    events = dict(poller.poll(timeout=50))
                except Exception:
                    continue
                if self._sub_from_slm in events and events[self._sub_from_slm] == zmq.POLLIN:
                    while True:
                        try:
                            msg = self._sub_from_slm.recv_pyobj(flags=zmq.NOBLOCK)
                        except zmq.Again:
                            break
                        except Exception:
                            break
                        try:
                            self._inbound_queues.put_nowait(msg)
                        except Exception:
                            pass
        self._recv_from_slm_thread = threading.Thread(target=_recv_loop, daemon=True)
        self._recv_from_slm_thread.start()
        print(f"[LLMServer] SUB from SLM started on port {port}, loaded successfully")

    def enqueue_to_slm(self, obj):
        try:
            self.queue_to_slm.put_nowait(obj)
        except Exception:
            try:
                self.queue_to_slm.put(obj)
            except Exception:
                pass
    
    def process_batch_results(rank: int, batch: ScheduleBatch, result, scheduler: Scheduler, outbound_queue: Optional[mp.Queue] = None):
        batch.output_ids = result.next_token_ids
        next_token_ids = result.next_token_ids.tolist()
        req_to_send = []

        for req, next_token_id in zip(batch.reqs, next_token_ids):
            req.status = "notneed"
            waiting_req = WaitingReq(rid=req.rid,new_token_ids=[next_token_id],sampling_params=None)
            req_to_send.append(waiting_req)
        if rank == 0:
            try:
                outbound_queue.put_nowait(req_to_send)
            except Exception:
                # Fallback to blocking put if needed
                try:
                    outbound_queue.put(req_to_send)
                except Exception:
                    pass
    
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