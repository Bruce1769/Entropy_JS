import uuid
import torch
import time
from collections import deque
import zmq
import pickle
from tqdm import tqdm
from typing import Optional, Tuple, Union, List, Dict
import multiprocessing as mp
import torch.distributed as dist
import atexit
import signal
import os
import threading
import queue
import sys
import json
from multiprocessing import Value
import nvtx

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
from sglang.srt.utils import broadcast_pyobj, point_to_point_pyobj

from r2r.models.recorder import GenerationRecord, GenerationRecorder
from r2r.utils.config import (
    QUICK_COLOR,
    REFERENCE_COLOR,
    RESET,
)
from r2r.utils.switching import (
    _leftmost_argmax_index,
    append_entropy_lookahead_score_log,
    create_switching_strategy,
    EntropyLookaheadSwitching,
    EntropyVarianceJsSwitching,
    SlidingWindowEntropyJsSwitching,
    SlidingWindowEntropySwitching,
)
from r2r.utils.token_manager import SGLangTokenManager
from r2r.utils.dataclass import ModelOutputs
from r2r.utils.sampling import sample_token
from r2r.utils.metrics import (
    compute_entropy,
    compute_js_divergence_logits,
    compute_js_divergence_topk_union,
    log_prob_of_token,
)
from r2r.models.sglang_patch.schedule_req import (
    WaitingReq,
    SimpleSamplingParams,
    EntropyLookaheadRpc,
    EntropyLookaheadResp,
    NextTokenJsRpc,
    NextTokenJsAbortRpc,
    NextTokenJsResp,
    SlidingWindowJsRpc,
    SlidingWindowJsResp,
)

_el_rpc_fail_last_log = 0.0
_el_rpc_fail_pending = 0

_AGENT_DEBUG_LOG_PATH = "/remote-home/pxl/.cursor/debug-7a73bd.log"
_AGENT_DEBUG_SESSION_ID = "7a73bd"


def _agent_debug_log(run_id: str, hypothesis_id: str, location: str, message: str, data: dict) -> None:
    try:
        payload = {
            "sessionId": _AGENT_DEBUG_SESSION_ID,
            "runId": run_id,
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
        }
        with open(_AGENT_DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _log_entropy_lookahead_rpc_fail_throttled(err: object, seq_i: int) -> None:
    """Avoid MB-sized logs when LLM OOMs every token."""
    global _el_rpc_fail_last_log, _el_rpc_fail_pending
    now = time.monotonic()
    _el_rpc_fail_pending += 1
    elapsed = now - _el_rpc_fail_last_log
    if _el_rpc_fail_last_log != 0.0 and elapsed < 15.0:
        return
    es = str(err).replace("\n", " ")[:180]
    print(
        f"[SLMServer] entropy lookahead LLM logprob RPC failed x{_el_rpc_fail_pending} "
        f"in ~{max(elapsed, 0.001):.1f}s, seq={seq_i}: {es}; falling back to entropy routing"
    )
    _el_rpc_fail_pending = 0
    _el_rpc_fail_last_log = now


def _tps_log_interval_s() -> float:
    """Wall-clock spacing for throughput lines; override with R2R_TPS_LOG_INTERVAL (seconds, min 0.25)."""
    try:
        return max(0.25, float(os.environ.get("R2R_TPS_LOG_INTERVAL", "4.0")))
    except ValueError:
        return 4.0


def _perf_probe_enabled() -> bool:
    return os.environ.get("R2R_PERF_PROBE", "").strip().lower() in ("1", "true", "yes", "on")


def _log_entropy_sum_enabled() -> bool:
    return os.environ.get("R2R_LOG_ENTROPY_SUM", "").strip().lower() in ("1", "true", "yes", "on")


def _maybe_log_entropy_sum(
    req: Req,
    entropy_path: list,
    entropy_path_sum: float,
    threshold: float,
    triggered: bool,
    window_full: bool = True,
    *,
    path_mean: Optional[float] = None,
    threshold_mean: Optional[float] = None,
) -> None:
    """实时打印滑动窗口熵（每步 decode）；可用 R2R_ENTROPY_SUM_MIN_INTERVAL_S 限频。
    When ``threshold_mean`` is set, trigger uses mean vs that threshold; log line includes mean."""
    try:
        min_iv = float(os.environ.get("R2R_ENTROPY_SUM_MIN_INTERVAL_S", "0"))
    except ValueError:
        min_iv = 0.0
    min_iv = max(0.0, min_iv)
    now = time.perf_counter()
    last = getattr(req, "_r2r_entropy_sum_log_t", None)
    if last is not None and (now - last) < min_iv:
        return
    setattr(req, "_r2r_entropy_sum_log_t", now)
    rid = getattr(req, "rid", "?")
    path_s = "[" + ", ".join(f"{x:.3f}" for x in entropy_path) + "]"
    full_tag = "full" if window_full else "partial"
    if threshold_mean is not None and path_mean is not None:
        print(
            f"[entropy_window] rid={rid} {full_tag} path_mean={path_mean:.4f} thr_mean={threshold_mean:.4f} "
            f"path_sum={entropy_path_sum:.4f} path={path_s} triggered={triggered}",
            flush=True,
        )
    else:
        print(
            f"[entropy_sum] rid={rid} {full_tag} path_sum={entropy_path_sum:.4f} thr={threshold} "
            f"path={path_s} triggered={triggered}",
            flush=True,
        )


def _log_current_tokens_per_second(
    scheduler: Scheduler,
    rank: int,
    source: str,
    token_count: int,
    interval_s: Optional[float] = None,
) -> None:
    if interval_s is None:
        interval_s = _tps_log_interval_s()
    if rank != 0 or token_count <= 0:
        return
    now = time.perf_counter()
    scheduler.current_tps_total_tokens = getattr(scheduler, "current_tps_total_tokens", 0) + int(token_count)
    scheduler.current_tps_slm_tokens = getattr(scheduler, "current_tps_slm_tokens", 0)
    scheduler.current_tps_llm_tokens = getattr(scheduler, "current_tps_llm_tokens", 0)
    if source == "slm":
        scheduler.current_tps_slm_tokens += int(token_count)
    elif source == "llm":
        scheduler.current_tps_llm_tokens += int(token_count)
    last = getattr(scheduler, "current_tps_last_time", None)
    if last is None:
        scheduler.current_tps_last_time = now
        return
    elapsed = now - last
    if elapsed < interval_s:
        return
    total = getattr(scheduler, "current_tps_total_tokens", 0)
    slm = getattr(scheduler, "current_tps_slm_tokens", 0)
    llm = getattr(scheduler, "current_tps_llm_tokens", 0)
    tps = total / max(elapsed, 1e-6)
    print(
        f"[current tokens/s] total={tps:.2f} "
        f"(slm_tokens={slm}, llm_tokens={llm}, window={elapsed:.1f}s)",
        flush=True,
    )
    scheduler.current_tps_last_time = now
    scheduler.current_tps_total_tokens = 0
    scheduler.current_tps_slm_tokens = 0
    scheduler.current_tps_llm_tokens = 0


class SLMServer:
    """SLM Server launched by SGLang"""

    @staticmethod
    def tree_cache_finished_req_safe(scheduler: Scheduler, req: Req) -> None:
        """Finish request KV lifecycle without crashing when ``req.last_node`` is None.

        ``ChunkCache.match_prefix`` returns ``last_device_node=None``. Radix
        ``cache_finished_req`` still calls ``dec_lock_ref(req.last_node)``, which
        raises ``AttributeError`` if the node was never set.
        """
        tc = scheduler.tree_cache
        if getattr(tc, "disable", False):
            tc.cache_finished_req(req)
            return
        if getattr(req, "last_node", None) is None:
            if req.req_pool_idx is None:
                return
            tok_len = len(req.origin_input_ids) + len(req.output_ids)
            if tok_len <= 0:
                scheduler.req_to_token_pool.free(req.req_pool_idx)
                return
            kv_indices = scheduler.req_to_token_pool.req_to_token[
                req.req_pool_idx, : tok_len - 1
            ]
            scheduler.token_to_kv_pool_allocator.free(kv_indices)
            scheduler.req_to_token_pool.free(req.req_pool_idx)
            return
        tc.cache_finished_req(req)

    def __init__(
        self,
        model_config: Dict,
        quick_sglang_kwargs: Dict,
        quick_num_gpus: int,
        req_port: int,
        ready_queue: Optional[mp.Queue] = None,
        switching_strategy: str = "neural",
        strategy_kwargs: Dict = {},
        mem_fraction_static: Optional[float] = None,
        llm_kvcache_size: Optional[Value] = None,
        master_port: Optional[int] = None,
        entropy_lookahead_query_queue: Optional[mp.Queue] = None,
        entropy_lookahead_reply_queue: Optional[mp.Queue] = None,
    ):
        self.quick_waiting_line = []
        self.is_reset_cache = False
        self.shutdown_loop = False
        self.batch = None
        self.new_reqs = []
        self.model_config = model_config
        self.quick_sglang_kwargs = quick_sglang_kwargs
        self.quick_num_gpus = quick_num_gpus
        self.req_port = req_port
        self.ready_queue = ready_queue
        self.switching_strategy = switching_strategy
        self.strategy_kwargs = strategy_kwargs
        self.master_port = master_port
        self.entropy_lookahead_query_queue = entropy_lookahead_query_queue
        self.entropy_lookahead_reply_queue = entropy_lookahead_reply_queue
        # Inter-server queues (outbound to LLM / inbound from LLM)
        self.queue_to_llm = mp.Queue()
        # Dedicated outbound sequence counter for messages sent to LLM
        # Per-rank inbound queues (LLM -> SLM); each rank consumes its own queue
        self._inbound_queues = mp.Queue()

        # ZMQ context (used for PUB/SUB sockets)
        self._ctx = zmq.Context.instance()

        # ================== New: PUB to system (SLM -> System) for finished reqs ==================
        # Create a dedicated mp.Queue to gather finished requests from workers,
        # and a thread to publish them over ZMQ in real time. Bind and expose port early,
        # before sending tokenizer back to controller.
        self._finished_reqs_queue: mp.Queue = mp.Queue()
        try:
            self._pub_finished = self._ctx.socket(zmq.PUB)
            self._pub_finished.setsockopt(zmq.LINGER, 0)
            self._pub_finished.setsockopt(zmq.SNDHWM, 100000)
            self.system_receive_port = self._pub_finished.bind_to_random_port("tcp://127.0.0.1")
        except Exception as e:
            print(f"[SLMServer] Failed to bind PUB to system: {e}")
            self._pub_finished = None
            self.system_receive_port = None
        # Sender thread for finished reqs
        self._send_finished_stop = threading.Event()
        def _send_finished_loop():
            time.sleep(0.05)
            while not self._send_finished_stop.is_set():
                try:
                    item = self._finished_reqs_queue.get(timeout=0.1)
                except Exception:
                    continue
                if item is None:
                    break
                if self._pub_finished is None:
                    continue
                try:
                    self._pub_finished.send_pyobj(item)
                    if isinstance(item, dict) and item.get("status") == "finished":
                        #region agent log
                        _agent_debug_log(
                            run_id=os.environ.get("R2R_DEBUG_RUN_ID", "run-unknown"),
                            hypothesis_id="H2",
                            location="slm_server.py:_send_finished_loop",
                            message="published_finished_event",
                            data={"rid": item.get("rid"), "status": item.get("status")},
                        )
                        #endregion
                except Exception:
                    pass
        self._send_finished_thread = threading.Thread(target=_send_finished_loop, daemon=True)
        self._send_finished_thread.start()

        # ================== Inter-server PUB (SLM -> LLM) BEFORE workers ==================
        try:
            self._pub_llm = self._ctx.socket(zmq.PUB)
            self._pub_llm.setsockopt(zmq.LINGER, 0)
            self._pub_llm.setsockopt(zmq.SNDHWM, 100000)
            # Bind now (B1) and expose port so controller can start LLM SUB later
            self.llm_recv_port = self._pub_llm.bind_to_random_port("tcp://127.0.0.1")
        except Exception as e:
            print(f"[SLMServer] Failed to bind PUB to LLM (early): {e}")
            self._pub_llm = None
            self.llm_recv_port = None

        # Send thread: pulls objects from queue_to_llm (start early so ready when workers spawn)
        self._send_llm_stop = threading.Event()
        def _send_llm_loop():
            time.sleep(0.05)
            while not self._send_llm_stop.is_set():
                try:
                    item = self.queue_to_llm.get(timeout=0.1)
                except Exception:
                    continue
                if item is None:
                    break
                if self._pub_llm is None:
                    continue
                try:
                    self._pub_llm.send_pyobj(item)
                except Exception:
                    pass
        self._send_llm_thread = threading.Thread(target=_send_llm_loop, daemon=True)
        self._send_llm_thread.start()

        # Placeholder for later SUB (LLM -> SLM) started by controller
        self._sub_from_llm = None
        self._recv_from_llm_thread = None
        self._recv_from_llm_stop = threading.Event()

        print(f"Loading quick model {self.model_config['quick']['model_name']}...")
        # readiness queue
        self.ready_queue = ready_queue

        # Register atexit & signal handlers for safe shutdown like LLMServer
        atexit.register(self.shutdown)
        def _sig_handler(sig, frame):
            try:
                self.shutdown()
            finally:
                os._exit(0)
        try:
            signal.signal(signal.SIGINT, _sig_handler)
            signal.signal(signal.SIGTERM, _sig_handler)
        except Exception:
            pass

        _quick_kw = dict(quick_sglang_kwargs or {})
        _quick_kw.pop("disable_radix_cache", None)
        quick_server_args = ServerArgs(
            model_path=self.model_config["quick"]["model_path"],
            disable_cuda_graph=False,
            disable_overlap_schedule=True,
            # ChunkCache: avoids RadixCache.cache_finished_req + dec_lock_ref(req.last_node)
            # when last_node is unset on hybrid/disagg finish paths.
            disable_radix_cache=True,
            mem_fraction_static=mem_fraction_static,
            **_quick_kw,
        )
        quick_server_args.tp_size = quick_num_gpus
        
        router_args = self.model_config.get("router", {})
        override_init_args = router_args.get("override_init_args", {})
        self.strategy_kwargs["override_init_args"] = override_init_args
        
        self.quick_model_procs = []
        for rank in range(quick_num_gpus):
            proc = mp.Process(
                target=self.quick_model_worker,
                args=(
                    rank, 
                    quick_num_gpus,
                    quick_server_args,
                    self.ready_queue,
                    self.switching_strategy,
                    self.strategy_kwargs,
                    self._inbound_queues,  # per-rank inbound msgs
                    self.queue_to_llm,    # outbound (for potential direct worker usage),
                    self._finished_reqs_queue,  # finished reqs back to system
                    req_port if rank == 0 else None,  # system SUB only on rank 0
                    llm_kvcache_size,
                    self.master_port,  # Pass master_port to worker
                    self.entropy_lookahead_query_queue,
                    self.entropy_lookahead_reply_queue,
                ),
            )
            proc.start()
            self.quick_model_procs.append(proc)

    
    def process_new_requests(reqs: List[Req], scheduler: Scheduler, rank: int, outbound_queue: Optional[mp.Queue] = None):
        if len(reqs) == 0:
            return
        for req in reqs:
            if req.status in ("SHUTDOWN", "RESET_CACHE"):
                if req.status == "SHUTDOWN":
                    if rank == 0:
                        outbound_queue.put_nowait([WaitingReq(status="SHUTDOWN",)])
                    return False
                elif req.status == "RESET_CACHE":
                    ok = scheduler.flush_cache()
                    print(f"[quick rank{scheduler.gpu_id}] Cache reset: {ok}")
                    if rank == 0:
                        outbound_queue.put_nowait([WaitingReq(rid=str(-1), new_token_ids=[], status="RESET_CACHE",)])
                    continue

            req.eos_token_ids = scheduler.model_config.hf_eos_token_id
            req.vocab_size = scheduler.model_config.vocab_size
            scheduler.waiting_queue.append(req)
        
        return True

    @staticmethod
    def quick_model_worker(
        rank,
        world_size: int,
        server_args: ServerArgs,
        ready_queue: Optional[mp.Queue] = None,
        switching_strategy: str = "neural",
        strategy_kwargs: Dict = {},
        inbound_queue: Optional[mp.Queue] = None,
        outbound_queue: Optional[mp.Queue] = None,
        finished_queue: Optional[mp.Queue] = None,
        req_port: Optional[int] = None,
        llm_kvcache_size: Optional[Value] = None,
        master_port: Optional[int] = None,
        entropy_lookahead_query_queue: Optional[mp.Queue] = None,
        entropy_lookahead_reply_queue: Optional[mp.Queue] = None,
    ):
        # Register signal handler to ensure finally block execution on terminate
        def _worker_sig_handler(signum, frame):
            sys.exit(0)
        signal.signal(signal.SIGTERM, _worker_sig_handler)
        
        # Set MASTER_ADDR and MASTER_PORT inside worker
        os.environ["MASTER_ADDR"] = "localhost"
        if master_port is not None:
            os.environ["MASTER_PORT"] = str(master_port)
        elif "MASTER_PORT" not in os.environ:
            raise RuntimeError("MASTER_PORT must be provided or set in environment")
        
        dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)

        port_args = PortArgs.init_new(server_args)
        scheduler = Scheduler(
            server_args=server_args,
            port_args=port_args,
            gpu_id=rank,
            tp_rank=rank,
            dp_rank=0,
            moe_ep_rank=0,
            pp_rank=0, # Pipeline parallelism is not Supported
            llm_kvcache_size=llm_kvcache_size,
        )
        # Setup system SUB socket on rank 0
        if req_port is not None:
            ctx = zmq.Context.instance()
            sub_socket = ctx.socket(zmq.SUB)
            sub_socket.setsockopt(zmq.LINGER, 0)
            sub_socket.connect(f"tcp://127.0.0.1:{req_port}")
            sub_socket.setsockopt(zmq.SUBSCRIBE, b"")
            poller = zmq.Poller()
            poller.register(sub_socket, zmq.POLLIN)
            scheduler.receive_from_system = sub_socket
        else:
            scheduler.receive_from_system = None

        # Initialize switching strategy
        router = create_switching_strategy(switching_strategy, **strategy_kwargs)

        # Notify readiness after subscription
        if ready_queue is not None:
            try:
                ready_queue.put(("READY", rank, scheduler.tokenizer if rank == 0 else None))
            except Exception as e:
                print(f"SLMServer Failed to send tokenizer from rank {rank}: {e}")

        print(f"Quick model worker {rank} started, waiting for requests...")

        SLMServer.init_batch_not_need(scheduler)
        pbar_dict = {}

        # event_loop
        try:
            while True:
                init_nvtx = False
                if inbound_queue is not None: # Process message from LLM
                    llm_reqs = SLMServer.recv_reqs_from_llm(
                        inbound_queue=inbound_queue,
                        scheduler=scheduler,
                    )
                    if llm_reqs:
                        nvtx.push_range("SLM")
                        init_nvtx = True
                        SLMServer.process_result_from_llm(rank, scheduler, llm_reqs, finished_queue, outbound_queue)
                
                recv_reqs = SLMServer.recv_requests(scheduler)
                if (
                    rank == 0
                    and recv_reqs
                    and os.environ.get("R2R_TRACE_REQ_FLOW", "").strip().lower() in ("1", "true", "yes", "on")
                ):
                    print(
                        f"[req-recv] count={len(recv_reqs)} "
                        f"rids={[getattr(r, 'rid', None) for r in recv_reqs[:8]]}",
                        flush=True,
                    )
                if rank == 0 and recv_reqs:
                    #region agent log
                    _agent_debug_log(
                        run_id=os.environ.get("R2R_DEBUG_RUN_ID", "run-unknown"),
                        hypothesis_id="H5",
                        location="slm_server.py:quick_model_worker_recv",
                        message="received_reqs_from_system",
                        data={
                            "num_reqs": len(recv_reqs),
                            "rids": [getattr(r, "rid", None) for r in recv_reqs[:8]],
                            "statuses": [getattr(r, "status", None) for r in recv_reqs[:8]],
                        },
                    )
                    #endregion
                ok = SLMServer.process_new_requests(recv_reqs, scheduler, rank, outbound_queue)
                if rank == 0 and recv_reqs:
                    #region agent log
                    _agent_debug_log(
                        run_id=os.environ.get("R2R_DEBUG_RUN_ID", "run-unknown"),
                        hypothesis_id="H5",
                        location="slm_server.py:quick_model_worker_post_process_new_requests",
                        message="requests_enqueued_to_waiting_queue",
                        data={
                            "recv_count": len(recv_reqs),
                            "waiting_queue_len": len(getattr(scheduler, "waiting_queue", []) or []),
                            "batch_not_need_len": len(getattr(getattr(scheduler, "batch_not_need", None), "reqs", []) or []),
                        },
                    )
                    #endregion
                if ok is False:
                    print(f"[quick rank{rank}] SHUTDOWN received, exiting...")
                    break

                if scheduler.waiting_queue or any(req.status == "need" for req in scheduler.batch_not_need.reqs) or (scheduler.last_batch and any(req.status == "need" for req in scheduler.last_batch.reqs)) and not init_nvtx:
                    nvtx.push_range("SLM")
                batch = scheduler.get_next_batch_to_run()
                if batch:
                    result, req_to_send = SLMServer.run_batch(
                        batch,
                        scheduler,
                        router,
                        entropy_lookahead_query_queue=entropy_lookahead_query_queue,
                        entropy_lookahead_reply_queue=entropy_lookahead_reply_queue,
                        finished_queue=finished_queue,
                        outbound_queue=outbound_queue,
                        rank=rank,
                    )
                    SLMServer.process_batch_results(batch, result, scheduler, finished_queue, outbound_queue, rank, req_to_send)
                    scheduler.last_batch=batch
                    if rank == 0:
                        SLMServer.update_tqdm(pbar_dict, batch, scheduler)
                    nvtx.pop_range()
        except (SystemExit, KeyboardInterrupt):
            pass
        except BaseException as e:
            print(f"[quick rank{rank}] SLM worker fatal error: {e}", flush=True)
            import traceback as _tb
            _tb.print_exc()
        finally:
            try:
                if dist.is_initialized():
                    dist.destroy_process_group()
            except Exception as e:
                print(f"[rank {rank}] destroy_process_group error: {e}")
    
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
        SLMServer.simple_prepare_for_extend(scheduler.batch_not_need)
        scheduler.batch_not_need.multimodal_inputs = []
        scheduler.batch_not_need.output_ids = torch.tensor([], dtype=torch.int64).to(
            scheduler.batch_not_need.device, non_blocking=True
        )
        scheduler.batch_not_need.orig_seq_lens = torch.tensor([], dtype=torch.int64).to(
            scheduler.batch_not_need.device, non_blocking=True
        )
        scheduler.batch_not_need.output_ids = torch.tensor([], dtype=torch.int64).to(
            scheduler.batch_not_need.device, non_blocking=True
        )
    
    @staticmethod
    def recv_requests(scheduler: Scheduler) -> List[Req]:
        """Receive results at tp_rank = 0 and broadcast it to all other TP ranks."""

        if scheduler.pp_rank == 0:
            if scheduler.attn_tp_rank == 0:
                recv_reqs = []

                while True:
                    try:
                        recv_req = scheduler.receive_from_system.recv_pyobj(zmq.NOBLOCK)
                    except zmq.ZMQError:
                        break
                    recv_reqs.append(recv_req)
            else:
                recv_reqs = None
        else:
            if scheduler.attn_tp_rank == 0:
                dp_offset = scheduler.attn_dp_rank * scheduler.attn_tp_size
                recv_reqs = point_to_point_pyobj(
                    [],
                    scheduler.pp_rank * scheduler.tp_size + dp_offset,
                    scheduler.world_group.device_group,
                    (scheduler.pp_rank - 1) * scheduler.tp_size + dp_offset,
                    scheduler.pp_rank * scheduler.tp_size + dp_offset,
                )
            else:
                recv_reqs = None

        if scheduler.input_blocker is not None:
            recv_reqs = scheduler.input_blocker.handle(recv_reqs)

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

    @staticmethod
    def process_result_from_llm(rank: int, scheduler: Scheduler, commit_msgs, finished_queue: Optional[mp.Queue] = None, outbound_queue: Optional[mp.Queue] = None):
        if rank == 0 and hasattr(scheduler, "n_generated_tokens"):
            scheduler.n_generated_tokens += len(commit_msgs)
        better_token_ids = {}
        returned_rid_list = []
        for waiting_req in commit_msgs:
            better_token_ids[waiting_req.rid] = waiting_req.new_token_ids[-1]
            returned_rid_list.append(waiting_req.rid)
        keep_indices = []
        not_keep_indices = []
        finished_reqs = []
        if scheduler.batch_not_need is not None:
            if scheduler.last_batch is None:
                return
            else:
                scheduler.last_batch.merge_batch(scheduler.batch_not_need)
            output_ids_list = []
            for i, req in enumerate(scheduler.last_batch.reqs):
                if req.rid in returned_rid_list:
                    if better_token_ids[req.rid] in scheduler.model_config.hf_eos_token_id:
                        scheduler.abort_request(AbortReq(req.rid))
                    req.output_ids.append(better_token_ids[req.rid])
                    req.llm_token_count = getattr(req, 'llm_token_count', 0) + 1
                    llm_reason = getattr(req, 'current_llm_reason', None)
                    if llm_reason == 'window_followup' and getattr(req, 'forced_llm_window_remaining', 0) > 0:
                        req.forced_llm_window_remaining -= 1
                    req.current_llm_reason = None
                    req.status = "need"
                    req.check_finished()
                    if req.finished():
                        SLMServer.tree_cache_finished_req_safe(scheduler, req)
                        finished_reqs.append(req)
                    keep_indices.append(i)
                elif req.status == "need":
                    keep_indices.append(i)
                output_ids_list.append(req.output_ids[-1] if req.output_ids else 1)
            
            scheduler.last_batch.output_ids = torch.tensor(output_ids_list, dtype=torch.int64).to(
                scheduler.last_batch.device, non_blocking=True
            )
            
            scheduler.last_batch.filter_batch(keep_indices=keep_indices)
            
            for i, req in enumerate(scheduler.batch_not_need.reqs):
                if req.status == "notneed" and req.rid not in returned_rid_list:
                    not_keep_indices.append(i)
            scheduler.batch_not_need.filter_batch(keep_indices=not_keep_indices)
            if finished_reqs and rank == 0:
                SLMServer.process_finished_requests(finished_reqs, scheduler.tokenizer, finished_queue, outbound_queue)
            if finished_reqs:
                for req in finished_reqs:
                    scheduler.issued_reqs.remove(req)
        _log_current_tokens_per_second(
            scheduler=scheduler,
            rank=rank,
            source="llm",
            token_count=len(returned_rid_list),
        )
    
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
        # KV indices must come from the real allocator (a running counter overflows the pool).
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

    @staticmethod
    def recv_reqs_from_llm(inbound_queue: Optional[mp.Queue], scheduler: Scheduler):

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

    @staticmethod
    def _slm_one_off_forward_logits(scheduler: Scheduler, context_ids: List[int]) -> torch.Tensor:
        """Full-sequence extend on a throwaway Req; returns next-token logits (vocab,)."""
        rid = f"_ela_{uuid.uuid4()}"
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
        SLMServer.simple_prepare_for_extend(new_batch)
        model_batch = new_batch.get_model_worker_batch()
        kv_locs = new_batch.out_cache_loc
        try:
            result = scheduler.tp_worker.forward_batch_generation(model_batch)
            logits_output = result[0] if isinstance(result, tuple) else result.logits_output
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
    def _sliding_entropy_deque_for_req(req: Req, maxlen: int) -> deque:
        dq = getattr(req, "_sliding_entropy_deque", None)
        if dq is None or dq.maxlen != maxlen:
            dq = deque(maxlen=maxlen)
            req._sliding_entropy_deque = dq
        return dq

    @staticmethod
    def _sliding_slm_logits_deque_for_req(req: Req, maxlen: int) -> deque:
        dq = getattr(req, "_sliding_slm_logits_deque", None)
        if dq is None or dq.maxlen != maxlen:
            dq = deque(maxlen=maxlen)
            req._sliding_slm_logits_deque = dq
        return dq

    @staticmethod
    def _sliding_window_truncate_slm_suffix(req: Req, router: SlidingWindowEntropySwitching, window_offset_k: int) -> None:
        """Drop SLM-generated tokens from the intervention index onward (prefix kept).

        Before the current decode append, ``output_ids[last_llm_loc:]`` has length L;
        the deque holds N entropies for the next-token positions ending at index L.
        ``window_offset_k`` is the index within the deque (0 = oldest in window) of the
        first token to replace—**leftmost index of maximum entropy** in the window.
        Keep ``L - N + 1 + window_offset_k`` tokens of the SLM suffix.
        """
        n = router.window_size
        ll = int(req.last_llm_loc) if req.last_llm_loc is not None else 0
        slm_seg = req.output_ids[ll:]
        L = len(slm_seg)
        if L == 0:
            return
        keep = L - n + 1 + int(window_offset_k)
        if keep < 0:
            keep = 0
        if keep > L:
            keep = L
        new_len = ll + keep
        if new_len < len(req.output_ids):
            del req.output_ids[new_len:]

    @staticmethod
    def _sliding_window_truncate_entire_window(req: Req, router: SlidingWindowEntropySwitching) -> None:
        """Drop the last N SLM-generated tokens (the full rolling window span).

        After this, LLM will fill in N new tokens (via N decode steps) before SLM resumes;
        the entropy deque is empty so the window slides forward with subsequent SLM steps.
        """
        n = router.window_size
        ll = int(req.last_llm_loc) if req.last_llm_loc is not None else 0
        slm_seg = req.output_ids[ll:]
        L = len(slm_seg)
        if L == 0:
            return
        keep = max(0, L - n)
        new_len = ll + keep
        if new_len < len(req.output_ids):
            del req.output_ids[new_len:]

    @staticmethod
    def _sliding_window_entropy_choices(
        router: SlidingWindowEntropySwitching,
        batch: ScheduleBatch,
        logits: torch.Tensor,
        next_token_ids: torch.Tensor,
        scheduler: Scheduler,
    ) -> torch.Tensor:
        """Main-path SLM entropies only; sliding deque of size N; scheme B: no threshold until full."""
        _ = (scheduler, next_token_ids)  # signature matches entropy lookahead hook
        device = logits.device
        batch_size = logits.shape[0]
        choices = torch.zeros(batch_size, dtype=torch.int, device=device)
        n = router.window_size
        for i in range(batch_size):
            req = batch.reqs[i]
            if not hasattr(req, "forced_llm_window_remaining"):
                req.forced_llm_window_remaining = 0
            if not hasattr(req, "current_llm_reason"):
                req.current_llm_reason = None
            if req.forced_llm_window_remaining > 0:
                req.current_llm_reason = "window_followup"
                choices[i] = 1
                continue

            L0 = logits[i, 0, :]
            H = float(compute_entropy(L0.unsqueeze(0)))
            dq = SLMServer._sliding_entropy_deque_for_req(req, n)
            dq.append(H)

            if len(dq) < n:
                choices[i] = 0
                req.current_llm_reason = None
                if _log_entropy_sum_enabled():
                    partial = list(dq)
                    ps = float(sum(partial))
                    pm = ps / len(partial) if partial else 0.0
                    _maybe_log_entropy_sum(
                        req=req,
                        entropy_path=partial,
                        entropy_path_sum=ps,
                        threshold=float(router.entropy_sum_threshold),
                        triggered=False,
                        window_full=False,
                        path_mean=pm,
                        threshold_mean=float(router.entropy_mean_threshold),
                    )
                continue

            vals = list(dq)
            entropy_path_sum = float(sum(vals))
            path_mean = entropy_path_sum / float(n)
            triggered = path_mean > float(router.entropy_mean_threshold)
            truncate_k = _leftmost_argmax_index(vals)

            if _log_entropy_sum_enabled():
                _maybe_log_entropy_sum(
                    req=req,
                    entropy_path=vals,
                    entropy_path_sum=entropy_path_sum,
                    threshold=float(router.entropy_sum_threshold),
                    triggered=triggered,
                    window_full=True,
                    path_mean=path_mean,
                    threshold_mean=float(router.entropy_mean_threshold),
                )
            choices[i] = 1 if triggered else 0
            if triggered:
                req.current_llm_reason = "window_trigger"
                entropy_path = list(dq)
                # Entropy at the chosen truncate anchor (leftmost argmax in window).
                h_at_truncate = float(vals[truncate_k])
                mode = router.intervention_mode
                if mode == "replace_full_window":
                    req._sliding_truncate_argmax_k = None
                    req._sliding_full_window_truncate_once = True
                elif getattr(router, "truncate_on_llm_trigger", False):
                    req._sliding_truncate_argmax_k = int(truncate_k)
                else:
                    req._sliding_truncate_argmax_k = None
                dq.clear()
                if mode in ("replace_window", "replace_full_window"):
                    req.forced_llm_window_remaining = max(0, n - 1)
                do_trunc = bool(getattr(router, "truncate_on_llm_trigger", False))
                rid_log = getattr(req, "rid", "?")
                if mode == "replace_full_window":
                    print(
                        f"[entropy_truncate_full_window] rid={rid_log} window_size={n} "
                        f"path_mean={float(path_mean):.6f} thr_mean={float(router.entropy_mean_threshold):.6f}",
                        flush=True,
                    )
                elif do_trunc:
                    print(
                        f"[entropy_truncate] rid={rid_log} truncate_k={int(truncate_k)} "
                        f"H_at_truncate={h_at_truncate:.6f} path_mean={float(path_mean):.6f}",
                        flush=True,
                    )
                append_entropy_lookahead_score_log(
                    router.score_log_path,
                    {
                        "event": "sliding_window_entropy_triggered",
                        "rid": getattr(req, "rid", None),
                        "seq_in_batch": i,
                        "window_size": n,
                        "entropy_path": entropy_path,
                        "entropy_path_sum": entropy_path_sum,
                        "entropy_path_mean": float(path_mean),
                        "entropy_mean_threshold": float(router.entropy_mean_threshold),
                        "entropy_sum_threshold_yaml": float(router.entropy_sum_threshold),
                        "intervention_mode": router.intervention_mode,
                        "truncate_on_llm_trigger": bool(
                            getattr(router, "truncate_on_llm_trigger", False)
                        ),
                        "truncate_window_offset_k": int(truncate_k),
                        "truncate_position_entropy": h_at_truncate,
                        "truncate_mode": (
                            "full_window"
                            if mode == "replace_full_window"
                            else ("argmax" if do_trunc else "none")
                        ),
                        "routed_to_llm": True,
                        "main_path_only": True,
                    },
                )
            else:
                req.current_llm_reason = None
        router.state.last_model = "reference" if choices.any().item() else "quick"
        return choices

    @staticmethod
    def _sliding_window_entropy_js_choices(
        router: SlidingWindowEntropyJsSwitching,
        batch: ScheduleBatch,
        logits: torch.Tensor,
        next_token_ids: torch.Tensor,
        scheduler: Scheduler,
        query_queue: mp.Queue,
        reply_queue: mp.Queue,
    ) -> torch.Tensor:
        """Mean entropy gate on the window, then JS(SLM||LLM) left-to-right (one LLM prefill for all n logits)."""
        _ = (scheduler, next_token_ids)
        device = logits.device
        batch_size = logits.shape[0]
        choices = torch.zeros(batch_size, dtype=torch.int, device=device)
        n = router.window_size
        for i in range(batch_size):
            req = batch.reqs[i]
            if not hasattr(req, "forced_llm_window_remaining"):
                req.forced_llm_window_remaining = 0
            if not hasattr(req, "current_llm_reason"):
                req.current_llm_reason = None
            if req.forced_llm_window_remaining > 0:
                req.current_llm_reason = "window_followup"
                choices[i] = 1
                continue

            L0 = logits[i, 0, :]
            H = float(compute_entropy(L0.unsqueeze(0)))
            dq = SLMServer._sliding_entropy_deque_for_req(req, n)
            lq = SLMServer._sliding_slm_logits_deque_for_req(req, n)
            dq.append(H)
            lq.append(L0.detach().cpu().float().clone())

            if len(dq) < n:
                choices[i] = 0
                req.current_llm_reason = None
                continue

            vals = list(dq)
            entropy_path_sum = float(sum(vals))
            path_mean = entropy_path_sum / float(n)
            if path_mean <= float(router.entropy_mean_threshold):
                choices[i] = 0
                req.current_llm_reason = None
                continue

            ll = int(req.last_llm_loc) if req.last_llm_loc is not None else 0
            base = list(req.origin_input_ids) + list(req.output_ids[:ll])
            slm_seg = list(req.output_ids[ll:])
            L = len(slm_seg)
            if L < n:
                choices[i] = 0
                req.current_llm_reason = None
                continue

            full_ids = base + slm_seg[:L]
            lq_list = list(lq)
            js_vals = []
            first_hit = None
            rpc_failed = False
            err_msg = ""
            qid = int(time.time_ns() % (1 << 62))
            rpc = SlidingWindowJsRpc(
                query_id=qid,
                full_ids=full_ids,
                base_len=len(base),
                window_size=n,
            )
            query_queue.put(rpc)
            try:
                resp = reply_queue.get(timeout=router.rpc_timeout_s)
            except Exception as e:
                err_msg = str(e)
                resp = None
            bad = (
                resp is None
                or not isinstance(resp, SlidingWindowJsResp)
                or not resp.ok
                or len(getattr(resp, "llm_logits", []) or []) < n
            )
            if bad:
                if not err_msg and resp is not None:
                    err_msg = str(getattr(resp, "error", "") or "bad_resp")
                elif not err_msg:
                    err_msg = "timeout_or_bad_resp"
                rpc_failed = True
            else:
                for j in range(n):
                    llm_t = torch.as_tensor(resp.llm_logits[j]).float()
                    if llm_t.dim() == 2:
                        llm_t = llm_t.squeeze(0)
                    js = compute_js_divergence_logits(
                        torch.as_tensor(lq_list[j]).float(), llm_t
                    )
                    js_vals.append(js)
                    if js > float(router.js_threshold):
                        first_hit = j
                        break

            if rpc_failed:
                choices[i] = 1
                req.current_llm_reason = "window_trigger_js_rpc_fail"
                truncate_k = 0
                mode = router.intervention_mode
                if mode == "replace_full_window":
                    req._sliding_truncate_argmax_k = None
                    req._sliding_full_window_truncate_once = True
                else:
                    req._sliding_truncate_argmax_k = int(truncate_k)
                dq.clear()
                lq.clear()
                if mode in ("replace_window", "replace_full_window"):
                    req.forced_llm_window_remaining = max(0, n - 1)
                append_entropy_lookahead_score_log(
                    router.score_log_path,
                    {
                        "event": "sliding_window_entropy_js_triggered",
                        "rid": getattr(req, "rid", None),
                        "seq_in_batch": i,
                        "rpc_ok": False,
                        "error": err_msg,
                        "entropy_path_mean": float(path_mean),
                    },
                )
                continue

            if first_hit is None:
                first_hit = max(range(n), key=lambda j: js_vals[j])

            truncate_k = int(first_hit)
            choices[i] = 1
            req.current_llm_reason = "window_trigger_js"
            mode = router.intervention_mode
            if mode == "replace_full_window":
                req._sliding_truncate_argmax_k = None
                req._sliding_full_window_truncate_once = True
            else:
                req._sliding_truncate_argmax_k = int(truncate_k)
            dq.clear()
            lq.clear()
            if mode in ("replace_window", "replace_full_window"):
                req.forced_llm_window_remaining = max(0, n - 1)

            # First LLM token can be sampled from JS prefill logits (no duplicate prefill for that step).
            reuse_idx = 0 if mode == "replace_full_window" else int(truncate_k)
            req._reuse_llm_first_logits = torch.as_tensor(
                resp.llm_logits[reuse_idx], dtype=torch.float32
            ).cpu().clone()

            rid_log = getattr(req, "rid", "?")
            print(
                f"[entropy_js] rid={rid_log} path_mean={float(path_mean):.6f} "
                f"js_vals={[round(x, 5) for x in js_vals]} truncate_k={truncate_k} "
                f"js_threshold={float(router.js_threshold):.6f}",
                flush=True,
            )
            append_entropy_lookahead_score_log(
                router.score_log_path,
                {
                    "event": "sliding_window_entropy_js_triggered",
                    "rid": getattr(req, "rid", None),
                    "seq_in_batch": i,
                    "window_size": n,
                    "entropy_path": vals,
                    "entropy_path_mean": float(path_mean),
                    "entropy_mean_threshold": float(router.entropy_mean_threshold),
                    "js_path": [float(x) for x in js_vals],
                    "js_llm_forwards": 0 if rpc_failed else int(n),
                    "js_threshold": float(router.js_threshold),
                    "truncate_window_offset_k": truncate_k,
                    "intervention_mode": router.intervention_mode,
                    "rpc_ok": True,
                    "routed_to_llm": True,
                    "reuse_first_llm_token_from_js_prefill": True,
                },
            )

        router.state.last_model = "reference" if choices.any().item() else "quick"
        return choices

    @staticmethod
    def _entropy_variance_js_choices(
        router: EntropyVarianceJsSwitching,
        batch: ScheduleBatch,
        logits: torch.Tensor,
        next_token_ids: torch.Tensor,
        scheduler: Scheduler,
        query_queue: mp.Queue,
        reply_queue: mp.Queue,
    ) -> torch.Tensor:
        """Entropy gate (H > threshold), then one LLM forward for next-token logits; JS > threshold -> LLM + reuse logits."""
        _ = (scheduler, next_token_ids)
        device = logits.device
        batch_size = logits.shape[0]
        choices = torch.zeros(batch_size, dtype=torch.int, device=device)
        _cooldown = max(0, int(os.environ.get("R2R_EVJS_COOLDOWN", "0")))
        _log_all = os.environ.get("R2R_LOG_EVJS_ALL", "").strip().lower() in ("1", "true", "yes", "on")
        for i in range(batch_size):
            req = batch.reqs[i]
            if not hasattr(req, "current_llm_reason"):
                req.current_llm_reason = None
            L0 = logits[i, 0, :]
            H = float(compute_entropy(L0.unsqueeze(0)))
            if not (H > router.entropy_threshold):
                choices[i] = 0
                req.current_llm_reason = None
                if _log_all:
                    print(
                        f"[evjs-gate] rid={getattr(req, 'rid', '?')} "
                        f"entropy={H:.6f} thr_entropy={float(router.entropy_threshold):.6f} "
                        f"-> gate=SLM",
                        flush=True,
                    )
                continue
            # --- RPC cooldown: skip expensive LLM RPC for N steps after last "stay-on-SLM" ---
            if _cooldown > 0:
                cd_remaining = getattr(req, "_evjs_rpc_cooldown", 0)
                if cd_remaining > 0:
                    req._evjs_rpc_cooldown = cd_remaining - 1
                    choices[i] = 0
                    req.current_llm_reason = None
                    if _log_all:
                        print(
                            f"[evjs-cooldown] rid={getattr(req, 'rid', '?')} "
                            f"entropy={H:.6f} cooldown_remaining={cd_remaining - 1} "
                            f"-> gate=SLM (skipped RPC)",
                            flush=True,
                        )
                    continue
            ctx = list(req.origin_input_ids) + list(req.output_ids)
            qid = int(time.time_ns() % (1 << 62)) + i
            rpc = NextTokenJsRpc(
                query_id=qid,
                context_ids=ctx,
                rid=str(getattr(req, "rid", qid)),
            )
            query_queue.put(rpc)
            err_msg = ""
            try:
                resp = reply_queue.get(timeout=router.rpc_timeout_s)
            except Exception as e:
                resp = None
                err_msg = str(e)
            bad = (
                resp is None
                or not isinstance(resp, NextTokenJsResp)
                or not resp.ok
                or (
                    resp.llm_logits is None
                    and (
                        resp.llm_topk_indices is None
                        or resp.llm_topk_probs is None
                        or resp.llm_tail_mass is None
                    )
                )
            )
            if bad:
                if not err_msg and resp is not None:
                    err_msg = str(getattr(resp, "error", "") or "bad_resp")
                elif not err_msg:
                    err_msg = "timeout_or_bad_resp"
                choices[i] = 0
                req.current_llm_reason = None
                if _cooldown > 0:
                    req._evjs_rpc_cooldown = _cooldown * 2
                try:
                    query_queue.put_nowait(NextTokenJsAbortRpc(rid=str(req.rid)))
                except Exception:
                    pass
                print(
                    f"[evjs-rpc-fail] rid={getattr(req, 'rid', '?')} "
                    f"entropy={H:.6f} thr_entropy={float(router.entropy_threshold):.6f} "
                    f"error={err_msg} cooldown={_cooldown * 2} -> route=SLM",
                    flush=True,
                )
                append_entropy_lookahead_score_log(
                    router.score_log_path,
                    {
                        "event": "entropy_variance_js",
                        "sub": "rpc_fail",
                        "rid": getattr(req, "rid", None),
                        "seq_in_batch": i,
                        "entropy": H,
                        "entropy_threshold": float(router.entropy_threshold),
                        "error": err_msg,
                        "routed_to_llm": False,
                    },
                )
                continue
            if resp.llm_logits is not None:
                llm_t = torch.as_tensor(resp.llm_logits).float().flatten()
                js = float(compute_js_divergence_logits(L0.cpu().float(), llm_t))
            else:
                topk_js_k = int(getattr(resp, "topk", 16) or 16)
                js = float(
                    compute_js_divergence_topk_union(
                        L0.cpu().float(),
                        q_topk_indices=list(resp.llm_topk_indices or []),
                        q_topk_probs=list(resp.llm_topk_probs or []),
                        q_tail_mass=float(resp.llm_tail_mass or 0.0),
                        top_k=topk_js_k,
                    )
                )
            if js > float(router.js_threshold):
                choices[i] = 1
                if resp.llm_logits is not None:
                    req._reuse_llm_first_logits = llm_t.cpu().clone()
                else:
                    # top-k payload cannot reconstruct full-vocab logits for reuse sampling.
                    req._reuse_llm_first_logits = None
                req.current_llm_reason = "entropy_variance_js"
                print(
                    f"[evjs-switch] rid={getattr(req, 'rid', '?')} "
                    f"entropy={H:.6f} thr_entropy={float(router.entropy_threshold):.6f} "
                    f"js={js:.6f} thr_js={float(router.js_threshold):.6f} -> route=LLM",
                    flush=True,
                )
                append_entropy_lookahead_score_log(
                    router.score_log_path,
                    {
                        "event": "entropy_variance_js",
                        "sub": "js_route",
                        "rid": getattr(req, "rid", None),
                        "seq_in_batch": i,
                        "entropy": H,
                        "entropy_threshold": float(router.entropy_threshold),
                        "js": js,
                        "js_threshold": float(router.js_threshold),
                        "routed_to_llm": True,
                        "reuse_first_llm_token_from_rpc": True,
                    },
                )
            else:
                choices[i] = 0
                req.current_llm_reason = None
                if _cooldown > 0:
                    req._evjs_rpc_cooldown = _cooldown
                try:
                    query_queue.put_nowait(NextTokenJsAbortRpc(rid=str(req.rid)))
                except Exception:
                    pass
                if _log_all:
                    print(
                        f"[evjs-keep] rid={getattr(req, 'rid', '?')} "
                        f"entropy={H:.6f} thr_entropy={float(router.entropy_threshold):.6f} "
                        f"js={js:.6f} thr_js={float(router.js_threshold):.6f} "
                        f"cooldown={_cooldown} -> route=SLM",
                        flush=True,
                    )
                append_entropy_lookahead_score_log(
                    router.score_log_path,
                    {
                        "event": "entropy_variance_js",
                        "sub": "js_below",
                        "rid": getattr(req, "rid", None),
                        "seq_in_batch": i,
                        "entropy": H,
                        "entropy_threshold": float(router.entropy_threshold),
                        "js": js,
                        "js_threshold": float(router.js_threshold),
                        "routed_to_llm": False,
                        "cooldown_set": _cooldown,
                    },
                )
        router.state.last_model = "reference" if choices.any().item() else "quick"
        return choices

    @staticmethod
    def _entropy_lookahead_choices(
        router: EntropyLookaheadSwitching,
        batch: ScheduleBatch,
        logits: torch.Tensor,
        next_token_ids: torch.Tensor,
        scheduler: Scheduler,
        query_queue: mp.Queue,
        reply_queue: mp.Queue,
    ) -> torch.Tensor:
        """Per-sequence S score; synchronous RPC to LLM worker (tp=1)."""
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
            # Path entropies: current-step H(L0) plus one per draft extend (length 1+n).
            entropy_path = [float(H)]
            temp = float(req.sampling_params.temperature)
            top_p = float(req.sampling_params.top_p)
            top_k = int(req.sampling_params.top_k)
            for _ in range(n):
                ctx_k = base_ctx + draft
                Lk = SLMServer._slm_one_off_forward_logits(scheduler, ctx_k)
                entropy_path.append(float(compute_entropy(Lk.unsqueeze(0))))
                tk = sample_token(Lk, temperature=temp, top_p=top_p, top_k=top_k)
                tid = int(tk.item()) if isinstance(tk, torch.Tensor) else int(tk)
                slm_lps.append(log_prob_of_token(Lk, tid))
                draft.append(tid)
            entropy_path_sum = float(sum(entropy_path))
            contexts = [base_ctx + draft[:k] for k in range(len(draft))]
            qid = int(time.time_ns() % (1 << 62))
            rpc = EntropyLookaheadRpc(query_id=qid, contexts=contexts, tokens=draft)
            query_queue.put(rpc)
            resp = reply_queue.get(timeout=router.rpc_timeout_s)
            if not isinstance(resp, EntropyLookaheadResp) or not resp.ok or len(resp.logprobs) != len(draft):
                err = getattr(resp, "error", "unknown")
                _log_entropy_lookahead_rpc_fail_throttled(err, i)
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
    def run_batch(
        batch: ScheduleBatch,
        scheduler: Scheduler,
        router,
        entropy_lookahead_query_queue: Optional[mp.Queue] = None,
        entropy_lookahead_reply_queue: Optional[mp.Queue] = None,
        finished_queue: Optional[mp.Queue] = None,
        outbound_queue: Optional[mp.Queue] = None,
        rank: int = 0,
    ):
        result = scheduler.run_batch(batch)
        batch, hidden_states, logits, next_token_ids = SLMServer.process_routing_input(batch, result)
        # Create a ModelOutputs object for switching strategy
        model_outputs = ModelOutputs(
            logits=logits,
            hidden_states=[hidden_states],  # dummy layer dimension
            token=next_token_ids[:, None],
        )
        if isinstance(router, EntropyLookaheadSwitching):
            if entropy_lookahead_query_queue is not None and entropy_lookahead_reply_queue is not None:
                model_choices = SLMServer._entropy_lookahead_choices(
                    router,
                    batch,
                    logits,
                    next_token_ids,
                    scheduler,
                    entropy_lookahead_query_queue,
                    entropy_lookahead_reply_queue,
                ).cpu()
            else:
                model_choices = router.route(model_outputs).cpu()
        elif isinstance(router, SlidingWindowEntropyJsSwitching):
            if entropy_lookahead_query_queue is not None and entropy_lookahead_reply_queue is not None:
                model_choices = SLMServer._sliding_window_entropy_js_choices(
                    router,
                    batch,
                    logits,
                    next_token_ids,
                    scheduler,
                    entropy_lookahead_query_queue,
                    entropy_lookahead_reply_queue,
                ).cpu()
            else:
                model_choices = router.route(model_outputs).cpu()
        elif isinstance(router, EntropyVarianceJsSwitching):
            if entropy_lookahead_query_queue is not None and entropy_lookahead_reply_queue is not None:
                model_choices = SLMServer._entropy_variance_js_choices(
                    router,
                    batch,
                    logits,
                    next_token_ids,
                    scheduler,
                    entropy_lookahead_query_queue,
                    entropy_lookahead_reply_queue,
                ).cpu()
            else:
                model_choices = router.route(model_outputs).cpu()
        elif isinstance(router, SlidingWindowEntropySwitching):
            model_choices = SLMServer._sliding_window_entropy_choices(
                router,
                batch,
                logits,
                next_token_ids,
                scheduler,
            ).cpu()
        else:
            model_choices = router.route(model_outputs).cpu()
        # TODO: merge router into sglang

        # Check if reference model is needed for any prompt
        reference_needed = torch.any(model_choices)
        req_to_send = []
        if reference_needed:
            for i, req in enumerate(batch.reqs):
                if model_choices[i] == 1:
                    req.status = "notneed"
                    did_truncate = False
                    if isinstance(router, SlidingWindowEntropyJsSwitching):
                        if getattr(req, "_sliding_full_window_truncate_once", False):
                            SLMServer._sliding_window_truncate_entire_window(req, router)
                            req._sliding_full_window_truncate_once = False
                            did_truncate = True
                        else:
                            k_trunc = getattr(req, "_sliding_truncate_argmax_k", None)
                            if k_trunc is not None:
                                SLMServer._sliding_window_truncate_slm_suffix(
                                    req, router, int(k_trunc)
                                )
                                req._sliding_truncate_argmax_k = None
                                did_truncate = True
                    elif isinstance(router, SlidingWindowEntropySwitching):
                        if getattr(req, "_sliding_full_window_truncate_once", False):
                            SLMServer._sliding_window_truncate_entire_window(req, router)
                            req._sliding_full_window_truncate_once = False
                            did_truncate = True
                        elif getattr(router, "truncate_on_llm_trigger", False):
                            k_trunc = getattr(req, "_sliding_truncate_argmax_k", None)
                            if k_trunc is not None:
                                SLMServer._sliding_window_truncate_slm_suffix(
                                    req, router, int(k_trunc)
                                )
                                req._sliding_truncate_argmax_k = None
                                did_truncate = True
                    if isinstance(router, SlidingWindowEntropySwitching) and req.last_llm_loc is not None:
                        req.last_llm_loc = min(int(req.last_llm_loc), len(req.output_ids))

                    reuse_logits = getattr(req, "_reuse_llm_first_logits", None)
                    if (
                        isinstance(router, (SlidingWindowEntropyJsSwitching, EntropyVarianceJsSwitching))
                        and reuse_logits is not None
                    ):
                        sp0 = req.sampling_params
                        tok = sample_token(
                            reuse_logits
                            if isinstance(reuse_logits, torch.Tensor)
                            else torch.as_tensor(reuse_logits, dtype=torch.float32),
                            temperature=float(sp0.temperature),
                            top_p=float(sp0.top_p),
                            top_k=int(sp0.top_k),
                        )
                        tok_i = int(tok) if isinstance(tok, int) else int(tok.item())
                        req._reuse_llm_first_logits = None
                        req.output_ids.append(tok_i)
                        req.llm_token_count = getattr(req, "llm_token_count", 0) + 1
                        if tok_i in scheduler.model_config.hf_eos_token_id:
                            scheduler.abort_request(AbortReq(req.rid))
                        req.check_finished()
                        if req.finished():
                            SLMServer.tree_cache_finished_req_safe(scheduler, req)
                            if rank == 0:
                                SLMServer.process_finished_requests(
                                    [req], scheduler.tokenizer, finished_queue, outbound_queue
                                )
                            try:
                                scheduler.issued_reqs.remove(req)
                            except Exception:
                                pass
                            continue

                    new_token_ids = []
                    if req.last_llm_loc is None:
                        req.last_llm_loc = 0
                        new_token_ids = list(req.origin_input_ids)
                    new_token_ids = new_token_ids + list(req.output_ids[req.last_llm_loc :])

                    st = getattr(req, "current_llm_reason", "need") or "need"
                    llm_used = int(getattr(req, "llm_token_count", 0) or 0) > 0
                    incremental_empty = (req.last_llm_loc is not None) and (
                        len(req.output_ids) <= int(req.last_llm_loc)
                    )
                    if llm_used and (did_truncate or incremental_empty or len(new_token_ids) == 0):
                        new_token_ids = list(req.origin_input_ids) + list(req.output_ids)
                        st = "RETRACT_AND_PREFILL"
                    elif st == "entropy_variance_js" and not llm_used and not did_truncate:
                        st = "EVJS_CONTINUE"

                    req.last_llm_loc = len(req.output_ids)
                    waiting_req = WaitingReq(
                        rid=req.rid,
                        new_token_ids=new_token_ids,
                        sampling_params=SimpleSamplingParams(
                            temperature=req.sampling_params.temperature,
                            top_k=req.sampling_params.top_k,
                            top_p=req.sampling_params.top_p,
                            max_new_tokens=1,
                        ),
                        status=st,
                    )
                    req_to_send.append(waiting_req)
        return result, req_to_send

    @staticmethod
    def process_routing_input(batch: ScheduleBatch, result):
        device = batch.seq_lens.device
        batch_size = batch.batch_size()
        is_prefill = (result.logits_output.hidden_states.shape[0] != batch_size)

        if is_prefill:
            # For prefill, use cumsum of extend_lens to get correct indices
            extend_lens = torch.tensor(batch.extend_lens, device=device)
            hidden_indices = torch.cumsum(extend_lens, dim=0) - 1
        else:
            # For decode, use sequential indices
            hidden_indices = torch.arange(batch_size, device=device)

        # Get hidden states for the relevant positions
        hidden_states = result.logits_output.hidden_states[hidden_indices, :][:, None, :] # batch_size, 1, hidden_size
        logits = result.logits_output.next_token_logits # batch_size, vocab_size
        next_token_ids = result.next_token_ids # batch_size

        return batch, hidden_states, logits[:, None, :], next_token_ids

    # NOTE: original recv_requests removed in favor of central queue distribution.
    
    @staticmethod
    def process_batch_results(batch: ScheduleBatch, result, scheduler: Scheduler, finished_queue: Optional[mp.Queue] = None, outbound_queue: Optional[mp.Queue] = None, rank: int = 0, req_to_send: Optional[List[WaitingReq]] = None):
        slm_generated_tokens = batch.batch_size() - len(req_to_send)
        if rank == 0:
            if not hasattr(scheduler, "last_status_time"):
                scheduler.last_status_time = time.perf_counter()
            if not hasattr(scheduler, "n_generated_tokens"):
                scheduler.n_generated_tokens = 0
            scheduler.n_generated_tokens += slm_generated_tokens
            gap_latency = time.perf_counter() - scheduler.last_status_time
            iv = _tps_log_interval_s()
            if gap_latency > iv:
                print(
                    f"[quick rank{rank}] throughput: {scheduler.n_generated_tokens / gap_latency:.2f} tokens/s",
                    flush=True,
                )
                scheduler.n_generated_tokens = 0
                scheduler.last_status_time = time.perf_counter()
                if _perf_probe_enabled() and batch.reqs:
                    parts = []
                    for req in batch.reqs[:6]:
                        if getattr(req, "status", None) == "notneed":
                            continue
                        inp = len(getattr(req, "origin_input_ids", None) or [])
                        out = len(getattr(req, "output_ids", None) or [])
                        parts.append(
                            f"rid={getattr(req, 'rid', '?')} seq={inp + out} "
                            f"slm={getattr(req, 'slm_token_count', 0)} "
                            f"llm={getattr(req, 'llm_token_count', 0)}"
                        )
                    if parts:
                        print("[R2R_PERF_PROBE] " + " | ".join(parts), flush=True)
        _log_current_tokens_per_second(
            scheduler=scheduler,
            rank=rank,
            source="slm",
            token_count=slm_generated_tokens,
        )
        batch.output_ids = result.next_token_ids
        finished_reqs = []

        for req, next_token_id in zip(batch.reqs, result.next_token_ids):
            if req.status == "notneed":
                continue
            if next_token_id in scheduler.model_config.hf_eos_token_id:
                scheduler.abort_request(AbortReq(req.rid))
            req.output_ids.append(next_token_id.item())
            # Track SLM token generation
            req.slm_token_count = getattr(req, 'slm_token_count', 0) + 1
            req.check_finished()
            cur_len = len(req.output_ids)
            milestone = cur_len // 128
            if rank == 0 and milestone > getattr(req, "_agent_debug_milestone", -1):
                req._agent_debug_milestone = milestone
                if os.environ.get("R2R_LOG_GENERATION_PREVIEW", "").strip().lower() in ("1", "true", "yes", "on"):
                    try:
                        tail_ids = list(req.output_ids[-24:])
                        tail_text = scheduler.tokenizer.decode(tail_ids) if tail_ids else ""
                        print(
                            f"[gen-preview] rid={getattr(req, 'rid', '?')} out_len={cur_len} tail={tail_text!r}",
                            flush=True,
                        )
                    except Exception:
                        pass
                #region agent log
                _agent_debug_log(
                    run_id=os.environ.get("R2R_DEBUG_RUN_ID", "run-unknown"),
                    hypothesis_id="H1",
                    location="slm_server.py:process_batch_results",
                    message="req_progress_milestone",
                    data={
                        "rid": getattr(req, "rid", None),
                        "output_len": cur_len,
                        "slm_token_count": int(getattr(req, "slm_token_count", 0)),
                        "llm_token_count": int(getattr(req, "llm_token_count", 0)),
                        "finished": bool(req.finished()),
                        "status": getattr(req, "status", None),
                    },
                )
                #endregion
            if req.finished():
                SLMServer.tree_cache_finished_req_safe(scheduler, req)
                finished_reqs.append(req)

        if len(finished_reqs) > 0 and rank == 0:
            SLMServer.process_finished_requests(finished_reqs, scheduler.tokenizer, finished_queue, outbound_queue)
        if len(finished_reqs) > 0:
            for req in finished_reqs:
                scheduler.issued_reqs.remove(req)
        if len(req_to_send) > 0:
            scheduler.check_batch_status(batch)
            if rank == 0:
                outbound_queue.put_nowait(req_to_send)

    @staticmethod
    def process_finished_requests(finished_reqs: List[Req], tokenizer, finished_queue: Optional[mp.Queue] = None, outbound_queue: Optional[mp.Queue] = None):
        """Process finished requests, e.g., logging or updating status."""
        for req in finished_reqs:
            #print(f"Request {req.rid} finished")
            #print(f"You: {req.origin_input_text}")
            #print(f"Bot: {tokenizer.decode(req.output_ids)}")
            #print("===")
            # Enqueue to system if queue is provided (send a lightweight serializable payload)
            outbound_queue.put_nowait([WaitingReq(
                rid=req.rid, 
                new_token_ids=[], 
                sampling_params=SimpleSamplingParams(),
                status="finished",
            )])
            if finished_queue is not None:
                slm_count = getattr(req, 'slm_token_count', 0)
                llm_count = getattr(req, 'llm_token_count', 0)
                total_count = slm_count + llm_count
                payload = {
                    "rid": getattr(req, "rid", None),
                    "origin_input_text": getattr(req, "origin_input_text", None),
                    "origin_input_ids": list(getattr(req, "origin_input_ids", [])),
                    "output_ids": list(getattr(req, "output_ids", [])),
                    "output_text": tokenizer.decode(getattr(req, "output_ids", [])),
                    "status": "finished",
                    "slm_token_count": slm_count,
                    "llm_token_count": llm_count,
                    "llm_ratio": llm_count / total_count if total_count > 0 else 0.0,
                }
                #region agent log
                _agent_debug_log(
                    run_id=os.environ.get("R2R_DEBUG_RUN_ID", "run-unknown"),
                    hypothesis_id="H4",
                    location="slm_server.py:process_finished_requests",
                    message="req_marked_finished",
                    data={
                        "rid": payload.get("rid"),
                        "output_len": len(payload.get("output_ids", [])),
                        "slm_token_count": int(slm_count),
                        "llm_token_count": int(llm_count),
                    },
                )
                #endregion
                try:
                    finished_queue.put_nowait(payload)
                except Exception:
                    try:
                        finished_queue.put(payload)
                    except Exception:
                        pass

    def shutdown(self):
        """Gracefully terminate quick model workers (mirrors LLMServer)."""
        # stop recv thread
        try:
            if hasattr(self, "_stop_event"):
                self._stop_event.set()
            if hasattr(self, "_recv_thread") and self._recv_thread.is_alive():
                self._recv_thread.join(timeout=2)
        except Exception:
            pass
        # stop finished-reqs send thread
        try:
            if hasattr(self, "_send_finished_stop"):
                self._send_finished_stop.set()
            if hasattr(self, "_finished_reqs_queue"):
                try:
                    self._finished_reqs_queue.put_nowait(None)
                except Exception:
                    pass
            if hasattr(self, "_send_finished_thread") and self._send_finished_thread.is_alive():
                self._send_finished_thread.join(timeout=2)
        except Exception:
            pass
        # stop inter-server send thread
        try:
            if hasattr(self, "_send_llm_stop"):
                self._send_llm_stop.set()
            if hasattr(self, "queue_to_llm"):
                try:
                    self.queue_to_llm.put_nowait(None)
                except Exception:
                    pass
            if hasattr(self, "_send_llm_thread") and self._send_llm_thread.is_alive():
                self._send_llm_thread.join(timeout=2)
        except Exception:
            pass
        # stop inter-server recv thread
        try:
            if hasattr(self, "_recv_from_llm_stop"):
                self._recv_from_llm_stop.set()
            if hasattr(self, "_recv_from_llm_thread") and self._recv_from_llm_thread and self._recv_from_llm_thread.is_alive():
                self._recv_from_llm_thread.join(timeout=2)
        except Exception:
            pass
        # close sockets
        try:
            if hasattr(self, "_pub_llm") and self._pub_llm is not None:
                self._pub_llm.close(0)
        except Exception:
            pass
        try:
            if hasattr(self, "_sub_from_llm") and self._sub_from_llm is not None:
                self._sub_from_llm.close(0)
        except Exception:
            pass
        try:
            if hasattr(self, "_pub_finished") and self._pub_finished is not None:
                self._pub_finished.close(0)
        except Exception:
            pass
        if hasattr(self, "quick_model_procs"):
            for p in self.quick_model_procs:
                try:
                    if p.is_alive():
                        p.terminate()
                        p.join(timeout=5)
                        if p.is_alive():
                            try:
                                p.kill()
                            except Exception:
                                pass
                except Exception:
                    pass

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass

    # ===== Controller-triggered: start SUB to receive from LLM (LLM -> SLM) =====
    def start_slm_sub(self, port: int):
        if port is None:
            print('[SLMServer] start_slm_sub called with None port')
            return
        if self._sub_from_llm is not None:
            return  # already started
        ctx = zmq.Context.instance()
        try:
            self._sub_from_llm = ctx.socket(zmq.SUB)
            self._sub_from_llm.setsockopt(zmq.LINGER, 0)
            self._sub_from_llm.connect(f"tcp://127.0.0.1:{port}")
            self._sub_from_llm.setsockopt(zmq.SUBSCRIBE, b"")
        except Exception as e:
            print(f"[SLMServer] Failed to connect SUB from LLM: {e}")
            self._sub_from_llm = None
            return
        def _recv_loop():
            poller = zmq.Poller()
            poller.register(self._sub_from_llm, zmq.POLLIN)
            while not self._recv_from_llm_stop.is_set():
                try:
                    events = dict(poller.poll(timeout=50))
                except Exception:
                    continue
                if self._sub_from_llm in events and events[self._sub_from_llm] == zmq.POLLIN:
                    while True:
                        try:
                            msg = self._sub_from_llm.recv_pyobj(flags=zmq.NOBLOCK)
                        except zmq.Again:
                            break
                        except Exception:
                            break
                        # Replicate to every rank's inbound queue to preserve identical ordering
                        
                        try:
                            self._inbound_queues.put_nowait(msg)
                        except Exception as e:
                            print(f"Failed to enqueue inbound msg to rank queue. Exception:{e}")
                            pass
        self._recv_from_llm_thread = threading.Thread(target=_recv_loop, daemon=True)
        self._recv_from_llm_thread.start()
        print(f"[SLMServer] SUB from LLM started on port {port}, loaded successfully")

    # Helper for user to enqueue outbound messages (optional direct use)
    def enqueue_to_llm(self, obj):
        try:
            self.queue_to_llm.put_nowait(obj)
        except Exception:
            try:
                self.queue_to_llm.put(obj)
            except Exception:
                pass

    @staticmethod
    def update_tqdm(pbar_dict: Dict[str, tqdm], batch: ScheduleBatch, scheduler: Scheduler):
        # Refresh reasoning progress bars
        current_rids = set()
        for req in batch.reqs+scheduler.batch_not_need.reqs:
            if not req.display_progress:
                continue
            current_rids.add(req.rid)
            if req.rid not in pbar_dict:
                pbar_dict[req.rid] = tqdm(total=req.sampling_params.max_new_tokens, desc=f"Req {req.rid}", leave=False)
            
            pbar_dict[req.rid].n = len(req.output_ids)
            pbar_dict[req.rid].refresh()
        
        # Close progress bars for requests that are no longer in the batch
        finished_rids = [rid for rid in pbar_dict if rid not in current_rids]
        for rid in finished_rids:
            pbar_dict[rid].close()
            del pbar_dict[rid]
