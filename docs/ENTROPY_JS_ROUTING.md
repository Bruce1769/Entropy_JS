# Entropy and Entropy–JS routing (R2R code paths)

This document maps **where routing decisions are implemented** in this tree (small model vs large model per token or step).

## Strategies (configuration)

- **YAML** (`config/`): `router.switching_strategy` and thresholds, e.g. `entropy`, `entropy_variance`, `entropy_variance_js`.
- **CLI** (`script/evaluate/hf_dataset_sglang.py`): `--threshold` (entropy), `--js_threshold`, `--variance_threshold`, etc., merged into `strategy_kwargs` per strategy.

## Core modules

| Strategy | Definition (math / `route`) | Hybrid runtime (SGLang worker) |
|------------|---------------------------|--------------------------------|
| `entropy` | `r2r/utils/switching.py` → `EntropySwitching.route` | `slm_server.run_batch` → `router.route` |
| `entropy_variance` | `EntropyVarianceSwitching.route` | same |
| `entropy_variance_js` | `EntropyVarianceJsSwitching.route` is **entropy-only fallback** when RPC queues are missing | **`SLMServer._entropy_variance_js_choices`** in `r2r/models/sglang_patch/slm_server.py`: entropy gate → LLM RPC → **JS(P_SLM, P_LLM)** vs `js_threshold` |

## Important files

1. **`r2r/utils/switching.py`** — strategy classes and `create_switching_strategy()`.
2. **`r2r/utils/metrics.py`** — `compute_entropy`, `compute_js_divergence_logits`.
3. **`r2r/models/sglang_patch/slm_server.py`** — `run_batch()` dispatches to `router.route` or `_entropy_variance_js_choices` / sliding-window variants; builds `req_to_send` to the LLM.
4. **`r2r/models/sglang_patch/llm_server.py`** — `NextTokenJsRpc` handling (`_next_token_js_llm_logits`).
5. **`r2r/models/sglang_patch/sl_disaggregation_system.py`** — hybrid system wiring, ZMQ finished-reqs.
6. **`r2r/models/sglang_patch/schedule_req.py`** — `NextTokenJsRpc` / `NextTokenJsResp` dataclasses.

## Note on `EntropyVarianceJsSwitching.route()`

For **`entropy_variance_js`** under hybrid serving, **JS is not computed inside `route()`**; it is computed in **`_entropy_variance_js_choices`** (needs SLM↔LLM RPC). The class `route()` is only used when RPC queues are not wired.

## Performance and environment knobs (EVJS)

- **`R2R_EVJS_TOPK`** (default `16`): LLM RPC returns compact top-k probs + tail mass instead of full-vocab logits for JS (`NextTokenJsResp` in `schedule_req.py`; `_next_token_js_llm_logits` in `llm_server.py`; JS via `compute_js_divergence_topk_union` in `metrics.py`).
- **`R2R_EVJS_COOLDOWN`** (default `0`): after an RPC concludes “stay on SLM” (JS below threshold), skip further LLM RPCs for this many decode steps even if entropy is high. Reduces synchronous RPC frequency dramatically.
- **Incremental KV on LLM side**: `_llm_one_off_forward_logits_persist` / `_llm_incremental_forward` reuse stashed KV when the context grows by a prefix extension, avoiding a full-context prefill on every RPC when possible.
- **Logging**: `R2R_LOG_EVJS_ALL`, `R2R_LOG_GENERATION_PREVIEW`, `R2R_TRACE_REQ_FLOW`, `R2R_DISPLAY_PROGRESS` (see `hf_dataset_sglang.py` and `slm_server.py`).

Example AIME launch script: `script/evaluate/run_aime_evjs_045_025.sh`.

## Upstream

This code is derived from **Roads to Rome (R2R)** — see `README.md` and the original repository [thu-nics/R2R](https://github.com/thu-nics/R2R).
