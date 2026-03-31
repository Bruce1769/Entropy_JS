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

## Upstream

This code is derived from **Roads to Rome (R2R)** — see `README.md` and the original repository [thu-nics/R2R](https://github.com/thu-nics/R2R).
