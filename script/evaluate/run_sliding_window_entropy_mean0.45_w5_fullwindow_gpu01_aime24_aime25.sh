#!/usr/bin/env bash
# Qwen3-0.6B + Qwen3-8B，滑动窗口 mean>0.45，window=5，intervention_mode=replace_full_window（整窗截断后 LLM 续写 N 步再回 SLM）。
# 结果：output/eval/sliding_window_entropy/mean0.45_w5_fullwindow_gpu01/run_<时间戳>/{aime24,aime25}
set -euo pipefail
R2R_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${R2R_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
PYTHON="${PYTHON:-python}"
export PYTHONUNBUFFERED=1
# 默认 8192：过长序列 + replace_full_window 易触发超长 LLM prefill/同步卡住；需要时再 export MAX_NEW_TOKENS=32768
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8192}"
CFG="${CFG:-config/Qwen3-0.6B+Qwen3-8B_sliding_window_entropy_mean0.45_w5_fullwindow.yaml}"
STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
BASE_OUT="${R2R_ROOT}/output/eval/sliding_window_entropy/mean0.45_w5_fullwindow_gpu01/run_${STAMP}"

# 使用 config 里 models/ 下本地权重时，可取消下行注释以禁止访问 Hugging Face Hub（需数据集已缓存）。
# export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE HF_DATASETS_OFFLINE 2>/dev/null || true

export R2R_PERF_PROBE="${R2R_PERF_PROBE:-1}"
export R2R_TPS_LOG_INTERVAL="${R2R_TPS_LOG_INTERVAL:-1.5}"
export R2R_LOG_ENTROPY_SUM="${R2R_LOG_ENTROPY_SUM:-1}"

_run() {
  local ds="$1"
  local out="${BASE_OUT}/${ds}"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] starting ${ds} -> ${out}"
  mkdir -p "${out}"
  "${PYTHON}" -u "${R2R_ROOT}/script/evaluate/hf_dataset_sglang.py" \
    --dataset "${ds}" \
    --config-path "${CFG}" \
    --generator sglang \
    --use_hybrid \
    --output_dir "${out}" \
    --tp_size 2 \
    --slm_tp_size 1 \
    --llm_tp_size 1 \
    --max_new_tokens "${MAX_NEW_TOKENS}"
}

_run aime24
_run aime25
echo "[$(date '+%Y-%m-%d %H:%M:%S')] done. outputs under ${BASE_OUT}"
