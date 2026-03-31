#!/usr/bin/env bash
# Qwen3-0.6B + Qwen3-8B，滑动窗口平均熵 > 0.3；window_size=5；GPU 0,1。
# 结果：output/eval/sliding_window_entropy/mean0.3_w5_gpu01/run_<时间戳>/{aime24,aime25}
set -euo pipefail
R2R_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${R2R_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
PYTHON="${PYTHON:-python}"
export PYTHONUNBUFFERED=1
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32768}"
CFG="${CFG:-config/Qwen3-0.6B+Qwen3-8B_sliding_window_entropy_mean0.3_w5.yaml}"
STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
BASE_OUT="${R2R_ROOT}/output/eval/sliding_window_entropy/mean0.3_w5_gpu01/run_${STAMP}"

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
