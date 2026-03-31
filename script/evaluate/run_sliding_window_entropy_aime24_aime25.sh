#!/usr/bin/env bash
# 滑动窗口熵路由 hybrid：依次跑 AIME 2024 + AIME 2025。
# 需要 2 张 GPU（slm_tp=1, llm_tp=1，总 tp_size=2）。
# 若曾设置 HF_HUB_OFFLINE，需能访问 Hub 检查模型文件，或确保模型已完整缓存。
#
# 生成长度：hf_dataset_sglang 默认 --max_new_tokens 32768。本脚本默认与之对齐；若需缩短单题耗时可
#   export MAX_NEW_TOKENS=8192
set -euo pipefail
R2R_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${R2R_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
PYTHON="${PYTHON:-python}"
export PYTHONUNBUFFERED=1
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32768}"
CFG="config/Qwen3-0.6B+Qwen3-8B_sliding_window_entropy.yaml"
STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
BASE_OUT="${R2R_ROOT}/output/eval/sliding_window_entropy/run_${STAMP}"

# 允许在线访问 Hugging Face（避免 hf_quant_config.json 检查失败）
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE HF_DATASETS_OFFLINE 2>/dev/null || true

# 探针：seq_len / slm vs llm 计数 + 更密的吞吐行（见 r2r/models/sglang_patch/slm_server.py）
export R2R_PERF_PROBE="${R2R_PERF_PROBE:-1}"
export R2R_TPS_LOG_INTERVAL="${R2R_TPS_LOG_INTERVAL:-1.5}"
# 每步打印滑动窗口熵 path 与 path_sum（日志会很多）；限频：export R2R_ENTROPY_SUM_MIN_INTERVAL_S=0.5
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
