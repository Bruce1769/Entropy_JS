#!/usr/bin/env bash
# =============================================================================
# 多机分布式跑评测：所有机器使用相同的 SUITE_NAME，结果会落在
#   $R2R_ROOT/output/eval/$SUITE_NAME/<run_id>/
# 跑完后在任意一台有完整子目录的机器上执行（或 nfs 汇总后）：
#   python script/evaluate/aggregate_and_plot_benchmarks.py --suite-dir output/eval/$SUITE_NAME
#
# 后台跑（nohup + 日志）：
#   export SUITE_NAME=... DATASET=... CUDA_VISIBLE_DEVICES=0,1
#   bash script/evaluate/distributed_jobs/run_job_nohup.sh run_01_slm.sh
# 日志目录：output/eval/$SUITE_NAME/logs/nohup_<脚本>_<时间>.log
# =============================================================================
set -euo pipefail

# distributed_jobs 在 repo/script/evaluate/distributed_jobs/，需三级回到仓库根
_DIST_JOBS="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
export R2R_ROOT="$(cd "${_DIST_JOBS}/../../.." && pwd)"
export EVAL_SCRIPT="${R2R_ROOT}/script/evaluate/hf_dataset_sglang.py"

# --- 多机必须一致（否则无法在同一 suite-dir 下聚合）---
# 仅目录名，与跑哪个数据集无关；建议自己改成可辨认的名字，例如:
#   method_suite_20260325_AMC   或   method_suite_shared_AIME
export SUITE_NAME="${SUITE_NAME:-method_suite_shared}"

# --- 按需修改 ---
export CONFIG_PATH="${CONFIG_PATH:-config/Qwen3-0.6B+Qwen3-8B.yaml}"
export DATASET="${DATASET:-aime}"
export PYTHON="${PYTHON:-python}"

# 速度对比建议：SLM-only / LLM-only 的 tp 与 hybrid 两侧分别对齐
export TP_SIZE="${TP_SIZE:-1}"
export SLM_TP_SIZE="${SLM_TP_SIZE:-1}"
export LLM_TP_SIZE="${LLM_TP_SIZE:-1}"

# 每台机器自己的 GPU（互不影响）
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# 可选附加参数（会按词切分传给 python），例如：--num_problems 20 --debug
export EXTRA_EVAL_FLAGS="${EXTRA_EVAL_FLAGS:-}"

export SUITE_DIR="${R2R_ROOT}/output/eval/${SUITE_NAME}"

_run_hf() {
  local out_subdir="$1"
  shift
  mkdir -p "${SUITE_DIR}/${out_subdir}"
  cd "${R2R_ROOT}"
  # shellcheck disable=SC2086
  "${PYTHON}" "${EVAL_SCRIPT}" \
    --dataset "${DATASET}" \
    --config-path "${CONFIG_PATH}" \
    --generator sglang \
    --output_dir "${SUITE_DIR}/${out_subdir}" \
    --tp_size "${TP_SIZE}" \
    --slm_tp_size "${SLM_TP_SIZE}" \
    --llm_tp_size "${LLM_TP_SIZE}" \
    "$@" \
    ${EXTRA_EVAL_FLAGS}
}
