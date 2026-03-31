#!/usr/bin/env bash
# Launch a single hybrid AIME eval into an existing method_suite directory (same layout as
# run_aime_benchmark_suite.py). Use when slm/ and llm/ already have combined_results.csv
# and you want r2r_neural + each entropy-variance YAML on separate machines.
#
# Typical: 2 GPUs per box → export CUDA_VISIBLE_DEVICES=0,1
#
# Usage (from repo root, or any cwd — script cd's to R2R root):
#   export METHOD_SUITE_BASE=/remote-home/pxl/R2R/output/eval/method_suite_20260323_154454
#   export CUDA_VISIBLE_DEVICES=0,1
#   bash script/evaluate/launch_method_suite_hybrid.sh neural --detach
#   bash script/evaluate/launch_method_suite_hybrid.sh e045 --detach
#
# Modes:
#   neural   → r2r_neural/           (base Qwen3-0.6B+Qwen3-8B.yaml + --use_hybrid)
#   e045     → r2r_Qwen3-0.6B_Qwen3-8B_entropy_0.45_variance_1e-5/
#   e095     → ..._0.95_...
#   e12      → ..._1.2_...
#   e15      → ..._1.5_...
#   e17      → ..._1.7_...
#   e20      → ..._2_variance_1e-5/
#
# Options:
#   --detach   nohup + log under $METHOD_SUITE_BASE/logs/<mode>.log
#   --dry-run  print command only
#
# After all jobs finish:
#   python script/evaluate/aggregate_and_plot_benchmarks.py --suite-dir "$METHOD_SUITE_BASE"
#
# FlashInfer JIT + conda GCC often breaks CUDA nvcc. This script defaults CC/CXX to
# /usr/bin when unset or conda-provided. Opt out: export R2R_SKIP_CUDA_HOST_COMPILER_FIX=1

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

if [[ "${R2R_SKIP_CUDA_HOST_COMPILER_FIX:-}" != "1" ]] && [[ -x /usr/bin/gcc ]] && [[ -x /usr/bin/g++ ]]; then
  if [[ -z "${CC:-}" ]] || [[ "${CC}" == *conda* ]]; then
    export CC=/usr/bin/gcc
  fi
  if [[ -z "${CXX:-}" ]] || [[ "${CXX}" == *conda* ]]; then
    export CXX=/usr/bin/g++
  fi
fi

BASE="${METHOD_SUITE_BASE:-}"
if [[ -z "$BASE" ]]; then
  echo "Set METHOD_SUITE_BASE to your method_suite_* directory." >&2
  exit 1
fi

PY="${PYTHON:-python}"
EVAL=( "$PY" script/evaluate/hf_dataset_sglang.py --dataset aime --generator sglang )

BASE_CFG="config/Qwen3-0.6B+Qwen3-8B.yaml"
DETACH=false
DRY=false
MODE=""

for a in "$@"; do
  case "$a" in
    --detach) DETACH=true ;;
    --dry-run) DRY=true ;;
    -*)
      echo "Unknown option: $a" >&2
      exit 1
      ;;
    *)
      if [[ -n "$MODE" ]]; then
        echo "Only one MODE allowed." >&2
        exit 1
      fi
      MODE="$a"
      ;;
  esac
done

if [[ -z "$MODE" ]]; then
  echo "Usage: $0 <neural|e045|e095|e12|e15|e17|e20> [--detach] [--dry-run]" >&2
  exit 1
fi

case "$MODE" in
  neural)
    REL_OUT="r2r_neural"
    CFG="$BASE_CFG"
    EXTRA=( --use_hybrid )
    ;;
  e045)
    REL_OUT="r2r_Qwen3-0.6B_Qwen3-8B_entropy_0.45_variance_1e-5"
    CFG="config/Qwen3-0.6B+Qwen3-8B_entropy_0.45_variance_1e-5.yaml"
    EXTRA=( --use_hybrid )
    ;;
  e095)
    REL_OUT="r2r_Qwen3-0.6B_Qwen3-8B_entropy_0.95_variance_1e-5"
    CFG="config/Qwen3-0.6B+Qwen3-8B_entropy_0.95_variance_1e-5.yaml"
    EXTRA=( --use_hybrid )
    ;;
  e12)
    REL_OUT="r2r_Qwen3-0.6B_Qwen3-8B_entropy_1.2_variance_1e-5"
    CFG="config/Qwen3-0.6B+Qwen3-8B_entropy_1.2_variance_1e-5.yaml"
    EXTRA=( --use_hybrid )
    ;;
  e15)
    REL_OUT="r2r_Qwen3-0.6B_Qwen3-8B_entropy_1.5_variance_1e-5"
    CFG="config/Qwen3-0.6B+Qwen3-8B_entropy_1.5_variance_1e-5.yaml"
    EXTRA=( --use_hybrid )
    ;;
  e17)
    REL_OUT="r2r_Qwen3-0.6B_Qwen3-8B_entropy_1.7_variance_1e-5"
    CFG="config/Qwen3-0.6B+Qwen3-8B_entropy_1.7_variance_1e-5.yaml"
    EXTRA=( --use_hybrid )
    ;;
  e20)
    REL_OUT="r2r_Qwen3-0.6B_Qwen3-8B_entropy_2_variance_1e-5"
    CFG="config/Qwen3-0.6B+Qwen3-8B_entropy_2_variance_1e-5.yaml"
    EXTRA=( --use_hybrid )
    ;;
  *)
    echo "Unknown MODE: $MODE" >&2
    exit 1
    ;;
esac

OUT="$BASE/$REL_OUT"
mkdir -p "$BASE/logs"

CMD=(
  "${EVAL[@]}"
  --config-path "$CFG"
  --output_dir "$OUT"
  "${EXTRA[@]}"
  --resume
  --tp_size 1 --slm_tp_size 1 --llm_tp_size 1
)

echo "OUT_DIR=$OUT"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
printf '%q ' "${CMD[@]}"
echo

if $DRY; then
  exit 0
fi

if $DETACH; then
  LOG="$BASE/logs/hybrid_${MODE}.log"
  nohup env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" "${CMD[@]}" >>"$LOG" 2>&1 &
  echo "Started PID $!  log: $LOG"
else
  exec env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" "${CMD[@]}"
fi
