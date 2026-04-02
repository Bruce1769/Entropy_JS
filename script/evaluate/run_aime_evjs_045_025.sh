#!/usr/bin/env bash
# AIME hybrid eval: entropy_variance_js with h=0.45, js=0.25.
# Requires 2 GPUs (SLM + LLM), models under ./models/ per config.
# Optional: FlashInfer JIT — set CC/CXX if your host compiler is not on PATH.
set -euo pipefail

MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-32768}
BATCH_SIZE=${BATCH_SIZE:-1}
CUDA_DEVICES=${CUDA_DEVICES:-0,1}

ENTROPY_THRESHOLD=0.45
JS_THRESHOLD=0.25

R2R_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${R2R_ROOT}"

PYTHON="${PYTHON:-python3}"
OUT="experiments/aime_evjs_h045_js025_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"

echo "[RUN] repo=${R2R_ROOT} output_dir=${OUT}"

env \
  PATH="${PATH}" \
  PYTHONUNBUFFERED=1 \
  HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}" \
  CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" \
  R2R_WAIT_LOG_EVERY_S=10 \
  R2R_GENERATE_TIMEOUT_S=7200 \
  R2R_TPS_LOG_INTERVAL=1 \
  R2R_EVJS_TOPK=16 \
  R2R_EVJS_COOLDOWN=64 \
  R2R_LOG_EVJS_ALL=1 \
  R2R_PERF_PROBE=1 \
  R2R_DISPLAY_PROGRESS=1 \
  R2R_LOG_GENERATION_PREVIEW=1 \
  R2R_TRACE_REQ_FLOW=1 \
  "${PYTHON}" -u "${R2R_ROOT}/script/evaluate/hf_dataset_sglang.py" \
    --use_hybrid \
    --dataset aime \
    --dataset_path GY2233/AIME-2024-2025 \
    --config-path config/Qwen3-0.6B+Qwen3-8B_entropy_variance_js_aime.yaml \
    --output_dir "$OUT" \
    --slm_tp_size 1 --llm_tp_size 1 --tp_size 1 \
    --batch_size "$BATCH_SIZE" \
    --threshold "$ENTROPY_THRESHOLD" \
    --js_threshold "$JS_THRESHOLD" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
  2>&1 | tee "$OUT/run.log"

echo "[DONE] log=$OUT/run.log"
