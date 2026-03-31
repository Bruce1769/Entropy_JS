#!/usr/bin/env bash
set -euo pipefail
cd /remote-home/pxl/R2R
# 仅用物理 GPU 0、1：小模型 -> cuda:0，大模型 -> cuda:1
export CUDA_VISIBLE_DEVICES=0,1
# warmup 的 rid=0 会走同一套 entropy_variance_js，易长时间占满 worker，首个评测 rid=1 一直 pending
export R2R_SKIP_WARMUP=1
OUT="/remote-home/pxl/R2R/experiments/aime2425_entropy_variance_js_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
echo "$OUT" > "$OUT/OUTPUT_DIR.txt"
nohup python script/evaluate/hf_dataset_sglang.py \
  --use_hybrid \
  --dataset aime \
  --dataset_path GY2233/AIME-2024-2025 \
  --config-path config/Qwen3-0.6B+Qwen3-8B_entropy_variance_js_aime.yaml \
  --output_dir "$OUT" \
  --slm_tp_size 1 \
  --llm_tp_size 1 \
  --tp_size 2 \
  --batch_size 1 \
  --threshold 0.45 \
  --js_threshold 0.1 \
  > "$OUT/run.log" 2>&1 &
echo $! > "$OUT/nohup_shell.pid"
echo "Started PID $(cat "$OUT/nohup_shell.pid") log $OUT/run.log"
