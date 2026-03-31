#!/usr/bin/env bash
# 机器 D…：熵路由单一阈值。用法: ./run_04_r2r_entropy.sh 0.45
# 输出目录名须与套件一致: r2r_entropy_0_45（小数点换成下划线）
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "${HERE}/00_env.sh"

T="${1:?用法: $0 <entropy_threshold>   例: $0 0.45}"
tid="${T//./_}"
_run_hf "r2r_entropy_${tid}" \
  --use_hybrid \
  --switching_strategy entropy \
  --threshold "${T}"
