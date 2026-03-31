#!/usr/bin/env bash
# 在设置好 SUITE_NAME 后执行，打印聚合命令（不跑评测）
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "${HERE}/00_env.sh"
echo "Suite 目录: ${SUITE_DIR}"
echo "聚合命令:"
echo "  cd ${R2R_ROOT} && ${PYTHON} script/evaluate/aggregate_and_plot_benchmarks.py --suite-dir ${SUITE_DIR}"
