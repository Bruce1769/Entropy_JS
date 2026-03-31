#!/usr/bin/env bash
# 机器 C：R2R + YAML 中的 neural router（默认）。输出目录名: r2r_neural
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "${HERE}/00_env.sh"
_run_hf r2r_neural --use_hybrid
