#!/usr/bin/env bash
# 机器 A：仅 quick（小模型）。输出目录名: slm
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "${HERE}/00_env.sh"
_run_hf slm --use_model quick
