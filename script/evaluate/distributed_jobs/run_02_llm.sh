#!/usr/bin/env bash
# 机器 B：仅 reference（大模型）。输出目录名: llm
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "${HERE}/00_env.sh"
_run_hf llm --use_model reference
