#!/usr/bin/env bash
# 使用「熵+方差」专用 YAML 跑 hybrid（与 run_aime_benchmark_suite --entropy-variance-configs 一致）。
# 用法: ./run_05_r2r_entropy_variance_yaml.sh /path/to/config_xxx.yaml
# 输出目录名: r2r_<yaml文件名stem，非字母数字转为下划线>
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "${HERE}/00_env.sh"

YAML="${1:?用法: $0 <path/to/router.yaml>}"
YAML_ABS="$(cd "$(dirname "$YAML")" && pwd)/$(basename "$YAML")"
[[ -f "$YAML_ABS" ]] || { echo "文件不存在: $YAML_ABS" >&2; exit 1; }

stem=$(basename "$YAML_ABS" .yaml)
safe=$(echo "$stem" | sed 's/[^a-zA-Z0-9._-]/_/g')
safe="${safe:0:120}"
[[ -n "$safe" ]] || safe="cfg"
_run_hf "r2r_${safe}" --use_hybrid --config-path "$YAML_ABS"
