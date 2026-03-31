#!/usr/bin/env bash
# 实时查看 hybrid 评测：日志里的吞吐 / 探针 + GPU 占用（两个窗口各跑一行命令即可）。
#
# 用法：
#   终端 A — 只看关键行（推荐）：
#     bash script/evaluate/probe_hybrid_live.sh /path/to/nohup.log
#
#   终端 B — GPU 1Hz 刷新：
#     watch -n1 'nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv'
#
# 评测进程需带探针（在启动评测前 export）：
#   export R2R_PERF_PROBE=1
#   export R2R_TPS_LOG_INTERVAL=1.5    # 吞吐行间隔（秒），默认 4；越小越“实时”
#
set -euo pipefail
LOG="${1:?用法: $0 <nohup_or_eval.log>}"
if [[ ! -f "$LOG" ]]; then
  echo "文件不存在: $LOG" >&2
  exit 1
fi
echo "==> tail -F (grep 过滤)  $LOG"
echo "    若长时间无输出，确认评测已设置 R2R_PERF_PROBE=1 且日志路径正确"
echo ""
stdbuf -oL tail -F "$LOG" | stdbuf -oL grep --line-buffered -E \
  'R2R_PERF_PROBE|throughput:|current tokens/s|Processing batches|SlidingWindow|SLDisaggregation|Error|Traceback|CUDA out|Killed'
