#!/usr/bin/env bash
# 用 nohup 在后台跑任意 distributed_jobs 下的评测脚本，日志写入当前 suite 的 logs/。
#
# 用法（先设好环境变量，与直接跑 run_01_slm.sh 相同）：
#   export SUITE_NAME=my_amc_run
#   export DATASET=amc
#   export CUDA_VISIBLE_DEVICES=0,1
#   bash script/evaluate/distributed_jobs/run_job_nohup.sh run_01_slm.sh
# 需要参数的脚本会把多余参数原样传入，例如：
#   bash .../run_job_nohup.sh run_05_r2r_entropy_variance_yaml.sh config/....yaml
#
# 可选：自定义日志目录
#   export R2R_NOHUP_LOG_DIR=/path/to/logs
#
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
source "${HERE}/00_env.sh"

JOB="${1:?用法: $0 <脚本名> [脚本参数...]  例如: $0 run_01_slm.sh}"
shift
[[ "$JOB" == *.sh ]] || JOB="${JOB}.sh"
JOB_PATH="${HERE}/${JOB}"
[[ -f "$JOB_PATH" ]] || { echo "找不到: $JOB_PATH" >&2; exit 1; }

LOG_ROOT="${R2R_NOHUP_LOG_DIR:-${SUITE_DIR}/logs}"
mkdir -p "$LOG_ROOT"
stamp="$(date +%Y%m%d_%H%M%S)"
log_base="${JOB%.sh}"
LOG_FILE="${LOG_ROOT}/nohup_${log_base}_${stamp}.log"

echo "Suite:     ${SUITE_DIR}"
echo "任务脚本:  ${JOB}"
echo "日志文件:  ${LOG_FILE}"
echo "启动中..."

nohup bash "$JOB_PATH" "$@" >"$LOG_FILE" 2>&1 &
pid=$!

echo "PID:       ${pid}"
echo "查看日志:  tail -f '${LOG_FILE}'"
