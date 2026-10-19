#!/bin/bash
# 顺序运行多个标注配置（每臂由 watchdog 守护：崩溃/卡死自动重启续跑）。
# 用法（服务器项目根目录 source server-env.sh 后）:
#   nohup bash scripts/chain.sh configs/ablation/full.yaml configs/probes/title-only.yaml ... \
#     >> /tmp/v3chain.log 2>&1 < /dev/null &
set -u
[ $# -ge 1 ] || { echo "用法: chain.sh <config...>"; exit 1; }
cd "$(dirname "$0")/.."
PY=.venv/bin/python
for CFG in "$@"; do
  EXP=$($PY -c "import sys; sys.path.insert(0, '.'); from genometag.config import load_config; print(load_config('$CFG').exp_name)")
  echo "[chain] $(date '+%F %T') === $CFG (exp=$EXP) 开始 ==="
  bash scripts/watchdog.sh "$CFG" "results/$EXP/annotations.jsonl" 900
  RC=$?
  if [ "$RC" -ne 0 ]; then
    echo "[chain] $CFG 看门狗放弃（rc=$RC）——继续下一臂"
  fi
done
echo "[chain] $(date '+%F %T') 链完成（$# 个配置）"
