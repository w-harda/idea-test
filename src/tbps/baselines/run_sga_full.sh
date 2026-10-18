#!/usr/bin/env bash
# 用户手动启动；SGA每目标单独攻击，失败立即停止，无GPU轮询或重试。
set -Eeuo pipefail
[[ "$#" == 1 ]] || { echo "需要显式 --allow-full"; exit 2; }
[[ "$1" == "--allow-full" ]] || exit 2
WORK=/home/lzf/ldx/projects/idea-test-tbps-sga-whitebox-fixed-20261018
PY=/home/lzf/ldx/envs/tbps-image/bin/python
CFG="$WORK/configs/framework/sga_whitebox.server.json"
CONTROL=/home/lzf/ldx/tmp/sga_whitebox_full_v2
BASE=/home/lzf/ldx/outputs/tbps-framework-sga-whitebox-v2
mkdir -p "$CONTROL"
exec 9>"$CONTROL/pipeline.lock"
flock -n 9 || { echo "已有SGA Full链"; exit 3; }
[[ ! -e "$CONTROL/started" && ! -e "$BASE/full" ]] || { echo "已有Full/启动许可，禁止覆盖或重试"; exit 4; }
cd "$WORK"
export PYTHONPATH="$WORK/src" PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0
export TMPDIR=/home/lzf/ldx/tmp OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
(set -o noclobber; date -Is >"$CONTROL/started")
echo "$$" >"$CONTROL/supervisor.pid"
ps -o pgid= -p "$$" | tr -d ' ' >"$CONTROL/supervisor.pgid"
CURRENT=START
state() { CURRENT="$1"; echo "$CURRENT" >"$CONTROL/phase"; printf '%s | %s\n' "$(date -Is)" "$CURRENT" | tee -a "$CONTROL/status.log"; }
trap 'state FAIL; exit 1' ERR
trap 'state CANCELLED; exit 143' TERM INT
for model in irra rasa aptm; do
  state "CHECK_SMOKE_$model"
  env CUDA_VISIBLE_DEVICES="" "$PY" -B -m tbps.sga_eval --config "$CFG" --action audit --scope smoke --model "$model" >"$CONTROL/$model-smoke-check.log" 2>&1
  grep -Fq "SGA_TBPS_CPU_AUDIT_PASS smoke $model" "$CONTROL/$model-smoke-check.log"
done
for model in irra rasa aptm; do
  state "RUNNING_FULL_$model"
  "$PY" -B -m tbps.sga_eval --config "$CFG" --action run --scope full --model "$model" --allow-full >"$CONTROL/$model-full.log" 2>&1
  grep -Fq "SGA_TBPS_EVALUATION_PASS full $model" "$CONTROL/$model-full.log"
  state "RUNNING_FULL_AUDIT_$model"
  env CUDA_VISIBLE_DEVICES="" "$PY" -B -m tbps.sga_eval --config "$CFG" --action audit --scope full --model "$model" >"$CONTROL/$model-full-audit.log" 2>&1
  grep -Fq "SGA_TBPS_CPU_AUDIT_PASS full $model" "$CONTROL/$model-full-audit.log"
done
state SUCCESS
echo "SUCCESS | SGA_THREE_MODEL_FULL_AND_CPU_AUDIT" | tee -a "$CONTROL/status.log"
