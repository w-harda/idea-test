#!/usr/bin/env bash
# 用户确认后手动启动；默认拒绝；不调度GPU、不重试、不训练。
set -Eeuo pipefail
[[ "${1:-}" == "--allow-full" && "$#" == 1 ]] || { echo "需要显式 --allow-full"; exit 2; }
WORK=/home/lzf/ldx/projects/idea-test-tbps-baseline-b3-20261018
PY=/home/lzf/ldx/envs/tbps-image/bin/python
CFG="$WORK/configs/framework/baseline_b3_ready.server.json"
BASE=/home/lzf/ldx/outputs/tbps-framework-b3-comparison-ready
cd "$WORK"
export PYTHONPATH="$WORK/src" PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0
export TMPDIR=/home/lzf/ldx/tmp
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
mkdir -p "$BASE/logs"
exec 9>"$BASE/.full-launch.lock"
flock -n 9 || { echo "已有B3 Full监督器"; exit 3; }
[[ ! -e "$BASE/full" ]] || { echo "Full目录已存在，禁止覆盖/自动重试"; exit 4; }
trap 'rc=$?; echo "$(date -Is) B3_FULL_FAILED exit=$rc"; exit "$rc"' ERR
# 三目标Smoke及独立CPU审计必须先完成；本轮资源受阻不能跳过此门禁。
for model in irra rasa aptm; do
  [[ -f "$BASE/smoke/$model/completed/summary.json" ]] || { echo "缺少成功Smoke: $model；禁止Full"; exit 5; }
  CUDA_VISIBLE_DEVICES="" "$PY" -B -m tbps.baselines.launch --config "$CFG" --scope smoke --action audit --model "$model" >"$BASE/logs/${model}_smoke_audit_before_full.log" 2>&1
done
"$PY" -B -m tbps.baselines.launch --config "$CFG" --scope full --action preflight >"$BASE/logs/preflight.log" 2>&1
"$PY" -B -m tbps.baselines.launch --config "$CFG" --scope full --action pixels --allow-full-pixel-transport >"$BASE/logs/pixels.log" 2>&1
for model in irra rasa aptm; do
  echo "$(date -Is) RUNNING $model"
  "$PY" -B -m tbps.baselines.launch --config "$CFG" --scope full --action run --model "$model" --allow-full-evaluation >"$BASE/logs/${model}_full.log" 2>&1
  grep -q "B3_EVALUATION_PASS full $model" "$BASE/logs/${model}_full.log"
  CUDA_VISIBLE_DEVICES="" "$PY" -B -m tbps.baselines.launch --config "$CFG" --scope full --action audit --model "$model" >"$BASE/logs/${model}_audit.log" 2>&1
  grep -q "B3_CPU_AUDIT_PASS full $model" "$BASE/logs/${model}_audit.log"
done
echo "$(date -Is) SUCCESS | B3_THREE_MODEL_FULL_AND_CPU_AUDIT"
