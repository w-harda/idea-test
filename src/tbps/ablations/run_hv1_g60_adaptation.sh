#!/usr/bin/env bash
# 仅H适配训练与内部Validation。禁止重试、Test或其他模型评估。
set -Eeuo pipefail
ROOT=/home/lzf/ldx/projects/idea-test-tbps-hv1-g60-adaptation-20261021
CONTROL=/home/lzf/ldx/tmp/hv1_g60_adaptation_epoch60_v1
BASE=/home/lzf/ldx/outputs/tbps-framework-hv1-g60-adaptation-epoch60
PY=/home/lzf/ldx/envs/tbps-image/bin/python
CFG="$ROOT/configs/framework/hv1_g60_adaptation.server.json"
mkdir -p "$CONTROL"
exec 9>"$CONTROL/lock"
flock -n 9 || { echo "duplicate supervisor"; exit 1; }
test ! -e "$CONTROL/started"
test ! -e "$BASE/training"
test -f "$BASE/gallery_audit/result/completed.json"
test -f "$BASE/smoke_audit/result/completed.json"
date -Is > "$CONTROL/started"
echo "$$" > "$CONTROL/supervisor.pid"
ps -o pgid= -p "$$" | tr -d ' ' > "$CONTROL/supervisor.pgid"
ps -o sid= -p "$$" | tr -d ' ' > "$CONTROL/supervisor.sid"
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$ROOT/src"
cd "$ROOT"
phase() { echo "$1" > "$CONTROL/phase"; echo "$(date -Is) | $1" | tee -a "$CONTROL/status.log"; }
trap 'rc=$?; echo "$(date -Is) | FAIL exit=$rc line=$LINENO" >> "$CONTROL/status.log"; echo "FAIL exit=$rc" > "$CONTROL/phase"; exit "$rc"' ERR
phase CPU_PREFLIGHT
CUDA_VISIBLE_DEVICES="" "$PY" -B -m tbps.ablations.hv1_g60_adaptation --config "$CFG" --action preflight > "$CONTROL/preflight.log" 2>&1
CHILD=""
stop_training() {
  if [[ -n "$CHILD" ]] && kill -0 "$CHILD" 2>/dev/null; then
    kill -TERM "$CHILD"
    wait "$CHILD" || true
  fi
  phase PARTIAL_USER_STOP
  exit 0
}
trap stop_training TERM INT
phase RUNNING_TRAIN_E11_E60
CUDA_VISIBLE_DEVICES=0 "$PY" -B -m tbps.ablations.hv1_g60_adaptation --config "$CFG" --action train --allow-gpu --allow-training > "$CONTROL/training.log" 2>&1 &
CHILD=$!
echo "$CHILD" > "$CONTROL/training.pid"
wait "$CHILD"
CHILD=""
grep -q '^HV1_G60_TRAINING_PASS ' "$CONTROL/training.log"
phase RUNNING_CPU_AUDIT
CUDA_VISIBLE_DEVICES="" "$PY" -B -m tbps.ablations.hv1_g60_adaptation --config "$CFG" --action audit > "$CONTROL/cpu_audit.log" 2>&1
grep -q '^HV1_G60_TRAINING_CPU_AUDIT_PASS ' "$CONTROL/cpu_audit.log"
phase WRITING_REPORT
CUDA_VISIBLE_DEVICES="" "$PY" -B -m tbps.ablations.hv1_g60_results > "$CONTROL/report.log" 2>&1
phase PUBLISHING_REPORT
git add -- docs/PROJECT_STATE.md docs/EXPERIMENT_REGISTRY.md docs/framework/results/hv1_g60_adaptation_epoch60
git diff --cached --check
git -c user.name=w-harda -c user.email=wharda@163.com commit -m "docs: 发布H10至H60 G60适配训练及Validation消融"
git push origin codex/hv1-g60-adaptation-epoch60
phase SUCCESS
echo "SUCCESS | HV1_G60_ADAPTATION_EPOCH60_VALIDATION_AND_CPU_AUDIT" | tee -a "$CONTROL/status.log"
