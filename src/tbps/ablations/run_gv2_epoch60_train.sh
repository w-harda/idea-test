#!/usr/bin/env bash
# 只管理冻结训练入口、完成后的CPU审计与报告；不包含训练算法或自动恢复。
set -Eeuo pipefail
ROOT=/home/lzf/ldx/projects/idea-test-tbps-gv2-epoch60-ablation-20261020
MANAGER=/home/lzf/ldx/projects/idea-test-tbps-gv2-epoch60-launch-20261020
CONTROL=/home/lzf/ldx/tmp/gv2_epoch60_train_v1
OUT=/home/lzf/ldx/outputs/tbps-framework-gv2-epoch60-ablation/train
CONFIG="$ROOT/configs/framework/gv2_epoch60.server.json"
PY=/home/lzf/ldx/envs/tbps-image/bin/python
PIN=635471b6d8a722616f87df16c300a8358861cd10
[[ "${1:-}" == --allow-training ]] || { echo "explicit authorization required"; exit 2; }
[[ -d "$CONTROL" && ! -e "$OUT" && ! -e "$CONTROL/started" ]]
[[ "$(git -C "$ROOT" rev-parse HEAD)" == "$PIN" ]]
[[ -z "$(git -C "$ROOT" status --porcelain)" ]]
[[ "$(cat "$CONTROL/preflight.exit")" == 0 ]]
grep -q '^GV2_EPOCH60_CPU_PREFLIGHT_PASS ' "$CONTROL/cpu_preflight.log"
exec 9>"$CONTROL/supervisor.lock"
flock -n 9 || exit 3
(set -o noclobber; date --iso-8601=seconds > "$CONTROL/started")
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$ROOT/src" CUDA_VISIBLE_DEVICES=0
cd "$ROOT"
child=0
stop=0
phase() {
    printf '%s\n' "$1" > "$CONTROL/phase.tmp"
    mv "$CONTROL/phase.tmp" "$CONTROL/phase"
    printf '%s | %s\n' "$(date --iso-8601=seconds)" "$*" >> "$CONTROL/status.log"
}
failed() {
    local rc=$?
    trap - ERR
    phase "FAIL exit=$rc line=${BASH_LINENO[0]:-unknown}"
    exit "$rc"
}
trap failed ERR
request_stop() {
    stop=1
    phase STOP_REQUESTED_EPOCH_BOUNDARY
    if (( child > 0 )); then kill -TERM "$child" 2>/dev/null || true; fi
}
trap request_stop TERM INT
run_child() {
    local logfile=$1
    shift
    "$@" >> "$logfile" 2>&1 &
    child=$!
    printf '%s\n' "$child" > "$CONTROL/child.pid"
    local rc=0
    while true; do
        if wait "$child"; then rc=0; break; else rc=$?; fi
        if (( stop )) && kill -0 "$child" 2>/dev/null; then continue; fi
        break
    done
    child=0
    return "$rc"
}
printf '%s\n' "$$" > "$CONTROL/supervisor.pid"
ps -o pgid= -p "$$" | tr -d ' ' > "$CONTROL/supervisor.pgid"
ps -o sid= -p "$$" | tr -d ' ' > "$CONTROL/supervisor.sid"
ps -o pid,ppid,pgid,sid,tty,lstart,args -p "$$" > "$CONTROL/process.txt"
git -C "$MANAGER" rev-parse HEAD > "$CONTROL/launcher.commit"
sha256sum "$0" "$MANAGER/src/tbps/ablations/gv2_epoch60_results.py" > "$CONTROL/launcher_sources.sha256"
phase CPU_PREFLIGHT_PASS
phase RUNNING_TRAIN
run_child "$CONTROL/train.log" "$PY" -B -u -m tbps.ablations.gv2_epochs --config "$CONFIG" --action train --allow-training
if (( stop )); then
    phase STOPPED_EPOCH_BOUNDARY
    exit 0
fi
grep -q '^GV2_EPOCH60_TRAINING_PASS$' "$CONTROL/train.log"
[[ -f "$OUT/completed/report.json" ]]
phase RUNNING_CPU_AUDIT
run_child "$CONTROL/cpu_audit.log" env CUDA_VISIBLE_DEVICES= "$PY" -B -u -m tbps.ablations.gv2_epochs --config "$CONFIG" --action train-audit
grep -q '^GV2_EPOCH60_TRAIN_CPU_AUDIT_PASS$' "$CONTROL/cpu_audit.log"
if (( stop )); then phase STOPPED; exit 0; fi
phase WRITING_TRAINING_REPORT
run_child "$CONTROL/report.log" env CUDA_VISIBLE_DEVICES= PYTHONPATH="$MANAGER/src" "$PY" -B "$MANAGER/src/tbps/ablations/gv2_epoch60_results.py" --control "$CONTROL"
phase PUBLISHING_REPORT
cd "$MANAGER"
git diff --check
[[ -z "$(git diff --cached --name-only)" ]]
git add -- docs/framework/results/gv2_epoch60_training docs/PROJECT_STATE.md docs/EXPERIMENT_REGISTRY.md
git diff --cached --check
git -c user.name=w-harda -c user.email=wharda@163.com commit -m "docs: publish completed G-v2 Epoch11-60 training audit"
git push origin codex/gv2-epoch60-launch
git rev-parse HEAD > "$CONTROL/completion.commit"
phase SUCCESS
printf '%s | SUCCESS | GV2_EPOCH60_TRAINING_AND_CPU_AUDIT\n' "$(date --iso-8601=seconds)" >> "$CONTROL/status.log"
