#!/usr/bin/env bash
# 仅在另行批准Full后启动；无轮询、无重试，不修改冻结算法。
set -Eeuo pipefail
[[ "$#" == 1 && "$1" == "--allow-full" ]] || { echo "需要明确 --allow-full；默认不启动实验"; exit 2; }
ROOT=/home/lzf/ldx/projects/idea-test-tbps-vlpt-full-20261019
FROZEN=/home/lzf/ldx/projects/idea-test-tbps-vlpt-full-20261019
SOURCE_PY=/home/lzf/ldx/envs/dra-irra-source/bin/python
TARGET_PY=/home/lzf/ldx/envs/tbps-image/bin/python
CONFIG="$FROZEN/configs/framework/dra_irra_full.server.json"
OUT=/home/lzf/ldx/outputs/tbps-framework-vlpt-official-irra/full
CONTROL=/home/lzf/ldx/tmp/vlpt_official_irra_full_v1
mkdir -p "$CONTROL"
exec 9>"$CONTROL/supervisor.lock"
flock -n 9 || { echo "已有同一Full监督器；拒绝重复启动"; exit 3; }
[[ ! -e "$CONTROL/started" && ! -e "$OUT" ]] || { echo "已有启动记录或Full输出；保留现场并拒绝重入"; exit 4; }
umask 077
(set -o noclobber; printf '%s\n' "$(date -Is)" > "$CONTROL/started")
printf '%s\n' "$$" > "$CONTROL/supervisor.pid"
ps -o pgid= -p "$$" | tr -d ' ' > "$CONTROL/supervisor.pgid"
ps -o sid= -p "$$" | tr -d ' ' > "$CONTROL/supervisor.sid"
git -C "$ROOT" rev-parse HEAD > "$CONTROL/code.commit"
export PYTHONPATH="$ROOT/src" PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd "$ROOT"
phase=STARTING
child=""
status() { printf '%s | %s\n' "$(date -Is)" "$*" | tee -a "$CONTROL/status.log"; }
failure() { local code="$1"; printf 'FAIL\n' > "$CONTROL/phase"; status "FAIL | phase=$phase exit=$code"; exit "$code"; }
trap 'code=$?; failure "$code"' ERR
trap '[[ -z "$child" ]] || kill -TERM "$child" 2>/dev/null || true; failure 130' INT TERM HUP
sha256sum "$0" "$ROOT/src/tbps/baselines/dra_full.py" "$CONFIG" > "$CONTROL/launch_sources.sha256"
stage() {
    phase="$1"; local log="$2" marker="$3"; shift 3
    printf '%s\n' "$phase" > "$CONTROL/phase"; status "BEGIN | $phase"
    "$@" > "$log" 2>&1 & child=$!
    printf '%s\n' "$child" > "$CONTROL/current_child.pid"
    if wait "$child"; then child=""; else local rc=$?; child=""; failure "$rc"; fi
    grep -Fq "$marker" "$log" || failure 5
    status "PASS | $phase"
}
stage CPU_PREFLIGHT "$CONTROL/preflight.log" DRA_FULL_CPU_PREFLIGHT_PASS env CUDA_VISIBLE_DEVICES="" "$TARGET_PY" -B -u -m tbps.baselines.dra_full --config "$CONFIG"
nvidia-smi > "$CONTROL/gpu_start.log"
free -b > "$CONTROL/ram_start.log"
df -B1 "$(dirname "$(dirname "$OUT")")" > "$CONTROL/disk_start.log"
df -i "$(dirname "$(dirname "$OUT")")" > "$CONTROL/inodes_start.log"
available=$(df -B1 --output=avail "$(dirname "$(dirname "$OUT")")" | tail -n 1 | tr -d ' ')
[[ "$available" -ge 37580963840 ]] || failure 6
ram_kib=$(awk '/MemAvailable:/{print $2}' /proc/meminfo)
[[ "$ram_kib" -ge 50331648 ]] || failure 7
env CUDA_VISIBLE_DEVICES="" "$TARGET_PY" -B -u -m tbps.baselines.dra_full --config "$CONFIG" --action resources > "$CONTROL/resources.log" 2>&1
mkdir -p "$OUT"
stage RUNNING_SOURCE "$OUT/source.log" "DRA_OFFICIAL_IRRA_SOURCE_PASS full" env CUDA_VISIBLE_DEVICES=0 "$SOURCE_PY" -B -u -m tbps.baselines.dra_full --config "$CONFIG" --action source --allow-full
for model in irra rasa aptm; do
    stage "RUNNING_${model^^}" "$OUT/$model.log" "DRA_NATIVE_TARGET_PASS full $model" env CUDA_VISIBLE_DEVICES=0 "$TARGET_PY" -B -u -m tbps.baselines.dra_full --config "$CONFIG" --action evaluate --model "$model" --allow-full
done
stage RUNNING_CPU_AUDIT "$OUT/audit.log" "DRA_INDEPENDENT_CPU_AUDIT_PASS full" env CUDA_VISIBLE_DEVICES="" "$TARGET_PY" -B -u -m tbps.baselines.dra_full --config "$CONFIG" --action audit
phase=SUCCESS; printf '%s\n' "$phase" > "$CONTROL/phase"
status "SUCCESS | VLPTATTACK_OFFICIAL_IRRA_SOURCE_THREE_TARGET_FULL_AND_CPU_AUDIT"
