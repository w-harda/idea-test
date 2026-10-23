#!/usr/bin/env bash
# 仅调用冻结native_cli；不实现训练、Loss、JVP或恢复逻辑。
set -Eeuo pipefail
ROOT=/home/lzf/ldx/projects/idea-test-tbps-gh-probe-shared-20261023
OPS=/home/lzf/ldx/projects/idea-test-tbps-gh-rng-smoke-20261023
CTL=/home/lzf/ldx/tmp/gh_rng_smoke_v2_20261023
PY=/home/lzf/ldx/envs/tbps-image/bin/python
CONFIG="$ROOT/configs/framework/gh_bidirectional_backend.probe_v2.shared.server.json"
export PYTHONPATH="$ROOT/src" PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TMPDIR="$CTL"
cd "$ROOT"
exec 9>"$CTL/execution.lock"
flock -n 9 || { echo "duplicate supervisor owns lock"; exit 90; }
test -f "$CTL/launch_consumed.json"
test ! -e "$CTL/supervisor_started"
set -o noclobber
printf '%s\n' "$(date --iso-8601=seconds)" > "$CTL/supervisor_started"
set +o noclobber
printf '%s\n' "$$" > "$CTL/supervisor.pid"
ps -o pgid= -p "$$" | tr -d ' ' > "$CTL/supervisor.pgid"
ps -o sid= -p "$$" | tr -d ' ' > "$CTL/supervisor.sid"
ps -o pid=,ppid=,pgid=,sid=,tty=,comm= -p "$$" > "$CTL/supervisor_identity"
status() {
  printf '%s\n' "$1" > "$CTL/phase"
  printf '%s | %s\n' "$(date --iso-8601=seconds)" "$1" >> "$CTL/status.log"
}
fail() {
  rc=$?; trap - ERR
  printf '%s | FAIL | exit=%s line=%s\n' "$(date --iso-8601=seconds)" "$rc" "$1" >> "$CTL/status.log"
  printf 'FAIL\n' > "$CTL/phase"
  exit "$rc"
}
trap 'fail "$LINENO"' ERR
status CPU_PREFLIGHT
CUDA_VISIBLE_DEVICES="" "$PY" -B -u -m tbps.alternating.native_cli \
  --config "$CONFIG" --action preflight --check-parent-states > "$CTL/smoke_cpu_preflight.log" 2>&1
grep -q GH_NATIVE_CPU_PREFLIGHT_PASS "$CTL/smoke_cpu_preflight.log"
status FINAL_RESOURCE_AND_RNG_GATE
CUDA_VISIBLE_DEVICES="" "$PY" -B -u "$OPS/operations/gh_smoke_checks.py" \
  --action ready > "$CTL/smoke_ready.json" 2> "$CTL/smoke_ready.err"
grep -q SMOKE_READY "$CTL/smoke_ready.json"
status RUNNING_A1_B_SMOKE
export CUDA_VISIBLE_DEVICES=0
/usr/bin/time -v -o "$CTL/smoke_process_resources.log" \
  "$PY" -B -u -m tbps.alternating.native_cli \
  --config "$CONFIG" --action smoke --allow-gpu-smoke > "$CTL/smoke.log" 2>&1
printf '0\n' > "$CTL/smoke.exit"
grep -q GH_NATIVE_TWO_ROUND_SMOKE_GENERATED "$CTL/smoke.log"
status RUNNING_CPU_AUDIT
CUDA_VISIBLE_DEVICES="" /usr/bin/time -v -o "$CTL/audit_process_resources.log" \
  "$PY" -B -u -m tbps.alternating.native_cli \
  --config "$CONFIG" --action audit-smoke > "$CTL/smoke_cpu_audit.log" 2>&1
printf '0\n' > "$CTL/audit.exit"
grep -q GH_NATIVE_TWO_ROUND_CPU_AUDIT_PASS "$CTL/smoke_cpu_audit.log"
status FINAL_HISTORY_AND_EVIDENCE_CHECK
CUDA_VISIBLE_DEVICES="" "$PY" -B -u "$OPS/operations/gh_smoke_checks.py" \
  --action final > "$CTL/final_evidence.json" 2> "$CTL/final_evidence.err"
status SUCCESS
printf '%s | SUCCESS | GH_TWO_ROUND_SMOKE_AND_CPU_AUDIT\n' "$(date --iso-8601=seconds)" >> "$CTL/status.log"
