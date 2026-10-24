#!/usr/bin/env bash
# 一次性B入口；此脚本不轮询、不重试，不启动Test或Full评估。
set -Eeuo pipefail
ROOT=/home/lzf/ldx/projects/idea-test-tbps-gh-formal-b-20261024
PY=/home/lzf/ldx/envs/tbps-image/bin/python
CFG="$ROOT/configs/framework/gh_b_formal.server.json"
SCOPE=${1:-}
case "$SCOPE" in pilot|train) ;; *) echo "usage: $0 pilot|train [--allow-training]"; exit 2 ;; esac
CTL="/home/lzf/ldx/tmp/gh_formal_b_${SCOPE}_v1"
mkdir -p "$CTL"
exec 9>"$CTL/supervisor.lock"
flock -n 9 || { echo "duplicate supervisor"; exit 2; }
[[ ! -e "$CTL/started" ]] || { echo "start consumed; no automatic retry"; exit 2; }
[[ "$SCOPE" != train || "${2:-}" == --allow-training ]] || { echo "explicit training flag required"; exit 2; }
export PYTHONPATH="$ROOT/src" PYTHONDONTWRITEBYTECODE=1
cd "$ROOT"
# 原生入口仍执行全部来源/资源/唯一Attempt门禁；监督器不替代原门禁。
"$PY" -B "$ROOT/operations/gh_formal_b.py" --config "$CFG" --action resources --scope "$SCOPE" > "$CTL/preflight.log" 2>&1
if [[ "$SCOPE" == train ]]; then
 "$PY" -B -c 'from tbps.alternating.formal_sources import load_formal; from tbps.alternating.formal_cli import ready_for_training; import sys; assert ready_for_training(load_formal(sys.argv[1]))' "$CFG"
fi
(set -o noclobber; printf '%s\n' "$(date -Iseconds)" > "$CTL/started")
printf '%s\n' "$$" > "$CTL/supervisor.pid"
ps -o pid=,ppid=,pgid=,sid=,tty= -p "$$" > "$CTL/process.txt"
trap 'rc=$?; printf "%s | FAIL exit=%s line=%s\n" "$(date -Iseconds)" "$rc" "$LINENO" >> "$CTL/status.log"; exit "$rc"' ERR
printf '%s | RUNNING_%s\n' "$(date -Iseconds)" "$SCOPE" >> "$CTL/status.log"
ARGS=(--config "$CFG" --action "$SCOPE")
if [[ "$SCOPE" == pilot ]]; then ARGS+=(--allow-gpu-pilot); else ARGS+=(--allow-training); fi
if "$PY" -u -B "$ROOT/operations/gh_formal_b.py" "${ARGS[@]}" > "$CTL/train.log" 2>&1; then
 :
else
 rc=$?
 if [[ "$rc" == 3 ]]; then
  printf '%s | STOPPED_COMMITTED_BOUNDARY\n' "$(date -Iseconds)" >> "$CTL/status.log"
  exit 0
 fi
 printf '%s | FAIL exit=%s\n' "$(date -Iseconds)" "$rc" >> "$CTL/status.log"
 exit "$rc"
fi
"$PY" -u -B "$ROOT/operations/gh_formal_b.py" --config "$CFG" --action audit --scope "$SCOPE" > "$CTL/cpu_audit.log" 2>&1
printf '%s | SUCCESS | GH_B_%s_AND_CPU_AUDIT\n' "$(date -Iseconds)" "$SCOPE" >> "$CTL/status.log"
