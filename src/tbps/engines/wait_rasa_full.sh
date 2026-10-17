#!/usr/bin/env bash
# 仅等待资源、单次启动现有RaSa Full及CPU审计；不干预任何外部GPU进程。
set -Eeuo pipefail
umask 077

W="${RASA_WORKTREE:-/home/lzf/ldx/projects/idea-test-tbps-rasa-materialized-eval-20261017}"
PY="${RASA_PYTHON:-/home/lzf/ldx/envs/tbps-image/bin/python}"
RUN="${RASA_WAIT_DIR:-/home/lzf/ldx/tmp/rasa_full_wait}"
OUT="${RASA_FULL_OUTPUT:-/home/lzf/ldx/outputs/tbps-framework-rasa-materialized-evaluation/full}"
INTERVAL=60
REQUIRED=3
MIN_FREE=8192
MAX_UTIL=10
phase=WAITING
sleeper=""

status() {
    phase="$1"
    printf '%s\n' "$phase" > "$RUN/phase"
    printf '%s | %s\n' "$(date '+%F %T%z')" "$*" >> "$RUN/status.log"
}

cancel_wait() {
    if [[ "$phase" == WAITING ]]; then
        [[ -z "$sleeper" ]] || kill -TERM "$sleeper" 2>/dev/null || true
        status CANCELLED "waiting cancelled; Full not triggered"
        exit 0
    fi
    printf '%s | cancellation ignored: no longer waiting (%s)\n' "$(date '+%F %T%z')" "$phase" >> "$RUN/status.log"
}

full_running() {
    local pid command
    while read -r pid; do
        [[ -n "$pid" ]] || continue
        command="$(ps -p "$pid" -o args= 2>/dev/null || true)"
        if [[ "$command" == *"--action run"* && "$command" == *"--scope full"* ]]; then
            return 0
        fi
    done < <(pgrep -f '[r]asa_eval.py|tbps.engines.[m]aterialized_rasa' || true)
    return 1
}

sample_gpu() {
    local raw processes free util pid name memory line
    other=0
    free=UNKNOWN
    util=UNKNOWN
    if ! raw="$(timeout 15 nvidia-smi -i 0 --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits 2>&1)"; then
        printf '%s | QUERY_FAILED | %s\n' "$(date '+%F %T%z')" "$raw" >> "$RUN/poll.log"
        return 1
    fi
    if ! processes="$(timeout 15 nvidia-smi -i 0 --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>&1)"; then
        printf '%s | QUERY_FAILED | %s\n' "$(date '+%F %T%z')" "$processes" >> "$RUN/poll.log"
        return 1
    fi
    IFS=, read -r free util <<< "$raw"
    free="${free//[[:space:]]/}"
    util="${util//[[:space:]]/}"
    while IFS=, read -r pid name memory; do
        [[ -n "${pid//[[:space:]]/}" ]] || continue
        name="$(printf '%s' "$name" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
        case "${name##*/}" in Xorg|Xwayland) continue ;; esac
        other=$((other+1))
    done <<< "$processes"
    printf '%s | GPU0 | free_mib=%s util_percent=%s other_compute=%s processes=%s\n' \
        "$(date '+%F %T%z')" "$free" "$util" "$other" "${processes//$'\n'/; }" >> "$RUN/poll.log"
    [[ "$free" =~ ^[0-9]+$ && "$util" =~ ^[0-9]+$ ]] || return 1
    (( free >= MIN_FREE && util <= MAX_UTIL && other == 0 ))
}

wait_tick() {
    sleep "$INTERVAL" &
    sleeper=$!
    wait "$sleeper"
    sleeper=""
}

main() {
    mkdir -p "$RUN"
    exec 9>"$RUN/wait.lock"
    if ! flock -n 9; then
        printf '%s | REFUSED | another waiter/Full supervisor owns lock\n' "$(date '+%F %T%z')" >> "$RUN/launch.log"
        return 73
    fi
    if [[ -e "$RUN/triggered" || -e "$OUT" ]] || full_running; then
        status REFUSED "existing trigger, formal output or running Full; no duplicate launch"
        return 73
    fi
    printf '%s\n' "$$" > "$RUN/wait.pid"
    trap cancel_wait TERM INT
    trap 'rc=$?; trap - ERR; if [[ "$phase" != FAILED && "$phase" != REFUSED && "$phase" != CANCELLED && "$phase" != SUCCESS ]]; then status FAILED "controller error exit=$rc; no retry"; fi; exit "$rc"' ERR
    status WAITING "pid=$$ interval=60 free>=8192 util<=10 no-compute consecutive=3"
    local consecutive=0
    while true; do
        if [[ -e "$OUT" ]] || full_running; then
            status REFUSED "Full appeared while waiting; exiting without launch"
            return 73
        fi
        if sample_gpu; then consecutive=$((consecutive+1)); else consecutive=0; fi
        printf '%s | STREAK | %s/%s\n' "$(date '+%F %T%z')" "$consecutive" "$REQUIRED" >> "$RUN/poll.log"
        if (( consecutive >= REQUIRED )); then break; fi
        wait_tick
    done
    # 持锁再次检查；noclobber持久化一次性触发记录，失败不会自动重试。
    if [[ -e "$OUT" ]] || full_running; then
        status REFUSED "Full already exists at launch boundary"
        return 73
    fi
    if ! (set -o noclobber; printf '%s | pid=%s\n' "$(date '+%F %T%z')" "$$" > "$RUN/triggered"); then
        status REFUSED "single-use trigger already consumed"
        return 73
    fi
    status RUNNING_FULL "resources satisfied 3 times; starting one Full"
    printf '%s | START_FULL | worktree=%s output=%s\n' "$(date '+%F %T%z')" "$W" "$OUT" >> "$RUN/launch.log"
    export TMPDIR=/home/lzf/ldx/tmp PYTHONDONTWRITEBYTECODE=1
    export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    export PYTHONPATH="$W/src"
    cd "$W"
    if CUDA_VISIBLE_DEVICES=0 "$PY" -B -u "$W/src/tbps/rasa_eval.py" \
        --config "$W/configs/framework/materialized_rasa.server.json" \
        --action run --scope full --allow-full-evaluation > "$RUN/full_gpu.log" 2>&1; then
        printf '%s | FULL_EXIT | 0\n' "$(date '+%F %T%z')" >> "$RUN/launch.log"
    else
        local rc=$?
        status FAILED "Full exit=$rc; no retry"
        printf '%s | FULL_EXIT | %s\n' "$(date '+%F %T%z')" "$rc" >> "$RUN/launch.log"
        return "$rc"
    fi
    if ! grep -Fxq 'RASA_MATERIALIZED_EVALUATION_PASS full' "$RUN/full_gpu.log"; then
        status FAILED "Full success marker missing; audit not started; no retry"
        return 1
    fi
    status RUNNING_CPU_AUDIT "Full PASS; starting independent CPU audit"
    if CUDA_VISIBLE_DEVICES="" "$PY" -B -u "$W/src/tbps/rasa_eval.py" \
        --config "$W/configs/framework/materialized_rasa.server.json" \
        --action audit --scope full > "$RUN/cpu_audit.log" 2>&1; then
        if grep -Fxq 'RASA_MATERIALIZED_CPU_AUDIT_PASS full' "$RUN/cpu_audit.log"; then
            status SUCCESS "RASA_FULL_AND_CPU_AUDIT_SUCCESS"
            printf '%s | SUCCESS | RASA_FULL_AND_CPU_AUDIT_SUCCESS\n' "$(date '+%F %T%z')" >> "$RUN/launch.log"
            return 0
        fi
        status FAILED "CPU audit success marker missing; no retry"
        return 1
    else
        local rc=$?
        status FAILED "CPU audit exit=$rc; no retry"
        return "$rc"
    fi
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then main "$@"; fi
