#!/usr/bin/env bash
# 独立一次性APTM实验链；仅调用原CLI，不改RaSa/模型/数据，不自动重试。
set -Eeuo pipefail
umask 077
W="${APTM_WORKTREE:-/home/lzf/ldx/projects/idea-test-tbps-aptm-materialized-eval-20261018}"
PY="${APTM_PYTHON:-/home/lzf/ldx/envs/tbps-image/bin/python}"
RUN="${APTM_WAIT_DIR:-/home/lzf/ldx/tmp/aptm_full_wait}"
OUT="${APTM_OUTPUT_BASE:-/home/lzf/ldx/outputs/tbps-framework-aptm-materialized-evaluation}"
RASA="${APTM_RASA_WAIT_DIR:-/home/lzf/ldx/tmp/rasa_full_wait}"
RASA_OUT="${APTM_RASA_FULL_DIR:-/home/lzf/ldx/outputs/tbps-framework-rasa-materialized-evaluation/full}"
INTERVAL=60
MIN_FREE=20000
MAX_UTIL=10
REQUIRED=3
phase=WAITING_RASA
sleeper=""
note() { local file="$1"; shift; printf '%s | %s\n' "$(date '+%F %T%z')" "$*" >> "$RUN/$file.log"; }
status() {
    phase="$1"
    printf '%s\n' "$phase" > "$RUN/phase"
    note status "$*"
}
fail() { status FAIL "$*; no retry"; note launch "FAIL | $*"; }
cancel_wait() {
    case "$phase" in
        WAITING_RASA|WAITING_GPU)
            [[ -z "$sleeper" ]] || kill -TERM "$sleeper" 2>/dev/null || true
            fail "cancelled while waiting; existing artifacts retained"; exit 0 ;;
        *) note status "TERM/INT ignored outside waiting phase=$phase" ;;
    esac
}
tick() { sleep "$INTERVAL" & sleeper=$!; wait "$sleeper"; sleeper=""; }
model_running() {
    local kind="$1" pid command pattern
    if [[ "$kind" == rasa ]]; then pattern='[r]asa_eval.py|tbps.engines.[m]aterialized_rasa'
    else pattern='[a]ptm_eval.py|tbps.engines.[m]aterialized_aptm'; fi
    while read -r pid; do
        [[ -n "$pid" ]] || continue
        command="$(ps -p "$pid" -o args= 2>/dev/null || true)"
        if [[ "$command" == *"--action run"* || "$command" == *"--action audit"* ]]; then return 0; fi
    done < <(pgrep -f "$pattern" || true)
    return 1
}
collision() {
    local stage="$1"
    model_running aptm && return 0
    [[ -e "$OUT/full" ]] && return 0
    [[ "$stage" != smoke || ! -e "$OUT/smoke" ]] || return 0
    return 1
}
gpu_ready() {
    local raw processes free util pid name memory other=0
    if ! raw="$(timeout 15 nvidia-smi -i 0 --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits 2>&1)"; then
        note gpu "QUERY_FAILED | $raw"; return 1
    fi
    if ! processes="$(timeout 15 nvidia-smi -i 0 --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>&1)"; then
        note gpu "QUERY_FAILED | $processes"; return 1
    fi
    IFS=, read -r free util <<< "$raw"
    free="${free//[[:space:]]/}"; util="${util//[[:space:]]/}"
    while IFS=, read -r pid name memory; do
        [[ -n "${pid//[[:space:]]/}" ]] || continue
        name="$(printf '%s' "$name" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
        case "${name##*/}" in Xorg|Xwayland) continue ;; esac
        other=$((other+1))
    done <<< "$processes"
    note gpu "GPU0 free_mib=$free util_percent=$util other_compute=$other processes=${processes//$'\n'/; }"
    [[ "$free" =~ ^[0-9]+$ && "$util" =~ ^[0-9]+$ ]] || return 1
    (( free >= MIN_FREE && util <= MAX_UTIL && other == 0 ))
}
verify_rasa_receipts() {
    # 标准库仅校验小型完成/执行回执，不加载模型、特征或重新审计/排名。
    python3 -B - "$RASA_OUT" "$RASA" "$RUN" <<'PY'
from pathlib import Path
import hashlib,json,sys,datetime
root,control,out=map(Path,sys.argv[1:])
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def verify(folder):
    manifest=folder/"manifest.json"; receipt=folder/"completed.json"
    m=json.loads(manifest.read_text());r=json.loads(receipt.read_text())
    assert m["schema"]=="tbps-artifact-1" and r["schema"]=="tbps-artifact-receipt-1"
    assert r["manifest_sha256"]==sha(manifest)
    for name,h in m["files_sha256"].items():
        p=(folder/name).resolve();assert p.is_relative_to(folder.resolve())
        assert sha(p)==h
    for ref in m["sources"]:assert sha(Path(ref["path"]))==ref["sha256"]
    return m
try:
    m=verify(root/"completed")
    assert m["artifact_id"]=="completed" and m["provenance"]["scope"]=="full"
    summary=json.loads((root/"completed/summary.json").read_text())
    assert (summary["gallery_count"],summary["query_count"])==(3074,6156)
    assert set(summary["groups"])=={"Clean","Image-only","Text-only","Joint"}
    executions=sorted(root.glob("execution-*"));assert executions
    verify(executions[-1])
    report=json.loads((executions[-1]/"report.json").read_text())
    assert report["status"]=="PASS" and report["model_frozen"] is True
    assert not any(root.glob("*.pending*"))
    paths=[root/"completed/manifest.json",root/"completed/completed.json",
           executions[-1]/"manifest.json",executions[-1]/"report.json",
           control/"phase",control/"launch.log",control/"cpu_audit.log"]
    evidence={"schema":"aptm-readonly-rasa-confirmation-1",
              "confirmed_at":datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "status":"SUCCESS | RASA_FULL_AND_CPU_AUDIT",
              "legacy_marker":"RASA_FULL_AND_CPU_AUDIT_SUCCESS accepted explicitly",
              "not_a_new_rasa_audit":True,"sha256":{str(p):sha(p) for p in paths}}
    dest=out/"rasa_verified.json"
    if dest.exists():
        old=json.loads(dest.read_text());assert old["sha256"]==evidence["sha256"]
    else:
        with dest.open("x") as f:json.dump(evidence,f,ensure_ascii=False,sort_keys=True)
    print("RASA_COMPLETION_AND_AUDIT_EVIDENCE_VERIFIED")
except Exception as e:
    print("RASA_RECEIPT_INVALID",repr(e),file=sys.stderr);sys.exit(2)
PY
}
rasa_ready() {
    local state
    state="$(cat "$RASA/phase" 2>/dev/null || true)"
    case "$state" in
        FAIL|FAILED|REFUSED|CANCELLED)
            note poll "RASA_FAILED phase=$state detail=$(tail -n 3 "$RASA/status.log" 2>/dev/null || true)"; return 2 ;;
    esac
    [[ "$state" == SUCCESS ]] || return 1
    # 旧监督器实际后缀为_SUCCESS，明确兼容，不改写原日志或假定新的JSON审计回执。
    grep -Eq '\| SUCCESS \| RASA_FULL_AND_CPU_AUDIT(_SUCCESS)?$' "$RASA/launch.log" || return 1
    grep -Fxq 'RASA_MATERIALIZED_EVALUATION_PASS full' "$RASA/full_gpu.log" || return 1
    grep -Fxq 'RASA_MATERIALIZED_CPU_AUDIT_PASS full' "$RASA/cpu_audit.log" || return 1
    if ! verify_rasa_receipts >> "$RUN/poll.log" 2>&1; then return 2; fi
    model_running rasa && return 1
    return 0
}
wait_resources() {
    local stage="$1" streak=0 ready code gpu_ok
    while true; do
        if collision "$stage"; then fail "existing APTM output/process before $stage"; return 73; fi
        ready=0
        if rasa_ready; then ready=1
        else
            code=$?
            if (( code == 2 )); then fail "RaSa failure or invalid completion/audit evidence; see poll.log"; return 1; fi
        fi
        gpu_ok=0; if gpu_ready; then gpu_ok=1; fi
        if (( ready == 0 )); then
            [[ "$phase" == WAITING_RASA ]] || status WAITING_RASA "RaSa not fully completed/exited; stage=$stage"
            streak=0
        else
            [[ "$phase" == WAITING_GPU ]] || status WAITING_GPU "stage=$stage free>=20000 util<=10 no-compute 3 consecutive"
            if (( gpu_ok )); then streak=$((streak+1)); else streak=0; fi
        fi
        note poll "stage=$stage rasa_ready=$ready gpu_ready=$gpu_ok streak=$streak/3"
        if (( streak >= REQUIRED )); then return 0; fi
        tick
    done
}
execute_stage() {
    local scope="$1" action="$2" stage="$3" file="$4" marker="$5" rc
    local flags=()
    if [[ "$action" == run ]]; then
        if [[ "$scope" == smoke ]]; then flags=(--allow-smoke); else flags=(--allow-full-evaluation); fi
    fi
    status "$stage" "scope=$scope action=$action"
    note launch "START | $stage"
    if CUDA_VISIBLE_DEVICES="$([[ "$action" == run ]] && echo 0 || true)" "$PY" -B -u "$W/src/tbps/aptm_eval.py" \
        --config "$W/configs/framework/materialized_aptm.server.json" --action "$action" --scope "$scope" "${flags[@]}" \
        > "$RUN/$file.log" 2>&1; then rc=0; else rc=$?; fi
    note launch "EXIT | $stage rc=$rc"
    if (( rc != 0 )); then fail "$stage exit=$rc; see $file.log"; return "$rc"; fi
    if ! grep -Fxq "$marker" "$RUN/$file.log"; then fail "$stage success marker missing"; return 1; fi
    if [[ ! -f "$OUT/$scope/completed/manifest.json" || ! -f "$OUT/$scope/completed/completed.json" ]]; then
        fail "$stage completion receipts missing"; return 1
    fi
}
main() {
    mkdir -p "$RUN"
    exec 9>"$RUN/wait.lock"
    if ! flock -n 9; then note launch "REFUSED | another APTM chain owns lock"; return 73; fi
    if [[ -e "$RUN/triggered" ]]; then note launch "REFUSED | chain already consumed; no retry"; return 73; fi
    if collision smoke; then fail "existing APTM Smoke/Full output or process"; return 73; fi
    printf '%s\n' "$$" > "$RUN/wait.pid"
    trap cancel_wait TERM INT
    trap 'rc=$?; trap - ERR; [[ "$phase" == FAIL ]] || fail "controller exit=$rc"; exit "$rc"' ERR
    export TMPDIR=/home/lzf/ldx/tmp PYTHONDONTWRITEBYTECODE=1
    export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    export PYTHONPATH="$W/src"
    cd "$W"
    status WAITING_RASA "pid=$$ interval=60 one chain; no G/H generation"
    wait_resources smoke
    if collision smoke; then fail "APTM appeared at Smoke boundary"; return 73; fi
    (set -o noclobber; printf '%s pid=%s\n' "$(date '+%F %T%z')" "$$" > "$RUN/triggered")
    execute_stage smoke run RUNNING_SMOKE smoke 'APTM_MATERIALIZED_EVALUATION_PASS smoke'
    execute_stage smoke audit RUNNING_SMOKE_AUDIT smoke_audit 'APTM_MATERIALIZED_CPU_AUDIT_PASS smoke'
    # 独立重新计数，不能继承Smoke前的空闲计数。
    wait_resources full
    if collision full; then fail "APTM appeared at Full boundary"; return 73; fi
    (set -o noclobber; printf '%s pid=%s\n' "$(date '+%F %T%z')" "$$" > "$RUN/full_triggered")
    execute_stage full run RUNNING_FULL full 'APTM_MATERIALIZED_EVALUATION_PASS full'
    execute_stage full audit RUNNING_FULL_AUDIT full_audit 'APTM_MATERIALIZED_CPU_AUDIT_PASS full'
    status SUCCESS "APTM_FULL_AND_CPU_AUDIT"
    note launch "SUCCESS | APTM_FULL_AND_CPU_AUDIT"
}
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then main "$@"; fi
