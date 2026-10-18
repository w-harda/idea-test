#!/usr/bin/env bash
# 用户手动nohup+setsid启动；一次资源观察，Smoke全部验收后才进入Full。
set -Eeuo pipefail
[[ "${1:-}" == "--allow-full" && "$#" == 1 ]] || { echo "需要显式 --allow-full（包含Smoke→审计→Full→审计）"; exit 2; }
WORK=/home/lzf/ldx/projects/idea-test-tbps-b3-shared-run-20261018
PY=/home/lzf/ldx/envs/tbps-image/bin/python
CFG="$WORK/configs/framework/baseline_b3_shared.server.json"
BASE=/home/lzf/ldx/outputs/tbps-framework-b3-shared-evaluation
CONTROL=/home/lzf/ldx/tmp/b3_shared_run
LEGACY=/home/lzf/ldx/outputs/tbps-framework-b3-comparison-ready
cd "$WORK"
export PYTHONPATH="$WORK/src" PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0
export TMPDIR=/home/lzf/ldx/tmp OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
mkdir -p "$CONTROL"
exec 9>"$CONTROL/pipeline.lock"
flock -n 9 || { echo "已有B3后台链，拒绝重复启动"; exit 3; }
[[ ! -e "$CONTROL/started" ]] || { echo "本次启动许可已消费，禁止覆盖或自动重试"; exit 4; }
for base in "$BASE" "$LEGACY"; do
  [[ ! -e "$base/full" ]] || { echo "已有B3 Full目录：$base/full"; exit 4; }
  for model in irra rasa aptm; do
    [[ ! -e "$base/smoke/$model" ]] || { echo "已有B3 Smoke Attempt/结果：$base/smoke/$model"; exit 4; }
  done
done
PGID=$(ps -o pgid= -p "$$" | tr -d ' ')
SID=$(ps -o sid= -p "$$" | tr -d ' ')
[[ "$PGID" == "$$" && "$SID" == "$$" ]] || { echo "请使用nohup setsid启动独立会话"; exit 2; }
(set -o noclobber; printf '%s\n' "$(date -Is)" >"$CONTROL/started")
printf '%s\n' "$$" >"$CONTROL/supervisor.pid"
printf '%s\n' "$PGID" >"$CONTROL/supervisor.pgid"
mkdir -p "$CONTROL/logs"
CURRENT=STARTING
state() {
  CURRENT="$1"
  printf '%s\n' "$CURRENT" >"$CONTROL/phase"
  printf '%s | %s | %s\n' "$(date -Is)" "$CURRENT" "${2:-}" | tee -a "$CONTROL/status.log"
}
failed() {
  local rc="$1" at="$CURRENT"
  trap - ERR
  state FAIL "stage=$at exit=$rc; 保留全部现场，不重试"
  exit "$rc"
}
cancelled() {
  trap - ERR TERM INT
  state CANCELLED "stage=$CURRENT; 用户终止独立进程组，不恢复或重试"
  exit 143
}
trap 'failed "$?"' ERR
trap cancelled TERM INT
run_stage() {
  local name="$1" log="$2" marker="$3"
  shift 3
  state "$name" "$log"
  printf '%s\n' "$log" >"$CONTROL/current_log"
  "$@" >"$log" 2>&1 &
  local child="$!"
  printf '%s\n' "$child" >"$CONTROL/child.pid"
  wait "$child"
  printf '0\n' >"$CONTROL/child.pid"
  grep -Fq "$marker" "$log"
  printf '%s | PASS | %s\n' "$(date -Is)" "$name" | tee -a "$CONTROL/status.log"
}
# 仅此处检查一次GPU及其他B3进程。高利用率/Ollama存在本身不拒绝。
state RESOURCE_CHECK "$CONTROL/startup_gpu.json"
printf '%s\n' "$CONTROL/startup_gpu.json" >"$CONTROL/current_log"
env CUDA_VISIBLE_DEVICES="" "$PY" -B -m tbps.baselines.launch --config "$CFG" --action resources >"$CONTROL/startup_gpu.json" 2>"$CONTROL/resource_errors.log" &
printf '%s\n' "$!" >"$CONTROL/child.pid"
wait "$(cat "$CONTROL/child.pid")"
printf '0\n' >"$CONTROL/child.pid"
export TBPS_B3_GPU_SNAPSHOT="$CONTROL/startup_gpu.json"
run_stage CPU_PREFLIGHT "$CONTROL/logs/preflight.log" "B3_CPU_PREFLIGHT_PASS full" \
  "$PY" -B -m tbps.baselines.launch --config "$CFG" --action preflight --scope full
for model in irra rasa aptm; do
  run_stage "RUNNING_SMOKE_$model" "$CONTROL/logs/${model}_smoke.log" "B3_EVALUATION_PASS smoke $model" \
    "$PY" -B -m tbps.baselines.launch --config "$CFG" --action run --scope smoke --model "$model" --allow-smoke
  run_stage "RUNNING_SMOKE_AUDIT_$model" "$CONTROL/logs/${model}_smoke_audit.log" "B3_CPU_AUDIT_PASS smoke $model" \
    env CUDA_VISIBLE_DEVICES="" "$PY" -B -m tbps.baselines.launch --config "$CFG" --action audit --scope smoke --model "$model"
done
# 仅重放/尺寸映射已存在的B3历史RGB；绝不加载Generator或重新攻击。
run_stage READ_HISTORICAL_B3_RGB "$CONTROL/logs/pixels.log" "B3_PIXELS_PASS full" \
  "$PY" -B -m tbps.baselines.launch --config "$CFG" --action pixels --scope full --allow-full-pixel-transport
for model in irra rasa aptm; do
  run_stage "RUNNING_FULL_$model" "$CONTROL/logs/${model}_full.log" "B3_EVALUATION_PASS full $model" \
    "$PY" -B -m tbps.baselines.launch --config "$CFG" --action run --scope full --model "$model" --allow-full-evaluation
  run_stage "RUNNING_FULL_AUDIT_$model" "$CONTROL/logs/${model}_full_audit.log" "B3_CPU_AUDIT_PASS full $model" \
    env CUDA_VISIBLE_DEVICES="" "$PY" -B -m tbps.baselines.launch --config "$CFG" --action audit --scope full --model "$model"
done
state SUCCESS "B3_THREE_MODEL_FULL_AND_CPU_AUDIT"
