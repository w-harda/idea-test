#!/usr/bin/env bash
# 仅供后续用户明确授权Full时运行；本阶段不部署、不启动。
set -Eeuo pipefail
[[ "${1:-}" == "--allow-full" && "$#" == 1 ]] || { echo "需显式 --allow-full；默认不执行"; exit 2; }
ROOT=/home/lzf/ldx/projects/idea-test-tbps-gv2-epoch-sweep-20261020
PY=/home/lzf/ldx/envs/tbps-image/bin/python
CONFIG="$ROOT/configs/framework/gv2_epoch_sweep.server.json"
CONTROL=/home/lzf/ldx/tmp/gv2_epoch_sweep_full_v1
OUT=/home/lzf/ldx/outputs/tbps-framework-gv2-epoch-sweep/full
[[ ! -e "$CONTROL" && ! -e "$OUT" ]] || { echo "已有控制目录或Full产物，拒绝重复启动"; exit 2; }
# Full的用户授权与先前Smoke验收是两个独立条件；缺Smoke时不创建Full控制目录。
PYTHONPATH="$ROOT/src" PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" "$PY" -B -c '
from pathlib import Path
from tbps.artifacts import ArtifactStore,read_json
base=Path("/home/lzf/ldx/outputs/tbps-framework-gv2-epoch-sweep")
store=ArtifactStore(base/"smoke_v1",new_base=base)
store.verify("completed")
r=read_json(store.root/"completed/report.json")
assert r["status"]=="PASS" and r["G_forwards"]==20 and r["H_forwards"]==0
assert r["epoch10_historical_max_abs_error"]<=1e-6 and r["full_started"] is False
assert not (store.root/"failure").exists()
for model in ("irra","rasa","aptm"):
    ArtifactStore(store.root/"epoch_0060"/model,new_base=base).verify("cpu_audit")
'
mkdir "$CONTROL"
printf '%s\n' "$$" > "$CONTROL/supervisor.pid"
ps -o pgid= -p "$$" | xargs > "$CONTROL/supervisor.pgid"
ps -o sid= -p "$$" | xargs > "$CONTROL/supervisor.sid"
touch "$CONTROL/started"
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$ROOT/src" CUDA_VISIBLE_DEVICES=0
cd "$ROOT"
phase() { printf '%s\n' "$1" > "$CONTROL/phase"; printf '%s | %s\n' "$(date --iso-8601=seconds)" "$1" >> "$CONTROL/status.log"; }
trap 'code=$?; phase "FAIL exit=$code line=$LINENO"; exit "$code"' ERR
run() { local name="$1"; shift; phase "$name"; "$PY" -B -u -m tbps.gv2_epoch_eval --config "$CONFIG" --scope full "$@" >> "$CONTROL/$name.log" 2>&1; }
run CPU_PREFLIGHT --action preflight
# Epoch10仅导出原已审计结果引用，绝不重新生成/评分。
for model in irra rasa aptm; do
  run "E10_REUSE_$model" --action target --epoch 10 --model "$model" --allow-full-evaluation
  CUDA_VISIBLE_DEVICES="" run "E10_AUDIT_REF_$model" --action audit --epoch 10 --model "$model"
done
for epoch in 20 30 40 50 60; do
  run "GALLERY_E$epoch" --action generate --epoch "$epoch" --allow-full-materialization
  for model in irra rasa aptm; do
    run "TARGET_E${epoch}_$model" --action target --epoch "$epoch" --model "$model" --allow-full-evaluation
    CUDA_VISIBLE_DEVICES="" run "CPU_AUDIT_E${epoch}_$model" --action audit --epoch "$epoch" --model "$model"
  done
done
phase "SUCCESS | GV2_FIVE_NEW_EPOCHS_THREE_TARGET_FULL_AND_CPU_AUDIT"
