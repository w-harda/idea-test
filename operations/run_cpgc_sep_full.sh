#!/usr/bin/env bash
# 独立一次性监督器：CPU来源→各方法真实Smoke与CPU审计→正式Full；首错停止。
set -Eeuo pipefail
ROOT=/home/lzf/ldx/projects/idea-test-tbps-cpgc-sep-20261028
CONTROL=/home/lzf/ldx/tmp/cpgc_sep_full_v1_20261028
CONFIG="$ROOT/configs/framework/cpgc_sep_tbps.server.json"
SOURCE_PY=/home/lzf/ldx/envs/sga-irra-source/bin/python
VICTIM_PY=/home/lzf/ldx/envs/tbps-image/bin/python
export PYTHONPATH="$ROOT/src"
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0
mkdir -p "$CONTROL"
exec 9>"$CONTROL/process.lock"
flock -n 9 || { echo "DUPLICATE_SUPERVISOR"; exit 2; }
[[ ! -e "$CONTROL/started" ]] || { echo "STARTED_ALREADY; no retry"; exit 2; }
[[ -z "$(git -C "$ROOT" status --porcelain)" ]] || { echo "SOURCE_TREE_DIRTY"; exit 2; }
# 首次即时资源核验；尚未创建实验Attempt或消耗started。
"$VICTIM_PY" -u -B -c 'import sys,json;from tbps.baselines.cpgc_sep.runtime import config,resources;print(json.dumps(resources(config(sys.argv[1]))))' "$CONFIG" > "$CONTROL/resources_before.json"
( set -C; date --iso-8601=seconds > "$CONTROL/started" )
echo "$$" > "$CONTROL/supervisor.pid"
ps -o pgid= -p "$$" | xargs > "$CONTROL/supervisor.pgid"
ps -o sid= -p "$$" | xargs > "$CONTROL/supervisor.sid"
git -C "$ROOT" rev-parse HEAD > "$CONTROL/execution_commit"
sha256sum "$CONFIG" > "$CONTROL/config.sha256"
status() { printf '%s | %s\n' "$(date --iso-8601=seconds)" "$*" | tee -a "$CONTROL/status.log"; printf '%s\n' "$*" > "$CONTROL/phase"; }
failure() { rc=$?; trap - ERR; status "FAIL exit=$rc line=$1"; printf '%s\n' "$rc" > "$CONTROL/exit"; exit "$rc"; }
trap 'failure "$LINENO"' ERR
run() {
 local label="$1" py="$2"; shift 2
 status "$label"
 "$py" -u -B -m tbps.baselines.cpgc_sep.cli --config "$CONFIG" "$@" > "$CONTROL/$label.log" 2>&1
 printf '0\n' > "$CONTROL/$label.exit"
}
for role in source victim; do
 py="$VICTIM_PY"; [[ "$role" != source ]] || py="$SOURCE_PY"
 status "CPU_PREFLIGHT_$role"
 "$py" -u -B -c 'import sys;from tbps.baselines.cpgc_sep.preflight import run;run(sys.argv[1],role=sys.argv[2])' "$CONFIG" "$role" > "$CONTROL/CPU_PREFLIGHT_$role.log" 2>&1
done
for method in cpgc sep; do
 for scope in smoke full; do
  authorization=--allow-gpu-smoke; [[ "$scope" != full ]] || authorization=--allow-full
  for ds in cuhk rstp icfg; do
   if [[ "$method" == cpgc ]]; then
    run "${method}_${scope}_${ds}_SOURCE" "$SOURCE_PY" --action cpgc-source --method cpgc --dataset "$ds" --scope "$scope" "$authorization"
   fi
   for victim in irra rasa; do
    if [[ "$method" == sep ]]; then
     run "${method}_${scope}_${ds}_${victim}_SOURCE" "$VICTIM_PY" --action sep-source --method sep --dataset "$ds" --victim "$victim" --scope "$scope" "$authorization"
    fi
    run "${method}_${scope}_${ds}_${victim}_TARGET" "$VICTIM_PY" --action target --method "$method" --dataset "$ds" --victim "$victim" --scope "$scope" "$authorization"
    run "${method}_${scope}_${ds}_${victim}_AUDIT" "$VICTIM_PY" --action audit --method "$method" --dataset "$ds" --victim "$victim" --scope "$scope"
   done
  done
 done
done
run RESULTS_SUMMARY "$VICTIM_PY" --action report
status "SUCCESS | CPGC_SEP_TWELVE_FULL_AND_CPU_AUDIT"
printf '0\n' > "$CONTROL/exit"
