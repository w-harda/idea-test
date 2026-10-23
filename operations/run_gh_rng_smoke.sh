#!/usr/bin/env bash
# 后台独立监督；WATCHING不消费Smoke，只有完整门禁后持久化单次启动记录。
set -Eeuo pipefail
ROOT=/home/lzf/ldx/projects/idea-test-tbps-gh-probe-shared-20261023
OPS=/home/lzf/ldx/projects/idea-test-tbps-gh-rng-smoke-20261023
CTL=/home/lzf/ldx/tmp/gh_smoke_wait_v1_20261023
export PYTHONPATH="$ROOT/src" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export TMPDIR="$CTL" CUDA_VISIBLE_DEVICES=""
cd "$ROOT"
exec /home/lzf/ldx/envs/tbps-image/bin/python -B -u "$OPS/operations/gh_smoke_watch.py" \
  --control "$CTL" --allow-monitor-and-smoke
