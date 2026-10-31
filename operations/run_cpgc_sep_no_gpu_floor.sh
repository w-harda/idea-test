#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/lzf/ldx/projects/idea-test-tbps-cpgc-sep-no-gpu-floor-20261101
exec /home/lzf/ldx/envs/tbps-image/bin/python -u -B "$ROOT/operations/cpgc_sep_throughput_run.py" --allow-no-gpu-floor
