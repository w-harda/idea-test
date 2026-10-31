#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/lzf/ldx/projects/idea-test-tbps-cpgc-sep-throughput-v3-20261031
exec /home/lzf/ldx/envs/tbps-image/bin/python -u -B "$ROOT/operations/cpgc_sep_throughput_run.py" --allow-v3
