#!/usr/bin/env bash
set -Eeuo pipefail
export PYTHONDONTWRITEBYTECODE=1
exec /home/lzf/ldx/envs/tbps-image/bin/python -u -B /home/lzf/ldx/projects/idea-test-tbps-cpgc-sep-resume-20261029/operations/cpgc_sep_resume.py --allow-resume
