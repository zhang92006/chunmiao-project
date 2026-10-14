#!/usr/bin/env bash
# Compatible batch contract: 10 workers x 20,001 = 200,010 published-agent tests.
# Pass --prepare-only to install/verify without starting simulations.
# Every batch has a new directory. Interrupted batches are retained, never overwritten.
set -euo pipefail
cd "$(dirname "$0")"
prepare_only=false
if [ "${1:-}" = "--prepare-only" ]; then prepare_only=true; shift; fi
workers=${1:-10}; episodes=${2:-20001}
python=${PYTHON:-python3.9}
command -v "$python" >/dev/null
if [ ! -x .venv_replication/bin/python ]; then
  "$python" -m venv .venv_replication
fi
.venv_replication/bin/python -m pip install --disable-pip-version-check -r configs/requirements-d2rl-baseline.txt
args=(--workers "$workers" --episodes-per-worker "$episodes")
if [ "$prepare_only" = false ]; then args+=(--run); fi
exec .venv_replication/bin/python -m scenario_reconstruction.run_published_d2rl "${args[@]}"
