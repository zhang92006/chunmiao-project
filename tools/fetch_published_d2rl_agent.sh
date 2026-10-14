#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m scenario_reconstruction.prepare_published_d2rl --fetch-agent
