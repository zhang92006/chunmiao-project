#!/usr/bin/env bash
# Run only inside a dedicated, already activated ARM64 Python 3.9 environment.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHON="${PYTHON:-python}"
"$PYTHON" -c 'import os,platform,sys; assert sys.version_info[:2]==(3,9), "Use Python 3.9"; assert platform.system()=="Darwin" and platform.machine()=="arm64", "Use native Apple Silicon Python"; assert os.environ.get("CONDA_DEFAULT_ENV") not in (None,"base") or os.environ.get("VIRTUAL_ENV"), "Activate a dedicated non-base environment first"'
# Gym 0.21 contains legacy requirement metadata; do not use modern isolated build tools.
"$PYTHON" -m pip install 'pip==23.2.1' 'setuptools==65.5.0' 'wheel==0.37.1'
"$PYTHON" -m pip install --no-build-isolation -r configs/requirements-highd-portable.txt
"$PYTHON" -m pip install --only-binary=:all: 'eclipse-sumo==1.27.0'
"$PYTHON" -m pip check
"$PYTHON" -c 'import gym,ray,torch,numpy,pandas,scipy,traci,sumolib; print("Training and SUMO Python imports passed")'
"$PYTHON" -m scenario_reconstruction.highd_portable verify --sumo
