#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export TMPDIR=/tmp OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
# Preserve completed seed 7 Single and failed startup artifacts. No recollection.
python -u -m scenario_reconstruction.highd_v47_shared_teacher_recovery run \
  --lock outputs/v47_shared_teacher_ppo_v1_lock.json \
  --data outputs/v47_shared_teacher_ppo_v1_data \
  --audit outputs/v47_shared_teacher_ppo_v1_audit.json \
  --previous outputs/v47_shared_teacher_ppo_v1_training \
  --output outputs/v47_shared_teacher_ppo_v1_training_recovery
