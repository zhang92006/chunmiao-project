#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export TMPDIR=/tmp OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

# Run once after review/commit. All destinations are exclusive-create.
# Replays only existing teacher trajectories; does not collect new data.
python -u -m scenario_reconstruction.highd_v47_shared_teacher freeze \
  --output outputs/v47_shared_teacher_ppo_v1_lock.json
python -u -m scenario_reconstruction.highd_v47_shared_teacher export \
  --lock outputs/v47_shared_teacher_ppo_v1_lock.json \
  --output outputs/v47_shared_teacher_ppo_v1_data
python -u -m scenario_reconstruction.highd_v47_shared_teacher audit \
  --lock outputs/v47_shared_teacher_ppo_v1_lock.json \
  --data outputs/v47_shared_teacher_ppo_v1_data --replay \
  --output outputs/v47_shared_teacher_ppo_v1_audit.json
python -u -m scenario_reconstruction.highd_v47_shared_teacher train \
  --lock outputs/v47_shared_teacher_ppo_v1_lock.json \
  --data outputs/v47_shared_teacher_ppo_v1_data \
  --audit outputs/v47_shared_teacher_ppo_v1_audit.json \
  --output outputs/v47_shared_teacher_ppo_v1_training
