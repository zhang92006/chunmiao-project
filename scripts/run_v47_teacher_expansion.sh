#!/usr/bin/env bash
# Researcher-run only. No overwrite, native retry, validation or PPO launch.
set -euo pipefail
cd "$(dirname "$0")/.."
export TMPDIR=/tmp OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

python -u -m scenario_reconstruction.highd_v47_action_teacher run \
  --lock outputs/v47_action_teacher_v1_lock.json \
  --output outputs/v47_action_teacher_v1_reproduction
python -u -m scenario_reconstruction.highd_v47_teacher_restore \
  --reproduction outputs/v47_action_teacher_v1_reproduction \
  --output outputs/v47_teacher_full_reproduction_v1_audit.json
python -u -m scenario_reconstruction.highd_v47_teacher_expansion freeze \
  --reproduction-audit outputs/v47_teacher_full_reproduction_v1_audit.json \
  --output outputs/v47_teacher_expansion_v1_lock.json
python -u -m scenario_reconstruction.highd_v47_teacher_expansion discover \
  --lock outputs/v47_teacher_expansion_v1_lock.json \
  --output outputs/v47_teacher_expansion_v1_bank
python -u -m scenario_reconstruction.highd_v47_teacher_expansion collect \
  --lock outputs/v47_teacher_expansion_v1_lock.json \
  --bank outputs/v47_teacher_expansion_v1_bank \
  --output outputs/v47_teacher_expansion_v1_data
python -u -m scenario_reconstruction.highd_v47_teacher_expansion audit \
  --lock outputs/v47_teacher_expansion_v1_lock.json \
  --bank outputs/v47_teacher_expansion_v1_bank \
  --data outputs/v47_teacher_expansion_v1_data --replay \
  --output outputs/v47_teacher_expansion_v1_audit.json
