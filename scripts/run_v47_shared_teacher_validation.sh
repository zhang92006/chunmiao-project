#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export TMPDIR=/tmp OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
# One prospective source cohort. Do not repeat over existing outputs/claim.
python -u -m scenario_reconstruction.highd_v47_shared_teacher_validation freeze \
  --output outputs/v47_shared_teacher_validation_v1_lock.json
python -u -m scenario_reconstruction.highd_v47_shared_teacher_validation bank \
  --lock outputs/v47_shared_teacher_validation_v1_lock.json \
  --output outputs/v47_shared_teacher_validation_v1_bank
python -u -m scenario_reconstruction.highd_v47_shared_teacher_validation run \
  --lock outputs/v47_shared_teacher_validation_v1_lock.json \
  --bank outputs/v47_shared_teacher_validation_v1_bank \
  --output outputs/v47_shared_teacher_validation_v1
python -u -m scenario_reconstruction.highd_v47_shared_teacher_validation audit \
  --lock outputs/v47_shared_teacher_validation_v1_lock.json \
  --bank outputs/v47_shared_teacher_validation_v1_bank \
  --result outputs/v47_shared_teacher_validation_v1 --replay \
  --output outputs/v47_shared_teacher_validation_v1_audit.json
