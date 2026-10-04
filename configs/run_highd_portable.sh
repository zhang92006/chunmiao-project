#!/usr/bin/env bash
# All long-running examples live here, not in the Markdown handoff.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHON="${PYTHON:-python}"
ACTION="${1:-verify}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/handoff_${ACTION}_$(date +%Y%m%d_%H%M%S)}"
SEED="${SEED:-1000}"
REPEATS="${REPEATS:-1}"
ITERATIONS="${ITERATIONS:-2}"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
case "$ACTION" in
  verify)
    "$PYTHON" -m scenario_reconstruction.highd_portable verify --sumo
    ;;
  smoke)
    "$PYTHON" -m scenario_reconstruction.highd_portable verify --sumo
    "$PYTHON" -m unittest tests.test_highd_portable tests.test_highd_dual_ndd_criticality tests.test_highd_critical_sequences tests.test_highd_mean_precision_policy
    "$PYTHON" -m scenario_reconstruction.highd_portable train --output "$OUTPUT_ROOT/train" --iterations 2
    "$PYTHON" -m scenario_reconstruction.highd_portable diagnose --output "$OUTPUT_ROOT/checkpoint_restore"
    "$PYTHON" -m scenario_reconstruction.highd_portable collect --output "$OUTPUT_ROOT/controls" --scenarios adjacent_approach_pair --seed 7 --repeats 1
    "$PYTHON" -m scenario_reconstruction.highd_portable prepare --collection "$OUTPUT_ROOT/controls" --output "$OUTPUT_ROOT/control_audit"
    "$PYTHON" -m scenario_reconstruction.highd_portable evaluate --output "$OUTPUT_ROOT/policy" --scenarios adjacent_approach_pair --seed 7 --repeats 1 --smoke
    ;;
  train)
    "$PYTHON" -m scenario_reconstruction.highd_portable train --output "$OUTPUT_ROOT" --iterations "$ITERATIONS" --seed "$SEED" "${@:2}"
    ;;
  compare)
    # Same four-scene mixture, seeds and repeat count for all three policies.
    "$PYTHON" -m scenario_reconstruction.highd_portable collect --output "$OUTPUT_ROOT/controls" --seed "$SEED" --repeats "$REPEATS"
    "$PYTHON" -m scenario_reconstruction.highd_portable prepare --collection "$OUTPUT_ROOT/controls" --output "$OUTPUT_ROOT/control_audit"
    "$PYTHON" -m scenario_reconstruction.highd_portable evaluate --output "$OUTPUT_ROOT/policy" --seed "$SEED" --repeats "$REPEATS" "${@:2}"
    ;;
  *) printf '%s\n' 'Supported actions: verify, smoke, train, compare'; exit 2 ;;
esac
