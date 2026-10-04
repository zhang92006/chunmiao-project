#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python_bin="${PYTHON:-python}"
task="${1:-verify}"
if [[ $# -gt 0 ]]; then shift; fi
case "$task" in
  verify) "$python_bin" -m scenario_reconstruction.highd_dynamic_handoff verify --sumo "$@" ;;
  smoke)
    "$python_bin" -m scenario_reconstruction.highd_dynamic_handoff verify --sumo
    "$python_bin" -m scenario_reconstruction.highd_dynamic_handoff train --output outputs/dynamic_local_smoke --iterations 2 "$@"
    "$python_bin" -m scenario_reconstruction.highd_dynamic_handoff evaluate --training-root outputs/dynamic_local_smoke --output outputs/dynamic_local_online_smoke --scenarios adjacent_approach_pair --seed 2000 --smoke
    ;;
  train) "$python_bin" -m scenario_reconstruction.highd_dynamic_handoff train "$@" ;;
  collect) "$python_bin" -m scenario_reconstruction.highd_dynamic_handoff collect "$@" ;;
  prepare) "$python_bin" -m scenario_reconstruction.highd_dynamic_handoff prepare "$@" ;;
  evaluate) "$python_bin" -m scenario_reconstruction.highd_dynamic_handoff evaluate "$@" ;;
  compare)
    "$python_bin" -m scenario_reconstruction.highd_dynamic_handoff collect --output outputs/dynamic_compare_controls --seed 2000 "$@"
    "$python_bin" -m scenario_reconstruction.highd_dynamic_handoff prepare --collection outputs/dynamic_compare_controls --output outputs/dynamic_compare_controls_audit
    "$python_bin" -m scenario_reconstruction.highd_dynamic_handoff evaluate --output outputs/dynamic_compare_policy --seed 2000 "$@"
    ;;
  *) echo 'Tasks: verify | smoke | train | collect | prepare | evaluate | compare' >&2; exit 2 ;;
esac
