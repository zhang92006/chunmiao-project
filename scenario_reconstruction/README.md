# Scenario Reconstruction

This package defines the accident-template, perturbation, simulation, and
Autoware-analysis layers used to rebuild cases in the project SUMO/NADE
environment. Repository installation instructions and dependency information
are in the root `README.md`.

Main components:

- `templates.py` defines the machine-readable template schema.
- `templates/autoware_cut_in.yaml` is a starter accident template.
- `validate_template.py` checks that a template is structurally usable before it
  is connected to SUMO initialization.
- `environment.py` defines `ScenarioNADE`, a NADE environment that initializes
  CAV/BV vehicles from a template instead of random NDD traffic.
- `run_template.py` runs one reconstructed scenario and keeps the standard
  `NADEInfoExtractor` logging path.

The template describes:

- `ego`: CAV initial lane, position, speed, and controller.
- `actors`: background vehicles that matter for the reconstructed accident.
- `events`: fault or maneuver events to replay in the simulator.
- `perturbations`: parameter ranges used to expand one Autoware accident into
  many training episodes.

Run:

```bash
python -m scenario_reconstruction.validate_template scenario_reconstruction/templates/autoware_cut_in.yaml
```

Generate scenario diagnostic episodes from variants (not likelihood-valid training data):

```bash
python -m scenario_reconstruction.run_batch scenario_reconstruction/templates/autoware_cut_in.yaml \
  --count 100 \
  --output_root data_analysis/raw_data/ScenarioReconstructionDiagnostics --seed 7
```

All completed episodes are retained, including non-critical safe runs. Training
weight dictionaries are intentionally empty for template-initialized scenarios:
there is no derived importance ratio for their initialization or scripted actions.
The legacy 10-D/one-action training environment rejects joint MultiBV records.
`weight_result=null` means unavailable, not a zero collision probability.

Only `forced_bv_action` currently has a tested execution path. Other fault names
remain in the descriptive schema, but runtime validation rejects them. Use
`validate_template --schema_only` only when inspecting an unimplemented description.
The example is hand-authored, not a verified reconstruction of the Autoware bag.

See [the research roadmap](../docs/科研提升路线图.md). Earlier implementation and
experiment notes in this directory are historical; their heuristic training
weights and first-agent MultiBV projection are no longer supported.
