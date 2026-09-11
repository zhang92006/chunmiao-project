# Scenario Reconstruction

This folder defines the accident template layer used before rebuilding cases in
the project SUMO/NADE environment.

The first implementation step is intentionally small:

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

Generate MultiBV K=2 training data from variants:

```bash
python -m scenario_reconstruction.run_batch scenario_reconstruction/templates/autoware_cut_in.yaml \
  --count 100 \
  --output_root data_analysis/raw_data/ScenarioReconstructionMultiBV \
  --multi_bv_num 2
```

The generated `episodes/` folder contains `crash/`, `tested_and_safe/`,
`crash_weight_dict.json`, and `safe_weight_dict.json`. It can be checked with:

```bash
python MultiBV_Data_Onboarding_Package/MultiBV_Data_Onboarding_Package/validate_multibv_dataset.py \
  data_analysis/raw_data/ScenarioReconstructionMultiBV/episodes
```

Next implementation steps:

1. Expand event execution beyond `forced_bv_action`.
2. Add perturbation sampling for template variants.
3. Add Autoware db3-to-template extraction.
