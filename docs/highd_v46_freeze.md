# Frozen native V46 NDE

Frozen at the user's request on 2026-10-03, from NDE code commit `5649f90`.
The immutable reference is identified by `configs/highd_paper_v46_freeze.json`:

`106a9fbc795c75c9d4f60a880da7264cb626f9163e5ec073b720e6b48c6f9db5`

This version retains V34 single-action marginals, one-second maneuvers and
explicit feasibility restriction. Supported dynamic following pairs use V46
interval-aligned, hierarchically calibrated rank dependence. Fixed seeds
7/19/29 complete 60 s without collision, H(speed,gap,rr)=
0.15716053/0.21796016/0.11598359. Freezing records a research reference; it does
not add validation beyond the documented development results.

The 9 MB Git LFS archive `source_data/highd_paper_v46/runtime.zip` contains the
exact fitted runtime tables, reference histogram and three pilot fleets.
It contains no raw highD tracks or training checkpoints. All eleven archive
members and twenty source/configuration files are checksum-locked. Historical
source paths are not needed; extraction uses repository-relative paths.

```bash
git lfs install
git lfs pull
python -m scenario_reconstruction.highd_v46_assets
```

This verifies code and archive hashes, extracts into
`outputs/highd_paper_v46_assets`, and refuses conflicting existing files.
The original model, lateral table, pair kernel, initialization and reference
are available in the `model`, `lateral`, `pair`, `initial_fleets` and `reference`
subdirectories. The selected flags remain `--supported-pairs-only`, full
dependence scale, no gap taper and continuity-preserving matching.

Future changes to the natural law require a new freeze identity. D2RL additions
must keep the frozen files intact, distinguish P from Q, and report their CAV,
initial-state and collision-event contracts separately.
