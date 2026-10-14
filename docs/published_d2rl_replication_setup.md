# Released two-lane D2RL: preparation and evening launch

This work prepares the released D2RL agent for the supplied script's **10 workers
× 20,001 episodes = 200,010 tests**. The full batch has not been launched or
scheduled. The released controller, importance weights, two-lane network,
400 m stopping criterion, 0.1 s simulation/action steps and 1 s lane change are
retained. The AV is the published **IDM/MOBIL** controller, including its lane
changes. This is separate from highD/V46 and its fixed-lane CAV baseline.

## Source alignment

- Published source: `michigan-traffic-lab/Dense-Deep-Reinforcement-Learning`,
  commit `087921a994b3f5176fb0d75667a0a11fa357cd2b`.
- The supplied `two_lane_replication.zip` has SHA-256
  `f19d163a7ac2f3766c54cb50ce6a26d12923edafe3ad34049cd765a522564cb3`.
  It contains the previously missing requirements, download wrapper, main
  arguments, acceleration check and result summarizer. Its own Git revision
  was not supplied; the archive checksum identifies what was inspected.
- All 16 probability/challenge/model assets match the archive byte for byte.
  Their checksums are committed in `configs/published_d2rl_assets.json`.
- The archive's active NDD, NADE proposal, tree-search, IDM/MOBIL, vehicle,
  observation, information extractor, two-lane map and configuration code
  match this checkout (normalizing text line endings). Its optional AEB,
  three-lane and dual-BV extensions are inactive for the supplied command.
- Its dependency file targets **macOS arm64 / SUMO 1.27.0**, explicitly as a
  platform workaround. This Windows environment instead pins the upstream
  **SUMO 1.14.1** family. The original slowDown adapter was physically checked
  on this build. This does not claim bitwise equivalence to SUMO 1.27.0.
- Python 3.9, Torch 1.12.1 CPU and NumPy 1.23.5 are isolated from existing
  environments. The NumPy-to-Torch bridge and released TorchScript execution
  work without an observation-conversion workaround. No Ray or training stack
  is needed for this exported checkpoint.

The batch wrappers add explicit episode counts, per-episode NumPy/Python/SUMO
seeds, failure propagation, atomic final weight saves, unique output folders
and completion checks. They are compatible replacements for the supplied
launch contract, not an unmodified copy of the archive. The archive's original
loop silently continued failed episodes and did not always flush its final
weights; these behaviors are not used for a fixed-size evaluation.

## Memory fix

The three large lane-change probability tables originally expanded to about
**4.56 GiB per process**, before the model and simulation. They now use sorted
coordinate keys and the **unchanged float64 values** in read-only memory maps.
All four table caches total **154.12 MB** (decimal); the three large ones total
153.46 MB. Lookup returns exact original probabilities, including implicit zeros.
Caches are built once before workers start and are ignored by Git.

The small original OL table remains dense because the author controller iterates
over it. A bounded query cache avoids repeating coordinate searches. The dense
backend is retained only for diagnostics through `D2RL_TABLE_BACKEND=dense`;
do not enable it for the ten-worker batch.

The 100-episode, two-worker preflight measured maximum worker RSS **257.14 MiB**
and private memory **181.83 MiB**, versus the earlier dense process measurement
of about 4.70 GiB RSS. These are short-run measurements, not a long-run leak test.
Windows virtual-environment launchers spawn interpreter children; the final
batch monitor includes those children. The first preflight's parent-only
aggregate RSS was invalid and is excluded from the committed result evidence.

The launcher requires **3.00 GiB available RAM for ten workers**. Aim for at
least 4 GiB before an overnight run. During preparation, ten-worker attempts
were refused because other applications left only about 2 GiB available;
no other application was stopped. A ten-worker throughput measurement has
therefore not yet been obtained.

## Install and prepare (Windows PowerShell)

From the repository root, using an existing Python 3.9 executable:

```powershell
.\tools\setup_published_d2rl.ps1 -Python39 'path\to\python39.exe'
.\run_two_lane_replication.ps1
```

The first command creates `.venv_replication`, installs the pinned inference
dependencies, verifies the published checkpoint and builds the shared caches.
The second only prints/verifies the 200,010-test plan. It starts no simulations.
Run `git lfs pull` first on another checkout to obtain the tracked probability
assets. The model is downloaded from the pinned upstream commit with SHA-256
verification; it is not committed here.

The required packages and exact experiment settings are recorded under
`configs/requirements-d2rl-baseline.txt` and
`configs/published_d2rl_replication.json`. A changed dependency version causes
the launcher to fail before simulation rather than silently change the runtime.

## Explicit evening launch

```powershell
.\run_two_lane_replication.ps1 -Run
```

This starts exactly **10 × 20,001** episodes, with seed base `20261014` and
global episode IDs `0..200009`. It uses CPUs; no training/GPU workload is launched.
The shell entry point is also retained:

```bash
bash run_two_lane_replication.sh --prepare-only
bash run_two_lane_replication.sh 10 20001
```

The shell entry point's default is an actual run, matching the supplied script;
the PowerShell entry point defaults to preparation. Windows is the platform
validated here. macOS arm64 is not validated with these Windows-aligned pins.

Each batch creates new timestamped directories under
`outputs/two_lane_replication/` and `data_analysis/raw_data/`. Outputs include
`plan.json`, per-worker logs, `memory.json`, `summary.json` and `weights/`.
Worker reports contain attempted/completed counts, errors, policy-call counts,
model hash, SUMO version and peak memory. A failed or incomplete worker prevents
a successful summary. Weights are saved every 50 episodes and on normal/error
exit; an OS-level forced termination can still lose the last unsaved block.

Ctrl-C terminates this batch's processes, including their Windows interpreter
children. Existing results are retained. A later launch starts a **new batch**;
automatic resume is not implemented. Do not combine partial runs as a complete
200,010-test result.

## Validation and time estimate

- `pip check`: no broken dependencies.
- Four exact probability-query comparisons against independent original dense
  slices and four physical acceleration cases: **8 passed** in the isolated
  environment. No probability fitting, float32 conversion or policy change.
- **100/100 episodes completed**, two workers × 50, no execution errors,
  **28 released-policy calls**, one recorded CAV collision with a finite
  importance weight. The collision is expected evidence for an accelerated
  safety test, not a failure of the environment setup.
- Batch wall time **43.10 s**; worker simulation-loop times **38.69 / 39.65 s**.
  This is about **0.78 s/test per worker with two workers running**.
  Ideal linear scaling would imply about **4.4 h** at ten workers. Allow
  roughly **6–10 h** for planning, conditional on sufficient free memory and
  CPU availability; ten-worker scaling and overnight memory stability remain
  unmeasured. Keep the machine awake and powered for the intended run.

The short preflight is runtime evidence only. Its one-event crash estimate and
RHW are not a reproduction of the published statistical result. A fixed
200,010-test budget does not guarantee a desired confidence interval. The final
summary reports the importance-weighted CAV crash probability, 90% normal
interval and RHW; raw accelerated collision share is a different quantity.
With zero observed weighted events, RHW is unavailable rather than zero.

Concise machine-readable validation evidence is in
`results/published_d2rl_preflight_20261014.json`; raw logs and episode data stay
outside version control.
