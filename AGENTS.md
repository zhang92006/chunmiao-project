# Project instructions

This repository follows the two-person research Git workflow documented by the team.

Before changing files:

1. Read `README.md` and the relevant document under `docs/` or `scenario_reconstruction/`.
2. Start from an up-to-date `main` and create one short-lived branch per task.
3. Use `feature/*`, `experiment/*`, `fix/*`, `refactor/*`, or `docs/*` branch names.
4. Never develop or commit directly on `main`.
5. Check that the working tree is clean before starting; preserve unrelated user changes.

While working:

1. Keep changes scoped to the task.
2. Do not commit secrets, API keys, local paths, datasets, checkpoints, caches, logs, or raw outputs.
3. Required large binary runtime data must use Git LFS or a documented external source with a checksum.
4. Commit reproducible configuration, seed, dataset version, and environment information.
5. Inspect `git diff` and stage explicit paths instead of routinely using `git add .`.
6. Use descriptive commits such as `feat: ...`, `fix: ...`, `exp: ...`, `docs: ...`, or `config: ...`.

Before finishing:

1. Run the relevant validation and tests.
2. Update documentation and configuration when behavior changes.
3. Push the task branch and open a Pull Request to `main`.
4. Record purpose, changes, experiment configuration, results, risks, and test evidence in the PR.
5. The other researcher reviews and merges, preferably with Squash and merge.
6. Important paper milestones must identify the exact commit and receive an annotated Git tag.

Only use `--force-with-lease` on your own temporary branch after a rebase. Never force-push `main`.

