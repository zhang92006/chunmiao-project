from __future__ import annotations

import argparse
import json
from pathlib import Path

from .prepare_training_data import prepare_crash_weight_dict, prepare_safe_weight_dict
from .run_template import run_template


def run_template_manifest(
    manifest_path: str | Path,
    experiment_path: str | Path,
    max_ndd_possi: float | None = 0.01,
    gui_episode: int | None = None,
    split: str | None = None,
    start: int = 0,
    limit: int | None = None,
) -> dict:
    manifest_path = Path(manifest_path)
    experiment_path = Path(experiment_path)
    for subdir in ("crash", "tested_and_safe", "rejected"):
        (experiment_path / subdir).mkdir(parents=True, exist_ok=True)

    with manifest_path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)

    records = []
    for record in manifest.get("records", []):
        if split is not None and record.get("split") != split:
            continue
        if record.get("status") not in (None, "generated", "template_created"):
            continue
        template_path = record.get("path") or record.get("template_path")
        if not template_path:
            continue
        normalized_record = dict(record)
        normalized_record["path"] = template_path
        records.append(normalized_record)
    if start < 0:
        raise ValueError("start must be non-negative")
    records = records[start:]
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive when provided")
        records = records[:limit]
    if gui_episode is not None:
        if gui_episode < 0 or gui_episode >= len(records):
            raise ValueError(f"gui_episode={gui_episode} is out of range 0..{len(records) - 1}")
        records_to_run = [(gui_episode, records[gui_episode])]
    else:
        records_to_run = list(enumerate(records))

    results = []
    for episode_id, record in records_to_run:
        template_path = record["path"]
        try:
            weight = run_template(
                template_path,
                episode=episode_id,
                experiment_path=str(experiment_path),
                gui=gui_episode is not None,
            )
            results.append(
                {
                    "episode": episode_id,
                    "template": template_path,
                    "status": "ok",
                    "weight_result": weight,
                }
            )
        except Exception as exc:
            results.append(
                {
                    "episode": episode_id,
                    "template": template_path,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    crash_weight_dict = {}
    safe_weight_dict = {}
    if gui_episode is None:
        crash_weight_dict = prepare_crash_weight_dict(
            experiment_path,
            max_ndd_possi=max_ndd_possi,
            multi_bv=True,
            agent_num=2,
        )
        safe_weight_dict = prepare_safe_weight_dict(
            experiment_path,
            multi_bv=True,
            agent_num=2,
        )
    summary = {
        "manifest": str(manifest_path),
        "experiment_path": str(experiment_path),
        "attempted": len(results),
        "successful_runs": sum(1 for item in results if item["status"] == "ok"),
        "failed_runs": sum(1 for item in results if item["status"] != "ok"),
        "training_ready_crashes": len(crash_weight_dict),
        "training_ready_safe": len(safe_weight_dict),
        "results": results,
    }
    with (experiment_path / "manifest_run_summary.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(summary, stream, indent=4)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run every template listed in a simple MultiBV manifest.")
    parser.add_argument(
        "manifest",
        help="Path to simple_multibv_manifest.json.",
    )
    parser.add_argument(
        "--experiment_path",
        default="data_analysis/raw_data/SimpleMultiBV25/episodes",
        help="Output directory where episode JSON files are written.",
    )
    parser.add_argument("--max_ndd_possi", type=float, default=0.01)
    parser.add_argument("--split", choices=("train", "validation", "test"))
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--gui_episode",
        type=int,
        default=None,
        help="Open one manifest record in SUMO GUI instead of running the full batch.",
    )
    args = parser.parse_args()

    summary = run_template_manifest(
        args.manifest,
        experiment_path=args.experiment_path,
        max_ndd_possi=args.max_ndd_possi,
        gui_episode=args.gui_episode,
        split=args.split,
        start=args.start,
        limit=args.limit,
    )
    print("Manifest run finished.")
    print(f"attempted={summary['attempted']}")
    print(f"successful_runs={summary['successful_runs']}")
    print(f"failed_runs={summary['failed_runs']}")
    print(f"training_ready_crashes={summary['training_ready_crashes']}")
    print(f"training_ready_safe={summary['training_ready_safe']}")


if __name__ == "__main__":
    main()
