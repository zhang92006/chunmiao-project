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
    repeats: int = 1,
    epsilon: float = 0.99,
    max_initial_primary_ttc_s: float | None = None,
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
    records_after_split = len(records)
    if max_initial_primary_ttc_s is not None:
        if max_initial_primary_ttc_s <= 0:
            raise ValueError("max_initial_primary_ttc_s must be positive")
        records = [
            record
            for record in records
            if (
                (ttc := _initial_primary_ttc_s(record["path"])) is not None
                and ttc <= max_initial_primary_ttc_s
            )
        ]
    if start < 0:
        raise ValueError("start must be non-negative")
    records = records[start:]
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive when provided")
        records = records[:limit]
    if repeats < 1:
        raise ValueError("repeats must be positive")
    if gui_episode is not None:
        if gui_episode < 0 or gui_episode >= len(records):
            raise ValueError(f"gui_episode={gui_episode} is out of range 0..{len(records) - 1}")
        records_to_run = [(gui_episode, records[gui_episode], 0)]
    else:
        records_to_run = [
            (repeat_index * len(records) + record_index, record, repeat_index)
            for repeat_index in range(repeats)
            for record_index, record in enumerate(records)
        ]

    results = []
    for episode_id, record, repeat_index in records_to_run:
        template_path = record["path"]
        try:
            weight = run_template(
                template_path,
                episode=episode_id,
                experiment_path=str(experiment_path),
                gui=gui_episode is not None,
                epsilon=epsilon,
            )
            results.append(
                {
                    "episode": episode_id,
                    "repeat": repeat_index,
                    "template": template_path,
                    "status": "ok",
                    "weight_result": weight,
                }
            )
        except Exception as exc:
            results.append(
                {
                    "episode": episode_id,
                    "repeat": repeat_index,
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
    joint_stats = _joint_rollout_stats(experiment_path)
    summary = {
        "manifest": str(manifest_path),
        "experiment_path": str(experiment_path),
        "records_after_split": records_after_split,
        "records_after_initial_primary_ttc_filter": len(records),
        "max_initial_primary_ttc_s": max_initial_primary_ttc_s,
        "attempted": len(results),
        "repeats": repeats,
        "successful_runs": sum(1 for item in results if item["status"] == "ok"),
        "failed_runs": sum(1 for item in results if item["status"] != "ok"),
        "training_ready_crashes": len(crash_weight_dict),
        "training_ready_safe": len(safe_weight_dict),
        **joint_stats,
        "results": results,
    }
    with (experiment_path / "manifest_run_summary.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(summary, stream, indent=4)
    return summary


def _initial_primary_ttc_s(template_path: str | Path) -> float | None:
    """Return initial CAV-to-primary TTC for a closing, same-lane pair.

    A missing value deliberately excludes the template from an explicit TTC-filtered
    rollout.  It avoids treating adjacent-lane geometry or a receding primary BV as
    a longitudinal collision opportunity.
    """
    try:
        with Path(template_path).open("r", encoding="utf-8") as stream:
            template = json.load(stream)
        ego = template["ego"]
        primary = next(actor for actor in template["actors"] if actor.get("id") == "BV_primary")
        if int(ego["lane_index"]) != int(primary["lane_index"]):
            return None
        gap_m = float(primary["position"]) - float(ego["position"])
        closing_speed_mps = float(ego["speed"]) - float(primary["speed"])
        if gap_m <= 0 or closing_speed_mps <= 0:
            return None
        return gap_m / closing_speed_mps
    except (KeyError, StopIteration, TypeError, ValueError, OSError, json.JSONDecodeError):
        return None


def _joint_rollout_stats(experiment_path: Path) -> dict[str, int]:
    """Summarize actual K-agent records without treating debug candidates as samples."""
    episode_count = 0
    joint_step_count = 0
    debug_step_count = 0
    max_selected_count = 0
    for episode_path in (
        sorted((experiment_path / "crash").glob("*.json"))
        + sorted((experiment_path / "tested_and_safe").glob("*.json"))
    ):
        with episode_path.open("r", encoding="utf-8") as stream:
            episode = json.load(stream)
        debug = episode.get("multibv_selection_debug_step_info", {})
        debug_step_count += len(debug)
        for value in debug.values():
            max_selected_count = max(
                max_selected_count, len(value.get("selected_candidate_ids", []))
            )
        joint_steps = 0
        for timestep, obs in episode.get("drl_obs_step_info", {}).items():
            if not isinstance(obs, dict) or not isinstance(obs.get("joint"), list):
                continue
            if not isinstance(obs.get("per_agent"), list):
                continue
            if len(episode.get("controlled_bv_ids_step_info", {}).get(timestep, [])) < 2:
                continue
            joint_steps += 1
        if joint_steps:
            episode_count += 1
            joint_step_count += joint_steps
    return {
        "joint_training_episode_count": episode_count,
        "joint_training_step_count": joint_step_count,
        "selection_debug_step_count": debug_step_count,
        "max_selected_bv_count": max_selected_count,
    }


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
        "--repeats",
        type=int,
        default=1,
        help="Independent rollouts per selected template; each receives a unique episode ID.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.99,
        help="Fixed NADE naturalistic-mixture probability used by every rollout.",
    )
    parser.add_argument(
        "--max_initial_primary_ttc_s",
        type=float,
        default=None,
        help=(
            "Keep only templates whose primary BV is ahead in the CAV lane, "
            "is being closed upon, and has initial TTC no greater than this value."
        ),
    )
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
        repeats=args.repeats,
        epsilon=args.epsilon,
        max_initial_primary_ttc_s=args.max_initial_primary_ttc_s,
    )
    print("Manifest run finished.")
    print(f"attempted={summary['attempted']}")
    print(f"successful_runs={summary['successful_runs']}")
    print(f"failed_runs={summary['failed_runs']}")
    print(f"training_ready_crashes={summary['training_ready_crashes']}")
    print(f"training_ready_safe={summary['training_ready_safe']}")


if __name__ == "__main__":
    main()
