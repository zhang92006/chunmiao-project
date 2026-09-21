from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Mapping

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
    epsilon: float | Mapping[str, float] = 0.99,
    proposal_mode: str = "joint_pair",
    max_initial_primary_ttc_s: float | None = None,
    require_context_blocking: bool = False,
    source_event_ids: set[int] | None = None,
    online_policy_path: str | None = None,
    frozen_epsilon_source: str = "template",
    simulation_seed: int | None = None,
    online_intervention_budget: int | None = None,
    online_max_proposal_likelihood_ratio: float | None = None,
    online_likelihood_ratio_guard_actor_ids: list[str] | None = None,
    stratified_allocation_path: str | Path | None = None,
) -> dict:
    if online_intervention_budget is not None:
        if online_policy_path is None:
            raise ValueError("online_intervention_budget requires --online_policy")
        if (isinstance(online_intervention_budget, bool)
                or int(online_intervention_budget) != online_intervention_budget
                or int(online_intervention_budget) < 1):
            raise ValueError("online_intervention_budget must be a positive integer")
        online_intervention_budget = int(online_intervention_budget)
    if online_max_proposal_likelihood_ratio is not None:
        if online_policy_path is None:
            raise ValueError(
                "online_max_proposal_likelihood_ratio requires --online_policy"
            )
        online_max_proposal_likelihood_ratio = float(
            online_max_proposal_likelihood_ratio
        )
        if (not math.isfinite(online_max_proposal_likelihood_ratio)
                or online_max_proposal_likelihood_ratio <= 1.0):
            raise ValueError(
                "online_max_proposal_likelihood_ratio must be finite and greater than one"
            )
    if online_likelihood_ratio_guard_actor_ids is not None:
        values = list(online_likelihood_ratio_guard_actor_ids)
        allowed = {"BV_primary", "BV_context"}
        if online_policy_path is None or online_max_proposal_likelihood_ratio is None:
            raise ValueError(
                "online_likelihood_ratio_guard_actor_ids requires an online policy and ratio limit"
            )
        if (not values or len(values) != len(set(values))
                or not set(values).issubset(allowed)):
            raise ValueError(
                "online_likelihood_ratio_guard_actor_ids must be unique BV_primary/BV_context ids"
            )
        online_likelihood_ratio_guard_actor_ids = sorted(values)
    online_policy = None
    if online_policy_path is not None:
        if split not in {"train", "validation"}:
            raise ValueError("Online development runs require train or validation; test remains locked")
        if proposal_mode != "factorized" or frozen_epsilon_source != "runtime":
            raise ValueError("Online epsilon requires factorized proposals and runtime epsilon")
        from .d2rl_online_policy import OnlineEpsilonPolicy
        online_policy = OnlineEpsilonPolicy(online_policy_path)
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
    if online_policy is not None:
        from .templates import load_template
        for record in records:
            template = load_template(record['path'])
            if template.bridge_metadata.get('source_split') != split:
                raise ValueError('Template source split disagrees with online run split')
            if template.bridge_metadata.get('not_for_d2rl_training') or any(
                event.params.get('search_only') or event.params.get('calibration_only')
                for event in template.events
            ):
                raise ValueError('Search/calibration intervention cannot enter online evaluation')
    if source_event_ids is not None:
        records = [
            record
            for record in records
            if _source_event_id(record["path"]) in source_event_ids
        ]
    records_after_source_event_filter = len(records)
    if require_context_blocking:
        records = [
            record for record in records
            if _context_blocking_potential(record["path"])
        ]
    records_after_context_blocking_filter = len(records)
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
    stratified_allocation = None
    if stratified_allocation_path is not None:
        if (gui_episode is not None or repeats != 1 or start != 0 or limit is not None
                or source_event_ids is not None or require_context_blocking
                or max_initial_primary_ttc_s is not None):
            raise ValueError(
                "Stratified allocation cannot be combined with GUI, repeats, slicing, "
                "or template filters"
            )
        stratified_allocation = json.loads(
            Path(stratified_allocation_path).read_text(encoding="utf-8")
        )
        planned = {
            str(Path(path)): int(item["additional_rollouts"])
            for path, item in stratified_allocation.get("templates", {}).items()
        }
        selected = {str(Path(record["path"])) for record in records}
        if set(planned) != selected:
            missing = sorted(selected - set(planned))
            extra = sorted(set(planned) - selected)
            raise ValueError(
                f"Stratified allocation template set mismatch; missing={missing}, extra={extra}"
            )
        if any(value < 1 for value in planned.values()):
            raise ValueError("Every stratified template requires at least one rollout")
        if sum(planned.values()) != int(stratified_allocation["allocation_total"]):
            raise ValueError("Stratified allocation total is inconsistent")
    if gui_episode is not None:
        if gui_episode < 0 or gui_episode >= len(records):
            raise ValueError(f"gui_episode={gui_episode} is out of range 0..{len(records) - 1}")
        records_to_run = [(gui_episode, records[gui_episode], 0)]
    elif stratified_allocation is not None:
        records_to_run = []
        episode_id = 0
        for record in records:
            for repeat_index in range(planned[str(Path(record["path"]))]):
                records_to_run.append((episode_id, record, repeat_index))
                episode_id += 1
    else:
        records_to_run = [
            (repeat_index * len(records) + record_index, record, repeat_index)
            for repeat_index in range(repeats)
            for record_index, record in enumerate(records)
        ]

    results = []
    for episode_id, record, repeat_index in records_to_run:
        template_path = record["path"]
        runtime_options = {}
        if (online_policy is not None or frozen_epsilon_source != "template"
                or simulation_seed is not None or online_intervention_budget is not None
                or online_max_proposal_likelihood_ratio is not None
                or online_likelihood_ratio_guard_actor_ids is not None):
            runtime_options = {
                "online_policy": online_policy,
                "frozen_epsilon_source": frozen_epsilon_source,
                "simulation_seed": None if simulation_seed is None else simulation_seed + episode_id,
                "online_intervention_budget": online_intervention_budget,
                "online_max_proposal_likelihood_ratio": (
                    online_max_proposal_likelihood_ratio
                ),
                "online_likelihood_ratio_guard_actor_ids": (
                    online_likelihood_ratio_guard_actor_ids
                ),
            }
            if any((experiment_path / folder / f"{episode_id}.json").exists()
                   for folder in ("crash", "tested_and_safe", "rejected")):
                raise FileExistsError("Use a fresh output directory; rollout files already exist")
        try:
            weight = run_template(
                template_path,
                episode=episode_id,
                experiment_path=str(experiment_path),
                gui=gui_episode is not None,
                epsilon=epsilon,
                proposal_mode=proposal_mode,
                **runtime_options,
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
    importance_weight_diagnostics = None
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
        diagnostics_path = experiment_path / "importance_weight_diagnostics.json"
        if diagnostics_path.is_file():
            with diagnostics_path.open("r", encoding="utf-8") as stream:
                importance_weight_diagnostics = json.load(stream)
    joint_stats = _joint_rollout_stats(experiment_path)
    summary = {
        "manifest": str(manifest_path),
        "experiment_path": str(experiment_path),
        "records_after_split": records_after_split,
        "source_event_ids": sorted(source_event_ids) if source_event_ids is not None else None,
        "records_after_source_event_filter": records_after_source_event_filter,
        "records_after_context_blocking_filter": records_after_context_blocking_filter,
        "require_context_blocking": require_context_blocking,
        "records_after_initial_primary_ttc_filter": len(records),
        "max_initial_primary_ttc_s": max_initial_primary_ttc_s,
        "attempted": len(results),
        "repeats": repeats,
        "online_policy": online_policy.metadata if online_policy is not None else None,
        "frozen_epsilon_source": frozen_epsilon_source,
        "simulation_seed_base": simulation_seed,
        "online_intervention_budget": online_intervention_budget,
        "online_max_proposal_likelihood_ratio": online_max_proposal_likelihood_ratio,
        "online_likelihood_ratio_guard_actor_ids": online_likelihood_ratio_guard_actor_ids,
        "stratified_allocation_path": (
            None if stratified_allocation_path is None else str(stratified_allocation_path)
        ),
        "stratified_allocation": (
            None if stratified_allocation is None else {
                "schema_version": stratified_allocation.get("schema_version"),
                "allocation_total": stratified_allocation["allocation_total"],
                "template_count": stratified_allocation["template_count"],
                "equal_template_mixture_weight": 1.0 / len(planned),
                "rollouts_by_template": planned,
            }
        ),
        "proposal_mode": proposal_mode,
        "epsilon": 1.0 if proposal_mode == "naturalistic" else epsilon,
        "successful_runs": sum(1 for item in results if item["status"] == "ok"),
        "failed_runs": sum(1 for item in results if item["status"] != "ok"),
        "training_ready_crashes": len(crash_weight_dict),
        "training_ready_safe": len(safe_weight_dict),
        "importance_weight_diagnostics": importance_weight_diagnostics,
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


def _context_blocking_potential(template_path: str | Path) -> bool:
    """Read the source-side adjacent-lane blocking flag without inferring one."""
    try:
        with Path(template_path).open("r", encoding="utf-8") as stream:
            template = json.load(stream)
        return bool(
            template["bridge_metadata"]["source_adaptive_critical_window"][
                "context_blocking_potential"
            ]
        )
    except (KeyError, TypeError, OSError, json.JSONDecodeError):
        return False


def _source_event_id(template_path: str | Path) -> int | None:
    """Return the declared SHRP2 event id, rather than inferring one from a filename."""
    try:
        with Path(template_path).open("r", encoding="utf-8") as stream:
            template = json.load(stream)
        return int(template["bridge_metadata"]["source_event_id"])
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
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
        "--epsilon_primary",
        type=float,
        default=None,
        help="Optional epsilon for BV_primary; requires --epsilon_context.",
    )
    parser.add_argument(
        "--epsilon_context",
        type=float,
        default=None,
        help="Optional epsilon for BV_context; requires --epsilon_primary.",
    )
    parser.add_argument(
        "--proposal_mode",
        choices=("naturalistic", "factorized", "joint_pair"),
        default="joint_pair",
        help="Multi-BV action proposal family used in every rollout.",
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
        "--source_event_id",
        action="append",
        type=int,
        default=None,
        help=(
            "Keep only explicitly listed SHRP2 source event ids. Repeat this option "
            "to perform a stratified empirical collision-rate evaluation."
        ),
    )
    parser.add_argument(
        "--require_context_blocking",
        action="store_true",
        help=(
            "Keep only templates whose source-selected context BV is in an "
            "adjacent lane and has declared escape-blocking potential."
        ),
    )
    parser.add_argument(
        "--gui_episode",
        type=int,
        default=None,
        help="Open one manifest record in SUMO GUI instead of running the full batch.",
    )
    parser.add_argument("--online_policy", default=None, help="Verified portable epsilon model (.pt)")
    parser.add_argument("--frozen_epsilon_source", choices=("template", "runtime"), default="template")
    parser.add_argument("--simulation_seed", type=int, default=None)
    parser.add_argument(
        "--online_intervention_budget",
        type=int,
        default=None,
        help="Maximum online-policy critical decisions per episode; then use naturalistic actions.",
    )
    parser.add_argument(
        "--online_max_proposal_likelihood_ratio",
        type=float,
        default=None,
        help="Per-BV upper bound on q(a|s)/p(a|s), enforced before sampling.",
    )
    parser.add_argument(
        "--online_likelihood_ratio_guard_actor",
        action="append",
        choices=("BV_primary", "BV_context"),
        default=None,
        help="Apply the likelihood-ratio bound only to this BV id; repeat if needed.",
    )
    parser.add_argument(
        "--stratified_allocation",
        default=None,
        help=(
            "Validation-only allocation JSON produced by d2rl_stratified_allocation; "
            "uses per-template rollout counts and requires --repeats 1."
        ),
    )
    args = parser.parse_args()

    if (args.epsilon_primary is None) != (args.epsilon_context is None):
        parser.error("--epsilon_primary and --epsilon_context must be provided together")
    epsilon = args.epsilon
    if args.epsilon_primary is not None:
        epsilon = {
            "BV_primary": args.epsilon_primary,
            "BV_context": args.epsilon_context,
        }

    summary = run_template_manifest(
        args.manifest,
        experiment_path=args.experiment_path,
        max_ndd_possi=args.max_ndd_possi,
        gui_episode=args.gui_episode,
        split=args.split,
        start=args.start,
        limit=args.limit,
        repeats=args.repeats,
        epsilon=epsilon,
        proposal_mode=args.proposal_mode,
        max_initial_primary_ttc_s=args.max_initial_primary_ttc_s,
        require_context_blocking=args.require_context_blocking,
        source_event_ids=set(args.source_event_id) if args.source_event_id is not None else None,
        online_policy_path=args.online_policy,
        frozen_epsilon_source=args.frozen_epsilon_source,
        simulation_seed=args.simulation_seed,
        online_intervention_budget=args.online_intervention_budget,
        online_max_proposal_likelihood_ratio=(
            args.online_max_proposal_likelihood_ratio
        ),
        online_likelihood_ratio_guard_actor_ids=(
            args.online_likelihood_ratio_guard_actor
        ),
        stratified_allocation_path=args.stratified_allocation,
    )
    print("Manifest run finished.")
    print(f"attempted={summary['attempted']}")
    print(f"successful_runs={summary['successful_runs']}")
    print(f"failed_runs={summary['failed_runs']}")
    print(f"training_ready_crashes={summary['training_ready_crashes']}")
    print(f"training_ready_safe={summary['training_ready_safe']}")


if __name__ == "__main__":
    main()
