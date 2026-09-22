"""Generate and evaluate auditable SUMO calibration candidates for SHRP2 rear-end seeds.

Calibration searches a limited set of initial gaps and declared interventions.
Schema v1 scripts primary-BV braking only.  Schema v2 also applies an explicit,
calibration-only CAV response hold; it is a reachability test, not a modeled
perception or controller delay.  A selected candidate is only a SUMO
calibration result and must never be labelled an exact SHRP2 crash replay.
"""
from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
from pathlib import Path
from typing import Any

from .run_template import run_template
from .templates import ScenarioTemplate

SELECTION_NUMERICAL_EPSILON = 1e-9
CALIBRATION_SOURCE_QUALITIES = {"high", "audited"}


def validate_config(config: dict[str, Any]) -> None:
    schema_version = config.get("schema_version")
    if schema_version not in (1, 2):
        raise ValueError("Only SHRP2 calibration schema_version 1 or 2 is supported")
    if config.get("source_split") != "train":
        raise ValueError("Collision calibration may use only the project train split")
    for name in ("gap_offsets_m", "braking_accelerations_mps2", "braking_start_times_s"):
        values = config.get(name)
        if not isinstance(values, list) or not values or any(
            isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
            for value in values
        ):
            raise ValueError(f"{name} must be a non-empty list of finite numbers")
    if any(value > 0 for value in config["braking_accelerations_mps2"]):
        raise ValueError(
            "braking_accelerations_mps2 values must be non-positive; zero denotes an explicit speed hold"
        )
    if schema_version == 2:
        for name in (
            "cav_override_accelerations_mps2",
            "cav_override_durations_s",
        ):
            values = config.get(name)
            if not isinstance(values, list) or not values or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                for value in values
            ):
                raise ValueError(f"{name} must be a non-empty list of finite numbers")
        if any(value <= 0 for value in config["cav_override_durations_s"]):
            raise ValueError("cav_override_durations_s values must be positive")
        if any(
            value < -4 or value > 2
            for value in config["cav_override_accelerations_mps2"]
        ):
            raise ValueError(
                "cav_override_accelerations_mps2 values must stay within [-4, 2]"
            )
        cav_start = config.get("cav_override_start_time_s")
        if (
            isinstance(cav_start, bool)
            or not isinstance(cav_start, (int, float))
            or not math.isfinite(cav_start)
            or cav_start < 0
        ):
            raise ValueError("cav_override_start_time_s must be finite and non-negative")
    for name in ("braking_duration_s", "minimum_initial_gap_m", "target_collision_time_s", "time_tolerance_s"):
        value = config.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not isinstance(config.get("max_candidates"), int) or config["max_candidates"] <= 0:
        raise ValueError("max_candidates must be a positive integer")


def generate_calibration_candidates(
    template_path: str | Path, output: str | Path, config: dict[str, Any]
) -> dict[str, Any]:
    """Generate a bounded grid of calibration templates without running SUMO."""
    validate_config(config)
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new or empty calibration output directory")
    source_path = Path(template_path)
    source = _load_json_template(source_path)
    source_template = ScenarioTemplate.from_dict(source)
    _validate_source_template(source, source_template, config)

    primary = _find_actor(source, "BV_primary")
    source_gap = float(primary["position"]) - float(source["ego"]["position"])
    schema_version = int(config["schema_version"])
    cav_accelerations = (
        config["cav_override_accelerations_mps2"]
        if schema_version == 2
        else [None]
    )
    cav_durations = (
        config["cav_override_durations_s"] if schema_version == 2 else [None]
    )
    combinations = list(
        itertools.product(
            config["gap_offsets_m"],
            config["braking_accelerations_mps2"],
            config["braking_start_times_s"],
            cav_accelerations,
            cav_durations,
        )
    )
    if len(combinations) > config["max_candidates"]:
        raise ValueError(
            f"Grid has {len(combinations)} candidates, above max_candidates={config['max_candidates']}"
        )

    records = []
    for index, (
        gap_offset,
        braking_accel,
        braking_start,
        cav_accel,
        cav_duration,
    ) in enumerate(combinations):
        initial_gap = source_gap + float(gap_offset)
        record = {
            "candidate_index": index,
            "gap_offset_m": float(gap_offset),
            "initial_gap_m": initial_gap,
            "braking_acceleration_mps2": float(braking_accel),
            "braking_start_time_s": float(braking_start),
            "braking_duration_s": float(config["braking_duration_s"]),
        }
        if schema_version == 2:
            record.update(
                {
                    "cav_override_acceleration_mps2": float(cav_accel),
                    "cav_override_start_time_s": float(
                        config["cav_override_start_time_s"]
                    ),
                    "cav_override_duration_s": float(cav_duration),
                }
            )
        if initial_gap < config["minimum_initial_gap_m"]:
            record.update({
                "status": "rejected_before_run",
                "reason": "initial gap violates minimum_initial_gap_m",
            })
            records.append(record)
            continue
        if braking_start + config["braking_duration_s"] > source_template.duration:
            record.update({
                "status": "rejected_before_run",
                "reason": "braking event extends beyond template duration",
            })
            records.append(record)
            continue
        if schema_version == 2 and (
            config["cav_override_start_time_s"] + float(cav_duration)
            > source_template.duration
        ):
            record.update({
                "status": "rejected_before_run",
                "reason": "CAV calibration action extends beyond template duration",
            })
            records.append(record)
            continue
        candidate = _calibrated_template(
            source,
            index,
            initial_gap,
            float(braking_accel),
            float(braking_start),
            config,
            cav_accel=None if cav_accel is None else float(cav_accel),
            cav_duration=None if cav_duration is None else float(cav_duration),
        )
        ScenarioTemplate.from_dict(candidate)
        path = output / "templates" / f"{candidate['template_id']}.json"
        _write_json(path, candidate)
        record.update({"status": "generated", "template_path": str(path)})
        records.append(record)

    manifest = {
        "status": "generated_only; pass --run to execute SUMO candidates",
        "source_template": str(source_path),
        "source_scenario_id": source.get("bridge_metadata", {}).get("source_scenario_id"),
        "source_split": config["source_split"],
        "schema_version": schema_version,
        "target_collision_time_s": float(config["target_collision_time_s"]),
        "time_tolerance_s": float(config["time_tolerance_s"]),
        "minimum_initial_gap_m": float(config["minimum_initial_gap_m"]),
        "initial_gap_semantics": (
            "longitudinal position delta from CAV to BV_primary; "
            "not physical bumper clearance"
        ),
        "candidate_count": len(records),
        "generated_count": sum(record["status"] == "generated" for record in records),
        "rejected_count": sum(record["status"] != "generated" for record in records),
        "records": records,
        "limitations": [
            "Candidate primary-BV longitudinal action is an explicitly declared simulation intervention, not inferred driver behavior.",
            "Generated files are calibration inputs and must not be inserted directly into D2RL training data.",
            "Selection uses only train-split inputs; validation is reserved for freezing the calibration protocol.",
        ],
    }
    if schema_version == 2:
        manifest["limitations"].insert(
            1,
            "The CAV action is a reachability intervention, not a perception-delay, control-delay, or human-response model.",
        )
    _write_json(output / "calibration_manifest.json", manifest)
    return manifest


def run_calibration(manifest_path: str | Path, max_candidates: int | None = None) -> dict[str, Any]:
    """Run generated candidates and write selection metrics beside the manifest."""
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    generated = [record for record in manifest["records"] if record["status"] == "generated"]
    if max_candidates is not None:
        if max_candidates <= 0:
            raise ValueError("max_candidates must be positive")
        generated = generated[:max_candidates]
    episode_root = manifest_path.parent / "episodes"
    outcomes = []
    for record in generated:
        index = int(record["candidate_index"])
        try:
            run_template(record["template_path"], episode=index, experiment_path=str(episode_root))
            episode_path, episode = _load_episode(episode_root, index)
            outcome = _outcome(record, episode_path, episode, manifest)
        except Exception as exc:
            outcome = {**record, "status": "run_error", "error": str(exc)}
        outcomes.append(outcome)

    return _write_calibration_summary(
        manifest_path,
        manifest,
        outcomes,
        status=(
            "calibration execution; selected records are SUMO calibration results, "
            "not exact SHRP2 replays"
        ),
    )


def rescore_calibration(manifest_path: str | Path) -> dict[str, Any]:
    """Re-evaluate completed episodes without launching SUMO again."""
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    generated = [record for record in manifest["records"] if record["status"] == "generated"]
    episode_root = manifest_path.parent / "episodes"
    outcomes = []
    for record in generated:
        index = int(record["candidate_index"])
        try:
            episode_path, episode = _load_episode(episode_root, index)
            outcome = _outcome(record, episode_path, episode, manifest)
        except Exception as exc:
            outcome = {**record, "status": "rescore_error", "error": str(exc)}
        outcomes.append(outcome)
    return _write_calibration_summary(
        manifest_path,
        manifest,
        outcomes,
        status=(
            "calibration rescore of completed episodes; selected records are SUMO "
            "calibration results, not exact SHRP2 replays"
        ),
    )


def _write_calibration_summary(
    manifest_path: Path,
    manifest: dict[str, Any],
    outcomes: list[dict[str, Any]],
    status: str,
) -> dict[str, Any]:
    selected = [record for record in outcomes if record.get("selection_status") == "selected"]
    summary = {
        "status": status,
        "manifest": str(manifest_path),
        "executed_count": len(outcomes),
        "selected_count": len(selected),
        "selection_rule": {
            "initial_gap_m_at_least": manifest["minimum_initial_gap_m"],
            "collision_result": True,
            "collision_actor": "BV_primary",
            "absolute_collision_time_error_s_at_most": manifest["time_tolerance_s"],
            "numerical_epsilon_s": SELECTION_NUMERICAL_EPSILON,
        },
        "outcomes": outcomes,
        "selected": selected,
    }
    _write_json(manifest_path.parent / "calibration_summary.json", summary)
    return summary


def _validate_source_template(source: dict[str, Any], template: ScenarioTemplate, config: dict[str, Any]) -> None:
    metadata = source.get("bridge_metadata", {})
    if metadata.get("source_quality") not in CALIBRATION_SOURCE_QUALITIES:
        raise ValueError(
            "Calibration requires source_quality='high' or a quality-audited SHRP2 bridge template"
        )
    if metadata.get("source_split") not in (None, config["source_split"]):
        raise ValueError("Template source_split does not match calibration source_split")
    if "rear_end" not in template.tags or template.map != "2Lane":
        raise ValueError("Calibration requires a 2Lane rear_end bridge template")
    if template.duration <= config["target_collision_time_s"]:
        raise ValueError("Template duration must exceed target_collision_time_s")


def _calibrated_template(
    source,
    index,
    initial_gap,
    braking_accel,
    braking_start,
    config,
    cav_accel=None,
    cav_duration=None,
):
    candidate = copy.deepcopy(source)
    candidate["template_id"] = (
        f"{source['template_id']}_cal_{index:03d}_gap_{initial_gap:.3f}_brake_{abs(braking_accel):.2f}_at_{braking_start:.2f}"
    )
    candidate["description"] = (
        "Calibration candidate derived from a high-quality or quality-audited SHRP2 rear-end initialization. "
        "Its scripted primary-BV longitudinal action is a declared SUMO intervention, not an exact crash replay."
    )
    primary = _find_actor(candidate, "BV_primary")
    primary["position"] = float(candidate["ego"]["position"]) + initial_gap
    candidate["events"] = [{
        "type": "forced_bv_action",
        "actor": "BV_primary",
        "start_time": braking_start,
        "duration": float(config["braking_duration_s"]),
        "params": {
            "lateral": "central",
            "longitudinal": braking_accel,
            "apply_once": False,
            "multi_bv_num": 2,
            "calibration_only": True,
            "not_for_d2rl_training": True,
        },
    }]
    if int(config["schema_version"]) == 2:
        candidate["events"].append({
            "type": "calibration_cav_action",
            "actor": candidate["ego"]["id"],
            "start_time": float(config["cav_override_start_time_s"]),
            "duration": cav_duration,
            "params": {
                "lateral": "central",
                "longitudinal": cav_accel,
                "apply_once": False,
                "calibration_only": True,
                "not_for_d2rl_training": True,
            },
        })
    candidate["perturbations"] = []
    candidate["calibration_metadata"] = {
        "protocol": (
            "rear_end_braking_grid_v1"
            if int(config["schema_version"]) == 1
            else "rear_end_reachability_grid_v2"
        ),
        "source_split": config["source_split"],
        "initial_gap_m": initial_gap,
        "initial_gap_semantics": (
            "longitudinal position delta from CAV to BV_primary; "
            "not physical bumper clearance"
        ),
        "braking_acceleration_mps2": braking_accel,
        "braking_start_time_s": braking_start,
        "braking_duration_s": config["braking_duration_s"],
        "primary_bv_action_semantics": (
            "negative value denotes braking; zero denotes an explicit speed hold"
        ),
        "target_collision_time_s": config["target_collision_time_s"],
        "not_for_d2rl_training": True,
    }
    if int(config["schema_version"]) == 2:
        candidate["description"] = (
            "Reachability candidate derived from a high-quality or quality-audited SHRP2 rear-end "
            "initialization. Its BV longitudinal action and CAV response hold are declared "
            "calibration interventions, not an exact crash replay or delay model."
        )
        candidate["calibration_metadata"].update({
            "cav_override_acceleration_mps2": cav_accel,
            "cav_override_start_time_s": config["cav_override_start_time_s"],
            "cav_override_duration_s": cav_duration,
            "cav_override_semantics": "calibration reachability intervention; not a delay model",
        })
    return candidate


def _outcome(record, episode_path, episode, manifest):
    collision_ids = list(episode.get("collision_id") or [])
    collision = bool(episode.get("collision_result"))
    end_time = float(episode.get("episode_info", {}).get("end_time", math.inf))
    error = end_time - float(manifest["target_collision_time_s"]) if collision else None
    is_target_collision = collision and "BV_primary" in collision_ids and "CAV" in collision_ids
    selected = (
        record["initial_gap_m"] >= manifest["minimum_initial_gap_m"]
        and is_target_collision
        and error is not None
        and abs(error) <= manifest["time_tolerance_s"] + SELECTION_NUMERICAL_EPSILON
    )
    return {
        **record,
        "status": "executed",
        "episode_path": str(episode_path),
        "collision_result": collision,
        "collision_ids": collision_ids,
        "end_time_s": end_time,
        "collision_time_error_s": error,
        "minimum_ttc_s": _minimum(episode.get("ttc_step_info", {})),
        "minimum_distance_m": _minimum(episode.get("distance_step_info", {})),
        "selection_status": "selected" if selected else "not_selected",
    }


def _load_episode(episode_root: Path, index: int):
    for category in ("crash", "tested_and_safe"):
        path = episode_root / category / f"{index}.json"
        if path.is_file():
            return path, json.loads(path.read_text(encoding="utf-8"))
    raise FileNotFoundError(f"No episode JSON written for calibration candidate {index}")


def _minimum(values: dict) -> float | None:
    return min((float(value) for value in values.values()), default=None)


def _find_actor(template: dict[str, Any], actor_id: str) -> dict[str, Any]:
    for actor in template.get("actors", []):
        if actor.get("id") == actor_id:
            return actor
    raise ValueError(f"Template has no {actor_id} actor")


def _load_json_template(path: Path) -> dict[str, Any]:
    if path.suffix.lower() != ".json":
        raise ValueError("Calibration currently accepts the JSON templates emitted by shrp2_sumo_bridge")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Template JSON must contain an object")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "template", nargs="?", help="High-quality rear-end JSON template from shrp2_sumo_bridge"
    )
    parser.add_argument("--output", help="New or empty calibration output directory")
    parser.add_argument("--config", default="configs/shrp2_rear_end_calibration.json")
    parser.add_argument("--run", action="store_true", help="Execute generated SUMO candidates")
    parser.add_argument("--max_candidates", type=int, help="Optional cap when --run is set")
    parser.add_argument(
        "--rescore",
        metavar="MANIFEST",
        help="Re-score existing completed episodes without launching SUMO",
    )
    args = parser.parse_args()
    if args.rescore:
        if args.template or args.output or args.run or args.max_candidates is not None:
            parser.error("--rescore cannot be combined with template, --output, --run, or --max_candidates")
        summary = rescore_calibration(args.rescore)
        print(json.dumps({
            "calibration_summary_path": str(Path(args.rescore).parent / "calibration_summary.json"),
            "executed_count": summary["executed_count"],
            "selected_count": summary["selected_count"],
        }, ensure_ascii=False, indent=2))
        return
    if not args.template or not args.output:
        parser.error("template and --output are required unless --rescore is used")
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    manifest = generate_calibration_candidates(args.template, args.output, config)
    result = {
        "manifest_path": str(Path(args.output) / "calibration_manifest.json"),
        "candidate_count": manifest["candidate_count"],
        "generated_count": manifest["generated_count"],
        "rejected_count": manifest["rejected_count"],
    }
    if args.run:
        execution = run_calibration(
            Path(args.output) / "calibration_manifest.json", max_candidates=args.max_candidates
        )
        result.update({
            "calibration_summary_path": str(Path(args.output) / "calibration_summary.json"),
            "executed_count": execution["executed_count"],
            "selected_count": execution["selected_count"],
        })
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
