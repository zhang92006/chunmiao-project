"""Build and run one frozen, fault-semantic SHRP2 rear-end validation scenario."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from .run_template import run_template
from .templates import ScenarioTemplate


def build_fault_validation_template(
    source_path: str | Path, output: str | Path, config: dict[str, Any]
) -> dict[str, Any]:
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new or empty fault-validation output directory")
    source = json.loads(Path(source_path).read_text(encoding="utf-8"))
    source_template = ScenarioTemplate.from_dict(source)
    if config.get("schema_version") != 1 or config.get("source_split") != "train":
        raise ValueError("Fault validation requires schema_version=1 and source_split='train'")
    if source.get("bridge_metadata", {}).get("source_quality") != "high" or "rear_end" not in source_template.tags:
        raise ValueError("Fault validation requires a high-quality SHRP2 rear_end bridge template")
    _validate_config(config, source_template.duration)
    candidate = copy.deepcopy(source)
    candidate["template_id"] = f"{source['template_id']}_fault_v3_control_delay"
    candidate["description"] = (
        "Fault-semantic validation derived from a frozen SHRP2 rear-end calibration. "
        "The CAV control delay is an explicit controller-boundary fault model, not an exact crash replay."
    )
    primary = next(actor for actor in candidate["actors"] if actor["id"] == "BV_primary")
    initial_gap = float(primary["position"]) - float(candidate["ego"]["position"]) + float(config["gap_offset_m"])
    primary["position"] = float(candidate["ego"]["position"]) + initial_gap
    candidate["events"] = [
        {
            "type": "forced_bv_action", "actor": "BV_primary",
            "start_time": float(config["bv_braking_start_time_s"]),
            "duration": float(config["bv_braking_duration_s"]),
            "params": {"lateral": "central", "longitudinal": float(config["bv_braking_acceleration_mps2"]), "apply_once": False, "calibration_only": True, "not_for_d2rl_training": True},
        },
        {
            "type": "control_delay", "actor": "CAV",
            "start_time": float(config["control_delay_start_time_s"]),
            "duration": float(config["control_delay_duration_s"]),
            "params": {"delay_s": float(config["control_delay_s"]), "initial_action": copy.deepcopy(config["initial_action"]), "fault_validation_only": True, "not_for_d2rl_training": True},
        },
    ]
    candidate["perturbations"] = []
    candidate["fault_validation_metadata"] = {
        "protocol": "shrp2_rear_end_fault_semantic_v3",
        "source_split": "train", "not_for_d2rl_training": True,
        "initial_position_delta_m": initial_gap,
        "target_collision_time_s": config["target_collision_time_s"],
        "control_delay_semantics": "controller output is replayed after delay_s; initial_action is applied before sufficient history exists",
    }
    ScenarioTemplate.from_dict(candidate)
    template_path = output / "template.json"
    template_path.parent.mkdir(parents=True, exist_ok=True)
    template_path.write_text(json.dumps(candidate, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {"template_path": str(template_path), "source_path": str(source_path), "config": config, "status": "generated_only"}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def run_fault_validation(manifest_path: str | Path) -> dict[str, Any]:
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = manifest_path.parent
    run_template(manifest["template_path"], episode=0, experiment_path=str(root / "episode"))
    episode_path = next(iter((root / "episode" / "crash").glob("0.json")), None)
    if episode_path is None:
        episode_path = root / "episode" / "tested_and_safe" / "0.json"
    episode = json.loads(episode_path.read_text(encoding="utf-8"))
    collision = bool(episode.get("collision_result"))
    end_time = float(episode.get("episode_info", {}).get("end_time", float("inf")))
    error = end_time - float(manifest["config"]["target_collision_time_s"]) if collision else None
    summary = {
        "status": "executed; fault-semantic validation, not an exact SHRP2 replay",
        "template_path": manifest["template_path"], "episode_path": str(episode_path),
        "collision_result": collision, "collision_ids": episode.get("collision_id") or [],
        "end_time_s": end_time, "collision_time_error_s": error,
        "selected": bool(collision and error is not None and abs(error) <= float(manifest["config"]["time_tolerance_s"]) + 1e-9),
        "fault_log_step_count": len(episode.get("cav_fault_step_info", {})),
    }
    (root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _validate_config(config: dict[str, Any], duration: float) -> None:
    required_positive = ("bv_braking_duration_s", "control_delay_duration_s", "control_delay_s", "target_collision_time_s", "time_tolerance_s")
    for name in required_positive:
        if not isinstance(config.get(name), (int, float)) or isinstance(config[name], bool) or config[name] <= 0:
            raise ValueError(f"{name} must be positive")
    if config["bv_braking_acceleration_mps2"] >= 0:
        raise ValueError("bv_braking_acceleration_mps2 must be negative")
    for name in ("bv_braking_start_time_s", "control_delay_start_time_s", "gap_offset_m"):
        if not isinstance(config.get(name), (int, float)) or isinstance(config[name], bool):
            raise ValueError(f"{name} must be numeric")
    if config["bv_braking_start_time_s"] + config["bv_braking_duration_s"] > duration or config["control_delay_start_time_s"] + config["control_delay_duration_s"] > duration:
        raise ValueError("fault event extends beyond scenario duration")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_template")
    parser.add_argument("--config", default="configs/shrp2_rear_end_fault_validation_v3.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    manifest = build_fault_validation_template(args.source_template, args.output, config)
    result: dict[str, Any] = {"manifest_path": str(Path(args.output) / "manifest.json")}
    if args.run:
        result["summary"] = run_fault_validation(Path(args.output) / "manifest.json")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
