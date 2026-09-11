from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from .augment_templates import generate_template_variants
from .templates import ScenarioTemplate


def generate_simple_multibv_scenarios(
    output_root: str | Path,
    variants_per_scene: int = 4,
    seed: int = 20260618,
) -> dict[str, Any]:
    output_root = Path(output_root)
    template_dir = output_root / "templates"
    template_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for scene_index, base in enumerate(_base_scenarios()):
        scene_dir = template_dir / base["template_id"]
        scene_dir.mkdir(parents=True, exist_ok=True)
        base_template = ScenarioTemplate.from_dict(base)
        base_path = scene_dir / f"{base['template_id']}_base.json"
        _write_json(base, base_path)
        records.append(
            {
                "scene_index": scene_index,
                "scene_id": base["template_id"],
                "variant_index": "base",
                "path": str(base_path),
            }
        )

        variants = generate_template_variants(
            base_template,
            count=variants_per_scene,
            seed=seed + scene_index,
        )
        for variant_index, variant in enumerate(variants):
            variant["template_id"] = f"{base['template_id']}_var_{variant_index:04d}"
            variant_path = scene_dir / f"{variant['template_id']}.json"
            _write_json(variant, variant_path)
            records.append(
                {
                    "scene_index": scene_index,
                    "scene_id": base["template_id"],
                    "variant_index": variant_index,
                    "path": str(variant_path),
                    "applied_perturbations": variant.get("applied_perturbations", {}),
                }
            )

    manifest = {
        "description": "Five simple MultiBV fault scenarios, each with base plus four variants.",
        "scene_count": len(_base_scenarios()),
        "variants_per_scene": variants_per_scene,
        "total_records": len(records),
        "multi_bv_num": 2,
        "records": records,
    }
    manifest_path = output_root / "simple_multibv_manifest.json"
    _write_json(manifest, manifest_path)
    return manifest


def _base_scenarios() -> list[dict[str, Any]]:
    scenarios = [
        _scenario(
            template_id="simple_multibv_cut_in_left",
            description="Two-BV left cut-in fault. The primary BV merges into the CAV lane while a second BV remains nearby as the auxiliary controlled vehicle.",
            ego_lane=1,
            ego_pos=400.0,
            ego_speed=32.0,
            primary_id="BV_cut_in",
            primary_lane=0,
            primary_pos=420.0,
            primary_speed=24.0,
            secondary_id="BV_right_follower",
            secondary_lane=0,
            secondary_pos=389.0,
            secondary_speed=25.0,
            lateral="left",
            longitudinal=0.0,
            start_time=0.2,
            duration=0.9,
        ),
        _scenario(
            template_id="simple_multibv_cut_in_close",
            description="Close-range two-BV cut-in. The cut-in BV starts only slightly ahead of the CAV and a second BV follows in the original lane.",
            ego_lane=1,
            ego_pos=400.0,
            ego_speed=30.0,
            primary_id="BV_cut_in",
            primary_lane=0,
            primary_pos=413.0,
            primary_speed=23.0,
            secondary_id="BV_right_follower",
            secondary_lane=0,
            secondary_pos=384.0,
            secondary_speed=26.0,
            lateral="left",
            longitudinal=-0.5,
            start_time=0.1,
            duration=0.8,
        ),
        _scenario(
            template_id="simple_multibv_brake_and_cut_in",
            description="Cut-in with braking fault. The primary BV both merges toward the CAV lane and decelerates while a nearby second BV is kept under MultiBV logging.",
            ego_lane=1,
            ego_pos=400.0,
            ego_speed=31.0,
            primary_id="BV_brake_cut_in",
            primary_lane=0,
            primary_pos=424.0,
            primary_speed=25.0,
            secondary_id="BV_adjacent_rear",
            secondary_lane=0,
            secondary_pos=392.0,
            secondary_speed=24.0,
            lateral="left",
            longitudinal=-2.5,
            start_time=0.3,
            duration=1.0,
        ),
        _scenario(
            template_id="simple_multibv_right_cut_in",
            description="Mirrored right cut-in. The CAV starts in the right lane and the primary BV cuts in from the left lane.",
            ego_lane=0,
            ego_pos=400.0,
            ego_speed=29.0,
            primary_id="BV_cut_in",
            primary_lane=1,
            primary_pos=417.0,
            primary_speed=22.0,
            secondary_id="BV_left_follower",
            secondary_lane=1,
            secondary_pos=386.0,
            secondary_speed=25.0,
            lateral="right",
            longitudinal=0.0,
            start_time=0.2,
            duration=0.9,
        ),
        _scenario(
            template_id="simple_multibv_lead_slowdown",
            description="Lead-vehicle slowdown with a second adjacent BV. The primary BV remains in-lane and brakes sharply while another BV provides the second controlled external vehicle.",
            ego_lane=1,
            ego_pos=400.0,
            ego_speed=28.0,
            primary_id="BV_lead_slow",
            primary_lane=1,
            primary_pos=426.0,
            primary_speed=21.0,
            secondary_id="BV_adjacent",
            secondary_lane=0,
            secondary_pos=410.0,
            secondary_speed=23.0,
            lateral="central",
            longitudinal=-3.0,
            start_time=0.4,
            duration=1.1,
        ),
    ]
    return scenarios


def _scenario(
    template_id: str,
    description: str,
    ego_lane: int,
    ego_pos: float,
    ego_speed: float,
    primary_id: str,
    primary_lane: int,
    primary_pos: float,
    primary_speed: float,
    secondary_id: str,
    secondary_lane: int,
    secondary_pos: float,
    secondary_speed: float,
    lateral: str,
    longitudinal: float,
    start_time: float,
    duration: float,
) -> dict[str, Any]:
    return {
        "template_id": template_id,
        "description": description,
        "map": "2Lane",
        "route": "route_0",
        "duration": 10.0,
        "tags": ["simple_multibv", "two_bv", "fault_seed"],
        "ego": {
            "id": "CAV",
            "role": "CAV",
            "route": "route_0",
            "lane_index": ego_lane,
            "position": ego_pos,
            "speed": ego_speed,
            "controller": "IDM",
        },
        "actors": [
            {
                "id": primary_id,
                "role": "BV",
                "route": "route_0",
                "lane_index": primary_lane,
                "position": primary_pos,
                "speed": primary_speed,
                "controller": "script",
            },
            {
                "id": secondary_id,
                "role": "BV",
                "route": "route_0",
                "lane_index": secondary_lane,
                "position": secondary_pos,
                "speed": secondary_speed,
                "controller": "IDM",
            },
        ],
        "events": [
            {
                "type": "forced_bv_action",
                "actor": primary_id,
                "start_time": start_time,
                "duration": duration,
                "params": {
                    "lateral": lateral,
                    "longitudinal": longitudinal,
                    "apply_once": False,
                    "multi_bv_num": 2,
                    "epsilon_placeholder": [0.5, 0.5],
                },
            }
        ],
        "perturbations": _perturbations(primary_id, secondary_id),
    }


def _perturbations(primary_id: str, secondary_id: str) -> list[dict[str, Any]]:
    return [
        {"field": "ego.speed", "distribution": "uniform", "low": -1.5, "high": 1.5},
        {
            "field": f"actors.{primary_id}.position",
            "distribution": "uniform",
            "low": -4.0,
            "high": 4.0,
        },
        {
            "field": f"actors.{primary_id}.speed",
            "distribution": "uniform",
            "low": -1.5,
            "high": 1.5,
        },
        {
            "field": f"actors.{secondary_id}.position",
            "distribution": "uniform",
            "low": -5.0,
            "high": 5.0,
        },
        {
            "field": f"actors.{secondary_id}.speed",
            "distribution": "uniform",
            "low": -1.5,
            "high": 1.5,
        },
        {
            "field": "events.forced_bv_action.start_time",
            "distribution": "uniform",
            "low": -0.1,
            "high": 0.4,
        },
    ]


def _write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(copy.deepcopy(data), stream, indent=4, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate simple K=2 MultiBV scenario templates.")
    parser.add_argument(
        "--output_root",
        default="data_analysis/raw_data/SimpleMultiBV25",
        help="Output directory for the 25 template records.",
    )
    parser.add_argument("--variants_per_scene", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260618)
    args = parser.parse_args()

    manifest = generate_simple_multibv_scenarios(
        output_root=args.output_root,
        variants_per_scene=args.variants_per_scene,
        seed=args.seed,
    )
    print(f"Generated {manifest['total_records']} template records.")
    print(f"Manifest: {Path(args.output_root) / 'simple_multibv_manifest.json'}")


if __name__ == "__main__":
    main()
