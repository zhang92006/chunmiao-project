from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Any

from .templates import ScenarioTemplate, load_template


def generate_template_variants(
    template: ScenarioTemplate,
    count: int,
    seed: int | None = None,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    variants = []
    base = _template_to_dict(template)
    for index in range(count):
        variant = copy.deepcopy(base)
        variant["template_id"] = f"{template.template_id}_var_{index:04d}"
        variant["source_template_id"] = template.template_id
        variant["variant_index"] = index
        applied = {}
        for perturbation in template.perturbations:
            delta = rng.uniform(perturbation.low, perturbation.high)
            _apply_delta(variant, perturbation.field, delta)
            applied[perturbation.field] = delta
        variant["applied_perturbations"] = applied
        ScenarioTemplate.from_dict(variant)
        variants.append(variant)
    return variants


def write_variants(
    variants: list[dict[str, Any]],
    output_dir: str | Path,
) -> list[Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    written = []
    for variant in variants:
        path = output_path / f"{variant['template_id']}.json"
        with path.open("w", encoding="utf-8") as stream:
            json.dump(variant, stream, indent=4)
        written.append(path)
    return written


def _template_to_dict(template: ScenarioTemplate) -> dict[str, Any]:
    return {
        "template_id": template.template_id,
        "description": template.description,
        "map": template.map,
        "route": template.route,
        "duration": template.duration,
        "tags": list(template.tags),
        "ego": vars(template.ego).copy(),
        "actors": [vars(actor).copy() for actor in template.actors],
        "events": [
            {
                "type": event.type,
                "actor": event.actor,
                "start_time": event.start_time,
                "duration": event.duration,
                "params": copy.deepcopy(event.params),
            }
            for event in template.events
        ],
        "perturbations": [vars(item).copy() for item in template.perturbations],
    }


def _apply_delta(template: dict[str, Any], field: str, delta: float) -> None:
    parts = field.split(".")
    if not parts:
        raise ValueError("Empty perturbation field.")

    if parts[0] == "ego":
        target = template["ego"]
        key = parts[1]
    elif parts[0] == "actors":
        actor_id = parts[1]
        target = _find_by_id(template["actors"], actor_id)
        key = parts[2]
    elif parts[0] == "events":
        event_type = parts[1]
        target = _find_by_type(template["events"], event_type)
        key = parts[2]
    else:
        raise ValueError(f"Unsupported perturbation root: {parts[0]}")

    if key not in target:
        raise ValueError(f"Perturbation field does not exist: {field}")
    target[key] = float(target[key]) + delta


def _find_by_id(items: list[dict[str, Any]], item_id: str) -> dict[str, Any]:
    for item in items:
        if item.get("id") == item_id:
            return item
    raise ValueError(f"Unknown actor in perturbation: {item_id}")


def _find_by_type(items: list[dict[str, Any]], item_type: str) -> dict[str, Any]:
    for item in items:
        if item.get("type") == item_type:
            return item
    raise ValueError(f"Unknown event type in perturbation: {item_type}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate scenario template variants.")
    parser.add_argument("template", help="Path to the base scenario template.")
    parser.add_argument("--count", type=int, default=10, help="Number of variants.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--output_dir",
        default="scenario_reconstruction/generated_templates",
        help="Directory to store generated JSON variants.",
    )
    args = parser.parse_args()

    template = load_template(args.template)
    variants = generate_template_variants(template, args.count, seed=args.seed)
    written = write_variants(variants, args.output_dir)
    print(f"Generated {len(written)} variants in {args.output_dir}")


if __name__ == "__main__":
    main()
