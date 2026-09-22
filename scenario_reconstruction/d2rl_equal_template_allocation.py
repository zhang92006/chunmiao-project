"""Create an equal-template rollout allocation from an existing allocation plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_equal_allocation(reference: dict, budget: int) -> dict:
    templates = sorted(reference.get("templates", {}))
    if not templates:
        raise ValueError("Reference allocation has no templates")
    if budget < len(templates):
        raise ValueError("Budget must provide at least one rollout per template")
    base, remainder = divmod(budget, len(templates))
    rows = {}
    for index, template in enumerate(templates):
        source = reference["templates"][template].get("source_event_id", "unknown")
        rows[template] = {
            "source_event_id": str(source),
            "additional_rollouts": base + int(index < remainder),
        }
    return {
        "schema_version": 1,
        "scope": (
            "Equal-template validation control derived from a frozen template set; "
            "not a real-world SHRP2 mixture."
        ),
        "allocation_method": "equal_template",
        "template_count": len(templates),
        "allocation_total": budget,
        "templates": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--budget", required=True, type=int)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    reference = json.loads(Path(args.reference).read_text(encoding="utf-8"))
    result = build_equal_allocation(reference, args.budget)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "template_count": result["template_count"],
        "allocation_total": result["allocation_total"],
        "rollouts_per_template": sorted({
            row["additional_rollouts"] for row in result["templates"].values()
        }),
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
