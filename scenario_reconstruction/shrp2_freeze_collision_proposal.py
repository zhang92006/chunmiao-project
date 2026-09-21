"""Freeze successful CEM discoveries into explicit factorized action proposals."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


ACTION_COUNT = 33


def freeze_collision_proposals(
    cem_summary_path: str | Path,
    output_dir: str | Path,
    epsilon: float = 0.05,
    proposal_version: str = "v1",
) -> dict:
    """Create train-eligible templates with fixed schedules and explicit q action PDFs."""
    if not 0.0 < epsilon < 1.0:
        raise ValueError("epsilon must lie strictly between zero and one")
    if not proposal_version or not proposal_version.replace("_", "").isalnum():
        raise ValueError("proposal_version must contain only letters, numbers, or underscores")
    cem_summary_path = Path(cem_summary_path)
    output_dir = Path(output_dir)
    template_dir = output_dir / "templates" / "train"
    template_dir.mkdir(parents=True, exist_ok=True)
    with cem_summary_path.open("r", encoding="utf-8") as stream:
        summary = json.load(stream)

    records = []
    excluded = []
    for source in summary.get("source_results", []):
        event_id = int(source["source_event_id"])
        best = source.get("best_candidate") or {}
        if int(best.get("target_collision_count", 0)) < 1:
            excluded.append({
                "event_id": event_id,
                "reason": "cem_found_no_target_collision",
            })
            continue
        final_generation = source["generations"][-1]
        distributions = final_generation["updated_distributions"]
        parameters = best["parameters"]
        source_template_path = Path(source["source_template"])
        with source_template_path.open("r", encoding="utf-8") as stream:
            template = json.load(stream)
        candidate = copy.deepcopy(template)
        proposal_id = f"shrp2_frozen_cem_{event_id}_{proposal_version}"
        candidate["template_id"] = proposal_id
        candidate["description"] = (
            "SHRP2 initialization with a frozen, explicit factorized categorical "
            "proposal learned from search-only CEM. Actions are sampled, not forced."
        )
        candidate["events"] = []
        candidate["perturbations"] = []
        tags = [
            tag for tag in candidate.get("tags", [])
            if tag not in {"collision_search_only", "not_for_d2rl_training"}
        ]
        if "frozen_collision_proposal" not in tags:
            tags.append("frozen_collision_proposal")
        candidate["tags"] = tags
        context_start = max(
            0.0,
            float(parameters["start_time_s"])
            + float(parameters["context_delay_s"]),
        )
        candidate.setdefault("bridge_metadata", {})["frozen_collision_proposal"] = {
            "schema_version": 1,
            "proposal_id": proposal_id,
            "source": "search_only_cem_elite_distribution",
            "source_cem_summary": str(cem_summary_path),
            "action_sampling": "per_step_factorized_defensive_mixture",
            "importance_semantics": "q=epsilon*p+(1-epsilon)*categorical_elite_pdf",
            "naturalistic_support_constraint": "runtime_mask_requires_p_gt_zero",
            "agents": {
                "BV_primary": {
                    "start_time_s": float(parameters["start_time_s"]),
                    "duration_s": float(parameters["primary_duration_s"]),
                    "epsilon": float(epsilon),
                    "action_pdf": _dense_action_pdf(
                        distributions["primary_action_id"]
                    ),
                },
                "BV_context": {
                    "start_time_s": context_start,
                    "duration_s": float(parameters["context_duration_s"]),
                    "epsilon": float(epsilon),
                    "action_pdf": _dense_action_pdf(
                        distributions["context_action_id"]
                    ),
                },
            },
        }
        candidate["bridge_metadata"]["collision_search_only"] = False
        candidate["bridge_metadata"]["not_for_d2rl_training"] = False
        output_path = template_dir / f"{proposal_id}.json"
        _write_json(output_path, candidate)
        records.append({
            "event_id": event_id,
            "split": "train",
            "category": candidate["bridge_metadata"].get("source_category"),
            "status": "template_created",
            "template_path": str(output_path),
            "proposal_id": proposal_id,
        })

    manifest_path = output_dir / "frozen_collision_proposal_manifest.json"
    _write_json(manifest_path, {
        "schema_version": 1,
        "proposal_mode": "factorized",
        "search_episodes_reused": False,
        "records": records,
    })
    result = {
        "schema_version": 1,
        "cem_summary": str(cem_summary_path),
        "epsilon": epsilon,
        "proposal_version": proposal_version,
        "frozen_proposal_count": len(records),
        "excluded_count": len(excluded),
        "records": records,
        "excluded": excluded,
        "manifest_path": str(manifest_path),
        "next_step": "fresh factorized rollout; never copy CEM search episodes",
    }
    _write_json(output_dir / "freeze_summary.json", result)
    return result


def _dense_action_pdf(entries: list[dict]) -> list[float]:
    result = [0.0] * ACTION_COUNT
    for entry in entries:
        action_id = int(entry["value"])
        if action_id < 0 or action_id >= ACTION_COUNT:
            raise ValueError(f"Invalid action id in frozen distribution: {action_id}")
        result[action_id] = float(entry["probability"])
    total = sum(result)
    if total <= 0:
        raise ValueError("Frozen action distribution has no probability mass")
    return [value / total for value in result]


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze successful CEM results into explicit factorized proposals."
    )
    parser.add_argument("cem_summary")
    parser.add_argument("--output", required=True)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--proposal_version", default="v1")
    args = parser.parse_args()
    result = freeze_collision_proposals(
        args.cem_summary,
        args.output,
        epsilon=args.epsilon,
        proposal_version=args.proposal_version,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
