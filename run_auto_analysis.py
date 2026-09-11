from __future__ import annotations

import argparse
import json
from pathlib import Path

from .autoware_summary import summarize_autoware_bag, write_summary
from .llm_template_generator import DEFAULT_QWEN_MODEL, generate_templates_from_summary
from .templates import load_template


def run_auto_analysis(
    metadata_path: str,
    db3_path: str,
    output_root: str,
    decoded_summary_path: str | None = None,
    template_count: int = 2,
    variant_count: int = 5,
    seed: int = 7,
    model: str = DEFAULT_QWEN_MODEL,
    use_llm: bool = True,
    allow_fallback: bool = True,
    skip_simulation: bool = False,
    max_ndd_possi: float | None = 0.01,
    target_end_time: float = 1.4,
    target_collision_actor: str = "BV_cut_in",
) -> dict:
    output_dir = Path(output_root)
    summary_dir = output_dir / "summary"
    template_dir = output_dir / "templates"
    batch_dir = output_dir / "batches"
    summary_dir.mkdir(parents=True, exist_ok=True)
    template_dir.mkdir(parents=True, exist_ok=True)
    batch_dir.mkdir(parents=True, exist_ok=True)

    autoware_summary = summarize_autoware_bag(metadata_path, db3_path)
    if decoded_summary_path:
        autoware_summary["decoded_signal_summary"] = _load_decoded_summary(
            decoded_summary_path
        )
    summary_path = write_summary(autoware_summary, summary_dir / "autoware_summary.json")

    template_paths = generate_templates_from_summary(
        autoware_summary,
        count=template_count,
        output_dir=template_dir,
        model=model,
        use_llm=use_llm,
        allow_fallback=allow_fallback,
    )
    manifest_path = template_dir / "template_generation_manifest.json"

    template_infos = []
    batch_summaries = []
    for template_path in template_paths:
        template = load_template(template_path)
        template_infos.append(
            {
                "template_id": template.template_id,
                "path": str(template_path),
                "actors": [actor.id for actor in template.actors],
                "events": [event.type for event in template.events],
                "perturbations": [item.field for item in template.perturbations],
            }
        )
        if skip_simulation:
            continue
        from .run_batch import run_batch

        batch_output = batch_dir / template.template_id
        batch_summaries.append(
            run_batch(
                str(template_path),
                count=variant_count,
                output_root=str(batch_output),
                seed=seed,
                max_ndd_possi=max_ndd_possi,
                target_end_time=target_end_time,
                target_collision_actor=target_collision_actor,
            )
        )

    result = {
        "metadata_path": metadata_path,
        "db3_path": db3_path,
        "decoded_summary_path": decoded_summary_path,
        "output_root": str(output_dir),
        "summary_path": str(summary_path),
        "template_generation_manifest": str(manifest_path),
        "template_count": len(template_paths),
        "variant_count": variant_count,
        "skip_simulation": skip_simulation,
        "templates": template_infos,
        "batches": _compact_batch_summaries(batch_summaries),
    }
    result["evaluation"] = _build_evaluation(manifest_path, result["batches"])
    with (output_dir / "auto_analysis_summary.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(result, stream, indent=4, ensure_ascii=False)
    return result


def _compact_batch_summaries(summaries: list[dict]) -> list[dict]:
    compact = []
    for summary in summaries:
        outcomes = summary.get("outcomes", [])
        compact.append(
            {
                "template": summary.get("template"),
                "output_root": summary.get("output_root"),
                "successful_runs": summary.get("successful_runs"),
                "failed_runs": summary.get("failed_runs"),
                "training_ready_crashes": summary.get("training_ready_crashes"),
                "best_outcome": outcomes[0] if outcomes else None,
            }
        )
    return compact


def _load_decoded_summary(path: str) -> dict:
    decoded_path = Path(path)
    with decoded_path.open("r", encoding="utf-8") as stream:
        data = json.load(stream)
    # Keep the LLM input compact while preserving real state evidence.
    return {
        "source_path": str(decoded_path),
        "registered_custom_type_count": data.get("registered_custom_type_count"),
        "topics": {
            topic: {
                "msgtype": topic_data.get("msgtype"),
                "decoded_count": topic_data.get("decoded_count"),
                "stats": topic_data.get("stats"),
                "samples": topic_data.get("samples", [])[:2],
            }
            for topic, topic_data in (data.get("topics") or {}).items()
        },
    }


def _build_evaluation(manifest_path: Path, batches: list[dict]) -> dict:
    manifest = {}
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as stream:
            manifest = json.load(stream)
    candidates = manifest.get("candidates", [])
    unusable_errors = {}
    for candidate in candidates:
        error = candidate.get("validation_error")
        if error:
            unusable_errors[error] = unusable_errors.get(error, 0) + 1
    return {
        "generation_source": manifest.get("source"),
        "model": manifest.get("model"),
        "llm_error": manifest.get("llm_error"),
        "requested_template_count": manifest.get("requested_template_count"),
        "raw_candidate_count": manifest.get("raw_candidate_count"),
        "usable_template_count": manifest.get("usable_template_count"),
        "unusable_template_count": manifest.get("unusable_template_count"),
        "unusable_error_counts": unusable_errors,
        "simulation_batch_count": len(batches),
        "successful_runs": sum(item.get("successful_runs") or 0 for item in batches),
        "failed_runs": sum(item.get("failed_runs") or 0 for item in batches),
        "training_ready_crashes": sum(
            item.get("training_ready_crashes") or 0 for item in batches
        ),
        "best_outcomes": [
            item.get("best_outcome") for item in batches if item.get("best_outcome")
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate accident templates from Autoware rosbag metadata/db3, "
            "then run variant generation and collision scoring."
        )
    )
    parser.add_argument("--metadata", default="metadata.yaml")
    parser.add_argument("--db3", default="2019_07_9_cuted_run01_0.db3")
    parser.add_argument(
        "--decoded_summary",
        default=None,
        help="Optional decoded Autoware CDR summary JSON to include in LLM input.",
    )
    parser.add_argument(
        "--output_root",
        default="data_analysis/raw_data/AutowareAutoAnalysis",
    )
    parser.add_argument("--templates", type=int, default=2)
    parser.add_argument("--variants", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--model", default=DEFAULT_QWEN_MODEL)
    parser.add_argument(
        "--no_llm",
        action="store_true",
        help="Use local fallback template generation instead of Qwen.",
    )
    parser.add_argument(
        "--disable_fallback",
        action="store_true",
        help="Do not generate fallback templates when Qwen is unavailable or unusable.",
    )
    parser.add_argument(
        "--skip_simulation",
        action="store_true",
        help="Only generate and validate templates; do not run SUMO batches.",
    )
    parser.add_argument("--max_ndd_possi", type=float, default=0.01)
    parser.add_argument("--target_end_time", type=float, default=1.4)
    parser.add_argument("--target_collision_actor", default="BV_cut_in")
    args = parser.parse_args()

    result = run_auto_analysis(
        args.metadata,
        args.db3,
        args.output_root,
        decoded_summary_path=args.decoded_summary,
        template_count=args.templates,
        variant_count=args.variants,
        seed=args.seed,
        model=args.model,
        use_llm=not args.no_llm,
        allow_fallback=not args.disable_fallback,
        skip_simulation=args.skip_simulation,
        max_ndd_possi=args.max_ndd_possi,
        target_end_time=args.target_end_time,
        target_collision_actor=args.target_collision_actor,
    )
    print("Auto analysis finished.")
    print(f"summary={Path(args.output_root) / 'auto_analysis_summary.json'}")
    print(f"templates={result['template_count']}")
    if args.skip_simulation:
        print("simulation=skipped")
    else:
        print(f"batches={len(result['batches'])}")


if __name__ == "__main__":
    main()
