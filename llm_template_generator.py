from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .templates import ScenarioTemplate


DEFAULT_QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
DEFAULT_QWEN_MODEL = "qwen-plus"


def generate_templates_from_summary(
    summary: dict,
    count: int,
    output_dir: str | Path,
    model: str = DEFAULT_QWEN_MODEL,
    use_llm: bool = True,
    allow_fallback: bool = True,
) -> list[Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    templates, generation = _generate_with_llm_or_fallback(
        summary,
        count=count,
        model=model,
        use_llm=use_llm,
        allow_fallback=allow_fallback,
    )
    written = []
    candidate_records = []
    for index, template in enumerate(templates):
        candidate = _normalize_template(template, index)
        record = {
            "candidate_index": index,
            "template_id": candidate.get("template_id"),
            "status": "unusable",
            "path": None,
            "validation_error": None,
            "raw_template": template,
            "candidate_template": candidate,
        }
        try:
            ScenarioTemplate.from_dict(candidate)
        except Exception as exc:
            record["validation_error"] = str(exc)
            candidate_records.append(record)
            continue

        path = output_path / f"{candidate['template_id']}.json"
        with path.open("w", encoding="utf-8") as stream:
            json.dump(candidate, stream, indent=4, ensure_ascii=False)
        written.append(path)
        record["status"] = "usable"
        record["path"] = str(path)
        candidate_records.append(record)
        if len(written) >= count:
            break

    manifest = {
        "source": generation["source"],
        "model": model if generation["source"] == "llm" else None,
        "llm_error": generation.get("error"),
        "raw_response": generation.get("raw_response"),
        "requested_template_count": count,
        "raw_candidate_count": len(templates),
        "usable_template_count": len(written),
        "unusable_template_count": sum(
            1 for item in candidate_records if item["status"] != "usable"
        ),
        "templates": [str(path) for path in written],
        "candidates": candidate_records,
    }
    with (output_path / "template_generation_manifest.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(manifest, stream, indent=4, ensure_ascii=False)
    return written


def _generate_with_llm_or_fallback(
    summary: dict,
    count: int,
    model: str,
    use_llm: bool,
    allow_fallback: bool,
) -> tuple[list[dict], dict]:
    if use_llm:
        api_key = _read_api_key()
        if api_key:
            try:
                templates, raw_response = _generate_with_qwen(
                    summary, count, model, api_key
                )
                return templates, {
                    "source": "llm",
                    "raw_response": raw_response,
                    "error": None,
                }
            except (TimeoutError, urllib.error.URLError, ValueError, KeyError) as exc:
                if not allow_fallback:
                    return [], {
                        "source": "llm_failed",
                        "raw_response": None,
                        "error": str(exc),
                    }
                print(f"LLM generation failed, using fallback templates: {exc}")
                return _fallback_templates(summary, count), {
                    "source": "fallback_after_llm_failure",
                    "raw_response": None,
                    "error": str(exc),
                }
        if not allow_fallback:
            return [], {
                "source": "llm_unavailable",
                "raw_response": None,
                "error": "DASHSCOPE_API_KEY or QWEN_API_KEY is not set.",
            }
    return _fallback_templates(summary, count), {
        "source": "fallback",
        "raw_response": None,
        "error": None if not use_llm else "DASHSCOPE_API_KEY or QWEN_API_KEY is not set.",
    }


def _read_api_key() -> str | None:
    return os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("QWEN_API_KEY")


def _generate_with_qwen(
    summary: dict,
    count: int,
    model: str,
    api_key: str,
) -> tuple[list[dict], str]:
    base_url = os.environ.get("QWEN_BASE_URL", DEFAULT_QWEN_BASE_URL)
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": _system_prompt(),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "requested_template_count": count,
                        "autoware_summary": summary,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
            },
        ],
        "temperature": 0.2,
    }
    request = urllib.request.Request(
        base_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    timeout = float(os.environ.get("QWEN_TIMEOUT", "120"))
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = json.loads(response.read().decode("utf-8"))
    content = data["choices"][0]["message"]["content"]
    templates = _parse_templates(content)
    return templates, content


def _system_prompt() -> str:
    return (
        "You generate accident reconstruction templates for a SUMO/NADE D2RL "
        "pipeline. Return JSON only. The top-level response must be a JSON array "
        "of template objects, with no markdown, no wrapper object, no prose, and "
        "no comments. Each template should describe a plausible accident "
        "hypothesis from the Autoware bag summary. Use these schema fields when "
        "possible: template_id, description, map, route, duration, tags, ego, "
        "actors, events, perturbations. You may infer map and route freely from "
        "the data, but mention assumptions in description or tags. Use CAV as "
        "the ego vehicle when possible. Use numeric position and speed in meters "
        "and m/s when possible. Use exact field names required by the schema. "
        "ego must contain id, role, route, lane_index, position, speed, controller. "
        "Each actor must contain id, role, route, lane_index, position, speed, "
        "controller. Each event must contain type, actor, start_time, duration, "
        "params. Each perturbation must contain field, distribution, low, high. "
        "position must be a single longitudinal float, not a coordinate array. "
        "speed must be a single float. role should be CAV for ego and BV for actors. "
        "For templates intended to run now, name the primary cut-in actor BV_cut_in, "
        "set ego controller to IDM, set the primary actor controller to script, "
        "and use forced_bv_action params with lateral equal to left or right, "
        "longitudinal as a numeric acceleration-like value, and apply_once as a boolean. "
        "The forced_bv_action lateral direction must move the actor toward the ego "
        "lane. In the current 2Lane setup, prefer ego lane_index 1, BV_cut_in "
        "lane_index 0, and lateral left, matching the executable SUMO cut-in path. "
        "If you choose ego lane_index 0 and BV_cut_in lane_index 1, use lateral "
        "right instead of left. For high-risk cut-in templates, place BV_cut_in "
        "ahead of CAV by about 10 to 30 meters and usually make CAV speed greater "
        "than or close to BV_cut_in speed. If decoded_signal_summary is present, "
        "derive speed, acceleration, trajectory, object pose, and event timing from "
        "those decoded records first. Do not invent high-speed values when the "
        "decoded vehicle/control data indicates low-speed motion. Use simulator "
        "projection only to map real Autoware coordinates into the current 2Lane "
        "schema, and explain that projection in description or tags. Prefer "
        "apply_once false only when the event represents an active maneuver window; "
        "otherwise follow the decoded evidence. "
        "Event start_time must be non-negative and event duration must be positive. "
        "Compute scenario duration and event duration from records when possible: "
        "scenario duration should come from autoware_summary.duration_s or from "
        "(max topic last_timestamp_ns - min topic first_timestamp_ns) / 1e9, clipped "
        "to a practical simulation window such as 6 to 20 seconds. Event duration "
        "should represent the fault or maneuver active window. If decoded event "
        "start/end timestamps are available, use (event_end_ns - event_start_ns) / "
        "1e9. If only frame counts and rates are available, use frame_count / "
        "relevant_topic_rate_hz. For cut-in forced_bv_action without decoded "
        "trajectory, first use "
        "template_generation_hints.forced_bv_action_duration_estimate_s if present. "
        "If it is absent, estimate duration from perception/control sampling as "
        "max(0.5, min(2.0, 8 / min(perception_objects.bag_rate_hz, "
        "control_command.bag_rate_hz))) seconds; if those rates are missing, use "
        "0.8 seconds. Add params.duration_basis explaining which records and formula "
        "you used. "
        "Perturbation fields must use paths like ego.speed, actors.BV_cut_in.position, "
        "actors.BV_cut_in.speed, or events.forced_bv_action.start_time. "
        "Do not use alternate names such as initial_speed_mps, initial_position_m, "
        "start_time_s, duration_s, actor_id, or nested actor events. Put all events "
        "in the top-level events array. Current simulator support is limited to 2Lane, "
        "route_0, lane_index 0/1, and executable forced_bv_action events; "
        "templates outside those limits are still useful as reasoning output and "
        "will be recorded as unusable until simulator support is expanded. "
        "Perturbations must be numeric uniform low/high additive deltas, not "
        "absolute final values. The simulator will add the sampled value to the "
        "base template field. Good examples: ego.speed low=-0.5 high=0.5, "
        "actors.BV_cut_in.position low=-3.0 high=3.0, "
        "events.forced_bv_action.start_time low=-0.3 high=0.3. Bad examples: "
        "using ego.speed low=3.5 high=4.26 when base ego.speed is already 4.26, "
        "or actors.BV_cut_in.position low=22 high=28 when base position is "
        "already 25."
    )


def _parse_templates(content: str) -> list[dict]:
    text = content.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"(\[[\s\S]*\])", text)
        if not match:
            raise ValueError("LLM response did not contain a JSON array.")
        value = json.loads(match.group(1))
    if not isinstance(value, list):
        raise ValueError("LLM response must be a JSON array.")
    return value


def _normalize_template(template: dict[str, Any], index: int) -> dict:
    normalized = dict(template)
    normalized.setdefault("template_id", f"autoware_llm_template_{index:04d}")
    normalized.setdefault("description", "LLM generated Autoware accident template.")
    normalized.setdefault("tags", ["autoware", "llm_generated"])
    normalized.setdefault("duration", 12.0)
    normalized.setdefault("perturbations", _default_perturbations())
    return normalized


def _fallback_templates(summary: dict, count: int) -> list[dict]:
    duration = float(
        (summary.get("template_generation_hints") or {}).get("target_duration_s")
        or 12.0
    )
    base_speed = 34.8
    templates = []
    for index in range(count):
        offset = float(index)
        templates.append(
            {
                "template_id": f"autoware_fallback_cut_in_{index:04d}",
                "description": (
                    "Fallback template inferred from Autoware bag topic summary. "
                    "It uses the current executable cut-in reconstruction path."
                ),
                "map": "2Lane",
                "route": "route_0",
                "duration": duration,
                "tags": ["autoware", "fallback", "cut_in"],
                "ego": {
                    "id": "CAV",
                    "role": "CAV",
                    "route": "route_0",
                    "lane_index": 1,
                    "position": 400.0,
                    "speed": base_speed - min(offset, 2.0),
                    "controller": "IDM",
                },
                "actors": [
                    {
                        "id": "BV_cut_in",
                        "role": "BV",
                        "route": "route_0",
                        "lane_index": 0,
                        "position": 418.0 + offset * 2.0,
                        "speed": 24.5 + offset * 0.5,
                        "controller": "script",
                    },
                    {
                        "id": "BV_right_follower",
                        "role": "BV",
                        "route": "route_0",
                        "lane_index": 0,
                        "position": 388.0 - offset * 2.0,
                        "speed": 23.5,
                        "controller": "IDM",
                    },
                ],
                "events": [
                    {
                        "type": "forced_bv_action",
                        "actor": "BV_cut_in",
                        "start_time": min(0.2 * offset, 1.0),
                        "duration": 0.8,
                        "params": {
                            "lateral": "left",
                            "longitudinal": 0.0,
                            "apply_once": False,
                        },
                    },
                    {
                        "type": "perception_dropout",
                        "actor": "BV_cut_in",
                        "start_time": 0.0,
                        "duration": 0.5,
                        "params": {"target": "CAV"},
                    },
                ],
                "perturbations": _default_perturbations(),
            }
        )
    return templates


def _default_perturbations() -> list[dict]:
    return [
        {
            "field": "ego.speed",
            "distribution": "uniform",
            "low": -2.0,
            "high": 2.0,
        },
        {
            "field": "actors.BV_cut_in.position",
            "distribution": "uniform",
            "low": -5.0,
            "high": 5.0,
        },
        {
            "field": "actors.BV_cut_in.speed",
            "distribution": "uniform",
            "low": -2.0,
            "high": 2.0,
        },
        {
            "field": "events.forced_bv_action.start_time",
            "distribution": "uniform",
            "low": 0.0,
            "high": 1.0,
        },
    ]
