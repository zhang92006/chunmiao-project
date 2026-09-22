"""Check candidate execution constraints against REAL train/calibration LC labels."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .highd_lane_change_baseline import _decision_rows, _direction_map
from .highd_ndd_baseline import _load_split_manifest
from .highd_lane_change_guard import gap_envelope, validate_config


def audit_recording(source_root, recording, context, guard):
    rows = _decision_rows(Path(source_root), recording,
        source_hz=context["source_frequency_hz"], target_hz=context["target_frequency_hz"],
        decision_lead_s=context["decision_lead_s"], execution_tail_s=context["execution_tail_s"])
    speed = rows.xVelocity.abs()
    leader = rows.precedingId > 0
    gap = np.where(leader, rows.dhw, context["maximum_gap_m"])
    rr = np.where(leader, rows.precedingXVelocity.abs() - speed, 0)
    eligible = (speed.between(*context["speed_range_mps"]) & (gap >= 0)
        & (gap <= context["maximum_gap_m"]) & (rr >= context["relative_speed_range_mps"][0])
        & (rr <= context["relative_speed_range_mps"][1]))
    if context.get("require_current_leader_for_lateral", False):
        eligible &= leader
    rows = rows.loc[eligible & (rows.action_index != 1)]
    tracks = pd.read_csv(Path(source_root)/"data"/f"{recording}_tracks.csv",
                         usecols=["frame", "id", "x", "width", "xVelocity"])
    lookup = tracks.set_index(["frame", "id"])
    direction = _direction_map(Path(source_root), recording)
    counts = Counter(real_lane_change_labels=len(rows), blocked_real_labels=0,
                     blocked_without_current_leader=0, labels_with_missing_neighbor=0)
    examples = []
    for _, row in rows.iterrows():
        side = "left" if row.action_index == 0 else "right"
        sign = -1 if direction[int(row.id)] == 1 else 1
        center = row.x + row.width/2
        findings = []
        missing = []
        ids = {int(row[side + relation + "Id"]) for relation in ("Preceding", "Following", "Alongside")}
        for neighbor_id in sorted(ids - {0}):
            key = (int(row.frame), neighbor_id)
            if key not in lookup.index:
                missing.append(neighbor_id)
                continue
            neighbor = lookup.loc[key]
            dx = sign*(neighbor.x + neighbor.width/2 - center)
            distance = abs(dx) - (neighbor.width + row.width)/2
            # Match runtime observation range; no future information is used in the check.
            if distance > context["maximum_gap_m"]:
                continue
            gap_rate = abs(neighbor.xVelocity) - abs(row.xVelocity)
            if dx < 0:
                gap_rate = -gap_rate
            envelope = gap_envelope(distance, gap_rate, guard)
            if envelope["reason"]:
                findings.append({"neighbor_id": neighbor_id, **envelope})
        if missing:
            counts["labels_with_missing_neighbor"] += 1
        if findings:
            counts["blocked_real_labels"] += 1
            counts["blocked_without_current_leader"] += int(row.precedingId == 0)
            for reason in {f["reason"] for f in findings}:
                counts[reason] += 1
        if findings or missing:
            examples.append({"vehicle_id": int(row.id), "frame": int(row.frame), "side": side,
                "has_current_leader": bool(row.precedingId > 0), "missing_neighbors": missing,
                "findings": findings})
    return {**dict(counts), "blocked_fraction": counts["blocked_real_labels"]/len(rows) if len(rows) else None,
            "examples": examples}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source_root", "manifest", "context_config", "guard_config", "output"):
        parser.add_argument("--"+name, required=True)
    parser.add_argument("--recordings", nargs="+", required=True)
    args = parser.parse_args()
    splits = _load_split_manifest(Path(args.manifest))
    allowed = set(splits.get("train", [])) | set(splits.get("calibration", []))
    if any(r not in allowed for r in args.recordings):
        parser.error("Guard development audit accepts train/calibration only")
    context = json.loads(Path(args.context_config).read_text(encoding="utf-8"))
    guard = json.loads(Path(args.guard_config).read_text(encoding="utf-8"))
    validate_config(guard)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    results = {}
    for recording in args.recordings:
        results[recording] = audit_recording(args.source_root, recording, context, guard)
        print(json.dumps({"recording": recording, **{k:v for k,v in results[recording].items() if k!='examples'}}), flush=True)
    summary = {"schema_version": 1, "guard_config": guard,
        "guard_config_sha256": hashlib.sha256(Path(args.guard_config).read_bytes()).hexdigest(),
        "context_config_sha256": hashlib.sha256(Path(args.context_config).read_bytes()).hexdigest(),
        "recordings": results,
        "scope": "Label compatibility at the existing 0.5s-before-crossing proxy, not human maneuver initiation. Blocking observed labels is evidence AGAINST directly deploying this hard constraint; zero labels in a bin is not proof of physical impossibility."}
    (output/"guard_label_audit.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
