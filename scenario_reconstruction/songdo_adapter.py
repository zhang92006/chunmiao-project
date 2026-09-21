"""Minimal, dependency-free adapter and quality audit for Songdo Traffic CSVs.

This module intentionally stops at measured trajectory normalization and audit.
It does not create collision labels, infer signal phases, or assign naturalistic
action probabilities for D2RL.
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import hashlib
import json
import math
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple


SCHEMA_VERSION = "songdo_normalized_trajectory_v1"
EXPECTED_FPS = 29.97
EXPECTED_DT_S = 1.0 / EXPECTED_FPS
REQUIRED_COLUMNS = (
    "Vehicle_ID",
    "Local_Time",
    "Drone_ID",
    "Ortho_X",
    "Ortho_Y",
    "Local_X",
    "Local_Y",
    "Latitude",
    "Longitude",
    "Vehicle_Length",
    "Vehicle_Width",
    "Vehicle_Class",
    "Vehicle_Speed",
    "Vehicle_Acceleration",
    "Road_Section",
    "Lane_Number",
    "Visibility",
)


def _optional_float(value: Optional[str]) -> Optional[float]:
    return None if value is None or value.strip() == "" else float(value)


def _optional_int(value: Optional[str]) -> Optional[int]:
    return None if value is None or value.strip() == "" else int(value)


def _seconds_since_midnight(value: str) -> float:
    hour, minute, seconds = value.split(":")
    return int(hour) * 3600.0 + int(minute) * 60.0 + float(seconds)


def _percentile(values: Sequence[float], probability: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _distribution(values: Sequence[float]) -> Dict[str, Optional[float]]:
    return {
        "min": min(values) if values else None,
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "max": max(values) if values else None,
    }


def _fraction(numerator: int, denominator: int) -> Optional[float]:
    return numerator / denominator if denominator else None


@dataclass(frozen=True)
class SongdoRecord:
    schema_version: str
    source_archive: str
    source_member: str
    split_group: str
    vehicle_id: int
    time_local: str
    time_s: float
    drone_id: int
    x_m: float
    y_m: float
    ortho_x_px: float
    ortho_y_px: float
    latitude_deg: float
    longitude_deg: float
    length_m: Optional[float]
    width_m: Optional[float]
    vehicle_class: int
    speed_mps: Optional[float]
    acceleration_mps2: Optional[float]
    road_section: Optional[str]
    lane_number: Optional[int]
    visible: bool


def split_group_for_member(member: str) -> str:
    """Return the indivisible date/intersection/session group for a CSV member."""

    return Path(member).stem


def normalize_row(
    row: Mapping[str, str], source_archive: str, source_member: str
) -> SongdoRecord:
    """Convert one official Songdo row to SI units without interpolation."""

    missing = [column for column in REQUIRED_COLUMNS if column not in row]
    if missing:
        raise ValueError(f"missing Songdo columns: {missing}")
    speed_kph = _optional_float(row["Vehicle_Speed"])
    return SongdoRecord(
        schema_version=SCHEMA_VERSION,
        source_archive=source_archive,
        source_member=source_member,
        split_group=split_group_for_member(source_member),
        vehicle_id=int(row["Vehicle_ID"]),
        time_local=row["Local_Time"],
        time_s=_seconds_since_midnight(row["Local_Time"]),
        drone_id=int(row["Drone_ID"]),
        x_m=float(row["Local_X"]),
        y_m=float(row["Local_Y"]),
        ortho_x_px=float(row["Ortho_X"]),
        ortho_y_px=float(row["Ortho_Y"]),
        latitude_deg=float(row["Latitude"]),
        longitude_deg=float(row["Longitude"]),
        length_m=_optional_float(row["Vehicle_Length"]),
        width_m=_optional_float(row["Vehicle_Width"]),
        vehicle_class=int(row["Vehicle_Class"]),
        speed_mps=None if speed_kph is None else speed_kph / 3.6,
        acceleration_mps2=_optional_float(row["Vehicle_Acceleration"]),
        road_section=row["Road_Section"].strip() or None,
        lane_number=_optional_int(row["Lane_Number"]),
        visible=row["Visibility"].strip() == "1",
    )


def csv_members(archive_path: Path, member_globs: Sequence[str]) -> List[str]:
    with zipfile.ZipFile(archive_path) as archive:
        names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
    if not member_globs:
        return sorted(names)
    return sorted(
        name for name in names if any(fnmatch.fnmatch(name, pat) for pat in member_globs)
    )


def iter_records(archive_path: Path, member: str) -> Iterator[SongdoRecord]:
    """Stream normalized records from one CSV inside an official daily ZIP."""

    with zipfile.ZipFile(archive_path) as archive, archive.open(member) as binary:
        text = (line.decode("utf-8-sig") for line in binary)
        reader = csv.DictReader(text)
        missing = [column for column in REQUIRED_COLUMNS if column not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{member} is missing Songdo columns: {missing}")
        for row in reader:
            yield normalize_row(row, archive_path.name, member)


def load_segmentation_pairs(path: Optional[Path]) -> Optional[set[Tuple[str, int]]]:
    if path is None:
        return None
    pairs: set[Tuple[str, int]] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            pairs.add((row["Section"].strip(), int(row["Lane"])))
    return pairs


def _new_track(record: SongdoRecord) -> dict:
    return {
        "first_s": record.time_s,
        "last_s": record.time_s,
        "rows": 0,
        "visible": 0,
        "lane_labeled": 0,
        "gaps": 0,
        "last_record": None,
        "length_min": None,
        "length_max": None,
        "width_min": None,
        "width_max": None,
    }


def audit_member(
    archive_path: Path,
    member: str,
    segmentation_pairs: Optional[set[Tuple[str, int]]] = None,
    window_seconds: float = 4.0,
) -> dict:
    """Audit one flight-session CSV while retaining only compact track state."""

    tracks: Dict[int, dict] = {}
    timestamps: Counter[str] = Counter()
    same_lane_counts: Counter[Tuple[str, str, int]] = Counter()
    missing = Counter()
    class_counts = Counter()
    speed_values: List[float] = []
    acceleration_values: List[float] = []
    finite_difference_speed_abs_error: List[float] = []
    adjacent_dt: List[float] = []
    length_by_class: Dict[int, List[float]] = defaultdict(list)
    width_by_class: Dict[int, List[float]] = defaultdict(list)
    invalid_lane_assignments = 0
    lane_labeled_count = 0
    same_section_lane_change_candidates = 0
    section_transition_count = 0
    nonpositive_time_steps = 0
    row_count = 0

    for record in iter_records(archive_path, member):
        row_count += 1
        timestamps[record.time_local] += 1
        class_counts[record.vehicle_class] += 1
        for field_name in (
            "length_m",
            "width_m",
            "speed_mps",
            "acceleration_mps2",
            "road_section",
            "lane_number",
        ):
            if getattr(record, field_name) is None:
                missing[field_name] += 1

        if record.speed_mps is not None:
            speed_values.append(record.speed_mps)
        if record.acceleration_mps2 is not None:
            acceleration_values.append(record.acceleration_mps2)
        if record.length_m is not None:
            length_by_class[record.vehicle_class].append(record.length_m)
        if record.width_m is not None:
            width_by_class[record.vehicle_class].append(record.width_m)

        lane_key = None
        if record.road_section is not None and record.lane_number is not None:
            lane_key = (record.road_section, record.lane_number)
            lane_labeled_count += 1
            same_lane_counts[(record.time_local, *lane_key)] += 1
            if segmentation_pairs is not None and lane_key not in segmentation_pairs:
                invalid_lane_assignments += 1

        track = tracks.setdefault(record.vehicle_id, _new_track(record))
        track["rows"] += 1
        track["last_s"] = record.time_s
        track["visible"] += int(record.visible)
        track["lane_labeled"] += int(lane_key is not None)
        for value_name in ("length", "width"):
            value = getattr(record, f"{value_name}_m")
            if value is not None:
                minimum_key = f"{value_name}_min"
                maximum_key = f"{value_name}_max"
                track[minimum_key] = value if track[minimum_key] is None else min(track[minimum_key], value)
                track[maximum_key] = value if track[maximum_key] is None else max(track[maximum_key], value)

        previous: Optional[SongdoRecord] = track["last_record"]
        if previous is not None:
            dt = record.time_s - previous.time_s
            adjacent_dt.append(dt)
            if dt <= 0:
                nonpositive_time_steps += 1
            else:
                if dt > 1.5 * EXPECTED_DT_S:
                    track["gaps"] += 1
                if (
                    dt <= 3.0 * EXPECTED_DT_S
                    and record.speed_mps is not None
                    and record.visible
                    and previous.visible
                ):
                    displacement = math.hypot(record.x_m - previous.x_m, record.y_m - previous.y_m)
                    finite_difference_speed_abs_error.append(
                        abs(displacement / dt - record.speed_mps)
                    )
            if (
                dt > 0
                and dt <= 3.0 * EXPECTED_DT_S
                and previous.road_section is not None
                and record.road_section is not None
            ):
                if previous.road_section != record.road_section:
                    section_transition_count += 1
                elif (
                    previous.lane_number is not None
                    and record.lane_number is not None
                    and previous.lane_number != record.lane_number
                ):
                    same_section_lane_change_candidates += 1
        track["last_record"] = record

    durations = [max(0.0, track["last_s"] - track["first_s"]) for track in tracks.values()]
    row_counts = [track["rows"] for track in tracks.values()]
    length_ranges = [
        track["length_max"] - track["length_min"]
        for track in tracks.values()
        if track["length_min"] is not None and track["length_max"] is not None
    ]
    width_ranges = [
        track["width_max"] - track["width_min"]
        for track in tracks.values()
        if track["width_min"] is not None and track["width_max"] is not None
    ]
    usable_tracks = sum(
        duration >= window_seconds
        and track["gaps"] == 0
        and _fraction(track["visible"], track["rows"]) >= 0.8
        and _fraction(track["lane_labeled"], track["rows"]) >= 0.8
        for duration, track in zip(durations, tracks.values())
    )
    vehicles_per_timestamp = list(timestamps.values())
    same_lane_occupancies = list(same_lane_counts.values())
    speed_domain_counts = {
        "below_5_mps": sum(value < 5.0 for value in speed_values),
        "5_to_below_20_mps": sum(5.0 <= value < 20.0 for value in speed_values),
        "20_to_40_mps_current_legacy_support": sum(
            20.0 <= value <= 40.0 for value in speed_values
        ),
        "above_40_mps": sum(value > 40.0 for value in speed_values),
    }

    return {
        "source_archive": archive_path.name,
        "source_member": member,
        "split_group": split_group_for_member(member),
        "row_count": row_count,
        "vehicle_count": len(tracks),
        "vehicle_class_row_counts": dict(sorted(class_counts.items())),
        "field_coverage": {
            field: {
                "present_rows": row_count - missing[field],
                "present_fraction": _fraction(row_count - missing[field], row_count),
            }
            for field in (
                "length_m",
                "width_m",
                "speed_mps",
                "acceleration_mps2",
                "road_section",
                "lane_number",
            )
        },
        "synchronization": {
            "timestamp_count": len(timestamps),
            "vehicles_per_timestamp": _distribution(vehicles_per_timestamp),
            "timestamps_with_at_least_2_vehicles": sum(value >= 2 for value in vehicles_per_timestamp),
            "timestamps_with_at_least_3_vehicles": sum(value >= 3 for value in vehicles_per_timestamp),
            "same_section_lane_groups_with_at_least_2_vehicles": sum(
                value >= 2 for value in same_lane_occupancies
            ),
        },
        "continuity": {
            "expected_dt_s": EXPECTED_DT_S,
            "adjacent_dt_s": _distribution(adjacent_dt),
            "nonpositive_time_steps": nonpositive_time_steps,
            "gap_count_gt_1_5_frames": sum(track["gaps"] for track in tracks.values()),
            "tracks_with_gap": sum(track["gaps"] > 0 for track in tracks.values()),
            "track_duration_s": _distribution(durations),
            "rows_per_track": _distribution(row_counts),
            "usable_track_count": usable_tracks,
            "usable_track_definition": (
                f"duration >= {window_seconds:g}s, no >1.5-frame gap, "
                "visibility >= 0.8, lane/section coverage >= 0.8"
            ),
        },
        "kinematics": {
            "reported_speed_mps": _distribution(speed_values),
            "speed_domain_row_counts": speed_domain_counts,
            "reported_acceleration_mps2": _distribution(acceleration_values),
            "position_difference_vs_reported_speed_abs_error_mps": _distribution(
                finite_difference_speed_abs_error
            ),
            "comparison_note": (
                "Finite differences use 0.01 m rounded positions while reported speeds are smoothed; "
                "this is an internal consistency diagnostic, not ground-truth error."
            ),
        },
        "dimensions": {
            "length_m_by_class": {
                str(key): _distribution(values) for key, values in sorted(length_by_class.items())
            },
            "width_m_by_class": {
                str(key): _distribution(values) for key, values in sorted(width_by_class.items())
            },
            "within_track_length_range_m": _distribution(length_ranges),
            "within_track_width_range_m": _distribution(width_ranges),
            "ground_truth_note": (
                "No per-vehicle dimension ground truth is present; ranges measure consistency, not accuracy."
            ),
        },
        "lane": {
            "labeled_rows": lane_labeled_count,
            "labeled_fraction": _fraction(lane_labeled_count, row_count),
            "invalid_segmentation_pair_rows": invalid_lane_assignments,
            "same_section_lane_change_candidates": same_section_lane_change_candidates,
            "section_transitions": section_transition_count,
            "candidate_note": (
                "Lane changes are candidates only and require geometry/trajectory validation."
            ),
        },
    }


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_archives(
    archives: Iterable[Path],
    member_globs: Sequence[str],
    segmentation_root: Optional[Path],
    window_seconds: float,
) -> dict:
    results = []
    sources = []
    for archive_path in archives:
        intersection = archive_path.stem.rsplit("_", 1)[-1]
        segmentation_path = (
            segmentation_root / f"{intersection}.csv" if segmentation_root is not None else None
        )
        segmentation_pairs = load_segmentation_pairs(segmentation_path)
        members = csv_members(archive_path, member_globs)
        if not members:
            raise ValueError(f"no CSV members matched in {archive_path}")
        sources.append(
            {
                "path": str(archive_path),
                "bytes": archive_path.stat().st_size,
                "md5": _md5(archive_path),
                "members": members,
                "segmentation": str(segmentation_path) if segmentation_path else None,
            }
        )
        for member in members:
            results.append(
                audit_member(
                    archive_path,
                    member,
                    segmentation_pairs=segmentation_pairs,
                    window_seconds=window_seconds,
                )
            )
    return {
        "schema_version": "songdo_quality_audit_v1",
        "adapter_schema_version": SCHEMA_VERSION,
        "expected_fps": EXPECTED_FPS,
        "sources": sources,
        "sessions": results,
        "totals": {
            "archive_count": len(sources),
            "session_count": len(results),
            "row_count": sum(item["row_count"] for item in results),
            "vehicle_track_count": sum(item["vehicle_count"] for item in results),
            "usable_track_count": sum(
                item["continuity"]["usable_track_count"] for item in results
            ),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", action="append", required=True, type=Path)
    parser.add_argument(
        "--member-glob",
        action="append",
        default=[],
        help="ZIP member glob; repeat for multiple sessions (default: all CSVs)",
    )
    parser.add_argument(
        "--segmentation-root",
        type=Path,
        help="Directory containing per-intersection segmentation CSVs",
    )
    parser.add_argument("--window-seconds", type=float, default=4.0)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    summary = audit_archives(
        archives=args.archive,
        member_globs=args.member_glob,
        segmentation_root=args.segmentation_root,
        window_seconds=args.window_seconds,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary["totals"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
