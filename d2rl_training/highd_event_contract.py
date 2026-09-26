"""Explicit endpoint semantics; independent of SUMO, Gym and fitted NDD assets."""
import hashlib
import json
import math


FIRST_EVENT = "highd_cav_in_first_collision_v1"
FIRST_SEQUENCE = "highd_first_collision_critical_sequence_v1"
FULL_SEQUENCE = "highd_full_critical_sequence_v1"
FIRST_EVENT_DEFINITION = {
    "name": FIRST_EVENT,
    "horizon_s": 8.0,
    "stop": "first_SUMO_collision_of_any_actors_or_horizon",
    "positive": "CAV_is_in_the_first_reported_collision_step",
    "bv_only": "competing_terminal_event_not_full_horizon_safe",
    "simultaneous_CAV_and_BV_collision": "positive",
    "denominator": "all_attempted_episodes_in_the_declared_initial_mixture",
    "excludes": "CAV_collisions_after_an_earlier_BV_only_collision",
}


def target_digest(spec):
    return hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def first_collision_label(sequence):
    """Derive a new event label; do not replace the full-horizon outcome."""
    horizon, end = sequence["horizon_s"], sequence["end_time_s"]
    if horizon != FIRST_EVENT_DEFINITION["horizon_s"] or not math.isfinite(end) or not 0 < end <= horizon + 1e-9:
        raise ValueError("Wrong first-collision horizon or stopping time")
    ids = set(map(str, sequence["colliding_actor_ids"]))
    if sequence["termination"] == "duration_reached":
        if ids or sequence["collision_result"] is not False or not sequence["outcome_complete"] or abs(end - horizon) > 1e-9:
            raise ValueError("Conflicting horizon-complete outcome")
        return "horizon_no_collision", False
    if sequence["termination"] != "collision" or len(ids) < 2:
        raise ValueError("Execution errors/support exits are not competing collisions")
    cav = str(sequence["cav_id"]) in ids
    if cav:
        if sequence["collision_result"] is not True or not sequence["outcome_complete"]:
            raise ValueError("CAV collision disagrees with recorded outcome")
        return "cav_first_collision", True
    if sequence["collision_result"] is not None or sequence["outcome_complete"]:
        raise ValueError("BV-only collision must retain an unknown full-horizon result")
    return "bv_only_competing_collision", False


def event_indicator(sequence):
    event = sequence.get("event_contract")
    if event is None:
        result = sequence["collision_result"]
        if type(result) is not bool:
            raise ValueError("Unknown full-horizon outcome is not a negative training label")
        return result
    if event != FIRST_EVENT:
        raise ValueError("Unknown event contract")
    kind, result = first_collision_label(sequence)
    if sequence.get("event_type") != kind or sequence.get("event_result") is not result:
        raise ValueError("First-collision label disagrees with terminal record")
    return result


def with_first_event(sequence):
    kind, result = first_collision_label(sequence)
    return dict(sequence, event_contract=FIRST_EVENT, event_type=kind, event_result=result)
