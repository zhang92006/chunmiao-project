"""Straight-road bumper relations. Parallel overlap is not a following gap."""
from __future__ import annotations

import numpy as np


SIDE_FEATURES = ("front_present", "front_gap", "front_relative_speed",
                 "alongside_present", "alongside_signed_center_distance", "alongside_relative_speed")
GEOMETRY_FEATURES = tuple(side + "_" + name for side in ("left", "right") for name in SIDE_FEATURES)


def longitudinal_relation(actor, other):
    distance = other["center_x"] - actor["center_x"]
    half_length = (actor["length"] + other["length"]) / 2
    if distance >= half_length:
        return "front", distance - half_length
    if distance <= -half_length:
        return "rear", -distance - half_length
    return "alongside", distance


def side_features(own, front, alongside, delta):
    """Vectorized raw-highD features; same function is used by runtime.

    Width in raw highD is vehicle LENGTH. Zero means missing, not zero gap.
    Front IDs may be nearly alongside; signed gap is preserved (bounded only
    as an explicit feature transform). No safety rejection/mask is applied.
    """
    def valid(other):
        return ((other.direction.to_numpy() == own.direction.to_numpy())
                & (other.laneId.to_numpy() == own.laneId.to_numpy() + delta)
                & np.isfinite(other.x.to_numpy()))
    has_front, has_side = valid(front), valid(alongside)
    center = own.x.to_numpy() + own.width.to_numpy() / 2
    fdx = (front.x.to_numpy() + front.width.to_numpy() / 2 - center) * own.travel_sign.to_numpy()
    sdx = (alongside.x.to_numpy() + alongside.width.to_numpy() / 2 - center) * own.travel_sign.to_numpy()
    gap = fdx - (own.width.to_numpy() + front.width.to_numpy()) / 2
    return np.column_stack([has_front, np.where(has_front, np.clip(gap, -20, 115), 115),
                            np.where(has_front, front.speed_mps - own.speed_mps, 0),
                            has_side, np.where(has_side, np.clip(sdx, -115, 115), 0),
                            np.where(has_side, alongside.speed_mps - own.speed_mps, 0)]).astype(np.float32)
