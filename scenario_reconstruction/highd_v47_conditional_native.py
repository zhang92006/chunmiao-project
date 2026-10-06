"""Read-only batch execution of the LAST frozen native decision.

This is not a risk proxy. It executes the same 15 substeps, IDM CAV, bounded
constant BV accelerations, lateral updates and first-collision stopping rule
as NativeExperiment.step_policy. It cannot be used to continue its endpoints:
natural histories and diagnostics are deliberately not synthesized.
"""
import copy

import numpy as np

from .highd_v46_d2rl import idm_acceleration


def terminal_decisions(road, actions):
    """Evaluate full BV action assignments without mutating the native prefix."""
    cars = road.vehicles
    ci = next(i for i, car in enumerate(cars) if car.id == road.cav.id)
    bi = np.array([i for i in range(len(cars)) if i != ci])
    codes = np.asarray(actions)
    if (road.crashes or codes.ndim != 2 or codes.shape[1] != len(bi)
            or not len(codes) or not np.issubdtype(codes.dtype, np.integer)
            or np.any((codes < 0) | (codes > 32))):
        raise ValueError('Require surviving prefix and complete valid BV action batches')
    n = len(codes)
    def field(name, dtype=float):
        return np.broadcast_to(np.array([getattr(c, name) for c in cars], dtype), (n, len(cars))).copy()
    x, v, y = field('x'), field('v'), field('y')
    lane, target, elapsed = field('lane', int), field('target', int), field('elapsed')
    length, width = field('length'), field('width')
    bv_ids = [cars[i].id for i in bi]
    onset = (codes == 0) | (codes == 32)
    start_lane = lane[:, bi].copy()
    onset_target = start_lane + np.where(codes == 0, 1, -1)
    target[:, bi] = np.where(onset, onset_target, target[:, bi])
    elapsed[:, bi] = np.where(onset, 0., elapsed[:, bi])
    acc = np.asarray(road.model.a)[np.clip(codes-1, 0, 30)]
    acc = np.where(onset, 0., acc)
    lower, upper = road.model.config['speed_grid'][:2]
    duration = road.model.config['lane_change_duration_s']
    lane_width = road.model.config['lane_width_m']
    alive = np.ones(n, bool); event = np.zeros(n, bool); competing = np.zeros(n, bool)
    stopped_at = np.ones(n); pairs = [[] for _ in range(n)]
    boundary = np.zeros(n, bool)
    left, right = np.triu_indices(len(cars), 1)
    dt = 1/15
    for k in range(15):
        active = np.flatnonzero(alive)
        if not len(active): break
        ax, av, ay = x[active].copy(), v[active].copy(), y[active].copy()
        ahead = ((ax[:, bi] > ax[:, ci, None])
                 & (np.abs(ay[:, bi]-ay[:, ci, None]) < (width[active][:, bi]+width[active, ci, None])/2))
        distances = np.where(ahead, ax[:, bi], np.inf)
        front = np.argmin(distances, axis=1)
        has = np.isfinite(distances[np.arange(len(active)), front])
        front_index = bi[front]
        gap = np.where(has, ax[np.arange(len(active)), front_index]-length[active, front_index]-ax[:, ci], np.inf)
        lead_v = np.where(has, av[np.arange(len(active)), front_index], av[:, ci])
        ca = idm_acceleration(av[:, ci], gap, lead_v, road.config['cav'])
        old = av[:, ci]; proposed = old+ca*dt; new = np.clip(proposed, 0, 40)
        t = np.full(len(active), dt); bounded = new != proposed
        np.divide(new-old, ca, out=t, where=bounded)
        ax[:, ci] += old*t+.5*ca*t*t+new*(dt-t); av[:, ci] = new
        old = av[:, bi].copy(); aa = acc[active]; proposed = old+aa*dt
        new = np.clip(proposed, lower, upper); bounded = np.abs(proposed-new) > 1e-10
        t = np.full_like(new, dt); np.divide(new-old, aa, out=t, where=bounded)
        distance = np.where(bounded, old*t+.5*aa*t*t+new*(dt-t), old*dt+.5*aa*dt**2)
        ax[:, bi] += distance; av[:, bi] = new
        atarget, alane, aelapsed = target[active][:, bi], lane[active][:, bi], elapsed[active][:, bi]
        moving = atarget >= 0
        aelapsed = np.where(moving, np.minimum(duration, aelapsed+dt), aelapsed)
        z = aelapsed/duration; fraction = 10*z**3-15*z**4+6*z**5
        ay[:, bi] = np.where(moving, (alane+(atarget-alane)*fraction)*lane_width, ay[:, bi])
        done = moving & (z >= 1-1e-12)
        alane = np.where(done, atarget, alane); atarget = np.where(done, -1, atarget)
        aelapsed = np.where(done, 0., aelapsed)
        # NativeExperiment overrides newly initiated lateral paths with its
        # one-decision linear trajectory after advance() at EACH substep.
        ay[:, bi] = np.where(onset[active],
                            (start_lane[active]+(onset_target[active]-start_lane[active])*(k+1)/15)*4,
                            ay[:, bi])
        x[active], v[active], y[active] = ax, av, ay
        lane[np.ix_(active, bi)], target[np.ix_(active, bi)], elapsed[np.ix_(active, bi)] = alane, atarget, aelapsed
        longitudinal = np.minimum(ax[:, left], ax[:, right])-np.maximum(ax[:, left]-length[active][:, left], ax[:, right]-length[active][:, right])
        lateral = np.abs(ay[:, left]-ay[:, right])-(width[active][:, left]+width[active][:, right])/2+1e-9
        boundary[active] |= ((np.abs(longitudinal-1e-9) < 1e-7) | (np.abs(lateral) < 1e-7)).any(axis=1)
        hit = (longitudinal > 1e-9) & (lateral < 0)
        crashed = hit.any(axis=1)
        for local in np.flatnonzero(crashed):
            row = active[local]
            pairs[row] = [[cars[l].id, cars[r].id] for l, r in zip(left[hit[local]], right[hit[local]])]
            event[row] = any(road.cav.id in pair for pair in pairs[row])
            competing[row] = not event[row]; alive[row] = False; stopped_at[row] = (k+1)/15
    # Never let vectorized floating-point ordering decide a near-boundary
    # endpoint. Re-execute those deterministic assignments with the original
    # scalar native method, without drawing actions or preparing again.
    for row in np.flatnonzero(boundary):
        clone = copy.copy(road)
        excluded = {'model','config','natural','rng','sample','action_pmfs'}
        clone.__dict__ = {k:copy.deepcopy(val) for k,val in road.__dict__.items() if k not in excluded}
        clone.model,clone.config=road.model,road.config
        clone.cav=next(c for c in clone.vehicles if c.id==road.cav.id)
        clone.action_pmfs=lambda:{}
        clone.sample=lambda *unused:dict(zip(bv_ids,codes[row]))
        clone.step_policy('natural',1.)
        pairs[row]=clone.crashes;event[row]=any(clone.cav.id in p for p in clone.crashes)
        competing[row]=bool(clone.crashes) and not event[row];stopped_at[row]=clone.time-road.time
        for name, array in [('x',x),('v',v),('y',y),('lane',lane),('target',target),('elapsed',elapsed)]:
            array[row]=[getattr(c,name) for c in clone.vehicles]
    return dict(event=event, competing=competing, collision_pairs=pairs, elapsed_s=stopped_at,
                scalar_boundary_replay=boundary,
                bv_ids=bv_ids, x=x, v=v, y=y, lane=lane, target=target, elapsed=elapsed)
