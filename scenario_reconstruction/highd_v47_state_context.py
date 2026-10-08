"""Pre-outcome traffic context for a separately versioned state model.

Vehicle x is the FRONT bumper. No source/actor identity, absolute position,
timestamp, RNG state, action label or continuation outcome is a predictor.
"""
import numpy as np

from . import highd_v47_state_features as legacy

RADIUS = 115.
CONTEXT_NAMES = (
    'cav_speed', 'front_present', 'front_nearness', 'front_closing', 'front_urgency',
    'rear_present', 'rear_nearness', 'rear_closing', 'rear_urgency',
    'front_chain_present', 'front_chain_nearness', 'front_chain_closing',
    'rear_chain_present', 'rear_chain_nearness', 'rear_chain_closing',
    'same_lane_local_density', 'adjacent_local_density', 'same_lane_speed_spread',
    'lane_entry_count', 'local_lateral_active', 'cav_lateral_offset',
    'pair_front_present', 'pair_front_nearness', 'pair_front_closing',
    'pair_rear_present', 'pair_rear_nearness', 'pair_rear_closing',
    'coefficient_risk_variance', 'coefficient_reference_mass',
)
FEATURE_NAMES = legacy.FEATURE_NAMES + CONTEXT_NAMES


def _bumper(subject, other, front):
    return (other['x']-other['length']-subject['x'] if front
            else subject['x']-subject['length']-other['x'])


def _neighbor(subject, cars, front, lane):
    candidates = [v for v in cars if v['id'] != subject['id']
        and (v['lane'] == lane or v['target'] == lane)
        and ((v['x'] > subject['x']) if front else (v['x'] <= subject['x']))
        and abs(v['x']-subject['x']) <= RADIUS]
    return min(candidates, key=lambda v: (abs(v['x']-subject['x']), v['v'], v['length'], v['y'])) if candidates else None


def _geometry(subject, other, front):
    if other is None: return [0., 0., 0., 0.]
    gap = max(0., _bumper(subject, other, front))
    closing = max(0., subject['v']-other['v'] if front else other['v']-subject['v'])
    return [1., 1/(1+gap/20), closing/10, closing/(1+gap)]


def state_features(state, ids, p, qa, risk):
    old = legacy.state_features(state, ids, p, qa, risk)
    cars = state['vehicles']; lookup = {v['id']: v for v in cars}; cav = lookup[state['cav_id']]
    if len(lookup) != len(cars): raise ValueError('Duplicate physical actor identity')
    for v in cars:
        if (not all(np.isfinite(v[k]) for k in ('x','v','length','y','lane','target'))
                or v['length'] <= 0 or v['v'] < 0):
            raise ValueError('Invalid pre-outcome traffic context')
    local = [v for v in cars if v['id'] != cav['id'] and abs(v['x']-cav['x']) <= RADIUS]
    same = [v for v in local if v['lane'] == cav['lane'] or v['target'] == cav['lane']]
    front = _neighbor(cav, cars, True, cav['lane']); rear = _neighbor(cav, cars, False, cav['lane'])
    fg = _geometry(cav, front, True); rg = _geometry(cav, rear, False)
    fc = _geometry(front, _neighbor(front,cars,True,front['lane']),True)[:3] if front else [0.]*3
    rc = _geometry(rear, _neighbor(rear,cars,False,rear['lane']),False)[:3] if rear else [0.]*3
    pair = [lookup[i] for i in ids]
    pf = _geometry(cav,_neighbor(cav,pair,True,cav['lane']),True)[:3]
    pr = _geometry(cav,_neighbor(cav,pair,False,cav['lane']),False)[:3]
    p, qa, risk = (np.asarray(v,float) for v in (p,qa,risk)); c = np.zeros_like(p)
    c[p > 0] = p[p > 0]**2/qa[p > 0]; g = c/c.sum()
    ref1 = int(np.argmax(p.sum(1))); ref2 = int(np.argmax(p.sum(0)))
    reference = np.zeros_like(p,dtype=bool); reference[ref1,:] = True; reference[:,ref2] = True
    extra = [cav['v']/40]+fg+rg+fc+rc+[
        len(same)/10, sum(abs(v['lane']-cav['lane']) == 1 for v in local)/10,
        float(np.std([cav['v']]+[v['v'] for v in same]))/10,
        sum(v['target'] == cav['lane'] and v['lane'] != cav['lane'] for v in local)/10,
        sum(v['target'] >= 0 for v in local)/10, abs(cav['y']-4*cav['lane'])/4,
    ]+pf+pr+[float(np.sum(g*(risk-np.sum(g*risk))**2)), float(g[reference].sum())]
    values = old['values']+extra
    if len(values) != len(FEATURE_NAMES) or not np.isfinite(values).all():
        raise ValueError('Traffic feature contract changed')
    return dict(names=list(FEATURE_NAMES), values=[float(v) for v in values],
        coefficient_mass=old['coefficient_mass'])
