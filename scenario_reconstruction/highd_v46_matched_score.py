"""Score actual frozen rollouts without replaying or changing their dynamics."""
from pathlib import Path
import numpy as np

from .highd_paper_data import histogram
from .highd_v46_matched_reference import hist2, action_counts


def score_states(trace, actions, hz=10):
    """Exact longitudinal interpolation between observed 15 Hz substeps.

    Linear lateral interpolation follows the actual frozen executor. Lane/target
    labels are left-continuous inside substeps, exact at observed boundaries.
    This is an observation conversion, not a second simulation/collision test.
    """
    end = (len(trace)-1)/15
    times = np.arange(int(np.floor(end*hz+1e-8))+1)/hz
    index = np.floor(times*15+1e-8).astype(int)
    result = trace[index].copy()
    delta = times-index/15
    partial = np.flatnonzero(delta > 1e-10)
    if len(partial):
        idx = index[partial]; dt = delta[partial, None]
        acceleration = actions[np.minimum(idx//15, len(actions)-1)]
        before = trace[idx, :, 2]
        proposed = before+acceleration*dt
        after = np.clip(proposed, 20, 40)
        hit = np.abs(proposed-after) > 1e-10
        bound_time = np.divide(after-before, acceleration, out=np.zeros_like(after), where=acceleration != 0)
        travel_time = np.where(hit, bound_time, dt)
        distance = before*travel_time+.5*acceleration*travel_time**2+after*(dt-travel_time)
        result[partial, :, 1] += distance
        result[partial, :, 2] = after
        fraction = (delta[partial]*15)[:, None]
        result[partial, :, 4] = trace[idx, :, 4]*(1-fraction)+trace[idx+1, :, 4]*fraction
    return times, result


def front_indices(states, physical=False):
    x = states[..., 1]; lane = states[..., 3]; target = states[..., 5]
    dx = x[:, None, :]-x[:, :, None]
    if physical:
        y = states[..., 4]; width = states[..., 7]
        compatible = np.abs(y[:, None, :]-y[:, :, None]) < (width[:, None, :]+width[:, :, None])/2
    else:
        compatible = (lane[:, None, :] == lane[:, :, None]) | (target[:, None, :] == lane[:, :, None])
    possible = compatible & (dx > 0)
    nearest = np.where(possible, dx, np.inf).argmin(axis=-1)
    return np.where(possible.any(axis=-1), nearest, -1)


def observations(states):
    front = front_indices(states); safe = np.maximum(front, 0)
    tick = np.arange(len(states))[:, None]
    leaders = states[tick, safe]
    gap = leaders[..., 1]-leaders[..., 6]-states[..., 1]
    rr = leaders[..., 2]-states[..., 2]
    valid = (front >= 0) & (gap > 0) & (gap <= 115)
    return front, gap, rr, valid


def state_counts(states, bins):
    _, gap, rr, valid = observations(states)
    return {'speed': histogram(states[..., 2].ravel(), *bins['speed']),
            'gap': histogram(gap[valid], *bins['gap']),
            'rr': histogram(rr[valid], *bins['rr']),
            'joint_state': hist2(gap[valid], rr[valid], bins['gap'], bins['rr'])}


def stability(trace):
    front, gap, rr, valid = observations(trace)
    quiet = trace[..., 5] < 0
    changed = np.ones_like(quiet)
    changed[1:] = ((front[1:] != front[:-1]) | (trace[1:, :, 3] != trace[:-1, :, 3]) |
                   ~quiet[1:] | ~quiet[:-1])
    last_change = np.maximum.accumulate(np.where(changed, np.arange(len(trace))[:, None], 0), axis=0)
    age = (np.arange(len(trace))[:, None]-last_change)/15
    return front, gap, rr, valid, age


def interaction_counts(trace, cutoff, bins):
    front, gap, rr, valid, age = stability(trace)
    speed = trace[..., 2]; front_actions = []; rear_actions = []
    for second in range(int(np.floor(cutoff+1e-8))):
        now = second*15; end = now+15
        if end >= len(trace):
            break
        for rear in np.flatnonzero(valid[now] & (np.abs(rr[now]) <= 20) & (age[now] >= 3)):
            leader = front[now, rear]
            if age[now, leader] < 3 or min(age[end, rear], age[end, leader]) < 1:
                continue
            # Match the frozen source selector: quiet both-stay intervals,
            # observed endpoints, both actors within acceleration support.
            a = speed[end, rear]-speed[now, rear]
            b = speed[end, leader]-speed[now, leader]
            if not (-4-1e-9 <= a <= 2+1e-9 and -4-1e-9 <= b <= 2+1e-9):
                continue
            rear_actions.append(-4+np.rint((a+4)/.2)*.2)
            front_actions.append(-4+np.rint((b+4)/.2)*.2)
    return action_counts(np.asarray(front_actions), np.asarray(rear_actions), bins)


def persistent_windows(trace, duration, horizon=60):
    front, _, _, valid, _ = stability(trace)
    result = {}
    width = round(duration*15)
    for left in range(0, min(len(trace)-width, horizon*15), width):
        end = left+width
        pairs = set()
        for rear in np.flatnonzero(valid[left]):
            leader = front[left, rear]
            if (np.all(front[left:end+1, rear] == leader) and np.all(valid[left:end+1, rear]) and
                    np.all(trace[left:end+1, [rear, leader], 5] < 0) and
                    np.all(trace[left:end+1, [rear, leader], 3] == trace[left, [rear, leader], 3])):
                pairs.add((int(rear), int(leader)))
        result[left/15] = pairs
    return result


def persistent_counts(sampled, times, windows, duration, bins):
    gaps = []; rates = []
    for start, pairs in windows.items():
        local = sampled[(times >= start) & (times < start+duration)]
        for rear, leader in pairs:
            gaps.extend(local[:, leader, 1]-local[:, leader, 6]-local[:, rear, 1])
            rates.extend(local[:, leader, 2]-local[:, rear, 2])
    gaps = np.asarray(gaps); rates = np.asarray(rates)
    return {'persistent_gap': histogram(gaps, *bins['gap']),
            'persistent_rr': histogram(rates, *bins['rr']),
            'persistent_joint_state': hist2(gaps, rates, bins['gap'], bins['rr'])}


def safety(trace, threshold):
    lead = front_indices(trace, physical=True); safe = np.maximum(lead, 0)
    leaders = trace[np.arange(len(trace))[:, None], safe]
    gap = leaders[..., 1]-leaders[..., 6]-trace[..., 1]
    closing = trace[..., 2]-leaders[..., 2]
    eligible = (lead >= 0) & (gap > 0) & (closing > 0)
    ttc = np.divide(gap, closing, out=np.full_like(gap, np.inf), where=eligible)
    finite = ttc[np.isfinite(ttc)]
    x = trace[..., 1]; length = trace[..., 6]; y = trace[..., 4]; width = trace[..., 7]
    longitudinal = np.minimum(x[:, None, :], x[:, :, None])-np.maximum(
        (x-length)[:, None, :], (x-length)[:, :, None])
    lateral = (width[:, None, :]+width[:, :, None])/2-np.abs(y[:, None, :]-y[:, :, None])
    unique = np.triu(np.ones(longitudinal.shape[-2:], dtype=bool), 1)
    overlap = (longitudinal > 1e-9) & (lateral > 1e-9) & unique
    return {'minimum_precontact_ttc_s': float(finite.min()) if len(finite) else None,
            'near_collision_episode': bool(np.any(ttc < threshold)),
            'ttc_below_threshold_actor_ticks': int((ttc < threshold).sum()),
            'initial_overlap_pairs': int(overlap[0].sum()),
            'observed_overlap_pair_ticks': int(overlap.sum()),
            'maximum_contact_longitudinal_overlap_m': float(longitudinal[overlap].max()) if overlap.any() else 0.,
            'maximum_contact_lateral_overlap_m': float(lateral[overlap].max()) if overlap.any() else 0.}


def load_pair(root, seed, config):
    from .highd_v46_assets import read, digest
    reports = {}; arrays = {}; sampled = {}; times = {}
    for arm in ('single', 'dual'):
        reports[arm] = read(Path(root)/f'{arm}_seed{seed}.json')
        path = Path(root)/f'{arm}_seed{seed}.npz'
        if digest(path) != reports[arm]['trace_sha256']:
            raise ValueError('Modified rollout trace')
        with np.load(path) as source:
            arrays[arm] = {key: source[key] for key in ('states', 'actions')}
        times[arm], sampled[arm] = score_states(**{'trace': arrays[arm]['states'], 'actions': arrays[arm]['actions']})
    common = min(reports[arm]['elapsed_s'] for arm in reports)
    width = config['persistent_alternative_window_s']; bins = config['bins']
    windows = {arm: persistent_windows(arrays[arm]['states'], width) for arm in reports}
    shared = {t: windows['single'][t] & windows['dual'][t] for t in set(windows['single']) & set(windows['dual'])}
    strict = {arm: persistent_windows(arrays[arm]['states'], config['persistent_duration_s']) for arm in reports}
    strict_shared = strict['single'].get(0., set()) & strict['dual'].get(0., set())
    result = {}
    for arm in reports:
        t = times[arm]; s = sampled[arm]
        # Half-open pooled windows exclude t=60; terminal observations at an
        # exact scoring tick are excluded as well (no collision-state pooling).
        matched = t < common-1e-9
        available = t < min(reports[arm]['elapsed_s'], config['horizon_s'])-1e-9
        primary = state_counts(s[matched], bins)
        primary.update(interaction_counts(arrays[arm]['states'], common, bins))
        primary.update(persistent_counts(s, t, shared, width, bins))
        by_time = {}
        for point in config['diagnostic_times_s']:
            index = int(point*10)
            if reports[arm]['elapsed_s'] >= point-1e-9 and index < len(s):
                by_time[str(point)] = state_counts(s[index:index+1], bins)
        result[arm] = {'primary': primary, 'available': state_counts(s[available], bins), 'by_time': by_time,
                       'safety': safety(arrays[arm]['states'], config['near_collision_ttc_s']),
                       'persistent_60s_pairs': len(strict[arm].get(0., set())),
                       'persistent_5s_windows': sum(map(len, windows[arm].values())),
                       'report': reports[arm]}
    result['matched'] = {'common_duration_s': common, 'common_persistent_60s_pairs': len(strict_shared),
                         'common_persistent_5s_windows': sum(map(len, shared.values()))}
    return result
