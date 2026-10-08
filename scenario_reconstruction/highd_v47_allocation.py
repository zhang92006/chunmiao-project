"""Collision-probability calibration and exact constrained first-action allocation.

The continuation stays natural, so its Bernoulli second moment is its collision
probability, not that probability squared. This opt-in contract samples Q
factors directly; it is deliberately not a V46 joint-H/checkpoint contract.
"""
import numpy as np

from .highd_paper_discrete_pairs import inverse_cdf
from .highd_v46_cross_unit import natural_log_probability
from .highd_v46_pair_synergy import pair_marginal
from .highd_v46_pair_witness import conditional_actions

CONTRACT = 'v47_first_action_conditional_M2_allocation_v1'


def natural_factors(p):
    p = np.asarray(p, float)
    if (p.ndim != 2 or not p.size or not np.isfinite(p).all()
            or np.any(p < 0) or not np.isclose(p.sum(), 1., atol=1e-12, rtol=1e-12)):
        raise ValueError('Expected normalized finite nonnegative joint P')
    p1 = p.sum(axis=1)
    p2 = np.divide(p, p1[:, None], out=np.zeros_like(p), where=p1[:, None] > 0)
    return p, p1, p2


def waterfill(p, second, epsilon):
    """Minimize sum(p**2 * second / q), q >= epsilon*p, sum(q)=1.

    KKT gives q=max(epsilon*p, c*p*sqrt(second)). An active-set solution
    avoids iterative optimization tolerances on the very small rare-event scale.
    An all-zero objective returns P, a declared neutral choice among minimizers.
    """
    p, second = np.asarray(p, float), np.asarray(second, float)
    if (p.ndim != 1 or second.shape != p.shape or not p.size
            or not np.isfinite(p).all() or not np.isfinite(second).all()
            or np.any(p < 0) or np.any(second < 0)
            or not np.isclose(p.sum(), 1., atol=1e-12, rtol=1e-12)
            or not np.isfinite(epsilon) or not 0 <= epsilon <= 1):
        raise ValueError('Invalid water-filling problem')
    scale = float(np.max(second[p > 0]))
    if scale == 0 or epsilon == 1:
        return p.copy()
    score = np.zeros_like(p)
    supported = p > 0
    score[supported] = p[supported] * np.sqrt(second[supported] / scale)
    lower = epsilon * p
    active = score > 0
    q = lower.copy()
    for _ in range(len(p) + 1):
        mass = 1. - float(lower[~active].sum())
        trial = mass * score[active] / score[active].sum()
        bound = trial < lower[active]
        if not bound.any():
            q[active] = trial
            return q
        active[np.flatnonzero(active)[bound]] = False
    raise ArithmeticError('Water-filling active set did not converge')


def check_q(p, q1, q2, epsilon):
    p, p1, p2 = natural_factors(p)
    q1, q2, eps = np.asarray(q1, float), np.asarray(q2, float), np.asarray(epsilon, float)
    if (q1.shape != p1.shape or q2.shape != p.shape or eps.shape != (2,)
            or not all(np.isfinite(x).all() for x in (q1, q2, eps))
            or np.any(eps < .05) or np.any(eps > 1)
            or np.any(q1 < 0) or np.any(q2 < 0)
            or np.any(q1[p1 == 0] != 0) or np.any(q2[p == 0] != 0)
            or np.any(q1[p1 > 0] <= 0) or np.any(q2[p > 0] <= 0)
            or not np.isclose(q1.sum(), 1., atol=1e-12, rtol=1e-12)
            or not np.allclose(q2.sum(1), (p1 > 0).astype(float), atol=1e-12, rtol=1e-12)
            or np.any(q1 < eps[0]*p1*(1-1e-12))
            or np.any(q2 < eps[1]*p2*(1-1e-12))):
        raise ValueError('Invalid Q support, normalization or factor lower bounds')
    return p1, p2, q1, q2


def second_moment(p, failure, q):
    p, _, _ = natural_factors(p)
    failure, q = np.asarray(failure, float), np.asarray(q, float)
    if (failure.shape != p.shape or q.shape != p.shape
            or not np.isfinite(failure).all() or np.any(failure < 0) or np.any(failure > 1)
            or not np.isfinite(q).all() or np.any(q < 0)
            or not np.isclose(q.sum(), 1., atol=1e-12, rtol=1e-12)
            or np.any(q[p > 0] <= 0) or np.any(q[p == 0] != 0)):
        raise ValueError('Invalid failure probabilities or supported Q')
    supported = p > 0
    return float(np.sum(p[supported] * (p[supported]/q[supported]) * failure[supported]))


def allocate(p, failure, epsilon=(.1, .1), first_only=False):
    """Global optimum under independent conditional-factor lower bounds.

    Solve each conditional row first. Its minimized second moment is then
    the cost for allocating the first factor. No shared-H marginal constraint
    is imposed: a first-factor floor row can still tilt its second factor.
    """
    p, p1, p2 = natural_factors(p)
    failure, eps = np.asarray(failure, float), np.asarray(epsilon, float)
    if (failure.shape != p.shape or not np.isfinite(failure).all()
            or np.any(failure < 0) or np.any(failure > 1) or eps.shape != (2,)
            or not np.isfinite(eps).all() or np.any(eps < .05) or np.any(eps > 1)):
        raise ValueError('Invalid calibrated probabilities or epsilon')
    q2, costs = p2.copy(), np.zeros_like(p1)
    for a in np.flatnonzero(p1 > 0):
        if not first_only:
            q2[a] = waterfill(p2[a], failure[a], eps[1])
        supported = p2[a] > 0
        costs[a] = np.sum(p2[a, supported] * (p2[a, supported]/q2[a, supported]) * failure[a, supported])
    q1 = waterfill(p1, costs, eps[0])
    check_q(p, q1, q2, eps)
    return q1, q2


def factor_tv(p, q1, q2, epsilon=(.1, .1)):
    p1, p2, q1, q2 = check_q(p, q1, q2, epsilon)
    return [float(.5*np.abs(q1-p1).sum()),
            float(np.sum(q1 * .5*np.abs(q2-p2).sum(axis=1)))]


def draw_allocation(units, ids, q1, q2, epsilon, rng):
    """Q(full)=q1*q2*P(rest|controlled), including correlated natural partners."""
    if len(ids) != 2 or len(set(ids)) != 2:
        raise ValueError('Two distinct controlled actors required')
    p, same = pair_marginal(units, ids)
    p1, p2, q1, q2 = check_q(p, q1, q2, epsilon)
    a = inverse_cdf(q1, float(rng.random()))
    b = inverse_cdf(q2[a], float(rng.random()))
    actions = conditional_actions(units, dict(zip(ids, (a, b))), rng)
    pf, qf = [float(p1[a]), float(p2[a, b])], [float(q1[a]), float(q2[a, b])]
    log_weight = float(np.log(np.asarray(pf)/qf).sum())
    full_p = natural_log_probability(units, actions)
    rest = full_p - float(np.log(p[a, b]))
    full_q = float(np.log(qf).sum()) + rest
    if not np.isclose(full_p-full_q, log_weight, atol=1e-10, rtol=1e-10):
        raise ArithmeticError('Full and factor likelihoods differ')
    return actions, {'contract': CONTRACT, 'p_factors': pf, 'q_factors': qf,
        'factor_lower_bounds': list(epsilon), 'log_weight': log_weight,
        'full_log_p': full_p, 'full_log_q': full_q, 'cancelled_log_conditional': rest,
        'same_natural_unit': same, 'action_indices': [int(a), int(b)],
        'affected_natural_units': [list(u['ids']) for u in units if set(ids).intersection(u['ids'])]}


def bin_indices(risk, interaction, config):
    risk, interaction = np.asarray(risk, float), np.asarray(interaction, float)
    edges = np.asarray(config['risk_edges'], float)
    if (risk.shape != interaction.shape or not np.isfinite(risk).all()
            or not np.isfinite(interaction).all() or np.any(risk < 0) or np.any(risk > 1)
            or np.any(interaction < 0) or edges.ndim != 1 or len(edges) < 2
            or edges[0] != 0 or edges[-1] != 1 or np.any(np.diff(edges) <= 0)
            or not np.isfinite(edges).all() or not 0 < config['interaction_threshold'] <= 1):
        raise ValueError('Invalid predeclared calibration bins')
    risk_bin = np.searchsorted(edges[1:-1], risk, side='right')
    return 2*risk_bin + (interaction >= config['interaction_threshold']).astype(int)


def fit_calibrator(rows, config, excluded_source=None):
    """Source-held-out, natural-action weighted bin regression with shrinkage.

    The shrinkage mass is a fixed regularizer, not a Beta posterior sample size.
    Shared seeds across behavior arms are retained but not counted as independent
    traffic sources. Probabilities are bin averages, not certified point risks.
    """
    count = 2*(len(config['risk_edges'])-1)
    floor, regularizer = config['probability_floor'], config['regularization_mass']
    if not 0 < floor < .5 or not np.isfinite(regularizer) or regularizer <= 0:
        raise ValueError('Invalid probability floor or shrinkage')
    selected = [r for r in rows if r['source_seed'] != excluded_source]
    if not selected:
        raise ValueError('No independent source remains for fitting')
    sources = sorted({r['source_seed'] for r in selected})
    design = np.array([r['design_mass'] for r in selected], float)
    p = np.array([r['p'] for r in selected], float)
    behavior = np.array([r['behavior_q'] for r in selected], float)
    labels = np.array([r['event'] for r in selected], float)
    bins = np.array([r['bin'] for r in selected], int)
    if (not all(np.isfinite(x).all() for x in (design, p, behavior, labels))
            or np.any(design <= 0) or np.any(p <= 0) or np.any(p > 1)
            or np.any(behavior <= 0) or np.any(behavior > 1)
            or np.any((labels != 0) & (labels != 1)) or np.any(bins < 0) or np.any(bins >= count)):
        raise ValueError('Invalid calibration observations')
    weights = design/design.sum() * p/behavior
    mass = np.bincount(bins, weights=weights, minlength=count)
    positives = np.bincount(bins, weights=weights*labels, minlength=count)
    prior = float(np.clip(positives.sum()/mass.sum(), floor, 1-floor))
    probabilities = np.clip((positives + regularizer/count*prior)/(mass + regularizer/count), floor, 1-floor)
    return {'source_seeds': sources, 'excluded_source': excluded_source,
        'probabilities': probabilities.tolist(), 'global_probability': prior,
        'weighted_bin_mass': mass.tolist(), 'weighted_bin_events': positives.tolist(),
        'bin_source_counts': [len({r['source_seed'] for r in selected if r['bin'] == b}) for b in range(count)],
        'bin_event_source_counts': [len({r['source_seed'] for r in selected if r['bin'] == b and r['event']}) for b in range(count)],
        'event_sources': sorted({r['source_seed'] for r in selected if r['event']}),
        'configuration': dict(config), 'observations': len(selected),
        'scope': 'Experimental natural-action weighted bin regression; no posterior intervals or deployment approval'}
