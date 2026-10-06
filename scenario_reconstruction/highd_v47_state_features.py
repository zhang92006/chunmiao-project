"""Outcome-blind state features and source-grouped J prediction.

This is an opt-in offline bank model, not a source-ID table or natural reset
density. Regression shrinkage is a working model; no fitted error bar is a
native-risk confidence bound. IS observations/weights are never clipped.
"""
import numpy as np

from .highd_v47_allocation import waterfill

FEATURE_NAMES = (
    'log_coefficient_mass', 'coefficient_risk_mean', 'coefficient_risk_squared',
    'coefficient_high_risk_mass', 'natural_risk_mean', 'natural_high_risk_mass',
    'supported_maximum_risk', 'pair_nearness', 'pair_closing_speed',
    'pair_lateral_active', 'pair_brackets_cav',
)


def state_features(state, ids, p, qa, risk):
    p, qa, risk = (np.asarray(v, float) for v in (p, qa, risk))
    if (p.ndim != 2 or p.shape != qa.shape or p.shape != risk.shape
            or not all(np.isfinite(v).all() for v in (p, qa, risk))
            or np.any(p < 0) or np.any(qa < 0) or np.any(qa[p > 0] <= 0)
            or not np.isclose(p.sum(), 1) or not np.isclose(qa.sum(), 1)
            or np.any((risk < 0) | (risk > 1)) or len(set(ids)) != 2):
        raise ValueError('Invalid pre-outcome state/action features')
    c = np.zeros_like(p); c[p > 0] = p[p > 0]*(p[p > 0]/qa[p > 0])
    total = float(c.sum()); g = c/total
    cars = {v['id']: v for v in state['vehicles']}; cav = cars[state['cav_id']]
    pair = [cars[i] for i in ids]; offsets = [v['x']-cav['x'] for v in pair]
    closing = [max(0., (cav['v']-v['v']) if x > 0 else (v['v']-cav['v']))
               for v, x in zip(pair, offsets)]
    x = [np.log1p(total), np.sum(g*risk), np.sum(g*risk*risk), np.sum(g*(risk >= .5)),
         np.sum(p*risk), np.sum(p*(risk >= .5)), risk[p > 0].max(),
         np.mean([1/(1+abs(v)/20) for v in offsets]), np.mean(closing)/10,
         float(any(v['target'] >= 0 for v in pair)), float(offsets[0]*offsets[1] < 0)]
    if not np.isfinite(x).all() or total <= 0:
        raise ValueError('Invalid coefficient mass or physical features')
    # No seed, source ID, actor ID, absolute position, timestamp or outcome.
    return dict(names=list(FEATURE_NAMES), values=[float(v) for v in x], coefficient_mass=total)


def _arrays(rows):
    if not rows or len({r['source_seed'] for r in rows}) != len(rows):
        raise ValueError('One complete state per independent source required')
    x = np.array([r['features']['values'] for r in rows], float)
    c = np.array([r['features']['coefficient_mass'] for r in rows], float)
    j = np.array([r['J'] for r in rows], float)
    n = np.array([r['effective_measurement_roots'] for r in rows], float)
    if (x.shape != (len(rows), len(FEATURE_NAMES)) or not all(np.isfinite(v).all() for v in (x,c,j,n))
            or any(r['features']['names'] != list(FEATURE_NAMES) for r in rows)
            or np.any(c <= 0) or np.any(j < 0) or np.any(j > 2*c+1e-12) or np.any(n <= 0)):
        raise ValueError('Invalid complete source training measurements')
    return x, c, j, n


def fit(rows, ridge, prior_roots=16., probability_floor=1e-8):
    """Equal-source log-fraction ridge with deterministic measurement shrinkage.

    J/C is a c-weighted native event probability. Independent branching roots
    estimate it, possibly above one in a finite sample. Pseudocount shrinkage
    uses the design's effective root count, not tree leaves or observed events.
    """
    x, c, j, n = _arrays(rows)
    if ridge <= 0 or prior_roots <= 0 or not 0 < probability_floor < .5:
        raise ValueError('Invalid frozen model regularization')
    prior = float(np.clip(np.mean(j/c), probability_floor, 1-probability_floor))
    fraction = np.clip((n*j/c+prior_roots*prior)/(n+prior_roots), probability_floor, 1-probability_floor)
    center = x.mean(0); scale = x.std(0); scale[scale < 1e-12] = 1.
    z = np.column_stack([np.ones(len(x)), (x-center)/scale])
    penalty = np.diag([0.]+[ridge]*x.shape[1])
    beta = np.linalg.solve(z.T@z+penalty, z.T@np.log(fraction))
    return dict(contract='v47_source_feature_log_J_v1', feature_names=list(FEATURE_NAMES),
        center=center.tolist(), scale=scale.tolist(), coefficients=beta.tolist(),
        ridge=float(ridge), prior_roots=float(prior_roots), probability_floor=probability_floor,
        fitting_sources=sorted(r['source_seed'] for r in rows), training_global_fraction=prior)


def predict(model, features):
    if (model['contract'] != 'v47_source_feature_log_J_v1' or model['feature_names'] != list(FEATURE_NAMES)
            or any(f['names'] != list(FEATURE_NAMES) for f in features)):
        raise ValueError('Feature contract changed after model freeze')
    if not features: return np.array([], float)
    x = np.array([f['values'] for f in features], float); c = np.array([f['coefficient_mass'] for f in features])
    center = np.array(model['center']); scale = np.array(model['scale']); beta = np.array(model['coefficients'])
    if (x.shape[1:] != (len(FEATURE_NAMES),) or center.shape != scale.shape or center.shape != x.shape[1:]
            or beta.shape != (x.shape[1]+1,) or not all(np.isfinite(v).all() for v in (x,c,center,scale,beta))
            or np.any(scale <= 0) or np.any(c <= 0)):
        raise ValueError('Invalid frozen feature model')
    log_fraction = beta[0]+((x-center)/scale)@beta[1:]
    floor = model['probability_floor']
    return c*np.exp(np.clip(log_fraction, np.log(floor), np.log(1-floor)))


def grouped_fit(rows, cfg):
    """All outcomes from a source stay in one deterministic training-only fold.

    Select ridge by the average common-measurement held-out objective ratio.
    OOF reports are development evidence, not independent validation intervals.
    """
    _arrays(rows)
    if len(rows) < cfg['folds']*2: raise ValueError('Too few independent fitting sources')
    seeds = sorted(r['source_seed'] for r in rows)
    shuffled = np.random.default_rng(cfg['fold_seed']).permutation(seeds)
    assignment = {int(s): i % cfg['folds'] for i,s in enumerate(shuffled)}
    reports = []; predictions = {}
    for ridge in cfg['ridge_grid']:
        values = []; oof = {}
        for fold in range(cfg['folds']):
            training = [r for r in rows if assignment[r['source_seed']] != fold]
            checking = [r for r in rows if assignment[r['source_seed']] == fold]
            model = fit(training, ridge, cfg['prior_roots'], cfg['probability_floor'])
            js = predict(model, [r['features'] for r in checking])
            a = np.full(len(checking), 1/len(checking)); q = waterfill(a, js, 1/cfg['allocation_loss_cap'])
            actual = np.array([r['J'] for r in checking]); base = float(a@actual)
            ratio = float((a*a/q)@actual)/base if base > 0 else None
            values.append(dict(fold=fold, fitting_sources=model['fitting_sources'],
                checking_sources=[r['source_seed'] for r in checking], M2_ratio_point=ratio))
            oof.update({r['source_seed']: float(j) for r,j in zip(checking,js)})
        ratios = [r['M2_ratio_point'] for r in values if r['M2_ratio_point'] is not None]
        reports.append(dict(ridge=ridge, folds=values, mean_ratio_point=float(np.mean(ratios)) if ratios else None))
        predictions[ridge] = oof
    eligible = [r for r in reports if r['mean_ratio_point'] is not None]
    if not eligible: raise ValueError('No training fold measured positive J; do not open validation')
    chosen = min(eligible, key=lambda r:(r['mean_ratio_point'], -r['ridge']))
    model = fit(rows, chosen['ridge'], cfg['prior_roots'], cfg['probability_floor'])
    return dict(model=model, fold_assignments={str(k):v for k,v in assignment.items()}, cross_fitted_reports=reports,
        selected_ridge=chosen['ridge'], selected_OOF_ratio_point=chosen['mean_ratio_point'],
        OOF_predictions={str(k):v for k,v in predictions[chosen['ridge']].items()},
        measured_positive_sources=int(sum(r['J'] > 0 for r in rows)),
        scope='Whole-source training cross-fit; no evaluation labels or source-ID predictors')
