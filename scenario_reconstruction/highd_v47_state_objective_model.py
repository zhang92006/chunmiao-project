"""Nested source-held-out state allocation: log surrogate versus direct M2.

Direct scores are relative allocation priorities, not calibrated J predictions.
The empirical training objective retains all weighted J values without clipping.
No held-out source enters fitting, scaling, priors or hyperparameter selection.
"""
import numpy as np
from scipy.optimize import minimize

from .highd_v47_allocation import waterfill


def arrays(rows):
    if not rows or len({r['source_seed'] for r in rows}) != len(rows):
        raise ValueError('One locked state per independent source required')
    names = rows[0]['features']['names']
    x = np.array([r['features']['values'] for r in rows],float)
    c = np.array([r['features']['coefficient_mass'] for r in rows],float)
    j = np.array([r['J'] for r in rows],float)
    n = np.array([r['effective_measurement_roots'] for r in rows],float)
    if (x.shape != (len(rows),len(names)) or any(r['features']['names'] != names for r in rows)
            or not all(np.isfinite(v).all() for v in (x,c,j,n)) or np.any(c <= 0)
            or np.any(j < 0) or np.any(j > 2*c+1e-12) or np.any(n <= 0)):
        raise ValueError('Invalid locked state measurements')
    return x,c,j,n


def objective_gradient(beta, z, alpha, j, floor, penalty):
    """Analytic derivative inside each water-fill active set.

On recipients q_k=M alpha_k sqrt(score_k)/sum_rec alpha sqrt(score),
so d log(q_k)/d beta = (z_k - sum_rec q*z/M)/2.
    Donors remain at their deterministic floor and have zero derivative.
    Active-set boundaries can be nonsmooth; optimizer failures remain explicit.
"""
    logscore = z@beta; scores = np.exp(np.maximum(logscore-logscore.max(),-700))
    q = waterfill(alpha,scores,floor); base = float(alpha@j)
    if base <= 0: return penalty*float(beta@beta)/2, penalty*beta
    terms = alpha*alpha*j/q
    active = q > floor*alpha*(1+1e-10)
    mean_z = (q[active,None]*z[active]).sum(0)/q[active].sum()
    gradient = -.5*(terms[active,None]*(z[active]-mean_z)).sum(0)/base + penalty*beta
    return float(terms.sum()/base+penalty*(beta@beta)/2), gradient


def fit(rows, method, regularization, cfg):
    x,c,j,n = arrays(rows)
    if method not in ('log_J','direct_M2') or regularization <= 0:
        raise ValueError('Invalid predeclared model family')
    center = x.mean(0); scale = x.std(0); scale[scale < 1e-12] = 1.
    z = (x-center)/scale; prior = None
    status = dict(success=True, message='Closed-form ridge', iterations=0)
    if method == 'log_J':
        f = cfg['probability_floor']; roots = cfg['prior_roots']
        prior = float(np.clip(np.mean(j/c),f,1-f))
        fraction = np.clip((n*j/c+roots*prior)/(n+roots),f,1-f)
        zz = np.column_stack([np.ones(len(rows)),z])
        beta = np.linalg.solve(zz.T@zz+np.diag([0.]+[regularization]*x.shape[1]),zz.T@np.log(fraction))
    else:
        alpha = np.full(len(rows),1/len(rows)); floor = 1/cfg['allocation_loss_cap']
        options = cfg['optimizer']
        result = minimize(objective_gradient,np.zeros(x.shape[1]),
            args=(z,alpha,j,floor,regularization),jac=True,method='L-BFGS-B',
            bounds=[(-options['coefficient_bound'],options['coefficient_bound'])]*x.shape[1],
            options=dict(maxiter=options['maxiter'],ftol=options['ftol'],gtol=options['gtol']))
        status = dict(success=bool(result.success),message=str(result.message),iterations=int(result.nit),
            penalized_objective=float(result.fun),gradient_norm=float(np.linalg.norm(result.jac)))
        # Explicit neutral fallback, never promote a failed optimizer's last iterate.
        beta = result.x if result.success else np.zeros(x.shape[1])
    return dict(contract='v47_nested_allocation_score_v1',method=method,
        feature_names=rows[0]['features']['names'],center=center.tolist(),scale=scale.tolist(),
        coefficients=beta.tolist(),regularization=float(regularization),
        probability_floor=cfg['probability_floor'],training_global_fraction=prior,
        fitting_sources=sorted(r['source_seed'] for r in rows),optimizer=status)


def predict(model, features):
    if not features: return np.array([],float)
    if (model['contract'] != 'v47_nested_allocation_score_v1'
            or any(f['names'] != model['feature_names'] for f in features)):
        raise ValueError('Locked inference feature contract changed')
    x = np.array([f['values'] for f in features],float)
    center,scale,beta = (np.array(model[k],float) for k in ('center','scale','coefficients'))
    if (center.shape != scale.shape or center.shape != x.shape[1:] or np.any(scale <= 0)
            or not all(np.isfinite(v).all() for v in (x,center,scale,beta))):
        raise ValueError('Invalid frozen allocation model')
    z = (x-center)/scale
    if model['method'] == 'log_J':
        if beta.shape != (x.shape[1]+1,): raise ValueError('Invalid log model coefficients')
        f = model['probability_floor']; c = np.array([v['coefficient_mass'] for v in features])
        return c*np.exp(np.clip(beta[0]+z@beta[1:],np.log(f),np.log(1-f)))
    if model['method'] != 'direct_M2' or beta.shape != (x.shape[1],):
        raise ValueError('Invalid direct model coefficients')
    logscore = z@beta
    # A common shift does not affect water-fill. This is numerical score scaling,
    # NOT clipping an observed IS weight, J label, target mass or contribution.
    return np.exp(np.maximum(logscore-logscore.max(),-700))


def folds(rows, count, seed):
    if len(rows) < count*2: raise ValueError('Too few independent sources for nested folds')
    seeds = sorted(r['source_seed'] for r in rows)
    return {int(s):i % count for i,s in enumerate(np.random.default_rng(seed).permutation(seeds))}


def comparison(rows, scores, cap):
    alpha = np.full(len(rows),1/len(rows)); q = waterfill(alpha,scores,1/cap)
    actual = np.array([r['J'] for r in rows]); base = float(alpha@actual)
    candidate = float((alpha*alpha/q)@actual)
    batches = []
    for b in range(2):
        j = np.array([r['batch_estimates'][b]['J'] for r in rows])
        batches.append(dict(baseline_M2=float(alpha@j),candidate_M2=float((alpha*alpha/q)@j)))
    d = alpha*(alpha/q-1); variance = np.array([r['variance'] for r in rows])
    return dict(baseline_M2=base,candidate_M2=candidate,ratio_point=candidate/base if base else None,
        objective_difference=candidate-base,objective_MCSE=float(np.sqrt((d*d)@variance)),
        batches=batches,rows=[dict(source_seed=r['source_seed'],score=float(s),q_over_alpha=float(v),J=r['J'])
            for r,s,v in zip(rows,scores,q/alpha)])


def select(rows, method, grid, cfg, count, seed):
    assignment = folds(rows,count,seed); reports = []
    for reg in grid:
        cases = []
        for f in range(count):
            training = [r for r in rows if assignment[r['source_seed']] != f]
            checking = [r for r in rows if assignment[r['source_seed']] == f]
            model = fit(training,method,reg,cfg)
            report = comparison(checking,predict(model,[r['features'] for r in checking]),cfg['allocation_loss_cap'])
            cases.append(dict(fold=f,size=len(checking),fitting_sources=model['fitting_sources'],
                checking_sources=[r['source_seed'] for r in checking],optimizer=model['optimizer'],comparison=report))
        baseline = sum(c['size']*c['comparison']['baseline_M2'] for c in cases)/len(rows)
        candidate = sum(c['size']*c['comparison']['candidate_M2'] for c in cases)/len(rows)
        reports.append(dict(regularization=reg,aggregate_ratio_point=candidate/baseline if baseline else None,folds=cases))
    usable = [r for r in reports if r['aggregate_ratio_point'] is not None
        and all(c['optimizer']['success'] for c in r['folds'])]
    chosen = min(usable,key=lambda r:(r['aggregate_ratio_point'],-r['regularization'])) if usable else None
    return dict(selected_regularization=chosen['regularization'] if chosen else max(grid),
        status='selected' if chosen else 'no_usable_positive_objective',grid_reports=reports)


def nested_fit(rows, method, grid, cfg):
    arrays(rows); assignment = folds(rows,cfg['outer_folds'],cfg['outer_seed']); reports = []
    for f in range(cfg['outer_folds']):
        training = [r for r in rows if assignment[r['source_seed']] != f]
        checking = [r for r in rows if assignment[r['source_seed']] == f]
        selected = select(training,method,grid,cfg,cfg['inner_folds'],cfg['inner_seed']+f)
        model = fit(training,method,selected['selected_regularization'],cfg)
        report = comparison(checking,predict(model,[r['features'] for r in checking]),cfg['allocation_loss_cap'])
        reports.append(dict(fold=f,size=len(checking),inner_selection=selected,model=model,
            checking_sources=[r['source_seed'] for r in checking],comparison=report))
    baseline = sum(r['size']*r['comparison']['baseline_M2'] for r in reports)/len(rows)
    candidate = sum(r['size']*r['comparison']['candidate_M2'] for r in reports)/len(rows)
    batches = []
    for b in range(2):
        base = sum(r['size']*r['comparison']['batches'][b]['baseline_M2'] for r in reports)/len(rows)
        cand = sum(r['size']*r['comparison']['batches'][b]['candidate_M2'] for r in reports)/len(rows)
        batches.append(dict(baseline_M2=base,candidate_M2=cand,ratio_point=cand/base if base else None))
    final_selection = select(rows,method,grid,cfg,cfg['outer_folds'],cfg['outer_seed'])
    final_model = fit(rows,method,final_selection['selected_regularization'],cfg)
    positive = sum(r['J'] > 0 for r in rows)
    successful = (all(r['model']['optimizer']['success'] and r['inner_selection']['status'] == 'selected' for r in reports)
        and final_model['optimizer']['success'] and final_selection['status'] == 'selected')
    gate = bool(successful and positive >= cfg['minimum_positive_sources']
        and all(b['ratio_point'] is not None and b['ratio_point'] < 1 for b in batches))
    return dict(method=method,outer_assignments={str(k):v for k,v in assignment.items()},outer_reports=reports,
        aggregate_ratio_point=candidate/baseline if baseline else None,batch_objectives=batches,
        measured_positive_sources=int(positive),development_gate_passed=gate,
        final_selection=final_selection,model=final_model,
        inference='Features only; relative score for direct M2, not a calibrated event probability',
        evidence_scope='Nested whole-source development on consumed training labels; NOT fresh validation')
