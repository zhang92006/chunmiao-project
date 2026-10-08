"""Fresh-source feature Q_S: separate training, model freeze and validation.

No old fixed-bank loader is called. Pure branching/native estimators are reused
under new source/target locks. Production, Q_A, NDE and PPO remain unchanged.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

import numpy as np

from . import highd_v47_decision_branching as branching
from . import highd_v47_continuation_moments as continuation
from . import highd_v47_objective_bounds as bounds
from . import highd_v47_state_features as features
from .highd_paper_discrete_pairs import PairModelQueries, SINGLE_OPTIONS
from .highd_v46_assets import digest, ensure_assets, read, save
from .highd_v46_cross_unit import proposal_factors
from .highd_v46_d2rl import experiment_target, identity
from .highd_v46_multistate_q import dependencies, environment
from .highd_v46_pair_synergy import pair_marginal, response_risk
from .highd_v46_source_selector_validation import discover, select_per_source
from .highd_v46_state_discovery import audit_gates
from .highd_v47_action_teacher import exploration_h
from .highd_v47_allocation import waterfill
from .highd_v47_allocation_certificate import natural_root, iid_comparison
from .highd_v47_state_moments import action_strata
from .highd_v47_state_moments_v2 import integer_budget

DEFAULT = 'configs/highd_v47_source_transfer_v1.json'
# The central declaration includes consumed development, validation and reserved
# final-test sources. Users cannot remove these exclusions through a config.
FORBIDDEN = ((100000,101023),(110000,110015),(120000,120031),(140000,140007),
    (160000,160015),(170000,170007),(180000,180031),(190000,190031),(200000,200063),
    (210000,210031),(220000,220031),(230000,230063),(240000,240255),(250000,250255))


def protocol(config):
    cfg = read(config)
    required = dict(version=1, bank_method='mixed5_top2', selector='frozen_strict_B_one_per_source',
        epsilon=[.1,.1], risk_edges=[0,.2,.4,.6,.8,1], horizon_s=4,
        roots_per_state=128, coefficient_singletons=12, batches=2, workers=4,
        allocation_loss_cap=1.1, maximum_loss_root_bound=2e-5,
        maximum_natural_roots_per_batch=150000, confidence=.95,
        natural_seed_offset=5000000, iid_draws_per_arm=2112, iid_seed_start=90000000,
        schedule_seed=260019, iid_state_rng_seed=270019)
    if any(cfg.get(k) != v for k,v in required.items()):
        raise ValueError('Explicitly version changed source-transfer design')
    source_sets = {}
    for split, start in [('train',260000),('validation',270000)]:
        cohort = cfg[split]
        if cohort != dict(seed_start=start, episodes=256):
            raise ValueError('Use declared new source-disjoint train/validation cohorts')
        source_sets[split] = set(range(start,start+cohort['episodes']))
        if any(source_sets[split] & set(range(a,b+1)) for a,b in FORBIDDEN):
            raise ValueError('Consumed or final-test sources rejected')
    if source_sets['train'] & source_sets['validation']:
        raise ValueError('Source leakage between fitting and validation')
    if (cfg['measurement_seed_starts'] != {'train':[50000000,60000000], 'validation':[70000000,80000000]}
            or cfg['continuation'] != read('configs/highd_v47_continuation_moments_v1.json')['continuation']
            or cfg['model'] != dict(folds=4, fold_seed=260017, ridge_grid=[1.,10.,100.], prior_roots=16.,
                probability_floor=1e-8, allocation_loss_cap=1.1, minimum_positive_training_sources=5)
            or cfg['statistics'] != dict(bootstrap_repeats=2000, bootstrap_seed=270017, confidence=.95,
                minimum_positive_sources=5, maximum_censored_fraction=.1)
            or cfg['environment'] != environment()):
        raise ValueError('Frozen model/measurement/statistics/environment differs')
    dc = read(cfg['discovery_config']); base = read(dc['base_discovery_config']); physical = read(base['physical_config'])
    if ([dc['bank'][k] for k in ('geometry_states','uniform_states','phase_states')] != [2,2,1]
            or dc['pairs']['ranked'] != 2 or dc['pairs']['audit'] != 1):
        raise ValueError('Retain bounded mixed5/top-two/audit discovery')
    deps = dependencies(cfg,dc,base)
    # Include transitive physics, density, measurement and feature code. Imported
    # old entry points are NOT used to load their fixed-bank data.
    for name in ('highd_v46_multistate_q.py','highd_v46_source_selector_validation.py',
        'highd_v46_native_calibration.py','highd_paper_discrete.py','highd_paper_dynamics.py',
        'highd_v47_state_features.py','highd_v47_action_teacher.py','highd_v47_allocation.py',
        'highd_v47_state_moments.py','highd_v47_state_moments_v2.py','highd_v47_continuation_moments.py',
        'highd_v47_branching_moments.py','highd_v47_decision_branching.py','highd_v47_conditional_native.py',
        'highd_v47_objective_bounds.py','highd_v47_allocation_certificate.py','highd_v47_state_allocation.py',
        'highd_v46_collision_coverage.py'):
        deps[name] = digest(Path(__file__).with_name(name))
    deps['continuation_config'] = digest('configs/highd_v47_continuation_moments_v1.json')
    assets, frozen = ensure_assets()
    lock = dict(config_sha256=digest(config), implementation_sha256=digest(__file__), dependencies=deps,
        environment=environment(), natural_target_sha256=frozen['natural_target_sha256'],
        physical_target_sha256=identity(experiment_target(frozen,physical)),
        forbidden_source_ranges=[list(v) for v in FORBIDDEN])
    return cfg,dc,base,physical,assets,lock


def completed_training(path, cfg, current):
    root = Path(path); summary = read(root/'summary.json')
    if (not summary['complete'] or summary['split'] != 'train' or summary['config'] != cfg
            or any(summary['lock'].get(k) != v for k,v in current.items())):
        raise ValueError('Complete matching new-source training evidence required')
    for name,sha in summary['artifact_hashes'].items():
        if Path(name).name != name or digest(root/name) != sha:
            raise ValueError('Training artifact changed: '+name)
    # Reproduce fitting rather than trusting an editable model artifact.
    if not (root/'model.json').exists():
        raise ValueError('Training completed without a fit; leave validation unopened')
    fitted = read(root/'model.json')
    if fitted != fit_model(read(root/'measurement_plan.json'), read_records(root)):
        raise ValueError('Training feature fit no longer reproduces')
    expected_sources = set(range(cfg['train']['seed_start'],cfg['train']['seed_start']+cfg['train']['episodes']))
    if (set(fitted['model']['fitting_sources']) != {s['plan']['source_seed'] for s in read(root/'measurement_plan.json')['states']}
            or not set(fitted['model']['fitting_sources']) <= expected_sources):
        raise ValueError('Model fitting includes sources outside new training')
    return root,summary,fitted


def check_training_audit(path, root):
    record = read(path)
    if (not record.get('verified') or record.get('split') != 'train'
            or record.get('summary_sha256') != digest(root/'summary.json')
            or record.get('audit_implementation_sha256') != digest(Path(__file__).with_name('highd_v47_source_transfer_audit.py'))
            or record.get('predetermined_native_root_replays') != 2*read(root/'summary.json')['selected_sources']):
        raise ValueError('Complete independent training audit/replays required before validation')


def freeze(config, split, output, training=None, training_audit=None):
    cfg,*_,lock = protocol(config)
    if split == 'validation':
        if training is None: raise ValueError('Freeze validation with complete training, not an old bank model')
        root,summary,fitted = completed_training(training,cfg,lock)
        if (fitted['measured_positive_sources'] < cfg['model']['minimum_positive_training_sources']
                or fitted['selected_OOF_ratio_point'] >= 1):
            raise ValueError('Training source-held-out review failed; leave validation unopened')
        if training_audit is None:
            raise ValueError('Independent training audit required before validation')
        check_training_audit(training_audit,root)
        lock.update(training_path=str(root), training_summary_sha256=digest(root/'summary.json'),
            model_sha256=digest(root/'model.json'), benefit_bet_rate=fitted['benefit_bet']['rate'],
            training_audit_path=str(training_audit), training_audit_sha256=digest(training_audit))
    lock.update(split=split, sources_opened=False,
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    save(output,lock)
    return lock


def prepare_cases(bank, rows, source_seeds, model, physical, base, cfg):
    selections = select_per_source(rows,source_seeds)['B']; cases = []
    for seed in source_seeds:
        index = selections[seed]
        if index is None: continue
        row = rows[index]; state = bank[row['state_index']]
        if state['seed'] != seed: raise ValueError('Selected source/state mismatch')
        cars = [SimpleNamespace(**v) for v in state['vehicles']]
        road = SimpleNamespace(vehicles=cars,cav=next(v for v in cars if v.id==state['cav_id']),model=model,config=physical)
        p,_ = pair_marginal(state['units'],row['actor_ids'])
        risk = response_risk(road,row['actor_ids'],{int(k):np.array(v) for k,v in state['pmfs'].items()},base['proxy'],reactive=True)
        audit = audit_gates(p,risk,base['selection'])
        if any(identity(audit[k]) != identity(row[k]) for k in audit):
            raise ValueError('Original strict selection audit changed')
        h,_ = exploration_h(p,risk,cfg['risk_edges']); *_,q1,q2 = proposal_factors(p,h,cfg['epsilon']); qa = q1[:,None]*q2
        f = features.state_features(state,row['actor_ids'],p,qa,risk)
        support = np.flatnonzero(p.ravel() > 0); c = p.ravel()[support]*(p.ravel()[support]/qa.ravel()[support])
        # Top coefficient cells isolated BEFORE native outcomes; all remaining
        # supported cells retained, including zero-score cells.
        heads = sorted(support[np.argsort(-c,kind='stable')[:cfg['coefficient_singletons']]].tolist())
        hs = action_strata(p,qa,risk,[.2,.4,.6,.8],[-6,-3,-1,0],cfg['roots_per_state'],heads)
        counts = integer_budget([v['coefficient_mass']**2 for v in hs],cfg['roots_per_state'])
        for v,n in zip(hs,counts): v['draws'] = n
        cases.append(dict(state=state, P=p.tolist(), risk=risk.tolist(), H=h.tolist(), QA=qa.tolist(), features=f,
            strata=hs, plan=dict(case_id=len(cases),state_index=row['state_index'],row_index=index,
                source_seed=seed,time_s=state['time_s'],actor_ids=row['actor_ids'],state_sha256=identity(state))))
    for case in cases: case['plan']['target_mass'] = 1/len(cases)
    return cases,selections


def measurement_design(cases, cfg, split, fitted=None):
    alpha = np.full(len(cases),1/len(cases)) if cases else np.array([])
    prediction = features.predict(fitted['model'],[c['features'] for c in cases]) if fitted and cases else None
    q = waterfill(alpha,prediction,1/cfg['allocation_loss_cap']) if prediction is not None else alpha.copy()
    d = alpha*(alpha/q-1) if cases else np.array([])
    states = []
    for i,c in enumerate(cases):
        side = ('benefit' if d[i] < 0 else 'loss' if d[i] > 0 else 'neutral') if split=='validation' else 'training'
        hs = [dict(h) for h in c['strata']]
        if side in ('loss','neutral'):
            for h in hs: h['draws'] = max(2,int(np.ceil(abs(d[i])*h['coefficient_mass']/cfg['maximum_loss_root_bound'])))
        states.append(dict(state_id=i,plan=c['plan'],features=c['features'],strata=hs,side=side,
            signed_objective_coefficient=float(d[i])))
    natural_count = sum(h['draws'] for s in states if s['side'] in ('loss','neutral') for h in s['strata'])
    if natural_count > cfg['maximum_natural_roots_per_batch']:
        raise ValueError('Pre-outcome objective budget exceeds declared cap; do not extend this run')
    schedules = []
    for batch,start in enumerate(cfg['measurement_seed_starts'][split]):
        natural_index = 0
        for s in states:
            rng = np.random.default_rng(np.random.SeedSequence([cfg['schedule_seed'],batch,s['plan']['source_seed']]))
            rows = []
            for h in s['strata']:
                cells = rng.choice(h['cells'],h['draws'],p=h['measurement_probabilities'])
                for j,cell in enumerate(cells):
                    r = dict(stratum_id=h['stratum_id'],within_stratum_draw=j,
                        action_indices=[int(a) for a in np.unravel_index(cell,(33,33))])
                    if s['side'] in ('training','benefit'):
                        seed = start+64*(s['state_id']*cfg['roots_per_state']+len(rows))
                        r.update(initial_seed=seed,step_seeds=list(range(seed+1,seed+21)),
                            terminal_seeds=list(range(seed+21,seed+37)),control_uniforms=rng.random(21).tolist())
                    else:
                        r.update(native_seed=start+cfg['natural_seed_offset']+natural_index); natural_index += 1
                    rows.append(r)
            schedules.append(dict(batch=batch,state_id=s['state_id'],side=s['side'],rows=rows))
    sampling = {}
    if split == 'validation' and cases:
        rng = np.random.default_rng(cfg['iid_state_rng_seed']); uniforms = rng.random(cfg['iid_draws_per_arm'])
        sampling = {name:np.searchsorted(np.cumsum(v),uniforms,side='right').clip(max=len(v)-1).tolist()
                    for name,v in [('target_iid',alpha),('allocated_iid',q)]}
    limits = [s['signed_objective_coefficient']*h['coefficient_mass']/h['draws'] for s in states
              if s['side']=='loss' for h in s['strata']]
    return dict(states=states,schedules=schedules,target_masses=alpha.tolist(),candidate_q=q.tolist(),
        predicted_J=prediction.tolist() if prediction is not None else None,
        iid_schedules=sampling, natural_roots_per_batch=natural_count,
        loss_bet_rate=.8/max(limits) if limits else 1.,
        benefit_bet_rate=fitted['benefit_bet']['rate'] if fitted else None,
        error_per_side_per_batch=(1-cfg['confidence'])/(2*cfg['batches']),
        roots=sum(len(s['rows']) for s in schedules),
        target='Uniform selected fresh source and its one outcome-blind strict-B state; not natural reset density')


def estimate(design, records):
    lookup = {(r['batch'],r['state_id']):r for r in records}; estimates = []
    for s in design['states']:
        batches = []; count = 0
        for b in (0,1):
            rows = lookup[(b,s['state_id'])]['rows']; j = v = 0.
            expected = {(h['stratum_id'],a) for h in s['strata'] for a in range(h['draws'])}
            if len(rows) != len(expected) or {(r['stratum_id'],r['within_stratum_draw']) for r in rows} != expected:
                raise ValueError('Incomplete or duplicate root schedule')
            for h in s['strata']:
                z = np.array([r['measurement_value'] for r in rows if r['stratum_id']==h['stratum_id']])
                bound = 2. if s['side'] in ('training','benefit') else 1.
                if not np.isfinite(z).all() or np.any(z<0) or np.any(z>bound): raise ValueError('Invalid root value')
                c = h['coefficient_mass']; j += c*z.mean(); v += c*c*z.var(ddof=1)/len(z)
            count += sum(r['measurement_value'] > 0 for r in rows)
            batches.append(dict(J=float(j),variance=float(v)))
        total = sum(h['coefficient_mass'] for h in s['strata'])
        n_eff = 2*total**2/sum(h['coefficient_mass']**2/h['draws'] for h in s['strata'])
        estimates.append(dict(state_id=s['state_id'],source_seed=s['plan']['source_seed'],features=s['features'],
            J=float(np.mean([b['J'] for b in batches])),variance=sum(b['variance'] for b in batches)/4,
            batch_estimates=batches,positive_roots=count,effective_measurement_roots=float(n_eff)))
    return estimates


def fit_model(design, records):
    rows = estimate(design,records); cfg = design['model_config']
    fitted = features.grouped_fit(rows,cfg)
    a = np.array(design['target_masses']); js = features.predict(fitted['model'],[r['features'] for r in rows])
    q = waterfill(a,js,1/cfg['allocation_loss_cap']); d = a*(a/q-1)
    by_key = {(r['batch'],r['state_id']):r['rows'] for r in records}; pilot = [[],[]]
    for s in design['states']:
        i = s['state_id']
        if d[i] >= 0: continue
        weights = {h['stratum_id']:-d[i]*h['coefficient_mass']/h['draws'] for h in s['strata']}
        for b in (0,1): pilot[b].extend(weights[r['stratum_id']]*r['measurement_value'] for r in by_key[(b,i)])
    fitted['benefit_bet'] = bounds.choose_pilot_bet(pilot,np.logspace(1,9,81),design['error_per_side_per_batch'])
    fitted['training_measurements'] = rows
    fitted['fitted_candidate_q'] = q.tolist()
    return fitted


def objective_reports(design, records):
    by_key = {(r['batch'],r['state_id']):r['rows'] for r in records}; reports = []
    for b in (0,1):
        benefit = []; loss = []; support = []
        for s in design['states']:
            weights = {h['stratum_id']:abs(s['signed_objective_coefficient'])*h['coefficient_mass']/h['draws'] for h in s['strata']}
            for r in by_key[(b,s['state_id'])]:
                w = weights[r['stratum_id']]; x = w*r['measurement_value']
                if s['side']=='benefit': benefit.append(x)
                elif s['side']=='loss': loss.append(x); support.append(w)
        lo = bounds.benefit_lower(benefit,design['benefit_bet_rate'],design['error_per_side_per_batch'])
        hi = bounds.loss_upper(loss,support,design['loss_bet_rate'],design['error_per_side_per_batch'])
        reports.append(dict(batch=b,**bounds.objective_difference(lo,hi)))
    return reports


def source_comparison(design, measurements, all_sources, cfg):
    if not measurements: return dict(status='no_selected_sources',interval=None,source_generalization_supported=False)
    all_sources = list(all_sources); present = {r['source_seed']:r for r in measurements}
    lookup = {s['plan']['source_seed']:i for i,s in enumerate(design['states'])}
    alpha = np.array(design['target_masses']); q = np.array(design['candidate_q']); j = np.array([r['J'] for r in measurements])
    point_a = float(alpha@j); point_q = float((alpha*alpha/q)@j)
    predictions = np.array(design['predicted_J']); rng = np.random.default_rng(cfg['bootstrap_seed']); ratios = []
    source_count = len([r for r in measurements if r['positive_roots'] > 0]); valid = 0
    for _ in range(cfg['bootstrap_repeats']):
        sampled = rng.choice(all_sources,len(all_sources),replace=True)
        selected = [lookup[int(s)] for s in sampled if int(s) in present]
        positive_unique = {int(s) for s in sampled if int(s) in present and present[int(s)]['positive_roots'] > 0}
        if not selected or len(positive_unique) < cfg['minimum_positive_sources']: continue
        a = np.full(len(selected),1/len(selected)); qs = waterfill(a,predictions[selected],1.1)
        base = float(a@j[selected]); proposed = float((a*a/qs)@j[selected])
        if base > 0: ratios.append(proposed/base); valid += 1
    censored = cfg['bootstrap_repeats']-valid
    permitted = source_count >= cfg['minimum_positive_sources'] and censored/cfg['bootstrap_repeats'] <= cfg['maximum_censored_fraction']
    interval = np.quantile(ratios,[(1-cfg['confidence'])/2,(1+cfg['confidence'])/2]).tolist() if permitted and ratios else None
    contributions = alpha*alpha/q*j; total = float(contributions.sum())
    shares = contributions/total if total > 0 else None
    d = alpha*(alpha/q-1); v = np.array([r['variance'] for r in measurements])
    return dict(target_M2=point_a,candidate_M2=point_q,M2_ratio_point=point_q/point_a if point_a else None,
        difference_empirical_MCSE=float(np.sqrt((d*d)@v)),
        paired_source_bootstrap_ratio_interval=interval,valid_replicates=valid,censored_replicates=censored,
        measured_positive_sources=source_count,all_source_episodes=len(all_sources),selected_sources=len(measurements),
        maximum_source_M2_share=float(shares.max()) if shares is not None else None,
        effective_M2_sources=float(1/(shares@shares)) if shares is not None else None,
        source_generalization_supported=bool(interval is not None and interval[1] < 1),
        uncertainty_scope='Approximate whole-source bootstrap with frozen predictions and re-normalized Q_S; root MCSE separate. Not a distribution-free population guarantee.')


def _collect(task):
    schedule,case,assets,physical,cfg = task; rows = []
    for r in schedule['rows']:
        if schedule['side'] in ('training','benefit'):
            value = branching.root_continuation(continuation._MODEL,Path(assets),physical,case['state'],case['plan']['actor_ids'],r,cfg)
            value['measurement_value'] = value['branching_value']
        else:
            value = natural_root(continuation._MODEL,Path(assets),physical,case['state'],case['plan']['actor_ids'],r)
        rows.append(dict(r,**value))
    return dict(schedule,rows=rows)


def compact_record(record):
    """Keep independent root values in memory; full native trees stay on disk."""
    return {**{k:v for k,v in record.items() if k!='rows'},'rows':[
        {k:v for k,v in r.items() if k in ('stratum_id','within_stratum_draw','measurement_value')}
        for r in record['rows']]}


def read_records(root):
    return [compact_record(read(p)) for p in sorted(Path(root).glob('batch_*_state_*.json'))]


def run(config, split, lock_path, output):
    cfg,dc,base,physical,assets,current = protocol(config); lock = read(lock_path)
    if (lock.get('split') != split or lock.get('sources_opened') is not False
            or any(lock.get(k) != v for k,v in current.items())):
        raise ValueError('Source transfer changed after freeze')
    fitted = None
    if split == 'validation':
        root,_,fitted = completed_training(lock['training_path'],cfg,current)
        if digest(lock['training_audit_path']) != lock['training_audit_sha256']:
            raise ValueError('Training audit changed after validation freeze')
        check_training_audit(lock['training_audit_path'],root)
        if (fitted['measured_positive_sources'] < cfg['model']['minimum_positive_training_sources']
                or fitted['selected_OOF_ratio_point'] >= 1
                or digest(root/'summary.json') != lock['training_summary_sha256'] or digest(root/'model.json') != lock['model_sha256']
                or fitted['benefit_bet']['rate'] != lock['benefit_bet_rate']):
            raise ValueError('Model changed after validation freeze')
    out = Path(output); out.mkdir(parents=True,exist_ok=False); started = time.perf_counter()
    save(out/'opened_protocol.json',dict(config=cfg,lock=lock,rule_lock_sha256=digest(lock_path)))
    model = PairModelQueries(assets/'model',assets/'lateral',**SINGLE_OPTIONS)
    discovery_cfg = dict(cfg,source=dict(cfg[split],split=split))
    bank,rows,discovery = discover(discovery_cfg,dc,base,physical,model,assets)
    source_seeds = list(range(cfg[split]['seed_start'],cfg[split]['seed_start']+cfg[split]['episodes']))
    if [e['seed'] for e in discovery['episodes']] != source_seeds: raise ValueError('Incomplete new source cohort')
    save(out/'state_bank.json',dict(split=split,states=bank)); save(out/'pair_audit.json',dict(rows=rows))
    cases,selections = prepare_cases(bank,rows,source_seeds,model,physical,base,cfg)
    for c in cases: save(out/f'case_{c["plan"]["case_id"]:03d}_design.json',c)
    save(out/'case_plan.json',dict(source_seeds=source_seeds,selections=selections,
        cases=[c['plan'] for c in cases],selection_uses_native_outcomes=False))
    if not cases: raise ValueError('No eligible sources; retain discovery artifacts and do not backfill')
    design = measurement_design(cases,cfg,split,fitted); design['model_config'] = cfg['model']
    design['conditional_target_sha256'] = identity([(c['plan']['state_sha256'],c['plan']['actor_ids'],c['plan']['target_mass']) for c in cases])
    save(out/'measurement_plan.json',design)  # ALL schedules before first new native root.
    records = []
    for b in (0,1):
        with ProcessPoolExecutor(max_workers=cfg['workers'],mp_context=multiprocessing.get_context('spawn'),
                initializer=continuation._init_worker,initargs=(str(assets),)) as pool:
            jobs = {pool.submit(_collect,(s,cases[s['state_id']],str(assets),physical,cfg)) for s in design['schedules'] if s['batch']==b}
            for future in as_completed(jobs):
                r = future.result(); save(out/f'batch_{b}_state_{r["state_id"]:03d}.json',r)
                records.append(compact_record(r)); jobs.remove(future)
                print(dict(phase=split,batch=b,completed=len(records),total=len(cases)*2),flush=True)
    measured = estimate(design,records); comparison = certificates = smoke = None
    fitting_status = None
    if split == 'train':
        if len(cases) < cfg['model']['folds']*2 or not any(r['J'] > 0 for r in measured):
            fitting_status = 'insufficient_training_sources_or_all_zero_observations'
        else:
            fitted = fit_model(design,records); save(out/'model.json',fitted)
            fitting_status = 'fitted_training_only_model'
    else:
        certificates = objective_reports(design,records)
        comparison = source_comparison(design,measured,source_seeds,cfg['statistics'])
        # The smoke arms run irrespective of the measurement outcome.
        continuation._MODEL = model
        iid_cfg = dict(iid_continuation_seed_start=cfg['iid_seed_start'])
        samples,smoke = iid_comparison(iid_cfg,assets,physical,[c['plan'] for c in cases],cases,design)
        for arm,data in samples.items(): save(out/f'{arm}_sequences.json',data)
    result = dict(complete=True,split=split,config=cfg,lock=lock,discovery=discovery,
        natural_sources=len(source_seeds),selected_sources=len(cases),bank_states=len(bank),bounded_pair_audits=len(rows),
        conditional_target_sha256=design['conditional_target_sha256'],independent_measurement_roots=design['roots'],
        measurements=measured,objective_certificates=certificates,source_comparison=comparison,iid_smoke=smoke,
        model_sha256=(digest(out/'model.json') if (out/'model.json').exists() else None) if split=='train' else lock['model_sha256'],
        fitting_status=fitting_status,
        fixed_new_bank_improvement_certified=all(r['improvement_certified'] for r in certificates) if certificates else False,
        transfer_review_passed=bool(certificates and all(r['improvement_certified'] for r in certificates)
            and comparison['source_generalization_supported']),
        total_wall_s=time.perf_counter()-started,production_changed=False,ppo_ready=False,training_exported=False,
        artifact_hashes={p.name:digest(p) for p in sorted(out.glob('*.json'))})
    save(out/'summary.json',result)
    return {k:result[k] for k in ('split','selected_sources','independent_measurement_roots','transfer_review_passed','ppo_ready')}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['freeze-train','train','freeze-validation','validate'])
    p.add_argument('--config',default=DEFAULT); p.add_argument('--lock'); p.add_argument('--training'); p.add_argument('--training-audit'); p.add_argument('--output',required=True)
    args = p.parse_args(); split = 'train' if args.mode in ('freeze-train','train') else 'validation'
    if args.mode.startswith('freeze'):
        r = freeze(args.config,split,args.output,args.training,args.training_audit)
        print(dict(split=split,sources_opened=False,implementation_sha256=r['implementation_sha256']))
    else:
        if not args.lock: p.error('run requires --lock')
        print(run(args.config,split,args.lock,args.output))


if __name__ == '__main__': main()
