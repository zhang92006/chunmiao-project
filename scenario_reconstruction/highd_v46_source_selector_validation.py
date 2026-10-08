"""Locked A/B source-selector validation on unchanged V46 native dynamics."""
import argparse
from pathlib import Path
import subprocess
from types import SimpleNamespace
import time

import numpy as np

from .highd_v46_assets import digest, ensure_assets, read, save
from .highd_paper_discrete_pairs import PairModelQueries, SINGLE_OPTIONS
from .highd_v46_cross_unit import positive_interaction, proposal_factors
from .highd_v46_d2rl import NativeExperiment, experiment_target, identity
from .highd_v46_discovery_v2 import choose_states
from .highd_v46_multistate_q import environment, dependencies, sample_continuation, stratum_summary
from .highd_v46_native_calibration import audit_state
from .highd_v46_pair_synergy import pair_marginal, response_risk, snapshot
from .highd_v46_pair_witness import continue_state
from .highd_v46_state_discovery import audit_gates, screen_pairs
from .highd_v46_statistics import episode_values
from .highd_v46_training_expansion import load_calibration
from .highd_v46_source_statistics import cluster_bootstrap, summarize_confirmation, summarize_sources


DEFAULT_CONFIG = 'configs/highd_v46_source_selector_validation_v1.json'
ARMS = {'natural': [1., 1.], 'first_factor_only': [.1, 1.], 'second_factor_only': [1., .1], 'dual': [.1, .1]}


def protocol(config):
    cfg = read(config); dc = read(cfg['discovery_config']); base = read(dc['base_discovery_config'])
    physical = read(base['physical_config'])
    source = cfg['source']; seeds = set(range(source['seed_start'], source['seed_start']+source['episodes']))
    if (source['split'] != 'validation' or len(seeds) < 2
            or any(seeds & set(range(a,b+1)) for a,b in cfg['forbidden_source_ranges'])
            or seeds & set(range(100000,101024))):
        raise ValueError('Source validation needs fresh, nonempty, disjoint episodes')
    if (cfg['selectors'] != {'A': 'absolute_risk_only', 'B': 'absolute_risk_with_production_strict_gate'}
            or cfg['rank_score'] != 'maximum_supported_proxy_risk'
            or cfg['tie_break'] != 'Ascending time then ordered actor IDs'
            or cfg['maximum_states_per_source'] != 1 or cfg['zero_interaction_mass'] != 'Natural fallback with no replacement'
            or cfg['bank_method'] != 'mixed5_top2' or cfg['arms'] != ARMS or cfg['horizon_s'] != 4
            or cfg['replicates_per_case_arm'] < 2):
        raise ValueError('Unsupported frozen selector/proposal contract')
    if ([dc['bank'][k] for k in ('geometry_states','uniform_states','phase_states')] != [2,2,1]
            or dc['pairs']['ranked'] != 2 or dc['pairs']['audit'] != 1 or base['proxy']['horizon_s'] != 4):
        raise ValueError('Keep mixed5 and bounded rank 1 + rank 2 + audit')
    confirmation = cfg['confirmation']; stats = cfg['statistics']; ready = cfg['screening_readiness']
    if (confirmation['maximum_source_episodes'] < 1 or confirmation['replicates_per_corner'] < 2
            or confirmation['minimum_native_interaction'] <= 0
            or not 0 < stats['confidence'] < 1 or stats['bootstrap_repeats'] < 2
            or stats['minimum_event_sources_for_intervals'] < 3 or stats['minimum_events_for_intervals'] < 1
            or not 0 < stats['minimum_effective_factor_tv'] < 1
            or ready['minimum_event_sources'] < 3 or not 0 < ready['maximum_source_second_moment_fraction'] < 1
            or ready['minimum_effective_second_moment_sources'] < 1):
        raise ValueError('Invalid confirmation/statistical guards')
    qseeds = set(range(cfg['continuation_seed_start'],cfg['continuation_seed_start']+source['episodes']*cfg['replicates_per_case_arm']))
    cseeds = set(range(confirmation['continuation_seed_start'],confirmation['continuation_seed_start']+source['episodes']*confirmation['replicates_per_corner']))
    if min(qseeds | cseeds) < 0 or qseeds & cseeds:
        raise ValueError('Continuation and mechanism seed schedules overlap')
    if environment() != cfg['environment']:
        raise ValueError('Use the pinned Python/dependency/thread environment')
    deps = dependencies(cfg, dc, base)
    deps.update({name: digest(Path(__file__).with_name(name)) for name in
                 ('highd_v46_multistate_q.py','highd_v46_native_calibration.py','highd_v46_training_expansion.py',
                  'highd_v46_conditional_baseline.py','highd_v46_source_statistics.py')})
    return cfg, dc, base, physical, deps


def freeze(config, output):
    cfg, dc, base, physical, deps = protocol(config)
    _, frozen = ensure_assets(); seen = set(); training = []
    for entry in cfg['training_locks']:
        training_cfg = read(entry['config'])
        summary, seeds, _, _, _, _ = load_calibration(entry['path'], training_cfg, entry['summary_sha256'])
        if seen & seeds or summary['natural_target_sha256'] != frozen['natural_target_sha256']:
            raise ValueError('Training source overlap or natural target mismatch')
        if summary['experiment_target_sha256'] != identity(experiment_target(frozen, physical)):
            raise ValueError('Training physical target differs')
        seen.update(seeds)
        training.append({'summary_sha256': entry['summary_sha256'], 'source_seeds': sorted(seeds)})
        cases = training_cfg['cases']
        old_count = (cases['maximum_source_episodes']*cases['maximum_states_per_source']+cases['abstention_control_episodes'])*training_cfg['replicates_per_case_arm']
        old_seeds = set(range(training_cfg['continuation_seed_start'],training_cfg['continuation_seed_start']+old_count))
        for start,count in [(cfg['continuation_seed_start'],cfg['source']['episodes']*cfg['replicates_per_case_arm']),
                            (cfg['confirmation']['continuation_seed_start'],cfg['source']['episodes']*cfg['confirmation']['replicates_per_corner'])]:
            if old_seeds & set(range(start,start+count)):
                raise ValueError('Validation continuation schedule overlaps training')
    cohort = cfg['source']
    if seen & set(range(cohort['seed_start'],cohort['seed_start']+cohort['episodes'])):
        raise ValueError('Validation overlaps training source episodes')
    result = {'config_identity': identity(cfg), 'config_sha256': digest(config),
              'implementation_sha256': digest(__file__), 'dependencies': deps,
              'environment': environment(), 'natural_target_sha256': frozen['natural_target_sha256'],
              'experiment_target_sha256': identity(experiment_target(frozen, physical)),
              'training_locks': training, 'validation_sources_opened': False,
              'code_commit': subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).resolve().parents[1],text=True).strip(),
              'scope': 'Immutable source-validation rule lock before opening any fresh source'}
    save(output, result)
    return result


def check_lock(config, lock_path):
    cfg, dc, base, physical, deps = protocol(config); lock = read(lock_path)
    for field,value in [('config_identity',identity(cfg)),('config_sha256',digest(config)),
                        ('implementation_sha256',digest(__file__)),('dependencies',deps),('environment',environment())]:
        if lock.get(field) != value:
            raise ValueError('Rule changed after freeze: '+field)
    if lock.get('validation_sources_opened') is not False:
        raise ValueError('Expected a pre-validation rule lock')
    if [r['summary_sha256'] for r in lock['training_locks']] != [e['summary_sha256'] for e in cfg['training_locks']]:
        raise ValueError('Training locks differ')
    return cfg, dc, base, physical, lock


def select_per_source(rows, source_seeds):
    """Same Rmax ordering; only B applies the existing research gate."""
    result = {method: {} for method in ('A','B')}
    universe = set(source_seeds)
    if len({(r['seed'],r['time_s'],tuple(r['actor_ids'])) for r in rows}) != len(rows):
        raise ValueError('Duplicate candidate state-pair')
    for r in rows:
        risk = r['ranker_features'][2]
        if r['seed'] not in universe or not np.isfinite(risk) or not 0 <= risk <= 1:
            raise ValueError('Candidate source or absolute risk differs')
    for seed in source_seeds:
        ranked = sorted([i for i,r in enumerate(rows) if r['seed'] == seed],
            key=lambda i: (-rows[i]['ranker_features'][2],rows[i]['time_s'],rows[i]['actor_ids']))
        result['A'][seed] = ranked[0] if ranked else None
        result['B'][seed] = next((i for i in ranked if rows[i]['views']['production_strict']),None)
    return result


def discover(cfg, dc, base, physical, model, assets):
    bank, rows, episodes = [], [], []; start = time.perf_counter()
    class Observed(NativeExperiment):
        def choose(self, units, actors, pmfs, mode):
            if mode != 'natural': raise ValueError('Natural source law required')
            screened = screen_pairs(self,pmfs,base['screen'])
            state = snapshot(self,units,pmfs,seed,{'source_selector_validation':True})
            states.append(dict(state,screen_pairs=screened,screen_score=screened[0]['score'] if screened else 0.))
            return None
    cohort = cfg['source']
    for seed in range(cohort['seed_start'],cohort['seed_start']+cohort['episodes']):
        states = []; road = Observed(model,assets/'pair',assets/'lateral',seed,physical)
        while road.time < physical['horizon_s']-1e-9:
            if not road.step_policy('natural',1.): break
        chosen = choose_states(states,'mixed5_top2',dc['bank'],physical['horizon_s'],base['proxy']['horizon_s'],
                               np.random.default_rng(np.random.SeedSequence([dc['bank']['random_seed'],seed])))
        bank.extend(dict(states[i],discovery_role=chosen[i]) for i in sorted(chosen))
        episodes.append({'seed':seed,'natural_states':len(states),'bank_states':len(chosen),
                         'natural_cav_event':any(road.cav.id in p for p in road.crashes),
                         'natural_bv_only_event':bool(road.crashes) and not any(road.cav.id in p for p in road.crashes)})
    for i,state in enumerate(bank):
        rows.extend(audit_state(state,model,physical,base,i,dc))
        if (i+1)%10 == 0 or i+1 == len(bank):
            print({'phase':'source_validation_discovery','states':i+1,'total':len(bank),'pairs':len(rows)},flush=True)
    return bank,rows,{'episodes':episodes,'cached_pair_matrices':len(rows),'wall_s':time.perf_counter()-start}


def proposal(p, risk, arms):
    """Research-gate ablation; existing H/Q algebra and support remain exact."""
    p = np.asarray(p,float); risk = np.asarray(risk,float)
    if risk.shape != p.shape or not np.isfinite(risk).all() or np.any(risk < 0) or np.any(risk > 1):
        raise ValueError('Invalid risk matrix')
    score = p*positive_interaction(p,risk); mass = float(score.sum())
    h = score/mass if mass > 0 else p.copy()
    factors = {}
    for arm,eps in arms.items():
        p1,p2,_,_,q1,q2 = proposal_factors(p,h,eps)
        factors[arm] = [float(.5*np.abs(p1-q1).sum()),float(np.sum(q1*.5*np.abs(p2-q2).sum(1)))]
    return h,mass,factors


def prepare(state,row,model,physical,base,arms):
    cars = [SimpleNamespace(**c) for c in state['vehicles']]
    road = SimpleNamespace(vehicles=cars,cav=next(c for c in cars if c.id==state['cav_id']),model=model,config=physical)
    units = [{'ids':u['ids'],'p':np.asarray(u['p'])} for u in state['units']]
    p,_ = pair_marginal(units,row['actor_ids'])
    risk = response_risk(road,row['actor_ids'],{int(k):np.asarray(v) for k,v in state['pmfs'].items()},base['proxy'],reactive=True)
    audit = audit_gates(p,risk,base['selection'])
    if any(identity(audit[k]) != identity(row[k]) for k in audit):
        raise ValueError('Selected proxy audit no longer reproduces')
    if not np.isclose(risk[p>0].max(),row['ranker_features'][2],atol=1e-12,rtol=0):
        raise ValueError('Cached absolute risk differs')
    h,mass,factors = proposal(p,risk,arms)
    chosen = {'ids':row['actor_ids'],'h':h} if mass > 0 else None
    a0,b0 = row['metrics']['baseline_actions']; a,b = row['metrics']['max_interaction_actions']
    corners = [(a0,b0),(a,b0),(a0,b),(a,b)]
    supported = mass > 0 and all(p[x,y] > 0 for x,y in corners)
    return chosen, {'mass':mass,'factor_tv':factors['dual'],'arm_factor_tv':factors,
                    'natural_fallback':mass == 0,'confirmation_corners':corners if supported else None}


def case_plan(bank,rows,seeds,selections,confirmation):
    plans,lookup = [],{}
    for seed in seeds:
        for method in ('A','B'):
            index = selections[method][seed]
            if index is None: continue
            if index not in lookup:
                r = rows[index]; state = bank[r['state_index']]
                if state['seed'] != seed or state['time_s'] != r['time_s']:
                    raise ValueError('Source/state identity mismatch')
                lookup[index] = len(plans)
                plans.append({'case_id':len(plans),'row_index':index,'state_index':r['state_index'],
                              'source_seed':seed,'time_s':r['time_s'],'actor_ids':r['actor_ids'],
                              'state_sha256':identity(state),'selected_for':[]})
            plans[lookup[index]]['selected_for'].append(method)
    pool = sorted({p['source_seed'] for p in plans})
    rng = np.random.default_rng(confirmation['source_selection_seed'])
    probe_seeds = sorted(int(s) for s in rng.choice(pool,min(len(pool),confirmation['maximum_source_episodes']),replace=False))
    for p in plans: p['confirmation_planned'] = p['source_seed'] in probe_seeds
    return plans,probe_seeds


def mechanism(model,assets,physical,state,plan,cfg,scale):
    corners = plan['confirmation_corners']
    if corners is None:
        return {'status':'no_positive_supported_contrast','source_seed':plan['source_seed']}
    count = cfg['confirmation']['replicates_per_corner']; start = cfg['confirmation']['continuation_seed_start']
    start += (plan['source_seed']-cfg['source']['seed_start'])*count
    risks,events = [],[]
    for seed in range(start,start+count):
        samples = [continue_state(model,assets,physical,state,dict(zip(plan['actor_ids'],corner)),seed,4,scale) for corner in corners]
        risks.append([s['clearance_risk'] for s in samples]); events.append([int(s['cav_first_collision']) for s in samples])
    values = np.asarray(risks); interaction = values[:,3]-values[:,1]-values[:,2]+values[:,0]
    return {'status':'measured','source_seed':plan['source_seed'],'native_interaction_mean':float(interaction.mean()),
            'fixed_state_interaction_mcse':float(interaction.std(ddof=1)/np.sqrt(count)),
            'replicate_clearance_risks':risks,'replicate_cav_events':events,
            'scope':'Forced supported corners; separate from probabilistic Q and ordinary IS estimates'}


def run(config,lock_path,output):
    cfg,dc,base,physical,lock = check_lock(config,lock_path)
    assets,frozen = ensure_assets()
    if frozen['natural_target_sha256'] != lock['natural_target_sha256'] or identity(experiment_target(frozen,physical)) != lock['experiment_target_sha256']:
        raise ValueError('Locked physical/natural target differs')
    output = Path(output); output.mkdir(parents=True,exist_ok=False); start = time.perf_counter()
    save(output/'opened_protocol.json',{'lock_sha256':digest(lock_path),'config':cfg})
    model = PairModelQueries(assets/'model',assets/'lateral',**SINGLE_OPTIONS)
    bank,rows,discovery = discover(cfg,dc,base,physical,model,assets)
    seeds = list(range(cfg['source']['seed_start'],cfg['source']['seed_start']+cfg['source']['episodes']))
    if [e['seed'] for e in discovery['episodes']] != seeds:
        raise ValueError('Incomplete natural source cohort')
    selector_cpu = time.process_time()
    selections = select_per_source(rows,seeds); plans,probe_seeds = case_plan(bank,rows,seeds,selections,cfg['confirmation'])
    chosen_cases = []
    for plan in plans:
        chosen,info = prepare(bank[plan['state_index']],rows[plan['row_index']],model,physical,base,cfg['arms'])
        plan.update(info); chosen_cases.append(chosen)
    selector_cpu = time.process_time()-selector_cpu
    save(output/'state_bank.json',{'split':'validation','states':bank})
    save(output/'pair_audit.json',{'rows':rows})
    # All selections, fallback decisions, masses and probe cases are fixed before native outcomes.
    save(output/'case_plan.json',{'source_seeds':seeds,'selection_uses_native_outcomes':False,
                                'selections':selections,'cases':plans,'confirmation_source_seeds':probe_seeds})
    case_results,confirmations = {},{}
    stats = {**cfg['statistics'],'rhw_target':.3,'minimum_events':cfg['statistics']['minimum_events_for_intervals']}
    for plan,chosen in zip(plans,chosen_cases):
        state = bank[plan['state_index']]; arms = {}; n = cfg['replicates_per_case_arm']
        seed_start = cfg['continuation_seed_start']+(plan['source_seed']-cfg['source']['seed_start'])*n
        for arm,eps in cfg['arms'].items():
            samples = [sample_continuation(model,assets,physical,state,chosen,eps,seed,cfg,base['proxy']['clearance_scale_m']) for seed in range(seed_start,seed_start+n)]
            summary = stratum_summary(samples,stats); values = episode_values(samples)
            arms[arm] = {'events':summary['positive_episodes'],'trials':n,'mean':float(values.mean()),
                         'second_moment':float(np.mean(values**2)),
                         'mean_mc_variance':float(values.var(ddof=1)/n),'likelihood_diagnostics':summary}
            save(output/f'case_{plan["case_id"]:03d}_{arm}.json',{'plan':plan,'epsilon':eps,'sequences':samples})
        case_results[plan['case_id']] = arms
        if plan['confirmation_planned']:
            record = mechanism(model,assets,physical,state,plan,cfg,base['proxy']['clearance_scale_m'])
            confirmations[plan['case_id']] = record
            save(output/f'confirmation_{plan["case_id"]:03d}.json',record)
        print({'phase':'source_native_validation','case':plan['case_id']+1,'total':len(plans),
               'source':plan['source_seed'],'dual_events':arms['dual']['events']},flush=True)
    groups,summaries = {},{}
    for method in ('A','B'):
        selected = {p['source_seed']:p for p in plans if method in p['selected_for']}; records,probes = [],[]
        for seed in seeds:
            p = selected.get(seed)
            records.append({'source_seed':seed,'case_id':p['case_id'] if p else None,
                            'factor_tv':p['factor_tv'] if p else None,'arm_factor_tv':p['arm_factor_tv'] if p else None,
                            'arms':case_results[p['case_id']] if p else None})
            if seed in probe_seeds:
                probes.append(confirmations[p['case_id']] if p else {'source_seed':seed,'status':'selector_abstained'})
        groups[method] = records
        summaries[method] = {'arms':{arm:summarize_sources(records,arm,cfg['screening_readiness'],cfg['statistics']) for arm in cfg['arms']},
                             'confirmation':summarize_confirmation(probes,cfg['confirmation']['minimum_native_interaction'])}
    save(output/'source_records.json',groups)
    artifacts = {p.name:digest(p) for p in output.iterdir() if p.is_file()}
    result = {'complete':True,'split':'validation','config':cfg,'config_sha256':digest(config),
              'implementation_sha256':digest(__file__),'rule_lock_sha256':digest(lock_path),'dependencies':lock['dependencies'],
              'natural_target_sha256':lock['natural_target_sha256'],'experiment_target_sha256':lock['experiment_target_sha256'],
              'discovery':discovery,'case_count':len(plans),'selectors':summaries,
              'selection_and_reaudit_cpu_s':selector_cpu,
              'source_cluster_uncertainty':cluster_bootstrap(groups,summaries,cfg['statistics']),
              'artifacts':artifacts,'ppo_trained':False,'training_manifest_exported':False,'population_acceleration':None,
              'decision':{'winner_selected':None,'ppo_ready':False,'scope':'Source screening evidence requires separate training data and independent model evaluation'},
              'total_wall_s':time.perf_counter()-start}
    save(output/'summary.json',result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['freeze','run'])
    parser.add_argument('--config',default=DEFAULT_CONFIG)
    parser.add_argument('--lock')
    parser.add_argument('--output',required=True)
    args = parser.parse_args()
    if args.command == 'run' and not args.lock: parser.error('run requires --lock')
    result = freeze(args.config,args.output) if args.command == 'freeze' else run(args.config,args.lock,args.output)
    print({'command':args.command,'validation_sources_opened':args.command=='run','ppo_ready':False})


if __name__ == '__main__': main()
