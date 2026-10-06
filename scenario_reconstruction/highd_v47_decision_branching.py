"""Early decision-layer branching with terminal natural-action integration.

The estimand is the frozen teacher action second moment with NATURAL futures.
G mixes whole P/T prefixes through the third second; the last decision is
natural. All roots, not terminal action cells, define measurement uncertainty.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from pathlib import Path
import subprocess
import time

import numpy as np

from . import highd_v47_branching_moments as parent
from .highd_v46_assets import digest, read, save
from .highd_v46_cross_unit import natural_log_probability
from .highd_v46_d2rl import identity
from .highd_v46_pair_synergy import pair_marginal
from .highd_v46_pair_witness import conditional_actions, restore_prepared_state
from .highd_paper_discrete_pairs import inverse_cdf
from .highd_v47_conditional_native import terminal_decisions

DEFAULT = 'configs/highd_v47_decision_branching_v1.json'


def combine_children(children, eta, uniform):
    """Unbiased conditional variance-removal recursion; retain finite negatives."""
    if len(children) != 4 or not 0 < eta < 1 or not 0 <= uniform < 1:
        raise ValueError('Invalid decision tree node')
    z = np.array([c['branching_value'] for c in children])
    d = np.array([c['removed_conditional_variance_unbiased'] for c in children])
    p, t = z[:2], z[2:]; sp, st = p.var(ddof=1), t.var(ddof=1)
    contrast = (p.mean()-t.mean())**2-sp/2-st/2
    vs = eta*(sp+d[:2].mean())+(1-eta)*(st+d[2:].mean())+eta*(1-eta)*contrast
    vb = eta**2*sp/2+(1-eta)**2*st/2
    selected = 0 if uniform < eta else 2
    return dict(branching_value=float(eta*p.mean()+(1-eta)*t.mean()),
                single_value=children[selected]['single_value'], selected_child=selected,
                removed_conditional_variance_unbiased=float(vs-vb))


def integrate_terminal(road, prefix, seed, uniform):
    """Integrate one causal BV action; draw its correlated partners conditionally.

    Reusing the SAME partner uniforms in every forced action cell defines a
    legitimate conditional-MC coupling, not independent event observations.
    Last-step law is P for both prefix mixture components, so its ratio cancels.
    """
    pmfs = road.action_pmfs(); units, actors = road.prepare(pmfs)
    nearest = min((abs(c.x-road.cav.x), c.id) for c in road.vehicles if c.id != road.cav.id)[1]
    unit = next(u for u in units if nearest in u['ids']); p = np.array(unit['p'])
    marginal = p if p.ndim == 1 else p.sum(axis=1 if unit['ids'][0] == nearest else 0)
    supported = np.flatnonzero(marginal > 0)
    assignments = [conditional_actions(units, {nearest: int(a)}, np.random.default_rng(seed)) for a in supported]
    bv_ids = [c.id for c in road.vehicles if c.id != road.cav.id]
    codes = np.array([[a[i] for i in bv_ids] for a in assignments], int)
    outcome = terminal_decisions(road, codes)
    probabilities = marginal[supported]; f = float(probabilities@outcome['event'])
    weight = prefix['future_weight']; selected_action = inverse_cdf(marginal, uniform)
    selected = int(np.flatnonzero(supported == selected_action)[0])
    return dict(actor_id=nearest, action_indices=supported.tolist(), probabilities=probabilities.tolist(),
                bv_ids=bv_ids, all_action_indices=codes.tolist(),
                full_natural_action_logs=[natural_log_probability(units, a) for a in assignments],
                event_results=outcome['event'].tolist(), competing_results=outcome['competing'].tolist(),
                elapsed_s=outcome['elapsed_s'].tolist(), collision_pairs=outcome['collision_pairs'],
                scalar_boundary_replays=int(np.sum(outcome.get('scalar_boundary_replay', []))),
                conditional_event_value=f, selected_action_row=selected,
                branching_value=weight*f, single_value=weight*float(outcome['event'][selected]),
                removed_conditional_variance_unbiased=weight**2*f*(1-f))


def root_continuation(model, assets, physical, state, ids, schedule, cfg):
    road = restore_prepared_state(model, assets, physical, state, schedule['initial_seed'])
    pmfs = {int(k): np.asarray(p) for k, p in state['pmfs'].items()}; initial = None
    def first_sample(*unused):
        nonlocal initial
        a = conditional_actions(state['units'], dict(zip(ids, schedule['action_indices'])), road.rng)
        p, _ = pair_marginal(state['units'], ids)
        initial = natural_log_probability(state['units'], a)-float(np.log(p[tuple(schedule['action_indices'])]))
        return a
    road.action_pmfs = lambda: pmfs; road.sample = first_sample
    road.step_policy('natural', 1.); del road.action_pmfs; del road.sample
    origin = state['time_s']; end = origin+4
    def node(current, steps, index):
        prefix = parent.pack_path(current, steps, initial, origin)
        header = dict(node_index=index, prefix=prefix, state_sha256=parent.dynamic_identity(current))
        if current.crashes:
            return dict(header, status='stopped', children=[], branching_value=prefix['weighted_event'],
                        single_value=prefix['weighted_event'], removed_conditional_variance_unbiased=0.)
        if current.time >= origin+3-1e-9:
            terminal = integrate_terminal(current, prefix, schedule['terminal_seeds'][index-5], schedule['control_uniforms'][index])
            return dict(header, status='terminal_integration', children=[], terminal=terminal,
                        **{k:terminal[k] for k in ('branching_value','single_value','removed_conditional_variance_unbiased')})
        eta = prefix['future_weight']/2; children = []
        for j in range(4):
            child_index = 4*index+j+1
            child = parent.clone_native(current, schedule['step_seeds'][child_index-1])
            if parent.dynamic_identity(child) != header['state_sha256']:
                raise ValueError('Decision branch changed native prefix')
            record = parent.advance_future(child, j >= 2, end, cfg['continuation'])
            children.append(node(child, steps+[record], child_index))
        return dict(header, status='branched', posterior_natural_fraction=eta, children=children,
                    **combine_children(children, eta, schedule['control_uniforms'][index]))
    tree = node(road, [], 0)
    return dict(tree=tree, **{k:tree[k] for k in ('branching_value','single_value','removed_conditional_variance_unbiased')})


def design_for(old, cfg):
    design = {k:v for k,v in old.items() if k != 'max_suffixes'}; schedules = []
    for b, start in enumerate(cfg['seed_starts']):
        original = [s for s in old['schedules'] if s['batch'] == b]
        for s in original:
            i = s['state_id']; rows = []
            rng = np.random.default_rng(np.random.SeedSequence([cfg['schedule_seed'], b, i]))
            action_rows=[]
            for h in old['states'][i]['strata']:
                cells=rng.choice(h['cells'],h['draws'],p=h['measurement_probabilities'])
                for within,cell in enumerate(cells):
                    action_rows.append(dict(stratum_id=h['stratum_id'],within_stratum_draw=within,
                        action_indices=[int(a) for a in np.unravel_index(cell,(33,33))]))
            for k, action in enumerate(action_rows):
                base = start+64*(i*cfg['roots_per_state']+k)
                rows.append(dict(action, initial_seed=base,
                    step_seeds=list(range(base+1, base+21)), terminal_seeds=list(range(base+21, base+37)),
                    control_uniforms=rng.random(21).tolist()))
            schedules.append(dict(batch=b, state_id=i, rows=rows))
    design['schedules'] = schedules
    design['maximum_future_prefix_steps'] = 20*design['independent_roots']
    design['maximum_terminal_native_decisions'] = 16*33*design['independent_roots']
    return design


def inputs(config):
    cfg = read(config)
    required = dict(version=1, roots_per_state=256, batches=2, seed_starts=[8000000,9000000],
                    schedule_seed=800017, workers=4, confidence=.95, allocation_loss_cap=1.1)
    if any(cfg.get(k) != v for k,v in required.items()):
        raise ValueError('Explicitly version changed decision branching design')
    old_cfg, physical, assets, plans, ads, old_design, old_lock = parent.inputs(cfg['parent_config'])
    root = Path(cfg['parent_path']); summary = read(root/'summary.json')
    if (digest(root/'summary.json') != cfg['parent_summary_sha256'] or not summary['complete']
            or summary['config'] != old_cfg or any(summary['lock'].get(k) != v for k,v in old_lock.items())):
        raise ValueError('Completed branching evidence changed')
    for name, sha in summary['artifact_hashes'].items():
        if Path(name).name != name or digest(root/name) != sha: raise ValueError('Branching artifact changed')
    # Initial actions are NEW outcome-blind draws under the original c/C_h law.
    design = design_for(old_design, cfg); cfg = dict(cfg, continuation=old_cfg['continuation'])
    lock = dict(config_sha256=digest(config), implementation_sha256=digest(__file__),
                terminal_implementation_sha256=digest(Path(__file__).with_name('highd_v47_conditional_native.py')),
                parent_lock=old_lock, parent_summary_sha256=cfg['parent_summary_sha256'], design_sha256=identity(design))
    return cfg, physical, assets, plans, ads, design, lock


def _collect(task):
    schedule, state, plan, assets, physical, cfg = task
    rows = [dict(r, **root_continuation(parent.parent.parent._MODEL, Path(assets), physical,
                state, plan['actor_ids'], r, cfg)) for r in schedule['rows']]
    return dict(batch=schedule['batch'], state_id=schedule['state_id'], rows=rows)


def tree_counts(tree):
    if tree['status'] == 'terminal_integration':
        return 0, len(tree['terminal']['probabilities'])
    prefix_steps = terminal_steps = 0
    for child in tree['children']:
        a,b=tree_counts(child); prefix_steps+=1+a; terminal_steps+=b
    return prefix_steps, terminal_steps


def freeze(config, output):
    *_, design, lock = inputs(config)
    lock['code_commit'] = subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    save(output,lock)
    return {k:design[k] for k in ('independent_roots','maximum_future_prefix_steps','maximum_terminal_native_decisions')}


def run(config, lock_path, output):
    cfg, physical, assets, plans, ads, design, expected = inputs(config); lock=read(lock_path)
    if any(lock.get(k)!=v for k,v in expected.items()): raise ValueError('Frozen decision branching changed')
    out=Path(output); out.mkdir(parents=True,exist_ok=False)
    save(out/'opened_protocol.json',dict(config=cfg,lock=lock)); save(out/'measurement_plan.json',design)
    records=[]; started=time.perf_counter()
    for b in (0,1):
        tasks=[(s,ads[s['state_id']]['state'],plans[s['state_id']],str(assets),physical,cfg)
               for s in design['schedules'] if s['batch']==b]
        with ProcessPoolExecutor(max_workers=cfg['workers'],mp_context=multiprocessing.get_context('spawn'),
                initializer=parent.parent.parent._init_worker,initargs=(str(assets),)) as pool:
            for future in as_completed([pool.submit(_collect,t) for t in tasks]):
                r=future.result(); records.append(r); save(out/f'batch_{b}_state_{r["state_id"]:03d}.json',r)
                print(dict(completed_cases=len(records),total=66),flush=True)
        if b==0:
            error=(1-cfg['confidence'])/(4*sum(len(s['strata']) for s in design['states']))
            es={r['state_id']:parent.estimate(design['states'][r['state_id']]['strata'],r['rows'],'branching',error) for r in records}
            allocation=parent.protected_allocation(design['target_masses'],
                [es[i]['bank_simultaneous_interval'][0] for i in range(len(plans))],
                [es[i]['bank_simultaneous_interval'][1] for i in range(len(plans))],cfg['allocation_loss_cap'])
            save(out/'batch0_allocation_before_batch1.json',allocation)
    records.sort(key=lambda r:(r['batch'],r['state_id']))
    counts=[tree_counts(r['tree']) for x in records for r in x['rows']]
    result=dict(complete=True,config=cfg,lock=lock,**parent.summarize(cfg,design,records),
        independent_roots=design['independent_roots'], future_prefix_decisions=sum(c[0] for c in counts),
        terminal_integrated_decisions=sum(c[1] for c in counts),total_wall_s=time.perf_counter()-started,
        production_changed=False,ppo_ready=False,source_generalization_validated=False,
        artifact_hashes={p.name:digest(p) for p in sorted(out.glob('*.json'))})
    if result['uncertainty_aware_allocation']!=read(out/'batch0_allocation_before_batch1.json'):
        raise ValueError('Allocation changed after evaluation outcomes')
    save(out/'summary.json',result)
    return {a:{k:x[k] for k in ('target_M2','empirical_MCSE')} for a,x in result['arms'].items()}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['freeze','run'])
    p.add_argument('--config',default=DEFAULT);p.add_argument('--lock');p.add_argument('--output',required=True)
    a=p.parse_args(); print(freeze(a.config,a.output) if a.mode=='freeze' else run(a.config,a.lock,a.output))
