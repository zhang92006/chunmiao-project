"""Independent read-only objective, schedule, native and likelihood audit."""
import argparse
import json
from pathlib import Path

import numpy as np

from . import highd_v47_allocation_certificate as native
from .highd_v46_assets import digest, read
from .highd_v46_collision_coverage import audit_sample
from .highd_v46_cross_unit import proposal_factors
from .highd_v46_pair_synergy import pair_marginal
from .highd_v47_decision_branching_audit import reconstruct_tree


def reconstruct_natural(row, state, ids):
    steps=row['steps']; assert 1<=len(steps)<=4
    assert steps[0]['initial'] and all(not s['initial'] for s in steps[1:])
    assert row['future_weight']==1. and 0<row['elapsed_s']<=4+1e-12
    initial=steps[0]['all_action_indices']
    assert [initial[str(i)] if str(i) in initial else initial[i] for i in ids]==row['action_indices']
    initial_full=0.
    if 'units' in state:
        for u in state['units']:
            indices=tuple(initial.get(str(i),initial.get(i)) for i in u['ids'])
            initial_full+=np.log(np.asarray(u['p'])[indices])
        p,_=pair_marginal(state['units'],ids)
        np.testing.assert_allclose([initial_full,initial_full-np.log(p[tuple(row['action_indices'])])],
            [steps[0]['full_log_p'],row['initial_cancelled_log_conditional']],rtol=1e-12,atol=1e-12)
    for j,s in enumerate(steps):
        assert s['time_s']==state['time_s']+j
        assert len({i for u in s['units'] for i in u['ids']})==sum(len(u['ids']) for u in s['units'])
        ps=[]
        for u in s['units']:
            assert 0<u['probability']<=1+1e-12
            assert u['actions']==[s['all_action_indices'].get(str(i),s['all_action_indices'].get(i)) for i in u['ids']]
            ps.append(u['probability'])
        np.testing.assert_allclose(np.log(ps).sum(),s['full_log_p'],rtol=1e-12,atol=1e-12)
    event=any(state['cav_id'] in p for p in row['collision_pairs'])
    assert row['event_result']==event and row['measurement_value']==float(event)
    assert row['competing_result']==(bool(row['collision_pairs']) and not event)
    return float(event)


def independent_objective(design, records):
    result=[]; a=np.array(design['target_masses']);q=np.array(design['candidate_q']);d=a*(a/q-1)
    for b in (0,1):
        positive=[];negative=[];root_limits=[];js=[];vs=[]
        for i,s in enumerate(design['states']):
            j=v=0.
            for h in s['strata']:
                group=[r for r in records[(b,i)]['rows'] if r['stratum_id']==h['stratum_id']]
                assert len(group)==h['draws'] and {r['within_stratum_draw'] for r in group}==set(range(len(group)))
                z=np.array([r['measurement_value'] for r in group]);c=h['coefficient_mass'];n=len(z)
                j+=c*z.sum()/n;v+=c*c*np.sum((z-z.mean())**2)/(n*(n-1))
                if d[i]>0:
                    positive.extend(d[i]*c*z/n);root_limits.extend([d[i]*c/n]*n)
                elif d[i]<0:negative.extend(-d[i]*c*z/n)
            js.append(j);vs.append(v)
        error=design['error_per_side_per_batch'];lb=design['benefit_bet']['rate'];lu=design['loss_bet_rate']
        low=max(0.,(sum(np.log1p(lb*x) for x in negative)-np.log(1/error))/lb)
        assert all(0<=x<=h and lu*h<1 for x,h in zip(positive,root_limits))
        high=min(sum(root_limits),(np.log(1/error)-sum(np.log1p(-lu*x) for x in positive))/lu)
        result.append(dict(batch=b,lower_benefit=float(low),upper_loss=float(high),
            difference_upper=float(high-low),difference_point=float(sum(positive)-sum(negative)),
            target_M2=float(a@js),candidate_M2=float((a*a/q)@js),
            difference_MCSE=float(np.sqrt((d*d)@vs))))
    return result


def audit(config, directory, replay=False):
    cfg,physical,assets,plans,ads,design,expected=native.inputs(config)
    root=Path(directory);summary=read(root/'summary.json')
    assert summary['complete'] and summary['config']==cfg and read(root/'measurement_plan.json')==design
    assert all(summary['lock'].get(k)==v for k,v in expected.items())
    for name,sha in summary['artifact_hashes'].items():
        assert Path(name).name==name and digest(root/name)==sha
    a=np.array(design['target_masses']);q=np.array(design['candidate_q']);d=a*(a/q-1)
    assert np.isclose(q.sum(),1.,rtol=0,atol=1e-12) and np.max(a/q)<=1.1+1e-13
    assert all(s['side']==('benefit' if d[i]<0 else 'loss' if d[i]>0 else 'neutral') for i,s in enumerate(design['states']))
    all_seeds=set();records={};replays=0;max_log_error=0.;roots=0
    if replay:native.parent.parent.parent.parent._init_worker(str(assets))
    for schedule in design['schedules']:
        b,i=schedule['batch'],schedule['state_id'];r=read(root/f'batch_{b}_state_{i:03d}.json')
        assert len(r['rows'])==len(schedule['rows']);records[(b,i)]=r;roots+=len(r['rows'])
        for plan,row in zip(schedule['rows'],r['rows']):
            assert all(row[k]==v for k,v in plan.items())
            if schedule['side']=='benefit':
                seeds=[row['initial_seed']]+row['step_seeds']+row['terminal_seeds']
                value=reconstruct_tree(row['tree'],row,ads[i]['state']['cav_id']);max_log_error=max(max_log_error,value[3])
                np.testing.assert_allclose(value[1],row['measurement_value'],rtol=1e-10,atol=1e-25)
            else:
                seeds=[row['native_seed']];reconstruct_natural(row,ads[i]['state'],plans[i]['actor_ids'])
            assert not set(seeds)&all_seeds;all_seeds.update(seeds)
        if replay:
            index=(17*i+31*b)%len(r['rows']);plan=schedule['rows'][index]
            if schedule['side']=='benefit':
                value=native.parent.root_continuation(native.parent.parent.parent.parent._MODEL,assets,physical,
                    ads[i]['state'],plans[i]['actor_ids'],plan,cfg)
                value['measurement_value']=value['branching_value']
            else:value=native.natural_root(native.parent.parent.parent.parent._MODEL,assets,physical,
                ads[i]['state'],plans[i]['actor_ids'],plan)
            assert json.loads(json.dumps(value))=={k:v for k,v in r['rows'][index].items() if k not in plan}
            replays+=1
    objective=independent_objective(design,records)
    for stored,e in zip(summary['batches'],objective):
        np.testing.assert_allclose([e[k] for k in ('difference_upper','difference_point','target_M2','candidate_M2','difference_MCSE')],
            [stored[k] for k in ('M2_difference_upper','M2_difference_point','target_M2','candidate_M2','difference_empirical_MCSE')],rtol=1e-11,atol=1e-25)
        np.testing.assert_allclose([e['lower_benefit'],e['upper_loss']],[stored['benefit']['lower'],stored['loss']['upper']],rtol=1e-11,atol=1e-25)
    certified=all(e['difference_upper']<0 for e in objective)
    assert summary['allocation_improvement_certified']==certified
    assert summary['recommended_q']==(design['candidate_q'] if certified else design['target_masses'])
    assert summary['family_confidence']==1-4*design['error_per_side_per_batch']
    physical_keys=set();iid_draws=0
    for arm in ('target_iid','allocated_iid'):
        rows=read(root/f'{arm}_sequences.json')['sequences'];qs=a if arm=='target_iid' else q
        assert len(rows)==cfg['iid_draws_per_arm'];z=[]
        for draw,(i,row) in enumerate(zip(design['iid_schedules'][arm],rows)):
            assert row['draw_id']==draw and row['state_id']==i and row['seed']==cfg['iid_continuation_seed_start']+draw
            assert row['seed'] not in all_seeds;physical_keys.add((draw,i));iid_draws+=1
            audit_sample(row,[.1,.1]);log=row['initial_likelihood_audit'];pa,pb=log['action_indices']
            p1,p2,h1,h2,q1,q2=proposal_factors(ads[i]['P'],ads[i]['H'],[.1,.1])
            np.testing.assert_allclose([p1[pa],p2[pa,pb],h1[pa],h2[pa,pb],q1[pa],q2[pa,pb]],
                log['p_factors']+log['h_factors']+log['q_factors'],rtol=1e-12,atol=1e-25)
            state_log=np.log(a[i]/qs[i]);full=state_log+log['full_log_p']-log['full_log_q']
            max_log_error=max(max_log_error,abs(full-row['total_log_weight']))
            np.testing.assert_allclose([a[i],qs[i],state_log,full],
                [row[k] for k in ('target_state_mass','generation_state_mass','state_log_weight','total_log_weight')],rtol=1e-11,atol=1e-12)
            z.append(float(row['event_result'])*np.exp(full))
        report=summary['iid_smoke']['arms'][arm]
        np.testing.assert_allclose([np.mean(z),np.mean(np.square(z)),np.var(z,ddof=1)/len(z)],
            [report[k] for k in ('mean','pooled_second_moment','estimator_variance')],rtol=1e-11,atol=1e-25)
        assert sum(r['event_result'] for r in rows)==report['events']
    assert len(physical_keys)==summary['iid_smoke']['physical_paths']
    return dict(verified=True,summary_sha256=digest(root/'summary.json'),artifact_hashes_verified=len(summary['artifact_hashes']),
        independent_measurement_roots=roots,iid_draws=iid_draws,predetermined_native_root_replays=replays,
        max_full_factor_log_error=max_log_error,objective=objective,allocation_improvement_certified=certified)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default=native.DEFAULT)
    p.add_argument('--run',default='outputs/v47_allocation_certificate_v2');p.add_argument('--replay',action='store_true')
    a=p.parse_args();print(audit(a.config,a.run,a.replay))
