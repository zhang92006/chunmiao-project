"""Read-only density, tree, terminal-integration and allocation reconstruction."""
import argparse
import json
from pathlib import Path

import numpy as np

from .highd_v46_assets import digest,read
from .highd_v47_branching_moments_audit import reconstruct_estimate
from .highd_v47_continuation_moments_audit import reconstruct_path
from .highd_v47_decision_branching import inputs


def check_root_values(values, stored, scale):
    # Values/likelihoods keep the original tight check. Variance removal is
    # a subtraction of two nearly equal quadratic terms in the frozen writer;
    # relative error against a near-zero difference is ill-conditioned. Allow
    # only floating-point roundoff at the original quadratic signal scale.
    np.testing.assert_allclose(values[:2],stored[:2],rtol=1e-10,atol=1e-25)
    np.testing.assert_allclose(values[2],stored[2],rtol=1e-10,
        atol=128*np.finfo(float).eps*scale)


def reconstruct_tree(node,schedule,cav_id):
    error=reconstruct_path(node['prefix']);prefix=node['prefix'];index=node['node_index']
    if node['status']=='stopped':
        assert prefix['collision_pairs'] and not node['children']
        values=(prefix['weighted_event'],prefix['weighted_event'],0.)
    elif node['status']=='terminal_integration':
        assert not node['children'] and prefix['elapsed_s']==3.
        t=node['terminal'];p=np.array(t['probabilities']);y=np.array(t['event_results'],bool)
        assert np.all(p>0) and np.isclose(p.sum(),1.,rtol=0,atol=1e-12)
        assert len(p)==len(set(t['action_indices']))<=33
        assignments=np.array(t['all_action_indices'])
        assert assignments.shape==(len(p),len(t['bv_ids']))
        assert np.array_equal(assignments[:,t['bv_ids'].index(t['actor_id'])],t['action_indices'])
        assert np.isfinite(t['full_natural_action_logs']).all()
        assert np.all((np.array(t['elapsed_s'])>0)&(np.array(t['elapsed_s'])<=1+1e-12))
        for j,pairs in enumerate(t['collision_pairs']):
            assert bool(y[j])==any(cav_id in pair for pair in pairs)
            assert t['competing_results'][j]==(bool(pairs) and not y[j])
        selected=int(np.searchsorted(np.cumsum(p),schedule['control_uniforms'][index],side='right'))
        assert selected==t['selected_action_row']
        f=float(p@y);weight=float(np.exp(-np.logaddexp(np.log(.5),np.log(.5)+prefix['full_log_t']-prefix['full_log_p'])))
        values=(weight*y[selected],weight*f,weight**2*f*(1-f))
        np.testing.assert_allclose(f,t['conditional_event_value'],rtol=1e-12,atol=1e-30)
        np.testing.assert_allclose(values,[t[k] for k in ('single_value','branching_value','removed_conditional_variance_unbiased')],rtol=1e-11,atol=1e-25)
    else:
        assert node['status']=='branched' and len(node['children'])==4
        eta=float(np.exp(-np.logaddexp(0.,prefix['full_log_t']-prefix['full_log_p'])))
        np.testing.assert_allclose(eta,node['posterior_natural_fraction'],rtol=1e-11)
        children=[]
        for j,child in enumerate(node['children']):
            assert child['node_index']==4*index+j+1
            assert child['prefix']['steps'][:-1]==prefix['steps']
            assert child['prefix']['initial_cancelled_log_conditional']==prefix['initial_cancelled_log_conditional']
            value=reconstruct_tree(child,schedule,cav_id);error=max(error,value[3]);children.append(value[:3])
        children=np.array(children);p,t=children[:2,1],children[2:,1]
        mp,mt=p.sum()/2,t.sum()/2;sp=((p-mp)**2).sum();st=((t-mt)**2).sum()
        # Algebraically identical NONNEGATIVE expression avoids vs-vb
        # cancellation; it does not change the stored experiment or estimates.
        removed=(eta*sp/2+(1-eta)*st/2+eta*(1-eta)*(mp-mt)**2
                 +eta*children[:2,2].mean()+(1-eta)*children[2:,2].mean())
        selected=0 if schedule['control_uniforms'][index]<eta else 2
        assert selected==node['selected_child']
        values=(children[selected,0],eta*mp+(1-eta)*mt,removed)
    scale=max(values[0]**2,values[1]**2,
              float(np.max(children[:,1]**2)) if node['status']=='branched' else 0.)
    check_root_values(values,[node[k] for k in ('single_value','branching_value','removed_conditional_variance_unbiased')],scale)
    assert -1e-12<=values[1]<=2+1e-12
    return tuple(map(float,values))+(error,)


def audit(config,directory,replay=False):
    cfg,physical,assets,plans,ads,design,expected=inputs(config)
    root=Path(directory);summary=read(root/'summary.json')
    assert summary['complete'] and summary['config']==cfg and read(root/'measurement_plan.json')==design
    assert all(summary['lock'].get(k)==v for k,v in expected.items())
    for name,sha in summary['artifact_hashes'].items():
        assert Path(name).name==name and digest(root/name)==sha
    error=(1-cfg['confidence'])/(4*sum(len(s['strata']) for s in design['states']))
    estimates={(e['batch'],e['state_id'],e['arm']):e for e in summary['state_batch_estimates']}
    measurements={};max_error=0.;seeds=set();replays=0
    if replay:
        from . import highd_v47_decision_branching as native
        native.parent.parent.parent._init_worker(str(assets))
    for schedule in design['schedules']:
        b,i=schedule['batch'],schedule['state_id'];stored=read(root/f'batch_{b}_state_{i:03d}.json')
        assert len(stored['rows'])==len(schedule['rows'])==256
        for planned,row in zip(schedule['rows'],stored['rows']):
            assert all(row[k]==v for k,v in planned.items())
            ss=[row['initial_seed']]+row['step_seeds']+row['terminal_seeds'];assert not set(ss)&seeds;seeds.update(ss)
            values=reconstruct_tree(row['tree'],row,ads[i]['state']['cav_id']);max_error=max(max_error,values[3])
            check_root_values(values[:3],[row[k] for k in ('single_value','branching_value','removed_conditional_variance_unbiased')],
                              max(values[0]**2,values[1]**2))
        for arm in ('single','branching'):
            e=estimates[(b,i,arm)];v=reconstruct_estimate(design['states'][i]['strata'],stored['rows'],arm,error)
            np.testing.assert_allclose(v[:2],[e['J'],e['estimator_variance']],rtol=1e-12,atol=1e-30)
            np.testing.assert_allclose(v[2],e['bank_simultaneous_interval'],rtol=1e-12,atol=1e-30)
            np.testing.assert_allclose(v[3],e['removed_conditional_variance_unbiased'],rtol=1e-10,atol=1e-25)
            measurements[(b,i,arm)]=v
        if replay:
            index=(17*i+31*b)%256
            reconstructed=native.root_continuation(native.parent.parent.parent._MODEL,assets,physical,
                ads[i]['state'],plans[i]['actor_ids'],schedule['rows'][index],cfg)
            assert json.loads(json.dumps(reconstructed))=={k:v for k,v in stored['rows'][index].items() if k not in schedule['rows'][index]}
            replays+=1
    a=np.array(design['target_masses']);info={}
    for arm in ('single','branching'):
        js=[];vs=[]
        for i in range(len(plans)):
            x,y=measurements[(0,i,arm)],measurements[(1,i,arm)]
            j,v=(x[0]+y[0])/2,(x[1]+y[1])/4;s=summary['arms'][arm]['states'][i]
            np.testing.assert_allclose([j,v],[s['J'],s['estimator_variance']],rtol=1e-12,atol=1e-30)
            js.append(j);vs.append(v)
        np.testing.assert_allclose([a@js,(a*a)@vs],[summary['arms'][arm]['target_M2'],summary['arms'][arm]['target_measurement_variance']],rtol=1e-12)
        info[arm]=dict(target_M2=float(a@js),measurement_variance=float((a*a)@vs))
    allocation=read(root/'batch0_allocation_before_batch1.json');assert allocation==summary['uncertainty_aware_allocation']
    q=np.array(allocation['upper_envelope_candidate']);assert np.isclose(q.sum(),1.) and np.max(a/q)<=1.1+1e-14
    train=np.array([measurements[(0,i,'branching')][2] for i in range(len(plans))]);d=a*(a/q-1)
    upper=float(np.where(d>=0,d*train[:,1],d*train[:,0]).sum())
    np.testing.assert_allclose(upper,allocation['worst_case_M2_difference_upper'],rtol=1e-12)
    if not allocation['strict_improvement_certified']:
        np.testing.assert_array_equal(allocation['recommended_q'],a)
    test=np.array([measurements[(1,i,'branching')][0] for i in range(len(plans))]);comparison=summary['independent_batch_allocation_comparison']
    np.testing.assert_allclose([a@test,(a*a/q)@test],[comparison['target_M2'],comparison['candidate_M2']],rtol=1e-12)
    assert len(seeds)==37*summary['independent_roots']
    return dict(verified=True,summary_sha256=digest(root/'summary.json'),artifact_hashes_verified=len(summary['artifact_hashes']),
                independent_roots=summary['independent_roots'],predetermined_native_root_replays=replays,
                max_full_factor_log_error=max_error,arms=info,allocation=allocation)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='configs/highd_v47_decision_branching_v1.json')
    p.add_argument('--run',default='outputs/v47_decision_branching_v1');p.add_argument('--replay',action='store_true')
    a=p.parse_args();print(audit(a.config,a.run,a.replay))
