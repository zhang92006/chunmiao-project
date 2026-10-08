"""Read-only reconstruction of new-source feature allocation evidence."""
import argparse
import json
from pathlib import Path

import numpy as np

from . import highd_v47_source_transfer as native
from .highd_v46_assets import digest,read,save
from .highd_v46_collision_coverage import audit_sample
from .highd_v46_cross_unit import proposal_factors
from .highd_v47_allocation_certificate_audit import reconstruct_natural
from .highd_v47_decision_branching_audit import reconstruct_tree


def audit(config,directory,replay=False,native=native):
    # The prospective v2 validation adapter supplies the same reconstruction
    # contract without mutating global modules or the frozen v1 collector.
    cfg,dc,base,physical,assets,current=native.protocol(config)
    root=Path(directory);summary=read(root/'summary.json');split=summary['split'];lock=summary['lock']
    assert summary['complete'] and summary['config']==cfg and split in ('train','validation')
    assert all(lock.get(k)==v for k,v in current.items())
    assert set(summary['artifact_hashes'])=={p.name for p in root.glob('*.json') if p.name!='summary.json'}
    for name,sha in summary['artifact_hashes'].items():
        assert Path(name).name==name and digest(root/name)==sha
    opened=read(root/'opened_protocol.json');assert opened['lock']==lock and opened['config']==cfg
    source_seeds=list(range(cfg[split]['seed_start'],cfg[split]['seed_start']+cfg[split]['episodes']))
    plan=read(root/'case_plan.json');assert plan['source_seeds']==source_seeds and not plan['selection_uses_native_outcomes']
    bank=read(root/'state_bank.json');rows=read(root/'pair_audit.json')['rows'];assert bank['split']==split
    model=native.PairModelQueries(assets/'model',assets/'lateral',**native.SINGLE_OPTIONS)
    cases,selections=native.prepare_cases(bank['states'],rows,source_seeds,model,physical,base,cfg)
    assert plan['selections']==json.loads(json.dumps(selections)) and plan['cases']==[c['plan'] for c in cases]
    assert len(cases)==summary['selected_sources'] and summary['natural_sources']==len(source_seeds)
    for c in cases:assert c==read(root/f'case_{c["plan"]["case_id"]:03d}_design.json')
    fitted=None
    if split=='validation':
        training,_,fitted=native.completed_training(lock['training_path'],cfg,current)
        assert digest(training/'summary.json')==lock['training_summary_sha256']
        assert digest(training/'model.json')==lock['model_sha256']
    design=native.measurement_design(cases,cfg,split,fitted);design['model_config']=cfg['model']
    design['conditional_target_sha256']=native.identity([(c['plan']['state_sha256'],c['plan']['actor_ids'],c['plan']['target_mass']) for c in cases])
    assert design==read(root/'measurement_plan.json')
    records=native.read_records(root);by_key={(r['batch'],r['state_id']):r for r in records}
    assert len(by_key)==len(records)==len(design['schedules'])
    all_seeds=set();max_error=0.;replays=0;independent=[]
    for schedule in design['schedules']:
        b,i=schedule['batch'],schedule['state_id'];record=read(root/f'batch_{b}_state_{i:03d}.json');case=cases[i]
        assert len(record['rows'])==len(schedule['rows'])
        for planned,row in zip(schedule['rows'],record['rows']):
            assert all(row[k]==v for k,v in planned.items())
            if schedule['side'] in ('training','benefit'):
                seeds=[row['initial_seed']]+row['step_seeds']+row['terminal_seeds']
                values=reconstruct_tree(row['tree'],row,case['state']['cav_id']);max_error=max(max_error,values[3])
                np.testing.assert_allclose(values[1],row['measurement_value'],rtol=1e-10,atol=1e-25)
            else:
                seeds=[row['native_seed']];reconstruct_natural(row,case['state'],case['plan']['actor_ids'])
            assert not set(seeds)&all_seeds;all_seeds.update(seeds)
        j=variance=0.
        for h in design['states'][i]['strata']:
            group=[r for r in record['rows'] if r['stratum_id']==h['stratum_id']]
            z=np.array([r['measurement_value'] for r in group]);n=len(z);c=h['coefficient_mass']
            assert n==h['draws'] and {r['within_stratum_draw'] for r in group}==set(range(n))
            mean=z.sum()/n;j+=c*mean;variance+=c*c*np.sum((z-mean)**2)/(n*(n-1))
        independent.append(dict(batch=b,state_id=i,J=float(j),variance=float(variance)))
        if replay:
            index=(17*i+31*b)%len(record['rows']);planned=schedule['rows'][index]
            if schedule['side'] in ('training','benefit'):
                value=native.branching.root_continuation(model,assets,physical,case['state'],case['plan']['actor_ids'],planned,cfg)
                value['measurement_value']=value['branching_value']
            else:value=native.natural_root(model,assets,physical,case['state'],case['plan']['actor_ids'],planned)
            assert json.loads(json.dumps(value))=={k:v for k,v in record['rows'][index].items() if k not in planned}
            replays+=1
    measured=native.estimate(design,records);assert measured==summary['measurements']
    for r in independent:
        np.testing.assert_allclose([r['J'],r['variance']],
            [measured[r['state_id']]['batch_estimates'][r['batch']][k] for k in ('J','variance')],rtol=1e-12,atol=1e-25)
    if split=='train' and summary['model_sha256'] is not None:
        fitted=native.fit_model(design,records);assert fitted==read(root/'model.json')
        assert fitted['model']['fitting_sources']==sorted(c['plan']['source_seed'] for c in cases)
        assert digest(root/'model.json')==summary['model_sha256']
        for report in fitted['cross_fitted_reports']:
            for fold in report['folds']:assert not set(fold['fitting_sources'])&set(fold['checking_sources'])
    iid_draws=0
    if split=='validation':
        certificates=native.objective_reports(design,records);assert certificates==summary['objective_certificates']
        a=np.array(design['target_masses']);q=np.array(design['candidate_q']);d=a*(a/q-1)
        # Independent product/log algebra, not a call to the bound helpers.
        for b,report in enumerate(certificates):
            benefit=[];loss=[];support=[]
            for i,s in enumerate(design['states']):
                for h in s['strata']:
                    group=[r for r in by_key[(b,i)]['rows'] if r['stratum_id']==h['stratum_id']]
                    w=abs(d[i])*h['coefficient_mass']/len(group)
                    if d[i]<0:benefit.extend(w*r['measurement_value'] for r in group)
                    elif d[i]>0:loss.extend(w*r['measurement_value'] for r in group);support.extend([w]*len(group))
            lb=design['benefit_bet_rate'];lu=design['loss_bet_rate'];error=design['error_per_side_per_batch']
            lower=max(0.,(sum(np.log1p(lb*x) for x in benefit)-np.log(1/error))/lb)
            upper=min(sum(support),(np.log(1/error)-sum(np.log1p(-lu*x) for x in loss))/lu)
            np.testing.assert_allclose([lower,upper,upper-lower],
                [report['benefit']['lower'],report['loss']['upper'],report['M2_difference_upper']],rtol=1e-11,atol=1e-25)
        comparison=native.source_comparison(design,measured,source_seeds,cfg['statistics'])
        assert comparison==summary['source_comparison']
        physical_paths=set()
        for arm in ('target_iid','allocated_iid'):
            samples=read(root/f'{arm}_sequences.json')['sequences'];qs=a if arm=='target_iid' else q;z=[]
            assert len(samples)==cfg['iid_draws_per_arm']
            for draw,(i,row) in enumerate(zip(design['iid_schedules'][arm],samples)):
                assert (row['draw_id'],row['state_id'],row['seed'])==(draw,i,cfg['iid_seed_start']+draw)
                assert row['seed'] not in all_seeds;physical_paths.add((draw,i));iid_draws+=1
                audit_sample(row,[.1,.1]);log=row['initial_likelihood_audit'];u,v=log['action_indices']
                p1,p2,h1,h2,q1,q2=proposal_factors(cases[i]['P'],cases[i]['H'],[.1,.1])
                np.testing.assert_allclose([p1[u],p2[u,v],h1[u],h2[u,v],q1[u],q2[u,v]],
                    log['p_factors']+log['h_factors']+log['q_factors'],rtol=1e-12,atol=1e-25)
                full=np.log(a[i]/qs[i])+log['full_log_p']-log['full_log_q'];max_error=max(max_error,abs(full-row['total_log_weight']))
                np.testing.assert_allclose(full,row['total_log_weight'],rtol=1e-11,atol=1e-12)
                z.append(float(row['event_result'])*np.exp(full))
            stored=summary['iid_smoke']['arms'][arm]
            np.testing.assert_allclose([np.mean(z),np.mean(np.square(z)),np.var(z,ddof=1)/len(z)],
                [stored[k] for k in ('mean','pooled_second_moment','estimator_variance')],rtol=1e-11,atol=1e-25)
        assert len(physical_paths)==summary['iid_smoke']['physical_paths']
        assert summary['transfer_review_passed']==(all(r['improvement_certified'] for r in certificates) and comparison['source_generalization_supported'])
    assert len(all_seeds)>0 and not summary['ppo_ready'] and not summary['production_changed']
    return dict(verified=True,split=split,summary_sha256=digest(root/'summary.json'),
        artifact_hashes_verified=len(summary['artifact_hashes']),independent_measurement_roots=design['roots'],
        predetermined_native_root_replays=replays,max_full_factor_log_error=max_error,iid_draws=iid_draws,
        audit_implementation_sha256=digest(__file__))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default=native.DEFAULT)
    p.add_argument('--run',required=True);p.add_argument('--replay',action='store_true');p.add_argument('--output');a=p.parse_args()
    result=audit(a.config,a.run,a.replay)
    if a.output:save(a.output,result)
    print(result)
