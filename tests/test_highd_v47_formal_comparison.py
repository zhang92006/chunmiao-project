import copy
from types import SimpleNamespace
import numpy as np
import pytest

from scenario_reconstruction import highd_v47_formal_comparison as runner
from scenario_reconstruction import highd_v47_formal_native as native
from scenario_reconstruction import highd_v47_formal_statistics as stats
from scenario_reconstruction.highd_v46_assets import read,save,digest
from scenario_reconstruction.highd_v46_d2rl import NativeExperiment
from scenario_reconstruction.highd_v47_fixed_alpha_contract import state_schedule
from scenario_reconstruction.highd_v47_formal_training_metrics import diagnostics
from test_highd_v47_fixed_alpha_ppo import dataset,write,FixedRng
from d2rl_training.highd_v46_sequence_env import V46SequenceEnv


def cfg():return read(runner.DEFAULT)


def test_source_disjoint_protocol_and_no_state_learning():
    c=cfg();r=runner.ranges(c)
    assert all(not r[a]&r[b] for a in r for b in r if a!=b)
    assert min(set().union(*r.values()))>c['historical_namespace_excluded'][1]
    assert c['state_model'] is None and c['state_likelihood_ratio']==1
    assert c['training']['seeds']==[7,17,29,41,53]
    assert c['training']['iterations']==200


def test_dual_never_defines_common_critical_state_gate(monkeypatch):
    calls=[];single={'ids':[1]};pair={'ids':[1,2]}
    def choose(self,u,a,p,mode):calls.append(mode);return single if mode=='single' else pair
    monkeypatch.setattr(NativeExperiment,'choose',choose)
    assert native.common_choice(None,[],{}, {},'dual')==pair
    assert calls==['single','dual']
    calls.clear()
    monkeypatch.setattr(NativeExperiment,'choose',lambda self,u,a,p,mode:None if mode=='single' else pair)
    assert native.common_choice(None,[],{}, {},'dual') is None


def test_dual_fallback_is_identical_original_single_choice(monkeypatch):
    single={'ids':[1]};different={'ids':[3]}
    monkeypatch.setattr(NativeExperiment,'choose',lambda self,u,a,p,mode:single if mode=='single' else different)
    assert native.common_choice(None,[],{}, {},'dual') is single


def test_dual_fallback_has_exact_inactive_factor_and_partner_law(monkeypatch):
    p=np.array([[.15,.25],[.1,.5]]);h=np.array([.8,.2]);unit=dict(ids=[1,2],p=p)
    cars={1:SimpleNamespace(x=0.,length=5.,v=30.)}
    road=SimpleNamespace(config={'horizon_s':60},time=12.,log_weight=0.,steps=[],rng=FixedRng([.2,.7,.2]))
    monkeypatch.setattr(native,'common_choice',lambda *args:dict(ids=[1],h=h,mass=.1,relative_contrast=.2,unit=unit))
    monkeypatch.setattr(native,'observation',lambda *args:[0.]*14)
    actions,step,audit=native.act(road,[unit],cars,{},'dual',lambda obs:[.1,.7])
    assert step['factor_mask']==[1,0]
    assert [step[k][1] for k in ('p_factors','h_factors','q_factors')]==[1.,1.,1.]
    assert step['actual_QP_TV'][1]==0
    assert audit['log_p']-audit['log_q']==pytest.approx(step['log_weight'])
    assert road.log_weight==pytest.approx(step['log_weight'])
    assert set(actions)=={1,2}


def report(m2,supported=True):return dict(episodes=1024,weighted_second_moment=m2,uncertainty_supported=supported)


def test_validation_keeps_zero_when_no_supported_gain_and_uses_native_M2():
    assert stats.select_checkpoint({0:report(2),10:report(3),20:report(2)})[0]==0
    assert stats.select_checkpoint({0:report(2),10:report(0,False),20:report(1)})[0]==20
    assert stats.select_checkpoint({0:report(2,False),10:report(1)})[0]==0
    assert stats.select_checkpoint({0:report(2),10:report(1),20:report(1)})[0]==10
    with pytest.raises(ValueError):stats.select_checkpoint({10:report(1)})
    with pytest.raises(ValueError):stats.select_checkpoint({0:report(2),10:dict(report(1),episodes=2048)})


def test_neutral_states_have_support_and_matched_schedule():
    law=[dict(case_id='a',source_seed=1,alpha=.25),dict(case_id='b',source_seed=1,alpha=.25),dict(case_id='c',source_seed=2,alpha=.5)]
    assert state_schedule(law,100,7)==state_schedule(copy.deepcopy(law),100,7)
    assert len(set(state_schedule(law,100,7)))==3


def row(i,event=True):
    return dict(source_seed=i,seed=i,event_result=event,generation_log_weight=0.,cpu_s=.1,steps=[],native_decisions=[dict(log_p=-1.,log_q=-1.)])


def test_sparse_event_moments_do_not_produce_precision_or_promoted_checkpoint():
    c=cfg();r=stats.describe([row(i,i==0) for i in range(50)],c)
    assert r['weighted_second_moment']==pytest.approx(.02)
    assert r['relative_half_width'] is None and r['uncertainty_supported'] is False
    assert r['maximum_source_M2_share']==1.
    assert all(v is None for v in r['required_tests_observed'].values())
    assert not stats.readiness([dict(mode=m,seed=s,iteration=0,metrics=r) for m in ('single','dual') for s in c['training']['seeds']],c)['passed']


def test_paired_bootstrap_rejects_trajectory_resampling_as_sources():
    c=cfg();a=[row(1),row(1)]
    with pytest.raises(ValueError,match='one full reset'):stats.paired_comparison(a,a,c)


def test_common_reward_scale_and_diagnostics_preserve_replay_objective(tmp_path):
    d=dataset('dual')
    for s in d['sequences']:
        for step in s['steps']:
            step['factor_pmfs']=[dict(p=[.4,.6],h=[.7,.3])]*2
    env=V46SequenceEnv(dict(sequence_manifest=write(tmp_path,d),log_reward_scale=0.))
    diag=diagnostics(env,lambda obs:[.1,.1])
    assert env.log_scale==0.
    assert diag['empirical_scaled_second_moment']==pytest.approx(env.empirical_objective(lambda obs:[.1,.1]))
    assert diag['empirical_second_moment']==diag['empirical_scaled_second_moment']
    assert diag['dataset_event_count']==4
    assert diag['epsilon_quantiles_by_active_factor']==[[.1,.1,.1]]*2
    assert diag['conditional_QP_TV_at_logged_action_rows']==pytest.approx([.27,.27])


def test_final_gate_never_starts_native_work_on_failed_selection(tmp_path,monkeypatch):
    c=cfg();lock={'x':1}
    monkeypatch.setattr(runner,'checked',lambda *a:(c,{},None,lock))
    monkeypatch.setattr(runner,'native_tasks',lambda *a:pytest.fail('Opened FinalTest prematurely'))
    r=stats.describe([row(i,False) for i in range(10)],c)
    selected=[dict(mode=m,seed=s,iteration=0,metrics=r) for m in ('single','dual') for s in c['training']['seeds']]
    save(tmp_path/'lock.json',dict(lock=lock,rules_frozen=True,selected=selected))
    with pytest.raises(ValueError,match='stop-condition'):runner.final_test('cfg','frozen',tmp_path/'lock.json',tmp_path/'out')
    assert not (tmp_path/'out').exists()


def test_historical_namespace_or_split_overlap_is_rejected_before_assets(tmp_path):
    c=cfg();c['source_splits']['train']['seed_start']=200000
    save(tmp_path/'cfg.json',c)
    with pytest.raises(ValueError,match='Historical'):runner.protocol(tmp_path/'cfg.json')
    c=cfg();c['source_splits']['validation']=c['source_splits']['train'].copy()
    save(tmp_path/'overlap.json',c)
    with pytest.raises(ValueError,match='overlap'):runner.protocol(tmp_path/'overlap.json')


def test_native_pair_factors_full_probability_and_distribution_TV(monkeypatch):
    p=np.array([[.15,.25],[.1,.5]]);h=np.array([[.3,.3],[.3,.1]])
    unit=dict(ids=[1,2],p=p);cars={i:SimpleNamespace(x=i*10.,length=5.,v=30.) for i in (1,2)}
    road=SimpleNamespace(config={'horizon_s':60},time=12.,log_weight=0.,steps=[],rng=FixedRng([.2,.7,.3,.4]))
    monkeypatch.setattr(native,'common_choice',lambda *args:dict(ids=[1,2],h=h,mass=.1,relative_contrast=.2,unit=unit))
    monkeypatch.setattr(native,'observation',lambda *args:[0.]*14)
    _,step,audit=native.act(road,[unit],cars,{},'dual',lambda obs:[.2,.4])
    p1=p.sum(1);h1=h.sum(1);q1=.2*p1+.8*h1
    p2=p/p1[:,None];h2=h/h1[:,None];q2=.4*p2+.6*h2
    a,b=step['action_indices']
    assert np.exp(audit['log_p'])==pytest.approx(p[a,b])
    assert np.exp(audit['log_q'])==pytest.approx(q1[a]*q2[a,b])
    assert np.exp(step['log_weight'])==pytest.approx(p[a,b]/(q1[a]*q2[a,b]))
    assert step['actual_QP_TV']==pytest.approx([.5*np.abs(q1-p1).sum(),np.sum(q1*.5*np.abs(q2-p2).sum(1))])


def test_degenerate_and_underflow_samples_never_claim_precision():
    c=cfg();ones=stats.describe([row(i) for i in range(30)],c)
    assert ones['uncertainty_supported'] is False
    assert all(not p['precision_reached'] for curve in ones['convergence_curves'].values() for p in curve)
    rare=[dict(row(i),generation_log_weight=-1000.) for i in range(30)]
    observed=stats.describe(rare,c)
    assert observed['raw_event_rate']==1 and observed['native_CAV_events']==30
    assert observed['event_contribution_underflow_count']==30 and observed['weighted_event_sources']==0
    assert not observed['uncertainty_supported']


def test_readiness_cannot_drop_unfavorable_training_seeds():
    c=cfg();r=stats.describe([row(i,i%2==0) for i in range(50)],c)
    assert 'Every predeclared' in stats.readiness([],c)['reasons'][0]
    selected=[dict(mode=m,seed=s,iteration=10,metrics=r) for m in ('single','dual') for s in c['training']['seeds']]
    assert 'Every predeclared' in stats.readiness(selected[:-1],c)['reasons'][0]


def test_final_audit_reconstructs_all_arms_without_new_sources(tmp_path,monkeypatch):
    c=cfg();c['final_test']['bootstrap_replicates']=10
    c['source_splits']['final_test']=dict(seed_start=1000200000,episodes=30)
    lock=dict(config=c);root=tmp_path/'result';root.mkdir()
    monkeypatch.setattr(runner,'checked',lambda *a:(c,{},None,lock))
    records=[row(i,i%2==0) for i in sorted(runner.ranges(c)['final_test'])]
    for s in records:s.update(state_likelihood_ratio=1.,state_log_weight=0.,initial_fleet=[],cav_id=1)
    labels=['fixed_'+m for m in ('natural','single','dual')]+[m+'_seed'+str(s) for s in c['training']['seeds'] for m in ('single','dual')]
    for label in labels:
        hashes={}
        for i,s in enumerate(records):
            p=root/label/('draw_%05d.json'%i);save(p,s);hashes[str(p.relative_to(root))]=digest(p)
        save(root/(label+'_artifacts.json'),dict(hashes=hashes))
        save(root/(label+'_plan.json'),dict(tasks=[dict(seed=s['seed'],case_id=None) for s in records]))
    summary=dict(complete=True,lock=lock,fixed_controls={m:stats.describe(records,c) for m in ('natural','single','dual')},
        paired_seed_comparisons={str(seed):stats.paired_comparison(records,records,c) for seed in c['training']['seeds']})
    summary['final_readiness']=stats.final_readiness(summary['paired_seed_comparisons'],c)
    save(root/'summary.json',summary)
    assert runner.audit_final('cfg','frozen',root,tmp_path/'audit.json')['verified']
    assert read(tmp_path/'audit.json')['native_replays']==0
    # Changing a raw zero-event endpoint still invalidates the audit.
    (root/'fixed_single/draw_00001.json').write_text('{}')
    with pytest.raises(ValueError,match='artifact changed'):runner.audit_final('cfg','frozen',root,tmp_path/'invalid.json')


def test_report_renders_all_seeds_and_preserves_censored_AF(tmp_path):
    from scenario_reconstruction.highd_v47_formal_report import render
    c=cfg();c['final_test']['bootstrap_replicates']=10;lock=dict(config=c)
    records=[row(i,i==0) for i in range(30)]
    pairs={str(seed):stats.paired_comparison(records,records,c) for seed in c['training']['seeds']}
    root=tmp_path/'final';save(root/'summary.json',dict(complete=True,lock=lock,paired_seed_comparisons=pairs,conclusion='not established'))
    save(tmp_path/'audit.json',dict(verified=True,native_replays=13,lock=lock,summary_sha256=digest(root/'summary.json')))
    assert render(root,tmp_path/'audit.json',tmp_path/'report')['paired_seeds']==5
    report=read(tmp_path/'report/report.json')
    assert len(report['paired_seeds'])==5 and len(report['figures'])==2
    assert all(v is None for r in report['paired_seeds'] for v in r['AF_observed'].values())


def test_final_evidence_cannot_be_promoted_from_a_favorable_ratio_alone():
    c=cfg();r=stats.describe([row(i,i==0) for i in range(30)],c)
    pairs={str(seed):dict(single=r,dual=r,supports_dual_lower_M2=True) for seed in c['training']['seeds']}
    screen=stats.final_readiness(pairs,c)
    assert not screen['passed'] and any('source concentration' in s for s in screen['reasons'])
