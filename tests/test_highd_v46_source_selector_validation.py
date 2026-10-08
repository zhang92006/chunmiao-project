import copy

import numpy as np
import pytest

from scenario_reconstruction.highd_v46_assets import read, save
from scenario_reconstruction.highd_v46_source_selector_validation import (
    DEFAULT_CONFIG, case_plan, check_lock, proposal, select_per_source,
)
from scenario_reconstruction.highd_v46_source_statistics import (
    cluster_bootstrap, interval, summarize_confirmation, summarize_sources,
)


def config():
    return read(DEFAULT_CONFIG)


def candidate(seed, time=5., ids=(1,2), risk=.9, strict=False, state_index=0):
    return {'seed':seed,'time_s':time,'actor_ids':list(ids),'state_index':state_index,
            'ranker_features':[1.,.1,risk,0.],'views':{'production_strict':strict}}


def source(seed, events=0, m2=0., mean=0., case=True, tv=(.5,.6)):
    return {'source_seed':seed,'case_id':seed if case else None,'factor_tv':list(tv) if case else None,
            'arm_factor_tv':{a:list(tv) for a in config()['arms']} if case else None,
            'arms':{a:{'events':events,'trials':64,'mean':mean,'second_moment':m2,'mean_mc_variance':.01}
                    for a in config()['arms']} if case else None}


def summaries(groups,cfg):
    return {method:{'arms':{arm:summarize_sources(records,arm,cfg['screening_readiness'],cfg['statistics'])
                           for arm in cfg['arms']}} for method,records in groups.items()}


def test_A_and_B_differ_only_by_research_eligibility():
    rows=[candidate(1,risk=.9),candidate(1,time=6.,risk=.8,strict=True),candidate(2,strict=True,state_index=1)]
    result=select_per_source(rows,[1,2,3])
    assert result=={'A':{1:0,2:2,3:None},'B':{1:1,2:2,3:None}}
    for r in rows:r['native_events']=1000
    assert select_per_source(rows,[1,2,3])==result
    tied=[candidate(1,time=6.,ids=(3,4)),candidate(1,time=5.,ids=(3,4)),candidate(1,time=5.,ids=(1,2))]
    assert select_per_source(tied,[1])['A'][1]==2
    with pytest.raises(ValueError):select_per_source([rows[0],rows[0]],[1])


def test_plan_deduplicates_shared_pair_and_locks_probe_without_outcomes():
    cfg=config(); seeds=[1,2,3]
    bank=[{'seed':1,'time_s':5.},{'seed':1,'time_s':6.},{'seed':2,'time_s':5.}]
    rows=[candidate(1),candidate(1,time=6.,strict=True,state_index=1),candidate(2,strict=True,state_index=2)]
    selection=select_per_source(rows,seeds)
    plans,probe=case_plan(bank,rows,seeds,selection,cfg['confirmation'])
    assert len(plans)==3 and probe==[1,2]
    assert plans[2]['selected_for']==['A','B']
    assert all(p['confirmation_planned'] for p in plans)
    assert not any(p['source_seed']==3 for p in plans)


def test_ungated_Q_support_normalization_and_zero_mass_natural_fallback():
    p=np.array([[.7,.1],[.1,.1]])
    risk=np.array([[0.,0.],[0.,1.]])
    h,mass,tvs=proposal(p,risk,config()['arms'])
    assert mass==pytest.approx(.1) and h[1,1]==1.
    assert tvs['dual'][0]>0 and tvs['dual'][1]>0
    assert tvs['natural']==[0.,0.]
    assert tvs['first_factor_only'][1]==0.
    assert tvs['second_factor_only'][0]==0.
    from scenario_reconstruction.highd_v46_cross_unit import proposal_factors
    q1,q2=proposal_factors(p,h,[.1,.1])[-2:];q=q1[:,None]*q2
    assert q.sum()==pytest.approx(1.) and np.all(q>=.01*p)
    assert np.sum(q*p/q)==pytest.approx(1.)
    h,mass,tvs=proposal(p,np.zeros_like(p),config()['arms'])
    assert mass==0 and np.array_equal(h,p)
    assert all(x==[0.,0.] for x in tvs.values())


def test_source_weights_mcse_and_concentration_include_selected_zeros():
    cfg=config(); records=[source(1,events=2,m2=9.,mean=1.),source(2,events=1,m2=1.,mean=0.),source(3),source(4,case=False)]
    report=summarize_sources(records,'dual',cfg['screening_readiness'],cfg['statistics'])
    assert report['candidate_source_fraction']==.75
    n=report['native']
    assert n['event_source_episodes']==2 and n['observed_event_source_yield']==.5
    assert n['conditional_second_moment']==pytest.approx(10/3)
    assert n['conditional_is_mean']==pytest.approx(1/3)
    assert n['maximum_source_contribution']==pytest.approx(.9)
    assert n['effective_second_moment_sources_descriptive']==pytest.approx(1/(.9**2+.1**2))
    assert n['fixed_state_mean_mcse']==pytest.approx(np.sqrt(.03)/3)
    assert n['by_source'][2]['contribution_fraction']==0.
    assert not n['source_diversity_guard'] and not n['event_interval_guard']


def test_natural_fallback_events_do_not_pass_effective_pair_diversity():
    cfg=config(); records=[source(i,events=8,m2=1.,tv=(0.,0.)) for i in range(8)]
    n=summarize_sources(records,'dual',cfg['screening_readiness'],cfg['statistics'])['native']
    assert n['event_source_episodes']==8 and n['effective_pair_event_source_episodes']==0
    assert not n['source_diversity_guard']


def test_no_cases_and_zero_events_remain_distinct():
    cfg=config()
    none=summarize_sources([source(1,case=False)],'dual',cfg['screening_readiness'],cfg['statistics'])
    zero=summarize_sources([source(1)],'dual',cfg['screening_readiness'],cfg['statistics'])
    assert none['native'] is None
    assert zero['native']['cav_events']==0
    assert zero['native']['maximum_source_contribution'] is None


def test_cluster_bootstrap_preserves_pairing_source_count_and_sparse_guards():
    cfg=config();cfg['statistics']['bootstrap_repeats']=100
    records=[source(1,events=2,m2=4.),source(2),source(3,case=False)]
    groups={'A':records,'B':copy.deepcopy(records)}
    result=cluster_bootstrap(groups,summaries(groups,cfg),cfg['statistics'])
    assert result['unit']=='complete source episode'
    assert result['paired']['candidate_source_fraction_B_minus_A']['interval']==[0.,0.]
    assert result['selectors']['A']['conditional_second_moment']['interval'] is None
    assert result['selectors']['A']['conditional_is_mean']['undefined_replicates']>0
    assert result['paired']['observed_event_source_yield_B_minus_A']['status']=='insufficient_event_sources'
    bad=copy.deepcopy(groups);bad['B'].reverse()
    with pytest.raises(ValueError):cluster_bootstrap(bad,summaries(bad,cfg),cfg['statistics'])


def test_cluster_intervals_when_multiple_event_sources_present():
    cfg=config();cfg['statistics']['bootstrap_repeats']=100
    groups={'A':[source(i,events=5,m2=float(i+1),mean=.1) for i in range(8)],
            'B':[source(i,events=6,m2=float(i+2),mean=.2) for i in range(8)]}
    result=cluster_bootstrap(groups,summaries(groups,cfg),cfg['statistics'])
    assert result['selectors']['A']['conditional_is_mean']['interval']==pytest.approx([.1,.1])
    assert result['selectors']['B']['conditional_is_mean']['interval']==pytest.approx([.2,.2])
    assert result['paired']['source_weighted_event_rate_B_minus_A']['interval']==pytest.approx([1/64,1/64])
    assert result['selectors']['A']['candidate_source_fraction']['status']=='degenerate_observed_boundary'
    assert result['selectors']['A']['candidate_source_fraction']['interval'] is None


def test_confirmation_missing_is_not_a_native_zero():
    r=summarize_confirmation([{'status':'measured','native_interaction_mean':.1},
                             {'status':'measured','native_interaction_mean':-.1},
                             {'status':'selector_abstained'}],.05)
    assert r['confirmation_rate_among_measured']==.5
    assert r['confirmed_source_yield_in_planned_subset']==pytest.approx(1/3)
    assert r['unmeasured_reasons']=={'selector_abstained':1}


def test_supported_native_corners_use_common_rng_and_separate_risk_contrast(monkeypatch):
    from scenario_reconstruction import highd_v46_source_selector_validation as module
    cfg=config();cfg['confirmation']['replicates_per_corner']=2
    plan={'source_seed':240002,'actor_ids':[1,2],'confirmation_corners':[(0,0),(1,0),(0,1),(1,1)]}
    calls=[]
    def continuation(model,assets,physical,state,forced,seed,horizon,scale):
        corner=(forced[1],forced[2]);calls.append((corner,seed))
        return {'clearance_risk':{(0,0):.1,(1,0):.2,(0,1):.3,(1,1):.8}[corner],
                'cav_first_collision':corner==(1,1),'bv_only_collision':False}
    monkeypatch.setattr(module,'continue_state',continuation)
    result=module.mechanism(None,None,None,{},plan,cfg,2.)
    assert result['native_interaction_mean']==pytest.approx(.4)
    assert result['fixed_state_interaction_mcse']==0.
    assert [s for _,s in calls]==[3900004]*4+[3900005]*4
    assert result['replicate_cav_events']==[[0,0,0,1]]*2
    plan['confirmation_corners']=None
    assert module.mechanism(None,None,None,{},plan,cfg,2.)['status']=='no_positive_supported_contrast'
    assert len(calls)==8


@pytest.mark.parametrize('mutation',['reserved','old_validation','old_training','mode','budget','seed_overlap','environment'])
def test_protocol_rejects_leakage_and_contract_changes(tmp_path,mutation,monkeypatch):
    from scenario_reconstruction import highd_v46_source_selector_validation as module
    cfg=config()
    if mutation=='reserved':cfg['source']['seed_start']=100000
    elif mutation=='old_validation':cfg['source']['seed_start']=210000
    elif mutation=='old_training':cfg['source']['seed_start']=230000
    elif mutation=='mode':cfg['source']['split']='train'
    elif mutation=='budget':cfg['maximum_states_per_source']=2
    elif mutation=='seed_overlap':cfg['confirmation']['continuation_seed_start']=cfg['continuation_seed_start']
    else:cfg['environment']['numpy']='wrong'
    save(tmp_path/'config.json',cfg)
    with pytest.raises(ValueError):module.protocol(tmp_path/'config.json')


def test_full_source_validation_path_and_rule_lock(tmp_path,monkeypatch):
    from scenario_reconstruction import highd_v46_source_selector_validation as module
    from scenario_reconstruction.highd_v46_d2rl import identity
    cfg=config();cfg['source']['episodes']=3;cfg['replicates_per_case_arm']=2
    cfg['confirmation']['replicates_per_corner']=2;cfg['statistics']['bootstrap_repeats']=20
    path=tmp_path/'config.json';save(path,cfg)
    monkeypatch.setattr(module,'environment',lambda:cfg['environment'])
    monkeypatch.setattr(module,'ensure_assets',lambda:(tmp_path,{'natural_target_sha256':'P'}))
    monkeypatch.setattr(module,'experiment_target',lambda *args:{'P':'P'})
    monkeypatch.setattr(module,'load_calibration',lambda directory,*args:
        ({'natural_target_sha256':'P','experiment_target_sha256':identity({'P':'P'})},
         {200000} if 'calibration' in str(directory) else {230000},[],[],[],{}))
    module.freeze(path,tmp_path/'lock.json')
    seeds=list(range(240000,240003));bank=[{'seed':seeds[0],'time_s':5.},{'seed':seeds[0],'time_s':6.},
                                       {'seed':seeds[1],'time_s':5.}]
    rows=[candidate(seeds[0]),candidate(seeds[0],time=6.,strict=True,state_index=1),
          candidate(seeds[1],strict=True,state_index=2)]
    monkeypatch.setattr(module,'PairModelQueries',lambda *args,**kwargs:None)
    monkeypatch.setattr(module,'discover',lambda *args:(bank,rows,{'episodes':[{'seed':s} for s in seeds]}))
    monkeypatch.setattr(module,'prepare',lambda state,row,*args:
        ({'ids':[1,2],'h':np.ones((2,2))/4},{'mass':.1,'factor_tv':[.5,.6],
         'arm_factor_tv':{a:[.5,.6] for a in cfg['arms']},'natural_fallback':False,
         'confirmation_corners':None}))
    calls=[]
    def sample(model,assets,physical,state,chosen,eps,seed,*args):
        calls.append((state['seed'],seed,tuple(eps)))
        return {'seed':seed,'event_result':seed%2==0,'generation_log_weight':0.,'bv_only_collision':False,
                'initial_likelihood_audit':{'full_log_p':0.,'full_log_q':0.},
                'clearance_risk':.1,'cpu_s':.01,'wall_s':.01}
    monkeypatch.setattr(module,'sample_continuation',sample)
    result=module.run(path,tmp_path/'lock.json',tmp_path/'out')
    assert result['complete'] and result['case_count']==3 and not result['ppo_trained']
    for method in ('A','B'):
        n=result['selectors'][method]['arms']['dual']
        assert n['all_source_episodes']==3 and n['candidate_source_episodes']==2
        assert n['native']['event_source_episodes']==2
    assert len(calls)==3*4*2
    assert {seed for src,seed,eps in calls if src==240000}=={2900000,2900001}
    assert {seed for src,seed,eps in calls if src==240001}=={2900002,2900003}
    assert not (tmp_path/'out/sequence_manifest.json').exists()
    assert not result['training_manifest_exported'] and result['decision']['ppo_ready'] is False
    plan=read(tmp_path/'out/case_plan.json')
    assert plan['selection_uses_native_outcomes'] is False
    assert plan['cases'][2]['selected_for']==['A','B']
    with pytest.raises(FileExistsError):module.run(path,tmp_path/'lock.json',tmp_path/'out')
    changed=copy.deepcopy(read(tmp_path/'lock.json'));changed['implementation_sha256']='wrong'
    save(tmp_path/'wrong_lock.json',changed)
    with pytest.raises(ValueError,match='Rule changed'):check_lock(path,tmp_path/'wrong_lock.json')
    changed=copy.deepcopy(read(tmp_path/'lock.json'));changed['dependencies']['proxy']='wrong'
    save(tmp_path/'wrong_dependencies.json',changed)
    with pytest.raises(ValueError,match='Rule changed'):check_lock(path,tmp_path/'wrong_dependencies.json')
    changed=copy.deepcopy(cfg);changed['continuation_seed_start']=2500000
    save(tmp_path/'overlap_config.json',changed)
    with pytest.raises(ValueError,match='schedule overlaps training'):
        module.freeze(tmp_path/'overlap_config.json',tmp_path/'overlap_lock.json')
    assert not (tmp_path/'overlap_lock.json').exists()
