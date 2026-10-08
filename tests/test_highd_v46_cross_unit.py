from collections import Counter
from itertools import product
from types import SimpleNamespace
import numpy as np
import pytest
from scenario_reconstruction.highd_v46_cross_unit import (
    positive_interaction,proposal_factors,natural_log_probability,draw_intervention,CrossUnitExperiment,
)
from scenario_reconstruction.highd_v46_pair_synergy import pair_marginal
from scenario_reconstruction.highd_v46_assets import read,save,digest,ensure_assets,ROOT
from scenario_reconstruction.highd_v46_d2rl import NativeExperiment
from d2rl_training.highd_v46_sequence_env import V46SequenceEnv,log_weight


def units():
    return [{'ids':[1,2],'p':np.array([[.4,.1],[.2,.3]])},
            {'ids':[3,4],'p':np.array([[.1,.2],[.3,.4]])}]


def conditional_mass(partition,forced,actions):
    """Independent enumeration oracle, conditioning each entire natural unit."""
    mass=1.
    for unit in partition:
        p=unit['p'];denominator=0.
        for outcome in np.ndindex(p.shape):
            if all(k not in forced or forced[k]==a for k,a in zip(unit['ids'],outcome)):
                denominator+=p[outcome]
        numerator=p[tuple(actions[k] for k in unit['ids'])]
        mass*=numerator/denominator
    return mass


@pytest.mark.parametrize('ids',[[1,3],[2,3],[1,2],[2,1]])
@pytest.mark.parametrize('epsilon',[[.1,.2],[1.,1.],[.1,1.]])
def test_full_joint_q_normalization_support_and_exact_is(ids,epsilon):
    partition=units();p,_=pair_marginal(partition,ids)
    h=p*np.array([[.1,.2],[.3,2.]]);h/=h.sum()
    p1,p2,h1,h2,q1,q2=proposal_factors(p,h,epsilon)
    total=0.;event_natural=0.;event_is=0.;mean_weight=0.
    for values in product(range(2),repeat=4):
        actions=dict(zip([1,2,3,4],values));a,b=[actions[i] for i in ids]
        natural=np.prod([u['p'][tuple(actions[k] for k in u['ids'])] for u in partition])
        q=q1[a]*q2[a,b]*conditional_mass(partition,dict(zip(ids,(a,b))),actions)
        assert q>0
        weight=p1[a]*p2[a,b]/(q1[a]*q2[a,b])
        assert np.isclose(natural/q,weight)
        total+=q;mean_weight+=q*weight
        event=(actions[2]==1 and actions[4]==0)
        event_natural+=natural*event;event_is+=q*weight*event
        if epsilon==[1.,1.]:assert np.isclose(natural,q)
    assert np.isclose(total,1.) and np.isclose(mean_weight,1.)
    assert np.isclose(event_natural,event_is)


def test_actual_draws_follow_full_q_and_preserve_conditional_partner():
    partition=units();ids=[2,3];p,_=pair_marginal(partition,ids)
    h=p*np.array([[.1,.2],[.3,2.]]);h/=h.sum();rng=np.random.default_rng(1701)
    q1,q2=proposal_factors(p,h,[.1,.2])[-2:];counts=Counter()
    for _ in range(6000):
        actions,logged=draw_intervention(partition,ids,h,[.1,.2],rng)
        counts[tuple(actions[i] for i in (1,2,3,4))]+=1
        assert np.isclose(logged['log_weight'],log_weight(logged,[.1,.2]))
        assert np.isclose(logged['full_log_p']-logged['full_log_q'],logged['log_weight'])
    for values in product(range(2),repeat=4):
        actions=dict(zip([1,2,3,4],values));a,b=[actions[i] for i in ids]
        expected=q1[a]*q2[a,b]*conditional_mass(partition,dict(zip(ids,(a,b))),actions)
        assert abs(counts[values]/6000-expected)<.025


def test_zero_h_rows_fall_back_to_natural_conditional_and_no_support_expansion():
    p=np.array([[.4,.1],[.2,.3]]);h=np.array([[0.,0.],[0.,1.]])
    _,p2,_,h2,q1,q2=proposal_factors(p,h,[.05,.05])
    np.testing.assert_allclose(h2[0],p2[0]);assert np.isclose((q1[:,None]*q2).sum(),1.)
    with pytest.raises(ValueError):proposal_factors(np.diag([.5,.5]),np.ones((2,2))/4,[.1,.1])
    with pytest.raises(ValueError):proposal_factors(p,h,[0.,.1])


def test_interaction_score_ignores_single_actor_effect_and_unsupported_corners():
    p=np.ones((2,2))/4
    np.testing.assert_array_equal(positive_interaction(p,np.array([[0.,0.],[1.,1.]])),0.)
    np.testing.assert_array_equal(positive_interaction(p,np.array([[0.,0.],[0.,1.]])),[[0.,0.],[0.,1.]])
    assert not positive_interaction(np.diag([.5,.5]),np.array([[0.,0.],[0.,1.]])).any()


def test_logged_cross_unit_sequence_is_valid_offline_replay(tmp_path):
    road=CrossUnitExperiment.__new__(CrossUnitExperiment)
    cars=[SimpleNamespace(id=i,x=float(i*10),v=25.,lane=0,length=5.) for i in range(5)]
    road.vehicles=cars;road.cav=cars[0];road.time=0.;road.cav_acceleration=0.;road.last_acceleration={}
    road.rng=np.random.default_rng(1);road.steps=[];road.log_weight=0.;road.max_intervened=0;road.diagnostics=Counter()
    road.config={'horizon_s':60.};partition=units();ids=[1,3];p,_=pair_marginal(partition,ids)
    h=p*np.array([[.1,.2],[.3,2.]]);h/=h.sum()
    road.prepare=lambda pmfs:(partition,{c.id:c for c in cars})
    road.select_pair=lambda *args:{'ids':ids,'p':p,'h':h,'mass':.2,'contrast':1.,'factor_influence':[.2,.3],'metrics':{},'same_natural_unit':False}
    road.sample({},'dual',[.1,.1])
    step=road.steps[0]
    assert not step['same_natural_unit'] and step['affected_natural_units']==[[1,2],[3,4]]
    data={'contract':'highd_v46_native_d2rl_sequence_v2','observation_contract':'native_cav6_bv4x2_context_v2',
        'action_dimension':2,'intervention_mode':'dual','natural_target_sha256':'P','experiment_target_sha256':'target',
        'sequences':[{'seed':1,'steps':road.steps,'event_result':True,'generation_log_weight':road.log_weight}]}
    save(tmp_path/'sequences.json',data)
    save(tmp_path/'sequence_manifest.json',{'data_path':'sequences.json','sha256':digest(tmp_path/'sequences.json'),
        **{k:data[k] for k in ('natural_target_sha256','experiment_target_sha256','intervention_mode')}})
    env=V46SequenceEnv({'sequence_manifest':str(tmp_path/'sequence_manifest.json')})
    assert env.reset().shape==(22,)
    _,reward,done,_=env.step([.1,.1]);assert done and np.isclose(reward,-1.)


def test_natural_branch_preserves_existing_seeded_native_rollout():
    from scenario_reconstruction.highd_paper_discrete_pairs import PairModelQueries,SINGLE_OPTIONS
    archive=ROOT/'source_data/highd_paper_v46/runtime.zip'
    if not archive.exists() or archive.stat().st_size<1000:pytest.skip('V46 LFS runtime required')
    assets,_=ensure_assets();model=PairModelQueries(assets/'model',assets/'lateral',**SINGLE_OPTIONS)
    physical=read(ROOT/'configs/highd_v46_dual_context_v2.json');cfg=read(ROOT/'configs/highd_v46_cross_unit_v1.json')
    old=NativeExperiment(model,assets/'pair',assets/'lateral',140000,physical)
    new=CrossUnitExperiment(model,assets/'pair',assets/'lateral',140000,physical,proposal_config=cfg)
    for _ in range(4):
        assert old.step_policy('natural',1.)==new.step_policy('natural',[1.,1.])
        assert [vars(c) for c in old.vehicles]==[vars(c) for c in new.vehicles]
        assert old.natural.histories==new.natural.histories
    assert new.log_weight==0. and not new.steps
