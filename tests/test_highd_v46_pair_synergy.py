from collections import Counter
from types import SimpleNamespace
import json
import copy
import numpy as np
import pytest

from scenario_reconstruction.highd_paper_dynamics import Vehicle, advance
from scenario_reconstruction.highd_v46_d2rl import NativeExperiment
from scenario_reconstruction.highd_v46_pair_synergy import (
    pair_metrics, pair_marginal, select_pair, bounded_motion, response_risk,
    candidate_pairs, state_features,
)
from scenario_reconstruction.highd_v46_pair_witness import conditional_actions

THRESHOLDS={'minimum_delta':.01,'minimum_interaction':.05,
            'minimum_joint_excess':.05,'minimum_factor_tv':.001,'minimum_synergy_mass':1e-6}


def test_single_actor_effect_is_not_synergy():
    p=np.ones((2,2))/4
    result=pair_metrics(p,np.array([[0.,0.],[1.,1.]]),THRESHOLDS)
    assert result['max_delta_second']==0
    assert result['max_positive_interaction']==0
    assert not result['synergy_eligible']


def test_joint_only_danger_has_both_leverage_and_interaction():
    result=pair_metrics(np.ones((2,2))/4,np.array([[0.,0.],[0.,1.]]),THRESHOLDS)
    assert result['C_i_increment']==result['C_j_increment']==0
    assert result['C_ij_increment']==result['joint_excess_over_single_sum']==1
    assert result['positive_interaction_mass']==.25
    assert result['synergy_eligible']
    assert select_pair([{'actor_ids':[1,2],'metrics':result}])['actor_ids']==[1,2]


def test_additive_risk_and_unsupported_corners_do_not_create_synergy():
    result=pair_metrics(np.ones((2,2))/4,np.array([[0.,.4],[.4,.8]]),THRESHOLDS)
    assert result['max_positive_interaction']==0
    assert not result['synergy_eligible']
    result=pair_metrics(np.diag([.5,.5]),np.array([[0.,0.],[0.,1.]]),THRESHOLDS)
    assert result['max_positive_interaction']==0
    assert not result['synergy_eligible']


def test_natural_second_baseline_is_conditional_not_independent():
    p=np.array([[.45,.05],[.05,.45]])
    r=np.array([[0.,1.],[0.,1.]])
    result=pair_metrics(p,r,THRESHOLDS)
    assert np.isclose(result['max_delta_second'],.9)
    assert result['max_positive_interaction']==0


def test_exact_marginal_preserves_natural_joint_and_other_partners():
    p=np.array([[.4,.1],[.2,.3]])
    units=[{'ids':[1,2],'p':p},{'ids':[3],'p':np.array([.3,.7])}]
    joint,same=pair_marginal(units,[2,1])
    np.testing.assert_array_equal(joint,p.T);assert same
    joint,same=pair_marginal(units,[2,3])
    np.testing.assert_allclose(joint,np.outer(p.sum(0),[.3,.7]));assert not same
    with pytest.raises(ValueError):pair_marginal(units,[2,9])


def test_cross_unit_candidates_and_old_pair_coverage():
    cars=[Vehicle(0,0,25,0),Vehicle(1,10,25,0),Vehicle(2,20,25,1),Vehicle(3,100,25,0)]
    road=SimpleNamespace(vehicles=cars,cav=cars[0])
    pairs=candidate_pairs(road,[{'ids':[1,3]},{'ids':[2]}],{'nearest_actors':2,'observation_m':115})
    assert pairs==[(1,2),(1,3)]


def test_bounded_motion_matches_native_kinematics():
    cars=[Vehicle(1,0,39.9,0),Vehicle(2,20,20.1,0)]
    expected=copy.deepcopy(cars);cfg={'speed_grid':[20.,40.,1.],'lane_change_duration_s':1.,'lane_width_m':4.}
    advance(expected,[2.,-4.],.2,cfg)
    x,v=bounded_motion(np.array([[0.,20.]]),np.array([[39.9,20.1]]),np.array([[2.,-4.]]),.2,20.,40.)
    np.testing.assert_allclose(x[0],[c.x for c in expected]);np.testing.assert_allclose(v[0],[c.v for c in expected])


def make_road():
    road=NativeExperiment.__new__(NativeExperiment)
    road.vehicles=[Vehicle(0,0.,30.,0),Vehicle(1,18.,26.,0),Vehicle(2,35.,24.,0)]
    road.cav=road.vehicles[0]
    road.config={'cav':{'accel':2.,'decel':4.,'emergency_decel':9.,'tau':.2,'min_gap':2.,'desired_speed':40.}}
    road.model=SimpleNamespace(a=np.linspace(-4.,2.,31),config={'speed_grid':[20.,40.,1.],
        'lane_change_duration_s':1.,'lane_width_m':4.},summary={'idm':{'parameters':[2.,30.,4.,4.,2.,1.2]}})
    return road


def test_first_second_proxy_matches_native_and_never_mutates_state():
    road=make_road();before=copy.deepcopy([vars(c) for c in road.vehicles])
    pmfs={c.id:np.eye(33)[21] for c in road.vehicles}
    cfg={'step_s':1/15,'horizon_s':1.,'clearance_scale_m':2.}
    reactive=response_risk(road,[1,2],pmfs,cfg,True)
    means=response_risk(road,[1,2],pmfs,cfg,False)
    np.testing.assert_allclose(reactive,means)
    assert before==[vars(c) for c in road.vehicles]
    road.action_pmfs=lambda:pmfs;road.sample=lambda *args:{1:1,2:21}
    road.time=0.;road.diagnostics=Counter();road.minimum_ttc=np.inf;road.minimum_gap=np.inf
    road.crashes=[];road.collision_context=[]
    road.step_policy('natural',1.)
    assert np.isclose(reactive[1,21],np.exp(-max(road.minimum_gap,0)/2),atol=1e-12)


def test_response_continuation_changes_joint_matrix():
    road=make_road();pmfs={c.id:np.eye(33)[21] for c in road.vehicles}
    cfg={'step_s':.2,'horizon_s':4.,'clearance_scale_m':2.}
    a=response_risk(road,[1,2],pmfs,cfg,True);b=response_risk(road,[1,2],pmfs,cfg,False)
    assert np.max(np.abs(a-b))>.01
    assert np.isfinite(a).all() and a.min()>=0 and a.max()<=1


def test_pool_filter_is_predraw_geometry_and_supported_cut_in_only():
    road=make_road();pmfs={c.id:np.eye(33)[21] for c in road.vehicles}
    cfg={'closing_rate_mps':4.,'closing_gap_m':40.,'cut_in_distance_m':30.,
         'cut_in_probability':.001,'gap_m':10.,'ttc_s':4.}
    features=state_features(road,pmfs,cfg)
    assert features['stratum']=='high_risk'
    assert 'low_ttc' in features['reasons']
    assert 'cut_in_opportunity' not in features['reasons']


def test_invalid_probability_or_risk_is_rejected():
    for p,r in [(np.ones((2,2)),np.zeros((2,2))),
                (np.ones((2,2))/4,np.full((2,2),np.nan))]:
        with pytest.raises(ValueError):pair_metrics(p,r,THRESHOLDS)


def test_forcing_cross_unit_actors_preserves_unforced_conditional_partners():
    units=[{'ids':[1,2],'p':[[.4,.1],[.1,.4]]},{'ids':[3],'p':[.5,.5]}]
    rng=np.random.default_rng(71)
    draws=[conditional_actions(units,{1:0,3:1},rng) for _ in range(4000)]
    assert all(d[1]==0 and d[3]==1 for d in draws)
    assert abs(np.mean([d[2]==0 for d in draws])-.8)<.03
    draws=[conditional_actions(units,{2:1},rng) for _ in range(4000)]
    assert abs(np.mean([d[1]==1 for d in draws])-.8)<.03


def test_forced_zero_probability_corner_is_rejected():
    units=[{'ids':[1,2],'p':[[.5,0.],[0.,.5]]}]
    with pytest.raises(ValueError):conditional_actions(units,{1:0,2:1},np.random.default_rng(1))


def test_prepared_snapshot_replays_frozen_native_continuation():
    from scenario_reconstruction.highd_v46_assets import ensure_assets,read,ROOT
    from scenario_reconstruction.highd_paper_discrete_pairs import PairModelQueries,SINGLE_OPTIONS
    from scenario_reconstruction.highd_v46_pair_synergy import snapshot
    from scenario_reconstruction.highd_v46_pair_witness import continue_state
    archive=ROOT/'source_data/highd_paper_v46/runtime.zip'
    if not archive.exists() or archive.stat().st_size<1000:
        pytest.skip('Native integration requires the documented V46 LFS archive')
    assets,_=ensure_assets();model=PairModelQueries(assets/'model',assets/'lateral',**SINGLE_OPTIONS)
    cfg=read(ROOT/'configs/highd_v46_dual_context_v2.json');saved=[]
    class Observed(NativeExperiment):
        def choose(self,units,actors,pmfs,mode):
            if self.time==5.:
                saved.append(json.loads(json.dumps(snapshot(self,units,pmfs,120013,{}))))
            return None
    road=Observed(model,assets/'pair',assets/'lateral',120013,cfg)
    while road.time<5.:assert road.step_policy('natural',1.)
    road.minimum_gap=np.inf
    while road.time<9. and not road.crashes:road.step_policy('natural',1.)
    assert len(saved)==1
    restored=continue_state(model,assets,cfg,saved[0],{},None,4,2)
    assert np.isclose(restored['clearance_risk'],np.exp(-max(road.minimum_gap,0)/2),atol=1e-12,rtol=0)
    assert restored['cav_first_collision']==any(road.cav.id in p for p in road.crashes)
