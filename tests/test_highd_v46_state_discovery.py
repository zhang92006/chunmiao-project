from types import SimpleNamespace
import copy
import numpy as np
import pytest
from scenario_reconstruction.highd_v46_assets import read,ROOT
from scenario_reconstruction.highd_v46_d2rl import identity
from scenario_reconstruction.highd_v46_state_discovery import (
    screen_pairs,choose_bank,audit_candidates,audit_gates,validate_split,
)


def config():
    return read(ROOT/'configs/highd_v46_state_discovery_v1.json')


def test_fast_screen_finds_bracketing_pair_without_collision_labels():
    cars=[SimpleNamespace(id=i,x=x,v=25.,lane=0,length=5.) for i,x in enumerate([0.,-15.,15.,50.])]
    road=SimpleNamespace(cav=cars[0],vehicles=cars)
    pmfs={c.id:np.eye(33)[21] for c in cars}
    pairs=screen_pairs(road,pmfs,config()['screen'])
    assert pairs[0]['actor_ids']==[1,2] and pairs[0]['mechanism']=='bracket'
    assert pairs[0]['score']>pairs[-1]['score']
    # Geometry ranking has no random source or trajectory outcome input.
    assert pairs==screen_pairs(road,pmfs,config()['screen'])


def test_unsupported_lateral_actor_has_no_direct_pressure():
    cars=[SimpleNamespace(id=0,x=0.,v=25.,lane=0,length=5.),
          SimpleNamespace(id=1,x=-15.,v=30.,lane=0,length=5.),
          SimpleNamespace(id=2,x=15.,v=23.,lane=1,length=5.)]
    road=SimpleNamespace(cav=cars[0],vehicles=cars);pmfs={c.id:np.eye(33)[21] for c in cars}
    assert screen_pairs(road,pmfs,config()['screen'])[0]['score']==0.
    pmfs[2]=np.eye(33)[32]*.01+np.eye(33)[21]*.99
    assert screen_pairs(road,pmfs,config()['screen'])[0]['score']>0.


def test_one_second_bank_recovers_off_grid_peak_and_enforces_spacing():
    states=[{'time_s':float(t),'screen_score':1. if t==2 else .5 if t==7 else .1} for t in range(10)]
    cfg=config()['bank']
    assert choose_bank(states,'geometry_1s',cfg,np.random.default_rng(1))==[2,7]
    assert choose_bank(states,'geometry_5s',cfg,np.random.default_rng(1))==[0,5]
    a=choose_bank(states,'random_1s',cfg,np.random.default_rng(1))
    b=choose_bank(states,'random_1s',cfg,np.random.default_rng(1))
    assert a==b and len(a)==2 and a[1]-a[0]>=5
    for s in states:s['future_collision']=True
    assert choose_bank(states,'geometry_1s',cfg,np.random.default_rng(1))==[2,7]


def test_pair_audit_has_diversity_and_random_competitor_without_duplicates():
    rows=[{'actor_ids':[i,i+10],'score':1-i*.1,'mechanism':kind} for i,kind in enumerate(['bracket','bracket','lateral','same_side'])]
    selected=audit_candidates(rows,config()['audit'],np.random.default_rng(1))
    assert [r['mechanism'] for r in selected[:2]]==['bracket','lateral']
    assert selected[2]['selection']=='random_audit'
    assert len({tuple(r['actor_ids']) for r in selected})==3


def test_gate_funnel_separates_local_cooperation_from_global_maximum_excess():
    p=np.ones((3,3))/9;r=np.zeros((3,3));r[2,0]=r[0,2]=.9;r[1,1]=1.
    result=audit_gates(p,r,config()['selection'])
    assert not result['views']['production_strict']
    assert result['first_failed_gate']=='global_excess'
    assert result['views']['drop_global_excess']
    assert result['views']['local_interaction_actual_H']


def test_gate_funnel_rejects_pure_single_actor_risk():
    result=audit_gates(np.ones((2,2))/4,np.array([[0.,0.],[1.,1.]]),config()['selection'])
    assert not any(result['views'].values())
    assert result['first_failed_gate']=='leverage'


def test_split_lock_prevents_source_episode_leakage_and_post_train_changes():
    cfg=config();deps={'proxy':'hash'}
    lock={'split':'train','config_identity':identity(cfg),'implementation_sha256':'impl','dependencies':deps}
    validate_split(cfg,'validation',lock,'impl',deps)
    with pytest.raises(ValueError):validate_split(cfg,'validation',None,'impl',deps)
    with pytest.raises(ValueError):validate_split(cfg,'validation',lock,'changed',deps)
    with pytest.raises(ValueError):validate_split(cfg,'validation',lock,'impl',{'proxy':'changed'})
    changed=copy.deepcopy(cfg);changed['screen']['gap_scale_m']=25
    with pytest.raises(ValueError):validate_split(changed,'validation',lock,'impl',deps)
    changed=copy.deepcopy(cfg);changed['validation']=changed['train'].copy()
    with pytest.raises(ValueError):validate_split(changed,'train')
    changed=copy.deepcopy(cfg);changed['train']['seed_start']=100000
    with pytest.raises(ValueError):validate_split(changed,'train')


def test_native_witness_selection_is_episode_diverse_and_respects_remaining_time():
    from scenario_reconstruction.highd_v46_discovery_witness import select_witnesses
    cfg=read(ROOT/'configs/highd_v46_discovery_witness_v1.json')
    rows=[{'seed':s,'time_s':t,'actor_ids':[1,2],'methods':['geometry_1s'],
           'views':{'local_interaction_actual_H':True},'selection':'ranked',
           'metrics':{'positive_interaction_mass':m}}
          for s,t,m in [(1,59,10.),(1,20,1.),(1,30,.9),(2,40,.5),(3,40,.4)]]
    selected=select_witnesses(rows,cfg)
    assert [(r['seed'],r['time_s']) for r in selected]==[(1,20),(2,40)]
