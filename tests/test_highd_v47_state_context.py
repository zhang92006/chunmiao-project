import copy

import numpy as np
import pytest

from scenario_reconstruction import highd_v47_state_context as m


def fixture():
    state = dict(seed=260000,time_s=20.,cav_id=5,vehicles=[
        dict(id=5,x=100.,v=30.,lane=1,length=5.,y=4.,target=-1),
        dict(id=1,x=88.,v=32.,lane=1,length=5.,y=4.,target=-1),
        dict(id=2,x=115.,v=28.,lane=1,length=5.,y=4.,target=-1),
        dict(id=3,x=125.,v=27.,lane=1,length=5.,y=4.,target=-1),
        dict(id=4,x=96.,v=29.,lane=0,length=5.,y=0.,target=1)])
    p=np.array([[.4,.1],[.2,.3]]); qa=np.array([[.1,.2],[.3,.4]]); r=np.array([[.1,.2],[.3,.9]])
    return state,p,qa,r


def test_actual_front_bumper_geometry_and_separate_chain():
    s,p,q,r=fixture(); f=m.state_features(s,[1,2],p,q,r); x=dict(zip(f['names'],f['values']))
    assert x['front_nearness']==pytest.approx(1/(1+10/20))
    assert x['front_closing']==pytest.approx(.2)
    # The lane-entering actor is the immediate rear neighbor, not pair actor 1.
    assert x['rear_nearness']==1.
    assert x['pair_rear_nearness']==pytest.approx(1/(1+7/20))
    assert x['pair_rear_closing']==pytest.approx(.2)
    assert x['front_chain_nearness']==pytest.approx(1/(1+5/20))
    assert x['lane_entry_count']==.1


def test_identity_translation_order_pair_permutation_invariance():
    s,p,q,r=fixture(); f=m.state_features(s,[1,2],p,q,r)
    altered=copy.deepcopy(s); altered.update(seed=999999,time_s=59.,cav_id=50)
    for v in altered['vehicles']: v['id']*=10; v['x']+=1000.
    altered['vehicles'].reverse()
    assert f==m.state_features(altered,[10,20],p,q,r)
    transpose=m.state_features(s,[2,1],p.T,q.T,r.T)
    assert f['names']==transpose['names']
    assert np.allclose(f['values'],transpose['values'],rtol=1e-14,atol=1e-14)


def test_no_neighbors_finite_and_no_target_or_outcome_predictors():
    s,p,q,r=fixture(); s['vehicles']=s['vehicles'][:3]
    for v in s['vehicles'][1:]: v['x']+=1000.
    f=m.state_features(s,[1,2],p,q,r); x=dict(zip(f['names'],f['values']))
    assert len(f['names'])==40 and np.isfinite(f['values']).all()
    assert x['front_present']==x['rear_present']==x['front_chain_present']==0
    forbidden=('source','seed','actor','time','outcome','absolute_x')
    assert not any(any(v in name for v in forbidden) for name in f['names'])


@pytest.mark.parametrize('field,value',[('length',0),('x',float('nan')),('v',-1)])
def test_invalid_physical_context_rejected(field,value):
    s,p,q,r=fixture(); s['vehicles'][1][field]=value
    with pytest.raises(ValueError): m.state_features(s,[1,2],p,q,r)
