import copy
import numpy as np
import pytest

from scenario_reconstruction import highd_v47_allocation_certificate_audit as a
from scenario_reconstruction import highd_v47_allocation_certificate as m
from test_highd_v47_allocation_certificate import fixture


def natural():
    return dict(steps=[dict(initial=True,time_s=1.,all_action_indices={'1':2,'2':3},
        units=[dict(ids=[1,2],actions=[2,3],probability=.1)],full_log_p=np.log(.1))],
        action_indices=[2,3],future_weight=1.,elapsed_s=.2,collision_pairs=[[0,1]],
        event_result=True,competing_result=False,measurement_value=1.)


def test_independent_natural_density_and_competing_endpoint_checks():
    r=natural();s=dict(time_s=1.,cav_id=0)
    assert a.reconstruct_natural(r,s,[1,2])==1
    r['collision_pairs']=[[1,2]];r['event_result']=False;r['measurement_value']=0.;r['competing_result']=True
    assert a.reconstruct_natural(r,s,[1,2])==0
    r['steps'][0]['full_log_p']=0.
    with pytest.raises(AssertionError):a.reconstruct_natural(r,s,[1,2])


def test_prepared_initial_natural_law_and_conditional_cancellation_reconstructed():
    p=np.zeros((5,5));p[2,3]=.1;p[1,1]=.9
    s=dict(time_s=1.,cav_id=0,units=[dict(ids=[1,2],p=p)])
    r=natural();r['initial_cancelled_log_conditional']=0.
    assert a.reconstruct_natural(r,s,[1,2])==1
    r['initial_cancelled_log_conditional']=.1
    with pytest.raises(AssertionError):a.reconstruct_natural(r,s,[1,2])


@pytest.mark.parametrize('field,value',[('future_weight',2),('measurement_value',0.),('event_result',False),
    ('action_indices',[2,4]),('elapsed_s',5),('competing_result',True)])
def test_corrupt_natural_record_is_rejected(field,value):
    r=natural();r[field]=value
    with pytest.raises(AssertionError):a.reconstruct_natural(r,dict(time_s=1.,cav_id=0),[1,2])


def test_objective_reconstruction_agrees_with_primary_without_reusing_its_bounds():
    old,pilot,cfg=fixture();d=m.fit_and_design(old,pilot,cfg);records={}
    for b in (0,1):
        for s in d['states']:
            rows=[]
            for h in s['strata']:
                rows.extend(dict(stratum_id=h['stratum_id'],within_stratum_draw=n,
                    measurement_value=float(n%2) if s['side']=='loss' else .2+.1*(n%2)) for n in range(h['draws']))
            records[(b,s['state_id'])]=dict(rows=rows)
    x=m.summarize(d,records);y=a.independent_objective(d,records)
    for b in (0,1):
        assert y[b]['difference_upper']==pytest.approx(x['batches'][b]['M2_difference_upper'])
        assert y[b]['difference_MCSE']==pytest.approx(x['batches'][b]['difference_empirical_MCSE'])
        assert y[b]['target_M2']==pytest.approx(x['batches'][b]['target_M2'])
    broken=copy.deepcopy(records);broken[(0,0)]['rows'][0]['within_stratum_draw']=1
    with pytest.raises(AssertionError):a.independent_objective(d,broken)
