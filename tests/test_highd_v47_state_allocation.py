import numpy as np
import pytest
from scenario_reconstruction.highd_v47_state_allocation import state_allocation, schedule, summarize


def test_optimal_sqrt_allocation_and_floor():
    p = np.array([.25, .75]); m = np.array([100., 1.])
    q, _ = state_allocation(p, m, .2, 0)
    np.testing.assert_allclose(q, p*np.sqrt(m)/np.sum(p*np.sqrt(m)))
    assert np.sum(p*p*m/q) < np.sum(p*m)
    q, _ = state_allocation(p, [1, 0], .2, 0)
    np.testing.assert_allclose(q, [.85, .15])


def test_full_likelihood_exact_enumeration():
    ps = np.array([.2, .8]); qs = np.array([.7, .3])
    pa = np.array([[.9, .1], [.4, .6]]); qa = np.array([[.2, .8], [.8, .2]])
    f = np.array([[.1, .5], [.2, .9]])
    w = ps[:, None]*pa/(qs[:, None]*qa)
    assert np.isclose(np.sum(qs[:, None]*qa*w), 1)
    assert np.isclose(np.sum(qs[:, None]*qa*f*w), np.sum(ps[:, None]*pa*f))
    conditional_m2 = np.sum(pa*pa*f/qa, axis=1)
    assert np.isclose(np.sum(qs[:, None]*qa*f*w*w), np.sum(ps*ps*conditional_m2/qs))
    assert not np.isclose(np.sum(qs[:, None]*qa*f*pa/qa), np.sum(ps[:, None]*pa*f))


def test_schedule_is_fixed_and_stratified_complete():
    a = [.25, .75]; q = [.8, .2]
    s = schedule(a, q, 2000, 71)
    assert s == schedule(a, q, 2000, 71)
    assert np.bincount(s['equal_state_stratified']).tolist() == [1000, 1000]
    assert abs(np.mean(np.array(s['allocated_iid']) == 0)-.8) < .04


def test_zero_pilot_keeps_target():
    q, _ = state_allocation([.2, .8], [0, 0])
    np.testing.assert_array_equal(q, [.2, .8])


@pytest.mark.parametrize('a,m', [([0, 1], [1, 1]), ([.5,.5], [1,-1]), ([.5,.5], [1,np.nan])])
def test_invalid_models(a, m):
    with pytest.raises(ValueError):
        state_allocation(a, m)


def test_stratified_variance_not_iid_pooled_variance():
    plans = [{'source_seed': 1, 'target_mass': .2}, {'source_seed': 2, 'target_mass': .8}]
    rows = [{'state_id': i, 'event_result': True, 'generation_log_weight': 0.,
             'total_log_weight': np.log(plans[i]['target_mass']/.5)} for i in [0,0,1,1]]
    r = summarize(rows, plans, 'equal_state_stratified')
    assert r['mean'] == 1 and r['estimator_variance'] == 0
    assert summarize(rows, plans, 'target_iid')['estimator_variance'] > 0
