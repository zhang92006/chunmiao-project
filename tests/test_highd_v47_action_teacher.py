import numpy as np
import pytest
from scenario_reconstruction.highd_v47_action_teacher import exploration_h, union_cases
from scenario_reconstruction.highd_v46_cross_unit import proposal_factors


def test_uniform_bins_not_natural_probability_weighted():
    p = np.array([[.8, .19], [.01, 0.]])
    h, bins = exploration_h(p, np.array([[0., .1], [1., .5]]), [0, .2, .4, .6, .8, 1])
    np.testing.assert_allclose(h, [[.25, .25], [.5, 0]])
    assert bins[1, 0] == 4
    q1, q2 = proposal_factors(p, h, [.1, .1])[-2:]
    q = q1[:, None]*q2
    assert np.all(q >= .01*p) and q[1, 1] == 0
    assert np.isclose(q.sum(), 1) and np.isclose(np.sum(q[p > 0]*p[p > 0]/q[p > 0]), 1)


def test_single_bin_and_tiny_natural_support():
    p = np.array([[1-1e-30, 1e-30]])
    h, _ = exploration_h(p, np.zeros_like(p), [0, .2, 1])
    np.testing.assert_allclose(h, [[.5, .5]])


@pytest.mark.parametrize('risk', [np.array([[np.nan]]), np.array([[-.1]]), np.array([[1.1]])])
def test_reject_invalid_risk(risk):
    with pytest.raises(ValueError):
        exploration_h(np.ones((1, 1)), risk, [0, .2, 1])


def test_union_retains_negatives_distinct_states_and_pairs():
    a = {'state': {'seed': 1, 't': 1}, 'row': {'actor_ids': [1, 2]}, 'event': False}
    b = {'state': {'seed': 1, 't': 2}, 'row': {'actor_ids': [1, 2]}, 'event': False}
    c = {'state': a['state'], 'row': {'actor_ids': [2, 3]}, 'event': True}
    result = union_cases([a, b], [a, c])
    assert len(result) == 3 and result[0]['state'] == a['state']
