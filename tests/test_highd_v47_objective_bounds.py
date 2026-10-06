from itertools import product

import numpy as np
import pytest

from scenario_reconstruction import highd_v47_objective_bounds as m


def test_non_iid_finite_laws_have_e_value_at_most_one_and_correct_coverage():
    # Enumerate heterogeneous root laws, not fake IID leaves.
    laws = [([0., .2], [.7, .3]), ([.1, .4], [.9, .1]), ([0., .6], [.5, .5])]
    mu = sum(np.dot(x, p) for x, p in laws); bounds = [.2, .4, .6]
    for rate in (.1, .9, 1.5):
        for error in (.01, .1, .5):
            eb = el = fail_b = fail_l = 0.
            for indices in product((0, 1), repeat=3):
                x = [laws[i][0][j] for i, j in enumerate(indices)]
                prob = np.prod([laws[i][1][j] for i, j in enumerate(indices)])
                eb += prob*np.exp(-rate*mu)*np.prod(1+rate*np.array(x))
                el += prob*np.exp(rate*mu)*np.prod(1-rate*np.array(x))
                fail_b += prob*(m.benefit_lower(x, rate, error)['lower'] > mu)
                fail_l += prob*(m.loss_upper(x, bounds, rate, error)['upper'] < mu)
            assert eb <= 1+1e-14 and el <= 1+1e-14
            assert fail_b <= error+1e-14 and fail_l <= error+1e-14


def test_large_benefit_does_not_need_a_tuned_observation_bound():
    result = m.benefit_lower([1e6]*100, 1., .01)
    assert 0 < result['lower'] < result['point']


def test_shared_leaves_cannot_be_treated_as_independent_roots():
    # Two copies of ONE Bernoulli prefix violate the e-value guarantee if
    # incorrectly multiplied as independent observations. Use one root value.
    assert np.exp(-.2)*(.9+.1*4) > 1
    assert np.exp(-.1)*(.9+.1*2) <= 1


def test_zero_observations_are_not_zero_true_loss():
    x = [0.]*100; b = [.001]*100; rate = .8/.001
    r = m.loss_upper(x, b, rate, .0125)
    assert r['upper'] == pytest.approx(np.log(80)/rate)
    assert r['upper'] > 0 and r['point'] == 0
    assert m.benefit_lower(x, 100., .0125)['lower'] == 0


def test_empty_objective_side_has_known_zero_bound():
    assert m.loss_upper([], [], .1, .05)['upper'] == 0
    assert m.benefit_lower([], .1, .05)['lower'] == 0


def test_objective_difference_and_family_error():
    benefit = m.benefit_lower([.01]*1000, 100., .0125)
    loss = m.loss_upper([0.]*100, [.001]*100, 800., .0125)
    r = m.objective_difference(benefit, loss)
    assert r['M2_difference_upper'] < 0 and r['improvement_certified']
    assert r['family_error'] == .025


@pytest.mark.parametrize('x,rate,error', [([-1], 1, .05), ([np.nan], 1, .05),
    ([1], 0, .05), ([1], np.inf, .05), ([1], 1, 0), ([[1]], 1, .05)])
def test_reject_invalid_nonnegative_bets(x, rate, error):
    with pytest.raises(ValueError): m.benefit_lower(x, rate, error)


@pytest.mark.parametrize('x,b,rate', [([.2], [.1], 1), ([0], [-1], 1),
    ([0], [1], 1), ([0], [np.inf], .1), ([0], [], 1)])
def test_loss_support_is_designed_not_observed(x, b, rate):
    with pytest.raises(ValueError): m.loss_upper(x, b, rate, .05)


def test_pilot_rate_is_deterministic_and_not_an_evaluation_bound():
    r = m.choose_pilot_bet([[0, .1]*100, [0, .05]*100], [1, 10, 100], .0125)
    assert r == m.choose_pilot_bet([[0, .1]*100, [0, .05]*100], [1, 10, 100], .0125)
    assert r['rate'] in (1, 10, 100)
    assert 'Training-only' in r['interpretation']
    with pytest.raises(ValueError): m.choose_pilot_bet([], [1, 10], .05)
    with pytest.raises(ValueError): m.choose_pilot_bet([[0]], [10, 1], .05)


def test_known_sign_decomposition_equals_same_target_moment_difference():
    alpha=np.array([.1,.2,.3,.4]); q=np.array([.11,.21,.38,.30]); j=np.array([.01,.02,.1,.3])
    d=alpha*(alpha/q-1)
    loss=np.sum(np.maximum(d,0)*j); benefit=np.sum(np.maximum(-d,0)*j)
    assert loss-benefit == pytest.approx(np.sum(alpha**2*j/q)-alpha@j)
    # Mean is the SAME, so the M2 difference also equals IID centered variance
    # difference multiplied by N. A source-law change would break this fact.
    mu=.001; n=100
    assert loss-benefit == pytest.approx(n*((np.sum(alpha**2*j/q)-mu**2)/n-(alpha@j-mu**2)/n))
