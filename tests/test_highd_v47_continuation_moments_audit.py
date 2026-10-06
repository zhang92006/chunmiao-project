import copy

import numpy as np
import pytest

from scenario_reconstruction.highd_v47_continuation_moments_audit import (
    reconstruct_estimate, reconstruct_path, same_draw_variance,
)


def path():
    p = [.2, .3]; t = [.4, .6]; conditional = -.7
    lp, lt = float(np.log(p).sum()+conditional), float(np.log(t).sum()+conditional)
    w = 1/(.5+.5*4)
    return dict(steps=[dict(actor_ids=[2, 1], p_factors=p, t_factors=t,
                     full_log_p=lp, full_log_t=lt, log_t_over_p=float(np.log(4.)),
                     cancelled_log_conditional=conditional)],
                initial_cancelled_log_conditional=-5., full_log_p=lp, full_log_t=lt,
                log_t_over_p=float(np.log(4.)), future_weight=w, weighted_event=w,
                event_result=True, bv_only_collision=False, collision_pairs=[[0, 1]])


def test_independent_path_reconstruction_includes_cancelled_initial_partners():
    assert reconstruct_path(path()) < 1e-12
    bad = copy.deepcopy(path()); bad['weighted_event'] *= bad['future_weight']
    with pytest.raises(AssertionError): reconstruct_path(bad)
    bad = copy.deepcopy(path()); bad['full_log_t'] += .1
    with pytest.raises(AssertionError): reconstruct_path(bad)


def test_independent_weighted_variance_and_interval_reconstruction():
    strata = [dict(stratum_id=0, coefficient_mass=2., draws=3)]
    z = [0., .4, .8]
    rows = [dict(stratum_id=0, natural=dict(event_result=bool(v)),
                 defensive=dict(weighted_event=v)) for v in z]
    j, v, bounds = reconstruct_estimate(strata, rows, 'defensive', .01)
    assert j == pytest.approx(.8)
    assert v == pytest.approx(4*np.var(z, ddof=1)/3)
    np.testing.assert_array_equal(bounds, [0., 2.])


def test_same_draw_true_variance_difference_recovers_exact_path_law():
    # P event probability .1; T event .7; G event .4.
    # Weights: .25 on event, 1.5 on negative. G Z^2=.025 vs P Y^2=.1.
    strata = [dict(stratum_id=0, coefficient_mass=1., draws=2)]
    delta = vn = vg = 0.
    for first, second in __import__('itertools').product([0., .25], repeat=2):
        probability = (.4 if first else .6)*(.4 if second else .6)
        rows = [dict(stratum_id=0, defensive=dict(weighted_event=z)) for z in (first, second)]
        c = same_draw_variance(strata, rows, .05)
        delta += probability*c['variance_difference']
        vn += probability*c['natural_variance_same_draws_unbiased']
        vg += probability*c['defensive_variance_same_draws']
        assert not c['certified_improvement']  # two observations cannot certify it
    assert vn == pytest.approx(.1*.9/2)
    assert vg == pytest.approx((.025-.1**2)/2)
    assert delta == pytest.approx((.025-.1)/2)
