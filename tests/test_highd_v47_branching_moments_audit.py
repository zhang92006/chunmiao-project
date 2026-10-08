import numpy as np
import pytest

from scenario_reconstruction import highd_v47_branching_moments as m
from scenario_reconstruction.highd_v47_branching_moments_audit import reconstruct_estimate
from scenario_reconstruction.highd_v47_branching_moments_audit import root_values


@pytest.mark.parametrize('arm', ['single', 'branching'])
def test_independent_root_variance_reconstruction(arm):
    hs=[dict(stratum_id=0,draws=3,coefficient_mass=.5)]
    rs=[dict(stratum_id=0,within_stratum_draw=i,single_value=x,branching_value=x/2,
             removed_conditional_variance_unbiased=.001) for i,x in enumerate([0.,.2,1.])]
    a=reconstruct_estimate(hs,rs,arm,.01); b=m.estimate(hs,rs,arm,.01)
    np.testing.assert_allclose(a[:2],[b['J'],b['estimator_variance']])
    np.testing.assert_allclose(a[2],b['bank_simultaneous_interval'])
    assert a[3]==pytest.approx(b['removed_conditional_variance_unbiased'])


def root_fixture():
    step=dict(actor_ids=[1,2],p_factors=[.2,.3],t_factors=[.4,.6],
              full_log_p=float(np.log(.06)),full_log_t=float(np.log(.24)),
              cancelled_log_conditional=0.,log_t_over_p=float(np.log(4.)))
    def path(event):
        return dict(steps=[step],full_log_p=step['full_log_p'],full_log_t=step['full_log_t'],
                    log_t_over_p=step['log_t_over_p'],initial_cancelled_log_conditional=-1.,
                    future_weight=.4,weighted_event=.4*event,event_result=event,
                    bv_only_collision=False,collision_pairs=[[0,1]] if event else [],elapsed_s=2.)
    branches=[dict(component='P' if i<2 else 'T',fork_state_sha256='same',suffix_steps=0,path=path(i>=2)) for i in range(4)]
    return dict(prefix=path(False),terminal_prefix=False,branches=branches,
                posterior_natural_fraction=.2,selected_branch=2,control_uniform=.3,
                fork_state_sha256='same',single_value=.4,branching_value=.32,
                removed_conditional_variance_unbiased=.0256)


def test_independent_audit_checks_posterior_weight_and_root_mean():
    r=root_fixture(); single,branch,delta,error=root_values(r)
    np.testing.assert_allclose([single,branch,delta],[.4,.32,.0256])
    assert error<1e-12
    r['posterior_natural_fraction']=.5
    with pytest.raises(AssertionError):root_values(r)


def test_audit_rejects_squaring_or_double_counting_future_weight():
    r=root_fixture();r['branches'][2]['path']['weighted_event']*=.4
    with pytest.raises(AssertionError):root_values(r)
