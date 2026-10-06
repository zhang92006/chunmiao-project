import numpy as np
import pytest

from scenario_reconstruction import highd_v47_branching_moments as m
from scenario_reconstruction.highd_v47_branching_moments_audit import reconstruct_estimate


@pytest.mark.parametrize('arm', ['single', 'branching'])
def test_independent_root_variance_reconstruction(arm):
    hs=[dict(stratum_id=0,draws=3,coefficient_mass=.5)]
    rs=[dict(stratum_id=0,within_stratum_draw=i,single_value=x,branching_value=x/2,
             removed_conditional_variance_unbiased=.001) for i,x in enumerate([0.,.2,1.])]
    a=reconstruct_estimate(hs,rs,arm,.01); b=m.estimate(hs,rs,arm,.01)
    np.testing.assert_allclose(a[:2],[b['J'],b['estimator_variance']])
    np.testing.assert_allclose(a[2],b['bank_simultaneous_interval'])
    assert a[3]==pytest.approx(b['removed_conditional_variance_unbiased'])
