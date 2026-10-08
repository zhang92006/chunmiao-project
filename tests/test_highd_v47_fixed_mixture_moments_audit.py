import numpy as np
import pytest

from scenario_reconstruction import highd_v47_fixed_mixture_moments as m
from scenario_reconstruction.highd_v47_fixed_mixture_moments_audit import reconstruct


def rows():
    return [dict(stratum_id=0, within_stratum_draw=i, fixed_tilted_component=i >= 2,
                 natural=dict(event_result=i % 2 == 1),
                 iid_defensive=dict(weighted_event=z), fixed_defensive=dict(weighted_event=z))
            for i, z in enumerate([0., .2, 1., 1.5])]


@pytest.mark.parametrize('arm', m.ARMS)
def test_independent_arithmetic_matches_correct_estimators(arm):
    hs = [dict(stratum_id=0, draws=4, coefficient_mass=.8)]
    x = reconstruct(hs, rows(), arm, .01); e = m.estimate(hs, rows(), arm, .01)
    np.testing.assert_allclose([x['J'], x['variance']], [e['J'], e['estimator_variance']])
    np.testing.assert_allclose(x['bounds'], e['bank_simultaneous_interval'])
    if arm == 'fixed_defensive':
        assert x['removed'] == pytest.approx(e['removed_between_component_variance_unbiased'])


def test_independent_audit_rejects_incorrect_fixed_component_counts():
    r = rows(); r[1]['fixed_tilted_component'] = True
    with pytest.raises(AssertionError): reconstruct([dict(stratum_id=0, draws=4, coefficient_mass=1.)], r, 'fixed_defensive', .01)
