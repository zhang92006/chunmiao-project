import pytest

from scenario_reconstruction.highd_v47_state_moments_v2_audit import (
    checked_moments, precision_record,
)


def state(j=1., bounds=(.6, 1.4)):
    return dict(state_id=0, J=j, bank_simultaneous_interval=bounds,
                relative_empirical_MCSE=0., relative_batch_gap=0.)


def test_zero_events_and_batch_agreement_do_not_certify_precision():
    s = state(0., (0., 1.))
    assert not precision_record(s)['precision_supported']
    assert not precision_record(state(1., (.001, 9.)))['precision_supported']
    with pytest.raises(ValueError, match='retain target'):
        checked_moments(dict(complete=True, arms={'equal': {'states': [s]}}), 'equal')


def test_full_bank_is_not_filtered_to_precise_states():
    summary = dict(complete=True, arms={'equal': {'states': [state(), state(0., (0., 2.))]}})
    with pytest.raises(ValueError):
        checked_moments(summary, 'equal')
    summary['arms']['equal']['states'] = [state()]
    assert checked_moments(summary, 'equal').tolist() == [1.]
    summary['complete'] = False
    with pytest.raises(ValueError, match='Incomplete'):
        checked_moments(summary, 'equal')


@pytest.mark.parametrize('bounds', [(1., 0.), (-1., 1.), (0., float('inf'))])
def test_invalid_uncertainty_is_rejected(bounds):
    with pytest.raises(ValueError):
        precision_record(state(bounds=bounds))
