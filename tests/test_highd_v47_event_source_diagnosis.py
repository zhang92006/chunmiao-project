import pytest
from scenario_reconstruction.highd_v47_event_source_diagnosis import select_rows


def row(seed, risk, t=1, ids=(1, 2), event=False):
    return {'seed': seed, 'ranker_features': [1, 0, risk, 0], 'time_s': t,
            'actor_ids': ids, 'event': event}


def test_selection_not_outcome_or_gate_based():
    rows = [row(1, .8, event=False), row(1, .6, event=True), row(2, .2)]
    out = select_rows(rows, [1, 2, 3], .5)
    assert out[0]['row_index'] == 0
    assert out[1]['maximum_risk'] == .2 and not out[1]['probe_selected']
    assert out[2]['row_index'] is None and not out[2]['probe_selected']
    for r in rows:
        r['event'] = not r['event']
    assert out == select_rows(rows, [1, 2, 3], .5)


def test_stable_ties_threshold_and_source_exclusion():
    rows = [row(1, .5, 2), row(1, .5, 1, (2, 3)), row(1, .5, 1), row(240000, 1)]
    out = select_rows(rows, [1], .5)
    assert len(out) == 1 and out[0]['row_index'] == 2 and out[0]['probe_selected']


@pytest.mark.parametrize('seeds,threshold', [([1, 1], .5), ([1], -1), ([1], 2)])
def test_invalid_plan(seeds, threshold):
    with pytest.raises(ValueError):
        select_rows([], seeds, threshold)
