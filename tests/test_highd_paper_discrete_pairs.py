from collections import Counter
from types import SimpleNamespace

import numpy as np

from scenario_reconstruction.highd_paper_discrete import DiscreteRoad
from scenario_reconstruction.highd_paper_dynamics import Vehicle
from scenario_reconstruction.highd_paper_discrete_pairs import DiscretePairs


def test_pair_order_does_not_reassign_logged_acceleration():
    road = DiscreteRoad.__new__(DiscreteRoad)
    road.vehicles = [Vehicle(0, 0., 30., 0), Vehicle(1, 100., 30., 0)]
    road.model = SimpleNamespace(a=np.linspace(-4., 2., 31), config={
        'speed_grid': [20., 40.], 'lane_change_duration_s': 1., 'lane_width_m': 4.})
    road.pair_sampler = SimpleNamespace(draw=lambda *args: {1: 31, 0: 1})
    road.action_pmfs = lambda: {}
    road.rng = np.random.default_rng(7)
    road.diagnostics = Counter()
    road.time = road.max_action_error = 0.
    road.decisions = []
    assert road.step()
    assert road.decisions[0]['accelerations'] == {0: -4., 1: 2.}
    np.testing.assert_allclose([c.v for c in road.vehicles], [26., 32.])
    np.testing.assert_allclose([c.x for c in road.vehicles], [28., 131.])


def test_unsupported_old_pair_does_not_block_supported_neighbor_pair():
    cars = [Vehicle(0, 0., 30., 0), Vehicle(1, 20., 30., 0), Vehicle(2, 50., 30., 0)]
    sampler = DiscretePairs.__new__(DiscretePairs)
    sampler.model = {'cells': {str(g): {'weights': [1., 0., 0., 0., 0., 0.],
        'independent_mass': .1, 'reason': 'calibrated'} for g in range(12)}}
    sampler.histories = {0: ((0, 1), 3.), 1: ((0, 2), 3.), 2: ((0, None), 3.)}
    sampler.previous = {('following', 0, 1)}
    sampler.direct = lambda car, front: car.id != 0
    sampler.mode = 'fitted'
    sampler.dependence_scale = 1.
    sampler.gap_taper = False
    sampler.supported_pairs_only = True
    sampler.pair_selection = 'continuity'
    sampler.records = []
    sampler.maximum_marginal_error = 0.
    actions = sampler.draw(cars, {c.id: np.ones(33)/33 for c in cars}, np.random.default_rng(7))
    assert sampler.records[-1]['pairs'] == [[1, 2]]
    assert ('following', 0, 1) in sampler.records[-1]['removed']
    assert sorted(actions) == [0, 1, 2]
    assert sorted(i for unit in sampler.records[-1]['units'] for i in unit['actor_ids']) == [0, 1, 2]
    assert sampler.maximum_marginal_error < 1e-12


def test_weighted_matching_chooses_two_edges_over_one_larger_edge():
    from scenario_reconstruction.highd_paper_pair_matching import maximum_weight_following
    edges = [{'actor_ids': (i, i+1), 'relation': 'following', 'weight': w}
             for i, w in enumerate([.9, 1., .9])]
    selected = maximum_weight_following(edges, {('following', 1, 2)})
    assert [c['actor_ids'] for c in selected] == [(0, 1), (2, 3)]


def test_interval_training_uses_forward_speed_and_excludes_truncated_tracks(monkeypatch):
    import pandas as pd
    from scenario_reconstruction import highd_paper_pair_calibration as calibration
    frames = np.arange(1, 151)
    t = (frames-1)/25
    rows = []
    for actor in (1, 2):
        rows.append(pd.DataFrame({'frame': frames, 'id': actor,
            'xVelocity': (30.+.4*t) if actor == 1 else (33.+.8*t),
            'yVelocity': 0., 'xAcceleration': 0., 'dhw': 20.+3*t+.2*t*t,
            'precedingXVelocity': 33.+.8*t, 'precedingId': 2 if actor == 1 else 0,
            'frontSightDistance': 200., 'laneId': 1}))
    tracks = pd.concat(rows, ignore_index=True)
    meta = pd.DataFrame({'id': [1, 2], 'drivingDirection': [2, 2], 'class': ['Car', 'Car']})
    monkeypatch.setattr(calibration, 'sha', lambda path: 'synthetic')
    monkeypatch.setattr(calibration.pd, 'read_csv', lambda path, **kw:
                        meta.copy() if path.name.endswith('tracksMeta.csv') else tracks.copy())
    model = SimpleNamespace(summary={'train_recordings': []}, config={'vehicle_classes': ['Car']},
        v=np.arange(20., 41.), g=np.array([20., 40.]), rr=np.array([-20., 0., 20.]), a=np.linspace(-4., 2., 31),
        query_rows_custom=lambda rows: np.ones((len(rows), 31))/31,
        arrays={'ff_counts': np.ones((21, 31)), 'ff_probability': np.ones((21, 31))/31,
                'cf_counts': np.ones((21, 2, 3, 31)), 'cf_probability': np.ones((21, 2, 3, 31))/31})
    instant, _ = calibration.samples(model, '.', '99', return_data=True, decision_hz=1)
    interval, info = calibration.samples(model, '.', '99', return_data=True, decision_hz=1, one_second_actions=True)
    assert len(interval['a']) > 0
    assert np.all(instant['a'] == 20)
    assert np.all(interval['a'] == 22)
    assert np.all(interval['b'] == 24)
    assert info['future_interval_excluded'] > 0
    assert len(interval['a']) < len(instant['a'])
