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
    sampler.records = []
    sampler.maximum_marginal_error = 0.
    actions = sampler.draw(cars, {c.id: np.ones(33)/33 for c in cars}, np.random.default_rng(7))
    assert sampler.records[-1]['pairs'] == [[1, 2]]
    assert ('following', 0, 1) in sampler.records[-1]['removed']
    assert sorted(actions) == [0, 1, 2]
    assert sorted(i for unit in sampler.records[-1]['units'] for i in unit['actor_ids']) == [0, 1, 2]
    assert sampler.maximum_marginal_error < 1e-12
