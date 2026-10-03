from collections import Counter
from types import SimpleNamespace

import numpy as np

from scenario_reconstruction.highd_paper_discrete import DiscreteRoad
from scenario_reconstruction.highd_paper_dynamics import Vehicle


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
