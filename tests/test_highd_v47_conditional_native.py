import copy
from types import SimpleNamespace

import numpy as np
import pytest

from scenario_reconstruction.highd_paper_dynamics import Vehicle
from scenario_reconstruction.highd_v46_d2rl import NativeExperiment
from scenario_reconstruction.highd_v47_conditional_native import terminal_decisions


def toy_road(cars):
    road = NativeExperiment.__new__(NativeExperiment)
    road.vehicles = copy.deepcopy(cars); road.cav = road.vehicles[0]
    road.model = SimpleNamespace(a=np.linspace(-4, 2, 31),
        config=dict(speed_grid=[20., 40., 1.], lane_change_duration_s=1., lane_width_m=4.))
    road.config = dict(cav=dict(accel=2., decel=4., emergency_decel=9., tau=.2, min_gap=2., desired_speed=40.))
    road.time = 10.; road.crashes = []; road.diagnostics = dict(lane_change_onsets=0)
    road.minimum_gap = road.minimum_ttc = np.inf
    road.action_pmfs = lambda: {}
    return road


def scalar(road, actions):
    clone = copy.deepcopy(road)
    clone.sample = lambda *unused: dict(zip([c.id for c in clone.vehicles[1:]], actions))
    clone.step_policy('natural', 1.)
    return clone


def assert_native_batch(road, codes):
    original = [vars(c).copy() for c in road.vehicles]
    result = terminal_decisions(road, codes)
    for i, action in enumerate(codes):
        native = scalar(road, action)
        assert result['collision_pairs'][i] == native.crashes
        assert bool(result['event'][i]) == any(native.cav.id in p for p in native.crashes)
        assert bool(result['competing'][i]) == (bool(native.crashes) and not result['event'][i])
        assert result['elapsed_s'][i] == pytest.approx(native.time-road.time, abs=1e-12)
        for field in ('x', 'v', 'y', 'elapsed'):
            np.testing.assert_allclose(result[field][i], [getattr(c, field) for c in native.vehicles], rtol=1e-13, atol=1e-12)
        for field in ('lane', 'target'):
            np.testing.assert_array_equal(result[field][i], [getattr(c, field) for c in native.vehicles])
    assert [vars(c) for c in road.vehicles] == original
    return result


def test_vectorized_last_decision_matches_960_scalar_native_paths():
    rng = np.random.default_rng(471901)
    totals = dict(event=0, competing=0, negative=0)
    for k in range(80):
        n = int(rng.integers(3, 10)); cars = []
        for i in range(n):
            lane = int(rng.integers(0, 3)); x = float(rng.uniform(-40, 40))
            cars.append(Vehicle(id=i, x=x, v=float(rng.uniform(20, 40)), lane=lane, y=4.*lane,
                                width=float(rng.uniform(1.8, 2.4)), length=float(rng.uniform(4, 6))))
            if i and rng.random() < .3:
                cars[-1].target = (lane+1) % 3; cars[-1].elapsed = float(rng.uniform(0, .99))
                z = cars[-1].elapsed; cars[-1].y = (lane+(cars[-1].target-lane)*(10*z**3-15*z**4+6*z**5))*4
        road = toy_road(cars); codes = rng.integers(0, 33, size=(12, n-1))
        result = assert_native_batch(road, codes)
        totals['event'] += result['event'].sum(); totals['competing'] += result['competing'].sum()
        totals['negative'] += (~(result['event'] | result['competing'])).sum()
    assert all(v > 0 for v in totals.values())


def test_speed_bounds_and_simultaneous_competing_endpoint():
    road = toy_road([Vehicle(0, 0., 0., 0, y=0.), Vehicle(1, 3., 20., 0, y=0.),
                     Vehicle(2, 3.1, 40., 0, y=0.), Vehicle(3, -30., 40., 2, y=8.)])
    result = assert_native_batch(road, np.array([[1, 31, 0], [32, 1, 31], [1, 1, 1]]))
    assert result['event'].any()


def test_near_collision_boundary_reverts_to_original_scalar_execution():
    road=toy_road([Vehicle(0,-200.,20.,0,y=0.),Vehicle(1,0.,20.,0,y=0.),
                   Vehicle(2,3.,20.,1,y=2.+1e-9)])
    result=assert_native_batch(road,np.array([[21,21],[1,21]]))
    assert result['scalar_boundary_replay'].any()


@pytest.mark.parametrize('codes', [np.array([[1.2, 1.]]), np.array([[33, 1]]), np.array([[1]]), np.empty((0, 2), int)])
def test_reject_incomplete_or_invalid_action_batches(codes):
    road = toy_road([Vehicle(0, 0, 20, 0), Vehicle(1, 30, 20, 0), Vehicle(2, -30, 20, 0)])
    with pytest.raises(ValueError): terminal_decisions(road, codes)
