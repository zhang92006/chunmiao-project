import unittest
import numpy as np

from scenario_reconstruction.highd_paper_dynamics import Vehicle, Road, advance
from scenario_reconstruction.highd_paper_data import histogram
from scenario_reconstruction.highd_v46_matched_runtime import state
from scenario_reconstruction.highd_v46_matched_score import score_states, state_counts, persistent_windows
from scenario_reconstruction.highd_v46_matched_report import compare, bootstrap_weights


class MatchedFidelityTests(unittest.TestCase):
    def test_observation_interpolation_handles_speed_bound_exactly(self):
        car = Vehicle(0, 0., 39.99, 0)
        trace = [state([car])]
        cfg = {'speed_grid': [20, 40], 'lane_change_duration_s': 1, 'lane_width_m': 4}
        for _ in range(3):
            advance([car], [2.], 1/15, cfg); trace.append(state([car]))
        times, scored = score_states(np.asarray(trace), np.array([[2.]]))
        self.assertEqual(times[1], .1)
        expected = 39.99*.005+.5*2*.005**2+40*.095
        self.assertAlmostEqual(scored[1, 0, 1], expected)
        self.assertEqual(scored[1, 0, 2], 40.)

    def test_vector_score_matches_frozen_neighbor_semantics(self):
        from types import SimpleNamespace
        road = Road.__new__(Road)
        road.model = SimpleNamespace(g=[0., 115.])
        road.vehicles = [Vehicle(0, 0, 30, 0), Vehicle(1, 20, 28, 0),
                         Vehicle(2, 50, 32, 1, target=0)]
        sample = road.sample(); bins = {'speed': [20,40,1], 'gap': [0,115,1], 'rr': [-20,20,1]}
        counts = state_counts(state(road.vehicles)[None], bins)
        for key in bins:
            np.testing.assert_array_equal(counts[key], histogram(sample[key], *bins[key]))

    def test_persistent_pair_requires_entire_interval_and_endpoint(self):
        cars = [Vehicle(0, 0, 30, 0), Vehicle(1, 20, 30, 0)]
        trace = np.repeat(state(cars)[None], 76, axis=0)
        self.assertEqual(persistent_windows(trace, 5)[0.], {(0,1)})
        self.assertEqual(persistent_windows(trace[:-1], 5), {})
        trace[30,1,5] = 1
        self.assertEqual(persistent_windows(trace, 5)[0.], set())

    def test_paired_bootstrap_has_zero_width_when_arms_identical(self):
        counts = np.array([[1,2,3], [4,2,1], [7,3,2]])
        report = compare(counts, counts, [2,3,4], 'hellinger', bootstrap_weights(3,100,7))
        self.assertEqual(report['mean_seed_delta_ci95'], [0.,0.])
        self.assertEqual(report['pooled_delta_ci95'], [0.,0.])


if __name__ == '__main__':
    unittest.main()
