import unittest

import numpy as np
import pandas as pd

from scenario_reconstruction.highd_lane_change_longitudinal_audit import window_metrics, describe_events


class MotionAuditTests(unittest.TestCase):
    def window(self, sign=1):
        return pd.DataFrame({'frame': np.arange(100,126),
            'xVelocity': sign*(30-np.arange(26)/25), 'xAcceleration': np.full(26, -sign)})

    def test_both_directions_measure_deceleration_not_signed_velocity(self):
        for sign,direction in [(1,2),(-1,1)]:
            m = window_metrics(self.window(sign),100,25,1,direction)
            self.assertAlmostEqual(m['mean_acceleration_from_speed_mps2'], -1)
            self.assertAlmostEqual(m['mean_recorded_acceleration_mps2'], -1)
            self.assertAlmostEqual(m['zero_acceleration_endpoint_speed_error_mps'], 1)

    def test_incomplete_and_nonfinite_windows_rejected(self):
        with self.assertRaisesRegex(ValueError,'incomplete'):
            window_metrics(self.window().iloc[:-1],100,25,1,2)
        with self.assertRaisesRegex(ValueError,'incomplete'):
            window_metrics(self.window().drop(index=5),100,25,1,2)
        window=self.window()
        window.loc[1,'xAcceleration']=float('nan')
        with self.assertRaisesRegex(ValueError,'nonfinite'):
            window_metrics(window,100,25,1,2)
        with self.assertRaisesRegex(ValueError,'direction'):
            window_metrics(self.window(-1),100,25,1,2)

    def test_summary_counts_events_and_reports_empty_cohort(self):
        m = window_metrics(self.window(),100,25,1,2)
        rows=[dict(m,recording='01',vehicle_id=1),dict(m,recording='01',vehicle_id=1)]
        report=describe_events(rows)
        self.assertEqual(report['event_count'],2)
        self.assertEqual(report['vehicle_count'],1)
        self.assertEqual(report['mean_deceleration_counts']['1'],2)
        self.assertEqual(describe_events([])['event_count'],0)
