import copy
import unittest

from scenario_reconstruction.highd_original_ndd import longitudinal_observation


class GapAlignmentTests(unittest.TestCase):
    def test_truck_front_positions_are_converted_to_measured_gap(self):
        obs = {'Ego': {'position': [100,42]},
               'Lead': {'position': [119.88,42], 'distance': 3},
               'RightLead': {'position': [120,46], 'distance': 15}}
        saved=copy.deepcopy(obs)
        corrected=longitudinal_observation(obs, 5)
        self.assertAlmostEqual(corrected['Lead']['position'][0]-corrected['Ego']['position'][0]-5,3)
        self.assertEqual(obs,saved)
        self.assertEqual(corrected['RightLead'],obs['RightLead'])

    def test_equal_length_and_free_flow_unchanged(self):
        for obs in [{'Ego':{'position':[100,42]},'Lead':None},
                    {'Ego':{'position':[100,42]},'Lead':{'position':[108,42],'distance':3}}]:
            self.assertEqual(longitudinal_observation(obs,5),obs)

    def test_negative_gap_retained_and_nan_rejected(self):
        obs={'Ego':{'position':[100,42]},'Lead':{'position':[104,42],'distance':-1}}
        self.assertEqual(longitudinal_observation(obs,5)['Lead']['position'][0],104)
        obs['Lead']['distance']=float('nan')
        with self.assertRaises(ValueError):
            longitudinal_observation(obs,5)
