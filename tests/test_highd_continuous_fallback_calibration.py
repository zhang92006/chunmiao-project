import unittest
import numpy as np

from scenario_reconstruction.highd_continuous_fallback_calibration import (
    blend_probabilities, choose, score_all, support_weight,
)


class ContinuousFallbackTests(unittest.TestCase):
    def test_weight_and_blend_include_empty_state_reference(self):
        n=np.array([0,1,9])
        np.testing.assert_allclose(support_weight(n,3),[0,.25,.75])
        highd=np.tile([.8,.2],(3,1)); reference=np.tile([.1,.9],(3,1))
        blend=blend_probabilities(highd,reference,n,3)
        np.testing.assert_allclose(blend[0],reference[0])
        np.testing.assert_allclose(blend.sum(axis=-1),1)
        for tau in [0,-1,float('nan'),float('inf')]:
            with self.assertRaises(ValueError): support_weight(n,tau)

    def test_score_reports_zeros_without_flooring_and_brier_uses_all(self):
        counts=np.array([[2,1],[0,3]])
        probability=np.array([[.5,.5],[1.,0.]])
        result=score_all(counts,probability)
        self.assertEqual(result['observations'],6)
        self.assertEqual(result['zero_probability_observations'],3)
        self.assertEqual(result['positive_support_observations'],3)
        self.assertAlmostEqual(result['nll_on_positive_support'],np.log(2))
        self.assertIsNotNone(result['brier_all_observations'])

    def test_selection_prioritizes_support_then_brier(self):
        def metric(nll,brier,zero=0):
            return {'nll_on_positive_support':nll,'brier_all_observations':brier,
                    'zero_probability_observations':zero}
        base={'name':'base','overall':metric(2,1,2),'tail':metric(3,1,1)}
        bad={'name':'bad','overall':metric(1,.9,0),'tail':metric(4,1.1,0)}
        good={'name':'good','overall':metric(1.5,.9,0),'tail':metric(2.5,.9,0)}
        selected,status=choose([base,bad,good],True)
        self.assertIs(selected,good)
        self.assertEqual(status,'calibration_screen_only')
        self.assertIsNone(choose([base,good],False)[0])

    def test_highd_parent_distribution_can_cover_empty_fine_cells(self):
        observed=np.array([[0,2],[1,0]])
        parent=np.array([[.25,.75],[.75,.25]])
        result=score_all(observed,parent)
        self.assertEqual(result['zero_probability_observations'],0)
        self.assertEqual(result['positive_support_observations'],3)
