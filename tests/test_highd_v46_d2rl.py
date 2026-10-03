"""Small checks for the new intervention budget and exact P/Q factors."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np

from scenario_reconstruction.highd_v46_d2rl import NativeExperiment
from d2rl_training.highd_v46_sequence_env import log_weight


class ProposalTests(unittest.TestCase):
    def draw(self,mode,epsilon,seed):
        p=np.zeros((33,33));p[20:22,20:22]=[[.6,.1],[.1,.2]]
        h=np.zeros_like(p);h[21,21]=1.
        road=NativeExperiment.__new__(NativeExperiment)
        road.cav=SimpleNamespace(id=0);road.vehicles=[road.cav,SimpleNamespace(id=1),SimpleNamespace(id=2)]
        road.config={'epsilon_min':.05};road.rng=np.random.default_rng(seed)
        road.steps=[];road.log_weight=0.;road.max_intervened=0;road.time=0.
        unit={'ids':[1,2],'p':p}
        road.prepare=lambda pmfs:([unit],{})
        selected=[2] if mode=='single' else [1,2]
        road.choose=lambda *args:{'unit':unit,'ids':selected,'h':h.sum(axis=0) if mode=='single' else h,'mass':.2}
        with patch('scenario_reconstruction.highd_v46_d2rl.observation',return_value=[0.]*14):
            actions=road.sample({},mode,epsilon)
        return p,h,road,actions

    def test_single_changes_one_factor_even_when_selected_bv_is_second(self):
        for seed in range(8):
            p,h,road,actions=self.draw('single',.3,seed)
            step=road.steps[0];self.assertEqual(step['actor_ids'],[2]);self.assertEqual(road.max_intervened,1)
            self.assertEqual(step['natural_unit_ids'],[2,1]);self.assertEqual(len(step['p_factors']),1)
            j=actions[2];expected=p[:,j].sum()/(.3*p[:,j].sum()+.7*h[:,j].sum())
            self.assertAlmostEqual(np.exp(road.log_weight),expected)
            self.assertAlmostEqual(log_weight(step,[.3]),road.log_weight)

    def test_dual_conditional_product_and_zero_critical_row(self):
        for seed in range(8):
            p,h,road,actions=self.draw('dual',.3,seed);i,j=actions[1],actions[2]
            step=road.steps[0];self.assertEqual(road.max_intervened,2)
            p1=p.sum(axis=1);h1=h.sum(axis=1)
            h2=h[i]/h1[i] if h1[i]>0 else p[i]/p1[i]
            q=(.3*p1[i]+.7*h1[i])*(.3*p[i,j]/p1[i]+.7*h2[j])
            self.assertAlmostEqual(np.prod(step['q_factors']),q)
            self.assertAlmostEqual(np.prod(step['p_factors']),p[i,j])
            self.assertAlmostEqual(np.exp(road.log_weight),p[i,j]/q)

    def test_epsilon_one_has_unit_importance_weight(self):
        for mode in ('single','dual'):
            _,_,road,_=self.draw(mode,1.,7)
            self.assertEqual(road.log_weight,0.)


if __name__=='__main__':unittest.main()
