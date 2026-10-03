"""Likelihood, causality and rare-event precision regressions for V46 v2."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np

from scenario_reconstruction.highd_v46_assets import digest,save
from scenario_reconstruction.highd_v46_contract import (
    CONTEXT_OBSERVATION, enrich_observation, independent_split, replay_observation,
)
from scenario_reconstruction.highd_v46_d2rl import NativeExperiment,experiment_target,identity
from scenario_reconstruction.highd_v46_statistics import summarize_values,convergence,compare_collections
from scenario_reconstruction.highd_v46_workflow import merge_collections
from scenario_reconstruction.highd_v46_fixed_baseline import select_fixed
from scenario_reconstruction.highd_v46_policy import NumpyPolicy
from d2rl_training.highd_v46_sequence_env import V46SequenceEnv,log_weight


def dataset(seed=1,positive=True):
    step={'p_factors':[.5,1.],'h_factors':[1.,1.],'q_factors':[.55,1.],
          'generation_epsilon':[.9,.9],'actor_ids':[1],'factor_mask':[1,0],
          'observation':enrich_observation([0.]*14,.1,.5,0.,0.,60.,[1,0])}
    weight=log_weight(step,[.9,.9])
    step2=copy.deepcopy(step);step2['observation'][16]=weight/30.
    seq={'seed':seed,'steps':[step,step2],'event_result':positive,'generation_log_weight':2*weight}
    return {'contract':'highd_v46_native_d2rl_sequence_v2','observation_contract':CONTEXT_OBSERVATION,
            'action_dimension':2,'intervention_mode':'dual','natural_target_sha256':'natural',
            'experiment_target_sha256':'physical','proposal_structure_sha256':'structure','sequences':[seq]}


def write_collection(root,data):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    save(root/'sequences.json',data)
    save(root/'sequence_manifest.json',{'data_path':'sequences.json','sha256':digest(root/'sequences.json'),
         **{k:data[k] for k in ('natural_target_sha256','experiment_target_sha256','intervention_mode')}})
    return root/'sequence_manifest.json'


class LikelihoodAndSplitTests(unittest.TestCase):
    def test_dual_fallback_reorders_pair_and_ignores_second_head(self):
        p=np.zeros((33,33));p[20:22,20:22]=[[.6,.1],[.1,.2]]
        h=np.zeros(33);h[21]=1.
        outcomes=[]
        for second in (.05,1.):
            road=NativeExperiment.__new__(NativeExperiment)
            road.cav=SimpleNamespace(id=0);road.vehicles=[road.cav,SimpleNamespace(id=1),SimpleNamespace(id=2)]
            road.config={'epsilon_min':.05};road.rng=np.random.default_rng(7)
            road.steps=[];road.log_weight=0.;road.max_intervened=0;road.time=0.
            unit={'ids':[1,2],'p':p};road.prepare=lambda pmfs:([unit],{})
            road.choose=lambda *args:{'unit':unit,'ids':[2],'h':h,'mass':.2}
            with patch('scenario_reconstruction.highd_v46_d2rl.observation',return_value=[0.]*14):
                actions=road.sample({},'dual',.3,policy=lambda obs:[.3,second])
            step=road.steps[0]
            self.assertEqual(step['natural_unit_ids'],[2,1]);self.assertEqual(step['factor_mask'],[1,0])
            self.assertEqual(step['p_factors'][1],1.);self.assertEqual(step['q_factors'][1],1.)
            self.assertAlmostEqual(log_weight(step,[.3,second]),road.log_weight)
            self.assertEqual(road.max_intervened,1)
            outcomes.append((actions,road.log_weight))
        self.assertEqual(outcomes[0],outcomes[1])

    def test_joint_is_recovers_natural_event_probability(self):
        p=np.array([[.6,.1],[.1,.2]]);h=np.array([[0.,0.],[0.,1.]])
        p1=p.sum(axis=1);h1=h.sum(axis=1);total=0.
        for i in range(2):
            p2=p[i]/p1[i];h2=h[i]/h1[i] if h1[i] else p2
            for j in range(2):
                q=(.2*p1[i]+.8*h1[i])*(.4*p2[j]+.6*h2[j])
                event=i==1 and j==1
                total+=q*event*p[i,j]/q
        self.assertAlmostEqual(total,.2)

    def test_no_actionable_pair_falls_back_to_nearest_actor_candidates(self):
        road=NativeExperiment.__new__(NativeExperiment)
        road.cav=SimpleNamespace(id=0,x=0.)
        actors={i:SimpleNamespace(id=i,x=float(i*10)) for i in range(4)}
        road.config={'dual_single_fallback':True,'criticality':{
            'observation_m':115.,'max_candidates':2,'minimum_critical_mass':1e-6,'minimum_action_contrast':1e-6}}
        units=[{'ids':[1],'p':np.ones(33)/33}, {'ids':[2,3],'p':np.ones((33,33))/1089}]
        def score(road,ids,pmfs,cfg):
            if len(ids)==2:return np.ones((33,33))*.5
            result=np.zeros(33);result[0]=1. if ids[0]==1 else .5
            return result
        with patch('scenario_reconstruction.highd_v46_d2rl.challenge',side_effect=score):
            selected=road.choose(units,actors,{},'dual')
        self.assertEqual(selected['ids'],[1])
        self.assertEqual(road.selection_diagnostics['eligible_pair_candidates'],0)
        self.assertEqual(road.selection_diagnostics['evaluated_candidates'],3)

    def test_replay_uses_current_prefix_and_never_mutates_logged_observation(self):
        data=dataset();original=copy.deepcopy(data['sequences'][0]['steps'])
        with tempfile.TemporaryDirectory() as root:
            env=V46SequenceEnv({'sequence_manifest':str(write_collection(root,data))})
            first=env.reset();self.assertEqual(first[16],0.)
            second,_,done,_=env.step([.1,.8])
            self.assertFalse(done)
            expected=log_weight(original[0],[.1,.8])/30.
            self.assertAlmostEqual(second[16],expected,places=7)
            self.assertNotAlmostEqual(second[16],original[1]['observation'][16],places=5)
            seen=[]
            def policy(obs):
                seen.append(float(obs[16]));return [.1,.8]
            objective=env.empirical_objective(policy)
            self.assertAlmostEqual(seen[1],expected,places=7)
            self.assertAlmostEqual(objective,env.loss(env.sequences[0],2*log_weight(original[0],[.1,.8])))
        self.assertEqual(data['sequences'][0]['steps'],original)

    def test_future_weight_and_inactive_likelihood_are_rejected(self):
        for mutation in ('future','inactive'):
            data=dataset()
            if mutation=='future':data['sequences'][0]['steps'][0]['observation'][16]=1.
            else:
                step=data['sequences'][0]['steps'][0]
                step['p_factors'][1]=step['h_factors'][1]=step['q_factors'][1]=.5
            with tempfile.TemporaryDirectory() as root:
                manifest=write_collection(root,data)
                with self.assertRaises(ValueError):V46SequenceEnv({'sequence_manifest':str(manifest)})

    def test_split_rejects_overlap_duplicates_and_proxy_changes(self):
        train=dataset(1);valid=dataset(2)
        independent_split(train,valid)
        for bad in (dataset(1),dict(valid,proposal_structure_sha256='other'),
                    dict(valid,sequences=valid['sequences']*2)):
            with self.assertRaises(ValueError):independent_split(train,bad)

    def test_physical_target_excludes_proposal_tuning(self):
        cfg={'horizon_s':60,'cav':{'tau':.2},'generation_epsilon':.1,'criticality':{'horizon_s':4}}
        other={**cfg,'generation_epsilon':.99,'criticality':{'horizon_s':2}}
        lock={'natural_target_sha256':'frozen'}
        self.assertEqual(identity(experiment_target(lock,cfg)),identity(experiment_target(lock,other)))
        self.assertNotEqual(identity(experiment_target(lock,cfg)),identity(experiment_target(lock,{**cfg,'horizon_s':30})))

    def test_merge_keeps_every_episode_and_rejects_duplicate_seeds(self):
        with tempfile.TemporaryDirectory() as root:
            a=write_collection(Path(root)/'a',dataset(1));b=write_collection(Path(root)/'b',dataset(2,False))
            result=merge_collections([a,b],Path(root)/'merged')
            self.assertEqual(result['episodes'],2);self.assertEqual(result['positive_episodes'],1)
            with self.assertRaises(ValueError):merge_collections([a,a],Path(root)/'bad')

    def test_fixed_baseline_is_frozen_and_marks_validation_seeds(self):
        with tempfile.TemporaryDirectory() as root:
            manifest=write_collection(Path(root)/'validation',dataset(50000))
            out=Path(root)/'fixed';report=select_fixed(manifest,out,grid=[.1,.9])
            self.assertEqual(report['epsilon'],[.1,.1])
            policy=NumpyPolicy(out)
            self.assertEqual(policy(dataset()['sequences'][0]['steps'][0]['observation']),[.1,.1])
            self.assertEqual(policy.metadata['validation_seeds'],[50000])
            negative=write_collection(Path(root)/'negative',dataset(50001,False))
            with self.assertRaises(ValueError):select_fixed(negative,Path(root)/'bad')


class PrecisionTests(unittest.TestCase):
    def test_zero_events_do_not_report_convergence_or_zero_width_interval(self):
        result=summarize_values(np.zeros(32))
        self.assertEqual(result['precision_status'],'insufficient_events')
        self.assertIsNone(result['relative_half_width']);self.assertIsNone(result['confidence_interval_normal'])
        self.assertIsNone(result['projected_required_tests'])
        required,curve=convergence(np.zeros(32));self.assertIsNone(required)
        self.assertFalse(any(row['precision_reached'] for row in curve))

    def test_full_denominator_and_normal_half_width(self):
        result=summarize_values([0.,.2],minimum_events=2)
        self.assertAlmostEqual(result['weighted_first_collision_mean'],.1)
        self.assertAlmostEqual(result['weighted_second_moment'],.02)
        self.assertAlmostEqual(result['weighted_standard_error'],.1)
        self.assertAlmostEqual(result['relative_half_width'],1.6448536269514722)

    def test_minimum_events_and_sustained_suffix(self):
        required,_=convergence(np.ones(100),minimum_events=20)
        self.assertEqual(required,20)
        # A late dominant contribution invalidates an earlier precision crossing.
        required,curve=convergence(np.r_[np.ones(100),10000.],minimum_events=20)
        self.assertTrue(curve[99]['precision_reached']);self.assertIsNone(required)

    def test_comparison_requires_shared_P_and_retains_censored_bootstraps(self):
        with tempfile.TemporaryDirectory() as root:
            natural=dataset(1,False);natural['intervention_mode']='natural';natural['sequences'][0]['steps']=[]
            a=write_collection(Path(root)/'natural',natural)
            b=write_collection(Path(root)/'dual',dataset(2,False))
            report=compare_collections({'natural':a,'dual':b},Path(root)/'report',bootstrap=10)
            self.assertIsNone(report['arms']['dual']['test_acceleration_observed'])
            self.assertEqual(report['arms']['dual']['bootstrap']['budget_censored_repeats'],10)
            wrong=dataset(3);wrong['experiment_target_sha256']='different'
            c=write_collection(Path(root)/'wrong',wrong)
            with self.assertRaises(ValueError):compare_collections({'natural':a,'dual':c},Path(root)/'bad',bootstrap=0)


class MaskedPolicyTests(unittest.TestCase):
    def test_inactive_output_has_no_log_probability_gradient(self):
        import torch
        from scenario_reconstruction.highd_v46_action_dist import V46MaskedBeta
        logits=torch.zeros((2,4),requires_grad=True)
        model=SimpleNamespace(active_factor_mask=torch.tensor([[1.,0.],[1.,1.]]))
        dist=V46MaskedBeta(logits,model)
        dist.logp(torch.tensor([[-.2,.5],[-.2,.5]])).sum().backward()
        self.assertEqual(float(logits.grad[0,1]),0.)
        self.assertEqual(float(logits.grad[0,3]),0.)
        self.assertGreater(abs(float(logits.grad[1,1])),0.)

    def test_inactive_output_cannot_change_entropy_or_kl(self):
        import torch
        from scenario_reconstruction.highd_v46_action_dist import V46MaskedBeta
        model=SimpleNamespace(active_factor_mask=torch.tensor([[1.,0.]]))
        a=V46MaskedBeta(torch.zeros((1,4)),model)
        b=V46MaskedBeta(torch.tensor([[0.,1.,0.,2.]]),model)
        self.assertAlmostEqual(float(a.entropy()),float(b.entropy()))
        self.assertEqual(float(a.kl(b)),0.)


if __name__=='__main__':unittest.main()
