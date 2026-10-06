import copy

import numpy as np
import pytest

from scenario_reconstruction.highd_v47_decision_branching_audit import reconstruct_tree


def terminal_fixture():
    step=dict(actor_ids=[1,2],p_factors=[.2,.3],t_factors=[.4,.6],
              full_log_p=float(np.log(.06)),full_log_t=float(np.log(.24)),
              cancelled_log_conditional=0.,log_t_over_p=float(np.log(4.)))
    prefix=dict(steps=[step],full_log_p=step['full_log_p'],full_log_t=step['full_log_t'],
                log_t_over_p=step['log_t_over_p'],initial_cancelled_log_conditional=-1.,
                future_weight=.4,weighted_event=0.,event_result=False,bv_only_collision=False,
                collision_pairs=[],elapsed_s=3.)
    t=dict(actor_id=1,action_indices=[1,2],probabilities=[.75,.25],bv_ids=[1,2],
           all_action_indices=[[1,1],[2,3]],full_natural_action_logs=[-2.,-3.],
           event_results=[False,True],competing_results=[False,False],elapsed_s=[1.,.2],
           collision_pairs=[[],[[0,2]]],selected_action_row=1,conditional_event_value=.25,
           single_value=.4,branching_value=.1,removed_conditional_variance_unbiased=.03)
    node=dict(status='terminal_integration',node_index=5,prefix=prefix,children=[],terminal=t,
              single_value=.4,branching_value=.1,removed_conditional_variance_unbiased=.03)
    return node,dict(control_uniforms=[.9]*21)


def test_terminal_density_is_prefix_weight_once_and_root_not_action_count():
    node,schedule=terminal_fixture();s,b,d,error=reconstruct_tree(node,schedule,0)
    np.testing.assert_allclose([s,b,d],[.4,.1,.03]);assert error<1e-12
    corrupted=copy.deepcopy(node);corrupted['terminal']['branching_value']*=.4
    with pytest.raises(AssertionError):reconstruct_tree(corrupted,schedule,0)


def test_audit_checks_natural_marginal_and_competing_endpoint():
    node,schedule=terminal_fixture();node['terminal']['probabilities']=[.5,.25]
    with pytest.raises(AssertionError):reconstruct_tree(node,schedule,0)
    node,schedule=terminal_fixture();node['terminal']['competing_results'][1]=True
    with pytest.raises(AssertionError):reconstruct_tree(node,schedule,0)


def test_shared_prefix_early_endpoints_and_tree_posterior():
    def prefix(event,bv,steps):
        return dict(steps=steps,full_log_p=0.,full_log_t=0.,log_t_over_p=0.,initial_cancelled_log_conditional=-1.,
                    future_weight=1.,weighted_event=float(event),event_result=event,bv_only_collision=bv,
                    collision_pairs=[[0,1]] if event else [[1,2]] if bv else [],elapsed_s=2. if steps else 1.)
    step=dict(actor_ids=[],full_log_p=0.,full_log_t=0.,log_t_over_p=0.)
    children=[dict(status='stopped',node_index=i+1,prefix=prefix(i>=2,i<2,[step]),children=[],
                   single_value=float(i>=2),branching_value=float(i>=2),removed_conditional_variance_unbiased=0.) for i in range(4)]
    root=dict(status='branched',node_index=0,prefix=prefix(False,False,[]),children=children,
              posterior_natural_fraction=.5,selected_child=2,single_value=1.,branching_value=.5,removed_conditional_variance_unbiased=.25)
    np.testing.assert_allclose(reconstruct_tree(root,dict(control_uniforms=[.9]*21),0)[:3],[1.,.5,.25])
    root['posterior_natural_fraction']=.3
    with pytest.raises(AssertionError):reconstruct_tree(root,dict(control_uniforms=[.9]*21),0)
