import copy
import numpy as np
import pytest

from scenario_reconstruction.highd_v46_assets import save, digest
from scenario_reconstruction.highd_v46_contract import CONDITIONAL_OBSERVATION, independent_split
from scenario_reconstruction.highd_v46_conditional_baseline import make_sequences, conditional_observation
from d2rl_training.highd_v46_sequence_env import V46SequenceEnv, log_weight


def fixture():
    state={'cav_id':0,'vehicles':[{'id':i,'x':float(i*12),'v':20.,'lane':0,'length':5.} for i in range(3)]}
    row={'actor_ids':[1,2],'metrics':{'positive_interaction_mass':.01}}
    cases=[{'source_seed':s,'time_s':float(i),'row':row,'observation':conditional_observation(state,row)} for i,s in enumerate([11,11,22])]
    audit={'p_factors':[.5,.5],'h_factors':[1.,0.],'q_factors':[.95,.05],'generation_epsilon':[.1,.1]}
    weight=log_weight(audit,[.1,.1])
    raw=[[{'seed':10+i*2+j,'initial_likelihood_audit':audit,'generation_log_weight':weight,'event_result':event,'bv_only_collision':False}
          for j,event in enumerate(events)] for i,events in enumerate(([True,False],[False,False],[True,True]))]
    sequences=make_sequences(cases,raw)
    return {'contract':'highd_v46_conditional_d2rl_sequence_v1','observation_contract':CONDITIONAL_OBSERVATION,
            'action_dimension':2,'intervention_mode':'dual','sampling_scope':'conditional_state_bank',
            'natural_target_sha256':'P','experiment_target_sha256':'conditional_family','conditional_target_sha256':'fixed_law',
            'proposal_structure_sha256':'structure','sequences':sequences}


def write(tmp_path,data):
    save(tmp_path/'sequences.json',data)
    save(tmp_path/'manifest.json',{'data_path':'sequences.json','sha256':digest(tmp_path/'sequences.json'),
        **{k:data[k] for k in ('natural_target_sha256','experiment_target_sha256','conditional_target_sha256','intervention_mode')}})
    return tmp_path/'manifest.json'


def test_source_weighted_objective_and_stratified_error(tmp_path):
    data=fixture();env=V46SequenceEnv({'sequence_manifest':write(tmp_path,data),'replay_positive_fraction':.8})
    assert [s['target_mass'] for s in data['sequences']]==[.125,.125,.125,.125,.25,.25]
    values=env.objective_values(lambda obs:[.1,.1])
    assert values.mean()==pytest.approx(.625)
    assert env.objective_standard_error(values)==pytest.approx(.125)
    env.reset();_,reward,done,_=env.step([.1,.1])
    assert done and np.isfinite(reward)
    # Zero-H events stay finite and carry their full likelihood; no clipping.
    assert np.isfinite(env.empirical_objective(lambda obs:[.05,.05]))


@pytest.mark.parametrize('mutation',['mass','case_owner','two_steps','population_contract','observation','future_prefix'])
def test_reject_invalid_conditional_replay(tmp_path,mutation):
    data=fixture()
    if mutation=='mass':data['sequences'][0]['target_mass']+=.1
    elif mutation=='case_owner':data['sequences'][-1]['case_id']='0'
    elif mutation=='two_steps':data['sequences'][0]['steps']*=2
    elif mutation=='population_contract':data['contract']='highd_v46_native_d2rl_sequence_v2'
    elif mutation=='observation':data['sequences'][0]['steps'][0]['observation']=[0.]*14
    else:data['sequences'][0]['steps'][0]['observation'][16]=1.
    with pytest.raises(ValueError):V46SequenceEnv({'sequence_manifest':write(tmp_path,data)})


def test_continuation_seed_disjointness_does_not_hide_source_leakage():
    training=fixture();validation=copy.deepcopy(training)
    for s in validation['sequences']:s['seed']+=1000
    with pytest.raises(ValueError,match='source episodes'):independent_split(training,validation)
    for s in validation['sequences']:s['source_seed']+=1000
    independent_split(training,validation)


def test_observation_is_predraw_and_missing_accelerations_explicit():
    data=fixture();obs=data['sequences'][0]['steps'][0]['observation']
    assert len(obs)==22 and obs[1]==obs[9]==obs[13]==0.
    assert obs[16]==0. and obs[17]==1. and obs[18:20]==[1.,1.]
    assert len(data['sequences'])==6 and sum(s['event_result'] for s in data['sequences'])==3


def test_conditional_fixed_baseline_carries_source_provenance(tmp_path):
    from scenario_reconstruction.highd_v46_fixed_baseline import select_fixed
    from scenario_reconstruction.highd_v46_policy import NumpyPolicy
    manifest=write(tmp_path,fixture());select_fixed(manifest,tmp_path/'fixed',[.1,.9])
    policy=NumpyPolicy(tmp_path/'fixed')
    assert policy.metadata['validation_source_seeds']==[11,22]
    assert policy.metadata['sampling_scope']=='conditional_state_bank'
    assert policy.metadata['observation_contract']==CONDITIONAL_OBSERVATION
