from types import SimpleNamespace

import pytest
import ray.cloudpickle as pickle

from scenario_reconstruction import highd_v47_shared_teacher_recovery as recovery
from scenario_reconstruction.highd_v46_assets import save,digest


def test_compact_cell_preserves_model_and_live_environment():
    check=SimpleNamespace(observation_contract='contract',dataset=bytes(2_000_000))
    def make():
        class Model:
            def forward(self,x):return x,check.observation_contract
        return Model
    model=make();original=model.forward
    before=len(pickle.dumps(model))
    recovery.compact_forward(model)
    assert model().forward(3)==original(model(),3)==(3,'contract')
    assert len(pickle.dumps(model))<10_000<before
    assert len(check.dataset)==2_000_000
    assert original.__closure__[0].cell_contents is check
    assert model.forward.__code__ is original.__code__


def test_unknown_closure_rejected():
    class Wrong:
        def forward(self,x):return x
    with pytest.raises(ValueError):recovery.compact_forward(Wrong)


def fixture(tmp_path):
    prior=tmp_path/'prior';data=tmp_path/'data';audit=tmp_path/'audit.json';config=tmp_path/'ppo.json'
    save(config,dict(iterations=200));save(audit,dict(verified=True))
    save(data/'summary.json',dict(readiness=dict(modes={'single':dict(log_reward_scale=2.),'dual':dict(log_reward_scale=3.)})))
    lock={'locked':True};cfg=dict(ppo_config=str(config),training_seeds=[7])
    save(prior/'opened_protocol.json',dict(lock=lock,data_summary_sha256=digest(data/'summary.json'),audit_sha256=digest(audit)))
    save(prior/'ppo_config.json',dict(iterations=200,log_reward_scale=3.))
    return prior,cfg,lock,data,audit


def test_empty_failed_startup_is_pending_and_completed_is_skipped(tmp_path,monkeypatch):
    prior,cfg,lock,data,audit=fixture(tmp_path)
    save(prior/'single_seed7'/'training_summary.json',dict(complete=True))
    (prior/'dual_seed7').mkdir()
    (prior/'dual_seed7.log').write_text('Received message larger than max')
    calls=[]
    monkeypatch.setattr(recovery,'completed_run',lambda *args:calls.append(args))
    ppo,done,pending,hashes=recovery.plan(prior,cfg,lock,data,audit)
    assert list(done)==['single_seed7'] and pending==[dict(name='dual_seed7',mode='dual',seed=7)]
    assert len(calls)==1 and ppo['log_reward_scale']==3.
    assert 'dual_seed7.log' in hashes


def test_partial_training_is_never_silently_restarted(tmp_path):
    prior,cfg,lock,data,audit=fixture(tmp_path)
    save(prior/'single_seed7'/'iteration_0001.json',dict(iteration=1))
    with pytest.raises(ValueError,match='checkpoint recovery'):recovery.plan(prior,cfg,lock,data,audit)


def test_unknown_startup_failure_is_not_automatically_retried(tmp_path):
    prior,cfg,lock,data,audit=fixture(tmp_path)
    (prior/'single_seed7').mkdir();(prior/'single_seed7.log').write_text('Unknown failure')
    with pytest.raises(ValueError,match='documented'):recovery.plan(prior,cfg,lock,data,audit)


def test_changed_previous_inputs_rejected(tmp_path):
    prior,cfg,lock,data,audit=fixture(tmp_path)
    with pytest.raises(ValueError,match='provenance'):recovery.plan(prior,cfg,{'locked':False},data,audit)


def test_registration_adapter_restores_on_error(monkeypatch):
    from ray.rllib.models import ModelCatalog
    old=ModelCatalog.register_custom_model
    with pytest.raises(RuntimeError):
        with recovery.small_model_registration():raise RuntimeError('test')
    assert ModelCatalog.register_custom_model is old
