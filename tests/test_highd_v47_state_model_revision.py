import pytest

from scenario_reconstruction import highd_v47_state_model_revision as m
from scenario_reconstruction.highd_v46_assets import read


@pytest.mark.parametrize('field,value',[('native_draws',1),('primary_family','legacy_log_J'),('families',[])])
def test_version_and_zero_native_boundary(field,value,monkeypatch):
    cfg=read(m.DEFAULT); cfg[field]=value
    monkeypatch.setattr(m,'read',lambda _:cfg)
    with pytest.raises(ValueError,match='version'): m.protocol(m.DEFAULT)


def test_unchanged_declared_four_families_and_nested_design():
    cfg=read(m.DEFAULT)
    assert cfg['model']==m.MODEL_CONFIG
    assert cfg['families']==m.FAMILIES
    assert cfg['primary_family']=='context_direct_M2' and cfg['native_draws']==0
    assert cfg['model']['outer_seed']==260017 and cfg['model']['inner_seed']!=260017


def test_input_hash_rejects_other_corpus_before_loader(tmp_path):
    path=tmp_path/'summary.json'; path.write_text('{}')
    audit=tmp_path/'audit.json'; audit.write_text('{}')
    with pytest.raises(ValueError,match='pinned training'):
        m.inputs(read(m.DEFAULT),{}, {},tmp_path,audit)


def test_ablation_families_dont_select_winner_after_outcomes(monkeypatch):
    called=[]
    def fake(rows,method,grid,cfg):
        called.append((rows,method,grid)); return dict(development_gate_passed=method=='log_J')
    monkeypatch.setattr(m.models,'nested_fit',fake)
    reports=m.fit_all(dict(legacy=['old'],context=['new']),read(m.DEFAULT))
    assert list(reports)==m.FAMILIES
    assert not reports['context_direct_M2']['development_gate_passed']
    assert reports['context_log_J']['development_gate_passed']
    assert called[2]==(['old'],'direct_M2',[.01,.1,1.])
