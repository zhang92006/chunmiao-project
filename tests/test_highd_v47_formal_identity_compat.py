import copy
import sys
import pytest
from scenario_reconstruction import highd_v47_formal_comparison as runner
from scenario_reconstruction import highd_v47_formal_identity_compat as compat
from scenario_reconstruction.highd_v46_assets import read,save,digest
from scenario_reconstruction.highd_v46_d2rl import identity


def fixture_bank(tmp_path):
    cfg=read(runner.DEFAULT);lock=dict(config=cfg)
    state=dict(seed=1000000000,pmfs={'2':[.5,.5],'10':[.1,.9]},last_acceleration={2:.1,10:-.2})
    case=dict(case_id='0',source_seed=1000000000,alpha=1.,state=state)
    law=[dict(case_id='0',source_seed=1000000000,alpha=1.,state_sha256=identity(state))]
    root=tmp_path/'bank'
    save(root/'bank.json',dict(cases=[case],state_law=law,split='train',lock=lock))
    save(root/'summary.json',dict(complete=True,lock=lock,split='train',states=1,bank_sha256=digest(root/'bank.json')))
    return root,lock


def rewrite_bank(root,b):
    # Test-only tampering with a consistent OUTER hash still fails inner identity.
    (root/'bank.json').unlink();save(root/'bank.json',b)
    summary=read(root/'summary.json');summary['bank_sha256']=digest(root/'bank.json')
    (root/'summary.json').unlink();save(root/'summary.json',summary)


def test_writer_integer_sort_and_reader_string_sort_are_different():
    state={'last_acceleration':{2:.1,10:.2}}
    stored={'last_acceleration':{'2':.1,'10':.2}}
    before=copy.deepcopy(stored)
    assert identity(state)!=identity(stored)
    assert compat.pre_json_state_identity(stored)==identity(state)
    assert stored==before


def test_adapter_accepts_exact_legacy_hash_without_changing_bytes_or_law(tmp_path):
    root,lock=fixture_bank(tmp_path);original={p.name:digest(p) for p in root.iterdir()}
    with pytest.raises(ValueError,match='identity/mass'):runner.load_bank(root,lock,'train')
    b,e=compat.inspect_bank(root,lock,'train')
    assert e['pre_JSON_integer_key_hash_matches']==1 and e['serialized_hash_matches']==0
    assert b==read(root/'bank.json') and list(b['cases'][0]['state']['last_acceleration'])==['2','10']
    assert {p.name:digest(p) for p in root.iterdir()}==original


def test_already_matching_serialized_hash_is_accepted(tmp_path):
    root,lock=fixture_bank(tmp_path);b=read(root/'bank.json')
    b['state_law'][0]['state_sha256']=identity(b['cases'][0]['state']);rewrite_bank(root,b)
    assert compat.inspect_bank(root,lock,'train')[1]['serialized_hash_matches']==1


@pytest.mark.parametrize('field',['state_value','acceleration_value','case_id','source_seed','alpha','state_hash'])
def test_real_data_changes_remain_rejected(tmp_path,field):
    root,lock=fixture_bank(tmp_path);b=read(root/'bank.json');c=b['cases'][0]
    if field=='state_value':c['state']['pmfs']['2'][0]=.6
    elif field=='acceleration_value':c['state']['last_acceleration']['2']=.2
    elif field=='state_hash':b['state_law'][0]['state_sha256']='bad'
    else:c[field]=42
    rewrite_bank(root,b)
    with pytest.raises(ValueError):compat.inspect_bank(root,lock,'train')


@pytest.mark.parametrize('key',['02','-2','2.0','actor2','٢'])
def test_actor_key_conversion_is_not_a_general_hash_bypass(key):
    with pytest.raises(ValueError,match='Noncanonical'):compat.pre_json_state_identity({'last_acceleration':{key:1.}})


def test_outer_checksum_and_whole_source_split_stay_required(tmp_path):
    root,lock=fixture_bank(tmp_path)
    with pytest.raises(ValueError,match='split'):compat.inspect_bank(root,lock,'validation')
    (root/'bank.json').write_text('{}')
    with pytest.raises(ValueError,match='checksum'):compat.inspect_bank(root,lock,'train')


def test_delegation_is_scoped_to_bank_reader_and_records_adapter(tmp_path,monkeypatch):
    root,lock=fixture_bank(tmp_path);_,e=compat.inspect_bank(root,lock,'train')
    monkeypatch.setattr(compat,'adapter_identity',lambda:{'version':1})
    save(tmp_path/'compat.json',dict(verified=True,adapter={'version':1},parameters_changed=False,
        parent_artifacts_rewritten=False,parent_lock_sha256=identity(lock),training_bank=e))
    monkeypatch.setattr(runner,'checked',lambda *args:(lock['config'],{},None,lock))
    original=runner.load_bank;previous=sys.argv;called=[]
    def main():
        called.append(sys.argv[1:]);assert runner.load_bank is compat.load_bank
        assert runner.load_bank(root,lock,'train')['cases'][0]['alpha']==1.
    monkeypatch.setattr(runner,'main',main)
    args=['audit-train','--frozen','frozen','--bank',str(root),'--output',str(tmp_path/'audit.json')]
    compat.delegate(args,tmp_path/'compat.json')
    assert called==[args] and runner.load_bank is original and sys.argv is previous
    sidecar=read(tmp_path/'audit.json_identity_compatibility.json')
    assert sidecar['bank_evidence']==e and sidecar['adapter']=={'version':1}


def test_changed_adoption_record_stops_before_running_parent(tmp_path,monkeypatch):
    monkeypatch.setattr(compat,'adapter_identity',lambda:{'version':1})
    save(tmp_path/'bad.json',dict(verified=True,adapter={'version':2},parameters_changed=False,parent_artifacts_rewritten=False))
    monkeypatch.setattr(runner,'main',lambda:pytest.fail('Should not execute'))
    with pytest.raises(ValueError,match='record missing/changed'):compat.delegate([],tmp_path/'bad.json')


def test_versioned_parent_pins_match_without_opening_native_sources():
    assert compat.adapter_identity()['version']==1


def test_wrapper_cannot_rediscover_existing_training_sources(tmp_path,monkeypatch):
    monkeypatch.setattr(compat,'adapter_identity',lambda:{'version':1})
    save(tmp_path/'compat.json',dict(verified=True,adapter={'version':1},parameters_changed=False,parent_artifacts_rewritten=False))
    monkeypatch.setattr(runner,'checked',lambda *args:pytest.fail('Rediscovery should stop before parent checks or simulation'))
    with pytest.raises(ValueError,match='cannot rediscover'):
        compat.delegate(['bank','--split','train','--frozen','frozen','--output',str(tmp_path/'out')],tmp_path/'compat.json')
    assert not (tmp_path/'out').exists()
