"""Versioned JSON-key compatibility; preserve the frozen v1 implementation/data.

Run prepare once, then use this entry point instead of formal_comparison for
the remaining commands. The only injected adapter is the bank identity reader.
Every original frozen-rule check still runs; no hash is edited or skipped.
"""
import argparse
from contextlib import contextmanager
from pathlib import Path
import subprocess
import sys

from .highd_v46_assets import read,save,digest,ROOT
from .highd_v46_d2rl import identity
from .highd_v47_fixed_alpha_contract import state_schedule
from . import highd_v47_formal_comparison as runner

CONFIG=ROOT/'configs/highd_v47_formal_identity_compat_v1.json'


def adapter_identity():
    cfg=read(CONFIG)
    if cfg['version']!=1 or cfg['sampling_change'] or cfg['new_outcomes_during_prepare']:
        raise ValueError('Representation-only compatibility required')
    for key in ('parent_config','parent_runner'):
        if digest(ROOT/cfg[key])!=cfg[key+'_sha256']:raise ValueError('Compatibility parent changed: '+key)
    return dict(version=1,config_sha256=digest(CONFIG),adapter_sha256=digest(__file__),
        parent_config_sha256=cfg['parent_config_sha256'],parent_runner_sha256=cfg['parent_runner_sha256'])


def pre_json_state_identity(state):
    """Reproduce the writer's INTEGER sort order, not an arbitrary alternate hash."""
    accelerations=state.get('last_acceleration')
    if not isinstance(accelerations,dict):raise ValueError('Missing original acceleration mapping')
    restored={}
    for key,value in accelerations.items():
        if not isinstance(key,str) or not key.isascii() or not key.isdecimal() or str(int(key))!=key:
            raise ValueError('Noncanonical actor ID in acceleration mapping')
        if int(key) in restored:raise ValueError('Duplicate acceleration actor ID')
        restored[int(key)]=value
    # All values, all other mappings and the returned original bank stay intact.
    return identity(dict(state,last_acceleration=restored))


def inspect_bank(path,lock,split):
    root=Path(path);summary=read(root/'summary.json');b=read(root/'bank.json')
    sha=digest(root/'bank.json')
    if not summary['complete'] or summary['lock']!=lock or summary['split']!=split or sha!=summary['bank_sha256']:
        raise ValueError('Bank checksum/split mismatch')
    if b['split']!=split or b['lock']!=lock or len(b['cases'])!=len(b['state_law']) or len(b['cases'])!=summary['states']:
        raise ValueError('Incomplete bank identities')
    if not {c['source_seed'] for c in b['cases']}<=runner.ranges(lock['config'])[split]:
        raise ValueError('Bank source belongs to another split')
    restored=0
    for c,s in zip(b['cases'],b['state_law']):
        if c['case_id']!=s['case_id'] or c['source_seed']!=s['source_seed'] or c['alpha']!=s['alpha']:
            raise ValueError('State identity/mass mismatch at case '+str(c['case_id']))
        actual=identity(c['state'])
        if actual!=s['state_sha256']:
            if pre_json_state_identity(c['state'])!=s['state_sha256']:
                raise ValueError('State content changed at case '+str(c['case_id']))
            restored+=1
    state_schedule(b['state_law'],1,0)
    return b,dict(split=split,bank_sha256=sha,summary_sha256=digest(root/'summary.json'),
        states=len(b['cases']),pre_JSON_integer_key_hash_matches=restored,
        serialized_hash_matches=len(b['cases'])-restored,state_law_sha256=identity(b['state_law']))


def load_bank(path,lock,split):
    return inspect_bank(path,lock,split)[0]


def prepare(frozen,bank,output):
    adapter=adapter_identity();cfg,_,_,lock=runner.checked(runner.DEFAULT,frozen)
    _,evidence=inspect_bank(bank,lock,'train')
    report=dict(verified=True,adapter=adapter,parent_lock_sha256=identity(lock),
        training_bank=evidence,code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        fresh_sources_opened=False,parameters_changed=False,parent_artifacts_rewritten=False)
    save(output,report)
    return dict(verified=True,states=evidence['states'],
        restored_identity_count=evidence['pre_JSON_integer_key_hash_matches'],fresh_sources_opened=False)


@contextmanager
def bank_reader_adapter():
    original=runner.load_bank
    runner.load_bank=load_bank
    try:yield
    finally:runner.load_bank=original


def delegate(arguments,compatibility):
    record=read(compatibility);adapter=adapter_identity()
    if not record['verified'] or record['adapter']!=adapter or record['parameters_changed'] or record['parent_artifacts_rewritten']:
        raise ValueError('Compatibility record missing/changed; do not repin an existing record')
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('command',choices=['collect-train','audit-train','train','bank','validate','seal','final-test','audit-final'])
    parser.add_argument('--config',default=runner.DEFAULT);parser.add_argument('--frozen',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--bank');parser.add_argument('--split')
    parsed,_=parser.parse_known_args(arguments)
    if parsed.command=='bank' and parsed.split!='validation':raise ValueError('Preserve the existing training bank; compatibility cannot rediscover training sources')
    _,_,_,lock=runner.checked(parsed.config,parsed.frozen)
    if identity(lock)!=record['parent_lock_sha256']:raise ValueError('Compatibility belongs to another frozen experiment')
    bank_evidence=None
    if parsed.bank:
        split='validation' if parsed.command=='validate' else 'train'
        _,bank_evidence=inspect_bank(parsed.bank,lock,split)
        if split=='train' and bank_evidence!=record['training_bank']:raise ValueError('Prepared training bank changed')
    # Sidecar precedes native/PPO work; neither its budget nor the parent changes.
    output=Path(parsed.output)
    save(output.with_name(output.name+'_identity_compatibility.json'),dict(
        adapter=adapter,compatibility_record_sha256=digest(compatibility),parent_lock_sha256=identity(lock),
        arguments=arguments,bank_evidence=bank_evidence,scope='Only bank hash interpretation changes; original frozen entry point executes'))
    previous=sys.argv
    try:
        sys.argv=[previous[0]]+arguments
        with bank_reader_adapter():runner.main()
    finally:sys.argv=previous


def main():
    if len(sys.argv)>1 and sys.argv[1]=='prepare':
        p=argparse.ArgumentParser('Declare the representation-only adapter without native simulation')
        p.add_argument('command',choices=['prepare']);p.add_argument('--frozen',required=True)
        p.add_argument('--bank',required=True);p.add_argument('--output',required=True)
        a=p.parse_args();print(prepare(a.frozen,a.bank,a.output),flush=True)
    else:
        p=argparse.ArgumentParser(add_help=False);p.add_argument('--identity-compat',required=True)
        a,remaining=p.parse_known_args();delegate(remaining,a.identity_compat)


if __name__=='__main__':main()
