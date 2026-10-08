"""Reproduce training-only source model ablations without new native outcomes."""
import argparse
from pathlib import Path
import subprocess

import numpy as np

from . import highd_v47_source_transfer as parent
from . import highd_v47_state_context as context
from . import highd_v47_state_features as legacy
from . import highd_v47_state_objective_model as models
from .highd_v46_assets import digest, read, save
from .highd_v46_d2rl import identity
from .highd_v46_multistate_q import environment

DEFAULT = 'configs/highd_v47_state_model_revision_v1.json'
FAMILIES = ['legacy_log_J','context_log_J','legacy_direct_M2','context_direct_M2']
MODEL_CONFIG = dict(outer_folds=4,outer_seed=260017,inner_folds=3,inner_seed=261017,
    log_grid=[1.,10.,100.],direct_grid=[.01,.1,1.],prior_roots=16.,probability_floor=1e-8,
    allocation_loss_cap=1.1,minimum_positive_sources=5,
    optimizer=dict(maxiter=400,ftol=1e-12,gtol=1e-8,coefficient_bound=8.))


def protocol(config):
    cfg = read(config)
    if (cfg['version'] != 1 or cfg['native_draws'] != 0 or cfg['families'] != FAMILIES
            or cfg['primary_family'] != 'context_direct_M2' or cfg['model'] != MODEL_CONFIG
            or digest(cfg['parent_config']) != cfg['parent_config_sha256']):
        raise ValueError('Explicitly version changed training model design')
    old_cfg,*_,old_lock = parent.protocol(cfg['parent_config'])
    dependencies = {name:digest(Path(__file__).with_name(name)) for name in (
        'highd_v47_state_context.py','highd_v47_state_objective_model.py','highd_v47_allocation.py')}
    return cfg,old_cfg,old_lock,dict(config_sha256=digest(config),implementation_sha256=digest(__file__),
        dependencies=dependencies,environment=environment(),parent_lock=old_lock,
        source_opening_allowed=False,native_draws=0)


def inputs(cfg, old_cfg, old_lock, training, audit):
    root = Path(training)
    if digest(root/'summary.json') != cfg['training_summary_sha256'] or digest(audit) != cfg['training_audit_sha256']:
        raise ValueError('Use the complete pinned training and independent native audit')
    # This is a DEVELOPMENT loader, not a failed model's validation guard bypass.
    # It verifies original code, all native artifacts, labels and original fit.
    root,summary,_ = parent.completed_training(root,old_cfg,old_lock)
    parent.check_training_audit(audit,root)
    if summary['conditional_target_sha256'] != cfg['conditional_target_sha256']:
        raise ValueError('Declared conditional target changed')
    rows = []; richer = []
    for measurement in summary['measurements']:
        case = read(root/('case_%03d_design.json' % measurement['state_id']))
        source = measurement['source_seed']
        if (source != case['plan']['source_seed'] or case['state']['seed'] != source
                or not 260000 <= source <= 260255 or case['plan']['target_mass'] != 1/summary['selected_sources']):
            raise ValueError('Training source/state/target mismatch')
        args = (case['state'],case['plan']['actor_ids'],case['P'],case['QA'],case['risk'])
        if legacy.state_features(*args) != measurement['features']:
            raise ValueError('Original pre-outcome feature reconstruction changed')
        rows.append(measurement); richer.append(dict(measurement,features=context.state_features(*args)))
    return dict(legacy=rows,context=richer)


def fit_all(design, cfg):
    reports = {}
    for family in FAMILIES:
        feature_set,method = family.split('_',1)
        grid = cfg['model']['log_grid' if method == 'log_J' else 'direct_grid']
        reports[family] = models.nested_fit(design[feature_set],method,grid,cfg['model'])
    return reports


def freeze(config, training, audit, output):
    cfg,old_cfg,old_lock,lock = protocol(config)
    design = inputs(cfg,old_cfg,old_lock,training,audit)
    lock.update(training_path=str(Path(training)),training_audit_path=str(Path(audit)),
        training_summary_sha256=digest(Path(training)/'summary.json'),training_audit_sha256=digest(audit),
        design_sha256=identity(design),code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        status='Frozen training-only fitting design; labels already consumed, no new outcomes')
    save(output,lock); return lock


def checked(lock, config):
    frozen = read(lock); cfg,old_cfg,old_lock,current = protocol(config)
    if any(frozen.get(k) != v for k,v in current.items()): raise ValueError('Model design changed after freeze')
    design = inputs(cfg,old_cfg,old_lock,frozen['training_path'],frozen['training_audit_path'])
    if identity(design) != frozen['design_sha256']: raise ValueError('Feature/measurement design changed')
    return frozen,cfg,design


def run(lock, config, output):
    frozen,cfg,design = checked(lock,config); out = Path(output); out.mkdir(parents=True,exist_ok=False)
    save(out/'opened_protocol.json',dict(lock=frozen,lock_sha256=digest(lock),config=cfg))
    save(out/'design.json',design)
    reports = fit_all(design,cfg); save(out/'models.json',reports)
    primary = reports[cfg['primary_family']]
    summary = dict(complete=True,config=cfg,lock=frozen,design_sha256=identity(design),
        selected_sources=len(design['legacy']),new_native_draws=0,source_opening_allowed=False,
        primary_family=cfg['primary_family'],development_gate_passed=primary['development_gate_passed'],
        metrics={name:{k:report[k] for k in ('aggregate_ratio_point','batch_objectives','measured_positive_sources',
            'development_gate_passed','final_selection')} for name,report in reports.items()},
        validation_opened=False,production_changed=False,ppo_ready=False,training_exported=False,
        artifact_hashes={n:digest(out/n) for n in ('opened_protocol.json','design.json','models.json')})
    save(out/'summary.json',summary); return summary


def audit(config, directory, output):
    root = Path(directory); summary = read(root/'summary.json'); opened = read(root/'opened_protocol.json')
    cfg,old_cfg,old_lock,current = protocol(config); frozen = opened['lock']
    if (not summary['complete'] or summary['config'] != cfg or opened['config'] != cfg or summary['lock'] != frozen
            or any(frozen.get(k) != v for k,v in current.items())):
        raise ValueError('Completed matching model revision required')
    if (any(summary.get(k) is not False for k in ('validation_opened','production_changed','ppo_ready','training_exported'))
            or summary['new_native_draws'] != 0 or summary['primary_family'] != cfg['primary_family']
            or summary['source_opening_allowed'] is not False):
        raise ValueError('Training-only boundary or primary family changed')
    for name,sha in summary['artifact_hashes'].items():
        if Path(name).name != name or digest(root/name) != sha: raise ValueError('Changed revision artifact')
    design = inputs(cfg,old_cfg,old_lock,frozen['training_path'],frozen['training_audit_path'])
    if identity(design) != frozen['design_sha256'] or design != read(root/'design.json'):
        raise ValueError('Feature extraction no longer reproduces')
    reconstructed = fit_all(design,cfg)
    if reconstructed != read(root/'models.json'): raise ValueError('Nested fitting no longer reproduces')
    # Independently recompute held-out masses/objectives; do not trust logged
    # ratios or a favorable fitted full-bank objective as a generalization test.
    maximum_error = 0.
    for family,report in reconstructed.items():
        lookup = {r['source_seed']:r for r in design[family.split('_',1)[0]]}
        weighted_base = np.zeros(2); weighted_candidate = np.zeros(2)
        for fold in report['outer_reports']:
            checking = [lookup[s] for s in fold['checking_sources']]
            if set(fold['checking_sources']) & set(fold['model']['fitting_sources']): raise ValueError('Outer source leakage')
            for grid in fold['inner_selection']['grid_reports']:
                for inner in grid['folds']:
                    if set(fold['checking_sources']) & (set(inner['fitting_sources'])|set(inner['checking_sources'])):
                        raise ValueError('Outer outcomes entered inner selection')
            scores = models.predict(fold['model'],[r['features'] for r in checking])
            alpha = np.full(len(checking),1/len(checking)); q = models.waterfill(alpha,scores,1/cfg['model']['allocation_loss_cap'])
            if np.min(q/alpha) < 1/1.1-1e-12: raise ValueError('Protected state floor violated')
            for b in (0,1):
                j = np.array([r['batch_estimates'][b]['J'] for r in checking])
                base = float(np.mean(j)); candidate = float(sum(alpha[i]**2*j[i]/q[i] for i in range(len(j))))
                logged = fold['comparison']['batches'][b]
                maximum_error = max(maximum_error,abs(base-logged['baseline_M2']),abs(candidate-logged['candidate_M2']))
                weighted_base[b] += len(j)*base/len(lookup); weighted_candidate[b] += len(j)*candidate/len(lookup)
        if not np.allclose(weighted_candidate/weighted_base,[v['ratio_point'] for v in report['batch_objectives']],rtol=1e-12,atol=1e-14):
            raise ValueError('Aggregate held-source objective changed')
        expected = {k:report[k] for k in summary['metrics'][family]}
        if summary['metrics'][family] != expected: raise ValueError('Summary differs from reconstructed models')
    if summary['development_gate_passed'] != reconstructed[cfg['primary_family']]['development_gate_passed']:
        raise ValueError('Primary decision changed')
    record = dict(verified=True,summary_sha256=digest(root/'summary.json'),
        artifact_hashes_verified=len(summary['artifact_hashes']),independent_training_sources=len(design['legacy']),
        original_native_audit_sha256=cfg['training_audit_sha256'],new_native_draws=0,
        maximum_objective_reconstruction_error=maximum_error,validation_opened=False,
        implementation_sha256=digest(__file__))
    save(output,record); return record


def main():
    parser = argparse.ArgumentParser(__doc__); parser.add_argument('--config',default=DEFAULT)
    sub = parser.add_subparsers(dest='command',required=True)
    p = sub.add_parser('freeze'); p.add_argument('--training',required=True); p.add_argument('--training-audit',required=True); p.add_argument('--output',required=True)
    p = sub.add_parser('run'); p.add_argument('--lock',required=True); p.add_argument('--output',required=True)
    p = sub.add_parser('audit'); p.add_argument('--directory',required=True); p.add_argument('--output',required=True)
    args = parser.parse_args()
    if args.command == 'freeze': result = freeze(args.config,args.training,args.training_audit,args.output)
    elif args.command == 'run': result = run(args.lock,args.config,args.output)
    else: result = audit(args.config,args.directory,args.output)
    if args.command == 'run':
        print(dict(complete=result['complete'],development_gate_passed=result['development_gate_passed'],
            ratios={k:v['aggregate_ratio_point'] for k,v in result['metrics'].items()},validation_opened=False))
    else: print(result)


if __name__ == '__main__': main()
