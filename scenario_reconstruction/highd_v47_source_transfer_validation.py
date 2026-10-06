"""Versioned validation adapter; leave the frozen v1 training code intact.

The v1 bootstrap path erroneously passes an M2 loss cap (>1) as a probability
floor. This prospective adapter corrects only that reporting path, before any
validation source opens. Training loaders, model, measurement and Q are reused
without modifying their frozen implementation or loosening their checks.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from pathlib import Path
import subprocess
import time

import numpy as np

from . import highd_v47_source_transfer as parent
from .highd_v46_assets import digest, read, save
# The shared read-only auditor uses these unchanged pure functions.
from .highd_v47_source_transfer import (PairModelQueries, SINGLE_OPTIONS,
    prepare_cases, measurement_design, read_records, estimate, objective_reports,
    identity, branching, natural_root)

DEFAULT = 'configs/highd_v47_source_transfer_validation_v2.json'
PARENT_SHA = '0ee2da48365041db841a8faa78698544a46e82f24ce0fc4d9025600bddc1c82b'


def protocol(config):
    declaration = read(config)
    expected = dict(version=2, parent_config=parent.DEFAULT, parent_config_sha256=PARENT_SHA,
        change='Validation-only correction: bootstrap water-filling uses reciprocal allocation_loss_cap as its probability floor, matching the frozen candidate. No fitting, source, continuation, budget, seed or acceptance-rule change.',
        training_unchanged=True, validation_sources_previously_unopened=True)
    if declaration != expected or digest(parent.DEFAULT) != PARENT_SHA:
        raise ValueError('Explicitly version changes beyond the validation floor correction')
    cfg, dc, base, physical, assets, old_lock = parent.protocol(parent.DEFAULT)
    lock = dict(old_lock, validation_protocol_sha256=digest(config),
        validation_implementation_sha256=digest(__file__),
        shared_audit_sha256=digest(Path(__file__).with_name('highd_v47_source_transfer_audit.py')),
        validation_audit_sha256=digest(Path(__file__).with_name('highd_v47_source_transfer_validation_audit.py')))
    return cfg, dc, base, physical, assets, lock


def completed_training(path, cfg, current):
    original_cfg, *_, original_lock = parent.protocol(parent.DEFAULT)
    if cfg != original_cfg or any(current.get(k) != v for k, v in original_lock.items()):
        raise ValueError('Frozen v1 training identity changed')
    return parent.completed_training(path, original_cfg, original_lock)


def checked_training(path, audit_path, cfg, current):
    root, summary, fitted = completed_training(path, cfg, current)
    if (fitted['measured_positive_sources'] < cfg['model']['minimum_positive_training_sources']
            or fitted['selected_OOF_ratio_point'] >= 1):
        raise ValueError('Training source-held-out review failed; leave validation unopened')
    parent.check_training_audit(audit_path, root)
    return root, summary, fitted


def freeze(config, training, training_audit, output):
    cfg, *_, lock = protocol(config)
    root, _, fitted = checked_training(training, training_audit, cfg, lock)
    lock.update(split='validation', sources_opened=False, validation_version=2,
        training_path=str(root), training_summary_sha256=digest(root/'summary.json'),
        model_sha256=digest(root/'model.json'), benefit_bet_rate=fitted['benefit_bet']['rate'],
        training_audit_path=str(training_audit), training_audit_sha256=digest(training_audit),
        code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip())
    save(output, lock)
    return lock


def source_comparison(design, measurements, all_sources, cfg):
    """Same declared whole-source statistic, with the correct Q_S floor.

    Resampling keeps roots within sources, includes abstentions and recomputes
    normalization. It does not refit the model or resample tree descendants.
    """
    if not measurements:
        return dict(status='no_selected_sources', interval=None, source_generalization_supported=False)
    all_sources = list(all_sources)
    present = {r['source_seed']: r for r in measurements}
    lookup = {s['plan']['source_seed']: i for i, s in enumerate(design['states'])}
    alpha, q = (np.array(design[k]) for k in ('target_masses', 'candidate_q'))
    j = np.array([r['J'] for r in measurements])
    point_a = float(alpha@j); point_q = float((alpha*alpha/q)@j)
    predictions = np.array(design['predicted_J'])
    rng = np.random.default_rng(cfg['bootstrap_seed']); ratios = []
    source_count = len([r for r in measurements if r['positive_roots'] > 0])
    # This is a probability FLOOR, not the reciprocal's M2 loss CAP.
    floor = 1/parent.read(parent.DEFAULT)['allocation_loss_cap']
    for _ in range(cfg['bootstrap_repeats']):
        sampled = rng.choice(all_sources, len(all_sources), replace=True)
        selected = [lookup[int(s)] for s in sampled if int(s) in present]
        positive_unique = {int(s) for s in sampled
            if int(s) in present and present[int(s)]['positive_roots'] > 0}
        if not selected or len(positive_unique) < cfg['minimum_positive_sources']:
            continue
        a = np.full(len(selected), 1/len(selected))
        qs = parent.waterfill(a, predictions[selected], floor)
        base = float(a@j[selected]); proposed = float((a*a/qs)@j[selected])
        if base > 0: ratios.append(proposed/base)
    valid = len(ratios); censored = cfg['bootstrap_repeats']-valid
    permitted = (source_count >= cfg['minimum_positive_sources']
        and censored/cfg['bootstrap_repeats'] <= cfg['maximum_censored_fraction'])
    interval = (np.quantile(ratios, [(1-cfg['confidence'])/2, (1+cfg['confidence'])/2]).tolist()
        if permitted and ratios else None)
    contributions = alpha*alpha/q*j; total = float(contributions.sum())
    shares = contributions/total if total > 0 else None
    d = alpha*(alpha/q-1); v = np.array([r['variance'] for r in measurements])
    return dict(target_M2=point_a, candidate_M2=point_q,
        M2_ratio_point=point_q/point_a if point_a else None,
        difference_empirical_MCSE=float(np.sqrt((d*d)@v)),
        paired_source_bootstrap_ratio_interval=interval, valid_replicates=valid, censored_replicates=censored,
        measured_positive_sources=source_count, all_source_episodes=len(all_sources), selected_sources=len(measurements),
        maximum_source_M2_share=float(shares.max()) if shares is not None else None,
        effective_M2_sources=float(1/(shares@shares)) if shares is not None else None,
        source_generalization_supported=bool(interval is not None and interval[1] < 1),
        uncertainty_scope='Approximate whole-source bootstrap with frozen predictions and re-normalized Q_S; root MCSE separate. Not a distribution-free population guarantee.')


def run(config, lock_path, output):
    cfg, dc, base, physical, assets, current = protocol(config); lock = read(lock_path)
    if (lock.get('split') != 'validation' or lock.get('sources_opened') is not False
            or lock.get('validation_version') != 2 or any(lock.get(k) != v for k, v in current.items())):
        raise ValueError('Versioned validation changed after freeze')
    root, _, fitted = checked_training(lock['training_path'], lock['training_audit_path'], cfg, current)
    if (digest(root/'summary.json') != lock['training_summary_sha256']
            or digest(root/'model.json') != lock['model_sha256']
            or digest(lock['training_audit_path']) != lock['training_audit_sha256']
            or fitted['benefit_bet']['rate'] != lock['benefit_bet_rate']):
        raise ValueError('Training model/audit changed after validation freeze')
    out = Path(output); out.mkdir(parents=True, exist_ok=False); started = time.perf_counter()
    save(out/'opened_protocol.json', dict(config=cfg, validation_protocol=read(config),
        lock=lock, rule_lock_sha256=digest(lock_path)))
    model = PairModelQueries(assets/'model', assets/'lateral', **SINGLE_OPTIONS)
    bank, rows, discovery = parent.discover(dict(cfg, source=dict(cfg['validation'], split='validation')),
        dc, base, physical, model, assets)
    seeds = list(range(cfg['validation']['seed_start'], cfg['validation']['seed_start']+cfg['validation']['episodes']))
    if [e['seed'] for e in discovery['episodes']] != seeds:
        raise ValueError('Incomplete new validation cohort')
    save(out/'state_bank.json', dict(split='validation', states=bank)); save(out/'pair_audit.json', dict(rows=rows))
    cases, selections = prepare_cases(bank, rows, seeds, model, physical, base, cfg)
    for c in cases: save(out/f'case_{c["plan"]["case_id"]:03d}_design.json', c)
    save(out/'case_plan.json', dict(source_seeds=seeds, selections=selections,
        cases=[c['plan'] for c in cases], selection_uses_native_outcomes=False))
    if not cases: raise ValueError('No eligible sources; retain discovery without backfill')
    design = measurement_design(cases, cfg, 'validation', fitted); design['model_config'] = cfg['model']
    design['conditional_target_sha256'] = identity([(c['plan']['state_sha256'], c['plan']['actor_ids'], c['plan']['target_mass']) for c in cases])
    save(out/'measurement_plan.json', design)
    records = []
    for b in (0, 1):
        with ProcessPoolExecutor(max_workers=cfg['workers'], mp_context=multiprocessing.get_context('spawn'),
                initializer=parent.continuation._init_worker, initargs=(str(assets),)) as pool:
            jobs = {pool.submit(parent._collect, (s, cases[s['state_id']], str(assets), physical, cfg))
                for s in design['schedules'] if s['batch'] == b}
            for future in as_completed(jobs):
                r = future.result(); save(out/f'batch_{b}_state_{r["state_id"]:03d}.json', r)
                records.append(parent.compact_record(r)); jobs.remove(future)
                print(dict(phase='validation_v2', batch=b, completed=len(records), total=2*len(cases)), flush=True)
    measured = estimate(design, records); certificates = objective_reports(design, records)
    comparison = source_comparison(design, measured, seeds, cfg['statistics'])
    parent.continuation._MODEL = model
    samples, smoke = parent.iid_comparison(dict(iid_continuation_seed_start=cfg['iid_seed_start']),
        assets, physical, [c['plan'] for c in cases], cases, design)
    for arm, data in samples.items(): save(out/f'{arm}_sequences.json', data)
    result = dict(complete=True, split='validation', config=cfg, validation_protocol=read(config), lock=lock,
        discovery=discovery, natural_sources=len(seeds), selected_sources=len(cases), bank_states=len(bank),
        bounded_pair_audits=len(rows), conditional_target_sha256=design['conditional_target_sha256'],
        independent_measurement_roots=design['roots'], measurements=measured, objective_certificates=certificates,
        source_comparison=comparison, iid_smoke=smoke, model_sha256=lock['model_sha256'], fitting_status=None,
        fixed_new_bank_improvement_certified=all(r['improvement_certified'] for r in certificates),
        transfer_review_passed=bool(all(r['improvement_certified'] for r in certificates)
            and comparison['source_generalization_supported']), total_wall_s=time.perf_counter()-started,
        production_changed=False, ppo_ready=False, training_exported=False,
        artifact_hashes={p.name: digest(p) for p in sorted(out.glob('*.json'))})
    save(out/'summary.json', result)
    return {k: result[k] for k in ('split', 'selected_sources', 'independent_measurement_roots', 'transfer_review_passed', 'ppo_ready')}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['freeze', 'run']); parser.add_argument('--config', default=DEFAULT)
    parser.add_argument('--training'); parser.add_argument('--training-audit'); parser.add_argument('--lock')
    parser.add_argument('--output', required=True); args = parser.parse_args()
    if args.mode == 'freeze':
        if not args.training or not args.training_audit: parser.error('complete training and independent audit required')
        record = freeze(args.config, args.training, args.training_audit, args.output)
        print(dict(validation_version=2, sources_opened=False, model_sha256=record['model_sha256']))
    else:
        if not args.lock: parser.error('run requires frozen validation lock')
        print(run(args.config, args.lock, args.output))
