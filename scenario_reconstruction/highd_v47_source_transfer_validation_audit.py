"""Read-only v2 validation audit using the unchanged reconstruction engine."""
import argparse
from pathlib import Path

from . import highd_v47_source_transfer_validation as validation
from .highd_v46_assets import digest, read, save
from .highd_v47_source_transfer_audit import audit as reconstruct


def audit(config, directory, replay=False):
    root = Path(directory); summary = read(root/'summary.json')
    opened = read(root/'opened_protocol.json')
    if (summary['split'] != 'validation' or summary['validation_protocol'] != read(config)
            or opened['validation_protocol'] != read(config) or summary['lock']['validation_version'] != 2):
        raise ValueError('Versioned validation record required')
    cfg, *_, current = validation.protocol(config)
    lock = summary['lock']
    validation.checked_training(lock['training_path'], lock['training_audit_path'], cfg, current)
    if digest(lock['training_audit_path']) != lock['training_audit_sha256']:
        raise ValueError('Training audit changed')
    record = reconstruct(config, directory, replay, native=validation)
    record.update(shared_audit_implementation_sha256=record['audit_implementation_sha256'],
        audit_implementation_sha256=digest(__file__), validation_version=2,
        validation_implementation_sha256=digest(validation.__file__), validation_protocol_sha256=digest(config))
    return record


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--config', default=validation.DEFAULT)
    p.add_argument('--run', required=True); p.add_argument('--replay', action='store_true'); p.add_argument('--output')
    a = p.parse_args(); result = audit(a.config, a.run, a.replay)
    if a.output: save(a.output, result)
    print(result)
