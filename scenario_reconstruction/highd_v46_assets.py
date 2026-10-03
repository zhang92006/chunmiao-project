"""Portable, checksum-locked V46 runtime assets; no licensed raw tracks needed."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / 'configs/highd_paper_v46_freeze.json'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def ensure_assets(destination=None):
    lock = read(LOCK)
    for rel, expected in lock['code_sha256'].items():
        if digest(ROOT / rel) != expected:
            raise ValueError('Frozen NDE source changed: '+rel)
    archive = ROOT / lock['archive']
    if digest(archive) != lock['archive_sha256']:
        raise ValueError('Runtime archive missing or changed; run git lfs pull')
    destination = Path(destination) if destination else ROOT / 'outputs/highd_paper_v46_assets'
    destination = destination.resolve()
    with zipfile.ZipFile(archive) as source:
        if set(source.namelist()) != set(lock['files']):
            raise ValueError('Unexpected archive contents')
        for rel, expected in lock['files'].items():
            parts = PurePosixPath(rel)
            if parts.is_absolute() or '..' in parts.parts or ':' in rel or '\\' in rel:
                raise ValueError('Unsafe archive member')
            path = destination.joinpath(*parts.parts)
            if path.exists():
                if digest(path) != expected:
                    raise ValueError('Existing runtime file differs: '+str(path))
            else:
                raw = source.read(rel)
                if hashlib.sha256(raw).hexdigest() != expected:
                    raise ValueError('Archive member checksum mismatch')
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open('xb') as stream:
                    stream.write(raw)
    return destination, lock


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output')
    args = parser.parse_args()
    destination, lock = ensure_assets(args.output)
    print(json.dumps({'assets': str(destination), 'natural_target_sha256': lock['natural_target_sha256'],
                      'verified_files': len(lock['files']), 'nde_code_commit': lock['nde_code_commit']}))
