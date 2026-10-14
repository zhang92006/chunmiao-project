"""Exact read-only lookup of published COO tables without dense expansion.

The cache contains original float64 values, not a fitted/compressed approximation.
Build it once before launching workers; workers memory-map the same cache files.
"""
import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / 'outputs/published_ndd_cache'


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def cache_paths(source):
    source = Path(source)
    lock = json.loads((ROOT/'configs/published_d2rl_assets.json').read_text(encoding='utf-8'))
    relative = source.resolve().relative_to(ROOT).as_posix()
    expected = lock['files'][relative]['sha256']
    return CACHE/(source.stem+'-'+expected[:16]), expected


def prepare_table(source):
    source = Path(source)
    destination, expected = cache_paths(source)
    if sha256(source) != expected:
        raise ValueError('Published source checksum differs: '+str(source))
    manifest = destination/'manifest.json'
    if manifest.exists():
        info = json.loads(manifest.read_text())
        for name, digest in info['cache_sha256'].items():
            if sha256(destination/name) != digest:
                raise ValueError('Cache checksum differs: '+str(destination/name))
        return info
    destination.mkdir(parents=True, exist_ok=True)
    with np.load(source, allow_pickle=False) as stored:
        shape = tuple(int(x) for x in stored['shape'])
        fill = stored['fill_value'].item()
        if fill != 0 or shape[-1] != 2:
            raise ValueError('Expected zero-filled published two-action table')
        values = stored['data']
        coords = stored['coords']
    keys = np.ravel_multi_index(tuple(coords), shape)
    del coords
    # Published COO files are canonical. Reject duplicates instead of changing
    # floating-point summation order during an alleged exact replication.
    if len(keys) > 1 and np.any(keys[1:] <= keys[:-1]):
        order = np.argsort(keys, kind='stable')
        keys, values = keys[order], values[order]
        if np.any(keys[1:] == keys[:-1]):
            raise ValueError('Duplicate COO coordinates require an explicit conversion')
    keys = keys.astype(np.uint32 if np.prod(shape) <= 2**32 else np.uint64)
    for name, array in [('keys.npy', keys), ('values.npy', values)]:
        temporary = destination/(name+'.tmp')
        with temporary.open('wb') as stream:
            np.save(stream, array, allow_pickle=False)
        os.replace(temporary, destination/name)
    info = {'schema': 1, 'source_sha256': expected, 'shape': shape,
            'dtype': str(values.dtype), 'nnz': len(values), 'fill_value': fill,
            'cache_sha256': {name: sha256(destination/name) for name in ('keys.npy','values.npy')},
            'cache_bytes': sum((destination/name).stat().st_size for name in ('keys.npy','values.npy')),
            'dense_bytes': int(np.prod(shape))*values.dtype.itemsize}
    temporary = destination/'manifest.json.tmp'
    temporary.write_text(json.dumps(info, indent=2)+'\n', encoding='utf-8')
    os.replace(temporary, manifest)
    return info


class CompactProbabilityTable:
    def __init__(self, source):
        directory, expected = cache_paths(source)
        manifest = directory/'manifest.json'
        if not manifest.exists():
            raise RuntimeError('Run python -m scenario_reconstruction.prepare_published_d2rl first')
        info = json.loads(manifest.read_text())
        if info['source_sha256'] != expected:
            raise ValueError('Incorrect table cache')
        self.shape = tuple(info['shape'])
        self.ndim = len(self.shape)
        self.dtype = np.dtype(info['dtype'])
        self.keys = np.load(directory/'keys.npy', mmap_mode='r', allow_pickle=False)
        self.values = np.load(directory/'values.npy', mmap_mode='r', allow_pickle=False)

    def __len__(self):
        return self.shape[0]

    @lru_cache(maxsize=4096)
    def _row(self, indices):
        linear = 0
        for value, size in zip(indices, self.shape[:-1]):
            if not isinstance(value, (int, np.integer)):
                raise TypeError('Prefix indices must be integers')
            value = int(value)
            if value < 0:
                value += size
            if not 0 <= value < size:
                raise IndexError(indices)
            linear = linear*size+value
        targets = np.array([2*linear, 2*linear+1], dtype=self.keys.dtype)
        positions = np.searchsorted(self.keys, targets)
        answer = np.zeros(2, dtype=self.dtype)
        inside = positions < len(self.keys)
        which = np.flatnonzero(inside)
        which = which[self.keys[positions[which]] == targets[which]]
        answer[which] = self.values[positions[which]]
        answer.flags.writeable = False
        return answer

    def __getitem__(self, index):
        if not isinstance(index, tuple) or len(index) != self.ndim:
            raise IndexError('Only a full state index and trailing action index/slice are supported')
        answer = self._row(tuple(index[:-1]))
        return answer[index[-1]]

    def to_dense(self):
        result = np.zeros(self.shape, dtype=self.dtype)
        result.ravel()[self.keys] = self.values
        result.flags.writeable = False
        return result


def load_table(source):
    table = CompactProbabilityTable(source)
    # The small OL table is iterable in the original controller.
    if table.ndim == 4 or os.environ.get('D2RL_TABLE_BACKEND') == 'dense':
        return table.to_dense()
    return table
