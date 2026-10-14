"""Verify published assets and prepare shared lookup caches; never run episodes."""
import argparse
import json
import urllib.request
from pathlib import Path
from .published_probability_tables import ROOT, prepare_table, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fetch-agent', action='store_true')
    args = parser.parse_args()
    lock = json.loads((ROOT/'configs/published_d2rl_assets.json').read_text())
    model = ROOT/'checkpoints/2lane_400m/model.pt'
    if args.fetch_agent and not model.exists():
        url = 'https://raw.githubusercontent.com/michigan-traffic-lab/Dense-Deep-Reinforcement-Learning/'
        url += lock['upstream_commit']+'/checkpoints/2lane_400m/model.pt'
        model.parent.mkdir(parents=True, exist_ok=True)
        data = urllib.request.urlopen(url, timeout=60).read()
        import hashlib
        if hashlib.sha256(data).hexdigest() != lock['files']['checkpoints/2lane_400m/model.pt']['sha256']:
            raise ValueError('Downloaded model differs from the published pin')
        with model.open('xb') as stream:
            stream.write(data)
    for relative, entry in lock['files'].items():
        path = ROOT/relative
        if not path.exists() or sha256(path) != entry['sha256']:
            raise ValueError('Missing/modified published asset: '+relative)
    tables = {}
    for source in sorted((ROOT/'source_data/NDD_DATA/LC').glob('*.npz')):
        tables[source.name] = prepare_table(source)
        print(json.dumps({'prepared':source.name,'cache_MB':tables[source.name]['cache_bytes']/1e6}),flush=True)
    output = ROOT/'outputs/published_d2rl_preparation.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({'assets_verified':len(lock['files']),'model_sha256':sha256(model),
        'upstream_commit':lock['upstream_commit'],'tables':tables},indent=2)+'\n')
    print('Ready: assets and shared caches verified. No simulation started.',flush=True)


if __name__ == '__main__':
    main()
