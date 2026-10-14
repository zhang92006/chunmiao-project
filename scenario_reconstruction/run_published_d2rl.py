"""Prepare by default. --run explicitly launches the requested bounded batch."""
import argparse
import datetime
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import psutil
from .published_probability_tables import ROOT


def check_runtime():
    if sys.version_info[:2] != (3, 9):
        raise RuntimeError('Use the isolated Python 3.9 replication environment')
    versions = {}
    for line in (ROOT/'configs/requirements-d2rl-baseline.txt').read_text().splitlines():
        if '==' not in line or line.startswith('#'):
            continue
        package, expected = line.split('==')
        actual = importlib.metadata.version(package)
        if actual != expected:
            raise RuntimeError(f'{package}: expected {expected}, installed {actual}; run setup first')
        versions[package] = actual
    return versions


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--workers',type=int,default=10)
    parser.add_argument('--episodes-per-worker',type=int,default=20001)
    parser.add_argument('--name',default='published_d2rl_repro_long')
    parser.add_argument('--seed',type=int,default=20261014)
    a=parser.parse_args()
    if not (1<=a.workers<=32) or a.episodes_per_worker<1:
        parser.error('Invalid worker/episode count')
    versions = check_runtime()
    subprocess.run([sys.executable,'-m','scenario_reconstruction.prepare_published_d2rl','--fetch-agent'],cwd=ROOT,check=True)
    import sumo
    environment=os.environ.copy()
    environment.update(SUMO_HOME=sumo.SUMO_HOME,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1',D2RL_TABLE_BACKEND='compact')
    plan={'workers':a.workers,'episodes_per_worker':a.episodes_per_worker,
          'total_episodes':a.workers*a.episodes_per_worker,'run_requested':bool(a.run),
          'simulation_started':False,'seed_base':a.seed,'packages':versions,
          'available_memory_GiB':psutil.virtual_memory().available/2**30,
          'python':sys.version,'sumo_home':sumo.SUMO_HOME}
    print(json.dumps(plan,indent=2),flush=True)
    if not a.run:
        return
    minimum=a.workers*256*2**20+512*2**20
    if psutil.virtual_memory().available<minimum:
        raise RuntimeError(f'Need at least {minimum/2**30:.2f} GiB available RAM for {a.workers} workers; close other applications or defer the run. No batch started.')
    # Keep shell and PowerShell wrappers behaviorally identical, with collision
    # handling and acceleration semantics checked against the installed SUMO.
    subprocess.run([sys.executable,'-m','pytest','tests/test_simulator_acceleration.py','-q'],cwd=ROOT,env=environment,check=True)
    stamp=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).strftime('%Y-%m-%d_%H%M%S')
    root=ROOT/'data_analysis/raw_data'/('Experiment-'+a.name+'_'+stamp)
    root.mkdir(parents=True,exist_ok=False)
    logs=ROOT/'outputs/two_lane_replication'/(a.name+'_'+stamp)
    logs.mkdir(parents=True,exist_ok=False)
    (logs/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    (root/'batch_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    processes=[];streams=[];start=time.perf_counter();samples=[]
    try:
        for worker in range(a.workers):
            stream=(logs/f'worker_{worker}.log').open('w',encoding='utf-8');streams.append(stream)
            cmd=[sys.executable,'main.py','--mode','D2RL','--lanes','2','--av_model','IDM',
                 '--episode_num',str(a.episodes_per_worker),'--worker_id',str(worker),'--seed',str(a.seed),
                 '--experiment_name',a.name,'--experiment_path',str(root)]
            processes.append(subprocess.Popen(cmd,cwd=ROOT,env=environment,stdout=stream,stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0))
        plan['simulation_started']=True
        for destination in (logs/'plan.json',root/'batch_plan.json'):
            destination.write_text(json.dumps(plan,indent=2)+'\n')
        last_print=0
        while any(p.poll() is None for p in processes):
            failed=[p for p in processes if p.poll() not in (None,0)]
            if failed:
                raise RuntimeError('A worker failed; see logs: '+str(logs))
            rss=private=0
            seen=set()
            for process in processes:
                if process.poll() is None:
                    try:
                        parent=psutil.Process(process.pid)
                        # Windows venv python.exe is a small redirector; the
                        # actual interpreter is its child, which must be counted.
                        for active in [parent]+parent.children(recursive=True):
                            if active.pid in seen:
                                continue
                            seen.add(active.pid)
                            memory=active.memory_info()
                            rss+=memory.rss;private+=getattr(memory,'private',memory.rss)
                    except psutil.NoSuchProcess:
                        pass
            elapsed=time.perf_counter()-start
            samples.append({'seconds':elapsed,'rss_sum_bytes':rss,'private_sum_bytes':private,
                            'available_bytes':psutil.virtual_memory().available})
            if elapsed-last_print>=30:
                print(json.dumps({'elapsed_s':round(elapsed,1),'active_workers':sum(p.poll() is None for p in processes),
                    'rss_sum_MiB':round(rss/2**20,1),'private_sum_MiB':round(private/2**20,1)}),flush=True)
                last_print=elapsed
            time.sleep(.5)
        if any(p.returncode!=0 for p in processes):
            raise RuntimeError('At least one worker failed')
    finally:
        for p in processes:
            if p.poll() is None:
                # Stop only this batch's workers, including venv children.
                try:
                    for child in psutil.Process(p.pid).children(recursive=True):
                        child.terminate()
                except psutil.NoSuchProcess:
                    pass
                p.terminate()
        for p in processes:
            p.wait()
        for stream in streams:
            stream.close()
        (logs/'memory.json').write_text(json.dumps(samples,indent=2)+'\n')
    from .d2rl_baseline_summary import summarize
    summary=summarize(root,plan['total_episodes'])
    summary.update(batch_wall_seconds=time.perf_counter()-start,
                   peak_sum_rss_MiB=max((s['rss_sum_bytes'] for s in samples),default=0)/2**20,
                   peak_sum_private_MiB=max((s['private_sum_bytes'] for s in samples),default=0)/2**20,
                   minimum_available_GiB=min((s['available_bytes'] for s in samples),default=0)/2**30)
    (logs/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (logs/'weights').mkdir()
    for p in root.glob('weight*.npy'):
        shutil.copyfile(p,logs/'weights'/p.name)
    print(json.dumps(summary,indent=2),flush=True)
    print('RESULT_FOLDER '+str(logs),flush=True)


if __name__=='__main__':
    main()
