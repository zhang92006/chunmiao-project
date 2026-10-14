"""Bounded, auditable evaluation of the released two-lane D2RL agent."""
import argparse
import contextlib
import json
import os
from pathlib import Path
import random
import time
import traceback


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    os.replace(temporary, path)


def run(args):
    import sumo
    os.environ['SUMO_HOME'] = sumo.SUMO_HOME
    import numpy as np
    import psutil
    import torch
    import libsumo
    from conf import conf
    from envs.nade import NADE
    from controller.treesearchnadecontroller import TreeSearchNADEBackgroundController
    from mtlsp.simulator import Simulator
    from scenario_reconstruction.published_probability_tables import ROOT, sha256

    if args.lanes != 2 or args.av_model != 'IDM':
        raise ValueError('This entry point is pinned to the published two-lane IDM AV')
    torch.set_num_threads(1)
    conf.experiment_config.update(mode=args.mode, experiment_name=args.experiment_name, AV_model='IDM')
    conf.simulation_config.update(gui_flag=False, epsilon_setting='drl' if args.mode=='D2RL' else 'fixed')
    model = ROOT/'checkpoints/2lane_400m/model.pt'
    lock = json.loads((ROOT/'configs/published_d2rl_assets.json').read_text())
    if sha256(model) != lock['files']['checkpoints/2lane_400m/model.pt']['sha256']:
        raise ValueError('The agent does not match the released checkpoint')
    if args.mode == 'D2RL':
        conf.discriminator_agent = conf.load_discriminator_agent(checkpoint_path=str(model))
    root = Path(args.experiment_path) if args.experiment_path else ROOT/'data_analysis/raw_data'/('Experiment-'+args.experiment_name+'_'+time.strftime('%Y-%m-%d'))
    root.mkdir(parents=True, exist_ok=True)
    for name in ('crash','tested_and_safe','rejected','conf'):
        (root/name).mkdir(exist_ok=True)
    weight_path = root/f'weight{args.worker_id}.npy'
    result_path = root/f'worker{args.worker_id}.json'
    if weight_path.exists() or result_path.exists():
        raise FileExistsError('Refusing to overwrite a previous worker run: '+str(root))
    start = time.perf_counter()
    report = {'worker_id':args.worker_id, 'requested':args.episode_num, 'completed':0,
              'attempted':0,'errors':[], 'model_sha256':sha256(model),'mode':args.mode,
              'av_model':'IDM','lanes':2,'seed_base':args.seed,'sumo_version':libsumo.getVersion(),
              'upstream_commit':lock['upstream_commit'],'max_rss_bytes':0,'max_private_bytes':0,
              'policy_calls':0,'collision_episodes':0,'wall_seconds':0.,'probability_backend':os.environ.get('D2RL_TABLE_BACKEND','compact')}
    if args.mode == 'D2RL':
        original = conf.discriminator_agent.compute_action
        def counted(observation):
            report['policy_calls'] += 1
            return original(observation)
        conf.discriminator_agent.compute_action = counted
    weights = []
    process = psutil.Process()
    def checkpoint():
        temp = weight_path.with_suffix('.npy.tmp')
        with temp.open('wb') as stream:
            np.save(stream, np.asarray(weights,dtype=np.float64),allow_pickle=False)
        os.replace(temp,weight_path)
        report['wall_seconds']=time.perf_counter()-start
        atomic_json(result_path,report)
    try:
        for local in range(args.episode_num):
            episode = args.worker_id*args.episode_num+local
            seed = args.seed+episode
            np.random.seed(seed);random.seed(seed)
            conf.simulation_config['sumo_seed']=seed
            env = NADE(BVController=TreeSearchNADEBackgroundController,cav_model='IDM')
            sim = Simulator(sumo_net_file_path='./maps/2LaneHighway/2LaneHighway.net.xml',
                sumo_config_file_path='./maps/2LaneHighway/2LaneHighwayHighSpeed.sumocfg',
                num_tries=1,step_size=.1,action_step_size=.1,lc_duration=1,
                track_cav=False,sublane_flag=True,gui_flag=False,output=[],experiment_path=str(root))
            sim.bind_env(env)
            report['attempted'] += 1
            sim.run(episode)
            value = float(env.info_extractor.weight_result)
            if not np.isfinite(value) or value < 0:
                raise ValueError('Invalid collision importance weight')
            weights.append(value)
            report['completed'] += 1
            report['collision_episodes'] += int(1 in (sim.stop_reason or {}))
            mem=process.memory_info()
            report['max_rss_bytes']=max(report['max_rss_bytes'],mem.rss)
            report['max_private_bytes']=max(report['max_private_bytes'],getattr(mem,'private',mem.rss))
            if (local+1)%50==0:
                checkpoint()
                print(json.dumps({'worker':args.worker_id,'completed':len(weights),'wall_s':report['wall_seconds']}),flush=True)
    except BaseException as error:
        report['errors'].append({'at_attempt':report['attempted'],'error':repr(error),'traceback':traceback.format_exc()})
        raise
    finally:
        checkpoint()
        with contextlib.suppress(Exception):
            libsumo.close()
    print(json.dumps({'worker':args.worker_id,'completed':len(weights),'wall_s':report['wall_seconds']}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['NDE','D2RL'],default='D2RL')
    parser.add_argument('--lanes',type=int,choices=[2],default=2)
    parser.add_argument('--av_model',choices=['IDM'],default='IDM')
    parser.add_argument('--episode_num',type=int,default=20001)
    parser.add_argument('--worker_id',type=int,default=0)
    parser.add_argument('--experiment_name',default='published_d2rl_repro_long')
    parser.add_argument('--experiment_path')
    parser.add_argument('--seed',type=int,default=20261014)
    args=parser.parse_args()
    if args.episode_num < 1 or args.worker_id < 0:
        parser.error('Need positive episode count and nonnegative worker ID')
    run(args)


if __name__=='__main__':
    main()
