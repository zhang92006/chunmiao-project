"""Whole-road basic / optimized NDE rollout and fixed-bin distribution diagnostics."""
import argparse
from pathlib import Path
import platform

import numpy as np

from .highd_paper_data import histogram, read, sha, write
from .highd_paper_dynamics import EmpiricalModel, Road


def metrics(counts, reference):
    counts=np.asarray(counts,dtype=float);reference=np.asarray(reference,dtype=float)
    if counts.shape != reference.shape or reference.sum() <= 0:
        raise ValueError('Incompatible reference histogram')
    if counts.sum() <= 0:
        return {'hellinger':None,'probability_mae':None,'samples':0,'counts':counts.astype(int).tolist()}
    p=counts/counts.sum();q=reference/reference.sum()
    return {'hellinger':float(np.linalg.norm(np.sqrt(p)-np.sqrt(q))/np.sqrt(2)),
            'probability_mae':float(np.mean(np.abs(p-q))),
            'samples':int(counts.sum()),'counts':counts.astype(int).tolist()}


def run(model_root, output, seeds=None, duration=None, optimized=None, warmup=0.,reference=None,initial_bank=None,optimization_mode='ff',pair_model=None,pair_mode='fitted',lane_model=None,onset_model=None,timing_model=None,center_observation=False,lateral_model=None,strict_phase_support=False,reference_onset_support=False,temporal_model=None,surface_model=None,following_response_model=None,conditional_crossing_timing=False,lane_change_context=False):
    m=EmpiricalModel(model_root,optimized,optimization_mode);c=m.config
    if pair_model is not None and optimized is not None:
        raise ValueError('Following coupling is fitted against the frozen empirical baseline, not optimized marginals')
    if lane_model is not None and pair_model is None:
        raise ValueError('Lane extension requires the declared following-pair model')
    if onset_model is not None and lane_model is None:
        raise ValueError('Conditional onset requires lane-action runtime')
    if center_observation and onset_model is None:
        raise ValueError('Center observation currently requires explicit conditional onset candidate')
    if lateral_model is not None and (onset_model is None or not center_observation):
        raise ValueError('Lateral feedback requires onset model and center observations')
    if lateral_model is not None and timing_model is not None:
        raise ValueError('Choose either fixed timing or feedback execution, not both')
    if strict_phase_support and lane_model is None:
        raise ValueError('Phase support option requires lane model')
    if reference_onset_support and onset_model is None:
        raise ValueError('Reference onset support requires conditional onset model')
    if temporal_model is not None and (onset_model is None or not center_observation or lateral_model is not None):
        raise ValueError('Temporal candidate currently requires conditional onset, center observations and fixed-path execution')
    if surface_model is not None and (onset_model is None or not center_observation or lateral_model is not None or temporal_model is not None):
        raise ValueError('Surface candidate requires conditional onset, center observations, fixed path, and no temporal overlay')
    if following_response_model is not None and (onset_model is None or not center_observation or any(v is not None for v in (lateral_model,temporal_model,surface_model))):
        raise ValueError('Following response requires onset, center observations, fixed path, and no other quiet marginal overlay')
    if conditional_crossing_timing:
        if onset_model is None or not center_observation or any(v is not None for v in (lateral_model,temporal_model,surface_model,following_response_model)):
            raise ValueError('Conditional timing requires explicit onset, center observation and no other execution/marginal overlay')
        if read(Path(onset_model)/'onset_model.json').get('model_kind')!='censored_motion_commitment':
            raise ValueError('Conditional timing requires the fitted confirmation-time model')
    target_front_candidate=lane_model is not None and read(Path(lane_model)/'lane_model.json').get('lane_model_kind')=='actual_target_front_mixture'
    if target_front_candidate and (onset_model is None or not center_observation or not strict_phase_support or any(v is not None for v in (lateral_model,temporal_model,surface_model,following_response_model))):
        raise ValueError('Actual target-front model requires onset, center observations, strict phase support and no other marginal/execution overlay')
    if lane_change_context and not target_front_candidate:raise ValueError('Lane-change context diagnostic requires the declared target-front parent')
    if timing_model is not None:
        if lane_model is None or onset_model is None:
            raise ValueError('Timing candidate requires the conditional onset extension')
        timing=read(Path(timing_model)/'timing_model.json')
        if timing['base_model_sha256']!=m.summary['model_sha256'] or not np.isfinite(timing['duration_s']) or timing['duration_s']<=0:
            raise ValueError('Incompatible timing model')
        c['lane_change_duration_s']=float(timing['duration_s'])
    if reference is not None:
        ref=read(Path(reference)/'reference_summary.json')
        if ref['model_sha256']!=m.summary['model_sha256'] or sha(Path(reference)/'reference.npz')!=ref['reference_sha256']:
            raise ValueError('Reference artifact changed or belongs to a different model')
        with np.load(Path(reference)/'reference.npz') as arrays:
            m.arrays.update({key:arrays[key] for key in arrays.files})
    if 'reference_rr_counts' not in m.arrays:
        raise ValueError('Frozen model lacks rr reference; supply --reference from the train-only reference exporter')
    seeds=c['pilot_seeds'] if seeds is None else seeds
    duration=c['pilot_duration_s'] if duration is None else duration
    if optimization_mode=='ff_cf_candidate' and duration>60:
        raise ValueError('Unaccepted CF candidates are short diagnostic experiments only')
    dt=1/c['decision_hz']
    if not seeds or len(set(seeds)) != len(seeds) or duration <= 0 or warmup < 0 or warmup >= duration:
        raise ValueError('Distinct seeds and 0 <= warmup < duration required')
    root=Path(output);root.mkdir(parents=True,exist_ok=False)
    times=sorted(set([0.,float(duration)]+[float(t) for t in c['diagnostic_times_s']+[120,300,450,600,750,900] if t<=duration]))
    ticks={int(round(t/dt)):t for t in times}
    references={key:m.arrays['reference_'+key+'_counts'] for key in ('speed','gap','rr')}
    counts={str(t):dict({key:np.zeros_like(ref,dtype=np.uint64) for key,ref in references.items()},reached_runs=0) for t in times}
    collection={key:np.zeros_like(ref,dtype=np.uint64) for key,ref in references.items()}
    specs={'speed':(20,40,1),'gap':(0,115,1),'rr':(-20,20,1)}
    windows=[(left,right) for left,right in ((0.,10.),(10.,30.),(30.,60.)) if right<=duration]
    rows=[]
    for seed in seeds:
        snapshot=None if initial_bank is None else read(Path(initial_bank)/f'initial_seed{seed}.json')
        sampler=None
        road_class=Road
        if lane_model is not None:
            from .highd_paper_lane_runtime import LanePairs,LaneRoad
            sampler=LanePairs(pair_model,lane_model,m.summary['model_sha256'],pair_mode,c['lane_width_m'],onset_model,center_observation)
            sampler.strict_phase_support=strict_phase_support
            sampler.reference_onset_support=reference_onset_support
            road_class=LaneRoad
            if temporal_model is not None:
                from .highd_paper_temporal_runtime import TemporalPairs,TemporalRoad
                sampler=TemporalPairs(pair_model,lane_model,m.summary['model_sha256'],pair_mode,c['lane_width_m'],onset_model,center_observation,temporal_root=temporal_model)
                sampler.strict_phase_support=strict_phase_support
                sampler.reference_onset_support=reference_onset_support
                road_class=TemporalRoad
            if surface_model is not None:
                from .highd_paper_probability_surface import SurfacePairs,SurfaceRoad
                sampler=SurfacePairs(pair_model,lane_model,m.summary['model_sha256'],pair_mode,c['lane_width_m'],onset_model,center_observation,surface_root=surface_model)
                sampler.strict_phase_support=strict_phase_support
                sampler.reference_onset_support=reference_onset_support
                road_class=SurfaceRoad
            if following_response_model is not None:
                from .highd_paper_following_response_runtime import ResponsePairs
                sampler=ResponsePairs(pair_model,lane_model,m.summary['model_sha256'],pair_mode,c['lane_width_m'],onset_model,center_observation,response_root=following_response_model,empirical_model=m)
                sampler.strict_phase_support=strict_phase_support
                sampler.reference_onset_support=reference_onset_support
            if conditional_crossing_timing:
                from .highd_paper_conditional_timing import ConditionalTimingRoad
                road_class=ConditionalTimingRoad
            if target_front_candidate:
                from .highd_paper_target_front_runtime import TargetFrontRoad,TargetFrontTimingRoad
                road_class=TargetFrontTimingRoad if conditional_crossing_timing else TargetFrontRoad
            if lane_change_context:
                from .highd_paper_lane_context import ContextRoad,ContextTimingRoad
                road_class=ContextTimingRoad if conditional_crossing_timing else ContextRoad
        elif pair_model is not None:
            from .highd_paper_pairs import NativePairs
            sampler=NativePairs(pair_model,m.summary['model_sha256'],pair_mode)
        if lateral_model is not None:
            from .highd_paper_feedback_runtime import FeedbackRoad
            road=FeedbackRoad(m,seed,snapshot,sampler,lateral_model)
        else:
            road=road_class(m,seed,snapshot,sampler)
        if snapshot is None:
            snapshot={'seed':seed,'vehicles':[vars(car).copy() for car in road.vehicles],'diagnostics':road.initialization,
                      'model_sha256':m.summary['model_sha256'],'rng_state_after_initialization':road.rng.bit_generator.state}
        write(root/f'initial_seed{seed}.json',snapshot)
        timeline={}
        window_counts={f'{left:g}_{right:g}':{key:np.zeros_like(ref,dtype=np.uint64) for key,ref in references.items()} for left,right in windows}
        for tick in range(int(round(duration/dt))+1):
            sample=None
            if tick in ticks:
                sample=road.sample();slot=counts[str(ticks[tick])]
                timeline[str(ticks[tick])]={}
                for key in references:
                    observed=histogram(sample[key],*specs[key]).astype(np.uint64)
                    slot[key]+=observed
                    timeline[str(ticks[tick])][key]=metrics(observed,references[key])
                slot['reached_runs']+=1
            if tick*dt >= warmup-1e-9:
                sample=road.sample() if sample is None else sample
                for key in references:
                    collection[key]+=histogram(sample[key],*specs[key]).astype(np.uint64)
            active=[f'{left:g}_{right:g}' for left,right in windows if left-1e-9<=tick*dt<right-1e-9]
            if active:
                sample=road.sample() if sample is None else sample
                for key in references:
                    observed=histogram(sample[key],*specs[key]).astype(np.uint64)
                    for name in active:
                        window_counts[name][key]+=observed
            if tick==int(round(duration/dt)) or not road.step():
                break
        row={'seed':seed,'initialization':road.initialization,'elapsed_s':road.time,
             'complete':not bool(road.crashes) and getattr(road,'execution_failure',None) is None and road.time>=duration-1e-8,'collision_pairs':road.crashes,
             'execution_failure':getattr(road,'execution_failure',None),
             'collision_context':road.crash_context,
             'diagnostics':dict(road.diagnostics),'max_unbounded_sampled_executed_acceleration_error':road.max_action_error,
             'initial_fleet_sha256':sha(root/f'initial_seed{seed}.json'),'by_time':timeline}
        row['fixed_windows']={f'{left:g}_{right:g}':{
            'start_s':left,'end_s':right,'complete_exposure':road.time>=right-1e-8 and not ((road.crashes or getattr(road,'execution_failure',None)) and road.time<=right+1e-8),
            'metrics':{key:metrics(window_counts[f'{left:g}_{right:g}'][key],ref) for key,ref in references.items()}}
            for left,right in windows}
        if sampler is not None:
            write(root/f'pair_decisions_seed{seed}.json',{'seed':seed,'decisions':sampler.records})
            row['pair_audit']={'decision_count':len(sampler.records),
                'maximum_marginal_error':sampler.maximum_marginal_error,
                'maximum_simultaneous_pairs':max((len(r['pairs']) for r in sampler.records),default=0),
                'pair_additions':sum(len(r['added']) for r in sampler.records),
                'pair_removals':sum(len(r['removed']) for r in sampler.records),
                'maximum_selected_probability_reconstruction_error':getattr(sampler,'maximum_probability_error',None),
                'natural_only_q_equals_p':True}
        if lateral_model is not None:
            row['maneuver_events']=road.maneuver_events
            row['unfinished_lateral_states']=road.motion
        rows.append(row);write(root/f'run_seed{seed}.json',row)
        label=m.mode if pair_model is None else ('lane93_' if lane_model else 'dynamic_pairs_')+pair_mode
        print(f'mode={label} seed={seed} vehicles={len(road.vehicles)} elapsed={road.time:.1f} collision={bool(road.crashes)} execution_failure={getattr(road,"execution_failure",None)}',flush=True)
    scored={t:dict({key:metrics(slot[key],ref) for key,ref in references.items()},
                  reached_runs=slot['reached_runs'],requested_runs=len(seeds),full_cohort=slot['reached_runs']==len(seeds)) for t,slot in counts.items()}
    result={'schema_version':1,'mode':m.mode if pair_model is None else 'empirical_dynamic_following_pairs','model_sha256':m.summary['model_sha256'],
            'pair_model_sha256':None if pair_model is None else sha(Path(pair_model)/'pair_model.json'),
            'pair_dependence_mode':None if pair_model is None else pair_mode,
            'lane_model_sha256':None if lane_model is None else sha(Path(lane_model)/'lane_model.json'),
            'onset_model_sha256':None if onset_model is None else sha(Path(onset_model)/'onset_model.json'),
            'timing_model_sha256':None if timing_model is None else sha(Path(timing_model)/'timing_model.json'),
            'lateral_model_sha256':None if lateral_model is None else sha(Path(lateral_model)/'lateral_model.json'),
            'temporal_model_sha256':None if temporal_model is None else sha(Path(temporal_model)/'temporal_model.json'),
            'surface_model_sha256':None if surface_model is None else sha(Path(surface_model)/'pair_model.json'),
            'following_response_model_sha256':None if following_response_model is None else sha(Path(following_response_model)/'response_model.json'),
            'conditional_crossing_timing':conditional_crossing_timing,
            'lane_change_context':lane_change_context,
            'lane_change_duration_s':c['lane_change_duration_s'],
            'center_lane_observation':center_observation,
            'strict_phase_support':strict_phase_support,
            'reference_onset_support':reference_onset_support,
            'action_contract':'a_index*3+lateral_choice; 93 actions' if lane_model else '33 paper actions',
            'lateral_sha256':m.summary['lateral_sha256'],'full_declared_train':m.summary['full_declared_train'],
            'training_recordings':m.summary['train_recordings'],'seeds':seeds,'duration_s':duration,'warmup_s':warmup,
            'engine':'direct longitudinal kinematics + prescribed quintic lateral trajectory',
            'implementation_sha256':{'runner':sha(__file__),'dynamics':sha(Path(__file__).with_name('highd_paper_dynamics.py'))},
            'environment':{'python':platform.python_version(),'numpy':np.__version__},
            'by_time':scored,'collection':{key:metrics(collection[key],ref) for key,ref in references.items()},
            'reference_sha256':None if reference is None else ref['reference_sha256'],
            'initial_bank':None if initial_bank is None else str(initial_bank),
            'runs':rows,'complete_run_count':sum(row['complete'] for row in rows),
            'scope':'Train-target development diagnostic; not independent generalization. All vehicles initialized across two lanes, no donor / replay / learned joint action. Finite fleet on an unbounded straight road, no arrival process. Histograms include tail bins; gap is bumper-to-bumper for observed leaders within 115 m. Collision-terminated runs are reported; later snapshots are survivor-only and cannot be accepted as full-cohort fidelity. Pooled snapshots are correlated; no sample-count claim of independent evidence.'}
    if pair_model is not None:
        result['scope']='Train-only fitted following coupling with dynamic disjoint pairs, preserving both full 33-action single marginals. Unsupported lane interactions remain independent. Natural-only P=Q, no D2RL intervention. Fixed initialization and score bins; pilot only, not heldout behavioral acceptance. '+result['scope'].replace('no donor / replay / learned joint action','no donor or trajectory replay')
    if lane_model is not None:
        result['mode']='native_lane93_'+pair_mode
        result['scope']='New 93-action natural-law candidate. Quiet stay probabilities and lateral hazards preserved, lane-onset/active accelerations and lane dependence changed. Active phase fits plus onset pre-boundary proxy. No zero-acceleration lock, implicit safety brake, altered collision accounting or altered score bins. Finite fleet, same initial scenes; collision-truncated pooled scores are not long-run fidelity acceptance. Natural-only P=Q, no D2RL proposal.'
        result['implementation_sha256']['lane_runtime']=sha(Path(__file__).with_name('highd_paper_lane_runtime.py'))
    if onset_model is not None:
        result['scope']='Quiet P(a) exactly frozen; new learned conditional P(d|a,s) replaces quiet onset hazard and active-phase onset proxy. Full longitudinal rank kernel composed with lateral conditionals, matching fitted quiet acceleration marginals. '+result['scope']
        result['scope']=result['scope'].replace('Quiet stay probabilities and lateral hazards preserved, lane-onset/active accelerations and lane dependence changed. Active phase fits plus onset pre-boundary proxy.', 'Active phase fits retained, 1 s path unchanged.')
        if read(Path(onset_model)/'onset_model.json').get('model_kind')=='censored_motion_commitment':
            result['scope']+=' Conditional onset factors sustained motion initiation and competing crossing/return confirmation hazards. Incomplete source episodes contribute observed survival intervals. Censoring assumptions and omitted noncommitted lateral wiggles remain model limitations.'
            result['implementation_sha256']['onset_censoring']=sha(Path(__file__).with_name('highd_paper_onset_censoring.py'))
    if timing_model is not None:
        result['scope']=result['scope'].replace('1 s path unchanged','train-only median duration with symmetric quintic approximation')
    if conditional_crossing_timing:
        result['scope']=result['scope'].replace('train-only median duration with symmetric quintic approximation','twice the fitted conditional mean crossing time as quintic duration')
        result['scope']+=' Duration is a deterministic function of pre-onset geometry and selected action, fixed for that maneuver. No sampled acceleration alteration, subsequent collision check, or new probability factor. Symmetric path and conditional mean remain execution approximations.'
        result['engine']='direct longitudinal kinematics + conditional mean crossing-time quintic trajectory'
        result['implementation_sha256']['conditional_timing']=sha(Path(__file__).with_name('highd_paper_conditional_timing.py'))
    if center_observation:
        result['scope']+=' Control and pairing queries use center-lane membership, matching source laneId rather than assigning both lanes throughout the maneuver. Legacy scoring preserved for comparability; not a claim of calibrated lateral shape.'
    if lane_model is not None and 'interaction_marginals' in read(Path(lane_model)/'lane_model.json'):
        result['scope']=result['scope'].replace('Quiet P(a) exactly frozen;', 'Quiet P(a) frozen only outside selected active-lane interaction pairs;')
        result['scope']+=' Both members of a supported active-lane pair use contextual acceleration marginals and a refitted coupling, with explicitly logged 1% train-prior support. Not a marginal-preserving modification of the full original natural law.'
    if lateral_model is not None:
        lateral=read(Path(lateral_model)/'lateral_model.json')
        result['lateral_probability_contract']={k:lateral[k] for k in ('action_axis','coefficients','sigma','model_kind','gating_coefficients','kp','damping_ratio','launch_model','mode_persistence','phase_goal_coefficients','progress_force_coefficients') if k in lateral}
        result['engine']='direct longitudinal kinematics + sampled lateral acceleration feedback'
        result['implementation_sha256']['feedback_runtime']=sha(Path(__file__).with_name('highd_paper_feedback_runtime.py'))
        if 'mode_persistence' in lateral:
            result['implementation_sha256']['lateral_memory']=sha(Path(__file__).with_name('highd_paper_lateral_memory.py'))
            result['scope']+=' Maneuver memory is a causal latent-mode posterior updated from sampled actions; action probability integrates modes, not one selected component.'
        result['action_contract']+=' + conditional lateral acceleration bins for active/launching cars'
        result['scope']+=' Fixed-time lateral path REPLACED by sampled state-responsive lateral accelerations, with explicit conditional probability included in scene likelihood. Returns and holds possible; returning longitudinal state uses reference backoff. Road departures are failed runs, not deleted samples. Original score bins and legacy score sampling retained.'
    if strict_phase_support:
        result['scope']+=' Additional phase/interaction marginal tilts and lane dependence apply only to directly populated empirical states; unsupported actors retain their queried reference law. This is support backoff, not a collision-conditioned action veto.'
    if reference_onset_support:
        result['scope']+=' Conditional onset retains original reference exclusion of an alongside target-lane vehicle. This changes the initiation action support BEFORE sampling and is included in its normalized PMF, not an executed-action safety override. Active maneuvers are not interrupted. Empirical positive labels excluded by this rule require separate reporting.'
    if temporal_model is not None:
        result['scope']=result['scope'].replace('Quiet P(a) frozen only outside selected active-lane interaction pairs;', 'Quiet longitudinal law is history-conditioned when supported;')
        result['scope']+=' Explicit P(a_t|s_t,a_previous) for quiet stable histories, with declared 1% train-prior support and refitted following kernels. Mixed-history pairs use independence; lane-pair marginals stay on the declared lane model. Initial history absent; no smoothing after sampling. Temporal inputs and selected conditional lateral probabilities are logged for separate probability reconstruction.'
        result['implementation_sha256']['temporal_runtime']=sha(Path(__file__).with_name('highd_paper_temporal_runtime.py'))
    if surface_model is not None:
        result['scope']=result['scope'].replace('Quiet P(a) frozen only outside selected active-lane interaction pairs;', 'Quiet longitudinal PMFs use an explicit continuous surface after stable history;')
        result['scope']+=' Trilinear mixture of frozen empirical corner PMFs and actual-state fallback mass at unsupported corners. Applied before sampling, no sampled/executed action interpolation. Original 33-action reference and selected lane-interaction marginal laws retained; following coupling refitted. Mixed-surface pairs use independence.'
        result['implementation_sha256']['surface_runtime']=sha(Path(__file__).with_name('highd_paper_probability_surface.py'))
    if following_response_model is not None:
        result['scope']=result['scope'].replace('Quiet P(a) frozen only outside selected active-lane interaction pairs;', 'Quiet P(a) additionally depends on both current states for eligible selected following pairs;')
        result['scope']+=' Following role tilts preserve zero action support; both states must be directly populated and quiet with 3 s causal history. Coupling refitted on these marginals. Same new marginals in independent control; lateral conditionals and execution retained.'
        result['implementation_sha256']['following_response']=sha(Path(__file__).with_name('highd_paper_following_response.py'))
        result['implementation_sha256']['following_response_runtime']=sha(Path(__file__).with_name('highd_paper_following_response_runtime.py'))
    if target_front_candidate:
        result['scope']=result['scope'].replace('Active phase fits retained, ', '').replace('Both members of a supported active-lane pair use contextual acceleration marginals and a refitted coupling, with explicitly logged 1% train-prior support.', 'The quiet member of a supported active-lane pair retains contextual acceleration with declared 1% train-prior support; the focal member uses the target-front mixture without that prior.')
        result['scope']+=' Active focal acceleration uses a fitted convex mixture of current and actual target-front frozen reference laws, separately from the selected lane partner. This replaces all older focal phase/interaction corrections. No focal prior added; unknown target states retain the current reference. Partner correction and lane dependence retain the fitted direct-support rule.'
        result['implementation_sha256']['target_front']=sha(Path(__file__).with_name('highd_paper_target_front.py'))
        result['implementation_sha256']['target_front_runtime']=sha(Path(__file__).with_name('highd_paper_target_front_runtime.py'))
    if lane_change_context:
        result['scope']+=' Three-actor context is established in the sampled onset step and persists until completion. Target rear sees the incoming car from the next decision (0.1 s clock), before center crossing, even outside selected pairs. Receiver PMF mixes frozen current/virtual leader laws with w=clipped lateral progress. Old selected-partner tilt is bypassed for these receivers. No parameter fitting, joint-kernel change, additional random draw, acceleration override or lane-change veto. This is an uncalibrated relationship diagnostic, not an accepted natural-law model.'
        result['implementation_sha256']['lane_context']=sha(Path(__file__).with_name('highd_paper_lane_context.py'))
    write(root/'rollout_summary.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True);p.add_argument('--output',required=True)
    p.add_argument('--optimized');p.add_argument('--seeds',type=int,nargs='+');p.add_argument('--pair-model')
    p.add_argument('--pair-mode',choices=['fitted','independent_control'],default='fitted')
    p.add_argument('--lane-model')
    p.add_argument('--onset-model')
    p.add_argument('--timing-model')
    p.add_argument('--center-observation',action='store_true')
    p.add_argument('--lateral-model')
    p.add_argument('--strict-phase-support',action='store_true')
    p.add_argument('--reference-onset-support',action='store_true')
    p.add_argument('--temporal-model')
    p.add_argument('--surface-model')
    p.add_argument('--following-response-model')
    p.add_argument('--conditional-crossing-timing',action='store_true')
    p.add_argument('--lane-change-context',action='store_true')
    p.add_argument('--reference');p.add_argument('--initial-bank');p.add_argument('--optimization-mode',choices=['ff','ff_cf','ff_cf_candidate'],default='ff')
    p.add_argument('--duration-s',type=float);p.add_argument('--warmup-s',type=float,default=0.)
    args=p.parse_args();r=run(args.model,args.output,args.seeds,args.duration_s,args.optimized,args.warmup_s,args.reference,args.initial_bank,args.optimization_mode,args.pair_model,args.pair_mode,args.lane_model,args.onset_model,args.timing_model,args.center_observation,args.lateral_model,args.strict_phase_support,args.reference_onset_support,args.temporal_model,args.surface_model,args.following_response_model,args.conditional_crossing_timing,args.lane_change_context)
    print({t:{k:slot[k]['hellinger'] for k in ('speed','gap','rr')} for t,slot in r['by_time'].items()})
