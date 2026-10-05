"""Frozen highD/Single/Dual temporal correlations on common six-second spells."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from .highd_v46_assets import ROOT, ensure_assets, read, save, digest
from .highd_v46_matched_score import score_states, persistent_windows, observations
from .highd_v46_matched_report import rho, bootstrap_weights

MODES = ('interval_1s', 'interval_1s_centered', 'instantaneous')
MEASURES = ('auto_front', 'auto_rear', 'front_to_rear')


def moments(x,y):
    return np.array([len(x),x.sum(),y.sum(),np.dot(x,x),np.dot(y,y),np.dot(x,y)])


def accumulate(intervals, instantaneous, lags):
    result = np.zeros((3,3,len(lags),6))
    if not len(intervals):return result
    intervals = np.asarray(intervals); instantaneous = np.asarray(instantaneous)
    arrays = [intervals, intervals-intervals.mean(axis=1,keepdims=True), instantaneous]
    for mode, data in enumerate(arrays):
        for index,lag in enumerate(lags):
            anchor = 6; shifted = anchor+round(2*lag)
            for measure,(first,second) in enumerate(((0,0),(1,1),(0,1))):
                result[mode,measure,index] = moments(data[:,anchor,first],data[:,shifted,second])
    return result


def arrays_from_window(velocities, accelerations, hz):
    source_time = np.arange(len(velocities))/hz
    grid = np.arange(13)/2
    speed = np.column_stack([np.interp(grid,source_time,velocities[:,i]) for i in (0,1)])
    interval = speed[2:]-speed[:-2]
    instant = np.column_stack([np.interp(grid[:11],source_time,accelerations[:,i]) for i in (0,1)])
    return interval,instant


def source_record(source, rec, frozen, settings):
    base = Path(source)/'data'
    hashes = {kind:digest(base/f'{rec}_{kind}.csv') for kind in ('tracks','tracksMeta','recordingMeta')}
    if hashes != frozen['source_sha256'][rec]:raise ValueError('Changed source recording')
    meta = pd.read_csv(base/f'{rec}_tracksMeta.csv')
    rows = pd.read_csv(base/f'{rec}_tracks.csv',usecols=['frame','id','laneId','xVelocity','yVelocity',
        'xAcceleration','precedingId','precedingXVelocity','dhw']).sort_values(['id','frame']).reset_index(drop=True)
    frame=rows.frame.to_numpy(); actor=rows.id.to_numpy(); leader=rows.precedingId.to_numpy()
    lane=rows.laneId.to_numpy(); speed=rows.xVelocity.abs().to_numpy(); gap=rows.dhw.to_numpy()
    direction=rows.id.map(dict(zip(meta.id,meta.drivingDirection))).to_numpy()
    acceleration=rows.xAcceleration.to_numpy()*np.where(direction==1,-1.,1.)
    supported=meta.loc[meta['class'].isin(['Car','Truck']) & (meta.numFrames>=75),'id']
    domain=rows.id.isin(supported).to_numpy() & (speed>=20) & (speed<=40)
    lookup=pd.MultiIndex.from_frame(rows[['frame','id']])
    front=lookup.get_indexer(pd.MultiIndex.from_arrays([frame,leader])); safe=np.maximum(front,0)
    quiet=rows.yVelocity.abs().to_numpy()<.2
    valid=(front>=0)&domain&domain[safe]&(lane==lane[safe])&(direction==direction[safe])&quiet&quiet[safe]
    valid &= (gap>0)&(gap<=115)&(np.abs(speed[safe]-speed)<=20)
    change=np.ones(len(rows),bool)
    change[1:]=((actor[1:]!=actor[:-1])|(frame[1:]!=frame[:-1]+1)|(leader[1:]!=leader[:-1])|
        (lane[1:]!=lane[:-1])|(lane[safe][1:]!=lane[safe][:-1])|
        (leader[safe][1:]!=leader[safe][:-1])|~valid[1:]|~valid[:-1])
    starts=np.flatnonzero(change); ends=np.r_[starts[1:],len(rows)]
    intervals=[]; instantaneous=[]; excluded=0; eligible=0
    width=150
    for left,right in zip(starts,ends):
        if not valid[left] or right-left<=width:continue
        first=((int(frame[left])+width-1)//width)*width
        for start in range(first,int(frame[right-1])-width+1,width):
            idx=left+start-int(frame[left]); positions=np.arange(idx,idx+width+1)
            v=np.column_stack([speed[safe[positions]],speed[positions]])
            a=np.column_stack([acceleration[safe[positions]],acceleration[positions]])
            interval,instant=arrays_from_window(v,a,25); eligible+=1
            if np.any(interval < -4-1e-9) or np.any(interval > 2+1e-9):excluded+=1;continue
            intervals.append(interval);instantaneous.append(instant)
    return accumulate(intervals,instantaneous,settings['lags_s']), {
        'source_sha256':hashes,'eligible_windows':eligible,'action_support_excluded':excluded,'accepted_windows':len(intervals)}


def simulation_windows(trace):
    windows=persistent_windows(trace,6)
    front,gap,rr,_=observations(trace)
    result={}
    for start,pairs in windows.items():
        lo=round(start*15);hi=lo+90+1
        result[start]={ (rear,lead) for rear,lead in pairs
            if np.all(front[lo:hi,lead]==front[lo,lead]) and np.all(np.abs(rr[lo:hi,rear])<=20)}
    return result


def simulation_pair(root,seed,settings):
    root=Path(root); report=read(root/f'pair_seed{seed}.json')
    arrays={}; windows={}; grid={}
    for arm in ('single','dual'):
        path=root/f'{arm}_seed{seed}.npz'
        if digest(path)!=report[arm]['trace_sha256']:raise ValueError('Trace changed')
        with np.load(path) as f:arrays[arm]={k:f[k] for k in ('states','actions')}
        windows[arm]=simulation_windows(arrays[arm]['states'])
        _,grid[arm]=score_states(arrays[arm]['states'],arrays[arm]['actions'],hz=2)
    cutoff=min(report[a]['elapsed_s'] for a in arrays)
    full=all(report[a]['complete'] for a in arrays)
    intervals={a:[] for a in arrays}; instants={a:[] for a in arrays}; accepted=0; excluded=0
    for start in sorted(set(windows['single'])&set(windows['dual'])):
        if start+6>cutoff+1e-9 or (not full and start+6>=cutoff-1e-9):continue
        for rear,lead in sorted(windows['single'][start]&windows['dual'][start]):
            group={}
            for arm in arrays:
                idx=round(start*2); ids=[lead,rear]
                v=grid[arm][idx:idx+13,:,2][:,ids]
                interval=v[2:]-v[:-2]
                times=start+np.arange(11)/2; a=arrays[arm]['actions'][np.floor(times).astype(int)][:,ids].copy()
                velocity=v[:11]
                a[((velocity>=40-1e-9)&(a>0))|((velocity<=20+1e-9)&(a<0))]=0.
                group[arm]=(interval,a)
            if any(np.any(v[0]<-4-1e-9) or np.any(v[0]>2+1e-9) for v in group.values()):
                excluded+=1;continue
            for arm,(interval,instant) in group.items():intervals[arm].append(interval);instants[arm].append(instant)
            accepted+=1
    return {a:accumulate(intervals[a],instants[a],settings['lags_s']) for a in arrays}, {
        'seed':seed,'common_windows':accepted,'action_support_excluded':excluded,'common_cutoff_s':cutoff}


def curves(groups,settings):
    result={}; boot={}
    for name,blocks in groups.items():
        blocks=np.asarray(blocks)
        weights=bootstrap_weights(len(blocks),settings['bootstrap_replicates'],
                                  settings['bootstrap_seed']+int(name=='highd'))
        samples=(weights@blocks.reshape(len(blocks),-1)).reshape(-1,3,3,len(settings['lags_s']),6)
        boot[name]=np.moveaxis(np.asarray([rho(samples[:,i,j,k]) for i in range(3) for j in range(3)
                                          for k in range(len(settings['lags_s']))]).reshape(3,3,-1,len(samples)), -1,0)
        total=blocks.sum(axis=0)
        result[name]=[]
        for i,mode in enumerate(MODES):
            for j,measure in enumerate(MEASURES):
                for k,lag in enumerate(settings['lags_s']):
                    r=float(rho(total[i,j,k])[0]); values=boot[name][:,i,j,k]
                    result[name].append({'mode':mode,'measure':measure,'lag_s':lag,'rho':r,
                        'ci95':np.nanquantile(values,[.025,.975]).tolist(),'windows':int(total[i,j,k,0])})
    differences=[]
    for i,mode in enumerate(MODES):
        for j,measure in enumerate(MEASURES):
            for k,lag in enumerate(settings['lags_s']):
                d=boot['dual'][:,i,j,k]-boot['single'][:,i,j,k]
                error=np.abs(boot['dual'][:,i,j,k]-boot['highd'][:,i,j,k])-np.abs(boot['single'][:,i,j,k]-boot['highd'][:,i,j,k])
                differences.append({'mode':mode,'measure':measure,'lag_s':lag,
                    'dual_minus_single_ci95':np.nanquantile(d,[.025,.975]).tolist(),
                    'dual_minus_single_absolute_error_vs_highd_ci95':np.nanquantile(error,[.025,.975]).tolist()})
    return result,differences


def plot(root,curves):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(13,7),constrained_layout=True)
    for row,mode in enumerate(('interval_1s','interval_1s_centered')):
        for col,measure in enumerate(MEASURES):
            ax=axes[row,col]
            for group,color in [('highd','#333333'),('single','#3266ad'),('dual','#d36926')]:
                points=[p for p in curves[group] if p['mode']==mode and p['measure']==measure and (p['lag_s']>=0 or measure=='front_to_rear')]
                x=np.array([p['lag_s'] for p in points]);y=np.array([p['rho'] for p in points]);ci=np.array([p['ci95'] for p in points])
                ax.plot(x,y,marker='o',label=group,color=color);ax.fill_between(x,ci[:,0],ci[:,1],color=color,alpha=.12)
            ax.set(title=measure.replace('_',' '),xlabel='Lag (s)',ylabel='Correlation'+(' (window-centered)' if row else ''))
            ax.axhline(0,color='gray',lw=.7);ax.grid(alpha=.2)
    axes[0,0].legend();fig.suptitle('Frozen V46 temporal diagnostic: forward 1 s acceleration; pointwise block-bootstrap 95% CI')
    fig.savefig(root/'temporal_correlations.png',dpi=160);fig.savefig(root/'temporal_correlations.pdf');plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default=str(ROOT/'configs/highd_v46_ni_temporal_v1.json'))
    p.add_argument('--source-root',required=True);p.add_argument('--rollout',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();cfg=read(a.config);settings=cfg['temporal'];assets,lock=ensure_assets()
    frozen=read(assets/'model/fit_summary.json');root=Path(a.output);root.mkdir(parents=True,exist_ok=False)
    groups={'highd':[],'single':[],'dual':[]};source_info={};simulation_info=[]
    for rec in frozen['train_recordings']:
        counts,info=source_record(a.source_root,rec,frozen,settings)
        groups['highd'].append(counts);source_info[rec]=info
        print({'recording':rec,'windows':info['accepted_windows']},flush=True)
    for index,seed in enumerate(range(40000,40100),1):
        counts,info=simulation_pair(a.rollout,seed,settings)
        for arm in ('single','dual'):groups[arm].append(counts[arm])
        simulation_info.append(info)
        if index%20==0:print({'episode_pairs':index},flush=True)
    values,differences=curves(groups,settings)
    np.savez_compressed(root/'block_moments.npz',**{k:np.asarray(v) for k,v in groups.items()})
    save(root/'summary.json',{'protocol':cfg,'config_sha256':digest(a.config),'implementation_sha256':digest(__file__),
        'natural_target_sha256':lock['natural_target_sha256'],'curves':values,'paired_differences':differences,
        'source_records':source_info,'simulation_pairs':simulation_info,
        'scope':'Diagnostic selected stable 6s common-pair windows. Positive lag does not identify causal reaction delay. Half-second overlapping measurement/held-action effects and highD trajectory smoothing must be considered.'})
    plot(root,values)
    print({'finished':True,'highd_windows':sum(i['accepted_windows'] for i in source_info.values()),
           'common_simulation_windows':sum(i['common_windows'] for i in simulation_info)},flush=True)


if __name__=='__main__':main()
