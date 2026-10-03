"""Train-only empirical behavior and whole-road initialization distributions.

No imports from SUMO, frozen NDD, donor retrieval, or joint-BV controllers.
highD dhw is bumper gap; rr is leader speed minus subject speed.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter
from scipy.optimize import least_squares, minimize


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def axis(spec):
    return np.linspace(spec[0], spec[1], round((spec[1]-spec[0])/spec[2])+1)


def index(values, grid):
    return np.rint((np.asarray(values)-grid[0])/(grid[1]-grid[0])).astype(int)


def clock(frames, source_hz, decision_hz):
    return ((frames+1)*decision_hz)//source_hz > (frames*decision_hz)//source_hz


def idm(parameters, speed, gap, rr):
    a, v0, delta, b, s0, headway = parameters
    desired = s0+np.maximum(0., speed*headway-speed*rr/(2*np.sqrt(a*b)))
    interaction = np.where(np.isfinite(gap), (desired/np.maximum(gap, .05))**2, 0.)
    return a*(1-(speed/v0)**delta-interaction)


def smooth_counts(counts, width, eligible=None):
    """Smooth frequencies BEFORE normalization; empty states remain explicit."""
    values = counts.astype(float)
    if eligible is not None:
        values *= eligible[..., None]
    values = uniform_filter(values, size=width, mode='constant', cval=0.)
    # Running-sum roundoff must not turn truly empty count cells into tiny
    # signed "supported" distributions after row normalization.
    values[np.abs(values) < 1e-10] = 0.
    values = np.maximum(values,0.)
    if eligible is not None:
        values *= eligible[..., None]
    total = values.sum(axis=-1, keepdims=True)
    return np.divide(values, total, out=np.zeros_like(values), where=total > 0).astype(np.float32)


def inevitable(gap, rr, braking):
    return gap <= np.minimum(rr, 0.)**2/(2*braking)


def onset_labels(tracks, config):
    """Offline future lane crossing identifies labels; runtime consumes no future."""
    result = np.zeros(len(tracks), np.int8)
    available = clock(tracks.frame.to_numpy(), config['source_hz'], config['decision_hz'])
    fps = config['source_hz']
    for _, part in tracks.groupby('id', sort=False):
        ids = part.index.to_numpy()
        frames, lanes = part.frame.to_numpy(), part.laneId.to_numpy()
        vy = np.abs(part.yVelocity.to_numpy())
        for cross in np.flatnonzero(lanes[1:] != lanes[:-1])+1:
            if abs(lanes[cross]-lanes[cross-1]) != 1:
                continue
            start = cross
            lower = frames[cross]-round(config['lane_onset_max_lookback_s']*fps)
            while start > 0 and frames[start-1] >= lower and vy[start-1] >= config['lane_onset_lateral_speed_mps']:
                start -= 1
            candidates = np.flatnonzero(available[ids[:start+1]])
            if not len(candidates):
                continue
            decision = int(candidates[-1])
            left = (part.direction.iloc[0] == 1 and lanes[cross] > lanes[cross-1]) or (part.direction.iloc[0] == 2 and lanes[cross] < lanes[cross-1])
            result[ids[decision]] = 1 if left else -1
            end = np.searchsorted(frames, frames[cross]+round(config['lane_completion_tail_s']*fps), side='right')
            available[ids[decision+1:end]] = False
    return result, available


def context_key(speed, gap, rr, front, rear, config):
    """Four target-lane neighbor cases; missing neighbors aren't distant cars."""
    vstep, gstep = config['lateral_state_speed_step_mps'], config['lateral_state_gap_step_m']
    code = int(front is not None)*2+int(rear is not None)
    vals = [code, round(speed/vstep), round(gap/gstep), round(rr/vstep)]
    if front is not None:
        vals.extend([round(front[0]/gstep), round(front[1]/vstep)])
    if rear is not None:
        vals.extend([round(rear[0]/gstep), round(rear[1]/vstep)])
    return tuple(vals)


def histogram(values, first, last, step):
    edges = first-step/2+np.arange(round((last-first)/step)+2)*step
    return np.bincount(np.searchsorted(np.round(edges, 10), np.round(values, 10), side='right'), minlength=len(edges)+1)


def fit(source_root, config_path, output, recordings=None):
    c = read(config_path)
    recs = list(c['train_recordings']) if recordings is None else list(recordings)
    if not recs or len(set(recs)) != len(recs) or set(recs)-set(c['train_recordings']):
        raise ValueError('Unique declared TRAIN recordings required')
    root = Path(output); root.mkdir(parents=True, exist_ok=False)
    v, g, rr, a = [axis(c[k+'_grid']) for k in ('speed', 'gap', 'rr', 'acceleration')]
    cf = np.zeros((len(v), len(g), len(rr), len(a)), np.uint32)
    ff = np.zeros((len(v), len(a)), np.uint32)
    init = np.zeros((len(v), len(g), len(rr)), np.uint32)
    speeds, gaps = np.zeros(len(v), np.uint64), np.zeros(118, np.uint64)
    eval_speed = np.zeros(23, np.uint64)
    eval_rr = np.zeros(43, np.uint64)
    lateral = {}; totals = Counter(); sources = {}; fit_samples = []; mobil_samples = []
    rng = np.random.default_rng(1901)
    columns = ['frame','id','x','y','width','height','xVelocity','yVelocity','xAcceleration',
        'dhw','precedingXVelocity','precedingId','frontSightDistance','laneId',
        'leftPrecedingId','leftAlongsideId','leftFollowingId','rightPrecedingId','rightAlongsideId','rightFollowingId']
    for rec in recs:
        base = Path(source_root)/'data'
        sources[rec] = {kind: sha(base/(rec+'_'+kind+'.csv')) for kind in ('tracks','tracksMeta','recordingMeta')}
        meta = pd.read_csv(base/(rec+'_recordingMeta.csv')).iloc[0]
        if int(meta.frameRate) != c['source_hz']:
            raise ValueError('Unexpected source clock')
        tm = pd.read_csv(base/(rec+'_tracksMeta.csv'))
        supported = tm.loc[tm['class'].isin(c['vehicle_classes']) & (tm.numFrames >= c['minimum_track_s']*c['source_hz']), 'id']
        t = pd.read_csv(base/(rec+'_tracks.csv'), usecols=columns)
        # Preserve unsupported neighbors for geometry; only subjects are filtered.
        direction = dict(zip(tm.id, tm.drivingDirection))
        t['direction'] = t.id.map(direction)
        t = t.sort_values(['id','frame']).reset_index(drop=True)
        source_lanes = {int(d): set(part.laneId.unique()) for d, part in t.groupby('direction')}
        labels, decisions = onset_labels(t, c)
        t['lane_action'] = labels
        change = t.id.ne(t.id.shift()) | t.precedingId.ne(t.precedingId.shift()) | t.frame.ne(t.frame.shift()+1)
        segment = change.cumsum()
        lengths = segment.groupby(segment).transform('size').to_numpy()
        rows = t.loc[clock(t.frame.to_numpy(), c['source_hz'], c['decision_hz']) & t.id.isin(supported)].copy()
        speed = np.abs(rows.xVelocity.to_numpy())
        acc = rows.xAcceleration.to_numpy()*np.where(rows.direction.to_numpy()==1, -1., 1.)
        gap = rows.dhw.to_numpy(); rate = np.abs(rows.precedingXVelocity.to_numpy())-speed
        domain = np.isfinite(speed) & (speed >= v[0]) & (speed <= v[-1])
        has = rows.precedingId.to_numpy() > 0
        observed = has | (rows.frontSightDistance.to_numpy() >= g[-1])
        following = has & np.isfinite(gap) & (gap > 0) & (gap <= g[-1])
        free = observed & ~following
        supported_cf = following & np.isfinite(rate) & (rate >= rr[0]) & (rate <= rr[-1])
        leader_domain = (speed+rate >= v[0]) & (speed+rate <= v[-1])
        vi = index(speed[domain], v); np.add.at(speeds, vi, 1)
        eval_speed += histogram(speed[domain], 20., 40., 1.).astype(np.uint64)
        gap_ref = domain & following
        gaps += histogram(gap[gap_ref], 0., 115., 1.).astype(np.uint64)
        eval_rr += histogram(rate[gap_ref], -20.,20.,1.).astype(np.uint64)
        totals['observed_regime_denominator'] += int((domain & observed).sum())
        totals['following_regime_numerator'] += int((domain & observed & following).sum())
        totals['censored_leader_absence_rows'] += int((domain & ~observed).sum())
        init_mask = domain & supported_cf & leader_domain
        np.add.at(init, (index(speed[init_mask],v), index(gap[init_mask],g), index(rate[init_mask],rr)), 1)
        longitudinal = domain & np.isfinite(acc) & (acc >= a[0]) & (acc <= a[-1]) & (np.abs(rows.yVelocity.to_numpy()) < c['lane_onset_lateral_speed_mps'])
        stable = lengths[rows.index.to_numpy()] >= c['minimum_stable_leader_s']*c['source_hz']
        mask = longitudinal & supported_cf & stable
        np.add.at(cf, (index(speed[mask],v),index(gap[mask],g),index(rate[mask],rr),index(acc[mask],a)),1)
        mask_ff = longitudinal & free & stable
        np.add.at(ff,(index(speed[mask_ff],v),index(acc[mask_ff],a)),1)
        totals['cf_rows'] += int(mask.sum()); totals['ff_rows'] += int(mask_ff.sum())
        sample = np.flatnonzero(mask | mask_ff)
        sample = rng.choice(sample, size=min(len(sample), max(1,c['idm_max_fit_samples']//len(recs))), replace=False)
        fit_samples.append(np.column_stack([speed[sample],np.where(following[sample],gap[sample],np.inf),rate[sample],acc[sample]]))
        lookup = t.set_index(['frame','id'])
        eligible_indices = rows.index.to_numpy()[domain & following & decisions[rows.index.to_numpy()]]
        raw = t.loc[eligible_indices]
        for side in ('left','right'):
            # Lookup neighbors vectorially; missing IDs and opposite directions remain absent.
            nbs = {}
            for role in ('Preceding','Following','Alongside'):
                ids = raw[side+role+'Id'].to_numpy(int)
                aligned = lookup.reindex(pd.MultiIndex.from_arrays([raw.frame.to_numpy(),ids]))
                nbs[role] = aligned
            # Use arrays in the frequency loop: no full-record scans or pandas
            # row construction per candidate. Keep a bounded fallback fit sample.
            vectors = {role: part[['x','width','xVelocity','direction']].to_numpy()
                       for role, part in nbs.items()}
            selected = set(rng.choice(len(raw), size=min(len(raw), max(1,c['idm_max_fit_samples']//(2*len(recs)))), replace=False))
            for i, row in enumerate(raw.itertuples()):
                delta = (1 if row.direction==1 else -1)*(1 if side=='left' else -1)
                if row.laneId+delta not in source_lanes[int(row.direction)]:
                    continue
                if np.isfinite(vectors['Alongside'][i,0]):
                    totals['lateral_alongside_excluded'] += 1
                    continue
                sv = abs(row.xVelocity); sg = row.dhw; sr = abs(row.precedingXVelocity)-sv
                sign = -1 if row.direction==1 else 1
                subject_front = sign*(row.x+row.width/2)+row.width/2
                neighbors = []
                for role in ('Preceding','Following'):
                    nx, nl, nv, nd = vectors[role][i]
                    if not np.isfinite(nx) or nd != row.direction:
                        neighbors.append(None); continue
                    nb_front = sign*(nx+nl/2)+nl/2
                    ng = nb_front-nl-subject_front if role=='Preceding' else subject_front-row.width-nb_front
                    if ng < 0 or ng > g[-1]:
                        neighbors.append(None); continue
                    nr = abs(nv)-sv if role=='Preceding' else sv-abs(nv)
                    neighbors.append((float(ng),float(nr)))
                key = context_key(sv,sg,sr,*neighbors,c)
                positive = int(row.lane_action == (1 if side=='left' else -1))
                entry = lateral.setdefault(key,[0,0]); entry[0] += 1; entry[1] += positive
                front, rear = neighbors
                if i in selected:
                    mobil_samples.append([sv,sg,sr,np.inf if front is None else front[0],0. if front is None else front[1],
                                          np.inf if rear is None else rear[0],0. if rear is None else rear[1],positive])
        print(f'train_recording={rec} cf={totals["cf_rows"]} ff={totals["ff_rows"]} lateral_states={len(lateral)}',flush=True)
    sample = np.concatenate(fit_samples)
    if not len(sample) or not totals['observed_regime_denominator'] or not init.sum():
        raise ValueError('No usable longitudinal / initialization training support')
    def residual(par):
        return np.clip(idm(par,sample[:,0],sample[:,1],sample[:,2]),a[0],a[-1])-sample[:,3]
    fit_idm = least_squares(residual,[.8,37.,3.,1.3,.1,.8],bounds=([.1,25.,1.,.3,0.,.1],[2.,45.,6.,4.,5.,3.]),max_nfev=150)
    parameters = fit_idm.x.tolist()
    std = float(np.sqrt(np.mean(residual(fit_idm.x)**2)))
    active_bounds = fit_idm.active_mask.tolist()
    # MOBIL gain parameters are fitted only to train exposure. No SPMD fallback table.
    ms = np.asarray(mobil_samples).reshape(-1,8)
    my = ms[:,7]
    old = np.clip(idm(fit_idm.x,ms[:,0],ms[:,1],ms[:,2]),a[0],a[-1])
    new = np.clip(idm(fit_idm.x,ms[:,0],ms[:,3],ms[:,4]),a[0],a[-1])
    # Declared stochastic MOBIL-inspired fallback: subject incentive and target
    # follower acceleration, fitted on highD. Not the complete MOBIL formula.
    follower = np.where(np.isfinite(ms[:,5]), np.clip(idm(fit_idm.x,ms[:,0]-ms[:,6],ms[:,5],ms[:,6]),a[0],a[-1]),0.)
    mx = np.column_stack([new-old,follower])
    def loss(par):
        gain = mx[:,0]+par[0]*mx[:,1]
        score = np.clip((gain-par[1])/par[2]+par[3],-30,30)
        return float(np.mean(np.logaddexp(0,score)-my*score))
    mobil = minimize(loss,[.1,.2,1.,-5.],bounds=[(0.,1.),(-2.,2.),(.1,5.),(-15.,0.)],method='L-BFGS-B') if len(my) else None
    admissible = ~inevitable(g[:,None],rr[None,:],c['inevitable_state_deceleration_mps2'])
    pcf = smooth_counts(cf,c['moving_average_width'],np.broadcast_to(admissible,cf.shape[:-1]))
    pff = smooth_counts(ff,c['moving_average_width'])
    np.savez_compressed(root/'empirical.npz',speed_axis=v,gap_axis=g,rr_axis=rr,acceleration_axis=a,
        cf_counts=cf,ff_counts=ff,initial_joint_counts=init,initial_speed_counts=speeds,
        cf_probability=pcf,ff_probability=pff,reference_speed_counts=eval_speed,reference_gap_counts=gaps,reference_rr_counts=eval_rr)
    write(root/'lateral.json',{'states':[[list(key),value] for key,value in lateral.items()],
        'onset':'backtrack contiguous lateral motion before crossing; align to preceding decision tick',
        'alongside_policy':'excluded empirical; MOBIL fallback', 'smoothing':'axis-neighbor moving average frequencies'})
    summary = {'config':c,'train_recordings':recs,'full_declared_train':recs==c['train_recordings'],
        'source_sha256':sources,'counts':dict(totals),'initial_following_probability':totals['following_regime_numerator']/totals['observed_regime_denominator'],
        'model_sha256':sha(root/'empirical.npz'),'lateral_sha256':sha(root/'lateral.json'),
        'implementation_sha256':sha(__file__),'idm':{'parameters':parameters,'residual_std':std,'converged':bool(fit_idm.success),'fit_rows':len(sample),
                                                  'active_bounds':active_bounds,'identifiability_warning':bool(np.any(fit_idm.active_mask))},
        'mobil':{'parameters':mobil.x.tolist() if mobil else [0.,.2,1.,-15.], 'converged':bool(mobil.success) if mobil else False,'fit_exposures':len(my),'fit_positive_onsets':int(my.sum()),
                 'contract':'highD-fitted stochastic MOBIL-inspired incentive hazard; not a full calibrated deterministic MOBIL'},
        'scope':'Train-only marginal actions and initialization. Observed no-leader states require sufficient front sight. Neighbor-context transport across road lane counts remains an assumption. Population speed preservation after sequential generation is not guaranteed.'}
    write(root/'fit_summary.json',summary)
    return summary


def export_reference(source_root, model_root, output):
    """Add rr reference to an existing frozen model without refitting actions."""
    model_root=Path(model_root);summary=read(model_root/'fit_summary.json');c=summary['config']
    speed=np.zeros(23,np.uint64);gap=np.zeros(118,np.uint64);rr=np.zeros(43,np.uint64)
    for rec in summary['train_recordings']:
        base=Path(source_root)/'data'
        for kind in ('tracks','tracksMeta','recordingMeta'):
            if sha(base/(rec+'_'+kind+'.csv'))!=summary['source_sha256'][rec][kind]:
                raise ValueError('Training source changed after model fitting')
        tm=pd.read_csv(base/(rec+'_tracksMeta.csv'))
        supported=tm.loc[tm['class'].isin(c['vehicle_classes'])&(tm.numFrames>=c['minimum_track_s']*c['source_hz']),'id']
        t=pd.read_csv(base/(rec+'_tracks.csv'),usecols=['frame','id','xVelocity','dhw','precedingXVelocity','precedingId'])
        rows=t.loc[clock(t.frame.to_numpy(),c['source_hz'],c['decision_hz'])&t.id.isin(supported)]
        v=np.abs(rows.xVelocity.to_numpy());g=rows.dhw.to_numpy();rate=np.abs(rows.precedingXVelocity.to_numpy())-v
        domain=(v>=20)&(v<=40)&np.isfinite(v)
        cf=domain&(rows.precedingId.to_numpy()>0)&np.isfinite(g)&(g>0)&(g<=115)
        speed+=histogram(v[domain],20,40,1).astype(np.uint64)
        gap+=histogram(g[cf],0,115,1).astype(np.uint64)
        rr+=histogram(rate[cf],-20,20,1).astype(np.uint64)
    with np.load(model_root/'empirical.npz') as fitted:
        if not np.array_equal(speed,fitted['reference_speed_counts']) or not np.array_equal(gap,fitted['reference_gap_counts']):
            raise ValueError('Reference exposure differs from original frozen fit')
    root=Path(output);root.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(root/'reference.npz',reference_speed_counts=speed,reference_gap_counts=gap,reference_rr_counts=rr)
    r={'model_sha256':summary['model_sha256'],'reference_sha256':sha(root/'reference.npz'),
       'speed_samples':int(speed.sum()),'following_samples':int(gap.sum()),'rr_samples':int(rr.sum()),
       'speed_gap_reference_identical_to_frozen_fit':True,'train_recordings':summary['train_recordings'],
       'scope':'Exact original reference exposure; rr uses -20..20 m/s unit bins plus two tails. No probability model refit.'}
    write(root/'reference_summary.json',r)
    return r


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--config',default='configs/highd_paper_native_v1.json');p.add_argument('--recordings',nargs='+')
    p.add_argument('--reference-model')
    args=p.parse_args()
    if args.reference_model:
        print(export_reference(args.source_root,args.reference_model,args.output))
    else:
        result=fit(args.source_root,args.config,args.output,args.recordings)
        print(json.dumps({k:result[k] for k in ('train_recordings','full_declared_train','counts','initial_following_probability','idm','mobil')},indent=2))
