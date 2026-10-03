"""Independent empirical NDE: simultaneous actions, direct longitudinal dynamics.

An unbounded two-lane straight road with a finite initialized fleet. No SUMO,
trajectory replay, donor patches, joint-action model or implicit safety driver.
Positions are front bumpers. Lateral motion is a prescribed quintic path, not
the paper's bicycle model; speed bounds and collision termination are explicit.
"""
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.special import expit, ndtr

from .highd_paper_data import context_key, idm, index, read, sha


def normalize(values):
    values = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(values)) or np.any(values < 0) or values.sum() <= 0:
        raise ValueError('Invalid probability mass')
    return values / values.sum()


@dataclass
class Vehicle:
    id: int
    x: float
    v: float
    lane: int
    length: float = 5.
    width: float = 2.
    y: float = 0.
    target: int = -1
    elapsed: float = 0.


class EmpiricalModel:
    def __init__(self, root, optimized=None, optimization_mode='ff'):
        self.root = Path(root)
        self.summary = read(self.root/'fit_summary.json')
        self.config = self.summary['config']
        if sha(self.root/'empirical.npz') != self.summary['model_sha256'] or sha(self.root/'lateral.json') != self.summary['lateral_sha256']:
            raise ValueError('Model artifact differs from recorded checksum')
        with np.load(self.root/'empirical.npz') as arrays:
            self.arrays = {key: arrays[key] for key in arrays.files}
        self.v, self.g, self.rr, self.a = [self.arrays[k+'_axis'] for k in ('speed','gap','rr','acceleration')]
        self.lateral = {tuple(key): value for key,value in read(self.root/'lateral.json')['states']}
        self.mode = 'empirical'
        if optimized is not None:
            report = read(Path(optimized)/'optimization_summary.json')
            if optimization_mode not in ('ff','ff_cf','ff_cf_candidate'):
                raise ValueError('Unknown optimization mode')
            required=(report.get('ff_usable',report['usable']) if optimization_mode=='ff' else
                      report.get('ff_cf_candidate_usable',False) if optimization_mode=='ff_cf_candidate' else
                      report.get('ff_cf_usable',False))
            if report['input_model_sha256'] != self.summary['model_sha256'] or not required:
                raise ValueError('Unusable optimization or incompatible empirical model')
            if 'optimized_model_sha256' in report and sha(Path(optimized)/'optimized.npz')!=report['optimized_model_sha256']:
                raise ValueError('Optimized model checksum mismatch')
            with np.load(Path(optimized)/'optimized.npz') as opt:
                self.arrays['ff_probability'] = opt['ff_only_probability'] if optimization_mode=='ff' and 'ff_only_probability' in opt else opt['ff_probability']
                if optimization_mode in ('ff_cf','ff_cf_candidate'):
                    self.arrays['cf_probability']=opt['cf_probability']
            self.mode = 'candidate_ff_cf' if optimization_mode=='ff_cf_candidate' else 'stationary_ff_cf' if optimization_mode=='ff_cf' else 'stationary_ff_only'
        self.initial_speed_probability = normalize(self.arrays['initial_speed_counts'])

    def fallback(self, v, g=np.inf, rr=0.):
        par = self.summary['idm']
        mean = float(np.clip(idm(par['parameters'],v,g,rr), self.a[0], self.a[-1]))
        std = max(par['residual_std'], .05)
        edges = np.r_[-np.inf, (self.a[:-1]+self.a[1:])/2, np.inf]
        return normalize(np.diff(ndtr((edges-mean)/std)))

    def longitudinal(self, v, gap=None, rr=0.):
        vi = int(np.clip(index(v,self.v),0,len(self.v)-1))
        if gap is None or gap > self.g[-1]:
            p = self.arrays['ff_probability'][vi]
            status = 'empirical_ff' if self.mode == 'empirical' else 'stationary_ff'
        elif 0 <= gap <= self.g[-1] and self.rr[0] <= rr <= self.rr[-1]:
            p = self.arrays['cf_probability'][vi,int(index(gap,self.g)),int(index(rr,self.rr))]
            status = 'candidate_cf' if self.mode=='candidate_ff_cf' else 'stationary_cf' if self.mode=='stationary_ff_cf' else 'empirical_cf'
        else:
            p = np.zeros(len(self.a))
            status = 'cf_out_of_support'
        if p.sum() <= 0:
            return self.fallback(v,np.inf if gap is None else gap,rr), 'idm_fallback'
        return normalize(p), status

    @lru_cache(maxsize=100000)
    def lateral_frequency(self, key):
        exposure, events = self.lateral.get(key,[0,0])
        radius = self.config['lateral_smoothing_radius']
        # Axis-neighbor smoothing is deliberately sparse, rather than a huge
        # hypercube with exponential work in the eight-dimensional contexts.
        for feature in range(1,len(key)):
            for offset in range(-radius,radius+1):
                if offset:
                    neighbor = list(key); neighbor[feature] += offset
                    n,e = self.lateral.get(tuple(neighbor),[0,0])
                    exposure += n; events += e
        return (events/exposure, 'empirical_lateral') if exposure else (None,'mobil_fallback')

    def lateral_probability(self, v, gap, rr, front, rear):
        key = context_key(v,gap,rr,front,rear,self.config)
        probability, status = self.lateral_frequency(key)
        if probability is not None:
            return probability,status
        par = self.summary['idm']['parameters']
        old = float(np.clip(idm(par,v,gap,rr),self.a[0],self.a[-1]))
        new = float(np.clip(idm(par,v,np.inf if front is None else front[0],0. if front is None else front[1]),self.a[0],self.a[-1]))
        follower = 0. if rear is None else float(np.clip(idm(par,v-rear[1],rear[0],rear[1]),self.a[0],self.a[-1]))
        politeness,threshold,temperature,intercept = self.summary['mobil']['parameters']
        return float(expit((new-old+politeness*follower-threshold)/temperature+intercept)),status

    @lru_cache(maxsize=101)
    def initialization_distribution(self, vi):
        # Conditional frequencies retain gap / relative-speed dependence.
        counts = self.arrays['initial_joint_counts'][vi].astype(float)
        status = 'exact_speed_bin'
        if not counts.sum():
            lo,hi = max(0,vi-1),min(len(self.v),vi+2)
            counts = self.arrays['initial_joint_counts'][lo:hi].sum(axis=0).astype(float)
            status = 'adjacent_speed_smoothing'
        if not counts.sum():
            return None,status
        # Domain / positive-gap rejection conditions are applied BEFORE sampling;
        # this is an explicit conditional renormalization, not clipped values.
        admissible = (self.g[:,None] > 0) & (self.v[vi]+self.rr[None,:] >= self.v[0]) & (self.v[vi]+self.rr[None,:] <= self.v[-1])
        counts *= admissible
        if not counts.sum():
            return None,'no_admissible_conditional_support'
        return normalize(counts.ravel()),status


def initialize(model, rng):
    c = model.config
    vehicles, diagnostics = [],Counter()
    for lane in range(c['lane_count']):
        x = float(rng.uniform(0,c['initial_zone_m']))
        v = float(rng.choice(model.v,p=model.initial_speed_probability))
        while x < c['initial_span_m']:
            vehicles.append(Vehicle(len(vehicles),x,v,lane,c['vehicle_length_m'],c['vehicle_width_m'],lane*c['lane_width_m']))
            if rng.random() < model.summary['initial_following_probability']:
                vi = int(index(v,model.v))
                probability,status = model.initialization_distribution(vi)
                diagnostics['following_'+status] += 1
                if probability is None:
                    raise RuntimeError('Whole-road initialization lacks conditional support; no independent-gap substitution permitted')
                gi,ri = np.unravel_index(rng.choice(len(probability),p=probability),(len(model.g),len(model.rr)))
                gap,rate = float(model.g[gi]),float(model.rr[ri])
                x += gap+c['vehicle_length_m']; v += rate
            else:
                diagnostics['free'] += 1
                x += model.g[-1]+rng.uniform(0,c['initial_zone_m'])+c['vehicle_length_m']
                v = float(rng.choice(model.v,p=model.initial_speed_probability))
    diagnostics['vehicles'] = len(vehicles)
    diagnostics['initial_collisions'] = len(collisions(vehicles))
    if diagnostics['initial_collisions']:
        raise RuntimeError('Invalid initial geometry')
    return vehicles,dict(diagnostics)


def collisions(vehicles):
    result = []
    for i,left in enumerate(vehicles):
        for right in vehicles[i+1:]:
            longitudinal = min(left.x,right.x)-max(left.x-left.length,right.x-right.length)
            if longitudinal > 1e-9 and abs(left.y-right.y) < (left.width+right.width)/2-1e-9:
                result.append([left.id,right.id])
    return result


def neighbors(vehicles, subject, lane, observation=115.):
    # Source and target lanes both count while a maneuver is active.
    candidates = [car for car in vehicles if car.id != subject.id and (car.lane == lane or car.target == lane)]
    ahead = sorted((car for car in candidates if car.x > subject.x),key=lambda car:car.x)
    behind = sorted((car for car in candidates if car.x <= subject.x),key=lambda car:car.x,reverse=True)
    front = ahead[0] if ahead else None
    rear = behind[0] if behind else None
    fg = None if front is None else front.x-front.length-subject.x
    rg = None if rear is None else subject.x-subject.length-rear.x
    alongside = (fg is not None and fg < 0) or (rg is not None and rg < 0)
    f = (fg,front.v-subject.v) if fg is not None and fg <= observation else None
    r = (rg,subject.v-rear.v) if rg is not None and rg <= observation else None
    return f,r,alongside


def advance(vehicles, accelerations, dt, config):
    """Execute constant sampled acceleration exactly until an explicit speed bound."""
    bounded = 0; max_error = 0.; total_distance = 0.
    lower,upper = config['speed_grid'][:2]
    for car,acc in zip(vehicles,accelerations):
        before = car.v
        proposed = before+acc*dt
        after = float(np.clip(proposed,lower,upper))
        if abs(proposed-after) > 1e-10:
            bounded += 1
            time_to_bound = (after-before)/acc
            distance = before*time_to_bound+.5*acc*time_to_bound**2+after*(dt-time_to_bound)
        else:
            distance = before*dt+.5*acc*dt**2
            max_error = max(max_error,abs((after-before)/dt-acc))
        car.v = after; car.x += distance; total_distance += distance
        if car.target >= 0:
            car.elapsed = min(config['lane_change_duration_s'],car.elapsed+dt)
            z = car.elapsed/config['lane_change_duration_s']
            fraction = 10*z**3-15*z**4+6*z**5
            car.y = (car.lane+(car.target-car.lane)*fraction)*config['lane_width_m']
            if z >= 1-1e-12:
                car.lane = car.target; car.target = -1; car.elapsed = 0.
    return bounded,max_error,total_distance


class Road:
    def __init__(self, model, seed, initial_snapshot=None, pair_sampler=None):
        self.model=model; self.rng=np.random.default_rng(seed)
        if initial_snapshot is None:
            self.vehicles,self.initialization = initialize(model,self.rng)
        else:
            if initial_snapshot['seed']!=seed or initial_snapshot['model_sha256']!=model.summary['model_sha256']:
                raise ValueError('Initial fleet bank does not match the seed/model')
            self.vehicles=[Vehicle(**row) for row in initial_snapshot['vehicles']]
            self.initialization=initial_snapshot['diagnostics']
            self.rng.bit_generator.state=initial_snapshot['rng_state_after_initialization']
        self.diagnostics=Counter(); self.max_action_error=0.; self.time=0.
        self.crashes=[]; self.crash_context=[]
        self.pair_sampler=pair_sampler

    def action_pmfs(self):
        m=self.model;c=m.config;pmfs={}
        # All queries use the same pre-step state, including all target lanes.
        for car in self.vehicles:
            if car.target >= 0:
                self.diagnostics['lane_change_locked'] += 1
                continue
            front,_,_ = neighbors(self.vehicles,car,car.lane,m.g[-1])
            gap,rate = (None,0.) if front is None else front
            p,status=m.longitudinal(car.v,gap,rate); self.diagnostics[status] += 1
            sides=[]
            if front is not None and gap > 0:
                for delta in (1,-1):
                    target=car.lane+delta
                    if not 0 <= target < c['lane_count']:
                        sides.append(0.); continue
                    tf,tr,alongside=neighbors(self.vehicles,car,target,m.g[-1])
                    if alongside:
                        self.diagnostics['lateral_overlap_unavailable'] += 1
                        sides.append(0.); continue
                    probability,lstatus=m.lateral_probability(car.v,gap,rate,tf,tr)
                    self.diagnostics[lstatus] += 1; sides.append(probability)
            else:
                sides=[0.,0.]; self.diagnostics['no_leader_no_lane_change'] += 1
            if sum(sides)>1:
                sides=list(np.asarray(sides)/sum(sides))
                self.diagnostics['lateral_renormalization'] += 1
            pmfs[car.id]=normalize(np.r_[sides[0],(1-sum(sides))*p,sides[1]])
        return pmfs

    def step(self):
        m=self.model; c=m.config; actions=[]; changes=[]
        before={car.id:vars(car).copy() for car in self.vehicles}
        pmfs=self.action_pmfs()
        decisions=(self.pair_sampler.draw(self.vehicles,pmfs,self.rng) if self.pair_sampler is not None else
                   {car.id:int(self.rng.choice(len(pmfs[car.id]),p=pmfs[car.id])) for car in self.vehicles if car.id in pmfs})
        for car in self.vehicles:
            if car.id not in decisions:
                actions.append(0.);continue
            decision=decisions[car.id]
            if decision in (0,len(m.a)+1):
                changes.append((car,car.lane+(1 if decision==0 else -1)))
                actions.append(0.); self.diagnostics['lane_change_onsets'] += 1
            else:
                actions.append(float(m.a[decision-1]))
        for car,target in changes:
            car.target=target;car.elapsed=0.
        self.last_step_actions={car.id:float(action) for car,action in zip(self.vehicles,actions)}
        bounded,error,distance=advance(self.vehicles,actions,1/c['decision_hz'],c)
        self.diagnostics['speed_bound_actions'] += bounded
        self.diagnostics['vehicle_decisions'] += len(self.vehicles)
        self.diagnostics['distance_m'] += distance
        self.max_action_error=max(self.max_action_error,error)
        self.time += 1/c['decision_hz']
        self.crashes=collisions(self.vehicles)
        if self.crashes:
            involved={identifier for pair in self.crashes for identifier in pair}
            self.crash_context=[{'before':before[car.id],'after':vars(car).copy(),
                                 'sampled_acceleration':float(action),
                                 'lane_change_started_this_step':any(changed.id==car.id for changed,_ in changes)}
                                for car,action in zip(self.vehicles,actions) if car.id in involved]
        return not bool(self.crashes)

    def sample(self):
        speed=[car.v for car in self.vehicles]
        gap=[];rate=[];all_gap=[]
        for car in self.vehicles:
            front,_,_=neighbors(self.vehicles,car,car.lane,np.inf)
            if front is not None:
                all_gap.append(front[0])
                if 0 < front[0] <= self.model.g[-1]:
                    gap.append(front[0]);rate.append(front[1])
        return {'speed':speed,'gap':gap,'rr':rate,'all_gap':all_gap}
