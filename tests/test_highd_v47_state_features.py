import copy
import json

import numpy as np
import pytest

from scenario_reconstruction import highd_v47_state_features as m
from scenario_reconstruction.highd_v47_allocation import waterfill


def fixture():
    state = dict(seed=260000,time_s=10.,cav_id=5,vehicles=[
        dict(id=5,x=100.,v=24.,target=-1),dict(id=1,x=90.,v=25.,target=-1),dict(id=2,x=115.,v=22.,target=1)])
    p = np.array([[.4,.1],[.2,.3]]); qa = np.array([[.1,.2],[.3,.4]]); risk = np.array([[.1,.2],[.3,.9]])
    return state,p,qa,risk


def rows():
    result = []
    for i in range(24):
        x = np.zeros(len(m.FEATURE_NAMES)); x[1] = i/24
        result.append(dict(source_seed=260000+i,features=dict(names=list(m.FEATURE_NAMES),values=x.tolist(),coefficient_mass=2.),
            J=2e-5*np.exp(8*i/24),effective_measurement_roots=256.))
    return result


def config():
    return dict(folds=4,fold_seed=260017,ridge_grid=[1.,10.,100.],prior_roots=16.,probability_floor=1e-8,allocation_loss_cap=1.1)


def test_feature_algebra_and_identity_invariance():
    s,p,q,r = fixture(); f = m.state_features(s,[1,2],p,q,r); c = p*p/q
    assert f['coefficient_mass']==pytest.approx(c.sum())
    assert f['values'][1]==pytest.approx(np.sum(c*r)/c.sum())
    assert f['values'][3]==pytest.approx(c[1,1]/c.sum())
    assert f['values'][-2:]==[1.,1.]
    changed = copy.deepcopy(s); changed.update(seed=999999,time_s=51.,cav_id=50)
    for v in changed['vehicles']: v['id']*=10; v['x']+=1000.
    assert f==m.state_features(changed,[10,20],p,q,r)
    assert f==m.state_features(s,[2,1],p.T,q.T,r.T)


@pytest.mark.parametrize('field', ['support','normalization','nan','risk','ids'])
def test_invalid_feature_laws_rejected(field):
    s,p,q,r = fixture(); ids=[1,2]
    if field=='support': q[0,0]=0
    if field=='normalization': p*=2
    if field=='nan': r[0,0]=np.nan
    if field=='risk': r[0,0]=-1
    if field=='ids': ids=[1,1]
    with pytest.raises(ValueError): m.state_features(s,ids,p,q,r)


def test_log_fit_is_positive_serializable_and_reproducible():
    rs = rows(); model = m.fit(rs,10.); pred = m.predict(model,[r['features'] for r in rs])
    assert np.all(pred>0) and pred[-1]>pred[0]
    assert model==m.fit(rs,10.)
    changed = [dict(r,source_seed=999000+i) for i,r in enumerate(rs)]
    assert np.array_equal(pred,m.predict(m.fit(changed,10.),[r['features'] for r in rs]))
    assert m.predict(model,[]).size==0


def test_zero_observations_regularized_without_claiming_zero_native_risk():
    rs = rows(); rs[0]['J']=0
    model = m.fit(rs,1.); pred=m.predict(model,[r['features'] for r in rs])
    assert pred[0]>0 and model['probability_floor']==1e-8
    a=np.full(len(rs),1/len(rs)); q=waterfill(a,pred,1/1.1)
    assert np.min(q/a)>=1/1.1-1e-14 and np.max(a/q)<=1.1+1e-14


def test_whole_source_fold_exclusion_includes_scaler_and_prior():
    rs=rows(); cfg=config(); fitted=m.grouped_fit(rs,cfg)
    assert len(fitted['OOF_predictions'])==len(rs)
    assert fitted==json.loads(json.dumps(fitted))  # Locked model round-trip must reproduce.
    for report in fitted['cross_fitted_reports']:
        for fold in report['folds']:
            assert not set(fold['checking_sources'])&set(fold['fitting_sources'])
            assert len(fold['checking_sources'])+len(fold['fitting_sources'])==len(rs)
    # Change every outcome in a checking fold. That fold's own OOF predictions
    # remain EXACTLY unchanged, including prior/scaling; no validation fitting.
    assignment=fitted['fold_assignments']; changed=copy.deepcopy(rs)
    for r in changed:
        if assignment[str(r['source_seed'])]==0: r['J']*=100
    fold0=[r for r in rs if assignment[str(r['source_seed'])]==0]
    training=[r for r in rs if assignment[str(r['source_seed'])]!=0]
    training2=[r for r in changed if assignment[str(r['source_seed'])]!=0]
    for ridge in cfg['ridge_grid']:
        assert m.fit(training,ridge)==m.fit(training2,ridge)
        assert np.array_equal(m.predict(m.fit(training,ridge),[r['features'] for r in fold0]),
            m.predict(m.fit(training2,ridge),[r['features'] for r in fold0]))


def test_empty_allzero_duplicate_source_and_bad_contract_fail():
    with pytest.raises(ValueError): m.fit([],1.)
    rs=rows(); rs.append(rs[0])
    with pytest.raises(ValueError): m.fit(rs,1.)
    rs=rows()
    for r in rs: r['J']=0
    with pytest.raises(ValueError,match='No training fold'): m.grouped_fit(rs,config())
    model=m.fit(rows(),1.); model['feature_names']=['source_seed']
    with pytest.raises(ValueError): m.predict(model,[rows()[0]['features']])
