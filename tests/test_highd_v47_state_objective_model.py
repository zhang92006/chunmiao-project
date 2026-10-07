import copy
import json

import numpy as np
import pytest

from scenario_reconstruction import highd_v47_state_objective_model as m
from scenario_reconstruction.highd_v47_state_model_revision import MODEL_CONFIG


def rows():
    rs=[]
    for i in range(24):
        x=i/24; j=1e-4*np.exp(6*x)
        rs.append(dict(source_seed=260000+i,features=dict(names=['geometry','constant'],values=[x,0.],coefficient_mass=2.),
            J=j,variance=j*j/100,effective_measurement_roots=256.,batch_estimates=[dict(J=.8*j),dict(J=1.2*j)]))
    return rs


@pytest.mark.parametrize('beta',[np.array([0.,0.]),np.array([1.3,-.2]),np.array([-1.,2.])])
def test_direct_gradient_matches_finite_difference_with_floor(beta):
    z=np.array([[-1.,0.],[-.5,1.],[.2,-.7],[1.,.5]])
    a=np.array([.1,.2,.3,.4]); j=np.array([.001,.003,.1,2.])
    value,gradient=m.objective_gradient(beta,z,a,j,1/1.1,.1)
    numeric=[]
    for i in range(len(beta)):
        delta=np.eye(len(beta))[i]*1e-6
        numeric.append((m.objective_gradient(beta+delta,z,a,j,1/1.1,.1)[0]
            -m.objective_gradient(beta-delta,z,a,j,1/1.1,.1)[0])/2e-6)
    assert np.isfinite(value) and np.allclose(gradient,numeric,atol=2e-7,rtol=2e-5)


@pytest.mark.parametrize('method,reg',[('log_J',10.),('direct_M2',.1)])
def test_serializable_reproducible_feature_only_models(method,reg):
    rs=rows(); model=m.fit(rs,method,reg,MODEL_CONFIG); features=[r['features'] for r in rs]
    score=m.predict(model,features)
    assert model==m.fit(rs,method,reg,MODEL_CONFIG) and model==json.loads(json.dumps(model))
    assert model['optimizer']['success'] and np.all(score>0) and score[-1]>score[0]
    changed=[dict(r,source_seed=990000+i) for i,r in enumerate(rs)]
    assert np.array_equal(score,m.predict(m.fit(changed,method,reg,MODEL_CONFIG),features))
    q=m.waterfill(np.full(len(rs),1/len(rs)),score,1/1.1)
    assert np.min(q*len(rs))>=1/1.1-1e-13


def test_nested_outer_exclusion_includes_hyperparameter_selection():
    rs=rows(); cfg=dict(MODEL_CONFIG,log_grid=[1.,10.])
    fitted=m.nested_fit(rs,'log_J',cfg['log_grid'],cfg)
    altered=copy.deepcopy(rs)
    for r in altered:
        if fitted['outer_assignments'][str(r['source_seed'])]==0:
            r['J']*=10
            for b in r['batch_estimates']: b['J']*=10
    changed=m.nested_fit(altered,'log_J',cfg['log_grid'],cfg)
    old=fitted['outer_reports'][0]; new=changed['outer_reports'][0]
    assert old['model']==new['model'] and old['inner_selection']==new['inner_selection']
    assert [r['score'] for r in old['comparison']['rows']]==[r['score'] for r in new['comparison']['rows']]
    for fold in fitted['outer_reports']:
        checking=set(fold['checking_sources'])
        assert not checking & set(fold['model']['fitting_sources'])
        for report in fold['inner_selection']['grid_reports']:
            for inner in report['folds']:
                assert not checking & (set(inner['fitting_sources'])|set(inner['checking_sources']))


def test_aggregate_objective_not_unweighted_mean_of_fold_ratios():
    rs=rows(); fitted=m.nested_fit(rs,'direct_M2',[.1],MODEL_CONFIG)
    reports=fitted['outer_reports']
    base=sum(r['size']*r['comparison']['baseline_M2'] for r in reports)
    cand=sum(r['size']*r['comparison']['candidate_M2'] for r in reports)
    assert fitted['aggregate_ratio_point']==pytest.approx(cand/base)
    assert fitted['aggregate_ratio_point'] < 1
    # A favorable point objective alone cannot override any failed inner fit.
    if any(r['inner_selection']['status'] != 'selected' for r in reports):
        assert not fitted['development_gate_passed']


def test_zero_measurement_bank_neutral_and_never_passes():
    rs=rows()
    for r in rs:
        r['J']=0
        for b in r['batch_estimates']: b['J']=0
    fitted=m.nested_fit(rs,'direct_M2',[.1],MODEL_CONFIG)
    assert not fitted['development_gate_passed'] and fitted['aggregate_ratio_point'] is None
    assert np.array_equal(m.predict(fitted['model'],[r['features'] for r in rs]),np.ones(len(rs)))


def test_failed_optimizer_explicit_neutral_fallback(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(m,'minimize',lambda *a,**k:SimpleNamespace(success=False,message='fixture failure',nit=1,
        fun=1.,jac=np.ones(2),x=np.ones(2)))
    model=m.fit(rows(),'direct_M2',.1,MODEL_CONFIG)
    assert not model['optimizer']['success']
    assert model['coefficients']==[0.,0.]


def test_bad_source_or_contract_rejected():
    rs=rows()
    with pytest.raises(ValueError): m.fit(rs+[rs[0]],'direct_M2',.1,MODEL_CONFIG)
    model=m.fit(rs,'log_J',10.,MODEL_CONFIG); model['feature_names']=['source_seed']
    with pytest.raises(ValueError): m.predict(model,[rs[0]['features']])
