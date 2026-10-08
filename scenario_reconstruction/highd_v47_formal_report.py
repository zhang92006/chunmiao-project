"""Render all paired-seed curves from an audited FinalTest; no native sampling."""
import argparse
from importlib.metadata import version
from pathlib import Path
from .highd_v46_assets import read,save,digest


def render(result,audit,output):
    root=Path(result);summary=read(root/'summary.json');verified=read(audit)
    if (not summary['complete'] or not verified['verified'] or verified['native_replays']!=13
            or verified['summary_sha256']!=digest(root/'summary.json') or verified['lock']!=summary['lock']):
        raise ValueError('Run audit-final --replay against this complete FinalTest before rendering')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cfg=summary['lock']['config'];seeds=cfg['training']['seeds'];pairs=summary['paired_seed_comparisons']
    if set(pairs)!=set(map(str,seeds)):raise ValueError('Every predeclared seed must appear')
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    fig,axes=plt.subplots(len(seeds),3,figsize=(13,2.7*len(seeds)),squeeze=False,constrained_layout=True)
    for row,seed in enumerate(seeds):
        for mode,color in (('single','#3366aa'),('dual','#c05535')):
            curve=pairs[str(seed)][mode]['convergence_curves']['0.3']
            for col,(field,title) in enumerate((('M2','Event M2'),('estimator_variance','Estimator variance'),('relative_half_width','90% RHW'))):
                points=[p for p in curve if p[field] is not None and p[field]>0]
                ax=axes[row,col]
                if points:ax.plot([p['tests'] for p in points],[p[field] for p in points],label=mode.title(),color=color)
                ax.set_yscale('log');ax.set_title('Seed %s — %s'%(seed,title));ax.set_xlabel('Original-reset episodes');ax.grid(alpha=.2)
                if col==2:
                    for target in cfg['statistics']['rhw_targets']:ax.axhline(target,color='grey',linewidth=.6,linestyle=':')
        axes[row,0].legend()
    fig.suptitle('Matched FinalTest: all paired seeds (plug-in curves, not variance certificates)')
    fig.savefig(out/'paired_convergence.png',dpi=160);fig.savefig(out/'paired_convergence.svg');plt.close(fig)
    table=[]
    for seed in seeds:
        pair=pairs[str(seed)]
        table.append(dict(seed=seed,M2_ratio_point=pair['M2_ratio_point'],
            M2_ratio_interval=pair['paired_source_M2_ratio_interval'],bootstrap_censored=pair['bootstrap_censored'],
            variance_ratio_point=pair['estimator_variance_ratio_point'],AF_observed=pair['acceleration_observed'],
            N_single=pair['single']['required_tests_observed'],N_dual=pair['dual']['required_tests_observed'],
            single_source_M2_shares=pair['single']['source_M2_shares'],dual_source_M2_shares=pair['dual']['source_M2_shares']))
    save(out/'report.json',dict(summary_sha256=digest(root/'summary.json'),audit_sha256=digest(audit),
        plotting_environment={'matplotlib':version('matplotlib')},renderer_sha256=digest(__file__),
        paired_seeds=table,conclusion=summary['conclusion'],
        figures={p.name:digest(p) for p in sorted(out.glob('paired_convergence.*'))},
        scope='All seeds retained. Undefined RHW/AF/intervals stay null. Raw event rates are proposal diagnostics.'))
    return dict(rendered=True,paired_seeds=len(table),conclusion=summary['conclusion'])


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--result',required=True);p.add_argument('--audit',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();print(render(a.result,a.audit,a.output))


if __name__=='__main__':main()
