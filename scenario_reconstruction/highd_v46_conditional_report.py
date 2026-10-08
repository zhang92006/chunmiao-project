"""Plot actual PPO logs; lower second moment is better than a rising reward alone."""
import argparse
from pathlib import Path
from .highd_v46_assets import read, save


def report(run, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    run, output = Path(run), Path(output)
    summary = read(run/'ppo/training_summary.json')
    rows = [read(p) for p in sorted((run/'ppo').glob('iteration_*.json'))]
    if not rows: raise ValueError('No completed training iterations')
    fixed_ratio = None
    if (run/'selected_fixed/selection.json').exists():
        fixed = read(run/'selected_fixed/selection.json')
        initial = [summary['config']['initial_epsilon']]*summary['action_dimension']
        reference = next(r['scaled_second_moment'] for r in fixed['candidates'] if r['epsilon']==initial)
        selected = next(r['scaled_second_moment'] for r in fixed['candidates'] if r['epsilon']==fixed['epsilon'])
        # Each tool may choose a different numerical reward scale; ratios cancel it.
        fixed_ratio = selected/reference
    output.mkdir(parents=True, exist_ok=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    axes[0].plot([r['iteration'] for r in rows], [r['reward_mean'] for r in rows])
    axes[0].set(xlabel='PPO iteration', ylabel='Mean scaled replay reward', title='Training reward (corrected replay)')
    for key, denominator, label in [
        ('empirical_scaled_second_moment', summary['fixed_epsilon_scaled_second_moment'], f"Training ({summary['positive_episodes']} events)"),
        ('validation_scaled_second_moment', summary.get('validation_fixed_scaled_second_moment'), f"Validation ({summary['validation_positive_episodes']} events)")]:
        valid = [r for r in rows if key in r]
        if valid and denominator:
            axes[1].plot([0]+[r['iteration'] for r in valid], [1.]+[r[key]/denominator for r in valid], marker='o', label=label)
    axes[1].axhline(1., color='gray', linestyle='--', label='Initial fixed (0.1,0.1)')
    if fixed_ratio is not None:
        axes[1].axhline(fixed_ratio, color='purple', linestyle='-.', label='Validation-tuned fixed')
    axes[1].axvline(summary['selected_iteration'], color='green', linestyle=':', label='Selected checkpoint')
    axes[1].set(xlabel='PPO iteration', ylabel='Second moment / initial fixed', title='Lower is better; conditional target')
    axes[1].legend(fontsize=8)
    for ax in axes: ax.grid(alpha=.2)
    fig.savefig(output/'training_curves.png', dpi=160); plt.close(fig)
    compact={k: summary[k] for k in ('completed_iterations','selected_iteration','resubstitution_ratio','validation_ratio','portable_policy_max_error','sampling_scope')}
    compact.update(last_training_ratio=rows[-1]['empirical_scaled_second_moment']/summary['fixed_epsilon_scaled_second_moment'],
                   selected_fixed_validation_ratio=fixed_ratio,
                   ppo_over_selected_fixed_validation_ratio=summary['validation_ratio']/fixed_ratio if fixed_ratio is not None else None)
    compact['scope']='Conditional training/validation curves; not population efficiency evidence'
    save(output/'curve_summary.json',compact)
    return compact


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();print(report(args.run,args.output))


if __name__=='__main__':main()
