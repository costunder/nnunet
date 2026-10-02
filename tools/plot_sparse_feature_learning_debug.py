"""Plot measured sparse-L0 DEBUG learning without invoking torch or a model.

Input is a completed 48/96/192 comparison report. The four scientific panels
show train/held-out P-U half-tie pair-win and whole-case pairwise ranking loss.
This is short-cohort learning evidence, never graph optimality or CP efficacy.
"""
import argparse
import json
import math
from pathlib import Path


BUDGETS = (48, 96, 192)
STEPS = (0, 20, 40, 60)


def finite_number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'Finite measured number required: {label}')
    return float(value)


def measured_curves(report):
    """Reject absent, partial, differently scheduled or unbound results."""
    if (report.get('debug') is not True or report.get('actual_CT') is not True
            or report.get('actual_CUDA') is not True
            or report.get('complete_original_case_P_and_U128') is not True
            or report.get('physical_batch') != 32
            or report.get('full_training') is not False
            or report.get('optimal_graph_size_established') is not False):
        raise ValueError('Completed actual-CT CUDA DEBUG physical32/full P-U128 report required')
    train_cases = report.get('train_cases', [])
    held_out = report.get('validation_cases', [])
    if (len(train_cases) != 4 or len(held_out) != 1
            or len(set(train_cases)) != len(train_cases)
            or set(train_cases) & set(held_out)):
        raise ValueError('The measured four-train/one-held-out original case split is required')
    budgets = report.get('budgets', {})
    if set(budgets) != {str(count) for count in BUDGETS}:
        raise ValueError('All three measured node budgets48/96/192 are required')
    hashes = {budgets[str(count)].get('initial_model_sha256') for count in BUDGETS}
    if len(hashes) != 1 or None in hashes or '' in hashes:
        raise ValueError('All budgets must have the same recorded initial model hash')
    curves = {}
    for count in BUDGETS:
        result = budgets[str(count)]
        if result.get('continuous_optimizer') is not True:
            raise ValueError(f'Continuous Adam result required for budget{count}')
        snapshots = result.get('evaluation', [])
        steps = tuple(snapshot.get('step') for snapshot in snapshots)
        if steps != STEPS:
            raise ValueError(f'Exact measured steps{STEPS} required for budget{count}; received{steps}')
        updates = result.get('actual_updates', [])
        if tuple(update.get('step') for update in updates) != tuple(range(1, STEPS[-1] + 1)):
            raise ValueError(f'All60 continuous measured updates required for budget{count}')
        curves[count] = {split: {'pair_win': [], 'rank_loss': []}
                         for split in ('train', 'validation')}
        for snapshot in snapshots:
            if snapshot.get('model_optimizer_RNG_preserved') is not True:
                raise ValueError(f'Evaluation preservation evidence missing at budget{count}, step{snapshot["step"]}')
            for split, expected in (('train', train_cases), ('validation', held_out)):
                evaluated = snapshot.get(split, {})
                details = evaluated.get('cases', [])
                if (len(details) != len(expected)
                        or {case.get('case_id') for case in details} != set(expected)
                        or any(case.get('unobserved') != 128 for case in details)
                        or any(case.get('all_original_case_observations') is not True for case in details)):
                    raise ValueError(f'Whole original case P-U coverage missing: budget{count}, {split}')
                evaluation = evaluated.get('evaluation', {})
                if (evaluation.get('current_native_CNN_reencoded') is not True
                        or evaluation.get('support_train_only') is not True
                        or evaluation.get('complete_selected_case_P_U') is not True):
                    raise ValueError(f'Current native/held-out evaluation binding missing: budget{count}, {split}')
                pair_win = finite_number(evaluated.get('diagnostics', {}).get('pair_win_with_half_ties'),
                                         f'budget{count}/{split}/half-tie pair-win')
                loss = finite_number(evaluated.get('metrics', {}).get('ranking_pairwise_loss'),
                                     f'budget{count}/{split}/pairwise rank loss')
                if not 0 <= pair_win <= 1 or loss < 0:
                    raise ValueError('Measured pair-win/loss is outside its defined range')
                curves[count][split]['pair_win'].append(pair_win)
                curves[count][split]['rank_loss'].append(loss)
    # Each branch must see the same observations in the same actual updates.
    reference_schedule = [update.get('record_ids') for update in budgets[str(BUDGETS[0])]['actual_updates']]
    if any(not ids or len(ids) > 32 for ids in reference_schedule):
        raise ValueError('Explicit physical32 update record IDs required')
    for count in BUDGETS[1:]:
        schedule = [update.get('record_ids') for update in budgets[str(count)]['actual_updates']]
        if schedule != reference_schedule:
            raise ValueError('Node budgets used different actual observation schedules')
    return curves


def plot(report, output):
    output = Path(output)
    if output.suffix.lower() != '.png':
        raise ValueError('Standalone PNG output is required')
    if output.exists():
        raise FileExistsError(f'Existing result preserved: {output}')
    curves = measured_curves(report)
    # Matplotlib is loaded only for plotting; --help and report validation do
    # not import torch, initialize CUDA, execute inference or touch checkpoints.
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    from matplotlib.ticker import FormatStrFormatter, MaxNLocator

    colors = {48: '#0072B2', 96: '#D55E00', 192: '#009E73'}
    markers = {48: 'o', 96: 's', 192: '^'}
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.titlesize': 11, 'axes.labelsize': 10,
                         'legend.fontsize': 10, 'savefig.facecolor': 'white'})
    figure, axes = plt.subplots(2, 2, figsize=(10.8, 6.8), sharex=True,
                                sharey='row', layout='constrained')
    for column, (split, title) in enumerate((('train', 'Training: 4 cases'),
                                            ('validation', 'Held-out: 1 case'))):
        axes[0, column].set_title(title, fontweight='normal')
        for count in BUDGETS:
            for row, key in ((0, 'pair_win'), (1, 'rank_loss')):
                axes[row, column].plot(STEPS, curves[count][split][key],
                    color=colors[count], marker=markers[count], markersize=5,
                    linewidth=1.6, label=f'{count} context nodes')
        axes[0, column].axhline(.5, color='0.45', linewidth=.9, linestyle='--',
                               label='Chance pair-win = 0.5')
        axes[0, column].set_ylim(0, 1)
        axes[1, column].set_xlabel('Continuous optimizer update')
        axes[1, column].yaxis.set_major_formatter(FormatStrFormatter('%.5f'))
        for row in range(2):
            axes[row, column].set_xticks(STEPS)
            axes[row, column].grid(axis='y', color='0.87', linewidth=.65)
            axes[row, column].spines[['top', 'right']].set_visible(False)
            axes[row, column].yaxis.set_major_locator(MaxNLocator(nbins=5))
    axes[0, 0].set_ylabel('P-U pair-win (ties count 0.5)')
    axes[1, 0].set_ylabel('Whole-case pairwise rank loss')
    losses = [value for count in BUDGETS for split in ('train', 'validation')
              for value in curves[count][split]['rank_loss']]
    lower, upper = min(losses), max(losses)
    padding = max((upper - lower) * .12, max(abs(lower), abs(upper), 1) * 1e-6)
    axes[1, 0].set_ylim(max(0, lower - padding), upper + padding)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc='outside lower center', ncols=4, frameon=False)
    figure.suptitle('DEBUG continuous ranking | 4 training cases / 1 held-out case\n'
        'All original P + 128 U per case | physical batch32 | same initial weights\n'
        'Short-cohort results: graph optimality and CP efficacy are NOT established',
        fontsize=12, fontweight='normal')
    description = ('Actual CT CUDA DEBUG comparison: three context-node budgets48,96,192; '
        'continuous steps0,20,40,60. Four training cases and one held-out case. '
        'All original observed positives and128 unobserved comparison locations per case. '
        'Top row shows P-U half-tie pair-win; bottom row shows whole-case pairwise loss. '
        'Columns are train and held-out. Dashed0.5 reference appears only on pair-win. '
        'No conclusion about optimum graph size or CP efficacy.')
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation avoids a check/write race and cannot overwrite a
    # completed scientific figure from this or another experiment.
    try:
        with output.open('xb') as stream:
            figure.savefig(stream, format='png', dpi=180,
                           metadata={'Title': 'Sparse L0 continuous ranking DEBUG',
                                     'Description': description})
    finally:
        plt.close(figure)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True,
                        help='Completed actual measured DEBUG JSON report')
    parser.add_argument('--output', type=Path, required=True,
                        help='New standalone PNG; an existing file is never overwritten')
    args = parser.parse_args()
    with args.report.open('r', encoding='utf-8') as stream:
        report = json.load(stream)
    result = plot(report, args.output)
    print('DEBUG plot:', result)
    print('Measured short-cohort learning only; graph optimality/CP efficacy not established.')


if __name__ == '__main__':
    main()
