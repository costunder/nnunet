"""Observed ranking progress against the same untrained validation baseline."""
import csv
import math
from pathlib import Path

METRICS = ('ranking_pairwise_loss', 'ranking_mrr', 'ranking_recall_at_1',
           'ranking_recall_at_5', 'ranking_recall_at_10')


def measurement(epoch, step, metrics, baseline):
    if any(not math.isfinite(float(metrics[k])) for k in METRICS):
        raise ValueError('Nonfinite ranking history')
    if any(metrics[k] != baseline[k] for k in ('ranking_pairs', 'ranking_evaluable_cases', 'ranking_cases_without_observed')):
        raise ValueError('Validation comparison population changed')
    return dict(epoch=epoch, step=step, **{k:float(metrics[k]) for k in METRICS},
        mrr_change_from_initial=float(metrics['ranking_mrr']-baseline['ranking_mrr']),
        recall_at_1_change_from_initial=float(metrics['ranking_recall_at_1']-baseline['ranking_recall_at_1']),
        rank_loss_change_from_initial=float(metrics['ranking_pairwise_loss']-baseline['ranking_pairwise_loss']))


def write_history(root, rows):
    """Rebuild from checkpoint-bound history, including earlier resumed epochs."""
    if not rows:
        return
    path = Path(root) / 'validation_history.csv'
    temporary = path.with_suffix('.csv.tmp')
    with temporary.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)


def progress_line(row):
    return (f"Change vs initial | MRR={row['mrr_change_from_initial']:+.4f} "
            f"R@1={row['recall_at_1_change_from_initial']:+.4f} "
            f"rank_loss={row['rank_loss_change_from_initial']:+.4f} (lower is better) "
            '| same validation cohort; CP benefit not measured')
