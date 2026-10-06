"""Compact saved-result summary for declared arms; no GPU, checkpoint, or data loading."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
import statistics

ARMS = ('selected', 'native', 'native_fixed', 'native_listwise')
TRAINING_FORMAT = 'original_v1_comparison_controls_training_v1'


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def invocation_paths(root, arm):
    if arm not in ARMS:
        raise ValueError('Unknown declared comparison arm')
    name = re.compile(re.escape(arm) + r'_[0-9a-f]{32}\.json')
    return [path for path in (root / 'invocations').glob(arm + '_*.json')
            if name.fullmatch(path.name)]


def metric_text(report, count):
    if (report.get('source_problems') != count or not report.get('full_candidate_evaluation')
            or not report.get('joint_upper_once_per_source_problem')
            or len(report.get('rows', [])) != count
            or any(row['candidate_count'] != 129 for row in report['rows'])):
        raise ValueError('Incomplete full129 joint validation receipt')
    metrics = report['metrics']
    keys = ('mrr', 'top1', 'pair_win', 'pair_loss')
    if not all(math.isfinite(metrics[key]) for key in keys):
        raise ValueError('Nonfinite evaluation metric')
    return ' '.join(f'{key}={metrics[key]:.6f}' for key in keys)


def summarize(root, arms=ARMS):
    if not arms or len(set(arms)) != len(arms) or any(arm not in ARMS for arm in arms):
        raise ValueError('Select unique declared comparison arms')
    experiment = read(root / 'experiment.json')
    content = {key: value for key, value in experiment.items() if key != 'sha256'}
    actual = hashlib.sha256(json.dumps(content, sort_keys=True, separators=(',', ':'),
                                      allow_nan=False).encode()).hexdigest()
    if experiment.get('format') != 'v19_matched_comparison_controls_v1' or actual != experiment.get('sha256'):
        raise ValueError('Not a byte-bound v1.9 comparison experiment')
    count = sum(source['partition'] == 'val' for source in experiment['samples'])
    if not count:
        raise ValueError('No declared held-out source problems')
    print(f'v1.9 {root.name} | DEBUG={experiment["debug"]} | epochs={experiment["epochs"]} | full129 patient macro')
    for arm in arms:
        paths = invocation_paths(root, arm)
        if not paths:
            print(f'{arm}: no completed/paused invocation; active curves remain in {arm}/curve.jsonl')
            continue
        invocation = read(max(paths, key=lambda path: path.stat().st_mtime_ns))
        result = invocation['result']
        if (invocation['contract_sha256'] != actual or result['arm'] != arm
                or result.get('format') != TRAINING_FORMAT or result.get('debug') != experiment['debug']):
            raise ValueError('Arm result belongs to another contract')
        print(f'{arm} | status={result["status"]} | epochs={result["completed_epochs"]} | updates={result["updates"]}')
        directory = root / arm
        initial = directory / 'validation_epoch_000.json'
        if initial.exists():
            print('  INIT: ' + metric_text(read(initial), count))
        curve_file = directory / 'curve.jsonl'
        curve = [json.loads(line) for line in curve_file.read_text(encoding='utf8').splitlines() if line] if curve_file.exists() else []
        if curve:
            if len({row['epoch'] for row in curve}) != len(curve):
                raise ValueError('Repeated completed epoch in curve')
            best_epoch = result['best']['epoch']
            best = next(row for row in curve if row['epoch'] == best_epoch)
            last = curve[-1]
            print(f'  BEST epoch{best_epoch}: ' + metric_text(best['validation129'], count))
            print(f'  LAST epoch{last["epoch"]}: ' + metric_text(last['validation129'], count))
            print(f'  epoch-min median={statistics.median(row["epoch_wall_seconds"] for row in curve)/60:.3f}')
        coverage = result['trained_comparisons']
        counts = list(coverage['counts_by_source'].values())
        print(f'  objective={result["comparison_policy"]["objective"]}; unique-comparison/source min/mean/max='
              f'{min(counts)}/{statistics.mean(counts):.1f}/{max(counts)}; '
              f'target={coverage["expected_unique_U_per_source"]}; full-training={result["full_training"]}; quality={result["quality_verified"]}')
        if result['resume'].get('completed_checkpoint_no_repeat_verified'):
            print('  completed reexecution: zero additional optimizer updates')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--arms', choices=ARMS, nargs='+', default=ARMS)
    arguments = parser.parse_args()
    summarize(arguments.experiment.resolve(strict=True), arguments.arms)


if __name__ == '__main__':
    main()
