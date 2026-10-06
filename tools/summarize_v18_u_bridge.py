"""Read existing v1.8 JSON receipts and print a compact result summary.

No neural imports, checkpoints, source-data access, GPU queries, or file writes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


EXPERIMENT_FORMAT = 'v18_u_bridge_matched_experiment_v1'
TRAINING_FORMAT = 'original_v1_u_bridge_training_v1'
METRICS = ('mrr', 'top1', 'pair_win', 'pair_loss')


def read(path):
    with path.open(encoding='utf8') as stream:
        return json.load(stream)


def validated_experiment(root):
    value = read(root / 'experiment.json')
    if value.get('format') != EXPERIMENT_FORMAT:
        raise ValueError('Not a v1.8 matched experiment.json')
    expected = value.get('sha256')
    content = {key: item for key, item in value.items() if key != 'sha256'}
    actual = hashlib.sha256(json.dumps(content, sort_keys=True, separators=(',', ':'),
                                      allow_nan=False).encode()).hexdigest()
    if actual != expected: raise ValueError('experiment.json content checksum changed')
    if type(value.get('debug')) is not bool or type(value.get('epochs')) is not int:
        raise ValueError('Missing explicit experiment DEBUG/epoch contract')
    return value


def metrics(value):
    result = value['metrics']
    for key in METRICS:
        if not math.isfinite(result[key]): raise ValueError('Nonfinite recorded metric: ' + key)
    return {key: result[key] for key in METRICS}


def full_validation(value, count):
    if not value.get('full_candidate_evaluation') or not value.get('joint_upper_once_per_source_problem'):
        raise ValueError('Recorded validation does not certify full129 joint scoring')
    if value.get('source_problems') != count:
        raise ValueError('Full-validation source count disagrees with experiment')
    for row in value.get('rows', ()):
        if row.get('candidate_count') != 129: raise ValueError('Incomplete recorded validation candidate row')
    return metrics(value)


def latest_report(root, arm, contract):
    candidates = []
    for name in ('training_complete.json', 'paused.json'):
        path = root / arm / name
        if path.exists(): candidates.append((path.stat().st_mtime_ns, path, False))
    for path in (root / 'invocations').glob(arm + '_*.json'):
        candidates.append((path.stat().st_mtime_ns, path, True))
    if not candidates: return None, None
    _, path, invocation = max(candidates, key=lambda item: (item[0], str(item[1])))
    value = read(path)
    if invocation:
        if value.get('contract_sha256') != contract: raise ValueError('Latest invocation belongs to another experiment')
        value = value['result']
    if value.get('format') != TRAINING_FORMAT or value.get('arm') != arm:
        raise ValueError('Latest arm report has an incompatible identity: ' + str(path))
    return value, path


def curve(path, validation_count):
    rows = []
    if not path.exists(): return rows
    with path.open(encoding='utf8') as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip(): continue
            value = json.loads(line)
            if rows and value['epoch'] <= rows[-1]['epoch']:
                raise ValueError(f'Nonincreasing/repeated epoch in {path.name}:{line_number}')
            rows.append(dict(epoch=value['epoch'], update=value['update'],
                validation=full_validation(value['validation129'], validation_count),
                train7=metrics(value['train7_live_before_updates']),
                wall=value['epoch_wall_seconds'], optimization=value['optimization_loader_save_seconds'],
                validation_seconds=value['validation_seconds']))
    return rows


def metric_text(value):
    return (f'MRR={value["mrr"]:.6f} top1={value["top1"]:.6f} '
            f'pair-win={value["pair_win"]:.6f} pair-loss={value["pair_loss"]:.6f}')


def summarize(root):
    experiment = validated_experiment(root)
    samples = experiment['samples']
    train = [row for row in samples if row['partition'] == 'train']
    validation = [row for row in samples if row['partition'] == 'val']
    if not train or not validation: raise ValueError('Missing original train/validation source inventory')
    print(f'v1.8 {root.name} | DEBUG={experiment["debug"]} | epochs={experiment["epochs"]} '
          '| metrics=patient macro, source-anchor proxy')
    for arm in ('selected', 'native'):
        directory = root / arm
        report, report_path = latest_report(root, arm, experiment['sha256'])
        rows = curve(directory / 'curve.jsonl', len(validation))
        if report and report['debug'] != experiment['debug']:
            raise ValueError('Arm DEBUG contract differs from experiment')
        print(f'{arm} | reported-status={report["status"] if report else "NO_REPORT"} '
              f'| train-sources={len(train)} val-sources={len(validation)} '
              f'| completed-epochs={report["completed_epochs"] if report else "unavailable"} '
              f'| updates={report["updates"] if report else "unavailable"}')
        initial = directory / 'validation_epoch_000.json'
        if initial.exists():
            value = read(initial)
            print('  INIT full129: ' + metric_text(full_validation(value, len(validation))))
        else: print('  INIT full129: pending')
        if rows:
            latest = rows[-1]
            best_epoch = (report.get('best') or {}).get('epoch') if report else None
            best = next((row for row in rows if row['epoch'] == best_epoch), None)
            if best_epoch is not None and best is None: raise ValueError('BEST epoch is missing from curve')
            if best is None:
                best = max(rows, key=lambda row: (row['validation']['mrr'], row['validation']['top1'],
                                                 -row['validation']['pair_loss']))
            print(f'  BEST full129 epoch={best["epoch"]}: ' + metric_text(best['validation']))
            print(f'  LAST full129 epoch={latest["epoch"]}: ' + metric_text(latest['validation']))
            print(f'  TRAIN7 live before updates epoch={latest["epoch"]}: ' + metric_text(latest['train7']))
            minutes = [row['wall'] / 60 for row in rows]
            print(f'  epoch-min mean={statistics.mean(minutes):.3f} last={minutes[-1]:.3f} '
                  f'total={sum(minutes):.3f}; last opt/load/save={latest["optimization"]/60:.3f} '
                  f'full129-val={latest["validation_seconds"]/60:.3f}')
        else: print('  BEST/LAST full129 and TRAIN7: no completed epoch')
        if report:
            coverage = report.get('trained_comparisons')
            if coverage:
                counts = coverage['counts_by_source']; keys = coverage['keys_by_source']
                if set(counts) != {str(row['index']) for row in train} or set(keys) != set(counts):
                    raise ValueError('Coverage receipt changed the source inventory')
                if any(count != len(set(keys[index])) for index, count in counts.items()):
                    raise ValueError('Coverage counts disagree with recorded unique comparison keys')
                unique = set().union(*(set(values) for values in keys.values()))
                print(f'  trained-U unique-key-union={len(unique)}; per-source '
                      f'min/mean/max={min(counts.values())}/{statistics.mean(counts.values()):.1f}/{max(counts.values())} '
                      f'of {coverage["expected_unique_U_per_source"]}; '
                      f'complete-coverage={coverage["complete_expected_coverage"]}')
            else: print('  trained-U coverage: unavailable in this report')
            print(f'  full-training={report["full_training"]} full-evaluation={report["full_evaluation"]} '
                  f'CP-quality-verified={report["CP_quality_verified"]}; report={report_path.relative_to(root)}')
            resume = report.get('resume', {})
            if resume.get('completed_checkpoint_no_repeat_verified'):
                print('  completed-resume: no repeated updates/epochs; model/optimizer/schedule/RNG unchanged')
        else: print('  full-training/full-evaluation/coverage: no saved arm report')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    arguments = parser.parse_args()
    try:
        summarize(arguments.experiment.resolve(strict=True))
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(2, f'Cannot summarize v1.8 receipts: {type(error).__name__}: {error}\n')


if __name__ == '__main__':
    main()
