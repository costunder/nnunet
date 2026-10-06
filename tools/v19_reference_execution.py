"""Read an active production v1.8 run's immutable execution metadata only."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


FORMAT = 'v18_u_bridge_matched_experiment_v1'
ARMS = ('selected', 'native')


def _read(path):
    if not path.is_file():
        raise FileNotFoundError(f'Reference metadata is unavailable: {path}')
    value = json.loads(path.read_text(encoding='utf8'))
    if not isinstance(value, dict):
        raise ValueError(f'Reference metadata must be a JSON object: {path}')
    return value


def _positive_integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return value


def _batches(value, name):
    if (not isinstance(value, list) or not value
            or any(type(n) is not int or n < 1 for n in value)
            or value != sorted(set(value))):
        raise ValueError(f'{name} must contain unique increasing positive batch sizes')
    return value


def _state_hash(value):
    if (not isinstance(value, str) or len(value) != 64
            or any(c not in '0123456789abcdef' for c in value)):
        raise ValueError('Reference calibration needs a valid initial state SHA256')
    return value


def reference_execution(root):
    """Validate batch/worker provenance without locks, checkpoints, or writes.

    experiment.json and calibration.json are published before training and do
    not change during the active run. No running process or result is modified.
    """
    root = Path(root).resolve(strict=True)
    if not root.is_dir():
        raise ValueError('Reference experiment must be a directory')
    experiment = _read(root / 'experiment.json')
    content = {key: value for key, value in experiment.items() if key != 'sha256'}
    checksum = hashlib.sha256(json.dumps(content, sort_keys=True, separators=(',', ':'),
                                        allow_nan=False).encode()).hexdigest()
    if (experiment.get('format') != FORMAT or experiment.get('sha256') != checksum):
        raise ValueError('Reference is not a sealed v1.8 experiment')
    if experiment.get('debug') is not False or type(experiment.get('epochs')) is not int or experiment['epochs'] != 40:
        raise ValueError('Reference must be a production forty-epoch run, not DEBUG')
    workers = _positive_integer(experiment.get('workers'), 'Reference workers', minimum=2)
    calibration = _read(root / 'calibration.json')
    if calibration.get('contract_sha256') != checksum:
        raise ValueError('Reference calibration belongs to another experiment')
    if calibration.get('accepted') is not True or calibration.get('arm') != 'matched_both':
        raise ValueError('Reference needs an accepted two-arm calibration')
    batch = _positive_integer(calibration.get('physical_batch'), 'Reference physical batch')
    explicit = _batches(calibration.get('explicit_candidates'), 'Explicit candidates')
    accepted = _batches(calibration.get('accepted_common'), 'Accepted common candidates')
    if not set(accepted).issubset(explicit) or batch not in explicit or batch not in accepted:
        raise ValueError('Reference physical batch was not explicitly accepted by both arms')
    reports = calibration.get('reports')
    if not isinstance(reports, dict) or set(reports) != set(ARMS):
        raise ValueError('Both reference arm measurements are required')
    states, selected_ids = [], []
    for arm in ARMS:
        report = reports[arm]
        if (not isinstance(report, dict) or report.get('arm') != arm
                or report.get('original_model_and_RNG_preserved') is not True):
            raise ValueError(f'{arm}: unchanged-model calibration provenance required')
        rows = report.get('reports')
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            raise ValueError(f'{arm}: actual calibration rows required')
        chosen = [row for row in rows if row.get('physical_batch') == batch]
        if len(chosen) != 1 or chosen[0].get('accepted') is not True:
            raise ValueError(f'{arm}: exactly one accepted chosen-batch measurement required')
        row = chosen[0]
        if type(row['physical_batch']) is not int:
            raise ValueError(f'{arm}: measured physical batch must be an integer')
        ids = row.get('actual_sample_indices')
        if (not isinstance(ids, list) or len(ids) != batch
                or any(type(index) is not int or index < 0 for index in ids)
                or len(set(ids)) != batch):
            raise ValueError(f'{arm}: physical batch must contain distinct actual source indices')
        rate = row.get('samples_per_second')
        if (isinstance(rate, bool) or not isinstance(rate, (int, float))
                or not math.isfinite(rate) or rate <= 0):
            raise ValueError(f'{arm}: positive finite measured throughput required')
        states.append(_state_hash(report.get('initial_state_sha256')))
        selected_ids.append(ids)
    if states[0] != states[1]:
        raise ValueError('Reference arms were calibrated from different initial neural states')
    if selected_ids[0] != selected_ids[1]:
        raise ValueError('Reference arms measured different source problems')
    return dict(root=str(root), contract_sha256=checksum, physical_batch=batch,
                workers=workers, calibration_initial_state_sha256=states[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--json', action='store_true', help='Emit the validated reference receipt')
    arguments = parser.parse_args()
    receipt = reference_execution(arguments.reference)
    if arguments.json:
        print(json.dumps(receipt, sort_keys=True, allow_nan=False))
    else:
        print(f'{receipt["physical_batch"]} {receipt["workers"]}')


if __name__ == '__main__':
    main()
