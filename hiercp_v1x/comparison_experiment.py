"""v1.9 comparison controls; preserve frozen v1.8 inputs and engine bytes."""
from __future__ import annotations

import math

from .u_bridge_experiment import (
    FILES as V18_FILES, Budget, digest, read, sha, write_new, lock,
    prepare_inputs, prepared_data_path,
)

FORMAT = 'v19_matched_comparison_controls_v1'
TRAINING_FORMAT = 'original_v1_comparison_controls_training_v1'
ARMS = ('selected', 'native', 'native_fixed', 'native_listwise')
FILES = V18_FILES + (
    'hiercp_v1x/comparison_experiment.py', 'hiercp_v1x/comparison_data.py',
    'hiercp_v1x/comparison_training.py', 'tools/run_v19_comparison.py',
    'config/v19_comparison_controls.json',
)


def joint_calibration(reports, candidates):
    """Use one genuinely measured physical batch accepted by all four arms."""
    if set(reports) != set(ARMS):
        raise ValueError('All four real-arm measurements are required')
    if (not candidates or candidates != sorted(set(candidates))
            or any(type(n) is not int or n < 1 for n in candidates)):
        raise ValueError('Explicit unique positive physical batch candidates required')
    tables = {}
    for arm in ARMS:
        report = reports[arm]
        if report.get('arm') != arm or report.get('original_model_and_RNG_preserved') is not True:
            raise ValueError(f'{arm}: actual unchanged-model calibration identity required')
        table = {}
        for row in report['reports']:
            size = row['physical_batch']
            if size not in candidates or size in table:
                raise ValueError('Calibration has undeclared or duplicate physical batches')
            if type(row.get('accepted')) is not bool:
                raise ValueError('Explicit measured acceptance required')
            if row['accepted']:
                rate = row.get('samples_per_second')
                if (isinstance(rate, bool) or not isinstance(rate, (int, float))
                        or not math.isfinite(rate) or rate <= 0):
                    raise ValueError('Actual positive finite measured throughput required')
                ids = row.get('actual_sample_indices', [])
                if len(ids) != size or len(set(ids)) != size or any(type(index) is not int for index in ids):
                    raise ValueError('Calibration physical batch differs from real source count')
            table[size] = row
        if set(table) != set(candidates):
            raise ValueError(f'{arm}: every explicit candidate must have a measured receipt')
        tables[arm] = table
    accepted = [n for n in candidates if all(tables[arm][n]['accepted'] for arm in ARMS)]
    if not accepted:
        raise RuntimeError('No explicit physical batch passed all four arms; reports preserved')
    for size in accepted:
        reference = tables['selected'][size]['actual_sample_indices']
        if any(tables[arm][size]['actual_sample_indices'] != reference for arm in ARMS):
            raise ValueError('Matched calibration source order differs between arms')
    winner = max(accepted, key=lambda n: min(tables[arm][n]['samples_per_second'] for arm in ARMS))
    return dict(physical_batch=winner, accepted=True, arm='matched_four',
                explicit_candidates=list(candidates), accepted_common=accepted,
                selection='max minimum measured samples/sec across all four arms', reports=reports)
