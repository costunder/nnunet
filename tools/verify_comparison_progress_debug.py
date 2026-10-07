"""Actual CT/CUDA regression for execution changes; no production training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--input-report', type=Path, required=True,
                   help='Completed actual-CT input/repeated-CUDA diagnostic, retained separately')
    a = p.parse_args()
    from hiercp_v1x import comparison_data
    from hiercp_v1x.comparison_inputs import local_mask_provider
    from hiercp_v1x.comparison_views import parallel_view_provider
    from hiercp_v1x.comparison_data_timing import timed_provider
    from hiercp_v1x.comparison_execution import comparison_execution
    from tools.current_gpu import current_device_selection
    from tools.verify_comparison_cache_debug import parse, verify, sha, read
    from unittest.mock import patch
    import torch

    inputs = read(a.input_report)
    if (inputs.get('exact_inputs') is not True
            or inputs.get('parameters') != 10434532
            or not all(inputs.get('preserved', {}).values())
            or {row['candidate_count'] for row in inputs.get('inputs', [])} != {8, 129}):
        raise ValueError('Actual complete train8/validation129 input equivalence evidence required')
    input_sha = sha(a.input_report)
    args = parse(['--gpu', str(a.gpu), '--output', str(a.output)])
    baseline = ROOT/'work/comparison_cache_DEBUG_20261007_r2/native_fixed'
    checkpoint = baseline/'checkpoint_latest.pt'
    before = sha(checkpoint)
    Provider = timed_provider(parallel_view_provider(local_mask_provider(comparison_data.ComparisonData)), a.output/'input_timing.jsonl')
    with current_device_selection(), comparison_execution(), patch.object(comparison_data, 'ComparisonData', Provider):
        report = verify(args)
    old = torch.load(checkpoint, map_location='cpu', weights_only=False)
    new = torch.load(a.output/'native_fixed/checkpoint_latest.pt', map_location='cpu', weights_only=False)
    if old['model'].keys() != new['model'].keys():
        raise AssertionError('Model parameter inventory changed')
    differences = {k: float((old['model'][k].double()-new['model'][k].double()).abs().max())
                   for k in old['model'] if old['model'][k].numel()}
    matched = all(torch.equal(old['model'][k], new['model'][k]) for k in old['model'])
    metric_differences = []
    for epoch in range(3):
        left = read(baseline/f'validation_epoch_{epoch:03d}.json')['metrics']
        right = read(a.output/'native_fixed'/f'validation_epoch_{epoch:03d}.json')['metrics']
        metric_differences.append(dict(epoch=epoch, old=left, new=right, equal=left == right))
    result = dict(scope='actual_CT_CUDA_DEBUG_only', baseline=str(baseline),
        model_parameters=10434532, optimizer_updates=2, physical_batch=2, workers=4,
        model_bitwise_equal=matched, max_parameter_absolute_difference=max(differences.values()),
        full129_metrics=metric_differences, original_checkpoint_preserved=sha(checkpoint)==before,
        actual_input_evidence=dict(path=str(a.input_report), sha256=input_sha,
            exact_values_and_layout_equal=True,
            unchanged_baseline_repeat_equal=[row['baseline_repeat_equal']
                for row in inputs['forward_diagnostics']]),
        mechanics_smoke_passed=True,
        historical_comparison_scope='Diagnostic only: bitwise CUDA weights/metrics are not promised. '
            'The separate repeated unchanged-input test records baseline numerical variability; '
            'it does not prove every historical training difference has that cause.',
        quality_verified=False, production_training_started=False,
        limitation='One validation source in this preserved fixture; validates values and CUDA path, not production prefetch speedup')
    (a.output/'equivalence.json').write_text(json.dumps(result, indent=2), encoding='utf8')
    if not result['original_checkpoint_preserved']:
        raise AssertionError('Original checkpoint changed')
    if sha(a.input_report) != input_sha:
        raise AssertionError('Referenced input diagnostic changed')
    print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
