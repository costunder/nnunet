"""Apply the exact historical six-metric definitions to saved predictions.

Only model_validation_dir is adapted for isolated results. Metric functions,
matching, empty-mask behavior, bootstrap counts and cohorts remain unchanged.
"""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys

EXPECTED = {'online_eval_v2.py': '93c703edb20e074645ef443f6ca2bd7db03268bbb90fda8591f337993592270b',
    'online_eval_provenance.py': 'f624ffe37adc708ba5495b1b6bb9c59df74b841109442994379acd175ba75d91'}


def load_evaluator(root):
    """Admit original private exports or the one Git-pinned historical package."""
    root=Path(root).resolve(strict=True);source=root/'tools';historical=False
    module_name='tools.online_eval_v2'
    if(source/'historical_v24').is_dir():
        source=source/'historical_v24';module_name='tools.historical_v24.online_eval_v2';historical=True
    for name,checksum in EXPECTED.items():
        if hashlib.sha256((source/name).read_bytes()).hexdigest()!=checksum:
            raise ValueError('Historical metric implementation hash mismatch: '+name)
    sys.path.insert(0,str(root))
    if historical:
        # Frozen v1 checkout files use CRLF. Bind the exact LF historical
        # sibling for this import, then restore the process's previous alias.
        provenance=importlib.import_module('tools.historical_v24.online_eval_provenance')
        marker=object();previous=sys.modules.get('tools.online_eval_provenance',marker)
        sys.modules['tools.online_eval_provenance']=provenance
        try:evaluator=importlib.import_module(module_name)
        finally:
            if previous is marker:sys.modules.pop('tools.online_eval_provenance')
            else:sys.modules['tools.online_eval_provenance']=previous
    else:
        evaluator=importlib.import_module(module_name)
        provenance=importlib.import_module('tools.online_eval_provenance')
    if(Path(evaluator.__file__).resolve()!=source/'online_eval_v2.py'
            or Path(provenance.__file__).resolve()!=source/'online_eval_provenance.py'
            or any(Path(getattr(evaluator,name).__globals__['__file__']).resolve()!=source/'online_eval_provenance.py'
                for name in('build_evaluation_contract','contract_comparability','prepare_new_output','verify_evaluation_contract'))):
        raise ValueError('Wrong historical evaluator/provenance import')
    return evaluator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluator-root', type=Path, required=True)
    parser.add_argument('--prediction', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--checkpoint', choices=('BEST', 'FINAL'), default='BEST')
    parser.add_argument('--trainer', default='nnUNetTrainer_250epochs_FrozenV23CP')
    parser.add_argument('--model-label', default='v2.3 GNN CP -> nnUNet')
    parser.add_argument('--gpu-label', default='GPU1')
    args = parser.parse_args()
    expected=EXPECTED
    evaluator=load_evaluator(args.evaluator_root)
    trainer = args.trainer
    original = evaluator.model_validation_dir

    def validation_dir(paths, dataset_id, outer_fold, nn_cfg, trainer_name):
        if trainer_name == trainer:
            if dataset_id != 730 or outer_fold != 0:
                raise ValueError('Original full outer-fold0 required')
            return args.prediction.resolve(strict=True)
        return original(paths, dataset_id, outer_fold, nn_cfg, trainer_name)

    evaluator.model_validation_dir = validation_dir
    options = evaluator.parser().parse_args(['--project', str(args.project), '--hier-trainer', trainer,
        '--output', str(args.output)])
    evaluator.evaluate(options)
    summary = json.loads((args.output / 'summary.json').read_text())
    basic = summary['criteria']['dice_ge_0p10']['basic_cp']
    reference = {'recall': 121/172, 'precision': 121/247, 'f1': 242/419, 'fp_per_case': 126/26}
    for key, value in reference.items():
        if abs(basic[key] - value) > 1e-12:
            raise ValueError('Historical Basic CP reproduction failed: ' + key)
    if abs(summary['case_tumor_dice']['basic_mean'] - 0.645377888302566) > 1e-12:
        raise ValueError('Historical Basic CP tumor Dice changed')
    if abs(summary['lesion_dice_quality']['basic_mean'] - 0.4898810630076888) > 1e-12:
        raise ValueError('Historical Basic CP lesion Dice changed')
    rows = []
    for side, key, label in [('basic', 'basic_cp', 'Basic CP (historical predictions)'),
                              ('hier', 'hiercp', args.model_label + ' ' + args.checkpoint + ' (' + args.gpu_label + ')')]:
        values = summary['criteria']['dice_ge_0p10'][key]
        rows.append(dict(model=label, tumor_dice=summary['case_tumor_dice'][side + '_mean'],
            lesion_mean_dice=summary['lesion_dice_quality'][side + '_mean'],
            **{name: values[name] for name in ('recall', 'precision', 'f1', 'fp_per_case')}))
    with (args.output / 'six_metric_comparison.json').open('x') as stream:
        json.dump(dict(rows=rows, criterion='dice_ge_0p10', patient_count=26,
            evaluator_sha256=expected, historical_basic_reproduced=True,
            prediction_override=str(args.prediction), checkpoint=args.checkpoint,
            wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()), stream, indent=2)
    print('SIX_METRIC_COMPARISON ' + json.dumps(rows), flush=True)


if __name__ == '__main__':
    main()
