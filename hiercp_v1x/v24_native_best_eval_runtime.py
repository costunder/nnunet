"""Require native full validation to use the exact BEST loaded last.

Native checkpoint loading and full-volume validation execute unchanged. FINAL
remains the completed-training witness; it is never an evaluation fallback.
"""
from __future__ import annotations

import copy
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import sys
import types

FORMAT = 'v24_native_best_loaded_full26_validation_v1'
RECEIPT = 'best_validation_complete.json'
BEST = 'checkpoint_best.pth'
TRAIN_COMMAND = ('from hiercp_v1x.v24_native_best_eval_runtime import run_training_entry; '
                 'run_training_entry()')
ROOT = Path(__file__).resolve().parents[1]


def _sha(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError('Regular BEST evaluation artifact required: ' + str(path))
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 2**20), b''):
            digest.update(block)
    return digest.hexdigest()


def _stat(path):
    value = Path(path).stat()
    return [value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns]


def _exact(path, *, directory=False):
    path = Path(path).absolute()
    if path.resolve(strict=True) != path:
        raise ValueError('Exact BEST evaluation path required: ' + str(path))
    if any(parent.is_symlink() for parent in (path, *path.parents)):
        raise ValueError('Symlinks cannot establish BEST evaluation provenance')
    if not (path.is_dir() if directory else path.is_file()):
        raise ValueError('Missing BEST evaluation artifact: ' + str(path))
    return path


def _cases(values):
    values = list(values)
    if (len(values) != 26 or len(set(values)) != 26
            or any(not isinstance(case, str) or not case or Path(case).name != case for case in values)):
        raise ValueError('Complete unchanged outer26 BEST validation cohort required')
    return sorted(values)


def _function_source_proof(function):
    """Bind the actual callable to its complete source-compiled code object."""
    from .v24_readonly_native_storage import code_fingerprint
    path = _exact(inspect.getsourcefile(function))
    compiled = compile(path.read_bytes(), str(path), 'exec', dont_inherit=True)
    matches = []
    def collect(code):
        for child in code.co_consts:
            if isinstance(child, types.CodeType):
                if (child.co_name == function.__code__.co_name
                        and child.co_firstlineno == function.__code__.co_firstlineno):
                    matches.append(child)
                collect(child)
    collect(compiled)
    if len(matches) != 1 or code_fingerprint(matches[0]) != code_fingerprint(function.__code__):
        raise ValueError('Actual native BEST evaluation method differs from its original source')
    return dict(path=str(path), sha256=_sha(path), stat=_stat(path),
                method=function.__name__, code_sha256=code_fingerprint(function.__code__))


def _guard_sources(proofs, *, full_hash=False):
    for proof in proofs:
        path = _exact(proof['path'])
        if _stat(path) != proof['stat'] or (full_hash and _sha(path) != proof['sha256']):
            raise ValueError('Admitted BEST evaluation source changed: ' + str(path))


def _method_sources(function):
    result = [_function_source_proof(function)]
    # The existing validation-only GPU1 helper adds its own last-BEST guard.
    # Preserve that complete callable and additionally bind its native closure.
    if Path(result[0]['path']).name == 'run_v24_best_validation.py':
        closure = inspect.getclosurevars(function).nonlocals
        originals = [value for name, value in closure.items()
            if name in ('original_load', 'original_validation') and inspect.isfunction(value)]
        if len(originals) != 1:
            raise ValueError('Validation-only BEST helper lacks its original native method binding')
        result.append(_function_source_proof(originals[0]))
    return result


def _install(trainer_class, expected_cases, *, bank_sha256, source_proofs=()):
    expected_cases = _cases(expected_cases)
    existing = trainer_class.__dict__.get('_v24_best_evaluation_binding')
    if existing is not None:
        if (existing['cases'] != expected_cases or existing['bank_sha256'] != bank_sha256
                or trainer_class.load_checkpoint is not existing['load']
                or trainer_class.perform_actual_validation is not existing['validate']):
            raise ValueError('Existing BEST evaluation guard differs')
        _guard_sources(existing['sources'])
        return copy.deepcopy(existing['contract'])
    original_load = trainer_class.load_checkpoint
    original_validate = trainer_class.perform_actual_validation
    sources = [*source_proofs,
        dict(path=str(Path(__file__).resolve()), sha256=_sha(__file__), stat=_stat(__file__)),
        *_method_sources(original_load), *_method_sources(original_validate)]
    contract = dict(format=FORMAT, checkpoint_name=BEST, validation_cases=expected_cases,
        bank_sha256=bank_sha256, original_checkpoint_load_called=True,
        original_full_validation_called=True, final_checkpoint_fallback=False,
        original_training_math_unchanged=True, source_proofs=copy.deepcopy(sources))
    state_name = '_v24_last_actual_checkpoint_load'

    def load(self, filename, *args, **kwargs):
        _guard_sources(sources)
        # Native accepts state dictionaries for other callers. They cannot
        # prove an on-disk BEST, so full validation rejects them explicitly.
        path = _exact(filename) if isinstance(filename, (str, os.PathLike)) else None
        before = None if path is None else dict(path=str(path), sha256=_sha(path), stat=_stat(path))
        setattr(self, state_name, None)
        result = original_load(self, filename, *args, **kwargs)
        _guard_sources(sources)
        if path is not None:
            if _stat(path) != before['stat'] or _sha(path) != before['sha256']:
                raise ValueError('Checkpoint changed during original native loading')
            setattr(self, state_name, before)
        return result

    def validate(self, *args, **kwargs):
        _guard_sources(sources, full_hash=True)
        fold = _exact(self.output_folder, directory=True)
        checkpoint = fold / BEST
        if not checkpoint.is_file():
            raise ValueError('Native BEST checkpoint missing; FINAL evaluation fallback is forbidden')
        checkpoint = _exact(checkpoint)
        loaded = getattr(self, state_name, None)
        if (not loaded or loaded['path'] != str(checkpoint)
                or loaded['stat'] != _stat(checkpoint) or loaded['sha256'] != _sha(checkpoint)):
            raise ValueError('Refusing full validation: exact native BEST must be loaded last and unchanged')
        explicit = os.environ.get('V24_EVAL_BEST')
        if explicit and (_exact(explicit) != checkpoint
                or loaded['sha256'] != os.environ.get('V24_EVAL_BEST_SHA256')):
            raise ValueError('Explicit validation-only BEST admission differs')
        _, actual_cases = self.do_split()
        if _cases(actual_cases) != expected_cases:
            raise ValueError('Native BEST validation differs from the complete outer26 cohort')
        epoch = int(self.current_epoch)
        if not 0 < epoch <= 250:
            raise ValueError('Native BEST checkpoint epoch must be within the full250 training contract')
        receipt = fold / RECEIPT
        if receipt.exists() or receipt.is_symlink():
            raise FileExistsError('Completed BEST validation provenance is immutable: ' + str(receipt))
        prediction_output = fold / 'validation'
        if prediction_output.exists():
            _exact(prediction_output, directory=True)
            if any(prediction_output.iterdir()):
                raise FileExistsError('Fresh empty validation output required; previous FINAL/unknown predictions cannot become BEST')
        result = original_validate(self, *args, **kwargs)
        _guard_sources(sources, full_hash=True)
        if (_stat(checkpoint) != loaded['stat'] or _sha(checkpoint) != loaded['sha256']
                or _cases(self.do_split()[1]) != expected_cases
                or getattr(self, state_name, None) != loaded):
            raise ValueError('BEST/checkpoint/cohort changed during original full validation')
        predictions = _exact(fold / 'validation', directory=True)
        actual = {path.name.removesuffix('.nii.gz') for path in predictions.glob('*.nii.gz')}
        if actual != set(expected_cases):
            raise ValueError('Original native BEST validation did not produce all and only outer26 predictions')
        summary = _exact(predictions / 'summary.json')
        document = dict(contract, native_fold=str(fold), checkpoint_path=str(checkpoint),
            checkpoint_sha256=loaded['sha256'], checkpoint_stat=loaded['stat'], best_current_epoch=epoch,
            complete=True, native_validation_returned_successfully=True,
            predictions_path=str(predictions),
            predictions_sha256={case:_sha(predictions / (case + '.nii.gz')) for case in expected_cases},
            summary_sha256=_sha(summary), GT_passed_to_predictor=False, CP_at_inference=False)
        from .transition_v1_data import _publish_new_json
        _publish_new_json(receipt, document)
        return result

    trainer_class.load_checkpoint = load
    trainer_class.perform_actual_validation = validate
    trainer_class._v24_best_evaluation_binding = dict(cases=expected_cases, bank_sha256=bank_sha256,
        load=load, validate=validate, sources=sources, contract=contract)
    _guard_sources(sources)
    return copy.deepcopy(contract)


def install_best_evaluation_guard():
    """Install only in the native evaluation process, with sealed cohort input."""
    module = importlib.import_module('nnunetv2.training.nnUNetTrainer.nnUNetTrainer')
    path = _exact(os.environ['ONLINE_CP_BANK'])
    bank = json.loads(path.read_text(encoding='utf8'))
    expected = _cases(bank['split']['outer_val'])
    explicit_cases = os.environ.get('V24_EVAL_CASES')
    if explicit_cases and _cases(json.loads(explicit_cases)) != expected:
        raise ValueError('Validation-only BEST request changed the admitted bank cohort')
    proof = dict(path=str(path), sha256=_sha(path), stat=_stat(path))
    return _install(module.nnUNetTrainer, expected, bank_sha256=proof['sha256'], source_proofs=[proof])


def validate_best_validation_receipt(receipt_path, *, fold, expected_cases, bank_sha256):
    """Require completed actual native BEST inference, never a FINAL receipt."""
    receipt_path = _exact(receipt_path)
    fold = _exact(fold, directory=True)
    checkpoint = _exact(fold / BEST)
    value = json.loads(receipt_path.read_text(encoding='utf8'))
    cases = _cases(expected_cases)
    predictions = _exact(fold / 'validation', directory=True)
    sources = value.get('source_proofs', [])
    if (not isinstance(sources, list)
            or not {'load_checkpoint', 'perform_actual_validation'}.issubset(
                {item.get('method') for item in sources if isinstance(item, dict)})
            or not any(item.get('path') == str(Path(__file__).resolve())
                and item.get('sha256') == _sha(__file__) for item in sources if isinstance(item, dict))):
        raise ValueError('Actual native BEST loader, validation and guard source provenance required')
    if (receipt_path != fold / RECEIPT or value.get('format') != FORMAT
            or value.get('checkpoint_name') != BEST or value.get('checkpoint_path') != str(checkpoint)
            or value.get('native_fold') != str(fold) or value.get('bank_sha256') != bank_sha256
            or value.get('checkpoint_sha256') != _sha(checkpoint) or value.get('checkpoint_stat') != _stat(checkpoint)
            or value.get('complete') is not True or value.get('native_validation_returned_successfully') is not True
            or value.get('original_checkpoint_load_called') is not True
            or value.get('original_full_validation_called') is not True
            or value.get('final_checkpoint_fallback') is not False
            or value.get('GT_passed_to_predictor') is not False or value.get('CP_at_inference') is not False
            or value.get('validation_cases') != cases or value.get('predictions_path') != str(predictions)
            or set(value.get('predictions_sha256', {})) != set(cases)
            or {path.name.removesuffix('.nii.gz') for path in predictions.glob('*.nii.gz')} != set(cases)
            or any(value['predictions_sha256'][case] != _sha(predictions / (case + '.nii.gz')) for case in cases)
            or value.get('summary_sha256') != _sha(predictions / 'summary.json')):
        raise ValueError('Complete full26 evaluation provenance from exact native BEST required')
    _guard_sources(sources, full_hash=True)
    return value


def run_training_entry():
    if '--val_best' not in sys.argv[1:] or sys.argv[1:].count('--val_best') != 1:
        raise ValueError('Native full validation requires explicit --val_best; FINAL fallback is forbidden')
    install_best_evaluation_guard()
    from nnunetv2.run.run_training import run_training_entry as original_entry
    return original_entry()
