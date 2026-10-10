"""DEBUG clone admission for the authentic historical three-counter CP trainer.

The actual native AMP retry step retains every instruction and equation. Its
one counter-name constant drops only the raw-target transport counter, which
the SHA-pinned historical preprocessed trainer never owned. Production native
training does not call this helper.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import inspect
from pathlib import Path
import textwrap
from types import FunctionType

from . import v24_native_gradient_runtime as gradient

FORMAT = 'v24_historical_Basic_three_CP_counters_native_clone_DEBUG_v1'
HISTORICAL_SOURCE_SHA256 = 'a3df56435fedff46fbc42a502596fdade2632d0211ea412630dba1555f32567d'
HISTORICAL_CLASS = '_nnUNetTrainer_250epochs_OnlineCP'
ORIGINAL_COUNTERS = ('_online_cp_events', '_online_cp_samples', '_online_schedule_hash', '_online_native_transport')
MATCHED_COUNTERS = ORIGINAL_COUNTERS[:-1]


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _stat(path):
    value = Path(path).stat()
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def _source_contract(path, expected_sha256):
    path = Path(path).absolute()
    if path.resolve(strict=True) != path or any(p.is_symlink() for p in (path, *path.parents)) or not path.is_file():
        raise ValueError('Exact immutable historical trainer source required for clone counter admission')
    before = _stat(path)
    if _sha(path) != expected_sha256:
        raise ValueError('Historical trainer source SHA differs from admitted three-counter contract')
    text = path.read_text(encoding='utf8')
    tree = ast.parse(text)
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == HISTORICAL_CLASS]
    constructors = [] if len(classes) != 1 else [node for node in classes[0].body
        if isinstance(node, ast.FunctionDef) and node.name == '__init__']
    if len(constructors) != 1:
        raise ValueError('Original historical CP constructor counter ownership is unproved')
    names = [target.attr for node in ast.walk(constructors[0]) if isinstance(node, ast.Assign)
        for target in node.targets if isinstance(target, ast.Attribute)
        and isinstance(target.value, ast.Name) and target.value.id == 'self'
        and target.attr.startswith('_online_')]
    if (len(names) != 3 or set(names) != set(MATCHED_COUNTERS)
            or any(isinstance(node, ast.Attribute) and node.attr == ORIGINAL_COUNTERS[-1] for node in ast.walk(tree))):
        raise ValueError('Exactly the original three historical CP counters and no raw transport counter required')
    if _stat(path) != before:
        raise ValueError('Historical trainer source changed during clone counter admission')
    return path, before


def _replace_counter_constant(original):
    source = ast.parse(textwrap.dedent(inspect.getsource(original)))
    if (original.__name__ != gradient.STEP
            or ast.dump(source, include_attributes=False) != ast.dump(gradient._source_function(gradient.STEP), include_attributes=False)):
        raise ValueError('Complete original native AMP clone-step AST required')
    assignments = [node for node in ast.walk(source) if isinstance(node, ast.Assign)
        and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == 'counter_names']
    value = None if len(assignments) != 1 else assignments[0].value
    if (not isinstance(value, ast.Tuple) or any(not isinstance(item, ast.Constant) for item in value.elts)
            or tuple(item.value for item in value.elts) != ORIGINAL_COUNTERS):
        raise ValueError('Exact original four-counter-name assignment required')
    constants = original.__code__.co_consts
    positions = [index for index, item in enumerate(constants) if item == ORIGINAL_COUNTERS]
    if len(positions) != 1:
        raise ValueError('Unique original counter tuple constant required')
    position = positions[0]
    changed = tuple(MATCHED_COUNTERS if index == position else item for index, item in enumerate(constants))
    result = original.__code__.replace(co_consts=changed)
    if (result.co_code != original.__code__.co_code
            or result.replace(co_consts=constants) != original.__code__):
        raise ValueError('Only native DEBUG counter tuple constant may change; complete AMP code must remain identical')
    return result, position


def _clone_step(original, historical_trainer_source, *, expected_sha256, verify_actual_constructor):
    path, before = _source_contract(historical_trainer_source, expected_sha256)
    code, constant_position = _replace_counter_constant(original)
    corrected = gradient.clone_step(original)
    helper = corrected.__globals__[gradient.HELPER]
    own = Path(__file__).absolute(); own_stat = _stat(own); own_sha = _sha(own)
    contract = dict(format=FORMAT, debug=True, production_optimizer_updates=0,
        original_historical_trainer=str(path), original_historical_trainer_sha256=expected_sha256,
        original_counter_names=list(ORIGINAL_COUNTERS), matched_counter_names=list(MATCHED_COUNTERS),
        removed_counter='absent original historical raw-target transport counter only',
        replaced_code_constant_index=constant_position, instruction_bytes_preserved=True,
        original_loss_gradient_clipping_scaler_optimizer_and_AMP_retry_equations_preserved=True,
        original_model_and_batch_RNG_retry_equations_preserved=True,
        production_training_math_changed=False,
        source_files_sha256={'hiercp_v1x/v24_matched_native_gradient.py': own_sha,
            **gradient.gradient_runtime_contract()['source_files_sha256']})

    def admitted_helper(trainer, parameters):
        if _stat(path) != before or _stat(own) != own_stat:
            raise ValueError('Admitted historical DEBUG counter runtime source changed')
        if verify_actual_constructor:
            method = type(trainer).__init__
            actual = inspect.getsourcefile(method)
            if (actual is None or Path(actual).absolute() != path
                    or method.__qualname__ != HISTORICAL_CLASS + '.__init__'):
                raise ValueError('Actual clone trainer must inherit the admitted original historical constructor')
            from .v24_native_best_eval_runtime import _function_source_proof
            constructor = _function_source_proof(method)
            if constructor['sha256'] != expected_sha256:
                raise ValueError('Actual inherited historical constructor source proof differs')
        else:
            constructor = None
        if hasattr(trainer, ORIGINAL_COUNTERS[-1]):
            raise ValueError('Raw-target transport ownership cannot be admitted as historical preprocessed CP')
        proof = helper(trainer, parameters)
        proof['matched_CP_counter_runtime'] = copy.deepcopy(contract)
        if constructor is not None:
            proof['matched_CP_counter_runtime']['actual_original_constructor'] = constructor
        return proof

    result = FunctionType(code, dict(corrected.__globals__, **{gradient.HELPER: admitted_helper}),
        original.__name__, original.__defaults__, original.__closure__)
    result.__kwdefaults__ = copy.deepcopy(original.__kwdefaults__)
    return result


def clone_step(original, *, historical_trainer_source):
    return _clone_step(original, historical_trainer_source,
        expected_sha256=HISTORICAL_SOURCE_SHA256, verify_actual_constructor=True)


def _clone_step_debug(original, *, historical_trainer_source, fixture_sha256):
    """Explicit CPU fixture admission; never used by the production CLI."""
    return _clone_step(original, historical_trainer_source,
        expected_sha256=fixture_sha256, verify_actual_constructor=False)
