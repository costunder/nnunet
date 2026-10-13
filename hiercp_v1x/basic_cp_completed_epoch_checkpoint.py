"""Private native saver for a checkpoint taken AFTER on_epoch_end increments.

Only this post-increment latest save serializes current_epoch without the
native +1. Native BEST/final saves and original modules remain unchanged.
"""
from __future__ import annotations

import ast
import copy
import inspect
import math
import textwrap

_CACHE = {}
_LOGGER_FIELDS = frozenset(('mean_fg_dice', 'ema_fg_dice', 'dice_per_class_or_region',
                           'train_losses', 'val_losses', 'lrs',
                           'epoch_start_timestamps', 'epoch_end_timestamps'))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _private_completed_saver(function):
    _require(inspect.isfunction(function) and function.__closure__ is None,
             'Native checkpoint saver must be a closure-free function; wrapped guards cannot be bypassed')
    cached = _CACHE.get(function)
    if cached is not None and cached[0] is function.__code__:
        return cached[1]
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    _require(len(tree.body) == 1 and isinstance(tree.body[0], ast.FunctionDef)
             and tree.body[0].name == function.__name__ and not tree.body[0].decorator_list,
             'Exact undecorated actual native checkpoint function source required')
    original_tree = copy.deepcopy(tree)
    fields = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for index, key in enumerate(node.keys):
                if isinstance(key, ast.Constant) and key.value == 'current_epoch':
                    fields.append((node, index))
    _require(len(fields) == 1, 'Exactly one native serialized current_epoch dictionary field required')
    dictionary, index = fields[0]; value = dictionary.values[index]
    _require(isinstance(value, ast.BinOp) and isinstance(value.op, ast.Add)
             and isinstance(value.left, ast.Attribute) and value.left.attr == 'current_epoch'
             and isinstance(value.left.value, ast.Name) and value.left.value.id == 'self'
             and isinstance(value.right, ast.Constant) and type(value.right.value) is int and value.right.value == 1,
             'Actual native current_epoch serialization must be exactly self.current_epoch + 1')
    dictionary.values[index] = copy.deepcopy(value.left)
    # Recompile the ORIGINAL source first and verify that this AST represents
    # the actual method bytecode, including every native guard/state branch.
    namespace = dict(function.__globals__)
    filename = function.__code__.co_filename
    exec(compile(ast.fix_missing_locations(original_tree), filename, 'exec'), namespace)
    reconstructed = namespace[function.__name__]
    _require(reconstructed.__code__.co_code == function.__code__.co_code
             and reconstructed.__code__.co_consts == function.__code__.co_consts,
             'Inspected checkpoint source differs from the actual native function bytecode')
    namespace = dict(function.__globals__)
    exec(compile(ast.fix_missing_locations(tree), filename, 'exec'), namespace)
    saver = namespace[function.__name__]
    saver.__defaults__ = function.__defaults__
    saver.__kwdefaults__ = copy.copy(function.__kwdefaults__)
    saver.__annotations__ = dict(function.__annotations__)
    saver.__qualname__ = function.__qualname__
    saver.__module__ = function.__module__
    _CACHE[function] = (function.__code__, saver)
    return saver


def save_completed_epoch(trainer, filename):
    """Save full native model/optimizer/scaler/logger at a proved completed epoch.

    Trainer.current_epoch is never modified. Logger.get_checkpoint and native
    state_dict guards execute once, in the privately cloned original method.
    """
    epoch = trainer.current_epoch
    _require(type(epoch) is int and epoch > 0, 'Post-epoch checkpoint requires a positive completed current_epoch')
    logging = getattr(trainer.logger, 'my_fantastic_logging', None)
    _require(isinstance(logging, dict) and _LOGGER_FIELDS.issubset(logging),
             'Actual native logger must contain all eight completed-epoch arrays')
    _require(all(isinstance(value, (list, tuple)) and len(value) == epoch for value in logging.values()),
             'Every actual logger array must match the completed current_epoch')
    starts, ends = logging['epoch_start_timestamps'], logging['epoch_end_timestamps']
    _require(all(math.isfinite(float(start)) and math.isfinite(float(end)) and float(end) >= float(start)
                 for start, end in zip(starts, ends)), 'Every completed epoch must have finite ordered start/end timestamps')
    bound = trainer.save_checkpoint
    _require(inspect.ismethod(bound) and bound.__self__ is trainer,
             'Actual bound trainer.save_checkpoint method required')
    saver = _private_completed_saver(bound.__func__)
    return saver(trainer, filename)
