"""Explicit v1.9 policies bound to private copies of the frozen v1.8 engine.

Only candidate policy and the approved listwise objective differ. Original
model, AdamW/cosine40, RandomSampler, AMP retry, cursor/RNG/checkpoint and joint
full129 validation bytecode are reused without modifying shared module globals.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import FunctionType

import torch
from torch.nn import functional as F

from . import u_bridge_training as engine
from .comparison_experiment import ARMS, TRAINING_FORMAT

FORMAT = TRAINING_FORMAT
ENGINE_SHA256 = '9bcb9d67fc2b69861028b5241110270a144ee0ea849c573ab757da7ba391a8cc'
POLICY_FORMAT = 'v19_comparison_candidate_objective_policy_v1'
capture_rng = engine.capture_rng
restore_rng = engine.restore_rng
atomic_save = engine.atomic_save
cpu_copy = engine.cpu_copy
digest = engine.digest
CalibrationRejected = engine.CalibrationRejected


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _seal(value):
    result = copy.deepcopy(value)
    result['sha256'] = hashlib.sha256(json.dumps(result, sort_keys=True, separators=(',', ':'),
                                                allow_nan=False).encode()).hexdigest()
    return result


def arm_policy(arm):
    if arm not in ARMS: raise ValueError('Explicit v1.9 comparison arm required')
    if _sha(engine.__file__) != ENGINE_SHA256:
        raise ValueError('Frozen v1.8 training engine source changed')
    kind = 'listwise_cross_entropy' if arm == 'native_listwise' else 'pairwise_softplus'
    schedule = ('original_selected_seven' if arm == 'selected' else
                'fixed_native_U0_to_U6' if arm == 'native_fixed' else 'native_seven_rotating_across_epochs')
    loss = ('mean cross_entropy of eight logits with source P target index0'
            if kind == 'listwise_cross_entropy' else 'mean over source problems of mean seven softplus(U-P)')
    return _seal(dict(format=POLICY_FORMAT, version='v1.9', arm=arm,
        candidate_schedule=schedule, training_candidates=8, comparisons_per_source=7,
        expected_unique_U_per_source=7 if arm in ('selected', 'native_fixed') else 128,
        fixed_native_indices=list(range(7)) if arm == 'native_fixed' else None,
        native_rotation='((epoch-1)*7)%128, consecutive seven wrapped indices',
        objective=kind, objective_kind=kind, ranking_reduction=loss, consistency_weight=.1,
        loss=loss + ' + 0.1 original genuine two-view consistency',
        natural_loss_scale_matched=False,
        loss_scale_confound='raw listwise CE and mean pairwise softplus have different natural loss/gradient scales; no rescaling',
        validation='fixed P+128 native U, original score_inference_chunked, one joint upper per source',
        best=['patient_macro_MRR', 'patient_macro_top1', '-patient_macro_pairwise_softplus_loss'],
        reported_pair_loss='shared pairwise softplus diagnostic; native_listwise ranking term is raw CE',
        positive_index=0, original_model_parameters=10434532, geometry_corruption=False,
        engine_sha256=ENGINE_SHA256, facade_sha256=_sha(__file__),
        engine_binding='private globals; original run_arm/calibrate_batches code objects; no shared global mutation',
        architecture_changed=False, optimizer_or_schedule_changed=False, quality_verified=False))


def expected_keys(arm, epoch, *, full=False):
    if arm not in ARMS or type(epoch) is not int or epoch < 0:
        raise ValueError('Explicit arm and nonnegative actual view epoch required')
    if full: return ('P', *(f'U:{i}' for i in range(128)))
    if epoch < 1: raise ValueError('Training view epoch is one-based')
    if arm == 'selected': return ('P', *(f'S:{i}' for i in range(7)))
    start = 0 if arm == 'native_fixed' else (epoch - 1) * 7 % 128
    return ('P', *(f'U:{(start+i)%128}' for i in range(7)))


def objective(scores, consistency, *, arm, expected_candidates=8):
    if arm not in ARMS or expected_candidates != 8:
        raise ValueError('Approved source-P/seven-U training objective required')
    if arm != 'native_listwise':
        return engine.pair_objective(scores, consistency, expected_candidates=8)
    if (not isinstance(scores, (list, tuple)) or not scores or
            any(not torch.is_tensor(s) or s.ndim != 1 or s.numel() != 8 for s in scores)):
        raise ValueError('Eight ordered logits per original source problem required')
    if not torch.is_tensor(consistency) or consistency.numel() != 1:
        raise ValueError('Actual original two-view consistency scalar required')
    logits = torch.stack(scores).float()
    targets = torch.zeros(len(scores), dtype=torch.long, device=logits.device)
    ranking = F.cross_entropy(logits, targets, reduction='mean')
    return ranking + .1 * consistency.float(), dict(ranking=ranking, consistency=consistency.float())


class _PolicyProvider:
    """Verify declared candidate keys while preserving actual view epoch/data."""
    def __init__(self, provider, arm): self._provider, self._arm = provider, arm
    def __getattr__(self, name): return getattr(self._provider, name)
    def candidate_keys(self, index, arm, epoch, full=False):
        if arm != self._arm: raise ValueError('Candidate request crossed comparison arms')
        keys = tuple(self._provider.candidate_keys(index, arm, epoch, full=full))
        if keys != expected_keys(arm, epoch, full=full): raise ValueError('Provider candidate policy differs from sealed arm')
        return keys
    def batch(self, indices, arm, epoch, training, full=False):
        if arm != self._arm: raise ValueError('Physical batch request crossed comparison arms')
        ids = list(indices)
        result = self._provider.batch(ids, arm, epoch, training, full=full)
        keys = tuple(self.candidate_keys(index, arm, epoch, full=full) for index in ids)
        if tuple(getattr(result, 'bridge_indices', ())) != tuple(ids):
            raise ValueError('Physical batch lost original source identities')
        if tuple(getattr(result, 'bridge_candidate_keys', ())) != keys:
            raise ValueError('Actual batch candidate order differs from declared policy')
        return result


def _bind_config(config, policy):
    result = copy.deepcopy(config)
    if 'comparison_policy' in result and result['comparison_policy'] != policy:
        raise ValueError('Configuration comparison policy differs; no silent resume migration')
    result['comparison_policy'] = copy.deepcopy(policy)
    return result


def _normalize_report(value, policy):
    result = copy.deepcopy(value)
    result['comparison_policy'] = copy.deepcopy(policy)
    result['objective_kind'] = policy['objective_kind']
    if 'trained_comparisons' in result:
        coverage = result['trained_comparisons']
        expected = policy['expected_unique_U_per_source']
        coverage['expected_unique_U_per_source'] = expected
        coverage['complete_expected_coverage'] = all(n == expected for n in coverage['counts_by_source'].values())
        coverage['candidate_schedule'] = policy['candidate_schedule']
    return result


def _engine_functions(policy):
    """Private namespace bindings; never assign to the frozen engine module."""
    namespace = vars(engine).copy()
    def bound_objective(scores, consistency, *, expected_candidates=8):
        return objective(scores, consistency, arm=policy['arm'], expected_candidates=expected_candidates)
    def write_receipt(path, value):
        result = _normalize_report(value, policy)
        if Path(path).name == 'execution_contract.json':
            result['loss'] = policy['loss']
            result['ranking_term'] = policy['objective_kind']
            result['expected_unique_U_per_source'] = policy['expected_unique_U_per_source']
        engine._write_new(path, result)
    def append_receipt(path, value):
        result = copy.deepcopy(value)
        result['comparison_policy_sha256'] = policy['sha256']
        result['objective_kind'] = policy['objective_kind']
        if 'ranking' in result: result['ranking_term'] = policy['objective_kind']
        engine._append(path, result)
    def save_checkpoint(path, payload):
        result = dict(payload)
        result['comparison_policy'] = copy.deepcopy(policy)
        result.pop('content_sha256', None)
        result['content_sha256'] = engine.digest(result)
        engine.atomic_save(path, result)
    namespace.update(ARMS=ARMS, FORMAT=FORMAT, pair_objective=bound_objective,
                     _write_new=write_receipt, _append=append_receipt, atomic_save=save_checkpoint)
    functions = {}
    for name in ('run_arm', 'calibrate_batches'):
        original = getattr(engine, name)
        if original.__closure__ is not None: raise RuntimeError('Frozen engine acquired unexpected closure dependencies')
        bound = FunctionType(original.__code__, namespace, name, original.__defaults__, original.__closure__)
        bound.__kwdefaults__ = copy.deepcopy(original.__kwdefaults__)
        bound.__annotations__ = copy.deepcopy(original.__annotations__)
        bound.__doc__ = original.__doc__
        if bound.__code__ is not original.__code__: raise RuntimeError('Engine bytecode was not reused exactly')
        functions[name] = bound
    return functions


def calibrate_batches(net, provider, config, *, arm, candidates, workers, budget, debug=False):
    policy = arm_policy(arm)
    functions = _engine_functions(policy)
    selected, receipt = functions['calibrate_batches'](net, _PolicyProvider(provider, arm),
        _bind_config(config, policy), arm=arm, candidates=candidates, workers=workers, budget=budget, debug=debug)
    receipt = _normalize_report(receipt, policy)
    receipt['format'] = FORMAT
    return selected, receipt


def run_arm(net, provider, config, *, arm, output, physical_batch, workers, epochs, identity, budget, debug=False):
    policy = arm_policy(arm); bound_config = _bind_config(config, policy)
    calibration = bound_config['u_bridge_runtime']['batch_calibration']
    if calibration.get('comparison_policy') != policy or calibration.get('format') != FORMAT:
        raise ValueError('Actual calibration belongs to another comparison/loss policy')
    bound_identity = copy.deepcopy(identity)
    if 'comparison_policy' in bound_identity and bound_identity['comparison_policy'] != policy:
        raise ValueError('Identity comparison policy differs; no silent resume migration')
    bound_identity['comparison_policy'] = copy.deepcopy(policy)
    result = _engine_functions(policy)['run_arm'](net, _PolicyProvider(provider, arm), bound_config,
        arm=arm, output=output, physical_batch=physical_batch, workers=workers, epochs=epochs,
        identity=bound_identity, budget=budget, debug=debug)
    return _normalize_report(result, policy)
