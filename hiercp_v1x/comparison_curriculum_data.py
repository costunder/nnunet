"""Explicit candidate plans over the preserved comparison preparation stack.

This adapter changes candidate selection only. Original samples, full canonical
geometry, ordered disjoint graph collation, view RNG and exact cache bindings
remain in the prepared provider. The frozen rotating-arm verifier is explicitly
unwrapped, never represented as having accepted a new curriculum schedule.

Plans must enclose prefetch and its cpu_staging scope. Drain/close both before
leaving a plan. A violated lifetime poisons the adapter and prevents new work;
restoration waits for already running calls/staging to finish naturally.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
from dataclasses import dataclass
import hashlib
import importlib
import json
import re
import threading

ARMS = ('selected', 'native', 'native_fixed', 'native_listwise')
FULL_KEYS = ('P', *(f'U:{number}' for number in range(128)))
FORMAT = 'comparison_explicit_curriculum_candidate_plan_v1'
_HEX = re.compile(r'[0-9a-f]{64}')
_REGISTRY_LOCK = threading.RLock()
_BORROWED = {}


def _keys(values, *, arm, mode):
    if isinstance(values, (str, bytes)):
        raise TypeError('Supply the ordered candidate sequence, not a string')
    result = tuple(values)
    if (not result or result[0] != 'P' or any(not isinstance(k, str) for k in result)
            or len(set(result)) != len(result)):
        raise ValueError('One original P first and distinct ordered candidate keys are required')
    if mode == 'training' and len(result) != 8:
        raise ValueError('Training preserves exactly P plus seven real candidates')
    if mode == 'stage_validation' and not 2 <= len(result) <= 129:
        raise ValueError('Stage validation requires P plus the declared nonempty pool, at most128 U')
    prefix, limit = ('S', 7) if arm == 'selected' else ('U', 128)
    allowed = {f'{prefix}:{i}' for i in range(limit)}
    if any(key not in allowed for key in result[1:]):
        raise ValueError('Candidate key is outside this arm\'s original frozen population')
    return result


@dataclass(frozen=True)
class CandidatePlan:
    arm: str
    policy_sha256: str
    keys: tuple[str, ...]
    mode: str
    epoch: int

    def __post_init__(self):
        if self.arm not in ARMS or self.mode not in ('training', 'stage_validation'):
            raise ValueError('Explicit comparison arm and training/stage_validation mode required')
        if not isinstance(self.policy_sha256, str) or not _HEX.fullmatch(self.policy_sha256):
            raise ValueError('The separately sealed curriculum policy SHA256 is required')
        if type(self.epoch) is not int or self.epoch < 0 or (self.mode == 'training' and self.epoch < 1):
            raise ValueError('Explicit actual view epoch required; training is one-based')
        object.__setattr__(self, 'keys', _keys(self.keys, arm=self.arm, mode=self.mode))

    def receipt(self):
        result = dict(format=FORMAT, arm=self.arm, policy_sha256=self.policy_sha256,
            candidate_keys=list(self.keys), mode=self.mode, actual_view_epoch=self.epoch,
            legacy_expected_keys_verifier_applied=False, full_validation_keys=list(FULL_KEYS))
        result['sha256'] = hashlib.sha256(json.dumps(result, sort_keys=True,
            separators=(',', ':'), allow_nan=False).encode()).hexdigest()
        return result


def _unwrap(provider, arm):
    """Unwrap only the exact frozen facade; reject unknown wrapper lookalikes."""
    cls = type(provider)
    if cls.__module__ == 'hiercp_v1x.comparison_training' and cls.__name__ == '_PolicyProvider':
        module = importlib.import_module('hiercp_v1x.comparison_training')
        if cls is not module._PolicyProvider:
            raise TypeError('Unrecognized replacement for the frozen comparison policy facade')
        if vars(provider).get('_arm') != arm:
            raise ValueError('The frozen policy facade belongs to a different arm')
        return vars(provider)['_provider'], True
    if '_provider' in vars(provider):
        raise TypeError('An unknown provider facade cannot be silently unwrapped')
    return provider, False


class _State:
    def __init__(self, provider, original_class, arm, policy_sha256, unwrapped):
        self.provider, self.original_class = provider, original_class
        self.arm, self.policy_sha256, self.unwrapped = arm, policy_sha256, unwrapped
        self.lock = threading.RLock()
        self.plan = None
        self.active_calls = self.staging = 0
        self.poisoned = self.restore_requested = self.restored = False

    def _restore_if_drained(self):
        if self.restore_requested and not self.active_calls and not self.staging and not self.restored:
            self.provider.__class__ = self.original_class
            self.restored = True
            with _REGISTRY_LOCK:
                _BORROWED.pop(id(self.provider), None)

    @contextmanager
    def call(self):
        with self.lock:
            if self.poisoned or self.restore_requested:
                raise RuntimeError('Candidate-plan scope is closing/invalid; no new preparation may start')
            self.active_calls += 1
            plan = self.plan
        try:
            yield plan
        finally:
            with self.lock:
                self.active_calls -= 1
                self._restore_if_drained()

    def keys(self, index, arm, epoch, full, plan):
        if arm != self.arm:
            raise ValueError('Candidate request crossed explicitly declared curriculum arms')
        if type(epoch) is not int or epoch < 0 or type(full) is not bool:
            raise ValueError('Actual nonnegative view epoch and explicit full-validation flag required')
        self.provider._example(index)  # Preserve source/index admission, even for readiness checks.
        if full:
            return FULL_KEYS
        if plan is None:
            raise RuntimeError('A non-full request requires an active explicit candidate plan')
        if epoch != plan.epoch:
            raise ValueError('Candidate request uses another actual view epoch than the immutable plan')
        return plan.keys

    def request(self, indices, arm, epoch, training, full, plan):
        ids = tuple(indices)
        if not ids or len(set(ids)) != len(ids) or any(type(i) is not int for i in ids):
            raise ValueError('A complete nonempty batch of distinct original source indices is required')
        if type(training) is not bool or (training and full):
            raise ValueError('Full129 is evaluation only; training remains eight candidates')
        keys = tuple(self.keys(i, arm, epoch, full, plan) for i in ids)
        if not full and training != (plan.mode == 'training'):
            raise ValueError('Training flag differs from the explicitly active plan mode')
        return ids, keys


def _planned_class(original, state):
    class PlannedProvider(original):
        def candidate_keys(self, index, arm, epoch, full=False):
            with state.call() as plan:
                return state.keys(index, arm, epoch, full, plan)

        def batch(self, indices, arm, epoch, training, full=False):
            with state.call() as plan:
                ids, keys = state.request(indices, arm, epoch, training, full, plan)
                result = super().batch(ids, arm, epoch, training, full=full)
                if (tuple(result.counts) != tuple(map(len, keys))
                        or tuple(getattr(result, 'bridge_indices', ())) != ids
                        or tuple(tuple(k) for k in getattr(result, 'bridge_candidate_keys', ())) != keys
                        or tuple(getattr(result, 'bridge_source_ids', ())) !=
                            tuple(self._example(i)['id'] for i in ids)):
                    raise ValueError('Actual collation changed planned candidate counts/order/source ownership')
                result.curriculum_candidate_plan = (dict(format=FORMAT, arm=state.arm,
                    policy_sha256=state.policy_sha256, mode='full_validation',
                    actual_view_epoch=epoch, candidate_keys=list(FULL_KEYS),
                    legacy_expected_keys_verifier_applied=False) if full else plan.receipt())
                return result

        def cached_batch_ready(self, indices, arm, epoch, *, training, full=False):
            with state.call() as plan:
                ids, _ = state.request(indices, arm, epoch, training, full, plan)
                # Crucial: original readiness executes with this actual self,
                # so layout bindings see the same keys as the eventual batch.
                return super().cached_batch_ready(ids, arm, epoch, training=training, full=full)

        def cold_staging_bytes(self, indices):
            with state.call():
                return super().cold_staging_bytes(indices)

        @contextmanager
        def cpu_staging(self, slots):
            with state.call():
                with state.lock:
                    state.staging += 1
                try:
                    with super().cpu_staging(slots):
                        yield
                finally:
                    with state.lock:
                        state.staging -= 1
                        state._restore_if_drained()

    PlannedProvider.__name__ = 'ExplicitCurriculum' + original.__name__
    return PlannedProvider


class CurriculumProvider:
    """Owning facade; plan epochs are actual view epochs, never policy stages."""
    def __init__(self, state):
        self._state = state

    def __getattr__(self, name):
        with self._state.lock:
            if self._state.restored or self._state.restore_requested:
                raise RuntimeError('Curriculum provider context is closed')
        return getattr(self._state.provider, name)

    @property
    def candidate_plan_receipt(self):
        with self._state.lock:
            return None if self._state.plan is None else self._state.plan.receipt()

    @property
    def legacy_policy_facade_unwrapped(self):
        return self._state.unwrapped

    def report(self):
        """Keep preparation statistics, explicitly distinguish the new policy."""
        state = self._state
        with state.call():
            result = copy.deepcopy(state.provider.report())
            previous = result.pop('comparison_controls', None)
            if previous is not None:
                result['legacy_controls_reference_only'] = previous
            result['curriculum_candidate_execution'] = dict(arm=state.arm,
                policy_sha256=state.policy_sha256,
                candidate_plan=None if state.plan is None else state.plan.receipt(),
                frozen_policy_facade_explicitly_unwrapped=state.unwrapped,
                legacy_expected_keys_verifier_applied=False,
                full_validation_candidates=129,
                model_graph_geometry_view_RNG_and_preparation_unchanged=True)
            return result

    @contextmanager
    def plan(self, keys, *, mode, epoch):
        state = self._state
        plan = CandidatePlan(state.arm, state.policy_sha256, keys, mode, epoch)
        with state.lock:
            if (state.plan is not None or state.active_calls or state.staging or state.poisoned
                    or state.restore_requested or state.restored):
                raise RuntimeError('Drain/close the previous plan and all CPU staging before changing candidates')
            state.plan = plan
        try:
            yield self
        finally:
            with state.lock:
                if state.active_calls or state.staging:
                    state.poisoned = True
                    raise RuntimeError('Candidate plan left before prefetch/preparation drained; new work disabled')
                state.plan = None


@contextmanager
def curriculum_provider(provider, arm, policy_sha256):
    """Borrow one real prepared provider; restore its exact class on scope end.

No object/state copy is used: original LRU accounting, budget registration,
cached runtime proxies, shared candidate workers and publication locks continue
to refer to the same provider. Callers must not use its frozen facade in parallel.
"""
    if arm not in ARMS or not isinstance(policy_sha256, str) or not _HEX.fullmatch(policy_sha256):
        raise ValueError('Explicit arm and separately sealed curriculum policy SHA256 required')
    underlying, unwrapped = _unwrap(provider, arm)
    for method in ('batch', 'candidate_keys', 'sample', '_example', 'cached_batch_ready',
                   'cold_staging_bytes', 'cpu_staging'):
        if not callable(getattr(underlying, method, None)):
            raise TypeError('The actual prepared comparison provider lacks required method: '+method)
    original = type(underlying)
    state = _State(underlying, original, arm, policy_sha256, unwrapped)
    with _REGISTRY_LOCK:
        if id(underlying) in _BORROWED:
            raise RuntimeError('This prepared provider already belongs to an active candidate adapter')
        _BORROWED[id(underlying)] = state
        try:
            underlying.__class__ = _planned_class(original, state)
        except (TypeError, AttributeError):
            _BORROWED.pop(id(underlying), None)
            raise TypeError('Prepared provider cannot support an isolated instance-class candidate adapter')
    facade = CurriculumProvider(state)
    try:
        yield facade
    finally:
        with state.lock:
            state.restore_requested = True
            unclosed = state.plan is not None or state.active_calls or state.staging
            if unclosed:
                state.poisoned = True
            state._restore_if_drained()
        if unclosed:
            raise RuntimeError('Curriculum scope ended with an undrained plan/staging; restoration deferred until drain')


@contextmanager
def candidate_plan(provider, arm, policy_sha256, keys, *, mode, epoch):
    """Flat runtime convenience; epoch is the actual sampling/view epoch.

Example: with candidate_plan(provider, arm, policy_sha, keys,
                             mode='stage_validation', epoch=29) as phase:
                 phase.batch(ids, arm, 29, False, full=False)
    """
    with curriculum_provider(provider, arm, policy_sha256) as adapted:
        with adapted.plan(keys, mode=mode, epoch=epoch) as phase:
            yield phase
