"""UNIT: explicit candidate plans retain complete prepared-provider contracts."""
from contextlib import contextmanager
from copy import deepcopy
import sys
import threading
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from hiercp_v1x.comparison_curriculum_data import (
    CandidatePlan, FULL_KEYS, candidate_plan, curriculum_provider,
)

POLICY = '1'*64


def keys(start=0, size=7):
    return ('P', *(f'U:{i}' for i in range(start, start+size)))


class PreparedFixture:
    """UNIT explicit graph identities, full node/edge arrays, and layout cache.

    No CUDA/model claim: this fixture exposes the same self-dispatch dependency
    as the prepared production stack, including its readiness/worker methods.
    """
    def __init__(self):
        self.workers = 16
        self.layout = {}
        self.stats = dict(builds=0, hits=0, staging_entries=0)
        self.examples_by_index = {76:dict(id='liver_46:0', index=76),
                                  77:dict(id='liver_46:1', index=77)}
        self.mode = None
        self.tamper = False
        self.started = self.release = None

    def _example(self, index):
        if type(index) is not int or index not in self.examples_by_index:
            raise IndexError('Unknown signed source')
        return self.examples_by_index[index]

    def examples(self, partition):
        return list(self.examples_by_index.values()) if partition == 'train' else []

    def report(self):
        return dict(self.stats,comparison_controls=dict(schedule='legacy_rotating'))

    def candidate_keys(self, index, arm, epoch, full=False):
        if full:
            return FULL_KEYS
        if arm == 'selected':
            return ('P', *(f'S:{i}' for i in range(7)))
        return keys(0 if arm == 'native_fixed' else (epoch-1)*7)

    def sample(self, index, requested, epoch, training):
        if self.started is not None:
            self.started.set()
            if not self.release.wait(5):
                raise RuntimeError('UNIT worker did not receive explicit release')
        binding = (index, tuple(requested))
        if binding not in self.layout:
            self.layout[binding] = [dict(identity=(index,key),
                nodes=tuple(range(17)), edges=tuple((i,i+1) for i in range(16))) for key in requested]
            self.stats['builds'] += 1
        else:
            self.stats['hits'] += 1
        return dict(graphs=deepcopy(self.layout[binding]), actual_epoch=epoch,
                    views=[(index,number,epoch,view) for number in range(len(requested)) for view in (0,1)],
                    training=training)

    def batch(self, indices, arm, epoch, training, full=False):
        requested = tuple(self.candidate_keys(i, arm, epoch, full=full) for i in indices)
        samples = [self.sample(i,k,epoch,training) for i,k in zip(indices,requested)]
        return SimpleNamespace(counts=tuple(len(k)-(1 if self.tamper else 0) for k in requested),
            bridge_indices=tuple(indices), bridge_candidate_keys=requested,
            bridge_source_ids=tuple(self._example(i)['id'] for i in indices), samples=samples)

    def cached_batch_ready(self, indices, arm, epoch, *, training, full=False):
        return all((i,tuple(self.candidate_keys(i, arm, epoch, full=full))) in self.layout for i in indices)

    def cold_staging_bytes(self, indices):
        return sum(1000000 for i in indices if self._example(i))

    @contextmanager
    def cpu_staging(self, slots):
        if self.mode is not None:
            raise RuntimeError('UNIT staging already active')
        self.mode = slots
        self.stats['staging_entries'] += 1
        try:
            yield
        finally:
            self.mode = None


def frozen_fixture_class():
    # Standalone UNIT injection for the lazy import identity boundary; no torch
    # dependency or substitution is used in the production adapter.
    class _PolicyProvider:
        def __init__(self, provider, arm):
            self._provider, self._arm = provider, arm

        def __getattr__(self, name):
            return getattr(self._provider, name)

        def batch(self, indices, arm, epoch, training, full=False):
            result = self._provider.batch(indices, arm, epoch, training, full=full)
            wanted = FULL_KEYS if full else keys((epoch-1)*7)
            if any(tuple(k) != wanted for k in result.bridge_candidate_keys):
                raise ValueError('UNIT frozen strict schedule rejected new keys')
            return result

    _PolicyProvider.__module__ = 'hiercp_v1x.comparison_training'
    return _PolicyProvider


class CandidatePlanTests(unittest.TestCase):
    def test_all_four_initial_plans_preserve_real_eight_graphs(self):
        for arm in ('selected', 'native', 'native_fixed', 'native_listwise'):
            with self.subTest(arm=arm):
                provider = PreparedFixture()
                requested = ('P', *(f'S:{i}' for i in range(7))) if arm == 'selected' else keys()
                with candidate_plan(provider, arm, POLICY, requested, mode='training', epoch=9) as phase:
                    batch = phase.batch([76,77], arm, 9, True)
                    self.assertEqual(batch.counts, (8,8))
                    self.assertEqual(batch.bridge_candidate_keys, (requested,requested))
                    self.assertEqual([s['actual_epoch'] for s in batch.samples], [9,9])
                    for source in batch.samples:
                        for graph in source['graphs']:
                            self.assertEqual(len(graph['nodes']),17)
                            self.assertEqual(len(graph['edges']),16)
                self.assertIs(type(provider),PreparedFixture)

    def test_stage_pool_and_full_validation_are_separate(self):
        provider = PreparedFixture()
        with candidate_plan(provider,'native_listwise',POLICY,keys(size=63),
                            mode='stage_validation',epoch=29) as phase:
            stage = phase.batch([76,77],'native_listwise',29,False)
            full = phase.batch([76,77],'native_listwise',0,False,full=True)
            self.assertEqual(stage.counts,(64,64))
            self.assertEqual(stage.bridge_candidate_keys,(keys(size=63),)*2)
            self.assertEqual(full.counts,(129,129))
            self.assertEqual(full.bridge_candidate_keys,(FULL_KEYS,)*2)
            self.assertEqual(full.curriculum_candidate_plan['mode'],'full_validation')
            with self.assertRaises(ValueError):
                phase.batch([76],'native_listwise',29,True)

    def test_layout_readiness_follows_new_keys_and_preserves_view_epoch(self):
        provider = PreparedFixture()
        with curriculum_provider(provider,'native',POLICY) as adapted:
            with adapted.plan(keys(),mode='training',epoch=1) as phase:
                phase.batch([76],'native',1,True)
                self.assertTrue(phase.cached_batch_ready([76],'native',1,training=True))
            with adapted.plan(keys(49),mode='training',epoch=8) as phase:
                self.assertFalse(phase.cached_batch_ready([76],'native',8,training=True))
                cold = phase.batch([76],'native',8,True)
                self.assertTrue(phase.cached_batch_ready([76],'native',8,training=True))
                replay = phase.batch([76],'native',8,True)
                self.assertEqual(cold.samples,replay.samples)
                self.assertEqual(cold.bridge_candidate_keys,(keys(49),))
            with adapted.plan(keys(49),mode='training',epoch=9) as phase:
                self.assertTrue(phase.cached_batch_ready([76],'native',9,training=True))
                next_view = phase.batch([76],'native',9,True)
                self.assertEqual(next_view.samples[0]['graphs'],cold.samples[0]['graphs'])
                self.assertNotEqual(next_view.samples[0]['views'],cold.samples[0]['views'])
        self.assertEqual(provider.stats['builds'],2)

    def test_original_state_budget_staging_and_instance_are_preserved(self):
        provider = PreparedFixture()
        before = provider.__dict__
        with candidate_plan(provider,'native',POLICY,keys(49),mode='training',epoch=8) as phase:
            self.assertIs(provider.__dict__,before)
            self.assertEqual(phase.cold_staging_bytes([76,77]),2000000)
            with phase.cpu_staging(2):
                self.assertEqual(provider.mode,2)
                phase.batch([76,77],'native',8,True)
            self.assertEqual(provider.mode,None)
        self.assertIs(provider.__dict__,before)
        self.assertEqual(provider.stats['staging_entries'],1)

    def test_exact_frozen_facade_unwrapped_and_restored(self):
        module = ModuleType('hiercp_v1x.comparison_training')
        module._PolicyProvider = frozen_fixture_class()
        provider = PreparedFixture()
        frozen = module._PolicyProvider(provider,'native')
        with patch.dict(sys.modules, {'hiercp_v1x.comparison_training':module}):
            with candidate_plan(frozen,'native',POLICY,keys(21),mode='training',epoch=8) as phase:
                self.assertTrue(phase.legacy_policy_facade_unwrapped)
                batch = phase.batch([76],'native',8,True)
                self.assertEqual(batch.bridge_candidate_keys,(keys(21),))
                self.assertFalse(batch.curriculum_candidate_plan['legacy_expected_keys_verifier_applied'])
                report = phase.report()
                self.assertNotIn('comparison_controls',report)
                self.assertEqual(report['legacy_controls_reference_only']['schedule'],'legacy_rotating')
                self.assertFalse(report['curriculum_candidate_execution']['legacy_expected_keys_verifier_applied'])
                with self.assertRaises(ValueError):
                    frozen.batch([76],'native',8,True)
            self.assertEqual(frozen.batch([76],'native',8,True).bridge_candidate_keys,(keys(49),))

    def test_unknown_facade_and_wrong_arm_are_not_unwrapped(self):
        class Other:
            def __init__(self): self._provider = PreparedFixture()
        with self.assertRaises(TypeError):
            with curriculum_provider(Other(),'native',POLICY):
                self.fail('Unknown wrapper accepted')
        module = ModuleType('hiercp_v1x.comparison_training')
        module._PolicyProvider = frozen_fixture_class()
        with patch.dict(sys.modules, {'hiercp_v1x.comparison_training':module}):
            with self.assertRaises(ValueError):
                with curriculum_provider(module._PolicyProvider(PreparedFixture(),'selected'),'native',POLICY):
                    self.fail('Cross-arm wrapper accepted')

    def test_reject_stale_epoch_flags_cross_arm_and_graph_loss(self):
        provider = PreparedFixture()
        with candidate_plan(provider,'native',POLICY,keys(49),mode='training',epoch=8) as phase:
            for args in (([76],'native',9,True), ([76],'selected',8,True),
                         ([76],'native',8,False), ([76,76],'native',8,True)):
                with self.subTest(args=args),self.assertRaises(ValueError):
                    phase.batch(*args)
            with self.assertRaises(ValueError):
                phase.cached_batch_ready([76],'native',7,training=True)
            with self.assertRaises(ValueError):
                phase.batch([76],'native',8,True,full=True)
            provider.tamper = True
            with self.assertRaises(ValueError):
                phase.batch([76],'native',8,True)
        self.assertIs(type(provider),PreparedFixture)

    def test_no_plan_allows_only_full129(self):
        provider = PreparedFixture()
        with curriculum_provider(provider,'selected',POLICY) as adapted:
            with self.assertRaises(RuntimeError):
                adapted.batch([76],'selected',1,True)
            full = adapted.batch([76],'selected',0,False,full=True)
            self.assertEqual(full.bridge_candidate_keys,(FULL_KEYS,))

    def test_plan_immutable_copied_and_policy_hash_separates_cache_context(self):
        mutable = list(keys())
        first = CandidatePlan('native',POLICY,mutable,'training',1)
        mutable[-1] = 'U:99'
        self.assertEqual(first.keys,keys())
        second = CandidatePlan('native','2'*64,keys(),'training',1)
        self.assertNotEqual(first.receipt()['sha256'],second.receipt()['sha256'])
        self.assertNotEqual(first.receipt()['sha256'],
            CandidatePlan('native',POLICY,keys(),'training',2).receipt()['sha256'])

    def test_malformed_keys_and_policy_are_rejected(self):
        cases = [keys(size=6), ('P','U:0','U:0',*keys()[3:]), ('U:1',*keys()[1:]),
                 ('P',*keys()[1:7],'U:128'), ('P',*keys()[1:7],'S:0')]
        for candidate in cases:
            with self.subTest(candidate=candidate),self.assertRaises(ValueError):
                CandidatePlan('native',POLICY,candidate,'training',1)
        with self.assertRaises(ValueError):
            CandidatePlan('native','unsealed',keys(),'training',1)
        with self.assertRaises(ValueError):
            CandidatePlan('native',POLICY,('P',),'stage_validation',0)

    def test_nested_adapter_and_plan_rejected(self):
        provider = PreparedFixture()
        with curriculum_provider(provider,'native',POLICY) as adapted:
            with self.assertRaises(RuntimeError):
                with curriculum_provider(provider,'native',POLICY): self.fail('Nested adapter')
            with adapted.plan(keys(),mode='training',epoch=1):
                with self.assertRaises(RuntimeError):
                    with adapted.plan(keys(7),mode='training',epoch=2): self.fail('Nested plan')
        self.assertIs(type(provider),PreparedFixture)
        with self.assertRaises(RuntimeError):
            adapted.batch([76],'native',1,True)

    def test_undrained_work_rejects_transition_and_restores_after_natural_drain(self):
        provider = PreparedFixture()
        provider.started, provider.release = threading.Event(), threading.Event()
        errors = []
        def prepare(phase):
            try: phase.batch([76],'native',1,True)
            except Exception as error: errors.append(error)
        thread = None
        try:
            with self.assertRaises(RuntimeError):
                with candidate_plan(provider,'native',POLICY,keys(),mode='training',epoch=1) as phase:
                    thread = threading.Thread(target=prepare,args=(phase,))
                    thread.start()
                    self.assertTrue(provider.started.wait(2))
            self.assertIsNot(type(provider),PreparedFixture)
        finally:
            provider.release.set()
            if thread is not None: thread.join(5)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors,[])
        self.assertIs(type(provider),PreparedFixture)


if __name__ == '__main__':
    unittest.main()
