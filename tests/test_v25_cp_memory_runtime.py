"""CPU UNIT lifetime/AST regressions; synthetic arrays are not patient results."""
import ast
from collections import OrderedDict
import copy
import inspect
from pathlib import Path
import tempfile
from types import SimpleNamespace
import threading
import unittest
from unittest.mock import patch
import weakref

import numpy as np

from hiercp_v1x import v25_cp as cp, v25_factory as factory
from hiercp_v1x import v25_preparation as preparation
from hiercp_v1x import v25_cp_memory_runtime as runtime


class Payload:
    pass


class Budget:
    rss_bytes = 64*2**30
    cuda_bytes = 40*2**30
    def __init__(self, checks=None):self.checks = [] if checks is None else checks
    def check(self):self.checks.append('original_hard_check')


class CPMemoryLifetimeUnit(unittest.TestCase):
    def setUp(self):
        self.original_case = preparation.AcceleratedNativeInputs._case
        self.addCleanup(setattr, preparation.AcceleratedNativeInputs, '_case', self.original_case)
        self.addCleanup(runtime._INSTALLED.clear)

    def test_normalization_proves_full_scientific_AST_and_declared_sizes_unchanged(self):
        tree, counts = runtime._transform_cp(cp.score_v25_cp_bank, authorized_B=True)
        original = ast.parse(inspect.getsource(cp.score_v25_cp_bank))
        self.assertEqual(runtime._dump(runtime._normal(tree)), runtime._dump(original))
        self.assertEqual(counts['release_batch_loop_aliases'], 1)
        self.assertEqual(counts['release_publication_payload'], 1)
        text = ast.unparse(runtime._normal(tree))
        for expected in ('physical=4', 'workers=4', 'max_workers=4', '128', '642'):
            self.assertIn(expected, text)
        case, counts = runtime._transform_case(factory.V25NativeInputs._case)
        self.assertEqual(counts, dict(release_snapshot_inputs=1, release_evicted_raw_alias=1))
        self.assertNotIn('del organ', ast.unparse(runtime._normal(case)))

    def activation_fixture(self, *, transformed):
        if transformed:
            tree, _ = runtime._transform_cp(cp.score_v25_cp_bank)
        else:tree = ast.parse(inspect.getsource(cp.score_v25_cp_bank))
        target = runtime._dump(ast.parse('for plan, item, graphs in prepared_batch: pass').body[0].target)
        loop = next(node for node in ast.walk(tree) if isinstance(node, ast.For)
            and runtime._dump(node.target) == target)
        statements = [copy.deepcopy(loop)]
        if transformed:
            statements.extend(ast.parse('del item, graphs').body)
        statements.extend(ast.parse('active.clear();active_graphs.clear();prepared_batch.clear();return check()').body)
        fixture = ast.parse('def exercise(prepared_batch, check):\n active={};active_graphs={};batch_plans=[]\n').body[0]
        fixture.body.extend(statements)
        namespace = {}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[fixture], type_ignores=[])),
            'CPU_UNIT_actual_CP_activation_subtree', 'exec'), namespace)
        item, graphs = Payload(), Payload();item.values = (1, 2, 3)
        refs = weakref.ref(item), weakref.ref(graphs)
        plan = SimpleNamespace(case_id='synthetic', donor_component=1, entry='entry')
        batch = [(plan, item, graphs)];del item, graphs
        observed = namespace['exercise'](batch, lambda:tuple(ref() is None for ref in refs))
        return observed

    def test_actual_batch_activation_loop_releases_last_raw_and_graph_before_next_batch(self):
        self.assertEqual(self.activation_fixture(transformed=False), (False, False))
        self.assertEqual(self.activation_fixture(transformed=True), (True, True))

    def test_actual_factory_case_drops_evicted_workspace_and_organ_before_RSS_guard(self):
        # Execute the complete original and transformed _case method against
        # synthetic decoded voxels and explicit local UNIT file witnesses.
        import hiercp.common as common
        results = [];decoded_alive_during_EDT = []
        with tempfile.TemporaryDirectory(prefix='UNIT_cp_lifetime_', dir=Path.cwd()/'tmp') as directory:
            image_path = Path(directory)/'image';label_path = Path(directory)/'label'
            image_path.write_bytes(b'UNIT');label_path.write_bytes(b'UNIT')
            for transformed in (False, True):
                old = Payload();old_ref = weakref.ref(old)
                organ_refs = []
                inputs = SimpleNamespace(_lock=threading.RLock(),
                    _build_lock=lambda key:threading.RLock(), guard_input_files=lambda case:None,
                    raw={'synthetic':dict(image=str(image_path),label=str(label_path),image_sha256='UNIT',label_sha256='UNIT')},
                    _raw_cache=OrderedDict(old=(old, 1)),_raw_bytes=1,raw_resident_bytes=1,
                    _donors={},_regions={},_file_proofs={})
                del old
                loaded = SimpleNamespace(image=np.ones((4,4,4),dtype=np.float32),
                    label=np.ones((4,4,4),dtype=np.uint8),spacing=np.ones(3),image_affine=np.eye(4))
                decoded_ref = weakref.ref(loaded.image)
                from hiercp_v1x.v24_inputs import RecipientContext
                original_depth = RecipientContext.organ_depth.func
                def depth(context):
                    decoded_alive_during_EDT.append(decoded_ref() is not None)
                    return original_depth(context)
                def snapshot(*args):
                    organ_refs.append(weakref.ref(args[2]))
                    return factory.recipient_context(*args)
                observed = []
                inputs._rss_locked = lambda:observed.append((old_ref() is None,organ_refs[0]() is None))
                if transformed:tree, _ = runtime._transform_case(factory.V25NativeInputs._case)
                else:tree = ast.parse(__import__('textwrap').dedent(inspect.getsource(factory.V25NativeInputs._case)))
                function = runtime._compile(tree, factory.V25NativeInputs._case,
                    dict(factory.V25NativeInputs._case.__globals__,sha=lambda path:'UNIT',recipient_context=snapshot))
                with patch.object(common,'load_case',return_value=loaded), \
                        patch.object(RecipientContext.organ_depth,'func',side_effect=depth):
                    value = function(inputs, 'synthetic')
                self.assertEqual(value['context'].image.shape, (4,4,4))
                self.assertEqual(value['context'].image.sum(), 64)
                self.assertFalse(value['context'].image.flags.writeable)
                results.append(observed)
        self.assertEqual(results, [[(False,False)],[(True,True)]])
        self.assertEqual(decoded_alive_during_EDT, [True,False])

    def process(self, value):
        return SimpleNamespace(memory_info=lambda:SimpleNamespace(rss=value()))

    def inputs(self, events):
        return SimpleNamespace(rss_bytes=64*2**30,raw_resident_bytes=8*2**30,
            memory_coordinator=SimpleNamespace(providers=[SimpleNamespace(resident_bytes=32*2**30)]),
            _rss=lambda:events.append('original_input_coordinator'))

    def test_pressure_releases_only_cache_reference_and_retains_active_input_values(self):
        events = [];payload = Payload();payload.values = np.arange(8)
        ref = weakref.ref(payload);cache = OrderedDict(patient=payload)
        active = payload;del payload
        values = active.values.copy();rss = [50*2**30]
        def release():cache.clear();events.append('collection_cache_released');return 100
        budget = runtime._LifetimeBudget(Budget(events), self.inputs(events), release)
        def trim():events.append('allocator_hint');rss[0] = 40*2**30;return {'supported':True,'applied':True}
        with patch.object(runtime.psutil,'Process',side_effect=lambda:self.process(lambda:rss[0])), \
                patch.object(runtime.gc,'collect',return_value=0),patch.object(runtime,'_trim_heap',side_effect=trim):
            budget.check()
        self.assertIs(ref(),active);np.testing.assert_array_equal(active.values,values)
        self.assertEqual(events,['collection_cache_released','allocator_hint','original_input_coordinator','original_hard_check'])
        self.assertEqual(budget.rss_bytes,64*2**30)
        self.assertEqual(budget.profile['last_reclamation']['after_RSS_bytes'],40*2**30)

    def test_inactive_collection_cycle_is_dead_before_allocator_check(self):
        events = [];payload = Payload();payload.cycle = payload
        reference = weakref.ref(payload);cache = OrderedDict(patient=payload);del payload
        collect = runtime.gc.collect
        def release():cache.clear();return 100
        budget = runtime._LifetimeBudget(Budget(events),self.inputs(events),release)
        rss = [50*2**30]
        def trim():
            self.assertIsNone(reference());rss[0] = 20*2**30
            return {'supported':True,'applied':True}
        with patch.object(runtime.psutil,'Process',side_effect=lambda:self.process(lambda:rss[0])), \
                patch.object(runtime.gc,'collect',side_effect=collect),patch.object(runtime,'_trim_heap',side_effect=trim):
            budget.check()
        self.assertEqual(budget.profile['collection_bytes_released'],100)

    def test_true_active_overflow_still_calls_original_hard_error_and_never_reduces_batch(self):
        events = [];inputs = self.inputs(events)
        original = Budget(events)
        original.check = lambda:(_ for _ in ()).throw(MemoryError('original64GiB'))
        budget = runtime._LifetimeBudget(original,inputs,lambda:0)
        with patch.object(runtime.psutil,'Process',side_effect=lambda:self.process(lambda:70*2**30)), \
                patch.object(runtime.gc,'collect',return_value=0),patch.object(runtime,'_trim_heap',return_value={'supported':False}), \
                self.assertRaisesRegex(MemoryError,'original64GiB'):
            budget.check()
        self.assertEqual(budget.rss_bytes,64*2**30)

    def test_no_pressure_means_no_extra_eviction_or_allocator_work(self):
        budget = runtime._LifetimeBudget(Budget(),self.inputs([]),lambda:(_ for _ in ()).throw(AssertionError('eviction')))
        with patch.object(runtime.psutil,'Process',side_effect=lambda:self.process(lambda:48*2**30)), \
                patch.object(runtime.gc,'collect',side_effect=AssertionError('GC')):
            budget.check()
        self.assertEqual(budget.profile['pressure_events'],0)
        self.assertEqual(budget._budget.checks,['original_hard_check'])

    def test_installer_private_B_resource_route_and_frozen_SHA_declaration_preserved(self):
        from hiercp_v1x.v25_authorized_early_downstream import _private_clone
        clone = _private_clone(cp)
        before = {name:path.read_bytes() for name,path in
            [('v25_cp.py',Path(cp.__file__)),('v25_factory.py',Path(factory.__file__))]}
        declaration = cp._implementation_sources()
        proof = runtime.install_runtime(cp_module=clone,
            expected_frozen_source_sha256={name:__import__('hashlib').sha256(value).hexdigest() for name,value in before.items()})
        self.assertEqual(proof,runtime.install_runtime(cp_module=clone))
        self.assertEqual(declaration,cp._implementation_sources())
        self.assertEqual(before,{name:(Path(cp.__file__).parent/name).read_bytes() for name in before})
        text = ast.unparse(runtime._transform_cp(cp.score_v25_cp_bank,authorized_B=True)[0])
        self.assertIn('from .v25_B_early_downstream_resources import BEarlyDownstreamAllocationLease as AllocationLease',text)
        self.assertNotIn('from .v25_server_resources import AllocationLease',text)
        self.assertEqual(proof['physical_patient_batch'],4)
        self.assertEqual(proof['CPU_workers'],4)
        self.assertEqual(proof['RSS_limit_bytes'],64*2**30)

    def test_foreign_scorer_override_and_wrong_frozen_SHA_fail_closed(self):
        from hiercp_v1x.v25_authorized_early_downstream import _private_clone
        clone = _private_clone(cp)
        with self.assertRaisesRegex(ValueError,'source SHA'):
            runtime.install_runtime(cp_module=clone,expected_frozen_source_sha256={})
        clone.score_v25_cp_bank = lambda **kwargs:None
        with self.assertRaisesRegex(ValueError,'Unknown existing CP scorer'):
            runtime.install_runtime(cp_module=clone)

    def test_changed_release_anchor_fails_exact_AST_whitelist(self):
        source = inspect.getsource(cp.score_v25_cp_bank).replace('for plan, item, graphs in prepared_batch:',
            'for plan, changed_item, graphs in prepared_batch:')
        with patch.object(runtime.inspect,'getsource',return_value=source),self.assertRaisesRegex(ValueError,'AST whitelist'):
            runtime._transform_cp(cp.score_v25_cp_bank)


if __name__ == '__main__':unittest.main()
