"""CPU DEBUG fixtures; no clinical generation, CUDA, training or scale claim."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
from scipy import ndimage as ndi
from concurrent.futures import Future

from hiercp_v1x import basic_cp_ablation_bank as bank


ROOT = Path(__file__).resolve().parents[1]
TMP = ROOT/"tmp"


def entry_fixture(count=2):
    return dict(source_data=np.arange(8, dtype=np.float32).reshape(1, 2, 2, 2),
                source_mask=np.ones((2, 2, 2), dtype=np.uint8),
                anchor_offset=np.ones(3, dtype=np.int16),
                candidate_centers=np.array([(i+2, 2, 2) for i in range(count)], dtype=np.int32),
                candidate_raw_centers=np.array([(i+2, 2, 2) for i in range(count)], dtype=np.int32),
                scores=np.arange(count, dtype=np.float32)/max(count, 1),
                source_component=np.array([1], dtype=np.int16),
                source_diameter_mm=np.array([8.0], dtype=np.float32))


def index_fixture():
    # Explicit CPU DEBUG catalogue has the required full642/81 cardinality.
    groups = {f"liver_{i}": [] for i in range(81)}
    for i in range(642):
        groups[f"liver_{i%81}"].append(f"entries/liver_{i%81}__component_{i:03d}.npz")
    return dict(format="hiercp_online_bank_v2", candidate_count=128,
                source_entries=642, eligible_cases=81, total_candidates=82176,
                entries_by_case=groups, minimum_liver_coverage=.85,
                liver_label=1, tumor_label=2, draw_count=256, attempts=4, cp_probability=.5)


class EntryGeometryTests(unittest.TestCase):
    def setUp(self):
        self.entry = entry_fixture()
        self.seg = np.ones((1, 12, 12, 12), dtype=np.int16)

    def extend(self, proposals, target=4):
        return bank.extend_entry(self.entry, proposals, self.seg, liver_label=1,
            tumor_label=2, minimum_coverage=.85, original_count=2, candidate_count=target)

    def test_source_scores_and_original_prefix_are_byte_exact(self):
        result, proof = self.extend([(np.array(p), np.array(p)) for p in [(5, 5, 5), (8, 8, 8)]])
        for key, value in self.entry.items():
            actual = result[key][:2] if key.startswith("candidate_") else result[key]
            self.assertEqual(actual.dtype, value.dtype)
            self.assertEqual(actual.tobytes(), value.tobytes())
        self.assertEqual(result['scores'].shape, (2,))
        self.assertFalse(proof['additional_GNN_scores_generated'])
        self.assertFalse(proof['scores_used'])

    def test_raw_and_preprocessed_duplicate_rejection(self):
        proposals = [(np.array(r), np.array(p)) for r,p in [
            ((2,2,2),(6,6,6)), ((6,6,6),(2,2,2)),
            ((7,7,7),(7,7,7)), ((7,7,7),(9,9,9)), ((9,9,9),(9,9,9))]]
        result, proof = self.extend(proposals)
        self.assertEqual(proof['duplicate_raw'], 2)
        self.assertEqual(proof['duplicate_preprocessed'], 1)
        np.testing.assert_array_equal(result['candidate_centers'][2:], [(7,7,7),(9,9,9)])

    def test_exact_coverage_threshold_and_tumor_mask(self):
        source = np.ones((1,1,20), dtype=np.uint8)
        seg = np.ones((1,2,2,24), dtype=np.int16)
        seg[0,0,0,17:20] = 0
        valid = lambda: bank.preprocessed_valid(np.array([0,0,0]),source,np.array([0,0,0]),
            seg,liver_label=1,tumor_label=2,minimum_coverage=.85)
        self.assertTrue(valid())  # 17/20 exactly .85
        seg[0,0,0,16] = 0
        self.assertFalse(valid())
        seg[0,0,0,:20] = 1
        seg[0,0,0,0] = 2
        self.assertFalse(valid())

    def test_full_source_bounds_even_where_mask_is_empty(self):
        self.entry['source_mask'][0] = 0
        self.assertFalse(bank.preprocessed_valid(np.array([0,0,0]), self.entry['source_mask'],
            self.entry['anchor_offset'],self.seg,liver_label=1,tumor_label=2,minimum_coverage=.85))

    def test_tumor_outside_pasted_mask_does_not_reject(self):
        mask = np.zeros((2,2,2), dtype=np.uint8)
        mask[0,0,0] = 1
        self.seg[0,5,5,5] = 2
        self.assertTrue(bank.preprocessed_valid(np.array([5,5,5]),mask,np.ones(3,dtype=int),
            self.seg,liver_label=1,tumor_label=2,minimum_coverage=.85))

    def test_shortfall_never_publishes_smaller_bank(self):
        with self.assertRaisesRegex(bank.BankContractError, 'no source dropped'):
            self.extend([(np.array([5,5,5]),np.array([5,5,5]))])

    def test_original_illegal_position_fails(self):
        self.seg[0,1,1,1] = 2
        with self.assertRaisesRegex(bank.BankContractError, 'Original prefix'):
            self.extend([])

    def test_nonfinite_old_score_is_rejected_not_replaced(self):
        self.entry['scores'][0] = np.nan
        with self.assertRaisesRegex(bank.BankContractError, 'finite scores'):
            self.extend([])

    def test_preservation_detects_patch_score_and_prefix_mutation(self):
        result, _ = self.extend([(np.array(p),np.array(p)) for p in [(5,5,5),(8,8,8)]])
        for key in ['source_data','source_mask','anchor_offset','scores','candidate_centers']:
            damaged = copy.deepcopy(result)
            damaged[key].flat[0] += 1
            with self.assertRaises(bank.BankContractError):
                bank.verify_preservation(self.entry,damaged,original_count=2,candidate_count=4)

    def test_full_production128_to256_array_contract(self):
        entry = entry_fixture(128)
        seg = np.ones((1,260,4,4), dtype=np.int16)
        proposals = [(np.array([i,2,2]),np.array([i,2,2])) for i in range(130,258)]
        result, proof = bank.extend_entry(entry,proposals,seg,liver_label=1,tumor_label=2,
            minimum_coverage=.85,original_count=128,candidate_count=256)
        self.assertEqual(result['candidate_centers'].shape,(256,3))
        self.assertEqual(result['scores'].shape,(128,))
        self.assertTrue(proof['original_fields_and_first128_preserved'])


class HistoricalStaticCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        receipt = ROOT.parents[1]/'outputs/v24_gpu6_recovery_20261011/inspect_basic256_historical_code_READONLY_20261013_1791836048882408400.log'
        actual = json.loads(receipt.read_text(encoding='utf8'))
        source = actual['sources']['hiercp/common.py']
        if source['sha256'] != bank.COMMON_SHA256:
            raise AssertionError('Exact historical common.py receipt required')
        text = source['snippets']['build_candidate_pool']
        namespace = dict(LoadedCase=object,SourceTumor=object,CandidateInfo=lambda **values: types.SimpleNamespace(**values),
            np=np,ndi=ndi,context_ring_mask=lambda mask,width: ndi.binary_dilation(mask,iterations=width)&~mask,
            distance_to_mask_mm=lambda mask,spacing: ndi.distance_transform_edt(~mask,sampling=spacing),
            slices_for_center=lambda center,shape,volume: tuple(slice(c-s//2,c-s//2+s) for c,s in zip(center,shape)),
            sample_candidate_centers=lambda mask,shape,**kwargs: [(3,3,3),(8,8,8),(12,12,12)])
        exec(compile(text,'<verified historical common.py CPU DEBUG>','exec'),namespace)
        cls.original = staticmethod(namespace['build_candidate_pool'])

    def setUp(self):
        self.occupied = np.zeros((16,16,16),dtype=bool)
        self.occupied[3,3,3] = True
        self.spacing = np.array([1.,2.,1.],dtype=np.float32)
        self.case = types.SimpleNamespace(shape=self.occupied.shape,spacing=self.spacing,
            image=np.arange(self.occupied.size,dtype=np.float32).reshape(self.occupied.shape))
        self.source = types.SimpleNamespace(patch_mask=np.ones((2,2,2),dtype=bool),voxel_count=8)
        self.kwargs = dict(placement_mask=np.ones_like(self.occupied),full_organ_mask=np.ones_like(self.occupied),
            occupied_mask=self.occupied,organ_distance=np.ones(self.occupied.shape,dtype=np.float32),
            num_candidates=2,max_draws=100,min_liver_coverage=.85,occupied_clearance_vox=2,min_center_separation_mm=1.)

    def test_exact_old_function_outputs_preserved_and_static_work_once(self):
        cached, counts = bank.cached_historical_pool(self.original,self.occupied,self.spacing)
        original, original_distance = self.original(self.case,self.source,rng=np.random.default_rng(42),**self.kwargs)
        first, first_distance = cached(self.case,self.source,rng=np.random.default_rng(42),**self.kwargs)
        second, second_distance = cached(self.case,self.source,rng=np.random.default_rng(42),**self.kwargs)
        self.assertEqual([vars(c) for c in original],[vars(c) for c in first])
        self.assertEqual([vars(c) for c in first],[vars(c) for c in second])
        np.testing.assert_array_equal(original_distance,first_distance)
        self.assertIs(first_distance,second_distance)
        self.assertEqual(counts['occupied_distance_computations'],1)
        self.assertEqual(counts['forbidden_computations'],1)
        self.assertFalse(first_distance.flags.writeable)
        self.assertIsNot(cached.__globals__,self.original.__globals__)
        self.assertIs(cached.__code__,self.original.__code__)

    def test_other_patient_or_clearance_cannot_hit_stale_cache(self):
        cached, _ = bank.cached_historical_pool(self.original,self.occupied,self.spacing)
        cached(self.case,self.source,rng=np.random.default_rng(42),**self.kwargs)
        other = dict(self.kwargs,occupied_mask=self.occupied.copy())
        with self.assertRaisesRegex(bank.BankContractError,'input changed'):
            cached(self.case,self.source,rng=np.random.default_rng(42),**other)
        other = dict(self.kwargs,occupied_clearance_vox=3)
        with self.assertRaisesRegex(bank.BankContractError,'geometry changed'):
            cached(self.case,self.source,rng=np.random.default_rng(42),**other)

    def test_other_spacing_cannot_hit_stale_distance(self):
        cached, _ = bank.cached_historical_pool(self.original,self.occupied,self.spacing)
        self.case.spacing = np.array([2.,2.,2.])
        with self.assertRaisesRegex(bank.BankContractError,'input changed'):
            cached(self.case,self.source,rng=np.random.default_rng(42),**self.kwargs)


class FullPopulationAndPublicationTests(unittest.TestCase):
    def setUp(self):
        TMP.mkdir(exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(prefix='DEBUG_bank256_',dir=TMP)
        self.root = Path(self.directory.name)

    def tearDown(self):
        self.directory.cleanup()

    def test_original_full_source_catalogue_required(self):
        original = index_fixture()
        bank.validate_original_index(original)
        for field,value in [('candidate_count',127),('source_entries',641),('eligible_cases',80),('cp_probability',.6)]:
            changed = copy.deepcopy(original)
            changed[field] = value
            with self.assertRaises(bank.BankContractError):
                bank.validate_original_index(changed)
        changed = copy.deepcopy(original)
        changed['entries_by_case']['liver_0'][0] = changed['entries_by_case']['liver_1'][0]
        with self.assertRaisesRegex(bank.BankContractError,'identity list'):
            bank.validate_original_index(changed)

    def test_no_source_path_escape(self):
        with self.assertRaises(bank.BankContractError):
            bank.safe_entry(self.root,'../outside.npz')
        with self.assertRaises(bank.BankContractError):
            bank.safe_entry(self.root,'entries/fake.json')

    def test_exclusive_publication_and_readback(self):
        path = self.root/'proof.json'
        bank.write_json_exclusive(path,{'complete':False})
        self.assertEqual(bank.read_json(path),{'complete':False})
        with self.assertRaises(FileExistsError):
            bank.write_json_exclusive(path,{'complete':True})
        bank.publish_progress(self.root,{'status':'PREPARING','source_completed':1})
        bank.publish_progress(self.root,{'status':'COMPLETE','source_completed':642})
        self.assertEqual(bank.read_json(self.root/'generation_progress.json')['source_completed'],642)
        self.assertEqual(list(self.root.glob('*.tmp')),[])

    def test_worker_metadata_initialized_once_not_per_patient(self):
        with patch.object(bank,'_WORKER_PROVIDER',None),patch.object(bank,'_WORKER_BINDING',None), \
                patch.object(bank,'HistoricalGeometryProvider') as provider:
            bank._initialize_worker({'DEBUG':True})
            self.assertEqual(provider.call_count,1)
            with self.assertRaisesRegex(bank.BankContractError,'initialize exactly once'):
                bank._initialize_worker({'DEBUG':True})

    def test_same_initialized_worker_handles_two_patients_and_npz_roundtrip(self):
        original = self.root/'original'
        output = self.root/'output'
        original.mkdir()
        output.mkdir()
        provider = types.SimpleNamespace(bank={'liver_label':1,'tumor_label':2},
            load_patient=lambda case: np.ones((1,260,4,4),dtype=np.int16),
            proposals=lambda case,entry: ((np.array([i,2,2]),np.array([i,2,2])) for i in range(130,258)),
            last_proposal_diagnostics={'graph_rejected':0},
            patient_cache={'forbidden_computations':1,'occupied_distance_computations':1},
            verify_patient_inputs=lambda: None)
        binding = {'bank_index':str(original/'index.json')}
        for case in ['DEBUG_A','DEBUG_B']:
            with (original/(case+'.npz')).open('xb') as stream:
                np.savez(stream,**entry_fixture(128))
        with patch.object(bank,'_WORKER_PROVIDER',provider),patch.object(bank,'_WORKER_BINDING',binding), \
                patch.object(bank,'HistoricalGeometryProvider') as forbidden_constructor:
            for case in ['DEBUG_A','DEBUG_B']:
                rows = bank._build_patient(binding,case,[case+'.npz'],str(output))
                self.assertEqual(len(rows),1)
                result = bank.read_entry(output/(case+'.npz'))
                self.assertEqual(result['candidate_centers'].shape,(256,3))
                self.assertEqual(result['scores'].shape,(128,))
                self.assertTrue((output/'entry_receipts'/(case+'.json')).is_file())
            forbidden_constructor.assert_not_called()

    def test_generation_failure_has_visible_receipt_without_publish_index(self):
        original = self.root/'original'
        original.mkdir()
        bank.write_json_exclusive(original/'index.json',index_fixture())
        output = self.root/'new'

        class FailingExecutor:
            def __init__(self,**options):
                self.options = options
            def __enter__(self): return self
            def __exit__(self,*args):
                # Failure must be visible while executor shutdown is still pending.
                self.assert_failure_visible()
                return False
            def submit(self,*args):
                future = Future()
                future.set_exception(bank.BankContractError('DEBUG exact geometry shortfall'))
                return future
            @staticmethod
            def assert_failure_visible():
                if bank.read_json(output/'generation_progress.json')['status'] != 'FAILED':
                    raise AssertionError('Failure hidden until executor shutdown')
                if bank.read_json(output/'preparation_abort.json')['stop_remaining'] is not True:
                    raise AssertionError('No cooperative worker abort before shutdown')

        with patch.object(bank,'file_sha',return_value=bank.BANK_SHA256), \
                patch.object(bank,'cpu_affinity',return_value=[0,1,2,3]), \
                patch.object(bank,'process_tree_rss',return_value=100), \
                patch.object(bank,'ProcessPoolExecutor',FailingExecutor):
            with self.assertRaisesRegex(bank.BankContractError,'geometry shortfall'):
                bank.build_bank({'bank_index':str(original/'index.json')},output,workers=4,rss_limit_gib=64)
        self.assertEqual(bank.read_json(output/'generation_progress.json')['status'],'FAILED')
        self.assertFalse(bank.read_json(output/'preparation_failed.json')['complete'])
        self.assertFalse((output/'index.json').exists())
        self.assertFalse((output/'complete.json').exists())

    def test_abort_marker_stops_worker_before_loading_another_patient(self):
        output=self.root/'output';output.mkdir()
        bank.write_json_exclusive(output/'preparation_abort.json',
            {'format':bank.FORMAT,'stop_remaining':True,'error':'DEBUG lease expired'})
        provider=types.SimpleNamespace(load_patient=lambda case: self.fail('Patient decoded after abort'))
        binding={'bank_index':str(self.root/'index.json')}
        with patch.object(bank,'_WORKER_PROVIDER',provider),patch.object(bank,'_WORKER_BINDING',binding):
            with self.assertRaisesRegex(bank.BankContractError,'cooperatively stopped'):
                bank._build_patient(binding,'DEBUG',['entry.npz'],str(output))

    def test_initializer_checks_abort_before_importing_historical_geometry(self):
        output=self.root/'output';output.mkdir()
        bank.write_json_exclusive(output/'preparation_abort.json',
            {'format':bank.FORMAT,'stop_remaining':True,'error':'DEBUG source failed'})
        with patch.object(bank,'_WORKER_PROVIDER',None),patch.object(bank,'_WORKER_BINDING',None), \
                patch.object(bank,'_PREPARATION_ABORT_ROOT',None),patch.object(bank,'HistoricalGeometryProvider') as provider:
            with self.assertRaisesRegex(bank.BankContractError,'cooperatively stopped'):
                bank._initialize_worker({'DEBUG':True},str(output))
            provider.assert_not_called()

    def test_source_failure_includes_patient_entry_component_and_original_cause(self):
        original=self.root/'original';output=self.root/'output';original.mkdir();output.mkdir()
        path=original/'DEBUG_source.npz';np.savez(path,**entry_fixture(128))
        def unavailable(*args):raise ValueError('DEBUG scientific input missing')
        provider=types.SimpleNamespace(bank={'liver_label':1,'tumor_label':2},
            load_patient=lambda cid:np.ones((1,270,4,4),dtype=np.int16),
            proposals=unavailable,last_proposal_diagnostics={'attempts':5,'accepted_added':106})
        binding={'bank_index':str(original/'index.json')}
        with patch.object(bank,'_WORKER_PROVIDER',provider),patch.object(bank,'_WORKER_BINDING',binding):
            with self.assertRaisesRegex(bank.BankContractError,'case=DEBUG entry=DEBUG_source.npz component=1') as caught:
                bank._build_patient(binding,'DEBUG',['DEBUG_source.npz'],str(output))
        self.assertIsInstance(caught.exception.__cause__,ValueError)
        self.assertIn('scientific input missing',str(caught.exception))

    def test_optional_controller_lease_cached_and_verified_without_signals(self):
        from hiercp_v1x import basic_cp_ablation_resources as resources
        path = self.root/'lease.json'
        bank.write_json_exclusive(path,{'DEBUG':True})
        with patch.dict(os.environ,{'BASIC_CP_LEASE_REQUEST':str(path)}), \
                patch.object(bank,'_PREPARATION_LEASE',None),patch.object(bank,'_PREPARATION_LEASE_BINDING',None), \
                patch.object(resources,'BasicLease') as lease:
            bank.preparation_lease_verify()
            bank.preparation_lease_verify()
            lease.assert_called_once_with({'DEBUG':True})
            self.assertEqual(lease.return_value.verify.call_count,2)
            path.write_text('{"DEBUG":false}',encoding='utf8')
            with self.assertRaisesRegex(bank.BankContractError,'request changed'):
                bank.preparation_lease_verify()

    def test_controller_expiry_error_propagates_cooperatively(self):
        from hiercp_v1x import basic_cp_ablation_resources as resources
        path = self.root/'lease.json'
        bank.write_json_exclusive(path,{'DEBUG':True})
        with patch.dict(os.environ,{'BASIC_CP_LEASE_REQUEST':str(path)}), \
                patch.object(bank,'_PREPARATION_LEASE',None),patch.object(bank,'_PREPARATION_LEASE_BINDING',None), \
                patch.object(resources,'BasicLease') as lease:
            lease.return_value.verify.side_effect = RuntimeError('DEBUG expired owned lifetime')
            with self.assertRaisesRegex(RuntimeError,'expired owned lifetime'):
                bank.preparation_lease_verify()

    def test_standalone_without_lease_environment_has_no_resource_side_effect(self):
        from hiercp_v1x import basic_cp_ablation_resources as resources
        with patch.dict(os.environ,{},clear=True),patch.object(bank,'_PREPARATION_LEASE',None), \
                patch.object(resources,'BasicLease') as lease:
            bank.preparation_lease_verify()
            lease.assert_not_called()

    def test_final_build_refuses_serial_unverified_sha_or_existing_output(self):
        source = self.root/'original'
        source.mkdir()
        index = source/'index.json'
        bank.write_json_exclusive(index,index_fixture())
        binding = {'bank_index':str(index)}
        with self.assertRaisesRegex(bank.BankContractError,'SHA differs'):
            bank.build_bank(binding,self.root/'new',workers=4,rss_limit_gib=64)
        with patch.object(bank,'file_sha',return_value=bank.BANK_SHA256), \
                patch.object(bank,'cpu_affinity',return_value=[0,1,2,3]):
            with self.assertRaisesRegex(bank.BankContractError,'>=2'):
                bank.build_bank(binding,self.root/'new',workers=1,rss_limit_gib=64)
            with self.assertRaisesRegex(bank.BankContractError,'CPU affinity'):
                bank.build_bank(binding,self.root/'new',workers=5,rss_limit_gib=64)
            with self.assertRaisesRegex(bank.BankContractError,'RAM limit'):
                bank.build_bank(binding,self.root/'new',workers=4,rss_limit_gib=0)
            (self.root/'new').mkdir()
            with self.assertRaises(FileExistsError):
                bank.build_bank(binding,self.root/'new',workers=4,rss_limit_gib=64)


class HistoricalExtendedSearchTests(unittest.TestCase):
    """CPU DEBUG provider tests seed order and full256, not clinical geometry."""
    def setUp(self):
        self.calls=[]
        self.provider=object.__new__(bank.HistoricalGeometryProvider)
        def seed(*parts):
            if len(parts)>3 and parts[-2]=='online_pool':
                self.calls.append(parts);return int(parts[-1])
            return 42
        self.provider.benchmark=types.SimpleNamespace(stable_seed=seed,
            _source_from_component=lambda *args:object(),
            _map_raw_point=lambda raw,*args:np.asarray(raw,dtype=np.int32))
        self.provider.global_seed=42;self.provider.bank={'attempts':4,'draw_count':256,'liver_label':1,'tumor_label':2}
        self.provider.generation=dict(bank.GENERATION)
        self.provider.case=types.SimpleNamespace(spacing=np.ones(3))
        self.provider.components=object();self.provider.graph=object();self.provider.prototype=object()
        self.provider.regions=types.SimpleNamespace(full_organ_mask=np.ones(1,dtype=bool),organ_depth=np.ones(1))
        self.provider.ct_clip=(-200.,250.);self.provider.plans={'transpose_forward':[0,1,2]}
        self.provider.properties={};self.provider.preseg=np.ones((1,270,4,4),dtype=np.int16)
        self.provider.placement_mask=np.ones(1,dtype=bool);self.provider.occupied_mask=np.zeros(1,dtype=bool)
        self.provider.prepare=lambda *args,**kwargs:object()
        self.provider.specs=lambda pool,*args,**kwargs:[None]*len(pool)
        self.provider.geometry_error=RuntimeError
        self.provider.local=lambda *args,**kwargs:object()
        self.pool_attempt=0
        self.pool_parameters=[]
        def pool(*args,**kwargs):
            attempt=self.pool_attempt;self.pool_attempt+=1
            self.pool_parameters.append(kwargs)
            # Firstfour existing pools have only original-prefix positions.
            center=(2+attempt,2,2) if attempt<4 else (130+attempt-4,2,2)
            return [types.SimpleNamespace(center=center)],None
        self.provider.patient_pool=pool

    def test_initial_four_exhausted_then_deterministic_more_pools_complete256(self):
        original=entry_fixture(128)
        result,proof=bank.extend_entry(original,self.provider.proposals('DEBUG',original),self.provider.preseg,
            liver_label=1,tumor_label=2,minimum_coverage=.85,original_count=128,candidate_count=256)
        self.assertEqual(result['candidate_centers'].shape,(256,3))
        self.assertEqual(result['scores'].shape,(128,))
        self.assertEqual([parts[-1] for parts in self.calls],list(range(132)))
        self.assertEqual(self.calls[:4],[(42,'DEBUG',1,'online_pool',i) for i in range(4)])
        self.assertEqual(self.provider.last_proposal_diagnostics['accepted_added'],128)
        self.assertEqual(self.provider.last_proposal_diagnostics['raw_duplicate'],4)
        self.assertEqual(self.provider.last_proposal_diagnostics['attempts'],132)
        self.assertTrue(proof['original_fields_and_first128_preserved'])
        self.assertTrue(all(p['min_liver_coverage']==.85 and p['occupied_clearance_vox']==2
            and p['min_center_separation_mm']==12. and p['num_candidates']==256
            and p['max_draws']==128000 for p in self.pool_parameters))

    def test_continuing_search_never_bypasses_cooperative_resource_failure(self):
        original=entry_fixture(128)
        def lease():
            if len(self.calls)>=4:raise RuntimeError('DEBUG host allocation expired')
        with patch.object(bank,'preparation_lease_verify',side_effect=lease):
            with self.assertRaisesRegex(RuntimeError,'allocation expired'):
                list(self.provider.proposals('DEBUG',original))
        self.assertEqual(len(self.calls),4)
        self.assertEqual(self.provider.last_proposal_diagnostics['accepted_added'],0)


class PreparedBankReuseTests(unittest.TestCase):
    """Full642 CPU DEBUG catalogue tests actual hashing/arrays; no clinical claim."""
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix='prepared256_debug_', dir=TMP)
        cls.root = Path(cls.temporary.name).resolve()
        cls.original, cls.extended = cls.root/'original', cls.root/'extended'
        cls.original.mkdir(); cls.extended.mkdir()
        original = index_fixture()
        cls.original_index = cls.original/'index.json'
        bank.write_json_exclusive(cls.original_index, original)
        cls.original_sha = bank.file_sha(cls.original_index)
        cls.binding_path = cls.root/'binding.json'
        cls.binding = {'bank_index': str(cls.original_index), 'DEBUG_fixture': True}
        bank.write_json_exclusive(cls.binding_path, cls.binding)
        cls.binding_sha = bank.file_sha(cls.binding_path)
        entry = entry_fixture(128)
        extended = entry_fixture(256)
        extended['scores'] = entry['scores'].copy()
        proof = bank.verify_preservation(entry, extended, original_count=128, candidate_count=256)
        rows = []
        for cid, names in original['entries_by_case'].items():
            for name in names:
                old, new = cls.original/name, cls.extended/name
                old.parent.mkdir(parents=True,exist_ok=True);new.parent.mkdir(parents=True,exist_ok=True)
                np.savez(old,**entry);np.savez(new,**extended)
                rows.append({'case_id':cid,'entry':name,'original_entry_sha256':bank.file_sha(old),
                    'original_stat':bank.stat_identity(old),'new_entry_sha256':bank.file_sha(new),'proof':proof})
        manifest = {'format':bank.FORMAT,'entries':sorted(rows,key=lambda r:r['entry']),
                    'source_entries':642,'eligible_cases':81}
        bank.write_json_exclusive(cls.extended/'preservation_manifest.json',manifest)
        ablation = {'format':bank.FORMAT,'uniform_candidate_count':256,'scores_used':False,
            'historical_score_count':128,'original_candidate_count':128,
            'additional_GNN_scores_generated':False,'original_index':str(cls.original_index),
            'original_index_sha256':cls.original_sha,'geometry':bank.GENERATION,
            'raw_pool_draw_count':256,'raw_pool_max_draws':128000,'raw_pool_attempts':4,
            'historical_graph_geometry_required':True,
            'only_intervention':'uniform candidate universe128->256; original source/order/patch/RNG/appearance/CP probability retained'}
        index = dict(original,candidate_count=256,total_candidates=164352,ablation=ablation,
                     manifest=str(cls.extended/'preservation_manifest.json'))
        bank.write_json_exclusive(cls.extended/'index.json',index)
        started = {'format':bank.FORMAT,'binding':cls.binding,'original_index_sha256':cls.original_sha,
            'full_sources':642,'original_candidate_count':128,'candidate_count':256,
            'CUDA_model_or_scoring_executed':False}
        bank.write_json_exclusive(cls.extended/'preparation_started.json',started)
        complete = {'format':bank.FORMAT,'complete':True,'candidate_count':256,'source_entries':642,
            'eligible_cases':81,'total_candidates':164352,'index_sha256':bank.file_sha(cls.extended/'index.json'),
            'original_index_sha256':cls.original_sha,
            'preservation_manifest_sha256':bank.file_sha(cls.extended/'preservation_manifest.json'),
            'all_original_arrays_and_first128_preserved':True,'all_256_unique_raw_and_preprocessed':True,
            'all_added128_historical_raw_preprocessed_graph_valid':True,
            'additional_GNN_scores_generated':False,'training_executed':False,'CUDA_initialized':False,
            'raw_CT_preprocessing_executed':False,'patient_region_cache_rebuilt':False,
            'shared_preprocessed_arrays_written':False,
            'entries_sha256':{row['entry']:row['new_entry_sha256'] for row in rows}}
        bank.write_json_exclusive(cls.extended/'complete.json',complete)
        cls.documents = {name:(cls.extended/name).read_bytes() for name in
            ['index.json','complete.json','preservation_manifest.json','preparation_started.json']}
        cls.entry_name = sorted(complete['entries_sha256'])[0]
        cls.entry_bytes = (cls.extended/cls.entry_name).read_bytes()
        cls.partial_receipts = {}
        for row in sorted(rows,key=lambda row:row['entry'])[:2]:
            path=cls.extended/'entry_receipts'/Path(row['entry']).with_suffix('.json')
            path.parent.mkdir(parents=True,exist_ok=True);bank.write_json_exclusive(path,row)
            cls.partial_receipts[path]=path.read_bytes()

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def setUp(self):
        for name, data in self.documents.items():(self.extended/name).write_bytes(data)
        (self.extended/self.entry_name).write_bytes(self.entry_bytes)
        for path,data in self.partial_receipts.items():path.write_bytes(data)
        self.constant = patch.object(bank,'BANK_SHA256',self.original_sha)
        self.constant.start();self.addCleanup(self.constant.stop)

    def witness(self):
        return {'index':str(self.extended/'index.json'),
            'index_sha256':bank.file_sha(self.extended/'index.json'),
            'complete_sha256':bank.file_sha(self.extended/'complete.json'),
            'manifest_sha256':bank.file_sha(self.extended/'preservation_manifest.json'),
            'preparation_started_sha256':bank.file_sha(self.extended/'preparation_started.json')}

    def replace_document(self,name,value):
        (self.extended/name).write_text(json.dumps(value),encoding='utf8')

    def refresh_complete(self):
        value = bank.read_json(self.extended/'complete.json')
        value.update(index_sha256=bank.file_sha(self.extended/'index.json'),
            preservation_manifest_sha256=bank.file_sha(self.extended/'preservation_manifest.json'))
        self.replace_document('complete.json',value)

    def verify(self,witness=None):
        return bank.verify_prepared_bank(witness or self.witness(),self.binding_path,self.binding_sha)

    def test_full642_readonly_reuse_returns_fresh_derivative_not_training_claim(self):
        before = {str(p):bank.stat_identity(p) for p in self.original.rglob('*') if p.is_file()}
        result = self.verify()
        self.assertEqual(result['source_entries'],642)
        self.assertEqual(result['source_index_sha256'],self.original_sha)
        self.assertEqual(result['index'],str(self.extended/'index.json'))
        self.assertTrue(result['all642_entry_sha_verified'])
        self.assertTrue(result['full_geometry_preflight_still_required'])
        self.assertFalse(result['bank_regenerated']);self.assertFalse(result['training_executed'])
        self.assertEqual(before,{str(p):bank.stat_identity(p) for p in self.original.rglob('*') if p.is_file()})

    def test_partial_bank_and_geometry_claim_changes_fail_closed(self):
        for key,value in [('complete',False),('source_entries',641),
                          ('all_added128_historical_raw_preprocessed_graph_valid',False)]:
            complete=json.loads(self.documents['complete.json']);complete[key]=value
            self.replace_document('complete.json',complete)
            with self.assertRaisesRegex(bank.BankContractError,'Complete full642'):
                self.verify()

    def test_binding_and_complete_sha_must_match_actual_generation(self):
        stale=self.witness();stale['complete_sha256']='0'*64
        with self.assertRaisesRegex(bank.BankContractError,'SHA differs'):self.verify(stale)
        started=bank.read_json(self.extended/'preparation_started.json')
        started['binding']['DEBUG_fixture']=False
        self.replace_document('preparation_started.json',started)
        with self.assertRaisesRegex(bank.BankContractError,'binding/full-source'):self.verify()

    def test_source_population_and_other_interventions_cannot_be_relabelled(self):
        index=bank.read_json(self.extended/'index.json');index['cp_probability']=.6
        self.replace_document('index.json',index);self.refresh_complete()
        with self.assertRaisesRegex(bank.BankContractError,'more than declared'):self.verify()

    def test_completed_extended_search_requires_truthful_matching_policy_receipts(self):
        index=bank.read_json(self.extended/'index.json')
        index['ablation'].update(raw_pool_attempts=None,additional_candidate_search=bank.ADDITIONAL_CANDIDATE_SEARCH)
        self.replace_document('index.json',index);self.refresh_complete()
        with self.assertRaisesRegex(bank.BankContractError,'extended-search policy'):self.verify()
        started=bank.read_json(self.extended/'preparation_started.json')
        started['additional_candidate_search']=bank.ADDITIONAL_CANDIDATE_SEARCH
        self.replace_document('preparation_started.json',started)
        complete=bank.read_json(self.extended/'complete.json')
        complete['additional_candidate_search']=bank.ADDITIONAL_CANDIDATE_SEARCH
        self.replace_document('complete.json',complete)
        result=self.verify()
        self.assertEqual(result['additional_candidate_search'],bank.ADDITIONAL_CANDIDATE_SEARCH)

    def test_manifest_array_proof_and_original_metadata_must_be_actual(self):
        for field in ['proof','original_stat']:
            manifest=json.loads(self.documents['preservation_manifest.json'])
            if field=='proof':manifest['entries'][0]['proof']['original_array_sha256']['scores']='0'*64
            else:manifest['entries'][0]['original_stat'][2]+=1
            self.replace_document('preservation_manifest.json',manifest);self.refresh_complete()
            with self.assertRaises(bank.BankContractError):self.verify()

    def test_changed_source_patch_detected_even_when_outer_hashes_recomputed(self):
        path=self.extended/self.entry_name;entry=bank.read_entry(path)
        entry['source_data'].flat[0]+=1;np.savez(path,**entry)
        digest=bank.file_sha(path)
        manifest=bank.read_json(self.extended/'preservation_manifest.json')
        manifest['entries'][0]['new_entry_sha256']=digest
        self.replace_document('preservation_manifest.json',manifest)
        complete=bank.read_json(self.extended/'complete.json');complete['entries_sha256'][self.entry_name]=digest
        self.replace_document('complete.json',complete);self.refresh_complete()
        with self.assertRaisesRegex(bank.BankContractError,'Original field/prefix bytes changed'):self.verify()

    def fresh_partial_copy(self):
        path=self.root/('fresh_partial_'+str(bank.time.time_ns()))
        path.mkdir();(path/'entry_receipts').mkdir()
        return path

    def test_partial_reuse_copies_only_receipted_sources_without_old_writes(self):
        output=self.fresh_partial_copy()
        before={str(path):bank.stat_identity(path) for path in self.extended.rglob('*') if path.is_file()}
        rows=bank.copy_verified_partial_bank(self.binding,self.extended,output)
        self.assertEqual(len(rows),2)  # Other640 unreceipted NPZs are preserved, not claimed complete.
        self.assertEqual(len(list(output.rglob('*.npz'))),2)
        self.assertFalse((output/'complete.json').exists())
        self.assertFalse(bank.read_json(output/'partial_reuse_admission.json')['complete'])
        for row in rows:
            old,new=self.extended/row['entry'],output/row['entry']
            self.assertEqual(old.read_bytes(),new.read_bytes())
            self.assertNotEqual(bank.stat_identity(old)[:2],bank.stat_identity(new)[:2])
        self.assertEqual(before,{str(path):bank.stat_identity(path) for path in self.extended.rglob('*') if path.is_file()})

    def test_failed55_sources_remain_reusable_without_claiming_completed642_bank(self):
        manifest=bank.read_json(self.extended/'preservation_manifest.json')
        created=[]
        for index,row in enumerate(manifest['entries'][:55]):
            receipt=self.extended/'entry_receipts'/Path(row['entry']).with_suffix('.json')
            if not receipt.exists():
                receipt.parent.mkdir(parents=True,exist_ok=True)
                bank.write_json_exclusive(receipt,row);created.append(receipt)
            if index<28:
                previous=bank.read_json(receipt)
                previous['partial_reuse']={'previous_root':'/CPU_DEBUG/older28',
                    'previous_receipt_sha256':'0'*64,'previous_entry_sha256':row['new_entry_sha256'],
                    'new_copy_byte_verified':True}
                receipt.write_text(json.dumps(previous),encoding='utf8')
        try:
            output=self.fresh_partial_copy()
            before={str(path):bank.stat_identity(path) for path in self.extended.rglob('*') if path.is_file()}
            rows=bank.copy_verified_partial_bank(self.binding,self.extended,output)
            self.assertEqual(len(rows),55)
            self.assertEqual({row['entry'] for row in rows},{row['entry'] for row in manifest['entries'][:55]})
            for row in rows:
                old_receipt=self.extended/'entry_receipts'/Path(row['entry']).with_suffix('.json')
                self.assertEqual(row['partial_reuse']['previous_receipt_sha256'],bank.file_sha(old_receipt))
                self.assertEqual(bank.file_sha(output/row['entry']),row['new_entry_sha256'])
            admission=bank.read_json(output/'partial_reuse_admission.json')
            self.assertEqual(admission['verified_reused_sources'],55)
            self.assertEqual(admission['required_full_sources'],642)
            self.assertFalse(admission['complete'])
            self.assertFalse((output/'complete.json').exists())
            self.assertEqual(before,{str(path):bank.stat_identity(path) for path in self.extended.rglob('*') if path.is_file()})
        finally:
            for receipt in created:receipt.unlink()  # Own CPU DEBUG fixture files only.
            for path,data in self.partial_receipts.items():path.write_bytes(data)

    def test_partial_binding_or_completed_source_checksum_changes_are_rejected(self):
        started=bank.read_json(self.extended/'preparation_started.json');started['binding']['DEBUG_fixture']=False
        self.replace_document('preparation_started.json',started)
        with self.assertRaisesRegex(bank.BankContractError,'binding/geometry'):
            bank.copy_verified_partial_bank(self.binding,self.extended,self.fresh_partial_copy())
        (self.extended/'preparation_started.json').write_bytes(self.documents['preparation_started.json'])
        receipt=next(iter(self.partial_receipts));row=bank.read_json(receipt);row['new_entry_sha256']='0'*64
        receipt.write_text(json.dumps(row),encoding='utf8')
        with self.assertRaisesRegex(bank.BankContractError,'SHA/stat'):
            bank.copy_verified_partial_bank(self.binding,self.extended,self.fresh_partial_copy())

    def test_partial_array_tamper_is_rejected_even_with_updated_receipt_sha(self):
        path=self.extended/self.entry_name;entry=bank.read_entry(path);entry['scores'][0]+=.01
        np.savez(path,**entry)
        receipt=next(iter(self.partial_receipts));row=bank.read_json(receipt)
        row['new_entry_sha256']=bank.file_sha(path);receipt.write_text(json.dumps(row),encoding='utf8')
        with self.assertRaisesRegex(bank.BankContractError,'Original field/prefix bytes changed'):
            bank.copy_verified_partial_bank(self.binding,self.extended,self.fresh_partial_copy())

    def test_reused_patient_skips_all_ct_load_and_graph_work(self):
        output=self.fresh_partial_copy()
        rows=bank.copy_verified_partial_bank(self.binding,self.extended,output)
        row=rows[0];name=row['entry'];case=row['case_id']
        provider=types.SimpleNamespace(load_patient=lambda cid:self.fail('Reused patient unnecessarily decoded'))
        with patch.object(bank,'_WORKER_PROVIDER',provider),patch.object(bank,'_WORKER_BINDING',self.binding):
            actual=bank._build_patient(self.binding,case,[name],str(output),{name:row})
        self.assertEqual(actual,[row])

    def test_copied_receipt_change_prevents_worker_skipping_source(self):
        output=self.fresh_partial_copy()
        row=bank.copy_verified_partial_bank(self.binding,self.extended,output)[0]
        receipt=output/'entry_receipts'/Path(row['entry']).with_suffix('.json')
        receipt.write_text('{}',encoding='utf8')
        provider=types.SimpleNamespace(load_patient=lambda cid:self.fail('Tampered reuse decoded before rejection'))
        with patch.object(bank,'_WORKER_PROVIDER',provider),patch.object(bank,'_WORKER_BINDING',self.binding):
            with self.assertRaisesRegex(bank.BankContractError,'copied source/receipt changed'):
                bank._build_patient(self.binding,row['case_id'],[row['entry']],str(output),{row['entry']:row})


if __name__ == '__main__':
    unittest.main()
