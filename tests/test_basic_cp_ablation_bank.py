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
            def __exit__(self,*args): return False
            def submit(self,*args):
                future = Future()
                future.set_exception(bank.BankContractError('DEBUG exact geometry shortfall'))
                return future

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


if __name__ == '__main__':
    unittest.main()
