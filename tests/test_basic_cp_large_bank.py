"""CPU DEBUG fixtures for finite full-scope search and immutable nested banks.

No clinical CT preparation, GPU scoring, training or final performance claim.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

from hiercp_v1x import basic_cp_large_bank as large
from hiercp_v1x import basic_cp_ablation_bank as historical

ROOT = Path(__file__).resolve().parents[1]


def sample_candidate_centers(*args, **kwargs):
    raise AssertionError("DEBUG original sampler must be replaced in private namespace")


def frozen_pool_fixture(mask, minimum, *, requested=3):
    """DEBUG frozen predicate: only centers passing the original threshold."""
    return [center for center in sample_candidate_centers(mask,(1,1,1),
        rng=np.random.default_rng(4),requested=requested,max_draws=100)
        if sum(center) >= minimum]


def source_fixture(count):
    return dict(source_data=np.arange(8,dtype=np.float32).reshape(1,2,2,2),
        source_mask=np.ones((2,2,2),dtype=np.uint8),anchor_offset=np.ones(3,dtype=np.int16),
        candidate_centers=np.asarray([(i+2,2,2) for i in range(count)],dtype=np.int32),
        candidate_raw_centers=np.asarray([(i+2,2,2) for i in range(count)],dtype=np.int32),
        scores=np.arange(128,dtype=np.float32)/128,
        source_component=np.array([1],dtype=np.int16),source_diameter_mm=np.array([8],dtype=np.float32))


class FiniteSearchTests(unittest.TestCase):
    def test_each_liver_voxel_once_across_pool_boundaries(self):
        mask = np.zeros((4,5,6),dtype=bool)
        mask[1:3,1:4,2:5] = True
        flat = np.flatnonzero(mask)
        sampler = large.FiniteLiverCenters(mask,flat,42)
        observed = []
        while not sampler.exhausted:
            observed.extend(sampler.sampler(mask,(1,1,1),rng=np.random.default_rng(1),requested=5,max_draws=10))
        self.assertEqual(len(observed),len(flat))
        self.assertEqual(len(set(observed)),len(flat))
        self.assertEqual(set(observed),set(map(tuple,np.column_stack(np.nonzero(mask)))))
        self.assertTrue(sampler.diagnostics['full_liver_domain_exhausted'])

    def test_seeded_order_repeatability_and_original_catalogue_immutability(self):
        mask = np.ones((3,4,5),dtype=bool)
        flat = np.flatnonzero(mask)
        before = flat.tobytes()
        first,second,other = [large.FiniteLiverCenters(mask,flat,seed) for seed in (42,42,43)]
        np.testing.assert_array_equal(first.order,second.order)
        self.assertFalse(np.array_equal(first.order,other.order))
        self.assertEqual(flat.tobytes(),before)

    def test_supplier_clone_retains_raw_predicate_bytecode_and_globals(self):
        mask = np.ones((3,3,3),dtype=bool)
        sampler = large.FiniteLiverCenters(mask,np.flatnonzero(mask),2)
        original = frozen_pool_fixture.__globals__['sample_candidate_centers']
        adapted = large.finite_pool(frozen_pool_fixture,sampler.sampler)
        self.assertIs(adapted.__code__,frozen_pool_fixture.__code__)
        self.assertTrue(all(sum(center) >= 4 for center in adapted(mask,4,requested=27)))
        self.assertIs(frozen_pool_fixture.__globals__['sample_candidate_centers'],original)
        self.assertTrue(sampler.exhausted)

    def test_interrupting_a_raw_pool_does_not_drop_unconsumed_voxels(self):
        mask = np.ones((3,3,3),dtype=bool)
        sampler = large.FiniteLiverCenters(mask,np.flatnonzero(mask),2)
        stream = sampler.sampler(mask,(1,1,1),rng=np.random.default_rng(1),requested=27,max_draws=100)
        prefix = [next(stream),next(stream)]
        stream.close()
        remainder = list(sampler.sampler(mask,(1,1,1),rng=np.random.default_rng(1),requested=27,max_draws=100))
        self.assertEqual(len(set(prefix+remainder)),27)
        self.assertEqual(len(prefix+remainder),27)


class PrefixStreamTests(unittest.TestCase):
    def test_one_stream_three_nested_byte_exact_prefixes_scores_stay128(self):
        original,prefix = source_fixture(128),source_fixture(256)
        proposals = ((np.array((i+2,2,2),dtype=np.int32),np.array((i+2,2,2),dtype=np.int32))
                     for i in range(256,2048))
        published = {}
        large.collect_prefixes(original,prefix,proposals,targets=large.COUNTS,
                               publish=lambda count,result,proof:published.update({count:(result,proof)}))
        self.assertEqual(set(published),set(large.COUNTS))
        for count,(result,proof) in published.items():
            self.assertEqual(result['scores'].shape,(128,))
            self.assertEqual(result['candidate_centers'].shape,(count,3))
            self.assertTrue(proof['all_admitted_prefix_centers_byte_preserved'])
            np.testing.assert_array_equal(result['candidate_centers'][:256],prefix['candidate_centers'])
            for key in original:
                new = result[key][:128] if key.startswith('candidate_') else result[key]
                self.assertEqual(new.tobytes(),original[key].tobytes())
        for small,big in ((512,1024),(1024,2048)):
            np.testing.assert_array_equal(published[small][0]['candidate_centers'],
                                          published[big][0]['candidate_centers'][:small])

    def test_finite_exhaustion_does_not_publish_smaller_requested_bank(self):
        original = source_fixture(128)
        published = []
        proposals = ((np.array((i+2,2,2),dtype=np.int32),np.array((i+2,2,2),dtype=np.int32))
                     for i in range(128,513))
        with self.assertRaisesRegex(large.BankContractError,'measured513.*required1024'):
            large.collect_prefixes(original,original,proposals,targets=large.COUNTS,
                                   publish=lambda count,*args:published.append(count))
        self.assertEqual(published,[512])

    def test_duplicate_or_changed256_prefix_fail_visibly(self):
        old,prefix = source_fixture(128),source_fixture(256)
        prefix['candidate_centers'][0,0] += 1
        with self.assertRaises(large.BankContractError):
            large.collect_prefixes(old,prefix,iter(()),targets=large.COUNTS,publish=lambda *args:None)
        proposals = iter([(old['candidate_raw_centers'][0],np.array((1000,2,2)))])
        with self.assertRaisesRegex(large.BankContractError,'duplicate'):
            large.collect_prefixes(old,old,proposals,targets=large.COUNTS,publish=lambda *args:None)


class FiniteProviderTests(unittest.TestCase):
    def test_measured_unavailable_full_domain_has_no_endless_retry_or_fallback(self):
        provider = object.__new__(large.LargeGeometryProvider)
        provider.global_seed = 42
        provider.bank = {'liver_label':1,'tumor_label':2}
        provider.generation = historical.GENERATION
        provider.case = types.SimpleNamespace(spacing=np.ones(3))
        provider.components = object()
        provider.benchmark = types.SimpleNamespace(stable_seed=lambda *args:42,
            _source_from_component=lambda *args:object(),
            _map_raw_point=lambda raw,*args:np.asarray(raw,dtype=np.int32))
        provider.regions = types.SimpleNamespace(full_organ_mask=np.ones((2,2,2),dtype=bool),
                                                 organ_depth=np.ones((2,2,2)))
        provider.graph = provider.prototype = object()
        provider.ct_clip = (-200,250)
        provider.prepare = lambda *args,**kwargs:object()
        provider.placement_mask = np.ones((2,2,2),dtype=bool)
        provider.flat_liver = np.flatnonzero(provider.placement_mask)
        provider.occupied_mask = np.zeros((2,2,2),dtype=bool)
        provider.patient_pool = frozen_pool_fixture
        provider.specs = lambda pool,*args,**kwargs:[None]*len(pool)
        provider.preseg = np.ones((1,2,2,2),dtype=np.int16)
        provider.plans = {'transpose_forward':[0,1,2]}
        provider.properties = {}
        provider.local = lambda *args,**kwargs:None
        provider.geometry_error = RuntimeError
        old = source_fixture(128)
        old['source_mask'] = np.ones((1,1,1),dtype=np.uint8)
        old['anchor_offset'] = np.zeros(3,dtype=np.int16)
        # Actual raw pool semantics stay in its private cloned namespace.
        def pool(case,source,**kwargs):
            return frozen_pool_fixture(kwargs['placement_mask'],0,requested=256),None
        # Supply a frozen no-closure fixture with expected pool signature.
        provider.patient_pool = finite_provider_pool_fixture
        with self.assertRaisesRegex(large.BankContractError,'Finite candidate domain exhausted') as caught:
            list(provider.proposals('DEBUG',old))
        record = json.loads(str(caught.exception).split('exhausted ',1)[1])
        self.assertEqual(record['available_unique_raw_and_preprocessed_graph_valid'],136)
        self.assertEqual(record['required_candidate_count'],2048)
        self.assertTrue(record['all_liver_center_voxels_checked'])
        self.assertEqual(provider.last_proposal_diagnostics['finite_search']['finite_raw_centers_examined'],8)


def finite_provider_pool_fixture(case,source,*,placement_mask,rng,num_candidates,max_draws,**kwargs):
    pool = []
    for center in sample_candidate_centers(placement_mask,(1,1,1),rng=rng,
                                           requested=num_candidates*20,max_draws=max_draws):
        pool.append(types.SimpleNamespace(center=center))
        if len(pool) == num_candidates:
            break
    return pool,None


class FullCatalogueAdmissionTests(unittest.TestCase):
    """All642 DEBUG source identities test proof admission, not clinical efficacy."""
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix='large_bank_DEBUG_',dir=ROOT/'tmp')
        cls.root = Path(cls.temporary.name).resolve()
        cls.original_root = cls.root/'original'
        cls.shared = cls.root/'shared'
        cls.bank_root = cls.shared/'512'
        cls.original_root.mkdir();cls.bank_root.mkdir(parents=True)
        groups = {f'liver_{i}':[] for i in range(81)}
        for i in range(642):
            groups[f'liver_{i%81}'].append(f'entries/liver_{i%81}__component_{i:03d}.npz')
        cls.original_index = cls.original_root/'index.json'
        cls.index = dict(format='hiercp_online_bank_v2',candidate_count=128,source_entries=642,
            eligible_cases=81,total_candidates=82176,entries_by_case=groups,minimum_liver_coverage=.85,
            liver_label=1,tumor_label=2,draw_count=256,attempts=4,cp_probability=.5)
        historical.write_json_exclusive(cls.original_index,cls.index)
        cls.original_sha = historical.file_sha(cls.original_index)
        cls.binding = {'bank_index':str(cls.original_index),'DEBUG_fixture':True}
        cls.binding_path = cls.root/'binding.json'
        historical.write_json_exclusive(cls.binding_path,cls.binding)
        cls.binding_sha = historical.file_sha(cls.binding_path)
        started = dict(format=large.FORMAT,binding=cls.binding,candidate_count=512,
            candidate_counts=list(large.COUNTS),full_sources=642,additional_candidate_search=large.SEARCH,
            CUDA_model_or_scoring_executed=False,prepared256_reuse_admission=None)
        historical.write_json_exclusive(cls.bank_root/'preparation_started.json',started)
        old,new = source_fixture(128),source_fixture(512)
        proof = historical.verify_preservation(old,new,original_count=128,candidate_count=512)
        proof.update(admitted_prefix_count=128,all_admitted_prefix_centers_byte_preserved=True)
        rows = []
        for case,names in groups.items():
            for name in names:
                original_path,new_path = cls.original_root/name,cls.bank_root/name
                original_path.parent.mkdir(exist_ok=True);new_path.parent.mkdir(exist_ok=True)
                for path,entry in ((original_path,old),(new_path,new)):
                    with path.open('xb') as stream:np.savez(stream,**entry)
                old_sha = historical.file_sha(original_path)
                old_stat = historical.stat_identity(original_path)
                rows.append(dict(entry=name,case_id=case,original_stat=old_stat,
                    original_entry_sha256=old_sha,new_entry_sha256=historical.file_sha(new_path),proof=proof,
                    admitted_prefix=dict(path=str(original_path),sha256=old_sha,stat=old_stat)))
        cls.manifest_path = cls.bank_root/'preservation_manifest.json'
        historical.write_json_exclusive(cls.manifest_path,dict(format=large.FORMAT,
            entries=sorted(rows,key=lambda row:row['entry']),source_entries=642,eligible_cases=81))
        extended_index = copy.deepcopy(cls.index)
        with patch.object(historical,'BANK_SHA256',cls.original_sha):
            extended_index.update(candidate_count=512,total_candidates=642*512,
                ablation=large._ablation(cls.original_index,512,cls.shared),manifest=str(cls.manifest_path))
        cls.index_path = cls.bank_root/'index.json'
        historical.write_json_exclusive(cls.index_path,extended_index)
        cls.complete_path = cls.bank_root/'complete.json'
        cls.complete = dict(format=large.FORMAT,complete=True,candidate_count=512,source_entries=642,
            eligible_cases=81,total_candidates=642*512,index_sha256=historical.file_sha(cls.index_path),
            original_index_sha256=cls.original_sha,preservation_manifest_sha256=historical.file_sha(cls.manifest_path),
            entries_sha256={row['entry']:row['new_entry_sha256'] for row in rows},
            all_original_arrays_and_first128_preserved=True,all_centers_unique_raw_and_preprocessed=True,
            all_added_historical_raw_preprocessed_graph_valid=True,additional_GNN_scores_generated=False,
            training_executed=False,CUDA_initialized=False,raw_CT_preprocessing_executed=False,
            patient_region_cache_rebuilt=False,shared_preprocessed_arrays_written=False,
            additional_candidate_search=large.SEARCH,shared_generation_root=str(cls.shared),
            prepared256_reuse_admission=None)
        historical.write_json_exclusive(cls.complete_path,cls.complete)
        cls.witness = dict(index=str(cls.index_path),index_sha256=historical.file_sha(cls.index_path),
            complete_sha256=historical.file_sha(cls.complete_path),manifest_sha256=historical.file_sha(cls.manifest_path),
            preparation_started_sha256=historical.file_sha(cls.bank_root/'preparation_started.json'))

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def admit(self,witness=None,count=None):
        with patch.object(historical,'BANK_SHA256',self.original_sha):
            return large.verify_prepared_bank(self.witness if witness is None else witness,
                self.binding_path,self.binding_sha,candidate_count=count)

    def test_actual_full642_admission_requires_all_original_and_new_arrays(self):
        receipt = self.admit(count=512)
        self.assertEqual(receipt['source_entries'],642)
        self.assertEqual(receipt['total_candidates'],642*512)
        self.assertTrue(receipt['all642_entry_sha_verified'])
        self.assertTrue(receipt['full_geometry_preflight_still_required'])
        self.assertFalse(receipt['training_executed'])

    def test_count_and_proof_hash_mismatch_fail_closed(self):
        with self.assertRaisesRegex(large.BankContractError,'candidate count differs'):
            self.admit(count=1024)
        witness = dict(self.witness,complete_sha256='0'*64)
        with self.assertRaisesRegex(large.BankContractError,'evidenceSHA'):
            self.admit(witness)

    def test_generation_does_not_admit_toy_or_unknown_source_bank(self):
        with self.assertRaisesRegex(large.BankContractError,'Original128 indexSHA'):
            large.build_shared_banks(self.binding,self.root/'forbidden',workers=4,rss_limit_gib=64)
        self.assertFalse((self.root/'forbidden').exists())


if __name__ == '__main__':
    unittest.main()
