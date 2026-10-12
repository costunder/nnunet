"""Explicit CPU UNIT metadata/tensor tests; no real CT/CUDA/model result claim."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

from hiercp_v1x import v25_cp as cp
from hiercp_v1x.v24_matched_basic_cp_scoring import matched_plan
from tests.test_v24_matched_basic_cp_scoring import PlanPopulation_DEBUG,payload_DEBUG


def declaration_UNIT():
    entries=['CPU_UNIT/source_'+str(i)+'.npz' for i in range(642)]
    batches=[entries[i:i+4] for i in range(0,640,4)]+[entries[640:]]
    return dict(CPU_UNIT_SYNTHETIC_METADATA=True,format=cp.STORE_FORMAT,
        cp_contract=cp.CP_CONTRACT,best_provenance={'arm':'A'},
        source_batches=batches,source_bank={'entries':{name:{} for name in entries}})


class DurableCompleteCPUnit(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory();self.root=Path(self.temporary.name)/'scores'
        self.addCleanup(self.temporary.cleanup);self.declaration=declaration_UNIT()

    def store(self):
        store=cp.ScoreStore(self.root,self.declaration);self.addCleanup(store.close);return store

    def test_restarts_reuse_only_whole_original_batches_without_overwrite(self):
        first=self.store();batch=self.declaration['source_batches'][0]
        for entry in batch:first.save_row(dict(CPU_UNIT_SYNTHETIC_METADATA=True,entry=entry,scores=list(range(128))))
        first.finish_batch(batch);before={path:cp._sha(path) for path in (self.root/'scores').iterdir()}
        torn=self.declaration['source_batches'][1]
        first.save_row(dict(CPU_UNIT_SYNTHETIC_METADATA=True,entry=torn[0],scores=list(range(128))))
        first.close();second=self.store()
        self.assertEqual(set(second.batch(batch)),set(batch));self.assertEqual(second.resumed_sources,4)
        self.assertIsNone(second.batch(torn));self.assertEqual(second.completed_sources,4)
        self.assertEqual(before,{path:cp._sha(path) for path in before})
        changed=dict(entry=batch[0],scores=[0]*128)
        with self.assertRaisesRegex(ValueError,'no overwrite'):second.save_row(changed)

    def test_partial_batches_cannot_commit_or_claim_full_complete(self):
        store=self.store();batch=self.declaration['source_batches'][0]
        store.save_row(dict(entry=batch[0],scores=list(range(128))))
        with self.assertRaises(ValueError):store.finish_batch(batch)
        with self.assertRaises(ValueError):store.finish_batch(batch[:2])
        with self.assertRaises(ValueError):store.finish(dict(complete_sources=128,complete_positions=16384))
        self.assertFalse((self.root/'score_overlay.json').exists())

    def test_changed_request_and_row_bytes_are_rejected(self):
        store=self.store();batch=self.declaration['source_batches'][0]
        for entry in batch:store.save_row(dict(entry=entry,scores=list(range(128))))
        store.finish_batch(batch);store.close()
        value=json.loads(store.row_path(batch[0]).read_text());value['row']['scores'][0]=1000
        store.row_path(batch[0]).write_text(json.dumps(value))
        again=self.store()
        with self.assertRaisesRegex(ValueError,'row changed'):again.batch(batch)
        again.close();different=copy.deepcopy(self.declaration);different['cp_contract']['epochs']=249
        with self.assertRaises(ValueError):cp.ScoreStore(self.root,different)

    def test_concurrent_writer_and_subset_are_refused(self):
        self.store()
        with self.assertRaises(RuntimeError):cp.ScoreStore(self.root,self.declaration)
        subset=declaration_UNIT();subset['source_batches']=subset['source_batches'][:-1]
        with self.assertRaisesRegex(ValueError,'All642'):cp.ScoreStore(self.root/'other',subset)

    def test_progress_is_observed_durable_count_not_total_hint(self):
        store=self.store();store.progress(status='RUNNING')
        progress=cp._read(self.root/'progress.json')
        self.assertEqual(progress['completed_sources'],0);self.assertEqual(progress['total_sources'],642)
        self.assertEqual(progress['actor']['pid'],__import__('os').getpid())
        self.assertNotIn('full_native_training_completed',progress)


class OwnBESTAndCompleteScoreCPUUnit(unittest.TestCase):
    def test_actual_checkpoint_bytes_and_full40_680_selection_are_required(self):
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'CPU_UNIT_checkpoint.bin';path.write_bytes(b'CPU_UNIT_METADATA_ONLY')
            value=dict(CPU_UNIT_SYNTHETIC_METADATA=True,completed_epochs=40,optimizer_updates=680,
                selected_by='fixed_full128_validation_only',model_sha256='a'*64,
                context_contract_sha256='b'*64,full_training=True,debug=False,
                checkpoint=str(path),checkpoint_sha256=cp._sha(path),arm='B',request_sha256='c'*64)
            self.assertEqual(cp.validate_best_provenance(value,model_sha256='a'*64,context_contract_sha256='b'*64),value)
            for field,bad in (('completed_epochs',39),('optimizer_updates',679),('arm','GPU5'),
                    ('selected_by','active_validation'),('debug',True),('checkpoint_sha256','0'*64)):
                with self.subTest(field=field),self.assertRaises(ValueError):
                    cp.validate_best_provenance(dict(value,**{field:bad}),model_sha256='a'*64,context_contract_sha256='b'*64)
            path.write_bytes(b'CPU_UNIT_CHANGED')
            with self.assertRaises(ValueError):cp.validate_best_provenance(value,model_sha256='a'*64,context_contract_sha256='b'*64)

    def test_actual_full_P_plus128_vector_only_and_original_argmax(self):
        payload=payload_DEBUG();plan=matched_plan('case','CPU_UNIT/source.npz',payload,PlanPopulation_DEBUG(5))
        scores=np.linspace(-3,2,len(plan.record_ids));row=cp.cp_score_row(plan,scores,payload,
            original_sha256='a'*64,model_sha256='b'*64,
            input_binding={'CPU_UNIT_SYNTHETIC_METADATA':True},workload={'CPU_UNIT_SYNTHETIC_METADATA':True})
        self.assertEqual(row['selected_index'],127);self.assertEqual(len(row['scores']),128)
        self.assertEqual(row['joint_observed_P'],5);self.assertFalse(row['recipient_GT_used_in_forward'])
        for bad in (scores[:128],np.full(len(scores),np.nan)):
            with self.assertRaises(ValueError):cp.cp_score_row(plan,bad,payload,original_sha256='a'*64,
                model_sha256='b'*64,input_binding={},workload={})

    def test_no_vessel_or_old_depth_policy_enters_v25_scorer(self):
        source=Path(cp.__file__).read_text()
        self.assertNotIn('v30_model',source);self.assertNotIn('VesselScorer',source)
        self.assertIn('resource_guard();return original_encode',source)
        self.assertEqual(cp.CP_CONTRACT['total_sources'],642)
        self.assertEqual(cp.CP_CONTRACT['epochs'],250)


if __name__=='__main__':unittest.main()
