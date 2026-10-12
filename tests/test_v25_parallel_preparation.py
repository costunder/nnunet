"""CPU UNIT integrity of four-way dispatch, not production native CT evidence."""
from contextlib import nullcontext
import copy
import gzip
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import uuid

import torch

from hiercp_v1x import v25_parallel_preparation as parallel
from hiercp_v1x.contracts import canonical_hash
from hiercp_v1x.v24_inputs import tensor_digest, query_inputs
from hiercp_v22.contracts import sha


ROOT = Path(__file__).resolve().parents[1]


class ParallelPreparationCPUUnit(unittest.TestCase):
    def setUp(self):
        self.root = ROOT/'tmp'/('UNIT_v25_parallel_'+uuid.uuid4().hex)
        self.root.mkdir(parents=True,exist_ok=False)
        self.addCleanup(self.cleanup_owned)
        self.cases = ['UNIT_case%d'%index for index in range(8)]
        self.rows = {case:[dict(id=case+':'+str(index),case_id=case,center=[index,1,1],
            donor_case_id='UNIT_donor',donor_component=1) for index in range(3)] for case in self.cases}

    def cleanup_owned(self):
        if self.root.parent != ROOT/'tmp' or not self.root.name.startswith('UNIT_v25_parallel_'):
            raise ValueError('CPU UNIT cleanup escaped owned namespace')
        for path in sorted(self.root.rglob('*'),key=lambda value:len(value.parts),reverse=True):
            if not path.is_relative_to(self.root) or path.is_symlink():
                raise ValueError('Unexpected CPU UNIT cleanup path')
            path.rmdir() if path.is_dir() else path.unlink()
        self.root.rmdir()

    def inputs(self):
        inputs = SimpleNamespace(root=self.root,runtime=dict(workers=4,minimum_free_disk_gib=0),
            population=SimpleNamespace(case=lambda case,count:SimpleNamespace(query_rows=self.rows[case])),
            inventory=dict(records=[row for rows in self.rows.values() for row in rows]),
            config_sha256='1'*64,source_sha256='2'*64,bank_sha256='3'*64,
            context_contract=dict(contract_sha256='4'*64),rss_bytes=parallel.WORKER_RSS_BYTES,
            raw={case:dict(spacing=[1.,1.,1.]) for case in self.cases},
            _input_bindings={},_file_proofs={},_rss=lambda:None)
        def binding(plan):
            case=plan.query_rows[0]['case_id']
            value={'case_id':case,'UNIT_NOT_CT':True}
            inputs._input_bindings[case]=value
            return value
        inputs.input_binding=binding
        inputs._local=self.record
        inputs._binding_header=lambda:dict(case_ids=self.cases,raw_files={})
        inputs._validate_binding=lambda case,value:self.assertEqual(value,{'case_id':case,'UNIT_NOT_CT':True})
        inputs._publish_input_bindings=lambda:'5'*64
        return inputs

    @staticmethod
    def record(row):
        # Real tensor serialization/hash tests, intentionally tiny CPU UNIT
        # tensors; these are never submitted as model training evidence.
        value=float(row['center'][0])
        record=dict(observation_id=row['id'],center=row['center'],
            source_patch=torch.ones((2,2,2),dtype=torch.float16),
            target_patch=torch.full((2,2,2),value,dtype=torch.float16),
            source_local=dict(nodes=torch.ones((3,2))),target_local=dict(nodes=torch.ones((4,2))))
        record['tensor_sha256']=tensor_digest({key:record[key] for key in
            ('source_patch','target_patch','source_local','target_local')})
        return record

    @staticmethod
    def views(record,*,epoch):
        return ((SimpleNamespace(node_stores=[SimpleNamespace(num_nodes=7)],
            edge_stores=[SimpleNamespace(num_edges=6)]),)*2,record['source_patch'],record['target_patch'])

    def run_cases(self,inputs,cases):
        def writer(root,minimum_free_bytes):
            class UnitWriter:
                def write(writer,relative,record,source_key):
                    reference=dict(path='shared_sources/UNIT.pt.gz',sha256='6'*64,content_sha256='7'*64)
                    path=root/relative;path.parent.mkdir(parents=True,exist_ok=True)
                    payload=dict(record,shared_source=reference)
                    stream=io.BytesIO();torch.save(payload,stream)
                    with path.open('xb') as output:output.write(gzip.compress(stream.getvalue(),mtime=0))
                    return dict(path=relative,sha256=sha(path),shared_source=reference,bounds=dict(nodes=7,edges=6,bytes=64))
            return UnitWriter()
        with patch.object(parallel,'publication_lock',return_value=nullcontext()), \
             patch.object(parallel,'materialize_pair',side_effect=self.views), \
             patch('hiercp_v22.storage.GraphWriter',side_effect=writer), \
             patch('hiercp_v22.storage.load_record',side_effect=lambda root,relative:self.read_raw(root/relative)):
            return parallel.prepare_cases(inputs,cases,lock_path=self.root/'UNIT.lock')

    @staticmethod
    def read_raw(path):
        with gzip.open(path,'rb') as stream:
            value=torch.load(io.BytesIO(stream.read()),map_location='cpu',weights_only=False)
        value.pop('shared_source')
        return value

    def receipt(self,inputs,cases,records,shard):
        return parallel.shard_receipt(inputs,cases,records,shard=shard,execution_sha256='8'*64)

    def test_patient_split_is_complete_disjoint_and_keeps_original_order(self):
        shards=parallel.split_cases(self.cases)
        self.assertEqual(shards,[self.cases[index::4] for index in range(4)])
        self.assertEqual({case for shard in shards for case in shard},set(self.cases))
        self.assertEqual(sum(map(len,shards)),len(self.cases))
        with self.assertRaisesRegex(ValueError,'unique'):parallel.split_cases(['duplicate','duplicate'])

    def test_existing_partial_records_are_read_only_and_missing_rows_are_built(self):
        inputs=self.inputs()
        first=self.run_cases(inputs,self.cases[:1])
        before={entry['path']:(self.root/'local'/entry['path']).read_bytes() for entry in first}
        with patch.object(inputs,'_local',side_effect=self.record) as builder:
            again=self.run_cases(inputs,self.cases[:2])
        self.assertEqual(builder.call_count,3)
        self.assertEqual({entry['id'] for entry in again},{row['id'] for case in self.cases[:2] for row in self.rows[case]})
        for relative,content in before.items():self.assertEqual((self.root/'local'/relative).read_bytes(),content)

    def test_tensor_mutation_fails_and_preserves_partial_file(self):
        inputs=self.inputs();entries=self.run_cases(inputs,self.cases[:1])
        path=self.root/'local'/entries[0]['path']
        with gzip.open(path,'rb') as stream:record=torch.load(io.BytesIO(stream.read()),weights_only=False)
        record['target_patch'].fill_(99)
        stream=io.BytesIO();torch.save(record,stream);path.write_bytes(gzip.compress(stream.getvalue(),mtime=0))
        changed=path.read_bytes()
        with self.assertRaisesRegex(ValueError,'tensor proof differs'):
            self.run_cases(inputs,self.cases[:1])
        self.assertEqual(path.read_bytes(),changed)

    def test_four_receipts_merge_exact_original_index_and_all_query_ids(self):
        inputs=self.inputs();shards=parallel.split_cases(self.cases)
        receipts=[self.receipt(inputs,cases,self.run_cases(inputs,cases),shard) for shard,cases in enumerate(shards)]
        with patch('hiercp_v22.storage.load_record',side_effect=lambda root,relative:self.read_raw(root/relative)):
            index=parallel.merge_shards(inputs,receipts,shards,execution_sha256='8'*64)
        value=json.loads(index.read_text())
        self.assertTrue(value['complete']);self.assertFalse(value['debug'])
        self.assertEqual(len(value['records']),24)
        self.assertEqual([entry['id'] for entry in value['records']],sorted(row['id'] for row in inputs.inventory['records']))
        self.assertEqual(set(inputs._input_bindings),set(self.cases))
        self.assertEqual(value['input_bindings_sha256'],'5'*64)

    def test_resealed_incomplete_shard_cannot_publish_a_complete_index(self):
        inputs=self.inputs();shards=parallel.split_cases(self.cases)
        receipts=[self.receipt(inputs,cases,self.run_cases(inputs,cases),shard) for shard,cases in enumerate(shards)]
        receipts[0]['records'].pop()
        receipts[0]['content_sha256']=canonical_hash({key:value for key,value in receipts[0].items() if key!='content_sha256'})
        with self.assertRaisesRegex(ValueError,'omitted or duplicated'):
            parallel.merge_shards(inputs,receipts,shards,execution_sha256='8'*64)
        self.assertFalse((self.root/'local/index.json').exists())

    def test_closed_receipt_rejects_different_execution_source(self):
        inputs=self.inputs();shards=parallel.split_cases(self.cases)
        receipts=[self.receipt(inputs,cases,self.run_cases(inputs,cases),shard) for shard,cases in enumerate(shards)]
        with self.assertRaisesRegex(ValueError,'identity/proof differs'):
            parallel.merge_shards(inputs,receipts,shards,execution_sha256='9'*64)
        self.assertFalse((self.root/'local/index.json').exists())

    def test_resealed_false_tensor_claim_is_checked_against_real_serialized_tensor(self):
        inputs=self.inputs();shards=parallel.split_cases(self.cases)
        receipts=[self.receipt(inputs,cases,self.run_cases(inputs,cases),shard) for shard,cases in enumerate(shards)]
        receipts[0]['records'][0]['tensor_sha256']='0'*64
        receipts[0]['content_sha256']=canonical_hash({key:value for key,value in receipts[0].items() if key!='content_sha256'})
        with patch('hiercp_v22.storage.load_record',side_effect=lambda root,relative:self.read_raw(root/relative)):
            with self.assertRaisesRegex(ValueError,'actual tensor/shared-source proof differs'):
                parallel.merge_shards(inputs,receipts,shards,execution_sha256='8'*64)
        self.assertFalse((self.root/'local/index.json').exists())

    def test_resealed_foreign_worker_pid_cannot_publish_production_union(self):
        inputs=self.inputs();shards=parallel.split_cases(self.cases)
        receipts=[self.receipt(inputs,cases,self.run_cases(inputs,cases),shard) for shard,cases in enumerate(shards)]
        _,manifest=self.manifest();slots=[manifest['slots'][arm] for arm in 'ABCD']
        pids=[100,101,102,103]
        for shard,receipt in enumerate(receipts):
            receipt['execution_identity']=dict(PID=pids[shard],create_time=1.,
                command=['CPU_UNIT_NOT_ACTUAL_SERVER_ACTOR'],UID=41833,
                CPU_affinity=slots[shard]['CPU_affinity'],
                resource_request_path=slots[shard]['resource_request_path'],
                resource_request_sha256=slots[shard]['resource_request_sha256'],execution_manifest_sha256='a'*64)
            receipt['content_sha256']=canonical_hash({key:value for key,value in receipt.items() if key!='content_sha256'})
        receipts[0]['execution_identity']['PID']=999
        receipts[0]['content_sha256']=canonical_hash({key:value for key,value in receipts[0].items() if key!='content_sha256'})
        with self.assertRaisesRegex(ValueError,'Actual shard PID/CPU/resource/manifest'):
            parallel.merge_shards(inputs,receipts,shards,execution_sha256='8'*64,
                execution_slots=slots,worker_PIDs=pids,manifest_sha256='a'*64)
        self.assertFalse((self.root/'local/index.json').exists())

    def test_cooperative_admission_stop_preserves_every_existing_record(self):
        inputs=self.inputs();self.run_cases(inputs,self.cases[:1])
        files={str(path):path.read_bytes() for path in (self.root/'local').rglob('*.gz')}
        stop=SimpleNamespace(is_set=lambda:True)
        with self.assertRaisesRegex(RuntimeError,'partial canonical files preserved'):
            parallel.prepare_cases(inputs,self.cases[1:2],lock_path=self.root/'UNIT.lock',cancellation=stop)
        self.assertEqual(files,{str(path):path.read_bytes() for path in (self.root/'local').rglob('*.gz')})

    def test_memory_guard_accounts_all_workers_and_shared_cache_without_pbs_claim(self):
        value=parallel.memory_admission([10*2**30]*4,8*2**30,coordinator_rss=2*2**30)
        self.assertEqual(value['total_accounted_bytes'],50*2**30)
        self.assertFalse(value['scheduler_reservation_proven'])
        with self.assertRaisesRegex(MemoryError,'64GiB'):
            parallel.memory_admission([65*2**30,0,0,0],0)
        with self.assertRaisesRegex(MemoryError,'32GiB'):
            parallel.memory_admission([0]*4,33*2**30)
        with self.assertRaisesRegex(MemoryError,'288GiB'):
            parallel.memory_admission([64*2**30]*4,32*2**30,coordinator_rss=1)
        with self.assertRaisesRegex(ValueError,'Four actual'):
            parallel.memory_admission([0]*3,0)

    def test_worker_header_keeps_scientific_identity_outside_execution_details(self):
        inputs=self.inputs();header=parallel.index_header(inputs)
        self.assertNotIn('shard',header);self.assertNotIn('CPU_affinity',header)
        self.assertNotIn('execution_source_sha256',header)
        self.assertEqual(header['source_sha256'],inputs.source_sha256)
        self.assertEqual(header['context_contract_sha256'],inputs.context_contract['contract_sha256'])

    def manifest(self):
        slots={}
        for index,(arm,gpu) in enumerate(zip('ABCD',(6,5,1,4))):
            proof=self.root/('UNIT_direct_'+arm+'.json')
            proof.write_text(json.dumps(dict(CPU_UNIT=True,not_actual_resource_evidence=True)))
            slots[arm]=dict(resource_request_path=str(proof),resource_request_sha256=sha(proof),
                CPU_affinity=list(range(index*6,index*6+6)),gpu=gpu,arm=arm)
        value=dict(native_experiment=str(self.root/'UNIT_native'),inventory_path=str(self.root/'UNIT_inventory'),
            group_root=str(self.root),execution_root=str(self.root/'UNIT_execution'),slots=slots)
        path=self.root/'UNIT_manifest.json';path.write_text(json.dumps(value))
        return path,value

    def test_manifest_dispatches_four_own_cpu_sets_and_all_proofs_outside_scientific_binding(self):
        path,value=self.manifest();inputs=self.inputs()
        with patch.object(parallel,'prepare_four_way',return_value='CPU_UNIT_RESULT') as prepare:
            self.assertEqual(parallel.prepare_local_parallel(inputs,path),'CPU_UNIT_RESULT')
        args=prepare.call_args.kwargs
        self.assertEqual([slot['gpu'] for slot in args['execution_slots']],[6,5,1,4])
        self.assertEqual(args['resource_admission']['CPU_affinity'],list(range(24)))
        self.assertFalse(args['resource_admission']['scheduler_reservation_inferred'])
        self.assertEqual(args['resource_admission']['resource_manifest_sha256'],sha(path))

    def test_overlapping_cpu_slots_fail_before_any_worker_launch(self):
        path,value=self.manifest();value['slots']['B']['CPU_affinity']=value['slots']['A']['CPU_affinity']
        path.write_text(json.dumps(value))
        with patch.object(parallel,'prepare_four_way') as prepare:
            with self.assertRaisesRegex(ValueError,'disjoint admitted CPU'):
                parallel.prepare_local_parallel(self.inputs(),path)
        prepare.assert_not_called()


if __name__=='__main__':unittest.main()
