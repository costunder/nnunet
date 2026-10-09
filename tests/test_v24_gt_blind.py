"""Actual archived operators on explicit CPU UNIT fixtures, not experiment results."""
from __future__ import annotations
import copy
import hashlib
import json
import multiprocessing
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
import unittest
import uuid
from zipfile import ZipFile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def _concurrent_upper_publisher(output, graph, prototype, audit, rows, binding, barrier, results):
    """Independent CPU processes share only the cache publication directory."""
    from hiercp_v1x.v24_geometry import V24UpperGeometryCache
    from hiercp_v1x.v24_inputs import tensor_digest
    torch.set_num_threads(1)
    plan=SimpleNamespace(query_rows=rows,record_ids=tuple(row['id'] for row in rows))
    def build(plan,provider):
        barrier.wait(timeout=30)
        return graph.clone(),prototype.clone(),copy.deepcopy(audit)
    cache=V24UpperGeometryCache(None,output,build,input_binding=lambda plan:binding,
        guard_inputs=lambda plan,proof:None,workers=2,resident_bytes=2**30,rss_bytes=2*2**30)
    try:
        first=cache.get(plan)
        second=cache.get(plan)
        results.put(dict(digest=tensor_digest((first[0].to_dict(),first[1].to_dict())),
            private=first[0]['candidate'].raw_x.data_ptr()!=second[0]['candidate'].raw_x.data_ptr()))
    except Exception as error:
        results.put(dict(error=repr(error)))
        raise


class RecipientGTBlindDebug(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(min(4,cls.threads))
        archive = ROOT/'versions/v1/pipeline_v1_source.zip'
        from hiercp_v1x.historical_checkpoint import _activate
        from hiercp_v1x import bounded_scope
        import sys
        # The working original package is pinned byte-for-byte to the archive.
        # This CPU-only process installs wrappers in memory, never edits source.
        import hiercp
        if 'hiercp' in sys.modules:
            cls.snapshot = Path(sys.modules['hiercp'].__file__).resolve().parents[1]
            active = bounded_scope._ACTIVE
            if active is None:
                bounded_scope._verify_snapshot(cls.snapshot)
                active = bounded_scope.install(10,expected_snapshot_root=cls.snapshot)
            else:
                # Prior actual PyG forwards create generated propagation
                # modules outside the archive directory. Reuse the installed
                # scope only after rechecking every original archived byte;
                # do not reset modules or alter the production verifier.
                from hiercp_v1x.contracts import V1_ARCHIVE_SHA256
                if hashlib.sha256(archive.read_bytes()).hexdigest()!=V1_ARCHIVE_SHA256:
                    raise ValueError('Original archive changed during combined CPU UNIT')
                with ZipFile(archive) as zipped:
                    files={name:hashlib.sha256(zipped.read(name)).hexdigest()
                        for name in zipped.namelist() if name.startswith('hiercp/') and name.endswith('.py')}
                if files!=active['original_module_sha256']:
                    raise ValueError('Installed CPU UNIT original source binding differs')
                for name,checksum in files.items():
                    path=cls.snapshot/name
                    if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest()!=checksum:
                        raise ValueError('Original source changed during combined CPU UNIT: '+name)
            _, cls.scope = _activate(cls.snapshot,active)
        else:
            cls.snapshot = ROOT/'outputs'/('v24_GT_blind_original_CPU_UNIT_'+uuid.uuid4().hex)
            cls.snapshot.mkdir(parents=True)
            with ZipFile(archive) as zipped:
                for name in zipped.namelist():
                    if not ((name.startswith('hiercp/') and name.endswith('.py')) or name=='config/train.json'): continue
                    relative = PurePosixPath(name)
                    if relative.is_absolute() or '..' in relative.parts or ':' in name or '\\' in name:
                        raise ValueError('Unsafe pinned original archive path')
                    target = cls.snapshot.joinpath(*relative.parts)
                    target.parent.mkdir(parents=True,exist_ok=True)
                    target.write_bytes(zipped.read(name))
            _, cls.scope = _activate(cls.snapshot,None)
        from hiercp_v1x.v24_inputs import recipient_context, prepare_donor, build_local_record
        from hiercp_v1x.v24_geometry import build_recipient_regions, build_upper_graphs
        from hiercp_v1x.v24_model import build_gt_free_model
        from hiercp.schema import graph_config_from_dict
        from hiercp_v1x.bounded_scope import configure
        from hiercp_v22.data import sources
        from hiercp.prototype import build_prototype_bank
        base=json.loads((cls.snapshot/'config/train.json').read_text())
        cls.config=graph_config_from_dict(configure(base['graph'],10.).to_dict())
        cls.model_config=copy.deepcopy(base['model'])
        cls.model_config.update(local_layers=3,patient_layers=2,prototype_layers=2,dense_batch_size=4)
        cls.clip=tuple(base['ct_clip'])
        rng=np.random.default_rng(42); shape=(32,32,32)
        cls.CT=rng.normal(65,18,shape).astype(np.float32)
        cls.organ=np.zeros(shape,bool);cls.organ[3:29,3:29,3:29]=True
        cls.spacing=np.array([1.5,1.5,1.5],np.float32);affine=np.diag([1.5,1.5,1.5,1.])
        lab=cls.organ.astype(np.uint8);lab[11:15,12:16,11:15]=2
        cls.donor=SimpleNamespace(image=cls.CT.copy(),label=lab,spacing=cls.spacing.copy(),
            shape=shape,paths=SimpleNamespace(case_id='UNIT_train_donor'),image_affine=affine.copy(),label_affine=affine.copy())
        cls.source=sources(cls.donor,2,20)[0][0]
        cls.recipient=recipient_context('UNIT_val_recipient',cls.CT,cls.organ,cls.spacing,affine)
        cls.prepared=prepare_donor(cls.donor,cls.source,config=cls.config,seed=42,ct_clip=cls.clip)
        cls.donor_context=recipient_context('UNIT_train_donor',cls.CT,cls.organ,cls.spacing,affine)
        cls.donor_regions=build_recipient_regions(cls.donor_context,config=cls.config,seed=42,ct_clip=cls.clip)
        cls.regions=build_recipient_regions(cls.recipient,config=cls.config,seed=42,ct_clip=cls.clip)
        cls.bank=build_prototype_bank([(cls.donor.paths.case_id,cls.donor_regions.region_features)],
            config=cls.config,rng=np.random.default_rng(42))
        cls.rows=[dict(id='UNIT_val:'+str(i),case_id=cls.recipient.case_id,center=list(center),
            donor_case_id=cls.donor.paths.case_id,donor_component=cls.source.component_id,
            target=int(i==0),component=i+1 if i==0 else None)
            for i,center in enumerate([(13,14,13),(18,17,16),(10,20,18)])]
        cls.graph,cls.prototype,cls.audit=build_upper_graphs(cls.recipient,cls.donor,cls.source,cls.regions,
            cls.donor_regions,cls.rows,cls.bank,config=cls.config,ct_clip=cls.clip,
            training_case_ids=(cls.donor.paths.case_id,))
        cls.records=[build_local_record(cls.recipient,cls.donor,cls.source,cls.prepared,row,
            config=cls.config,seed=42,ct_clip=cls.clip,scope_contract=cls.scope['contract_sha256']) for row in cls.rows]
        torch.manual_seed(42)
        cls.net=build_gt_free_model(cls.model_config,original_snapshot_root=cls.snapshot)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_recipient_object_excludes_GT_and_freezes_explicit_anatomy(self):
        with self.assertRaises(AttributeError): _=self.recipient.label
        self.assertFalse(self.recipient.image.flags.writeable)
        self.assertFalse(self.recipient.organ_mask.flags.writeable)
        self.assertNotIn('label_sha256',self.recipient.binding())

    def test_all_upper_inputs_invariant_to_recipient_tumor_annotations_and_PU_labels(self):
        from hiercp_v1x.v24_inputs import recipient_context,tensor_digest
        from hiercp_v1x.v24_geometry import build_recipient_regions,build_upper_graphs
        reference=tensor_digest((self.graph.to_dict(),self.prototype.to_dict()))
        for variant in ('erase','permute','all_organ_tumor'):
            # Supervision variants exist outside the model-input contract.
            label=self.organ.astype(np.uint8)
            if variant=='permute':label[18:22,5:9,16:20]=2
            if variant=='all_organ_tumor':label[self.organ]=2
            context=recipient_context(self.recipient.case_id,self.CT,self.organ,self.spacing,self.recipient.image_affine)
            rows=copy.deepcopy(self.rows)
            for row in rows:row['target']=1-row['target'];row['component']=999
            regions=build_recipient_regions(context,config=self.config,seed=42,ct_clip=self.clip)
            graph,prototype,audit=build_upper_graphs(context,self.donor,self.source,regions,self.donor_regions,rows,
                self.bank,config=self.config,ct_clip=self.clip,training_case_ids=(self.donor.paths.case_id,))
            self.assertEqual(reference,tensor_digest((graph.to_dict(),prototype.to_dict())))
            self.assertFalse(audit['recipient_GT_used_in_forward'])

    def test_original_L0_patches_topology_views_invariant_to_recipient_tumor_GT(self):
        from hiercp_v1x.v24_inputs import recipient_context,build_local_record,materialize_pair,tensor_digest
        reference=self.records[0]
        context=recipient_context(self.recipient.case_id,self.CT,self.organ,self.spacing,self.recipient.image_affine)
        row=dict(self.rows[0],target=0,component=None)
        candidate=build_local_record(context,self.donor,self.source,self.prepared,row,config=self.config,seed=42,
            ct_clip=self.clip,scope_contract=self.scope['contract_sha256'])
        self.assertEqual(reference['tensor_sha256'],candidate['tensor_sha256'])
        a,b=materialize_pair(reference,epoch=29),materialize_pair(candidate,epoch=29)
        self.assertEqual(tensor_digest((tuple(v.to_dict() for v in a[0]),a[1],a[2])),
                         tensor_digest((tuple(v.to_dict() for v in b[0]),b[1],b[2])))

    def test_leak_modules_removed_and_all_retained_depth_width_heads_preserved(self):
        self.assertFalse(any('lesion' in name for name,_ in self.net.named_parameters()))
        self.assertEqual(self.net.patient_encoder.liver_project[0].in_features,13)
        self.assertEqual(len(self.net.local_encoder.blocks),3)
        self.assertEqual(len(self.net.patient_encoder.blocks),2)
        self.assertEqual(len(self.net.prototype_encoder.blocks),2)
        self.assertEqual(self.net.hidden_dim,128)
        self.assertTrue(all(conv.heads==4 for encoder in (self.net.local_encoder,self.net.patient_encoder,self.net.prototype_encoder)
                            for block in encoder.blocks for conv in block.conv.convs.values()))

    def test_actual_full_original_neural_path_scores_invariant_and_all_parameters_connected(self):
        from hiercp_v1x.v24_inputs import materialize_pair,collate
        from torch_geometric.data import Batch
        batch=collate([(materialize_pair(record,epoch=29),i) for i,record in enumerate(self.records)])
        self.net.eval();self.net.zero_grad(set_to_none=True)
        local=self.net.local_encoder
        source,target=local.encode_dense_maps(batch.source_patches,batch.source_index,batch.target_patches)
        fields=local.forward_graph(batch.graph,source[batch.graph_observation_index],target[batch.graph_observation_index])
        views=tuple({key:value[view::2] for key,value in fields.items()} for view in (0,1))
        merged={key:(views[0][key]+views[1][key])/2 for key in fields}
        upper=SimpleNamespace(patient_batch=Batch.from_data_list([self.graph.clone()]),
            prototype_batch=Batch.from_data_list([self.prototype.clone()]),counts=(len(self.rows),))
        score=self.net._score_upper(upper,merged)[0]
        # Even a legacy liver count in the removed slot cannot influence scores.
        changed=self.graph.clone();changed['liver'].raw_x[:,10]=135.
        changed['candidate'].raw_x[:,4]=torch.tensor([0.,1.,2.])
        variant=SimpleNamespace(patient_batch=Batch.from_data_list([changed]),
            prototype_batch=Batch.from_data_list([self.prototype.clone()]),counts=(len(self.rows),))
        second=self.net._score_upper(variant,merged)[0]
        torch.testing.assert_close(score,second,rtol=0,atol=0)
        loss=torch.nn.functional.softplus(score[1:]-score[0]).mean()+.1*self.net._view_consistency(*views)
        loss.backward()
        missing=[name for name,p in self.net.named_parameters() if p.requires_grad and p.grad is None]
        nonfinite=[name for name,p in self.net.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
        self.assertEqual(missing,[])
        self.assertEqual(nonfinite,[])
        before={name:p.detach().clone() for name,p in self.net.named_parameters()}
        optimizer=torch.optim.AdamW(self.net.parameters(),lr=1e-4,weight_decay=1e-4)
        optimizer.step()
        for module in ('local_encoder','patient_encoder','prototype_encoder','patient_readout','score_head'):
            changed=any(not torch.equal(before[name],p.detach()) for name,p in self.net.named_parameters()
                        if name.startswith(module+'.'))
            self.assertTrue(changed,module)

    def test_changed_actual_CT_and_organ_change_cache_binding(self):
        from hiercp_v1x.v24_inputs import recipient_context
        CT=self.CT.copy();CT[10,20,18]+=5
        changed=recipient_context(self.recipient.case_id,CT,self.organ,self.spacing,self.recipient.image_affine)
        self.assertNotEqual(self.recipient.binding(),changed.binding())
        organ=self.organ.copy();organ[4,4,4]=False
        changed=recipient_context(self.recipient.case_id,self.CT,organ,self.spacing,self.recipient.image_affine)
        self.assertNotEqual(self.recipient.binding(),changed.binding())

    def test_heldout_donor_and_prototype_bank_rejected(self):
        from hiercp_v1x.v24_geometry import build_upper_graphs
        with self.assertRaisesRegex(ValueError,'train-only'):
            build_upper_graphs(self.recipient,self.donor,self.source,self.regions,self.donor_regions,self.rows,
                self.bank,config=self.config,ct_clip=self.clip,training_case_ids=('UNIT_other_train',))

    def test_legacy_GT_exposed_graph_is_not_accepted_by_actual_model(self):
        from torch_geometric.data import Batch
        graph=self.graph.clone();graph['lesion'].raw_x=torch.ones((1,14));graph['lesion'].num_nodes=1
        graph[('candidate','near','lesion')].edge_index=torch.tensor([[0],[0]])
        graph[('candidate','near','lesion')].edge_attr=torch.ones((1,12))
        upper=SimpleNamespace(patient_batch=Batch.from_data_list([graph]),
            prototype_batch=Batch.from_data_list([self.prototype.clone()]),counts=(len(self.rows),))
        with self.assertRaisesRegex(ValueError,'Incompatible'):
            self.net._score_upper(upper,{})

    def _cache(self):
        from hiercp_v1x.v24_geometry import V24UpperGeometryCache
        from hiercp_v1x.v24_inputs import tensor_digest
        output=ROOT/'outputs'/('v24_upper_cache_CPU_UNIT_'+uuid.uuid4().hex)
        state=dict(builds=0,source_valid=True,donor='actual_UNIT_donor_CT_mask_1')
        plan=SimpleNamespace(query_rows=copy.deepcopy(self.rows),record_ids=tuple(row['id'] for row in self.rows))
        def binding(plan):
            return dict(recipient=self.recipient.binding(),donor={'CT_mask_sha256':state['donor']},
                prototype_bank_sha256=self.bank.fingerprint(),config_sha256=tensor_digest(self.config.to_dict()),
                source_sha256='CPU_UNIT_exact_original_source')
        def guard(plan,proof):
            if not state['source_valid']:raise ValueError('UNIT actual input/source file changed')
        def builder(plan,provider):
            state['builds']+=1
            return self.graph.clone(),self.prototype.clone(),copy.deepcopy(self.audit)
        return V24UpperGeometryCache(None,output,builder,input_binding=binding,guard_inputs=guard,
            workers=2,resident_bytes=2**30,rss_bytes=2*2**30),plan,state

    def test_upper_cache_private_clones_and_supervision_independent_keys(self):
        from hiercp_v1x.v24_inputs import tensor_digest
        cache,plan,state=self._cache()
        first=cache.get(plan);expected=tensor_digest((first[0].to_dict(),first[1].to_dict()))
        first[0]['candidate'].raw_x.add_(100)
        for row in plan.query_rows:row['target']=1-row['target'];row['component']=999
        second=cache.get(plan)
        self.assertEqual(state['builds'],1)
        self.assertEqual(expected,tensor_digest((second[0].to_dict(),second[1].to_dict())))
        self.assertEqual(len(list(cache.output.glob('*.pt'))),1)

    def test_upper_cache_source_guard_and_replaced_file_never_silently_hit(self):
        cache,plan,state=self._cache();cache.get(plan)
        state['source_valid']=False
        with self.assertRaisesRegex(ValueError,'source file changed'):cache.get(plan)
        state['source_valid']=True
        path=next(cache.output.glob('*.pt'))
        with path.open('ab') as stream:stream.write(b'changed immutable CPU UNIT file')
        with self.assertRaisesRegex(ValueError,'file changed'):cache.get(plan)

    def test_upper_cache_donor_input_change_creates_separate_namespace(self):
        cache,plan,state=self._cache();cache.get(plan)
        state['donor']='actual_UNIT_donor_CT_mask_2'
        cache.get(plan)
        self.assertEqual(state['builds'],2)
        self.assertEqual(len(list(cache.output.glob('*.pt'))),2)

    def test_upper_cache_independent_processes_publish_one_complete_equal_payload(self):
        from hiercp_v1x.v24_inputs import tensor_digest
        cache,plan,state=self._cache()
        binding=cache.input_binding(plan)
        ctx=multiprocessing.get_context('spawn')
        barrier,results=ctx.Barrier(2),ctx.Queue()
        workers=[ctx.Process(target=_concurrent_upper_publisher,
            args=(str(cache.output),self.graph,self.prototype,self.audit,
                  plan.query_rows,binding,barrier,results)) for _ in range(2)]
        for worker in workers:worker.start()
        for worker in workers:
            worker.join(timeout=30)
            self.assertFalse(worker.is_alive(),'CPU UNIT publisher did not finish')
            self.assertEqual(worker.exitcode,0)
        actual=[results.get(timeout=5) for _ in workers]
        expected=tensor_digest((self.graph.to_dict(),self.prototype.to_dict()))
        self.assertTrue(all(row==dict(digest=expected,private=True) for row in actual),actual)
        self.assertEqual(len(list(cache.output.glob('*.pt'))),1)
        self.assertEqual(list(cache.output.glob('.v24-upper-publish-*.tmp')),[])
        admitted=cache.get(plan)
        self.assertEqual(expected,tensor_digest((admitted[0].to_dict(),admitted[1].to_dict())))
        self.assertEqual(state['builds'],0)

    def test_upper_cache_concurrent_differing_payload_is_rejected_without_overwrite(self):
        cache,plan,state=self._cache()
        cache.get(plan)
        path,key,binding=cache._path(plan)
        before=path.read_bytes()
        graph=self.graph.clone();graph['candidate'].raw_x.add_(1)
        from hiercp_v1x.v24_geometry import FORMAT
        from hiercp_v1x.v24_inputs import tensor_digest
        value=dict(format=FORMAT,query_sha256=key,input_binding=binding,graph=graph,
            prototype=self.prototype.clone(),audit=copy.deepcopy(self.audit),
            tensor_sha256=tensor_digest((graph.to_dict(),self.prototype.to_dict())))
        with self.assertRaisesRegex(ValueError,'different tensors'):
            cache._publish(path,value,plan,key,binding)
        self.assertEqual(path.read_bytes(),before)
        self.assertEqual(list(cache.output.glob('.v24-upper-publish-*.tmp')),[])

    def test_shared_memory_guard_evicts_before_process_RSS_check(self):
        from unittest.mock import patch
        cache,_,_=self._cache()
        state={'rss':cache.rss_bytes+1,'calls':0}
        def trim():
            state['calls']+=1;state['rss']=cache.rss_bytes-1
        cache.memory_guard=trim
        process=SimpleNamespace(memory_info=lambda:SimpleNamespace(rss=state['rss']))
        with patch('psutil.Process',return_value=process):cache._guard()
        self.assertEqual(state['calls'],1)
        cache.memory_guard=None;state['rss']=cache.rss_bytes+1
        with patch('psutil.Process',return_value=process):
            with self.assertRaises(MemoryError):cache._guard()


if __name__=='__main__':unittest.main()
