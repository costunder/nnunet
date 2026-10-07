"""Synthetic topology fixtures; actual CT costs are in the separate report."""
from tests.artifacts import unit_artifact_root
import copy,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import torch
from hiercp_v222.v1_cache import configuration
from hiercp_v222.v1_local import V1LocalEncoder
from hiercp_v222.model import PromptGraphModel,supervised_loss
from tools.v222_review_contracts import installed
from tools.v22_artifacts import tree_hash
from l0_ezsp.config import load_profile
from l0_ezsp.data import collate as fine_collate
from l0_ezsp.diagnostic import ResourceBudget
from l0_regions.preparation import fixed_profile,binding,prepare
from l0_regions.data import collate,save_new,load,validate_item
from l0_regions.materialization import seal_materialization
from l0_regions.resident import RegionResidentCache
from l0_regions.encoder import RegionSAGEEncoder
from tests.test_l0_ezsp_debug import synthetic


@unittest.skipUnless(torch.cuda.is_available(),'Official GPU partition required')
class FixedHierarchy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(8);torch.manual_seed(42)
        cls.cfg,cls.base=configuration()
        with installed('stride4'):cls.reference=V1LocalEncoder(cls.base)
        cls.budget=ResourceBudget(6*2**30,12*2**30,180)
        cls.fine=fine_collate([(synthetic(),0),(synthetic(),1)]).to('cuda')
        cls.profile=fixed_profile(load_profile(reg_scale1=.02,reg_scale2=.02))
        cnn_hash=tree_hash(cls.reference.dense_encoder.state_dict())
        cls.bindings=[binding(record=dict(id=str(i),sha256='synthetic',shared_source={},center=[0,0,0],donor_case_id='fixture',donor_component=1),
            dataset_index=i,cache_sha256='synthetic',frozen_cnn_sha256=cnn_hash,profile=cls.profile,view_epoch=0,view_index=0,feature_evidence='untrained_plumbing_only') for i in range(2)]
        receipts=[dict(dataset_index=i,record_id=str(i),record_sha256='synthetic',cache_sha256='synthetic',
            shared_source={},center=[0,0,0],donor_case_id='fixture',donor_component=1,view_epoch=0,view_index=0) for i in range(2)]
        cls.fine=seal_materialization(cls.fine,receipts)
        frozen=copy.deepcopy(cls.reference).cuda().eval().requires_grad_(False)
        cls.items,cls.audit=prepare(cls.fine,frozen,cls.profile,cls.bindings,cls.budget,allow_unvalidated_profile=True)

    def model(self):return RegionSAGEEncoder(self.reference,seed=42,resource_budget=self.budget,allow_unvalidated_profile=True).cuda()

    def test_cache_identity_and_tamper_rejected(self):
        folder=tempfile.mkdtemp(prefix='fixed_region_fixture_',dir=unit_artifact_root())
        # Keep this small corruption fixture as evidence; never delete user paths.
        if folder:
            path=Path(folder)/'pair.pt';save_new(path,self.items[0])
            result=load(path,self.bindings[0]);self.assertEqual(result['binding'],self.bindings[0])
            for key in ('view','center','record_sha256','frozen_cnn_sha256','preparation_source_sha256'):
                wrong=copy.deepcopy(self.bindings[0]);wrong[key]='changed'
                with self.assertRaises(ValueError):load(path,wrong)
            with path.open('ab') as f:f.write(b'corruption')
            with self.assertRaises(ValueError):load(path,self.bindings[0])

    def test_invalid_coverage_and_edges_rejected(self):
        for mutation in ('mass','parent','grid','edge'):
            item=copy.deepcopy(self.items[0]);role='tumor_surface'
            if mutation=='mass':item['scales'][0][role]['mass'][0]+=1
            elif mutation=='parent':item['fine'][role]['parent'][0]=-1
            elif mutation=='grid':item['fine'][role]['grid'][0,0]=float('nan')
            else:
                rel=next(iter(item['edges'][0]));item['edges'][0][rel]=torch.tensor([[999999],[0]])
            with self.assertRaises(ValueError):validate_item(item)

    def test_rebatch_output_and_no_fine_edges(self):
        net=self.model().eval()
        with torch.no_grad():
            together=net(collate(self.items,[0,1]).to('cuda'))
            alone=torch.cat([net(collate([item],[i]).to('cuda')) for i,item in enumerate(self.items)])
        torch.testing.assert_close(together,alone,rtol=1e-4,atol=1e-5)
        self.assertFalse(any('edge' in key for key in self.items[0]['fine']['tumor_surface']))

    def test_both_typed_quotients_have_exact_original_witnesses(self):
        from l0_ezsp.ops import quotient
        batch=collate(self.items,[0,1]).to('cuda')
        for rel in self.fine.graph.edge_types:
            expected,_,_,_=quotient(self.fine.graph[rel].edge_index,batch.fine[rel[0]]['parent'],batch.fine[rel[2]]['parent'],same_type=rel[0]==rel[2])
            torch.testing.assert_close(expected,batch.graphs[0][rel].edge_index)
            second,_,_,_=quotient(expected,batch.regions[0][rel[0]].parent,batch.regions[0][rel[2]].parent,same_type=rel[0]==rel[2])
            torch.testing.assert_close(second,batch.graphs[1][rel].edge_index)

    def test_live_cnn_gradient_no_partition_in_update(self):
        local=self.model();net=PromptGraphModel(self.cfg,self.base,{},local_encoder=local).cuda().train()
        batch=collate(self.items,[0,1]).to('cuda')
        original=[batch.fine[k]['parent'].clone() for k in batch.fine]
        before=next(local.dense_encoder.parameters()).detach().clone()
        # Any accidental online merge must fail, including checkpoint backward.
        with patch('l0_ezsp.encoder.partition',side_effect=AssertionError('online partition forbidden')):
            embeddings=net.local(batch);classes=torch.tensor([0,1],device='cuda')
            output=net.predict_embeddings(embeddings,net.prepare_support(embeddings,classes,classes))
            loss=supervised_loss(output,classes);loss.backward()
            for prefix in ('local.core.dense_encoder','local.core.blocks.0.conv','local.core.blocks.1.conv','local.core.blocks.2.conv','l1','l2'):
                grads=[p.grad for n,p in net.named_parameters() if n.startswith(prefix)]
                self.assertTrue(grads and all(g is not None and torch.isfinite(g).all() for g in grads),prefix)
                self.assertTrue(any(bool((g!=0).any()) for g in grads),prefix)
            torch.optim.AdamW(net.parameters(),lr=1e-4).step()
        self.assertFalse(torch.equal(before,next(local.dense_encoder.parameters())))
        for old,k in zip(original,batch.fine):self.assertTrue(torch.equal(old,batch.fine[k]['parent']))
        with self.assertRaises(RuntimeError):net.state_dict()

    def test_failed_profile_not_normal_sample(self):
        batch=collate(self.items,[0,1]).to('cuda');batch.profile_exceeded=True
        net=RegionSAGEEncoder(self.reference,seed=42,resource_budget=self.budget,allow_unvalidated_profile=False).cuda()
        with self.assertRaises(ValueError):net(batch)

    def test_materialized_record_view_and_content(self):
        from l0_regions.materialization import verify_materialization
        for key in ('record_id','view'):
            bindings=copy.deepcopy(self.bindings)
            bindings[0][key]='other' if key=='record_id' else dict(epoch=999,index=99)
            with self.assertRaises(ValueError):verify_materialization(self.fine,bindings)
        with self.assertRaises(ValueError):verify_materialization(self.fine,list(reversed(self.bindings)))
        bad=copy.deepcopy(self.fine);bad.target_patches[0,0,0,0,0]+=1
        with self.assertRaises(ValueError):verify_materialization(bad,self.bindings)

    def test_duplicate_record_and_wrong_indices(self):
        with self.assertRaises(ValueError):collate([self.items[0],self.items[0]],[0,1])
        with self.assertRaises(ValueError):collate(self.items,[1,0])

    def test_missing_second_relation_even_with_resealed_receipt(self):
        from l0_regions.data import seal_item,profile_status
        item=copy.deepcopy(self.items[0])
        rel=next(r for r,e in item['edges'][1].items() if e.shape[1])
        item['edges'][1][rel]=item['edges'][1][rel][:,:-1]
        item['profile_exceeded']=profile_status(item);seal_item(item)
        with self.assertRaisesRegex(ValueError,'exact typed quotient'):validate_item(item)

    def test_profile_flag_cannot_override_audit(self):
        from l0_regions.data import seal_item
        item=copy.deepcopy(self.items[0]);role=next(iter(item['audit']['scale1']['roles']))
        groups=item['audit']['scale1']['roles'][role]['group_diagnostics']
        next(g for g in groups if g['owner']==0)['variance_excess']['clusters']=1
        item['profile_exceeded']=False;seal_item(item)
        with self.assertRaisesRegex(ValueError,'profile flag'):validate_item(item)

    def test_resident_reuse_mutation_and_budget(self):
        folder=Path(tempfile.mkdtemp(prefix='resident_fixture_',dir=unit_artifact_root()))
        paths=[folder/f'{i}.pt' for i in range(2)]
        for p,item in zip(paths,self.items):save_new(p,item)
        cache=RegionResidentCache(max_tensor_bytes=32*2**20,workers=2)
        first=cache.get(paths,self.bindings,[0,1])
        with patch('l0_regions.resident.load',side_effect=AssertionError('Repeated disk load')):
            self.assertIs(first,cache.get(paths,self.bindings,[0,1]))
        first.target_patches[0,0,0,0,0]+=1
        with self.assertRaisesRegex(ValueError,'mutated'):cache.get(paths,self.bindings,[0,1])
        with self.assertRaises(MemoryError):RegionResidentCache(max_tensor_bytes=1,workers=2).get(paths,self.bindings,[0,1])
        verified=load(paths[0],self.bindings[0]);verified['source_patch'][0,0,0,0]+=1
        with self.assertRaisesRegex(ValueError,'mutated'):validate_item(verified)

    def test_preparation_dependencies_isolated_bytes(self):
        from l0_regions.preparation import PREPARATION_FILES,preparation_fingerprint
        folder=Path(tempfile.mkdtemp(prefix='fingerprint_fixture_',dir=unit_artifact_root()))
        for name in PREPARATION_FILES:
            p=folder/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('fixture',encoding='utf-8')
        before=preparation_fingerprint(folder,{})
        for name in ('l0_ezsp/encoder.py','tools/v222_review_contracts.py'):
            p=folder/name;p.write_text('changed isolated fixture',encoding='utf-8')
            self.assertNotEqual(before,preparation_fingerprint(folder,{}))
            p.write_text('fixture',encoding='utf-8')

    def test_missing_verification_rejected_at_all_boundaries(self):
        folder=Path(tempfile.mkdtemp(prefix='missing_signature_',dir=unit_artifact_root()))
        paths=[folder/f'{i}.pt' for i in range(2)]
        for p,item in zip(paths,self.items):save_new(p,item)
        net=self.model().eval()
        for mutation in ('none','ct','metadata','replacement'):
            with self.subTest(mutation=mutation):
                cache=RegionResidentCache(max_tensor_bytes=32*2**20,workers=2)
                batch=cache.get(paths,self.bindings,[0,1])
                del batch._verified_signature
                if mutation=='ct':batch.target_patches[0,0,0,0,0]+=1
                elif mutation=='metadata':batch.bindings[0]['center'][0]+=1
                elif mutation=='replacement':batch.target_patches=batch.target_patches.clone()
                with self.assertRaisesRegex(ValueError,'signature missing'):cache.get(paths,self.bindings,[0,1])
                for device in ('cpu','cuda'):
                    with self.assertRaisesRegex(ValueError,'signature missing'):batch.to(device)
                with patch.object(net.core,'encode_dense_maps',side_effect=AssertionError('CNN must not run')) as cnn:
                    with self.assertRaisesRegex(ValueError,'signature missing'):net(batch)
                    cnn.assert_not_called()
                self.assertFalse(hasattr(batch,'_verified_signature'))

    def test_valid_cold_hit_transfer_and_storage_accounting(self):
        from l0_regions.resident import check_verified,storage_bytes
        folder=Path(tempfile.mkdtemp(prefix='valid_signature_',dir=unit_artifact_root()))
        paths=[folder/f'{i}.pt' for i in range(2)]
        serialized=sum(save_new(p,item) for p,item in zip(paths,self.items))
        cache=RegionResidentCache(max_tensor_bytes=32*2**20,workers=2)
        batch=cache.get(paths,self.bindings,[0,1])
        with patch('l0_regions.resident.load',side_effect=AssertionError('hit reload')),patch('l0_regions.resident.collate',side_effect=AssertionError('hit recollate')):
            self.assertIs(batch,cache.get(paths,self.bindings,[0,1]))
        self.assertTrue(check_verified(batch.to('cpu')))
        gpu=batch.to('cuda');self.assertTrue(check_verified(gpu))
        self.assertEqual(storage_bytes(batch),storage_bytes([batch,batch]))
        self.assertEqual(storage_bytes(batch),storage_bytes(gpu))
        self.assertNotEqual(serialized,storage_bytes(batch))
        with torch.no_grad():self.assertEqual(tuple(self.model().eval()(gpu).shape),(2,128))


if __name__=='__main__':unittest.main(verbosity=2)
