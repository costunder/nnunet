"""Synthetic DEBUG contracts, not clinical validation or reduced production settings."""
import copy
import unittest
import numpy as np
import torch
from hiercp_v222.contracts import load_config, validate_identities
from hiercp_v222.inputs import context_patch, topology
from hiercp_v222.data import support_for_query, fit_contract, assert_no_duplicate_ct
from hiercp_v222.model import PromptGraphModel, supervised_loss

def debug_contract():
    # Small spatial fixture ONLY for adversarial unit tests; model depth/width retained.
    return dict(format='v222_fixed_blind_v1', fit_partition='inner_train', blind_radius_mm=3.0,
                context_width_mm=3.0, node_spacing_mm=2.5, edge_radius_mm=6.0, patch_size=48, ct_clip=[-200.,250.])

class LeakageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        cls.cfg, cls.base = load_config()

    def test_native_blind_invariance_and_visible_sensitivity(self):
        rng = np.random.default_rng(42)
        image = rng.normal(40, 20, (24, 24, 24)).astype(np.float32)
        center = np.array([12.,12.,12.]); spacing = np.array([1.,1.5,2.])
        grid = np.indices(image.shape).transpose(1,2,3,0)
        mask = np.linalg.norm((grid-center)*spacing, axis=-1) <= 3
        changed = image.copy(); changed[mask] = 999999
        a = context_patch(image, spacing, center, debug_contract())
        b = context_patch(changed, spacing, center, debug_contract())
        np.testing.assert_array_equal(a,b)
        changed[~mask] += 30
        self.assertFalse(np.array_equal(a, context_patch(changed, spacing, center, debug_contract())))
        with self.assertRaises(TypeError): context_patch(image, spacing, center, debug_contract(), target_mask=mask)

    def test_no_query_label_or_patient_geometry_input(self):
        model = PromptGraphModel(self.cfg, self.base, debug_contract()).eval()
        s = torch.randn(8,128); owners = torch.tensor([0,0,0,0,1,1,1,1]); cls = torch.tensor([0,1,0,1,0,1,0,1])
        with torch.no_grad():
            state = model.prepare_support(s,owners,cls)
            before = state['labels'].clone()
            q = torch.randn(3,128)
            out = model.predict_embeddings(q,state)
            loss_a = supervised_loss(out,torch.tensor([0,0,0]))
            loss_b = supervised_loss(out,torch.tensor([1,1,1]))
            torch.testing.assert_close(out['logits'], model.predict_embeddings(q,state)['logits'],rtol=0,atol=0)
            model.predict_embeddings(q+20,state)
            torch.testing.assert_close(before,state['labels'],rtol=0,atol=0)
            self.assertNotEqual(float(loss_a),float(loss_b))
        with self.assertRaises(TypeError): model.predict_embeddings(q,state,query_targets=torch.ones(3))

    def test_patient_group_exclusion_and_merge(self):
        memory = dict(patient_groups=['A','A','B','C'], owners=torch.tensor([0,1,2,3]),
                      classes=torch.tensor([0,1,0,1]), embeddings=torch.arange(512).reshape(4,128).float())
        s, owners, labels = support_for_query(memory,'A')
        torch.testing.assert_close(s,memory['embeddings'][2:]); self.assertEqual(owners.tolist(),[0,1])
        _, owners, _ = support_for_query(memory,'D')
        self.assertEqual(owners.tolist(),[0,0,1,2])
        with self.assertRaises(ValueError): support_for_query(dict(memory,patient_groups=['A','A','B','B']),'A')

    def test_alias_patient_across_partitions_rejected(self):
        split = dict(inner_train=['a','b'],inner_val=['c'],outer_train=['a','b','c'],outer_val=['d'])
        identity = dict(format='hiercp_patient_identity_v1',cases={c:dict(patient_group=c,identity_basis='DEBUG',annotation_complete=True) for c in 'abcd'})
        validate_identities(identity,split)
        identity['cases']['d']['patient_group']='a'
        with self.assertRaises(ValueError): validate_identities(identity,split)

    def test_duplicate_decoded_ct_rejected(self):
        split = dict(inner_train=['a'],inner_val=['b'],outer_val=['c'])
        with self.assertRaises(ValueError): assert_no_duplicate_ct({c:dict(decoded_ct_sha256='same') for c in 'abc'},split)

    def test_blind_fit_never_uses_validation(self):
        split = dict(inner_train=['a'])
        inv = dict(a=dict(positives=[dict(extent_mm=8)]),b=dict(positives=[dict(extent_mm=999)]))
        contract = fit_contract(inv,split,self.base)
        self.assertEqual(contract['blind_radius_mm'],10)
        inv['b']['positives'][0]['extent_mm']=99999
        self.assertEqual(fit_contract(inv,split,self.base),contract)

    def test_full_model_gradient_and_optimizer_path(self):
        torch.manual_seed(42)
        model = PromptGraphModel(self.cfg,self.base,debug_contract()).train()
        optimizer = torch.optim.AdamW(model.parameters(),lr=1e-4)
        before = {n:p.detach().clone() for n,p in model.named_parameters()}
        patches = torch.randn(2,1,48,48,48)
        out = model(patches,torch.randn(8,128),torch.tensor([0,0,0,0,1,1,1,1]),torch.tensor([0,1,0,1,0,1,0,1]))
        loss = supervised_loss(out,torch.tensor([0,1])); loss.backward()
        missing = [n for n,p in model.named_parameters() if p.grad is None]
        self.assertEqual(missing,[])
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))
        optimizer.step()
        for prefix in ['local.dense','local.project','local.pool_gate','label_seed']+[f'local.blocks.{i}' for i in range(3)]+[f'l1.{i}' for i in range(2)]+[f'l2.{i}' for i in range(2)]:
            self.assertTrue(any(not torch.equal(before[n],p) for n,p in model.named_parameters() if n.startswith(prefix)),prefix)

    def test_relation_and_l2_affect_output(self):
        torch.manual_seed(8)
        model = PromptGraphModel(self.cfg,self.base,debug_contract()).eval()
        s=torch.randn(8,128); owners=torch.tensor([0,0,0,0,1,1,1,1]); labels=torch.tensor([0,1,0,1,0,1,0,1]); q=torch.randn(3,128)
        with torch.no_grad():
            state=model.prepare_support(s,owners,labels)
            first=model.predict_embeddings(q,state)['logits']
            altered=model.predict_embeddings(q,model.prepare_support(s,owners,1-labels))['logits']
            no_alignment=model.predict_embeddings(q,dict(state,labels=state['local_labels']))['logits']
        self.assertFalse(torch.allclose(first,altered))
        self.assertFalse(torch.allclose(first,no_alignment))

    def test_grouped_loader_visits_every_row_without_cross_group_batch(self):
        from hiercp_v222.training import PatientBatchSampler
        class Fixture:
            rows=[dict(patient_group=g) for g in ['A']*5+['B']*3+['C']*7]
        fixture=Fixture();sampler=PatientBatchSampler(fixture,4,42)
        for epoch in (0,1):
            sampler.epoch=epoch;batches=list(sampler)
            flat=[i for batch in batches for i in batch]
            self.assertEqual(sorted(flat),list(range(len(fixture.rows))))
            self.assertTrue(all(len({fixture.rows[i]['patient_group'] for i in batch})==1 for batch in batches))
            self.assertEqual(len(batches),len(sampler))

    def test_legacy_checkpoint_and_bank_rejected(self):
        from hiercp_v222.contracts import validate_checkpoint
        from hiercp_v222.bank import validate_catalog
        with self.assertRaises(ValueError): validate_checkpoint({'format':'old'})
        with self.assertRaises(ValueError): validate_checkpoint({'format':'hiercp_v222_blinded_context_r1','debug':True})
        with self.assertRaises(ValueError): validate_catalog({'pipeline_version':'old'})

    def test_trilinear_values_and_gradients_match_grid_sample(self):
        model=PromptGraphModel(self.cfg,self.base,debug_contract())
        a=torch.randn(2,32,12,12,12,requires_grad=True)
        b=a.detach().clone().requires_grad_(True)
        custom=model.local.sample_features(a)
        grid=model.local.grid[None,:,None,None].expand(2,-1,-1,-1,-1)
        reference=torch.nn.functional.grid_sample(b,grid,align_corners=True,padding_mode='border')
        reference=reference[:,:,:,0,0].transpose(1,2).reshape(-1,32)
        torch.testing.assert_close(custom,reference,rtol=1e-5,atol=1e-6)
        weights=torch.randn_like(custom)
        (custom*weights).sum().backward();(reference*weights).sum().backward()
        torch.testing.assert_close(a.grad,b.grad,rtol=1e-5,atol=1e-6)

    def test_chunked_gat_preserves_full_graph_values_and_gradients(self):
        from hiercp_v222.model import chunked_gat
        model=PromptGraphModel(self.cfg,self.base,debug_contract())
        local=model.local;layer=local.blocks[0].eval();reference=copy.deepcopy(layer)
        n=len(local.grid)
        a=torch.randn(2,n,128,requires_grad=True);b=a.detach().clone().requires_grad_(True)
        edges=(local.edge[None]+torch.arange(2)[:,None,None]*n).permute(1,0,2).reshape(2,-1)
        actual=chunked_gat(layer,a,local.neighbors,local.neighbor_valid,7,checkpoint_chunks=True)
        expected=reference(b.reshape(-1,128),edges).reshape(2,n,128)
        torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)
        weight=torch.randn_like(actual)
        (actual*weight).sum().backward();(expected*weight).sum().backward()
        torch.testing.assert_close(a.grad,b.grad,rtol=3e-5,atol=3e-6)
        for p,q in zip(layer.parameters(),reference.parameters()):
            torch.testing.assert_close(p.grad,q.grad,rtol=1e-4,atol=2e-5)

if __name__=='__main__': unittest.main()
