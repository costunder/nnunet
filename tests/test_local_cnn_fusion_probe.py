"""Full-size operator checks, not trained CT ranking evidence."""
import copy
import json
from pathlib import Path
import unittest

import torch

from tests.test_local_cnn_l0_probe import fixture
from tools.local_cnn_fusion_probe import (
    BasisBank, fusion_basis, missing_batches, probe_fusion, support_indices, validate_scales,
)


class FusionProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)

    def test_cuda_unchanged_cnn_output_once_and_detached_basis(self):
        if not torch.cuda.is_available():
            self.skipTest('Real CUDA parity required')
        local, batch, _ = fixture('cuda')
        calls=[]
        handle=local.cnn.register_forward_hook(lambda *args:calls.append(1))
        before=copy.deepcopy(local.state_dict())
        with torch.no_grad():
            recipient,fused=fusion_basis(local,batch)
        self.assertEqual(len(calls),1);handle.remove()
        with torch.no_grad():
            expected=local(batch)
        torch.testing.assert_close(fused,expected,atol=2e-6,rtol=2e-5)
        self.assertEqual(recipient.shape,(3,128));self.assertFalse(recipient.requires_grad)
        for key,value in before.items():
            if torch.is_tensor(value):torch.testing.assert_close(value,local.state_dict()[key],rtol=0,atol=0)
            else:self.assertEqual(value,local.state_dict()[key])

    def test_coverage_and_no_duplicate_encoding(self):
        rows=[dict(id=str(i),case_id='c'+str(i//3),donor_case_id='d',donor_component=1) for i in range(7)]
        bank=BasisBank(rows);r=torch.randn(2,128);f=torch.randn(2,128)
        bank.add([0,2],r,f)
        self.assertEqual(missing_batches(rows,range(7),bank,2),[[1],[3,4],[5],[6]])
        with self.assertRaisesRegex(ValueError,'Repeated'):
            bank.add([0],r[:1],f[:1])
        with self.assertRaisesRegex(ValueError,'Incomplete'):
            bank.get([0,1],'cpu')
        rr,ff=bank.get([2,0],'cpu')
        torch.testing.assert_close(rr,r[[1,0]]);torch.testing.assert_close(ff,f[[1,0]])

    def test_full_support_excludes_recipient_and_donor_and_keeps_every_other_row(self):
        rows=[dict(patient_group=p,donor_group=d,target=t) for p,d,t in
              [('query','other',0),('a','query',1),('a','z',0),('a','z',1),('b','z',0),('b','z',1)]]
        self.assertEqual(support_indices(rows,'query'),[2,3,4,5])
        with self.assertRaisesRegex(ValueError,'two other'):
            support_indices(rows[1:],'a')

    def test_explicit_scales_and_detached_basis_required(self):
        for value in ([],[0,0],[float('nan')],[-1]):
            with self.assertRaisesRegex(ValueError,'explicit'):validate_scales(value)
        bank=BasisBank([dict(id='x')])
        with self.assertRaisesRegex(ValueError,'detached'):
            bank.add([0],torch.randn(1,128,requires_grad=True),torch.randn(1,128))

    def test_train_mode_and_mixed_donor_rejected(self):
        local,batch,_=fixture();local.train()
        with self.assertRaisesRegex(ValueError,'eval'):fusion_basis(local,batch)
        local.eval();batch.donor[1]=1;batch.validate()
        with self.assertRaisesRegex(ValueError,'Same-donor'):fusion_basis(local,batch)

    def test_cuda_identical_formula_for_support_query_and_fresh_plan_per_branch(self):
        if not torch.cuda.is_available():self.skipTest('Real CUDA downstream parity required')
        from hiercp_v222.model import PromptGraphModel
        cfg=json.loads(Path('config/prompt_graph_v222_v1_l0.json').read_text())
        base=json.loads(Path('config/train.json').read_text())
        torch.manual_seed(814)
        net=PromptGraphModel(cfg,base,{},local_encoder=torch.nn.Identity()).cuda().eval()
        qr,qf=torch.randn(5,128,device='cuda'),torch.randn(5,128,device='cuda')
        sr,sf=torch.randn(12,128,device='cuda'),torch.randn(12,128,device='cuda')
        owners=torch.arange(3,device='cuda').repeat_interleave(4)
        classes=torch.tensor([0,1,0,1]*3,device='cuda');truth=torch.tensor([1,0,0,0,1],device='cuda')
        seen=[];original=net.fit_support_clusters
        def fit(emb,*args):
            seen.append(emb.clone());return original(emb,*args)
        net.fit_support_clusters=fit
        with torch.no_grad():
            report=probe_fusion(net,qr,qf,sr,sf,owners,classes,scales=[0,.25,1],batch=2,
                truth=truth,support_record_ids=[str(i) for i in range(12)],query_group='query')
        self.assertEqual(len(seen),4)
        for actual,expected in zip(seen,[sf,sr,sr+.25*sf,sr+sf]):
            torch.testing.assert_close(actual,expected,atol=0,rtol=0)
        for row,q,s in zip(report['branches'],[qf,qr,qr+.25*qf,qr+qf],seen):
            with torch.no_grad():
                state=net.prepare_support(s,owners,classes)
                logits=torch.cat([net.predict_embeddings(chunk,state)['logits'] for chunk in q.split(2)])
                expected=float((logits[:,1]-logits[:,0]).std(unbiased=False))
            self.assertAlmostEqual(row['score']['score_std'],expected,places=6)
            self.assertTrue(row['query_support_formula_identical'])
            self.assertTrue(row['support_plan_refitted'])
        self.assertIsNone(report['production_compatible_lambda'])


if __name__=='__main__':unittest.main()
