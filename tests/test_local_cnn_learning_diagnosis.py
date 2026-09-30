"""Diagnostic selection/math tests; no production accuracy claims."""
import csv,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import torch
from tools.diagnose_local_cnn_learning import history_matches,find_run,read_history,score_summary,spread,checkpoint_for,trace_head

FIELDS=['epoch','ranking_pairwise_loss','ranking_mrr','ranking_recall_at_1','ranking_recall_at_5','ranking_recall_at_10']
class Checks(unittest.TestCase):
    def setUp(self):
        parent=Path('work').resolve()
        self.temp=tempfile.TemporaryDirectory(prefix='learning_diagnosis_unit_',dir=parent)
        self.root=Path(self.temp.name)
        assert self.root.is_relative_to(parent)
        self.addCleanup(self.temp.cleanup)
    def fixture(self,name='m20',mrr=.21271):
        root=self.root/'experiments'/name;(root/'attempts/0001').mkdir(parents=True)
        (root/'inventory').mkdir();(root/'inventory/index.json').write_text('{}')
        (root/'experiment.json').write_text(json.dumps(dict(attempts=[dict(output='attempts/0001',resume_from=None)])))
        p=root/'attempts/0001/validation_history.csv'
        with p.open('w',newline='') as f:
            w=csv.writer(f);w.writerow(FIELDS);w.writerow([17,.693147,mrr,.014814,.05185,.10370])
        return root,p
    def test_rounded_history_and_all_metrics(self):
        _,p=self.fixture();ref={17:dict(zip(FIELDS[1:],[.6931,.213,.015,.052,.104]))}
        self.assertTrue(history_matches(p,ref));ref[17]['ranking_recall_at_10']=.1
        self.assertFalse(history_matches(p,ref))
    def test_discovery_not_global_latest(self):
        root,p=self.fixture();self.fixture('m30',mrr=.3)
        self.assertEqual(find_run(self.root,read_history(p)),root.resolve())
    def test_duplicate_matches_rejected(self):
        _,p=self.fixture();self.fixture('copy')
        with self.assertRaisesRegex(ValueError,'found 2'):find_run(self.root,read_history(p))
    def test_checkpoint_uses_experiment_ledger(self):
        root,_=self.fixture();cp=root/'attempts/0001/checkpoint_latest.pt';cp.write_bytes(b'UNIT FIXTURE')
        (root/'unrelated.pt').write_bytes(b'NOT SELECTED')
        self.assertEqual(checkpoint_for(root),cp)
    def test_no_checkpoint_is_not_new_training(self):
        root,_=self.fixture()
        with self.assertRaisesRegex(ValueError,'no optimization'):checkpoint_for(root)
    def test_constant_scores_report_ties(self):
        r=score_summary(torch.zeros(4),torch.tensor([1,0,0,1]))
        self.assertAlmostEqual(r['mean_pairwise_loss'],.69314718,places=6)
        self.assertEqual(r['exact_tie_rate'],1);self.assertEqual(r['score_std'],0)
    def test_rank_separation_report(self):
        r=score_summary(torch.tensor([2.,-1.,-2.,3.]),torch.tensor([1,0,0,1]))
        self.assertEqual(r['pair_win_rate'],1);self.assertGreater(r['mean_positive_minus_unobserved'],0)
    def test_spread_detects_same_direction(self):
        x=torch.tensor([[1.,2.],[2.,4.]])
        r=spread(x);self.assertGreater(r['centered_energy'],0)
        self.assertLess(r['normalized_centered_energy'],1e-12)
    def test_cuda_head_trace_matches_production(self):
        from hiercp_v222.model import PromptGraphModel
        if not torch.cuda.is_available():self.skipTest('CUDA operator parity requires GPU')
        torch.set_num_threads(4);torch.manual_seed(123)
        cfg=json.loads(Path('config/prompt_graph_v222_v1_l0.json').read_text())
        base=json.loads(Path('config/train.json').read_text())
        net=PromptGraphModel(cfg,base,{},local_encoder=torch.nn.Identity()).cuda().eval()
        # Synthetic fixed-size operator inputs, not CT or trained predictions.
        support=(torch.randn(12,128,device='cuda'),torch.arange(3,device='cuda').repeat_interleave(4),torch.tensor([0,1,0,1]*3,device='cuda'))
        query=torch.randn(4,128,device='cuda')
        with torch.no_grad():
            measured,_=trace_head(net,query,support,2)
            logits=net.predict_embeddings(query,net.prepare_support(*support))['logits']
        torch.testing.assert_close(measured,logits[:,1]-logits[:,0],atol=2e-6,rtol=2e-5)

if __name__=='__main__':unittest.main()
