"""Explicit synthetic mathematical/selection unit tests; no medical scores."""
import unittest
from types import SimpleNamespace
import numpy as np
import torch
from tools.v22_rank_objective import rank_loss,ranking_metrics,RankingContext,resolve_objective,configuration,OBJECTIVE
from tools.v22_rank_recommendation import rank_then_filter
from tools.run_v222_server import stages
from pathlib import Path


class RankingTests(unittest.TestCase):
    def test_gradient_promotes_observed_and_orders_relative_to_unknown(self):
        scores=torch.tensor([-.3,.7,-.2],requires_grad=True)
        value,pairs=rank_loss(scores,torch.tensor([1,0,0]),torch.zeros(3,dtype=torch.long),torch.ones(3,dtype=torch.bool))
        self.assertEqual(int(pairs),2);value.backward()
        self.assertLess(float(scores.grad[0]),0)
        self.assertTrue((scores.grad[1:]>0).all())
        self.assertAlmostEqual(float(scores.grad.sum()),0.,places=6)

    def test_reference_positive_works_when_current_batch_has_no_positive(self):
        scores=torch.tensor([1.,0.,-.2],requires_grad=True)
        value,pairs=rank_loss(scores,torch.tensor([1,0,0]),torch.zeros(3,dtype=torch.long),torch.tensor([False,True,False]))
        self.assertEqual(int(pairs),1);value.backward()
        self.assertGreater(float(scores.grad[1]),0);self.assertEqual(float(scores.grad[2]),0)

    def test_case_boundaries_and_no_positive_are_explicit(self):
        scores=torch.tensor([1.,0.,5.,6.],requires_grad=True)
        value,pairs=rank_loss(scores,torch.tensor([1,0,0,0]),torch.tensor([0,0,1,1]),torch.ones(4,dtype=torch.bool))
        self.assertEqual(int(pairs),1);value.backward();self.assertTrue(torch.equal(scores.grad[2:],torch.zeros(2)))
        empty,count=rank_loss(scores,torch.zeros(4,dtype=torch.long),torch.zeros(4,dtype=torch.long),torch.ones(4,dtype=torch.bool))
        self.assertEqual(int(count),0);self.assertEqual(float(empty.detach()),0)

    def test_metrics_do_not_force_positives_to_top_or_reward_ties(self):
        metrics,rows=ranking_metrics([.1,.9,.8,.1],[1,0,0,0],['A','A','A','B'])
        self.assertEqual(rows[0]['observed_ranks'],[3]);self.assertEqual(metrics['ranking_recall_at_1'],0)
        self.assertEqual(metrics['ranking_cases_without_observed'],1)
        metrics,rows=ranking_metrics([1.,1.,1.],[1,0,0],['A']*3)
        self.assertEqual(rows[0]['observed_ranks'],[3])

    def test_complete_case_reference_not_only_minibatch(self):
        data=SimpleNamespace(rows=[dict(id='a',case_id='A',target=1),dict(id='b',case_id='A',target=0),dict(id='c',case_id='B',target=0)])
        memory=dict(record_ids=['c','a','b'],embeddings=torch.tensor([[3.],[1.],[2.]],requires_grad=True))
        context=RankingContext(data,memory)
        embeddings,positions,observed,cases=context.reference([1],torch.device('cpu'))
        self.assertEqual(embeddings.tolist(),[[1.],[2.]])
        self.assertFalse(embeddings.requires_grad);self.assertEqual(positions.tolist(),[1]);self.assertEqual(observed.tolist(),[1,0])

    def test_objective_resume_never_relabels_old_training(self):
        self.assertEqual(resolve_objective(),OBJECTIVE);self.assertEqual(resolve_objective({}),'observation_ce')
        with self.assertRaisesRegex(ValueError,'new run'):resolve_objective({},OBJECTIVE)
        saved=dict(training_objective=OBJECTIVE,ranking_contract=configuration())
        self.assertEqual(resolve_objective(saved),OBJECTIVE)
        saved['ranking_contract']['ranking_weight']=2
        with self.assertRaisesRegex(ValueError,'contract'):resolve_objective(saved)

    def test_server_forwards_explicit_ranking_contract(self):
        plan=dict(stages(Path('/m'),Path('/o'),32,9,optimized=True,process_loader=True,
            cache=Path('/cache/index.json'),training_objective=OBJECTIVE))
        command=plan['gnn_training']
        self.assertEqual(command[command.index('--training-objective')+1],OBJECTIVE)


class ExclusionTests(unittest.TestCase):
    def test_filter_all_tumors_after_ranking_not_just_first_place(self):
        label=np.ones((9,9,9),dtype=np.uint8);label[1,1,1]=label[5,5,5]=2
        out=rank_then_filter([.4,.9,.8],[[1,1,1],[3,3,3],[5,5,5]],np.ones((1,1,1),bool),[0,0,0],label,min_liver_coverage=1)
        self.assertEqual(out['selected_index'],1)
        self.assertEqual([r['index'] for r in out['ranked_candidates']],[1,2,0])
        self.assertEqual([r['model_score'] for r in out['ranked_candidates']],[.9,.8,.4])
        self.assertFalse(out['score_override'])

    def test_full_footprint_overlap_even_with_clear_center(self):
        label=np.ones((9,9,9),dtype=np.uint8);label[4,4,5]=2
        out=rank_then_filter([.9,.1],[[4,4,4],[1,1,1]],np.ones((3,3,3),bool),[1,1,1],label,min_liver_coverage=1)
        self.assertEqual(out['selected_index'],1)
        first=out['ranked_candidates'][0]
        self.assertFalse(first['observed_tumor_at_center']);self.assertEqual(first['tumor_overlap_voxels'],1)

    def test_no_admitted_candidate_keeps_original(self):
        label=np.full((3,3,3),2,dtype=np.uint8)
        out=rank_then_filter([.9],[[1,1,1]],np.ones((1,1,1),bool),[0,0,0],label,min_liver_coverage=1)
        self.assertTrue(out['keep_original']);self.assertIsNone(out['selected_index'])


if __name__=='__main__':unittest.main()
