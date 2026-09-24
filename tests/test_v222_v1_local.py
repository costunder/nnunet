"""Unit checks for the v1 local integration, not medical validation."""
import json
from pathlib import Path
import unittest
import torch
from hiercp_v222.v1_local import (V1LocalEncoder, materialize, model,
                                  score_candidates, support_for_recipient)

ROOT = Path(__file__).resolve().parents[1]


class V1LocalTests(unittest.TestCase):
    def test_retired_features_and_interior_are_absent(self):
        base=json.loads((ROOT/'config/train.json').read_text())
        with torch.device('meta'):
            local=V1LocalEncoder(base)
        self.assertEqual(sum(p.numel() for p in local.parameters()),4718420)
        self.assertEqual(local.dense_encoder.stem.block[0].in_channels,1)
        self.assertEqual(len(local.blocks),3)
        self.assertNotIn('tumor_interior',local.project)
        self.assertTrue(all(v[0].in_features==32 for v in local.project.values()))
        self.assertFalse(any('lin_edge' in n for n,_ in local.named_parameters()))

    def test_pair_batch_is_required(self):
        base=json.loads((ROOT/'config/train.json').read_text())
        with torch.device('meta'):
            local=V1LocalEncoder(base)
        with self.assertRaisesRegex(TypeError,'paired CT'):
            local(torch.empty(2,1,48,48,48,device='meta'))

    def test_legacy_cache_is_not_relabeled(self):
        with self.assertRaisesRegex(ValueError,'raw-CT paired'):
            materialize({'format':'canonical-full-v22','center_masking':True})

    def test_both_recipient_and_donor_identity_are_excluded(self):
        # Deliberately synthetic unit-test identity table, never model input data.
        memory=dict(patient_groups=['A','B','C','D'],donor_groups=['E','A','E','E','E'],
            owners=torch.tensor([0,1,1,2,3]),classes=torch.tensor([0,1,0,1,0]),
            embeddings=torch.arange(5*128).reshape(5,128).float())
        embeddings,owners,classes=support_for_recipient(memory,'A')
        torch.testing.assert_close(embeddings,memory['embeddings'][2:])
        self.assertEqual(owners.tolist(),[0,1,2])
        self.assertEqual(classes.tolist(),[0,1,0])
        del memory['donor_groups']
        with self.assertRaisesRegex(ValueError,'donor patient group'):
            support_for_recipient(memory,'A')

    def test_l1_l2_classes_and_counts_unchanged(self):
        cfg=json.loads((ROOT/'config/prompt_graph_v222_v1_l0.json').read_text())
        base=json.loads((ROOT/'config/train.json').read_text())
        with torch.device('meta'):
            network=model(cfg,base)
        self.assertEqual(len(network.l1),2)
        self.assertEqual(len(network.l2),2)
        self.assertEqual(sum(p.numel() for p in network.parameters())-4718420,832386)
        self.assertEqual(cfg['cp_probability'],.8)
        self.assertEqual(cfg['seed'],42)

    def test_incomplete_candidate_pool_fails(self):
        dummy=torch.nn.Linear(1,1).eval()  # invalid API fixture, no prediction.
        with self.assertRaisesRegex(ValueError,'all 128'):
            score_candidates(dummy,[],{},query_group='A',batch_size=4)


if __name__=='__main__':
    unittest.main()
