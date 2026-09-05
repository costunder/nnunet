"""DEBUG canonical/materialized scorer plumbing; no medical graph or bank build."""
from types import SimpleNamespace
import unittest
import numpy as np
import torch

from tools.online_scoring import PendingBankScorer


class CanonicalBankScoringDebugTests(unittest.TestCase):
    def scorer(self):
        scorer = PendingBankScorer.__new__(PendingBankScorer)
        scorer.torch, scorer.device = torch, torch.device("cpu")
        scorer.amp, scorer.pin_memory = False, False
        scorer.candidate_count, scorer.chunk_size = 128, 4

        def collate(samples):
            for sample in samples:
                if "local_graphs" not in sample:
                    sample["local_graphs"] = sample.pop("target_locals")
            return SimpleNamespace(case_ids=[v["case_id"] for v in samples],
                counts=[len(v["local_graphs"]) for v in samples], sample_count=len(samples))

        scorer.collate = collate
        scorer.model = SimpleNamespace(score_inference_chunked=lambda batch, local_chunk_size:
            [torch.arange(128, dtype=torch.float32) for _ in batch.counts])
        return scorer

    def test_canonical_then_materialized_input_preserves_candidate_order(self):
        scorer = self.scorer()
        samples = [dict(case_id="debug_train", target_locals=[None] * 128)]
        first = scorer._infer(samples)
        second = scorer._infer(samples)
        np.testing.assert_array_equal(first[0], np.arange(128))
        np.testing.assert_array_equal(first[0], second[0])

    def test_canonical_input_cannot_silently_reduce_candidate_count(self):
        with self.assertRaisesRegex(ValueError, "candidate count"):
            self.scorer()._infer([dict(case_id="debug_train", target_locals=[None] * 127)])


if __name__ == "__main__":
    unittest.main()
