"""DEBUG numerical calibration checks; not graph throughput or final training."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from hiercp.loss import CurriculumConfig
from hiercp.pipeline import _physical_batch_candidates, _measure_batch_candidates


class DebugBatch:
    def __init__(self, count):
        self.count = count

    def to(self, *args, **kwargs):
        return self

    def difficulty_list(self):
        return tuple(torch.tensor([0, 1, 2, 3]) for _ in range(self.count))


class DebugDataset:
    def __init__(self, files, **kwargs):
        self.files = files

    def __getitem__(self, index):
        return index


class DebugNumericalModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([0.1, -0.2, 0.4, 0.0]))

    def forward(self, batch):
        return SimpleNamespace(scores=tuple(self.weight * (i + 1) for i in range(batch.count)),
                               consistency=self.weight.square().mean())


class PhysicalCalibrationDebugTests(unittest.TestCase):
    def test_candidates_cover_whole_cohort_without_a_hidden_two_sample_cap(self):
        self.assertEqual(_physical_batch_candidates({"batch_size_candidates": "powers_of_two_to_cohort"}, 13),
                         [1, 2, 4, 8, 13])
        self.assertEqual(_physical_batch_candidates({"batch_size_candidates": "powers_of_two_to_cohort"}, 16),
                         [1, 2, 4, 8, 16])
        with self.assertRaises(ValueError):
            _physical_batch_candidates({"batch_size_candidates": "powers_of_two_to_cohort"}, 1)

    def test_real_loss_calibration_preserves_weights_and_rng(self):
        with tempfile.TemporaryDirectory(prefix="debug_batch_calibration_") as tmp:
            paths = [Path(tmp) / str(i) for i in range(2)]
            for path in paths:
                path.write_bytes(b"debug calibration inventory, not a medical cache")
            model = DebugNumericalModel()
            original = model.weight.detach().clone()
            rng = torch.random.get_rng_state().clone()
            selected, trials = _measure_batch_candidates(
                torch_module=torch, model=model, dataset_type=DebugDataset,
                collate_fn=lambda samples: DebugBatch(len(samples)), train_files=paths,
                candidates=[1, 2], repeats=2, max_vram_fraction=0.9,
                device=torch.device("cpu"), use_amp=False, seed=42,
                optimizer_kwargs={"lr": 0.01, "weight_decay": 0.01},
                fused_optimizer=False, trainable_parameters=list(model.parameters()),
                curriculum_config=CurriculumConfig(), consistency_weight=0.1)
            self.assertIn(selected, (1, 2))
            self.assertTrue(all(row["status"] == "accepted" for row in trials))
            self.assertTrue(torch.equal(model.weight, original))
            self.assertTrue(torch.equal(torch.random.get_rng_state(), rng))


if __name__ == "__main__":
    unittest.main()
