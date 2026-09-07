"""DEBUG numerical calibration checks; not graph throughput or final training."""
import tempfile
import unittest
from unittest.mock import patch
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

    def pin_memory(self):
        return self

    def difficulty_list(self):
        return tuple(torch.tensor([0, 1, 2, 3]) for _ in range(self.count))


class DebugDataset:
    def __init__(self, files, **kwargs):
        self.files = files

    def __getitem__(self, index):
        return index

    def set_epoch(self, epoch):
        self.epoch = epoch


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
                curriculum_config=CurriculumConfig(), consistency_weight=0.1, epochs=40)
            self.assertIn(selected, (1, 2))
            self.assertTrue(all(row["status"] == "accepted" for row in trials))
            self.assertTrue(torch.equal(model.weight, original))
            self.assertTrue(torch.equal(torch.random.get_rng_state(), rng))
            self.assertTrue(all(row["measured_view_epochs"] == [1, 40] for row in trials))
            self.assertTrue(all(row["optimizer_buffers_retained_between_steps"] for row in trials))
            self.assertTrue(all(row["materialization_and_transfer_seconds"] >= 0 for row in trials))

    def test_peak_headroom_uses_worst_repeat_not_only_last(self):
        class DebugTorchProxy:
            cuda = SimpleNamespace(empty_cache=lambda: None, reset_peak_memory_stats=lambda device: None,
                                   synchronize=lambda device: None)

            def __getattr__(self, name):
                return getattr(torch, name)

        snapshots = [{"cuda_peak_allocated_bytes": peak, "cuda_total_bytes": 1000}
                     for peak in (200, 100, 50, 60)]
        with tempfile.TemporaryDirectory(prefix="debug_calibration_peak_") as tmp:
            paths = [Path(tmp) / str(i) for i in range(2)]
            for path in paths:
                path.write_bytes(b"DEBUG nonmedical inventory")
            model = DebugNumericalModel()
            with patch("hiercp.tensor.cuda_memory_snapshot", side_effect=snapshots):
                selected, trials = _measure_batch_candidates(
                    torch_module=DebugTorchProxy(), model=model, dataset_type=DebugDataset,
                    collate_fn=lambda samples: DebugBatch(len(samples)), train_files=paths,
                    candidates=[1, 2], repeats=2, max_vram_fraction=0.15,
                    device=torch.device("cuda"), use_amp=False, seed=42,
                    optimizer_kwargs={"lr": 0.01, "weight_decay": 0.01},
                    fused_optimizer=False, trainable_parameters=list(model.parameters()),
                    curriculum_config=CurriculumConfig(), consistency_weight=0.1)
        self.assertEqual(selected, 2)
        self.assertEqual(trials[0]["peak_vram_bytes"], 200)
        self.assertEqual(trials[0]["status"], "rejected_vram_headroom")


if __name__ == "__main__":
    unittest.main()
