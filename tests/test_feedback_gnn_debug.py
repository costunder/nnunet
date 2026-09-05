"""DEBUG synthetic graphs only; no clinical observations or final training.

Widths/patches use the existing explicit debug fixture; all 3/2/2 hierarchy
blocks are retained. The full-128-pool test checks cardinality, not medical
performance. CPU physical-batch calibration is actually executed once.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch

from test_hierarchy_model_debug import debug_batch, debug_model
from hiercp.feedback import (BankGraphProvider, FeedbackGNNRuntime, MEASUREMENT,
                             observed_difficulty_loss, tensor_state_sha256, validate_observations)
from hiercp.model import FeedbackDifficultyModel


class DebugGraphProvider:
    def __init__(self, counts=(4, 3)):
        batch = debug_batch(counts)
        local = batch.local_batch.to_data_list()
        second = batch.local_batch_view2.to_data_list()
        patients, populations = batch.patient_batch.to_data_list(), batch.prototype_batch.to_data_list()
        self.samples, self.counts, self.entry_cases = {}, {}, {}
        offset = 0
        for i, count in enumerate(counts):
            entry, case = f"debug_entry_{i}", f"debug_case_{i}"
            self.entry_cases[entry], self.counts[entry] = case, count
            self.samples[entry] = {"case_id": case, "source_patch": batch.source_patches[i],
                                   "target_patches": batch.target_patches[offset:offset + count],
                                   "local_graphs": local[offset:offset + count],
                                   "local_graphs_view2": second[offset:offset + count],
                                   "patient_graph": patients[i], "prototype_graph": populations[i],
                                   "difficulties": torch.ones(count, dtype=torch.long)}
            offset += count

    def get(self, entry):
        return copy.deepcopy(self.samples[entry])


def debug_runtime(counts=(4, 3)):
    provider = DebugGraphProvider(counts)
    config = json.loads((Path(__file__).parents[1] / "config" / "online_cp_feedback_gnn.json").read_text())
    return FeedbackGNNRuntime(model=FeedbackDifficultyModel(debug_model()), provider=provider,
                              config=config, identity={"train_case_ids": list(provider.entry_cases.values())},
                              device="cpu", num_workers=0, local_chunk_size=5)


def debug_records(provider, epoch=0):
    return [{"entry_id": entry, "case_id": provider.entry_cases[entry], "candidate_index": index,
             "error": float((index + 1) / (count + 2)), "phase": "train", "timing": "pre_update", "epoch": epoch}
            for entry, count in provider.counts.items() for index in range(count)]


def debug_progress(epoch):
    return {"completed_epoch": epoch, "optimizer_steps": (epoch + 1) * 9, "network_sha256": "a" * 64}


class FeedbackGNNDebugTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_debug_all_levels_gradient_update_and_quality_immutable(self):
        quality = debug_model().eval()
        before = tensor_state_sha256(quality.state_dict())
        difficulty = FeedbackDifficultyModel(quality).train()
        self.assertEqual(quality.hidden_dim, difficulty.hierarchy.hidden_dim)
        self.assertEqual(sum(p.numel() for p in difficulty.parameters()),
                         sum(p.numel() for p in quality.parameters()) + 1)
        old = {name: value.detach().clone() for name, value in difficulty.named_parameters()}
        batch = debug_batch()
        scores = difficulty(batch)
        groups = {str(i): [{"candidate_index": j, "error": (j + 1) / (len(value) + 2)}
                          for j in range(len(value))] for i, value in enumerate(scores)}
        loss = observed_difficulty_loss(scores, list(groups), groups)
        optimizer = torch.optim.AdamW(difficulty.parameters(), lr=1e-3)
        loss.backward()
        for name, parameter in difficulty.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        optimizer.step()
        for encoder, blocks in (("local_encoder", 3), ("patient_encoder", 2), ("prototype_encoder", 2)):
            for block in range(blocks):
                prefix = f"hierarchy.{encoder}.blocks.{block}."
                self.assertTrue(any(not torch.equal(old[name], value) for name, value in difficulty.named_parameters()
                                    if name.startswith(prefix)), prefix)
        bias_name = f"hierarchy.score_head.{len(difficulty.hierarchy.score_head) - 1}.bias"
        self.assertFalse(torch.equal(old[bias_name], dict(difficulty.named_parameters())[bias_name]))
        self.assertEqual(before, tensor_state_sha256(quality.state_dict()))

    def test_debug_observed_only_vectorized_loss_matches_duplicate_exposure_reference(self):
        first = torch.tensor([.3, -.8, 2.], requires_grad=True)
        second = torch.tensor([-.5, .9], requires_grad=True)
        groups = {"a": [{"candidate_index": 1, "error": .2}, {"candidate_index": 1, "error": .8}],
                  "b": [{"candidate_index": 0, "error": .6}]}
        actual = observed_difficulty_loss([first, second], ["a", "b"], groups)
        expected = torch.nn.functional.binary_cross_entropy_with_logits(
            torch.stack([first[1], first[1], second[0]]), torch.tensor([.2, .8, .6]))
        self.assertTrue(torch.equal(actual, expected))
        actual.backward()
        self.assertEqual(first.grad[0], 0.)
        self.assertEqual(first.grad[2], 0.)
        self.assertEqual(second.grad[1], 0.)

    def test_debug_rejects_validation_wrong_case_index_epoch_and_nonfinite_targets(self):
        provider = DebugGraphProvider()
        base = debug_records(provider)[0]
        changes = ({"phase": "val"}, {"timing": "post_update"}, {"case_id": "heldout"},
                   {"candidate_index": -1}, {"candidate_index": 4}, {"candidate_index": True},
                   {"error": float("nan")}, {"error": 1.01}, {"epoch": 1})
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_observations([dict(base, **change)], epoch=0, entry_cases=provider.entry_cases,
                                      candidate_counts=provider.counts, train_cases=set(provider.entry_cases.values()))

    def test_debug_real_calibration_update_predict_and_caller_rng_isolation(self):
        runtime = debug_runtime()
        self.assertEqual(runtime.predict(0), (None, None))
        caller_rng = torch.get_rng_state().clone()
        python_rng, numpy_rng = random.getstate(), np.random.get_state()
        report = runtime.update(debug_records(runtime.provider), 0, nnunet_progress=debug_progress(0))
        predictions, provenance = runtime.predict(1)
        self.assertTrue(torch.equal(caller_rng, torch.get_rng_state()))
        self.assertEqual(python_rng, random.getstate())
        after_numpy = np.random.get_state()
        self.assertEqual(numpy_rng[0], after_numpy[0])
        self.assertTrue(np.array_equal(numpy_rng[1], after_numpy[1]))
        self.assertEqual(numpy_rng[2:], after_numpy[2:])
        self.assertEqual(report["status"], "updated")
        self.assertTrue(report["all_parameters_connected"])
        self.assertEqual(sum(report["actual_physical_batches"]), 2)
        self.assertEqual([trial["physical_batch_size"] for trial in runtime.calibration["update"]["trials"]], [1, 2])
        self.assertEqual(set(predictions), set(runtime.provider.entry_cases))
        self.assertEqual({key: len(value) for key, value in predictions.items()}, runtime.provider.counts)
        self.assertEqual(provenance["measurement_definition"], MEASUREMENT)
        self.assertEqual(provenance["trained_through_epoch"], 0)
        self.assertEqual(provenance["prediction_epoch"], 1)
        self.assertEqual(next(runtime.model.parameters()).device.type, "cpu")
        with self.assertRaisesRegex(ValueError, "already consumed"):
            runtime.update(debug_records(runtime.provider), 0, nnunet_progress=debug_progress(0))

    def test_debug_resume_rejects_identity_change_and_preserves_next_update(self):
        first = debug_runtime()
        fixed = {"physical_batch_size": 2}
        with mock.patch.object(first, "_calibrate", return_value=fixed):
            first.update(debug_records(first.provider), 0, nnunet_progress=debug_progress(0))
        state = first.state_dict()
        second = debug_runtime()
        altered = copy.deepcopy(state)
        altered["identity"]["train_case_ids"] = ["heldout"]
        with self.assertRaisesRegex(ValueError, "identity"):
            second.load_state_dict(altered)
        second.load_state_dict(state)
        for runtime in (first, second):
            with mock.patch.object(runtime, "_calibrate", return_value=fixed):
                runtime.update(debug_records(runtime.provider, epoch=1), 1, nnunet_progress=debug_progress(1))
        self.assertEqual(tensor_state_sha256(first.model.state_dict()), tensor_state_sha256(second.model.state_dict()))
        self.assertEqual(first.records, second.records)
        self.assertEqual(first.optimizer_steps, second.optimizer_steps)

    def test_debug_trained_then_no_observations_resume_preserves_training_lineage_without_predictions(self):
        runtime = debug_runtime()
        with mock.patch.object(runtime, "_calibrate", return_value={"physical_batch_size": 2}):
            runtime.update(debug_records(runtime.provider), 0, nnunet_progress=debug_progress(0))
        model_before = tensor_state_sha256(runtime.model.state_dict())
        observations_before = copy.deepcopy(runtime.records)
        steps_before = runtime.optimizer_steps
        newer_student = dict(debug_progress(1), network_sha256="b" * 64)
        with mock.patch.object(runtime, "_calibrate", side_effect=AssertionError("no observations must not calibrate")):
            runtime.update([], 1, nnunet_progress=newer_student)
            self.assertEqual(runtime.predict(2), (None, None))
        self.assertEqual(runtime.completed_epoch, 1)
        self.assertEqual(runtime.trained_through_epoch, 0)
        self.assertEqual(runtime.nnunet_progress, debug_progress(0))
        self.assertNotEqual(runtime.nnunet_progress["network_sha256"], newer_student["network_sha256"])
        saved = runtime.state_dict()
        restored = debug_runtime()
        restored.load_state_dict(saved)
        self.assertEqual(restored.completed_epoch, 1)
        self.assertEqual(restored.trained_through_epoch, 0)
        self.assertEqual(restored.nnunet_progress, debug_progress(0))
        self.assertEqual(restored.records, observations_before)
        self.assertEqual(restored.optimizer_steps, steps_before)
        self.assertEqual(tensor_state_sha256(restored.model.state_dict()), model_before)
        with mock.patch.object(restored, "_calibrate", side_effect=AssertionError("stale model must not repredict")):
            self.assertEqual(restored.predict(2), (None, None))

    def test_debug_prediction_covers_all_128_candidates_in_every_entry(self):
        runtime = debug_runtime((128, 128))
        runtime.local_chunk_size = 37  # Debug streaming crosses a source boundary.
        # A genuine synthetic observation update, not a forged trained state.
        records = [debug_records(runtime.provider)[0], debug_records(runtime.provider)[128]]
        with mock.patch.object(runtime, "_calibrate", return_value={"physical_batch_size": 2}):
            runtime.update(records, 0, nnunet_progress=debug_progress(0))
            predictions, _ = runtime.predict(1)
        self.assertEqual({key: len(value) for key, value in predictions.items()}, runtime.provider.counts)
        self.assertTrue(all(np.isfinite(values).all() for values in predictions.values()))

    def test_debug_full_128_pool_chunk_logits_and_every_parameter_gradient_equivalence(self):
        full = FeedbackDifficultyModel(debug_model()).train()
        chunked = copy.deepcopy(full)
        batch = debug_batch((128, 128))
        direct = full(copy.deepcopy(batch))
        streamed = chunked(copy.deepcopy(batch), local_chunk_size=37)
        grouped = {str(i): [{"candidate_index": j, "error": (j + 1) / (len(values) + 2)}
                           for j in range(len(values))] for i, values in enumerate(direct)}
        for left, right in zip(direct, streamed):
            torch.testing.assert_close(left, right, rtol=3e-4, atol=3e-5)
        observed_difficulty_loss(direct, list(grouped), grouped).backward()
        observed_difficulty_loss(streamed, list(grouped), grouped).backward()
        maximum = 0.
        for (name, left), (other, right) in zip(full.named_parameters(), chunked.named_parameters()):
            self.assertEqual(name, other)
            self.assertIsNotNone(left.grad, name)
            self.assertIsNotNone(right.grad, name)
            torch.testing.assert_close(left.grad, right.grad, rtol=5e-3, atol=2e-5, msg=name)
            maximum = max(maximum, float((left.grad - right.grad).abs().max()))
        print(f"[DEBUG FeedbackChunk] physical_batch=2 candidate_pools=[128,128] "
              f"all_parameter_gradient_max_absolute_difference={maximum:.9g}", flush=True)

    def test_debug_failed_partial_update_cannot_be_checkpointed_or_predicted(self):
        runtime = debug_runtime()
        with mock.patch.object(runtime, "_calibrate", side_effect=RuntimeError("debug measured failure")):
            with self.assertRaisesRegex(RuntimeError, "debug measured failure"):
                runtime.update(debug_records(runtime.provider), 0, nnunet_progress=debug_progress(0))
        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            runtime.state_dict()
        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            runtime.predict(1)

    def test_debug_raw_center_component_binding_reconstructs_real_candidate_statistics(self):
        from hiercp.common import CasePaths
        from hiercp.schema import GraphBuildConfig

        with tempfile.TemporaryDirectory(prefix="hiercp_feedback_debug_") as temporary:
            root = Path(temporary)
            path = root / "debug_entry.npz"
            centers = np.asarray([[7, 7, 7], [8, 7, 7]], dtype=np.int32)
            np.savez(path, candidate_raw_centers=centers, scores=np.asarray([.1, .2]), source_component=[1])
            label = np.ones((12, 12, 12), dtype=np.int16)
            label[3:5, 3:5, 3:5] = 2
            image = np.arange(label.size, dtype=np.float32).reshape(label.shape)
            case = SimpleNamespace(paths=CasePaths("debug", root / "image", root / "label"), image=image,
                                   label=label, shape=label.shape, spacing=np.ones(3))
            provider = BankGraphProvider.__new__(BankGraphProvider)
            provider.counts, provider.raw_paths = {"entry": 2}, {"debug": (root / "image", root / "label")}
            provider.train = {"labels": {"liver": 1, "tumor": 2}, "generation": {"source_pad": 1}}
            provider.graph_config, provider.ct_clip = GraphBuildConfig(), (-200., 250.)
            provider.seed, provider.prototype, provider.cache = 42, object(), root / "cache"
            captured = {}
            def inference(loaded, source, candidates, prototype, **kwargs):
                captured.update(source=source, candidates=candidates, kwargs=kwargs)
                return {"candidate_centers": torch.tensor([item.center for item in candidates])}, []
            regions = SimpleNamespace(full_organ_mask=np.ones(label.shape, dtype=bool),
                                      organ_depth=np.ones(label.shape, dtype=np.float32))
            with mock.patch("hiercp.common.load_case", return_value=case), \
                 mock.patch("hiercp.region.load_or_build_patient_regions", return_value=regions), \
                 mock.patch("hiercp.cache.build_inference_sample", side_effect=inference), \
                 mock.patch.object(provider, "_assert_sources"):
                result = provider._build("entry", path, "debug")
            self.assertTrue(np.array_equal(result["candidate_centers"].numpy(), centers))
            self.assertEqual(captured["source"].component_id, 1)
            self.assertEqual(captured["source"].voxel_count, 8)
            for candidate in captured["candidates"]:
                self.assertEqual(candidate.liver_coverage, 1.)
                self.assertTrue(math_is_finite_positive(candidate.context_std_hu))
                self.assertGreater(candidate.occupied_distance_mm, 0.)


def math_is_finite_positive(value):
    return bool(np.isfinite(value) and value > 0)


if __name__ == "__main__":
    unittest.main()
