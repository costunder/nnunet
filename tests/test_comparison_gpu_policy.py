"""CPU UNIT fixtures for execution policy/state guards; not CUDA measurements."""
from contextlib import ExitStack, nullcontext
from types import SimpleNamespace
import random
import unittest
from unittest.mock import patch

import numpy as np
import torch

from hiercp_v1x import comparison_gpu_policy as policy


class UnitNet(torch.nn.Module):
    """Explicit tiny UNIT model; never passed off as the research architecture."""
    def __init__(self):
        super().__init__()
        self.local_encoder = torch.nn.Linear(1, 1)
        self.local_encoder.dense_batch_size = 4
        self.local_encoder.checkpoint_dense_encoder = True
        self.local_encoder.checkpoint_local_blocks = True
        self.register_buffer("counter", torch.tensor(0.))
        self.register_buffer("transient_counter", torch.tensor(0.), persistent=False)
        self.fail_retained = False
        self.bad_retained = False

    def forward(self, batch):
        self.counter.add_(1)
        self.transient_counter.add_(1)
        random.random()
        np.random.random()
        noise = torch.rand(()) * .01
        if self.fail_retained and not self.local_encoder.checkpoint_dense_encoder:
            raise torch.cuda.OutOfMemoryError("UNIT explicitly injected candidate OOM")
        logits = self.local_encoder(torch.arange(16.).reshape(16, 1)).flatten() + noise
        if self.bad_retained and not self.local_encoder.checkpoint_dense_encoder:
            logits = logits * 2
        return SimpleNamespace(scores=list(logits.reshape(2, 8)), consistency=logits.mean().square())


def unit_batch():
    return SimpleNamespace(counts=[8, 8], local_batch_view2=object(),
        source_patches=torch.zeros(2, 5, 1, 1, 1), target_patches=torch.zeros(16, 5, 1, 1, 1),
        bridge_indices=(11, 29))


def unit_loss(output):
    return torch.stack(output.scores).square().mean() + .1 * output.consistency


class GpuPolicyUnitTests(unittest.TestCase):
    def setUp(self):
        # CPU UNIT tests must not initialize a real CUDA context even when the
        # test host has a GPU; individual tests inject the entire CUDA surface.
        availability = patch.object(torch.cuda, "is_available", return_value=False)
        availability.start()
        self.addCleanup(availability.stop)

    def cuda_unit_mocks(self):
        stack = ExitStack()
        values = {"is_available": True, "synchronize": None, "reset_peak_memory_stats": None,
            "mem_get_info": (14*1024**3, 16*1024**3), "memory_allocated": 1024**3,
            "max_memory_allocated": 2*1024**3, "max_memory_reserved": 3*1024**3,
            "empty_cache": None, "get_device_name": "CPU UNIT injected CUDA interface",
            "get_rng_state_all": [], "set_rng_state_all": None}
        for name, value in values.items():
            stack.enter_context(patch.object(torch.cuda, name, return_value=value))
        stack.enter_context(patch.object(policy, "_cuda_device", return_value=torch.device("cpu")))
        stack.enter_context(patch.object(torch, "autocast", return_value=nullcontext()))
        return stack

    def test_only_three_runtime_attributes_can_change(self):
        net = UnitNet()
        before = policy.digest(net.state_dict())
        parameters = tuple(id(value) for value in net.parameters())
        settings = dict(dense_batch_size=16, checkpoint_dense_encoder=False, checkpoint_local_blocks=False)
        policy.apply_execution_settings(net, settings)
        self.assertEqual(policy.execution_settings(net), settings)
        self.assertEqual(before, policy.digest(net.state_dict()))
        self.assertEqual(parameters, tuple(id(value) for value in net.parameters()))
        with self.assertRaises(ValueError): policy.apply_execution_settings(net, dict(settings, hidden_dim=64))
        with self.assertRaises(ValueError): policy.apply_execution_settings(net, dict(settings, dense_batch_size=0))
        with self.assertRaises(ValueError): policy.apply_execution_settings(net, dict(settings, checkpoint_local_blocks=0))

    def test_chunks_cover_actual_batch_without_reducing_baseline_or_capping(self):
        self.assertEqual(policy.dense_chunk_candidates(4, 16), [4, 8, 16])
        self.assertEqual(policy.dense_chunk_candidates(8, 35), [8, 16, 32, 35])
        self.assertEqual(policy.dense_chunk_candidates(8, 3), [8])
        flags = policy._flag_candidates(policy.execution_settings(UnitNet()))
        self.assertEqual(len(flags), 4)
        self.assertEqual({row["dense_batch_size"] for row in flags}, {4})

    def test_full_calibration_preserves_state_grads_rng_modes_and_settings(self):
        net = UnitNet().eval()
        net.local_encoder.train()
        next(net.parameters()).grad = torch.full_like(next(net.parameters()), .75)
        before = policy.digest(net.state_dict())
        rng = policy.capture_rng()
        grads = [None if p.grad is None else p.grad.clone() for p in net.parameters()]
        modes = [module.training for module in net.modules()]
        original = policy.execution_settings(net)
        notifications = []
        with self.cuda_unit_mocks():
            selected, receipt = policy.calibrate_execution_policy(net, unit_batch(), unit_loss, amp=False,
                cuda_limit_bytes=12*1024**3, notify=notifications.append)
        self.assertEqual(before, policy.digest(net.state_dict()))
        self.assertEqual(policy.digest(rng), policy.digest(policy.capture_rng()))
        self.assertEqual(modes, [module.training for module in net.modules()])
        self.assertEqual(original, policy.execution_settings(net))
        self.assertEqual(float(net.transient_counter), 0.)
        for expected, parameter in zip(grads, net.parameters()):
            if expected is None: self.assertIsNone(parameter.grad)
            else: self.assertTrue(torch.equal(expected, parameter.grad))
        self.assertEqual(len(receipt["reports"]), 6)
        self.assertEqual(len(notifications), 7)
        self.assertTrue(all(isinstance(message, str) for message in notifications))
        self.assertIn("source batch unchanged", notifications[-1])
        self.assertTrue(all(row["accepted"] for row in receipt["reports"]))
        self.assertEqual(receipt["workload"]["physical_source_batch"], 2)
        self.assertEqual(receipt["workload"]["candidate_graphs"], 16)
        self.assertEqual(receipt["optimizer_updates"], 0)
        self.assertEqual(receipt["selected_policy"], selected)
        self.assertEqual(receipt["numerical_equivalence"]["rtol"], 1e-4)
        self.assertGreater(receipt["headroom_bytes"], 1024**3)

    def test_candidate_oom_is_reported_with_original_state_preserved(self):
        net = UnitNet()
        net.fail_retained = True
        before = policy.digest(net.state_dict())
        with self.cuda_unit_mocks():
            selected, receipt = policy.calibrate_execution_policy(net, unit_batch(), unit_loss,
                amp=False, cuda_limit_bytes=12*1024**3)
        self.assertTrue(selected["checkpoint_dense_encoder"])
        rejected = [row for row in receipt["reports"] if not row["accepted"]]
        self.assertEqual(len(rejected), 2)
        self.assertTrue(all("CUDA OOM rejected" in row["error"] for row in rejected))
        self.assertEqual(before, policy.digest(net.state_dict()))

    def test_changed_outputs_or_gradients_cannot_win_through_speed(self):
        net = UnitNet()
        net.bad_retained = True
        with self.cuda_unit_mocks():
            selected, receipt = policy.calibrate_execution_policy(net, unit_batch(), unit_loss,
                amp=False, cuda_limit_bytes=12*1024**3)
        self.assertTrue(selected["checkpoint_dense_encoder"])
        rejected = [row for row in receipt["reports"] if not row["accepted"]]
        self.assertEqual(len(rejected), 2)
        self.assertTrue(all("equivalence rejected" in row["error"] for row in rejected))

    def test_baseline_oom_is_fatal_and_restores_state(self):
        net = UnitNet().eval()
        before = policy.digest(net.state_dict())
        rng = policy.capture_rng()
        with self.cuda_unit_mocks(), patch.object(net, "forward", side_effect=torch.cuda.OutOfMemoryError("UNIT baseline")):
            with self.assertRaises(policy.ExecutionCalibrationRejected) as caught:
                policy.calibrate_execution_policy(net, unit_batch(), unit_loss,
                    amp=False, cuda_limit_bytes=12*1024**3)
        self.assertEqual(len(caught.exception.reports), 1)
        self.assertFalse(caught.exception.reports[0]["accepted"])
        self.assertEqual(before, policy.digest(net.state_dict()))
        self.assertEqual(policy.digest(rng), policy.digest(policy.capture_rng()))
        self.assertFalse(net.training)

    def test_programming_error_is_not_swallowed_and_restores_state(self):
        net = UnitNet().eval()
        original = policy.execution_settings(net)
        before = policy.digest(net.state_dict())
        def failing_loss(output):
            raise ValueError("UNIT objective failed")
        with self.cuda_unit_mocks():
            with self.assertRaisesRegex(ValueError, "UNIT objective failed"):
                policy.calibrate_execution_policy(net, unit_batch(), failing_loss,
                    amp=False, cuda_limit_bytes=12*1024**3)
        self.assertEqual(before, policy.digest(net.state_dict()))
        self.assertEqual(original, policy.execution_settings(net))
        self.assertFalse(net.training)

    def test_rejects_missing_views_coverage_cpu_and_unsafe_budget(self):
        net, batch = UnitNet(), unit_batch()
        with patch.object(torch.cuda, "is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "Actual CUDA"):
                policy.calibrate_execution_policy(net, batch, unit_loss, amp=False, cuda_limit_bytes=12*1024**3)
        with self.cuda_unit_mocks():
            batch.local_batch_view2 = None
            with self.assertRaisesRegex(ValueError, "genuine two views"):
                policy.calibrate_execution_policy(net, batch, unit_loss, amp=False, cuda_limit_bytes=12*1024**3)
            with self.assertRaisesRegex(ValueError, "two measured"):
                policy.calibrate_execution_policy(net, unit_batch(), unit_loss, amp=False,
                    cuda_limit_bytes=12*1024**3, repeats=1)

    def test_releases_unused_allocator_cache_before_driver_free_memory_check(self):
        calls = []
        with self.cuda_unit_mocks(), patch.object(torch.cuda, "empty_cache", side_effect=lambda: calls.append("release")), \
             patch.object(torch.cuda, "mem_get_info", side_effect=lambda device: (calls.append("free") or (14*1024**3, 16*1024**3))):
            policy.calibrate_execution_policy(UnitNet(), unit_batch(), unit_loss,
                amp=False, cuda_limit_bytes=12*1024**3)
        self.assertEqual(calls[:2], ["release", "free"])

    def test_original_can_continue_when_it_fits_hard_budget_but_not_optimization_headroom(self):
        net = UnitNet()
        original = policy.execution_settings(net)
        with self.cuda_unit_mocks():
            selected, receipt = policy.calibrate_execution_policy(net, unit_batch(), unit_loss,
                amp=False, cuda_limit_bytes=2*1024**3)
        self.assertEqual(selected, original)
        self.assertEqual(receipt["calibration_status"], "original_retained_no_headroom")
        self.assertEqual(len(receipt["reports"]), 1)
        self.assertFalse(receipt["reports"][0]["within_preferred_headroom"])

    def test_measured_noise_has_no_multiplier_and_global_guard_cannot_hide_distributed_drift(self):
        reference = dict(scores=torch.zeros(2, 8), consistency=torch.tensor(0.), loss=torch.tensor(1.))
        gradient = torch.ones(10000)*.001
        noise = dict(outputs={name: torch.ones_like(value)*.0003 for name, value in reference.items()},
                     gradient_tensors_l2={"weight": .0005}, gradient_global_l2=.0005)
        observed = {name: value+.0002 for name, value in reference.items()}
        result = policy._equivalence(observed, {"weight": gradient+.000001},
            (reference, {"weight": gradient}), rtol=.01, atol=.0001, noise=noise)
        self.assertTrue(result["accepted"])
        # Every changed element is below atol, yet the whole-vector direction
        # moved by 5%, exceeding 1% plus the independently measured 0.5% noise.
        rejected = policy._equivalence(observed, {"weight": gradient+.00005},
            (reference, {"weight": gradient}), rtol=.01, atol=.0001, noise=noise)
        self.assertFalse(rejected["accepted"])
        self.assertFalse(rejected["gradient_global_accepted"])
        self.assertEqual(rejected["elementwise_gradient_failed_parameters"], [])
        bad_output = {name: value+.02 for name, value in reference.items()}
        rejected = policy._equivalence(bad_output, {"weight": gradient},
            (reference, {"weight": gradient}), rtol=.01, atol=.0001, noise=noise)
        self.assertFalse(rejected["accepted"])
        self.assertIn("scores", rejected["failed_outputs"])

    def test_changed_rng_consumption_cannot_be_selected(self):
        net = UnitNet()
        original_forward = net.forward
        def extra_rng(batch):
            result = original_forward(batch)
            if not net.local_encoder.checkpoint_dense_encoder:
                torch.rand(())
            return result
        with self.cuda_unit_mocks(), patch.object(net, "forward", side_effect=extra_rng):
            selected, receipt = policy.calibrate_execution_policy(net, unit_batch(), unit_loss,
                amp=False, cuda_limit_bytes=12*1024**3)
        self.assertTrue(selected["checkpoint_dense_encoder"])
        rejected = [row for row in receipt["reports"] if not row["accepted"]]
        self.assertEqual(len(rejected), 2)
        self.assertTrue(all(not row["trials"][0]["equivalence"]["rng_consumption_matches_reference"] for row in rejected))


if __name__ == "__main__":
    unittest.main()
