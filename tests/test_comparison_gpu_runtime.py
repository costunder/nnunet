"""CPU UNIT policy-admission checks only; no actual CT or CUDA measurement."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from hiercp_v1x import comparison_gpu_runtime as runtime


ORIGINAL = dict(dense_batch_size=4, checkpoint_dense_encoder=True, checkpoint_local_blocks=True)
SELECTED = dict(dense_batch_size=16, checkpoint_dense_encoder=False, checkpoint_local_blocks=True)


def unit_workload(cost=100):
    graph = dict(graphs=16, nodes=dict(total=cost, max=cost // 2),
                 edges=dict(total=cost * 3, max=cost),
                 node_types={"UNIT_node": cost}, edge_types={"UNIT_relation": cost * 3})
    return dict(source_patch_shape=[2, 5, 48, 48, 48],
                target_patch_shape=[16, 5, 48, 48, 48], counts=[8, 8],
                **{name: copy.deepcopy(graph) for name in runtime.GRAPH_NAMES})


class GpuRuntimeAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory(prefix="UNIT_gpu_policy_", dir=Path(__file__).resolve().parents[1])
        self.addCleanup(self.directory.cleanup)
        self.net = SimpleNamespace(local_encoder=SimpleNamespace(**ORIGINAL))
        self.loss = lambda output: output["UNIT_original_loss"]
        self.batch = object()
        self.notices = []
        self.controller = runtime.ComparisonGpuRuntime(
            self.net, self.loss, amp=True, cuda_limit_bytes=40 * 2**30,
            output=self.directory.name, notify=self.notices.append)
        self.probe = patch.object(runtime.policy, "calibrate_execution_policy",
            return_value=(copy.deepcopy(SELECTED), dict(unit_only=True, optimizer_steps=0))).start()
        self.addCleanup(patch.stopall)

    def receipts(self):
        path = Path(self.directory.name) / "gpu_execution.jsonl"
        return [json.loads(line) for line in path.read_text(encoding="utf8").splitlines()]

    def test_constructor_complete_resume_performs_no_probe_or_write(self):
        self.probe.assert_not_called()
        self.assertEqual(self.controller.current_settings(), ORIGINAL)
        self.assertFalse((Path(self.directory.name) / "gpu_execution.jsonl").exists())

    def test_first_actual_batch_uses_exact_bound_objective_and_original_budget(self):
        workload = unit_workload()
        before = copy.deepcopy(workload)
        row = self.controller.ensure(self.batch, workload)
        self.probe.assert_called_once_with(self.net, self.batch, self.loss, amp=True,
            cuda_limit_bytes=40 * 2**30, reserve_bytes=None, notify=self.notices.append)
        self.assertEqual(self.controller.current_settings(), SELECTED)
        self.assertEqual(workload, before)
        self.assertEqual(row["inventory"]["source_problems"], 2)
        self.assertFalse(row["calibration"].get("actual_CUDA_measured", False))
        self.assertEqual(row["calibration"]["optimizer_steps"], 0)
        self.assertIn("not worst-case", row["workload_admission"])
        self.assertEqual(len(self.notices), 2)
        self.assertIn("GPU execution calibration", self.notices[0])
        self.assertIn("GPU execution selected", self.notices[1])
        self.assertTrue(all(isinstance(value, str) for value in self.notices))

    def test_smaller_inventory_reuses_policy_and_validation_restore_is_reapplied(self):
        self.controller.ensure(self.batch, unit_workload())
        self.controller.restore_original()
        self.assertEqual(self.controller.current_settings(), ORIGINAL)
        self.assertIsNone(self.controller.ensure(self.batch, unit_workload(80)))
        self.probe.assert_called_once()
        self.assertEqual(self.controller.current_settings(), SELECTED)
        receipt = self.controller.current_receipt()
        receipt["settings"]["dense_batch_size"] = 99
        self.assertEqual(self.controller.current_settings(), SELECTED)

    def test_larger_unknown_uses_original_and_logs_once_per_inventory(self):
        self.controller.ensure(self.batch, unit_workload())
        row = self.controller.ensure(self.batch, unit_workload(120))
        self.assertEqual(self.controller.current_settings(), ORIGINAL)
        self.assertEqual(row["event"], "unmeasured_larger_workload_original_execution")
        self.assertIsNone(self.controller.ensure(self.batch, unit_workload(120)))
        self.probe.assert_called_once()
        events = [row["event"] for row in self.receipts()]
        self.assertEqual(events.count("unmeasured_larger_workload_original_execution"), 1)
        self.controller.ensure(self.batch, unit_workload(130))
        self.assertEqual(sum("larger unmeasured workloads" in value for value in self.notices), 1)
        self.assertEqual(sum(row["event"] == "unmeasured_larger_workload_original_execution"
                             for row in self.receipts()), 2)

    def test_doubled_highwater_probes_same_complete_batch_without_changing_loss(self):
        self.controller.ensure(self.batch, unit_workload())
        larger_batch = object()
        self.controller.ensure(larger_batch, unit_workload(200))
        self.assertEqual(self.probe.call_count, 2)
        self.assertIs(self.probe.call_args.args[1], larger_batch)
        self.assertIs(self.probe.call_args.args[2], self.loss)
        self.assertEqual(self.controller.current_settings(), SELECTED)
        starts = [value for value in self.notices if "GPU execution calibration" in value]
        self.assertIn("first actual batch", starts[0])
        self.assertIn("larger measured workload", starts[1])
        # Probe scheduling grows from measured coverage, without reducing any
        # model, graph, source, epoch or candidate setting.
        self.controller.ensure(self.batch, unit_workload(220))
        self.assertEqual(self.probe.call_count, 2)
        self.assertEqual(self.controller.current_settings(), ORIGINAL)

    def test_optimized_oom_quarantines_inventory_and_original_oom_is_fatal(self):
        self.controller.ensure(self.batch, unit_workload())
        error = RuntimeError("UNIT injected CUDA OOM")
        self.assertTrue(self.controller.reject_optimized_oom(unit_workload(), error))
        self.assertEqual(self.controller.current_settings(), ORIGINAL)
        self.assertIsNone(self.controller.ensure(self.batch, unit_workload()))
        self.probe.assert_called_once()
        self.assertFalse(self.controller.reject_optimized_oom(unit_workload(), error))
        self.controller.ensure(self.batch, unit_workload(80))
        self.assertEqual(self.controller.current_settings(), SELECTED)
        rows = self.receipts()
        self.assertEqual(rows[-1]["event"], "optimized_execution_OOM_original_retry")
        self.assertIn("caller restores RNG", rows[-1]["retry_scope"])

    def test_calibration_failure_restores_original_and_propagates(self):
        failure = runtime.policy.ExecutionCalibrationRejected([dict(accepted=False, unit_only=True)])
        def fail(*args, **kwargs):
            runtime.policy.apply_execution_settings(self.net, SELECTED)
            raise failure
        self.probe.side_effect = fail
        with self.assertRaises(runtime.policy.ExecutionCalibrationRejected) as caught:
            self.controller.ensure(self.batch, unit_workload())
        self.assertIs(caught.exception, failure)
        self.assertEqual(self.controller.current_settings(), ORIGINAL)
        self.assertEqual(self.receipts()[-1]["candidate_reports"], failure.reports)

    def test_invalid_or_missing_full_graph_inventory_cannot_be_benchmarked(self):
        cases = []
        row = unit_workload(); del row["local_batch_view2"]; cases.append(row)
        row = unit_workload(); row["counts"] = [1, 1]; cases.append(row)
        row = unit_workload(); row["patient_batch"]["edges"]["total"] = -1; cases.append(row)
        row = unit_workload(); row["source_patch_shape"][0] = 0; cases.append(row)
        for row in cases:
            with self.subTest(row=row):
                with self.assertRaises(ValueError):
                    self.controller.ensure(self.batch, row)
        self.probe.assert_not_called()


if __name__ == "__main__":
    unittest.main()
