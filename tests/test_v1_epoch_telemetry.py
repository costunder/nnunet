"""Epoch-telemetry CPU metadata/invariant units; no neural or CUDA PASS claim.

Archived source is parsed without importing the medical pipeline. Fake events
test timing bookkeeping only, and never serve as measured GPU results.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path
import pickle
import shutil
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch
import uuid
import weakref
from zipfile import ZipFile

from hiercp_v1x import epoch_telemetry as telemetry


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "versions/v1/pipeline_v1_source.zip"


def archived_run_train_source():
    with ZipFile(ARCHIVE) as archive:
        raw = archive.read("hiercp/pipeline.py")
    source = raw.decode("utf-8")
    function = next(node for node in ast.parse(source).body
                    if isinstance(node, ast.FunctionDef) and node.name == "run_train")
    return raw, "\n".join(source.splitlines()[function.lineno - 1:function.end_lineno]) + "\n"


def batch_timing(sample_count=2):
    return SimpleNamespace(sample_count=sample_count, **{telemetry.BATCH_TIMING_KEY: {
        "sampling_worker_seconds_sum": 0.3,
        "sampling_worker_sample_seconds_min": 0.1,
        "sampling_worker_sample_seconds_max": 0.2,
        "sampling_measured_samples": sample_count,
        "collate_worker_seconds_sum": 0.04,
    }})


class EpochTelemetrySourceMetadataUnits(unittest.TestCase):
    def test_archive_and_function_are_pinned_and_overlay_is_reversible(self):
        archive_before = ARCHIVE.read_bytes()
        raw, source = archived_run_train_source()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), telemetry.PIPELINE_SHA256)
        self.assertEqual(hashlib.sha256(source.encode()).hexdigest(), telemetry.RUN_TRAIN_SHA256)
        rewritten, identity = telemetry.overlay_source(source)
        for before, after in reversed(telemetry.SOURCE_REWRITES):
            self.assertEqual(rewritten.count(after), 1)
            rewritten = rewritten.replace(after, before, 1)
        self.assertEqual(rewritten, source)
        self.assertEqual(ARCHIVE.read_bytes(), archive_before)
        self.assertFalse(identity["per_batch_cuda_synchronization_added"])
        self.assertFalse(identity["rng_operations_added"])

    def test_changed_source_and_missing_expected_pattern_fail_closed(self):
        _, source = archived_run_train_source()
        with self.assertRaisesRegex(telemetry.TelemetryError, "source hash mismatch"):
            telemetry.overlay_source(source.replace("ranking_loss + consistency_weight", "ranking_loss - consistency_weight"))
        # Hash bypass here exists only to exercise the second independent guard.
        broken = source.replace("        for batch_index, batch in enumerate(loader):\n", "        for batch_index, batch in enumerate(loader):  # metadata UNIT\n")
        broken_digest = hashlib.sha256(broken.encode()).hexdigest()
        with patch.object(telemetry, "RUN_TRAIN_SHA256", broken_digest):
            with self.assertRaisesRegex(telemetry.TelemetryError, "expected one source pattern"):
                telemetry.overlay_source(broken)

    def test_compute_calls_optimizer_rng_and_sync_order_are_preserved(self):
        _, source = archived_run_train_source()
        rewritten, _ = telemetry.overlay_source(source)
        # Remove telemetry-only expressions and restore the two loader wrappers.
        class RemoveTelemetry(ast.NodeTransformer):
            def visit_Expr(self, node):
                if isinstance(node.value, ast.Call):
                    target = node.value.func
                    if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
                        if target.value.id == "_v1x_pass":
                            return None
                        if target.value.id == "result" and target.attr == "update" and node.value.args:
                            value = node.value.args[0]
                            if isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute) and isinstance(value.func.value, ast.Name) and value.func.value.id == "_v1x_pass":
                                return None
                        if target.value.id == "_v1x_epoch_telemetry":
                            if target.attr in ("save_checkpoint", "report"):
                                node.value.func = node.value.args.pop(0)
                                return self.generic_visit(node)
                            return None
                return self.generic_visit(node)

            def visit_Assign(self, node):
                if any(isinstance(target, ast.Name) and target.id == "_v1x_pass" for target in node.targets):
                    return None
                return self.generic_visit(node)

            def visit_Call(self, node):
                if isinstance(node.func, ast.Name) and node.func.id == "enumerate" and node.args:
                    arg = node.args[0]
                    if isinstance(arg, ast.Call) and isinstance(arg.func, ast.Attribute) and isinstance(arg.func.value, ast.Name) and arg.func.value.id == "_v1x_pass":
                        node.args[0] = arg.args[0]
                if isinstance(node.func, ast.Name) and node.func.id == "DataLoader":
                    node.keywords = [item for item in node.keywords if item.arg != "worker_init_fn"]
                    for item in node.keywords:
                        if item.arg == "collate_fn":
                            item.value = ast.Name(id="collate_samples", ctx=ast.Load())
                return self.generic_visit(node)

        original_tree = ast.parse(source)
        recovered_tree = RemoveTelemetry().visit(ast.parse(rewritten))
        self.assertEqual(ast.dump(recovered_tree, include_attributes=False),
                         ast.dump(original_tree, include_attributes=False))
        self.assertEqual(rewritten.count("torch.cuda.synchronize(device)"), source.count("torch.cuda.synchronize(device)"))
        helper_tree = ast.parse(inspect_helper_source(telemetry.PassTelemetry))
        synchronization_calls = [node for node in ast.walk(helper_tree)
                                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                                 and node.func.attr == "synchronize"]
        self.assertEqual(synchronization_calls, [])

    def test_worker_hooks_are_picklable_for_spawn(self):
        for function in (telemetry.telemetry_worker_init, telemetry.timed_collate_samples):
            self.assertIs(pickle.loads(pickle.dumps(function)), function)


def inspect_helper_source(value):
    import inspect
    return inspect.getsource(value)


class EpochTelemetrySamplingMetadataUnits(unittest.TestCase):
    def test_sampling_and_collator_called_once_preserving_original_objects(self):
        fake_package = ModuleType("hiercp")
        fake_data = ModuleType("hiercp.data")
        fake_package.data = fake_data
        calls = []
        graph = object()
        tensor = object()
        sample = {"graph": graph, "tensor": tensor}

        def materialize(*args, **kwargs):
            calls.append(("sample", args, kwargs))
            return sample

        def collate(samples):
            calls.append(("collate", samples))
            return SimpleNamespace(sample_count=len(samples), graph=graph, tensor=tensor)

        fake_data.materialize_sample_views = materialize
        fake_data.collate_samples = collate
        with patch.dict(sys.modules, {"hiercp": fake_package, "hiercp.data": fake_data}):
            telemetry.prepare_sampling_capture()
            telemetry.telemetry_worker_init(0)  # Idempotent with a forked worker.
            measured = fake_data.materialize_sample_views("original", training=True, epoch=17, global_seed=42)
            batch = telemetry.timed_collate_samples([measured])
        self.assertIs(measured, sample)
        self.assertIs(batch.graph, graph)
        self.assertIs(batch.tensor, tensor)
        self.assertEqual(set(sample) - {telemetry.SAMPLE_TIMING_KEY}, {"graph", "tensor"})
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0], ("sample", ("original",), {"training": True, "epoch": 17, "global_seed": 42}))
        self.assertIs(calls[1][1][0], sample)
        self.assertGreaterEqual(getattr(batch, telemetry.BATCH_TIMING_KEY)["sampling_worker_seconds_sum"], 0)

    def test_missing_sampling_is_an_error_and_never_a_fake_zero(self):
        fake_package = ModuleType("hiercp")
        fake_data = ModuleType("hiercp.data")
        fake_package.data = fake_data
        fake_data.collate_samples = lambda samples: self.fail("Original collator must not run on missing timing")
        with patch.dict(sys.modules, {"hiercp": fake_package, "hiercp.data": fake_data}):
            with self.assertRaisesRegex(telemetry.TelemetryError, "missing"):
                telemetry.timed_collate_samples([{"actual_graph": object()}])

    def test_cpu_pass_reports_gpu_unavailable_and_counts_actual_samples(self):
        monitor = telemetry.PassTelemetry(None, SimpleNamespace(type="cpu"), True)
        batches = [batch_timing(), batch_timing(1)]
        for batch in monitor.iter_loader(batches):
            monitor.begin_compute()
            monitor.end_compute()
        report = monitor.finish(expected_samples=3, expected_batches=2)
        self.assertIsNone(report["gpu_optimization_seconds"])
        self.assertEqual(report["gpu_compute_timing_status"], "NOT_APPLICABLE_CPU")
        self.assertEqual(report["sampling_measured_samples"], 3)
        self.assertAlmostEqual(report["sampling_worker_seconds_sum"], 0.6)
        self.assertGreaterEqual(report["loader_wait_seconds"], 0)

    def test_measurement_coverage_mismatch_is_fatal(self):
        monitor = telemetry.PassTelemetry(None, SimpleNamespace(type="cpu"), True)
        list(monitor.iter_loader([batch_timing()]))
        with self.assertRaisesRegex(telemetry.TelemetryError, "every original"):
            monitor.finish(expected_samples=3, expected_batches=1)
        missing = telemetry.PassTelemetry(None, SimpleNamespace(type="cpu"), True)
        with self.assertRaisesRegex(telemetry.TelemetryError, "no actual worker"):
            list(missing.iter_loader([SimpleNamespace(sample_count=1)]))

    def test_previous_batch_is_released_before_next_loader_fetch(self):
        class ReferencedBatch:
            pass

        owner = self

        class LifetimeLoader:
            def __init__(self):
                self.calls = 0
                self.reference = None

            def __len__(self):
                return 1

            def __iter__(self):
                return self

            def __next__(self):
                self.calls += 1
                if self.calls == 1:
                    batch = ReferencedBatch()
                    batch.__dict__.update(batch_timing().__dict__)
                    self.reference = weakref.ref(batch)
                    return batch
                owner.assertIsNone(self.reference(), "Telemetry retained the previous graph batch during the next fetch")
                raise StopIteration

        loader = LifetimeLoader()
        monitor = telemetry.PassTelemetry(None, SimpleNamespace(type="cpu"), True)
        iterator = monitor.iter_loader(loader)
        batch = next(iterator)
        self.assertIs(loader.reference(), batch)
        del batch  # Mirrors the original epoch_pass release before fetching.
        with self.assertRaises(StopIteration):
            next(iterator)
        self.assertEqual(loader.calls, 2)
        self.assertIsNone(loader.reference())
        monitor.finish(expected_samples=2, expected_batches=1)

    def test_fake_cuda_events_bookkeeping_has_no_sync_or_gpu_quality_claim(self):
        records = []

        class FakeEvent:
            def __init__(self, *, enable_timing):
                records.append(("create", enable_timing))

            def record(self, stream):
                records.append(("record", stream))

            def elapsed_time(self, end):
                records.append(("elapsed", end))
                return 2.5

        fake_cuda = SimpleNamespace(Event=FakeEvent, current_stream=lambda device: "UNIT_fake_stream")
        monitor = telemetry.PassTelemetry(SimpleNamespace(cuda=fake_cuda), SimpleNamespace(type="cuda"), True)
        for batch in monitor.iter_loader([batch_timing()]):
            monitor.begin_compute()
            monitor.end_compute()
        self.assertFalse(any(item[0] == "elapsed" for item in records))
        report = monitor.finish(expected_samples=2, expected_batches=1)
        self.assertEqual(report["gpu_optimization_seconds"], 0.0025)
        self.assertEqual([item[0] for item in records], ["create", "create", "record", "record", "elapsed"])


class EpochTelemetryOutputMetadataUnits(unittest.TestCase):
    def setUp(self):
        (ROOT / "work").mkdir(exist_ok=True)
        # Python 3.13's Windows tempfile mode=0o700 may fail with PermissionError
        # and retry 300,000 times in a restricted host. A fresh explicit mkdir
        # fails immediately, without changing ACLs or broadening permissions.
        self.output = ROOT / "work" / ("v1_epoch_telemetry_METADATA_UNIT_" + uuid.uuid4().hex)
        self.output.mkdir()

    def tearDown(self):
        target = self.output.resolve(strict=True)
        if target.parent != (ROOT / "work").resolve() or not target.name.startswith("v1_epoch_telemetry_METADATA_UNIT_"):
            raise AssertionError("Refusing cleanup outside the exact task-owned metadata UNIT directory")
        shutil.rmtree(target)

    def test_exclusive_jsonl_preserves_original_payload_and_records_save_scope(self):
        session = telemetry.EpochTelemetry(self.output, {"UNIT": "metadata_only"}, {"format": telemetry.FORMAT})
        original_payload = {"epoch": 2, "train": {"mrr": 0.5, "cuda_peak_allocated_bytes": 123},
                            "validation": {"mrr": 0.6, "cuda_peak_allocated_bytes": 234}, "best_checkpoint_updated": False}
        snapshot = copy.deepcopy(original_payload)
        printed = []
        saved = []
        session.begin_epoch(2)
        session.save_checkpoint(lambda payload, path: saved.append((payload, path)), {"original": True}, self.output / "last.pt")
        session.report(lambda name, payload: printed.append((name, payload)), "EpochPostRun", original_payload)
        session.save_checkpoint(lambda payload, path: saved.append((payload, path)), {"original_final": True}, self.output / "best.pt")
        session.report(lambda name, payload: printed.append((name, payload)), "PostRun", {"full_training_complete": False})
        session.close()
        session.close()
        self.assertEqual(original_payload, snapshot)
        rows = [json.loads(line) for line in session.record_path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual([row["event"] for row in rows], ["TelemetryInstalled", "EpochPostRun", "PostRun"])
        self.assertEqual(rows[1]["train"], snapshot["train"])
        self.assertEqual(rows[1]["validation"], snapshot["validation"])
        self.assertEqual(rows[1]["telemetry"]["epoch_peak_vram_allocated_bytes"], 234)
        self.assertEqual(len(rows[1]["telemetry"]["checkpoint_saves"]), 1)
        self.assertEqual(len(rows[2]["telemetry"]["finalization_checkpoint_saves"]), 1)
        second = telemetry.EpochTelemetry(self.output, {}, {})
        second.close()
        self.assertNotEqual(second.record_path, session.record_path)
        self.assertEqual(len(saved), 2)

    def test_wrong_epoch_and_nonfinite_metadata_are_rejected(self):
        with self.assertRaises(ValueError):
            telemetry.EpochTelemetry(self.output, {"bad": float("nan")}, {})
        session = telemetry.EpochTelemetry(self.output, {}, {})
        try:
            session.begin_epoch(1)
            with self.assertRaisesRegex(telemetry.TelemetryError, "disagrees"):
                session.report(lambda *args: None, "EpochPostRun", {"epoch": 2})
        finally:
            session.close()

    def test_install_fails_before_output_on_nonpinned_pipeline(self):
        wrong = self.output / "wrong_pipeline.py"
        wrong.write_text("def run_train(args):\n    return None\n", encoding="utf-8")
        module = SimpleNamespace(__file__=str(wrong), run_train=lambda args: None)
        with self.assertRaisesRegex(telemetry.TelemetryError, "file hash mismatch"):
            telemetry.install(module, self.output / "uncreated", metadata={})
        self.assertFalse((self.output / "uncreated").exists())

    def test_install_compiles_in_memory_without_source_rewrite_or_training(self):
        raw, _ = archived_run_train_source()
        path = self.output / "pipeline.py"
        path.write_bytes(raw)
        tree = ast.parse(raw.decode("utf-8"))
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run_train")
        module = ModuleType("UNIT_unimported_pipeline")
        module.__file__ = str(path)
        isolated = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), function], type_ignores=[])
        ast.fix_missing_locations(isolated)
        exec(compile(isolated, str(path), "exec"), module.__dict__)
        original = module.run_train
        session = telemetry.install(module, self.output / "records", metadata={"scope": "CPU metadata UNIT only"})
        try:
            self.assertIsNot(module.run_train, original)
            self.assertIs(module.run_train.__wrapped__, original)
            self.assertIs(module.run_train.__globals__, module.__dict__)
            self.assertEqual(path.read_bytes(), raw)
            self.assertEqual(session.identity["run_train_source_sha256"], telemetry.RUN_TRAIN_SHA256)
            with self.assertRaisesRegex(telemetry.TelemetryError, "already installed"):
                telemetry.install(module, self.output / "records", metadata={})
        finally:
            session.close()


if __name__ == "__main__":
    unittest.main()
