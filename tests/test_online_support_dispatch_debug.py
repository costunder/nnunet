"""DEBUG dispatch fixtures only: no real split, preparation, GPU or training."""
import contextlib
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from tools import online_cp_benchmark as online


class OnlineSupportDispatchDebugTests(unittest.TestCase):
    def layout(self, root):
        project = root / "DEBUG checkout with spaces"
        medical = root / "DEBUG medical"
        paired = project / "work/nested/feedback_recovered/paired"
        return SimpleNamespace(
            project=project, medical=medical, paired=paired,
            train_config=paired.parent / "recovery/train_config.json",
            nnunet_config=project / "config/nnunet.json",
            outer_splits=paired / "outer_splits.json",
            gnn=lambda fold: paired / f"folds/fold_{fold}/gnn",
        )

    @staticmethod
    def write_split(layout):
        layout.outer_splits.parent.mkdir(parents=True, exist_ok=True)
        layout.outer_splits.write_text("DEBUG split placeholder; outer_split verifier is mocked", encoding="utf-8")

    @staticmethod
    def publish_debug_assets(layout, fold):
        for name in ("split.json", "prototype.pt", "model.pt", "model.last.pt", "causality.json",
                     "causality.json.preflight.json", "graphs/complete.json"):
            path = layout.gnn(fold) / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("DEBUG bytes; never loaded as a model or graph", encoding="utf-8")

    def check_command(self, argv, layout, stage, fold, device=None):
        self.assertEqual(argv[:4], [sys.executable, "-m", "tools.paired_benchmark", stage])
        expected = {"--project-root": layout.project, "--medical-root": layout.medical,
                    "--work": layout.paired, "--train-config": layout.train_config,
                    "--nnunet-config": layout.nnunet_config, "--outer-fold": fold}
        for flag, value in expected.items():
            self.assertEqual(argv[argv.index(flag) + 1], str(value))
        self.assertNotIn(str(layout.medical / "pairedcp"), argv)
        self.assertNotIn("--overwrite", argv)
        self.assertNotIn("--resume", argv)
        if device is None:
            self.assertNotIn("--device", argv)
        else:
            self.assertEqual(argv[argv.index("--device") + 1], device)

    def test_missing_split_uses_exact_checkout_work_and_configs_without_legacy_wrapper(self):
        with tempfile.TemporaryDirectory(prefix="debug_online_split_dispatch_") as tmp:
            layout = self.layout(Path(tmp))
            def runner(argv, **kwargs):
                self.check_command(argv, layout, "split", 2)
                self.assertEqual(kwargs, {"cwd": layout.project, "dry_run": False})
                self.write_split(layout)
            with mock.patch.object(online, "run_command", side_effect=runner) as run, \
                 mock.patch.object(online, "outer_split", return_value={"DEBUG": True}) as verify:
                self.assertTrue(online.ensure_outer_split(layout, 2, False))
            self.assertEqual(run.call_count, 1)
            verify.assert_called_once_with(layout, 2)
            self.assertFalse((layout.medical / "pairedcp").exists())

    def test_existing_split_is_verified_without_recreation(self):
        with tempfile.TemporaryDirectory(prefix="debug_online_split_reuse_") as tmp:
            layout = self.layout(Path(tmp)); self.write_split(layout)
            before = layout.outer_splits.read_bytes()
            with mock.patch.object(online, "run_command") as run, \
                 mock.patch.object(online, "outer_split", return_value={"DEBUG": True}) as verify:
                self.assertTrue(online.ensure_outer_split(layout, 4, False))
            self.assertEqual(run.call_count, 0)
            verify.assert_called_once_with(layout, 4)
            self.assertEqual(layout.outer_splits.read_bytes(), before)

    def test_split_dry_run_has_no_publication_and_preserves_requested_device(self):
        with tempfile.TemporaryDirectory(prefix="debug_online_split_dryrun_") as tmp:
            layout = self.layout(Path(tmp))
            with mock.patch.object(online, "run_command") as run, \
                 mock.patch.object(online, "outer_split") as verify, contextlib.redirect_stdout(io.StringIO()):
                self.assertFalse(online.ensure_outer_split(layout, 1, True, device="cuda:3"))
            self.check_command(run.call_args.args[0], layout, "split", 1, "cuda:3")
            self.assertTrue(run.call_args.kwargs["dry_run"])
            self.assertEqual(verify.call_count, 0)
            self.assertFalse(layout.outer_splits.exists())

    def test_missing_support_generates_in_requested_recovery_root_with_one_train_config(self):
        with tempfile.TemporaryDirectory(prefix="debug_online_support_dispatch_") as tmp:
            layout = self.layout(Path(tmp)); self.write_split(layout)
            legacy = layout.project / "work/paired_benchmark/DEBUG_old_result"
            legacy.parent.mkdir(parents=True); legacy.write_bytes(b"preserve legacy")
            calls = []
            def runner(argv, **kwargs):
                calls.append((list(argv), kwargs))
                if argv[3] == "gnn-train":
                    self.publish_debug_assets(layout, 2)
            with mock.patch.object(online, "run_command", side_effect=runner), \
                 mock.patch.object(online, "outer_split", return_value={"DEBUG": True}), \
                 mock.patch.object(online, "_verified_gnn_causality", return_value={"DEBUG": True}) as verify:
                self.assertTrue(online.ensure_support_assets(layout, 2, "cuda:3", False))
            self.assertEqual(len(calls), 2)
            for (argv, kwargs), stage in zip(calls, ("gnn-prepare", "gnn-train")):
                self.check_command(argv, layout, stage, 2, "cuda:3")
                self.assertEqual(kwargs["cwd"], layout.project)
                self.assertFalse(kwargs["dry_run"])
            self.assertEqual(calls[1][1]["env"]["PYTORCH_CUDA_ALLOC_CONF"], "expandable_segments:True")
            verify.assert_called_once_with(layout, 2)
            self.assertEqual(legacy.read_bytes(), b"preserve legacy")

    def test_missing_support_and_split_dry_run_does_not_claim_ready(self):
        with tempfile.TemporaryDirectory(prefix="debug_online_support_split_dryrun_") as tmp:
            layout = self.layout(Path(tmp))
            with mock.patch.object(online, "run_command") as run, \
                 mock.patch.object(online, "_verified_gnn_causality") as verify, contextlib.redirect_stdout(io.StringIO()):
                self.assertFalse(online.ensure_support_assets(layout, 0, "cuda:4", True))
            self.assertEqual(run.call_count, 1)
            self.check_command(run.call_args.args[0], layout, "split", 0, "cuda:4")
            self.assertEqual(verify.call_count, 0)
            self.assertFalse(layout.project.exists())

    def test_support_dry_run_records_only_exact_prepare_train_commands(self):
        with tempfile.TemporaryDirectory(prefix="debug_online_support_dryrun_") as tmp:
            layout = self.layout(Path(tmp)); self.write_split(layout)
            with mock.patch.object(online, "run_command") as run, \
                 mock.patch.object(online, "outer_split", return_value={"DEBUG": True}), \
                 mock.patch.object(online, "_verified_gnn_causality") as verify, contextlib.redirect_stdout(io.StringIO()):
                self.assertFalse(online.ensure_support_assets(layout, 3, "cuda:0", True))
            self.assertEqual(run.call_count, 2)
            for call, stage in zip(run.call_args_list, ("gnn-prepare", "gnn-train")):
                self.check_command(call.args[0], layout, stage, 3, "cuda:0")
                self.assertTrue(call.kwargs["dry_run"])
            self.assertEqual(verify.call_count, 0)
            self.assertFalse(layout.gnn(3).exists())

    def test_valid_existing_support_is_reverified_and_invalid_existing_support_not_overwritten(self):
        for error in (None, online.OnlineBenchmarkError("DEBUG stale causality")):
            with self.subTest(error=error), tempfile.TemporaryDirectory(prefix="debug_online_support_verify_") as tmp:
                layout = self.layout(Path(tmp)); self.write_split(layout); self.publish_debug_assets(layout, 2)
                with mock.patch.object(online, "run_command") as run, \
                     mock.patch.object(online, "outer_split", return_value={"DEBUG": True}), \
                     mock.patch.object(online, "_verified_gnn_causality", side_effect=error) as verify, \
                     contextlib.redirect_stdout(io.StringIO()):
                    if error is None:
                        self.assertTrue(online.ensure_support_assets(layout, 2, "cuda:0", False))
                    else:
                        with self.assertRaisesRegex(online.OnlineBenchmarkError, "stale causality"):
                            online.ensure_support_assets(layout, 2, "cuda:0", False)
                self.assertEqual(run.call_count, 0)
                self.assertEqual(verify.call_count, 1)

    def test_prepare_failure_stops_training_and_unpublished_outputs_are_not_accepted(self):
        for failure in (True, False):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory(prefix="debug_online_support_failure_") as tmp:
                layout = self.layout(Path(tmp)); self.write_split(layout)
                error = subprocess.CalledProcessError(5, ["DEBUG prepare"]) if failure else None
                with mock.patch.object(online, "run_command", side_effect=error) as run, \
                     mock.patch.object(online, "outer_split", return_value={"DEBUG": True}), \
                     mock.patch.object(online, "_verified_gnn_causality") as verify:
                    with self.assertRaises(subprocess.CalledProcessError if failure else online.OnlineBenchmarkError):
                        online.ensure_support_assets(layout, 0, "cuda:0", False)
                self.assertEqual(run.call_count, 1 if failure else 2)
                self.assertEqual(verify.call_count, 0)


if __name__ == "__main__":
    unittest.main()
