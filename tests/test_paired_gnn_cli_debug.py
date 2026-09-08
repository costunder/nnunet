"""DEBUG CLI contracts only; no real dataset, GNN training, or GPU execution.

Exercise the production command builder against both real child parsers. A
recording runner replaces subprocess execution, not argument validation.
"""
from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import unittest
from unittest import mock

from hiercp import pipeline
from tools import causality, paired_benchmark as paired


ROOT = Path(__file__).resolve().parents[1]


class PairedGnnCliDebugTests(unittest.TestCase):
    def setUp(self):
        root = ROOT / "DEBUG command paths with spaces"
        self.layout = paired.Layout(
            project=root / "code",
            medical=root / "Medical",
            data=root / "Medical/Data",
            benchmark=root / "work/paired",
            source_work=root / "work/source",
            train_config=root / "recovery/train_config.json",
            nnunet_config=root / "config/nnunet.json",
            outer_splits=root / "outer_splits.json",
            profiles_csv=root / "profiles.csv",
            nnroot=root / "nnunetv2",
            raw=root / "nnunetv2/nnUNet_raw",
            preprocessed=root / "nnunetv2/nnUNet_preprocessed",
            results=root / "nnunetv2/nnUNet_results",
            logs=root / "logs",
        )
        self.config = json.loads((ROOT / "config/train.json").read_text(encoding="utf-8"))

    def commands(self, fold=0, overwrite=False):
        with mock.patch.object(paired, "run_command") as runner:
            paired.gnn_train(self.layout, fold, self.config, "cuda:0", overwrite, True)
        self.assertEqual(runner.call_count, 2)
        return runner.call_args_list

    def test_real_child_parsers_accept_commands_for_each_fold(self):
        for fold in range(5):
            for overwrite in (False, True):
                with self.subTest(fold=fold, overwrite=overwrite):
                    train_call, audit_call = self.commands(fold, overwrite)
                    train_argv = train_call.args[0]
                    audit_argv = audit_call.args[0]
                    self.assertEqual(train_argv[1:4], ["-m", "hiercp.pipeline", "train"])
                    self.assertEqual(audit_argv[1:3], ["-m", "tools.causality"])
                    train = pipeline.build_parser().parse_args(train_argv[3:])
                    audit = causality.build_parser().parse_args(audit_argv[3:])
                    paths = paired.gnn_paths(self.layout, fold)
                    for argv, args in ((train_argv, train), (audit_argv, audit)):
                        flags = [value for value in argv if value.startswith("--")]
                        self.assertEqual(len(flags), len(set(flags)), "Duplicate child CLI option")
                        self.assertEqual(args.cache_dir, str(paths["graphs"]))
                        self.assertEqual(args.checkpoint, str(paths["model"]))
                        self.assertEqual(args.prototype_bank, str(paths["prototype"]))
                        self.assertEqual(args.run_mode, "benchmark")
                        self.assertEqual(args.device, "cuda:0")
                        self.assertEqual(args.seed, int(self.config["seed"]) + fold)
                        self.assertEqual(args.overwrite, overwrite)
                    self.assertEqual(train.config, str(self.layout.train_config))
                    self.assertEqual(train.epochs, int(self.config["training"]["epochs"]))
                    self.assertEqual(audit.output, str(paths["causality"]))
                    self.assertEqual(audit.split, "val")
                    self.assertTrue(audit.strict)
                    self.assertEqual(audit.max_batches, 0)
                    self.assertIsNone(audit.batch_size)
                    self.assertIsNone(audit.num_workers)
                    for call in (train_call, audit_call):
                        self.assertEqual(call.kwargs, {
                            "cwd": self.layout.project, "log": paths["log"], "dry_run": True,
                        })

    def test_original_missing_audit_options_are_rejected_by_real_parser(self):
        audit_argv = self.commands()[1].args[0][3:]
        for option in ("--prototype-bank", "--run-mode"):
            with self.subTest(option=option):
                malformed = list(audit_argv)
                offset = malformed.index(option)
                del malformed[offset:offset + 2]
                error = io.StringIO()
                with contextlib.redirect_stderr(error), self.assertRaises(SystemExit) as failure:
                    causality.build_parser().parse_args(malformed)
                self.assertEqual(failure.exception.code, 2)
                self.assertIn(option, error.getvalue())

    def test_failed_training_does_not_launch_audit(self):
        with mock.patch.object(paired, "run_command", side_effect=paired.BenchmarkError("DEBUG training failure")) as runner:
            with self.assertRaisesRegex(paired.BenchmarkError, "DEBUG training failure"):
                paired.gnn_train(self.layout, 0, self.config, "cuda:0", False, True)
        self.assertEqual(runner.call_count, 1)
        self.assertEqual(runner.call_args.args[0][2], "hiercp.pipeline")

    def test_failed_audit_propagates_without_forced_overwrite_or_extra_commands(self):
        with mock.patch.object(paired, "run_command", side_effect=[None, paired.BenchmarkError("DEBUG audit failure")]) as runner:
            with self.assertRaisesRegex(paired.BenchmarkError, "DEBUG audit failure"):
                paired.gnn_train(self.layout, 0, self.config, "cuda:0", False, True)
        self.assertEqual(runner.call_count, 2)
        for call in runner.call_args_list:
            self.assertNotIn("--overwrite", call.args[0])


if __name__ == "__main__":
    unittest.main()
