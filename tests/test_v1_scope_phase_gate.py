"""Scope DEBUG CLI/static contract units; no CUDA or training is launched."""
from __future__ import annotations

import ast
import contextlib
import io
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools import run_v1_scope_probe
from tools import probe_v1_scope_time


ROOT = Path(__file__).resolve().parents[1]


def arguments():
    return ["scope_UNIT_cli", "--gpu", "3", "--experiment", "UNIT_NATIVE_EXPERIMENT",
            "--output", "UNIT_NEW_DEBUG_OUTPUT", "--margin-mm", "10", "20", "30",
            "--physical-batch", "2", "--workers", "2", "--updates", "1",
            "--cuda-gib", "12", "--rss-gib", "32"]


def qualified(call):
    if isinstance(call, ast.Name):
        return call.id
    if isinstance(call, ast.Attribute):
        return qualified(call.value) + "." + call.attr
    return ""


class ScopePhaseGateUnits(unittest.TestCase):
    def test_scope_adapter_is_bound_to_current_package_for_relative_imports(self):
        adapter, identity = probe_v1_scope_time.current_adapter()
        self.assertEqual(adapter.__package__, 'hiercp_v1x')
        self.assertEqual(Path(identity['path']).resolve(), ROOT / 'hiercp_v1x/bounded_scope.py')
        import hiercp_v1x.contracts
        self.assertEqual(Path(hiercp_v1x.contracts.__file__).resolve(), ROOT / 'hiercp_v1x/contracts.py')

    def test_every_run_defining_argument_is_mandatory_without_hidden_default(self):
        required = ("--gpu", "--experiment", "--output", "--margin-mm", "--physical-batch",
                    "--workers", "--updates", "--cuda-gib", "--rss-gib")
        for option in required:
            values = arguments()
            index = values.index(option)
            end = index + 1
            while end < len(values) and not values[end].startswith("--"):
                end += 1
            del values[index:end]
            with self.subTest(option=option), patch("sys.argv", values), \
                 contextlib.redirect_stderr(io.StringIO()), \
                 patch("hiercp_v1x.scope_inputs.prepare_debug_inputs") as prepare, \
                 patch.object(run_v1_scope_probe.subprocess, "run") as execute:
                with self.assertRaises(SystemExit) as error:
                    run_v1_scope_probe.main()
                self.assertEqual(error.exception.code, 2)
                prepare.assert_not_called()
                execute.assert_not_called()

    def test_single_debug_cost_subprocess_preserves_physical_scope_batch_and_resources(self):
        source, samples = Path("UNIT_SOURCE"), Path("UNIT_SAMPLES")
        with patch("sys.argv", arguments()), contextlib.redirect_stdout(io.StringIO()), \
             patch("hiercp_v1x.scope_inputs.prepare_debug_inputs", return_value=(source, samples)) as prepare, \
             patch.object(run_v1_scope_probe.subprocess, "run", return_value=SimpleNamespace(returncode=0)) as execute:
            run_v1_scope_probe.main()
        prepare.assert_called_once()
        self.assertEqual(prepare.call_args.args[2], 2)
        execute.assert_called_once()
        command = execute.call_args.args[0]
        self.assertEqual(Path(command[2]).name, "probe_v1_scope_time.py")
        self.assertEqual(command[command.index("--margin-mm") + 1:command.index("--physical-batch")],
                         ["10.0", "20.0", "30.0"])
        for option, value in (("--gpu", "3"), ("--physical-batch", "2"), ("--workers", "2"),
                              ("--updates", "1"), ("--cuda-gib", "12.0"), ("--rss-gib", "32.0")):
            self.assertEqual(command[command.index(option) + 1], value)
        self.assertNotIn("--epochs", command)
        self.assertNotIn("--resume", command)
        self.assertEqual(execute.call_args.kwargs["env"]["PYTHONDONTWRITEBYTECODE"], "1")
        self.assertIn("isolated_bytecode_lookup", execute.call_args.kwargs["env"]["PYTHONPYCACHEPREFIX"])

    def test_invalid_debug_update_or_serial_worker_never_starts_measurement(self):
        for option, value in (("--updates", "0"), ("--updates", "5"), ("--workers", "0"), ("--workers", "1")):
            values = arguments()
            values[values.index(option) + 1] = value
            with self.subTest(option=option, value=value), patch("sys.argv", values), \
                 patch("hiercp_v1x.scope_inputs.prepare_debug_inputs") as prepare, \
                 patch.object(run_v1_scope_probe.subprocess, "run") as execute:
                with self.assertRaisesRegex(ValueError, "DEBUG"):
                    run_v1_scope_probe.main()
                prepare.assert_not_called()
                execute.assert_not_called()

    def test_existing_output_error_stops_before_any_child_is_started(self):
        with patch("sys.argv", arguments()), \
             patch("hiercp_v1x.scope_inputs.prepare_debug_inputs", side_effect=FileExistsError("UNIT preserved")), \
             patch.object(run_v1_scope_probe.subprocess, "run") as execute:
            with self.assertRaises(FileExistsError):
                run_v1_scope_probe.main()
            execute.assert_not_called()

    def test_failed_cost_probe_does_not_launch_training_or_automatic_retry(self):
        with patch("sys.argv", arguments()), contextlib.redirect_stdout(io.StringIO()), \
             patch("hiercp_v1x.scope_inputs.prepare_debug_inputs", return_value=(Path("UNIT_SOURCE"), Path("UNIT_SAMPLES"))), \
             patch.object(run_v1_scope_probe.subprocess, "run", return_value=SimpleNamespace(returncode=7)) as execute:
            with self.assertRaisesRegex(RuntimeError, r"failed \(7\).*preserved"):
                run_v1_scope_probe.main()
            execute.assert_called_once()

    def test_runner_and_cost_probe_never_call_training_or_save_checkpoints(self):
        paths = (ROOT / "tools/run_v1_scope_probe.py", ROOT / "tools/probe_v1_scope_time.py")
        forbidden = {"run_train", "execute_stage", "run_experiment", "torch.save", "save_checkpoint",
                     "save_checkpoint_atomic", "save_state_dict", "prepare_generation_inputs"}
        for path in paths:
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    name = qualified(node.func)
                    self.assertNotIn(name, forbidden, f"{path.name}: {name}")
                    self.assertFalse("nnUNetTrainer" in name or "nnunet" in name.lower(), name)
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    names = [x.name for x in node.names]
                    if isinstance(node, ast.ImportFrom) and node.module:
                        names.append(node.module)
                    self.assertFalse(any("nested_execution" in x for x in names),
                                     "Deferred-attribute optimization must not change first scope experiment")
                    self.assertFalse(any("nnunet" in x.lower() for x in names), str(names))
                if isinstance(node, ast.Call):
                    for key in node.keywords:
                        if key.arg in ("production_ready", "quality_verified", "full_training", "full_evaluation"):
                            self.assertIsInstance(key.value, ast.Constant)
                            self.assertIs(key.value.value, False)

    def test_measurement_reports_are_new_files_and_not_production_artifacts(self):
        path = ROOT / "tools/probe_v1_scope_time.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        write = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == "write_new")
        calls = [x for x in ast.walk(write) if isinstance(x, ast.Call) and qualified(x.func).endswith(".open")]
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].args[0].value, "x")
        self.assertIn("no epoch extrapolation", path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
