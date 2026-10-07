"""UNIT sealed-argument restoration; synthetic metadata, no neural/CT run."""
from __future__ import annotations

import copy
from contextlib import redirect_stderr
from io import StringIO
import importlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from hiercp_v1x import comparison_experiment, u_bridge_experiment
from tools import resume_comparison_cached as runner


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf8")


def reseal(path, manifest):
    manifest = copy.deepcopy(manifest)
    manifest.pop("sha256", None)
    manifest["sha256"] = u_bridge_experiment.digest(manifest)
    write_json(path, manifest)
    return manifest


def unit_fixture(base, *, family="u_bridge", debug=False, external_data=False):
    root = base / "experiment"
    root.mkdir()
    repository = base / "UNIT_fake_helpers"
    repository.mkdir()
    module = u_bridge_experiment if family == "u_bridge" else comparison_experiment
    helpers = {}
    for name in module.FILES:
        path = repository / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("UNIT metadata-only source-binding fixture: " + name, encoding="utf8")
        helpers[name] = u_bridge_experiment.sha(path)
    for name in ("tools/resume_comparison_cached.py", "hiercp_v1x/host_memory.py",
                 "hiercp_v1x/preparation_reuse.py", "hiercp_v1x/comparison_execution.py",
                 "hiercp_v1x/comparison_runtime.py", "hiercp_v1x/comparison_progress.py",
                 "hiercp_v1x/comparison_inputs.py", "hiercp_v1x/comparison_views.py",
                 "hiercp_v1x/comparison_data_timing.py", "hiercp_v1x/comparison_preparation.py",
                 "hiercp_v1x/comparison_source_cache.py", "hiercp_v1x/comparison_upper_cache.py",
                 "hiercp_v1x/comparison_sample_cache.py", "hiercp_v1x/comparison_gpu_policy.py",
                 "hiercp_v1x/comparison_gpu_runtime.py"):
        path = repository / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("UNIT additive execution provenance fixture: " + name, encoding="utf8")
    inventory = base / "UNIT_inventory.json"
    write_json(inventory, dict(unit=True, no_actual_CT_or_prediction=True))
    baseline = base / "UNIT_preserved_baseline"
    baseline.mkdir()
    data = base / "UNIT_external_prepared_data" if external_data else root / "data"
    data.mkdir()
    manifest = dict(format=module.FORMAT, debug=debug, epochs=2 if debug else 40,
        helpers=helpers, baseline=dict(inventory_sha256=u_bridge_experiment.sha(inventory),
                                     baseline=str(baseline)),
        workers=16, explicit_batch_candidates=[1, 2, 4, 8, 16, 32],
        cuda_gib=40., rss_gib=192., resident_gib=128., validation_local_chunk=8,
        prepared_data_root=str(data),
        config=dict(model=dict(hidden_dim=128, heads=4, local_layers=3, patient_layers=2,
                               prototype_layers=2),
                    graph=dict(adaptive_roi_margin_mm=10., context_outer_radius_mm=10.),
                    candidate_contract=dict(train_candidates=8, eval_candidates=129)))
    manifest = reseal(root / "experiment.json", manifest)
    arguments = SimpleNamespace(gpu=3, arm="selected", experiment=root, inventory=inventory,
        cache_sources=[base / "UNIT_other_arm_data"], debug_fixture=None, debug_config=None,
        debug_source=None, debug_bank=None)
    if debug:
        for name in ("fixture", "config", "source", "bank"):
            value = base / ("UNIT_original_debug_" + name)
            value.mkdir() if name in ("fixture", "source") else value.write_text("UNIT metadata-only input")
            setattr(arguments, "debug_" + name, value)
    return root, repository, inventory, manifest, arguments


def snapshot(root):
    return {p.relative_to(root).as_posix(): (p.stat().st_size, u_bridge_experiment.sha(p))
            for p in root.rglob("*") if p.is_file()}


class CachedResumeArgumentsTests(unittest.TestCase):
    def directory(self):
        return TemporaryDirectory(prefix="UNIT_cached_resume_", dir=Path(__file__).resolve().parents[1])

    def test_restore_every_sealed_production_execution_setting_without_shrinking(self):
        with self.directory() as directory:
            base = Path(directory)
            root, repository, inventory, manifest, supplied = unit_fixture(base)
            before = snapshot(base)
            with patch.object(runner, "ROOT", repository):
                actual, family, restored, namespace = runner.sealed_arguments(supplied)
            self.assertEqual(actual, manifest)
            self.assertEqual(family, "u_bridge")
            self.assertEqual(restored.gpu, 3)
            self.assertEqual(restored.arm, "selected")
            self.assertEqual(restored.experiment, root.resolve())
            self.assertEqual(restored.inventory, inventory)
            self.assertEqual(restored.baseline, Path(manifest["baseline"]["baseline"]))
            self.assertEqual(restored.batch_candidates, [1, 2, 4, 8, 16, 32])
            self.assertEqual(restored.workers, 16)
            self.assertEqual((restored.cuda_gib, restored.rss_gib, restored.resident_gib), (40., 192., 128.))
            self.assertEqual(restored.validation_local_chunk, 8)
            self.assertEqual(namespace, root / "data")
            self.assertIsNone(restored.prepared_data)
            self.assertFalse(restored.debug)
            self.assertIsNone(restored.debug_epochs)
            self.assertIsNone(restored.debug_pause_updates)
            self.assertEqual(actual["epochs"], 40)
            self.assertEqual(actual["config"]["candidate_contract"], dict(train_candidates=8, eval_candidates=129))
            self.assertEqual(snapshot(base), before)

    def test_restore_original_external_cache_namespace_without_relocating_it(self):
        with self.directory() as directory:
            base = Path(directory)
            root, repository, _, manifest, supplied = unit_fixture(base, external_data=True)
            before = (root / "experiment.json").read_bytes()
            with patch.object(runner, "ROOT", repository):
                _, _, restored, namespace = runner.sealed_arguments(supplied)
            self.assertEqual(namespace, Path(manifest["prepared_data_root"]))
            self.assertEqual(restored.prepared_data, namespace)
            self.assertEqual((root / "experiment.json").read_bytes(), before)

    def test_old_contract_without_prepared_data_field_keeps_own_data_namespace(self):
        with self.directory() as directory:
            base = Path(directory)
            root, repository, _, manifest, supplied = unit_fixture(base)
            manifest.pop("prepared_data_root")
            reseal(root / "experiment.json", manifest)
            with patch.object(runner, "ROOT", repository):
                _, _, restored, namespace = runner.sealed_arguments(supplied)
            self.assertEqual(namespace, root / "data")
            self.assertIsNone(restored.prepared_data)

    def test_v19_accepts_each_declared_arm_with_identical_locked_settings(self):
        with self.directory() as directory:
            base = Path(directory)
            _, repository, _, manifest, supplied = unit_fixture(base, family="comparison")
            with patch.object(runner, "ROOT", repository):
                for arm in comparison_experiment.ARMS:
                    with self.subTest(arm=arm):
                        supplied.arm = arm
                        actual, family, restored, _ = runner.sealed_arguments(supplied)
                        self.assertEqual(family, "comparison")
                        self.assertEqual(restored.arm, arm)
                        self.assertEqual(restored.batch_candidates, manifest["explicit_batch_candidates"])
                        self.assertEqual(restored.workers, manifest["workers"])
                        self.assertEqual(actual["epochs"], 40)

    def test_debug_requires_all_four_explicit_original_inputs(self):
        with self.directory() as directory:
            base = Path(directory)
            _, repository, _, _, supplied = unit_fixture(base, debug=True)
            with patch.object(runner, "ROOT", repository):
                for name in ("fixture", "config", "source", "bank"):
                    missing = copy.copy(supplied)
                    setattr(missing, "debug_" + name, None)
                    with self.subTest(missing=name), self.assertRaisesRegex(ValueError, "Explicit original actual-CT DEBUG"):
                        runner.sealed_arguments(missing)
                _, _, restored, _ = runner.sealed_arguments(supplied)
            self.assertTrue(restored.debug)
            self.assertEqual(restored.debug_epochs, 2)
            self.assertIsNone(restored.baseline)
            for name in ("fixture", "config", "source", "bank"):
                self.assertEqual(getattr(restored, "debug_" + name), getattr(supplied, "debug_" + name))

    def test_debug_inputs_cannot_enter_production(self):
        with self.directory() as directory:
            base = Path(directory)
            _, repository, _, _, supplied = unit_fixture(base)
            supplied.debug_bank = base / "UNIT_unwanted_debug_bank"
            with patch.object(runner, "ROOT", repository), self.assertRaisesRegex(ValueError, "cannot enter production"):
                runner.sealed_arguments(supplied)

    def test_contract_digest_tamper_rejected_without_manifest_write(self):
        with self.directory() as directory:
            base = Path(directory)
            root, repository, _, manifest, supplied = unit_fixture(base)
            manifest["workers"] = 2
            write_json(root / "experiment.json", manifest)
            before = snapshot(base)
            with patch.object(runner, "ROOT", repository), self.assertRaisesRegex(ValueError, "contract changed"):
                runner.sealed_arguments(supplied)
            self.assertEqual(snapshot(base), before)

    def test_changed_helper_bytes_or_helper_inventory_are_rejected(self):
        with self.directory() as directory:
            base = Path(directory)
            root, repository, _, manifest, supplied = unit_fixture(base)
            helper = next(iter(manifest["helpers"]))
            path = repository / helper
            original = path.read_bytes()
            path.write_bytes(original + b"\nUNIT changed source")
            with patch.object(runner, "ROOT", repository), self.assertRaisesRegex(ValueError, "source changed"):
                runner.sealed_arguments(supplied)
            path.write_bytes(original)
            manifest["helpers"].pop(helper)
            reseal(root / "experiment.json", manifest)
            with patch.object(runner, "ROOT", repository), self.assertRaisesRegex(ValueError, "helper inventory changed"):
                runner.sealed_arguments(supplied)

    def test_changed_original128_inventory_is_rejected(self):
        with self.directory() as directory:
            base = Path(directory)
            _, repository, inventory, _, supplied = unit_fixture(base)
            inventory.write_text("UNIT changed candidate inventory")
            with patch.object(runner, "ROOT", repository), self.assertRaisesRegex(ValueError, "Original128 candidate inventory"):
                runner.sealed_arguments(supplied)

    def test_v18_rejects_v19_only_arms_and_unknown_family(self):
        with self.directory() as directory:
            base = Path(directory)
            root, repository, _, manifest, supplied = unit_fixture(base)
            with patch.object(runner, "ROOT", repository):
                for arm in ("native_fixed", "native_listwise"):
                    supplied.arm = arm
                    with self.subTest(arm=arm), self.assertRaisesRegex(ValueError, "Arm does not belong"):
                        runner.sealed_arguments(supplied)
                supplied.arm = "selected"
                manifest["format"] = "UNIT_unrecognized_family"
                reseal(root / "experiment.json", manifest)
                with self.assertRaisesRegex(ValueError, "Arm does not belong"):
                    runner.sealed_arguments(supplied)

    def test_shortened_production_epoch_count_rejected_even_when_sealed(self):
        with self.directory() as directory:
            base = Path(directory)
            root, repository, _, manifest, supplied = unit_fixture(base)
            manifest["epochs"] = 39
            reseal(root / "experiment.json", manifest)
            with patch.object(runner, "ROOT", repository), self.assertRaisesRegex(ValueError, "Original40epoch"):
                runner.sealed_arguments(supplied)

    def test_programmatic_unknown_arm_is_rejected_for_v19(self):
        with self.directory() as directory:
            base = Path(directory)
            _, repository, _, _, supplied = unit_fixture(base, family="comparison")
            supplied.arm = "UNIT_unknown_arm"
            with patch.object(runner, "ROOT", repository), self.assertRaisesRegex(ValueError, "arm"):
                runner.sealed_arguments(supplied)

    def test_controller_failure_restores_budget_provider_and_fields_without_manifest_mutation(self):
        from hiercp_v1x import u_bridge_fields
        for family, class_name, controller_name in (
                ("u_bridge", "UBridgeData", "tools.run_v18_u_bridge"),
                ("comparison", "ComparisonData", "tools.run_v19_comparison")):
            with self.subTest(family=family), self.directory() as directory:
                base = Path(directory)
                root, repository, _, _, supplied = unit_fixture(base, family=family)
                source_cache = supplied.cache_sources[0]
                source_cache.mkdir()
                (source_cache / "UNIT_completed_input.txt").write_text("UNIT preserved read-only source fixture")
                source_before = snapshot(source_cache)
                manifest_before = (root / "experiment.json").read_bytes()
                experiment_module = importlib.import_module("hiercp_v1x." + family + "_experiment")
                data_module = importlib.import_module("hiercp_v1x." + family + "_data")
                controller = importlib.import_module(controller_name)
                original_budget = experiment_module.Budget
                original_provider = getattr(data_module, class_name)
                original_fields = u_bridge_fields.cached_fields
                def failing_controller(args):
                    self.assertIsNot(experiment_module.Budget, original_budget)
                    self.assertIsNot(getattr(data_module, class_name), original_provider)
                    self.assertTrue(issubclass(getattr(data_module, class_name), original_provider))
                    self.assertIsNot(u_bridge_fields.cached_fields, original_fields)
                    self.assertEqual(args.batch_candidates, [1, 2, 4, 8, 16, 32])
                    self.assertEqual(args.workers, 16)
                    raise RuntimeError("UNIT injected controller failure before model execution")
                with patch.object(runner, "ROOT", repository), \
                        patch("tools.local_cnn_device.select") as selected_gpu, \
                        patch.object(controller, "run", side_effect=failing_controller), \
                        self.assertRaisesRegex(RuntimeError, "UNIT injected controller failure"):
                    runner.run(supplied)
                selected_gpu.assert_called_once_with(3)
                self.assertIs(experiment_module.Budget, original_budget)
                self.assertIs(getattr(data_module, class_name), original_provider)
                self.assertIs(u_bridge_fields.cached_fields, original_fields)
                self.assertEqual((root / "experiment.json").read_bytes(), manifest_before)
                self.assertEqual(snapshot(source_cache), source_before)
                self.assertFalse((root / ".pipeline.lock").exists())
                self.assertFalse((root / "data/.data.lock").exists())
                self.assertEqual(len(list((root / "execution_overrides").glob("*.json"))), 1)

    def test_independent_route_failure_restores_registered_provider_and_field_function(self):
        from hiercp_v1x import host_memory, u_bridge_fields, u_bridge_data
        from tools import run_v18_independent
        with self.directory() as directory:
            base = Path(directory)
            root, repository, _, manifest, supplied = unit_fixture(base)
            supplied.cache_sources[0].mkdir()
            write_json(root / "continuation.json", dict(request=dict(
                source_root=str(base / "UNIT_original_source"), arm="selected",
                contract_sha256=manifest["sha256"], data_root=str(root / "data"))))
            original_registered = host_memory.pressure_aware_provider
            original_fields = u_bridge_fields.cached_fields
            original_data = u_bridge_data.UBridgeData
            manifest_before = (root / "experiment.json").read_bytes()
            def failing_independent(args):
                self.assertIsNot(host_memory.pressure_aware_provider, original_registered)
                self.assertIsNot(u_bridge_fields.cached_fields, original_fields)
                self.assertIs(u_bridge_data.UBridgeData, original_data)
                wrapped = host_memory.pressure_aware_provider(original_data)
                self.assertTrue(issubclass(wrapped, original_data))
                self.assertEqual(args.arm, "selected")
                self.assertEqual(args.source_experiment, base / "UNIT_original_source")
                raise RuntimeError("UNIT independent controller failure; no model execution")
            with patch.object(runner, "ROOT", repository), patch("tools.local_cnn_device.select"), \
                    patch.object(run_v18_independent, "run", side_effect=failing_independent), \
                    self.assertRaisesRegex(RuntimeError, "UNIT independent controller failure"):
                runner.run(supplied)
            self.assertIs(host_memory.pressure_aware_provider, original_registered)
            self.assertIs(u_bridge_fields.cached_fields, original_fields)
            self.assertIs(u_bridge_data.UBridgeData, original_data)
            self.assertEqual((root / "experiment.json").read_bytes(), manifest_before)

    def test_cli_does_not_accept_unknown_arm_or_ad_hoc_scale_changes(self):
        common = ["--gpu", "3", "--arm", "native", "--experiment", "UNIT_root", "--inventory", "UNIT_inventory",
                  "--cache-sources", "UNIT_other_data"]
        for extra in (["--workers", "2"], ["--batch-candidates", "1"], ["--cuda-gib", "8"],
                      ["--rss-gib", "256"], ["--resident-gib", "16"], ["--validation-local-chunk", "1"]):
            with self.subTest(extra=extra), redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                runner.parse(common + extra)
        unknown = common.copy()
        unknown[3] = "UNIT_unknown_arm"
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            runner.parse(unknown)


if __name__ == "__main__":
    unittest.main()
