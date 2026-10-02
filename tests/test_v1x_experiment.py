"""Runtime metadata/integrity UNIT tests, without CT decoding or model execution.

The three empty NIfTI-named files per input kind are filename-only fixtures for
cohort accounting.  No fixture is passed to preparation, a model, or training.
Task-owned files stay under work/v1x_runtime_metadata_UNIT_* for traceability.
"""
import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import uuid
from zipfile import ZipFile

from hiercp_v1x.contracts import ContractError, STAGES, canonical_hash, verify_archive
from hiercp_v1x.experiment import (
    ROOT, bind_launch, command_plan, digest, execute_stage, freeze_execution, initialize,
    load_suite, normalize_split, overlay, owned_lock, prepare_generation_inputs,
)


class RuntimeMetadataUnitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.owned = ROOT / "work" / ("v1x_runtime_metadata_UNIT_" + uuid.uuid4().hex)
        cls.medical = cls.owned / "medical"
        cls.medical.mkdir(parents=True, exist_ok=False)
        for kind in ("image", "labels"):
            directory = cls.medical / "Data" / kind
            directory.mkdir(parents=True, exist_ok=False)
            for case in ("unit_train", "unit_val", "unit_outer"):
                suffix = "_0000.nii.gz" if kind == "image" else ".nii.gz"
                (directory / (case + suffix)).write_bytes(b"")
        cls.split = {"seed": 42, "train": ["unit_train"], "val": ["unit_val"],
                     "outer_validation_excluded": ["unit_outer"]}
        cls.split_path = cls.owned / "split.json"
        cls.split_path.write_text(json.dumps(cls.split), encoding="utf-8")
        cls.suite = cls.owned / "suite"
        cls.manifest = initialize(cls.suite, cls.medical, cls.split_path)

    def setUp(self):
        # Never start subprocesses even if a test assertion fails downstream.
        self.unexpected_subprocess = patch("hiercp_v1x.experiment.subprocess.Popen",
                                          side_effect=AssertionError("UNIT must not launch a model"))
        self.unexpected_subprocess.start()
        self.addCleanup(self.unexpected_subprocess.stop)

    def test_initialize_is_idempotent_and_preserves_results(self):
        marker = self.suite / "results" / "v1.0" / "UNIT_existing_result.txt"
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_bytes(b"owned-unit-result")
        first = digest(self.suite / "manifest.json")
        repeated = initialize(self.suite, self.medical, self.split_path)
        self.assertEqual(repeated, self.manifest)
        self.assertEqual(digest(self.suite / "manifest.json"), first)
        self.assertEqual(marker.read_bytes(), b"owned-unit-result")

    def test_original_archive_and_declared_overlay_are_exact(self):
        proof = verify_archive(ROOT)
        with ZipFile(ROOT / "versions/v1/pipeline_v1_source.zip") as zipped:
            for stage in STAGES:
                for name in zipped.namelist():
                    expected = zipped.read(name) + overlay(stage, name)
                    self.assertEqual((self.suite / "source" / stage / name).read_bytes(), expected)
                helper = self.suite / "source" / stage / "hiercp_v1x/models.py"
                self.assertEqual(helper.read_bytes(), (ROOT / "hiercp_v1x/models.py").read_bytes())
        self.assertEqual(proof["verified_files"], 202)

    def test_snapshot_python_import_closure_and_standalone_model_helper(self):
        source = self.suite / "source" / "v1.3"
        local_roots = {path.name for path in source.iterdir() if path.is_dir()}
        for file in source.rglob("*.py"):
            tree = ast.parse(file.read_text(encoding="utf-8-sig"), filename=str(file))
            for node in ast.walk(tree):
                modules = ([x.name for x in node.names] if isinstance(node, ast.Import)
                           else [node.module] if isinstance(node, ast.ImportFrom) and not node.level and node.module
                           else [])
                for module in modules:
                    if module.split(".")[0] not in local_roots:
                        continue
                    path = source.joinpath(*module.split("."))
                    self.assertTrue(path.with_suffix(".py").is_file() or path.is_dir(),
                                    f"{file.relative_to(source)} imports missing local {module}")
        tree = ast.parse((source / "hiercp_v1x/models.py").read_text(encoding="utf-8"))
        names = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module]
        self.assertFalse(any(x.startswith("hiercp.") or x == "hiercp" for x in names))

    def test_source_tampering_stops_before_launch(self):
        path = self.suite / "source/v1.2/hiercp/model.py"
        original = path.read_bytes()
        try:
            path.write_bytes(original + b"\n# UNIT tampering\n")
            with self.assertRaisesRegex(ContractError, "Snapshot changed"):
                command_plan(self.suite, "v1.2", "train")
        finally:
            path.write_bytes(original)

    def test_unlisted_source_injection_rejected(self):
        path = self.suite / "source/v1.3/UNIT_unlisted_source.py"
        path.write_text("# Metadata UNIT fixture; never imported\n", encoding="utf-8")
        try:
            with self.assertRaisesRegex(ContractError, "Unlisted or missing snapshot"):
                load_suite(self.suite)
        finally:
            path.unlink()  # Only this test's new source fixture.

    def test_config_tampering_stops_before_launch(self):
        path = self.suite / "configs/v1.2.json"
        original = path.read_bytes()
        try:
            config = json.loads(original)
            config["training"]["pairwise_weight"] = 0
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ContractError, "undeclared"):
                command_plan(self.suite, "v1.2", "train")
        finally:
            path.write_bytes(original)

    def test_split_tampering_stops_before_launch(self):
        path = self.suite / "shared/split.json"
        original = path.read_bytes()
        try:
            config = json.loads(original)
            config["train"] = []
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ContractError, "Common split"):
                load_suite(self.suite)
        finally:
            path.write_bytes(original)

    def test_split_overlap_seed_and_missing_cases_rejected(self):
        for bad in (
            {"seed": 43, "train": ["a"], "val": ["b"]},
            {"seed": 42, "train": ["a"], "val": ["a"]},
            {"seed": 42, "train": ["a", "a"], "val": ["b"]},
            {"seed": 42, "train": [], "val": ["b"]},
            {"seed": 43, "inner_train": ["a"], "inner_val": ["b"],
             "outer_train": ["a", "b"], "outer_val": ["c"]},
        ):
            with self.assertRaises(ContractError):
                normalize_split(bad)
        omitted = self.owned / "omitted.json"
        omitted.write_text(json.dumps({"seed": 42, "train": ["unit_train"], "val": ["unit_val"]}), encoding="utf-8")
        with self.assertRaisesRegex(ContractError, "every image and label"):
            initialize(self.owned / "must_not_initialize", self.medical, omitted)
        self.assertFalse((self.owned / "must_not_initialize").exists())

    def test_changed_initialization_contract_never_overwrites(self):
        path = self.owned / "different_split.json"
        altered = copy.deepcopy(self.split)
        altered["train"], altered["val"] = altered["val"], altered["train"]
        path.write_text(json.dumps(altered), encoding="utf-8")
        with self.assertRaisesRegex(ContractError, "different data/split"):
            initialize(self.suite, self.medical, path)

    def test_ambiguous_native_file_names_rejected(self):
        duplicate = self.medical / "Data/image/unit_train_0000.nii"
        duplicate.write_bytes(b"")
        try:
            with self.assertRaisesRegex(ContractError, "Ambiguous native"):
                initialize(self.owned / "ambiguous_must_not_initialize", self.medical, self.split_path)
        finally:
            duplicate.unlink()  # Only the UNIT fixture created immediately above.

    def test_unsupported_native_file_convention_rejected_before_preparation(self):
        invalid = self.medical / "Data/image/UNIT_wrong_convention.nii"
        invalid.write_bytes(b"")
        try:
            with self.assertRaises(ContractError):
                initialize(self.owned / "unsupported_must_not_initialize", self.medical, self.split_path)
        finally:
            invalid.unlink()

    def test_manifest_cannot_omit_stages_or_listed_sources(self):
        path = self.suite / "manifest.json"
        original = path.read_bytes()
        try:
            for mutation in (lambda m: m["stages"].pop("v1.3"),
                             lambda m: m["stages"]["v1.3"]["source_hashes"].pop("hiercp/model.py")):
                altered = json.loads(original)
                mutation(altered)
                plain = {k: v for k, v in altered.items() if k != "manifest_sha256"}
                altered["manifest_sha256"] = canonical_hash(plain)
                path.write_text(json.dumps(altered), encoding="utf-8")
                with self.assertRaises(ContractError):
                    load_suite(self.suite)
        finally:
            path.write_bytes(original)

    def test_command_plans_have_independent_checkpoints_shared_readonly_cache(self):
        checkpoints = set()
        for stage in STAGES:
            cwd, commands = command_plan(self.suite, stage, "train", python="UNIT_python")
            self.assertEqual(cwd, self.suite / "source" / stage)
            self.assertEqual(len(commands), 1)
            command = commands[0]
            self.assertEqual(command[:4], ["UNIT_python", "-u", "-m", "hiercp.pipeline"])
            self.assertNotIn("--overwrite", command)
            self.assertNotIn("--epochs", command)
            self.assertNotIn("--max-cases", command)
            checkpoints.add(command[command.index("--checkpoint") + 1])
            self.assertEqual(command[command.index("--cache-dir") + 1], str(self.suite / "shared/cache"))
            self.assertEqual(command[command.index("--device") + 1], "cuda")
        self.assertEqual(len(checkpoints), 4)
        cwd, commands = command_plan(self.suite, "v1.3", "prepare")
        self.assertEqual(cwd, self.suite / "source/v1.0")
        self.assertEqual([x[4] for x in commands], ["prepare-prototypes", "prepare"])

    def test_execution_lock_accepts_only_identical_baseline_measurement(self):
        trial = self.owned / "lock_fixture"
        (trial / "results/v1.0").mkdir(parents=True)
        (trial / "configs").mkdir()
        (trial / "configs/v1.0.json").write_bytes((self.suite / "configs/v1.0.json").read_bytes())
        calibration = dict(format="hiercp_preflight_calibration_v2", selected_batch_size=32,
                           selected_num_workers=8, resource_fingerprint={"gpu": "UNIT_metadata_only"},
                           identity={"seed": 42, "cache_dir": str(trial / "shared/cache"),
                                     "checkpoint_path": str(trial / "results/v1.0/checkpoint_best.pt")})
        path = trial / "results/v1.0/checkpoint_best.pt.preflight.json"
        path.write_text(json.dumps(calibration), encoding="utf-8")
        first = freeze_execution(trial)
        self.assertEqual(freeze_execution(trial), first)
        calibration["selected_batch_size"] = 16
        path.write_text(json.dumps(calibration), encoding="utf-8")
        with self.assertRaisesRegex(ContractError, "calibration changed"):
            freeze_execution(trial)

    def test_launch_binding_rejects_changed_cache_and_existing_unbound_weights(self):
        trial = self.owned / "launch_fixture"
        config = json.loads((self.suite / "configs/v1.0.json").read_text(encoding="utf-8"))
        # Digests are metadata fixtures; no checkpoint or prototype tensor exists.
        with patch("hiercp_v1x.experiment.digest", return_value="a" * 64):
            original = bind_launch(trial, "v1.0", self.manifest, config, None)
            self.assertEqual(bind_launch(trial, "v1.0", self.manifest, config, None), original)
        with patch("hiercp_v1x.experiment.digest", return_value="b" * 64):
            with self.assertRaisesRegex(ContractError, "no exact resume"):
                bind_launch(trial, "v1.0", self.manifest, config, None)
        empty = self.owned / "unbound_launch_fixture"
        with patch("hiercp_v1x.experiment.digest", return_value="a" * 64), \
                patch.object(Path, "glob", return_value=[empty / "never_created.pt"]):
            with self.assertRaisesRegex(ContractError, "Unbound pre-existing weights"):
                bind_launch(empty, "v1.0", self.manifest, config, None)

    def test_owned_lock_collision_and_exception_cleanup(self):
        path = self.owned / "single_owned.lock"
        with owned_lock(path):
            with self.assertRaises(FileExistsError):
                with owned_lock(path):
                    self.fail("second invocation acquired same lock")
        self.assertFalse(path.exists())
        with self.assertRaisesRegex(RuntimeError, "UNIT interrupted"):
            with owned_lock(path):
                raise RuntimeError("UNIT interrupted")
        self.assertFalse(path.exists())

    def test_child_failure_uses_foreground_argv_preserves_outputs_and_clears_own_lock(self):
        child = SimpleNamespace(pid=999999, wait=lambda: 1)
        with patch("hiercp_v1x.experiment.subprocess.Popen", return_value=child) as popen:
            with self.assertRaisesRegex(RuntimeError, "outputs preserved"):
                execute_stage(self.suite, "v1.0", "prepare")
        args, kwargs = popen.call_args
        self.assertIsInstance(args[0], list)
        self.assertEqual(args[0][1:4], ["-u", "-m", "hiercp.pipeline"])
        self.assertNotIn("shell", kwargs)
        self.assertNotIn("start_new_session", kwargs)
        self.assertEqual(kwargs["env"]["PYTHONDONTWRITEBYTECODE"], "1")
        self.assertNotIn("PYTHONPATH", kwargs["env"])
        self.assertFalse((self.suite / "shared/prepare.lock").exists())

    def test_generation_uses_links_only_and_excludes_outer_test(self):
        prepare_generation_inputs(self.suite, self.manifest)
        prepare_generation_inputs(self.suite, self.manifest)
        for kind in ("image", "labels"):
            paths = list((self.suite / "shared/generation_data" / kind).iterdir())
            self.assertEqual(len(paths), 2)
            self.assertFalse(any("unit_outer" in x.name for x in paths))
            for path in paths:
                self.assertEqual(path.stat().st_ino, (self.medical / "Data" / kind / path.name).stat().st_ino)

    def test_successful_train_collects_after_writer_lock_is_released(self):
        child = SimpleNamespace(pid=999999, wait=lambda: 0)
        def collect(experiment, stage):
            self.assertFalse((Path(experiment) / "results" / stage / "run.lock").exists())
            return {"metrics": {"mrr": 0.5}, "UNIT_only": True}
        with patch("hiercp_v1x.experiment.subprocess.Popen", return_value=child), \
                patch("hiercp_v1x.experiment.bind_launch"), \
                patch("hiercp_v1x.experiment.freeze_execution", return_value={"UNIT_only": True}), \
                patch("hiercp_v1x.results.collect_result", side_effect=collect) as collector:
            result = execute_stage(self.suite, "v1.0", "train")
        collector.assert_called_once_with(self.suite, "v1.0")
        self.assertEqual(result["metrics"], {"mrr": 0.5})
        self.assertEqual(result["comparison_report"], str(self.suite / "results/v1.0/comparison_report.json"))

    def test_generation_collects_and_checks_bound_file_hashes_before_child_launch(self):
        # A source-text file is a hash-only proof fixture; never treated as weights.
        path = self.suite / "source/v1.0/README.md"
        proof = {"checkpoints": {"best": {"path": str(path), "sha256": digest(path)}}}
        child = SimpleNamespace(pid=999999, wait=lambda: 0)
        def collect(experiment, stage):
            self.assertFalse((Path(experiment) / "results" / stage / "run.lock").exists())
            return copy.deepcopy(proof)
        with patch("hiercp_v1x.results.collect_result", side_effect=collect) as collector, \
                patch("hiercp_v1x.experiment.prepare_generation_inputs") as link_inputs, \
                patch("hiercp_v1x.experiment.subprocess.Popen", return_value=child) as popen:
            execute_stage(self.suite, "v1.0", "generate")
        collector.assert_called_once()
        link_inputs.assert_called_once()
        popen.assert_called_once()
        proof["checkpoints"]["best"]["sha256"] = "0" * 64
        with patch("hiercp_v1x.results.collect_result", side_effect=collect), \
                patch("hiercp_v1x.experiment.subprocess.Popen") as popen:
            with self.assertRaisesRegex(ContractError, "changed before launch"):
                execute_stage(self.suite, "v1.0", "generate")
        popen.assert_not_called()


if __name__ == "__main__":
    unittest.main()
