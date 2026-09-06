"""DEBUG orchestration only: opaque byte fixtures are never medical/model data.

The repository's complete train configuration is copied unchanged. Only case
discovery/classification, resource probes, subprocesses and graph migration /
publication validation are explicit DEBUG doubles. Source hashing, independent
file copying, pilot binding, configuration generation and receipt checks are real.
No image decoder, checkpoint loader, graph builder or training process is run.
"""
from contextlib import ExitStack
import copy
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from hiercp import cache, common, preparation_runtime, donor_preflight
from tools import feedback_preparation_recovery as recovery


class RecoveryOrchestrationDebugTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="DEBUG_recovery_flow_")
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        self.project = root / "DEBUG checkout"
        self.source = self.project / "work/failed"
        self.target = self.project / "work/recovered"
        self.medical = root / "DEBUG medical"
        self.fold = 2  # Exercise base-seed versus fold-specific seed propagation.
        self.relative_gnn = Path(f"paired/folds/fold_{self.fold}/gnn")
        self.old_gnn = self.source / self.relative_gnn
        self.gnn = self.target / self.relative_gnn
        repository = Path(__file__).resolve().parents[1]
        # Copy real source/config bytes for actual code identity, never execute
        # the temporary checkout and never reduce any graph/model configuration.
        for original in [repository / "config/train.json",
                         repository / "tools/feedback_preparation_recovery.py",
                         *sorted((repository / "hiercp").rglob("*.py"))]:
            destination = self.project / original.relative_to(repository)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original, destination)
        self.config = recovery.read_json(self.project / "config/train.json")
        self.plan = {"project_root": self.project, "run_root": self.target,
                     "medical_root": self.medical, "outer_fold": self.fold,
                     "dataset_id": 760, "seed": 17, "python_executable": sys.executable,
                     "train_config": self.target / "recovery/train_config.json",
                     "minimum_free_bytes": 1}  # DEBUG filesystem probe only.
        self.split = {"train": ["DEBUG_DONOR_TRAIN", "DEBUG_NO_DONOR_TRAIN"],
                      "val": ["DEBUG_DONOR_VAL"],
                      "outer_validation_excluded": ["DEBUG_OUTER_HELDOUT"]}
        self.selected = self.split["train"] + self.split["val"]
        self.paths = []
        for case_id in self.selected:
            image_path = self.medical / "Data" / f"{case_id}.image.bytes"
            label_path = self.medical / "Data" / f"{case_id}.label.bytes"
            for path in (image_path, label_path):
                self._bytes(path, f"DEBUG opaque fixture; not NIfTI: {path.name}".encode())
            self.paths.append(SimpleNamespace(case_id=case_id, image_path=image_path,
                                               label_path=label_path))
        self.sources = cache._source_contract(self.paths, self.selected)
        old_config = {"state": "failed", "run_mode": "benchmark", "subset_active": False,
                      "selected_case_ids": self.selected, "train_case_ids": self.split["train"],
                      "val_case_ids": self.split["val"], "graph_config": self.config["graph"],
                      "labels": self.config["labels"], "ct_clip": self.config["ct_clip"],
                      "source_cases": self.sources, "seed": self.config["seed"] + self.fold,
                      **self.config["cache"],
                      "difficulty_fractions": {
                          "easy": self.config["cache"]["easy_fraction"],
                          "inter": self.config["cache"]["inter_fraction"],
                          "intra_corrupted": self.config["cache"]["intra_fraction"]}}
        old_plan = {"project_root": str(self.project), "run_root": str(self.source),
                    "medical_root": str(self.medical), "commands": [
                        {"name": "gnn-prepare", "argv": ["DEBUG", "--outer-fold", str(self.fold)]},
                        {"name": "plan", "argv": ["DEBUG", "--dataset-id", "760"]},
                        {"name": "train_full", "argv": ["DEBUG", "--seed", "17"]}]}
        recovery.write_new(self.source / "launch_plan.json", old_plan)
        recovery.write_new(self.old_gnn / "split.json", self.split)
        recovery.write_new(self.old_gnn / "graphs/config.json", old_config)
        for path in [self.source / "paired/outer_splits.json", self.source / "paired/case_profiles.csv",
                     *[self.old_gnn / name for name in (
                         "prototype.pt", "metadata.json", "manifest.csv", "regions/DEBUG_region.bytes")]]:
            self._bytes(path, b"DEBUG opaque original; never decoded as a model or graph")
        self._bytes(self.old_gnn / "graphs/manifest.csv", (
            "case_id,sample_index,status\nDEBUG_DONOR_TRAIN,0,error\n"
            "DEBUG_NO_DONOR_TRAIN,,donor_ineligible\nDEBUG_DONOR_VAL,1,error\n").encode())
        self.before_source = self._tree(self.source)
        self.before_medical = self._tree(self.medical)
        self.env = {"CUDA_VISIBLE_DEVICES": "3", "DEBUG_ORCHESTRATION": "true"}
        self.events = []
        self.commands = []
        self.requests = []
        self.fail_prepare = False
        self.peak_rss = 512 * 1024 * 1024  # Explicit DEBUG probe value, not a measurement.
        stack = ExitStack()
        self.addCleanup(stack.close)
        for module, name, callback in (
            (common, "discover_cases", self._discover_debug),
            (donor_preflight, "audit_donor_headers", self._headers_debug),
            (cache, "build_donor_eligibility", self._eligibility_debug),
            (cache, "migrate_failed_hierarchical_cache", self._migrate_debug),
            (cache, "validate_cache_migration", self._validate_migration_debug),
            (cache, "validate_cache_publication", self._validate_publication_debug),
            (preparation_runtime, "snapshot", self._snapshot_debug),
            (recovery.importlib.metadata, "version", lambda name: "DEBUG-version-" + name),
        ):
            stack.enter_context(mock.patch.object(module, name, side_effect=callback))

    @staticmethod
    def _bytes(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)

    @staticmethod
    def _tree(root):
        return {str(path.relative_to(root)): path.read_bytes()
                for path in root.rglob("*") if path.is_file()}

    def _discover_debug(self, data_dir, *, case_ids, run_mode):
        self.events.append("source_scan")
        self.assertEqual(Path(data_dir), self.medical / "Data")
        self.assertEqual(case_ids, self.selected)
        self.assertEqual(run_mode, "benchmark")
        return self.paths

    def _eligibility_debug(self, **kwargs):
        self.events.append("eligibility")
        self.assertEqual(kwargs["selected_case_ids"], self.selected)
        self.assertEqual(kwargs["source_cases"], self.sources)
        self.assertEqual(kwargs["workers"], "auto")
        rows = []
        for source in self.sources:
            eligible = source["case_id"] != "DEBUG_NO_DONOR_TRAIN"
            tumor_count = 100 if eligible else 0
            # Synthetic metadata only; no volume allocated. A large component
            # exercises increased memory ceiling while graph geometry stays fixed.
            rows.append({**source, "shape": [128, 128, 128], "spacing_mm": [1.0, 1.0, 1.0],
                         "label_histogram": {"0": 128 ** 3 - 1000 - tumor_count,
                                             "1": 1000, "2": tumor_count},
                         "component_bbox_shapes": [[80, 90, 85]] if eligible else [],
                         "eligible": eligible, "reason": "configured_tumor_label_present"
                         if eligible else "configured_tumor_label_absent"})
        result = {"format": cache.DONOR_ELIGIBILITY_FORMAT, "selected_case_ids": self.selected,
                  "eligible_case_ids": [row["case_id"] for row in rows if row["eligible"]],
                  "ineligible_case_ids": ["DEBUG_NO_DONOR_TRAIN"],
                  "labels": self.config["labels"], "cases": rows}
        result["contract_sha256"] = cache._cache_config_fingerprint(result)
        # Real schema/checksum validation, independent of the classification double.
        cache.validate_donor_eligibility(result, selected_case_ids=self.selected,
                                        source_cases=self.sources, labels=self.config["labels"])
        return result

    def _headers_debug(self, **kwargs):
        self.events.append("header_preflight")
        self.assertEqual(kwargs["case_paths"], self.paths)
        self.assertEqual(kwargs["selected_case_ids"], self.selected)
        self.assertEqual(kwargs["workers"], "auto")
        return {}  # Explicit header-decoder double; real header tests are separate.

    def _snapshot_debug(self):
        self.events.append("resource_probe")
        return {"available_memory_bytes": 1024 ** 4, "host_total_memory_bytes": 2 * 1024 ** 4,
                "cpu_capacity": 32, "debug_probe": True}

    def _assert_prepare_contract(self, kwargs):
        self.assertEqual(kwargs["source_cache_dir"], self.old_gnn / "graphs")
        self.assertEqual(kwargs["destination_cache_dir"], self.gnn / "graphs")
        arguments = kwargs["prepare_kwargs"]
        self.assertEqual(arguments["train_case_ids"], self.split["train"])
        self.assertEqual(arguments["val_case_ids"], self.split["val"])
        self.assertEqual(arguments["seed"], self.config["seed"] + self.fold)
        self.assertEqual(arguments["total_candidates"], self.config["cache"]["total_candidates"])
        self.assertEqual(arguments["candidate_pool_size"], self.config["cache"]["candidate_pool_size"])
        self.assertIsNone(arguments["max_cases"])
        self.assertFalse(arguments["overwrite"])
        self.assertEqual(arguments["donor_eligibility"]["selected_case_ids"], self.selected)
        self.assertEqual(len(arguments["donor_eligibility"]["eligible_case_ids"]), 2)

    def _migrate_debug(self, **kwargs):
        self.events.append("migration")
        self._assert_prepare_contract(kwargs)
        recovery.write_new(self.gnn / "graphs/migration.json",
                           {"state": "ready_for_prepare", "debug_double": True})

    def _validate_migration_debug(self, **kwargs):
        self.events.append("verify_migration")
        self._assert_prepare_contract(kwargs)
        self.assertEqual(recovery.read_json(self.gnn / "graphs/migration.json")["state"], "ready_for_prepare")

    def _validate_publication_debug(self, path):
        self.events.append("verify_publication")
        self.assertEqual(Path(path), self.gnn / "graphs")
        self.assertEqual(recovery.read_json(Path(path) / "complete.json"), {"debug_double": True})

    def _runner_debug(self, argv, *, cwd, env, check):
        self.assertEqual(Path(cwd), self.project)
        self.assertTrue(check)
        self.assertEqual(argv[:3], [sys.executable, "-B", "-m"])
        self.commands.append((list(argv), dict(env)))
        if argv[3:5] == ["tools.feedback_preparation_recovery", "profile"]:
            self.events.append("profile")
            self.assertEqual(env, self.env)
            request = recovery.read_json(argv[argv.index("--request") + 1])
            self.requests.append(request)
            self.assertEqual(request["config"]["seed"], self.config["seed"] + self.fold)
            self.assertEqual(request["code_sha256"], recovery.preparation_code_identity(self.project))
            self.assertEqual(request["source"], next(row for row in self.sources
                                                     if row["case_id"] == request["case_id"]))
            result = {"format": "hiercp_full_size_resource_pilot_v1", "calibration_only": True,
                      "training_performed": False, "case_id": request["case_id"],
                      "sample_index": request["sample_index"], "request_sha256": recovery.value_sha(request),
                      "roi_budget": request["config"]["graph"]["adaptive_roi_max_voxels"],
                      "candidate_count": self.config["cache"]["total_candidates"],
                      "measurement": {"status": "complete", "sampling_error": None,
                                      "sampled_peak_rss_bytes": self.peak_rss, "elapsed_seconds": 0.25,
                                      "debug_double": True}}
            recovery.write_new(argv[argv.index("--output") + 1], result)
            return
        self.assertEqual(argv[3:5], ["tools.paired_benchmark", "gnn-prepare"])
        self.events.append("prepare")
        for flag, value in (("--project-root", self.project), ("--medical-root", self.medical),
                            ("--work", self.target / "paired"), ("--outer-fold", self.fold),
                            ("--train-config", self.plan["train_config"]), ("--device", "cuda:0")):
            self.assertEqual(argv.count(flag), 1)
            self.assertEqual(argv[argv.index(flag) + 1], str(value))
        self.assertEqual(env, {**self.env, "HIERCP_PREPARE_MEASURED_CASE_RSS_BYTES": str(self.peak_rss)})
        self.assertNotIn("--overwrite", argv)
        if self.fail_prepare:
            raise RuntimeError("DEBUG preparation interruption")
        for name in ("config.json", "index.json", "complete.json"):
            recovery.write_new(self.gnn / "graphs" / name, {"debug_double": True})
        self._bytes(self.gnn / "graphs/manifest.csv", b"DEBUG publication double, not graphs")

    def _run(self):
        return recovery.prepare_recovery(self.plan, self.source, runner=self._runner_debug, env=self.env)

    def test_complete_real_file_flow_preserves_full_configuration_and_originals(self):
        receipt = self._run()
        self.assertEqual(self.events, ["source_scan", "header_preflight", "eligibility", "resource_probe", "profile", "profile",
                                       "resource_probe", "migration", "prepare", "verify_publication"])
        self.assertEqual([r["case_id"] for r in self.requests], ["DEBUG_DONOR_TRAIN", "DEBUG_DONOR_VAL"])
        proposed = recovery.read_json(self.plan["train_config"])
        self.assertGreater(proposed["graph"]["adaptive_roi_max_voxels"], self.config["graph"]["adaptive_roi_max_voxels"])
        unchanged = copy.deepcopy(proposed)
        unchanged["graph"]["adaptive_roi_max_voxels"] = self.config["graph"]["adaptive_roi_max_voxels"]
        unchanged["runtime"]["prepare_workers"] = self.config["runtime"]["prepare_workers"]
        self.assertEqual(unchanged, self.config)
        self.assertEqual(self.before_source, self._tree(self.source))
        self.assertEqual(self.before_medical, self._tree(self.medical))
        self.assertEqual(self.env, {"CUDA_VISIBLE_DEVICES": "3", "DEBUG_ORCHESTRATION": "true"})
        self.assertFalse(receipt["training_performed"])
        self.assertTrue(receipt["original_results_preserved"])
        self.assertEqual(receipt["train_config"], str(self.plan["train_config"]))
        recovery.verify_identity(receipt["source_identity"])
        for name, expected in receipt["files"].items():
            self.assertEqual(recovery.digest(self.target / name), expected)
        self.assertEqual((self.gnn / "regions/DEBUG_region.bytes").read_bytes(),
                         (self.old_gnn / "regions/DEBUG_region.bytes").read_bytes())
        # Destination really is independent, not a hard link to the original.
        (self.gnn / "prototype.pt").write_bytes(b"DEBUG destination-only mutation")
        self.assertEqual(self.before_source, self._tree(self.source))

    def test_complete_reuse_reverifies_without_relaunch_or_writes(self):
        first = self._run()
        before = self._tree(self.target)
        self.events.clear()
        self.commands.clear()
        self.assertEqual(self._run(), first)
        self.assertEqual(self.events, ["source_scan", "verify_migration", "verify_publication"])
        self.assertEqual(self.commands, [])
        self.assertEqual(self._tree(self.target), before)

    def test_header_failures_stop_before_raw_hash_copy_or_voxel_classification(self):
        with mock.patch.object(donor_preflight, "audit_donor_headers",
                               side_effect=ValueError("DEBUG header failures")), \
                mock.patch.object(cache, "_source_contract") as source_hash, \
                mock.patch.object(recovery, "_copy_verified") as copy_files, \
                mock.patch.object(cache, "build_donor_eligibility") as classify:
            with self.assertRaisesRegex(ValueError, "DEBUG header failures"):
                self._run()
            source_hash.assert_not_called()
            copy_files.assert_not_called()
            classify.assert_not_called()
        self.assertEqual(self.commands, [])
        self.assertFalse(Path(self.plan["train_config"]).exists())
        self.assertFalse((self.target / "recovery/donor_eligibility.json").exists())
        self.assertEqual(self.before_source, self._tree(self.source))
        self.assertEqual(self.before_medical, self._tree(self.medical))

    def test_complete_reuse_rejects_changed_raw_source_before_relaunch(self):
        self._run()
        self.paths[0].label_path.write_bytes(b"DEBUG changed source label bytes")
        self.events.clear()
        self.commands.clear()
        with self.assertRaisesRegex(ValueError, "Actual source images/labels differ"):
            self._run()
        self.assertEqual(self.events, ["source_scan"])
        self.assertEqual(self.commands, [])

    def test_complete_reuse_rejects_changed_original_identity(self):
        self._run()
        (self.old_gnn / "prototype.pt").write_bytes(b"DEBUG changed original prototype")
        self.events.clear()
        with self.assertRaisesRegex(ValueError, "initial identity"):
            self._run()
        self.assertEqual(self.events, [])

    def test_complete_reuse_rejects_modified_target_artifact(self):
        self._run()
        self.plan["train_config"].write_bytes(b"DEBUG changed config")
        self.commands.clear()
        with self.assertRaisesRegex(ValueError, "completion artifact changed"):
            self._run()
        self.assertEqual(self.commands, [])

    def test_interrupted_preparation_reuses_bound_pilots_and_revalidates_migration(self):
        self.fail_prepare = True
        with self.assertRaisesRegex(RuntimeError, "DEBUG preparation interruption"):
            self._run()
        self.assertFalse((self.target / "recovery/complete.json").exists())
        self.assertEqual(self.before_source, self._tree(self.source))
        self.fail_prepare = False
        self.events.clear()
        self.commands.clear()
        receipt = self._run()
        self.assertEqual(self.events, ["source_scan", "header_preflight", "resource_probe", "resource_probe",
                                       "verify_migration", "prepare", "verify_publication"])
        self.assertEqual(len(self.commands), 1)
        self.assertFalse(receipt["training_performed"])

    def test_interrupted_preparation_rejects_tampered_successful_pilot(self):
        self.fail_prepare = True
        with self.assertRaisesRegex(RuntimeError, "DEBUG preparation interruption"):
            self._run()
        output = next((self.target / "recovery/pilots").glob("*.measurement.json"))
        result = recovery.read_json(output)
        result["request_sha256"] = "0" * 64
        output.write_text(json.dumps(result), encoding="utf-8")
        self.fail_prepare = False
        self.commands.clear()
        with self.assertRaisesRegex(ValueError, "bind the entire current request"):
            self._run()
        self.assertEqual(self.commands, [])
        self.assertFalse((self.target / "recovery/complete.json").exists())


if __name__ == "__main__":
    unittest.main()
