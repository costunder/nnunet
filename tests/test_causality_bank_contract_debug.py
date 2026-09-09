"""DEBUG-only bank-reader contract integration; no medical/GPU training claims.

Both real readers validate real Torch/JSON files, cache publication hashes and
canonical inventory. Only expensive model construction/prototype deserialization
use labeled stand-ins. Synthetic historic timing/allocation metadata exercises
receipt arithmetic; it does not claim measured production performance.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

from hiercp import cache
from hiercp.tensor import torch_load_compat
from tools import causality, online_cp_benchmark, online_cp_argmax_benchmark
from tools.causality_resources import audit_host_budget, build_audit_inventory


READERS = (online_cp_benchmark, online_cp_argmax_benchmark)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


class CausalityBankContractDebugTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="DEBUG_causality_bank_")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.graphs = self.root / "gnn" / "graphs"
        self.graphs.mkdir(parents=True)
        self.gnn = self.graphs.parent
        self.checkpoint_path = self.gnn / "model.pt"
        self.prototype_path = self.gnn / "prototype.pt"
        self.preflight_path = self.gnn / "causality.json.preflight.json"
        self.report_path = self.gnn / "causality.json"
        self.train_ids = ["DEBUG_train"]
        self.val_ids = ["DEBUG_val_a", "DEBUG_val_b"]
        self.ids = self.train_ids + self.val_ids
        self.prototype_path.write_bytes(b"DEBUG prototype; not a trained model")
        self.prototype = SimpleNamespace(training_case_ids=self.train_ids,
                                         fingerprint=lambda: "DEBUG_prototype")
        self.model = torch.nn.Linear(2, 2)
        self.layout = SimpleNamespace(gnn=lambda fold: self.gnn,
            train_config=self.root / "training.json", outer_splits=self.root / "outer.json")
        write_json(self.layout.train_config, {"seed": 42, "debug_metadata_fixture": True})
        write_json(self.layout.outer_splits, {"format": online_cp_benchmark.OUTER_SPLIT_FORMAT,
            "splits": [{"fold": 0, "train": self.ids, "val": ["DEBUG_outer_holdout"]}]})
        write_json(self.gnn / "split.json", {"format": online_cp_benchmark.GNN_SPLIT_FORMAT,
            "outer_fold": 0, "train": self.train_ids, "val": self.val_ids,
            "outer_validation_excluded": ["DEBUG_outer_holdout"]})
        self.publish_cache()
        self.publish_checkpoint()
        self.publish_audit()

    def publish_cache(self):
        sources = []
        for case_id in self.ids:
            # These explicit metadata fixtures are not disguised NIfTI images.
            image, label = (self.root / f"{case_id}.{kind}.debug" for kind in ("image", "label"))
            image.write_bytes(f"DEBUG synthetic image {case_id}".encode())
            label.write_bytes(f"DEBUG synthetic label {case_id}".encode())
            sources.append({"case_id": case_id, "image_sha256": sha(image), "label_sha256": sha(label)})
        donor = {"format": cache.DONOR_ELIGIBILITY_FORMAT, "debug_metadata_fixture": True,
            "selected_case_ids": self.ids, "eligible_case_ids": self.ids,
            "ineligible_case_ids": [], "labels": {"liver": 1, "tumor": 2},
            "cases": [{**source, "shape": [2, 2, 2], "spacing_mm": [1.0, 1.0, 1.0],
                "label_histogram": {"0": 0, "1": 7, "2": 1}, "component_bbox_shapes": [[1, 1, 1]],
                "eligible": True, "reason": "configured_tumor_label_present"} for source in sources]}
        donor["contract_sha256"] = cache._cache_config_fingerprint(donor)
        config = {"format": cache.CACHE_FORMAT, "integrity_format": "sha256_v1",
            "index_format": cache.CACHE_INDEX_FORMAT, "debug_metadata_fixture": True,
            "run_mode": "benchmark", "subset_active": False, "train_case_ids": self.train_ids,
            "val_case_ids": self.val_ids, "selected_case_ids": self.ids, "samples_per_case": 1,
            "total_candidates": 2, "source_cases": sources, "labels": {"liver": 1, "tumor": 2},
            "donor_eligibility": donor, "donor_contract_sha256": donor["contract_sha256"],
            "prototype_fingerprint": self.prototype.fingerprint(),
            "prototype_artifact_sha256": sha(self.prototype_path), "ct_clip": [-200.0, 250.0],
            "graph_config": {"debug_metadata_fixture": True, "depths": [3, 2, 2]}}
        fingerprint = cache._cache_config_fingerprint(config)
        config.update(config_fingerprint=fingerprint, state="ready_nonproduction",
            prototype_bank=str(self.prototype_path), progress_format=cache.CACHE_PROGRESS_FORMAT)
        records, entries = {}, []
        for index, source in enumerate(sources):
            case_id = source["case_id"]
            local = {"format": "canonical-full-v22", "nodes": {"DEBUG": {
                "x": torch.zeros(index + 2, 3)}}, "edges": {}}
            path = self.graphs / f"{case_id}__000.pt"
            torch.save({"debug_metadata_fixture": True, "source_local": local,
                "target_locals": [local, local], "source_patch": torch.zeros(1, 2, 2, 2)}, path)
            split = "train" if case_id in self.train_ids else "val"
            entry = {"case_id": case_id, "sample_index": 0, "path": path.name, "split": split,
                "source_image_sha256": source["image_sha256"],
                "source_label_sha256": source["label_sha256"],
                "artifact_sha256": sha(path), "file_size": path.stat().st_size}
            entries.append(entry)
            records[(case_id, 0)] = cache._progress_row(**{key: value for key, value in entry.items()
                if key != "split"}, split_name=split, status="ok", config_fingerprint=fingerprint)
            records[(case_id, None)] = cache._progress_row(case_id=case_id, sample_index=None,
                split_name=split, status="ok", config_fingerprint=fingerprint,
                source_image_sha256=source["image_sha256"], source_label_sha256=source["label_sha256"])
        common = {"cache_format": cache.CACHE_FORMAT, "config_fingerprint": fingerprint,
            "prototype_fingerprint": self.prototype.fingerprint(), "run_mode": "benchmark",
            "donor_contract_sha256": donor["contract_sha256"], "donor_case_ids": self.ids,
            "expected_entries": len(entries)}
        write_json(self.graphs / "config.json", config)
        write_json(self.graphs / "index.json", {**common, "format": cache.CACHE_INDEX_FORMAT,
                                                "entries": entries})
        cache._atomic_progress_manifest_save(records, self.graphs / "manifest.csv")
        write_json(self.graphs / "complete.json", {**common, "format": cache.CACHE_COMPLETE_FORMAT,
            "selected_case_ids": self.ids, "samples_per_case": 1, "entries": len(entries),
            **{f"{name}_sha256": sha(self.graphs / filename) for name, filename in
               (("config", "config.json"), ("index", "index.json"), ("manifest", "manifest.csv"))}})
        self.config = config
        self.index = cache.validate_cache_publication(self.graphs)

    def publish_checkpoint(self):
        parameters = sorted(name for name, _ in self.model.named_parameters())
        resources = {"DEBUG_historic_training_resources": True}
        self.checkpoint = {"debug_metadata_fixture": True, "method": "hiercp-full",
            "framework": "torch_geometric", "training_complete": True,
            "completed_epoch": 40, "target_epochs": 40,
            "model_kwargs": {"in_features": 2, "out_features": 2}, "state_dict": self.model.state_dict(),
            "graph_config": self.config["graph_config"], "ct_clip": (-200.0, 250.0),
            "prototype_fingerprint": self.prototype.fingerprint(), "prototype_training_cases": self.train_ids,
            "gradient_connectivity": {"format": "hiercp_gradient_connectivity_v1", "verified": True,
                "expected_parameter_count": len(parameters), "connected_parameter_count": len(parameters),
                "connected_parameters": parameters, "missing_parameters": []},
            "training_signature": {"format": "hiercp_training_signature_v1", "run_mode": "benchmark",
                "ablation_mode": "full", "target_epochs": 40, "batch_setting": "auto", "worker_setting": "auto",
                "calibration_resource_fingerprint": resources,
                "train_cache_files": sorted(e["path"] for e in self.index["entries"] if e["split"] == "train"),
                "val_cache_files": sorted(e["path"] for e in self.index["entries"] if e["split"] == "val")},
            "preflight_calibration": {"format": causality.TRAINING_PREFLIGHT_FORMAT,
                "resource_fingerprint": resources, "identity": {"checkpoint_path": str(self.checkpoint_path),
                    "cache_dir": str(self.graphs), "run_mode": "benchmark",
                    "batch_calibration_max_vram_fraction": 0.9, "loader_calibration_batches": 2,
                    "prefetch_factor": 2, "pin_memory": False},
                "batch_trials": [{"batch_size": value, "status": "DEBUG_fixture"} for value in (1, 2)],
                "worker_trials": [{"num_workers": value, "status": "DEBUG_fixture"} for value in (0, 2)]}}
        torch.save(self.checkpoint, self.checkpoint_path)
        self.checkpoint = torch_load_compat(self.checkpoint_path, map_location="cpu")

    def publish_audit(self, device="cpu"):
        selected, entries = causality._cache_entries_for_split(self.graphs, self.index, "val")
        self.inventory = build_audit_inventory(selected)
        self.artifacts = causality._artifact_contract(checkpoint_path=self.checkpoint_path,
            checkpoint=self.checkpoint, prototype_path=self.prototype_path,
            signature=self.checkpoint["training_signature"], cache_dir=self.graphs,
            cache_config=self.config, selected_entries=entries, run_mode="benchmark", split="val",
            max_batches=0, seed=42, permutation_tolerance=1e-4, response_threshold=1e-4, strict=True)
        plan = causality._training_measurement_plan(self.checkpoint,
            checkpoint_path=self.checkpoint_path, cache_dir=self.graphs, run_mode="benchmark")
        identity = causality._build_preflight_identity(self.artifacts, plan,
            device=device, repeats=3, inventory=self.inventory)
        # Only receipt creation reads this labeled synthetic historic snapshot.
        # Reader validation below cannot call any current machine/GPU probe.
        with mock.patch("hiercp.training_resources.snapshot", return_value={
                "available_memory_bytes": 1_000_000_000, "DEBUG_historic_snapshot": True}):
            batches = [{"batch_size": size, "status": "accepted", "repeats": 3,
                "cohort_size": 2, "completed_samples": 3 * size, "elapsed_seconds": 1.0,
                "samples_per_second": float(3 * size), "peak_vram_fraction": None,
                "host_measurement": {"status": "complete", "DEBUG_synthetic_measurement": True},
                "host_budget": audit_host_budget(self.inventory, selected, size, 0, 1, False)}
                for size in (1, 2)]
            workers = [{"num_workers": number, "status": "accepted", "samples": 4,
                "measurement_batches": 2, "elapsed_seconds": 2.0 if number == 0 else 1.0,
                "samples_per_second": 2.0 if number == 0 else 4.0,
                "host_measurement": {"status": "complete", "DEBUG_synthetic_measurement": True},
                "host_budget": audit_host_budget(self.inventory, selected, 2, number, 2, False)}
                for number in (0, 2)]
        self.preflight = {"format": causality.PREFLIGHT_FORMAT, "identity": identity,
            "resource_fingerprint": {"selected_device": device, "DEBUG_historic_resources": True},
            "selected_batch_size": 2, "selected_num_workers": 2,
            "batch_trials": batches, "worker_trials": workers}
        verdict = {key: True for key in ("permutation_invariant", "upper_position_shortcut_blocked",
            "upper_clearance_shortcut_blocked", "shortcut_safety_supported", "target_context_sensitive",
            "context_causality_supported", "spatial_edge_sensitive", "topology_sensitive")}
        self.report = {"format": causality.REPORT_FORMAT, "debug_metadata_fixture": True,
            "status": "complete", "strict_pass": True, "checkpoint": str(self.checkpoint_path),
            "split": "val", "cache_files": 2, "evaluated_batches": 1,
            "thresholds": {"permutation_tolerance": 1e-4, "response_threshold": 1e-4},
            "clean": {"samples": 2}, "conditions": {name: {"samples": 2} for name in causality.CONDITIONS},
            "verdict": verdict}
        self.seal()

    def seal(self):
        """Rehash changed DEBUG receipts, so negative tests exercise deeper guards."""
        self.preflight["identity_sha256"] = causality._value_sha256(self.preflight["identity"])
        write_json(self.preflight_path, self.preflight)
        contract = causality._input_contract(self.artifacts, preflight_path=self.preflight_path,
            preflight=self.preflight, batch_size=self.preflight["selected_batch_size"],
            num_workers=self.preflight["selected_num_workers"])
        self.report.update(input_contract=contract, input_contract_sha256=causality._value_sha256(contract))
        self.report["resource_preflight"] = {"path": str(self.preflight_path),
            "artifact_sha256": sha(self.preflight_path), "contract_sha256": causality._value_sha256(self.preflight),
            "physical_batch_size": self.preflight["selected_batch_size"],
            "num_workers": self.preflight["selected_num_workers"],
            "batch_trials": copy.deepcopy(self.preflight["batch_trials"]),
            "worker_trials": copy.deepcopy(self.preflight["worker_trials"]),
            "input_inventory": copy.deepcopy(self.inventory)}
        write_json(self.report_path, self.report)

    def verify(self, *, error=None):
        before = {path: sha(path) for path in self.root.rglob("*") if path.is_file()}
        try:
            with mock.patch("hiercp.model.HierarchicalPyGPlacementModel", side_effect=torch.nn.Linear), \
                 mock.patch.object(causality.PrototypeBank, "load", return_value=self.prototype), \
                 mock.patch("hiercp.training_resources.snapshot", side_effect=AssertionError("Reader must not remeasure")), \
                 mock.patch("torch.cuda.is_available", side_effect=AssertionError("Reader must not inspect current GPU")):
                for reader in READERS:
                    with self.subTest(reader=reader.__name__):
                        if error is None:
                            self.assertEqual(reader._verified_gnn_causality(self.layout, 0), self.report)
                        else:
                            with self.assertRaisesRegex(reader.OnlineBenchmarkError, error):
                                reader._verified_gnn_causality(self.layout, 0)
        finally:
            self.assertEqual({path: sha(path) for path in self.root.rglob("*") if path.is_file()}, before)

    def test_new_v2_report_passes_both_real_readers_without_rewriting(self):
        self.assertIsInstance(self.checkpoint["ct_clip"], tuple)
        self.assertIsInstance(json.loads((self.graphs / "config.json").read_text())["ct_clip"], list)
        self.verify()

    def test_historic_cuda_report_can_be_verified_without_a_current_gpu(self):
        self.publish_audit(device="cuda:0")
        self.verify()

    def test_legacy_identity_or_receipt_is_rejected_even_after_rehashing(self):
        for field in ("identity", "receipt"):
            with self.subTest(field=field):
                self.publish_audit()
                if field == "identity":
                    self.preflight["identity"]["format"] = "hiercp_causality_preflight_identity_v1"
                else:
                    self.preflight["format"] = "hiercp_causality_preflight_v1"
                self.seal()
                self.verify(error="preflight")

    def test_forged_bounds_and_inventory_fail_despite_recomputed_receipt_hashes(self):
        mutations = (lambda identity: identity["batch_input_upper_bounds"].__setitem__("2", 1),
                     lambda identity: identity.update(input_inventory_sha256="0" * 64),
                     lambda identity: identity.update(input_inventory_sample_count=3))
        for mutate in mutations:
            self.publish_audit()
            mutate(self.preflight["identity"])
            self.seal()
            self.verify(error="identity")

    def test_saved_plan_must_match_checkpoint_and_all_trial_arithmetic(self):
        for mutate in (
            lambda p: p["identity"]["measurement_plan"].update(prefetch_factor=3),
            lambda p: p["identity"]["measurement_plan"].update(batch_candidates=[2, 1]),
            lambda p: p["batch_trials"][1]["host_budget"].update(estimated_input_bytes=1),
            lambda p: p["worker_trials"][1].update(samples=3),
            lambda p: p.update(selected_batch_size=1),
        ):
            self.publish_audit()
            mutate(self.preflight)
            self.seal()
            self.verify(error="identity|host budget|measurement|selected batch")

    def test_report_hash_links_and_input_contract_cannot_be_detached(self):
        for mutate in (
            lambda r: r.update(input_contract_sha256="0" * 64),
            lambda r: r["input_contract"]["execution"].update(physical_batch_size=1),
            lambda r: r["resource_preflight"].update(artifact_sha256="0" * 64),
            lambda r: r["resource_preflight"].update(contract_sha256="0" * 64),
            lambda r: r["resource_preflight"].update(batch_trials=[]),
        ):
            self.publish_audit()
            mutate(self.report)
            write_json(self.report_path, self.report)
            self.verify(error="report|contract")

    def test_nested_inventory_and_historic_device_must_match_verified_identity(self):
        self.report["resource_preflight"]["input_inventory"]["rows"][0]["input_bytes_upper_bound"] += 1
        write_json(self.report_path, self.report)
        self.verify(error="resource/preflight")
        self.publish_audit()
        self.preflight["resource_fingerprint"]["selected_device"] = "cuda:0"
        self.seal()
        self.verify(error="device|resource")

    def test_strict_verdict_and_entire_validation_cohort_are_still_required(self):
        for mutate in (
            lambda r: r["verdict"].update(permutation_invariant=False),
            lambda r: r["verdict"].update(spatial_edge_sensitive=False, topology_sensitive=False),
            lambda r: r.update(strict_pass=False),
            lambda r: r["clean"].update(samples=1),
            lambda r: r["conditions"].pop(next(iter(causality.CONDITIONS))),
            lambda r: r.update(evaluated_batches=2),
        ):
            self.publish_audit()
            mutate(self.report)
            write_json(self.report_path, self.report)
            self.verify(error="strict|cohort|sample|condition|batch|verdict")

    def test_checkpoint_and_artifact_guards_are_real_not_success_stubs(self):
        for field, value, error in (("training_complete", False, "completed full training"),
                                    ("ct_clip", (-200.0, 251.0), "ct_clip"),
                                    ("prototype_fingerprint", "changed", "prototype fingerprint")):
            with self.subTest(field=field):
                original = self.checkpoint[field]
                self.checkpoint[field] = value
                torch.save(self.checkpoint, self.checkpoint_path)
                self.verify(error=error)
                self.checkpoint[field] = original
                torch.save(self.checkpoint, self.checkpoint_path)
        self.publish_audit()
        with self.prototype_path.open("ab") as handle:
            handle.write(b"DEBUG corruption")
        self.verify(error="SHA-256")


if __name__ == "__main__":
    unittest.main()
