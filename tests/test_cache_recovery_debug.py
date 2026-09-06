"""Isolated DEBUG recovery tests; synthetic NIfTI only, no medical training."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import shutil
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import nibabel as nib
import numpy as np
import torch

from hiercp import cache
from hiercp.region import graph_config_budget_compatible
from hiercp.schema import GraphBuildConfig
from hiercp.prototype import PrototypeBank
from hiercp.region import PatientRegionData, save_patient_regions, _region_cache_metadata
from hiercp.common import load_case, stable_case_seed


class DonorEligibilityDebugTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="hiercp-donor-debug-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def case(self, name, values, *, label_affine=None):
        image_path, label_path = self.root / f"{name}_image.nii.gz", self.root / f"{name}_label.nii.gz"
        image = np.full(values.shape, 100, dtype=np.float32)
        nib.save(nib.Nifti1Image(image, np.eye(4)), image_path)
        nib.save(nib.Nifti1Image(values, np.eye(4) if label_affine is None else label_affine), label_path)
        return SimpleNamespace(case_id=name, image_path=image_path, label_path=label_path)

    def collect(self, paths):
        selected = [path.case_id for path in paths]
        sources = cache._source_contract(paths, selected)
        # Scheduler itself has separate real-runtime tests. This scaffold executes
        # every supplied case and isolates raw-label/contract behavior only.
        def debug_jobs(*, tasks, function, commit, **kwargs):
            for task in tasks:
                commit(function(task))
        with patch("hiercp.preparation_runtime.run_case_jobs", side_effect=debug_jobs):
            contract = cache.build_donor_eligibility(
                case_paths=paths, selected_case_ids=selected, source_cases=sources,
                liver_label=1, tumor_label=2, workers=2, report_path=self.root / "debug_runtime.json")
        return contract, sources

    def test_full_cohort_is_preserved_and_only_label_present_is_donor(self):
        tumor = np.ones((5, 6, 7), dtype=np.int16)
        tumor[1:3, 2:5, 3:5] = 2
        absent = np.ones((5, 6, 7), dtype=np.int16)
        contract, sources = self.collect([self.case("train_tumor", tumor), self.case("val_absent", absent)])
        self.assertEqual(contract["selected_case_ids"], ["train_tumor", "val_absent"])
        self.assertEqual(contract["eligible_case_ids"], ["train_tumor"])
        self.assertEqual(contract["ineligible_case_ids"], ["val_absent"])
        self.assertEqual(contract["cases"][0]["component_bbox_shapes"], [[2, 3, 2]])
        self.assertEqual(contract["cases"][1]["reason"], "configured_tumor_label_absent")
        self.assertEqual(len(sources), 2)
        self.assertEqual(contract["cases"][0]["label_histogram"]["2"], 12)

    def test_component_connectivity_matches_six_connected_source_selection(self):
        values = np.ones((4, 4, 4), dtype=np.int16)
        values[1, 1, 1] = values[2, 2, 2] = 2
        contract, _ = self.collect([self.case("diagonal", values)])
        self.assertEqual(contract["cases"][0]["component_bbox_shapes"], [[1, 1, 1], [1, 1, 1]])

    def test_raw_fractional_nonfinite_and_unknown_labels_fail_before_cast(self):
        for value in (1.5, np.nan, np.inf, 3, -1):
            with self.subTest(value=value):
                values = np.ones((3, 3, 3), dtype=np.float32)
                values[1, 1, 1] = value
                with self.assertRaisesRegex(ValueError, "nonfinite/noninteger|unsupported"):
                    self.collect([self.case("bad", values)])

    def test_affine_mismatch_and_source_changes_fail(self):
        values = np.ones((3, 3, 3), dtype=np.int16)
        affine = np.eye(4)
        affine[0, 3] = 1
        with self.assertRaisesRegex(ValueError, "affine mismatch"):
            self.collect([self.case("shifted", values, label_affine=affine)])
        paths = self.case("changed", values)
        sources = cache._source_contract([paths], [paths.case_id])
        paths.label_path.write_bytes(b"DEBUG changed input")
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            cache.build_donor_eligibility(case_paths=[paths], selected_case_ids=[paths.case_id], source_cases=sources,
                                         liver_label=1, tumor_label=2, workers=2,
                                         report_path=self.root / "changed_runtime.json")

    def test_float32_header_roundoff_is_audited_without_modifying_sources(self):
        values = np.ones((3, 4, 5), dtype=np.uint8)
        values[1, 1, 1] = 2
        paths = self.case("roundoff", values)
        image_affine = np.diag([0.64453125, 0.64453125, 0.70000005, 1.0])
        image_affine[:3, 3] = [-163.35547, -295.35547, 54.399963]
        label_affine = np.diag([0.6445313, 0.6445313, 0.7, 1.0])
        label_affine[:3, 3] = [-163.3555, -295.3555, 54.399963]
        for path, data, affine in ((paths.image_path, values.astype(np.float32), image_affine),
                                   (paths.label_path, values, label_affine)):
            nii = nib.Nifti1Image(data, affine)
            nii.header.set_xyzt_units("mm")
            nib.save(nii, path)
        original = {path: path.read_bytes() for path in (paths.image_path, paths.label_path)}
        contract, sources = self.collect([paths])
        self.assertEqual(contract["cases"][0]["geometry_audit"]["accepted_as"], "roundoff")
        self.assertEqual(contract["eligible_case_ids"], [paths.case_id])
        self.assertEqual(original, {path: path.read_bytes() for path in original})
        altered = copy.deepcopy(contract)
        altered["cases"][0]["geometry_audit"]["accepted_as"] = "unverified"
        with self.assertRaisesRegex(ValueError, "checksum"):
            cache.validate_donor_eligibility(altered, selected_case_ids=[paths.case_id],
                source_cases=sources, labels={"liver": 1, "tumor": 2})

    def test_partition_histogram_and_source_tampering_are_rejected(self):
        values = np.ones((3, 3, 3), dtype=np.int16)
        values[1, 1, 1] = 2
        original, sources = self.collect([self.case("tumor", values)])
        modifications = (
            lambda c: c["eligible_case_ids"].clear(),
            lambda c: c["cases"][0].update(eligible=False),
            lambda c: c["cases"][0]["label_histogram"].update({"2": -1}),
            lambda c: c["cases"][0].update(label_sha256="a" * 64),
            lambda c: c["cases"][0].update(component_bbox_shapes=[]),
        )
        for mutate in modifications:
            altered = copy.deepcopy(original)
            mutate(altered)
            altered["contract_sha256"] = cache._cache_config_fingerprint({k: v for k, v in altered.items() if k != "contract_sha256"})
            with self.assertRaises(ValueError):
                cache.validate_donor_eligibility(altered, selected_case_ids=["tumor"], source_cases=sources,
                                                 labels={"liver": 1, "tumor": 2})

    def test_budget_compatibility_only_allows_increase_without_semantic_changes(self):
        original = GraphBuildConfig().to_dict()
        increased = {**original, "adaptive_roi_max_voxels": original["adaptive_roi_max_voxels"] + 1}
        self.assertTrue(graph_config_budget_compatible(original, original))
        self.assertTrue(graph_config_budget_compatible(original, increased))
        self.assertFalse(graph_config_budget_compatible(increased, original))
        for key, value in (("patch_size", 49), ("context_radius_mm", 29), ("sampled_context_nodes", 1)):
            self.assertFalse(graph_config_budget_compatible(original, {**increased, key: value}))
        self.assertEqual(original, GraphBuildConfig().to_dict())

    def test_ineligible_terminal_is_not_a_missing_self_donor_sample(self):
        fingerprint = "a" * 64
        row = cache._progress_row(case_id="absent", sample_index=None, split_name="val",
                                  status="donor_ineligible", config_fingerprint=fingerprint)
        kwargs = dict(root=self.root, records={("absent", None): row}, selected_case_ids=["absent"],
                      samples_per_case=2, fingerprint=fingerprint)
        self.assertTrue(cache._progress_is_complete(**kwargs, donor_case_ids=[]))
        self.assertFalse(cache._progress_is_complete(**kwargs))
        row["status"] = "no_tumor"
        self.assertFalse(cache._progress_is_complete(**kwargs, donor_case_ids=[]))


class CacheMigrationDebugTests(DonorEligibilityDebugTests):
    """Structural cache/prototype fixtures, explicitly not trained model outputs."""
    def setUp(self):
        super().setUp()
        self.old = self.root / "legacy" / "graphs"
        self.new = self.root / "recovery" / "graphs"
        self.old.mkdir(parents=True)
        self.graph = GraphBuildConfig()
        self.paths = []
        for name in ("donor", "absent", "pending"):
            labels = np.ones((5, 6, 7), dtype=np.int16)
            if name != "absent":
                labels[1:3, 2:4, 2:4] = 2
            self.paths.append(self.case(name, labels))
        self.ids = [p.case_id for p in self.paths]
        self.sources = cache._source_contract(self.paths, self.ids)
        self.donor, _ = self.collect(self.paths)
        self.original_bank = self.old.parent / "prototypes" / "bank.pt"
        self.new_bank = self.new.parent / "prototypes" / "bank.pt"
        self.regions = self.old.parent / "regions"
        self.new_regions = self.new.parent / "regions"
        self.bank = PrototypeBank(np.ones((16, 18), np.float32), np.ones((16, 16), np.float32),
            np.zeros(16, np.float32), np.ones(16, np.float32), np.empty((2, 0), np.int64), ("absent", "donor"))
        self.bank.save(self.original_bank)
        manifest = self.original_bank.parent / "manifest.csv"
        train_sources = sorted(self.sources[:2], key=lambda row: row["case_id"])
        cache._atomic_manifest_save([{**row, "status": "ok"} for row in train_sources], manifest)
        prototype_metadata = {"format": cache.PROTOTYPE_METADATA_FORMAT, "integrity_format": "sha256_v1",
            "training_cases": ["absent", "donor"], "source_cases": train_sources, "seed": 42,
            "graph_config": self.graph.to_dict(), "labels": {"liver": 1, "tumor": 2},
            "ct_clip": [-200.0, 250.0], "region_cache_format": cache.REGION_CACHE_FORMAT,
            "state": "ready", "prototype_fingerprint": self.bank.fingerprint(),
            "prototype_sha256": cache._sha256_file(self.original_bank), "manifest_sha256": cache._sha256_file(manifest)}
        self.write_json(self.original_bank.parent / "metadata.json", prototype_metadata)
        for path in self.paths:
            case = load_case(path)
            data = PatientRegionData(np.ones(case.shape, bool), np.ones(case.shape, np.float32),
                np.zeros(case.shape, np.int16), np.ones((24, 16), np.float32), np.ones((24, 3), np.float32),
                np.empty((2, 0), np.int64), np.ones((24, 3), np.float32))
            meta = _region_cache_metadata(case, liver_label=1, tumor_label=2, config=self.graph,
                ct_clip=(-200, 250), seed=stable_case_seed(42, path.case_id, cache.REGION_CACHE_SEED_SALT))
            save_patient_regions(data, self.regions / path.case_id, metadata=meta)
        shutil.copytree(self.original_bank.parent, self.new_bank.parent)
        shutil.copytree(self.regions, self.new_regions)
        self.kwargs = dict(data_dir=self.root, cache_dir=self.new, region_cache_dir=self.new_regions,
            bank_path=self.new_bank, train_case_ids=self.ids[:2], val_case_ids=self.ids[2:], graph_config=self.graph,
            liver_label=1, tumor_label=2, source_selection="random", source_pad=4, samples_per_case=2,
            total_candidates=8, candidate_pool_size=128, easy_fraction=.3, inter_fraction=.3, intra_fraction=.4,
            max_draws=20000, min_liver_coverage=.9, occupied_clearance_vox=2,
            min_center_separation_mm=8, ct_clip=(-200, 250), seed=42, max_cases=None, overwrite=False,
            workers=2, run_mode="debug", donor_eligibility=self.donor)
        legacy = cache._cache_expected_metadata(self.kwargs, bank=self.bank, sources=self.sources,
            selected_case_ids=self.ids, donor_eligibility=self.donor)
        del legacy["donor_eligibility"], legacy["donor_contract_sha256"]
        self.old_fingerprint = cache._cache_config_fingerprint(legacy)
        self.legacy = {**legacy, "config_fingerprint": self.old_fingerprint, "state": "failed",
            "data_dir": str(self.root), "prototype_bank": str(self.original_bank),
            "region_cache_dir": str(self.regions), "progress_format": cache.CACHE_PROGRESS_FORMAT}
        self.write_json(self.old / "config.json", self.legacy)
        self.records = {}
        for case_id, count in (("donor", 2), ("pending", 1)):
            for index in range(count):
                payload = self.sample(case_id, index)
                path = self.old / f"{case_id}__{index:03d}.pt"
                cache._atomic_torch_save(payload, path)
                source = next(row for row in self.sources if row["case_id"] == case_id)
                self.records[(case_id, index)] = cache._progress_row(case_id=case_id, sample_index=index,
                    split_name="train" if case_id == "donor" else "val", status="ok", path=path.name,
                    candidates=8, config_fingerprint=self.old_fingerprint,
                    source_image_sha256=source["image_sha256"], source_label_sha256=source["label_sha256"],
                    artifact_sha256=cache._sha256_file(path), file_size=path.stat().st_size)
        for case_id in self.ids:
            source = next(row for row in self.sources if row["case_id"] == case_id)
            self.records[(case_id, None)] = cache._progress_row(case_id=case_id, sample_index=None,
                split_name="val" if case_id == "pending" else "train",
                status={"donor": "ok", "absent": "no_tumor", "pending": "sample_failure"}[case_id],
                config_fingerprint=self.old_fingerprint, source_image_sha256=source["image_sha256"],
                source_label_sha256=source["label_sha256"])
        cache._atomic_progress_manifest_save(self.records, self.old / "manifest.csv")
        self.discovery = patch("hiercp.cache.discover_cases", side_effect=lambda directory, *, case_ids, **kw:
            [path for path in self.paths if path.case_id in case_ids])
        self.discovery.start()
        self.addCleanup(self.discovery.stop)

    @staticmethod
    def write_json(path, value):
        path.write_text(json.dumps(value), encoding="utf-8")

    def sample(self, case_id, sample_index):
        source = next(row for row in self.sources if row["case_id"] == case_id)
        canonical = {"format": "canonical-full-v22", "nodes": {"DEBUG": {"x": torch.ones(3, 2)}}, "edges": {}}
        return {"format": cache.CACHE_FORMAT, "prototype_fingerprint": self.bank.fingerprint(),
            "config_fingerprint": self.old_fingerprint, "case_id": case_id, "sample_index": sample_index,
            "split": "val" if case_id == "pending" else "train", "source_component": 1,
            "source_image_sha256": source["image_sha256"], "source_label_sha256": source["label_sha256"],
            "graph_config": self.graph.to_dict(), "ct_clip": (-200, 250), "source_local": canonical,
            "target_locals": [copy.deepcopy(canonical) for _ in range(8)], "source_patch": torch.ones(5, 4, 4, 4),
            "target_patches": torch.ones(8, 5, 4, 4, 4), "difficulties": torch.arange(8) % 4,
            "corruptions": torch.zeros(8, dtype=torch.long), "candidate_centers": torch.ones(8, 3),
            "candidate_regions": torch.zeros(8, dtype=torch.long), "candidate_prototypes": torch.zeros(8, dtype=torch.long)}

    def migrate(self):
        return cache.migrate_failed_hierarchical_cache(source_cache_dir=self.old,
            destination_cache_dir=self.new, prepare_kwargs=self.kwargs)

    def test_migration_preserves_sources_and_only_relabels_verified_artifact_metadata(self):
        before = {str(path): cache._sha256_file(path) for path in self.old.parent.rglob("*") if path.is_file()}
        result = self.migrate()
        self.assertEqual(result["state"], "ready_for_prepare")
        self.assertEqual(len(list(self.new.glob("*.pt"))), 3)
        self.assertFalse((self.new / "complete.json").exists())
        config = json.loads((self.new / "config.json").read_text())
        self.assertEqual(config["selected_case_ids"], self.ids)
        self.assertEqual(config["donor_eligibility"]["eligible_case_ids"], ["donor", "pending"])
        new = cache._torch_load_cpu(self.new / "donor__000.pt")
        old = cache._torch_load_cpu(self.old / "donor__000.pt")
        torch.testing.assert_close(new["target_patches"], old["target_patches"], rtol=0, atol=0)
        self.assertNotEqual(new["config_fingerprint"], old["config_fingerprint"])
        self.assertEqual(before, {str(path): cache._sha256_file(path) for path in self.old.parent.rglob("*") if path.is_file()})
        cache.validate_cache_migration(source_cache_dir=self.old, destination_cache_dir=self.new, prepare_kwargs=self.kwargs)
        with self.assertRaises(FileExistsError):
            self.migrate()

    def test_config_or_artifact_mismatch_is_rejected_before_destination_creation(self):
        self.kwargs["seed"] = 43
        with self.assertRaises(ValueError):
            self.migrate()
        self.assertFalse(self.new.exists())
        self.kwargs["seed"] = 42
        (self.old / "donor__000.pt").write_bytes(b"DEBUG corruption")
        with self.assertRaises((ValueError, FileExistsError)):
            self.migrate()
        self.assertFalse(self.new.exists())

    def test_budget_increase_is_allowed_but_decrease_is_not(self):
        self.kwargs["graph_config"] = GraphBuildConfig(adaptive_roi_max_voxels=8_000_001)
        self.migrate()
        self.assertEqual(cache._torch_load_cpu(self.new / "donor__000.pt")["graph_config"]["adaptive_roi_max_voxels"], 8_000_001)
        self.kwargs["cache_dir"] = self.root / "decrease"
        self.kwargs["graph_config"] = GraphBuildConfig(adaptive_roi_max_voxels=7_999_999)
        with self.assertRaises(ValueError):
            cache.migrate_failed_hierarchical_cache(source_cache_dir=self.old, destination_cache_dir=self.kwargs["cache_dir"], prepare_kwargs=self.kwargs)
        self.assertFalse(self.kwargs["cache_dir"].exists())

    def test_migration_certificate_or_original_mutation_blocks_recovery(self):
        self.migrate()
        certificate = json.loads((self.new / "migration.json").read_text())
        certificate["state"] = "copying"
        self.write_json(self.new / "migration.json", certificate)
        with self.assertRaises(ValueError):
            cache.validate_cache_migration(source_cache_dir=self.old, destination_cache_dir=self.new, prepare_kwargs=self.kwargs)

    def test_prepare_reuses_every_migrated_success_and_builds_only_missing_donor(self):
        self.migrate()
        retained = {p.name: cache._sha256_file(p) for p in self.new.glob("*.pt")}
        calls = []
        def debug_sample(case, bank, regions, *, sample_index, **kwargs):
            # The expensive graph producer is an explicit structural fixture.
            # Actual resume, file hashing and publication functions run normally.
            calls.append((case.paths.case_id, sample_index))
            return self.sample(case.paths.case_id, sample_index)
        with patch("hiercp.cache.build_training_sample", side_effect=debug_sample):
            cache.prepare_hierarchical_cache(**self.kwargs)
        self.assertEqual(calls, [("pending", 1)])
        self.assertEqual(retained, {name: cache._sha256_file(self.new / name) for name in retained})
        index = cache.validate_cache_publication(self.new)
        self.assertEqual(index["expected_entries"], 4)
        self.assertEqual(index["donor_case_ids"], ["donor", "pending"])
        config = json.loads((self.new / "config.json").read_text())
        self.assertEqual(config["selected_case_ids"], ["donor", "absent", "pending"])
        self.assertEqual(config["state"], "ready_nonproduction")
        cache.validate_cache_migration(source_cache_dir=self.old, destination_cache_dir=self.new, prepare_kwargs=self.kwargs)

    def test_overlap_and_untracked_artifact_are_rejected(self):
        request = {**self.kwargs, "cache_dir": self.old / "nested"}
        with self.assertRaises(ValueError):
            cache.migrate_failed_hierarchical_cache(source_cache_dir=self.old,
                destination_cache_dir=request["cache_dir"], prepare_kwargs=request)
        self.assertFalse(request["cache_dir"].exists())
        torch.save(self.sample("donor", 0), self.old / "untracked.pt")
        with self.assertRaises((ValueError, FileExistsError)):
            self.migrate()
        self.assertFalse(self.new.exists())

    def test_original_or_retained_artifact_mutation_is_detected_after_migration(self):
        self.migrate()
        path = self.new / "donor__000.pt"
        payload = cache._torch_load_cpu(path)
        payload["target_patches"][0, 0, 0, 0, 0] = 7
        torch.save(payload, path)
        with self.assertRaises(ValueError):
            cache.validate_cache_migration(source_cache_dir=self.old, destination_cache_dir=self.new, prepare_kwargs=self.kwargs)


if __name__ == "__main__":
    unittest.main()
