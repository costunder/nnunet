"""DEBUG real reconstruction APIs on synthetic NIfTI, never clinical training.

This uses two anatomically valid synthetic candidates and the explicit smoke
geometry profile. Production's factory separately requires all 128 candidates;
the full-128 model/logit/gradient checks live in test_feedback_gnn_debug.py.
No completed/trained quality checkpoint is fabricated by this fixture.
"""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import nibabel as nib
import numpy as np
import torch
from scipy import ndimage as ndi

from custom_trainers.onlinecp_curriculum_contract import file_sha256
from hiercp.common import CasePaths, load_case, choose_source_tumor, build_candidate_pool, distance_to_mask_mm
from hiercp.feedback import BankGraphProvider, FeedbackGNNRuntime
from hiercp.feedback_patient import PreparedFeedbackPatient
from hiercp.feedback_resources import content_digest
from hiercp.sample import materialize_sample_views
from hiercp.prototype import build_prototype_bank
from hiercp.region import (REGION_CACHE_SEED_SALT, load_or_build_patient_regions,
                           _region_cache_metadata, save_patient_regions)
from tools.online_cp_benchmark import RAW_MARKER_NAME, stable_seed
from tools.smoke import _full_config


class FeedbackGraphBindingDebugTests(unittest.TestCase):
    def test_debug_synthetic_nifti_real_hierarchy_binding_cache_and_no_heldout_reads(self):
        with tempfile.TemporaryDirectory(prefix="hiercp_feedback_nifti_debug_") as temporary:
            root = Path(temporary)
            raw, bank_root = root / "raw", root / "bank"
            (raw / "imagesTr").mkdir(parents=True)
            (raw / "labelsTr").mkdir()
            bank_root.mkdir()
            case_id, heldout = "debug_train", "debug_heldout"
            shape = (40, 40, 40)
            z, y, x = np.indices(shape)
            liver = (z - 20) ** 2 / 17. ** 2 + (y - 20) ** 2 / 16. ** 2 + (x - 20) ** 2 / 15. ** 2 <= 1
            tumor = (z - 20) ** 2 + (y - 19) ** 2 + (x - 17) ** 2 <= 3.5 ** 2
            label = np.zeros(shape, dtype=np.int16)
            label[liver], label[tumor] = 1, 2
            image = np.random.default_rng(1).normal(70., 11., shape).astype(np.float32)
            image[tumor] += 24.
            image_path, label_path = raw / "imagesTr" / f"{case_id}_0000.nii.gz", raw / "labelsTr" / f"{case_id}.nii.gz"
            affine = np.diag([1.2, 1.1, 1., 1.])
            nib.save(nib.Nifti1Image(image, affine), image_path)
            nib.save(nib.Nifti1Image(label, affine), label_path)
            case = load_case(CasePaths(case_id, image_path, label_path))
            config = replace(_full_config(), sample_hops=3, max_lesions=None)
            config.validate()
            regions = load_or_build_patient_regions(
                case, liver_label=1, tumor_label=2, config=config, ct_clip=(-200., 250.),
                seed=stable_seed(42, case_id, REGION_CACHE_SEED_SALT), cache_dir=None, overwrite=False, mmap=False,
            )
            prototype = build_prototype_bank([(case_id, regions.region_features)], config=config,
                                             rng=np.random.default_rng(4))
            source, _, _ = choose_source_tumor(image, label, tumor_label=2, rng=np.random.default_rng(2),
                                               selection="largest", pad=3)
            candidates, _ = build_candidate_pool(
                case, source, placement_mask=label == 1, full_organ_mask=regions.full_organ_mask,
                occupied_mask=label == 2, organ_distance=regions.organ_depth, rng=np.random.default_rng(5),
                num_candidates=2, max_draws=80_000, min_liver_coverage=.8, occupied_clearance_vox=1,
                min_center_separation_mm=7.,
            )
            self.assertEqual(len(candidates), 2)
            centers = np.asarray([candidate.center for candidate in candidates], dtype=np.int32)
            entry = "debug_source.npz"
            np.savez(bank_root / entry, candidate_raw_centers=centers, scores=np.asarray([.1, .2], np.float32),
                     source_component=np.asarray([source.component_id], np.int16))
            second_entry = "debug_second_entry_same_patient.npz"
            (bank_root / second_entry).write_bytes((bank_root / entry).read_bytes())
            marker = {"train_ids": [case_id], "val_ids": [heldout], "source_cases": [
                {"case_id": case_id, "image_sha256": file_sha256(image_path), "label_sha256": file_sha256(label_path)},
                # Names-only held-out metadata: intentionally no held-out voxel
                # files exist. Feedback must never try to open/hash them.
                {"case_id": heldout, "image_sha256": "0" * 64, "label_sha256": "0" * 64},
            ]}
            (raw / RAW_MARKER_NAME).write_text(json.dumps(marker), encoding="utf-8")
            training = root / "debug_train_config.json"
            training.write_text(json.dumps({"seed": 42, "labels": {"liver": 1, "tumor": 2},
                                             "generation": {"source_pad": 3}}), encoding="utf-8")
            contract = {"files": {"train_config": {"path": str(training)}}, "outer_fold": 0,
                        "train_case_ids": [case_id], "validation_case_ids": [heldout],
                        "entry_sha256": {name: file_sha256(bank_root / name) for name in (entry, second_entry)}}
            index = {"candidate_count": 2, "entries_by_case": {case_id: [entry, second_entry]},
                     "raw_marker_sha256": file_sha256(raw / RAW_MARKER_NAME)}
            source_regions = root / "verified_quality_regions"
            native_metadata = _region_cache_metadata(
                case, liver_label=1, tumor_label=2, config=config, ct_clip=(-200., 250.),
                seed=stable_seed(42, case_id, REGION_CACHE_SEED_SALT))
            native_metadata["graph_config"].pop("patient_graph_contract")
            save_patient_regions(regions, source_regions / case_id, metadata=native_metadata)
            original_regions = {p: file_sha256(p) for p in source_regions.rglob("*") if p.is_file()}
            options = {"raw_data_root": str(raw), "graph_cache_dir": str(root / "graph_cache"),
                       "region_cache_dir": str(source_regions), "graph_cache_minimum_free_bytes": 0}
            metadata = {"graph_config": config.to_dict(), "ct_clip": [-200., 250.]}
            with mock.patch("hiercp.common.load_case", wraps=load_case) as actual_reader:
                provider = BankGraphProvider(config=options, bank_root=bank_root, contract=contract,
                                              index=index, checkpoint=metadata, prototype=prototype)
                component_map, _ = ndi.label(case.label == 2, structure=ndi.generate_binary_structure(3, 1))
                prior_context = PreparedFeedbackPatient(case, component_map, regions,
                                                         distance_to_mask_mm(case.label == 2, case.spacing), {})
                prior = materialize_sample_views(
                    provider._build(entry, bank_root / entry, case_id, prepared=prior_context),
                    training=False, epoch=0, global_seed=0)
                with mock.patch("hiercp.region.build_patient_regions", side_effect=AssertionError("verified regions rebuilt")):
                    inventory = provider.ensure_graph_cache()
                self.assertEqual(set(inventory), {entry, second_entry})
                self.assertEqual(provider.preparation_report["cold_entries"], 2)
                self.assertEqual(original_regions, {p: file_sha256(p) for p in source_regions.rglob("*") if p.is_file()})
                sample = provider.get(entry)
                self.assertEqual(content_digest(prior), content_digest(sample))
                self.assertEqual(actual_reader.call_count, 1)
                self.assertEqual(actual_reader.call_args.args[0].case_id, case_id)
                self.assertTrue(np.array_equal(sample["candidate_centers"].numpy(), centers))
                self.assertEqual(sample["source_component"], source.component_id)
                self.assertEqual(len(sample["local_graphs"]), 2)
                self.assertEqual(sample["patient_graph"]["candidate"].num_nodes, 2)
                self.assertEqual(sample["prototype_graph"]["candidate"].num_nodes, 2)
                self.assertGreater(sum(graph.num_edges for graph in sample["local_graphs"]), 0)
                before = {path.name: file_sha256(path) for path in provider.cache.iterdir() if path.is_file()}
                with mock.patch("hiercp.sample.build_local_view", side_effect=AssertionError("static cache rebuilt")):
                    restored = provider.get(entry)
                self.assertEqual(actual_reader.call_count, 1)
                self.assertTrue(torch.equal(sample["candidate_centers"], restored["candidate_centers"]))
                self.assertEqual(before, {path.name: file_sha256(path) for path in provider.cache.iterdir() if path.is_file()})
                forbidden = dict(index, entries_by_case={heldout: [entry]})
                with self.assertRaisesRegex(ValueError, "cohort"):
                    BankGraphProvider(config=options, bank_root=bank_root, contract=contract,
                                      index=forbidden, checkpoint=metadata, prototype=prototype)
                self.assertEqual(actual_reader.call_count, 1)
                receipt = provider.cache / (provider.store.key(entry, provider.binding) + ".json")
                receipt.write_text(json.dumps({"sha256": "0" * 64}), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "changed immutable"):
                    provider.get(entry)

    def test_debug_production_factory_refuses_partial_candidate_pool_before_loading_artifacts(self):
        config = json.loads((Path(__file__).parents[1] / "config" / "online_cp_feedback_gnn.json").read_text())
        with self.assertRaisesRegex(ValueError, "128-candidate"):
            FeedbackGNNRuntime.from_config(config, bank_root="unused", bank_contract={},
                                            bank_index={"candidate_count": 2}, device="cpu")


if __name__ == "__main__":
    unittest.main()
