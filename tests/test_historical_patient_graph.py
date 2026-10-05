"""Synthetic CT geometry UNIT checks, not real-data neural/quality evidence."""
from __future__ import annotations

import copy
from dataclasses import replace
from itertools import product
from types import SimpleNamespace
import unittest

import numpy as np
from scipy import ndimage as ndi
import torch

from hiercp import hierarchy
from hiercp.common import choose_source_tumor, normalized_position
from hiercp.curriculum import CandidateSpec
from hiercp.prototype import PrototypeBank
from hiercp.region import PatientRegionData
from hiercp.schema import GraphBuildConfig, PATIENT_EDGE_TYPES, PROTOTYPE_EDGE_TYPES
from hiercp_v22.data import donor_in_target_spacing
from hiercp_v1x.historical_patient_graph import build_external_hierarchy


def case_and_regions(case_id, shape, spacing):
    image = (np.indices(shape).sum(axis=0) * 3).astype(np.float32)
    label = np.ones(shape, np.uint8)
    case = SimpleNamespace(paths=SimpleNamespace(case_id=case_id), image=image,
                           label=label, spacing=np.asarray(spacing, np.float32))
    organ = np.ones(shape, bool)
    depth = ndi.distance_transform_edt(organ, sampling=spacing).astype(np.float32)
    assignments = np.zeros(shape, np.int64)
    assignments[shape[0] // 2:] = 1
    centers = np.asarray([[shape[0] // 4, shape[1] // 2, shape[2] // 2],
                          [3 * shape[0] // 4, shape[1] // 2, shape[2] // 2]], np.float32)
    features = np.tile(np.linspace(.1, .9, 16, dtype=np.float32), (2, 1))
    features[1] += .15
    features[:, 15] = 1.
    positions = np.stack([normalized_position(center, shape) for center in centers]).astype(np.float32)
    regions = PatientRegionData(organ, depth, assignments, features, positions,
                                np.asarray([[0, 1], [1, 0]], np.int64), centers)
    return case, regions


def unit_inputs(candidate_count=5):
    # Deliberately distinct volumes/spacing: donor anchor exceeds recipient X.
    donor, donor_regions = case_and_regions("UNIT-donor", (17, 18, 19), (2., 1., 3.))
    donor.label[12:14, 10:12, 9:11] = 2
    donor_source, _, _ = choose_source_tumor(donor.image, donor.label, tumor_label=2,
                                            rng=np.random.default_rng(4), pad=1)
    recipient, recipient_regions = case_and_regions("UNIT-recipient", (11, 12, 13), (1., 2., 1.5))
    recipient.label[2:5, 3:6, 4:7] = 2
    recipient.label[7:9, 8:10, 9:11] = 2
    target_source, _ = donor_in_target_spacing(donor_source, donor.spacing, recipient.spacing)
    config = GraphBuildConfig(num_regions=2, region_k=1, num_prototypes=2, prototype_k=1)
    bank = PrototypeBank(np.tile(np.linspace(.1, 1., 18, dtype=np.float32), (2, 1)),
        np.zeros((2, 16), np.float32), np.zeros(16, np.float32), np.ones(16, np.float32),
        np.asarray([[0, 1], [1, 0]], np.int64), ("UNIT-donor", "UNIT-training-other"))
    specs = []
    for center in list(product(range(1, 10), range(1, 11), range(1, 12)))[:candidate_count]:
        specs.append(CandidateSpec(center=center, difficulty=1, corruption=0,
            region_id=recipient_regions.region_at(center), prototype_id=0,
            liver_coverage=1., border_distance_mm=float(recipient_regions.organ_depth[center]),
            occupied_distance_mm=1., context_mean_hu=float(recipient.image[center]), context_std_hu=2.))
    return dict(recipient_case=recipient, donor_case=donor, donor_source=donor_source,
        recipient_source=target_source, specs=specs, recipient_regions=recipient_regions,
        donor_regions=donor_regions, bank=bank, graph_config=config, ct_clip=(-200., 250.),
        training_case_ids=("UNIT-donor", "UNIT-training-other"))


class HistoricalPatientGraphUnit(unittest.TestCase):
    def test_true_donor_frame_and_descriptor_equations(self):
        inputs = unit_inputs()
        graph, _, audit = build_external_hierarchy(**inputs)
        donor, source, regions = (inputs[key] for key in ("donor_case", "donor_source", "donor_regions"))
        donor.shape = donor.image.shape
        expected = hierarchy._source_raw(donor, source, regions,
            (donor.label == 2) & ~source.full_mask, ct_clip=inputs["ct_clip"])
        np.testing.assert_array_equal(graph["tumor"].raw_x.numpy()[0], expected)
        np.testing.assert_array_equal(graph["tumor"].pos.numpy()[0],
                                      normalized_position(source.anchor_center, donor.shape))
        self.assertGreater(source.anchor_center[0], inputs["recipient_case"].image.shape[0] - 1)
        self.assertEqual(graph["tumor"].region_index.tolist(), [-1])
        self.assertEqual(graph["tumor"].source_region_provenance.tolist(),
                         [regions.region_at(source.anchor_center)])
        self.assertFalse(audit["exact_original_external_donor_path"])
        self.assertFalse(audit["source_host_incidence"])

    def test_all_recipient_lesions_retained_and_match_original_equations(self):
        inputs = unit_inputs()
        graph, _, audit = build_external_hierarchy(**inputs)
        case, regions = inputs["recipient_case"], inputs["recipient_regions"]
        case.shape = case.image.shape
        first, _, _ = choose_source_tumor(case.image, case.label, tumor_label=2,
                                          selection="largest", rng=np.random.default_rng(1), pad=0)
        other_mask = (case.label == 2) & ~first.full_mask
        # Both references use genuine actual components; no empty fake source.
        other = copy.deepcopy(first)
        other.full_mask = other_mask
        row_other, pos_other, reg_other = hierarchy._lesions(case, first, regions,
            tumor_label=2, max_lesions=None, ct_clip=inputs["ct_clip"])
        row_first, pos_first, reg_first = hierarchy._lesions(case, other, regions,
            tumor_label=2, max_lesions=None, ct_clip=inputs["ct_clip"])
        np.testing.assert_array_equal(graph["lesion"].raw_x.numpy(), np.concatenate([row_first, row_other]))
        np.testing.assert_array_equal(graph["lesion"].pos.numpy(), np.concatenate([pos_first, pos_other]))
        np.testing.assert_array_equal(graph["lesion"].region_index.numpy(), np.concatenate([reg_first, reg_other]))
        self.assertEqual(audit["annotation_exposure"]["recipient_lesion_nodes"], 2)
        self.assertFalse(audit["annotation_blind"])
        self.assertIn("Observed P anchors", audit["annotation_exposure"]["warning"])

    def test_complete_130_candidates_dimensions_direction_and_original_edges(self):
        inputs = unit_inputs(130)
        graph, prototype, audit = build_external_hierarchy(**inputs)
        self.assertEqual(graph["candidate"].raw_x.shape, (130, 14))
        self.assertEqual(graph["region"].raw_x.shape[1], 16)
        self.assertEqual(prototype["prototype"].raw_x.shape[1], 18)
        self.assertEqual(set(graph.edge_types), set(PATIENT_EDGE_TYPES))
        self.assertEqual(set(prototype.edge_types), set(PROTOTYPE_EDGE_TYPES))
        self.assertEqual(graph["candidate", "near", "lesion"].edge_index.shape, (2, 260))
        self.assertEqual(audit["candidate_count"], 130)
        expected = hierarchy._knn(graph["candidate"].pos.numpy(), inputs["graph_config"].candidate_k)
        np.testing.assert_array_equal(graph["candidate", "spatial_neighbor", "candidate"].edge_index, expected)
        for edge_type in PATIENT_EDGE_TYPES:
            store = graph[edge_type]
            self.assertEqual(store.edge_attr.shape[1], 12)
            expected = hierarchy._patient_edge_attributes(graph, edge_type, store.edge_index.numpy())
            np.testing.assert_array_equal(store.edge_attr.numpy(), expected)

    def test_candidate_equations_use_actual_recipient_spacing_footprint(self):
        inputs = unit_inputs()
        graph, _, audit = build_external_hierarchy(**inputs)
        case = inputs["recipient_case"]
        case.shape = case.image.shape
        axis, anisotropy = hierarchy._principal_axis(inputs["donor_source"].full_mask,
                                                     inputs["donor_case"].spacing)
        expected = hierarchy._candidate_raw(case, inputs["recipient_source"], inputs["specs"][0],
            inputs["recipient_regions"], source_axis=axis, source_anisotropy=anisotropy,
            ct_clip=inputs["ct_clip"])
        np.testing.assert_array_equal(graph["candidate"].raw_x.numpy()[0], expected)
        self.assertNotEqual(audit["donor_mask_voxels"], audit["recipient_regridded_mask_voxels"])

    def test_prototype_is_exact_original_builder_and_no_source_mutation(self):
        inputs = unit_inputs()
        original_image, original_label = inputs["recipient_case"].image.copy(), inputs["recipient_case"].label.copy()
        graph, prototype, audit = build_external_hierarchy(**inputs)
        expected = hierarchy.build_prototype_graph(inputs["specs"], graph, inputs["recipient_regions"],
                                                   inputs["bank"], config=inputs["graph_config"])
        for node_type in prototype.node_types:
            for key, value in prototype[node_type].items():
                self.assertTrue(torch.equal(value, expected[node_type][key]))
        for edge_type in prototype.edge_types:
            self.assertTrue(torch.equal(prototype[edge_type].edge_index, expected[edge_type].edge_index))
            self.assertTrue(torch.equal(prototype[edge_type].edge_attr, expected[edge_type].edge_attr))
        np.testing.assert_array_equal(inputs["recipient_case"].image, original_image)
        np.testing.assert_array_equal(inputs["recipient_case"].label, original_label)
        self.assertFalse(audit["P_U_labels_in_forward"])
        self.assertEqual(audit["bank_fingerprint"], inputs["bank"].fingerprint())

    def test_rejects_bank_fit_on_held_out_recipient_or_wrong_training_set(self):
        for altered in (("UNIT-donor", "UNIT-recipient"), ("UNIT-donor",)):
            inputs = unit_inputs()
            inputs["bank"].training_case_ids = altered
            with self.assertRaisesRegex(ValueError, "training-only"):
                build_external_hierarchy(**inputs)

    def test_production_rejects_real_subset_and_explicit_debug_preserves_it(self):
        inputs = unit_inputs()
        # Actual UNIT bank provenance remains a one-case subset; never relabel
        # it as the complete training population merely to satisfy admission.
        inputs["bank"].training_case_ids = ("UNIT-training-other",)
        with self.assertRaisesRegex(ValueError, "training-only"):
            build_external_hierarchy(**inputs)
        graph, _, audit = build_external_hierarchy(**inputs, debug=True)
        self.assertEqual(graph["candidate"].raw_x.shape[0], len(inputs["specs"]))
        self.assertEqual(inputs["bank"].training_case_ids, ("UNIT-training-other",))
        self.assertTrue(audit["debug"])
        self.assertTrue(audit["prototype_training_subset_in_DEBUG"])
        self.assertFalse(audit["quality_verified"])
        self.assertEqual(audit["prototype_training_case_ids"], ["UNIT-training-other"])
        self.assertEqual(audit["declared_training_case_count"], 2)
        self.assertEqual(audit["prototype_fitted_training_case_count"], 1)

    def test_debug_still_rejects_held_out_bank_and_nontraining_donor(self):
        inputs = unit_inputs()
        inputs["bank"].training_case_ids = ("UNIT-recipient",)
        with self.assertRaisesRegex(ValueError, "training-only"):
            build_external_hierarchy(**inputs, debug=True)
        inputs = unit_inputs()
        inputs["training_case_ids"] = ("UNIT-training-other",)
        inputs["bank"].training_case_ids = ("UNIT-training-other",)
        with self.assertRaisesRegex(ValueError, "training-only"):
            build_external_hierarchy(**inputs, debug=True)
        inputs = unit_inputs()
        with self.assertRaisesRegex(ValueError, "explicit boolean"):
            build_external_hierarchy(**inputs, debug="True")

    def test_rejects_cropped_donor_and_forged_recipient_patch(self):
        inputs = unit_inputs()
        source = copy.deepcopy(inputs["donor_source"])
        source.full_mask[12, 10, 9] = False
        source.patch_mask = source.full_mask[source.patch_slices].copy()
        source.voxel_count = int(source.full_mask.sum())
        inputs["donor_source"] = source
        with self.assertRaisesRegex(ValueError, "complete actual annotated component"):
            build_external_hierarchy(**inputs)
        inputs = unit_inputs()
        target = copy.deepcopy(inputs["recipient_source"])
        target.patch_image.flat[0] += 1
        inputs["recipient_source"] = target
        with self.assertRaisesRegex(ValueError, "actual donor spacing transport"):
            build_external_hierarchy(**inputs)

    def test_rejects_wrong_region_and_outside_liver(self):
        inputs = unit_inputs()
        inputs["specs"][0] = replace(inputs["specs"][0], region_id=1)
        with self.assertRaisesRegex(ValueError, "recipient center and region"):
            build_external_hierarchy(**inputs)
        inputs = unit_inputs()
        inputs["specs"][0] = replace(inputs["specs"][0], center=(99, 0, 0))
        with self.assertRaisesRegex(ValueError, "recipient center and region"):
            build_external_hierarchy(**inputs)

    def test_lesion_guard_fails_without_silent_deletion(self):
        inputs = unit_inputs()
        inputs["graph_config"] = replace(inputs["graph_config"], max_lesions=1)
        before = inputs["recipient_case"].label.copy()
        with self.assertRaisesRegex(RuntimeError, "no lesions were dropped"):
            build_external_hierarchy(**inputs)
        np.testing.assert_array_equal(inputs["recipient_case"].label, before)


if __name__ == "__main__":
    unittest.main()
