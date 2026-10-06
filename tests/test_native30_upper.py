"""Synthetic geometry UNIT checks using byte-exact legacy source operators.

These tests exercise the graph adaptation mechanics and real PyG containers.
They are not actual-data inference, trained-model validation or quality evidence.
The original region cache loader is replaced only inside this UNIT profile.
"""
from __future__ import annotations

import copy
import hashlib
import importlib
from itertools import product
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scipy import ndimage as ndi
import torch

from hiercp_v1x.native30_upper import (_bank_audit, _runtime, _whole_case,
                                      build_native30_upper)


ROOT = Path(__file__).resolve().parents[1]
UNIT_LEGACY_REVISION = "a818158"


class ForwardInputRow(dict):
    """Make accidental P/U target access fail inside the graph builder."""
    def __getitem__(self, key):
        if key in {"class", "target", "label", "is_positive", "kind"}:
            raise AssertionError("P/U targets may not be read by the upper forward adapter")
        return super().__getitem__(key)


class Native30UpperUnit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="native30_upper_UNIT_")
        cls.source = Path(cls.temporary.name) / "original_source"
        package = cls.source / "hiercp"
        package.mkdir(parents=True)
        # The snapshot is a separate UNIT fixture. Production always uses the
        # caller's explicitly activated source and never this Git revision.
        for name in ("__init__", "common", "curriculum", "hierarchy", "prototype", "region", "schema", "tensor",
                     "model", "sample"):
            data = subprocess.run(["git", "-c", "safe.directory=" + ROOT.as_posix(), "show",
                                   f"{UNIT_LEGACY_REVISION}:hiercp/{name}.py"], cwd=ROOT,
                                  check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout
            (package / (name + ".py")).write_bytes(data)
        cls.full_model_configuration = json.loads(subprocess.run(
            ["git", "-c", "safe.directory=" + ROOT.as_posix(), "show",
             f"{UNIT_LEGACY_REVISION}:config/train.json"], cwd=ROOT, check=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout)["model"]
        cls.previous = {name: value for name, value in sys.modules.items()
                        if name == "hiercp" or name.startswith("hiercp.")}
        for name in cls.previous:
            del sys.modules[name]
        sys.path.insert(0, str(cls.source))
        cls.runtime, cls.source_hashes = _runtime(cls.source)

    @classmethod
    def tearDownClass(cls):
        for name in list(sys.modules):
            if name == "hiercp" or name.startswith("hiercp."):
                del sys.modules[name]
        sys.modules.update(cls.previous)
        sys.path.remove(str(cls.source))
        cls.temporary.cleanup()

    def case_regions(self, case_id, shape, spacing):
        image = (np.indices(shape).sum(axis=0) * 3).astype(np.float32)
        label = np.ones(shape, np.uint8)
        case = SimpleNamespace(paths=SimpleNamespace(case_id=case_id), image=image,
            label=label, spacing=np.asarray(spacing, np.float32), shape=shape)
        organ = np.ones(shape, bool)
        depth = ndi.distance_transform_edt(organ, sampling=spacing).astype(np.float32)
        region_labels = np.zeros(shape, np.int16)
        region_labels[shape[0] // 2:] = 1
        centers = np.asarray([[shape[0] / 4, shape[1] / 2, shape[2] / 2],
                              [3 * shape[0] / 4, shape[1] / 2, shape[2] / 2]], np.float32)
        positions = np.stack([self.runtime["common"].normalized_position(center, shape) for center in centers])
        features = np.tile(np.linspace(.1, .9, 16, dtype=np.float32), (2, 1))
        features[1] += .15
        features[:, 15] = 1.
        regions = self.runtime["region"].PatientRegionData(organ, depth, region_labels, features,
            positions.astype(np.float32), np.asarray([[0, 1], [1, 0]], np.int64), centers)
        return case, regions

    def inputs(self, candidate_count=130):
        donor, donor_regions = self.case_regions("UNIT-donor", (17, 18, 19), (2., 1., 3.))
        donor.label[12:15, 10:13, 9:12] = 2
        donor.label[2:4, 3:5, 4:6] = 2
        source, _, _ = self.runtime["common"].choose_source_tumor(donor.image, donor.label,
            tumor_label=2, selection="largest", rng=np.random.default_rng(4), pad=1)
        recipient, recipient_regions = self.case_regions("UNIT-val-00", (11, 12, 13), (1., 2., 1.5))
        recipient.label[2:5, 3:6, 4:7] = 2
        recipient.label[7:9, 8:10, 9:11] = 2
        # Explicit synthetic UNIT transport with a complete footprint and
        # different voxel count; never supplied as real-data evidence.
        target = SimpleNamespace(patch_mask=np.repeat(source.patch_mask, 2, axis=0),
            patch_image=np.repeat(source.patch_image, 2, axis=0))
        target.voxel_count = int(target.patch_mask.sum())
        config = self.runtime["schema"].GraphBuildConfig(num_regions=2, region_k=1,
            num_prototypes=2, prototype_k=1)
        training_ids = tuple(f"UNIT-val-{index:02d}" for index in range(17)) + ("UNIT-donor",) + tuple(
            f"UNIT-original-train-{index:03d}" for index in range(87))
        bank = self.runtime["prototype"].PrototypeBank(
            np.tile(np.linspace(.1, 1., 18, dtype=np.float32), (2, 1)),
            np.stack((np.zeros(16, np.float32), np.ones(16, np.float32))),
            np.zeros(16, np.float32), np.ones(16, np.float32),
            np.asarray([[0, 1], [1, 0]], np.int64), training_ids)
        bundle = SimpleNamespace(source=self.source, prototype_bank=bank,
            config=dict(seed=42, graph=config.to_dict(), ct_clip=[-200., 250.]),
            source_proof=dict(source_python_sha256={path.relative_to(self.source).as_posix():
                hashlib.sha256(path.read_bytes()).hexdigest() for path in (self.source / "hiercp").glob("*.py")}))
        centers = list(product(range(1, 10), range(1, 11), range(1, 12)))[:candidate_count]
        rows = [ForwardInputRow(id=f"UNIT-row-{index:03d}", case_id=recipient.paths.case_id,
                    donor_case_id=donor.paths.case_id, donor_component=source.component_id,
                    center=list(center), kind="P" if index < 2 else "U", label=index < 2)
                for index, center in enumerate(centers)]
        meta = dict(split=dict(inner_val=[f"UNIT-val-{index:02d}" for index in range(21)]))
        geometry = SimpleNamespace(source=self.source, rows=rows, meta=meta,
            case=lambda case_id: {recipient.paths.case_id: recipient, donor.paths.case_id: donor}[case_id],
            donor_source=lambda row: (source, target))
        return SimpleNamespace(bundle=bundle, geometry=geometry, rows=rows, donor=donor,
            recipient=recipient, source=source, target=target, config=config,
            recipient_regions=recipient_regions, donor_regions=donor_regions)

    def build(self, inputs, **kwargs):
        regions = {inputs.recipient.paths.case_id: inputs.recipient_regions,
                   inputs.donor.paths.case_id: inputs.donor_regions}
        def loader(case, **config):
            self.assertFalse(config["overwrite"])
            self.assertEqual(config["config"].to_dict(), inputs.config.to_dict())
            return regions[case.paths.case_id]
        with patch.object(self.runtime["region"], "load_or_build_patient_regions", side_effect=loader):
            return build_native30_upper(inputs.bundle, inputs.geometry, kwargs.get("rows", inputs.rows),
                output=Path(self.temporary.name) / "evaluation_output",
                **({"lesion_policy": kwargs["lesion_policy"]} if "lesion_policy" in kwargs else {}))

    def many_lesion_inputs(self):
        """34 recipient and 14 other donor components, each really disconnected."""
        inputs = self.inputs(candidate_count=131)
        inputs.recipient.label.fill(1)
        recipient_centers = list(product((1, 4, 7, 10), (1, 4, 7, 10), (1, 4, 7, 10)))[:34]
        for center in recipient_centers:
            inputs.recipient.label[center] = 2
        inputs.donor.label.fill(1)
        inputs.donor.label[inputs.source.full_mask] = 2
        donor_centers = list(product((1, 4, 7, 10), (1, 4, 7, 10), (16,)))[:14]
        for center in donor_centers:
            inputs.donor.label[center] = 2
        # Added real components change scipy's component numbering. Resolve the
        # unchanged largest source from the new complete annotation, rather than
        # retaining a stale synthetic component ID in the donor query contract.
        source, _, _ = self.runtime["common"].choose_source_tumor(inputs.donor.image, inputs.donor.label,
            tumor_label=2, selection="largest", rng=np.random.default_rng(4), pad=1)
        np.testing.assert_array_equal(source.full_mask, inputs.source.full_mask)
        inputs.source = source
        for row in inputs.rows:
            row["donor_component"] = source.component_id
        inputs.geometry.donor_source = lambda row: (inputs.source, inputs.target)
        inputs.config = self.runtime["schema"].GraphBuildConfig(num_regions=2, region_k=1,
            num_prototypes=2, prototype_k=1, max_lesions=12)
        inputs.bundle.config["graph"] = inputs.config.to_dict()
        connectivity = ndi.generate_binary_structure(3, 1)
        self.assertEqual(ndi.label(inputs.recipient.label == 2, structure=connectivity)[1], 34)
        self.assertEqual(ndi.label((inputs.donor.label == 2) & ~inputs.source.full_mask,
                                  structure=connectivity)[1], 14)
        return inputs

    def test_complete_cohort_and_loaded_legacy_relation_schema(self):
        inputs = self.inputs()
        graph, prototype, audit = self.build(inputs)
        self.assertEqual(graph["candidate"].raw_x.shape, (130, 14))
        self.assertEqual(graph["region"].raw_x.shape, (4, 16))
        self.assertEqual(graph["liver"].raw_x.shape, (2, 14))
        self.assertEqual(set(graph.edge_types), set(self.runtime["schema"].PATIENT_EDGE_TYPES))
        self.assertEqual(set(prototype.edge_types), set(self.runtime["schema"].PROTOTYPE_EDGE_TYPES))
        self.assertNotIn(("tumor", "conditions", "region"), graph.edge_types)
        self.assertEqual(audit["candidate_record_ids"], [row["id"] for row in inputs.rows])
        self.assertTrue(audit["annotation_aware_evaluation"])
        self.assertFalse(audit["original_single_patient_topology_equivalent"])
        self.assertFalse(audit["P_U_labels_in_forward"])

    def test_source_feature_equations_and_host_incidence_keep_real_donor_frame(self):
        inputs = self.inputs()
        graph, _, audit = self.build(inputs)
        h = self.runtime["hierarchy"]
        expected = h._source_raw(inputs.donor, inputs.source, inputs.donor_regions,
            (inputs.donor.label == 2) & ~inputs.source.full_mask, ct_clip=(-200., 250.))
        np.testing.assert_array_equal(graph["tumor"].raw_x.numpy()[0], expected)
        np.testing.assert_array_equal(graph["tumor"].pos.numpy()[0],
            h.normalized_position(inputs.source.anchor_center, inputs.donor.shape))
        self.assertGreater(inputs.source.anchor_center[0], inputs.recipient.shape[0] - 1)
        actual_region = inputs.recipient_regions.num_regions + inputs.donor_regions.region_at(inputs.source.anchor_center)
        self.assertEqual(graph["tumor"].region_index.tolist(), [actual_region])
        self.assertEqual(graph["tumor", "hosted_by", "region"].edge_index.tolist(), [[0], [actual_region]])
        self.assertEqual(graph["region"].frame_index[actual_region].item(), 1)
        self.assertTrue(audit["source_host_incidence"])

    def test_recipient_all_lesions_donor_other_lesions_and_physical_edges(self):
        inputs = self.inputs()
        graph, _, audit = self.build(inputs)
        self.assertEqual(audit["recipient_lesion_count"], 2)
        self.assertEqual(audit["donor_other_lesion_count"], 1)
        self.assertEqual(graph["lesion"].frame_index.tolist(), [0, 0, 1])
        self.assertEqual(graph["candidate", "near", "lesion"].edge_index.shape, (2, 260))
        self.assertEqual(graph["tumor", "coexists_with", "lesion"].edge_index.tolist(), [[0], [2]])
        cross_frame = {("tumor", "compatible_with", "candidate"), ("candidate", "matched_to", "tumor")}
        for edge_type in graph.edge_types:
            if edge_type in cross_frame:
                continue
            source, _, destination = edge_type
            index = graph[edge_type].edge_index
            self.assertTrue(torch.equal(graph[source].frame_index[index[0]],
                                        graph[destination].frame_index[index[1]]), edge_type)

    def test_all_edge_features_are_actual_original_equations(self):
        inputs = self.inputs()
        graph, _, _ = self.build(inputs)
        h = self.runtime["hierarchy"]
        for edge_type in graph.edge_types:
            store = graph[edge_type]
            expected = h._patient_edge_attributes(graph, edge_type, store.edge_index.numpy())
            np.testing.assert_array_equal(store.edge_attr.numpy(), expected)
        expected_neighbors = h._knn(graph["candidate"].pos.numpy(), inputs.config.candidate_k)
        np.testing.assert_array_equal(graph["candidate", "spatial_neighbor", "candidate"].edge_index.numpy(), expected_neighbors)

    def test_candidate_feature_equations_and_complete_transported_volume(self):
        inputs = self.inputs()
        graph, _, audit = self.build(inputs)
        from hiercp_v1x.native30_upper import _candidate_specs
        specs = _candidate_specs(self.runtime, inputs.recipient, inputs.target,
            inputs.recipient_regions, inputs.rows, inputs.bundle.prototype_bank, inputs.config, (-200., 250.))
        axis, anisotropy = self.runtime["hierarchy"]._principal_axis(inputs.source.full_mask, inputs.donor.spacing)
        expected = self.runtime["hierarchy"]._candidate_raw(inputs.recipient, inputs.target, specs[0],
            inputs.recipient_regions, source_axis=axis, source_anisotropy=anisotropy, ct_clip=(-200., 250.))
        np.testing.assert_array_equal(graph["candidate"].raw_x.numpy()[0], expected)
        self.assertEqual(audit["recipient_regridded_mask_voxels"], 2 * audit["donor_mask_voxels"])

    def test_original_full_bank_is_retained_with_17_of_21_overlap_disclosed(self):
        inputs = self.inputs()
        fingerprint = inputs.bundle.prototype_bank.fingerprint()
        _, prototype, audit = self.build(inputs)
        self.assertEqual(audit["prototype_fitted_case_count"], 105)
        self.assertEqual(audit["prototype_current_validation_overlap_case_count"], 17)
        self.assertEqual(audit["current_validation_case_count"], 21)
        self.assertFalse(audit["strict_current_validation_held_out_claim"])
        self.assertEqual(audit["bank_fingerprint"], fingerprint)
        self.assertEqual(inputs.bundle.prototype_bank.fingerprint(), fingerprint)
        # Both real frame region descriptors use the original population operator.
        expected = np.concatenate((inputs.recipient_regions.region_features, inputs.donor_regions.region_features))
        np.testing.assert_array_equal(prototype["region"].raw_x.numpy(), expected)
        assignments, _ = inputs.bundle.prototype_bank.assign(expected,
            top_k=inputs.config.prototype_top_m, temperature=inputs.config.prototype_temperature)
        np.testing.assert_array_equal(prototype["region", "assigned_to", "prototype"].edge_index.numpy()[1], assignments.ravel())

    def test_rejects_subset_chunk_reordered_or_changed_native_assignments(self):
        inputs = self.inputs()
        for rows in (inputs.rows[:-1], inputs.rows[::-1], inputs.rows + [inputs.rows[0]]):
            with self.assertRaisesRegex(ValueError, "entire unchanged ordered case"):
                _whole_case(inputs.geometry, rows)
        rows = copy.deepcopy(inputs.rows)
        rows[0]["center"] = [8, 8, 8]
        with self.assertRaisesRegex(ValueError, "entire unchanged ordered case"):
            _whole_case(inputs.geometry, rows)

    def test_rejects_old_bank_subset_without_relabeling_it(self):
        inputs = self.inputs()
        inputs.bundle.prototype_bank.training_case_ids = inputs.bundle.prototype_bank.training_case_ids[:-1]
        with self.assertRaisesRegex(ValueError, "complete original 105-case"):
            _bank_audit(inputs.bundle.prototype_bank, inputs.geometry.meta)

    def test_explicit_debug_accepts_actual_training_bank_without_production_relaxation(self):
        inputs = self.inputs()
        ids = ("UNIT-donor", "UNIT-original-train-000")
        inputs.bundle.prototype_bank.training_case_ids = ids
        inputs.bundle.receipt = {"debug": True}
        inputs.geometry.meta["split"]["inner_train"] = list(ids)
        _, _, audit = self.build(inputs)
        self.assertTrue(audit["debug"])
        self.assertTrue(audit["prototype_training_subset_in_DEBUG"])
        self.assertEqual(audit["prototype_training_case_ids"], list(ids))
        self.assertFalse(audit["quality_verified"])
        self.assertFalse(audit["production_ready"])
        with self.assertRaisesRegex(ValueError, "complete original 105-case"):
            _bank_audit(inputs.bundle.prototype_bank, inputs.geometry.meta)
        inputs.bundle.prototype_bank.training_case_ids = (inputs.recipient.paths.case_id,)
        with self.assertRaisesRegex(ValueError, "only actual declared training"):
            _bank_audit(inputs.bundle.prototype_bank, inputs.geometry.meta, debug=True)

    def test_source_hashes_match_active_files_and_wrong_root_is_rejected(self):
        for name, digest in self.source_hashes.items():
            self.assertEqual(digest, hashlib.sha256((self.source / "hiercp" / (name + ".py")).read_bytes()).hexdigest())
        with self.assertRaisesRegex(ValueError, "explicit original source"):
            _runtime(ROOT)

    def test_source_receipt_mismatch_fails_before_any_region_build(self):
        inputs = self.inputs()
        inputs.bundle.source_proof["source_python_sha256"]["hiercp/hierarchy.py"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "source changed since native30 loading"):
            self.build(inputs)

    def test_inputs_are_not_mutated_and_no_lesion_guard_drops_components(self):
        inputs = self.inputs()
        before = inputs.recipient.label.copy(), inputs.donor.label.copy(), inputs.source.full_mask.copy()
        self.build(inputs)
        for actual, expected in zip((inputs.recipient.label, inputs.donor.label, inputs.source.full_mask), before):
            np.testing.assert_array_equal(actual, expected)
        inputs.config = self.runtime["schema"].GraphBuildConfig(num_regions=2, region_k=1,
            num_prototypes=2, prototype_k=1, max_lesions=1)
        inputs.bundle.config["graph"] = inputs.config.to_dict()
        with self.assertRaisesRegex(RuntimeError, "no lesions were dropped"):
            self.build(inputs)
        np.testing.assert_array_equal(inputs.recipient.label, before[0])

    def test_saved_lesion_guard_remains_the_default_for_large_actual_components(self):
        inputs = self.many_lesion_inputs()
        before = copy.deepcopy(inputs.bundle.config)
        for policy in (None, "saved_guard"):
            options = {} if policy is None else {"lesion_policy": policy}
            with self.assertRaisesRegex(RuntimeError, "detected_recipient_lesions=34"):
                self.build(inputs, **options)
            self.assertEqual(inputs.bundle.config, before)

    def test_explicit_all_observed_keeps_34_recipient_and_14_donor_other_components(self):
        inputs = self.many_lesion_inputs()
        before = copy.deepcopy(inputs.bundle.config)
        original_arrays = (inputs.recipient.label.copy(), inputs.donor.label.copy(),
                           inputs.source.full_mask.copy())
        graph, _, audit = self.build(inputs, lesion_policy="all_observed")
        self.assertEqual(graph["candidate"].raw_x.shape, (131, 14))
        self.assertEqual(graph["lesion"].raw_x.shape, (48, 14))
        self.assertEqual(graph["lesion"].frame_index.tolist(), [0] * 34 + [1] * 14)
        self.assertEqual(audit["recipient_lesion_count"], 34)
        self.assertEqual(audit["donor_other_lesion_count"], 14)
        self.assertEqual(len(audit["recipient_component_ids_in_size_order"]), 34)
        self.assertEqual(len(set(audit["recipient_component_ids_in_size_order"])), 34)
        self.assertEqual(audit["candidate_record_ids"], [row["id"] for row in inputs.rows])
        self.assertEqual(graph["candidate", "near", "lesion"].edge_index.shape, (2, 131 * 34))
        self.assertEqual(graph["lesion", "near", "candidate"].edge_index.shape, (2, 131 * 34))
        self.assertEqual(graph["tumor", "coexists_with", "lesion"].edge_index.shape, (2, 14))
        self.assertEqual(set(graph["tumor", "coexists_with", "lesion"].edge_index[1].tolist()),
                         set(range(34, 48)))
        expected_donor = self.runtime["hierarchy"]._lesions(inputs.donor, inputs.source,
            inputs.donor_regions, tumor_label=2, max_lesions=None, ct_clip=(-200., 250.))
        np.testing.assert_array_equal(graph["lesion"].raw_x.numpy()[34:], expected_donor[0])
        admission = audit["lesion_admission"]
        for key, expected in dict(policy="all_observed", saved_max_lesions=12,
                effective_max_lesions=None, recipient_components_retained=34,
                donor_other_components_retained=14, saved_guard_exceeded=True,
                components_dropped=0, saved_configuration_unchanged=True).items():
            self.assertEqual(admission[key], expected)
        self.assertEqual(audit["graph_config"]["max_lesions"], 12)
        self.assertEqual(inputs.bundle.config, before)
        self.assertFalse(audit["P_U_labels_in_forward"])
        self.assertFalse(audit["query_GT_in_forward"])
        self.assertFalse(audit["original_single_patient_topology_equivalent"])
        self.assertFalse(audit["quality_verified"])
        for actual, expected in zip((inputs.recipient.label, inputs.donor.label, inputs.source.full_mask),
                                    original_arrays):
            np.testing.assert_array_equal(actual, expected)

    def test_all_observed_matches_saved_guard_when_no_component_limit_is_exceeded(self):
        inputs = self.inputs()
        guarded, guarded_prototype, guarded_audit = self.build(inputs)
        unlimited, unlimited_prototype, unlimited_audit = self.build(inputs, lesion_policy="all_observed")
        for left, right in ((guarded, unlimited), (guarded_prototype, unlimited_prototype)):
            for node_type in left.node_types:
                np.testing.assert_array_equal(left[node_type].raw_x.numpy(), right[node_type].raw_x.numpy())
            for edge_type in left.edge_types:
                self.assertTrue(torch.equal(left[edge_type].edge_index, right[edge_type].edge_index), edge_type)
                self.assertTrue(torch.equal(left[edge_type].edge_attr, right[edge_type].edge_attr), edge_type)
        self.assertEqual(guarded_audit["lesion_admission"]["policy"], "saved_guard")
        self.assertFalse(unlimited_audit["lesion_admission"]["saved_guard_exceeded"])

    def test_unknown_lesion_policy_rejected_before_region_loading(self):
        inputs = self.inputs()
        for policy in ("all", "ALL_OBSERVED", "truncate", "", None, 12):
            with self.subTest(policy=policy):
                with patch.object(self.runtime["region"], "load_or_build_patient_regions") as loader:
                    with self.assertRaisesRegex(ValueError, "lesion.policy|lesion_policy"):
                        build_native30_upper(inputs.bundle, inputs.geometry, inputs.rows,
                            output=Path(self.temporary.name) / "bad_policy_output", lesion_policy=policy)
                    loader.assert_not_called()

    @unittest.skipUnless(torch.cuda.is_available(), "Actual CUDA UNIT required; no CPU neural fallback")
    def test_full_original_model_cuda_scores_all_131_queries_with_all_48_lesions(self):
        """Untrained CUDA UNIT upper-path mechanics, never CT/quality evidence."""
        from torch_geometric.data import Batch
        original_model = importlib.import_module("hiercp.model")
        self.assertEqual(Path(original_model.__file__).resolve(), self.source / "hiercp" / "model.py")
        model = original_model.HierarchicalPyGPlacementModel(**self.full_model_configuration).cuda().eval()
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 10_050_543)
        inputs = self.many_lesion_inputs()
        graph, prototype, audit = self.build(inputs, lesion_policy="all_observed")
        batch = SimpleNamespace(patient_batch=Batch.from_data_list([graph]).to("cuda"),
            prototype_batch=Batch.from_data_list([prototype]).to("cuda"), counts=(131,))
        values = torch.arange(131 * model.hidden_dim, device="cuda", dtype=torch.float32).reshape(131, -1)
        # Deterministic synthetic UNIT embeddings stand in for an already encoded
        # L0. Only the complete original L1/L2/scalar-head CUDA path is exercised.
        local = {key: torch.sin(values * .01 + index)
                 for index, key in enumerate(original_model._LOCAL_EMBEDDING_KEYS)}
        with torch.no_grad():
            scores = model._score_upper(batch, local)
        torch.cuda.synchronize()
        self.assertEqual(len(scores), 1)
        self.assertEqual(scores[0].shape, (131,))
        self.assertTrue(bool(torch.isfinite(scores[0]).all()))
        self.assertEqual(batch.patient_batch["lesion"].raw_x.shape[0], 48)
        self.assertEqual(audit["lesion_admission"]["components_dropped"], 0)
        print("CUDA UNIT upper path | actual GPU=" + torch.cuda.get_device_name()
              + " | full original parameters=10050543 | queries=131 | recipient lesions=34"
              + " | donor other lesions=14 | untrained synthetic UNIT; no quality claim", flush=True)


if __name__ == "__main__":
    unittest.main()
