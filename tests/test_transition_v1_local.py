"""CPU metadata/geometry-mechanics UNIT tests; no neural quality assertion.

Synthetic metadata vectors are labeled UNIT. Real CT construction and actual
CUDA forward/loss/gradient/optimizer checks are separate DEBUG executions.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch_geometric.data import HeteroData

from hiercp_v1x import transition_v1_local as local
from hiercp_v1x.transition_v1_data import (
    FORMAT as DATA_FORMAT, NativeObservationDataset, OriginalInputProvider,
    assignment_digest, validate_population, _bytes, _sha, _publish_new_json,
)


def source_metadata():
    image = np.arange(12**3, dtype=np.float32).reshape(12, 12, 12)
    full = np.zeros_like(image, dtype=bool)
    full[3:6, 4:7, 5:8] = True
    label = np.ones_like(image, dtype=np.uint8)
    label[full] = 2
    slices = (slice(1, 8), slice(2, 9), slice(3, 10))
    affine = np.diag([2., 3., 4., 1.])
    affine[:3, 3] = [11., -7., 17.]
    case = SimpleNamespace(image=image, label=label, image_affine=affine,
                           spacing=np.array([2., 3., 4.]), paths=SimpleNamespace(case_id="UNIT-donor"))
    source = SimpleNamespace(full_mask=full, patch_mask=full[slices].copy(), patch_slices=slices,
        patch_image=image[slices].copy(), anchor_center=(4, 5, 6), component_id=1, voxel_count=27)
    return case, source


def full_population():
    # All105 cases/14102 rows are synthetic metadata. No CT or model execution.
    train = [f"UNIT-train-{index}" for index in range(84)]
    validation = [f"UNIT-validation-{index}" for index in range(21)]
    cases = train + validation
    rows, raw = [], []
    for ordinal, case in enumerate(cases):
        positives = 7 if ordinal < 32 else 6  # 662P + 105*128U =14102.
        donor = train[1] if case == train[0] else train[0]
        for index in range(128 + positives):
            rows.append(dict(id=f"{case}:{index}", case_id=case, patient_group="case:" + case,
                component=None if index < 128 else index - 127, center=[index, 0, 0], target=int(index >= 128),
                donor_case_id=donor, donor_component=1, donor_group="case:" + donor))
        raw.append(dict(case_id=case, positives=[dict(component=index+1) for index in range(positives)]))
    return dict(debug=False, records=rows, raw_records=raw,
                split=dict(inner_train=train, inner_val=validation, outer_train=cases),
                donor_pool=[dict(case_id=case, component_id=1) for case in train])


def graph_metadata(*, observation="UNIT-observation", view=0):
    graph = HeteroData()
    for role in local._ROLE_NAMES:
        graph[role].x = torch.zeros(1, 16)
        graph[role].grid = torch.zeros(1, 3)
    graph.transition_record_sha256 = "a" * 64
    graph.transition_observation_id = observation
    graph.transition_recipient_case = "UNIT-recipient"
    graph.transition_donor_case = "UNIT-donor"
    graph.transition_view_id = torch.tensor([view])
    graph.transition_epoch = torch.tensor([0])
    return graph


class TransitionLocalMetadataUnit(unittest.TestCase):
    def test_original_input_contract_is_not_ct_only_or_native_equivalence(self):
        contract = local.local_contract()
        self.assertEqual(contract["dense_input_shape"], [5, 48, 48, 48])
        self.assertEqual((contract["width"], contract["heads"], contract["local_layers"]), (128, 4, 3))
        self.assertEqual(contract["views_per_observation"], 2)
        self.assertEqual(contract["ROI_margin_mm"], 10.)
        self.assertEqual(contract["view_consistency_weight"], .1)
        self.assertFalse(contract["checkpoint_dense_encoder"])
        self.assertFalse(contract["checkpoint_local_blocks"])
        self.assertFalse(contract["native_equivalence"])
        self.assertFalse(contract["query_GT_in_forward"])
        self.assertIn("erasure", contract["target_CT"])
        self.assertEqual(local.source_identity(), local.source_identity())

    def test_source_audit_binds_actual_anchor_full_mask_ct_and_donor_mm(self):
        case, source = source_metadata()
        audit = local._source_audit(case, source)
        self.assertEqual(audit["original_mask_voxels"], 27)
        self.assertEqual(audit["original_anchor_native_ijk"], [4, 5, 6])
        self.assertEqual(audit["original_anchor_world_mm"], [19., 8., 41.])
        self.assertEqual(audit["original_full_mask_sha256"], local._hash_array(source.full_mask))
        for change in (
            lambda item: setattr(item, "anchor_center", (3, 5, 6)),
            lambda item: item.patch_mask.__setitem__((2, 2, 2), False),
            lambda item: item.patch_image.__setitem__((0, 0, 0), -9999),
            lambda item: setattr(item, "voxel_count", 26),
        ):
            altered = copy.deepcopy(source)
            change(altered)
            with self.assertRaises(ValueError):
                local._source_audit(case, altered)

    def test_physical_anchor_transport_records_both_orientation_frames(self):
        case, source = source_metadata()
        audit = local._source_audit(case, source)
        target_affine = np.array([[0., -2., 0., 29.], [1., 0., 0., -13.],
                                  [0., 0., 5., 3.], [0., 0., 0., 1.]])
        target = SimpleNamespace(spacing=np.array([1., 2., 5.]), image_affine=target_affine)
        center = np.array([2, 4, 6])
        result = local._physical_transport(audit, target, center, np.eye(3))
        world = np.asarray(result["donor_world_to_recipient_world_mm"])
        donor_anchor = np.asarray(audit["original_anchor_world_mm"])
        target_anchor = (target_affine @ np.r_[center, 1.])[:3]
        np.testing.assert_allclose(world @ np.r_[donor_anchor, 1.], np.r_[target_anchor, 1.])
        offset_mm = np.array([3., -7., 11.])
        target_frame = target_affine[:3, :3] / target.spacing[None]
        np.testing.assert_allclose((world @ np.r_[donor_anchor + offset_mm, 1.])[:3] - target_anchor,
                                   target_frame @ offset_mm)

    def test_query_class_metadata_is_excluded_from_neural_record_provenance(self):
        value = dict(observation_id="UNIT-observation", donor_group="UNIT-d", recipient_group="UNIT-r", debug=True,
                     target=1, query_GT=1, **{key: "a" * 64 for key in local._PROVENANCE_HASHES})
        files = dict(image_sha256="a" * 64, label_sha256="a" * 64)
        bound = local._provenance(value, donor_files=files, recipient_files=files)
        self.assertNotIn("target", bound)
        self.assertNotIn("query_GT", bound)
        value["donor_group"] = value["recipient_group"]
        with self.assertRaises(ValueError):
            local._provenance(value, donor_files=files, recipient_files=files)

    def test_content_digest_matches_native_and_detects_shared_tensor_mutation(self):
        from hiercp_v222.record_binding import graph_hash
        tensor = torch.arange(24, dtype=torch.float32).reshape(8, 3)
        record = dict(metadata="UNIT-only", tuple_key={('a', 'b', 'c'): tensor}, other=[True, 3, None])
        first = local._graph_hash(record)
        self.assertEqual(first, graph_hash(record))
        self.assertEqual(first, local._graph_hash(record))
        tensor[0, 0] += 1
        self.assertNotEqual(first, local._graph_hash(record))
        self.assertEqual(local._graph_hash(record), graph_hash(record))

    def test_complete_two_view_disjoint_collation_shares_one_original_source(self):
        source = torch.zeros(5, 48, 48, 48, dtype=torch.float16)
        target = torch.ones_like(source)
        items = []
        for index in range(2):
            views = tuple(graph_metadata(observation=f"UNIT-{index}", view=view) for view in (0, 1))
            items.append(((views, source, target), index))
        batch = local.collate(items)
        self.assertEqual(len(batch), 2)
        self.assertEqual(batch.graph.num_graphs, 4)
        self.assertEqual(batch.source_patches.shape, (1, 5, 48, 48, 48))
        self.assertEqual(batch.target_patches.shape, (2, 5, 48, 48, 48))
        self.assertTrue(torch.equal(batch.source_index, torch.tensor([0, 0])))
        self.assertTrue(torch.equal(batch.graph_observation_index, torch.tensor([0, 0, 1, 1])))
        self.assertTrue(torch.equal(batch.graph.transition_view_id, torch.tensor([0, 1, 0, 1])))
        items[0][0][0][1].transition_donor_case = "UNIT-wrong-donor"
        with self.assertRaises(ValueError):
            local.collate(items)

    def test_resident_byte_accounting_counts_shared_storage_once(self):
        value = torch.ones(1024)
        self.assertEqual(_bytes([value, value, value[:32]]), value.untyped_storage().nbytes())

    def test_full14102_population_and_all128U_are_required(self):
        meta = full_population()
        receipt = validate_population(meta, debug=False)
        self.assertEqual(receipt["total_assignments"], 14102)
        self.assertEqual(receipt["observed_P"], 662)
        self.assertEqual(receipt["unobserved_U"], 13440)
        self.assertTrue(receipt["full14102"])
        self.assertFalse(receipt["hidden_subset"])
        self.assertFalse(receipt["donor_redraw"])
        original = assignment_digest(meta["records"])
        meta["records"][0]["target"] = 1
        self.assertNotEqual(original, assignment_digest(meta["records"]))
        with self.assertRaises(ValueError):
            validate_population(meta, debug=False)

    def test_incremental_storage_resume_is_exact_and_preserves_uncommitted_files(self):
        # Pure mechanical UNIT: mocked graph construction/storage payloads,
        # actual file/hash/receipt publication. No CT/model/neural quality claim.
        rows = [dict(id=f"UNIT-case:{index}", case_id="UNIT-case", patient_group="UNIT-r",
                     component=None, center=[index, 0, 0], target=0, donor_case_id="UNIT-donor",
                     donor_component=1, donor_group="UNIT-d") for index in range(3)]
        with tempfile.TemporaryDirectory(prefix="UNIT_D_resume_", dir=local.ROOT / "work") as temporary:
            folder = Path(temporary)
            inventory = folder / "UNIT_native.json"
            inventory.write_text(json.dumps(rows), encoding="utf8")
            ds = object.__new__(NativeObservationDataset)
            ds.rows = rows
            ds.meta = dict(records=rows)
            ds.path, ds.index_sha256 = inventory, _sha(inventory)
            ds.assignment_sha256 = assignment_digest(rows)
            ds.debug, ds.base, ds.scope_contract, ds.case_ids = True, {}, "a" * 64, None
            ds.population = dict(debug=True, total_assignments=3)
            class UnitProvider(OriginalInputProvider):
                def _guard(self):
                    pass  # UNIT storage mechanics; resources are separately measured.
                def _records_for(self, requested):
                    self.prepared_ids = [row["id"] for row in requested]
                    return [dict(UNIT_payload=row["id"]) for row in requested]
            class UnitWriter:
                interrupted = True
                calls = 0
                def __init__(self, root, minimum_free_bytes):
                    self.root = root
                def write(self, relative, record, source_key):
                    file = self.root / relative
                    file.parent.mkdir(parents=True, exist_ok=True)
                    with file.open("xb") as stream:
                        stream.write(("UNIT_graph_file_" + record["UNIT_payload"]).encode())
                    source = self.root / "UNIT_shared_source.pt"
                    if not source.exists():
                        with source.open("xb") as stream:
                            stream.write(b"UNIT_metadata_storage_only")
                    type(self).calls += 1
                    if type(self).interrupted and type(self).calls == 2:
                        raise RuntimeError("UNIT explicit interrupted preparation")
                    return dict(path=relative, sha256=_sha(file), bounds=dict(nodes=0, edges=0, bytes=0),
                                shared_source=dict(path=source.name, sha256=_sha(source), content_sha256="b" * 64))
            output = folder / "canonical"
            unit_views = tuple(graph_metadata(view=view) for view in (0, 1))
            with patch("hiercp_v22.storage.GraphWriter", UnitWriter), patch.object(local, "materialize_pair", return_value=(unit_views, None, None)):
                first = UnitProvider(ds, workers=2, resident_bytes=1024**3, rss_bytes=2*1024**3)
                with self.assertRaisesRegex(RuntimeError, "UNIT explicit interrupted"):
                    first.preflight(output, minimum_free_bytes=1)
                self.assertFalse((output / "index.json").exists())
                self.assertEqual(len(list((output / "completed").glob("*.json"))), 1)
                uncommitted = list((output / "segments").rglob("000001.pt.gz"))[0]
                previous = uncommitted.read_bytes()
                UnitWriter.interrupted = False
                second = UnitProvider(ds, workers=2, resident_bytes=1024**3, rss_bytes=2*1024**3)
                result = second.preflight(output, minimum_free_bytes=1)
                self.assertEqual(second.prepared_ids, [rows[1]["id"], rows[2]["id"]])
                self.assertEqual(uncommitted.read_bytes(), previous)
                self.assertEqual(len(list((output / "completed").glob("*.json"))), 3)
                index = json.loads(result.read_text(encoding="utf8"))
                self.assertEqual(index["resumed_exact_completed_observations"], 1)
                self.assertEqual(index["prepared_observations"], 3)
                self.assertEqual(index["assignment_sha256"], ds.assignment_sha256)
                self.assertEqual([row["id"] for row in index["records"]], [row["id"] for row in rows])
                self.assertEqual(second.preflight(output, minimum_free_bytes=1), result)
                self.assertEqual(len(list((output / "segments").iterdir())), 2)
                self.assertEqual(index["format"], DATA_FORMAT)
            bound = folder / "UNIT_atomic.json"
            _publish_new_json(bound, {"UNIT": "original"})
            with self.assertRaises(FileExistsError):
                _publish_new_json(bound, {"UNIT": "replacement_refused"})
            self.assertEqual(json.loads(bound.read_text()), {"UNIT": "original"})
        meta = full_population()
        meta["records"].pop()
        with self.assertRaises(ValueError):
            validate_population(meta, debug=False)


if __name__ == "__main__":
    unittest.main()
