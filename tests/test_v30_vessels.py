"""UNIT contracts only: generated fixtures are not actual CT or training data.

The production store still requires all 131 entries; the tests create an
explicit disposable 131-case UNIT inventory, never a production subset mode.
"""
from contextlib import contextmanager
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import uuid

import nibabel as nib
import numpy as np
import torch

from hiercp_v1x import v30_vessels as vessels
from tools.export_vessel_graph import graph_from_skeleton


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@contextmanager
def unit_inventory():
    workspace = Path(__file__).resolve().parents[1]
    root = workspace / "tmp" / ("UNIT_v30_vessels_" + uuid.uuid4().hex)
    root.mkdir(parents=True)
    try:
        (root / "imagesTr").mkdir()
        (root / "vessels").mkdir()
        affine = np.array([[0., -2., 0., 20.], [3., 0., 0., -10.],
                           [0., 0., -4., 80.], [0., 0., 0., 1.]])
        shape = (9, 8, 7)
        # A native three-point component and a real isolated component. This
        # checks complete graph retention rather than invented gap connections.
        xyz = np.array([[1, 2, 3], [2, 2, 3], [3, 2, 3], [6, 5, 4]], dtype=np.int32)
        graph, arrays = graph_from_skeleton(xyz, affine, np.array([1., 2., 3., 4.]), [1, 1, 1, 2])
        entries = []
        for number in range(131):
            case_id = f"liver_{number}"
            image = nib.Nifti1Image(np.full(shape, number, dtype=np.float32), affine)
            image.header.set_xyzt_units("mm")
            ct = root / "imagesTr" / (case_id + ".nii.gz")
            nib.save(image, ct)
            case_dir = root / "vessels" / case_id
            case_dir.mkdir()
            np.savez_compressed(case_dir / "centerline_graph.npz", **arrays)
            record = dict(graph, case_id=case_id, native_shape_xyz=list(shape),
                affine_voxel_to_ras_mm=affine.tolist(), complete_centerline_coverage=True,
                topology_audit_passed=True, native_resolution_preserved=True,
                all_components_requested=True, source_files={"ct": {"sha256": digest(ct)}},
                sampling_ratio=1.0, component_pruning=False, gap_bridging=False,
                graph_node_cap=None, graph_edge_cap=None, anatomy_validated=False,
                vessel_tumor_overlap_voxels=number,
                npz_schema={name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                            for name, value in arrays.items()})
            (case_dir / "graph.json").write_text(json.dumps(record), encoding="utf-8")
            files = {name: {"path": (case_dir / name).relative_to(root).as_posix(),
                            "bytes": (case_dir / name).stat().st_size, "sha256": digest(case_dir / name)}
                     for name in ("graph.json", "centerline_graph.npz")}
            entries.append(dict(case_id=case_id, ct={"path": ct.relative_to(root).as_posix(), "sha256": digest(ct)},
                files=files, status="PASS", topology_audit_passed=True, complete_centerline_coverage=True,
                coordinate_system="RAS", coordinate_units="mm", unknown_ct_units_assumed_mm=False,
                counts=record["counts"]))
        index = root / "vessels" / "index.json"
        index.write_text(json.dumps(dict(format="msd_liver_vessel_inputs_v1", path_base="..",
            source_dataset_training_cases=131, available_graph_cases=131,
            full_dataset_processed=True, cases=entries)), encoding="utf-8")
        yield root, index, affine, arrays
    finally:
        if root.parent.resolve() != (workspace / "tmp").resolve() or not root.name.startswith("UNIT_v30_vessels_"):
            raise RuntimeError("UNIT cleanup escaped its owned disposable fixture directory")
        for path in sorted(root.rglob("*"), key=lambda value: len(value.parts), reverse=True):
            if not path.resolve().is_relative_to(root.resolve()):
                raise RuntimeError("UNIT cleanup encountered a path outside its fixture root")
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
        root.rmdir()


def rewrite_payload(root, index, case_id, *, modify_arrays=None, modify_graph=None, modify_entry=None):
    """Mutate only the explicitly generated UNIT fixture and rebind its hashes."""
    document = json.loads(index.read_text(encoding="utf-8"))
    entry = next(row for row in document["cases"] if row["case_id"] == case_id)
    directory = root / "vessels" / case_id
    graph = json.loads((directory / "graph.json").read_text(encoding="utf-8"))
    if modify_arrays is not None:
        with np.load(directory / "centerline_graph.npz", allow_pickle=False) as payload:
            arrays = {name: payload[name] for name in payload.files}
        modify_arrays(arrays)
        np.savez_compressed(directory / "centerline_graph.npz", **arrays)
        graph["npz_schema"] = {name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                               for name, value in arrays.items()}
    if modify_graph is not None:
        modify_graph(graph)
    (directory / "graph.json").write_text(json.dumps(graph), encoding="utf-8")
    for name in ("centerline_graph.npz", "graph.json"):
        entry["files"][name].update(bytes=(directory / name).stat().st_size, sha256=digest(directory / name))
    if modify_entry is not None:
        modify_entry(entry)
    index.write_text(json.dumps(document), encoding="utf-8")


class V30VesselUnitTests(unittest.TestCase):
    def test_complete_disjoint_batch_preserves_points_edges_and_isolated_components(self):
        with unit_inventory() as (_, index, affine, arrays):
            store = vessels.VesselStore(index)
            ids = ("liver_0", "liver_1", "liver_130")
            bindings = {case_id: store.binding_from_source(case_id) for case_id in ids}
            batch = store.batch(ids, recipient_bindings=bindings)
            self.assertEqual(batch.node_features.shape, (12, 12))
            self.assertEqual(batch.edge_attr.shape, (12, 8))
            self.assertEqual(batch.case_ids, ids)
            self.assertEqual(batch.case_ptr.tolist(), [0, 4, 8, 12])
            self.assertEqual(batch.component_ptr.tolist(), [0, 2, 4, 6])
            self.assertEqual(batch.component_patient_index.tolist(), [0, 0, 1, 1, 2, 2])
            self.assertEqual(batch.component_index.tolist(), [0, 0, 0, 1, 2, 2, 2, 3, 4, 4, 4, 5])
            self.assertTrue(torch.equal(batch.node_patient_index[batch.edge_index[0]],
                                        batch.node_patient_index[batch.edge_index[1]]))
            np.testing.assert_allclose(batch.pos_mm[:4], nib.affines.apply_affine(affine, arrays["voxel_xyz"]))
            self.assertEqual(batch.audit["skeleton_edges"], 6)
            self.assertEqual(batch.audit["skeleton_components"], 6)
            self.assertEqual(store.audit()["inventory_cases"], 131)

    def test_edge_geometry_and_component_statistics_are_real(self):
        with unit_inventory() as (_, index, _, _):
            store = vessels.VesselStore(index)
            batch = store.load("liver_0", recipient_binding=store.binding_from_source("liver_0"))
            source, target = batch.edge_index
            torch.testing.assert_close(batch.edge_attr[:, :3], batch.pos_mm[target] - batch.pos_mm[source])
            torch.testing.assert_close(batch.edge_attr[:, 3], torch.full((4,), 3.))
            torch.testing.assert_close(batch.edge_attr[:, 4:7].norm(dim=1), torch.ones(4))
            torch.testing.assert_close(batch.edge_attr[:, 7], batch.node_features[target, 3] - batch.node_features[source, 3])
            torch.testing.assert_close(batch.node_features[:, 3], torch.tensor([1., 2., 3., 4.]))
            torch.testing.assert_close(batch.node_features[:, 4], torch.tensor([1., 2., 1., 0.]))
            torch.testing.assert_close(batch.node_features[:, 8], torch.tensor([2., 2., 2., 4.]))
            self.assertEqual(batch.node_features[-1, 10:].tolist(), [0., 0.])

    def test_cache_returns_independent_storage_and_never_recomputes_graph(self):
        with unit_inventory() as (_, index, _, _):
            store = vessels.VesselStore(index)
            binding = store.binding_from_source("liver_0")
            first = store.load("liver_0", recipient_binding=binding)
            first.node_features.fill_(-500)
            first.pos_mm.fill_(-500)
            first.edge_index.fill_(0)
            first.audit["patient_audits"][0]["ct_file_sha256"] = "changed UNIT returned copy"
            with patch.object(store, "_read_graph", side_effect=AssertionError("static graph recomputed")), \
                    patch.object(vessels, "_sha256", side_effect=AssertionError("source was rehashed on cache hit")):
                second = store.load("liver_0", recipient_binding=binding)
            self.assertTrue((second.node_features[:, 3] > 0).all())
            self.assertNotEqual(second.audit["patient_audits"][0]["ct_file_sha256"], "changed UNIT returned copy")
            with self.assertRaises(ValueError):
                store._cache["liver_0"].node_features.setflags(write=True)

    def test_wrong_recipient_case_ct_hash_shape_and_affine_are_rejected(self):
        with unit_inventory() as (_, index, _, _):
            store = vessels.VesselStore(index)
            binding = store.binding_from_source("liver_0")
            changed_affine = np.asarray(binding.affine_voxel_to_ras_mm).copy()
            changed_affine[0, 3] += 1
            wrong = (replace(binding, case_id="liver_1"), replace(binding, ct_sha256="f" * 64),
                     replace(binding, native_shape_xyz=(8, 8, 7)),
                     replace(binding, affine_voxel_to_ras_mm=changed_affine))
            for candidate in wrong:
                with self.subTest(candidate=candidate), self.assertRaisesRegex(ValueError, "identity|differs"):
                    store.load("liver_0", recipient_binding=candidate)

    def test_subset_inventory_and_duplicates_are_rejected(self):
        with unit_inventory() as (_, index, _, _):
            document = json.loads(index.read_text())
            document["cases"].pop()
            index.write_text(json.dumps(document))
            with self.assertRaisesRegex(ValueError, "complete successful 131"):
                vessels.VesselStore(index)
            document["cases"].append(document["cases"][0])
            index.write_text(json.dumps(document))
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                vessels.VesselStore(index)

    def test_changed_source_and_inventory_invalidate_cache(self):
        with unit_inventory() as (root, index, _, _):
            store = vessels.VesselStore(index)
            binding = store.binding_from_source("liver_0")
            store.load("liver_0", recipient_binding=binding)
            path = root / "vessels" / "liver_0" / "centerline_graph.npz"
            with path.open("ab") as stream:
                stream.write(b"UNIT payload mutation")
            with self.assertRaisesRegex(ValueError, "changed"):
                store.load("liver_0", recipient_binding=binding)
            with index.open("a") as stream:
                stream.write(" ")
            with self.assertRaisesRegex(ValueError, "inventory changed"):
                store.audit()

    def test_payload_sha_mismatch_is_not_a_silent_fallback(self):
        with unit_inventory() as (root, index, _, _):
            path = root / "vessels" / "liver_0" / "centerline_graph.npz"
            with path.open("ab") as stream:
                stream.write(b"UNIT corrupt bytes")
            store = vessels.VesselStore(index)
            with self.assertRaisesRegex(ValueError, "payload SHA256/size mismatch"):
                store.load("liver_0", recipient_binding=store.binding_from_source("liver_0"))

    def test_wrong_physical_coordinates_and_artificial_edge_are_rejected(self):
        for kind in ("coordinates", "edge"):
            with self.subTest(kind=kind), unit_inventory() as (root, index, _, _):
                def mutate(arrays):
                    if kind == "coordinates":
                        arrays["point_ras_mm"][0, 0] += .01
                    else:
                        arrays["skeleton_edge_index"] = np.concatenate((arrays["skeleton_edge_index"], [[2, 3]]))
                rewrite_payload(root, index, "liver_0", modify_arrays=mutate)
                store = vessels.VesselStore(index)
                with self.assertRaisesRegex(ValueError, "RAS positions|artificial connections"):
                    store.load("liver_0", recipient_binding=store.binding_from_source("liver_0"))

    def test_omitted_real_edges_and_incorrect_component_ownership_are_rejected(self):
        for kind in ("omitted_edge", "component"):
            with self.subTest(kind=kind), unit_inventory() as (root, index, _, _):
                def mutate(arrays):
                    if kind == "omitted_edge":
                        arrays["skeleton_edge_index"] = arrays["skeleton_edge_index"][:1]
                    else:
                        arrays["skeleton_component_id"][-1] = arrays["skeleton_component_id"][0]
                rewrite_payload(root, index, "liver_0", modify_arrays=mutate)
                store = vessels.VesselStore(index)
                with self.assertRaisesRegex(ValueError, "omit native neighbors|merge disconnected"):
                    store.load("liver_0", recipient_binding=store.binding_from_source("liver_0"))

    def test_recipient_tumor_overlap_metadata_never_enters_features(self):
        with unit_inventory() as (_, index, _, _):
            store = vessels.VesselStore(index)
            bindings = [store.binding_from_source(case_id) for case_id in ("liver_0", "liver_130")]
            first = store.load("liver_0", recipient_binding=bindings[0])
            second = store.load("liver_130", recipient_binding=bindings[1])
            torch.testing.assert_close(first.node_features, second.node_features)
            torch.testing.assert_close(first.edge_attr, second.edge_attr)
            self.assertFalse(first.audit["patient_audits"][0]["recipient_tumor_overlap_used"])
            self.assertFalse(first.audit["patient_audits"][0]["ct_intensity_features"])

    def test_retained_legacy_schema_is_validated_from_actual_native_geometry(self):
        with unit_inventory() as (root, index, _, _):
            rewrite_payload(root, index, "liver_1", modify_graph=lambda graph: graph.pop("npz_schema"),
                modify_entry=lambda entry: (entry.pop("status"), entry.pop("complete_centerline_coverage")))
            store = vessels.VesselStore(index)
            batch = store.load("liver_1", recipient_binding=store.binding_from_source("liver_1"))
            self.assertTrue(batch.audit["patient_audits"][0]["legacy_schema_without_npz_declaration"])
            self.assertEqual(batch.node_features.shape, (4, 12))

    def test_full_inventory_parallel_preload_and_single_device_transfer(self):
        with unit_inventory() as (_, index, _, _):
            store = vessels.VesselStore(index)
            bindings = {case_id: store.binding_from_source(case_id) for case_id in store.case_ids}
            report = store.preload(recipient_bindings=bindings, workers=4)
            self.assertEqual(report["cached_cases"], 131)
            batch = store.batch(store.case_ids, recipient_bindings=bindings).to("cpu", non_blocking=True)
            self.assertEqual(batch.node_features.shape, (524, 12))
            self.assertEqual(batch.edge_index.shape, (2, 524))
            self.assertEqual(batch.component_patient_index.shape, (262,))
            for name in ("node_features", "pos_mm", "edge_index", "edge_attr", "node_patient_index",
                         "component_index", "component_patient_index", "case_ptr", "component_ptr"):
                self.assertEqual(getattr(batch, name).device.type, "cpu")

    def test_native_xyz_candidate_conversion_uses_rotation_reflection_and_spacing(self):
        affine = np.array([[0., -2., 0., 20.], [3., 0., 0., -10.],
                           [0., 0., -4., 80.], [0., 0., 0., 1.]])
        xyz = torch.tensor([[1, 2, 3], [6, 5, 4]], dtype=torch.int64)
        result = vessels.voxel_to_ras_mm(xyz, affine)
        torch.testing.assert_close(result, torch.tensor([[16., -7., 68.], [10., 8., 64.]]))
        self.assertEqual(result.device, xyz.device)
        np.testing.assert_allclose(vessels.voxel_to_ras_mm(xyz.numpy(), affine), result.numpy())
        with self.assertRaisesRegex(ValueError, "last dimension"):
            vessels.voxel_to_ras_mm(np.zeros((2, 4)), affine)

    def test_explicit_relocated_native_CT_uses_existing_files_with_unchanged_sha(self):
        with unit_inventory() as (root, index, _, _):
            original = root / "imagesTr"
            relocated = root / "Medical" / "Task03_Liver" / "imagesTr"
            relocated.mkdir(parents=True)
            ct_paths = {}
            for number in range(131):
                case_id = f"liver_{number}"
                previous = original / (case_id + ".nii.gz")
                current = relocated / previous.name
                previous.rename(current)
                ct_paths[case_id] = current
            with self.assertRaisesRegex(ValueError, "existing file"):
                vessels.VesselStore(index)
            store = vessels.VesselStore(index, ct_paths=ct_paths)
            binding = store.binding_from_source("liver_0")
            graph = store.load("liver_0", recipient_binding=binding)
            self.assertEqual(graph.node_features.shape, (4, 12))
            self.assertEqual(graph.audit["patient_audits"][0]["ct_source_path"], str(ct_paths["liver_0"].resolve()))
            self.assertIn("relocation", store.audit()["ct_path_policy"])
            self.assertFalse((original / "liver_0.nii.gz").exists())
            self.assertEqual(store.index_sha256, digest(index))

    def test_relocated_CT_mapping_rejects_wrong_cohort_and_changed_content(self):
        with unit_inventory() as (root, index, _, _):
            ct_paths = {f"liver_{number}": root / "imagesTr" / f"liver_{number}.nii.gz" for number in range(131)}
            wrong_cohort = dict(ct_paths)
            wrong_cohort["different_UNIT_case"] = wrong_cohort.pop("liver_130")
            with self.assertRaisesRegex(ValueError, "same complete 131-case"):
                vessels.VesselStore(index, ct_paths=wrong_cohort)
            ct_paths["liver_0"] = ct_paths["liver_1"]
            store = vessels.VesselStore(index, ct_paths=ct_paths)
            with self.assertRaisesRegex(ValueError, "CT source SHA256 mismatch"):
                store.binding_from_source("liver_0")


if __name__ == "__main__":
    unittest.main()
