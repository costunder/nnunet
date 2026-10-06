"""UNIT geometry/storage contracts only; no learned model or quality evidence."""
from collections import OrderedDict
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
from torch_geometric.data import HeteroData

from hiercp_v1x.native30_geometry import (
    FORMAT, Native30Geometry, ROLES, _json_sha, _sha, collate_native30,
    physical_transport, resident_bytes_of,
)
from tests.test_transition_evaluation import unit_inventory


def unit_graph(record, view):
    graph = HeteroData()
    for role in ROLES:
        graph[role].x = torch.ones(2, 16)
        graph[role].grid = torch.zeros(2, 3)
        graph[role].pos = torch.zeros(2, 3)
    graph[ROLES[0], "UNIT_relation", ROLES[0]].edge_index = torch.tensor([[0], [1]])
    graph.native30_record_id = record
    graph.native30_view = torch.tensor([view])
    graph.native30_epoch = torch.tensor([0])
    return graph


class GeometryMathUnit(unittest.TestCase):
    def test_resident_counts_complete_shared_numpy_tensor_storage_once(self):
        array = np.arange(100, dtype=np.float32)
        tensor = torch.from_numpy(array)
        self.assertEqual(resident_bytes_of((array[10:15], tensor[30:33], {"same": array})), array.nbytes)
        independent = tensor.clone()
        self.assertEqual(resident_bytes_of((tensor, independent)), 2 * array.nbytes)

    def test_native_axis_transport_uses_two_affine_frames_and_exact_anchor(self):
        donor_affine = np.array([[0., -2., 0., 10.], [1., 0., 0., -5.], [0., 0., 3., 40.], [0., 0., 0., 1.]])
        target_affine = np.diag([-2., 1., 4., 1.])
        target_affine[:3, 3] = [100., 20., -80.]
        donor = SimpleNamespace(image_affine=donor_affine, spacing=np.array([1., 2., 3.]))
        target = SimpleNamespace(image_affine=target_affine, spacing=np.array([2., 1., 4.]))
        source = SimpleNamespace(anchor_center=(3, 4, 5))
        result = physical_transport(donor, target, source, [7, 8, 9])
        world = np.asarray(result["donor_world_to_recipient_world_mm"])
        np.testing.assert_allclose(world @ np.r_[result["donor_anchor_world_mm"], 1.],
                                   np.r_[result["recipient_anchor_world_mm"], 1.])
        np.testing.assert_array_equal(result["donor_relative_mm_to_recipient_relative_mm"], np.eye(3))
        self.assertFalse(np.allclose(world[:3, :3], np.eye(3)))

    def test_two_view_union_shares_donor_patch_and_preserves_observation_order(self):
        source = torch.ones(5, 48, 48, 48, dtype=torch.float16)
        first = ((unit_graph("UNIT_a", 0), unit_graph("UNIT_a", 1)), source, source.clone())
        second = ((unit_graph("UNIT_b", 0), unit_graph("UNIT_b", 1)), source, source.clone())
        batch = collate_native30([(first, 12), (second, 8)])
        self.assertEqual(len(batch), 2)
        self.assertEqual(batch.graph.num_graphs, 4)
        self.assertEqual(tuple(batch.source_patches.shape), (1, 5, 48, 48, 48))
        self.assertEqual(batch.source_index.tolist(), [0, 0])
        self.assertEqual(batch.graph_observation_index.tolist(), [0, 0, 1, 1])
        self.assertEqual(batch.indices.tolist(), [12, 8])
        self.assertEqual(batch.to("cpu").graph.native30_view.tolist(), [0, 1, 0, 1])
        self.assertNotIn("target", batch.graph.keys())
        self.assertNotIn("observed", batch.graph.keys())

    def test_wrong_view_ownership_or_real_empty_role_is_refused(self):
        source = torch.ones(5, 48, 48, 48, dtype=torch.float16)
        first, second = unit_graph("UNIT_a", 0), unit_graph("UNIT_b", 1)
        with self.assertRaises(ValueError):
            collate_native30([(((first, second), source, source), 0)])
        second.native30_record_id = "UNIT_a"
        first[ROLES[0]].x = torch.empty(0, 16)
        with self.assertRaises(ValueError):
            collate_native30([(((first, second), source, source), 0)])


class NativeGeometryAdmissionUnit(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="native30_geometry_UNIT_")
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.modules = {}
        for name in ("schema", "common", "local", "sample", "spatial", "curriculum"):
            path = self.source / (name + ".py")
            path.write_text("# UNIT source bytes, never actual model evidence\n", encoding="utf8")
            self.modules[name] = SimpleNamespace(__file__=str(path))
        self.config = dict(seed=42, ct_clip=[-200., 250.], cache=dict(source_pad=4),
            graph=dict(patch_size=48, adaptive_roi_margin_mm=30., context_outer_radius_mm=28.,
                       context_shells_mm=[4., 12., 28.], sample_hops=2))
        def graph_config(value):
            actual = copy.deepcopy(value)
            return SimpleNamespace(**actual, to_dict=lambda: copy.deepcopy(actual))
        self.modules["schema"].graph_config_from_dict = graph_config
        self.modules["local"]._require_full_graph = lambda _: None
        self.input = self.root / "inventory" / "index.json"
        self.input.parent.mkdir()
        self.runtime_patch = patch("hiercp_v1x.native30_geometry._runtime", return_value=self.modules)
        self.runtime_patch.start()

    def tearDown(self):
        self.runtime_patch.stop()
        if self.root.resolve().parent != Path(tempfile.gettempdir()).resolve() or not self.root.name.startswith("native30_geometry_UNIT_"):
            raise RuntimeError("Refusing cleanup outside explicit UNIT temporary directory")
        self.temp.cleanup()

    def geometry(self, *, production=False, case_ids=None, config=None, output=None):
        self.input.write_text(json.dumps(unit_inventory(production=production)), encoding="utf8")
        return Native30Geometry(self.input, self.config if config is None else config, self.source,
            workers=2, resident_bytes=1024**3, rss_bytes=8*1024**3,
            output=output or self.root / "new_geometry", debug=not production, debug_case_ids=case_ids)

    def published_unit_cache(self, *, production=False):
        """Explicit UNIT storage fixtures, never raw-CT/model/evaluation evidence."""
        geometry = self.geometry(production=production, case_ids=None if production else ["val_0"])
        dense = torch.zeros(1, dtype=torch.float16).expand(5, 48, 48, 48)
        geometry._canonical = {}
        for row in geometry.rows:
            key = _json_sha((row["donor_case_id"], row["donor_component"]))
            source_path = geometry.output / "shared_sources" / (key + ".pt")
            if not source_path.exists():
                source_path.parent.mkdir(exist_ok=True)
                torch.save(dict(UNIT_storage_fixture=True, source_local=dict(UNIT_source=True),
                                source_patch=dense), source_path)
            path = geometry.output / "records" / (_json_sha(row["id"]) + ".pt")
            path.parent.mkdir(exist_ok=True)
            torch.save(dict(UNIT_storage_fixture=True, format=FORMAT, case_id=row["case_id"],
                donor_case_id=row["donor_case_id"], component_id=row["donor_component"],
                center=row["center"], graph_config=geometry.graph_config.to_dict(), seed=42,
                input_provenance=dict(observation_id=row["id"]),
                target_local=dict(UNIT_record_id=row["id"]), target_patch=dense), path)
            receipt = dict(request_sha256=geometry.request_sha256, row=row, id=row["id"],
                path=path.relative_to(geometry.output).as_posix(), sha256=_sha(path),
                shared_source=dict(path=source_path.relative_to(geometry.output).as_posix(),
                                   sha256=_sha(source_path)),
                sampled_view_nodes=[12, 12], sampled_view_edges=[1, 1],
                sampled_two_view_nodes=24, sampled_two_view_edges=2,
                measurement_epoch=0, actual_original_views=True)
            ledger = geometry._receipt_path(row)
            ledger.parent.mkdir(exist_ok=True)
            ledger.write_text(json.dumps(receipt), encoding="utf8")
            geometry._canonical[row["id"]] = receipt
        (geometry.output / "index.json").write_text(json.dumps(geometry._index_summary()), encoding="utf8")
        return geometry

    def reopen_unit_cache(self, geometry, *, read_only=True):
        return Native30Geometry(self.input, geometry.config, self.source, workers=2,
            resident_bytes=1024**3, rss_bytes=8*1024**3, output=geometry.output,
            debug=geometry.debug, debug_case_ids=geometry.case_ids, read_only=read_only)

    @staticmethod
    def cache_snapshot(root):
        return {path.relative_to(root).as_posix(): (_sha(path), path.stat().st_mtime_ns)
                for path in root.rglob("*") if path.is_file()}

    def test_read_only_whole21_publication_never_writes_decodes_deserializes_or_builds(self):
        published = self.published_unit_cache(production=True)
        before = self.cache_snapshot(published.output)
        forbidden = AssertionError("UNIT: read-only preparation must not enter a write/build/load path")
        with patch("pathlib.Path.mkdir", side_effect=forbidden), \
                patch("hiercp_v1x.native30_geometry._write_new", side_effect=forbidden), \
                patch("hiercp_v1x.native30_geometry._save_new", side_effect=forbidden), \
                patch("torch.load", side_effect=forbidden), \
                patch.object(Native30Geometry, "_records_for", side_effect=forbidden), \
                patch.object(Native30Geometry, "_preload", side_effect=forbidden), \
                patch.object(Native30Geometry, "_sample", side_effect=forbidden), \
                patch.object(Native30Geometry, "_build", side_effect=forbidden), \
                patch.object(Native30Geometry, "_persist", side_effect=forbidden):
            reused = self.reopen_unit_cache(published)
            receipt = reused.prepare()
        self.assertEqual(len(reused.case_ids), 21)
        self.assertEqual(receipt["validation_records"], 2728)
        self.assertEqual(receipt["geometry_builds"], 0)
        self.assertEqual(receipt["geometry_loads"], 0)
        self.assertEqual(receipt["raw_cases_decoded"], [])
        self.assertTrue(receipt["preparation_reused"])
        self.assertTrue(receipt["read_only"])
        proof = receipt["reuse_provenance"]
        self.assertTrue(proof["all_payload_sha256_verified"])
        self.assertEqual(proof["validated_observations"], 2728)
        self.assertGreater(proof["validated_payload_files"], 2728)
        self.assertFalse(proof["existing_files_written"])
        self.assertEqual(receipt["sampled_two_view_summary"]["nodes"]["sum"], 2728*24)
        self.assertEqual(self.cache_snapshot(published.output), before)

    def test_read_only_missing_directory_request_index_or_receipt_is_refused_without_repair(self):
        published = self.published_unit_cache()
        nonexistent = self.root / "UNIT_missing_cache"
        with self.assertRaises(ValueError):
            Native30Geometry(self.input, published.config, self.source, workers=2,
                resident_bytes=1024**3, rss_bytes=8*1024**3, output=nonexistent,
                debug=True, debug_case_ids=published.case_ids, read_only=True)
        self.assertFalse(nonexistent.exists())
        for path in (published.output / "request.json", published.output / "index.json",
                     published._receipt_path(published.rows[-1])):
            data = path.read_bytes()
            path.unlink()
            before = self.cache_snapshot(published.output)
            with self.subTest(missing=path.name), self.assertRaises((ValueError, OSError)):
                self.reopen_unit_cache(published)
            self.assertEqual(self.cache_snapshot(published.output), before)
            path.write_bytes(data)

    def test_read_only_partial_reordered_or_corrupt_index_is_refused(self):
        published = self.published_unit_cache()
        path = published.output / "index.json"
        original = json.loads(path.read_text(encoding="utf8"))
        for kind in ("partial", "reordered", "duplicate", "complete_flag", "numeric_flag", "request_hash", "bad_count"):
            index = copy.deepcopy(original)
            if kind == "partial":
                index["records"].pop()
            elif kind == "reordered":
                index["records"].reverse()
            elif kind == "duplicate":
                index["records"][-1] = index["records"][0]
            elif kind == "complete_flag":
                index["complete"] = False
            elif kind == "numeric_flag":
                index["complete"] = 1
            elif kind == "request_hash":
                index["request_sha256"] = "0"*64
            else:
                index["validation_records"] -= 1
            path.write_text(json.dumps(index), encoding="utf8")
            before = self.cache_snapshot(published.output)
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                self.reopen_unit_cache(published)
            self.assertEqual(self.cache_snapshot(published.output), before)

    def test_read_only_every_target_and_shared_payload_sha_is_verified_on_admission(self):
        published = self.published_unit_cache()
        last = published._canonical[published.rows[-1]["id"]]
        for ref in (last, last["shared_source"]):
            path = published.output / ref["path"]
            original = path.read_bytes()
            path.write_bytes(original + b"UNIT_tamper")
            before = self.cache_snapshot(published.output)
            with self.subTest(payload=ref["path"]), self.assertRaises(ValueError):
                self.reopen_unit_cache(published)
            self.assertEqual(self.cache_snapshot(published.output), before)
            path.write_bytes(original)

    def test_read_only_receipt_measurements_and_extra_completion_are_refused(self):
        published = self.published_unit_cache()
        row = published.rows[0]
        ledger = published._receipt_path(row)
        original = copy.deepcopy(published._canonical[row["id"]])
        for kind in ("sum", "boolean", "one_view", "epoch", "boolean_epoch", "payload_path", "sha"):
            changed = copy.deepcopy(original)
            if kind == "sum":
                changed["sampled_two_view_nodes"] += 1
            elif kind == "boolean":
                changed["sampled_view_nodes"][0] = True
            elif kind == "one_view":
                changed["sampled_view_edges"].pop()
            elif kind == "epoch":
                changed["measurement_epoch"] = 1
            elif kind == "boolean_epoch":
                changed["measurement_epoch"] = False
            elif kind == "payload_path":
                changed["path"] = published._canonical[published.rows[1]["id"]]["path"]
            else:
                changed["sha256"] = "invalid"
            ledger.write_text(json.dumps(changed), encoding="utf8")
            published._canonical[row["id"]] = changed
            (published.output / "index.json").write_text(json.dumps(published._index_summary()), encoding="utf8")
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                self.reopen_unit_cache(published)
        ledger.write_text(json.dumps(original), encoding="utf8")
        published._canonical[row["id"]] = original
        (published.output / "index.json").write_text(json.dumps(published._index_summary()), encoding="utf8")
        extra = published.output / "completed_records" / "UNIT_extra.json"
        extra.write_text(json.dumps(original), encoding="utf8")
        with self.assertRaises(ValueError):
            self.reopen_unit_cache(published)

    def test_read_only_get_uses_mmap_and_current_two_view_sampler_without_raw_preload_or_persistence(self):
        published = self.published_unit_cache()
        reused = self.reopen_unit_cache(published)
        before = self.cache_snapshot(published.output)
        self.modules["common"].stable_case_seed = lambda seed, case, salt: int(_json_sha([seed, case, salt])[:8], 16)
        sampler = Mock(side_effect=lambda source, target, config, seed: unit_graph(target["UNIT_record_id"], 0))
        self.modules["sample"].build_local_view = sampler
        real_load = torch.load
        forbidden = AssertionError("UNIT: cached get must not rebuild, persist or decode raw CT")
        with patch("torch.load", wraps=real_load) as loads, \
                patch.object(reused, "_preload", side_effect=forbidden), \
                patch.object(reused, "_build", side_effect=forbidden), \
                patch.object(reused, "_persist", side_effect=forbidden):
            batch = reused.get([1, 0], epoch=3)
        self.assertEqual(batch.indices.tolist(), [1, 0])
        self.assertEqual(batch.graph.num_graphs, 4)
        self.assertEqual(batch.graph.native30_epoch.tolist(), [3, 3, 3, 3])
        self.assertEqual(sampler.call_count, 4)
        self.assertEqual(len({call.kwargs["seed"] for call in sampler.call_args_list}), 4)
        self.assertEqual(len(loads.call_args_list), 3)
        self.assertTrue(all(call.kwargs == dict(map_location="cpu", weights_only=False, mmap=True)
                            for call in loads.call_args_list))
        self.assertEqual(reused.builds, 0)
        self.assertEqual(reused.loads, 2)
        self.assertEqual(self.cache_snapshot(published.output), before)

    def test_read_only_runtime_coverage_loss_and_metadata_mutation_never_rebuild(self):
        published = self.published_unit_cache()
        reused = self.reopen_unit_cache(published)
        reused._canonical.pop(reused.rows[0]["id"])
        with patch.object(reused, "_build", side_effect=AssertionError("UNIT no rebuild")), \
                self.assertRaises(ValueError):
            reused.get([0])
        with self.assertRaises(ValueError):
            reused.prepare()
        reused = self.reopen_unit_cache(published)
        with self.assertRaises(RuntimeError):
            reused._build(reused.rows[0])
        with self.assertRaises(RuntimeError):
            reused._persist(reused.rows[0], {})
        path = published.output / "index.json"
        path.write_bytes(path.read_bytes() + b"\n")
        with self.assertRaises(ValueError):
            reused.guard()

    def test_writable_resume_still_loads_existing_complete_cache_and_keeps_build_capability(self):
        published = self.published_unit_cache()
        resumed = self.reopen_unit_cache(published, read_only=False)
        self.assertFalse(resumed.read_only)
        self.assertEqual(set(resumed._canonical), set(published._canonical))
        self.assertIsNone(resumed._read_only_index)

    def test_production_keeps_all21_cases_every_P128U_zero_P_case(self):
        geometry = self.geometry(production=True)
        self.assertEqual(len(geometry.case_ids), 21)
        self.assertEqual(sum(row["target"] == 0 for row in geometry.rows), 21*128)
        self.assertEqual(sum(row["target"] == 1 for row in geometry.rows), 40)
        self.assertTrue(geometry.population["production_full_inner_val"])
        self.assertTrue(geometry.request["no_training_preparation"])
        self.assertFalse(geometry.request["query_GT_in_forward"])
        self.assertEqual(geometry.request["rows"], geometry.rows)
        self.assertEqual(geometry._raw, {})

    def test_production_subset_is_refused_and_explicit_debug_retains_full_selected_case(self):
        with self.assertRaises(ValueError):
            self.geometry(production=True, case_ids=["val_0"])
        geometry = self.geometry(case_ids=["val_0"])
        self.assertEqual(len(geometry.rows), 130)
        self.assertEqual(geometry.case_ids, ["val_0"])
        self.assertTrue(geometry.debug)
        with self.assertRaises(ValueError):
            geometry.case("train_99")

    def test_native30_and_context28_are_both_required_without_scope_rewrite(self):
        for field, value in (("adaptive_roi_margin_mm", 10.), ("context_outer_radius_mm", 30.), ("patch_size", 32)):
            config = copy.deepcopy(self.config)
            config["graph"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.geometry(config=config)

    def test_changed_cache_contract_preserves_previous_request(self):
        geometry = self.geometry()
        request = geometry.output / "request.json"
        before = request.read_bytes()
        changed = copy.deepcopy(self.config)
        changed["graph"]["sample_hops"] = 3
        with self.assertRaises(ValueError):
            self.geometry(config=changed)
        self.assertEqual(request.read_bytes(), before)

    def test_inventory_and_original_source_mutations_are_refused(self):
        geometry = self.geometry()
        before = self.input.read_bytes()
        self.input.write_bytes(before + b"\n")
        with self.assertRaises(ValueError):
            geometry.guard()
        self.input.write_bytes(before)
        source = Path(self.modules["sample"].__file__)
        source.write_text("# UNIT mutated only within test\n", encoding="utf8")
        with self.assertRaises(ValueError):
            geometry.guard()

    def test_whole_active_record_cannot_be_evicted_to_disguise_budget_failure(self):
        geometry = self.geometry()
        tensor = torch.ones(100)
        record = dict(complete_tensor=tensor)
        geometry._records["UNIT"] = record
        geometry.resident_bytes = tensor.numel()*tensor.element_size()-1
        with self.assertRaises(MemoryError):
            geometry._make_room(active_records=[record])
        self.assertIs(geometry._records["UNIT"], record)
        self.assertEqual(len(geometry.rows), 388)

    def test_cached_tensor_file_tamper_or_escape_is_refused(self):
        geometry = self.geometry()
        path = geometry.output / "UNIT_record.pt"
        path.write_bytes(b"UNIT bytes")
        before = _sha(path)
        self.assertEqual(geometry._cache_file("UNIT_record.pt", before), path)
        path.write_bytes(b"UNIT changed bytes")
        with self.assertRaises(ValueError):
            geometry._cache_file("UNIT_record.pt", before)
        outside = self.root / "UNIT_outside.pt"
        outside.write_bytes(b"UNIT bytes")
        with self.assertRaises(ValueError):
            geometry._cache_file("../UNIT_outside.pt", _sha(outside))

    def test_query_geometry_identity_is_bound_but_GT_is_not_required_by_public_donor_query(self):
        geometry = self.geometry()
        actual = geometry.rows[0]
        query = {key: actual[key] for key in ("id", "case_id", "patient_group", "center", "donor_case_id",
                                             "donor_component", "donor_group")}
        self.assertIs(geometry._native_row(query), actual)
        query["donor_component"] += 1
        with self.assertRaises(ValueError):
            geometry._native_row(query)
        for ids in ([], [0, 0], [-1], [len(geometry)], [True]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                geometry.get(ids)


if __name__ == "__main__":
    unittest.main()
