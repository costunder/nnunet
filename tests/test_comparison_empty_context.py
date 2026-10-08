"""Synthetic CPU UNIT checks; not actual-CT or full-training evidence."""
from __future__ import annotations

import copy
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
from hiercp_v1x import comparison_empty_context as adapter


def geometry():
    shape = (5, 5, 5)
    footprint = np.zeros(shape, dtype=bool)
    footprint[2, 2, 2] = True
    organ = np.zeros(shape, dtype=bool)
    organ[0, 0, 0] = True
    depth = np.zeros(shape, dtype=np.float32)
    depth[0, 0, 0] = 2.
    fields = SimpleNamespace(footprint=footprint, organ_mask=organ,
        organ_depth=depth, outside_tumor_mm=np.full(shape, 3., dtype=np.float32))
    config = SimpleNamespace(context_inner_radius_mm=2., context_outer_radius_mm=10.,
        boundary_depth_mm=3., context_liver_surface_separation_mm=1.)
    points = dict(context=np.empty((0, 3), dtype=np.int64),
        liver_surface=np.array([[0, 0, 0]], dtype=np.int64))
    spatial = SimpleNamespace(_field=lambda fields, *names:
        next(getattr(fields, name) for name in names if hasattr(fields, name)))
    return fields, points, config, spatial


def local_records():
    node = lambda count: {"x": np.zeros((count, 16), dtype=np.float16)}
    source = {"nodes": {"source_context": node(1)}}
    target = {"nodes": {"target_context": node(0), "target_liver_surface": node(1)},
              adapter.PROOF_KEY: adapter.geometry_proof(*geometry())}
    return source, target


class ComparisonRecipientAbsenceUnit(unittest.TestCase):
    def test_exact_policy_and_full_mask_proof(self):
        proof = adapter.geometry_proof(*geometry())
        self.assertEqual(proof["semantic_context_voxels"], 0)
        self.assertEqual(proof["liver_surface_voxels"], 1)
        self.assertEqual(proof["canonical_liver_surface_nodes"], 1)
        self.assertEqual(proof["policy_sha256"], adapter.identity()["policy_sha256"])
        self.assertEqual(proof["minimum_liver_depth_exclusive_mm"], 4.)
        self.assertEqual(proof["context_outer_radius_mm"], 10.)

    def test_present_semantic_context_cannot_be_omitted(self):
        fields, points, config, spatial = geometry()
        fields.organ_mask[1, 1, 1] = True
        fields.organ_depth[1, 1, 1] = 5.
        with self.assertRaisesRegex(ValueError, "Present real recipient"):
            adapter.geometry_proof(fields, points, config, spatial)

    def test_real_surface_is_required(self):
        fields, points, config, spatial = geometry()
        fields.organ_mask[:] = False
        with self.assertRaisesRegex(ValueError, "actual liver-surface"):
            adapter.geometry_proof(fields, points, config, spatial)
        fields, points, config, spatial = geometry()
        points["liver_surface"] = np.empty((0, 3), dtype=np.int64)
        with self.assertRaisesRegex(ValueError, "nonempty real liver-surface"):
            adapter.geometry_proof(fields, points, config, spatial)

    def test_nonfinite_or_mismatched_fields_are_rejected(self):
        fields, points, config, spatial = geometry()
        fields.organ_depth[1, 1, 1] = np.nan
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            adapter.geometry_proof(fields, points, config, spatial)
        fields.organ_depth = np.zeros((4, 5, 5), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "inconsistent shapes"):
            adapter.geometry_proof(fields, points, config, spatial)

    def test_absent_canonical_requires_current_comparison_policy(self):
        source, target = local_records()
        self.assertEqual(adapter.validate_local_proof(source, target), target[adapter.PROOF_KEY])
        for field, value in (("policy_sha256", "0" * 64), ("comparison_format", "D-only"),
                             ("semantic_context_voxels", 1), ("canonical_liver_surface_nodes", 2)):
            changed = copy.deepcopy(target)
            changed[adapter.PROOF_KEY][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                adapter.validate_local_proof(source, changed)
        legacy = copy.deepcopy(target)
        legacy[adapter._geometry.PROOF_KEY] = legacy.pop(adapter.PROOF_KEY)
        with self.assertRaises(ValueError):
            adapter.validate_local_proof(source, legacy)

    def test_source_context_remains_mandatory(self):
        source, target = local_records()
        source["nodes"]["source_context"]["x"] = np.zeros((0, 16), np.float16)
        with self.assertRaisesRegex(ValueError, "source context"):
            adapter.validate_local_proof(source, target)

    def test_old_nonempty_canonical_needs_no_new_marker_and_stays_unchanged(self):
        source, target = local_records()
        target["nodes"]["target_context"]["x"] = np.zeros((1, 16), np.float16)
        target.pop(adapter.PROOF_KEY)
        keys = set(target)
        self.assertIsNone(adapter.validate_local_proof(source, target))
        self.assertEqual(set(target), keys)
        target[adapter.PROOF_KEY] = adapter.geometry_proof(*geometry())
        with self.assertRaisesRegex(ValueError, "Nonempty recipient"):
            adapter.validate_local_proof(source, target)

    def test_wrong_geometry_thresholds_do_not_enter_local_view(self):
        source, target = local_records()
        for field, value in (("context_outer_radius_mm", 11.),
                             ("context_inner_radius_mm", 1.),
                             ("minimum_liver_depth_exclusive_mm", 3.)):
            changed = copy.deepcopy(target)
            changed[adapter.PROOF_KEY][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                adapter.validate_local_proof(source, changed)

    def test_identity_includes_both_implementations(self):
        import hashlib
        descriptor = adapter.identity()
        self.assertEqual(descriptor["module_sha256"], hashlib.sha256(Path(adapter.__file__).read_bytes()).hexdigest())
        self.assertEqual(descriptor["geometry_validator_sha256"],
            hashlib.sha256(Path(adapter._geometry.__file__).read_bytes()).hexdigest())
        descriptor["policy"] = "mutated caller copy"
        self.assertEqual(adapter.identity()["policy"], adapter.POLICY)

    def test_actual_original_runtime_views_batches_gradients_and_restoration(self):
        result = subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()), "--runtime-child"],
            cwd=ROOT, text=True, capture_output=True, timeout=150)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("comparison empty context CPU UNIT PASS", result.stdout)


def runtime_child():
    """Byte-original full-size local encoder, synthetic semantic graph UNIT."""
    import hashlib
    import importlib
    import json
    from concurrent.futures import ThreadPoolExecutor
    from tempfile import TemporaryDirectory
    from threading import Barrier
    from zipfile import ZipFile

    import torch
    from torch_geometric.data import Batch

    from hiercp_v1x import bounded_scope as scope
    from hiercp_v1x.contracts import V1_ARCHIVE_SHA256

    torch.set_num_threads(2)
    torch.manual_seed(42)
    archive = ROOT / "versions/v1/pipeline_v1_source.zip"
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == V1_ARCHIVE_SHA256
    protected = [archive, Path(scope.__file__), Path(adapter._geometry.__file__)]
    before_hash = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in protected}
    with TemporaryDirectory(prefix="UNIT_comparison_empty_", dir=ROOT) as directory:
        snapshot = Path(directory)
        with ZipFile(archive) as bundle:
            for name in bundle.namelist():
                if (name.startswith("hiercp/") and name.endswith(".py")) or name == "config/train.json":
                    path = (snapshot / name).resolve()
                    assert path.is_relative_to(snapshot.resolve())
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(bundle.read(name))
        sys.path.insert(0, str(snapshot))
        receipt = scope.install(10, snapshot)
        runtime = {name: importlib.import_module("hiercp." + name)
                   for name in ("schema", "spatial", "local", "sample", "model", "cache")}
        runtime.update(scope=receipt, snapshot=snapshot)
        schema, spatial, local, sample, model = (runtime[name]
            for name in ("schema", "spatial", "local", "sample", "model"))
        settings = json.loads((snapshot / "config/train.json").read_text())
        config = scope.configure(settings["graph"], 10)

        def node(count):
            return dict(x=torch.zeros((count, 16), dtype=torch.float16),
                grid=torch.zeros((count, 3), dtype=torch.float16),
                pos=torch.zeros((count, 3)), pos_mm=torch.zeros((count, 3)))

        nodes = {role: node(1) for role in schema.LOCAL_NODE_TYPES}
        edges = {edge: torch.empty((2, 0), dtype=torch.int32) for edge in schema.LOCAL_EDGE_TYPES}
        base = dict(format="canonical-full-v22", geometry_contract=spatial.LEVEL0_GEOMETRY_CONTRACT,
            v1x_bounded_scope_contract=receipt["contract_sha256"], v1x_bounded_scope_margin_mm=10.)
        source = dict(base, nodes={k: v for k, v in nodes.items() if k in schema.SOURCE_LOCAL_NODE_TYPES},
            edges={k: v for k, v in edges.items()
                   if k[0] in schema.SOURCE_LOCAL_NODE_TYPES and k[2] in schema.SOURCE_LOCAL_NODE_TYPES})
        target = dict(base, nodes={k: v for k, v in nodes.items() if k not in schema.SOURCE_LOCAL_NODE_TYPES},
            edges={k: v for k, v in edges.items() if k not in source["edges"]}, transform=torch.eye(3))
        original_views = [sample.build_local_view(source, target, config, seed=seed) for seed in (42, 43)]
        kwargs = {k: v for k, v in settings["model"].items() if k in ("hidden_dim", "heads", "dropout",
            "dense_base_channels", "dense_feature_dim", "dense_batch_size", "channels_last_3d",
            "checkpoint_local_blocks", "checkpoint_dense_encoder")}
        net = model.LocalTumorContextPyGEncoder(**kwargs, layers=settings["model"]["local_layers"]).eval()
        parameter_shapes = {k: tuple(v.shape) for k, v in net.state_dict().items()}
        parameter_count = sum(v.numel() for v in net.parameters())
        source_map = torch.randn((2, settings["model"]["dense_feature_dim"], 3, 3, 3))
        target_map = torch.randn_like(source_map)
        with torch.no_grad():
            original_output = net.forward_graph(Batch.from_data_list(original_views), source_map, target_map)
        watched = [(spatial, "validate_canonical_coordinates"), (local, "_prepare_local_target"),
                   (local, "build_local_graph"), (sample, "build_local_view"),
                   (runtime["cache"], "build_local_graph"),
                   (model.LocalTumorContextPyGEncoder, "forward_graph"),
                   (model.LocalTumorContextPyGEncoder, "_pool_context_shells")]
        originals = {(id(owner), key): getattr(owner, key) for owner, key in watched}
        fields, points, _, _ = geometry()
        points.update(surface=np.array([[2, 2, 2]]), interior=np.array([[2, 2, 2]]))

        def rejected(call, expected=ValueError):
            try:
                call()
            except expected:
                return
            raise AssertionError("Invalid absent-context UNIT input was accepted")

        with adapter.activated(runtime) as policy:
            rejected(lambda: adapter.activated(runtime).__enter__(), RuntimeError)
            assert policy == adapter.identity()
            assert all(getattr(owner, key) is not originals[id(owner), key] for owner, key in watched)
            barrier = Barrier(2)

            def coordinate_request(is_target):
                token = adapter._TARGET.set(is_target)
                try:
                    barrier.wait()
                    try:
                        spatial.validate_canonical_coordinates(fields, points, config)
                    except spatial.EmptyCanonicalNodeError:
                        return "rejected"
                    return "admitted"
                finally:
                    adapter._TARGET.reset(token)

            with ThreadPoolExecutor(max_workers=2) as pool:
                tasks = [pool.submit(coordinate_request, value) for value in (True, False)]
                assert [task.result() for task in tasks] == ["admitted", "rejected"]
            rejected(lambda: spatial.validate_canonical_coordinates(fields, points, config), spatial.EmptyCanonicalNodeError)
            changed_views = [sample.build_local_view(source, target, config, seed=seed) for seed in (42, 43)]
            for before, after in zip(original_views, changed_views):
                for role in (*schema.LOCAL_NODE_TYPES, *schema.LOCAL_EDGE_TYPES):
                    for key, value in before[role].items():
                        assert torch.equal(value, after[role][key]), (role, key)
            with torch.no_grad():
                changed_output = net.forward_graph(Batch.from_data_list(changed_views), source_map, target_map)
            for key, value in original_output.items():
                assert torch.equal(value, changed_output[key]), key
            empty = copy.deepcopy(target)
            empty["nodes"]["target_context"] = node(0)
            empty[adapter.PROOF_KEY] = adapter.geometry_proof(fields, points, config, spatial)
            views = [sample.build_local_view(source, empty, config, seed=seed) for seed in (42, 43)]
            assert all(v["target_context"].x.shape == (0, 16) for v in views)
            for rows in ((changed_views[0], views[1]), (views[0], changed_views[1]), tuple(views)):
                batch = Batch.from_data_list(list(rows))
                assert batch[adapter.GRAPH_PROOF_KEY].shape == (2, 4)
                net.zero_grad(set_to_none=True)
                output = net.forward_graph(batch, source_map, target_map)
                assert all(v.shape == (2, 128) and torch.isfinite(v).all() for v in output.values())
                output["fused"].square().mean().backward()
                for number in range(3):
                    gradient = net.empty_context_shell["target_context_c" + str(number)].grad
                    assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0
            invalid = Batch.from_data_list(views)
            invalid[adapter.GRAPH_POLICY_KEY][0] = "0" * 64
            rejected(lambda: net.forward_graph(invalid, source_map, target_map))
            invalid = Batch.from_data_list(views)
            invalid[adapter.GRAPH_PROOF_KEY][0, 0] = 0
            rejected(lambda: net.forward_graph(invalid, source_map, target_map))
            invalid = Batch.from_data_list(views)
            invalid["source_context"].batch = torch.empty(0, dtype=torch.long)
            rejected(lambda: net.forward_graph(invalid, source_map, target_map))
            rejected(lambda: net._pool_context_shells("source_context", torch.empty((0, 128)),
                torch.empty((0, 16)), torch.empty(0, dtype=torch.long), 2), RuntimeError)
            assert {k: tuple(v.shape) for k, v in net.state_dict().items()} == parameter_shapes
            assert sum(v.numel() for v in net.parameters()) == parameter_count

        assert all(getattr(owner, key) is originals[id(owner), key] for owner, key in watched)
        rejected(lambda: sample.build_local_view(source, empty, config, seed=42), RuntimeError)
        # Exception teardown and a later independent entry are both reversible.
        try:
            with adapter.activated(runtime):
                raise RuntimeError("UNIT deliberate owner failure")
        except RuntimeError as error:
            assert str(error) == "UNIT deliberate owner failure"
        assert adapter._ACTIVE is None
        assert all(getattr(owner, key) is originals[id(owner), key] for owner, key in watched)
        assert {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in protected} == before_hash
    print("comparison empty context CPU UNIT PASS; original model parameters", parameter_count)


if __name__ == "__main__":
    if sys.argv[1:] == ["--runtime-child"]:
        runtime_child()
    else:
        unittest.main()
