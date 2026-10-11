"""Read the complete saved native vessel graphs and form disjoint batches.

No graph is regenerated, pruned, connected across a gap, or typed as an artery
or vein. Static, bytes-backed CPU arrays are cached; callers receive independent
tensor storage. Geometry is RAS millimetres, and the first three node features
are relative to the *CT field of view*, not an inferred organ boundary. CT
intensity and recipient tumor annotation/overlap never enter these features.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, fields
import copy
import hashlib
import itertools
import json
from pathlib import Path
import threading

import nibabel as nib
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
import torch


FORMAT = "v30_full_native_vessel_disjoint_graph_v1"
EXPECTED_CASES = 131
NODE_FEATURE_NAMES = (
    "ct_fov_relative_r", "ct_fov_relative_a", "ct_fov_relative_s",
    "radius_edt_mm", "degree_26", "log1p_component_points",
    "log1p_component_real_edges", "component_patient_point_fraction",
    "component_mean_radius_mm", "log1p_component_total_edge_length_mm",
    "is_endpoint", "is_junction",
)
EDGE_FEATURE_NAMES = (
    "delta_r_mm", "delta_a_mm", "delta_s_mm", "length_mm",
    "unit_r", "unit_a", "unit_s", "target_minus_source_radius_mm",
)
NODE_FEATURE_DIM = len(NODE_FEATURE_NAMES)
EDGE_FEATURE_DIM = len(EDGE_FEATURE_NAMES)
feature_dims = {"node": NODE_FEATURE_DIM, "edge": EDGE_FEATURE_DIM}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _signature(path: Path) -> tuple:
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino


def _immutable(value) -> np.ndarray:
    array = np.ascontiguousarray(value)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _affine(value) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
            or not np.allclose(matrix[3], [0, 0, 0, 1], rtol=0, atol=1e-10)
            or abs(np.linalg.det(matrix[:3, :3])) < 1e-12):
        raise ValueError("Finite invertible homogeneous voxel-to-RAS-mm affine required")
    return matrix


def voxel_to_ras_mm(voxel_xyz, affine_voxel_to_ras_mm):
    """Vectorized native XYZ voxel-center conversion; no axis permutation.

    Torch inputs keep their device and floating dtype. Integer inputs become
    float32 on their existing device. NumPy inputs use float64 physical math.
    """
    matrix = _affine(affine_voxel_to_ras_mm)
    if isinstance(voxel_xyz, torch.Tensor):
        if voxel_xyz.ndim < 1 or voxel_xyz.shape[-1] != 3:
            raise ValueError("Native voxel coordinates must have last dimension 3 (XYZ)")
        dtype = voxel_xyz.dtype if voxel_xyz.is_floating_point() else torch.float32
        coords = voxel_xyz.to(dtype=dtype)
        transform = torch.as_tensor(matrix, dtype=dtype, device=coords.device)
        return coords @ transform[:3, :3].T + transform[:3, 3]
    coords = np.asarray(voxel_xyz, dtype=np.float64)
    if coords.ndim < 1 or coords.shape[-1] != 3 or not np.isfinite(coords).all():
        raise ValueError("Finite native voxel XYZ coordinates with last dimension 3 required")
    return coords @ matrix[:3, :3].T + matrix[:3, 3]


@dataclass(frozen=True)
class RecipientBinding:
    """Identity of the actual native recipient CT, independent of targets.

    ct_sha256 hashes the NIfTI *file*, not an augmented or normalized array.
    Candidate centers supplied to voxel_to_ras_mm must use this native XYZ grid.
    """
    case_id: str
    ct_sha256: str
    affine_voxel_to_ras_mm: tuple
    native_shape_xyz: tuple

    def __post_init__(self):
        matrix = _affine(self.affine_voxel_to_ras_mm)
        shape = tuple(self.native_shape_xyz)
        if (not isinstance(self.case_id, str) or not self.case_id
                or not isinstance(self.ct_sha256, str) or len(self.ct_sha256) != 64
                or any(char not in "0123456789abcdef" for char in self.ct_sha256)
                or len(shape) != 3 or any(not isinstance(v, (int, np.integer)) or v < 1 for v in shape)):
            raise ValueError("Explicit case ID, lowercase CT file SHA256 and positive native XYZ shape required")
        object.__setattr__(self, "affine_voxel_to_ras_mm", tuple(tuple(row) for row in matrix.tolist()))
        object.__setattr__(self, "native_shape_xyz", tuple(int(v) for v in shape))

    @classmethod
    def from_mapping(cls, value):
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("RecipientBinding or explicit binding mapping required")
        return cls(**{name: value[name] for name in (
            "case_id", "ct_sha256", "affine_voxel_to_ras_mm", "native_shape_xyz")})


@dataclass(frozen=True)
class VesselBatch:
    node_features: torch.Tensor
    pos_mm: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    node_patient_index: torch.Tensor
    component_index: torch.Tensor
    component_patient_index: torch.Tensor
    case_ids: tuple[str, ...]
    case_ptr: torch.Tensor
    component_ptr: torch.Tensor
    audit: dict

    def to(self, device, non_blocking=False):
        values = {field.name: (getattr(self, field.name).to(device, non_blocking=non_blocking)
                              if isinstance(getattr(self, field.name), torch.Tensor)
                              else copy.deepcopy(getattr(self, field.name))) for field in fields(self)}
        return VesselBatch(**values)

    def pin_memory(self):
        values = {field.name: (getattr(self, field.name).pin_memory()
                              if isinstance(getattr(self, field.name), torch.Tensor)
                              else copy.deepcopy(getattr(self, field.name))) for field in fields(self)}
        return VesselBatch(**values)


@dataclass(frozen=True)
class _CachedGraph:
    node_features: np.ndarray
    pos_mm: np.ndarray
    edge_index: np.ndarray
    edge_attr: np.ndarray
    component_index: np.ndarray
    component_count: int
    binding: RecipientBinding
    audit: dict


class VesselStore:
    """Strict full-131 inventory, with one-time source/payload SHA validation.

    A batch may select the actual patients in a training step; the store still
    admits only a complete 131-case cohort. ``preload`` can load independent
    patients concurrently with an explicit measured worker choice. Changed
    source files invalidate the store instead of silently rebuilding its cache.
    """
    feature_dims = feature_dims

    def __init__(self, index_path, *, ct_paths=None):
        self.index_path = Path(index_path).resolve()
        self._index_signature = _signature(self.index_path)
        self.index_sha256 = _sha256(self.index_path)
        index = json.loads(self.index_path.read_text(encoding="utf-8"))
        entries = index.get("cases", [])
        if (index.get("format") != "msd_liver_vessel_inputs_v1" or index.get("path_base") != ".."
                or index.get("source_dataset_training_cases") != EXPECTED_CASES
                or index.get("available_graph_cases") != EXPECTED_CASES
                or index.get("full_dataset_processed") is not True or len(entries) != EXPECTED_CASES):
            raise ValueError("V3.0 requires the complete successful 131-case vessel inventory")
        self.dataset_root = self.index_path.parent.parent.resolve()
        self._entries = {entry["case_id"]: copy.deepcopy(entry) for entry in entries}
        if len(self._entries) != EXPECTED_CASES:
            raise ValueError("Duplicate case IDs in complete vessel inventory")
        self.case_ids = tuple(entry["case_id"] for entry in entries)
        if ct_paths is not None and (not isinstance(ct_paths, Mapping)
                                     or set(ct_paths) != set(self.case_ids)):
            raise ValueError("Relocated CT paths must explicitly bind the same complete 131-case vessel cohort")
        self._ct_path_policy = ("explicit native CT relocation; published CT file SHA256 unchanged"
                                if ct_paths is not None else "published relative native CT path")
        self._paths, self._signatures = {}, {}
        self._cache, self._bindings = {}, {}
        self._lock = threading.RLock()
        self._case_locks = {case_id: threading.Lock() for case_id in self.case_ids}
        for case_id, entry in self._entries.items():
            # The retained original liver_1 entry predates these two optional
            # index fields. Its actual graph coverage is checked below, and its
            # saved bytes remain untouched. Explicit failure is never admitted.
            if (entry.get("status") not in (None, "PASS") or entry.get("topology_audit_passed") is not True
                    or entry.get("complete_centerline_coverage") not in (None, True)
                    or entry.get("coordinate_system") != "RAS" or entry.get("coordinate_units") != "mm"):
                raise ValueError(f"{case_id}: complete RAS-mm topology-audited graph required")
            paths = {"ct": (self._external_ct(ct_paths[case_id]) if ct_paths is not None
                            else self._resolve(entry["ct"]["path"]))}
            for name in ("centerline_graph.npz", "graph.json"):
                paths[name] = self._resolve(entry["files"][name]["path"])
                if paths[name].name != name or paths[name].parent.name != case_id:
                    raise ValueError(f"{case_id}: payload path/case identity mismatch")
            self._paths[case_id] = paths
            self._signatures[case_id] = {name: _signature(path) for name, path in paths.items()}

    def _resolve(self, relative):
        path = (self.dataset_root / relative).resolve()
        if not path.is_relative_to(self.dataset_root) or not path.is_file():
            raise ValueError("Vessel inventory path must name an existing file inside its dataset root")
        return path

    @staticmethod
    def _external_ct(value):
        # Explicit read-only relocation allows a server's existing Medical/
        # native CT files without copying 18 GiB into the exported graph tree.
        # Content SHA and header are still checked by _source_binding below.
        path = Path(value).absolute()
        if path.is_symlink() or not path.is_file():
            raise ValueError("Relocated native CT must be an explicitly supplied regular file")
        # is_file() above checks existence; Windows sandbox handles may deny
        # GetFinalPathNameByHandle used only by strict realpath resolution.
        # Opening, hashing and stat guarding the actual file remain mandatory.
        return path.resolve()

    def _guard(self, case_id):
        if case_id not in self._entries:
            raise KeyError(f"Unknown vessel case: {case_id}")
        if _signature(self.index_path) != self._index_signature:
            raise ValueError("Vessel inventory changed after store admission")
        for name, path in self._paths[case_id].items():
            if _signature(path) != self._signatures[case_id][name]:
                raise ValueError(f"{case_id}: admitted {name} changed; immutable vessel cache refused")

    def _source_binding(self, case_id):
        self._guard(case_id)
        with self._lock:
            cached = self._bindings.get(case_id)
        if cached is not None:
            return cached
        path = self._paths[case_id]["ct"]
        expected = self._entries[case_id]["ct"]["sha256"]
        if _sha256(path) != expected:
            raise ValueError(f"{case_id}: CT source SHA256 mismatch")
        image = nib.load(path)
        if len(image.shape) != 3:
            raise ValueError(f"{case_id}: native CT must be three-dimensional")
        entry = self._entries[case_id]
        spatial_unit = image.header.get_xyzt_units()[0]
        if spatial_unit != "mm" and not (spatial_unit == "unknown"
                                        and entry.get("unknown_ct_units_assumed_mm") is True):
            raise ValueError(f"{case_id}: CT does not establish millimetre spatial units")
        result = RecipientBinding(case_id, expected, image.affine, image.shape)
        self._guard(case_id)
        with self._lock:
            self._bindings[case_id] = result
        return result

    def binding_from_source(self, case_id):
        """Verify the actual indexed CT file and inspect only its NIfTI header."""
        with self._case_locks[case_id]:
            return self._source_binding(case_id)

    def _read_graph(self, case_id, binding):
        entry, paths = self._entries[case_id], self._paths[case_id]
        for name in ("graph.json", "centerline_graph.npz"):
            payload = entry["files"][name]
            if paths[name].stat().st_size != payload["bytes"] or _sha256(paths[name]) != payload["sha256"]:
                raise ValueError(f"{case_id}: {name} payload SHA256/size mismatch")
        graph = json.loads(paths["graph.json"].read_text(encoding="utf-8"))
        if (graph.get("case_id") != case_id or graph.get("coordinate_system") != "RAS"
                or graph.get("units") != "mm" or graph.get("topology_audit_passed") is not True
                or graph.get("complete_centerline_coverage") is not True
                or graph.get("source_files", {}).get("ct", {}).get("sha256") != binding.ct_sha256
                or graph.get("sampling_ratio") != 1.0 or graph.get("component_pruning") is not False
                or graph.get("gap_bridging") is not False
                or graph.get("graph_node_cap") is not None or graph.get("graph_edge_cap") is not None):
            raise ValueError(f"{case_id}: saved graph provenance/full-coverage contract mismatch")
        graph_affine = _affine(graph["affine_voxel_to_ras_mm"])
        if (tuple(graph["native_shape_xyz"]) != binding.native_shape_xyz
                or not np.allclose(graph_affine, binding.affine_voxel_to_ras_mm, rtol=0, atol=1e-8)):
            raise ValueError(f"{case_id}: graph/native CT affine or shape mismatch")
        with np.load(paths["centerline_graph.npz"], allow_pickle=False) as payload:
            schema = graph.get("npz_schema")
            if schema is None and graph.get("schema_version") != 1:
                raise ValueError(f"{case_id}: unknown graph schema without an NPZ declaration")
            if schema is not None and set(payload.files) != set(schema):
                raise ValueError(f"{case_id}: NPZ schema differs from saved extraction")
            arrays = {}
            for name in payload.files:
                value = payload[name]
                if schema is not None and (list(value.shape) != schema[name]["shape"]
                                           or str(value.dtype) != schema[name]["dtype"]):
                    raise ValueError(f"{case_id}: NPZ dtype/shape mismatch: {name}")
                arrays[name] = value
        required = ("voxel_xyz", "point_ras_mm", "radius_edt_mm", "degree_26",
                    "input_component_id", "skeleton_component_id", "skeleton_edge_index")
        if any(name not in arrays for name in required):
            raise ValueError(f"{case_id}: complete native centerline arrays required")
        voxels, positions = arrays["voxel_xyz"], arrays["point_ras_mm"]
        radius, degree = arrays["radius_edt_mm"], arrays["degree_26"]
        edges = arrays["skeleton_edge_index"]
        raw_components = arrays["input_component_id"]
        components = arrays["skeleton_component_id"]
        n = len(voxels)
        if (n == 0 or voxels.shape != (n, 3) or not np.issubdtype(voxels.dtype, np.integer)
                or len(np.unique(voxels, axis=0)) != n or np.any(voxels < 0)
                or np.any(voxels >= np.asarray(binding.native_shape_xyz))
                or positions.shape != (n, 3) or not np.isfinite(positions).all()
                or radius.shape != (n,) or not np.isfinite(radius).all() or np.any(radius <= 0)
                or any(array.shape != (n,) or not np.issubdtype(array.dtype, np.integer)
                       for array in (degree, raw_components, components))
                or np.any(raw_components < 1) or np.any(components < 1)
                or edges.ndim != 2 or edges.shape[1] != 2 or not np.issubdtype(edges.dtype, np.integer)):
            raise ValueError(f"{case_id}: invalid full native centerline geometry")
        recomputed = voxel_to_ras_mm(voxels, graph_affine)
        if not np.allclose(positions, recomputed, rtol=0, atol=1e-7):
            raise ValueError(f"{case_id}: saved RAS positions disagree with native voxel/affine conversion")
        if (np.any(edges < 0) or np.any(edges >= n) or np.any(edges[:, 0] >= edges[:, 1])
                or len(np.unique(edges, axis=0)) != len(edges)):
            raise ValueError(f"{case_id}: invalid or duplicate real native skeleton edges")
        native_edges = cKDTree(voxels).query_pairs(1, p=np.inf, output_type="ndarray")
        native_edges = np.asarray(native_edges, dtype=np.int64).reshape(-1, 2)
        native_edges = native_edges[np.lexsort((native_edges[:, 1], native_edges[:, 0]))]
        ordered_edges = edges[np.lexsort((edges[:, 1], edges[:, 0]))]
        if not np.array_equal(ordered_edges, native_edges):
            raise ValueError(f"{case_id}: saved edges omit native neighbors or add artificial connections")
        if len(edges) and (np.any(raw_components[edges[:, 0]] != raw_components[edges[:, 1]])
                           or np.any(components[edges[:, 0]] != components[edges[:, 1]])):
            raise ValueError(f"{case_id}: real native edge crosses component ownership")
        actual_degree = np.bincount(edges.reshape(-1), minlength=n)
        if not np.array_equal(degree, actual_degree):
            raise ValueError(f"{case_id}: degree does not match complete native edges")
        ids, component_index = np.unique(components, return_inverse=True)
        component_count = len(ids)
        adjacency = sparse.coo_matrix((np.ones(2 * len(edges)),
            (np.concatenate((edges[:, 0], edges[:, 1])), np.concatenate((edges[:, 1], edges[:, 0])))),
            shape=(n, n)).tocsr()
        actual_component_count = connected_components(adjacency, directed=False, return_labels=False)
        if actual_component_count != component_count:
            raise ValueError(f"{case_id}: saved component IDs merge disconnected native components")
        counts = dict(skeleton_points=n, skeleton_edges=len(edges), skeleton_components=component_count)
        if any(graph["counts"].get(key) != value or entry["counts"].get(key) != value
               for key, value in counts.items()):
            raise ValueError(f"{case_id}: inventory/extraction/native graph counts disagree")
        corners = np.asarray(list(itertools.product(*[(0, size - 1) for size in binding.native_shape_xyz])))
        physical_corners = voxel_to_ras_mm(corners, graph_affine)
        lower, upper = physical_corners.min(axis=0), physical_corners.max(axis=0)
        # Add the projected native voxel width to define a nonzero FOV extent,
        # including singleton dimensions and arbitrary rotations/reflections.
        extent = upper - lower + np.abs(graph_affine[:3, :3]).sum(axis=1)
        centered = (positions - (lower + upper) / 2) / extent
        comp_points = np.bincount(component_index, minlength=component_count)
        comp_edges = np.bincount(component_index[edges[:, 0]], minlength=component_count)
        real_lengths = np.linalg.norm(positions[edges[:, 1]] - positions[edges[:, 0]], axis=1)
        comp_length = np.bincount(component_index[edges[:, 0]], weights=real_lengths, minlength=component_count)
        comp_mean_radius = np.bincount(component_index, weights=radius, minlength=component_count) / comp_points
        node_features = np.column_stack((centered, radius, degree, np.log1p(comp_points[component_index]),
            np.log1p(comp_edges[component_index]), comp_points[component_index] / n,
            comp_mean_radius[component_index], np.log1p(comp_length[component_index]), degree == 1, degree > 2))
        directed = np.concatenate((edges, edges[:, ::-1]), axis=0)
        delta = positions[directed[:, 1]] - positions[directed[:, 0]]
        lengths = np.linalg.norm(delta, axis=1)
        edge_attr = np.column_stack((delta, lengths, delta / lengths[:, None],
                                    radius[directed[:, 1]] - radius[directed[:, 0]]))
        audit = dict(format=FORMAT, case_id=case_id, ct_file_sha256=binding.ct_sha256,
            ct_source_path=str(paths["ct"]), ct_path_policy=self._ct_path_policy,
            graph_json_sha256=entry["files"]["graph.json"]["sha256"],
            centerline_npz_sha256=entry["files"]["centerline_graph.npz"]["sha256"],
            **counts, directed_message_edges=len(directed),
            coordinate_system="RAS", coordinate_units="mm", native_shape_xyz=list(binding.native_shape_xyz),
            affine_voxel_to_ras_mm=[list(row) for row in binding.affine_voxel_to_ras_mm],
            node_feature_names=list(NODE_FEATURE_NAMES), edge_feature_names=list(EDGE_FEATURE_NAMES),
            coordinate_normalization="physical CT FOV center and projected voxel-inclusive extent",
            full_native_points_preserved=True, full_native_edges_preserved=True, all_components_preserved=True,
            ct_intensity_features=False, recipient_tumor_GT_used=False, recipient_tumor_overlap_used=False,
            artery_vein_typing=False, flow_direction_labels=False, anatomy_validated=graph.get("anatomy_validated") is True,
            source_organ_constraint="saved mask-derived graph; no labels read during load",
            legacy_schema_without_npz_declaration=graph.get("npz_schema") is None,
            cache="immutable bytes-backed static CPU arrays; independent returned tensor storage")
        self._guard(case_id)
        return _CachedGraph(_immutable(node_features.astype(np.float32)),
            _immutable(positions.astype(np.float32)), _immutable(directed.T.astype(np.int64)),
            _immutable(edge_attr.astype(np.float32)), _immutable(component_index.astype(np.int64)),
            component_count, binding, audit)

    def _get(self, case_id, recipient_binding):
        binding = RecipientBinding.from_mapping(recipient_binding)
        if binding.case_id != case_id:
            raise ValueError("Vessel graph and recipient case identity differ")
        self._guard(case_id)
        with self._case_locks[case_id]:
            actual = self._source_binding(case_id)
            if binding != actual:
                raise ValueError(f"{case_id}: recipient CT SHA256/native affine/shape differs from vessel source")
            with self._lock:
                cached = self._cache.get(case_id)
            if cached is None:
                cached = self._read_graph(case_id, actual)
                with self._lock:
                    self._cache[case_id] = cached
            return cached

    def load(self, case_id, *, recipient_binding):
        """Load one complete patient graph as a one-patient VesselBatch."""
        return self.batch((case_id,), recipient_bindings={case_id: recipient_binding})

    def preload(self, *, recipient_bindings, workers):
        """Cache the entire inventory, concurrently; never select a subset."""
        if not isinstance(workers, int) or workers < 1:
            raise ValueError("Explicit positive measured CPU/I/O worker count required")
        bindings = self._ordered_bindings(self.case_ids, recipient_bindings)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            tuple(executor.map(lambda pair: self._get(*pair), zip(self.case_ids, bindings)))
        return self.audit()

    @staticmethod
    def _ordered_bindings(case_ids, recipient_bindings):
        if isinstance(recipient_bindings, Mapping):
            return tuple(recipient_bindings[case_id] for case_id in case_ids)
        if not isinstance(recipient_bindings, Sequence) or len(recipient_bindings) != len(case_ids):
            raise ValueError("One explicit recipient binding per batched patient required")
        return tuple(recipient_bindings)

    def batch(self, case_ids, *, recipient_bindings):
        case_ids = tuple(case_ids)
        if not case_ids or len(set(case_ids)) != len(case_ids):
            raise ValueError("Nonempty unique actual patient case IDs required for disjoint graph batch")
        bindings = self._ordered_bindings(case_ids, recipient_bindings)
        graphs = tuple(self._get(case_id, binding) for case_id, binding in zip(case_ids, bindings))
        node_counts = np.asarray([len(graph.pos_mm) for graph in graphs], dtype=np.int64)
        component_counts = np.asarray([graph.component_count for graph in graphs], dtype=np.int64)
        case_ptr = np.concatenate(([0], np.cumsum(node_counts)))
        component_ptr = np.concatenate(([0], np.cumsum(component_counts)))
        # This is CPU collation, not sequential patient neural computation. All
        # patients enter one disjoint GPU forward/backward after a single .to().
        tensor = lambda array: torch.from_numpy(np.ascontiguousarray(array))
        result = VesselBatch(
            node_features=tensor(np.concatenate([graph.node_features for graph in graphs])),
            pos_mm=tensor(np.concatenate([graph.pos_mm for graph in graphs])),
            edge_index=tensor(np.concatenate([graph.edge_index + offset for graph, offset in zip(graphs, case_ptr[:-1])], axis=1)),
            edge_attr=tensor(np.concatenate([graph.edge_attr for graph in graphs])),
            node_patient_index=tensor(np.repeat(np.arange(len(graphs), dtype=np.int64), node_counts)),
            component_index=tensor(np.concatenate([graph.component_index + offset
                                                   for graph, offset in zip(graphs, component_ptr[:-1])])),
            component_patient_index=tensor(np.repeat(np.arange(len(graphs), dtype=np.int64), component_counts)),
            case_ids=case_ids, case_ptr=tensor(case_ptr), component_ptr=tensor(component_ptr),
            audit=dict(format=FORMAT, index_sha256=self.index_sha256, inventory_cases=EXPECTED_CASES,
                used_cases=len(case_ids), skeleton_points=int(node_counts.sum()),
                skeleton_edges=sum(graph.audit["skeleton_edges"] for graph in graphs),
                directed_message_edges=sum(graph.audit["directed_message_edges"] for graph in graphs),
                skeleton_components=int(component_counts.sum()), patient_audits=[copy.deepcopy(graph.audit) for graph in graphs]))
        return result

    def audit(self):
        """Report complete inventory and actual cache coverage without loading CT."""
        if _signature(self.index_path) != self._index_signature:
            raise ValueError("Vessel inventory changed after store admission")
        with self._lock:
            cached = tuple(self._cache)
        counts = [self._entries[case_id]["counts"] for case_id in self.case_ids]
        return dict(format=FORMAT, index_path=str(self.index_path), index_sha256=self.index_sha256,
            ct_path_policy=self._ct_path_policy,
            inventory_cases=EXPECTED_CASES, cached_cases=len(cached), source_verified_cases=len(self._bindings),
            case_ids=list(self.case_ids), feature_dims=dict(feature_dims),
            skeleton_points=sum(count["skeleton_points"] for count in counts),
            skeleton_edges=sum(count["skeleton_edges"] for count in counts),
            skeleton_components=sum(count["skeleton_components"] for count in counts),
            subset_enabled=False, native_sampling_ratio=1.0, graph_caps=False,
            component_pruning=False, artificial_connections=False, recipient_tumor_overlap_used=False,
            node_feature_names=list(NODE_FEATURE_NAMES), edge_feature_names=list(EDGE_FEATURE_NAMES))
