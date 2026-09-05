"""Patient-specific liver-region partition and reusable region descriptors.

The partition is computed from all liver voxels, not from tumor locations. This
keeps tumor-free candidate regions represented and prevents positive-label
leakage. Expensive case-level products are persisted as a cropped compressed archive
and reused by prototype fitting, cache construction, and generation. Reuse is
fail-closed: both source volumes and the compact artifact are content-addressed.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy import ndimage as ndi

from hiercp.common import (
    LoadedCase,
    bbox_of_mask,
    ct_normalize,
    normalized_position,
    organ_depth_mm,
    verify_loaded_case_source_signatures,
)
from hiercp.schema import REGION_FEATURE_DIM, UPPER_RAW_DIM, GraphBuildConfig


REGION_CACHE_FORMAT = "hiercp_patient_regions_v2"
REGION_CACHE_STORAGE = "compact_crop_npz_v1"
REGION_CACHE_FILENAME = "regions.npz"
REGION_CACHE_SEED_SALT = "patient_regions_v2"
REGION_CACHE_INTEGRITY_FORMAT = "sha256_v1"


@dataclass
class PatientRegionData:
    full_organ_mask: np.ndarray
    organ_depth: np.ndarray
    region_labels: np.ndarray
    region_features: np.ndarray
    region_positions: np.ndarray
    region_edge_index: np.ndarray
    region_centers_vox: np.ndarray

    @property
    def num_regions(self) -> int:
        return int(self.region_features.shape[0])

    def region_at(self, center: Sequence[int]) -> int:
        center_t = tuple(int(v) for v in center)
        value = int(self.region_labels[center_t])
        if value >= 0:
            return value
        position = normalized_position(center_t, self.region_labels.shape)
        return int(np.argmin(np.linalg.norm(self.region_positions - position[None], axis=1)))


def numpy_kmeans(
    values: np.ndarray,
    clusters: int,
    *,
    rng: np.random.Generator,
    iterations: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Small deterministic NumPy k-means with k-means++ initialization."""

    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("values must be a non-empty matrix")
    if isinstance(clusters, (bool, np.bool_)) or int(clusters) != clusters:
        raise ValueError(f"clusters must be an exact integer, got {clusters!r}")
    cluster_count = int(clusters)
    if cluster_count <= 0:
        raise ValueError(f"clusters must be positive, got {cluster_count}")
    if cluster_count > int(values.shape[0]):
        raise ValueError(
            "Configured k-means cluster count exceeds the available samples; "
            f"clusters={cluster_count}, samples={values.shape[0]}. The configured "
            "cluster count was not reduced."
        )
    if isinstance(iterations, (bool, np.bool_)) or int(iterations) != iterations:
        raise ValueError(f"iterations must be an exact integer, got {iterations!r}")
    iteration_count = int(iterations)
    if iteration_count <= 0:
        raise ValueError(f"iterations must be positive, got {iteration_count}")
    first = int(rng.integers(values.shape[0]))
    centers = [values[first]]
    minimum = np.sum((values - centers[0]) ** 2, axis=1)
    for _ in range(1, cluster_count):
        total = float(minimum.sum())
        if not np.isfinite(total) or total <= 0.0:
            raise ValueError(
                "Configured k-means clusters cannot be initialized from distinct "
                f"support; clusters={cluster_count}, initialized={len(centers)}, "
                f"minimum_distance_sum={total!r}."
            )
        index = int(rng.choice(values.shape[0], p=minimum / total))
        centers.append(values[index])
        minimum = np.minimum(minimum, np.sum((values - centers[-1]) ** 2, axis=1))
    center_array = np.asarray(centers, dtype=np.float32)
    labels = np.zeros(values.shape[0], dtype=np.int32)
    for iteration in range(iteration_count):
        distances = np.sum((values[:, None] - center_array[None]) ** 2, axis=-1)
        labels = np.argmin(distances, axis=1).astype(np.int32)
        updated = center_array.copy()
        for cluster in range(cluster_count):
            members = values[labels == cluster]
            if not members.size:
                raise RuntimeError(
                    "K-means produced an empty configured cluster; refusing random "
                    f"replacement: iteration={iteration}, cluster={cluster}, "
                    f"clusters={cluster_count}, samples={values.shape[0]}"
                )
            updated[cluster] = members.mean(axis=0)
        if np.allclose(updated, center_array, atol=1e-4):
            center_array = updated
            break
        center_array = updated
    return center_array, labels


def _context_only_image(
    case: LoadedCase,
    *,
    liver_label: int,
    tumor_label: int,
) -> np.ndarray:
    """Erase tumor intensity before region/prototype CT statistics are computed."""

    output = case.image.astype(np.float32, copy=True)
    tumor = case.label == int(tumor_label)
    if not np.any(tumor):
        return output
    liver_only = case.label == int(liver_label)
    global_values = output[liver_only]
    finite_values = output[np.isfinite(output)]
    global_fill = float(np.median(global_values)) if global_values.size else float(
        np.median(finite_values) if finite_values.size else 0.0
    )
    structure = ndi.generate_binary_structure(3, 1)
    components, count = ndi.label(tumor, structure=structure)
    for component_id in range(1, count + 1):
        component = components == component_id
        slices = bbox_of_mask(component, pad=5)
        local_component = component[slices]
        local_liver = liver_only[slices]
        ring = ndi.binary_dilation(
            local_component,
            structure=structure,
            iterations=4,
        ) & ~local_component & local_liver
        values = output[slices][ring]
        fill = float(np.mean(values)) if values.size >= 8 else global_fill
        local_output = output[slices]
        local_output[local_component] = fill
    return output


def _canonical_voxel_features(
    coordinates: np.ndarray,
    shape: Sequence[int],
    depth: np.ndarray,
) -> np.ndarray:
    denominator = np.maximum(np.asarray(shape, dtype=np.float32) - 1.0, 1.0)
    position = coordinates.astype(np.float32) / denominator[None] * 2.0 - 1.0
    depth_values = depth[tuple(coordinates.T)].astype(np.float32)
    scale = max(float(np.percentile(depth_values, 95)) if depth_values.size else 1.0, 1.0)
    normalized_depth = np.clip(depth_values / scale, 0.0, 2.0)[:, None]
    radial = np.linalg.norm(position, axis=1, keepdims=True) / np.sqrt(3.0)
    return np.concatenate([position, normalized_depth, radial], axis=1).astype(np.float32)


def _assign_in_chunks(
    values: np.ndarray,
    centers: np.ndarray,
    chunk_size: int = 100_000,
) -> np.ndarray:
    labels = np.empty(values.shape[0], dtype=np.int32)
    for start in range(0, values.shape[0], chunk_size):
        part = values[start : start + chunk_size]
        distances = np.sum((part[:, None] - centers[None]) ** 2, axis=-1)
        labels[start : start + chunk_size] = np.argmin(distances, axis=1).astype(np.int32)
    return labels


def _region_adjacency(
    region_labels: np.ndarray,
    num_regions: int,
    fallback_k: int,
    positions: np.ndarray,
) -> np.ndarray:
    pairs: set[tuple[int, int]] = set()
    for axis in range(3):
        left = [slice(None)] * 3
        right = [slice(None)] * 3
        left[axis] = slice(0, -1)
        right[axis] = slice(1, None)
        first = region_labels[tuple(left)]
        second = region_labels[tuple(right)]
        valid = (first >= 0) & (second >= 0) & (first != second)
        if np.any(valid):
            # Boundary voxels may number in the millions while the number of
            # distinct region pairs is at most K². Deduplicate in NumPy first
            # instead of performing one Python set insertion per voxel face.
            boundary_pairs = np.stack(
                [first[valid].ravel(), second[valid].ravel()], axis=1
            ).astype(np.int64, copy=False)
            for source, destination in np.unique(boundary_pairs, axis=0):
                pairs.add((int(source), int(destination)))
                pairs.add((int(destination), int(source)))
    if isinstance(fallback_k, (bool, np.bool_)) or int(fallback_k) != fallback_k:
        raise ValueError(f"region_k must be an exact integer, got {fallback_k!r}")
    k = int(fallback_k)
    if num_regions <= 1 or not 1 <= k < int(num_regions):
        raise ValueError(
            "region_k must satisfy 1 <= region_k < num_regions; "
            f"region_k={k}, num_regions={num_regions}"
        )
    distances = np.linalg.norm(positions[:, None] - positions[None], axis=-1)
    np.fill_diagonal(distances, np.inf)
    neighbors = np.argpartition(distances, kth=k - 1, axis=1)[:, :k]
    for destination in range(num_regions):
        for source in neighbors[destination]:
            pairs.add((int(source), int(destination)))
    if not pairs:
        return np.empty((2, 0), dtype=np.int64)
    return np.asarray(sorted(pairs), dtype=np.int64).T


def build_patient_regions(
    case: LoadedCase,
    *,
    liver_label: int,
    tumor_label: int,
    config: GraphBuildConfig,
    rng: np.random.Generator,
    ct_clip: tuple[float, float],
) -> PatientRegionData:
    config.validate()
    full_organ = (case.label == int(liver_label)) | (case.label == int(tumor_label))
    if not np.any(full_organ):
        raise ValueError(f"No liver mask in {case.paths.case_id}")
    depth = organ_depth_mm(full_organ, case.spacing)
    coordinates = np.column_stack(np.where(full_organ)).astype(np.int32)
    sample_count = min(int(config.region_sample_voxels), int(coordinates.shape[0]))
    sampled = coordinates[rng.choice(coordinates.shape[0], size=sample_count, replace=False)]
    sampled_features = _canonical_voxel_features(sampled, case.shape, depth)
    centers, _ = numpy_kmeans(
        sampled_features,
        config.num_regions,
        rng=rng,
        iterations=config.region_lloyd_iters,
    )
    assignments = _assign_in_chunks(
        _canonical_voxel_features(coordinates, case.shape, depth),
        centers,
    )
    num_regions = int(centers.shape[0])
    region_labels = np.full(case.shape, -1, dtype=np.int16)
    region_labels[tuple(coordinates.T)] = assignments.astype(np.int16)

    descriptor_image = _context_only_image(
        case,
        liver_label=liver_label,
        tumor_label=tumor_label,
    )
    ct_normalized = ct_normalize(descriptor_image, ct_clip)
    boundary = full_organ & (depth <= float(config.boundary_depth_mm))
    organ_volume = max(1, int(full_organ.sum()))
    feature_rows: list[np.ndarray] = []
    positions: list[np.ndarray] = []
    centers_vox: list[np.ndarray] = []

    for region_id in range(num_regions):
        members = coordinates[assignments == region_id]
        if members.size == 0:
            raise RuntimeError(
                "Full-organ assignment produced an empty configured region; refusing "
                f"the previous sampled-voxel fallback: region={region_id}, "
                f"regions={num_regions}, organ_voxels={coordinates.shape[0]}"
            )
        index = tuple(members.T)
        center_vox = members.astype(np.float32).mean(axis=0)
        position = normalized_position(center_vox, case.shape).astype(np.float32)
        depth_values = depth[index].astype(np.float32)
        ct_values = ct_normalized[index].astype(np.float32)
        extent = (members.max(axis=0) - members.min(axis=0) + 1).astype(np.float32)
        extent /= np.maximum(np.asarray(case.shape, dtype=np.float32), 1.0)
        normalized_members = members.astype(np.float32) / np.maximum(
            np.asarray(case.shape, dtype=np.float32) - 1.0, 1.0
        ) * 2.0 - 1.0
        radial = np.linalg.norm(normalized_members, axis=1) / np.sqrt(3.0)
        compactness = float(
            np.mean(np.linalg.norm(members.astype(np.float32) - center_vox[None], axis=1))
            / max(float(np.linalg.norm(case.shape)), 1.0)
        )
        row = np.asarray(
            [
                *position,
                np.clip(float(depth_values.mean()) / 80.0, 0.0, 2.0),
                np.clip(float(depth_values.std()) / 40.0, 0.0, 2.0),
                np.clip(float(depth_values.max()) / 100.0, 0.0, 2.0),
                float(ct_values.mean()),
                np.clip(float(ct_values.std()), 0.0, 2.0),
                float(members.shape[0] / organ_volume),
                float(np.mean(boundary[index])),
                np.clip(compactness * 8.0, 0.0, 2.0),
                *np.clip(extent, 0.0, 1.0),
                np.clip(float(radial.mean()), 0.0, 2.0),
                1.0,
            ],
            dtype=np.float32,
        )
        if row.shape != (REGION_FEATURE_DIM,):
            raise RuntimeError(f"Region feature dimension mismatch: {row.shape}")
        feature_rows.append(row)
        positions.append(position)
        centers_vox.append(center_vox.astype(np.float32))

    region_features = np.stack(feature_rows).astype(np.float32)
    region_positions = np.stack(positions).astype(np.float32)
    return PatientRegionData(
        full_organ_mask=full_organ.astype(bool, copy=False),
        organ_depth=depth.astype(np.float32, copy=False),
        region_labels=region_labels,
        region_features=region_features,
        region_positions=region_positions,
        region_edge_index=_region_adjacency(
            region_labels,
            num_regions,
            config.region_k,
            region_positions,
        ),
        region_centers_vox=np.stack(centers_vox).astype(np.float32),
    )


def _sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _region_cache_metadata(
    case: LoadedCase,
    *,
    liver_label: int,
    tumor_label: int,
    config: GraphBuildConfig,
    ct_clip: tuple[float, float],
    seed: int,
) -> dict[str, object]:
    verify_loaded_case_source_signatures(case)
    return {
        "format": REGION_CACHE_FORMAT,
        "integrity_format": REGION_CACHE_INTEGRITY_FORMAT,
        "case_id": case.paths.case_id,
        "image": dict(case.image_source_signature),
        "label": dict(case.label_source_signature),
        "shape": list(case.shape),
        "spacing": [float(value) for value in case.spacing],
        "labels": {"liver": int(liver_label), "tumor": int(tumor_label)},
        "graph_config": config.to_dict(),
        "ct_clip": [float(value) for value in ct_clip],
        "seed": int(seed),
    }


def _metadata_equal(actual: dict, expected: dict) -> bool:
    return all(actual.get(key) == value for key, value in expected.items())


def _save_array(path: Path, array: np.ndarray, *, overwrite: bool = False) -> None:
    """Legacy full-volume writer retained only for backward-compatible loading tests."""

    destination = path if path.suffix == ".npy" else Path(f"{path}.npy")
    try:
        with destination.open("wb" if overwrite else "xb") as handle:
            np.save(handle, np.ascontiguousarray(array), allow_pickle=False)
    except FileExistsError as exc:
        raise FileExistsError(
            "Refusing to replace an existing region-cache array without explicit "
            f"overwrite authorization: {destination}. Confirm the path, then pass "
            "overwrite=True, or choose a new cache directory."
        ) from exc


def _cache_path_exists(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def _remove_cache_path(path: Path, *, overwrite: bool, action: str) -> None:
    if not _cache_path_exists(path):
        return
    if not overwrite:
        raise FileExistsError(
            f"Refusing to {action} without explicit overwrite authorization: {path}. "
            "Confirm the path, then pass overwrite=True, or choose a new cache directory."
        )
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()


def _compact_crop(regions: PatientRegionData) -> tuple[tuple[slice, slice, slice], list[int], list[int]]:
    mask = np.asarray(regions.full_organ_mask, dtype=bool)
    if mask.ndim != 3 or not np.any(mask):
        raise ValueError("Cannot persist an empty/non-3D patient-region mask")
    coordinates = np.where(mask)
    start = [int(axis.min()) for axis in coordinates]
    stop = [int(axis.max()) + 1 for axis in coordinates]
    slices = tuple(slice(first, last) for first, last in zip(start, stop))
    return slices, start, stop


def save_patient_regions(
    regions: PatientRegionData,
    destination: str | os.PathLike[str],
    *,
    metadata: dict[str, object],
    overwrite: bool = False,
) -> None:
    """Atomically persist an exact liver-bounding-box cache in compressed form.

    The logical cache format remains ``hiercp_patient_regions_v2`` because the
    region partition and descriptors are unchanged. Only the physical storage
    representation changes. Outside the liver, ``organ_depth`` is exactly zero
    and ``region_labels`` is exactly -1, so those voxels need not be stored.
    """

    root = Path(destination)
    root.parent.mkdir(parents=True, exist_ok=True)
    temporary = root.with_name(f"{root.name}.tmp.{os.getpid()}")
    _remove_cache_path(
        temporary,
        overwrite=overwrite,
        action="remove an existing temporary region-cache path",
    )
    temporary.mkdir(parents=True)
    slices, crop_start, crop_stop = _compact_crop(regions)
    labels_crop = np.ascontiguousarray(regions.region_labels[slices], dtype=np.int16)
    depth_crop = np.ascontiguousarray(regions.organ_depth[slices], dtype=np.float32)
    if not np.array_equal(labels_crop >= 0, np.asarray(regions.full_organ_mask[slices], dtype=bool)):
        raise RuntimeError("Region labels and full-organ mask disagree inside the compact crop")
    archive_path = temporary / REGION_CACHE_FILENAME
    np.savez_compressed(
        archive_path,
        organ_depth=depth_crop,
        region_labels=labels_crop,
        region_features=np.ascontiguousarray(regions.region_features, dtype=np.float32),
        region_positions=np.ascontiguousarray(regions.region_positions, dtype=np.float32),
        region_edge_index=np.ascontiguousarray(regions.region_edge_index, dtype=np.int64),
        region_centers_vox=np.ascontiguousarray(regions.region_centers_vox, dtype=np.float32),
    )
    metadata_payload = dict(metadata)
    metadata_payload.update(
        {
            "integrity_format": REGION_CACHE_INTEGRITY_FORMAT,
            "storage": REGION_CACHE_STORAGE,
            "crop_start": crop_start,
            "crop_stop": crop_stop,
            "artifact_sha256": _sha256_file(archive_path),
        }
    )
    (temporary / "metadata.json").write_text(
        json.dumps(metadata_payload, indent=2), encoding="utf-8"
    )
    _remove_cache_path(
        root,
        overwrite=overwrite,
        action="replace an existing region cache",
    )
    temporary.replace(root)


def _load_compact_regions(
    root: Path, metadata: dict[str, object]
) -> PatientRegionData:
    try:
        shape = tuple(int(value) for value in metadata["shape"])
        start = tuple(int(value) for value in metadata["crop_start"])
        stop = tuple(int(value) for value in metadata["crop_stop"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid compact region-cache geometry: {root}") from exc
    if len(shape) != 3 or len(start) != 3 or len(stop) != 3:
        raise ValueError(f"Invalid compact region-cache dimensions: {root}")
    if any(first < 0 or last > size or last <= first for first, last, size in zip(start, stop, shape)):
        raise ValueError(f"Invalid compact region-cache crop bounds: {root}")
    slices = tuple(slice(first, last) for first, last in zip(start, stop))
    with np.load(root / REGION_CACHE_FILENAME, allow_pickle=False) as archive:
        labels_crop = np.asarray(archive["region_labels"], dtype=np.int16)
        depth_crop = np.asarray(archive["organ_depth"], dtype=np.float32)
        expected_crop = tuple(last - first for first, last in zip(start, stop))
        if labels_crop.shape != expected_crop or depth_crop.shape != expected_crop:
            raise ValueError(
                f"Compact region-cache crop shape mismatch: {root}: "
                f"labels={labels_crop.shape} depth={depth_crop.shape} expected={expected_crop}"
            )
        region_labels = np.full(shape, -1, dtype=np.int16)
        organ_depth = np.zeros(shape, dtype=np.float32)
        region_labels[slices] = labels_crop
        organ_depth[slices] = depth_crop
        return PatientRegionData(
            full_organ_mask=region_labels >= 0,
            organ_depth=organ_depth,
            region_labels=region_labels,
            region_features=np.asarray(archive["region_features"], dtype=np.float32),
            region_positions=np.asarray(archive["region_positions"], dtype=np.float32),
            region_edge_index=np.asarray(archive["region_edge_index"], dtype=np.int64),
            region_centers_vox=np.asarray(archive["region_centers_vox"], dtype=np.float32),
        )


def _load_legacy_regions(root: Path, *, mmap: bool) -> PatientRegionData:
    mmap_mode = "r" if mmap else None
    required = (
        "full_organ_mask.npy",
        "organ_depth.npy",
        "region_labels.npy",
        "region_features.npy",
        "region_positions.npy",
        "region_edge_index.npy",
        "region_centers_vox.npy",
    )
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete legacy region cache {root}: {missing}")
    return PatientRegionData(
        full_organ_mask=np.load(root / "full_organ_mask.npy", mmap_mode=mmap_mode, allow_pickle=False),
        organ_depth=np.load(root / "organ_depth.npy", mmap_mode=mmap_mode, allow_pickle=False),
        region_labels=np.load(root / "region_labels.npy", mmap_mode=mmap_mode, allow_pickle=False),
        region_features=np.load(root / "region_features.npy", mmap_mode=mmap_mode, allow_pickle=False),
        region_positions=np.load(root / "region_positions.npy", mmap_mode=mmap_mode, allow_pickle=False),
        region_edge_index=np.load(root / "region_edge_index.npy", mmap_mode=mmap_mode, allow_pickle=False),
        region_centers_vox=np.load(root / "region_centers_vox.npy", mmap_mode=mmap_mode, allow_pickle=False),
    )


def load_patient_regions(
    source: str | os.PathLike[str],
    *,
    mmap: bool = True,
) -> tuple[PatientRegionData, dict[str, object]]:
    root = Path(source)
    metadata_path = root / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Region-cache metadata is missing: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict) or metadata.get("format") != REGION_CACHE_FORMAT:
        raise ValueError(f"Unsupported region cache: {root}")
    compact = root / REGION_CACHE_FILENAME
    if compact.is_file():
        if metadata.get("storage") != REGION_CACHE_STORAGE:
            raise ValueError(f"Unknown compact region-cache storage marker: {root}")
        if metadata.get("integrity_format") != REGION_CACHE_INTEGRITY_FORMAT:
            raise ValueError(
                f"Region cache has no supported integrity contract: {root}. "
                "Rebuild it with explicit overwrite authorization."
            )
        expected_sha256 = str(metadata.get("artifact_sha256", ""))
        if re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None:
            raise ValueError(
                f"Region cache has no exact artifact SHA-256 contract: {root}. "
                "Rebuild it with explicit overwrite authorization."
            )
        actual_sha256 = _sha256_file(compact)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                "Region-cache artifact SHA-256 mismatch: "
                f"{compact}; expected={expected_sha256}, actual={actual_sha256}. "
                "Rebuild it with explicit overwrite authorization."
            )
        data = _load_compact_regions(root, metadata)
    else:
        raise ValueError(
            f"Legacy region cache has no exact artifact integrity contract: {root}. "
            "Rebuild it with explicit overwrite authorization."
        )
    return data, metadata


def load_or_build_patient_regions(
    case: LoadedCase,
    *,
    liver_label: int,
    tumor_label: int,
    config: GraphBuildConfig,
    ct_clip: tuple[float, float],
    seed: int,
    cache_dir: str | os.PathLike[str] | None = None,
    overwrite: bool = False,
    mmap: bool = True,
) -> PatientRegionData:
    """Load a case-level region cache or build it exactly once.

    The same deterministic seed/config is shared by prototype fitting, training
    cache construction, and generation.  This both removes duplicate work and
    guarantees that population prototypes correspond to the regions seen by
    the upper GNN.
    """

    expected = _region_cache_metadata(
        case,
        liver_label=liver_label,
        tumor_label=tumor_label,
        config=config,
        ct_clip=ct_clip,
        seed=seed,
    )
    if cache_dir is None:
        return build_patient_regions(
            case,
            liver_label=liver_label,
            tumor_label=tumor_label,
            config=config,
            rng=np.random.default_rng(seed),
            ct_clip=ct_clip,
        )

    case_root = Path(cache_dir) / case.paths.case_id
    if case_root.is_dir() and not overwrite:
        regions, metadata = load_patient_regions(case_root, mmap=mmap)
        if not _metadata_equal(metadata, expected):
            raise FileExistsError(
                f"Region cache is incompatible for {case.paths.case_id}: {case_root}. "
                "Confirm the path, then pass overwrite=True, or use a separate workspace."
            )
        verify_loaded_case_source_signatures(case)
        return regions
    if (case_root.exists() or case_root.is_symlink()) and not overwrite:
        raise FileExistsError(
            "Refusing to replace an existing non-directory region-cache path without "
            f"explicit overwrite authorization: {case_root}. Confirm the path, then "
            "pass overwrite=True, or use a separate workspace."
        )

    regions = build_patient_regions(
        case,
        liver_label=liver_label,
        tumor_label=tumor_label,
        config=config,
        rng=np.random.default_rng(seed),
        ct_clip=ct_clip,
    )
    verify_loaded_case_source_signatures(case)
    save_patient_regions(regions, case_root, metadata=expected, overwrite=overwrite)
    # Compact NPZ storage cannot be memory-mapped. Return the arrays that were
    # just computed instead of allocating a second full-volume copy. Existing
    # compact caches are reconstructed only when they are actually reused.
    return regions


def upper_geometry_vector(
    *,
    center: Sequence[int | float],
    shape: Sequence[int],
    border_distance_mm: float,
    occupied_distance_mm: float,
    context_mean_hu: float,
    context_std_hu: float,
    volume_vox: int,
    coverage: float,
    local_thickness_mm: float,
    surface_alignment: float,
    scale_mean: float,
    anisotropy: float,
    valid: float = 1.0,
    ct_clip: tuple[float, float] = (-200.0, 250.0),
) -> np.ndarray:
    """Return the shared 14-dimensional tumor/candidate/lesion descriptor."""

    position = normalized_position(center, shape)
    low, high = map(float, ct_clip)
    mean_norm = np.clip(
        (float(context_mean_hu) - low) / max(high - low, 1e-6) * 2.0 - 1.0,
        -1.0,
        1.0,
    )
    clearance = 2.0 if not np.isfinite(occupied_distance_mm) else np.clip(
        float(occupied_distance_mm) / 100.0, 0.0, 2.0
    )
    row = np.asarray(
        [
            *position,
            np.clip(float(border_distance_mm) / 80.0, 0.0, 2.0),
            clearance,
            mean_norm,
            np.clip(float(context_std_hu) / 150.0, 0.0, 2.0),
            np.clip(np.log1p(max(0, int(volume_vox))) / 12.0, 0.0, 2.0),
            np.clip(float(coverage), 0.0, 1.0),
            np.clip(float(local_thickness_mm) / 100.0, 0.0, 2.0),
            np.clip(float(surface_alignment), 0.0, 1.0),
            np.clip(float(scale_mean), 0.0, 2.0),
            np.clip(float(anisotropy) / 10.0, 0.0, 2.0),
            np.clip(float(valid), 0.0, 1.0),
        ],
        dtype=np.float32,
    )
    if row.shape != (UPPER_RAW_DIM,):
        raise RuntimeError(f"Upper feature dimension mismatch: {row.shape}")
    return row
