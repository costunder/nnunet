"""Shared NIfTI, geometry, candidate-sampling, and copy-paste utilities.

These routines preserve the existing project’s I/O conventions and hard
anatomical constraints while the isolated experiment changes only the
hierarchical PyTorch Geometric placement model.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

try:
    import nibabel as nib
except ModuleNotFoundError:  # allows synthetic/static tests without NIfTI I/O
    nib = None
import numpy as np
from scipy import ndimage as ndi


DEFAULT_CT_CLIP = (-200.0, 250.0)
CANDIDATE_SEARCH_VERSION = "hiercp_candidate_search_v2_failure_exhaustive"
CANDIDATE_DIAGNOSTICS_FORMAT = "hiercp_candidate_diagnostics_v1"


class CandidatePreparationError(RuntimeError):
    """An unavailable candidate curriculum with explicit, serializable evidence."""

    def __init__(self, reason: str, diagnostics: dict, *, message: str | None = None):
        if not isinstance(reason, str) or not reason:
            raise ValueError("Candidate failure reason must be a non-empty stable string")
        payload = {"format": CANDIDATE_DIAGNOSTICS_FORMAT,
                   "search_version": CANDIDATE_SEARCH_VERSION, **diagnostics}
        # Snapshot the evidence; reject NaN/opaque objects instead of hiding them.
        serialized = json.dumps(payload, sort_keys=True, allow_nan=False)
        self.reason = reason
        self.diagnostics = json.loads(serialized)
        super().__init__(f"{reason}: {message or 'Candidate preparation is unavailable'}. {serialized}")


@dataclass(frozen=True)
class CasePaths:
    case_id: str
    image_path: Path
    label_path: Path


@dataclass
class LoadedCase:
    paths: CasePaths
    image: np.ndarray
    label: np.ndarray
    image_affine: np.ndarray
    label_affine: np.ndarray
    image_header: object
    label_header: object
    spacing: np.ndarray
    image_source_signature: dict[str, object]
    label_source_signature: dict[str, object]

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(v) for v in self.image.shape)


@dataclass
class SourceTumor:
    component_id: int
    full_mask: np.ndarray
    patch_mask: np.ndarray
    patch_image: np.ndarray
    patch_slices: tuple[slice, slice, slice]
    anchor_center: tuple[int, int, int]
    centroid: tuple[float, float, float]
    voxel_count: int


@dataclass(frozen=True)
class CandidateInfo:
    center: tuple[int, int, int]
    slices: tuple[slice, slice, slice]
    liver_coverage: float
    border_distance_mm: float
    occupied_distance_mm: float
    context_mean_hu: float
    context_std_hu: float


@dataclass(frozen=True)
class PasteRecord:
    copy_index: int
    source_component: int
    source_voxels: int
    target_center: tuple[int, int, int]
    liver_coverage: float
    selection_score: float
    intensity_scale: float
    intensity_shift: float


def natural_key(text: str) -> list[object]:
    return [int(token) if token.isdigit() else token.lower() for token in re.split(r"(\d+)", text)]


def stable_case_seed(global_seed: int, case_id: str, salt: str = "") -> int:
    payload = f"{global_seed}|{case_id}|{salt}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "little") % (2**32)


def _file_content_signature(
    path: str | os.PathLike[str],
    *,
    chunk_size: int = 8 * 1024 * 1024,
) -> dict[str, object]:
    """Hash one stable regular file and bind the digest to its exact identity."""

    source = Path(path).resolve(strict=True)
    if not source.is_file():
        raise FileNotFoundError(f"Source path is not a regular file: {source}")
    before = source.stat()
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    after = source.stat()
    before_identity = (
        int(before.st_dev),
        int(before.st_ino),
        int(before.st_size),
        int(before.st_mtime_ns),
    )
    after_identity = (
        int(after.st_dev),
        int(after.st_ino),
        int(after.st_size),
        int(after.st_mtime_ns),
    )
    if before_identity != after_identity:
        raise RuntimeError(
            f"Source file changed while it was being hashed: {source}"
        )
    return {
        "kind": "file",
        "integrity_format": "sha256_v1",
        "path": str(source),
        "size": int(after.st_size),
        "mtime_ns": int(after.st_mtime_ns),
        "sha256": digest.hexdigest(),
    }


def synthetic_array_signature(array: np.ndarray) -> dict[str, object]:
    """Return an explicit in-memory signature used only by synthetic fixtures."""

    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(repr(tuple(int(value) for value in contiguous.shape)).encode("ascii"))
    digest.update(memoryview(contiguous).cast("B"))
    return {
        "kind": "synthetic_fixture",
        "integrity_format": "sha256_v1",
        "dtype": contiguous.dtype.str,
        "shape": [int(value) for value in contiguous.shape],
        "sha256": digest.hexdigest(),
    }


def verify_loaded_case_source_signatures(case: LoadedCase) -> None:
    """Fail if a loaded case no longer matches the bytes it was loaded from."""

    sources = (
        (
            "image",
            case.paths.image_path,
            case.image,
            case.image_source_signature,
        ),
        (
            "label",
            case.paths.label_path,
            case.label,
            case.label_source_signature,
        ),
    )
    for name, path, array, expected in sources:
        if not isinstance(expected, dict):
            raise ValueError(f"Loaded case {name} source signature is not an object")
        kind = expected.get("kind")
        if kind == "file":
            actual = _file_content_signature(path)
        elif kind == "synthetic_fixture":
            actual = synthetic_array_signature(array)
        else:
            raise ValueError(
                f"Loaded case {name} source signature has unsupported kind={kind!r}"
            )
        if actual != expected:
            raise RuntimeError(
                f"Loaded case {case.paths.case_id!r} {name} source changed after loading"
            )


def discover_cases(
    data_dir: str | os.PathLike[str],
    *,
    image_subdir: str = "image",
    label_subdir: str = "labels",
    max_cases: int | None = None,
    case_ids: Sequence[str] | None = None,
    run_mode: str = "production",
) -> list[CasePaths]:
    data_root = Path(data_dir)
    image_dir = data_root / image_subdir
    label_dir = data_root / label_subdir
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")
    if not label_dir.is_dir():
        raise FileNotFoundError(f"Label directory does not exist: {label_dir}")

    requested = set(case_ids or [])
    cases: list[CasePaths] = []
    for image_path in sorted(image_dir.glob("*_0000.nii.gz"), key=lambda p: natural_key(p.name)):
        if image_path.name.startswith("._"):
            continue
        case_id = image_path.name[: -len("_0000.nii.gz")]
        if requested and case_id not in requested:
            continue
        label_path = label_dir / f"{case_id}.nii.gz"
        if not label_path.is_file():
            raise FileNotFoundError(f"Missing label for {image_path.name}: {label_path}")
        cases.append(CasePaths(case_id, image_path.resolve(), label_path.resolve()))

    if requested:
        found = {case.case_id for case in cases}
        missing = sorted(requested - found, key=natural_key)
        if missing:
            raise FileNotFoundError(f"Requested cases were not found: {missing}")

    if max_cases is not None:
        mode = str(run_mode).strip().lower()
        if mode not in {"debug", "benchmark", "ablation", "smoke", "nonproduction"}:
            raise ValueError(
                "max_cases is forbidden in production case discovery. Use the full "
                "cohort, or explicitly select a labelled non-production run mode and "
                "a separate output directory."
            )
        limit = int(max_cases)
        if limit <= 0:
            raise ValueError(f"max_cases must be positive, got {max_cases}")
        total = len(cases)
        cases = cases[:limit]
        print(
            "[NonProductionSubset] "
            f"run_mode={mode} selected_cases={len(cases)} total_cases={total} "
            f"usage_ratio={len(cases) / total if total else 0.0:.6f}"
        )
    if not cases:
        raise RuntimeError(f"No '*_0000.nii.gz' image/label pairs found under {data_root}")
    return cases


def _slice_nii_to_3d(nii):
    if len(nii.shape) == 3:
        return nii
    if len(nii.shape) != 4:
        raise ValueError(f"Only 3D or single-channel 4D NIfTI is supported, got shape={nii.shape}")

    # Prefer a small trailing channel dimension, which is the nnU-Net convention.
    if nii.shape[-1] <= 4:
        return nii.slicer[..., 0]
    if nii.shape[0] <= 4:
        return nii.slicer[0, ...]
    raise ValueError(f"Cannot infer the channel axis for 4D NIfTI shape={nii.shape}")


def load_case(paths: CasePaths) -> LoadedCase:
    if nib is None:
        raise ModuleNotFoundError("nibabel is required for NIfTI loading; install it with `python -m pip install nibabel`.")
    image_source_signature = _file_content_signature(paths.image_path)
    label_source_signature = _file_content_signature(paths.label_path)
    image_nii = _slice_nii_to_3d(nib.load(str(paths.image_path)))
    label_nii = _slice_nii_to_3d(nib.load(str(paths.label_path)))

    image = np.asarray(image_nii.dataobj, dtype=np.float32, order="C")
    label = np.asarray(label_nii.dataobj, dtype=np.int16, order="C")
    if image.shape != label.shape:
        raise ValueError(
            f"Image/label shape mismatch for {paths.case_id}: {image.shape} vs {label.shape}"
        )
    if image.ndim != 3:
        raise ValueError(f"Expected a 3D case, got {image.shape}")

    spacing = np.asarray(image_nii.header.get_zooms()[:3], dtype=np.float32)
    if spacing.shape != (3,) or np.any(spacing <= 0):
        raise ValueError(f"Invalid voxel spacing for {paths.case_id}: {spacing}")

    image_affine = np.asarray(image_nii.affine)
    label_affine = np.asarray(label_nii.affine)
    image_header = image_nii.header.copy()
    label_header = label_nii.header.copy()
    post_image_signature = _file_content_signature(paths.image_path)
    post_label_signature = _file_content_signature(paths.label_path)
    if post_image_signature != image_source_signature:
        raise RuntimeError(
            f"Image source changed while case {paths.case_id!r} was being loaded"
        )
    if post_label_signature != label_source_signature:
        raise RuntimeError(
            f"Label source changed while case {paths.case_id!r} was being loaded"
        )

    return LoadedCase(
        paths=paths,
        image=image,
        label=label,
        image_affine=image_affine,
        label_affine=label_affine,
        image_header=image_header,
        label_header=label_header,
        spacing=spacing,
        image_source_signature=image_source_signature,
        label_source_signature=label_source_signature,
    )


def output_paths(out_dir: str | os.PathLike[str], case_id: str) -> tuple[Path, Path]:
    root = Path(out_dir)
    return root / "image" / f"{case_id}_0000.nii.gz", root / "labels" / f"{case_id}.nii.gz"


def ensure_output_dirs(out_dir: str | os.PathLike[str]) -> None:
    root = Path(out_dir)
    (root / "image").mkdir(parents=True, exist_ok=True)
    (root / "labels").mkdir(parents=True, exist_ok=True)


def save_case_pair(
    case: LoadedCase,
    image: np.ndarray,
    label: np.ndarray,
    out_dir: str | os.PathLike[str],
    *,
    overwrite: bool = False,
) -> None:
    if nib is None:
        raise ModuleNotFoundError("nibabel is required for NIfTI saving; install it with `python -m pip install nibabel`.")
    if image.shape != case.shape or label.shape != case.shape:
        raise ValueError("Refusing to save arrays whose shapes differ from the source case")
    image_path, label_path = output_paths(out_dir, case.paths.case_id)
    ensure_output_dirs(out_dir)
    destinations = (image_path, label_path)
    existing = [
        path for path in destinations if path.exists() or path.is_symlink()
    ]
    if existing and not overwrite:
        raise FileExistsError(
            "Refusing to replace an existing or partial case pair without explicit "
            f"overwrite authorization: {existing}"
        )
    directories = [path for path in existing if path.is_dir()]
    if directories:
        raise IsADirectoryError(
            f"Case-pair destination is occupied by a directory: {directories}"
        )

    temporary_paths: list[Path] = []
    created_destinations: list[Path] = []
    try:
        for destination in destinations:
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{destination.name}.",
                suffix=".nii.gz",
                dir=destination.parent,
            )
            os.close(descriptor)
            temporary_paths.append(Path(temporary_name))
        nib.save(
            nib.Nifti1Image(
                image.astype(np.float32, copy=False),
                case.image_affine,
                case.image_header,
            ),
            str(temporary_paths[0]),
        )
        nib.save(
            nib.Nifti1Image(
                label.astype(np.int16, copy=False),
                case.label_affine,
                case.label_header,
            ),
            str(temporary_paths[1]),
        )
        if overwrite:
            for temporary, destination in zip(temporary_paths, destinations):
                os.replace(temporary, destination)
        else:
            try:
                for temporary, destination in zip(temporary_paths, destinations):
                    os.link(temporary, destination)
                    created_destinations.append(destination)
            except Exception:
                for destination in created_destinations:
                    if destination.exists() or destination.is_symlink():
                        destination.unlink()
                raise
    finally:
        for temporary in temporary_paths:
            if temporary.exists() or temporary.is_symlink():
                temporary.unlink()


def pair_already_exists(out_dir: str | os.PathLike[str], case_id: str) -> bool:
    image_path, label_path = output_paths(out_dir, case_id)
    return image_path.exists() and label_path.exists()


def link_or_copy_case(
    paths: CasePaths,
    out_dir: str | os.PathLike[str],
    *,
    mode: str = "symlink",
    overwrite: bool = False,
) -> None:
    if mode not in {"symlink", "hardlink", "copy"}:
        raise ValueError(f"Unsupported mode: {mode}")
    ensure_output_dirs(out_dir)
    dst_image, dst_label = output_paths(out_dir, paths.case_id)

    for src, dst in ((paths.image_path, dst_image), (paths.label_path, dst_label)):
        if dst.exists() or dst.is_symlink():
            if not overwrite:
                continue
            dst.unlink()
        if mode == "symlink":
            dst.symlink_to(src.resolve())
        elif mode == "hardlink":
            os.link(src, dst)
        else:
            shutil.copy2(src, dst)


def bbox_of_mask(mask: np.ndarray, pad: int = 0) -> tuple[slice, slice, slice]:
    if mask.ndim != 3 or not np.any(mask):
        raise ValueError("bbox_of_mask requires a non-empty 3D mask")
    objects = ndi.find_objects(mask.astype(np.uint8, copy=False))
    if not objects or objects[0] is None:
        raise ValueError("Could not determine mask bounding box")
    base = objects[0]
    slices: list[slice] = []
    for axis, slc in enumerate(base):
        start = max(0, int(slc.start) - int(pad))
        stop = min(mask.shape[axis], int(slc.stop) + int(pad))
        slices.append(slice(start, stop))
    return tuple(slices)  # type: ignore[return-value]


def slices_for_center(
    center: Sequence[int], patch_shape: Sequence[int], volume_shape: Sequence[int]
) -> tuple[slice, slice, slice] | None:
    result: list[slice] = []
    for c, p, size in zip(center, patch_shape, volume_shape):
        start = int(c) - int(p) // 2
        stop = start + int(p)
        if start < 0 or stop > int(size):
            return None
        result.append(slice(start, stop))
    return tuple(result)  # type: ignore[return-value]


def choose_source_tumor(
    image: np.ndarray,
    label: np.ndarray,
    *,
    tumor_label: int,
    rng: np.random.Generator,
    selection: str = "random",
    pad: int = 4,
) -> tuple[SourceTumor, np.ndarray, int]:
    tumor_mask = label == int(tumor_label)
    structure = ndi.generate_binary_structure(3, 1)
    components, num_components = ndi.label(tumor_mask, structure=structure)
    if num_components == 0:
        raise ValueError("No tumor component exists in the label")

    sizes = np.bincount(components.ravel(), minlength=num_components + 1)[1:]
    if selection == "random":
        component_id = int(rng.integers(1, num_components + 1))
    elif selection == "largest":
        component_id = int(np.argmax(sizes) + 1)
    elif selection == "size_weighted":
        probabilities = sizes.astype(np.float64)
        probabilities /= probabilities.sum()
        component_id = int(rng.choice(np.arange(1, num_components + 1), p=probabilities))
    else:
        raise ValueError(f"Unknown source selection mode: {selection}")

    full_mask = components == component_id
    patch_slices = bbox_of_mask(full_mask, pad=pad)
    patch_mask = full_mask[patch_slices].astype(bool, copy=True)
    patch_image = image[patch_slices].astype(np.float32, copy=True)
    starts = np.array([slc.start for slc in patch_slices], dtype=np.int64)
    anchor = starts + np.asarray(patch_mask.shape, dtype=np.int64) // 2
    centroid = ndi.center_of_mass(full_mask)

    source = SourceTumor(
        component_id=component_id,
        full_mask=full_mask,
        patch_mask=patch_mask,
        patch_image=patch_image,
        patch_slices=patch_slices,
        anchor_center=tuple(int(v) for v in anchor),
        centroid=tuple(float(v) for v in centroid),
        voxel_count=int(full_mask.sum()),
    )
    return source, components, num_components


def centered_mask(mask: np.ndarray, output_shape: Sequence[int]) -> np.ndarray:
    """Center-crop/pad a 3D mask into ``output_shape``."""
    output_shape_t = tuple(int(v) for v in output_shape)
    out = np.zeros(output_shape_t, dtype=bool)

    src_slices: list[slice] = []
    dst_slices: list[slice] = []
    for src_size, dst_size in zip(mask.shape, output_shape_t):
        if src_size <= dst_size:
            src_start = 0
            dst_start = (dst_size - src_size) // 2
            length = src_size
        else:
            src_start = (src_size - dst_size) // 2
            dst_start = 0
            length = dst_size
        src_slices.append(slice(src_start, src_start + length))
        dst_slices.append(slice(dst_start, dst_start + length))

    out[tuple(dst_slices)] = mask[tuple(src_slices)]
    return out


def extract_centered_patch(
    array: np.ndarray,
    center: Sequence[int],
    patch_shape: Sequence[int],
    *,
    pad_value: float | int = 0,
) -> np.ndarray:
    """Extract a centered 3D patch, padding when it crosses a volume boundary."""
    patch_shape_t = tuple(int(v) for v in patch_shape)
    out = np.full(patch_shape_t, pad_value, dtype=array.dtype)

    src_slices: list[slice] = []
    dst_slices: list[slice] = []
    for c, p, size in zip(center, patch_shape_t, array.shape):
        start = int(c) - p // 2
        stop = start + p
        src_start = max(0, start)
        src_stop = min(int(size), stop)
        if src_stop <= src_start:
            return out
        dst_start = src_start - start
        dst_stop = dst_start + (src_stop - src_start)
        src_slices.append(slice(src_start, src_stop))
        dst_slices.append(slice(dst_start, dst_stop))
    out[tuple(dst_slices)] = array[tuple(src_slices)]
    return out


def insert_centered_patch(
    volume: np.ndarray,
    patch: np.ndarray,
    center: Sequence[int],
    *,
    weight: np.ndarray | None = None,
) -> None:
    """Insert a centered patch in-place, optionally alpha blending by ``weight``."""
    src_slices: list[slice] = []
    dst_slices: list[slice] = []
    for c, p, size in zip(center, patch.shape, volume.shape):
        start = int(c) - int(p) // 2
        stop = start + int(p)
        dst_start = max(0, start)
        dst_stop = min(int(size), stop)
        if dst_stop <= dst_start:
            return
        src_start = dst_start - start
        src_stop = src_start + (dst_stop - dst_start)
        src_slices.append(slice(src_start, src_stop))
        dst_slices.append(slice(dst_start, dst_stop))

    src = tuple(src_slices)
    dst = tuple(dst_slices)
    if weight is None:
        volume[dst] = patch[src]
    else:
        alpha = np.asarray(weight[src], dtype=np.float32)
        volume[dst] = (1.0 - alpha) * volume[dst] + alpha * patch[src]


def ct_normalize(array: np.ndarray, clip: tuple[float, float] = DEFAULT_CT_CLIP) -> np.ndarray:
    low, high = (float(clip[0]), float(clip[1]))
    if high <= low:
        raise ValueError(f"Invalid CT clip range: {clip}")
    clipped = np.clip(array.astype(np.float32, copy=False), low, high)
    return ((clipped - low) / (high - low) * 2.0 - 1.0).astype(np.float32, copy=False)


def ct_denormalize(array: np.ndarray, clip: tuple[float, float] = DEFAULT_CT_CLIP) -> np.ndarray:
    low, high = (float(clip[0]), float(clip[1]))
    return ((array.astype(np.float32, copy=False) + 1.0) * 0.5 * (high - low) + low).astype(
        np.float32, copy=False
    )


def context_ring_mask(mask: np.ndarray, width: int = 3) -> np.ndarray:
    if width <= 0:
        return ~mask
    structure = ndi.generate_binary_structure(3, 1)
    return ndi.binary_dilation(mask, structure=structure, iterations=int(width)) & ~mask


def context_stats_for_local_mask(
    image_patch: np.ndarray,
    organ_patch: np.ndarray,
    target_mask_patch: np.ndarray,
    *,
    ring_width: int = 3,
) -> tuple[float, float]:
    ring = context_ring_mask(target_mask_patch, width=ring_width) & organ_patch.astype(bool)
    values = image_patch[ring]
    if values.size < 8:
        values = image_patch[organ_patch.astype(bool) & ~target_mask_patch]
    if values.size == 0:
        values = image_patch.reshape(-1)
    return float(np.mean(values)), float(np.std(values))


def erase_mask_with_context(
    image_patch: np.ndarray,
    target_mask_patch: np.ndarray,
    organ_patch: np.ndarray,
    *,
    ring_width: int = 3,
) -> np.ndarray:
    out = image_patch.astype(np.float32, copy=True)
    mean, _ = context_stats_for_local_mask(out, organ_patch, target_mask_patch, ring_width=ring_width)
    out[target_mask_patch] = mean
    return out


def distance_to_mask_mm(mask: np.ndarray, spacing: Sequence[float]) -> np.ndarray:
    if np.any(mask):
        return ndi.distance_transform_edt(
            ~mask.astype(bool), sampling=np.asarray(spacing, dtype=np.float32)
        ).astype(np.float32, copy=False)
    return np.full(mask.shape, np.inf, dtype=np.float32)


def organ_depth_mm(organ_mask: np.ndarray, spacing: Sequence[float]) -> np.ndarray:
    return ndi.distance_transform_edt(
        organ_mask.astype(bool), sampling=np.asarray(spacing, dtype=np.float32)
    ).astype(np.float32, copy=False)


def normalized_position(center: Sequence[int | float], shape: Sequence[int]) -> np.ndarray:
    center_arr = np.asarray(center, dtype=np.float32)
    shape_arr = np.maximum(np.asarray(shape, dtype=np.float32) - 1.0, 1.0)
    return center_arr / shape_arr * 2.0 - 1.0


def _mask_bbox(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    objects = ndi.find_objects(mask.astype(np.uint8, copy=False))
    if not objects or objects[0] is None:
        raise ValueError("Cannot sample from an empty mask")
    slc = objects[0]
    low = np.asarray([s.start for s in slc], dtype=np.int64)
    high = np.asarray([s.stop for s in slc], dtype=np.int64)
    return low, high


def sample_candidate_centers(
    center_mask: np.ndarray,
    patch_shape: Sequence[int],
    *,
    rng: np.random.Generator,
    requested: int,
    max_draws: int,
    diagnostics: dict | None = None,
) -> Iterator[tuple[int, int, int]]:
    """Random-rejection sampling without constructing/permuting every liver voxel."""
    counts = diagnostics if diagnostics is not None else {}
    counts.update(random_draws=0, random_coordinates_examined=0,
                  random_duplicate_centers=0, random_center_mask_rejections=0,
                  random_centers_yielded=0)
    if requested <= 0 or not np.any(center_mask):
        return
    patch = np.asarray(patch_shape, dtype=np.int64)
    shape = np.asarray(center_mask.shape, dtype=np.int64)
    half = patch // 2
    valid_low = half
    valid_high = shape - (patch - half)  # exclusive

    bbox_low, bbox_high = _mask_bbox(center_mask)
    low = np.maximum(valid_low, bbox_low)
    high = np.minimum(valid_high, bbox_high)
    if np.any(high <= low):
        return

    seen: set[tuple[int, int, int]] = set()
    yielded = 0
    draws = 0
    while yielded < requested and draws < max_draws:
        batch = min(2048, max_draws - draws)
        random_values = rng.random((batch, 3), dtype=np.float64)
        coords = np.floor(low + random_values * (high - low)).astype(np.int64)
        draws += batch
        counts["random_draws"] = draws
        for row in coords:
            counts["random_coordinates_examined"] += 1
            center = tuple(int(v) for v in row)
            if center in seen:
                counts["random_duplicate_centers"] += 1
                continue
            seen.add(center)
            if not center_mask[center]:
                counts["random_center_mask_rejections"] += 1
                continue
            yielded += 1
            counts["random_centers_yielded"] = yielded
            yield center
            if yielded >= requested:
                break


def build_candidate_pool(
    case: LoadedCase,
    source: SourceTumor,
    *,
    placement_mask: np.ndarray,
    full_organ_mask: np.ndarray,
    occupied_mask: np.ndarray,
    organ_distance: np.ndarray,
    rng: np.random.Generator,
    num_candidates: int,
    max_draws: int,
    min_liver_coverage: float,
    occupied_clearance_vox: int,
    min_center_separation_mm: float,
    candidate_oversample_factor: int = 20,
    required_candidates: int | None = None,
    force_exhaustive: bool = False,
    excluded_centers: Iterable[Sequence[int]] = (),
    diagnostics: dict | None = None,
    exhaustive_working_memory_bytes: int = 64 * 1024 * 1024,
) -> tuple[list[CandidateInfo], np.ndarray]:
    """Keep successful legacy proposals; exhaustively extend only shortfalls.

    ``required_candidates`` controls whether extension is necessary, not the
    target pool size. Cache ranking needs seven negatives, whereas an online
    bank needs its complete 128-center pool. Extension still targets the full
    ``num_candidates`` and never changes the source, constraints or resolution.
    The scratch-byte setting tiles computation, not the searched domain.
    """
    target = int(num_candidates)
    required = target if required_candidates is None else int(required_candidates)
    if not 1 <= required <= target:
        raise ValueError("Require 1 <= required_candidates <= num_candidates")
    if int(exhaustive_working_memory_bytes) < 1024:
        raise ValueError("Exhaustive search scratch budget must be at least 1024 bytes")
    excluded = {tuple(int(value) for value in center) for center in excluded_centers}
    if any(len(center) != 3 for center in excluded):
        raise ValueError("Excluded candidate centers must have three coordinates")
    stats = diagnostics if diagnostics is not None else {}
    stats.clear()
    stats.update(format=CANDIDATE_DIAGNOSTICS_FORMAT, search_version=CANDIDATE_SEARCH_VERSION,
                 target_candidates=target, required_candidates=required,
                 source_component=int(source.component_id),
                 source_anchor=[int(value) for value in source.anchor_center],
                 source_patch_shape=[int(value) for value in source.patch_mask.shape],
                 source_voxels=int(source.voxel_count), max_draws=int(max_draws),
                 min_liver_coverage=float(min_liver_coverage),
                 occupied_clearance_vox=int(occupied_clearance_vox),
                 min_center_separation_mm=float(min_center_separation_mm),
                 excluded_center_count=len(excluded), exhaustive_used=False,
                 fullsearch_exhausted=False, accepted=0)
    random_rejections = {name: 0 for name in
                         ("bounds", "excluded", "forbidden_overlap", "liver_coverage", "center_separation")}
    stats["random_rejections"] = random_rejections
    if occupied_clearance_vox > 0:
        structure = ndi.generate_binary_structure(3, 1)
        forbidden = ndi.binary_dilation(
            occupied_mask,
            structure=structure,
            iterations=int(occupied_clearance_vox),
        )
    else:
        forbidden = occupied_mask.astype(bool, copy=False)

    occupied_distance = distance_to_mask_mm(occupied_mask, case.spacing)
    desired_raw = max(num_candidates * int(candidate_oversample_factor), num_candidates)
    accepted: list[CandidateInfo] = []
    source_ring = context_ring_mask(source.patch_mask, width=3)
    tested_flat: set[int] = set()
    stats["requested_raw_centers"] = int(desired_raw)

    def append_candidate(center, coverage, occupied_distance_mm):
        slc = slices_for_center(center, source.patch_mask.shape, case.shape)
        if slc is None:
            raise RuntimeError("Verified candidate unexpectedly falls outside the volume")
        roi_image = case.image[slc]
        roi_organ = full_organ_mask[slc]
        stats_mask = source_ring & roi_organ
        values = roi_image[stats_mask]
        if values.size < 8:
            values = roi_image[roi_organ & ~source.patch_mask]
        if values.size == 0:
            values = roi_image.reshape(-1)
        accepted.append(CandidateInfo(
            center=tuple(int(value) for value in center), slices=slc,
            liver_coverage=float(coverage), border_distance_mm=float(organ_distance[center]),
            occupied_distance_mm=float(occupied_distance_mm),
            context_mean_hu=float(np.mean(values)), context_std_hu=float(np.std(values))))

    for center in sample_candidate_centers(
        placement_mask,
        source.patch_mask.shape,
        rng=rng,
        requested=desired_raw,
        max_draws=max_draws,
        diagnostics=stats,
    ):
        tested_flat.add(int(np.ravel_multi_index(center, case.shape)))
        if center in excluded:
            random_rejections["excluded"] += 1
            continue
        slc = slices_for_center(center, source.patch_mask.shape, case.shape)
        if slc is None:
            random_rejections["bounds"] += 1
            continue
        roi_liver = placement_mask[slc]
        roi_forbidden = forbidden[slc]
        if np.any(source.patch_mask & roi_forbidden):
            random_rejections["forbidden_overlap"] += 1
            continue

        coverage = float(np.sum(source.patch_mask & roi_liver) / max(1, source.voxel_count))
        if coverage < float(min_liver_coverage):
            random_rejections["liver_coverage"] += 1
            continue
        occupied_distance_mm = float(occupied_distance[center])
        if occupied_distance_mm < float(min_center_separation_mm):
            random_rejections["center_separation"] += 1
            continue
        append_candidate(center, coverage, occupied_distance_mm)
        if len(accepted) >= num_candidates:
            break
    stats["legacy_accepted"] = len(accepted)
    if len(accepted) < target and (len(accepted) < required or force_exhaustive):
        stats["exhaustive_used"] = True
        stats["fullsearch_exhausted"] = _extend_candidates_exhaustively(
            case, source, placement_mask, forbidden, occupied_distance,
            min_liver_coverage=float(min_liver_coverage),
            min_center_separation_mm=float(min_center_separation_mm),
            tested_flat=tested_flat, excluded=excluded, target=target,
            accepted=accepted, append_candidate=append_candidate, diagnostics=stats,
            working_memory_bytes=int(exhaustive_working_memory_bytes))
    stats["accepted"] = len(accepted)
    stats["required_candidates_met"] = len(accepted) >= required
    stats["target_candidates_met"] = len(accepted) >= target
    return accepted, occupied_distance


def _extend_candidates_exhaustively(
    case, source, placement_mask, forbidden, occupied_distance, *,
    min_liver_coverage, min_center_separation_mm, tested_flat, excluded,
    target, accepted, append_candidate, diagnostics, working_memory_bytes,
) -> bool:
    """Lexicographic full-domain search with batched, tiled exact mask tests.

    No complete-volume coordinate array or center-by-footprint tensor is kept.
    Scratch tiles contain at most the configured bytes of index/mask matrices.
    Source offsets and one 2-D plane of center indices are additional storage.
    True means every legal placement center was checked (including legacy).
    """
    rejected = {name: 0 for name in
                ("already_tested_or_excluded", "center_separation", "forbidden_overlap", "liver_coverage")}
    diagnostics.update(exhaustive_rejections=rejected, exhaustive_centers_evaluated=0,
                       exhaustive_valid_centers_evaluated=0,
                       exhaustive_working_memory_bytes=working_memory_bytes,
                       exhaustive_max_matrix_elements=0,
                       exhaustive_filter_order=["center_separation", "forbidden_overlap", "liver_coverage"])
    shape = np.asarray(case.shape, dtype=np.int64)
    patch = np.asarray(source.patch_mask.shape, dtype=np.int64)
    half = patch // 2
    low, high = half, shape - (patch - half) + 1
    # +1 includes the last center accepted by slices_for_center. The legacy
    # sampler is intentionally unchanged, including its historical RNG bounds.
    if np.any(high <= low) or not np.any(placement_mask):
        return True
    footprint_indices = np.flatnonzero(source.patch_mask)
    if not footprint_indices.size:
        raise CandidatePreparationError("empty_source_footprint", diagnostics)
    plane_stride = int(shape[1] * shape[2])
    offsets = ((footprint_indices // int(patch[1] * patch[2]) - half[0]) * plane_stride
               + ((footprint_indices // int(patch[2])) % patch[1] - half[1]) * int(shape[2])
               + (footprint_indices % patch[2] - half[2])).astype(np.int64)
    skip = np.asarray(sorted(tested_flat | {
        int(np.ravel_multi_index(center, case.shape)) for center in excluded
        if all(0 <= value < size for value, size in zip(center, case.shape))}), dtype=np.int64)
    forbidden_flat, placement_flat = forbidden.reshape(-1), placement_mask.reshape(-1)
    distance_flat = occupied_distance.reshape(-1)
    # int64 index + gathered bool + reduction working allowance per element.
    matrix_elements = max(1, working_memory_bytes // 16)
    # Do not evaluate thousands of large footprints when only (at most) the
    # remaining pool entries are needed. Repeated batches still cover the full
    # legal domain when anatomy rejects them; this is not a candidate cap.
    center_batch = min(max(1, int(math.sqrt(matrix_elements))), target - len(accepted))
    width = int(high[2] - low[2])
    for x in range(int(low[0]), int(high[0])):
        plane_indices = np.flatnonzero(placement_mask[x, low[1]:high[1], low[2]:high[2]])
        for start in range(0, len(plane_indices), center_batch):
            plane = plane_indices[start:start + center_batch]
            centers = np.column_stack((np.full(len(plane), x, dtype=np.int64),
                                       plane // width + low[1], plane % width + low[2]))
            flat = centers[:, 0] * plane_stride + centers[:, 1] * shape[2] + centers[:, 2]
            old = np.isin(flat, skip, assume_unique=True, kind="sort")
            rejected["already_tested_or_excluded"] += int(old.sum())
            centers, flat = centers[~old], flat[~old]
            diagnostics["exhaustive_centers_evaluated"] += len(flat)
            separated = distance_flat[flat] >= min_center_separation_mm
            rejected["center_separation"] += int((~separated).sum())
            centers, flat = centers[separated], flat[separated]
            if not len(flat):
                continue
            blocked = np.zeros(len(flat), dtype=bool)
            covered = np.zeros(len(flat), dtype=np.int64)
            offset_batch = max(1, matrix_elements // len(flat))
            for offset_start in range(0, len(offsets), offset_batch):
                indices = flat[:, None] + offsets[None, offset_start:offset_start + offset_batch]
                diagnostics["exhaustive_max_matrix_elements"] = max(
                    diagnostics["exhaustive_max_matrix_elements"], int(indices.size))
                blocked |= np.any(forbidden_flat[indices], axis=1)
                covered += np.sum(placement_flat[indices], axis=1, dtype=np.int64)
            coverage = covered / max(1, source.voxel_count)
            rejected["forbidden_overlap"] += int(blocked.sum())
            liver_ok = coverage >= min_liver_coverage
            rejected["liver_coverage"] += int((~blocked & ~liver_ok).sum())
            valid = np.flatnonzero(~blocked & liver_ok)
            diagnostics["exhaustive_valid_centers_evaluated"] += len(valid)
            for index in valid:
                center = tuple(int(value) for value in centers[index])
                append_candidate(center, float(coverage[index]), float(distance_flat[flat[index]]))
                if len(accepted) >= target:
                    return False
    return True


def feather_alpha(mask: np.ndarray, border: int) -> np.ndarray:
    if border <= 0:
        return mask.astype(np.float32)
    distance_inside = ndi.distance_transform_edt(mask.astype(bool))
    return np.clip(distance_inside / float(border), 0.0, 1.0).astype(np.float32)


def paste_source(
    out_image: np.ndarray,
    out_label: np.ndarray,
    occupied_mask: np.ndarray,
    source: SourceTumor,
    candidate: CandidateInfo,
    *,
    tumor_label: int,
    rng: np.random.Generator,
    intensity_scale_range: tuple[float, float],
    intensity_shift_range: tuple[float, float],
    blend_border: int,
) -> tuple[float, float]:
    scale = float(rng.uniform(*intensity_scale_range))
    shift = float(rng.uniform(*intensity_shift_range))
    slc = candidate.slices

    roi = out_image[slc]
    alpha = feather_alpha(source.patch_mask, blend_border)
    transformed = source.patch_image * scale + shift
    out_image[slc] = (1.0 - alpha) * roi + alpha * transformed

    label_roi = out_label[slc]
    label_roi[source.patch_mask] = int(tumor_label)
    occupied_roi = occupied_mask[slc]
    occupied_roi[source.patch_mask] = True
    return scale, shift


def source_context_stats(
    case: LoadedCase,
    source: SourceTumor,
    full_organ_mask: np.ndarray,
) -> tuple[float, float]:
    roi_organ = full_organ_mask[source.patch_slices]
    return context_stats_for_local_mask(
        source.patch_image,
        roi_organ,
        source.patch_mask,
        ring_width=3,
    )


def write_manifest(rows: Iterable[dict[str, object]], path: str | os.PathLike[str]) -> None:
    rows_list = list(rows)
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    if not rows_list:
        path_obj.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows_list:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path_obj.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_list)


def parse_float_pair(values: Sequence[float]) -> tuple[float, float]:
    if len(values) != 2:
        raise ValueError("Expected exactly two numbers")
    low, high = float(values[0]), float(values[1])
    if high < low:
        raise ValueError(f"Expected low <= high, got {values}")
    return low, high


def choose_from_top_k(
    scores: np.ndarray,
    *,
    rng: np.random.Generator,
    top_k: int = 1,
    temperature: float = 0.0,
) -> int:
    if scores.ndim != 1 or scores.size == 0:
        raise ValueError("scores must be a non-empty 1D array")
    k = max(1, min(int(top_k), scores.size))
    top_indices = np.argpartition(scores, -k)[-k:]
    top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
    if k == 1 or temperature <= 0:
        return int(top_indices[0])
    logits = scores[top_indices].astype(np.float64) / float(temperature)
    logits -= np.max(logits)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum()
    return int(rng.choice(top_indices, p=probabilities))


def format_center(center: Sequence[int | float]) -> str:
    return ",".join(f"{float(v):.3f}" for v in center)


def common_generation_kwargs(args: object) -> dict[str, object]:
    """Extract normalized generation settings from an argparse namespace."""
    return {
        "min_liver_coverage": float(getattr(args, "min_liver_coverage")),
        "occupied_clearance_vox": int(getattr(args, "occupied_clearance_vox")),
        "min_center_separation_mm": float(getattr(args, "min_center_separation_mm")),
        "max_draws": int(getattr(args, "max_draws")),
    }
