#!/usr/bin/env python3
"""V21 adaptive-geometry and unequal-node batching regression."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, fields as dataclass_fields
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from scipy import ndimage as ndi
from torch.nn import functional as F

from hiercp.sample import sample_dense_features_variable
from hiercp.geometry import (
    V21_SCHEMA_VERSION,
    adaptive_coordinate_sets,
    build_patch_payload,
    run_geometry_smoke,
)


def _resolve_device(requested: str | torch.device) -> torch.device:
    if isinstance(requested, torch.device):
        return requested
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    return device


def reference_per_graph(
    feature_map: torch.Tensor,
    grid: torch.Tensor,
    node_batch: torch.Tensor,
) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    for graph_index in range(int(feature_map.shape[0])):
        graph_grid = grid[node_batch == graph_index].view(1, -1, 1, 1, 3)
        sampled = F.grid_sample(
            feature_map[graph_index : graph_index + 1],
            graph_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        chunks.append(sampled[0, :, :, 0, 0].transpose(0, 1))
    return torch.cat(chunks, dim=0)


def run_variable_node_sampling(device: str | torch.device = "cpu") -> dict[str, Any]:
    resolved = _resolve_device(device)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(2107)
    counts = torch.tensor([7, 13, 5], dtype=torch.long)
    node_batch = torch.repeat_interleave(torch.arange(3), counts).to(resolved)
    grid = torch.empty((int(counts.sum()), 3), dtype=torch.float32).uniform_(-1.0, 1.0).to(resolved)
    feature_map = torch.randn((3, 6, 9, 10, 11), generator=generator, dtype=torch.float32).to(resolved)
    feature_map.requires_grad_(True)

    actual = sample_dense_features_variable(feature_map, grid, node_batch)
    reference = reference_per_graph(feature_map, grid, node_batch)
    maximum_error = float((actual - reference).abs().max().detach().cpu())
    if actual.shape != (int(counts.sum()), 6):
        raise AssertionError(f"Unexpected sampled feature shape: {tuple(actual.shape)}")
    if not torch.allclose(actual, reference, rtol=1e-5, atol=1e-6):
        raise AssertionError(f"Variable-node sampler mismatch: max_error={maximum_error}")

    loss = actual.square().mean() + actual.mean()
    loss.backward()
    gradient = feature_map.grad
    if gradient is None or not bool(torch.isfinite(gradient).all()):
        raise AssertionError("Non-finite or missing sampler gradient")
    gradient_l1 = float(gradient.abs().sum().detach().cpu())
    if gradient_l1 <= 0.0:
        raise AssertionError("Sampler backward produced a zero gradient")

    missing_type_rejected = False
    try:
        sample_dense_features_variable(
            feature_map.detach(),
            grid[node_batch != 1],
            node_batch[node_batch != 1],
        )
    except ValueError:
        missing_type_rejected = True
    if not missing_type_rejected:
        raise AssertionError("Missing semantic node type was not rejected")

    model_method_checked = False
    model_maximum_error: float | None = None
    try:
        from hiercp.model import LocalTumorContextPyGEncoder
    except (ImportError, ModuleNotFoundError):
        pass
    else:
        model_output = LocalTumorContextPyGEncoder._sample_dense_features(
            feature_map.detach(), grid, node_batch
        )
        model_maximum_error = float((model_output - reference.detach()).abs().max().cpu())
        if model_maximum_error > 1e-6:
            raise AssertionError(
                f"Patched model sampler mismatch: max_error={model_maximum_error}"
            )
        model_method_checked = True

    return {
        "device": str(resolved),
        "node_counts": counts.tolist(),
        "total_nodes": int(counts.sum()),
        "feature_channels": int(actual.shape[1]),
        "maximum_reference_error": maximum_error,
        "gradient_l1": gradient_l1,
        "finite_backward": True,
        "missing_semantic_type_rejected": missing_type_rejected,
        "model_method_checked": model_method_checked,
        "model_maximum_reference_error": model_maximum_error,
    }


@dataclass(frozen=True)
class LatestPatchFields:
    model_input: np.ndarray
    intensity: np.ndarray
    footprint: np.ndarray
    organ_mask: np.ndarray
    organ_depth: np.ndarray
    tumor_signed: np.ndarray
    tumor_outer: np.ndarray
    outside_tumor_mm: np.ndarray
    inside_tumor_mm: np.ndarray
    organ_boundary_mm: np.ndarray
    tumor_normals: np.ndarray
    organ_normals: np.ndarray
    curvature: np.ndarray


@dataclass(frozen=True)
class LegacyPatchFields:
    model_input: np.ndarray
    ct_norm: np.ndarray
    organ: np.ndarray
    footprint: np.ndarray
    tumor_sdf_mm: np.ndarray
    tumor_sdf_norm: np.ndarray
    liver_depth_mm: np.ndarray
    liver_depth_norm: np.ndarray
    gradient: np.ndarray
    local_mean: np.ndarray
    local_std: np.ndarray
    tumor_normal: np.ndarray
    tumor_curvature: np.ndarray
    liver_normal: np.ndarray
    liver_curvature: np.ndarray
    outside_tumor_mm: np.ndarray


def _compatibility_config() -> SimpleNamespace:
    return SimpleNamespace(
        graph_schema_version=V21_SCHEMA_VERSION,
        adaptive_nodes=True,
        adaptive_source_full_shape=True,
        patch_size=16,
        context_radius_mm=8.0,
        adaptive_roi_margin_mm=10.0,
        adaptive_roi_max_radius_mm=24.0,
        adaptive_roi_max_voxels=400_000,
        surface_area_per_node_mm2=12.0,
        surface_nodes_min=12,
        surface_nodes_max=48,
        interior_volume_per_node_mm3=30.0,
        interior_nodes_min=8,
        interior_nodes_max=32,
        context_ratio=2.0,
        context_nodes_min=24,
        context_nodes_max=96,
        context_ratio_min_audit=1.25,
        context_inner_radius_mm=1.0,
        context_outer_radius_mm=8.0,
        context_liver_surface_separation_mm=0.25,
        context_radial_bins=3,
        context_azimuth_bins=6,
        context_elevation_bins=3,
        context_sector_max_fraction=0.25,
        outward_step_mm=1.5,
        outward_max_steps=8,
        outward_snap_radius_mm=2.0,
        outward_min_cosine=0.0,
        liver_anchor_ratio=0.5,
        liver_anchor_nodes_min=6,
        liver_anchor_nodes_max=24,
        liver_anchor_search_mm=24.0,
        boundary_depth_mm=3.0,
        context_candidate_pool_max=5_000,
        kdtree_leafsize=8,
    )


def _construct(cls: type[Any], payload: dict[str, np.ndarray]) -> Any:
    names = [item.name for item in dataclass_fields(cls)]
    missing = [name for name in names if name not in payload]
    if missing:
        raise AssertionError(f"PatchFields compatibility payload missing {missing}")
    return cls(**{name: payload[name] for name in names})


def run_patchfields_compatibility() -> dict[str, Any]:
    shape = (49, 49, 49)
    center = np.asarray(shape, dtype=np.int64) // 2
    grid = np.indices(shape, dtype=np.float32)
    displacement = grid - center[:, None, None, None]
    radius = np.sqrt(np.sum(displacement**2, axis=0))
    organ = radius <= 22.0
    tumor = radius <= 5.0
    coords = np.argwhere(tumor)
    lo = coords.min(axis=0)
    hi = coords.max(axis=0) + 1
    slices = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    footprint = tumor[slices]
    image = (grid[0] * 0.25 + grid[1] * 0.1 - grid[2] * 0.05).astype(np.float32)
    organ_depth = ndi.distance_transform_edt(organ).astype(np.float32)
    config = _compatibility_config()
    payload = build_patch_payload(
        image=image,
        center=tuple(int(value) for value in center),
        footprint=footprint,
        full_organ=organ,
        organ_depth=organ_depth,
        spacing=(1.0, 1.0, 1.0),
        config=config,
        erase_target=False,
        ct_clip=(-100.0, 200.0),
    )
    latest = _construct(LatestPatchFields, payload)
    legacy = _construct(LegacyPatchFields, payload)
    latest_nodes = adaptive_coordinate_sets(
        latest, config, np.random.default_rng(3201), (1.0, 1.0, 1.0)
    )
    legacy_nodes = adaptive_coordinate_sets(
        legacy, config, np.random.default_rng(3201), (1.0, 1.0, 1.0)
    )
    counts: dict[str, int] = {}
    for name in ("surface", "interior", "context", "liver_surface"):
        if not np.array_equal(latest_nodes[name], legacy_nodes[name]):
            raise AssertionError(f"PatchFields schema changed V21 {name} coordinates")
        counts[name] = int(len(latest_nodes[name]))
    if int(latest.footprint.sum()) != int(footprint.sum()):
        raise AssertionError("Latest PatchFields path lost tumor voxels")
    return {
        "latest_schema_fields": [item.name for item in dataclass_fields(LatestPatchFields)],
        "legacy_schema_fields": [item.name for item in dataclass_fields(LegacyPatchFields)],
        "paste_voxels": int(footprint.sum()),
        "graph_voxels": int(latest.footprint.sum()),
        "crop_voxel_loss": int(footprint.sum() - latest.footprint.sum()),
        "node_counts": counts,
        "identical_coordinates_across_schemas": True,
    }


def run_v21_smoke(device: str | torch.device = "cpu") -> dict[str, Any]:
    return {
        "status": "PASS",
        "geometry": run_geometry_smoke(),
        "patchfields_compatibility": run_patchfields_compatibility(),
        "variable_node_sampling": run_variable_node_sampling(device),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_v21_smoke(args.device)
    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
