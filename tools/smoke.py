#!/usr/bin/env python3
"""CUDA/CPU regression for canonical radius graphs and sampled HeteroGATv2."""
from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch


def _full_config():
    from hiercp.schema import GraphBuildConfig

    config = GraphBuildConfig(
        patch_size=20,
        context_radius_mm=9.0,
        context_shells_mm=(2.5, 5.5, 9.0),
        boundary_depth_mm=2.0,
        candidate_k=2,
        num_regions=5,
        region_sample_voxels=800,
        region_lloyd_iters=3,
        region_k=2,
        num_prototypes=3,
        prototype_top_m=2,
        prototype_k=2,
        prototype_lloyd_iters=3,
        max_lesions=3,
        graph_schema_version="full_v22",
        adaptive_source_full_shape=True,
        adaptive_roi_margin_mm=10.0,
        adaptive_roi_max_radius_mm=24.0,
        adaptive_roi_max_voxels=2_000_000,
        context_inner_radius_mm=1.5,
        context_outer_radius_mm=9.0,
        context_liver_surface_separation_mm=0.5,
        context_radial_bins=3,
        context_azimuth_bins=6,
        context_elevation_bins=3,
        liver_anchor_search_mm=24.0,
        kdtree_leafsize=16,
        canonical_full_graph=True,
        canonical_surface_spacing_mm=2.0,
        canonical_interior_spacing_mm=2.5,
        canonical_context_spacing_mm=2.5,
        canonical_liver_spacing_mm=4.0,
        canonical_node_limit=100_000,
        sample_context_nodes=16,
        sample_hops=1,
        sample_interface_radius_mm=4.5,
        sample_hop_radius_mm=3.5,
        surface_edge_radius_mm=3.0,
        interior_edge_radius_mm=4.0,
        context_edge_radius_mm=4.0,
        interface_edge_radius_mm=4.5,
        cross_edge_radius_mm=4.5,
        liver_edge_radius_mm=6.0,
        correspondence_radius_mm=3.5,
        canonical_relation_edge_limit=500_000,
        sample_relation_edge_limit=250_000,
    )
    config.validate()
    return config


def _canonical_sample():
    from hiercp.cache import CACHE_FORMAT
    from hiercp.common import (
        CasePaths,
        LoadedCase,
        build_candidate_pool,
        choose_source_tumor,
        organ_depth_mm,
        synthetic_array_signature,
    )
    from hiercp.curriculum import build_training_specs
    from hiercp.local import build_local_graph, prepare_local_source
    from hiercp.prototype import build_prototype_bank
    from hiercp.region import (
        REGION_CACHE_FILENAME,
        REGION_CACHE_FORMAT,
        REGION_CACHE_STORAGE,
        _context_only_image,
        build_patient_regions,
        load_patient_regions,
        save_patient_regions,
    )
    from hiercp.hierarchy import build_patient_graph, build_prototype_graph

    config = _full_config()
    shape = (40, 40, 40)
    z, y, x = np.indices(shape)
    center = np.asarray(shape, dtype=np.float32) / 2.0
    liver = (
        (z - center[0]) ** 2 / 17.0**2
        + (y - center[1]) ** 2 / 16.0**2
        + (x - center[2]) ** 2 / 15.0**2
        <= 1.0
    )
    tumor = (z - 20) ** 2 + (y - 19) ** 2 + (x - 17) ** 2 <= 3.5**2
    label = np.zeros(shape, dtype=np.int16)
    label[liver] = 1
    label[tumor] = 2
    image = np.random.default_rng(1).normal(70.0, 11.0, shape).astype(np.float32)
    image[tumor] += 24.0
    case = LoadedCase(
        paths=CasePaths("synthetic", Path("synthetic_image"), Path("synthetic_label")),
        image=image,
        label=label,
        image_affine=np.eye(4),
        label_affine=np.eye(4),
        image_header=None,
        label_header=None,
        spacing=np.asarray([1.2, 1.1, 1.0], dtype=np.float32),
        image_source_signature=synthetic_array_signature(image),
        label_source_signature=synthetic_array_signature(label),
    )
    source, _, _ = choose_source_tumor(
        image,
        label,
        tumor_label=2,
        rng=np.random.default_rng(2),
        selection="largest",
        pad=3,
    )
    descriptor_image = _context_only_image(case, liver_label=1, tumor_label=2)
    if np.allclose(descriptor_image[tumor], image[tumor]):
        raise RuntimeError("Region descriptors retain real tumor intensity")
    regions = build_patient_regions(
        case,
        liver_label=1,
        tumor_label=2,
        config=config,
        rng=np.random.default_rng(3),
        ct_clip=(-200.0, 250.0),
    )
    with TemporaryDirectory(prefix="hiercp_region_cache_smoke_") as cache_directory:
        cache_root = Path(cache_directory) / "synthetic"
        metadata = {
            "format": REGION_CACHE_FORMAT,
            "shape": list(shape),
            "case_id": "synthetic",
        }
        raw_bytes = sum(
            int(array.nbytes)
            for array in (
                regions.full_organ_mask,
                regions.organ_depth,
                regions.region_labels,
                regions.region_features,
                regions.region_positions,
                regions.region_edge_index,
                regions.region_centers_vox,
            )
        )
        save_patient_regions(regions, cache_root, metadata=metadata)
        loaded_regions, loaded_metadata = load_patient_regions(cache_root, mmap=True)
        if loaded_metadata.get("storage") != REGION_CACHE_STORAGE:
            raise RuntimeError("Compact region-cache storage marker is missing")
        if not (cache_root / REGION_CACHE_FILENAME).is_file():
            raise RuntimeError("Compact region-cache archive was not created")
        if any((cache_root / name).exists() for name in (
            "full_organ_mask.npy", "organ_depth.npy", "region_labels.npy"
        )):
            raise RuntimeError("Full-volume legacy arrays were written by the compact cache")
        for name in (
            "full_organ_mask",
            "organ_depth",
            "region_labels",
            "region_features",
            "region_positions",
            "region_edge_index",
            "region_centers_vox",
        ):
            original = np.asarray(getattr(regions, name))
            restored = np.asarray(getattr(loaded_regions, name))
            if not np.array_equal(original, restored):
                raise RuntimeError(f"Compact region-cache roundtrip changed {name}")
        stored_bytes = int((cache_root / REGION_CACHE_FILENAME).stat().st_size)
        if stored_bytes >= raw_bytes:
            raise RuntimeError(
                f"Compact region cache did not reduce storage: {stored_bytes} >= {raw_bytes}"
            )
    bank = build_prototype_bank(
        [
            ("synthetic", regions.region_features),
            ("synthetic_shifted", regions.region_features + 0.01),
        ],
        config=config,
        rng=np.random.default_rng(4),
    )
    full_organ = (label == 1) | (label == 2)
    organ_depth = organ_depth_mm(full_organ, case.spacing)
    candidates, _ = build_candidate_pool(
        case,
        source,
        placement_mask=label == 1,
        full_organ_mask=full_organ,
        occupied_mask=label == 2,
        organ_distance=organ_depth,
        rng=np.random.default_rng(5),
        num_candidates=20,
        max_draws=80_000,
        min_liver_coverage=0.8,
        occupied_clearance_vox=1,
        min_center_separation_mm=7.0,
    )
    specs = build_training_specs(
        case,
        source,
        candidates,
        regions,
        bank,
        total_candidates=6,
        easy_fraction=0.34,
        inter_fraction=0.33,
        intra_fraction=0.33,
        tumor_label=2,
        config=config,
        rng=np.random.default_rng(6),
    )
    if specs is None:
        raise RuntimeError("Synthetic curriculum could not be constructed")

    prepared = prepare_local_source(
        case,
        source,
        full_organ_mask=full_organ,
        organ_depth=organ_depth,
        config=config,
        rng=np.random.default_rng(10),
        ct_clip=(-200.0, 250.0),
    )
    built = [
        build_local_graph(
            case,
            source,
            spec,
            full_organ_mask=full_organ,
            organ_depth=organ_depth,
            config=config,
            rng=np.random.default_rng(11 + index),
            ct_clip=(-200.0, 250.0),
            prepared_source=prepared,
        )
        for index, spec in enumerate(specs)
    ]
    if any(item.graph is not None for item in built):
        raise RuntimeError("Cache builder materialised a graph before view sampling")
    if not all(item.source_local["format"] == "canonical-full-v22" for item in built):
        raise RuntimeError("Canonical full-graph source format is missing")
    from hiercp.schema import LOCAL_EDGE_TYPES
    full_edges = {**built[0].source_local["edges"], **built[0].target_local["edges"]}
    if set(full_edges) != set(LOCAL_EDGE_TYPES):
        raise RuntimeError("Canonical full graph does not contain every local relation")
    if not any(int(edge.shape[1]) > 0 for edge in full_edges.values()):
        raise RuntimeError("Canonical full graph contains no physical-radius edges")
    source_voxels = int(source.full_mask.sum())
    graph_voxels = int(built[0].source_local["footprint_voxels"])
    if source_voxels != graph_voxels:
        raise RuntimeError(
            f"Full tumor footprint was cropped: source={source_voxels}, graph={graph_voxels}"
        )
    positive_footprint = built[0].target_patch[1] > 0.5
    if not positive_footprint.any():
        raise RuntimeError("Positive virtual footprint is empty")
    if np.allclose(
        built[0].source_patch[0][positive_footprint],
        built[0].target_patch[0][positive_footprint],
    ):
        raise RuntimeError("Positive target context was not erased")

    patient_graph = build_patient_graph(
        case,
        source,
        specs,
        regions,
        tumor_label=2,
        config=config,
        ct_clip=(-200.0, 250.0),
    )
    prototype_graph = build_prototype_graph(
        specs,
        patient_graph,
        regions,
        bank,
        config=config,
    )
    sample = {
        "format": CACHE_FORMAT,
        "prototype_fingerprint": bank.fingerprint(),
        "case_id": "synthetic",
        "sample_index": 0,
        "split": "train",
        "source_component": int(source.component_id),
        "source_patch": torch.from_numpy(prepared.source_patch.astype(np.float16)),
        "target_patches": torch.from_numpy(
            np.stack([item.target_patch for item in built]).astype(np.float16)
        ),
        "source_local": built[0].source_local,
        "target_locals": [item.target_local for item in built],
        "patient_graph": patient_graph,
        "prototype_graph": prototype_graph,
        "difficulties": torch.tensor(
            [spec.difficulty for spec in specs], dtype=torch.long
        ),
        "corruptions": torch.tensor(
            [spec.corruption for spec in specs], dtype=torch.long
        ),
        "candidate_centers": torch.tensor(
            [spec.center for spec in specs], dtype=torch.long
        ),
        "candidate_regions": torch.tensor(
            [spec.region_id for spec in specs], dtype=torch.long
        ),
        "candidate_prototypes": torch.tensor(
            [spec.prototype_id for spec in specs], dtype=torch.long
        ),
        "graph_config": config.to_dict(),
        "ct_clip": (-200.0, 250.0),
    }
    return sample, config


def _identical_context_is_complete(
    first_ids: np.ndarray, second_ids: np.ndarray, positions: np.ndarray,
    anchors: np.ndarray, config,
) -> bool:
    """DEBUG proof of no free seed choice, not a blanket identical-view waiver.

    Full canonical retention is always legitimate. A smaller deterministic set
    is legitimate only when mandatory interface seeds exhaust the seed budget
    and BOTH views equal their independently reconstructed complete hop closure.
    If optional seed slots remain, identical partial views are still rejected.
    """
    from scipy.spatial import cKDTree

    count = int(positions.shape[0])
    if count == 0:
        return False
    first = np.sort(np.asarray(first_ids, dtype=np.int64))
    second = np.sort(np.asarray(second_ids, dtype=np.int64))
    all_ids = np.arange(count, dtype=np.int64)
    if np.array_equal(first, all_ids) and np.array_equal(second, all_ids):
        return True
    budget = int(config.sample_context_nodes)
    if budget <= 0 or count <= budget:
        return False  # These contracts require full retention, checked above.
    tree = cKDTree(positions)

    def neighbors(query, radius):
        rows = tree.query_ball_point(query, float(radius))
        return np.unique(np.asarray([item for row in rows for item in row], dtype=np.int64))

    required = neighbors(anchors, config.sample_interface_radius_mm)
    if required.size < budget:
        return False
    expected = required
    for _ in range(int(config.sample_hops)):
        expected = np.union1d(expected, neighbors(positions[expected], config.sample_hop_radius_mm))
    return np.array_equal(first, expected) and np.array_equal(second, expected)


def _assert_sampled_views(sample: dict, config) -> tuple[dict, bool]:
    from hiercp.sample import _edge_radius, _induced_edge_index, materialize_sample_views
    from hiercp.schema import LOCAL_EDGE_TYPES, LOCAL_NODE_TYPES

    materialized = materialize_sample_views(
        copy.deepcopy(sample), training=True, epoch=3, global_seed=42
    )
    first = materialized["local_graphs"]
    second = materialized["local_graphs_view2"]
    if len(first) != len(second) or not first:
        raise RuntimeError("Two sampled local views were not materialised")
    changed = False
    reduced = False
    all_contexts_complete = True
    for candidate_index, (graph_a, graph_b) in enumerate(zip(first, second)):
        source_local = sample["source_local"]
        target_local = sample["target_locals"][candidate_index]
        canonical_edges = {**source_local["edges"], **target_local["edges"]}
        all_nodes = {**source_local["nodes"], **target_local["nodes"]}
        for node_type in LOCAL_NODE_TYPES:
            full_count = int(graph_a.canonical_counts[0, LOCAL_NODE_TYPES.index(node_type)])
            sample_count = int(graph_a[node_type].num_nodes)
            if sample_count > full_count:
                raise RuntimeError(f"Sampled {node_type} exceeds canonical count")
            if node_type in {"source_context", "target_context"}:
                reduced |= sample_count < full_count
                changed |= not torch.equal(
                    graph_a[node_type].full_id, graph_b[node_type].full_id
                )
                anchors = source_local["nodes"]["tumor_surface"]["pos_mm"].cpu().numpy()
                if node_type == "target_context":
                    anchors = anchors @ target_local["transform"].cpu().numpy().T
                all_contexts_complete &= _identical_context_is_complete(
                    graph_a[node_type].full_id.cpu().numpy(),
                    graph_b[node_type].full_id.cpu().numpy(),
                    all_nodes[node_type]["pos_mm"].cpu().numpy(), anchors, config,
                )
            elif sample_count != full_count:
                raise RuntimeError(f"Anchor type {node_type} was sampled instead of preserved")
        transform = graph_a.target_transform[0].cpu().numpy()
        for relation_index, edge_type in enumerate(LOCAL_EDGE_TYPES):
            source_type, relation, destination_type = edge_type
            edge = graph_a[edge_type].edge_index.cpu().numpy()
            expected = _induced_edge_index(
                canonical_edges[edge_type],
                graph_a[source_type].full_id.cpu().numpy(),
                graph_a[destination_type].full_id.cpu().numpy(),
                source_count=int(all_nodes[source_type]["x"].shape[0]),
                destination_count=int(all_nodes[destination_type]["x"].shape[0]),
            )
            if not np.array_equal(edge, expected):
                raise RuntimeError(f"Sampled relation is not the exact induced edge set: {edge_type}")
            if int(graph_a.canonical_edge_counts[0, relation_index]) != int(
                canonical_edges[edge_type].shape[1]
            ):
                raise RuntimeError(f"Canonical edge count metadata changed: {edge_type}")
            if edge.shape[1] == 0:
                continue
            source = graph_a[source_type].pos_mm.cpu().numpy()
            destination = graph_a[destination_type].pos_mm.cpu().numpy()
            if (
                source_type == "tumor_surface" and relation == "interfaces_target"
            ) or (
                source_type == "source_context" and relation == "corresponds_to"
            ):
                source = source @ transform.T
            delta = destination[edge[1]] - source[edge[0]]
            maximum = float(np.linalg.norm(delta, axis=1).max())
            radius = float(_edge_radius(edge_type, config))
            if maximum > radius + 1e-4:
                raise RuntimeError(
                    f"Radius edge violation {edge_type}: {maximum:.4f} > {radius:.4f}"
                )
    if not reduced and not all_contexts_complete:
        raise RuntimeError("Synthetic canonical graph was never reduced by the view sampler")
    if not changed and not all_contexts_complete:
        raise RuntimeError("Two stochastic graph views selected identical node sets")
    if not changed:
        print("[OK] Identical debug views preserve the exact mandatory seed/hop closure; no free seed choice")
    return materialized, changed


def _assert_variable_dense_sampling(device: torch.device) -> None:
    from hiercp.sample import sample_dense_features_variable

    torch.manual_seed(17)
    counts = torch.tensor([7, 13, 5], device=device)
    node_batch = torch.repeat_interleave(torch.arange(3, device=device), counts)

    # Reproduce the production cache path exactly: dense maps are float32
    # outside AMP, while cached normalized node grids are float16.
    feature_map = torch.randn(3, 6, 5, 6, 7, device=device, requires_grad=True)
    grid = (torch.rand(int(counts.sum()), 3, device=device) * 2.0 - 1.0).half()
    sampled = sample_dense_features_variable(feature_map, grid, node_batch)
    if tuple(sampled.shape) != (25, 6):
        raise RuntimeError(f"Variable dense sample shape is wrong: {tuple(sampled.shape)}")
    if sampled.dtype != feature_map.dtype:
        raise RuntimeError(
            f"Variable dense sampler changed dtype: {sampled.dtype} != {feature_map.dtype}"
        )
    sampled.square().mean().backward()
    if feature_map.grad is None or not torch.isfinite(feature_map.grad).all():
        raise RuntimeError("Mixed-dtype variable-node sampler backward is not finite")

    # Also exercise the CUDA AMP direction: float16 feature maps with a
    # float32 grid. The sampler must normalize the grid dtype internally.
    if device.type == "cuda":
        amp_map = torch.randn(
            3, 6, 5, 6, 7, device=device, dtype=torch.float16, requires_grad=True
        )
        float_grid = grid.float()
        amp_sampled = sample_dense_features_variable(amp_map, float_grid, node_batch)
        if amp_sampled.dtype != torch.float16:
            raise RuntimeError(f"CUDA AMP sampler dtype is wrong: {amp_sampled.dtype}")
        amp_sampled.float().square().mean().backward()
        if amp_map.grad is None or not torch.isfinite(amp_map.grad).all():
            raise RuntimeError("CUDA AMP variable-node sampler backward is not finite")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    try:
        import torch_geometric
        from torch_geometric.data import Batch, HeteroData
        from torch_geometric.nn import GATv2Conv, HeteroConv
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "torch_geometric is not installed. Run: python -m tools.install"
        ) from exc

    from hiercp.data import HierarchicalCacheDataset, collate_samples
    from hiercp.loss import CurriculumConfig, curriculum_ranking_loss
    from hiercp.model import HierarchicalPyGPlacementModel
    from hiercp.pipeline import _checkpoint_selection_is_better
    from hiercp.schema import (
        PATIENT_POSITION_EDGE_COLUMNS,
        UPPER_FORBIDDEN_RAW_COLUMNS,
    )
    from hiercp.tensor import resolve_device

    first_selection = {
        "mrr": 1.0,
        "acc": 1.0,
        "margin": 0.1,
        "ranking": 0.3,
        "consistency": 0.02,
    }
    later_selection = {
        "mrr": 1.0,
        "acc": 1.0,
        "margin": 0.4,
        "ranking": 0.2,
        "consistency": 0.01,
    }
    lower_mrr = {
        "mrr": 0.99,
        "acc": 1.0,
        "margin": 10.0,
        "ranking": 0.0,
        "consistency": 0.0,
    }
    if not _checkpoint_selection_is_better(later_selection, first_selection):
        raise RuntimeError("Checkpoint margin tie-break did not replace an MRR tie")
    if _checkpoint_selection_is_better(lower_mrr, first_selection):
        raise RuntimeError("Checkpoint tie-break overrode the primary MRR metric")

    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"))
    device = resolve_device(args.device)
    _assert_variable_dense_sampling(device)
    canonical, config = _canonical_sample()
    materialized, _ = _assert_sampled_views(canonical, config)
    batch = collate_samples([materialized]).to(device)

    model = HierarchicalPyGPlacementModel(
        hidden_dim=16,
        heads=4,
        local_layers=2,
        patient_layers=1,
        prototype_layers=1,
        dropout=0.0,
        dense_base_channels=4,
        dense_feature_dim=8,
        dense_batch_size=3,
        channels_last_3d=True,
        checkpoint_local_blocks=True,
        checkpoint_dense_encoder=True,
    ).to(device)
    if not model.local_encoder.checkpoint_local_blocks:
        raise RuntimeError("Local GAT activation checkpointing is disabled")
    if not model.local_encoder.checkpoint_dense_encoder:
        raise RuntimeError("Dense encoder activation checkpointing is disabled")
    blocks = (
        model.local_encoder.blocks[0],
        model.patient_encoder.blocks[0],
        model.prototype_encoder.blocks[0],
    )
    for block in blocks:
        if not isinstance(block.conv, HeteroConv):
            raise RuntimeError("A hierarchy level is not using PyG HeteroConv")
        if not all(isinstance(module, GATv2Conv) for module in block.conv.convs.values()):
            raise RuntimeError("A typed relation is not using GATv2Conv")

    model.train()
    output = model(batch)
    if len(output.scores) != 1 or tuple(output.scores[0].shape) != (6,):
        raise RuntimeError(
            f"Unexpected score shape: {[tuple(score.shape) for score in output.scores]}"
        )
    for graph_batch in (
        output.local_batch,
        output.local_batch_view2,
        output.patient_batch,
        output.prototype_batch,
    ):
        if not isinstance(graph_batch, (Batch, HeteroData)):
            raise RuntimeError("A hierarchy level did not return PyG Batch/HeteroData")
    if not torch.isfinite(output.consistency) or float(output.consistency) < 0.0:
        raise RuntimeError("Sample-view consistency loss is invalid")
    ranking, metrics = curriculum_ranking_loss(
        output.scores,
        batch.difficulty_list(),
        epoch=30,
        config=CurriculumConfig(
            easy_epochs=2,
            inter_epochs=4,
            intra_epochs=6,
            model_mine_start_epoch=7,
        ),
    )
    loss = ranking + 0.1 * output.consistency
    loss.backward()
    for name, block in zip(("local", "patient", "prototype"), blocks):
        gradients = [p.grad for p in block.conv.parameters() if p.requires_grad]
        if not any(
            grad is not None
            and torch.isfinite(grad).all()
            and bool(torch.any(grad != 0))
            for grad in gradients
        ):
            raise RuntimeError(f"No finite non-zero gradient reached the {name} GAT block")

    model.eval()
    inference_batch = collate_samples([copy.deepcopy(materialized)])
    with torch.inference_mode():
        full_scores = model(copy.deepcopy(inference_batch)).scores[0].float()
        chunked_scores = model.score_inference_chunked(
            copy.deepcopy(inference_batch), local_chunk_size=2
        )[0].float()
    if not torch.allclose(full_scores, chunked_scores, rtol=1e-5, atol=1e-5):
        error = float((full_scores - chunked_scores).abs().max().cpu())
        raise RuntimeError(
            f"Chunked generation changed candidate scores: max_error={error}"
        )

    corrupted = copy.deepcopy(inference_batch)
    for graph, node_type in (
        (corrupted.patient_batch, "tumor"),
        (corrupted.patient_batch, "candidate"),
        (corrupted.prototype_batch, "candidate"),
    ):
        raw = graph[node_type].raw_x.clone()
        raw[:, list(UPPER_FORBIDDEN_RAW_COLUMNS)] = 123.0
        graph[node_type].raw_x = raw
    for edge_type in corrupted.patient_batch.edge_types:
        if "tumor" not in (edge_type[0], edge_type[2]):
            continue
        edge_attr = corrupted.patient_batch[edge_type].edge_attr.clone()
        edge_attr[:, list(PATIENT_POSITION_EDGE_COLUMNS)] = -321.0
        corrupted.patient_batch[edge_type].edge_attr = edge_attr
    with torch.inference_mode():
        corrupted_scores = model(corrupted).scores[0].float()
    if not torch.allclose(full_scores, corrupted_scores, rtol=1e-6, atol=1e-6):
        error = float((full_scores - corrupted_scores).abs().max().cpu())
        raise RuntimeError(
            f"Forbidden upper features still affect scores: max_error={error}"
        )

    with TemporaryDirectory(prefix="hiercp_full_smoke_") as temporary_directory:
        cache_path = Path(temporary_directory) / "synthetic__000.pt"
        torch.save(canonical, cache_path)
        dataset = HierarchicalCacheDataset(
            [cache_path], mmap=True, training=True, seed=42
        )
        dataset.set_epoch(1)
        epoch1 = dataset[0]
        seed1 = int(epoch1["local_graphs"][0].view_seed.item())
        dataset.set_epoch(2)
        epoch2 = dataset[0]
        seed2 = int(epoch2["local_graphs"][0].view_seed.item())
        if seed1 == seed2:
            raise RuntimeError("Epoch-dependent sampled graph view did not change")

    print("[OK] cached full radius graph + induced spatial HeteroGATv2 smoke test")
    print("torch:", torch.__version__)
    print("torch_geometric:", torch_geometric.__version__)
    print("device:", device)
    print("scores:", tuple(output.scores[0].shape))
    print("loss:", float(loss.detach().cpu()))
    print("consistency:", float(output.consistency.detach().cpu()))
    print("active_max_difficulty:", float(metrics["active_max_difficulty"].cpu()))
    print("local relations:", len(model.local_edge_types))
    print("canonical schema:", config.graph_schema_version)
    print("shortcut-safe upper features: PASS")
    print("chunked inference equivalence: PASS")


if __name__ == "__main__":
    main()
