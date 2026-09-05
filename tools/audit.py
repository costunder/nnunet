#!/usr/bin/env python3
"""Static audit for the canonical full-graph sampled HeteroGATv2 pipeline."""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import py_compile

from hiercp.schema import graph_config_from_dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _source(path: Path) -> str:
    if not path.is_file():
        raise RuntimeError(f"Required active file is missing: {path}")
    text = path.read_text(encoding="utf-8")
    ast.parse(text, filename=str(path))
    py_compile.compile(str(path), doraise=True)
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    args = parser.parse_args()
    root = Path(args.project_root).expanduser().resolve()
    if root != PROJECT_ROOT:
        raise SystemExit(f"This audit is bound to its package root: {PROJECT_ROOT}")

    paths = {
        "schema": root / "hiercp" / "schema.py",
        "region": root / "hiercp" / "region.py",
        "spatial": root / "hiercp" / "spatial.py",
        "sample": root / "hiercp" / "sample.py",
        "local": root / "hiercp" / "local.py",
        "cache": root / "hiercp" / "cache.py",
        "dataset": root / "hiercp" / "data.py",
        "model": root / "hiercp" / "model.py",
        "method": root / "hiercp" / "pipeline.py",
        "smoke": root / "tools" / "smoke.py",
        "causality": root / "tools" / "causality.py",
        "run": root / "run.py",
    }
    sources = {name: _source(path) for name, path in paths.items()}
    config_path = root / "config" / "train.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    graph = graph_config_from_dict(config["graph"])
    if config.get("method") != "hiercp-full":
        raise RuntimeError("config/train.json does not select method=hiercp-full")
    if graph.graph_schema_version != "full_v22" or not graph.canonical_full_graph:
        raise RuntimeError("config/train.json does not enable the canonical full graph")
    if float(config["training"].get("consistency_weight", 0.0)) <= 0.0:
        raise RuntimeError("Sample-view consistency loss is disabled")
    fixed_validation_epoch = int(
        config["training"].get("fixed_validation_epoch", 0)
    )
    if fixed_validation_epoch < int(
        config["training"].get("model_mine_start_epoch", 1)
    ):
        raise RuntimeError(
            "Fixed validation does not activate the complete negative curriculum"
        )
    if int(config["training"].get("checkpoint_metric_precision", -1)) < 0:
        raise RuntimeError("Checkpoint tie-break precision is not configured")
    if bool(config["training"].get("cuda_prefetch", True)):
        raise RuntimeError("CUDA input prefetch must be disabled for full graph batches")
    if not bool(config["model"].get("checkpoint_local_blocks", False)):
        raise RuntimeError("Local GAT activation checkpointing is disabled")
    if not bool(config["model"].get("checkpoint_dense_encoder", False)):
        raise RuntimeError("Dense encoder activation checkpointing is disabled")
    if int(config["generation"].get("local_candidate_chunk_size", 0)) < 1:
        raise RuntimeError("Generation local candidate chunking is disabled")

    required = {
        "schema": (
            'UPPER_FEATURE_POLICY: Final[str] = "shortcut_safe_upper_v1"',
            "UPPER_FORBIDDEN_RAW_COLUMNS",
            "PATIENT_POSITION_EDGE_COLUMNS",
        ),
        "region": (
            'REGION_CACHE_STORAGE = "compact_crop_npz_v1"',
            'REGION_CACHE_FILENAME = "regions.npz"',
            "np.savez_compressed",
            "_load_compact_regions",
            "_load_legacy_regions",
        ),
        "spatial": (
            "canonical_coordinate_sets",
            "radius_edges",
            "cross_radius_edges",
            "cKDTree",
            "query_pairs",
            "query_ball_tree",
        ),
        "sample": (
            "materialize_sample_views",
            "stable_view_seed",
            "sample_dense_features_variable",
            "build_local_view",
            "query_ball_point",
            "_induced_edge_index",
            "canonical_counts",
            "canonical_edge_counts",
            "sampled_counts",
        ),
        "local": (
            "canonical-full-v22",
            "canonical_coordinate_sets",
            "_canonical_edges",
            '"edges"',
            "graph=None",
        ),
        "cache": (
            'CACHE_FORMAT = "full-cache"',
            '"source_local"',
            '"target_locals"',
        ),
        "dataset": (
            "local_batch_view2",
            "materialize_sample_views",
            "set_epoch",
        ),
        "model": (
            "HeteroConv",
            "GATv2Conv",
            "forward_graph",
            "_view_consistency",
            "batch[node_type].batch",
            "checkpoint_local_blocks",
            "checkpoint_dense_encoder",
            "use_reentrant=False",
            "_mask_upper_shortcuts",
            "_mask_tumor_spatial_edge_attr",
            "score_inference_chunked",
        ),
        "method": (
            'CHECKPOINT_METHOD = "hiercp-full"',
            "consistency_weight",
            "train_dataset.set_epoch(epoch)",
            "_checkpoint_selection_record",
            "_checkpoint_selection_is_better",
            "fixed_validation_epoch",
            "val_margin",
            "upper_feature_policy",
            "local_candidate_chunk_size",
        ),
        "causality": (
            "node_order",
            "target_context",
            "edge_attr_zero",
            "topology_shuffle",
            "upper_position_noise",
            "upper_clearance_noise",
            "shortcut_safety_supported",
            "context_causality_supported",
        ),
        "run": (
            '"causality"',
            '"tools.causality"',
            '"hiercp-full"',
            "_prepare_storage_preflight",
            "shutil.disk_usage",
        ),
    }
    for name, fragments in required.items():
        missing = [fragment for fragment in fragments if fragment not in sources[name]]
        if missing:
            raise RuntimeError(f"Missing {name} full-graph fragments: {missing}")

    local_graph_sources = "\n".join(
        sources[name] for name in ("spatial", "sample", "local", "model")
    )
    forbidden = (
        "knn_graph(",
        "torch_cluster.knn",
        "_knn_edges(",
        "nodes_per_graph = node_total // graph_count",
        "hiercp.v21",
        "adaptive_balanced_v21",
    )
    offenders = [fragment for fragment in forbidden if fragment in local_graph_sources]
    if offenders:
        raise RuntimeError(
            "Legacy/fixed/k-NN local graph path remains active: " + ", ".join(offenders)
        )
    if "radius_edges(" in sources["sample"] or "cross_radius_edges(" in sources["sample"]:
        raise RuntimeError("Sampled views rebuild topology instead of inducing cached full edges")
    if any(
        fragment in sources["schema"]
        for fragment in ("sample_surface_nodes", "sample_interior_nodes", "sample_liver_nodes")
    ):
        raise RuntimeError("Anchor node sampling remains enabled; only context may be sampled")

    grid_samples = sum(
        sources[name].count("F.grid_sample(") for name in ("sample", "model")
    )
    if grid_samples != 1:
        raise RuntimeError(f"Expected one variable-node grid_sample site, found {grid_samples}")
    if (root / "hiercp" / "v21").exists() or (root / "tools" / "v21").exists():
        raise RuntimeError("Obsolete V21 package directory still exists")
    if (root / "config" / "v21.json").exists():
        raise RuntimeError("Obsolete config/v21.json still exists")

    for protected in ('"Data"', '"Data_aug"', '"Task03_Liver"'):
        if protected not in sources["run"]:
            raise RuntimeError(f"Workspace protection is missing {protected}")

    active_python = [
        path
        for folder in (root / "hiercp", root / "tools")
        for path in folder.rglob("*.py")
    ] + [root / "run.py"]
    for path in sorted(active_python):
        _source(path)

    print("[OK] canonical full-graph project audit")
    print("schema:", graph.graph_schema_version)
    print("local topology: KD-tree physical radius")
    print("training views: exact induced HeteroData x2; all anchors preserved")
    print("message passing: relation-specific GATv2")
    print("variable-node dense sampling: valid")
    print("active python files:", len(active_python))


if __name__ == "__main__":
    main()
