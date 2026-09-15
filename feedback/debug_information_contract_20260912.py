"""Read-only DEBUG information-contract probe; no clinical data or training.

Real in-memory synthetic cases feed actual region/prototype/curriculum/upper
graph builders and native PyG L1/L2 plus the actual conditioned score/readout
and ranking loss. L0 is an EXPLICIT BOUNDARY: fixed synthetic embeddings, not
a claim to run CNN/L0 or validate medical geometry. Production hidden width,
heads, 3/2/2 module depths, 24 regions, 16 prototypes, eight ranking candidates
are retained. No checkpoint is read/written; no optimizer step is performed.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch_geometric.data import Batch
import torch_geometric

from hiercp.common import (CasePaths, LoadedCase, choose_source_tumor,
                           build_candidate_pool)
from hiercp.curriculum import build_training_specs
from hiercp.hierarchy import build_patient_graph, build_prototype_graph
from hiercp.loss import CurriculumConfig, curriculum_ranking_loss
from hiercp.model import HierarchicalPyGPlacementModel, _mask_tumor_spatial_edge_attr
from hiercp.prototype import build_prototype_bank
from hiercp.region import build_patient_regions
from hiercp.schema import (GraphBuildConfig, UPPER_FORBIDDEN_RAW_COLUMNS,
                           PATIENT_POSITION_EDGE_COLUMNS)


def synthetic_case(index):
    shape = (40, 40, 40)
    z, y, x = np.indices(shape)
    liver = (z-20)**2/17.**2 + (y-20)**2/16.**2 + (x-20)**2/15.**2 <= 1
    source_mask = (z-20)**2 + (y-19)**2 + (x-17)**2 <= 3.5**2
    other_mask = (z-25)**2 + (y-25)**2 + (x-25)**2 <= 1.8**2
    label = np.zeros(shape, np.int16)
    label[liver], label[source_mask | other_mask] = 1, 2
    image = np.random.default_rng(10+index).normal(70., 11., shape).astype(np.float32)
    image[source_mask | other_mask] += 24.
    case = LoadedCase(CasePaths(f"debug_case_{index}", Path("UNUSED_IMAGE"), Path("UNUSED_LABEL")),
                      image, label, np.eye(4), np.eye(4), None, None,
                      np.array([1.2, 1.1, 1.], np.float32), {}, {})
    source, _, _ = choose_source_tumor(image, label, tumor_label=2,
                                      rng=np.random.default_rng(30+index), selection="largest", pad=3)
    return case, source


def run():
    torch.set_num_threads(2)  # Explicit DEBUG CPU profile; not a production recommendation.
    torch.manual_seed(6712)
    config = GraphBuildConfig()
    cases = [synthetic_case(i) for i in range(2)]
    region_sets = [build_patient_regions(case, liver_label=1, tumor_label=2, config=config,
                      rng=np.random.default_rng(40+i), ct_clip=(-200., 250.))
                   for i, (case, source) in enumerate(cases)]
    bank = build_prototype_bank([(case.paths.case_id, regions.region_features)
               for (case, source), regions in zip(cases, region_sets)], config=config,
               rng=np.random.default_rng(55))
    patients, populations, difficulties = [], [], []
    for i, ((case, source), regions) in enumerate(zip(cases, region_sets)):
        candidates, _ = build_candidate_pool(case, source, placement_mask=case.label == 1,
             full_organ_mask=regions.full_organ_mask, occupied_mask=case.label == 2,
             organ_distance=regions.organ_depth, rng=np.random.default_rng(60+i),
             num_candidates=7, max_draws=80_000, min_liver_coverage=.8,
             occupied_clearance_vox=1, min_center_separation_mm=7.)
        specs = build_training_specs(case, source, candidates, regions, bank,
             total_candidates=8, easy_fraction=.34, inter_fraction=.33, intra_fraction=.33,
             tumor_label=2, config=config, rng=np.random.default_rng(70+i))
        graph = build_patient_graph(case, source, specs, regions, tumor_label=2,
                                   config=config, ct_clip=(-200., 250.))
        patients.append(graph)
        populations.append(build_prototype_graph(specs, graph, regions, bank, config=config))
        difficulties.append(torch.tensor([s.difficulty for s in specs]))
    batch = SimpleNamespace(counts=(8, 8), patient_batch=Batch.from_data_list(patients),
                            prototype_batch=Batch.from_data_list(populations))
    model = HierarchicalPyGPlacementModel(hidden_dim=128, heads=4, local_layers=3,
              patient_layers=2, prototype_layers=2, dropout=.1, dense_base_channels=12,
              dense_feature_dim=32, dense_batch_size=4).eval()
    local = {key: torch.randn(16, 128) for key in (
        "tumor", "source_context", "source_relation", "fused", "target_context", "target_relation")}
    for key in ("tumor", "source_context", "source_relation"):
        local[key] = torch.randn(2, 128).repeat_interleave(8, dim=0)
    baseline = model._score_full(batch, local)
    changed = copy.deepcopy(batch)
    host = ("tumor", "hosted_by", "region")
    reverse = ("region", "hosts_tumor", "tumor")
    old_host = int(changed.patient_batch[host].edge_index[1, 0])
    new_host = (old_host + 1) % config.num_regions
    changed.patient_batch[host].edge_index[1, 0] = new_host
    changed.patient_batch[reverse].edge_index[0, 0] = new_host
    rewired = model._score_full(changed, local)
    # All feature and edge-attribute tensors remain exactly fixed: surgical
    # topology counterfactual, not a claim that these are physical patient pairs.
    for key in batch.patient_batch.node_types:
        assert torch.equal(batch.patient_batch[key].raw_x, changed.patient_batch[key].raw_x)
    for key in batch.patient_batch.edge_types:
        assert torch.equal(batch.patient_batch[key].edge_attr, changed.patient_batch[key].edge_attr)
    numeric = copy.deepcopy(batch)
    for graph, node in ((numeric.patient_batch, "tumor"), (numeric.patient_batch, "candidate"),
                        (numeric.prototype_batch, "candidate")):
        graph[node].raw_x[:, list(UPPER_FORBIDDEN_RAW_COLUMNS)] = 999.
        graph[node].pos.fill_(999.)
    for key in numeric.patient_batch.edge_types:
        if "tumor" in (key[0], key[2]):
            numeric.patient_batch[key].edge_attr[:, list(PATIENT_POSITION_EDGE_COLUMNS)] = 999.
    numeric_scores = model._score_full(numeric, local)
    px = model.patient_encoder.forward_raw(batch.patient_batch, local, batch.counts)
    px_changed = model.patient_encoder.forward_raw(changed.patient_batch, local, batch.counts)
    pz = model.prototype_encoder.forward_raw(batch.prototype_batch, px)
    pz_changed = model.prototype_encoder.forward_raw(changed.prototype_batch, px_changed)
    loss, parts = curriculum_ranking_loss(baseline, difficulties, epoch=30, config=CurriculumConfig())
    loss.backward()
    raw_targets = {
        "patient_encoder.tumor_project.0.weight": 3*128,
        "patient_encoder.candidate_project.0.weight": 3*128,
        "prototype_encoder.candidate_bridge.0.weight": 128,
        "score_head.0.weight": 12*128,
    }
    named = dict(model.named_parameters())
    raw_grad = {}
    for name, offset in raw_targets.items():
        p = named[name]
        cols = [offset+c for c in UPPER_FORBIDDEN_RAW_COLUMNS]
        raw_grad[name] = {"columns": cols, "zero_cells": p.shape[0]*len(cols),
            "masked_gradient_max_abs": float(p.grad[:, cols].abs().max()),
            "whole_tensor_gradient_max_abs": float(p.grad.abs().max())}
    edge_grad = {}
    for name, p in named.items():
        if (name.startswith("patient_encoder.blocks.") and "tumor" in name
                and name.endswith("lin_edge.weight") and p.ndim == 2 and p.shape[1] == 12):
            edge_grad[name] = {"columns": list(PATIENT_POSITION_EDGE_COLUMNS),
                "zero_cells": p.shape[0]*len(PATIENT_POSITION_EDGE_COLUMNS),
                "masked_gradient_max_abs": None if p.grad is None else
                    float(p.grad[:, list(PATIENT_POSITION_EDGE_COLUMNS)].abs().max()),
                "whole_tensor_gradient_max_abs": None if p.grad is None else float(p.grad.abs().max())}
    # Exact affine equivalence of removing only masked raw columns (all four
    # production-width consumers), separate from any topology redesign.
    migration = {}
    for name, offset in raw_targets.items():
        p = named[name]
        linear = model.get_submodule(name.rsplit(".", 1)[0])
        x = torch.randn(16, p.shape[1], dtype=p.dtype)
        forbidden = [offset+c for c in UPPER_FORBIDDEN_RAW_COLUMNS]
        x[:, forbidden] = 0.
        keep = [j for j in range(p.shape[1]) if j not in forbidden]
        before = torch.nn.functional.linear(x, p, linear.bias)
        after = torch.nn.functional.linear(x[:, keep], p[:, keep], linear.bias)
        migration[name] = float((before-after).abs().max())
    data = {
       "kind": "synthetic_DEBUG_native_upper_hierarchy_with_fixed_L0_embedding_boundary",
       "torch": torch.__version__, "pyg": torch_geometric.__version__,
       "cpu_count": os.cpu_count(), "torch_threads": torch.get_num_threads(),
       "cuda_available": torch.cuda.is_available(), "physical_synthetic_case_batch": 2,
       "hidden": 128, "heads": 4, "module_blocks": [3,2,2],
       "region_counts": [g["region"].num_nodes for g in patients],
       "prototype_counts": [g["prototype"].num_nodes for g in populations],
       "candidate_counts": [8,8], "patient_nodes": [g.num_nodes for g in patients],
       "patient_edges": [g.num_edges for g in patients], "old_host": old_host, "new_host": new_host,
       "same_region_flags": [bool(v) for v in (patients[0]["candidate"].region_index == old_host)],
       "numeric_mask_score_max_abs_delta": max(float((a-b).abs().max()) for a,b in zip(baseline,numeric_scores)),
       "topology_score_max_abs_delta": float((baseline[0]-rewired[0]).abs().max()),
       "topology_relative_score_max_abs_delta": float(((baseline[0]-baseline[0][0])-(rewired[0]-rewired[0][0])).abs().max()),
       "other_patient_score_max_abs_delta": float((baseline[1]-rewired[1]).abs().max()),
       "l1_candidate_state_max_abs_delta": float((px["candidate"]-px_changed["candidate"]).abs().max()),
       "l2_candidate_state_max_abs_delta": float((pz["candidate"]-pz_changed["candidate"]).abs().max()),
       "l2_prototype_state_max_abs_delta": float((pz["prototype"]-pz_changed["prototype"]).abs().max()),
       "ranking_loss": float(loss.detach()), "raw_gradient": raw_grad, "edge_gradient": edge_grad,
       "raw_affine_migration_float32_max_abs_delta": migration,
       "trained_checkpoint_used": False, "optimizer_step": False,
    }
    print(json.dumps(data, indent=2, allow_nan=False))


if __name__ == "__main__":
    run()
