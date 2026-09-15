"""Synthetic CPU DEBUG: source-address exclusion and complete upper context.

Native case/region/prototype/curriculum/graph builders and native PyG upper
forward/loss/backward run at H128/4 heads/24 regions/16 prototypes, physical B2.
Their L0 embeddings are an explicit fixed boundary, not CNN/medical validation.
Separate complete-model tests use the established small DEBUG CNN/local graph
fixture with all 3/2/2 blocks and all128 candidates; production is unchanged.
"""
from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch
from torch_geometric.data import Batch

from hiercp.common import CasePaths, LoadedCase, choose_source_tumor, build_candidate_pool
from hiercp.contracts import (
    ARCHITECTURE_VERSION, GEOMETRY_CONTRACT, PATIENT_GRAPH_CONTRACT,
    require_current_checkpoint,
)
from hiercp.curriculum import build_training_specs
from hiercp.data import HierarchicalCacheDataset
from hiercp.cache import CACHE_FORMAT
from hiercp.hierarchy import build_patient_graph, build_prototype_graph
from hiercp.loss import CurriculumConfig, curriculum_ranking_loss
from hiercp.model import HierarchicalPyGPlacementModel, _mask_upper_shortcuts, _mask_tumor_spatial_edge_attr
from hiercp.prototype import build_prototype_bank
from hiercp.region import build_patient_regions
from hiercp.schema import (
    GraphBuildConfig, PATIENT_EDGE_TYPES, PATIENT_POSITION_EDGE_COLUMNS,
    UPPER_CONTENT_COLUMNS, UPPER_FORBIDDEN_RAW_COLUMNS, LESION_CONTENT_COLUMNS,
)
from tools.causality import (
    _condition_source_address_noise, _condition_upper_node_order,
    _permute_node_type,
)
from tests.test_hierarchy_model_debug import debug_batch, debug_model


def native_upper_fixture():
    config = GraphBuildConfig()
    cases, regions = [], []
    z, y, x = np.indices((40, 40, 40))
    organ = (z-20)**2/17.**2 + (y-20)**2/16.**2 + (x-20)**2/15.**2 <= 1
    tumor = (z-20)**2 + (y-19)**2 + (x-17)**2 <= 3.5**2
    other = (z-25)**2 + (y-25)**2 + (x-25)**2 <= 1.8**2
    for index in range(2):
        label = np.zeros(organ.shape, np.int16)
        label[organ], label[tumor | other] = 1, 2
        image = np.random.default_rng(10+index).normal(70., 11., organ.shape).astype(np.float32)
        image[tumor | other] += 24.
        case = LoadedCase(CasePaths(f"debug_{index}", Path("UNUSED_IMAGE"), Path("UNUSED_LABEL")),
                          image, label, np.eye(4), np.eye(4), None, None,
                          np.asarray([1.2, 1.1, 1.], np.float32), {}, {})
        source, _, _ = choose_source_tumor(image, label, tumor_label=2,
            rng=np.random.default_rng(30+index), selection="largest", pad=3)
        cases.append((case, source))
        regions.append(build_patient_regions(case, liver_label=1, tumor_label=2,
            config=config, rng=np.random.default_rng(40+index), ct_clip=(-200., 250.)))
    bank = build_prototype_bank([(case.paths.case_id, region.region_features)
          for (case, _), region in zip(cases, regions)], config=config, rng=np.random.default_rng(55))
    patients, populations, difficulties = [], [], []
    for index, ((case, source), region) in enumerate(zip(cases, regions)):
        candidates, _ = build_candidate_pool(case, source, placement_mask=case.label == 1,
            full_organ_mask=region.full_organ_mask, occupied_mask=case.label == 2,
            organ_distance=region.organ_depth, rng=np.random.default_rng(60+index),
            num_candidates=7, max_draws=80_000, min_liver_coverage=.8,
            occupied_clearance_vox=1, min_center_separation_mm=7.)
        specs = build_training_specs(case, source, candidates, region, bank, total_candidates=8,
            easy_fraction=.34, inter_fraction=.33, intra_fraction=.33, tumor_label=2,
            config=config, rng=np.random.default_rng(70+index))
        patient = build_patient_graph(case, source, specs, region, tumor_label=2,
            config=config, ct_clip=(-200., 250.))
        patients.append(patient)
        populations.append(build_prototype_graph(specs, patient, region, bank, config=config))
        difficulties.append(torch.tensor([spec.difficulty for spec in specs]))
    batch = SimpleNamespace(counts=(8, 8), patient_batch=Batch.from_data_list(patients),
        prototype_batch=Batch.from_data_list(populations), difficulties=difficulties)
    generator = torch.Generator().manual_seed(6712)
    local = {key: torch.randn(16, 128, generator=generator) for key in (
        "tumor", "source_context", "source_relation", "fused", "target_context", "target_relation")}
    return batch, local


class SourceContentContractDebugTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        cls.batch, cls.local = native_upper_fixture()
        torch.manual_seed(6712)
        cls.model = HierarchicalPyGPlacementModel(hidden_dim=128, heads=4,
            local_layers=3, patient_layers=2, prototype_layers=2, dropout=.1,
            dense_base_channels=12, dense_feature_dim=32, dense_batch_size=4).eval()

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_native_builder_conditions_every_region_and_preserves_context(self):
        for graph, population in zip(self.batch.patient_batch.to_data_list(),
                                     self.batch.prototype_batch.to_data_list()):
            self.assertEqual(graph.patient_graph_contract, PATIENT_GRAPH_CONTRACT)
            self.assertEqual(graph["region"].num_nodes, 24)
            self.assertEqual(graph["candidate"].num_nodes, 8)
            self.assertEqual(graph["lesion"].num_nodes, 1)
            self.assertEqual(population["prototype"].num_nodes, 16)
            edge = graph[("tumor", "conditions", "region")].edge_index
            torch.testing.assert_close(edge, torch.stack((torch.zeros(24, dtype=torch.long), torch.arange(24))))
            torch.testing.assert_close(graph[("region", "context_for", "tumor")].edge_index, edge.flip(0))
            self.assertNotIn(("tumor", "hosted_by", "region"), graph.edge_types)
            self.assertEqual(graph["tumor"].region_index.tolist(), [-1])

    def test_native_address_and_forbidden_field_interventions_are_exact_noops(self):
        changed = _condition_source_address_noise(copy.deepcopy(self.batch), 193)
        for graph, node in ((changed.patient_batch, "tumor"), (changed.patient_batch, "candidate"),
                            (changed.prototype_batch, "candidate")):
            graph[node].raw_x[:, list(UPPER_FORBIDDEN_RAW_COLUMNS)] = 731.
        for edge in changed.patient_batch.edge_types:
            if "tumor" in (edge[0], edge[2]):
                changed.patient_batch[edge].edge_attr[:, list(PATIENT_POSITION_EDGE_COLUMNS)] = -191.
        with torch.no_grad():
            expected = self.model._score_full(self.batch, self.local)
            actual = self.model._score_full(changed, self.local)
        for first, second in zip(expected, actual):
            torch.testing.assert_close(first, second, rtol=0., atol=0.)

    def test_native_permuted_region_prototype_identifiers_preserve_scores(self):
        changed = _condition_upper_node_order(copy.deepcopy(self.batch), 791)
        with torch.no_grad():
            expected = self.model._score_full(self.batch, self.local)
            actual = self.model._score_full(changed, self.local)
        for first, second in zip(expected, actual):
            torch.testing.assert_close(first, second, rtol=3e-4, atol=3e-5)

    def test_native_recipient_context_changes_scores_without_cross_patient_influence(self):
        changed = copy.deepcopy(self.batch)
        for graph in (changed.patient_batch, changed.prototype_batch):
            graph["region"].raw_x[:24, 6] += torch.linspace(.1, .8, 24)
        with torch.no_grad():
            expected = self.model._score_full(self.batch, self.local)
            actual = self.model._score_full(changed, self.local)
        self.assertGreater(float(((expected[0]-expected[0][0])-(actual[0]-actual[0][0])).abs().max()), 1e-7)
        torch.testing.assert_close(expected[1], actual[1], rtol=0., atol=0.)

    def test_native_source_biology_is_still_active(self):
        changed = copy.deepcopy(self.batch)
        changed.patient_batch["tumor"].raw_x[0, 5] += .7
        with torch.no_grad():
            expected = self.model._score_full(self.batch, self.local)
            actual = self.model._score_full(changed, self.local)
        self.assertGreater(float((expected[0]-actual[0]).abs().max()), 1e-7)
        torch.testing.assert_close(expected[1], actual[1], rtol=0., atol=0.)

    def test_legacy_graph_marker_and_host_edges_are_rejected_before_scoring(self):
        for missing_marker in (True, False):
            changed = copy.deepcopy(self.batch)
            if missing_marker:
                del changed.patient_batch.patient_graph_contract
            else:
                changed.patient_batch[("tumor", "hosted_by", "region")].edge_index = torch.tensor([[0], [0]])
                changed.patient_batch[("tumor", "hosted_by", "region")].edge_attr = torch.zeros(1, 12)
            with self.assertRaisesRegex(ValueError, "legacy source-host"):
                self.model._score_full(changed, self.local)

    def test_legacy_checkpoint_and_missing_persisted_semantics_are_rejected(self):
        payload = {"architecture_version": ARCHITECTURE_VERSION, "geometry_contract": GEOMETRY_CONTRACT,
                   "graph_config": {"geometry_contract": GEOMETRY_CONTRACT}}
        with self.assertRaisesRegex(ValueError, "patient_graph_contract"):
            require_current_checkpoint(payload)
        payload["graph_config"]["patient_graph_contract"] = PATIENT_GRAPH_CONTRACT
        require_current_checkpoint(payload)
        legacy = dict(self.model.state_dict())
        legacy["_architecture_revision"] = torch.tensor(3)
        with self.assertRaisesRegex(RuntimeError, "(?i)legacy weights"):
            self.model.load_state_dict(legacy, strict=False)
        dataset = HierarchicalCacheDataset(["UNUSED_OLD_SAMPLE"])
        with mock.patch("hiercp.data.torch_load_compat", return_value={"format": CACHE_FORMAT, "graph_config": {}}), \
             mock.patch("hiercp.data.materialize_sample_views") as materialize:
            with self.assertRaisesRegex(ValueError, "patient_graph_contract"):
                dataset[0]
            materialize.assert_not_called()

    def test_packing_preserves_permitted_features_and_relation_specific_widths(self):
        raw = torch.arange(28., dtype=torch.float32).reshape(2, 14)
        torch.testing.assert_close(_mask_upper_shortcuts(raw), raw[:, UPPER_CONTENT_COLUMNS])
        self.assertEqual(self.model.patient_encoder.tumor_project[0].in_features, 3*128+10)
        self.assertEqual(self.model.patient_encoder.lesion_project[0].in_features, 12)
        self.assertEqual(tuple(LESION_CONTENT_COLUMNS), tuple(i for i in range(14) if i not in (4, 10)))
        self.assertEqual(self.model.score_head[0].in_features, 12*128+10)
        for block in self.model.patient_encoder.blocks:
            for edge in PATIENT_EDGE_TYPES:
                source_relation = "tumor" in (edge[0], edge[2])
                projection = block.conv.convs[edge].lin_edge
                self.assertEqual(projection.weight.shape[1], 7 if source_relation else 12)
                packed = _mask_tumor_spatial_edge_attr(edge, torch.randn(3, 12))
                self.assertEqual(packed.shape[1], projection.weight.shape[1])

    def test_native_upper_ranking_gradient_and_optimizer_update(self):
        self.model.zero_grad(set_to_none=True)
        before = {name: value.detach().clone() for name, value in self.model.named_parameters()
                  if not name.startswith("local_encoder.")}
        local = {key: value.detach().clone().requires_grad_() for key, value in self.local.items()}
        try:
            optimizer = torch.optim.SGD(self.model.trainable_parameters(), lr=.01)
            scores = self.model._score_full(self.batch, local)
            loss, _ = curriculum_ranking_loss(scores, self.batch.difficulties, epoch=30, config=CurriculumConfig())
            loss.backward()
            for name, value in self.model.named_parameters():
                if name in before:
                    self.assertIsNotNone(value.grad, name)
                    self.assertTrue(torch.isfinite(value.grad).all(), name)
            for block in self.model.patient_encoder.blocks:
                for relation in PATIENT_EDGE_TYPES:
                    if "tumor" in (relation[0], relation[2]):
                        gradient = block.conv.convs[relation].lin_edge.weight.grad
                        self.assertTrue((gradient.abs().sum(dim=0) > 0).all(), relation)
            for key in ("tumor", "source_context", "source_relation"):
                self.assertTrue((local[key].grad.abs().sum(dim=1) > 0).all(), key)
            optimizer.step()
            for prefix in ("patient_encoder.blocks.0.", "patient_encoder.blocks.1.",
                           "prototype_encoder.blocks.0.", "prototype_encoder.blocks.1.",
                           "patient_readout.", "population_readout.", "score_head."):
                self.assertTrue(any(not torch.equal(before[name], value.detach())
                    for name, value in self.model.named_parameters() if name.startswith(prefix)), prefix)
        finally:
            with torch.no_grad():
                for name, value in self.model.named_parameters():
                    if name in before:
                        value.copy_(before[name])
            self.model.zero_grad(set_to_none=True)

    def test_debug_packing_matches_zero_mask_affine_math_without_migrating_weights(self):
        # Algebra-only synthetic weights are not an old learned checkpoint.
        # The v4 graph meaning still requires fresh training, even though this
        # numerical input-packing step is equivalent to zero-column masking.
        for width, columns in ((14, UPPER_CONTENT_COLUMNS),
                               (12, tuple(i for i in range(12)
                                          if i not in PATIENT_POSITION_EDGE_COLUMNS))):
            raw = torch.randn(19, width, dtype=torch.float64)
            prefix = torch.randn(19, 7, dtype=torch.float64)
            weight = torch.randn(31, 7 + width, dtype=torch.float64)
            bias = torch.randn(31, dtype=torch.float64)
            masked = torch.zeros_like(raw)
            masked[:, columns] = raw[:, columns]
            retained = tuple(range(7)) + tuple(7 + column for column in columns)
            expected = torch.nn.functional.linear(torch.cat((prefix, masked), dim=1), weight, bias)
            actual = torch.nn.functional.linear(torch.cat((prefix, raw[:, columns]), dim=1),
                                                weight[:, retained], bias)
            torch.testing.assert_close(actual, expected, rtol=1e-13, atol=1e-13)

    def test_complete_debug_model_candidate_permutation_is_equivariant(self):
        batch = debug_batch()
        changed = copy.deepcopy(batch)
        patients, prototypes = batch.patient_batch.to_data_list(), batch.prototype_batch.to_data_list()
        permutations = [torch.arange(count-1, -1, -1) for count in batch.counts]
        indices, offset = [], 0
        for perm, count in zip(permutations, batch.counts):
            indices.append(perm + offset)
            offset += count
        order = torch.cat(indices)
        changed.patient_batch = Batch.from_data_list([_permute_node_type(g, "candidate", p)
                                 for g, p in zip(patients, permutations)])
        changed.prototype_batch = Batch.from_data_list([_permute_node_type(g, "candidate", p)
                                   for g, p in zip(prototypes, permutations)])
        changed.target_patches = batch.target_patches[order]
        for field in ("local_batch", "local_batch_view2"):
            graphs = getattr(batch, field).to_data_list()
            setattr(changed, field, Batch.from_data_list([graphs[int(i)] for i in order]))
        model = debug_model().eval()
        with torch.no_grad():
            expected, actual = model(batch).scores, model(changed).scores
        for first, second, perm in zip(expected, actual, permutations):
            torch.testing.assert_close(first[perm], second, rtol=3e-4, atol=3e-5)

    def test_complete_debug_model_preserves_128_candidates_and_backward(self):
        batch = debug_batch((128, 128), region_count=24, prototype_count=16)
        model = debug_model().eval()
        output = model(batch)
        self.assertEqual([score.numel() for score in output.scores], [128, 128])
        self.assertEqual([len(getattr(model, name).blocks) for name in (
            "local_encoder", "patient_encoder", "prototype_encoder")], [3, 2, 2])
        loss, _ = curriculum_ranking_loss(output.scores, batch.difficulties, epoch=30, config=CurriculumConfig())
        (loss + .1*output.consistency).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in model.trainable_parameters()))


if __name__ == "__main__":
    unittest.main()
