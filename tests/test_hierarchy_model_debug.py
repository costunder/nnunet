"""DEBUG only: artificial graph/tensor fixtures, never medical or final results.

All configured message-passing depths (3/2/2) are retained. Hidden/CNN widths
and spatial size below are explicitly test-only; production config is untouched.
Fixtures cover every relation and both populated/missing context shells so a
missing gradient is not excused by an accidentally unsupported debug graph.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace
import unittest

AVAILABLE = all(importlib.util.find_spec(name) is not None
                for name in ("torch", "torch_geometric", "scipy"))
if AVAILABLE:
    import torch
    from torch_geometric.data import Batch, HeteroData
    from hiercp.loss import CurriculumConfig, curriculum_ranking_loss
    from hiercp.model import (
        ABLATION_MODES, MODEL_ARCHITECTURE_VERSION, CandidateContextReadout,
        CompatibilityGatedGATv2Conv, HierarchicalPyGPlacementModel,
    )
    from hiercp.schema import (
        CONTEXT_SHELL_FEATURE_INDEX, LOCAL_NODE_TYPES, LOCAL_EDGE_TYPES,
        LOCAL_HANDCRAFTED_DIM, LOCAL_EDGE_DIM, PATIENT_NODE_TYPES,
        PATIENT_EDGE_TYPES, PATIENT_EDGE_DIM, PROTOTYPE_NODE_TYPES,
        PROTOTYPE_EDGE_TYPES, PROTOTYPE_EDGE_DIM, UPPER_RAW_DIM,
        REGION_FEATURE_DIM, PROTOTYPE_FEATURE_DIM,
    )


def debug_model(mode="full"):
    return HierarchicalPyGPlacementModel(
        hidden_dim=16, heads=4, local_layers=3, patient_layers=2, prototype_layers=2,
        dropout=0., dense_base_channels=4, dense_feature_dim=8, dense_batch_size=8,
        channels_last_3d=False, checkpoint_local_blocks=False,
        checkpoint_dense_encoder=False, ablation_mode=mode,
    )


def debug_relations(graph, edge_types, edge_dim, generator):
    for edge_type in edge_types:
        source, _, destination = edge_type
        ns, nd = graph[source].num_nodes, graph[destination].num_nodes
        if ns and nd:
            edge_index = torch.cartesian_prod(torch.arange(ns), torch.arange(nd)).T.contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        if edge_type[1] in ("belongs_to", "contains_candidate"):
            count = graph["candidate"].num_nodes
            membership = torch.stack([torch.arange(count), torch.arange(count) % graph["region"].num_nodes])
            edge_index = membership if source == "candidate" else membership.flip(0)
        graph[edge_type].edge_index = edge_index
        graph[edge_type].edge_attr = torch.randn(edge_index.shape[1], edge_dim, generator=generator)


def debug_batch(counts=(4, 3), *, empty_lesions=False, seed=914, patch_size=8):
    generator = torch.Generator().manual_seed(seed)
    patients, populations, locals_first, locals_second = [], [], [], []
    for count in counts:
        patient = HeteroData()
        sizes = {"tumor": 1, "candidate": count, "region": 3,
                 "lesion": 0 if empty_lesions else 2, "liver": 1}
        for node_type in PATIENT_NODE_TYPES:
            width = REGION_FEATURE_DIM if node_type == "region" else UPPER_RAW_DIM
            patient[node_type].raw_x = torch.randn(sizes[node_type], width, generator=generator)
            patient[node_type].num_nodes = sizes[node_type]
        debug_relations(patient, PATIENT_EDGE_TYPES, PATIENT_EDGE_DIM, generator)
        population = HeteroData()
        for node_type in PROTOTYPE_NODE_TYPES:
            if node_type == "prototype":
                population[node_type].raw_x = torch.randn(3, PROTOTYPE_FEATURE_DIM, generator=generator)
                population[node_type].num_nodes = 3
            else:
                population[node_type].raw_x = patient[node_type].raw_x.clone()
                population[node_type].num_nodes = sizes[node_type]
        debug_relations(population, PROTOTYPE_EDGE_TYPES, PROTOTYPE_EDGE_DIM, generator)
        patients.append(patient)
        populations.append(population)
    for candidate_index in range(sum(counts)):
        graph = HeteroData()
        for node_type in LOCAL_NODE_TYPES:
            context = node_type in ("source_context", "target_context")
            count = 6 if context else 3
            graph[node_type].x = torch.randn(count, LOCAL_HANDCRAFTED_DIM, generator=generator)
            graph[node_type].grid = torch.rand(count, 3, generator=generator) * 1.6 - .8
            graph[node_type].num_nodes = count
            if context:
                # Every graph lacks one real shell. Across the physical batch,
                # every shell also has populated graphs with multiple nodes.
                graph[node_type].x[:, CONTEXT_SHELL_FEATURE_INDEX] = (
                    (torch.arange(count) % 2 + candidate_index) % 3
                ).float() / 3.
        debug_relations(graph, LOCAL_EDGE_TYPES, LOCAL_EDGE_DIM, generator)
        locals_first.append(graph)
        second = copy.deepcopy(graph)
        for node_type in LOCAL_NODE_TYPES:
            second[node_type].x[:, 0] += .1 * (candidate_index + 1)
        locals_second.append(second)
    return SimpleNamespace(
        counts=tuple(counts), sample_count=len(counts),
        source_patches=torch.randn(len(counts), 5, patch_size, patch_size, patch_size, generator=generator),
        target_patches=torch.randn(sum(counts), 5, patch_size, patch_size, patch_size, generator=generator),
        local_batch=Batch.from_data_list(locals_first),
        local_batch_view2=Batch.from_data_list(locals_second),
        patient_batch=Batch.from_data_list(patients),
        prototype_batch=Batch.from_data_list(populations),
        difficulties=tuple(torch.tensor([0] + [1 + i % 3 for i in range(n - 1)]) for n in counts),
    )


@unittest.skipUnless(AVAILABLE, "DEBUG model tests require PyTorch/PyG/SciPy")
class ConditionedHierarchyDebugTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)  # Explicit test-only CPU execution profile.

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_debug_all_active_parameter_gradients_and_block_updates_all_modes(self):
        for mode in ABLATION_MODES:
            with self.subTest(mode=mode):
                torch.manual_seed(181)
                model, batch = debug_model(mode), debug_batch()
                optimizer = torch.optim.SGD(model.trainable_parameters(), lr=.02)
                optimizer_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
                self.assertEqual(optimizer_ids, {id(p) for p in model.parameters() if p.requires_grad})
                before = {name: p.detach().clone() for name, p in model.named_parameters() if p.requires_grad}
                output = model(batch)
                ranking, _ = curriculum_ranking_loss(output.scores, batch.difficulties, epoch=30, config=CurriculumConfig())
                (ranking + .1 * output.consistency).backward()
                missing = [name for name, p in model.named_parameters() if p.requires_grad and p.grad is None]
                self.assertEqual(missing, [], f"Disconnected active parameters: {missing}")
                for name, parameter in model.named_parameters():
                    if parameter.requires_grad:
                        self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                    else:
                        self.assertIsNone(parameter.grad, name)
                optimizer.step()
                active_prefixes = ["score_head"]
                for level, removed, layers in (("local", "no_local", 3), ("patient", "no_patient", 2), ("prototype", "no_population", 2)):
                    if mode != removed:
                        active_prefixes.extend(f"{level}_encoder.blocks.{i}." for i in range(layers))
                if mode != "no_patient":
                    active_prefixes.append("patient_readout.")
                if mode != "no_population":
                    active_prefixes.append("population_readout.")
                for prefix in active_prefixes:
                    changed = [not torch.equal(before[name], p.detach()) for name, p in model.named_parameters()
                               if p.requires_grad and name.startswith(prefix)]
                    self.assertTrue(changed and any(changed), f"No optimizer update in {mode}:{prefix}")

    def test_debug_chunked_and_physical_batch_equivalence_all_modes(self):
        for mode in ABLATION_MODES:
            with self.subTest(mode=mode), torch.no_grad():
                torch.manual_seed(222)
                model, batch = debug_model(mode).eval(), debug_batch()
                expected = model(copy.deepcopy(batch)).scores
                actual = model.score_inference_chunked(copy.deepcopy(batch), local_chunk_size=2)
                for left, right in zip(expected, actual):
                    torch.testing.assert_close(left, right, rtol=3e-4, atol=3e-5)
                # Evaluate the exact same cases separately; no nodes from another
                # patient may enter either GNN or candidate-conditioned attention.
                offset = 0
                for case_id, count in enumerate(batch.counts):
                    one = copy.deepcopy(batch)
                    one.counts, one.sample_count = (count,), 1
                    one.source_patches = batch.source_patches[case_id:case_id + 1]
                    one.target_patches = batch.target_patches[offset:offset + count]
                    one.patient_batch = Batch.from_data_list([batch.patient_batch.to_data_list()[case_id]])
                    one.prototype_batch = Batch.from_data_list([batch.prototype_batch.to_data_list()[case_id]])
                    one.local_batch = Batch.from_data_list(batch.local_batch.to_data_list()[offset:offset + count])
                    one.local_batch_view2 = Batch.from_data_list(batch.local_batch_view2.to_data_list()[offset:offset + count])
                    torch.testing.assert_close(model(one).scores[0], expected[case_id], rtol=3e-4, atol=3e-5)
                    offset += count

    def test_debug_empty_lesions_remain_empty_and_scores_finite(self):
        batch = debug_batch(empty_lesions=True)
        self.assertEqual(batch.patient_batch["lesion"].num_nodes, 0)
        model = debug_model()
        output = model(batch)
        self.assertTrue(all(torch.isfinite(score).all() for score in output.scores))
        ranking, _ = curriculum_ranking_loss(output.scores, batch.difficulties, epoch=30, config=CurriculumConfig())
        ranking.backward()
        self.assertEqual(batch.patient_batch["lesion"].num_nodes, 0)
        # Lesion-only parameters may be conditionally unused here; no fake node
        # or artificial auxiliary objective is used to force their gradients.

    def test_debug_final_node_states_change_candidate_relative_scores(self):
        torch.manual_seed(301)
        model, batch = debug_model().eval(), debug_batch()
        with torch.no_grad():
            baseline = model(copy.deepcopy(batch)).scores[0]
        for level, node_type in (("patient", "lesion"), ("patient", "liver"),
                                 ("patient", "region"), ("prototype", "prototype")):
            def perturb(_module, _args, output, key=node_type):
                modified = dict(output)
                modified[key] = output[key] + torch.linspace(-1., 1., 16)[None]
                return modified
            handle = getattr(model, f"{level}_encoder").blocks[-1].register_forward_hook(perturb)
            try:
                with torch.no_grad():
                    changed = model(copy.deepcopy(batch)).scores[0]
            finally:
                handle.remove()
            self.assertFalse(torch.allclose(changed - changed[0], baseline - baseline[0], rtol=1e-5, atol=1e-7),
                             f"Final {level}/{node_type} changed no relative candidate score")

    def test_debug_single_neighbor_compatibility_is_trainable_and_finite(self):
        torch.manual_seed(401)
        conv = CompatibilityGatedGATv2Conv((4, 4), 2, heads=2, edge_dim=3, add_self_loops=False, dropout=0.)
        source = torch.randn(1, 4, requires_grad=True)
        target = torch.randn(2, 4, requires_grad=True)
        edge = torch.tensor([[0, 0], [0, 1]])
        attr = torch.randn(2, 3, requires_grad=True)
        output, (_, weights) = conv((source, target), edge, attr, return_attention_weights=True)
        self.assertTrue(torch.isfinite(weights).all())
        self.assertTrue(((weights > 0) & (weights < 1)).all())
        output.square().sum().backward()
        for value in (conv.att.grad, conv.lin_r.weight.grad, conv.lin_edge.weight.grad, target.grad, attr.grad):
            self.assertTrue(torch.isfinite(value).all())
            self.assertGreater(torch.count_nonzero(value).item(), 0)
        # Saturated gates may legitimately have zero derivatives, but must not
        # produce infinities or NaNs. No clipping or replacement value is used.
        for extreme in (-1e4, 1e4):
            with torch.no_grad():
                conv.att.fill_(extreme)
                value, (_, attention) = conv((source, target), edge, attr, return_attention_weights=True)
                self.assertTrue(torch.isfinite(value).all())
                self.assertTrue(torch.isfinite(attention).all())
                self.assertTrue(((attention >= 0) & (attention <= 1)).all())

    def test_debug_population_retrieval_is_query_conditioned_and_patient_isolated(self):
        torch.manual_seed(511)
        readout = CandidateContextReadout(12, 8)
        query = torch.randn(4, 12, requires_grad=True)
        context = torch.randn(6, 8, requires_grad=True)
        owners = torch.tensor([0, 0, 1, 1])
        context_owners = torch.tensor([0, 0, 0, 1, 1, 1])
        attended, compatibility = readout(query, context, owners, context_owners)
        changed = context.detach().clone()
        changed[3:] += 100.
        isolated, _ = readout(query, changed, owners, context_owners)
        torch.testing.assert_close(attended[:2], isolated[:2])
        (attended.square().sum() + compatibility.square().sum()).backward()
        self.assertGreater(torch.count_nonzero(query.grad).item(), 0)
        self.assertGreater(torch.count_nonzero(context.grad).item(), 0)

    def test_debug_architecture_guard_rejects_legacy_even_non_strict(self):
        model = debug_model()
        self.assertEqual(model.architecture_version, MODEL_ARCHITECTURE_VERSION)
        legacy = dict(model.state_dict())
        legacy.pop("_architecture_revision")
        with self.assertRaisesRegex(RuntimeError, "legacy weights"):
            model.load_state_dict(legacy, strict=False)
        model.load_state_dict(model.state_dict())

    @unittest.skipUnless(os.environ.get("HIERCP_DEBUG_PRODUCTION_SMOKE") == "1",
                         "Explicit opt-in production-sized DEBUG smoke; not final training")
    def test_debug_production_sized_batch_two_one_optimizer_step(self):
        import psutil

        config = json.loads((Path(__file__).resolve().parents[1] / "config" / "train.json").read_text(encoding="utf-8"))
        production = config["model"]
        self.assertEqual([production[key] for key in (
            "hidden_dim", "heads", "local_layers", "patient_layers", "prototype_layers",
            "dense_base_channels", "dense_feature_dim",
        )], [128, 4, 3, 2, 2, 12, 32])
        self.assertEqual(config["graph"]["patch_size"], 48)
        torch.manual_seed(901)
        process = psutil.Process()
        started = time.perf_counter()
        model = HierarchicalPyGPlacementModel(**production)
        batch = debug_batch(patch_size=48)
        trainable = list(model.trainable_parameters())
        training = config["training"]
        learning_rate = training.get("learning_rate", training.get("lr"))
        optimizer = torch.optim.AdamW(trainable, lr=float(learning_rate), weight_decay=float(training["weight_decay"]))
        total_parameters = sum(p.numel() for p in model.parameters())
        print(json.dumps({
            "scope": "production-sized-model-debug-smoke", "actual_medical_data": False,
            "full_training": False, "device": "cpu", "model": production,
            "parameters": total_parameters, "trainable_parameters": sum(p.numel() for p in trainable),
            "physical_batch_size": 2, "candidate_counts": list(batch.counts),
            "source_shape": list(batch.source_patches.shape), "target_shape": list(batch.target_patches.shape),
            "debug_graph_scope": "3 regions, 3 prototypes, 2 lesions, 3/6 local nodes per type; all relations",
            "cpu_threads": torch.get_num_threads(), "available_ram_bytes": psutil.virtual_memory().available,
        }), flush=True)
        before = {name: p.detach().clone() for name, p in model.named_parameters() if p.requires_grad}
        output = model(batch)
        curriculum_fields = set(CurriculumConfig.__dataclass_fields__)
        curriculum = CurriculumConfig(**{key: value for key, value in training.items() if key in curriculum_fields})
        ranking, _ = curriculum_ranking_loss(
            output.scores, batch.difficulties, epoch=curriculum.model_mine_start_epoch, config=curriculum,
        )
        loss = ranking + float(training["consistency_weight"]) * output.consistency
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        missing = [name for name, p in model.named_parameters() if p.requires_grad and p.grad is None]
        self.assertEqual(missing, [])
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in trainable))
        optimizer.step()
        for prefix in ("local_encoder.", "patient_encoder.", "prototype_encoder.",
                       "patient_readout.", "population_readout.", "score_head."):
            self.assertTrue(any(not torch.equal(before[name], p.detach())
                                for name, p in model.named_parameters() if name.startswith(prefix)))
        memory = process.memory_info()
        print(json.dumps({
            "scope": "production-sized-model-debug-smoke", "status": "passed",
            "optimizer_steps": 1, "elapsed_seconds": time.perf_counter() - started,
            "rss_bytes": memory.rss, "process_peak_working_set_bytes": getattr(memory, "peak_wset", None),
            "peak_vram_bytes": None, "gpu_used": False,
            "parameters": total_parameters, "trainable_parameters": sum(p.numel() for p in trainable),
        }), flush=True)


if __name__ == "__main__":
    unittest.main()
