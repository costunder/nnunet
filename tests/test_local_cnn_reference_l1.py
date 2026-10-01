"""Short 128D reference-L1 UNIT checks; fixtures are not trained CT evidence.

The oracle compiles only MetaGNNLayer from the pinned, byte-verified official
audit source. It does not import the official project or substitute this
fixture for the 14,102-record training/evaluation cohort.
"""
import ast
import copy
import hashlib
import inspect
import json
import math
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax

from hiercp_v222.model import PromptGraphModel
from hiercp_v222.v1_execution import rng_state
from l0_local_cnn.model import LocalCNN
from l0_regions.training import hash_state
from tools.local_cnn_reference_l1 import (
    ReferenceRelationLayer, ReferencePromptGraphModel, clone_reference,
    joint_topology,
)


ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = ROOT / 'validation/prompt_graph_audit_20261001/official/models/metaGNN.py'
OFFICIAL_SHA = 'fd94b26236b0edddfa585fc69f2d2f03e5deafd29d2a2a1b2e30bbc4267d9074'


def official_class():
    """Independent official class AST, including official edge-wise bias."""
    source = OFFICIAL.read_bytes()
    if hashlib.sha256(source).hexdigest() != OFFICIAL_SHA:
        raise AssertionError('Pinned official MetaGNNLayer source changed')
    parsed = ast.parse(source.decode('utf-8'), filename=str(OFFICIAL))
    matches = [node for node in parsed.body
               if isinstance(node, ast.ClassDef) and node.name == 'MetaGNNLayer']
    if len(matches) != 1:
        raise AssertionError('Exactly one pinned official MetaGNNLayer required')
    namespace = dict(torch=torch, F=F, math=math, softmax=softmax,
                     MessagePassing=MessagePassing, __name__=__name__)
    module = ast.fix_missing_locations(ast.Module(body=matches, type_ignores=[]))
    exec(compile(module, str(OFFICIAL), 'exec'), namespace)
    return namespace['MetaGNNLayer']


def fixture(device='cpu', dtype=torch.float32):
    torch.manual_seed(1001)
    cfg = json.loads((ROOT / 'config/prompt_graph_v222_v1_l0.json').read_text())
    base = json.loads((ROOT / 'config/train.json').read_text())
    original = PromptGraphModel(cfg, base, {}, local_encoder=nn.Identity()).to(device, dtype=dtype).eval()
    query = torch.randn(128, 128, device=device, dtype=dtype)
    # Sixteen support subjects, sixteen observations each. No query target
    # is provided to graph generation or inference.
    support = (torch.randn(256, 128, device=device, dtype=dtype),
               torch.arange(16, device=device).repeat_interleave(16),
               torch.arange(256, device=device).remainder(2))
    return original, query, support


def independent_graph(support, owners, classes, query, label_seed):
    """Write the intended topology independently of candidate helpers."""
    n, p, nq = len(support), int(owners.max()) + 1, len(query)
    device = support.device
    data = torch.arange(n, device=device).repeat_interleave(2)
    cls = torch.arange(2, device=device).repeat(n)
    label = n + 2 * owners.repeat_interleave(2) + cls
    known = torch.stack((torch.zeros_like(cls),
                         torch.where(classes.repeat_interleave(2) == cls, 1, -1)), -1).to(support.dtype)
    labels = torch.arange(n, n + 2*p, device=device)
    query_ids = torch.arange(n + 2*p, n + 2*p + nq, device=device)
    query_sources = labels.repeat(nq)
    query_destinations = query_ids.repeat_interleave(2*p)
    loops = torch.arange(n + 2*p + nq, device=device)
    edges = torch.stack((torch.cat((data, label, query_sources, loops)),
                         torch.cat((label, data, query_destinations, loops))))
    unknown = support.new_zeros(len(query_sources), 2)
    unknown[:, 0] = 1
    attributes = torch.cat((known, known, unknown, support.new_zeros(len(loops), 2)))
    return torch.cat((support, label_seed.repeat(p, 1), query)), edges, attributes


def edge_rows(edges, attributes):
    return sorted(tuple(row) for row in torch.cat((edges.T.to(attributes.dtype), attributes), -1).cpu().tolist())


class ReferenceL1Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        cls.Oracle = official_class()

    def test_official_source_hash_and_no_foreign_project_imports(self):
        self.assertEqual(hashlib.sha256(OFFICIAL.read_bytes()).hexdigest(), OFFICIAL_SHA)
        self.assertIs(self.Oracle.__mro__[1], MessagePassing)
        self.assertEqual(self.Oracle.__name__, 'MetaGNNLayer')

    def _layer_parity(self, device, training):
        original, query, support = fixture(device)
        graph = independent_graph(*support, query[:32], original.label_seed)
        candidate = ReferenceRelationLayer(128, 4, .1).to(device)
        oracle = self.Oracle(2, 128, heads=4, dropout=.1, batch_norm=True).to(device)
        self.assertEqual(set(candidate.state_dict()), set(oracle.state_dict()))
        oracle.load_state_dict(candidate.state_dict(), strict=True)
        candidate.train(training); oracle.train(training)
        coefficients = torch.randn_like(graph[0])
        outputs, derivatives = [], []
        for layer in (candidate, oracle):
            nodes = graph[0].detach().clone().requires_grad_()
            # Official attention dropout and residual dropout use the same
            # tensor order; parity covers train BN and the stochastic stream.
            torch.manual_seed(912)
            if device == 'cuda':
                torch.cuda.manual_seed_all(912)
            output = layer(nodes, graph[1], graph[2])
            loss = (output * coefficients).mean()
            derivatives.append(torch.autograd.grad(loss, [nodes] + list(layer.parameters())))
            outputs.append(output)
        torch.testing.assert_close(*outputs, atol=3e-6, rtol=4e-5)
        for old, new in zip(*derivatives):
            torch.testing.assert_close(old, new, atol=2e-6, rtol=5e-4)
        for key, value in oracle.bn.state_dict().items():
            torch.testing.assert_close(value, candidate.bn.state_dict()[key], atol=2e-6, rtol=4e-5)

    def test_official_eval_output_and_all_parameter_gradient_parity_cpu(self):
        self._layer_parity('cpu', False)

    def test_official_joint_train_bn_dropout_and_gradient_parity_cpu(self):
        self._layer_parity('cpu', True)

    @unittest.skipUnless(torch.cuda.is_available(), 'Real CUDA unavailable')
    def test_official_eval_and_joint_train_output_gradient_parity_cuda(self):
        self._layer_parity('cuda', False)
        self._layer_parity('cuda', True)

    def test_edge_embedding_only_affects_attention_and_projection_bias_is_per_edge(self):
        layer = ReferenceRelationLayer(128, 4, 0.).eval()
        with torch.no_grad():
            for parameter in layer.parameters():
                parameter.zero_()
            layer.bn.weight.fill_(1)
            layer.out_proj.bias.fill_(.25)
            layer.lin_edge.bias.fill_(4)
        nodes = torch.zeros(4, 128)
        # Destination 0 receives three edges and destination 1 one edge.
        edges = torch.tensor([[1, 2, 3, 0], [0, 0, 0, 1]])
        attributes = torch.randn(4, 2)
        result = layer(nodes, edges, attributes)
        scale = 1 / math.sqrt(1 + layer.bn.eps)
        torch.testing.assert_close(result[0], torch.full((128,), .75*scale))
        torch.testing.assert_close(result[1], torch.full((128,), .25*scale))
        self.assertTrue(torch.equal(result[2:], torch.zeros_like(result[2:])))
        # Nonzero edge embedding cannot become value when all V are zero.
        with torch.no_grad():
            layer.out_proj.bias.zero_()
        self.assertTrue(torch.equal(layer(nodes, edges, attributes), torch.zeros_like(nodes)))

    def test_joint_topology_has_tf_u_and_one_self_loop_each_without_query_reverse(self):
        original, query, support = fixture()
        graph = joint_topology(*support, query, original.label_seed)
        expected = independent_graph(*support, query, original.label_seed)
        torch.testing.assert_close(graph['nodes'], expected[0], atol=0, rtol=0)
        self.assertEqual(edge_rows(graph['edge_index'], graph['edge_attr']), edge_rows(expected[1], expected[2]))
        self.assertEqual(graph['support_count'], 256)
        self.assertEqual(graph['label_count'], 32)
        self.assertEqual(graph['query_start'], 288)
        self.assertEqual(graph['edge_index'].shape, (2, 5536))
        edges = graph['edge_index']; features = graph['edge_attr']
        loops = edges[0] == edges[1]
        self.assertEqual(int(loops.sum()), 416)
        self.assertEqual(edges[0, loops].sort().values.tolist(), list(range(416)))
        self.assertTrue(torch.equal(features[loops], torch.zeros_like(features[loops])))
        from_query = edges[0] >= 288
        self.assertTrue(bool((edges[0, from_query] == edges[1, from_query]).all()))
        query_messages = (edges[1] >= 288) & ~loops
        self.assertTrue(bool(((edges[0, query_messages] >= 256) & (edges[0, query_messages] < 288)).all()))
        self.assertTrue(torch.equal(features[query_messages], torch.tensor([1., 0.]).expand(4096, 2)))

    def test_two_layer_joint_matches_official_sequence_and_input_histories(self):
        original, query, support = fixture()
        candidate, _ = clone_reference(original)
        candidate.eval()
        expected, edges, attributes = independent_graph(*support, query[:32], candidate.label_seed)
        histories = []
        for i, layer in enumerate(candidate.l1):
            histories.append(expected[256:288].detach().clone())
            oracle = self.Oracle(2, 128, heads=4, dropout=.1, batch_norm=True)
            oracle.load_state_dict(layer.state_dict(), strict=True); oracle.eval()
            expected = oracle(expected, edges, attributes)
            if i != len(candidate.l1) - 1:
                expected = F.gelu(expected)
        result = candidate.encode_joint(query[:32], *support, return_trace=True)
        self.assertEqual(len(result['histories']), 2)
        for old, new in zip(histories, result['histories']):
            torch.testing.assert_close(old, new, atol=3e-6, rtol=4e-5)
        torch.testing.assert_close(result['query'], expected[288:], atol=3e-6, rtol=4e-5)
        torch.testing.assert_close(result['local_labels'], expected[256:288].reshape(16, 2, 128), atol=3e-6, rtol=4e-5)
        torch.testing.assert_close(result['histories'][0], candidate.label_seed.repeat(16, 1), atol=0, rtol=0)

    def test_clone_preserves_original_rng_mode_gradients_l0_and_l2(self):
        original, _, _ = fixture(dtype=torch.float64)
        original.label_seed.grad = torch.full_like(original.label_seed, .125)
        original.local.train(True)
        before, caller_rng = hash_state(original.state_dict()), hash_state(rng_state())
        modes = tuple(module.training for module in original.modules())
        gradients = hash_state({name: value.grad for name, value in original.named_parameters()})
        candidate, metadata = clone_reference(original)
        self.assertIsInstance(candidate, ReferencePromptGraphModel)
        self.assertEqual(hash_state(original.state_dict()), before)
        self.assertEqual(hash_state(rng_state()), caller_rng)
        self.assertEqual(tuple(module.training for module in original.modules()), modes)
        self.assertEqual(hash_state({name: value.grad for name, value in original.named_parameters()}), gradients)
        self.assertTrue(metadata)
        self.assertFalse(metadata['exact_resume'])
        self.assertFalse(metadata['production_default_changed'])
        self.assertFalse(metadata['production_checkpoint_created'])
        self.assertEqual(metadata['unchanged_l0_l2_label_seed_sha256_before'],
                         metadata['unchanged_l0_l2_label_seed_sha256_after'])
        self.assertEqual(metadata['hidden_dimension'], 128)
        self.assertEqual(metadata['heads'], 4)
        self.assertEqual(metadata['layers'], 2)
        self.assertTrue(metadata['batch_norm_contract']['fresh_running_statistics'])
        self.assertFalse(metadata['batch_norm_contract']['l1_activation_checkpointing'])
        for name, value in candidate.named_parameters():
            self.assertEqual(value.dtype, torch.float64, name)
            self.assertIsNot(value, dict(original.named_parameters()).get(name), name)
            if name.startswith(('local.', 'l2.', 'l2_updates.')) or name == 'label_seed':
                torch.testing.assert_close(value, dict(original.named_parameters())[name], atol=0, rtol=0)
        self.assertFalse(any('.ff.' in name or '.update.' in name for name, _ in candidate.l1.named_parameters()))
        optimizer = torch.optim.AdamW(candidate.parameters(), lr=1e-4)
        self.assertEqual({id(value) for group in optimizer.param_groups for value in group['params']},
                         {id(value) for value in candidate.parameters()})

    def test_transfer_qkv_attention_permutation_and_new_bias_bn_identity(self):
        original, _, _ = fixture()
        candidate, metadata = clone_reference(original)
        for index, (old, new) in enumerate(zip(original.l1, candidate.l1)):
            torch.testing.assert_close(new.mlp_kqv.weight, torch.cat((old.q.weight, old.k.weight, old.v.weight)), atol=0, rtol=0)
            self.assertTrue(torch.equal(new.mlp_kqv.bias, torch.zeros_like(new.mlp_kqv.bias)))
            width = 32
            expected = torch.cat((old.attn[0].weight[:, width:2*width], old.attn[0].weight[:, :width], old.attn[0].weight[:, 2*width:]), 1)
            torch.testing.assert_close(new.att_mlp[0].weight, expected, atol=0, rtol=0)
            torch.testing.assert_close(new.att_mlp[0].bias, old.attn[0].bias, atol=0, rtol=0)
            torch.testing.assert_close(new.att_mlp[2].weight, old.attn[2].weight, atol=0, rtol=0)
            torch.testing.assert_close(new.att_mlp[2].bias, old.attn[2].bias, atol=0, rtol=0)
            torch.testing.assert_close(new.lin_edge.weight, old.edge.weight, atol=0, rtol=0)
            self.assertTrue(torch.equal(new.lin_edge.bias, torch.zeros_like(new.lin_edge.bias)))
            torch.testing.assert_close(new.out_proj.weight, old.update.out.weight, atol=0, rtol=0)
            # The copied bias is applied once per edge in this different
            # operator; metadata must expose that alteration explicitly.
            torch.testing.assert_close(new.out_proj.bias, old.update.out.bias, atol=0, rtol=0)
            self.assertEqual(metadata['transfers'][index]['copied'][f'l1.{index}.out_proj.bias'],
                             f'l1.{index}.update.out.bias')
            self.assertIn('per_edge_output_projection', metadata['altered_equations'])
            expected_removed = [f'l1.{index}.'+name for name, _ in old.named_parameters()
                                if name.startswith(('update.norm.', 'update.ff.', 'update.final.'))]
            self.assertEqual(metadata['transfers'][index]['removed_parameters'], expected_removed)
            self.assertTrue(torch.equal(new.bn.weight, torch.ones_like(new.bn.weight)))
            self.assertTrue(torch.equal(new.bn.bias, torch.zeros_like(new.bn.bias)))
            self.assertTrue(torch.equal(new.bn.running_mean, torch.zeros_like(new.bn.running_mean)))
            self.assertTrue(torch.equal(new.bn.running_var, torch.ones_like(new.bn.running_var)))

    def test_eval_128_queries_chunk32_permutation_and_label_invariance(self):
        original, query, support = fixture()
        candidate, _ = clone_reference(original); candidate.eval()
        with torch.no_grad():
            whole = candidate.encode_joint(query, *support)
            parts = [candidate.encode_joint(part, *support) for part in query.split(32)]
            torch.testing.assert_close(torch.cat([item['query'] for item in parts]), whole['query'], atol=3e-6, rtol=4e-5)
            for item in parts:
                torch.testing.assert_close(item['local_labels'], whole['local_labels'], atol=3e-6, rtol=4e-5)
            permutation = torch.randperm(128)
            permuted = candidate.encode_joint(query[permutation], *support)
            torch.testing.assert_close(permuted['query'][permutation.argsort()], whole['query'], atol=3e-6, rtol=4e-5)
            state = candidate.prepare_support(*support)
            full_logits = candidate.predict_embeddings(query, state)['logits']
            chunk_logits = torch.cat([candidate.predict_embeddings(part, state)['logits'] for part in query.split(32)])
            torch.testing.assert_close(full_logits, chunk_logits, atol=3e-6, rtol=4e-5)

    def test_train_executes_one_joint_batchnorm_call_each_layer_and_not_split_parity(self):
        original, query, support = fixture()
        candidate, _ = clone_reference(original); candidate.train()
        observed = []
        handles = [layer.bn.register_forward_pre_hook(lambda module, inputs, i=i: observed.append((i, len(inputs[0]))))
                   for i, layer in enumerate(candidate.l1)]
        candidate.encode_joint(query[:32], *support)
        for handle in handles:
            handle.remove()
        self.assertEqual(observed, [(0, 320), (1, 320)])
        self.assertEqual([int(layer.bn.num_batches_tracked) for layer in candidate.l1], [1, 1])
        # Query sinks have no reverse edges, but official training BN uses
        # the complete joint node table. Splitting BN is not equivalent.
        joint = ReferenceRelationLayer(128, 4, 0.).train()
        separate = copy.deepcopy(joint)
        all_nodes, edges, features = independent_graph(*support, query[:32], original.label_seed)
        joint_out = joint(all_nodes, edges, features)
        keep = (edges[0] < 288) & (edges[1] < 288)
        support_out = separate(all_nodes[:288], edges[:, keep], features[keep])
        self.assertGreater(float((joint_out[:288]-support_out).detach().abs().max()), 1e-4)

    def test_clone_failure_preserves_rng_original_state_and_old_teacher_rejected(self):
        original, query, support = fixture()
        old_plan = original.fit_support_clusters(*support)
        before, caller_rng = hash_state(original.state_dict()), hash_state(rng_state())
        with patch('tools.local_cnn_reference_l1.ReferenceRelationLayer', side_effect=RuntimeError('UNIT constructor error')):
            with self.assertRaisesRegex(RuntimeError, 'constructor error'):
                clone_reference(original)
        self.assertEqual(hash_state(original.state_dict()), before)
        self.assertEqual(hash_state(rng_state()), caller_rng)
        candidate, _ = clone_reference(original)
        with self.assertRaisesRegex(ValueError, 'own support-only teacher'):
            candidate.prepare_support(*support, cluster_plan=old_plan)
        state = candidate.prepare_support(*support)
        state['support_embeddings'] = state['support_embeddings'] + .1
        with self.assertRaisesRegex(ValueError, 'changed after teacher binding'):
            candidate.predict_embeddings(query[:32], state)

    def test_real_local_cnn_extra_state_survives_clone_and_remains_independent(self):
        original, _, _ = fixture()
        config = json.loads((ROOT / 'config/v22_local_cnn.json').read_text())
        original.local = LocalCNN(config).eval()
        before = hash_state(original.state_dict())
        caller_rng = hash_state(rng_state())
        candidate, metadata = clone_reference(original)
        self.assertEqual(hash_state(original.state_dict()), before)
        self.assertEqual(hash_state(rng_state()), caller_rng)
        self.assertEqual(candidate.state_dict()['local._extra_state'], config)
        self.assertEqual(hash_state(candidate.local.state_dict()), hash_state(original.local.state_dict()))
        self.assertEqual(metadata['unchanged_l0_l2_label_seed_sha256_before'],
                         metadata['unchanged_l0_l2_label_seed_sha256_after'])
        candidate.local.config['margin_mm'] = 20
        self.assertEqual(original.local.config['margin_mm'], 10)
        self.assertNotEqual(hash_state(candidate.local.state_dict()), hash_state(original.local.state_dict()))

    def test_optional_trace_uses_same_forward_and_teacher_restores_mixed_modes(self):
        original, query, support = fixture()
        candidate, _ = clone_reference(original)
        candidate.train(); candidate.l1[0].eval(); candidate.l2[1].eval()
        modes = tuple(module.training for module in candidate.modules())
        candidate.fit_support_clusters(*support)
        self.assertEqual(tuple(module.training for module in candidate.modules()), modes)
        candidate.eval()
        state = candidate.prepare_support(*support)
        plain = candidate.predict_embeddings(query[:32], state)
        traced = candidate.predict_embeddings(query[:32], state, return_trace=True)
        torch.testing.assert_close(plain['logits'], traced['logits'], atol=0, rtol=0)
        self.assertEqual(len(traced['query_stages']), 2)
        self.assertEqual([list(value.shape) for value in traced['query_stages']], [[32,128], [32,128]])
        self.assertNotIn('query_stages', plain)
        # Both variants use exactly one joint forward. Compare train outputs
        # and BN buffers using matched initial state and the same dropout seed.
        left = copy.deepcopy(candidate).train(); right = copy.deepcopy(candidate).train()
        torch.manual_seed(111)
        left_output = left.predict_embeddings(query[:32], state)
        torch.manual_seed(111)
        right_output = right.predict_embeddings(query[:32], state, return_trace=True)
        torch.testing.assert_close(left_output['logits'], right_output['logits'], atol=0, rtol=0)
        for key, value in left.state_dict().items():
            if '.bn.' in key:
                torch.testing.assert_close(value, right.state_dict()[key], atol=0, rtol=0)

    def test_inference_api_rejects_query_truth_and_invalid_explicit_data(self):
        original, query, support = fixture()
        candidate, _ = clone_reference(original)
        self.assertNotIn('query_targets', inspect.signature(candidate.encode_joint).parameters)
        self.assertNotIn('query_targets', inspect.signature(candidate.predict_embeddings).parameters)
        state = candidate.prepare_support(*support)
        with self.assertRaises(TypeError):
            candidate.predict_embeddings(query[:32], state, query_targets=torch.ones(32, dtype=torch.long))
        wrong = query.clone(); wrong[0, 0] = float('nan')
        with self.assertRaises((ValueError, FloatingPointError)):
            candidate.encode_joint(wrong, *support)
        wrong_owners = support[1].clone(); wrong_owners[0] = -1
        with self.assertRaises(ValueError):
            candidate.encode_joint(query, support[0], wrong_owners, support[2])

    def test_rank_gradient_reaches_query_l0_reference_l1_l2_and_label_seed(self):
        original, query, support = fixture()
        candidate, _ = clone_reference(original); candidate.eval()
        live_query = query[:32].detach().clone().requires_grad_()
        detached_support = support[0].detach()
        result = candidate.predict_embeddings(live_query, candidate.prepare_support(detached_support, support[1], support[2]))
        score = result['logits'][:, 1]-result['logits'][:, 0]
        loss = F.softplus(score[16:][None, :]-score[:16, None]).mean()
        named = [(name, value) for name, value in candidate.named_parameters() if value.requires_grad]
        gradients = torch.autograd.grad(loss, [live_query]+[value for _, value in named], allow_unused=True)
        self.assertTrue(torch.isfinite(gradients[0]).all())
        self.assertGreater(float(gradients[0].norm()), 0.)
        by_name = dict(zip((name for name, _ in named), gradients[1:]))
        for prefix in ('l1.', 'l2.', 'l2_updates.', 'label_seed'):
            selected = [(name, value) for name, value in by_name.items() if name.startswith(prefix)]
            self.assertTrue(selected, prefix)
            self.assertTrue(all(value is not None and bool(torch.isfinite(value).all()) for _, value in selected), prefix)
            self.assertGreater(sum(float(value.norm()) for _, value in selected), 0., prefix)
        self.assertIsNone(detached_support.grad)


if __name__ == '__main__':
    unittest.main()
