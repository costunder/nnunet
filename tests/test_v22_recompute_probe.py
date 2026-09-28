"""Short DEBUG unit/operator tests; synthetic graphs are not CT performance evidence."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from types import SimpleNamespace
import unittest
import numpy as np
import torch
from tools.benchmark_v22_recompute_debug import execution_policy, exact, frozen, probe_batches, stage_events


class ProbeTests(unittest.TestCase):
    def test_support_selection_reuses_resumed_episode_and_refits_only_next_group(self):
        from unittest.mock import Mock, patch
        from contextlib import nullcontext
        from tools.benchmark_v22_recompute_debug import episode_support
        net = SimpleNamespace(fit_support_clusters=Mock(return_value='new_plan'))
        state = dict(memory='full', last_group='A', plan='saved_plan')
        first, second = (object(), object(), object()), (object(), object(), object())
        with patch('tools.v222_review_contracts.grouped_support', side_effect=[first, second]) as select, \
             patch('torch.autocast', return_value=nullcontext()):
            support = episode_support(net, state, 'A', None)
            self.assertIs(support, first)
            self.assertIs(episode_support(net, state, 'A', support), first)
            self.assertEqual(state['plan'], 'saved_plan')
            net.fit_support_clusters.assert_not_called()
            support = episode_support(net, state, 'B', support)
            self.assertIs(support, second)
            self.assertEqual(select.call_count, 2)
            net.fit_support_clusters.assert_called_once_with(*second)
            self.assertEqual(state['plan'], 'new_plan')

    def test_scope_restores_on_failure_and_preserves_cnn(self):
        local = SimpleNamespace(checkpoint_local_blocks=True, checkpoint_dense_encoder=True)
        net = SimpleNamespace(local=local)
        with self.assertRaisesRegex(RuntimeError, 'probe'):
            with execution_policy(net, 'no_outer_l0'):
                self.assertFalse(local.checkpoint_local_blocks)
                self.assertTrue(local.checkpoint_dense_encoder)
                raise RuntimeError('probe')
        self.assertTrue(local.checkpoint_local_blocks)
        with self.assertRaises(ValueError):
            with execution_policy(net, 'other'):
                pass

    def test_probe_preserves_cursor_batch_and_rejects_epoch_wrap(self):
        from hiercp_v222.v1_training import groups
        data = SimpleNamespace(rows=[dict(patient_group='a', bounds=dict(edges=i)) for i in range(11)])
        state = dict(phase='optimization', training_calibrated=True, memory={'present': True},
                     batch=3, epoch=2, next_batch=1)
        self.assertEqual(probe_batches(data, state, 42, 3), list(groups(data, 3, 42, 2))[1:])
        with self.assertRaisesRegex(ValueError, 'Too few'):
            probe_batches(data, state, 42, 4)
        with self.assertRaisesRegex(ValueError, 'full support'):
            probe_batches(data, dict(state, phase='initial_memory'), 42, 3)
        with self.assertRaisesRegex(ValueError, 'At least'):
            probe_batches(data, state, 42, 1)
        self.assertEqual(probe_batches(data, state, 42, 1, profile_only=True),
                         list(groups(data, 3, 42, 2))[1:2])
        with self.assertRaisesRegex(ValueError, 'exactly one'):
            probe_batches(data, state, 42, 3, profile_only=True)
        with self.assertRaisesRegex(ValueError, 'full support'):
            probe_batches(data, dict(state, phase='initial_memory'), 42, 1, profile_only=True)

    def test_snapshot_does_not_alias_cpu_adam_or_numpy_rng(self):
        value = {'adam': torch.ones(3), 'rng': (np.arange(3), [torch.zeros(2)])}
        snapshot = frozen(value)
        self.assertTrue(exact(value, snapshot))
        value['adam'].add_(1)
        value['rng'][0][0] = 9
        self.assertEqual(snapshot['adam'].tolist(), [1, 1, 1])
        self.assertEqual(snapshot['rng'][0].tolist(), [0, 1, 2])
        self.assertFalse(exact(value, snapshot))

    def test_selective_policy_bypasses_only_requested_block_and_restores(self):
        from unittest.mock import Mock
        blocks = [Mock(return_value=i) for i in range(3)]
        original = Mock(return_value='checkpointed')
        local = SimpleNamespace(checkpoint_local_blocks=True, checkpoint_dense_encoder=True,
                                blocks=blocks, _run_local_block=original)
        with execution_policy(SimpleNamespace(local=local), 'no_outer_l0_block1'):
            self.assertEqual(local._run_local_block(blocks[0], {}, {}, {}), 'checkpointed')
            self.assertEqual(local._run_local_block(blocks[1], {}, {}, {}), 1)
            self.assertEqual(local._run_local_block(blocks[2], {}, {}, {}), 'checkpointed')
            self.assertTrue(local.checkpoint_dense_encoder)
            self.assertTrue(local.checkpoint_local_blocks)
        self.assertIs(local._run_local_block, original)
        self.assertEqual(original.call_count, 2)
        self.assertEqual([b.call_count for b in blocks], [0, 1, 0])

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA timing requires a GPU')
    def test_cuda_stage_events_and_profiler_api(self):
        local = torch.nn.Linear(128,128).cuda()
        start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        data = torch.randn(8,128,device='cuda')
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                torch.profiler.ProfilerActivity.CUDA]) as trace:
            with stage_events(local) as (stamps,calls):
                value = local(data)
                loss = value.square().mean()
                start.record()
                loss.backward()
                end.record()
                end.synchronize()
            self.assertEqual(len(calls),1)
            self.assertGreaterEqual(stamps['begin'].elapsed_time(stamps['end']),0)
            self.assertGreaterEqual(start.elapsed_time(stamps['gradient']),0)
            self.assertGreaterEqual(stamps['gradient'].elapsed_time(end),0)
        self.assertFalse(local._forward_hooks)
        self.assertFalse(local._forward_pre_hooks)
        self.assertIn('aten::',trace.key_averages().table(sort_by='self_cuda_time_total',row_limit=5))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA operator validation requires a GPU')
    def test_cuda_full_width_attention_checkpoint_parity(self):
        from hiercp.model import HeteroGATv2Block
        from hiercp_v22.local import CTOnlyEncoder
        from unittest.mock import patch
        prior = torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(True)
        try:
            torch.manual_seed(42)
            kinds = ('a', 'b')
            edges = {('a','to','b'): torch.stack((torch.arange(63)%17, torch.arange(63)%19)).cuda(),
                     ('b','to','a'): torch.stack((torch.arange(59)%19, torch.arange(59)%17)).cuda()}
            attrs = {key: None for key in edges}
            blocks = torch.nn.ModuleList([HeteroGATv2Block(node_types=kinds, edge_types=tuple(edges), dim=128,
                                     heads=4, edge_dim=None, dropout=.1) for _ in range(3)]).cuda().train()
            inputs = {'a': torch.randn(17,128,device='cuda'), 'b': torch.randn(19,128,device='cuda')}
            from types import MethodType
            local = SimpleNamespace(checkpoint_local_blocks=True, training=True, blocks=blocks)
            local._run_local_block = MethodType(CTOnlyEncoder._run_local_block, local)
            net = SimpleNamespace(local=local)
            results = []
            # DEBUG-only small workspace forces multiple real edge chunks.
            with patch('hiercp.model.EDGE_ATTENTION_WORKSPACE_BYTES', 8*4*128*7):
                for policy in ('baseline','no_outer_l0', 'no_outer_l0_block0', 'no_outer_l0_block1', 'no_outer_l0_block2'):
                    blocks.zero_grad(set_to_none=True)
                    torch.manual_seed(913)
                    x = {key: value.detach().clone().requires_grad_() for key,value in inputs.items()}
                    with execution_policy(net,policy), torch.autocast('cuda',dtype=torch.bfloat16):
                        out = x
                        for block in blocks:
                            out = local._run_local_block(block,out,edges,attrs)
                        loss = sum(value.float().square().mean() for value in out.values())
                    loss.backward()
                    results.append(frozen(dict(out=out, gradients={k:p.grad for k,p in blocks.named_parameters()},
                                               inputs={k:v.grad for k,v in x.items()}, rng=torch.cuda.get_rng_state())))
            for result in results[1:]:
                self.assertTrue(exact(results[0],result))
        finally:
            torch.use_deterministic_algorithms(prior)


if __name__ == '__main__':
    unittest.main()
