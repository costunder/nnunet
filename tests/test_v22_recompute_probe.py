"""Short DEBUG unit/operator tests; synthetic graphs are not CT performance evidence."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from types import SimpleNamespace
import unittest
import numpy as np
import torch
from tools.benchmark_v22_recompute_debug import execution_policy, exact, frozen, probe_batches


class ProbeTests(unittest.TestCase):
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

    def test_snapshot_does_not_alias_cpu_adam_or_numpy_rng(self):
        value = {'adam': torch.ones(3), 'rng': (np.arange(3), [torch.zeros(2)])}
        snapshot = frozen(value)
        self.assertTrue(exact(value, snapshot))
        value['adam'].add_(1)
        value['rng'][0][0] = 9
        self.assertEqual(snapshot['adam'].tolist(), [1, 1, 1])
        self.assertEqual(snapshot['rng'][0].tolist(), [0, 1, 2])
        self.assertFalse(exact(value, snapshot))

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
            block = HeteroGATv2Block(node_types=kinds, edge_types=tuple(edges), dim=128,
                                     heads=4, edge_dim=None, dropout=.1).cuda().train()
            inputs = {'a': torch.randn(17,128,device='cuda'), 'b': torch.randn(19,128,device='cuda')}
            local = SimpleNamespace(checkpoint_local_blocks=True, training=True)
            net = SimpleNamespace(local=local)
            results = []
            # DEBUG-only small workspace forces multiple real edge chunks.
            with patch('hiercp.model.EDGE_ATTENTION_WORKSPACE_BYTES', 8*4*128*7):
                for policy in ('baseline','no_outer_l0'):
                    block.zero_grad(set_to_none=True)
                    torch.manual_seed(913)
                    x = {key: value.detach().clone().requires_grad_() for key,value in inputs.items()}
                    with execution_policy(net,policy), torch.autocast('cuda',dtype=torch.bfloat16):
                        out = CTOnlyEncoder._run_local_block(local,block,x,edges,attrs)
                        loss = sum(value.float().square().mean() for value in out.values())
                    loss.backward()
                    results.append(frozen(dict(out=out, gradients={k:p.grad for k,p in block.named_parameters()},
                                               inputs={k:v.grad for k,v in x.items()}, rng=torch.cuda.get_rng_state())))
            self.assertTrue(exact(results[0],results[1]))
        finally:
            torch.use_deterministic_algorithms(prior)


if __name__ == '__main__':
    unittest.main()
