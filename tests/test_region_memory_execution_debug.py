"""Execution-only diagnostic controls; no model/data reductions."""
import unittest
from types import SimpleNamespace
import torch
from tools.compare_region_memory_debug import execution, assert_state_close
from hiercp_v222 import model
from l0_regions import sparse


class ExecutionChecks(unittest.TestCase):
    def test_metadata_and_tensor_parity(self):
        state={'tensor': torch.tensor([1.]), 'contract': {'view': 'stride4', 'debug': True}}
        assert_state_close(state, state)
        with self.assertRaises(AssertionError):
            assert_state_close(state, {**state, 'tensor': torch.tensor([2.])})
        with self.assertRaises(AssertionError):
            assert_state_close(state, {**state, 'contract': {'view': 'other', 'debug': True}})

    def test_context_restores_execution_on_error(self):
        net=SimpleNamespace(local=SimpleNamespace(core=SimpleNamespace(
            checkpoint_dense_encoder=True, checkpoint_local_blocks=True)))
        checkpoint, mm=model.checkpoint, sparse.segmented_mm
        with self.assertRaisesRegex(RuntimeError, 'intentional'):
            with execution(net, retain=True, workspace_bytes=256*2**20):
                self.assertFalse(net.local.core.checkpoint_dense_encoder)
                self.assertFalse(net.local.core.checkpoint_local_blocks)
                self.assertEqual(model.checkpoint(lambda x:x+1, 2, use_reentrant=False), 3)
                raise RuntimeError('intentional')
        self.assertTrue(net.local.core.checkpoint_dense_encoder)
        self.assertTrue(net.local.core.checkpoint_local_blocks)
        self.assertIs(model.checkpoint, checkpoint)
        self.assertIs(sparse.segmented_mm, mm)

    def test_checkpoint_options_are_not_silently_ignored(self):
        net=SimpleNamespace(local=SimpleNamespace(core=SimpleNamespace(
            checkpoint_dense_encoder=True, checkpoint_local_blocks=True)))
        with execution(net, retain=True, workspace_bytes=64*2**20):
            with self.assertRaises(ValueError):
                model.checkpoint(lambda x:x, 2, use_reentrant=True)
            with self.assertRaises(ValueError):
                model.checkpoint(lambda x:x, 2, context_fn=lambda:None)


if __name__=='__main__':
    unittest.main()
