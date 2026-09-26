"""CPU checkpoint unit fixtures, not medical model/accuracy evidence."""
import tempfile
from pathlib import Path
import unittest
import torch
import random
import numpy as np
from tools.v222_support_snapshot import AsyncSaver


class FixedSnapshotTests(unittest.TestCase):
    def test_complete_checkpoint_state_and_fresh_weights_after_scope(self):
        with tempfile.TemporaryDirectory() as folder:
            net=torch.nn.Linear(3,2)
            optimizer=torch.optim.AdamW(net.parameters())
            net(torch.ones(2,3)).sum().backward();optimizer.step()
            saver=AsyncSaver(Path(folder),net,optimizer,dict(debug=True))
            state=dict(phase='initial_memory',epoch=0,step=1,memory_next=2,memory_work=torch.ones(2,128))
            try:
                with saver.fixed_support():
                    frozen=saver.frozen_payload
                    saver.save(state);saver.flush()
                    state.update(memory_next=4,memory_work=torch.ones(4,128)*2)
                    expected_python=random.getstate()
                    expected_numpy=np.random.get_state()
                    saver.save(state);saver.flush()
                    self.assertIs(saver.frozen_payload,frozen)
                    value=torch.load(Path(folder)/'checkpoint_latest.pt',weights_only=False)
                    self.assertEqual(value['state']['memory_next'],4)
                    self.assertEqual(value['rng']['python'],expected_python)
                    for left,right in zip(value['rng']['numpy'],expected_numpy):
                        self.assertTrue(np.array_equal(left,right))
                    torch.testing.assert_close(value['model']['weight'],net.weight,rtol=0,atol=0)
                    self.assertEqual(value['optimizer']['state'].keys(),optimizer.state_dict()['state'].keys())
                self.assertIsNone(saver.frozen_payload)
                with torch.no_grad():net.weight.add_(1)
                state['phase']='optimization'
                saver.save(state);saver.flush()
                value=torch.load(Path(folder)/'checkpoint_latest.pt',weights_only=False)
                torch.testing.assert_close(value['model']['weight'],net.weight,rtol=0,atol=0)
            finally:saver.close()

    def test_unexpected_parameter_change_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            net=torch.nn.Linear(3,2);optimizer=torch.optim.AdamW(net.parameters())
            saver=AsyncSaver(Path(folder),net,optimizer,dict(debug=True))
            try:
                with saver.fixed_support():
                    with torch.no_grad():net.weight.add_(1)
                    with self.assertRaisesRegex(RuntimeError,'Model changed'):
                        saver.save(dict(phase='refresh_memory',epoch=0,step=1))
            finally:saver.close()

    def test_optimizer_mutation_is_rejected_and_scope_released(self):
        with tempfile.TemporaryDirectory() as folder:
            net=torch.nn.Linear(3,2);optimizer=torch.optim.AdamW(net.parameters())
            net(torch.ones(2,3)).sum().backward();optimizer.step()
            saver=AsyncSaver(Path(folder),net,optimizer,dict(debug=True))
            try:
                with saver.fixed_support():
                    next(iter(optimizer.state.values()))['exp_avg'].add_(1)
                    with self.assertRaisesRegex(RuntimeError,'Model changed'):
                        saver.save(dict(phase='refresh_memory',epoch=0,step=1))
                self.assertIsNone(saver.frozen_payload)
            finally:saver.close()


if __name__=='__main__':unittest.main()
