"""CPU UNIT checks of exact checkpoint identity, snapshots and failure safety."""
from __future__ import annotations

from contextlib import redirect_stderr
import errno
import io
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch

from hiercp_v1x.comparison_checkpoint import ComparisonCheckpointWriter
from hiercp_v1x.u_bridge_training import cpu_copy, digest


def payload():
    return dict(format='UNIT_original_checkpoint', identity_sha256='UNIT_identity',
                comparison_policy={'arm': 'UNIT_listwise', 'nested': [None, True, 1.25]},
                model={'weight': torch.arange(12, dtype=torch.float32).reshape(3, 4).T,
                       'bf16': torch.arange(3, dtype=torch.bfloat16),
                       'empty': torch.empty(0, dtype=torch.int64)},
                optimizer={'state': {0: {'step': torch.tensor(3.),
                                        'exp_avg': torch.arange(12.)}},
                           'param_groups': [{'lr': 0.001, 'params': [0]}]},
                scheduler={'last_epoch': 3}, scaler={'scale': 32768.},
                state={'phase': 'validation', 'position': 0, 'rows': []},
                rng={'numpy': np.arange(8, dtype=np.uint32), 'python': (1, 2, 3),
                     'torch': torch.arange(10, dtype=torch.uint8), 'cuda': []},
                shuffle_generator=torch.arange(5, dtype=torch.uint8))


class CheckpointWriterUnit(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory(prefix='UNIT_checkpoint_',
                                            dir=Path(__file__).resolve().parents[1])
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'latest.pt'
        self.writer = ComparisonCheckpointWriter()

    def read_verified(self, path=None):
        saved = torch.load(path or self.path, map_location='cpu', weights_only=False)
        checksum = saved.pop('content_sha256')
        self.assertEqual(checksum, digest(saved))
        return saved

    def test_exact_frozen_digest_plain_torch_load_and_best_copy(self):
        live = payload()
        expected = cpu_copy(live)
        best = self.path.with_name('best.pt')
        receipt = self.writer.save([self.path, best], live)
        self.assertFalse(receipt['static_snapshot_reused'])
        self.assertEqual(receipt['content_sha256'], digest(expected))
        self.assertEqual(digest(self.read_verified()), digest(expected))
        self.assertEqual(digest(self.read_verified(best)), digest(expected))
        self.assertEqual({p.name for p in self.path.parent.iterdir()}, {'latest.pt', 'best.pt'})

    def test_training_recaptures_even_if_state_version_did_not_change(self):
        live = payload()
        self.writer.save([self.path], live)
        # Training never relies on a static-generation witness, including when
        # custom optimizers update raw storage without incrementing _version.
        live['model']['weight'].data.add_(20)
        receipt = self.writer.save([self.path], live)
        self.assertFalse(receipt['static_snapshot_reused'])
        self.assertEqual(digest(self.read_verified()), digest(live))

    def test_eval_reuses_static_but_refreshes_every_cursor_rng_and_shuffle(self):
        live = payload()
        first = self.writer.save([self.path], live, static_generation=('validation', 3, 12))
        live['state']['position'] = 1
        live['state']['rows'].append({'loss': 0.33})
        live['rng']['numpy'][0] = 77
        live['rng']['torch'].add_(1)
        live['shuffle_generator'].add_(2)
        with patch('hiercp_v1x.comparison_checkpoint.cpu_copy', wraps=cpu_copy) as clone:
            second = self.writer.save([self.path], live, static_generation=('validation', 3, 12))
        self.assertTrue(second['static_snapshot_reused'])
        self.assertEqual(clone.call_count, 3)
        self.assertNotEqual(first['content_sha256'], second['content_sha256'])
        self.assertEqual(digest(self.read_verified()), digest(live))

    def test_eval_normal_tensor_mutation_metadata_and_replacement_refresh(self):
        live = payload()
        self.writer.save([self.path], live, static_generation='eval')
        mutations = [lambda: live['model']['weight'].add_(1),
                     lambda: live['optimizer']['state'][0]['exp_avg'].add_(2),
                     lambda: live['scheduler'].update(last_epoch=4),
                     lambda: live['optimizer']['param_groups'][0].update(lr=0.002),
                     lambda: live['model'].update(weight=live['model']['weight'].clone())]
        for mutate in mutations:
            mutate()
            receipt = self.writer.save([self.path], live, static_generation='eval')
            self.assertFalse(receipt['static_snapshot_reused'])
            self.assertEqual(digest(self.read_verified()), digest(live))

    def test_generation_change_and_invalidation_force_new_snapshot(self):
        live = payload()
        self.writer.save([self.path], live, static_generation=1)
        self.assertFalse(self.writer.save([self.path], live, static_generation=2)['static_snapshot_reused'])
        self.writer.invalidate()
        self.assertFalse(self.writer.save([self.path], live, static_generation=2)['static_snapshot_reused'])

    def test_snapshot_does_not_alias_caller_cpu_storage(self):
        live = payload()
        self.writer.save([self.path], live, static_generation='eval')
        before = digest(self.writer._static)
        live['model']['weight'].add_(7)
        live['optimizer']['state'][0]['exp_avg'].zero_()
        live['optimizer']['param_groups'][0]['lr'] = 99.
        self.assertEqual(digest(self.writer._static), before)

    def test_write_failure_preserves_previous_checkpoint_and_cleans_only_owned_partial(self):
        live = payload()
        self.writer.save([self.path], live)
        previous = self.path.read_bytes()
        unrelated = self.path.with_name('other_process.tmp')
        unrelated.write_bytes(b'UNIT_existing_other_file')
        def partial_then_quota(value, stream):
            stream.write(b'UNIT_partial_new_checkpoint')
            raise OSError(errno.EDQUOT, 'UNIT quota failure')
        with patch('hiercp_v1x.comparison_checkpoint.torch.save', side_effect=partial_then_quota):
            with self.assertRaises(OSError) as failure:
                self.writer.save([self.path], live)
        self.assertEqual(failure.exception.errno, errno.EDQUOT)
        self.assertEqual(self.path.read_bytes(), previous)
        self.assertEqual(unrelated.read_bytes(), b'UNIT_existing_other_file')
        self.assertEqual({p.name for p in self.path.parent.iterdir()}, {'latest.pt', 'other_process.tmp'})

    def test_fsync_failure_does_not_publish(self):
        with patch('hiercp_v1x.comparison_checkpoint.os.fsync', side_effect=OSError('UNIT fsync failure')):
            with self.assertRaisesRegex(OSError, 'UNIT fsync failure'):
                self.writer.save([self.path], payload())
        self.assertFalse(self.path.exists())
        self.assertEqual(list(self.path.parent.iterdir()), [])

    def test_cleanup_failure_reports_both_errors_without_masking_quota(self):
        output = io.StringIO()
        quota = OSError(errno.EDQUOT, 'UNIT quota failure')
        cleanup = OSError(errno.EACCES, 'UNIT cleanup denied')
        with redirect_stderr(output), \
             patch('hiercp_v1x.comparison_checkpoint.torch.save', side_effect=quota), \
             patch('hiercp_v1x.comparison_checkpoint.Path.unlink', side_effect=cleanup):
            with self.assertRaises(OSError) as failure:
                self.writer.save([self.path], payload())
        self.assertIs(failure.exception, quota)
        self.assertIs(failure.exception.comparison_secondary_errors[0][1], cleanup)
        self.assertIn('UNIT cleanup denied', output.getvalue())
        self.assertFalse(self.path.exists())

    def test_refuses_checksum_and_missing_state_and_duplicate_destinations(self):
        for invalid in (dict(payload(), content_sha256='stale'),
                        {key: value for key, value in payload().items() if key != 'rng'}):
            with self.assertRaises(ValueError):
                self.writer.save([self.path], invalid)
        with self.assertRaises(ValueError):
            self.writer.save([self.path, self.path], payload())
        with self.assertRaises(ValueError):
            self.writer.save([], payload())


if __name__ == '__main__':
    unittest.main()
