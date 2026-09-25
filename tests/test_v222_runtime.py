"""Execution-only concurrency, integrity and durability tests; no medical claims."""
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import torch
import numpy as np
from dataclasses import dataclass
from tools.v222_runtime_cache import TensorCache, CanonicalStore, CachedPairDataset
from tools.v222_runtime_execution import AsyncSaver, snapshot

ROOT = Path(__file__).resolve().parents[1] / 'work'


class CacheTests(unittest.TestCase):
    def test_prepared_donor_dataclass_and_numpy_views_are_budgeted(self):
        @dataclass
        class Prepared:
            mask: object
            context: object
        raw=np.ones(512,dtype=np.float32)
        cache=TensorCache(2048)
        cache.get('donor',lambda:Prepared(raw,raw[:100]))
        self.assertEqual(cache.bytes,raw.nbytes)
        cache.get('second',lambda:np.zeros(512,dtype=np.float32))
        self.assertEqual(cache.report()['resident_entries'],1)

    def test_view_cache_keeps_epoch_in_identity(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path=Path(directory)/'graph.pt'
            path.write_bytes(b'identity only')
            dataset=CachedPairDataset.__new__(CachedPairDataset)
            dataset.rows=[dict(path=path.name,sha256='test identity')]
            dataset.store=CanonicalStore(path.parent,4096)
            with patch.object(dataset,'record',return_value={'real_record_read_is_tested_separately':True}), \
                 patch('tools.v222_runtime_cache.materialize',side_effect=lambda value,epoch:torch.tensor([epoch])) as build:
                self.assertIs(dataset.item(0,0)[0],dataset.item(0,0)[0])
                self.assertEqual(dataset.item(0,1)[0].item(),1)
                self.assertEqual(build.call_count,2)
    def test_single_flight_and_different_keys_parallel(self):
        cache = TensorCache(4096)
        started, release = threading.Event(), threading.Event()
        calls = []
        def build():
            calls.append(1)
            started.set()
            self.assertTrue(release.wait(5))
            return torch.ones(4)
        with ThreadPoolExecutor(max_workers=4) as pool:
            first = pool.submit(cache.get, 'a', build)
            self.assertTrue(started.wait(5))
            second = pool.submit(cache.get, 'a', build)
            other = pool.submit(cache.get, 'b', lambda: torch.zeros(4))
            self.assertEqual(other.result(timeout=5).sum(), 0)
            release.set()
            self.assertIs(first.result(), second.result())
        self.assertEqual(len(calls), 1)

    def test_aliases_count_once_and_eviction_preserves_budget(self):
        cache = TensorCache(64)
        tensor = torch.ones(8)
        cache.get('a', lambda: tensor)
        cache.get('b', lambda: (tensor, tensor))
        self.assertEqual(cache.bytes, 32)
        cache.get('c', lambda: torch.ones(16))
        self.assertEqual(cache.bytes, 64)
        self.assertEqual(cache.report()['resident_entries'], 1)

    def test_failure_propagates_and_retry_is_not_poisoned(self):
        cache = TensorCache(64)
        with self.assertRaisesRegex(ValueError, 'bad'):
            cache.get('a', lambda: (_ for _ in ()).throw(ValueError('bad')))
        self.assertEqual(cache.get('a', lambda: torch.ones(1)).item(), 1)

    def test_file_identity_integrity_and_no_mutation(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            root = Path(directory)
            buffer = io.BytesIO()
            torch.save({'a': torch.arange(4)}, buffer)
            raw = gzip.compress(buffer.getvalue())
            path = root/'input.pt.gz'
            path.write_bytes(raw)
            digest = hashlib.sha256(raw).hexdigest()
            store = CanonicalStore(root, 4096)
            first, _ = store.read(path.name, digest)
            second, _ = store.read(path.name, digest)
            self.assertIs(first, second)
            path.write_bytes(raw+b'changed')
            with self.assertRaisesRegex(ValueError, 'SHA'):
                store.read(path.name, digest)
            with self.assertRaisesRegex(ValueError, 'escapes'):
                store.path_key('../escape', digest)


class SaverTests(unittest.TestCase):
    def test_snapshot_freezes_cpu_and_reuses_references(self):
        value = torch.tensor([4.])
        copy = snapshot({'a': value, 'b': value})
        value.add_(1)
        self.assertEqual(copy['a'].item(), 4)
        self.assertIs(copy['a'], copy['b'])

    def test_pending_snapshot_is_immutable_and_stop_waits(self):
        from hiercp_v222.v1_execution import atomic_torch
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            net = torch.nn.Linear(2, 1)
            optimizer = torch.optim.AdamW(net.parameters())
            saver = AsyncSaver(directory, net, optimizer, {'debug': True})
            began, allow = threading.Event(), threading.Event()
            expected = net.weight.detach().clone()
            def delayed(path, payload):
                began.set()
                if not allow.wait(5):
                    raise TimeoutError('Test writer wait')
                atomic_torch(path, payload)
            try:
                with patch('tools.v222_runtime_execution.original.atomic_torch', delayed):
                    receipt = saver.save(dict(phase='optimization', epoch=0, step=1))
                    self.assertTrue(receipt['pending'])
                    self.assertTrue(began.wait(5))
                    with torch.no_grad():
                        net.weight.add_(10)
                    (Path(directory)/'STOP_AFTER_BATCH').touch()
                    allow.set()
                    self.assertTrue(saver.stop_requested())
                saved = torch.load(Path(directory)/'checkpoint_latest.pt', weights_only=False)
                self.assertTrue(torch.equal(saved['model']['weight'], expected))
                status = json.loads((Path(directory)/'checkpoint_status.json').read_text())
                self.assertTrue(status['durable'])
                self.assertEqual(status['step'], 1)
            finally:
                allow.set()
                saver.close()

    def test_writer_error_preserves_previous_and_is_reported(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            net = torch.nn.Linear(2, 1)
            saver = AsyncSaver(directory, net, torch.optim.AdamW(net.parameters()), {'debug': True})
            saver.save(dict(phase='optimization', epoch=0, step=1))
            saver.flush()
            with patch('tools.v222_runtime_execution.original.atomic_torch', side_effect=OSError('disk failed')):
                saver.save(dict(phase='optimization', epoch=0, step=2))
                with self.assertRaisesRegex(OSError, 'disk failed'):
                    saver.flush()
                with self.assertRaisesRegex(OSError, 'disk failed'):
                    saver.close()
            saved = torch.load(Path(directory)/'checkpoint_latest.pt', weights_only=False)
            self.assertEqual(saved['state']['step'], 1)


class IntegrationTests(unittest.TestCase):
    def test_resume_embedded_workspace_survives_checkpoint_move_and_rejects_changes(self):
        from tools.run_v222_optimized import runtime_identity,resume_policy
        from tools.v222_gpu_workspace import policy_path
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            checkpoint=Path(directory)/'training/checkpoint_latest.pt'
            old=dict(state=dict(release_unused=True))
            self.assertEqual(resume_policy(old,checkpoint),(64,True))
            with self.assertRaisesRegex(ValueError,'workspace'):resume_policy(old,checkpoint,256)
            with self.assertRaisesRegex(ValueError,'allocator'):resume_policy(old,checkpoint,64,False)
            saved=dict(state=dict(release_unused=False),execution_policy=dict(
                workspace_mib=256,runtime_sha256=runtime_identity()))
            self.assertEqual(resume_policy(saved,checkpoint),(256,False))
            policy_path(checkpoint.parent).write_text(json.dumps(dict(workspace_mib=64)))
            with self.assertRaisesRegex(ValueError,'disagree'):resume_policy(saved,checkpoint)
            saved['execution_policy']['runtime_sha256']={}
            with self.assertRaisesRegex(ValueError,'runtime changed'):resume_policy(saved,checkpoint)

    def test_optimized_server_uses_new_backend_and_reuses_complete_cache(self):
        from tools.run_v222_server import stages
        default=dict(stages(Path('/medical'),Path('/run'),32,9,256,optimized=True))
        self.assertTrue(any('v222_prepare_optimized.py' in arg for arg in default['paired_cache']))
        self.assertTrue(any('run_v222_optimized.py' in arg for arg in default['gnn_training']))
        self.assertIn('--optimized-runtime',default['profile_DEBUG'])
        resumed=dict(stages(Path('/medical'),Path('/run'),32,9,256,optimized=True,
                            cache=Path('/old/index.json'),resume=Path('/old/checkpoint_latest.pt')))
        self.assertEqual(list(resumed),['resources','model_check','gnn_training'])
        self.assertNotIn('--workspace-mib',resumed['gnn_training'])
        self.assertNotIn('--release-unused',resumed['gnn_training'])
        self.assertIn(str(Path('/old/index.json')),resumed['gnn_training'])


if __name__ == '__main__':
    unittest.main()
