"""CPU DEBUG fault injection for a live, unpinned training BEST writer.

Tiny decoder adapters test filesystem publication; they are not clinical
checkpoints, native model validation, or training performance evidence.
"""
from pathlib import Path
import json
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools import persist_basic_cp_seven as durable
from tests.test_persist_basic_cp_seven import debug_torch, debug_shared_read_open


class TrainingBestRaceTests(unittest.TestCase):
    def fixture(self, root):
        source=root/'DEBUG_best.pth';source.write_bytes(b'DEBUG_WHOLE_STATE_NEW')
        folder=root/'cache/GPU6';folder.mkdir(parents=True)
        target=folder/'CURRENT.pth';target.write_bytes(b'DEBUG_PREVIOUS_VALID_FULL_STATE')
        previous={'sha256':durable.digest(target),'saved_resume_epoch':3,'role':'OWN_BEST_TRAINING_STATE'}
        for name in ('CURRENT.json','PENDING.json'):
            (folder/name).write_text(json.dumps(previous),encoding='utf8')
        cache=durable.Cache(root/'cache',10**12)
        return source,target,cache,{p:p.read_bytes() for p in folder.iterdir()}

    def test_actual_in_place_truncation_keeps_previous_full_state_and_receipts(self):
        with tempfile.TemporaryDirectory(prefix='DEBUG_BEST_RACE_',dir=Path(__file__).parent)as temp:
            source,target,cache,previous=self.fixture(Path(temp));count=[0]
            def guard(*args):
                count[0]+=1
                if count[0]==3:source.write_bytes(b'DEBUG_SHORT_WRITE')
            with patch.dict('sys.modules',torch=debug_torch()),patch.object(cache,'guard',side_effect=guard), \
                 patch.object(Path,'open',debug_shared_read_open),patch.object(durable,'fsync_directory'):
                with self.assertRaises(durable.CheckpointSourceInFlight):
                    cache.copy(source,target,trainer='DEBUG_NATIVE',role='OWN_BEST_TRAINING_STATE',
                        extra={'latest_on_tmpfs_N':4})
            self.assertEqual({p:p.read_bytes()for p in target.parent.iterdir()},previous)
            source.write_bytes(b'DEBUG_NEXT_STABLE_WHOLE_STATE')
            with patch.dict('sys.modules',torch=debug_torch(epoch=4)),patch.object(cache,'guard'), \
                 patch.object(durable,'fsync_directory'):
                proof=cache.copy(source,target,trainer='DEBUG_NATIVE',role='OWN_BEST_TRAINING_STATE',
                    extra={'latest_on_tmpfs_N':5})
            self.assertEqual(target.read_bytes(),source.read_bytes())
            self.assertEqual(proof['saved_resume_epoch'],4)
            self.assertTrue(proof['scaler_present']);self.assertEqual(proof['optimizer_states'],1)
            self.assertEqual(set(proof['logger_lengths'].values()),{4})

    def test_actual_decode_failure_is_transient_only_when_held_inode_changed(self):
        with tempfile.TemporaryDirectory(prefix='DEBUG_BEST_RACE_',dir=Path(__file__).parent)as temp:
            source,target,cache,previous=self.fixture(Path(temp))
            def load(stream,**kwargs):
                source.write_bytes(b'DEBUG_TRUNCATED_DURING_DECODE');raise EOFError('DEBUG interrupted writer')
            decoder=SimpleNamespace(load=load,cuda=SimpleNamespace(is_initialized=lambda:False))
            with patch.dict('sys.modules',torch=decoder),patch.object(cache,'guard'), \
                 patch.object(Path,'open',debug_shared_read_open):
                with self.assertRaises(durable.CheckpointSourceInFlight):
                    cache.copy(source,target,trainer='DEBUG_NATIVE',role='OWN_BEST_TRAINING_STATE')
            self.assertEqual({p:p.read_bytes()for p in target.parent.iterdir()},previous)

    def test_stable_corruption_is_reported_and_preserves_previous_bytes(self):
        with tempfile.TemporaryDirectory(prefix='DEBUG_BEST_RACE_',dir=Path(__file__).parent)as temp:
            source,target,cache,previous=self.fixture(Path(temp))
            decoder=SimpleNamespace(load=lambda *a,**k:(_ for _ in ()).throw(RuntimeError('DEBUG stable corruption')))
            with patch.dict('sys.modules',torch=decoder),patch.object(cache,'guard'):
                with self.assertRaisesRegex(RuntimeError,'DEBUG stable corruption')as raised:
                    cache.copy(source,target,trainer='DEBUG_NATIVE',role='OWN_BEST_TRAINING_STATE')
            self.assertNotIsInstance(raised.exception,durable.CheckpointSourceInFlight)
            self.assertEqual({p:p.read_bytes()for p in target.parent.iterdir()},previous)

    def test_pinned_final_and_generic_artifact_mutation_remain_hard_errors(self):
        with tempfile.TemporaryDirectory(prefix='DEBUG_BEST_RACE_',dir=Path(__file__).parent)as temp:
            source,target,cache,previous=self.fixture(Path(temp))
            def mutate(stream):source.write_bytes(b'DEBUG_IN_PLACE_CHANGE')
            with patch.dict('sys.modules',torch=debug_torch(mutate)),patch.object(cache,'guard'), \
                 patch.object(Path,'open',debug_shared_read_open):
                with self.assertRaisesRegex(ValueError,'Source inode changed'):
                    cache.copy(source,target,trainer='DEBUG_NATIVE',role='OWN_BEST_TRAINING_STATE',expected='0'*64)
            self.assertEqual({p:p.read_bytes()for p in target.parent.iterdir()},previous)

    def test_chunk_guards_reuse_observation_but_each_new_copy_rechecks_budget(self):
        with tempfile.TemporaryDirectory(prefix='DEBUG_CACHE_GUARD_',dir=Path(__file__).parent)as temp:
            root=Path(temp);item=root/'DEBUG_allocated';item.write_bytes(b'x'*1000)
            cache=durable.Cache(root,10**12);rglob=Path.rglob
            process=SimpleNamespace(memory_info=lambda:SimpleNamespace(rss=1))
            with patch('psutil.Process',return_value=process),patch.object(durable,'GIB',128), \
                 patch.object(durable.shutil,'disk_usage',return_value=SimpleNamespace(free=10**9)), \
                 patch.object(durable.time,'monotonic',return_value=100), \
                 patch.object(Path,'rglob',autospec=True,side_effect=rglob)as walks:
                cache.guard(1)
                for _ in range(100):cache.guard()
                self.assertEqual(walks.call_count,1)
                with self.assertRaisesRegex(ValueError,'exceeds8GiB'):cache.guard(25)
                self.assertEqual(walks.call_count,2)


if __name__=='__main__':unittest.main()
