"""CPU DEBUG tests: tiny state adapters prove file/race/receipt behavior only.

No tiny fixture is a clinical checkpoint, training result or throughput claim.
The torch decoder adapter isolates held-FD semantics from a native1GiB model.
"""
from __future__ import annotations
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools import persist_basic_cp_seven as durable


def debug_shared_read_open(path, mode='r', *args, **kwargs):
    """Windows DEBUG OS adapter gives held read FD Linux-like delete sharing."""
    if os.name!='nt' or mode!='rb': return _PATH_OPEN(path,mode,*args,**kwargs)
    import ctypes,msvcrt
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.CreateFileW.argtypes=(ctypes.c_wchar_p,ctypes.c_uint32,ctypes.c_uint32,
        ctypes.c_void_p,ctypes.c_uint32,ctypes.c_uint32,ctypes.c_void_p)
    kernel.CreateFileW.restype=ctypes.c_void_p
    handle=kernel.CreateFileW(str(path),0x80000000,7,None,3,0x80,None)
    if handle==ctypes.c_void_p(-1).value: raise ctypes.WinError(ctypes.get_last_error())
    return os.fdopen(msvcrt.open_osfhandle(handle,os.O_RDONLY|os.O_BINARY),'rb')


_PATH_OPEN=Path.open


class DebugTensor:
    def numel(self): return 8


def debug_state(epoch=3):
    logging={k:[.5]*epoch for k in durable.LOGGERS}
    logging['epoch_start_timestamps']=list(range(epoch))
    logging['epoch_end_timestamps']=[v+.25 for v in range(epoch)]
    return dict(current_epoch=epoch, logging=logging, network_weights={'DEBUG.weight':DebugTensor()},
        optimizer_state={'state':{0:{'DEBUG.momentum':1}},'param_groups':[{'params':[0],'lr':.01}]},
        grad_scaler_state={'scale':65536.,'growth_factor':2.,'backoff_factor':.5,'growth_interval':2000,'_growth_tracker':3},
        trainer_name='DEBUG_NATIVE',init_args={'DEBUG':True},
        _best_ema=.5,inference_allowed_mirroring_axes=(0,1,2))


def debug_torch(callback=None, epoch=3):
    def load(stream, **kwargs):
        if callback: callback(stream)
        stream.read()
        return debug_state(epoch)
    return SimpleNamespace(load=load,cuda=SimpleNamespace(is_initialized=lambda:False))


class DurableCheckpointTests(unittest.TestCase):
    def test_unfinished_best_is_whole_original_training_state_with_latest_epoch_separate(self):
        with tempfile.TemporaryDirectory(prefix='DEBUG_DURABLE_',dir=Path(__file__).parent) as folder:
            root=Path(folder); native=dict(root=str(root),native_root=str(root/'native'),
                baseline={'dataset_name':'DEBUG_DATA','plans_name':'DEBUG_PLANS'},trainer='DEBUG_NATIVE')
            goal=durable.fold(native); goal.mkdir(parents=True)
            best=goal/'checkpoint_best.pth'; best.write_bytes(b'DEBUG_NATIVE_FULL_BEST')
            latest=goal/'checkpoint_latest.pth'; latest.write_bytes(b'DEBUG_NATIVE_FULL_LATEST')
            with patch.dict('sys.modules',torch=debug_torch(epoch=38)):
                selected,role,extra=durable.select_checkpoint(native,unfinished='best')
            self.assertEqual(selected,best); self.assertEqual(role,'OWN_BEST_TRAINING_STATE')
            self.assertEqual(extra['latest_on_tmpfs_N'],38)
            self.assertFalse(extra['native_evaluation_complete']); self.assertFalse(extra['complete_latest_backup'])
            output=root/'cache'; output.mkdir(); cache=durable.Cache(output,10**12)
            with patch.dict('sys.modules',torch=debug_torch(epoch=17)),patch.object(cache,'guard'),patch.object(durable,'fsync_directory'):
                proof=cache.copy(best,output/'CURRENT.pth',trainer='DEBUG_NATIVE',role=role,extra=extra)
                self.assertEqual(proof['saved_resume_epoch'],17)
                self.assertEqual(proof['epochs_may_repeat_after_tmpfs_loss'],21)
                self.assertEqual((output/'CURRENT.pth').read_bytes(),best.read_bytes())
                newer=dict(proof,latest_on_tmpfs_N=39,epochs_may_repeat_after_tmpfs_loss=22)
                durable.publish(output/'CURRENT.json',newer); cache.recover_pending(output)
            self.assertEqual(durable.read(output/'CURRENT.json')['latest_on_tmpfs_N'],39)
            (goal/'checkpoint_final.pth').write_bytes(b'DEBUG_FINAL_PENDING26')
            with patch.dict('sys.modules',torch=debug_torch(epoch=39)):
                pending,role,pending_meta=durable.select_checkpoint(native,unfinished='best')
            self.assertEqual(pending,best); self.assertEqual(role,'OWN_BEST_TRAINING_STATE')
            self.assertFalse(pending_meta['native_evaluation_complete'])
            with patch.dict('sys.modules',torch=debug_torch(epoch=39)):
                selected,role,extra=durable.select_checkpoint(native,unfinished='latest')
            self.assertEqual(selected,latest); self.assertEqual(role,'TRAINING_LATEST_PENDING_BEST26')

    def test_candidate_bank_preserves_only_published_npz_and_keeps_empty_prefix_partial(self):
        import numpy as np
        from tests.test_basic_cp_large_bank import source_fixture
        from hiercp_v1x.basic_cp_ablation_bank import verify_preservation
        with tempfile.TemporaryDirectory(prefix='DEBUG_DURABLE_',dir=Path(__file__).parent) as folder:
            root=Path(folder); source=root/'DEBUG_bank'; original=root/'DEBUG_original'
            relative=Path('entries/DEBUG_source.npz'); base=original/relative; base.parent.mkdir(parents=True)
            old,new=source_fixture(128),source_fixture(512)
            np.savez(base,**old); (original/'index.json').write_text('{}',encoding='utf8')
            entry=source/relative; entry.parent.mkdir(parents=True); np.savez(entry,**new)
            np.savez(entry.parent/'DEBUG_UNPUBLISHED.npz',**new)
            receipt=source/'entry_receipts'/relative.with_suffix('.json'); receipt.parent.mkdir(parents=True)
            row=dict(entry=relative.as_posix(),original_entry_sha256=durable.digest(base),
                new_entry_sha256=durable.digest(entry),proof=verify_preservation(old,new,original_count=128,candidate_count=512))
            receipt.write_text(json.dumps(row),encoding='utf8')
            output=root/'cache'; output.mkdir(); cache=durable.Cache(output,10**12)
            with patch.object(cache,'guard'),patch.object(durable,'fsync_directory'), \
                 patch.object(durable,'exact',side_effect=lambda path,**kw:Path(path)):
                durable.copy_bank(cache,source,512,original/'index.json',{relative.as_posix():durable.digest(base)})
                empty=root/'DEBUG_EMPTY_1024'; (empty/'entry_receipts').mkdir(parents=True)
                durable.copy_bank(cache,empty,1024,original/'index.json',{relative.as_posix():durable.digest(base)})
            self.assertEqual((output/'banks/512'/relative).read_bytes(),entry.read_bytes())
            self.assertEqual((output/'banks/512/entry_receipts'/relative.with_suffix('.json')).read_bytes(),receipt.read_bytes())
            self.assertFalse((output/'banks/512/entries/DEBUG_UNPUBLISHED.npz').exists())
            saved=durable.read(output/'banks/512/durable_receipt.json')
            self.assertEqual(saved['source_entries'],1); self.assertFalse(saved['complete'])
            self.assertEqual(durable.read(output/'banks/1024/durable_receipt.json')['source_entries'],0)

    def test_same_count_bank_indices_keep_distinct_native_identity_namespaces(self):
        # DEBUG published partial fixture tests identity collision only. It is
        # explicitly not an admitted full642 clinical or geometry bank.
        import numpy as np
        from tests.test_basic_cp_large_bank import source_fixture
        from hiercp_v1x.basic_cp_ablation_bank import verify_preservation
        with tempfile.TemporaryDirectory(prefix='DEBUG_DURABLE_',dir=Path(__file__).parent) as folder:
            root=Path(folder); original=root/'DEBUG_original'; relative=Path('entries/DEBUG.npz')
            base=original/relative; base.parent.mkdir(parents=True); old=source_fixture(128); extended=source_fixture(512)
            np.savez(base,**old); (original/'index.json').write_text('{}',encoding='utf8')
            sources=[]
            for label in ('DEBUG_C','DEBUG_V3'):
                source=root/label; entry=source/relative; entry.parent.mkdir(parents=True); np.savez(entry,**extended)
                (source/'index.json').write_text(json.dumps({'DEBUG_ONLY':True,'generation_root':label}),encoding='utf8')
                receipt=source/'entry_receipts'/relative.with_suffix('.json'); receipt.parent.mkdir(parents=True)
                row=dict(entry=relative.as_posix(),original_entry_sha256=durable.digest(base),
                    new_entry_sha256=durable.digest(entry),proof=verify_preservation(old,extended,original_count=128,candidate_count=512))
                receipt.write_text(json.dumps(row),encoding='utf8'); sources.append(source)
            output=root/'cache'; output.mkdir(); cache=durable.Cache(output,10**12)
            with patch.object(cache,'guard'),patch.object(durable,'fsync_directory'), \
                 patch.object(durable,'exact',side_effect=lambda p,**kw:Path(p)):
                durable.copy_bank(cache,sources[1],512,original/'index.json',{relative.as_posix():durable.digest(base)})
                durable.copy_bank(cache,sources[0],512,original/'index.json',{relative.as_posix():durable.digest(base)},
                                  destination_name='native_GPU3_512')
            primary=durable.read(output/'banks/512/durable_receipt.json')
            native=durable.read(output/'banks/native_GPU3_512/durable_receipt.json')
            self.assertNotEqual(primary['source_index_sha256'],native['source_index_sha256'])
            self.assertEqual(primary['source'],str(sources[1])); self.assertEqual(native['source'],str(sources[0]))
            self.assertFalse(primary['complete']); self.assertFalse(native['complete'])
            self.assertEqual((output/'banks/512'/relative).read_bytes(),(output/'banks/native_GPU3_512'/relative).read_bytes())

    def test_full_epoch_model_optimizer_scaler_logger_validation_fails_closed(self):
        state=debug_state(); result=durable.checkpoint_info(state,'DEBUG_NATIVE')
        self.assertEqual(result['checkpoint_completed_epoch'],3)
        self.assertEqual(result['optimizer_states'],1)
        for key in durable.LOGGERS:
            bad=copy.deepcopy(state); bad['logging'][key].pop()
            with self.assertRaisesRegex(ValueError,'eight native logger'): durable.checkpoint_info(bad,'DEBUG_NATIVE')
        for field in ('network_weights','optimizer_state','grad_scaler_state'):
            bad=copy.deepcopy(state); bad[field]={}
            with self.assertRaises(ValueError): durable.checkpoint_info(bad,'DEBUG_NATIVE')
        for end in (float('nan'),float('inf'),-1):
            bad=copy.deepcopy(state); bad['logging']['epoch_end_timestamps'][-1]=end
            with self.assertRaisesRegex(ValueError,'finite ordered'): durable.checkpoint_info(bad,'DEBUG_NATIVE')
        with self.assertRaisesRegex(ValueError,'trainer'): durable.checkpoint_info(state,'WRONG_TRAINER')

    def test_held_source_fd_survives_name_advancement_and_copy_is_fsynced(self):
        with tempfile.TemporaryDirectory(prefix='DEBUG_DURABLE_',dir=Path(__file__).parent) as folder:
            root=Path(folder); source=root/'DEBUG_source.pth'; source.write_bytes(b'DEBUG_OLD_INODE')
            replacement=root/'DEBUG_replacement.pth'; replacement.write_bytes(b'DEBUG_NEW_INODE')
            output=root/'cache'; output.mkdir(); cache=durable.Cache(output,10**12); target=output/'GPU0/CURRENT.pth'
            def advance(stream):
                self.assertEqual(stream.read(),b'DEBUG_OLD_INODE')
                # Linux exercises actual atomic overwrite. Windows MoveFileEx
                # rejects an open destination, so DEBUG fault injection first
                # renames that held inode; both tests retain the same read FD.
                if os.name=='nt': source.rename(root/'DEBUG_retired_inode.pth')
                replacement.replace(source)
            with patch.dict('sys.modules',torch=debug_torch(advance)), patch.object(cache,'guard'), patch.object(Path,'open',debug_shared_read_open), \
                 patch.object(durable,'fsync_directory') as directory_sync, patch.object(durable.os,'fsync',wraps=durable.os.fsync) as sync:
                result=cache.copy(source,target,trainer='DEBUG_NATIVE',role='TRAINING_LATEST')
            self.assertEqual(target.read_bytes(),b'DEBUG_OLD_INODE')
            self.assertEqual(source.read_bytes(),b'DEBUG_NEW_INODE')
            self.assertEqual(result['sha256'],hashlib.sha256(b'DEBUG_OLD_INODE').hexdigest())
            self.assertTrue(result['source_named_path_advanced']); self.assertGreater(sync.call_count,0)
            self.assertGreater(directory_sync.call_count,0)
            self.assertEqual(durable.read(target.parent/'CURRENT.json')['sha256'],result['sha256'])

    def test_disk_budget_and_interruption_preserve_old_current_and_recover_pending(self):
        with tempfile.TemporaryDirectory(prefix='DEBUG_DURABLE_',dir=Path(__file__).parent) as folder:
            root=Path(folder); cache=durable.Cache(root,10**12)
            target=root/'GPU0/CURRENT.pth'; target.parent.mkdir(); target.write_bytes(b'DEBUG_VALID_OLD')
            source=root/'DEBUG_new.pth'; source.write_bytes(b'DEBUG_VALID_NEW')
            with patch('psutil.Process',return_value=SimpleNamespace(memory_info=lambda:SimpleNamespace(rss=1))), \
                 patch.object(durable.shutil,'disk_usage',return_value=SimpleNamespace(free=durable.GIB)):
                with self.assertRaisesRegex(ValueError,'1GiB free'): cache.copy(source,target)
            self.assertEqual(target.read_bytes(),b'DEBUG_VALID_OLD')
            count=[0]
            def interrupt(*args):
                count[0]+=1
                if count[0]==3: raise ValueError('DEBUG interrupted during block copy')
            with patch.object(cache,'guard',side_effect=interrupt),patch.object(durable,'fsync_directory'):
                with self.assertRaisesRegex(ValueError,'DEBUG interrupted'): cache.copy(source,target)
            self.assertEqual(target.read_bytes(),b'DEBUG_VALID_OLD')
            self.assertEqual(list(target.parent.glob('.CURRENT*')),[])
            pending={'format':durable.FORMAT,'sha256':durable.digest(target),'role':'TRAINING_LATEST'}
            (target.parent/'PENDING.json').write_text(json.dumps(pending),encoding='utf8')
            with patch.object(durable,'fsync_directory'): cache.recover_pending(target.parent)
            self.assertEqual(durable.read(target.parent/'CURRENT.json'),pending)
            target.write_bytes(b'DEBUG_CORRUPT')
            with self.assertRaisesRegex(ValueError,'metadata mismatch'): cache.recover_pending(target.parent)

    def test_final_and_best_require_full250_actual_best26_without_relabeling_best_epoch(self):
        from hiercp_v1x import v24_native_best_eval_runtime as best_runtime
        with tempfile.TemporaryDirectory(prefix='DEBUG_DURABLE_',dir=Path(__file__).parent) as folder:
            root=Path(folder); native=dict(root=str(root),native_root=str(root/'native'),
                baseline={'dataset_name':'DEBUG_DATA','plans_name':'DEBUG_PLANS'},trainer='DEBUG_NATIVE',
                split={'outer_val':[f'DEBUG_{i}' for i in range(26)]},bank_sha256='DEBUG_BANK')
            goal=durable.fold(native); goal.mkdir(parents=True)
            final=goal/'checkpoint_final.pth'; final.write_bytes(b'DEBUG_FINAL250')
            best=goal/'checkpoint_best.pth'; best.write_bytes(b'DEBUG_BEST_EARLIER')
            selected,role,extra=durable.select_checkpoint(native)
            self.assertEqual((selected,role),(goal/'checkpoint_latest.pth','TRAINING_LATEST_PENDING_BEST26'))
            output=root/'cache'; output.mkdir(); cache=durable.Cache(output,10**12)
            with patch.dict('sys.modules',torch=debug_torch(epoch=249)),patch.object(cache,'guard'),patch.object(durable,'fsync_directory'):
                with self.assertRaisesRegex(ValueError,'FINAL full250'): cache.copy(final,output/'CURRENT.pth',
                    trainer='DEBUG_NATIVE',role='DEBUG_FINAL_GUARD',extra={'required_epoch':250})
            self.assertFalse((output/'CURRENT.pth').exists())
            proof=dict(epochs=250,checkpoint='checkpoint_best.pth',fold=str(goal),evaluation_checkpoint_is_final=False,
                final_training_witness_sha256=durable.digest(final),best_sha256=durable.digest(best))
            (root/'training_and_BEST26_complete.json').write_text(json.dumps(proof),encoding='utf8')
            with patch.dict('sys.modules',torch=debug_torch(epoch=249)),patch.object(best_runtime,'validate_best_validation_receipt') as validate:
                with self.assertRaisesRegex(ValueError,'Native FINAL250'): durable.select_checkpoint(native)
                validate.assert_not_called()
            with patch.dict('sys.modules',torch=debug_torch(epoch=250)),patch.object(best_runtime,'validate_best_validation_receipt',side_effect=ValueError('Actual BEST26 invalid')):
                with self.assertRaisesRegex(ValueError,'Actual BEST26'): durable.select_checkpoint(native)
            with patch.dict('sys.modules',torch=debug_torch(epoch=250)),patch.object(best_runtime,'validate_best_validation_receipt',return_value={'checkpoint_sha256':durable.digest(best)}) as validate:
                selected,role,extra=durable.select_checkpoint(native)
                self.assertEqual((selected,role),(best,'BEST_EVALUATED26')); validate.assert_called_once()
                self.assertEqual(len(validate.call_args.kwargs['expected_cases']),26)
                with patch.dict('sys.modules',torch=debug_torch(epoch=17)),patch.object(cache,'guard'),patch.object(durable,'fsync_directory'):
                    result=cache.copy(best,output/'CURRENT.pth',expected=extra.pop('expected_sha256'),
                        trainer='DEBUG_NATIVE',role=role,extra=extra)
            self.assertEqual(result['checkpoint_completed_epoch'],17)
            self.assertEqual(result['training_completed_epochs'],250)
            self.assertEqual(result['evaluated_cases'],26)
            self.assertFalse(result['checkpoint_is_training_resume'])


if __name__=='__main__': unittest.main()
