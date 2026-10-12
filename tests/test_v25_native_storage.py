"""DEBUG CPU fixtures for byte admission; no V25 training/evaluation claim.

Small explicit fixture files exercise exact131 ownership and serialized-byte
arithmetic. The separate genuine-checkpoint inspector rejects these fixtures.
"""
import copy
from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from hiercp_v1x import v25_native_storage as storage
from hiercp_v1x import v24_readonly_native_storage as core
from tests import test_v24_readonly_native_storage as readonly_unit


TRAINER_SOURCE = '''import os
import torch
class nnUNetTrainer:
    def save_checkpoint(self, filename):
        checkpoint = {'DEBUG_CPU_FIXTURE_ONLY': True}
        torch.save(checkpoint, filename)
    def on_train_end(self):
        self.save_checkpoint(os.path.join(self.output_folder, 'checkpoint_final.pth'))
        os.remove(os.path.join(self.output_folder, 'checkpoint_latest.pth'))
    def on_epoch_end(self):
        self.save_checkpoint(os.path.join(self.output_folder, 'checkpoint_latest.pth'))
        self.save_checkpoint(os.path.join(self.output_folder, 'checkpoint_best.pth'))
'''


@contextmanager
def unlocked_fixture(path):
    """Windows UNIT substitute only; production lock is Linux flock."""
    yield


class V25StorageDEBUGUnit(unittest.TestCase):
    def setUp(self):
        fixture = readonly_unit.ReadonlyNativeStorageUnit()
        fixture.setUp(); self.addCleanup(fixture.doCleanups)
        self.fixture = fixture; self.root = fixture.root
        self.group = self.root/'V25_DEBUG_UNIT_group'; self.group.mkdir()
        self.arms = {arm: self.group/arm for arm in storage.ARMS}
        for path in self.arms.values(): path.mkdir()
        trainer = fixture.package/'training/nnUNetTrainer/nnUNetTrainer.py'
        trainer.parent.mkdir(parents=True, exist_ok=True)
        trainer.write_text(TRAINER_SOURCE, encoding='utf8')
        self.best = self.root/'measured_native/checkpoint_best.pth'
        self.best.parent.mkdir(); self.best.write_bytes(b'N'*37)
        self.final = self.best.with_name('checkpoint_final.pth'); self.final.write_bytes(b'F'*37)
        self.gnn = self.root/'UNIT_not_real_gnn.pt'; self.gnn.write_bytes(b'G'*19)
        self.probe = dict(format=storage.GNN_PROBE_FORMAT, file=core.proof(self.gnn),
            actual_serialization=True, full_optimizer_state=True, optimizer_parameters=537,
            model_parameter_count=13331792, serialized_checkpoint_bytes=19,
            observed_training_format='DEBUG_CPU_FIXTURE_ONLY', actual_v25_checkpoint=False,
            training_completion_claim=False)
        self.patch_size = patch.object(storage, 'NATIVE_CHECKPOINT_BYTES', 37)
        self.patch_size.start(); self.addCleanup(self.patch_size.stop)
        self.header = patch.object(core, '_open_header', lambda path: dict(
            shape=list(np.load(path, allow_pickle=False).shape), dtype=np.load(path, allow_pickle=False).dtype.str))
        self.header.start(); self.addCleanup(self.header.stop)
        self.free = patch.object(storage.shutil, 'disk_usage', return_value=SimpleNamespace(free=100*2**30))
        self.free.start(); self.addCleanup(self.free.stop)
        self.publication_lock = patch.object(storage, '_publication_lock', unlocked_fixture)
        self.publication_lock.start(); self.addCleanup(self.publication_lock.stop)
        self.arguments = dict(group_root=self.group, arm_roots=self.arms, inventory_path=fixture.inventory,
            baseline_preprocessed=fixture.pre, runtime_package=fixture.package,
            native_checkpoint_paths=[self.best, self.final], gnn_checkpoint_measurement=self.probe,
            additional_writable_bytes={arm: 1000 for arm in storage.ARMS})

    def admission(self):
        return storage.build_admission(**self.arguments)

    def test_four_complete_arms_and_global9_exact_byte_arithmetic(self):
        document = self.admission(); budget = document['budget']
        self.assertEqual(set(document['arm_roots']), set('ABCD'))
        self.assertEqual(len(document['preprocessed']['cases']), 131)
        self.assertEqual(len(document['preprocessed']['files']), 393)
        self.assertEqual(budget['measured_components']['native_global9_checkpoint_bytes'], 9*37)
        self.assertEqual(budget['measured_components']['gnn_global9_checkpoint_bytes'], 9*19)
        self.assertEqual(budget['required_free_bytes'], budget['new_writable_bytes_estimate']+3*2**30)
        self.assertFalse(budget['canonical_tmpfs_included']); self.assertEqual(budget['readonly_dataset_copies'], 0)
        self.assertFalse(document['gnn_checkpoint_measurement']['actual_v25_checkpoint'])

    def test_three_arm_legacy_and_duplicate_root_rejected(self):
        self.arguments['arm_roots'] = {key: value for key, value in self.arms.items() if key != 'D'}
        with self.assertRaisesRegex(ValueError, 'four'): self.admission()
        self.arguments['arm_roots'] = dict(self.arms, D=self.arms['A'])
        with self.assertRaisesRegex(ValueError, 'distinct'): self.admission()

    def test_full131_coverage_and_no_unpacked_duplicate(self):
        self.fixture.data.joinpath(self.fixture.val[-1]+'.pkl').unlink()
        with self.assertRaisesRegex(ValueError, 'full131'): self.admission()
        self.fixture.data.joinpath(self.fixture.val[-1]+'.pkl').write_bytes(b'{}')
        self.fixture.data.joinpath('unpacked.npy').write_bytes(b'DEBUG')
        with self.assertRaisesRegex(ValueError, 'duplicate'): self.admission()

    def test_complete_native_best_and_final_size_witnesses_required(self):
        self.arguments['native_checkpoint_paths'] = [self.best]
        with self.assertRaisesRegex(ValueError, 'BEST/final'): self.admission()
        self.arguments['native_checkpoint_paths'] = [self.best, self.final]
        self.final.write_bytes(b'F'*38)
        with self.assertRaisesRegex(ValueError, '819237798'): self.admission()

    def test_empty_optimizer_or_initial_only_measurement_not_admitted(self):
        self.arguments['gnn_checkpoint_measurement'] = dict(self.probe, full_optimizer_state=False)
        with self.assertRaisesRegex(ValueError, 'full optimizer'): self.admission()
        self.arguments['gnn_checkpoint_measurement'] = dict(self.probe, actual_serialization=False)
        with self.assertRaisesRegex(ValueError, 'serialization'): self.admission()

    def test_real_cpu_checkpoint_inspector_rejects_unpopulated_optimizer(self):
        import torch
        path = self.root/'UNIT_torch_initial_only.pt'
        torch.save(dict(model={'tiny': torch.ones(1)}, optimizer={'state': {}, 'param_groups': []}), path)
        with self.assertRaisesRegex(ValueError, '537'): storage.measure_gnn_checkpoint(path)

    def test_insufficient_whole_group_space_creates_no_lock_or_claim(self):
        with patch.object(storage.shutil, 'disk_usage', return_value=SimpleNamespace(free=2**30)):
            with self.assertRaisesRegex(OSError, 'four complete'): self.admission()
        self.assertEqual(list(self.group.glob('*.lock')), [])

    def test_admission_exclusive_and_original_baseline_bytes_unchanged(self):
        baseline = {path: core.sha(path) for path in self.fixture.pre.rglob('*') if path.is_file()}
        document = self.admission(); path = self.group/'storage_admission.json'
        proof = storage.seal_admission(path, document)
        self.assertEqual(core.sha(path), proof['sha256'])
        with self.assertRaises(FileExistsError): storage.seal_admission(path, document)
        self.assertEqual(baseline, {path: core.sha(path) for path in baseline})
        self.assertEqual(list(self.group.glob('*.tmp')), [])

    def test_static_data_same_size_mutation_and_inventory_rebind_fail(self):
        document = self.admission(); path = self.fixture.data/(self.fixture.train[0]+'.pkl')
        value = path.read_bytes(); path.write_bytes(b'X'*len(value))
        with self.assertRaises(ValueError): storage.verify_admission(document)

    def test_fabricated_global7_or_reserve_discount_rejected(self):
        document = self.admission()
        for key, value in (('native_checkpoint_slots', 7), ('gnn_checkpoint_slots', 12),
                           ('checkpoint_reserve_bytes', 0), ('readonly_dataset_copies', 4)):
            changed = copy.deepcopy(document); changed['budget'][key] = value
            with self.assertRaisesRegex(ValueError, 'global9'): storage.verify_admission(changed)

    def test_actual_written_space_credited_only_within_reserved_roots(self):
        document = self.admission()
        before = storage.check_aggregate_disk(document, self.arms['A'])
        self.arms['A'].joinpath('UNIT_own_log').write_bytes(b'X'*8192)
        after = storage.check_aggregate_disk(document, self.arms['A'])
        self.assertGreater(after['observed_own_output_bytes'], before['observed_own_output_bytes'])
        self.assertLess(after['required_free_bytes'], before['required_free_bytes'])
        with self.assertRaisesRegex(ValueError, 'reserved arm'):
            storage.check_aggregate_disk(document, self.root)

    def test_sequential_wave_blocks_second_arm_and_releases_normal_finish(self):
        document = self.admission()
        with storage.sequential_wave_lock(document, 'A') as witness:
            self.assertTrue(Path(witness['path']).is_file())
            with self.assertRaises(FileExistsError):
                with storage.sequential_wave_lock(document, 'B'): pass
        self.assertFalse(self.group.joinpath('sequential_native_wave.lease.json').exists())
        self.assertEqual(len(list(self.group.glob('sequential_native_wave_finished_*.json'))), 1)
        with storage.sequential_wave_lock(document, 'B'): pass

    def test_failed_wave_preserves_lease_and_blocks_next_launch(self):
        document = self.admission()
        with self.assertRaisesRegex(RuntimeError, 'UNIT failed'):
            with storage.sequential_wave_lock(document, 'A'):
                raise RuntimeError('UNIT failed stage; child state uncertain')
        self.assertTrue(self.group.joinpath('sequential_native_wave.lease.json').exists())
        with self.assertRaises(FileExistsError):
            with storage.sequential_wave_lock(document, 'B'): pass

    def test_gnn_wrap_preserves_atomic_function_and_rejects_cross_arm(self):
        from hiercp_v1x import u_bridge_training
        document = self.admission(); original = u_bridge_training.atomic_save
        import types
        module = types.ModuleType('DEBUG_UNIT_training'); module.atomic_save = original
        proof = storage.install_gnn_checkpoint_adapter(module, document, arm='A')
        self.assertTrue(proof['original_atomic_save_code_preserved'])
        self.assertIs(module.atomic_save.__wrapped__, original)
        self.assertIs(u_bridge_training.atomic_save, original)
        path = self.arms['A']/'training'; path.mkdir()
        with patch.object(storage, '_publication_lock', unlocked_fixture):
            module.atomic_save(path/'checkpoint_latest.pt', {'DEBUG_CPU_FIXTURE_ONLY': True})
        self.assertTrue((path/'checkpoint_latest.pt').is_file())
        with self.assertRaisesRegex(ValueError, 'own reserved'):
            module.atomic_save(self.arms['B']/'training/checkpoint_latest.pt', {})
        with self.assertRaisesRegex(ValueError, 'bound differently'):
            storage.install_gnn_checkpoint_adapter(module, document, arm='B')

    def test_failed_gnn_temp_preserved_without_downstream_publication(self):
        import types
        from hiercp_v1x.u_bridge_training import atomic_save
        document = self.admission(); module = types.ModuleType('DEBUG_failed_GNN'); module.atomic_save = atomic_save
        storage.install_gnn_checkpoint_adapter(module, document, arm='A')
        path = self.arms['A']/'training'; path.mkdir()
        temporary = path/'checkpoint_latest.pt.UNIT.tmp'; temporary.write_bytes(b'preserve')
        with patch.object(storage, '_publication_lock', unlocked_fixture):
            with self.assertRaisesRegex(RuntimeError, 'explicit recovery'):
                module.atomic_save(path/'checkpoint_latest.pt', {})
        self.assertEqual(temporary.read_bytes(), b'preserve')
        self.assertFalse((path/'checkpoint_latest.pt').exists())

    def test_private_native_publication_keeps_original_methods_and_final_order(self):
        document = self.admission(); arm = self.arms['A']; private = arm/'native/runtime'
        source = document['runtime_files'][storage.original_checkpoint.CHECKPOINT_SOURCE]
        target = private/'nnunetv2'/storage.original_checkpoint.CHECKPOINT_SOURCE
        core.copy_private(source, target)
        spec = importlib.util.spec_from_file_location('DEBUG_UNIT_private_native', target)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        cls = module.nnUNetTrainer; original = cls.on_train_end
        untouched = (cls.save_checkpoint, cls.on_epoch_end)
        proof = storage.install_checkpoint_adapters(document, private, cls)
        self.assertEqual(proof['global_peak_checkpoint_slots'], 9)
        self.assertIs(cls.on_train_end.__wrapped__, original)
        self.assertEqual((cls.save_checkpoint, cls.on_epoch_end), untouched)
        instance = cls(); folder = Path(document['checkpoint_coordination']['folds']['A'])
        folder.mkdir(parents=True); instance.output_folder = str(folder)
        folder.joinpath('checkpoint_latest.pth').write_bytes(b'UNIT_latest')
        with patch.object(storage, '_publication_lock', unlocked_fixture): instance.on_train_end()
        self.assertTrue(folder.joinpath('checkpoint_final.pth').exists())
        self.assertFalse(folder.joinpath('checkpoint_latest.pth').exists())
        with self.assertRaises(ValueError): instance.on_train_end()

    def test_v25_dataset_adapter_original_reader_full131_and_readonly_mode(self):
        document = self.admission(); arm = self.arms['A']
        name = 'training/dataloading/nnunet_dataset.py'
        target = arm/'native/runtime/nnunetv2'/name
        core.copy_private(document['runtime_files'][name], target)
        spec = importlib.util.spec_from_file_location('DEBUG_UNIT_v25_private_dataset', target)
        module = importlib.util.module_from_spec(spec)
        module.blosc2 = readonly_unit.UnitBlosc
        spec.loader.exec_module(module)
        cls = module.nnUNetDatasetBlosc2
        module.infer_dataset_class = lambda folder: cls
        trainer_module = SimpleNamespace(infer_dataset_class=module.infer_dataset_class)
        folder = arm/'native/nnUNet_preprocessed'/self.fixture.pre.name/'UNIT_fullres'
        folder.mkdir(parents=True)
        old = storage._DATASET_ADAPTER
        self.addCleanup(setattr, storage, '_DATASET_ADAPTER', old)
        storage._DATASET_ADAPTER = None
        original_reader = cls.load_case
        proof = storage.install_dataset_adapter(document, folder, dataset_module=module, trainer_module=trainer_module)
        self.assertTrue(proof['original_load_case_code_preserved'])
        self.assertEqual(proof['full_preprocessed_cases'], 131)
        dataset = cls(str(folder))
        self.assertEqual(set(dataset.identifiers), set(self.fixture.train+self.fixture.val))
        readonly_unit.UnitBlosc.calls.clear()
        data, seg, previous, properties = dataset.load_case(self.fixture.val[-1])
        self.assertIsNone(previous)
        self.assertFalse(data.flags.writeable); self.assertFalse(seg.flags.writeable)
        self.assertTrue(all(mode == 'r' for path, mode in readonly_unit.UnitBlosc.calls))
        self.assertEqual(list(folder.iterdir()), [])
        self.assertEqual(original_reader.__code__, storage._DATASET_ADAPTER.__globals__['_DATASET_ORIGINALS']['load'].__code__)
        self.assertIsNone(cls.unpack_dataset(str(folder)))

    def interrupted_wave(self, *, arm='A'):
        document = self.admission()
        old = dict(format='v25_exclusive_downstream_wave_v1', arm=arm, pid=777,
            create_time=123., command=['DEBUG_CPU_fixture', '--native'],
            UID=storage.os.getuid() if hasattr(storage.os, 'getuid') else None,
            admission_sha256=storage.canonical_hash(document), arm_root=document['arm_roots'][arm],
            request_sha256='a'*64)
        path = self.group/'sequential_native_wave.lease.json'; core.publish_json(path, old)
        child = dict(pid=778, create_time=124., command=['DEBUG_CPU_child', '--train'],
            UID=old['UID'], parent_actor={key: old[key] for key in ('pid','create_time','command','UID')},
            state='RUNNING', request_sha256='a'*64, log_path=str(self.arms[arm]/'native/train.log'))
        child_path = self.arms[arm]/'native_child.json'; core.publish_json(child_path, child)
        return document, old, path, child_path

    def test_released_same_arm_recovery_archives_original_and_reclaims_exclusively(self):
        import psutil
        document, old, path, child = self.interrupted_wave(); original = path.read_bytes()
        real_process = psutil.Process
        def process(pid=None):
            if pid in (777, 778): raise psutil.NoSuchProcess(pid)
            return real_process(pid)
        with patch.object(psutil, 'Process', side_effect=process):
            with storage.sequential_wave_lock(document, 'A', request_sha256='b'*64,
                    recover_released_same_arm=True, predecessor_child=child):
                self.assertNotEqual(path.read_bytes(), original)
        archives = list(self.group.glob('sequential_native_wave_released_*.json'))
        self.assertEqual(len(archives), 1)
        self.assertEqual(core.read(archives[0])['original_lease'], old)
        self.assertFalse(core.read(archives[0])['signals_sent'])

    def test_recovery_live_parent_or_orphan_child_preserves_original_lease(self):
        document, old, path, child = self.interrupted_wave(); original = path.read_bytes()
        for releases in ((False,), (True, False)):
            with patch.object(storage, '_released', side_effect=releases):
                with self.assertRaisesRegex(RuntimeError, 'alive'):
                    with storage.sequential_wave_lock(document, 'A', recover_released_same_arm=True,
                                                       predecessor_child=child): pass
            self.assertEqual(path.read_bytes(), original)
        self.assertEqual(list(self.group.glob('sequential_native_wave_released_*.json')), [])

    def test_recovery_wrong_arm_and_wrong_child_parent_refused(self):
        document, old, path, child = self.interrupted_wave(); original = path.read_bytes()
        with self.assertRaisesRegex(ValueError, 'same-arm'):
            with storage.sequential_wave_lock(document, 'B', recover_released_same_arm=True,
                                               predecessor_child=child): pass
        changed = core.read(child); changed['parent_actor']['create_time'] += 1
        child.write_text(json.dumps(changed), encoding='utf8')
        with patch.object(storage, '_released', return_value=True):
            with self.assertRaisesRegex(ValueError, 'old parent'):
                with storage.sequential_wave_lock(document, 'A', recover_released_same_arm=True,
                                                   predecessor_child=child): pass
        self.assertEqual(path.read_bytes(), original)

    def test_release_detector_reused_pid_zombie_and_same_birth_argv(self):
        import psutil
        actor = dict(pid=777,create_time=123.,command=['DEBUG_CPU'],UID=42)
        process = SimpleNamespace(create_time=lambda: 124.)
        with patch.object(psutil, 'Process', return_value=process): self.assertTrue(storage._released(actor))
        process = SimpleNamespace(create_time=lambda: 123., uids=lambda:SimpleNamespace(real=42),
            cmdline=lambda:['DEBUG_CPU'], status=lambda:psutil.STATUS_ZOMBIE)
        with patch.object(psutil, 'Process', return_value=process): self.assertTrue(storage._released(actor))
        process.status=lambda:psutil.STATUS_RUNNING
        with patch.object(psutil, 'Process', return_value=process): self.assertFalse(storage._released(actor))
        process.cmdline=lambda:['UNKNOWN_replaced_process']
        with patch.object(psutil, 'Process', return_value=process):
            with self.assertRaisesRegex(ValueError, 'UID/argv'): storage._released(actor)

    def test_pre_child_status_recovery_only_for_actual_admission_or_prepare_stage(self):
        document, old, path, child = self.interrupted_wave()
        child.unlink()  # Test-owned fixture was never an actual child.
        root = self.arms['A']/'downstream'; root.mkdir()
        status_path = root/'status.json'
        status = dict(format='v25_full_own_arm_downstream_status_v1', arm='A',
            actor={key:old[key] for key in ('pid','create_time','command','UID')},
            request_sha256='a'*64, stage='train', state='FAILED')
        core.publish_json(status_path, status)
        with patch.object(storage, '_released', return_value=True):
            with self.assertRaisesRegex(ValueError, 'no native subprocess'):
                with storage.sequential_wave_lock(document, 'A', request_sha256='a'*64,
                        recover_released_same_arm=True, predecessor_status=status_path): pass
            status['stage'] = 'prepare-native'
            status_path.write_text(json.dumps(status), encoding='utf8')
            with storage.sequential_wave_lock(document, 'A', request_sha256='a'*64,
                    recover_released_same_arm=True, predecessor_status=status_path): pass


if __name__ == '__main__':
    unittest.main()
