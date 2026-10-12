import hashlib
import json
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import unittest
from pathlib import Path


from tools.run_v25_parallel_pipeline import (GPU_MAPPING, native_command, stage_command,
    validate_manifest, execute_independent_arm, validate_slots_live)


def manifest(tmp_path):
    slots = {}
    for index, arm in enumerate('ABCD'):
        proof = tmp_path/(arm+'.json')
        proof.write_text(json.dumps({'test': arm}))
        slots[arm] = dict(arm=arm, gpu=GPU_MAPPING[arm], CPU_affinity=list(range(index*6, index*6+6)),
            resource_request_path=str(proof), resource_request_sha256=hashlib.sha256(proof.read_bytes()).hexdigest())
    return dict(format='v25_four_GPU_independent_execution_v1', group_root=str(tmp_path), debug=False, slots=slots)


class ParallelDispatchTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='DEBUG_v25_parallel_dispatch_')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def args(self):
        return SimpleNamespace(**{name: self.root/name for name in ('native_experiment', 'inventory',
            'stunet_checkpoint', 'input_cache', 'storage_admission', 'source_bank', 'baseline_preprocessed',
            'project', 'basic_validation', 'evaluator_root')}, group=self.root)

    def test_four_distinct_physical_GPUs_and_24_cores(self):
        value = manifest(self.root)
        self.assertEqual(validate_manifest(value, self.root), value)
        self.assertEqual([value['slots'][arm]['gpu'] for arm in 'ABCD'], [6, 5, 1, 4])

    def test_each_resource_slot_is_live_verified_before_any_stage(self):
        calls=[];value=manifest(self.root)
        validate_slots_live(value,lambda path,**kwargs:calls.append((path,kwargs)))
        self.assertEqual([call[1]['gpu'] for call in calls],[6,5,1,4])
        self.assertTrue(all(call[1]['force'] for call in calls))
        self.assertEqual([call[1]['cpu_affinity'] for call in calls],
            [value['slots'][arm]['CPU_affinity'] for arm in 'ABCD'])

    def test_rejects_serial_or_unbound_dispatch(self):
        for bad in ['GPU', 'CPU', 'arm', 'hash', 'debug', 'missing']:
            with self.subTest(bad=bad):
                value = manifest(self.root)
                if bad == 'GPU': value['slots']['B']['gpu'] = 6
                if bad == 'CPU': value['slots']['B']['CPU_affinity'] = value['slots']['A']['CPU_affinity']
                if bad == 'arm': value['slots']['B']['arm'] = 'A'
                if bad == 'hash': value['slots']['B']['resource_request_sha256'] = '0'*64
                if bad == 'debug': value['debug'] = True
                if bad == 'missing': value['slots'].pop('D')
                with self.assertRaises(ValueError): validate_manifest(value, self.root)

    def test_commands_keep_full_scientific_settings_and_each_own_GPU(self):
        value = manifest(self.root)
        args = SimpleNamespace(**{name: self.root/name for name in ('native_experiment', 'inventory',
            'stunet_checkpoint', 'input_cache', 'storage_admission', 'source_bank', 'baseline_preprocessed',
            'project', 'basic_validation', 'evaluator_root')}, group=self.root)
        for arm in 'ABCD':
            slot = value['slots'][arm]
            command = stage_command(args, slot, 'train')
            self.assertEqual(command[command.index('--gpu')+1], str(GPU_MAPPING[arm]))
            self.assertEqual(command[command.index('--arm')+1], arm)
            self.assertEqual(command[command.index('--input-cache')+1], str(args.input_cache))
            self.assertNotIn('--epochs', command)
            self.assertNotIn('--subset', command)
            native = native_command(args, slot)
            self.assertEqual(native[native.index('--gpu')+1], str(GPU_MAPPING[arm]))
            self.assertEqual(native[native.index('--output')+1], str(self.root/arm))

    def test_common_preparation_and_independent_prefix_have_distinct_flags(self):
        args = self.args();slot = manifest(self.root)['slots']['A']
        shared = stage_command(args, slot, 'prepare', shared_prepare=True, parallel_prepare=True)
        independent = stage_command(args, slot, 'prepare')
        self.assertIn('--shared-preparation', shared)
        self.assertIn('--parallel-prepare', shared)
        self.assertNotIn('--independent-prepare', shared)
        self.assertIn('--independent-prepare', independent)
        self.assertNotIn('--shared-preparation', independent)
        with self.assertRaises(ValueError):stage_command(args, slot, 'train', shared_prepare=True)

    def test_ready_D_trains_while_A_prefix_is_still_preparing(self):
        args = self.args();slots = manifest(self.root)['slots']
        for arm in 'AD':(self.root/arm).mkdir()
        (self.root/'D/preparation.json').write_text('{}')
        blocked = threading.Event();release = threading.Event();D_started = threading.Event()
        observed = []
        def run(slot, stage, command):
            observed.append((slot['arm'], stage))
            if slot['arm'] == 'A' and stage == 'prepare_inputs':
                blocked.set()
                if not release.wait(5):raise RuntimeError('DEBUG synchronization timeout')
            if slot['arm'] == 'D' and stage == 'train':D_started.set()
        with ThreadPoolExecutor(max_workers=2) as pool:
            A = pool.submit(execute_independent_arm, args, slots['A'], run, lambda *args:None)
            self.assertTrue(blocked.wait(2))
            D = pool.submit(execute_independent_arm, args, slots['D'], run, lambda *args:None)
            try:
                self.assertTrue(D_started.wait(2))
                self.assertFalse(A.done())
                D.result(timeout=2)
                self.assertIn(('D', 'native'), observed)
            finally:release.set()
            A.result(timeout=2)
        self.assertEqual([s for a,s in observed if a == 'D'], ['calibrate', 'train', 'scoring', 'native'])
        self.assertEqual([s for a,s in observed if a == 'A'], ['prepare_inputs', 'calibrate', 'train', 'scoring', 'native'])

    def test_failed_A_prefix_does_not_block_other_complete_chains(self):
        args = self.args();slots = manifest(self.root)['slots'];observed = []
        for arm in 'ABCD':
            (self.root/arm).mkdir()
            if arm != 'A':(self.root/arm/'preparation.json').write_text('{}')
        def run(slot, stage, command):
            if slot['arm'] == 'A' and stage == 'prepare_inputs':raise MemoryError('DEBUG full workspace pressure')
            observed.append((slot['arm'], stage))
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {arm:pool.submit(execute_independent_arm, args, slots[arm], run, lambda *args:None)
                for arm in 'ABCD'}
            with self.assertRaises(MemoryError):futures['A'].result(timeout=2)
            for arm in 'BCD':futures[arm].result(timeout=2)
        self.assertEqual({arm for arm,stage in observed if stage == 'native'}, set('BCD'))


if __name__ == '__main__':
    unittest.main()
