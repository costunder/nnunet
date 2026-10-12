import hashlib
import json
import tempfile
from types import SimpleNamespace
import unittest
from pathlib import Path


from tools.run_v25_parallel_pipeline import GPU_MAPPING, native_command, stage_command, validate_manifest


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

    def test_four_distinct_physical_GPUs_and_24_cores(self):
        value = manifest(self.root)
        self.assertEqual(validate_manifest(value, self.root), value)
        self.assertEqual([value['slots'][arm]['gpu'] for arm in 'ABCD'], [6, 5, 1, 4])

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


if __name__ == '__main__':
    unittest.main()
