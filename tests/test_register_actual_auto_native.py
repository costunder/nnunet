"""DEBUG CPU launch/lease fixtures; no actual processes, server or CUDA."""
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

import psutil


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('register_auto_native_DEBUG',
    ROOT / 'outputs/v24_gpu6_recovery_20261011/register_actual_auto_native_NEW.py')
helper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(helper)


class FakeChain:
    REGISTRATION_FORMAT = 'v24_scored_bank_to_native_chain_registration_v1'
    STATUS_FORMAT = 'v24_scored_bank_to_native_chain_status_v1'
    PendingMetadata = RuntimeError

    @staticmethod
    def read(path):
        return json.loads(Path(path).read_bytes())

    @staticmethod
    def sha(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    @staticmethod
    def publish(path, value):
        with Path(path).open('x', encoding='utf8') as stream:
            json.dump(value, stream)


class AutoNativeRegistrationCPU_DEBUG(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory(prefix='register_auto_native_DEBUG_', dir=ROOT / 'outputs')
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.chain = FakeChain()
        self.request = self.root / 'request.json'
        self.chain.publish(self.request, dict(DEBUG=True))
        self.command = ['DEBUG_python_NEVER_RUN', '--request', str(self.request)]
        self.affinity = [0, 1, 2, 3]
        self.actor = Mock(pid=70004)
        self.actor.create_time.return_value = 100.
        self.actor.cmdline.return_value = self.command
        self.actor.uids.return_value = SimpleNamespace(real=41833)
        self.actor.cpu_affinity.return_value = self.affinity
        self.actor.environ.return_value = {'CUDA_VISIBLE_DEVICES': ''}
        self.actor.status.return_value = psutil.STATUS_RUNNING
        self.witness = dict(pid=70004, create_time=100., command=self.command,
            request_sha256=self.chain.sha(self.request))

    def test_new_launch_then_repeat_reuses_exact_actor_without_second_spawn(self):
        def fake_spawn(command, **kwargs):
            self.assertEqual(command, self.command)
            self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'], '')
            self.assertTrue(callable(kwargs['preexec_fn']))
            self.chain.publish(self.root / 'status.json', dict(format=self.chain.STATUS_FORMAT,
                request_sha256=self.chain.sha(self.request), GPU_arm=4, status='WAITING_FOR_FULL642_SCORING',
                supervisor_pid=70004, supervisor_create_time=100.))
            return SimpleNamespace(pid=70004)
        with patch.object(helper.psutil, 'Process', return_value=self.actor), \
                patch.object(helper.subprocess, 'Popen', side_effect=fake_spawn) as spawn:
            first = helper.launch(self.chain, self.root, self.request, self.command, self.affinity,
                native=True, gpu=4, code=str(self.root))
            second = helper.launch(self.chain, self.root, self.request, self.command, self.affinity,
                native=True, gpu=4, code=str(self.root))
        self.assertEqual(spawn.call_count, 1)
        self.assertEqual(first['pid'], second['pid'])
        self.assertFalse(second['signals_sent'])

    def test_controller_lease_recovers_only_matching_live_PID_birth_command(self):
        lease = self.root / 'lease.json'
        self.chain.publish(lease, dict(supervisor_pid=70004, supervisor_create_time=100.,
            command=self.command, request_sha256=self.chain.sha(self.request)))
        intent = dict(command=self.command, request_sha256=self.chain.sha(self.request), created_at=99.)
        actor_path = self.root / 'controller_launch_actor.json'
        with patch.object(helper.psutil, 'Process', return_value=self.actor):
            recovered = helper.recover_actor(self.chain, self.root, intent, self.affinity, actor_path, lease)
        self.assertEqual(recovered, self.witness)
        self.actor.create_time.return_value = 101.
        with patch.object(helper.psutil, 'Process', return_value=self.actor), self.assertRaisesRegex(ValueError, 'binding differs'):
            helper.recover_actor(self.chain, self.root, intent, self.affinity, actor_path, lease)

    def test_unknown_outcome_does_not_spawn_or_create_actor_record(self):
        intent = dict(command=self.command, request_sha256=self.chain.sha(self.request), created_at=99.)
        actor_path = self.root / 'actor.json'
        with patch.object(helper.psutil, 'process_iter', return_value=[]), \
                patch.object(helper.subprocess, 'Popen') as spawn, self.assertRaisesRegex(RuntimeError, 'no second spawn'):
            helper.recover_actor(self.chain, self.root, intent, self.affinity, actor_path, None)
        spawn.assert_not_called()
        self.assertFalse(actor_path.exists())

    def test_dead_or_GPU_visible_actor_requires_explicit_recovery(self):
        with patch.object(helper.psutil, 'Process', side_effect=psutil.NoSuchProcess(70004)), \
                self.assertRaisesRegex(RuntimeError, 'explicit recovery'):
            helper.exact_actor(self.witness, self.affinity)
        self.actor.environ.return_value = {'CUDA_VISIBLE_DEVICES': '4'}
        with patch.object(helper.psutil, 'Process', return_value=self.actor), \
                self.assertRaisesRegex(ValueError, 'CUDA visibility'):
            helper.exact_actor(self.witness, self.affinity)


if __name__ == '__main__':
    unittest.main()
