"""CPU UNIT: C recovery isolation, preserved FAILED monitor and full chain."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from hiercp_v1x import v25_standalone_c_resources as resources
from tools import run_v25_c_continuation as continuation
from tools import run_v25_c_execution as execution


def actor(pid, command=None):
    return dict(pid=pid, create_time=float(pid), UID=41833,
        command=['python', 'CPU_DEBUG_actor.py'] if command is None else command)


def seal_document():
    group = (execution.ROOT/'tmp/CPU_DEBUG_C_admission').resolve()
    old = group/'old_code'; code = group/'C_new_code'; digest = 'a'*64
    value = dict(format=resources.FORMAT, selected_arms=['C'], group_root=str(group),
        private_code=str(code), previous_code=str(old), controller_identity=actor(900),
        previous_controller_identity=actor(800), peer_controller_identity=actor(700),
        previous_seal=dict(path=str(group/'old_seal.json'), sha256=digest),
        peer_seal=dict(path=str(group/'ABD_seal.json'), sha256=digest), slots={},
        preserved_files={str(group/'C/monitor/status.json'): digest},
        runtime_files={'CPU_DEBUG_runtime': digest}, created_at=1.,
        scientific_configuration_unchanged=True, no_process_signaled=True)
    command = ['/home/aicompetition06/.conda/envs/nnunet/bin/python3.10', '-B', '-u',
        str(old/'tools/run_v25_u128_reuse_execution.py'), '--continuation-seal', value['previous_seal']['path'],
        '--mode', 'train', '--arm', 'C', '--gpu', '1', '--output', str(group/'C')]
    value['slots']['C'] = dict(arm='C', gpu=1, CPU_affinity=list(range(16, 22)),
        direct_proof=dict(path=str(group/'C_direct.json'), sha256=digest),
        checkpoint=dict(path=str(group/'C/training/checkpoint_latest.pt'), sha256=digest,
            identity_sha256=digest, content_sha256=digest, epoch=5, updates=68,
            phase='training', train_position=0, evaluation_position=0, active_u=64, status='RUNNING'),
        previous_actor=actor(801, command),
        failed_monitor=dict(path=str(group/'C_FAILED_archive.json'), sha256=digest),
        failure_log=dict(path=str(group/'C/monitor/actual_failure.log'), sha256=digest),
        monitor_path=str(group/'C/monitor/C_recovery_CPU_DEBUG/status.json'))
    return value


class CIsolationTests(unittest.TestCase):
    def test_only_C_exact_independent_controllers_allowed(self):
        value = seal_document()
        self.assertIs(resources.validate_seal(value), value)
        for key, bad in (('selected_arms', ['A', 'C']), ('peer_controller_identity', value['controller_identity']),
                ('no_process_signaled', False)):
            altered = copy.deepcopy(value); altered[key] = bad
            with self.subTest(key=key), self.assertRaises(ValueError):
                resources.validate_seal(altered)

    def test_canonical_monitor_cannot_be_selected_or_unsealed(self):
        value = seal_document(); group = Path(value['group_root'])
        for path in (group/'C/monitor/status.json', group/'B/monitor/recovery/status.json',
                group/'C/monitor/recovery/other.json'):
            altered = copy.deepcopy(value); altered['slots']['C']['monitor_path'] = str(path)
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, 'Fresh C monitor'):
                resources.validate_seal(altered)
        value['preserved_files'] = {'unrelated': 'a'*64}
        with self.assertRaisesRegex(ValueError, 'canonical original C FAILED'):
            resources.validate_seal(value)

    def test_full_original_C_curriculum_checkpoint_contract_required(self):
        value = seal_document()
        for key, bad in (('active_u', 16), ('epoch', 41), ('updates', 681), ('train_position', 66),
                ('evaluation_position', 22), ('status', 'FAILED')):
            altered = copy.deepcopy(value); altered['slots']['C']['checkpoint'][key] = bad
            with self.subTest(key=key), self.assertRaises(ValueError):
                resources.validate_seal(altered)

    def test_original_failure_evidence_requires_old_actor_and_release(self):
        seal = seal_document(); slot = seal['slots']['C']; old = slot['previous_actor']
        monitor = dict(status='FAILED', stage='train', arm='C', controller_pid=800,
            child_pid=old['pid'], child_create_time=old['create_time'], UID=old['UID'],
            child_command=old['command'], stage_log=slot['failure_log']['path'])
        log = Mock(); log.read_text.return_value = (
            'RuntimeError: Exact live descendant PID/birth/argv/UID identity required')
        with patch.object(resources, '_guard_proof', return_value=monitor), \
                patch.object(resources.direct, '_owned_regular', return_value=log), \
                patch.object(resources, 'sha', return_value='a'*64), \
                patch.object(resources.original, '_released') as released:
            resources._validate_failed_slot(slot, seal)
            released.assert_called_once_with(old)
            monitor['controller_pid'] = 700
            with self.assertRaisesRegex(ValueError, 'original C FAILED'):
                resources._validate_failed_slot(slot, seal)

    def test_live_ABD_tree_is_in_RAM_union_and_overlap_rejected(self):
        seal = seal_document(); peer = Mock(); peer.pid = 700
        child = Mock(); child.pid = 701; identity = actor(701)
        peer.children.return_value = [child]
        peer.cpu_affinity.return_value = list(range(24))
        lease = SimpleNamespace(group=Path(seal['group_root']), managed={},
            request=dict(aggregate_CPU_affinity=list(range(24)), cgroup_admission={'CPU_DEBUG': True}))
        with patch.object(resources.ad, '_peer_controller', return_value=peer), \
                patch.object(resources.direct, 'capture_cgroup_admission', return_value={'CPU_DEBUG': True}), \
                patch.object(resources.terminal, '_descendant', return_value=identity):
            own = {900: Mock()}; union = resources._peer_tree(lease, seal, own)
            self.assertEqual(set(union), {900, 700, 701})
            self.assertEqual(set(own), {900})
            self.assertEqual(lease._observed_ABD_descendants, {701: identity})
            with self.assertRaisesRegex(RuntimeError, 'trees overlap'):
                resources._peer_tree(lease, seal, {700: peer})

    def test_missing_ABD_controller_requires_terminal_monitors_and_preserves_live_orphan_RAM(self):
        seal = seal_document(); identity = actor(701); child = Mock(); child.pid = 701
        lease = SimpleNamespace(group=Path(seal['group_root']), managed={},
            _observed_ABD_descendants={701: identity})
        with patch.object(resources.ad, '_peer_controller',
                side_effect=resources.direct.psutil.NoSuchProcess(700)), \
                patch.object(resources, '_terminal_peer_monitors') as terminal, \
                patch.object(resources.direct.psutil, 'Process', return_value=child), \
                patch.object(resources.terminal, '_descendant', return_value=identity):
            self.assertEqual(set(resources._peer_tree(lease, seal, {900: Mock()})), {900, 701})
            terminal.assert_called_once_with(lease, seal)
        with patch.object(resources.ad, '_peer_controller', return_value=None), \
                patch.object(resources, '_terminal_peer_monitors',
                    side_effect=RuntimeError('CPU_DEBUG missing genuine terminal monitor')):
            with self.assertRaisesRegex(RuntimeError, 'genuine terminal monitor'):
                resources._peer_tree(lease, seal, {})

    def test_initial_terminal_ABD_controller_requires_verified_terminal_monitors(self):
        seal = seal_document()
        contract = dict(controller_identity=seal['peer_controller_identity'], runtime_files={}, slots={})
        with patch.object(resources, '_guard_proof', return_value=contract), \
                patch.object(resources.ad, 'validate_seal', return_value=contract), \
                patch.object(resources.ad, '_peer_controller',
                    side_effect=resources.direct.psutil.NoSuchProcess(700)), \
                patch.object(resources, '_terminal_peer_monitors') as verified:
            receipt = resources.verify_initial_peer(seal)
            self.assertTrue(receipt['ABD_controller_terminal_with_verified_monitors'])
            self.assertEqual(receipt['ABD_identity'], seal['peer_controller_identity'])
            self.assertEqual(verified.call_count, 1)
            self.assertEqual(verified.call_args.args[0].group, Path(seal['group_root']))
            self.assertEqual(verified.call_args.args[1], seal)
        with patch.object(resources, '_guard_proof', return_value=contract), \
                patch.object(resources.ad, 'validate_seal', return_value=contract), \
                patch.object(resources.ad, '_peer_controller', return_value=None), \
                patch.object(resources, '_terminal_peer_monitors',
                    side_effect=RuntimeError('CPU_DEBUG incomplete terminal ABD monitors')):
            with self.assertRaisesRegex(RuntimeError, 'incomplete terminal ABD'):
                resources.verify_initial_peer(seal)

    def test_initial_live_peer_requires_original_runtime_digest_and_cgroup(self):
        seal = seal_document(); peer = Mock(); peer.pid = 700
        proofref = {'path': '/CPU_DEBUG/direct.json', 'sha256': 'a'*64}
        contract = dict(controller_identity=seal['peer_controller_identity'],
            runtime_files={'CPU_DEBUG/runtime.py': 'a'*64}, slots={'A': {'direct_proof': proofref}})
        proof = dict(aggregate_CPU_affinity=list(range(24)), cgroup_admission={'DEBUG': True})
        peer.cpu_affinity.return_value = list(range(24))
        with patch.object(resources, '_guard_proof', side_effect=lambda ref:
                    proof if ref == proofref else contract), \
                patch.object(resources.ad, 'validate_seal', return_value=contract), \
                patch.object(resources.direct, '_owned_regular', return_value=Mock()), \
                patch.object(resources, 'sha', return_value='a'*64), \
                patch.object(resources.ad, '_peer_controller', return_value=peer), \
                patch.object(resources.direct, 'capture_cgroup_admission', return_value={'DEBUG': True}):
            self.assertFalse(resources.verify_initial_peer(seal)['ABD_controller_terminal_with_verified_monitors'])
            peer.cpu_affinity.return_value = [1]
            with self.assertRaisesRegex(RuntimeError, 'CPU/cgroup changed'):
                resources.verify_initial_peer(seal)
        with patch.object(resources, '_guard_proof', return_value=contract), \
                patch.object(resources.ad, 'validate_seal', return_value=contract), \
                patch.object(resources.direct, '_owned_regular', return_value=Mock()), \
                patch.object(resources, 'sha', return_value='b'*64):
            with self.assertRaisesRegex(ValueError, 'runtime source changed'):
                resources.verify_initial_peer(seal)

    def test_gpu_query_receives_only_own_C_stage_tree(self):
        lease = object.__new__(resources.StandaloneCAllocationLease)
        own = {900: Mock()}; selected = {901: Mock()}
        with patch.object(resources, '_stage_tree', return_value=selected), \
                patch.object(resources._BASE, '_gpu') as query:
            lease._gpu(own, initial=False)
        query.assert_called_once_with(lease, selected, initial=False)

    def test_missing_fresh_monitor_is_initial_admission_only(self):
        seal = seal_document(); lease = SimpleNamespace()
        with patch.object(resources, 'load_seal', return_value=seal), \
                patch.object(resources.Path, 'exists', return_value=False), \
                patch.object(resources.direct, '_owned_regular') as read:
            self.assertEqual(resources._stage_tree(lease, {}), {})
            read.assert_not_called()
            lease._stage_monitor_observed = True
            with self.assertRaisesRegex(RuntimeError, 'monitor disappeared'):
                resources._stage_tree(lease, {})

    def test_gpu_monitor_read_uses_only_new_namespace_and_rejects_wrong_controller(self):
        seal = seal_document(); lease = SimpleNamespace(request={'controller_identity': actor(900)})
        file = Mock(); file.read_bytes.return_value = json.dumps({'controller_pid': 800}).encode()
        with patch.object(resources, 'load_seal', return_value=seal), \
                patch.object(resources.Path, 'exists', return_value=True), \
                patch.object(resources.direct, '_owned_regular', return_value=file) as read:
            with self.assertRaisesRegex(RuntimeError, 'monitor controller changed'):
                resources._stage_tree(lease, {})
        read.assert_called_once_with(Path(seal['slots']['C']['monitor_path']))
        self.assertNotEqual(read.call_args.args[0], Path(seal['group_root'])/'C/monitor/status.json')

    def test_terminal_ABD_monitors_all_require_exact_controller_and_real_completions(self):
        seal = seal_document(); lease = SimpleNamespace(group=Path(seal['group_root']))
        rows = {arm: dict(status='COMPLETE', stage='complete', arm=arm, controller_pid=700)
            for arm in ('A', 'B', 'D')}
        def file(path):
            out = Mock(); out.read_bytes.return_value = json.dumps(rows[Path(path).parts[-3]]).encode()
            return out
        with patch.object(resources, '_guard_proof', return_value={}), \
                patch.object(resources.direct, '_owned_regular', side_effect=file), \
                patch('tools.run_v25_u128_reuse_continuation.verify_arm_completion') as verified:
            resources._terminal_peer_monitors(lease, seal)
            self.assertEqual([call.args for call in verified.call_args_list],
                [(lease.group, arm) for arm in ('A', 'B', 'D')])
            rows['B']['status'] = 'RUNNING'
            with self.assertRaisesRegex(RuntimeError, 'actual terminal monitor'):
                resources._terminal_peer_monitors(lease, seal)

    def test_monitor_stage_binding_requires_own_C_wrapper_and_mode(self):
        seal = seal_document(); slot = seal['slots']['C']
        command = ['/home/aicompetition06/.conda/envs/nnunet/bin/python3.10', '-B', '-u',
            str(Path(seal['private_code'])/'tools/run_v25_c_execution.py'),
            '--continuation-seal', '/CPU_DEBUG/seal.json', '--arm', 'C', '--gpu', '1',
            '--output', str(Path(seal['group_root'])/'C'), '--allocation-request', slot['direct_proof']['path'],
            '--mode', 'train']
        identity = actor(901, command)
        with patch.dict(resources.os.environ, {resources.ENV_PATH: '/CPU_DEBUG/seal.json'}):
            resources._stage_binding(identity, seal, 'train')
            identity['command'][identity['command'].index('--arm')+1] = 'A'
            with self.assertRaisesRegex(RuntimeError, 'argument binding changed'):
                resources._stage_binding(identity, seal, 'train')

    def test_generated_full_chain_resumes_own_C_and_native_pipeline(self):
        value = seal_document(); slot = value['slots']['C']; group = Path(value['group_root'])
        args = SimpleNamespace(group=group, execution_manifest=group/'C_seal.json')
        for name in ('native_experiment', 'inventory', 'stunet_checkpoint', 'input_cache', 'storage_admission',
                'source_bank', 'baseline_preprocessed', 'project', 'basic_validation', 'evaluator_root'):
            setattr(args, name, group/name)
        train = continuation.stage_command(args, slot, 'train')
        score = continuation.stage_command(args, slot, 'score-cp')
        native = continuation.native_command(args, slot)
        self.assertEqual(train[3], str(continuation.ROOT/'tools/run_v25_c_execution.py'))
        self.assertEqual(train[train.index('--resume')+1], slot['checkpoint']['path'])
        self.assertNotIn('--resume', score)
        self.assertEqual(native[4], '--native')
        for command in (train, score, native):
            self.assertEqual(command[command.index('--arm')+1], 'C')
            self.assertEqual(command[command.index('--gpu')+1], '1')
            self.assertNotIn('--debug', command)

    def test_native_children_install_C_guard_and_reject_unknown_entry(self):
        command = [execution.sys.executable, '-B', '-u', '-c', execution.NATIVE_ENTRY]
        adapted = execution.adapt_native_child(command)
        self.assertTrue(adapted[4].startswith(execution.NATIVE_BOOTSTRAP))
        self.assertEqual(command[4], execution.NATIVE_ENTRY)
        clone = [execution.sys.executable, '-B', '-u',
            str(execution.ROOT/'tools/run_v25_downstream.py'), 'clone-debug-worker']
        adapted = execution.adapt_native_child(clone)
        self.assertEqual(adapted[3:5], [str(execution.ROOT/'tools/run_v25_c_execution.py'), '--native'])
        with self.assertRaisesRegex(ValueError, 'Unknown native child'):
            execution.adapt_native_child([execution.sys.executable, '-B', '-u', '-c', 'unrelated()'])


if __name__ == '__main__':
    unittest.main()
