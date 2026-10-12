"""CPU UNIT for exact ABD recovery admission and independent full chains."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import threading
import unittest
from unittest.mock import Mock, patch

from hiercp_v1x import v25_standalone_ad_resources as resources
from tools import run_v25_ad_continuation as continuation
from tools import run_v25_ad_execution as execution


def actor(pid, command=None):
    return dict(pid=pid, create_time=float(pid), UID=41833,
        command=['python', 'CPU_DEBUG_actor.py'] if command is None else command)


def seal_document():
    group = (execution.ROOT/'tmp/CPU_DEBUG_ABD_admission').resolve()
    old = group/'old_code'; code = group/'new_code'; digest = 'a'*64
    value = dict(format=resources.FORMAT, selected_arms=['A', 'B', 'D'], group_root=str(group),
        private_code=str(code), previous_code=str(old), controller_identity=actor(900),
        peer_controller_identity=actor(800), peer_seal=dict(path=str(group/'old_seal.json'), sha256=digest),
        peer_slots={}, slots={}, preserved_files={'frozen_source': digest},
        runtime_files={'CPU_DEBUG_runtime': digest}, created_at=1.,
        scientific_configuration_unchanged=True, no_process_signaled=True)
    for index, (arm, (gpu, cpus)) in enumerate(resources.MAPPING.items()):
        command = ['/home/aicompetition06/.conda/envs/nnunet/bin/python3.10', '-B', '-u',
            str(old/'tools/run_v25_u128_reuse_execution.py'), '--continuation-seal', value['peer_seal']['path'],
            '--mode', 'train', '--arm', arm, '--gpu', str(gpu), '--output', str(group/arm)]
        value['slots'][arm] = dict(arm=arm, gpu=gpu, CPU_affinity=cpus,
            direct_proof=dict(path=str(group/(arm+'_direct.json')), sha256=digest),
            checkpoint=dict(path=str(group/arm/'training/checkpoint_latest.pt'), sha256=digest,
                identity_sha256=digest, content_sha256=digest, epoch=5, updates=68,
                phase='training', train_position=0, evaluation_position=0,
                active_u=resources.original.INCREMENTS[arm], status='RUNNING'),
            previous_actor=actor(810+index, command),
            failed_monitor=dict(path=str(group/(arm+'_FAILED_archive.json')), sha256=digest),
            failure_log=dict(path=str(group/arm/'monitor/actual_failure.log'), sha256=digest))
    initial = actor(801)
    value['peer_slots']['C'] = dict(direct_proof=dict(path=str(group/'C_old_direct.json'), sha256=digest),
        initial_actor=initial, command_sha256=resources.command_sha(initial['command']))
    return value


class StandaloneADTests(unittest.TestCase):
    def test_exact_ABD_only_and_original_C_identity_SHA_are_required(self):
        value = seal_document()
        self.assertIs(resources.validate_seal(value), value)
        for key, altered in (('selected_arms', ['A', 'D']), ('selected_arms', ['A', 'B', 'C', 'D']),
                ('no_process_signaled', False), ('scientific_configuration_unchanged', False)):
            bad = copy.deepcopy(value); bad[key] = altered
            with self.subTest(key=key, altered=altered), self.assertRaises(ValueError):
                resources.validate_seal(bad)
        bad = copy.deepcopy(value); bad['peer_slots']['C']['initial_actor']['command'].append('different')
        with self.assertRaisesRegex(ValueError, 'argv SHA'):
            resources.validate_seal(bad)

    def test_original_GPU_CPU_and_128_candidate_contract_is_preserved(self):
        for arm in resources.MAPPING:
            for key, bad_value in (('gpu', 0), ('CPU_affinity', [0]),):
                value = seal_document(); value['slots'][arm][key] = bad_value
                with self.subTest(arm=arm, key=key), self.assertRaises(ValueError):
                    resources.validate_seal(value)
            value = seal_document(); value['slots'][arm]['checkpoint']['active_u'] = 1
            with self.subTest(arm=arm), self.assertRaises(ValueError):
                resources.validate_seal(value)

    def test_complete_checkpoint_receipt_preserves_dynamic_saved_cursor(self):
        value = seal_document()
        cp = value['slots']['A']['checkpoint']
        cp.update(epoch=5, updates=68, phase='train_probe', train_position=65,
            evaluation_position=64, active_u=16)
        resources.validate_seal(value)
        cp['evaluation_position'] = 66
        with self.assertRaisesRegex(ValueError, 'optimizer/RNG'):
            resources.validate_seal(value)
        cp.update(phase='full_validation', evaluation_position=22)
        with self.assertRaisesRegex(ValueError, 'optimizer/RNG'):
            resources.validate_seal(value)
        cp.update(evaluation_position=0, epoch=41)
        with self.assertRaises(ValueError):
            resources.validate_seal(value)

    def test_failed_own_monitor_log_identity_and_actual_release_are_required(self):
        value = seal_document()
        for arm in resources.MAPPING:
            slot = value['slots'][arm]; previous = slot['previous_actor']
            monitor = dict(status='FAILED', stage='train', arm=arm,
                controller_pid=value['peer_controller_identity']['pid'], child_pid=previous['pid'],
                child_create_time=previous['create_time'], UID=previous['UID'],
                child_command=previous['command'], stage_log=slot['failure_log']['path'])
            log = Mock(); log.read_text.return_value = ('MemoryError: Historical evaluation RSS budget exceeded; no cases skipped'
                if arm == 'D' else 'RuntimeError: Exact live descendant PID/birth/argv/UID identity required')
            with patch.object(resources, '_guard_proof', return_value=monitor), \
                    patch.object(resources.direct, '_owned_regular', return_value=log), \
                    patch.object(resources, 'sha', return_value='a'*64), \
                    patch.object(resources.original, '_released') as released:
                resources._validate_failed_slot(slot, value)
                released.assert_called_once_with(previous)
                monitor['child_create_time'] = 999.
                with self.assertRaisesRegex(ValueError, 'FAILED monitor'):
                    resources._validate_failed_slot(slot, value)

    def test_initial_C_actor_is_proven_live_without_signaling_it(self):
        value = seal_document(); peer = Mock(); child = Mock()
        peer.pid = 800; child.ppid.return_value = 800; child.parent.return_value = peer
        child.cpu_affinity.return_value = resources.original.MAPPING['C'][1]
        child.cmdline.return_value = value['peer_slots']['C']['initial_actor']['command']
        with patch.object(resources.direct, '_exact', side_effect=[peer, child]), \
                patch.object(resources.direct, '_actor', return_value=value['peer_controller_identity']):
            proof = resources.verify_initial_peer(value)
        self.assertTrue(proof['C_actor_preserved'])
        child.ppid.return_value = 999
        with patch.object(resources.direct, '_exact', side_effect=[peer, child]), \
                patch.object(resources.direct, '_actor', return_value=value['peer_controller_identity']):
            with self.assertRaisesRegex(RuntimeError, 'ancestry'):
                resources.verify_initial_peer(value)

    def test_RAM_includes_exact_peer_tree_but_C_is_not_in_GPU_allowlist(self):
        lease = resources.StandaloneADAllocationLease.__new__(resources.StandaloneADAllocationLease)
        own = {900: Mock()}; union = {**own, 800: Mock(), 801: Mock()}
        def original_resources(instance, controller, current, tree):
            self.assertIs(tree, union)
            instance.actual_resources = {'original': True}
        with patch.object(resources, 'validate_own_contracts', return_value=seal_document()), \
                patch.object(resources, '_peer_tree', return_value=union), \
                patch.object(resources._BASE, '_resources', new=original_resources):
            lease._resources(Mock(), Mock(), own)
        self.assertTrue(lease.actual_resources['original_C_tree_only_used_for_RAM'])
        self.assertEqual(lease.actual_resources['combined_ABD_C_RSS_members'], 3)
        selected = {901: Mock()}
        with patch.object(resources, '_stage_tree', return_value=selected), \
                patch.object(resources._BASE, '_gpu', return_value='verified') as gpu:
            self.assertEqual(lease._gpu(own, initial=False), 'verified')
        self.assertIs(gpu.call_args.args[1], selected)
        self.assertNotIn(801, selected)

    def test_GPU_stage_binding_rejects_sibling_even_if_CPU_affinity_matches(self):
        value = seal_document(); slot = value['slots']['A']; group = Path(value['group_root'])
        command = ['/home/aicompetition06/.conda/envs/nnunet/bin/python3.10', '-B', '-u',
            str(Path(value['private_code'])/'tools/run_v25_ad_execution.py'),
            '--continuation-seal', 'own_seal', '--mode', 'train', '--arm', 'A', '--gpu', '6',
            '--output', str(group/'A'), '--allocation-request', slot['direct_proof']['path']]
        identity = actor(901, command)
        resources._stage_binding(identity, 'A', 6, group/'A', value['private_code'],
            'own_seal', slot['direct_proof']['path'], 'train')
        command[command.index('--arm')+1] = 'B'
        with self.assertRaisesRegex(RuntimeError, 'binding changed'):
            resources._stage_binding(identity, 'A', 6, group/'A', value['private_code'],
                'own_seal', slot['direct_proof']['path'], 'train')

    def test_new_workers_retrigger_unchanged_RAM_admission_before_GPU_check(self):
        lease = resources.StandaloneADAllocationLease.__new__(resources.StandaloneADAllocationLease)
        lease.request = dict(controller_identity=actor(900)); lease.self_identity = actor(901)
        tree = {900: Mock(), 901: Mock()}; selected = {901: tree[901], 902: Mock()}; calls = []
        def stage(instance, current):
            current[902] = selected[902]
            return selected
        with patch.object(resources, '_stage_tree', side_effect=stage), \
                patch.object(resources.direct, '_exact', side_effect=lambda value: tree[value['pid']]), \
                patch.object(lease, '_resources', side_effect=lambda *a: calls.append('RAM')), \
                patch.object(resources._BASE, '_gpu', side_effect=lambda *a, **k: calls.append('GPU')):
            lease._gpu(tree, initial=False)
        self.assertEqual(calls, ['RAM', 'GPU'])

    def test_controller_commands_use_own_latest_and_no_prepare_or_recalibrate(self):
        value = seal_document(); args = SimpleNamespace(group=Path(value['group_root']),
            execution_manifest=Path('new_seal.json'), native_experiment=Path('native'), inventory=Path('inventory'),
            stunet_checkpoint=Path('official'), input_cache=Path('existing_all14102'),
            storage_admission=Path('all4storage'), source_bank=Path('all642'), baseline_preprocessed=Path('preprocessed'),
            project=Path('project'), basic_validation=Path('basic'), evaluator_root=Path('frozen_evaluator'))
        for arm, slot in value['slots'].items():
            command = continuation.stage_command(args, slot, 'train')
            self.assertEqual(command[command.index('--resume')+1], slot['checkpoint']['path'])
            self.assertEqual(command[command.index('--arm')+1], arm)
            self.assertNotIn('prepare', command); self.assertNotIn('calibrate', command)
            scoring = continuation.stage_command(args, slot, 'score-cp')
            self.assertNotIn('--resume', scoring)
            native = continuation.native_command(args, slot)
            self.assertEqual(native[4:7], ['--native', '--continuation-seal', str(args.execution_manifest)])
            self.assertIn(str(args.group/arm/'cp_bank/score_overlay.json'), native)

    def test_native_real_training_and_private_debug_worker_retain_guard(self):
        command = [execution.sys.executable, '-B', '-u', '-c', execution.NATIVE_ENTRY, '--full-native-args']
        adapted = execution.adapt_native_child(command)
        self.assertEqual(adapted[4], execution.NATIVE_BOOTSTRAP+execution.NATIVE_ENTRY)
        self.assertEqual(adapted[-1], '--full-native-args')
        debug = [execution.sys.executable, '-B', '-u',
            str(execution.ROOT/'tools/run_v25_downstream.py'), 'clone-debug-worker', '--full-debug-args']
        adapted = execution.adapt_native_child(debug)
        self.assertEqual(adapted[3:6], [str(execution.ROOT/'tools/run_v25_ad_execution.py'),
            '--native', 'clone-debug-worker'])
        with self.assertRaisesRegex(ValueError, 'Unknown native child'):
            execution.adapt_native_child([execution.sys.executable, '-B', '-u', '-c', 'unrelated()'])

    def test_native_cached_clone_callback_is_adapted_too(self):
        native = SimpleNamespace(_child=Mock(return_value=7), _MODULES={'NATIVE_PIPELINE': SimpleNamespace(_child=None)})
        original = native._child
        execution.install_native_children(native)
        self.assertIs(native._MODULES['NATIVE_PIPELINE']._child, native._child)
        command = [execution.sys.executable, '-B', '-u', '-c', execution.NATIVE_ENTRY]
        self.assertEqual(native._child(command, {'all': 'env'}, Path('log')), 7)
        self.assertEqual(original.call_args.args[0][4], execution.NATIVE_BOOTSTRAP+execution.NATIVE_ENTRY)

    def test_three_chains_start_concurrently_and_one_failure_preserves_others(self):
        barrier = threading.Barrier(3); seen = []; failed = []
        def chain(arm):
            seen.append(arm); barrier.wait(timeout=5)
            if arm == 'A':
                raise RuntimeError('CPU_DEBUG_A_failure')
        failures = continuation.run_chains(chain, failed.append)
        self.assertEqual(set(seen), {'A', 'B', 'D'})
        self.assertEqual([item['arm'] for item in failures], ['A'])
        self.assertEqual(failed[-1], failures)


class PeerControllerTerminalTests(unittest.TestCase):
    """CPU UNIT: a released C controller never grants unknown live admission."""

    def setUp(self):
        self.seal = seal_document()
        self.identity = self.seal['peer_controller_identity']
        self.peer = Mock()
        self.peer.pid = self.identity['pid']
        self.peer.create_time.return_value = self.identity['create_time']
        self.peer.uids.return_value = SimpleNamespace(real=self.identity['UID'])
        self.peer.status.return_value = resources.direct.psutil.STATUS_RUNNING
        self.peer.cmdline.return_value = self.identity['command']
        self.lease = SimpleNamespace(group=Path(self.seal['group_root']), managed={})
        self.monitor = dict(status='COMPLETE', stage='complete', arm='C',
            controller_pid=self.identity['pid'])
        self.monitor_file = Mock()
        self.monitor_file.read_bytes.side_effect = lambda: json.dumps(self.monitor).encode()
        self.own_tree = {900: Mock()}

    def controller(self):
        with patch.object(resources.direct.psutil, 'Process', return_value=self.peer):
            return resources._peer_controller(self.lease, self.identity)

    def terminal_tree(self, *, missing=False, verifier=None):
        process = (resources.direct.psutil.NoSuchProcess(self.identity['pid'])
            if missing else self.peer)
        with patch.object(resources.direct.psutil, 'Process', **(
                {'side_effect': process} if missing else {'return_value': process})), \
                patch.object(resources.direct, '_owned_regular', return_value=self.monitor_file), \
                patch.object(resources, 'sha', return_value='a'*64), \
                patch('tools.run_v25_u128_reuse_continuation.verify_arm_completion', verifier or Mock()):
            return resources._peer_tree(self.lease, self.seal, self.own_tree)

    def failed_monitor(self, stage='train'):
        command = ['/home/aicompetition06/.conda/envs/nnunet/bin/python3.10', '-B', '-u',
            str(Path(self.seal['previous_code'])/'tools/run_v25_u128_reuse_execution.py')]
        if stage == 'native':
            command += ['--native']
        command += ['--continuation-seal', self.seal['peer_seal']['path'],
            '--arm', 'C', '--gpu', '1', '--output', str(self.lease.group/'C'),
            '--allocation-request', self.seal['peer_slots']['C']['direct_proof']['path']]
        if stage != 'native':
            command += ['--mode', 'train' if stage == 'train' else 'score-cp']
        identity = actor(801, command)
        self.seal['peer_slots']['C']['initial_actor']['command'] = command[:4] + ['initial_train']
        self.monitor.update(status='FAILED', stage=stage, child_pid=identity['pid'],
            child_create_time=identity['create_time'], UID=identity['UID'],
            child_command=identity['command'])
        return identity

    def test_complete_live_controller_never_waits_or_becomes_terminal(self):
        with patch.object(resources.time, 'sleep') as sleep:
            self.assertIs(self.controller(), self.peer)
        sleep.assert_not_called()
        self.assertFalse(hasattr(self.lease, '_terminal_C_controller_identity'))

    def test_actual_same_identity_zombie_uses_complete_monitor_and_full_certificate(self):
        self.peer.status.return_value = resources.direct.psutil.STATUS_ZOMBIE
        verifier = Mock()
        self.assertEqual(self.terminal_tree(verifier=verifier), self.own_tree)
        verifier.assert_called_once_with(self.lease.group, 'C')
        self.assertEqual(self.lease._terminal_C_controller_identity, self.identity)
        self.peer.cmdline.assert_not_called()

    def test_disappeared_controller_still_requires_full_complete_certificate(self):
        verifier = Mock()
        self.assertEqual(self.terminal_tree(missing=True, verifier=verifier), self.own_tree)
        verifier.assert_called_once_with(self.lease.group, 'C')

    def test_empty_live_argv_then_actual_zombie_uses_verified_terminal_monitor(self):
        self.peer.cmdline.return_value = []
        state = {'status': resources.direct.psutil.STATUS_RUNNING}
        self.peer.status.side_effect = lambda: state['status']
        def yield_to_exit(delay):
            self.assertEqual(delay, .01)
            state['status'] = resources.direct.psutil.STATUS_ZOMBIE
        verifier = Mock()
        with patch.object(resources.time, 'sleep', side_effect=yield_to_exit) as sleep:
            self.assertEqual(self.terminal_tree(verifier=verifier), self.own_tree)
        sleep.assert_called_once_with(.01)
        self.peer.cmdline.assert_called_once()
        verifier.assert_called_once_with(self.lease.group, 'C')

    def test_empty_live_argv_then_exact_argv_retains_live_identity(self):
        self.peer.cmdline.side_effect = [[], self.identity['command']]
        with patch.object(resources.time, 'sleep') as sleep:
            self.assertIs(self.controller(), self.peer)
        sleep.assert_called_once_with(.01)
        self.assertFalse(hasattr(self.lease, '_terminal_C_controller_identity'))

    def test_persistent_live_empty_argv_fails_after_exactly_three_reads(self):
        self.peer.cmdline.return_value = []
        with patch.object(resources.time, 'sleep') as sleep:
            with self.assertRaisesRegex(RuntimeError, 'bounded observations'):
                self.controller()
        self.assertEqual(self.peer.cmdline.call_count, 3)
        self.assertEqual([call.args for call in sleep.call_args_list], [(.01,), (.01,)])
        self.assertFalse(hasattr(self.lease, '_terminal_C_controller_identity'))

    def test_live_or_zombie_reused_pid_birth_and_foreign_UID_fail(self):
        for status in (resources.direct.psutil.STATUS_RUNNING, resources.direct.psutil.STATUS_ZOMBIE):
            for change in ('birth', 'UID'):
                with self.subTest(status=status, change=change):
                    self.peer.status.return_value = status
                    self.peer.create_time.return_value = 999. if change == 'birth' else self.identity['create_time']
                    self.peer.uids.return_value = SimpleNamespace(real=1 if change == 'UID' else 41833)
                    with self.assertRaises(RuntimeError):
                        self.controller()
        self.assertFalse(hasattr(self.lease, '_terminal_C_controller_identity'))

    def test_birth_reuse_or_UID_change_between_empty_rereads_fails(self):
        self.peer.cmdline.return_value = []
        self.peer.create_time.side_effect = [self.identity['create_time'], 999.]
        with patch.object(resources.time, 'sleep'):
            with self.assertRaisesRegex(RuntimeError, 'PID/birth/UID identity changed'):
                self.controller()
        self.peer.create_time.side_effect = None
        self.peer.uids.side_effect = [SimpleNamespace(real=41833), SimpleNamespace(real=1)]
        with patch.object(resources.time, 'sleep'):
            with self.assertRaisesRegex(RuntimeError, 'Foreign-UID'):
                self.controller()

    def test_live_birth_change_after_argv_observation_fails(self):
        self.peer.create_time.side_effect = [self.identity['create_time'], 999.]
        with self.assertRaisesRegex(RuntimeError, 'changed during live observation'):
            self.controller()

    def test_nonempty_argv_tamper_fails_even_if_next_status_becomes_zombie(self):
        self.peer.cmdline.return_value = ['python', 'unrelated.py']
        self.peer.status.side_effect = [resources.direct.psutil.STATUS_RUNNING,
            resources.direct.psutil.STATUS_ZOMBIE]
        with self.assertRaisesRegex(RuntimeError, 'argv identity changed'):
            self.controller()
        self.assertFalse(hasattr(self.lease, '_terminal_C_controller_identity'))

    def test_terminal_metadata_reobservation_must_keep_exact_birth(self):
        self.peer.status.return_value = resources.direct.psutil.STATUS_ZOMBIE
        self.peer.create_time.side_effect = [self.identity['create_time'], 999.]
        with self.assertRaisesRegex(RuntimeError, 'changed during terminal observation'):
            self.controller()

    def test_access_denied_is_not_hidden_after_an_empty_observation(self):
        self.peer.cmdline.side_effect = [[], resources.direct.psutil.AccessDenied(self.peer.pid)]
        with patch.object(resources.time, 'sleep'):
            with self.assertRaises(resources.direct.psutil.AccessDenied):
                self.controller()

    def test_fake_zombie_exception_does_not_grant_terminal_controller_admission(self):
        self.peer.cmdline.side_effect = resources.direct.psutil.ZombieProcess(self.peer.pid)
        with self.assertRaisesRegex(RuntimeError, 'actual zombie status'):
            self.controller()
        self.peer.cmdline.side_effect = None
        self.peer.uids.side_effect = resources.direct.psutil.ZombieProcess(self.peer.pid)
        with self.assertRaisesRegex(RuntimeError, 'UID/birth could not be verified'):
            self.terminal_tree()

    def test_terminal_controller_cannot_regain_live_admission(self):
        self.lease._terminal_C_controller_identity = self.identity.copy()
        with self.assertRaisesRegex(RuntimeError, 'Previously terminal'):
            self.controller()

    def test_zombie_without_actual_C_terminal_monitor_is_rejected(self):
        self.peer.status.return_value = resources.direct.psutil.STATUS_ZOMBIE
        for status in ('RUNNING', 'STAGE_COMPLETE', None):
            self.monitor['status'] = status
            with self.subTest(status=status), self.assertRaisesRegex(RuntimeError, 'actual terminal monitor'):
                self.terminal_tree()

    def test_terminal_monitor_requires_own_controller_and_C_arm(self):
        self.peer.status.return_value = resources.direct.psutil.STATUS_ZOMBIE
        for key, bad in (('controller_pid', 999), ('arm', 'A')):
            saved = self.monitor[key]
            self.monitor[key] = bad
            with self.subTest(key=key), self.assertRaisesRegex(RuntimeError, 'exact terminal monitor'):
                self.terminal_tree()
            self.monitor[key] = saved

    def test_complete_monitor_requires_complete_stage_and_valid_certificate(self):
        self.peer.status.return_value = resources.direct.psutil.STATUS_ZOMBIE
        self.monitor['stage'] = 'train'
        with self.assertRaisesRegex(RuntimeError, 'complete stage'):
            self.terminal_tree()
        self.monitor['stage'] = 'complete'
        verifier = Mock(side_effect=ValueError('CPU_DEBUG incomplete BEST26 certificate'))
        with self.assertRaisesRegex(ValueError, 'incomplete BEST26'):
            self.terminal_tree(verifier=verifier)

    def test_complete_certificate_change_is_rejected_after_cached_verification(self):
        self.peer.status.return_value = resources.direct.psutil.STATUS_ZOMBIE
        self.lease._completed_C_certificate_sha256 = 'b'*64
        with self.assertRaisesRegex(ValueError, 'completion certificate changed'):
            self.terminal_tree()

    def test_failed_train_scoring_native_monitor_requires_exact_released_actor(self):
        self.peer.status.return_value = resources.direct.psutil.STATUS_ZOMBIE
        for stage in ('train', 'scoring', 'native'):
            previous = self.failed_monitor(stage)
            with self.subTest(stage=stage), patch.object(resources.original, '_released') as released:
                self.assertEqual(self.terminal_tree(), self.own_tree)
                released.assert_called_once_with(previous)
        with patch.object(resources.original, '_released', side_effect=RuntimeError('CPU_DEBUG actor still live')):
            with self.assertRaisesRegex(RuntimeError, 'actor still live'):
                self.terminal_tree()

    def test_failed_C_monitor_rejects_wrong_stage_mode_or_output(self):
        self.peer.status.return_value = resources.direct.psutil.STATUS_ZOMBIE
        previous = self.failed_monitor()
        self.monitor['stage'] = 'native'
        with self.assertRaisesRegex(RuntimeError, 'exact train/scoring/native stage'):
            self.terminal_tree()
        self.monitor['stage'] = 'train'
        previous['command'][previous['command'].index('--mode')+1] = 'prepare'
        with self.assertRaisesRegex(RuntimeError, 'stage/mode binding changed'):
            self.terminal_tree()
        previous['command'][previous['command'].index('--mode')+1] = 'train'
        previous['command'][previous['command'].index('--output')+1] = str(self.lease.group/'A')
        with self.assertRaisesRegex(RuntimeError, 'own output binding changed'):
            self.terminal_tree()

    def test_terminal_C_controller_retains_exact_observed_live_orphans_for_RAM(self):
        self.peer.status.return_value = resources.direct.psutil.STATUS_ZOMBIE
        orphan = actor(802, ['python', 'CPU_DEBUG_C_orphan.py'])
        child = Mock(); child.pid = 802
        self.lease._observed_C_descendants = {802: orphan}
        with patch.object(resources, '_peer_controller', return_value=None), \
                patch.object(resources.direct.psutil, 'Process', return_value=child), \
                patch.object(resources.direct, '_owned_regular', return_value=self.monitor_file), \
                patch.object(resources, 'sha', return_value='a'*64), \
                patch('tools.run_v25_u128_reuse_continuation.verify_arm_completion'), \
                patch.object(resources.terminal, '_descendant', return_value=orphan):
            union = resources._peer_tree(self.lease, self.seal, self.own_tree)
        self.assertEqual(set(union), {900, 802})
        self.assertIs(union[802], child)


if __name__ == '__main__':
    unittest.main()
