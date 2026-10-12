"""CPU UNIT: exact terminal recovery, full independent chains and isolation."""
import copy
from pathlib import Path
from types import SimpleNamespace
import threading
import unittest
from unittest.mock import Mock, patch

from hiercp_v1x import v25_all_arm_recovery_resources as resources
from tools import run_v25_all_arm_recovery_continuation as continuation
from tools import run_v25_all_arm_recovery_execution as execution


def actor(pid, command=None):
    return dict(pid=pid, create_time=float(pid), UID=41833,
        command=['python', 'CPU_DEBUG_actor.py'] if command is None else command)


def fixture():
    group = (execution.ROOT/'tmp/CPU_DEBUG_ALL_admission').resolve(); code = group/'ALL_code'
    digest = 'a'*64
    seal = dict(format=resources.FORMAT, selected_arms=list('ABCD'), group_root=str(group),
        private_code=str(code), controller_identity=actor(900),
        previous_seals={key: dict(path=str(group/(key+'_seal.json')), sha256=digest)
            for key in ('u128', 'ABD')}, slots={}, preserved_files={}, runtime_files={'CPU_DEBUG_runtime': digest},
        source_scientific23={str(code/f'source_{index}.py'): digest for index in range(23)},
        created_at=1., scientific_configuration_unchanged=True, no_process_signaled=True)
    seal['preserved_files'].update(seal['source_scientific23'])
    contracts = {}
    for key, pid in (('u128', 800), ('ABD', 700)):
        previous = group/(key+'_code')
        contracts[key] = dict(group_root=str(group), private_code=str(previous), controller_identity=actor(pid),
            runtime_files={str(previous/'CPU_DEBUG_runtime.py'): digest},
            preserved_files={str(previous/f'source_{index}.py'): digest for index in range(23)}, slots={})
    contracts['ABD'].update(peer_controller_identity=contracts['u128']['controller_identity'],
        peer_seal=seal['previous_seals']['u128'])
    for index, (arm, (gpu, cpus)) in enumerate(resources.MAPPING.items()):
        key = 'u128' if arm == 'C' else 'ABD'; prior = contracts[key]
        proof = dict(path=str(group/(key+'_'+arm+'_direct.json')), sha256=digest)
        prior['slots'][arm] = dict(direct_proof=proof)
        entry = 'run_v25_u128_reuse_execution.py' if arm == 'C' else 'run_v25_ad_execution.py'
        command = ['/home/aicompetition06/.conda/envs/nnunet/bin/python3.10', '-B', '-u',
            str(Path(prior['private_code'])/'tools'/entry), '--continuation-seal', seal['previous_seals'][key]['path'],
            '--mode', 'train', '--arm', arm, '--gpu', str(gpu), '--output', str(group/arm),
            '--allocation-request', proof['path']]
        seal['slots'][arm] = dict(arm=arm, gpu=gpu, CPU_affinity=cpus,
            direct_proof=dict(path=str(group/(arm+'_new_direct.json')), sha256=digest),
            checkpoint=dict(path=str(group/arm/'training/checkpoint_latest.pt'), sha256=digest,
                identity_sha256=digest, content_sha256=digest, epoch=5, updates=68,
                phase='training', train_position=0, evaluation_position=0,
                active_u=resources.original.INCREMENTS[arm], status='RUNNING'),
            previous_actor=actor(810+index, command),
            failed_monitor=dict(path=str(group/(arm+'_FAILED_archive.json')), sha256=digest),
            failure_log=dict(path=str(group/arm/'monitor/actual_failure.log'), sha256=digest),
            monitor_path=str(group/arm/'monitor/ALL_recovery_CPU_DEBUG/status.json'))
        seal['preserved_files'][str(group/arm/'monitor/status.json')] = digest
    return seal, contracts


class AllArmRecoveryTests(unittest.TestCase):
    def test_all_four_original_scales_and_distinct_monitor_paths_required(self):
        seal, _ = fixture(); self.assertIs(resources.validate_seal(seal), seal)
        for key, bad in (('selected_arms', ['A', 'B', 'C']), ('no_process_signaled', False)):
            altered = copy.deepcopy(seal); altered[key] = bad
            with self.subTest(key=key), self.assertRaises(ValueError):
                resources.validate_seal(altered)
        for arm in 'ABCD':
            altered = copy.deepcopy(seal)
            altered['slots'][arm]['monitor_path'] = str(Path(seal['group_root'])/arm/'monitor/status.json')
            with self.subTest(arm=arm), self.assertRaisesRegex(ValueError, 'no canonical monitor overwrite'):
                resources.validate_seal(altered)

    def test_every_canonical_failed_monitor_and_original23_sources_must_be_sealed(self):
        seal, _ = fixture()
        for name in ('canonical', 'source'):
            altered = copy.deepcopy(seal)
            if name == 'canonical':
                altered['preserved_files'].pop(str(Path(seal['group_root'])/'B/monitor/status.json'))
            else:
                altered['source_scientific23'].pop(next(iter(altered['source_scientific23'])))
            with self.subTest(name=name), self.assertRaises(ValueError):
                resources.validate_seal(altered)

    def test_exact_curriculum_and_checkpoint_cursors_preserved(self):
        seal, _ = fixture()
        for arm in 'ABCD':
            for key, bad in (('active_u', 1), ('updates', 681), ('train_position', 66)):
                altered = copy.deepcopy(seal); altered['slots'][arm]['checkpoint'][key] = bad
                with self.subTest(arm=arm, key=key), self.assertRaises(ValueError):
                    resources.validate_seal(altered)
        seal['slots']['A']['checkpoint'].update(phase='train_probe', train_position=65, evaluation_position=20,
            epoch=6, updates=102, active_u=16)
        seal['slots']['B']['checkpoint'].update(phase='train_probe', train_position=65, evaluation_position=0,
            epoch=7, updates=119, active_u=32)
        seal['slots']['C']['checkpoint'].update(epoch=5, updates=68, active_u=64)
        seal['slots']['D']['checkpoint'].update(phase='stage_validation', train_position=65,
            evaluation_position=0, epoch=4, updates=68, active_u=128)
        resources.validate_seal(seal)

    def test_previous_source_bytes_must_match_both_original_sealed_admissions(self):
        seal, contracts = fixture()
        with patch.object(resources, '_guard_proof', side_effect=lambda ref: contracts[
                    'u128' if ref == seal['previous_seals']['u128'] else 'ABD']), \
                patch.object(resources.original, 'validate_seal', side_effect=lambda value: value), \
                patch.object(resources.ad, 'validate_seal', side_effect=lambda value: value), \
                patch.object(resources.direct, '_owned_regular', side_effect=lambda name: name), \
                patch.object(resources, 'sha', return_value='a'*64):
            self.assertEqual(resources.previous_contracts(seal), contracts)
            contracts['ABD']['preserved_files'].pop(next(iter(contracts['ABD']['preserved_files'])))
            with self.assertRaisesRegex(ValueError, 'Original23 scientific source bytes'):
                resources.previous_contracts(seal)

    def test_wrong_prior_controller_or_original_runtime_digest_is_rejected(self):
        seal, contracts = fixture()
        with patch.object(resources, '_guard_proof', side_effect=lambda ref: contracts[
                    'u128' if ref == seal['previous_seals']['u128'] else 'ABD']), \
                patch.object(resources.original, 'validate_seal', side_effect=lambda value: value), \
                patch.object(resources.ad, 'validate_seal', side_effect=lambda value: value), \
                patch.object(resources.direct, '_owned_regular', side_effect=lambda name: name), \
                patch.object(resources, 'sha', return_value='b'*64):
            with self.assertRaisesRegex(ValueError, 'runtime source bytes changed'):
                resources.previous_contracts(seal)
            contracts['ABD']['peer_controller_identity'] = actor(999)
            with self.assertRaisesRegex(ValueError, 'original U128/ABD provenance'):
                resources.previous_contracts(seal)

    def test_own_actual_failure_traceback_original_actor_and_allocation_are_required(self):
        seal, contracts = fixture()
        for arm in 'ABCD':
            slot = seal['slots'][arm]; old = slot['previous_actor']; kind = 'u128' if arm == 'C' else 'ABD'
            monitor = dict(status='FAILED', stage='train', arm=arm,
                controller_pid=contracts[kind]['controller_identity']['pid'], child_pid=old['pid'],
                child_create_time=old['create_time'], UID=old['UID'], child_command=old['command'],
                stage_log=slot['failure_log']['path'])
            file = Mock(); file.read_text.return_value = (
                'RuntimeError: Exact live descendant PID/birth/argv/UID identity required' if arm == 'C'
                else 'ValueError: Admitted GT-free implementation/prototype input changed')
            with self.subTest(arm=arm), patch.object(resources, '_guard_proof', return_value=monitor), \
                    patch.object(resources.direct, '_owned_regular', return_value=file), \
                    patch.object(resources, 'sha', return_value='a'*64), \
                    patch.object(resources.original, '_released') as released:
                resources._validate_failed_slot(slot, seal, contracts)
                released.assert_called_once_with(old)
                file.read_text.return_value = 'ValueError: unrelated'
                with self.assertRaisesRegex(ValueError, 'no generic recovery fallback'):
                    resources._validate_failed_slot(slot, seal, contracts)
                file.read_text.return_value = ('RuntimeError: Exact live descendant PID/birth/argv/UID identity required'
                    if arm == 'C' else 'ValueError: Admitted GT-free implementation/prototype input changed')
                monitor['controller_pid'] = 999
                with self.assertRaisesRegex(ValueError, 'original failed own training'):
                    resources._validate_failed_slot(slot, seal, contracts)

    def test_initial_release_requires_all_four_stages_and_both_original_controllers(self):
        seal, contracts = fixture()
        with patch.object(resources, 'previous_contracts', return_value=contracts), \
                patch.object(resources, '_validate_failed_slot') as stages, \
                patch.object(resources.original, '_released') as controllers:
            receipt = resources.verify_initial_release(seal)
            self.assertEqual(stages.call_count, 4)
            self.assertEqual([call.args for call in controllers.call_args_list],
                [(contracts[key]['controller_identity'],) for key in ('u128', 'ABD')])
            self.assertTrue(receipt['source_timestamp_guard_unchanged'])
        with patch.object(resources, 'previous_contracts', return_value=contracts), \
                patch.object(resources, '_validate_failed_slot'), \
                patch.object(resources.original, '_released', side_effect=RuntimeError('CPU_DEBUG actor still live')):
            with self.assertRaisesRegex(RuntimeError, 'actor still live'):
                resources.verify_initial_release(seal)

    def test_gpu_admission_contains_only_the_own_stage_and_preserves_all_ram_tree(self):
        lease = object.__new__(resources.AllArmRecoveryAllocationLease)
        alltree = {900: Mock(), 901: Mock(), 902: Mock()}; own = {901: alltree[901]}
        with patch.object(resources, '_stage_tree', return_value=own), \
                patch.object(resources._BASE, '_gpu') as query:
            lease._gpu(alltree, initial=False)
        query.assert_called_once_with(lease, own, initial=False)
        self.assertEqual(set(alltree), {900, 901, 902})

    def test_fresh_monitor_missing_is_only_an_initial_admission(self):
        seal, _ = fixture(); lease = SimpleNamespace(request={'arm': 'B'})
        with patch.object(resources, 'load_seal', return_value=seal), \
                patch.object(resources.Path, 'exists', return_value=False), \
                patch.object(resources.direct, '_owned_regular') as read:
            self.assertEqual(resources._stage_tree(lease, {}), {})
            read.assert_not_called()
            lease._stage_monitor_observed = True
            with self.assertRaisesRegex(RuntimeError, 'monitor disappeared'):
                resources._stage_tree(lease, {})

    def test_full_chain_commands_bind_each_own_checkpoint_GPU_and_native_bootstrap(self):
        seal, _ = fixture(); group = Path(seal['group_root'])
        args = SimpleNamespace(group=group, execution_manifest=group/'ALL_seal.json')
        for name in ('native_experiment', 'inventory', 'stunet_checkpoint', 'input_cache', 'storage_admission',
                'source_bank', 'baseline_preprocessed', 'project', 'basic_validation', 'evaluator_root'):
            setattr(args, name, group/name)
        for arm, slot in seal['slots'].items():
            train = continuation.stage_command(args, slot, 'train')
            score = continuation.stage_command(args, slot, 'score-cp')
            native = continuation.native_command(args, slot)
            self.assertEqual(train[train.index('--resume')+1], slot['checkpoint']['path'])
            self.assertNotIn('--resume', score); self.assertEqual(native[4], '--native')
            for command in (train, score, native):
                self.assertEqual(command[3], str(continuation.ROOT/'tools/run_v25_all_arm_recovery_execution.py'))
                self.assertEqual(command[command.index('--arm')+1], arm)
                self.assertEqual(command[command.index('--gpu')+1], str(slot['gpu']))
                self.assertNotIn('--debug', command)

    def test_four_chains_start_concurrently_and_one_failure_does_not_cancel_others(self):
        barrier = threading.Barrier(4, timeout=2); completed = []; lock = threading.Lock(); reports = []
        def chain(arm):
            barrier.wait()
            if arm == 'B':
                raise RuntimeError('CPU_DEBUG exact failed stage')
            with lock:
                completed.append(arm)
        failures = continuation.run_chains(chain, reports.append)
        self.assertEqual(set(completed), {'A', 'C', 'D'})
        self.assertEqual([row['arm'] for row in failures], ['B'])
        self.assertEqual(len(reports), 1)

    def test_native_child_entries_install_exact_recovery_guard_and_reject_unknown_entry(self):
        command = [execution.sys.executable, '-B', '-u', '-c', execution.NATIVE_ENTRY]
        adapted = execution.adapt_native_child(command)
        self.assertTrue(adapted[4].startswith(execution.NATIVE_BOOTSTRAP))
        self.assertEqual(command[4], execution.NATIVE_ENTRY)
        clone = [execution.sys.executable, '-B', '-u',
            str(execution.ROOT/'tools/run_v25_downstream.py'), 'clone-debug-worker']
        self.assertEqual(execution.adapt_native_child(clone)[3:5],
            [str(execution.ROOT/'tools/run_v25_all_arm_recovery_execution.py'), '--native'])
        with self.assertRaisesRegex(ValueError, 'Unknown native child'):
            execution.adapt_native_child([execution.sys.executable, '-B', '-u', '-c', 'unrelated()'])


if __name__ == '__main__':
    unittest.main()
