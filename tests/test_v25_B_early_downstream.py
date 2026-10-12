"""CPU UNIT: truthful B-only early handoff, exact release and isolation."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

from hiercp_v1x import v25_B_early_downstream_resources as resources
from tools import run_v25_B_early_downstream_controller as controller


def identity(pid, command=None):
    return dict(pid=pid, create_time=float(pid), UID=41833,
        command=command or ['python', 'CPU_DEBUG_actor.py'])


def fixture():
    group = (controller.ROOT/'tmp/CPU_DEBUG_B_early').resolve()
    code = group/'new_code'; transition = group/'B_only_user_transition_CPU_DEBUG'
    digest = 'a'*64
    proof = lambda name: dict(path=str(transition/name), sha256=digest)
    seal = dict(format=resources.FORMAT, selected_arms=['B'], group_root=str(group),
        private_code=str(code), controller_identity=identity(900),
        controller_cgroup_admission=dict(controller_cgroup='0::/CPU_DEBUG',
            memory_limits=[dict(path='/sys/fs/cgroup/memory.max', limit_bytes=416*2**30)]),
        previous_all_seal=proof('ALL_seal.json'), authorization_request=proof('request.json'),
        release_receipt=proof('release.json'), scientific_request=proof('scientific_authorization.json'),
        slot=dict(arm='B', gpu=5, CPU_affinity=resources.CPUS,
            direct_proof=proof('original_B_direct.json'),
            monitor_path=str(transition/'execution/monitor/status.json')),
        preserved_files={}, runtime_files={str(code/resources.ENTRY): digest},
        source_scientific23={str(code/f'source_{i}.py'): digest for i in range(23)},
        created_at=1., scientific_configuration_unchanged=True,
        GNN_early_stop_authorized=True, processes_signaled_by_this_adapter=False)
    seal['preserved_files'].update(seal['source_scientific23'])
    for key in ('previous_all_seal', 'authorization_request', 'release_receipt', 'scientific_request'):
        seal['preserved_files'][seal[key]['path']] = digest
    seal['preserved_files'][seal['slot']['direct_proof']['path']] = digest
    for name in ('checkpoint_latest.pt', 'checkpoint_best.pt'):
        seal['preserved_files'][str(group/'B/training'/name)] = digest
    prior = dict(group_root=str(group), private_code=str(group/'old_code'),
        controller_identity=identity(800), slots={})
    request = dict(group_root=str(group), original_seal_path=seal['previous_all_seal']['path'],
        controller_identity=prior['controller_identity'], actors={}, B_finish_epoch=13,
        GNN_only_early_stop_authorized=True, CP_sources=642, CP_candidates_per_source=128,
        native_epochs=250, evaluation_cases=26, evaluation_checkpoint='checkpoint_best.pth',
        user_instruction='CPU_DEBUG explicit B-only downstream authorization')
    monitors = {}; states = {}
    for arm, (gpu, cpus) in resources.original.MAPPING.items():
        p = dict(path=str(transition/(arm+'_direct.json')), sha256=digest)
        if arm == 'B':
            p = seal['slot']['direct_proof']
        slot = dict(gpu=gpu, CPU_affinity=cpus, direct_proof=p,
            monitor_path=str(group/arm/'monitor/ALL_CPU_DEBUG/status.json'))
        prior['slots'][arm] = slot
        command = ['/home/aicompetition06/.conda/envs/nnunet/bin/python3.10', '-B', '-u',
            str(Path(prior['private_code'])/'tools/run_v25_all_arm_recovery_execution.py'),
            '--continuation-seal', seal['previous_all_seal']['path'], '--mode', 'train',
            '--arm', arm, '--gpu', str(gpu), '--output', str(group/arm), '--allocation-request', p['path']]
        actor = identity(801+ord(arm), command); request['actors'][arm] = actor
        monitors[slot['monitor_path']] = dict(status='RUNNING', stage='train', arm=arm,
            controller_pid=800, child_pid=actor['pid'], child_create_time=actor['create_time'],
            UID=41833, child_command=command)
        states[arm] = dict(epoch=14, phase='training', status='PAUSED', updates=222,
            completed_epochs=13, train_position=4, evaluation_position=0,
            BEST=dict(epoch=3), latest_path=str(group/arm/'training/checkpoint_latest.pt'))
        seal['preserved_files'][str(group/arm/'monitor/status.json')] = digest
        seal['preserved_files'][slot['monitor_path']] = digest
    release = dict(format='v25_user_B_only_checkpoint_stop_v1',
        status='PAUSED_ACD_B_READY_FOR_OWN_BEST', selected_downstream_arms=['B'],
        paused_arms=['A', 'C', 'D'], old_controller_released=True,
        registered_stage_actors_released=True, all_GNN_children_stopped_cooperatively=True,
        user_instruction=request['user_instruction'], states=states,
        signals=[dict(target=prior['controller_identity'], signals=['SIGSTOP', 'SIGTERM', 'SIGCONT'])])
    return seal, prior, request, release, monitors


class BEarlyResourceTests(unittest.TestCase):
    def test_only_original_B_GPU5_six_CPUs_and_full_downstream(self):
        seal, *_ = fixture(); self.assertIs(resources.validate_seal(seal), seal)
        for key, wrong in (('selected_arms', ['B', 'A']), ('GNN_early_stop_authorized', False),
                ('processes_signaled_by_this_adapter', True)):
            changed = copy.deepcopy(seal); changed[key] = wrong
            with self.subTest(key=key), self.assertRaises(ValueError):
                resources.validate_seal(changed)
        for key, wrong in (('gpu', 6), ('CPU_affinity', list(range(6)))):
            changed = copy.deepcopy(seal); changed['slot'][key] = wrong
            with self.subTest(key=key), self.assertRaises(ValueError):
                resources.validate_seal(changed)

    def test_original_sources_and_all_canonical_monitors_are_required(self):
        seal, *_ = fixture()
        for mode in ('source', 'monitor', 'best'):
            changed = copy.deepcopy(seal)
            if mode == 'source':
                changed['source_scientific23'].pop(next(iter(changed['source_scientific23'])))
            elif mode == 'monitor':
                changed['preserved_files'].pop(str(Path(changed['group_root'])/'A/monitor/status.json'))
            else:
                changed['preserved_files'].pop(str(Path(changed['group_root'])/'B/training/checkpoint_best.pt'))
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                resources.validate_seal(changed)

    def test_canonical_or_other_arm_monitor_overwrite_is_rejected(self):
        seal, *_ = fixture()
        for path in (Path(seal['group_root'])/'B/monitor/status.json',
                Path(seal['group_root'])/'A/monitor/new/status.json'):
            changed = copy.deepcopy(seal); changed['slot']['monitor_path'] = str(path)
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, 'no canonical overwrite'):
                resources.validate_seal(changed)

    def _release_patches(self, seal, prior, request, release, monitors):
        def read(proof):
            return request if proof == seal['authorization_request'] else release
        def owned(path):
            return SimpleNamespace(read_bytes=lambda: json.dumps(monitors[str(path)]).encode())
        return (patch.object(resources, 'previous_contract', return_value=prior),
            patch.object(resources, '_guard_proof', side_effect=read),
            patch.object(resources.direct, '_owned_regular', side_effect=owned),
            patch.object(resources, '_released'))

    def test_actual_partial_epoch_is_accepted_without_false_full40_claim(self):
        seal, prior, request, release, monitors = fixture()
        a, b, c, d = self._release_patches(seal, prior, request, release, monitors)
        with a, b, c, d as released:
            receipt = resources.verify_initial_release(seal)
            self.assertEqual(receipt['actual_B_GNN']['updates'], 222)
            self.assertEqual(receipt['actual_B_GNN']['completed_epochs'], 13)
            self.assertEqual(receipt['native_epochs'], 250)
            self.assertEqual(released.call_count, 5)
            self.assertFalse(receipt['processes_signaled_by_this_adapter'])

    def test_all_old_actor_and_controller_must_be_reaped(self):
        seal, prior, request, release, monitors = fixture()
        a, b, c, d = self._release_patches(seal, prior, request, release, monitors)
        with a, b, c, d as released:
            released.side_effect = RuntimeError('CPU_DEBUG old zombie not reaped')
            with self.assertRaisesRegex(RuntimeError, 'not reaped'):
                resources.verify_initial_release(seal)

    def test_release_cannot_hide_changed_actor_or_reduced_native_contract(self):
        for bad in ('actor', 'native', 'epoch', 'signal', 'status'):
            seal, prior, request, release, monitors = fixture()
            if bad == 'actor':
                request['actors']['C']['command'] = ['python', 'unrelated.py']
            elif bad == 'native':
                request['native_epochs'] = 2
            elif bad == 'epoch':
                release['states']['B'].update(completed_epochs=12, updates=204)
            elif bad == 'signal':
                release['signals'][0]['target'] = identity(999)
            else:
                release['states']['B']['status'] = 'RUNNING'
            a, b, c, d = self._release_patches(seal, prior, request, release, monitors)
            with self.subTest(bad=bad), a, b, c, d, self.assertRaises(ValueError):
                resources.verify_initial_release(seal)

    def test_digest_cache_hashes_once_and_checks_metadata_every_read(self):
        resources._IMMUTABLE.clear(); resources._EXPECTED.clear()
        (controller.ROOT/'tmp').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='CPU_DEBUG_B_witness_', dir=controller.ROOT/'tmp') as name:
            path = Path(name)/'immutable.pt'; path.write_bytes(b'actual CPU debug bytes')
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            with patch.object(resources.direct, '_owned_regular', side_effect=Path), \
                    patch.object(resources, 'sha', wraps=resources.sha) as hashed:
                self.assertEqual(resources.immutable_digest(path, digest), digest)
                self.assertEqual(resources.immutable_digest(path, digest), digest)
                self.assertEqual(hashed.call_count, 1)
                path.write_bytes(b'changed CPU debug bytes!')
                with self.assertRaisesRegex(ValueError, 'metadata changed'):
                    resources.immutable_digest(path, digest)
                self.assertEqual(hashed.call_count, 1)

    def test_digest_cache_rejects_replacement_and_new_expected_digest(self):
        resources._IMMUTABLE.clear(); resources._EXPECTED.clear()
        (controller.ROOT/'tmp').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='CPU_DEBUG_B_replace_', dir=controller.ROOT/'tmp') as name:
            path = Path(name)/'input.json'; path.write_bytes(b'original bytes')
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            with patch.object(resources.direct, '_owned_regular', side_effect=Path):
                resources.immutable_digest(path, digest)
                with self.assertRaisesRegex(ValueError, 'different expected'):
                    resources.immutable_digest(path, 'b'*64)
                new = Path(name)/'new.json'; new.write_bytes(path.read_bytes()); new.replace(path)
                with self.assertRaisesRegex(ValueError, 'metadata changed'):
                    resources.immutable_digest(path, digest)

    def test_gpu_query_admits_only_own_B_stage_while_RAM_keeps_controller_tree(self):
        lease = object.__new__(resources.BEarlyDownstreamAllocationLease)
        tree = {900: Mock(), 901: Mock(), 902: Mock()}; stage = {901: tree[901]}
        with patch.object(resources, '_stage_tree', return_value=stage), \
                patch.object(resources._BASE, '_gpu') as query:
            lease._gpu(tree, initial=False)
        query.assert_called_once_with(lease, stage, initial=False)
        self.assertEqual(set(tree), {900, 901, 902})

    def test_unregistered_or_changed_stage_cannot_own_CUDA(self):
        seal, *_ = fixture()
        command = ['/home/aicompetition06/.conda/envs/nnunet/bin/python3.10', '-B', '-u',
            str(Path(seal['private_code'])/resources.ENTRY), '--controller-seal', '/CPU_DEBUG/seal.json',
            '--early-request', seal['scientific_request']['path'], '--mode', 'score-cp', '--arm', 'B',
            '--gpu', '5', '--output', str(Path(seal['group_root'])/'B'),
            '--allocation-request', seal['slot']['direct_proof']['path']]
        with patch.dict(resources.os.environ, {resources.ENV_PATH: '/CPU_DEBUG/seal.json'}):
            resources._stage_binding(identity(901, command), seal, 'scoring')
            bad = list(command); bad[bad.index('--arm')+1] = 'A'
            with self.assertRaisesRegex(RuntimeError, 'argument binding'):
                resources._stage_binding(identity(901, bad), seal, 'scoring')
            bad = list(command); bad[bad.index('--mode')+1] = 'train'
            with self.assertRaisesRegex(RuntimeError, 'full642x128'):
                resources._stage_binding(identity(901, bad), seal, 'scoring')

    def test_original_request_bytes_and_lifetime_remain_unchanged_when_rebound(self):
        seal, prior, *_ = fixture()
        declaration = dict(controller_identity=prior['controller_identity'], arm='B', gpu=5,
            CPU_affinity=resources.CPUS, aggregate_CPU_affinity=list(range(24)),
            cgroup_admission={'CPU_DEBUG_old': True}, host_lifetime={'expires_at_unix': 999999.},
            aggregate_native_and_cache_RSS_bytes=416*2**30)
        original = copy.deepcopy(declaration)
        lease = object.__new__(resources.BEarlyDownstreamAllocationLease)
        lease.request = declaration; lease.path = Path(seal['slot']['direct_proof']['path'])
        lease.group = Path(seal['group_root']); lease.last_verified = None
        with patch.dict(resources.os.environ, {resources.ENV_PATH: '/CPU_DEBUG/seal.json', resources.ENV_SHA: 'a'*64}), \
                patch.object(resources, 'load_seal', return_value=seal), \
                patch.object(resources, 'previous_contract', return_value=prior), \
                patch.object(resources, 'verify_initial_release') as release, \
                patch.object(resources, 'immutable_digest', return_value='a'*64), \
                patch.object(resources.Path, 'read_bytes', return_value=b'CPU_DEBUG immutable seal bytes'), \
                patch.object(resources._BASE, 'verify', return_value={'CPU_DEBUG_verified': True}):
            lease.verify(force=True); lease.verify(force=True)
        self.assertEqual(declaration, original)
        self.assertEqual(lease.request['controller_identity'], seal['controller_identity'])
        self.assertEqual(lease.request['aggregate_CPU_affinity'], resources.CPUS)
        self.assertEqual(lease.request['host_lifetime'], original['host_lifetime'])
        self.assertEqual(lease.request['aggregate_native_and_cache_RSS_bytes'], 416*2**30)
        release.assert_called_once_with(seal)

    def test_original_stage_RSS_limits_count_all_owned_descendants(self):
        seal, *_ = fixture(); lease = object.__new__(resources.BEarlyDownstreamAllocationLease)
        lease.actual_resources = {}
        process = Mock(); process.status.return_value = 'running'
        for stage, rss in (('scoring', 65*2**30), ('native', 97*2**30)):
            process.memory_info.return_value = SimpleNamespace(rss=rss)
            file = SimpleNamespace(read_bytes=lambda: json.dumps(dict(stage=stage)).encode())
            with self.subTest(stage=stage), patch.object(resources._BASE, '_resources'), \
                    patch.object(resources, '_stage_tree', return_value={901: process}), \
                    patch.object(resources, 'load_seal', return_value=seal), \
                    patch.object(resources.direct, '_owned_regular', return_value=file), \
                    self.assertRaisesRegex(MemoryError, 'original64/96GiB'):
                lease._resources(Mock(), Mock(), {901: process})

    def test_registered_monitor_cannot_disappear_after_observation(self):
        seal, *_ = fixture(); lease = SimpleNamespace(_stage_monitor_observed=False)
        with patch.object(resources, 'load_seal', return_value=seal), \
                patch.object(resources.Path, 'exists', return_value=False):
            self.assertEqual(resources._stage_tree(lease, {}), {})
            lease._stage_monitor_observed = True
            with self.assertRaisesRegex(RuntimeError, 'monitor disappeared'):
                resources._stage_tree(lease, {})


class BEarlyControllerTests(unittest.TestCase):
    def test_successful_exit_needs_saved_completion_before_next_stage(self):
        declarations = [dict(name='scoring'), dict(name='native')]; events = []
        def execute(item):
            events.append(('execute', item['name']))
        def verify(stage):
            events.append(('verify', stage)); return dict(actual=stage)
        result = controller.run_chain(declarations, execute, verify)
        self.assertEqual(events, [('execute', 'scoring'), ('verify', 'scoring'),
            ('execute', 'native'), ('verify', 'native')])
        self.assertEqual(set(result), {'scoring', 'native'})

    def test_failed_or_partial_CP_cannot_start_native(self):
        declarations = [dict(name='scoring'), dict(name='native')]; executed = []
        with self.assertRaisesRegex(RuntimeError, '642 missing'):
            controller.run_chain(declarations, lambda item: executed.append(item['name']),
                lambda stage: (_ for _ in ()).throw(RuntimeError('CPU_DEBUG 642 missing')))
        self.assertEqual(executed, ['scoring'])
        with self.assertRaises(ValueError):
            controller.run_chain([dict(name='train'), dict(name='native')], Mock(), Mock())

    def test_commands_use_own_B_and_fresh_CP_overlay_with_no_GNN_restart(self):
        seal, *_ = fixture(); group = Path(seal['group_root'])
        args = SimpleNamespace(group=group, execution_manifest=group/'newseal.json')
        for key in ('native_experiment', 'inventory', 'stunet_checkpoint', 'input_cache',
                'storage_admission', 'source_bank', 'baseline_preprocessed', 'project',
                'basic_validation', 'evaluator_root'):
            setattr(args, key, group/key)
        cp_output = Path(seal['authorization_request']['path']).parent/'cp_bank'
        with patch.object(controller.Path, 'read_bytes', return_value=json.dumps(dict(cp_output=str(cp_output))).encode()):
            scoring = controller.stage_command(args, seal); native = controller.native_command(args, seal)
        for command in (scoring, native):
            self.assertEqual(command[3], str(controller.ROOT/'tools/run_v25_B_early_downstream_execution.py'))
            self.assertEqual(command[command.index('--arm')+1], 'B')
            self.assertEqual(command[command.index('--gpu')+1], '5')
            self.assertEqual(command[command.index('--output')+1], str(group/'B'))
            self.assertNotIn('--resume', command); self.assertNotIn('--debug', command)
        self.assertEqual(scoring[scoring.index('--mode')+1], 'score-cp')
        self.assertEqual(native[4], '--native')
        self.assertEqual(native[native.index('--score-overlay')+1], str(cp_output/'score_overlay.json'))


if __name__ == '__main__':
    unittest.main()
