"""Exact B-only resource rebinding after the authorized cooperative GNN stop.

The original direct proof and scientific files remain immutable. Only the
new controller identity, its six CPUs and its observed cgroup are rebound
in memory. The original host-lifetime and 416GiB admission stay in force.
This module never sends a process signal or writes an old monitor.
"""
from __future__ import annotations
import copy
import json
import os
from pathlib import Path

from . import v25_direct_resources as direct
from . import v25_all_arm_recovery_resources as previous
from . import v25_u128_reuse_resources as original
from . import v25_standalone_ad_resources as ad
from . import v25_terminal_descendant_resources as terminal

FORMAT = 'v25_exact_B_authorized_early_downstream_v1'
ENV_PATH = 'V25_B_EARLY_CONTROLLER_SEAL'
ENV_SHA = 'V25_B_EARLY_CONTROLLER_SEAL_SHA256'
CPUS = list(range(42, 48))
ENTRY = 'tools/run_v25_B_early_downstream_execution.py'
_BASE = direct.DirectAllocationLease
sha = original.sha
_guard_proof = original._guard_proof
_IMMUTABLE = {}
_EXPECTED = {}
_FIELDS = {'format', 'selected_arms', 'group_root', 'private_code', 'controller_identity',
    'controller_cgroup_admission', 'previous_all_seal', 'authorization_request',
    'release_receipt', 'scientific_request', 'slot', 'preserved_files', 'runtime_files',
    'source_scientific23', 'created_at', 'scientific_configuration_unchanged',
    'GNN_early_stop_authorized', 'processes_signaled_by_this_adapter'}


def _stat_identity(path):
    value = Path(path).stat()
    return tuple(getattr(value, key) for key in
        ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns', 'st_nlink'))


def immutable_digest(path, expected):
    """Hash once, then reject any inode/content-metadata change on each read."""
    path = direct._owned_regular(path); key = str(path)
    if not direct._checksum(expected):
        raise ValueError('An exact immutable SHA256 witness is required')
    prior = _EXPECTED.get(key)
    if prior is not None and prior != expected:
        raise ValueError('A sealed file acquired a different expected SHA256: ' + key)
    before = _stat_identity(path)
    if key in _IMMUTABLE:
        if _IMMUTABLE[key] != before:
            raise ValueError('Immutable file inode/content metadata changed: ' + key)
        return expected
    if sha(path) != expected or _stat_identity(path) != before:
        raise ValueError('Immutable file bytes changed during initial admission: ' + key)
    _EXPECTED[key] = expected; _IMMUTABLE[key] = before
    return expected


def validate_seal(value):
    if (not isinstance(value, dict) or set(value) != _FIELDS or value['format'] != FORMAT
            or value['selected_arms'] != ['B'] or value['scientific_configuration_unchanged'] is not True
            or value['GNN_early_stop_authorized'] is not True
            or value['processes_signaled_by_this_adapter'] is not False
            or not direct._identity_document(value['controller_identity'])):
        raise ValueError('Exact authorized B-only controller declaration required')
    group = Path(value['group_root']); code = Path(value['private_code'])
    if (not all(p.is_absolute() and '..' not in p.parts for p in (group, code))
            or code.parent != group):
        raise ValueError('Fresh B-only private code inside the original group required')
    for key in ('previous_all_seal', 'authorization_request', 'release_receipt', 'scientific_request'):
        if not original._proof_reference(value[key], group):
            raise ValueError('Exact immutable previous/authorization/release/scientific proof required')
    slot = value['slot']
    if (not isinstance(slot, dict) or set(slot) != {'arm', 'gpu', 'CPU_affinity', 'direct_proof', 'monitor_path'}
            or slot['arm'] != 'B' or slot['gpu'] != 5 or slot['CPU_affinity'] != CPUS
            or not original._proof_reference(slot['direct_proof'], group)):
        raise ValueError('Only original B GPU5 CPU42..47 may continue')
    monitor = Path(slot['monitor_path'])
    transition = Path(value['authorization_request']['path']).parent
    allowed_monitor = (monitor.parent.is_relative_to(group/'B/monitor')
        and monitor.parent != group/'B/monitor') or monitor.parent == transition/'execution/monitor'
    if (not monitor.is_absolute() or '..' in monitor.parts or monitor.name != 'status.json'
            or not allowed_monitor):
        raise ValueError('Fresh B-only monitor namespace required; no canonical overwrite')
    for table in ('preserved_files', 'runtime_files', 'source_scientific23'):
        if (not isinstance(value[table], dict) or not value[table]
                or any(not isinstance(p, str) or not Path(p).is_absolute() or '..' in Path(p).parts
                    or not direct._checksum(h) for p, h in value[table].items())):
            raise ValueError('Absolute immutable source/input/runtime receipts required')
    if (len(value['source_scientific23']) != 23
            or any(not Path(p).is_relative_to(code) or value['preserved_files'].get(p) != h
                for p, h in value['source_scientific23'].items())
            or not direct._checksum(value['runtime_files'].get(str(code/ENTRY)))):
        raise ValueError('Original23 source digests and exact new execution entry required')
    for arm in 'ABCD':
        if not direct._checksum(value['preserved_files'].get(str(group/arm/'monitor/status.json'))):
            raise ValueError('Every canonical historical monitor must remain sealed')
    for proof in (value['previous_all_seal'], value['authorization_request'], value['release_receipt'],
            value['scientific_request'], slot['direct_proof']):
        if value['preserved_files'].get(proof['path']) != proof['sha256']:
            raise ValueError('Every exact handoff proof must be in the preserved digest table')
    for name in ('checkpoint_latest.pt', 'checkpoint_best.pt'):
        if not direct._checksum(value['preserved_files'].get(str(group/'B/training'/name))):
            raise ValueError('Own released B latest and winning BEST checkpoint bytes must be pinned')
    admission = value['controller_cgroup_admission']
    if (not isinstance(admission, dict) or set(admission) != {'controller_cgroup', 'memory_limits'}
            or not isinstance(admission['controller_cgroup'], str) or not admission['controller_cgroup']
            or not isinstance(admission['memory_limits'], list) or not admission['memory_limits']):
        raise ValueError('Fresh observed controller cgroup and inherited416GiB limit required')
    for row in admission['memory_limits']:
        if (not isinstance(row, dict) or set(row) != {'path', 'limit_bytes'}
                or not isinstance(row['path'], str) or not row['path'].startswith('/sys/fs/cgroup/')
                or row['limit_bytes'] is not None and (type(row['limit_bytes']) is not int
                    or row['limit_bytes'] < 416*2**30)):
            raise ValueError('Fresh controller cannot exceed an actual cgroup memory limit')
    return value


def load_seal(path=None):
    path = direct._owned_regular(path or os.environ[ENV_PATH])
    immutable_digest(path, os.environ.get(ENV_SHA))
    value = validate_seal(json.loads(path.read_bytes()))
    for table in ('preserved_files', 'runtime_files'):
        for name, digest in value[table].items():
            immutable_digest(name, digest)
    return value


def previous_contract(seal):
    prior = previous.validate_seal(_guard_proof(seal['previous_all_seal']))
    if (prior['group_root'] != seal['group_root'] or prior['private_code'] == seal['private_code']
            or prior['controller_identity']['pid'] == seal['controller_identity']['pid']
            or prior['slots']['B']['direct_proof'] != seal['slot']['direct_proof']):
        raise ValueError('Exact original ALL lineage and unchanged own B direct proof required')
    # A finished training run legitimately changed latest checkpoint bytes.
    # Admit its original seal structurally, then check only immutable runtime
    # and scientific files; the new seal pins the actual final PAUSED bytes.
    previous.previous_contracts(prior)
    for name, digest in prior['runtime_files'].items():
        immutable_digest(name, digest)
    for path, digest in seal['source_scientific23'].items():
        old = str(Path(prior['private_code'])/Path(path).relative_to(seal['private_code']))
        if prior['source_scientific23'].get(old) != digest:
            raise ValueError('Original23 scientific source bytes differ from prior ALL admission')
        immutable_digest(old, digest)
    for arm, slot in prior['slots'].items():
        if not direct._checksum(seal['preserved_files'].get(slot['monitor_path'])):
            raise ValueError('Every original ALL stage monitor must remain sealed unchanged')
        if arm == 'B' and slot['monitor_path'] == seal['slot']['monitor_path']:
            raise ValueError('New B monitor cannot overwrite the prior ALL monitor')
    return prior


def _released(identity):
    """The stop guardian reaps all actors; zombies never admit new CUDA."""
    original._released(identity)


def verify_initial_release(seal):
    prior = previous_contract(seal)
    request = _guard_proof(seal['authorization_request'])
    release = _guard_proof(seal['release_receipt'])
    group = Path(seal['group_root'])
    if (request.get('group_root') != str(group)
            or request.get('original_seal_path') != seal['previous_all_seal']['path']
            or request.get('controller_identity') != prior['controller_identity']
            or request.get('GNN_only_early_stop_authorized') is not True
            or request.get('CP_sources') != 642 or request.get('CP_candidates_per_source') != 128
            or request.get('native_epochs') != 250 or request.get('evaluation_cases') != 26
            or request.get('evaluation_checkpoint') != 'checkpoint_best.pth'
            or not isinstance(request.get('user_instruction'), str) or not request['user_instruction']
            or set(request.get('actors', {})) != set('ABCD')
            or type(request.get('B_finish_epoch')) is not int or not 1 <= request['B_finish_epoch'] <= 40):
        raise ValueError('Actual explicit B-only early-stop authorization and full downstream contract required')
    if (release.get('format') != 'v25_user_B_only_checkpoint_stop_v1'
            or release.get('status') != 'PAUSED_ACD_B_READY_FOR_OWN_BEST'
            or release.get('selected_downstream_arms') != ['B'] or release.get('paused_arms') != ['A', 'C', 'D']
            or release.get('old_controller_released') is not True
            or release.get('registered_stage_actors_released') is not True
            or release.get('all_GNN_children_stopped_cooperatively') is not True
            or release.get('user_instruction') != request['user_instruction']
            or set(release.get('states', {})) != set('ABCD')
            or release.get('signals') != [dict(target=prior['controller_identity'],
                signals=['SIGSTOP', 'SIGTERM', 'SIGCONT'])]):
        raise ValueError('Exact cooperative PAUSED/released handoff receipt required')
    for arm in 'ABCD':
        identity = request['actors'][arm]
        if not direct._identity_document(identity):
            raise ValueError('Exact registered old own arm actor required')
        monitor = json.loads(direct._owned_regular(prior['slots'][arm]['monitor_path']).read_bytes())
        actor = dict(pid=monitor.get('child_pid'), create_time=monitor.get('child_create_time'),
            UID=monitor.get('UID'), command=monitor.get('child_command'))
        if (actor != identity or monitor.get('arm') != arm or monitor.get('stage') != 'train'
                or monitor.get('controller_pid') != prior['controller_identity']['pid']):
            raise ValueError('Frozen original ALL own training monitor/actor mismatch')
        command = identity['command']; slot = prior['slots'][arm]
        if (command[:4] != ['/home/aicompetition06/.conda/envs/nnunet/bin/python3.10', '-B', '-u',
                str(Path(prior['private_code'])/'tools/run_v25_all_arm_recovery_execution.py')]
                or any(ad._flag(command, flag) != expected for flag, expected in (
                    ('--arm', arm), ('--gpu', str(slot['gpu'])), ('--mode', 'train'),
                    ('--output', str(group/arm)), ('--continuation-seal', seal['previous_all_seal']['path']),
                    ('--allocation-request', slot['direct_proof']['path'])))):
            raise ValueError('Exact original ALL arm invocation required')
        state = release['states'][arm]
        if (state.get('status') != 'PAUSED' or state.get('latest_path') != str(group/arm/'training/checkpoint_latest.pt')
                or any(type(state.get(k)) is not int for k in
                    ('epoch', 'updates', 'completed_epochs', 'train_position', 'evaluation_position'))
                or not 0 <= state['completed_epochs'] <= 40 or not 1 <= state['epoch'] <= 41
                or not state['completed_epochs']*17 <= state['updates'] <= min(680, (state['completed_epochs']+1)*17)
                or not 0 <= state['train_position'] <= 65 or not 0 <= state['evaluation_position'] <= 65
                or state.get('phase') not in ('initial_full_validation', 'training', 'train_probe',
                    'stage_validation', 'full_validation', 'epoch_completion', 'complete')):
            raise ValueError('Actual full-population PAUSED cursor/update receipt required')
        _released(identity)
    if release['states']['B']['completed_epochs'] < request['B_finish_epoch']:
        raise ValueError('B did not complete the explicitly selected current epoch')
    _released(prior['controller_identity'])
    return dict(selected_downstream_arms=['B'], paused_arms=['A', 'C', 'D'],
        previous_controller_identity=prior['controller_identity'], previous_actors=request['actors'],
        actual_B_GNN=release['states']['B'], native_epochs=250, evaluated_cases=26,
        evaluated_checkpoint='checkpoint_best.pth', processes_signaled_by_this_adapter=False)


def _stage_binding(identity, seal, stage):
    command = identity['command']; slot = seal['slot']
    if command[:4] != ['/home/aicompetition06/.conda/envs/nnunet/bin/python3.10', '-B', '-u',
            str(Path(seal['private_code'])/ENTRY)]:
        raise RuntimeError('Exact sealed B-only downstream execution entry required')
    if any(ad._flag(command, key) != expected for key, expected in (
            ('--arm', 'B'), ('--gpu', '5'), ('--output', str(Path(seal['group_root'])/'B')),
            ('--allocation-request', slot['direct_proof']['path']), ('--controller-seal', os.environ[ENV_PATH]),
            ('--early-request', seal['scientific_request']['path']))):
        raise RuntimeError('Own B-only downstream actor argument binding changed')
    if stage == 'scoring':
        if '--native' in command or ad._flag(command, '--mode') != 'score-cp':
            raise RuntimeError('Only full642x128 scoring may run before native')
    elif stage != 'native' or command[4:5] != ['--native']:
        raise RuntimeError('Only exact own scoring/native actors may own GPU5')


def _stage_tree(lease, tree):
    seal = load_seal(); path = Path(seal['slot']['monitor_path'])
    if not path.exists():
        if getattr(lease, '_stage_monitor_observed', False):
            raise RuntimeError('Observed B-only monitor disappeared')
        return {}
    value = json.loads(direct._owned_regular(path).read_bytes()); lease._stage_monitor_observed = True
    if value.get('controller_pid') != seal['controller_identity']['pid'] or value.get('arm') != 'B':
        raise RuntimeError('Exact new B-only monitor controller/arm changed')
    if value.get('status') == 'COMPLETE' and value.get('stage') == 'complete':
        return {}
    actor = dict(pid=value.get('child_pid'), create_time=value.get('child_create_time'),
        UID=value.get('UID'), command=value.get('child_command'))
    if not direct._identity_document(actor):
        raise RuntimeError('Exact B-only monitored actor identity required')
    _stage_binding(actor, seal, value.get('stage'))
    try:
        root = direct._exact(actor)
    except direct.psutil.NoSuchProcess:
        return {}
    parent = root.parent()
    if parent is None or root.ppid() != seal['controller_identity']['pid'] or direct._actor(parent) != seal['controller_identity']:
        raise RuntimeError('B-only stage is outside its exact fresh controller tree')
    original._remember_live(lease, root, actor, tree, CPUS); selected = {root.pid: root}
    for child in root.children(recursive=True):
        try:
            actual = original._current_descendant(lease, child, actor, tree, CPUS)
        except direct.psutil.NoSuchProcess:
            continue
        if actual is not None:
            selected.update(actual)
    return selected


class BEarlyDownstreamAllocationLease(_BASE):
    def verify(self, *, force=False):
        if not getattr(self, '_B_rebound', False):
            seal = load_seal(); prior = previous_contract(seal)
            if (str(self.path) != seal['slot']['direct_proof']['path']
                    or immutable_digest(self.path, seal['slot']['direct_proof']['sha256']) != seal['slot']['direct_proof']['sha256']
                    or self.request['controller_identity'] != prior['controller_identity']
                    or self.request['arm'] != 'B' or self.request['gpu'] != 5
                    or self.request['CPU_affinity'] != CPUS or str(self.group) != seal['group_root']):
                raise ValueError('Original immutable own B direct declaration required before rebinding')
            verify_initial_release(seal)
            self.request = copy.deepcopy(self.request)
            self.request.update(controller_identity=seal['controller_identity'], aggregate_CPU_affinity=CPUS,
                cgroup_admission=seal['controller_cgroup_admission'])
            self._B_seal = seal
            self._B_seal_bytes = Path(os.environ[ENV_PATH]).read_bytes()
            self._B_rebound = True
        if (Path(os.environ[ENV_PATH]).read_bytes() != self._B_seal_bytes
                or immutable_digest(os.environ[ENV_PATH], os.environ.get(ENV_SHA)) != os.environ.get(ENV_SHA)):
            raise ValueError('Immutable own B controller admission changed')
        # Check large immutable checkpoint/source bytes at the same force/poll
        # cadence as qstat/GPU/RAM admission, not once per encoded candidate.
        now = direct.time.monotonic()
        seal = (load_seal() if force or self.last_verified is None
            or now-self.last_verified >= direct.POLL_SECONDS else self._B_seal)
        if (self.request['controller_identity'] != seal['controller_identity']
                or self.request['aggregate_CPU_affinity'] != CPUS
                or self.request['cgroup_admission'] != seal['controller_cgroup_admission']):
            raise ValueError('B-only fresh controller binding changed')
        return _BASE.verify(self, force=force)

    def _resources(self, controller, own, tree):
        _BASE._resources(self, controller, own, tree)
        selected = _stage_tree(self, tree)
        if selected:
            monitor = json.loads(direct._owned_regular(load_seal()['slot']['monitor_path']).read_bytes())
            limit = (96 if monitor['stage'] == 'native' else 64)*2**30
            rss = 0
            for process in selected.values():
                try:
                    if process.status() != direct.psutil.STATUS_ZOMBIE:
                        rss += process.memory_info().rss
                except direct.psutil.NoSuchProcess:
                    continue
            if rss > limit:
                raise MemoryError('Exact B-only stage tree exceeds its original64/96GiB RSS budget')
            self.actual_resources.update(own_B_stage_RSS_bytes=rss, own_B_stage_RSS_limit_bytes=limit)
        self.actual_resources.update(selected_arms=['B'], original_direct_proof_modified=False,
            original_monitors_modified=False, GPU0_processes_modified=False,
            processes_signaled_by_this_adapter=False, GNN_full40_claimed=False)

    def _gpu(self, tree, *, initial):
        before = set(tree); selected = _stage_tree(self, tree)
        if set(tree) != before:
            self._resources(direct._exact(self.request['controller_identity']), direct._exact(self.self_identity), tree)
        return _BASE._gpu(self, selected, initial=initial)


def install_from_environment():
    seal = load_seal()
    if direct.DirectAllocationLease not in (_BASE, BEarlyDownstreamAllocationLease):
        raise ValueError('Unknown B-only direct-resource class override')
    direct.DirectAllocationLease = BEarlyDownstreamAllocationLease
    terminal.install_runtime()
    return dict(format=FORMAT, source_sha256=sha(Path(__file__)), seal_sha256=os.environ[ENV_SHA],
        selected_arms=['B'], physical_GPU=5, CPU_affinity=CPUS, aggregate_RSS_bytes=416*2**30,
        GNN_scoring_RSS_bytes=64*2**30, native_RSS_bytes=96*2**30,
        inherited_host_lifetime=True, original_direct_proof_modified=False,
        original_monitors_modified=False, GPU0_processes_modified=False,
        processes_signaled_by_this_adapter=False, controller_identity=seal['controller_identity'])
