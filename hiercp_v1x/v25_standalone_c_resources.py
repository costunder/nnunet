"""Resume exact failed C without replacing the original monitor used by ABD.

The new C controller accounts its own tree and the unchanged ABD tree together.
Only its exact C stage can use GPU1. No process is signalled by this module.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
from types import SimpleNamespace

from . import v25_direct_resources as direct
from . import v25_terminal_descendant_resources as terminal
from . import v25_u128_reuse_resources as original
from . import v25_standalone_ad_resources as ad

FORMAT = 'v25_exact_C_standalone_continuation_v1'
ENV_PATH = 'V25_C_CONTINUATION_SEAL'
ENV_SHA = 'V25_C_CONTINUATION_SEAL_SHA256'
MAPPING = {'C': (1, list(range(16, 22)))}
_BASE = direct.DirectAllocationLease
_FIELDS = {'format', 'selected_arms', 'group_root', 'private_code', 'previous_code',
    'controller_identity', 'previous_controller_identity', 'previous_seal',
    'peer_controller_identity', 'peer_seal', 'slots', 'preserved_files',
    'runtime_files', 'created_at', 'scientific_configuration_unchanged', 'no_process_signaled'}
sha = original.sha
_guard_proof = original._guard_proof


def validate_seal(value):
    if (not isinstance(value, dict) or set(value) != _FIELDS
            or value['format'] != FORMAT or value['selected_arms'] != ['C']
            or set(value['slots']) != {'C'}
            or value['scientific_configuration_unchanged'] is not True
            or value['no_process_signaled'] is not True
            or not all(direct._identity_document(value[key]) for key in (
                'controller_identity', 'previous_controller_identity', 'peer_controller_identity'))
            or len({value[key]['pid'] for key in ('controller_identity',
                'previous_controller_identity', 'peer_controller_identity')}) != 3):
        raise ValueError('Exact separate C, original C and unchanged ABD controller identities required')
    group = Path(value['group_root']); code = Path(value['private_code']); old = Path(value['previous_code'])
    if (not all(p.is_absolute() and '..' not in p.parts for p in (group, code, old))
            or code.parent != group or old.parent != group or code == old):
        raise ValueError('Fresh private C code beside the original immutable code required')
    if not all(original._proof_reference(value[key], group) for key in ('previous_seal', 'peer_seal')):
        raise ValueError('Immutable original U128 and unchanged ABD seals required')
    for table in ('preserved_files', 'runtime_files'):
        if (not isinstance(value[table], dict) or not value[table]
                or any(not isinstance(p, str) or not direct._checksum(h)
                    for p, h in value[table].items())):
            raise ValueError('Immutable scientific/input/execution admissions required')
    slot = value['slots']['C']
    if (not isinstance(slot, dict) or set(slot) != {'arm', 'gpu', 'CPU_affinity', 'direct_proof',
            'checkpoint', 'previous_actor', 'failed_monitor', 'failure_log', 'monitor_path'}
            or slot['arm'] != 'C' or slot['gpu'] != 1 or slot['CPU_affinity'] != MAPPING['C'][1]
            or not direct._identity_document(slot['previous_actor'])
            or not all(original._proof_reference(slot[k], group)
                for k in ('direct_proof', 'failed_monitor', 'failure_log'))):
        raise ValueError('Exact failed C GPU1/six-core actor and immutable receipts required')
    monitor = Path(slot['monitor_path'])
    if (not monitor.is_absolute() or '..' in monitor.parts or monitor.name != 'status.json'
            or not monitor.parent.is_relative_to(group/'C/monitor')
            or monitor.parent == group/'C/monitor'):
        raise ValueError('Fresh C monitor subdirectory required; canonical FAILED monitor must stay unchanged')
    canonical = str(group/'C/monitor/status.json')
    if not direct._checksum(value['preserved_files'].get(canonical)):
        raise ValueError('The exact canonical original C FAILED monitor must be sealed unchanged')
    cp = slot['checkpoint']
    if (not isinstance(cp, dict) or set(cp) != {'path', 'sha256', 'identity_sha256', 'content_sha256',
            'epoch', 'updates', 'phase', 'train_position', 'evaluation_position', 'active_u', 'status'}
            or cp['path'] != str(group/'C/training/checkpoint_latest.pt')
            or not all(direct._checksum(cp[k]) for k in ('sha256', 'identity_sha256', 'content_sha256'))
            or any(type(cp[k]) is not int for k in
                ('epoch', 'updates', 'train_position', 'evaluation_position', 'active_u'))
            or not 1 <= cp['epoch'] <= 40 or not 0 <= cp['updates'] <= 680
            or not 0 <= cp['train_position'] <= 65
            or cp['phase'] not in ('initial_full_validation', 'training', 'train_probe',
                'stage_validation', 'full_validation', 'epoch_completion')
            or not 0 <= cp['evaluation_position'] <= (65 if cp['phase'] == 'train_probe' else 21)
            or cp['active_u'] not in range(32, 129, 32) or cp['status'] not in ('RUNNING', 'PAUSED')):
        raise ValueError('Own complete full40/680/B4 optimizer/RNG checkpoint receipt required')
    return value


def load_seal(path=None):
    path = direct._owned_regular(path or os.environ[ENV_PATH])
    if sha(path) != os.environ.get(ENV_SHA):
        raise ValueError('Immutable C continuation seal changed')
    value = validate_seal(json.loads(path.read_bytes()))
    for table in ('preserved_files', 'runtime_files'):
        for name, digest in value[table].items():
            if sha(direct._owned_regular(name)) != digest:
                raise ValueError('Sealed C source/input/old monitor changed: ' + name)
    return value


def _stage_binding(identity, seal, stage):
    slot = seal['slots']['C']; command = identity['command']
    if command[:4] != ['/home/aicompetition06/.conda/envs/nnunet/bin/python3.10', '-B', '-u',
            str(Path(seal['private_code'])/'tools/run_v25_c_execution.py')]:
        raise RuntimeError('Exact sealed C stage wrapper required')
    if any(ad._flag(command, key) != expected for key, expected in (
            ('--arm', 'C'), ('--gpu', '1'), ('--output', str(Path(seal['group_root'])/'C')),
            ('--continuation-seal', os.environ[ENV_PATH]),
            ('--allocation-request', slot['direct_proof']['path']))):
        raise RuntimeError('Own C stage argument binding changed')
    if stage in ('train', 'scoring'):
        if ad._flag(command, '--mode') != ('train' if stage == 'train' else 'score-cp'):
            raise RuntimeError('Exact own C train/scoring mode required')
    elif stage != 'native' or command[4:5] != ['--native']:
        raise RuntimeError('Only own monitored C train/scoring/native actors can own GPU1')


def _validate_failed_slot(slot, seal):
    monitor = _guard_proof(slot['failed_monitor'])
    actor = dict(pid=monitor.get('child_pid'), create_time=monitor.get('child_create_time'),
        UID=monitor.get('UID'), command=monitor.get('child_command'))
    if (monitor.get('status') != 'FAILED' or monitor.get('stage') != 'train'
            or monitor.get('controller_pid') != seal['previous_controller_identity']['pid']
            or actor != slot['previous_actor'] or monitor.get('arm') != 'C'
            or monitor.get('stage_log') != slot['failure_log']['path']):
        raise ValueError('Exact original C FAILED monitor, actor and log binding required')
    command = actor['command']
    if (command[:4] != ['/home/aicompetition06/.conda/envs/nnunet/bin/python3.10', '-B', '-u',
            str(Path(seal['previous_code'])/'tools/run_v25_u128_reuse_execution.py')]
            or any(ad._flag(command, key) != expected for key, expected in (
                ('--arm', 'C'), ('--gpu', '1'), ('--output', str(Path(seal['group_root'])/'C')),
                ('--mode', 'train'), ('--continuation-seal', seal['previous_seal']['path'])))):
        raise ValueError('Exact original released own C execution actor required')
    log = direct._owned_regular(slot['failure_log']['path'])
    if (sha(log) != slot['failure_log']['sha256']
            or 'RuntimeError: Exact live descendant PID/birth/argv/UID identity required' not in log.read_text()):
        raise ValueError('Actual original C empty-argv failure traceback required')
    original._released(actor)


def validate_own_contracts(lease):
    seal = load_seal(); slot = seal['slots']['C']
    if (lease.request['arm'] != 'C' or str(lease.path) != slot['direct_proof']['path']
            or sha(lease.path) != slot['direct_proof']['sha256']
            or lease.request['controller_identity'] != seal['controller_identity']
            or str(lease.group) != seal['group_root']):
        raise ValueError('Only exact own C proof and new separate controller are admitted')
    old = original.validate_seal(_guard_proof(seal['previous_seal']))
    peer = ad.validate_seal(_guard_proof(seal['peer_seal']))
    if (old['private_code'] != seal['previous_code'] or old['group_root'] != seal['group_root']
            or old['controller_identity'] != seal['previous_controller_identity']
            or peer['group_root'] != seal['group_root']
            or peer['controller_identity'] != seal['peer_controller_identity']
            or peer['peer_controller_identity'] != seal['previous_controller_identity']
            or peer['peer_seal'] != seal['previous_seal']):
        raise ValueError('Exact unchanged original C and continuing ABD contracts required')
    for contract in (old, peer):
        for name, digest in contract['runtime_files'].items():
            if sha(direct._owned_regular(name)) != digest:
                raise ValueError('Original C or continuing ABD runtime source changed')
    for arm in ('A', 'B', 'C', 'D'):
        proof = slot['direct_proof'] if arm == 'C' else peer['slots'][arm]['direct_proof']
        document = direct.validate_direct_document(_guard_proof(proof))
        controller = seal['controller_identity'] if arm == 'C' else seal['peer_controller_identity']
        gpu, cpus = original.MAPPING[arm]
        if (document['arm'] != arm or document['gpu'] != gpu or document['CPU_affinity'] != cpus
                or document['controller_identity'] != controller or document['group_root'] != seal['group_root']
                or any(document[key] != lease.request[key] for key in ('host_lifetime',
                    'aggregate_CPU_affinity', 'resource_guard_sha256', 'cgroup_admission'))):
            raise ValueError('Same complete four-arm RAM/lifetime/six-core direct contract required')
    _validate_failed_slot(slot, seal)
    return seal


def verify_initial_peer(seal):
    contract = ad.validate_seal(_guard_proof(seal['peer_seal']))
    if contract['controller_identity'] != seal['peer_controller_identity']:
        raise ValueError('Exact original ABD controller seal required before C launch')
    for name, digest in contract['runtime_files'].items():
        if sha(direct._owned_regular(name)) != digest:
            raise ValueError('Original ABD runtime source changed before C launch')
    observer = SimpleNamespace(group=Path(seal['group_root']), managed={})
    try:
        peer = ad._peer_controller(observer, seal['peer_controller_identity'])
    except direct.psutil.NoSuchProcess:
        peer = None
    if peer is None:
        _terminal_peer_monitors(observer, seal)
        return dict(ABD_controller_preserved=True, ABD_identity=seal['peer_controller_identity'],
            ABD_controller_terminal_with_verified_monitors=True,
            original_C_monitor_unchanged=True, no_process_signaled=True)
    proof = _guard_proof(contract['slots']['A']['direct_proof'])
    if (peer.cpu_affinity() != proof['aggregate_CPU_affinity']
            or direct.capture_cgroup_admission(peer.pid) != proof['cgroup_admission']):
        raise RuntimeError('Continuing initial ABD controller CPU/cgroup changed')
    return dict(ABD_controller_preserved=True, ABD_identity=seal['peer_controller_identity'],
        ABD_controller_terminal_with_verified_monitors=False,
        original_C_monitor_unchanged=True, no_process_signaled=True)


def _terminal_peer_monitors(lease, seal):
    peer = _guard_proof(seal['peer_seal'])
    for arm in ('A', 'B', 'D'):
        value = json.loads(direct._owned_regular(lease.group/arm/'monitor/status.json').read_bytes())
        if value.get('arm') != arm or value.get('controller_pid') != seal['peer_controller_identity']['pid']:
            raise RuntimeError('Terminal ABD controller requires every exact own terminal monitor')
        if value.get('status') == 'COMPLETE':
            if value.get('stage') != 'complete':
                raise RuntimeError('ABD completion requires its actual complete stage')
            from tools.run_v25_u128_reuse_continuation import verify_arm_completion
            verify_arm_completion(lease.group, arm)
        elif value.get('status') == 'FAILED':
            identity = dict(pid=value.get('child_pid'), create_time=value.get('child_create_time'),
                UID=value.get('UID'), command=value.get('child_command'))
            if not direct._identity_document(identity):
                raise RuntimeError('Exact released failed ABD actor required')
            ad._stage_binding(identity, arm, peer['slots'][arm]['gpu'], lease.group/arm,
                peer['private_code'], seal['peer_seal']['path'],
                peer['slots'][arm]['direct_proof']['path'], value.get('stage'))
            original._released(identity)
        else:
            raise RuntimeError('ABD controller vanished before every actual terminal monitor')


def _peer_tree(lease, seal, tree):
    # Reuse only the strict identity observation helper, never install ABD's lease.
    identity = seal['peer_controller_identity']; union = dict(tree)
    observed = getattr(lease, '_observed_ABD_descendants', None)
    if observed is None:
        observed = lease._observed_ABD_descendants = {}
    try:
        peer = ad._peer_controller(lease, identity)
    except direct.psutil.ZombieProcess as error:
        raise RuntimeError('Terminal ABD controller UID/birth could not be verified') from error
    except direct.psutil.NoSuchProcess:
        peer = None
    if peer is None:
        _terminal_peer_monitors(lease, seal)
        for old in observed.values():
            try:
                child = direct.psutil.Process(old['pid']); actual = terminal._descendant(lease, child)
            except direct.psutil.NoSuchProcess:
                continue
            if actual is None:
                continue
            if actual != old:
                raise RuntimeError('Surviving ABD child PID/birth/argv identity changed')
            if child.pid in union:
                raise RuntimeError('Separate C and ABD trees overlap')
            union[child.pid] = child
        return union
    if (peer.cpu_affinity() != lease.request['aggregate_CPU_affinity']
            or direct.capture_cgroup_admission(peer.pid) != lease.request['cgroup_admission']):
        raise RuntimeError('Continuing ABD controller CPU/cgroup changed')
    if peer.pid in union:
        raise RuntimeError('Separate C and ABD controller trees overlap')
    union[peer.pid] = peer
    for child in peer.children(recursive=True):
        try:
            actual = terminal._descendant(lease, child)
        except direct.psutil.NoSuchProcess:
            continue
        if actual is None:
            continue
        prior = lease.managed.get(child.pid)
        if prior is not None and prior != actual:
            raise RuntimeError('Continuing ABD descendant PID/birth/argv identity changed')
        if child.pid in union:
            raise RuntimeError('Separate C and ABD descendants overlap')
        lease.managed[child.pid] = actual; observed[child.pid] = actual; union[child.pid] = child
    return union


def _stage_tree(lease, tree):
    seal = load_seal(); slot = seal['slots']['C']; path = Path(slot['monitor_path'])
    if not path.exists():
        if getattr(lease, '_stage_monitor_observed', False):
            raise RuntimeError('Observed own C stage monitor disappeared')
        return {}
    value = json.loads(direct._owned_regular(path).read_bytes())
    lease._stage_monitor_observed = True
    if value.get('controller_pid') != lease.request['controller_identity']['pid']:
        raise RuntimeError('Own recovery C monitor controller changed')
    actor = dict(pid=value.get('child_pid'), create_time=value.get('child_create_time'),
        UID=value.get('UID'), command=value.get('child_command'))
    if value.get('status') == 'COMPLETE' and value.get('stage') == 'complete':
        return {}
    if not direct._identity_document(actor):
        raise RuntimeError('Exact own monitored C stage identity required')
    _stage_binding(actor, seal, value.get('stage'))
    try:
        root = direct._exact(actor)
    except direct.psutil.NoSuchProcess:
        return {}
    parent = root.parent()
    if (parent is None or root.ppid() != lease.request['controller_identity']['pid']
            or direct._actor(parent) != lease.request['controller_identity']):
        raise RuntimeError('Own C stage origin is outside its exact controller tree')
    original._remember_live(lease, root, actor, tree, slot['CPU_affinity'])
    selected = {root.pid: root}
    for child in root.children(recursive=True):
        try:
            actual = original._current_descendant(lease, child, actor, tree, slot['CPU_affinity'])
        except direct.psutil.NoSuchProcess:
            continue
        if actual is not None:
            selected.update(actual)
    return selected


class StandaloneCAllocationLease(_BASE):
    def _resources(self, controller, own, tree):
        seal = validate_own_contracts(self)
        union = _peer_tree(self, seal, tree)
        _BASE._resources(self, controller, own, union)
        self.actual_resources.update(selected_arms=['C'], own_C_tree_RSS_members=len(tree),
            combined_C_ABD_RSS_members=len(union), unchanged_ABD_tree_only_used_for_RAM=True,
            canonical_C_FAILED_monitor_modified=False, ABD_and_GPU0_processes_modified=False)

    def _gpu(self, tree, *, initial):
        before = set(tree); selected = _stage_tree(self, tree)
        if set(tree) != before:
            self._resources(direct._exact(self.request['controller_identity']),
                direct._exact(self.self_identity), tree)
        return _BASE._gpu(self, selected, initial=initial)


def install_from_environment():
    seal = load_seal()
    if direct.DirectAllocationLease not in (_BASE, StandaloneCAllocationLease):
        raise ValueError('Unknown C direct-resource class override')
    direct.DirectAllocationLease = StandaloneCAllocationLease
    terminal.install_runtime()
    return dict(format=FORMAT, source_sha256=sha(Path(__file__)), seal_sha256=os.environ[ENV_SHA],
        selected_arms=['C'], GNN_RSS_bytes_per_arm=64*2**30, aggregate_RSS_bytes=416*2**30,
        unchanged_ABD_tree_only_used_for_RAM=True, canonical_C_FAILED_monitor_modified=False,
        ABD_and_GPU0_processes_modified=False, controller_identity=seal['controller_identity'])
