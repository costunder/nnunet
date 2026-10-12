"""Resume only failed A/B/D; the exact unchanged C tree contributes RAM only."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path

from . import v25_direct_resources as direct
from . import v25_terminal_descendant_resources as terminal
from . import v25_u128_reuse_resources as original

FORMAT = 'v25_exact_ABD_standalone_continuation_v1'
ENV_PATH = 'V25_AD_CONTINUATION_SEAL'
ENV_SHA = 'V25_AD_CONTINUATION_SEAL_SHA256'
MAPPING = {'A': (6, list(range(32, 38))), 'B': (5, list(range(42, 48))),
    'D': (4, list(range(6)))}
_BASE = direct.DirectAllocationLease
_FIELDS = {'format', 'selected_arms', 'group_root', 'private_code', 'previous_code',
    'controller_identity', 'peer_seal', 'peer_controller_identity', 'peer_slots',
    'slots', 'preserved_files', 'runtime_files', 'created_at',
    'scientific_configuration_unchanged', 'no_process_signaled'}
sha = original.sha
_guard_proof = original._guard_proof


def command_sha(command):
    return hashlib.sha256(json.dumps(command, ensure_ascii=False,
        separators=(',', ':')).encode('utf8')).hexdigest()


def _proof(value, group):
    return original._proof_reference(value, group)


def validate_seal(value):
    if (not isinstance(value, dict) or set(value) != _FIELDS
            or value['format'] != FORMAT or value['selected_arms'] != ['A', 'B', 'D']
            or set(value['slots']) != set(MAPPING) or set(value['peer_slots']) != {'C'}
            or value['scientific_configuration_unchanged'] is not True
            or value['no_process_signaled'] is not True
            or not direct._identity_document(value['controller_identity'])
            or not direct._identity_document(value['peer_controller_identity'])
            or value['controller_identity']['pid'] == value['peer_controller_identity']['pid']):
        raise ValueError('Exact selected ABD controller and unchanged separate C controller required')
    group = Path(value['group_root']); code = Path(value['private_code']); old = Path(value['previous_code'])
    if (not all(p.is_absolute() and '..' not in p.parts for p in (group, code, old))
            or code.parent != group or old.parent != group or code == old):
        raise ValueError('Fresh ABD private code beside the unchanged C code required')
    if not _proof(value['peer_seal'], group):
        raise ValueError('Exact immutable previous U128 controller seal required')
    for table in ('preserved_files', 'runtime_files'):
        if (not isinstance(value[table], dict) or not value[table]
                or any(not isinstance(p, str) or not direct._checksum(h)
                    for p, h in value[table].items())):
            raise ValueError('Immutable scientific/input/execution admissions required')
    for arm, (gpu, cpus) in MAPPING.items():
        slot = value['slots'][arm]
        if (not isinstance(slot, dict) or set(slot) != {'arm', 'gpu', 'CPU_affinity',
                'direct_proof', 'checkpoint', 'previous_actor', 'failed_monitor', 'failure_log'}
                or slot['arm'] != arm or slot['gpu'] != gpu or slot['CPU_affinity'] != cpus
                or not direct._identity_document(slot['previous_actor'])
                or not all(_proof(slot[k], group) for k in ('direct_proof', 'failed_monitor', 'failure_log'))):
            raise ValueError('Exact failed own ABD actor, original GPU/six CPUs and immutable evidence required')
        cp = slot['checkpoint']
        if (not isinstance(cp, dict) or set(cp) != {'path', 'sha256', 'identity_sha256',
                'content_sha256', 'epoch', 'updates', 'phase', 'train_position',
                'evaluation_position', 'active_u', 'status'}
                or cp['path'] != str(group/arm/'training/checkpoint_latest.pt')
                or not all(direct._checksum(cp[k]) for k in ('sha256', 'identity_sha256', 'content_sha256'))
                or any(type(cp[k]) is not int for k in
                    ('epoch', 'updates', 'train_position', 'evaluation_position', 'active_u'))
                or not 1 <= cp['epoch'] <= 40 or not 0 <= cp['updates'] <= 680
                or not 0 <= cp['train_position'] <= 65
                or cp['phase'] not in ('initial_full_validation', 'training', 'train_probe',
                    'stage_validation', 'full_validation', 'epoch_completion')
                or not 0 <= cp['evaluation_position'] <= (65 if cp['phase'] == 'train_probe' else 21)
                or cp['active_u'] not in range(original.INCREMENTS[arm], 129, original.INCREMENTS[arm])
                or cp['status'] not in ('RUNNING', 'PAUSED')):
            raise ValueError('Own complete full40/680/B4 optimizer/RNG checkpoint receipt required')
    peer = value['peer_slots']['C']
    if (not isinstance(peer, dict) or set(peer) != {'direct_proof', 'initial_actor', 'command_sha256'}
            or not _proof(peer['direct_proof'], group)
            or not direct._identity_document(peer['initial_actor'])
            or peer['command_sha256'] != command_sha(peer['initial_actor']['command'])):
        raise ValueError('Exact continuing C actor PID/birth/UID/argv SHA required')
    return value


def load_seal(path=None):
    path = direct._owned_regular(path or os.environ[ENV_PATH])
    if sha(path) != os.environ.get(ENV_SHA):
        raise ValueError('Immutable ABD continuation seal changed')
    value = validate_seal(json.loads(path.read_bytes()))
    for table in ('preserved_files', 'runtime_files'):
        for name, digest in value[table].items():
            if sha(direct._owned_regular(name)) != digest:
                raise ValueError('Sealed ABD source/input changed: ' + name)
    return value


def _flag(command, name):
    if command.count(name) != 1 or command.index(name)+1 >= len(command):
        raise ValueError('Exact stage argument required: ' + name)
    return command[command.index(name)+1]


def _stage_binding(identity, arm, gpu, output, code, seal, proof, stage):
    command = identity['command']
    if command[:4] != ['/home/aicompetition06/.conda/envs/nnunet/bin/python3.10', '-B', '-u',
            str(Path(code)/'tools/run_v25_ad_execution.py')]:
        raise RuntimeError('Exact sealed ABD stage wrapper required')
    if any(_flag(command, k) != v for k, v in (
            ('--arm', arm), ('--gpu', str(gpu)), ('--output', str(output)),
            ('--continuation-seal', str(seal)), ('--allocation-request', str(proof)))):
        raise RuntimeError('Own ABD stage argument binding changed')
    if stage in ('train', 'scoring'):
        if _flag(command, '--mode') != ('train' if stage == 'train' else 'score-cp'):
            raise RuntimeError('Exact own ABD train/scoring mode required')
    elif stage == 'native':
        if command[4] != '--native':
            raise RuntimeError('Exact own ABD native bootstrap required')
    else:
        raise RuntimeError('Only exact monitored train/scoring/native ABD actors can own CUDA')


def _validate_failed_slot(slot, seal):
    monitor = _guard_proof(slot['failed_monitor'])
    actor = dict(pid=monitor.get('child_pid'), create_time=monitor.get('child_create_time'),
        UID=monitor.get('UID'), command=monitor.get('child_command'))
    if (monitor.get('status') != 'FAILED' or monitor.get('stage') != 'train'
            or monitor.get('controller_pid') != seal['peer_controller_identity']['pid']
            or actor != slot['previous_actor'] or monitor.get('arm') != slot['arm']
            or monitor.get('stage_log') != slot['failure_log']['path']):
        raise ValueError('Exact original own FAILED monitor/actor/log binding required')
    command = actor['command']; arm = slot['arm']
    if (command[:4] != ['/home/aicompetition06/.conda/envs/nnunet/bin/python3.10', '-B', '-u',
            str(Path(seal['previous_code'])/'tools/run_v25_u128_reuse_execution.py')]
            or any(_flag(command, k) != v for k, v in (
                ('--arm', arm), ('--gpu', str(slot['gpu'])), ('--output', str(Path(seal['group_root'])/arm)),
                ('--mode', 'train'), ('--continuation-seal', seal['peer_seal']['path'])))):
        raise ValueError('Exact original released own U128 execution actor required')
    text = direct._owned_regular(slot['failure_log']['path']).read_text()
    expected = ('MemoryError: Historical evaluation RSS budget exceeded; no cases skipped'
        if arm == 'D' else 'RuntimeError: Exact live descendant PID/birth/argv/UID identity required')
    if sha(slot['failure_log']['path']) != slot['failure_log']['sha256'] or expected not in text:
        raise ValueError('Actual observed original own failure traceback required')
    original._released(actor)


def validate_own_contracts(lease):
    seal = load_seal(); arm = lease.request['arm']
    if arm not in MAPPING:
        raise ValueError('This recovery admits ABD only; continuing C remains independent')
    own = seal['slots'][arm]
    if (str(lease.path) != own['direct_proof']['path'] or sha(lease.path) != own['direct_proof']['sha256']
            or lease.request['controller_identity'] != seal['controller_identity']
            or str(lease.group) != seal['group_root']):
        raise ValueError('Exact own new ABD direct proof/controller required')
    old = original.validate_seal(_guard_proof(seal['peer_seal']))
    if (old['private_code'] != seal['previous_code'] or old['group_root'] != seal['group_root']
            or old['controller_identity'] != seal['peer_controller_identity']):
        raise ValueError('Unchanged original live C U128 seal/controller required')
    for name, digest in old['runtime_files'].items():
        if sha(direct._owned_regular(name)) != digest:
            raise ValueError('Continuing original C runtime source changed')
    for selected in ('A', 'B', 'C', 'D'):
        ref = seal['peer_slots']['C']['direct_proof'] if selected == 'C' else seal['slots'][selected]['direct_proof']
        document = direct.validate_direct_document(_guard_proof(ref))
        expected_controller = seal['peer_controller_identity'] if selected == 'C' else seal['controller_identity']
        gpu, cpus = original.MAPPING[selected]
        if (document['arm'] != selected or document['gpu'] != gpu or document['CPU_affinity'] != cpus
                or document['controller_identity'] != expected_controller
                or document['group_root'] != seal['group_root']
                or any(document[k] != lease.request[k] for k in
                    ('host_lifetime', 'aggregate_CPU_affinity', 'resource_guard_sha256', 'cgroup_admission'))):
            raise ValueError('Same complete four-arm RAM/lifetime/six-core direct contract required')
        if selected == 'C':
            if ref != old['slots']['C']['direct_proof']:
                raise ValueError('Exact original C direct proof required')
        else:
            _validate_failed_slot(seal['slots'][selected], seal)
    return seal


def verify_initial_peer(seal):
    peer = direct._exact(seal['peer_controller_identity'])
    child = direct._exact(seal['peer_slots']['C']['initial_actor'])
    parent = child.parent()
    if (parent is None or child.ppid() != peer.pid or direct._actor(parent) != seal['peer_controller_identity']
            or child.cpu_affinity() != original.MAPPING['C'][1]):
        raise RuntimeError('Continuing initial C PID/birth/UID/argv/ancestry changed before ABD launch')
    return dict(C_actor_preserved=True, C_identity=seal['peer_slots']['C']['initial_actor'],
        C_command_sha256=command_sha(child.cmdline()), C_controller_identity=seal['peer_controller_identity'])


def _peer_tree(lease, seal, tree):
    identity = seal['peer_controller_identity']; union = dict(tree)
    observed = getattr(lease, '_observed_C_descendants', None)
    if observed is None:
        observed = lease._observed_C_descendants = {}
    try:
        peer = direct._exact(identity)
    except direct.psutil.NoSuchProcess:
        # The original controller can end with the already recorded ABD
        # failures. Do not call that full C completion. Verify C's own terminal
        # monitor and released stage, and retain every observed live orphan in
        # RAM accounting if any survived its controller.
        value = json.loads(direct._owned_regular(lease.group/'C/monitor/status.json').read_bytes())
        if value.get('controller_pid') != identity['pid'] or value.get('arm') != 'C':
            raise RuntimeError('Missing C controller requires its exact terminal monitor')
        if value.get('status') == 'COMPLETE':
            certificate = direct._owned_regular(lease.group/'C/downstream/completion.json')
            digest = sha(certificate)
            cached = getattr(lease, '_completed_C_certificate_sha256', None)
            if cached is None:
                from tools.run_v25_u128_reuse_continuation import verify_arm_completion
                verify_arm_completion(lease.group, 'C')
                lease._completed_C_certificate_sha256 = digest
            elif cached != digest:
                raise ValueError('Verified original C completion certificate changed')
        elif value.get('status') == 'FAILED':
            actor = dict(pid=value.get('child_pid'), create_time=value.get('child_create_time'),
                UID=value.get('UID'), command=value.get('child_command'))
            if not direct._identity_document(actor):
                raise RuntimeError('Exact released failed C actor required')
            if actor['command'][:4] != seal['peer_slots']['C']['initial_actor']['command'][:4]:
                raise RuntimeError('Original failed C wrapper identity changed')
            if any(_flag(actor['command'], key) != expected for key, expected in (
                    ('--arm', 'C'), ('--gpu', '1'), ('--output', str(lease.group/'C')),
                    ('--continuation-seal', seal['peer_seal']['path']),
                    ('--allocation-request', seal['peer_slots']['C']['direct_proof']['path']))):
                raise RuntimeError('Original failed C own output binding changed')
            original._released(actor)
        else:
            raise RuntimeError('Original C controller vanished before an actual terminal monitor')
        for old in observed.values():
            try:
                child = direct.psutil.Process(old['pid']); actual = terminal._descendant(lease, child)
            except direct.psutil.NoSuchProcess:
                continue
            if actual is None:
                continue
            if actual != old:
                raise RuntimeError('Surviving original C child PID/birth/argv identity changed')
            if child.pid in union:
                raise RuntimeError('Separate ABD and original C controller trees overlap')
            union[child.pid] = child
        return union
    if (peer.cpu_affinity() != lease.request['aggregate_CPU_affinity']
            or direct.capture_cgroup_admission(peer.pid) != lease.request['cgroup_admission']):
        raise RuntimeError('Original continuing C controller CPU/cgroup changed')
    if peer.pid in union:
        raise RuntimeError('Separate ABD and original C controller trees overlap')
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
            raise RuntimeError('Original C descendant PID/birth/argv identity changed')
        if child.pid in union:
            raise RuntimeError('Separate ABD and original C descendants overlap')
        lease.managed[child.pid] = actual; observed[child.pid] = actual; union[child.pid] = child
    return union


def _stage_tree(lease, tree):
    value = json.loads(direct._owned_regular(lease.group/lease.request['arm']/'monitor/status.json').read_bytes())
    if value.get('controller_pid') != lease.request['controller_identity']['pid']:
        return {}
    actor = dict(pid=value.get('child_pid'), create_time=value.get('child_create_time'),
        UID=value.get('UID'), command=value.get('child_command'))
    if not direct._identity_document(actor):
        raise RuntimeError('Exact own monitored ABD stage identity required')
    seal = load_seal(); slot = seal['slots'][lease.request['arm']]
    _stage_binding(actor, slot['arm'], slot['gpu'], lease.group/slot['arm'],
        seal['private_code'], os.environ[ENV_PATH], slot['direct_proof']['path'], value.get('stage'))
    try:
        root = direct._exact(actor)
    except direct.psutil.NoSuchProcess:
        return {}
    parent = root.parent()
    if (parent is None or root.ppid() != lease.request['controller_identity']['pid']
            or direct._actor(parent) != lease.request['controller_identity']):
        raise RuntimeError('Own ABD stage origin is outside its exact controller tree')
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


class StandaloneADAllocationLease(_BASE):
    def _resources(self, controller, own, tree):
        seal = validate_own_contracts(self)
        union = _peer_tree(self, seal, tree)
        _BASE._resources(self, controller, own, union)
        self.actual_resources.update(selected_arms=['A', 'B', 'D'],
            own_ABD_tree_RSS_members=len(tree), combined_ABD_C_RSS_members=len(union),
            original_C_tree_only_used_for_RAM=True, C_and_GPU0_processes_modified=False)

    def _gpu(self, tree, *, initial):
        before = set(tree)
        selected = _stage_tree(self, tree)
        if set(tree) != before:
            self._resources(direct._exact(self.request['controller_identity']),
                direct._exact(self.self_identity), tree)
        return _BASE._gpu(self, selected, initial=initial)


def install_from_environment():
    seal = load_seal()
    if direct.DirectAllocationLease not in (_BASE, StandaloneADAllocationLease):
        raise ValueError('Unknown ABD direct-resource class override')
    direct.DirectAllocationLease = StandaloneADAllocationLease
    terminal.install_runtime()
    return dict(format=FORMAT, source_sha256=sha(Path(__file__)), seal_sha256=os.environ[ENV_SHA],
        selected_arms=['A', 'B', 'D'], GNN_RSS_bytes_per_arm=64*2**30,
        aggregate_RSS_bytes=416*2**30, original_C_tree_only_used_for_RAM=True,
        C_and_GPU0_processes_modified=False, controller_identity=seal['controller_identity'])
