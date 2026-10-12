"""Exact four-arm terminal recovery with untouched scientific and input guards.

Every old actor must have been naturally released. New monitors have separate
namespaces; all four stages use a single new controller and aggregate RAM tree.
"""
from __future__ import annotations
import json
import os
from pathlib import Path

from . import v25_direct_resources as direct
from . import v25_terminal_descendant_resources as terminal
from . import v25_u128_reuse_resources as original
from . import v25_standalone_ad_resources as ad

FORMAT = 'v25_exact_all_arm_terminal_recovery_v1'
ENV_PATH = 'V25_ALL_ARM_RECOVERY_SEAL'
ENV_SHA = 'V25_ALL_ARM_RECOVERY_SEAL_SHA256'
MAPPING = dict(original.MAPPING)
_BASE = direct.DirectAllocationLease
_FIELDS = {'format', 'selected_arms', 'group_root', 'private_code', 'controller_identity',
    'previous_seals', 'slots', 'preserved_files', 'runtime_files', 'source_scientific23',
    'created_at', 'scientific_configuration_unchanged', 'no_process_signaled'}
sha = original.sha
_guard_proof = original._guard_proof


def validate_seal(value):
    if (not isinstance(value, dict) or set(value) != _FIELDS or value['format'] != FORMAT
            or value['selected_arms'] != ['A', 'B', 'C', 'D'] or set(value['slots']) != set(MAPPING)
            or value['scientific_configuration_unchanged'] is not True
            or value['no_process_signaled'] is not True
            or not direct._identity_document(value['controller_identity'])):
        raise ValueError('Exact all-four terminal recovery controller required')
    group = Path(value['group_root']); code = Path(value['private_code'])
    if (not all(p.is_absolute() and '..' not in p.parts for p in (group, code)) or code.parent != group):
        raise ValueError('Fresh exact all-four private code in the original group required')
    if (not isinstance(value['previous_seals'], dict) or set(value['previous_seals']) != {'u128', 'ABD'}
            or not all(original._proof_reference(v, group) for v in value['previous_seals'].values())):
        raise ValueError('Immutable exact previous U128 and ABD seals required')
    for table in ('preserved_files', 'runtime_files', 'source_scientific23'):
        if (not isinstance(value[table], dict) or not value[table]
                or any(not isinstance(p, str) or not direct._checksum(h) for p, h in value[table].items())):
            raise ValueError('Immutable source/input/runtime receipts required')
    if (len(value['source_scientific23']) != 23
            or any(not Path(path).is_absolute() or '..' in Path(path).parts
                or not Path(path).is_relative_to(code) or value['preserved_files'].get(path) != digest
                for path, digest in value['source_scientific23'].items())):
        raise ValueError('All original23 scientific source digests must be preserved in the new copy')
    for arm, (gpu, cpus) in MAPPING.items():
        slot = value['slots'][arm]
        if (not isinstance(slot, dict) or set(slot) != {'arm', 'gpu', 'CPU_affinity', 'direct_proof',
                'checkpoint', 'previous_actor', 'failed_monitor', 'failure_log', 'monitor_path'}
                or slot['arm'] != arm or slot['gpu'] != gpu or slot['CPU_affinity'] != cpus
                or not direct._identity_document(slot['previous_actor'])
                or not all(original._proof_reference(slot[key], group)
                    for key in ('direct_proof', 'failed_monitor', 'failure_log'))):
            raise ValueError('Exact original arm GPU/six-core released actor and receipts required')
        monitor = Path(slot['monitor_path'])
        if (not monitor.is_absolute() or '..' in monitor.parts or monitor.name != 'status.json'
                or not monitor.parent.is_relative_to(group/arm/'monitor')
                or monitor.parent == group/arm/'monitor'):
            raise ValueError('Fresh per-arm monitor subdirectory required; no canonical monitor overwrite')
        if not direct._checksum(value['preserved_files'].get(str(group/arm/'monitor/status.json'))):
            raise ValueError('Every original canonical FAILED monitor must be sealed unchanged')
        cp = slot['checkpoint']
        if (not isinstance(cp, dict) or set(cp) != {'path', 'sha256', 'identity_sha256', 'content_sha256',
                'epoch', 'updates', 'phase', 'train_position', 'evaluation_position', 'active_u', 'status'}
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
            raise ValueError('Own full40/680/B4 optimizer/RNG checkpoint receipt required for every arm')
    return value


def load_seal(path=None):
    path = direct._owned_regular(path or os.environ[ENV_PATH])
    if sha(path) != os.environ.get(ENV_SHA):
        raise ValueError('Immutable all-four recovery seal changed')
    value = validate_seal(json.loads(path.read_bytes()))
    for table in ('preserved_files', 'runtime_files'):
        for name, digest in value[table].items():
            if sha(direct._owned_regular(name)) != digest:
                raise ValueError('Sealed all-four source/input/original monitor changed: ' + name)
    return value


def previous_contracts(seal):
    u128 = original.validate_seal(_guard_proof(seal['previous_seals']['u128']))
    abd = ad.validate_seal(_guard_proof(seal['previous_seals']['ABD']))
    if (u128['group_root'] != seal['group_root'] or abd['group_root'] != seal['group_root']
            or abd['peer_controller_identity'] != u128['controller_identity']
            or abd['peer_seal'] != seal['previous_seals']['u128']
            or len({seal['controller_identity']['pid'], u128['controller_identity']['pid'],
                abd['controller_identity']['pid']}) != 3
            or seal['private_code'] in (u128['private_code'], abd['private_code'])):
        raise ValueError('Exact separate new controller and original U128/ABD provenance required')
    for contract in (u128, abd):
        for name, digest in contract['runtime_files'].items():
            if sha(direct._owned_regular(name)) != digest:
                raise ValueError('Previous immutable runtime source bytes changed')
        for path, digest in seal['source_scientific23'].items():
            relative = Path(path).relative_to(seal['private_code'])
            previous = str(Path(contract['private_code'])/relative)
            if contract['preserved_files'].get(previous) != digest or sha(direct._owned_regular(previous)) != digest:
                raise ValueError('Original23 scientific source bytes differ from the exact previous admission')
    return {'u128': u128, 'ABD': abd}


def _validate_failed_slot(slot, seal, contracts=None):
    contracts = previous_contracts(seal) if contracts is None else contracts
    arm = slot['arm']; kind = 'u128' if arm == 'C' else 'ABD'; prior = contracts[kind]
    monitor = _guard_proof(slot['failed_monitor'])
    actor = dict(pid=monitor.get('child_pid'), create_time=monitor.get('child_create_time'),
        UID=monitor.get('UID'), command=monitor.get('child_command'))
    if (monitor.get('status') != 'FAILED' or monitor.get('stage') != 'train'
            or monitor.get('controller_pid') != prior['controller_identity']['pid']
            or monitor.get('arm') != arm or actor != slot['previous_actor']
            or monitor.get('stage_log') != slot['failure_log']['path']):
        raise ValueError('Exact original failed own training monitor/actor/log required')
    entry = 'run_v25_u128_reuse_execution.py' if arm == 'C' else 'run_v25_ad_execution.py'
    command = actor['command']
    if (command[:4] != ['/home/aicompetition06/.conda/envs/nnunet/bin/python3.10', '-B', '-u',
            str(Path(prior['private_code'])/'tools'/entry)]
            or any(ad._flag(command, key) != expected for key, expected in (
                ('--arm', arm), ('--gpu', str(slot['gpu'])), ('--output', str(Path(seal['group_root'])/arm)),
                ('--mode', 'train'), ('--continuation-seal', seal['previous_seals'][kind]['path']),
                ('--allocation-request', prior['slots'][arm]['direct_proof']['path'])))):
        raise ValueError('Exact original previous arm wrapper and allocation binding required')
    path = direct._owned_regular(slot['failure_log']['path'])
    expected = ('RuntimeError: Exact live descendant PID/birth/argv/UID identity required' if arm == 'C'
        else 'ValueError: Admitted GT-free implementation/prototype input changed')
    if sha(path) != slot['failure_log']['sha256'] or expected not in path.read_text():
        raise ValueError('Exact observed own failure traceback required; no generic recovery fallback')
    original._released(actor)


def verify_initial_release(seal):
    contracts = previous_contracts(seal)
    for slot in seal['slots'].values():
        _validate_failed_slot(slot, seal, contracts)
    for prior in contracts.values():
        original._released(prior['controller_identity'])
    return dict(previous_controllers_and_all_four_stages_released=True,
        previous_controller_identities={key: value['controller_identity']
            for key, value in contracts.items()}, previous_actor_identities={
            arm: slot['previous_actor'] for arm, slot in seal['slots'].items()},
        original_source23_bytes_unchanged=True, source_timestamp_guard_unchanged=True,
        no_process_signaled=True)


def validate_own_contracts(lease):
    seal = load_seal(); arm = lease.request['arm']
    if arm not in MAPPING:
        raise ValueError('Only the exact original four arms are admitted')
    slot = seal['slots'][arm]
    if (str(lease.path) != slot['direct_proof']['path'] or sha(lease.path) != slot['direct_proof']['sha256']
            or lease.request['controller_identity'] != seal['controller_identity']
            or str(lease.group) != seal['group_root']):
        raise ValueError('Exact own all-four recovery proof/controller required')
    contracts = previous_contracts(seal)
    for selected, (gpu, cpus) in MAPPING.items():
        declared = seal['slots'][selected]
        document = direct.validate_direct_document(_guard_proof(declared['direct_proof']))
        if (document['arm'] != selected or document['gpu'] != gpu or document['CPU_affinity'] != cpus
                or document['controller_identity'] != seal['controller_identity']
                or document['group_root'] != seal['group_root']
                or any(document[key] != lease.request[key] for key in ('host_lifetime',
                    'aggregate_CPU_affinity', 'resource_guard_sha256', 'cgroup_admission'))):
            raise ValueError('Same complete four-arm RAM/lifetime/six-core direct contract required')
        _validate_failed_slot(declared, seal, contracts)
    # Both old controllers must also be gone before a new all-four controller owns CUDA.
    for prior in contracts.values():
        original._released(prior['controller_identity'])
    return seal


def _stage_binding(identity, seal, arm, stage):
    slot = seal['slots'][arm]; command = identity['command']
    if command[:4] != ['/home/aicompetition06/.conda/envs/nnunet/bin/python3.10', '-B', '-u',
            str(Path(seal['private_code'])/'tools/run_v25_all_arm_recovery_execution.py')]:
        raise RuntimeError('Exact sealed all-four stage wrapper required')
    if any(ad._flag(command, key) != expected for key, expected in (
            ('--arm', arm), ('--gpu', str(slot['gpu'])), ('--output', str(Path(seal['group_root'])/arm)),
            ('--continuation-seal', os.environ[ENV_PATH]),
            ('--allocation-request', slot['direct_proof']['path']))):
        raise RuntimeError('Own all-four stage argument binding changed')
    if stage in ('train', 'scoring'):
        if ad._flag(command, '--mode') != ('train' if stage == 'train' else 'score-cp'):
            raise RuntimeError('Exact own train/scoring mode required')
    elif stage != 'native' or command[4:5] != ['--native']:
        raise RuntimeError('Only exact own monitored train/scoring/native actors can own CUDA')


def _stage_tree(lease, tree):
    seal = load_seal(); arm = lease.request['arm']; slot = seal['slots'][arm]
    path = Path(slot['monitor_path'])
    if not path.exists():
        if getattr(lease, '_stage_monitor_observed', False):
            raise RuntimeError('Observed own all-four stage monitor disappeared')
        return {}
    value = json.loads(direct._owned_regular(path).read_bytes()); lease._stage_monitor_observed = True
    if value.get('controller_pid') != lease.request['controller_identity']['pid'] or value.get('arm') != arm:
        raise RuntimeError('Own recovery monitor controller/arm changed')
    if value.get('status') == 'COMPLETE' and value.get('stage') == 'complete':
        return {}
    actor = dict(pid=value.get('child_pid'), create_time=value.get('child_create_time'),
        UID=value.get('UID'), command=value.get('child_command'))
    if not direct._identity_document(actor):
        raise RuntimeError('Exact own monitored recovery stage identity required')
    _stage_binding(actor, seal, arm, value.get('stage'))
    try:
        root = direct._exact(actor)
    except direct.psutil.NoSuchProcess:
        return {}
    parent = root.parent()
    if (parent is None or root.ppid() != lease.request['controller_identity']['pid']
            or direct._actor(parent) != lease.request['controller_identity']):
        raise RuntimeError('Own stage origin is outside its exact new controller tree')
    original._remember_live(lease, root, actor, tree, slot['CPU_affinity']); selected = {root.pid: root}
    for child in root.children(recursive=True):
        try:
            actual = original._current_descendant(lease, child, actor, tree, slot['CPU_affinity'])
        except direct.psutil.NoSuchProcess:
            continue
        if actual is not None:
            selected.update(actual)
    return selected


class AllArmRecoveryAllocationLease(_BASE):
    def _resources(self, controller, own, tree):
        validate_own_contracts(self)
        _BASE._resources(self, controller, own, tree)
        self.actual_resources.update(selected_arms=['A', 'B', 'C', 'D'],
            original_canonical_monitors_modified=False, GPU0_processes_modified=False)

    def _gpu(self, tree, *, initial):
        before = set(tree); selected = _stage_tree(self, tree)
        if set(tree) != before:
            self._resources(direct._exact(self.request['controller_identity']),
                direct._exact(self.self_identity), tree)
        return _BASE._gpu(self, selected, initial=initial)


def install_from_environment():
    seal = load_seal()
    if direct.DirectAllocationLease not in (_BASE, AllArmRecoveryAllocationLease):
        raise ValueError('Unknown all-four direct-resource class override')
    direct.DirectAllocationLease = AllArmRecoveryAllocationLease
    terminal.install_runtime()
    return dict(format=FORMAT, source_sha256=sha(Path(__file__)), seal_sha256=os.environ[ENV_SHA],
        selected_arms=['A', 'B', 'C', 'D'], GNN_RSS_bytes_per_arm=64*2**30,
        aggregate_RSS_bytes=416*2**30, original_canonical_monitors_modified=False,
        source_timestamp_guard_unchanged=True, GPU0_processes_modified=False,
        controller_identity=seal['controller_identity'])
