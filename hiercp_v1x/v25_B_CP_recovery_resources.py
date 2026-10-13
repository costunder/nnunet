"""Fresh host admission for unchanged own B BEST and 529 durable CP rows.

This execution sidecar leaves every original source/proof byte unchanged.
Only the new controller, current host lifetime and resource observations are
bound in memory. Existing score rows remain immutable while missing original
B4 batches may append. No process signal, scheduler mutation or launch occurs.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import os
import re

from . import basic_cp_ablation_resources as basic
from . import v25_B_early_downstream_resources as previous
from . import v25_direct_resources as direct

FORMAT = 'v25_B_own_CP529_fresh_host_recovery_v1'
ENV_PATH = 'V25_B_CP_RECOVERY_SEAL'
ENV_SHA = ENV_PATH + '_SHA256'
ENTRY = 'tools/run_v25_B_CP_recovery_execution.py'
BEST_SHA = 'bf9dbd6c64bb15e052f4ac426e208b0cb66c32b019d4b8ad23dff9cf8ad5c740'
SOURCE_SHA = '9e5be3e3ab8f04fd23c166b3267bc4efbfb4b321ef974920dfc08fbf5dd275e6'
GPU_UUID = 'GPU-e41d3421-8819-0a49-4a97-6c4518a54f5f'
CPUS = list(range(42, 48))
_B_BASE = previous.BEarlyDownstreamAllocationLease
_DIRECT_BASE = previous._BASE
_FIELDS = {'format', 'B_controller_seal', 'previous_failed_execution', 'durable',
    'fresh_host_lifetime', 'PBS_shared_cache_root', 'gpu', 'gpu_uuid', 'CPU_affinity',
    'runtime_files', 'created_at', 'previous_results_preserved', 'processes_signaled'}
_VALIDATED = set()


def _reference(value):
    return (isinstance(value, dict) and set(value) == {'path', 'sha256'}
        and isinstance(value['path'], str) and Path(value['path']).is_absolute()
        and '..' not in Path(value['path']).parts and direct._checksum(value['sha256']))


def _read_proof(value):
    if not _reference(value):
        raise ValueError('Exact own absolute immutable recovery proof required')
    path = direct._owned_regular(value['path'])
    previous.immutable_digest(path, value['sha256'])
    return json.loads(path.read_bytes())


def _canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def validate_recovery_proof(value):
    if (not isinstance(value, dict) or set(value) != _FIELDS or value.get('format') != FORMAT
            or value.get('gpu') != 5 or value.get('gpu_uuid') != GPU_UUID
            or value.get('CPU_affinity') != CPUS or value.get('previous_results_preserved') is not True
            or value.get('processes_signaled') is not False
            or type(value.get('created_at')) not in (int, float)
            or not math.isfinite(value['created_at']) or value['created_at'] <= 0):
        raise ValueError('Exact own B/GPU5/CPU42..47/529-source recovery admission required')
    for key in ('B_controller_seal', 'fresh_host_lifetime'):
        if not _reference(value[key]):
            raise ValueError('Immutable original B seal and fresh host-lifetime references required')
    path = Path(value['PBS_shared_cache_root'])
    if not path.is_absolute() or '..' in path.parts:
        raise ValueError('Actual own private shared PBS observation namespace required')
    failed = value['previous_failed_execution']
    if (not isinstance(failed, dict) or set(failed) != {'controller', 'child', 'monitor', 'controller_seal'}
            or any(not direct._identity_document(failed[k]) for k in ('controller', 'child'))
            or any(not _reference(failed[k]) for k in ('monitor', 'controller_seal'))):
        raise ValueError('Exact failed own controller/child and immutable original monitor/seal required')
    durable = value['durable']
    if (not isinstance(durable, dict) or set(durable) != {'root', 'request', 'completed_batches',
                'completed_sources', 'completed_positions', 'committed_files_sha256'}
            or not isinstance(durable['root'], str) or not Path(durable['root']).is_absolute()
            or '..' in Path(durable['root']).parts or not _reference(durable['request'])
            or durable['completed_batches'] != 133 or durable['completed_sources'] != 529
            or durable['completed_positions'] != 67712
            or Path(durable['request']['path']) != Path(durable['root'])/'request.json'
            or not isinstance(durable['committed_files_sha256'], dict)
            or len(durable['committed_files_sha256']) != 662):
        raise ValueError('Exactly original133 complete batches/529 sources/67712 positions required')
    for table in ('runtime_files',):
        if not isinstance(value[table], dict) or not value[table]:
            raise ValueError('Separately sealed recovery resource and entry source bytes required')
    for table in (durable['committed_files_sha256'], value['runtime_files']):
        if any(not isinstance(p, str) or not Path(p).is_absolute() or '..' in Path(p).parts
                or not direct._checksum(digest) for p, digest in table.items()):
            raise ValueError('Absolute own immutable recovery file/SHA table required')
    for module in (__file__, basic.__file__):
        if str(Path(module).resolve()) not in value['runtime_files']:
            raise ValueError('Actual new sidecar and unchanged shared-PBS resource implementation must be pinned')
    return value


def _verify_failed_execution(value):
    failed = value['previous_failed_execution']
    monitor = _read_proof(failed['monitor'])
    original = _read_proof(failed['controller_seal'])
    controller = failed['controller']; child = failed['child']
    actual_child = dict(pid=monitor.get('child_pid'), create_time=monitor.get('child_create_time'),
        UID=monitor.get('UID'), command=monitor.get('child_command'))
    if (monitor.get('status') != 'FAILED' or monitor.get('arm') != 'B'
            or monitor.get('stage') not in ('scoring', 'native')
            or monitor.get('controller_pid') != controller['pid']
            or actual_child != child or original.get('controller_identity') != controller
            or monitor.get('B_early_downstream_seal') != failed['controller_seal']['path']
            or original.get('slot', {}).get('monitor_path') != failed['monitor']['path']):
        raise ValueError('Actual original failed B monitor/controller/child lineage differs')
    for identity in (controller, child):
        previous._released(identity)


def _verify_durable(value):
    """Read and verify only original immutable prefix; never open a writer."""
    durable = value['durable']; root = direct._owned_regular(durable['root'], directory=True)
    declaration = _read_proof(durable['request'])
    batches = declaration.get('source_batches', [])
    entries = [entry for batch in batches for entry in batch]
    source = declaration.get('source_bank', {})
    best = declaration.get('best_provenance', {})
    if (declaration.get('format') != 'v25_owned_BEST_Basic642_durable_score_store_v1'
            or len(batches) != 162 or len(entries) != 642 or len(set(entries)) != 642
            or any(not 2 <= len(batch) <= 4 for batch in batches)
            or set(entries) != set(source.get('entries', {})) or source.get('index_sha256') != SOURCE_SHA
            or source.get('complete_cases') != 81 or source.get('complete_sources') != 642
            or source.get('complete_positions') != 82176 or best.get('arm') != 'B'
            or best.get('checkpoint_sha256') != BEST_SHA or best.get('full_training') is not False
            or best.get('best_record', {}).get('epoch') != 3
            or best.get('best_record', {}).get('updates') != 51
            or len([e for batch in batches[:133] for e in batch]) != 529):
        raise ValueError('Original complete642/B4 schedule and own early B BEST e3/update51 required')
    contract = declaration.get('cp_contract', {})
    required = dict(total_sources=642, candidates_per_source=128, total_scored_candidates=82176,
        donor_policy='same_patient', cp_probability=.5, train_cases=105,
        validation_cases=26, physical_batch=2, patch=[128,128,128], epochs=250,
        selection_policy='argmax', paste_policy='historical_preprocessed_patch_hard_paste')
    if any(contract.get(key) != expected for key, expected in required.items()):
        raise ValueError('Original full Basic matched CP/native contract differs')
    expected_files = {}
    for batch_entries in batches[:133]:
        batch_path = root/'batches'/(_canonical(batch_entries)+'.json')
        batch = json.loads(direct._owned_regular(batch_path).read_bytes())
        expected_files[str(batch_path)] = durable['committed_files_sha256'].get(str(batch_path))
        if (batch.get('format') != declaration['format']
                or batch.get('request_sha256') != durable['request']['sha256']
                or batch.get('entries') != batch_entries
                or set(batch.get('score_rows_sha256', {})) != set(batch_entries)):
            raise ValueError('Original complete durable B4 batch changed')
        for entry in batch_entries:
            row_path = root/'scores'/(hashlib.sha256(entry.encode()).hexdigest()+'.json')
            payload = json.loads(direct._owned_regular(row_path).read_bytes()); row = payload.get('row', {})
            digest = durable['committed_files_sha256'].get(str(row_path))
            expected_files[str(row_path)] = digest
            original = source['entries'][entry]; scores = row.get('scores', [])
            binding = row.get('input_binding', {}); geometry = binding.get('source_context_geometry', {})
            if (batch['score_rows_sha256'][entry] != digest or payload.get('format') != declaration['format']
                    or payload.get('request_sha256') != durable['request']['sha256']
                    or payload.get('row_sha256') != _canonical(row)
                    or row.get('entry') != entry or row.get('case_id') != original['case_id']
                    or row.get('source_component') != original['source_component']
                    or row.get('original_sha256') != original['sha256']
                    or len(scores) != 128 or any(type(x) not in (int, float) or not math.isfinite(x) for x in scores)
                    or row.get('selected_index') != max(range(128), key=scores.__getitem__)
                    or row.get('model_sha256') != declaration.get('model_sha256')
                    or row.get('all_original128_CP_scored') is not True
                    or row.get('all_original_P_in_context') is not True
                    or row.get('recipient_GT_used_in_forward') is not False
                    or row.get('original_non_score_payloads_preserved') is not True
                    or binding.get('context_contract') != declaration.get('context_contract')
                    or binding.get('source_bank_entry_sha256') != original['sha256']
                    or binding.get('original_non_score_arrays_sha256') != original['non_score_arrays_sha256']
                    or geometry.get('profile_sha256') != declaration['context_contract']['contract_sha256']
                    or geometry.get('minimum_liver_depth_exclusive_mm') != 1.
                    or geometry.get('context_inner_radius_mm') != 0.
                    or geometry.get('context_outer_radius_mm') != 10.
                    or geometry.get('canonical_context_nodes', 0) <= 0):
                raise ValueError('Original own BEST/context/source/full128 score row changed: '+entry)
    if expected_files != durable['committed_files_sha256']:
        raise ValueError('Exact original133 batch and529 row digest table differs')
    return declaration


def _fresh_lifetime(value):
    lifetime = _read_proof(value['fresh_host_lifetime'])
    if (lifetime.get('format') != direct.OWNED_FORMAT or lifetime.get('UID') != 41833
            or lifetime.get('owner') != 'aicompetition06' or lifetime.get('node') != 'ece-a6gpu6'
            or lifetime.get('gpu') != 6 or not re.fullmatch(r'\d+\.ECE-util1', lifetime.get('job_id', ''))
            or type(lifetime.get('expires_at_unix')) not in (int, float)
            or not math.isfinite(lifetime['expires_at_unix'])):
        raise ValueError('Actual immutable owned GPU6 host lifetime required; GPU5 ownership remains direct')
    if lifetime['expires_at_unix'] - direct.time.time() < direct.EXPIRY_RESERVE_SECONDS:
        raise RuntimeError('Fresh host lifetime has less than180seconds; no recovery execution')
    return lifetime


def load_recovery_proof():
    path = direct._owned_regular(os.environ[ENV_PATH]); digest = os.environ.get(ENV_SHA)
    previous.immutable_digest(path, digest)
    value = validate_recovery_proof(json.loads(path.read_bytes()))
    if (os.environ.get(previous.ENV_PATH) != value['B_controller_seal']['path']
            or os.environ.get(previous.ENV_SHA) != value['B_controller_seal']['sha256']):
        raise ValueError('Exact separately sealed new B controller admission required')
    seal = previous.load_seal()
    if Path(value['durable']['root']).parent.parent != Path(seal['group_root']):
        raise ValueError('Original durable CP store must stay inside the original B experiment group')
    if value['gpu_uuid'] != _read_proof(seal['slot']['direct_proof'])['GPU_UUID']:
        raise ValueError('Original own direct physical GPU5 UUID changed')
    for p, expected in {**value['runtime_files'], **value['durable']['committed_files_sha256']}.items():
        previous.immutable_digest(direct._owned_regular(p), expected)
    _fresh_lifetime(value)
    key = (str(path), digest)
    if key not in _VALIDATED:
        _verify_failed_execution(value)
        declaration = _verify_durable(value)
        if declaration['best_provenance']['checkpoint'] != str(Path(seal['group_root'])/'B/training/checkpoint_best.pt'):
            raise ValueError('Only original own B GNN BEST may resume its CP scores')
        _VALIDATED.add(key)
    return value, seal


class _ObservationHelper(basic.BasicLease):
    """Reuse tested scheduler/process/CUDA observations without Basic math."""
    def _tree(self, controller, own):
        controller, own, tree = self.recovery_lease._tree()
        self.recovery_lease._resources(controller, own, tree)
        return self.recovery_lease._selected_stage_tree(tree)


def _observer(lease, value):
    observer = _ObservationHelper.__new__(_ObservationHelper)
    observer.recovery_lease = lease
    observer.request = dict(controller_identity=copy.deepcopy(lease.request['controller_identity']),
        cpu_affinity=CPUS, gpu=5, gpu_uuid=GPU_UUID, GPU_memory_total_MiB=46068)
    observer.self_identity = copy.deepcopy(lease.self_identity)
    observer.lifetime = copy.deepcopy(lease.lifetime)
    observer.monotonic_expiry = lease.monotonic_expiry
    observer._pbs_shared_root = Path(value['PBS_shared_cache_root'])
    observer._pbs_shared_header = dict(format=basic.PBS_CACHE_FORMAT, UID=41833,
        job_id=lease.lifetime['job_id'], node='ece-a6gpu6', owner='aicompetition06',
        expires_at_unix=lease.lifetime['expires_at_unix'],
        lifetime_proof_sha256=value['fresh_host_lifetime']['sha256'])
    observer._pbs_verified_at = observer._pbs_last_attempt = observer._pbs_observation = observer._pbs_timeout = None
    observer._managed = {}; observer._fork_pending = {}; observer._terminal_observations = {}; observer._cuda_observations = {}
    return observer


class FreshBLifetimeLease(_B_BASE):
    def _bind_fresh_recovery(self):
        value, seal = load_recovery_proof(); prior = previous.previous_contract(seal)
        if (str(self.path) != seal['slot']['direct_proof']['path']
                or previous.immutable_digest(self.path, seal['slot']['direct_proof']['sha256']) != seal['slot']['direct_proof']['sha256']
                or self.request['controller_identity'] != prior['controller_identity']
                or self.request['arm'] != 'B' or self.request['gpu'] != 5
                or self.request['CPU_affinity'] != CPUS or self.request['GPU_UUID'] != GPU_UUID
                or str(self.group) != seal['group_root']):
            raise ValueError('Original immutable own B direct declaration required before fresh lifetime binding')
        previous.verify_initial_release(seal)
        self.request = copy.deepcopy(self.request)
        self.request.update(controller_identity=seal['controller_identity'], aggregate_CPU_affinity=CPUS,
            cgroup_admission=seal['controller_cgroup_admission'],
            host_lifetime=dict(allocation_request=value['fresh_host_lifetime']['path'],
                allocation_request_sha256=value['fresh_host_lifetime']['sha256'],
                expires_at_unix=_fresh_lifetime(value)['expires_at_unix']))
        self.lifetime_path = direct._owned_regular(value['fresh_host_lifetime']['path'])
        self.lifetime = _fresh_lifetime(value)
        self.monotonic_expiry = direct.time.monotonic()+self.lifetime['expires_at_unix']-direct.time.time()
        self._B_seal = seal; self._B_seal_bytes = Path(os.environ[previous.ENV_PATH]).read_bytes(); self._B_rebound = True
        self._recovery_value = value
        self._observer = _observer(self, value)

    def verify(self, *, force=False):
        if not getattr(self, '_B_rebound', False):
            self._bind_fresh_recovery()
        # Poll old source/score metadata at the original30second admission rate;
        # complete SHA is cached once per process by the frozen B witness cache.
        now = direct.time.monotonic()
        if force or self.last_verified is None or now-self.last_verified >= direct.POLL_SECONDS:
            value, _ = load_recovery_proof()
            if value != self._recovery_value:
                raise ValueError('Immutable separately sealed B recovery binding changed')
        result = super().verify(force=force)
        remaining = min(self.lifetime['expires_at_unix']-direct.time.time(), self.monotonic_expiry-direct.time.monotonic())
        if remaining < direct.EXPIRY_RESERVE_SECONDS:
            raise RuntimeError('Fresh B host lifetime crossed180second boundary during resource verification')
        if direct.time.monotonic()-self._observer._pbs_verified_at >= basic.PBS_TRANSIENT_MAX_AGE_SECONDS:
            self._host_lifetime()
        self.actual_resources.update(fresh_B_recovery=True, durable_sources_reused=529,
            original_direct_proof_modified=False, original_host_lifetime_file_modified=False,
            processes_signaled=False, host_lifetime_job_id=self.lifetime['job_id'],
            host_lifetime_expires_at_unix=self.lifetime['expires_at_unix'], host_lifetime_remaining_seconds=remaining,
            host_lifetime_actual_PBS_observation=copy.deepcopy(self._observer._pbs_observation),
            host_lifetime_PBS_observation_age_seconds=direct.time.monotonic()-self._observer._pbs_verified_at,
            host_lifetime_PBS_warning=copy.deepcopy(self._observer._pbs_timeout))
        return result

    def _host_lifetime(self):
        self._observer._host_lifetime()

    def _tree(self):
        controller = direct._exact(self.request['controller_identity']); own = direct._exact(self.self_identity)
        tree = {controller.pid: controller}
        for process in controller.children(recursive=True):
            try:
                process = self._observer._descendant(process)
                if process is not None:
                    if process.cpu_affinity() != CPUS:
                        raise RuntimeError('Fresh B descendant left exact CPU42..47 allocation')
                    tree[process.pid] = process
            except direct.psutil.NoSuchProcess:
                continue
        if own.pid not in tree or controller.cpu_affinity() != CPUS or own.cpu_affinity() != CPUS:
            raise RuntimeError('Fresh B stage/controller lost exact authorized lineage/CPU affinity')
        return controller, own, tree

    def _selected_stage_tree(self, tree):
        seal = previous.load_seal(); path = Path(seal['slot']['monitor_path'])
        if not path.exists():
            if getattr(self, '_stage_monitor_observed', False):
                raise RuntimeError('Observed fresh B monitor disappeared')
            return {}
        monitor = json.loads(direct._owned_regular(path).read_bytes()); self._stage_monitor_observed = True
        if monitor.get('controller_pid') != seal['controller_identity']['pid'] or monitor.get('arm') != 'B':
            raise RuntimeError('Exact new B monitor controller/arm changed')
        if monitor.get('status') == 'COMPLETE' and monitor.get('stage') == 'complete':
            return {}
        actor = dict(pid=monitor.get('child_pid'), create_time=monitor.get('child_create_time'),
            UID=monitor.get('UID'), command=monitor.get('child_command'))
        if not direct._identity_document(actor):
            raise RuntimeError('Exact fresh B monitored scoring/native actor required')
        previous._stage_binding(actor, seal, monitor.get('stage'))
        try:
            root = direct._exact(actor)
        except direct.psutil.NoSuchProcess:
            return {}
        if root.ppid() != seal['controller_identity']['pid'] or root.parent() is None or direct._actor(root.parent()) != seal['controller_identity']:
            raise RuntimeError('Fresh B monitored stage is outside exact current controller tree')
        if root.cpu_affinity() != CPUS:
            raise RuntimeError('Exact B monitored stage CPU affinity changed')
        selected = {root.pid: root}
        for process in root.children(recursive=True):
            try:
                process = self._observer._descendant(process)
                if process is not None:
                    if process.cpu_affinity() != CPUS:
                        raise RuntimeError('Exact B CUDA-stage descendant CPU affinity changed')
                    selected[process.pid] = process
            except direct.psutil.NoSuchProcess:
                continue
        return selected

    def _resources(self, controller, own, tree):
        _DIRECT_BASE._resources(self, controller, own, tree)
        selected = self._selected_stage_tree(tree)
        if selected:
            monitor = json.loads(Path(previous.load_seal()['slot']['monitor_path']).read_bytes())
            limit = (96 if monitor['stage'] == 'native' else 64)*2**30
            rss = self._observer._rss(selected)
            if rss > limit:
                raise MemoryError('Exact fresh B stage exceeds original64/96GiB RSS budget')
            self.actual_resources.update(own_B_stage_RSS_bytes=rss, own_B_stage_RSS_limit_bytes=limit)
        self.actual_resources.update(selected_arms=['B'], original_direct_proof_modified=False,
            original_monitors_modified=False, GPU0_processes_modified=False, GNN_full40_claimed=False,
            processes_signaled_by_this_adapter=False)

    def _gpu(self, tree, *, initial):
        selected = self._selected_stage_tree(tree)
        result = self._observer._gpu(selected, idle=False)
        if initial and result['GPU_free_MiB'] < 40*1024:
            raise RuntimeError('Exact fresh B GPU5 initial40GiB free admission failed')
        # GPU observation can refresh the tree after an own stage starts. Check
        # resources again with fresh exact descendants rather than stale PIDs.
        controller, own, full = self._tree()
        self._resources(controller, own, full)
        self.actual_resources.update(result)


def install_from_environment():
    if direct.DirectAllocationLease not in (_DIRECT_BASE, _B_BASE, FreshBLifetimeLease):
        raise ValueError('Unknown existing direct-resource class override')
    if previous.BEarlyDownstreamAllocationLease not in (_B_BASE, FreshBLifetimeLease):
        raise ValueError('Unknown existing B-resource class override')
    if previous.ENTRY not in ('tools/run_v25_B_early_downstream_execution.py', ENTRY):
        raise ValueError('Unknown existing B downstream execution entry override')
    previous.ENTRY = ENTRY
    value, seal = load_recovery_proof()
    direct.DirectAllocationLease = FreshBLifetimeLease
    previous.BEarlyDownstreamAllocationLease = FreshBLifetimeLease
    previous.install_from_environment = install_from_environment
    return dict(format=FORMAT, selected_arms=['B'], physical_GPU=5, CPU_affinity=CPUS,
        fresh_host_lifetime=value['fresh_host_lifetime'], controller_identity=seal['controller_identity'],
        source_seal_sha256=os.environ[ENV_SHA], original529_complete_sources_preserved=True,
        original_scientific_files_changed=False, original_resource_source_files_changed=False,
        original_host_lifetime_files_changed=False, processes_signaled=False,
        scoring_RSS_bytes=64*2**30, native_RSS_bytes=96*2**30, aggregate_RSS_bytes=416*2**30)
