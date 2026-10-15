"""Immutable seven-controller RAM admission for the full Basic CP group.

Available RAM excludes the group's already-resident pages.  Admission therefore
adds back every exactly registered process tree once, rather than only the
calling arm.  Four native arms retain96GiB each; candidate arms3/4/5 share288GiB;
the unchanged32GiB shared reserve keeps the full group admission at704GiB.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import shutil
import stat
import time

import psutil

from . import basic_cp_seven_resources as resource
from . import basic_cp_gpu_placement as placement

GIB = resource.GIB
FORMAT = 'basic_cp_shared_resource_registry_v1'
GUARD_FILE = 'hiercp_v1x/basic_cp_group_resources.py'
AGGREGATE_CAPACITY = 704 * GIB
GROUP_RSS_LIMIT = 672 * GIB
CANDIDATE_RSS_LIMIT = 288 * GIB
NATIVE_RSS_LIMIT = 96 * GIB
SHARED_CPU_AFFINITY = [*range(24, 48), *range(56, 96)]
ARMS = {0:'multi5', 1:'multi7', 2:'multi10', 3:'candidates512',
        4:'candidates1024', 5:'candidates2048', 6:'blend50'}


class RAMAdmissionUnavailable(MemoryError):
    """Exact full-group capacity is temporarily unavailable before admission."""

    def __init__(self, *, host_total_bytes, available_bytes, group_resident_bytes, required_bytes):
        observed = available_bytes + group_resident_bytes
        self.observation = dict(host_RAM_bytes=host_total_bytes,
            host_available_RAM_bytes=available_bytes,
            group_process_tree_RSS_bytes=group_resident_bytes,
            observed_available_plus_group_RSS_bytes=observed,
            required_group_and_cache_capacity_bytes=required_bytes,
            shortfall_bytes=max(0, required_bytes-observed))
        super().__init__('Actual available RAM plus deduplicated registered group RSS cannot admit'
            + str(required_bytes//GIB) + 'GiB: ' + json.dumps(self.observation, sort_keys=True))


def _absolute(value):
    return isinstance(value,str) and Path(value).is_absolute() and '..' not in Path(value).parts


class BasicLease(resource.BasicLease):
    """Original eight-CPU/96GiB own guard with complete group RSS accounting."""

    def __init__(self, request):
        super().__init__(request)
        path=request.get('shared_resource_registry')
        if not _absolute(path) or GUARD_FILE not in request['code_sha256']:
            raise ValueError('Immutable shared registry path and exact group guard SHA required')
        path=Path(path)
        deadline=time.monotonic()+120
        while not path.exists():
            remaining=min(self.lifetime['expires_at_unix']-time.time(),self.monotonic_expiry-time.monotonic())
            if remaining<resource.EXPIRY_RESERVE_SECONDS:
                raise RuntimeError('Host lifetime crossed180seconds while waiting for immutable group registry')
            resource._exact(self.request['controller_identity'])
            resource._exact(self.self_identity)
            if time.monotonic()>=deadline:
                raise RuntimeError('Immutable seven-controller registry publication unavailable after120seconds')
            time.sleep(.1)
        self._admit(path,resource.sha(path))
        document=json.loads(path.read_bytes())
        if (not isinstance(document,dict) or document.get('format')!=FORMAT
                or not isinstance(document.get('registration'),dict)
                or set(document['registration'])!={str(gpu)for gpu in range(7)}):
            raise ValueError('All seven exact controller registry entries are required')
        self.registry_path=path
        self.registry_sha256=resource.sha(path)
        self.registration=copy.deepcopy(document['registration'])
        self.registered_requests={}
        admitted_placement=placement.admit_placement(self.request,self._admit)
        self._group_previous_trees={}
        self._group_previous_shared_stage={}
        controller_pids=[]
        canonical_source=None
        for gpu in range(7):
            row=self.registration[str(gpu)]
            if (not isinstance(row,dict) or not {'root','arm','controller_identity','request_sha256'}<=set(row)
                    or not _absolute(row['root']) or row['arm']!=admitted_placement[gpu]
                    or not resource._identity(row['controller_identity'])
                    or not resource._checksum(row['request_sha256'])):
                raise ValueError('Exact seven-arm registry root/arm/controller/request SHA required')
            request_path=Path(row['root'])/'request.json'
            self._admit(request_path,row['request_sha256'])
            admitted=json.loads(request_path.read_bytes())
            resource.validate_request(admitted)
            placement.validate_peer_placement(self.request,admitted,admitted_placement)
            if (admitted['root']!=row['root'] or admitted.get('arm')!=row['arm']
                    or admitted['gpu']!=gpu or admitted['controller_identity']!=row['controller_identity']
                    or admitted['uid']!=resource.UID or admitted['cpu_affinity']!=resource.CPUS[gpu]
                    or admitted['RSS_limit_bytes']!=NATIVE_RSS_LIMIT):
                raise ValueError('Registered controller/request changed exact native arm/GPU/UID/eightCPU/96GiB identity')
            reference=admitted['host_lifetime']
            if (reference['sha256']!=self.request['host_lifetime']['sha256']
                    or reference['expires_at_unix']!=self.lifetime['expires_at_unix']):
                raise ValueError('All group controllers require the same immutable owned PBS host lifetime')
            self._admit(reference['path'],reference['sha256'])
            lifetime=json.loads(Path(reference['path']).read_bytes())
            if (lifetime.get('UID')!=resource.UID or lifetime.get('node')!=resource.NODE
                    or lifetime.get('job_id')!=self.lifetime['job_id']):
                raise ValueError('Registered group controller moved to a different UID/node/PBS job')
            source=(admitted['bank'],admitted['bank_sha256'])
            if canonical_source is None:canonical_source=source
            if source!=canonical_source:
                raise ValueError('Every registered arm must retain the same original Basic source bank')
            self._admit(source[0],source[1])
            if 'shared_resource_registry'in admitted and admitted['shared_resource_registry']!=str(path):
                raise ValueError('Registered controller points to a different shared resource registry')
            self.registered_requests[gpu]=admitted
            controller_pids.append(row['controller_identity']['pid'])
        if len(set(controller_pids))!=7:
            raise ValueError('Seven independent exact controller PIDs required; duplicate registry actor forbidden')
        own=self.registered_requests[self.request['gpu']]
        for name,value in own.items():
            if name in ('bank','bank_sha256')and self.request.get(name)!=value:
                source_name='source_bank'if name=='bank'else'source_bank_sha256'
                if self.request.get(source_name)!=value:
                    raise ValueError('Native derivative bank lost its exact registered original source connection')
            elif self.request.get(name)!=value:
                raise ValueError('Calling lease differs from its own immutable registry request: '+name)
        self._registration_bytes=json.dumps(self.registration,sort_keys=True,separators=(',',':'))
        self._registered_requests_bytes=json.dumps(self.registered_requests,sort_keys=True,separators=(',',':'))

    def _inputs(self):
        super()._inputs()
        if hasattr(self,'_registration_bytes')and(
                json.dumps(self.registration,sort_keys=True,separators=(',',':'))!=self._registration_bytes
                or json.dumps(self.registered_requests,sort_keys=True,separators=(',',':'))!=self._registered_requests_bytes):
            raise ValueError('Immutable seven-controller registration changed in memory')

    def _controller(self,gpu):
        identity=self.registration[str(gpu)]['controller_identity']
        try:
            process=psutil.Process(identity['pid'])
            metadata=resource._metadata(process)
            if any(metadata[key]!=identity[key]for key in ('pid','create_time','UID')):
                raise RuntimeError('Registered group controller PID/birth/UID was reused or became foreign')
            state=process.status()
            if state in (psutil.STATUS_ZOMBIE,psutil.STATUS_DEAD):
                return None,dict(status=state,identity=copy.deepcopy(identity),natural_terminal=True)
            try:
                exact=resource._exact(identity)
            except RuntimeError:
                if self._terminal_after_change(process,metadata,reason='registered group controller crossed natural exit'):
                    return None,dict(status='NaturalTerminalObservation',identity=copy.deepcopy(identity),natural_terminal=True)
                raise
            return exact,dict(status='LIVE',identity=copy.deepcopy(identity))
        except psutil.NoSuchProcess:
            return None,dict(status='NoSuchProcess',identity=copy.deepcopy(identity),natural_terminal=True)

    def _controller_stage_snapshot(self,gpu):
        for attempt in range(2):
            controller,observation=self._controller(gpu)
            if controller is None:return None,observation,None,[]
            try:
                return controller,observation,controller.cpu_affinity(),controller.children(recursive=True)
            except psutil.NoSuchProcess:
                if attempt==0:continue
                # Rebind the numeric PID one last time. PID reuse or a living
                # changed actor still fails; only genuine release is accepted.
                controller,observation=self._controller(gpu)
                if controller is None:return None,observation,None,[]
                raise RuntimeError('Live registered group controller lost its bounded ancestry observation')

    def _group_tree(self,gpu):
        controller,observation,affinity,candidates=self._controller_stage_snapshot(gpu)
        admitted=self.registered_requests[gpu]
        expected=list(resource.CPUS[gpu])
        shared=False
        if controller is not None and gpu==5 and affinity==SHARED_CPU_AFFINITY:
            if (admitted.get('shared_bank_CPU_affinity')!=SHARED_CPU_AFFINITY
                    or admitted.get('shared_bank_RSS_limit_GiB')!=288):
                raise RuntimeError('Existing GPU5 shared producer lacks explicit64CPU/288GiB admission')
            expected=list(SHARED_CPU_AFFINITY);shared=True
        if controller is not None:
            if affinity!=expected:
                raise RuntimeError('Registered group controller changed its admitted native/shared CPU affinity')
            tree={controller.pid:controller}
        else:
            # A terminal controller is not permission to ignore its previously
            # observed still-live children or to invent a released RSS value.
            tree={}
            candidates=list(self._group_previous_trees.get(gpu,{}).values())
            if gpu==5 and self._group_previous_shared_stage.get(gpu,False):
                expected=list(SHARED_CPU_AFFINITY);shared=True
        for process in candidates:
            try:
                process=self._descendant(process)
                if process is not None:
                    if process.cpu_affinity()!=expected:
                        raise RuntimeError('Exact group descendant changed admitted CPU affinity')
                    tree[process.pid]=process
            except psutil.NoSuchProcess:
                continue
        self._group_previous_trees[gpu]=dict(tree)
        self._group_previous_shared_stage[gpu]=shared
        observation.update(actual_CPU_affinity=expected,shared_CPU_bank_stage=shared,
                           surviving_exact_descendant_PIDs=sorted(tree))
        return tree,observation

    def _resources(self,controller,own,tree):
        observed=resource.capture_cgroup_admission(controller.pid)
        if self._controller_cgroup is None:self._controller_cgroup=observed
        elif observed!=self._controller_cgroup:
            raise RuntimeError('Exact group controller cgroup/ancestor memory limits changed')
        declared=self.request.get('controller_cgroup_admission')
        if declared is not None and declared!=observed:
            raise RuntimeError('Group controller differs from its observed cgroup admission')
        current=resource.capture_cgroup_admission(own.pid)
        for admission in(observed,current):
            for row in admission['memory_limits']:
                if row['limit_bytes']is not None and row['limit_bytes']<AGGREGATE_CAPACITY:
                    raise MemoryError('Actual inherited cgroup cannot admit the complete704GiB group')
        own_rss=self._rss(tree)
        all_trees={};observations={};group_pids={};candidate_pids={};rss_by_gpu={}
        for gpu in range(7):
            row_tree,observation=self._group_tree(gpu)
            observations[str(gpu)]=observation;all_trees[gpu]=row_tree;total=0
            for pid,process in row_tree.items():
                try:
                    if process.status()in(psutil.STATUS_ZOMBIE,psutil.STATUS_DEAD):continue
                    if process.uids().real!=resource.UID:
                        raise RuntimeError('Foreign UID appeared in the exact registered group tree')
                    rss=process.memory_info().rss
                    total+=rss
                    if pid in group_pids and group_pids[pid]!=rss:
                        raise RuntimeError('Shared group PID RSS changed across overlapping controller trees')
                    group_pids[pid]=rss
                    if gpu in(3,4,5):candidate_pids[pid]=rss
                except psutil.NoSuchProcess:continue
            rss_by_gpu[str(gpu)]=total
            if not(gpu==5 and observation['shared_CPU_bank_stage'])and total>NATIVE_RSS_LIMIT:
                raise MemoryError('Registered native GPU'+str(gpu)+' tree exceeds its unchanged96GiB RSS budget')
        group_rss=sum(group_pids.values());candidate_rss=sum(candidate_pids.values())
        if candidate_rss>CANDIDATE_RSS_LIMIT:
            raise MemoryError('Candidate GPU3/4/5 trees exceed their shared288GiB RSS budget')
        if group_rss>GROUP_RSS_LIMIT:
            raise MemoryError('All registered group trees exceed their unchanged672GiB RSS budget')
        shared_tree=all_trees[5]if observations['5']['shared_CPU_bank_stage']else{}
        if shared_tree:
            apps=resource._query(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader,nounits'],timeout=5)
            for line in apps.splitlines():
                if not line.strip():continue
                cells=[value.strip()for value in line.split(',')]
                if len(cells)!=2 or not cells[0].isdigit():
                    raise RuntimeError('Shared CPU producer CUDA observation requires exact numeric PID evidence')
                if int(cells[0])in shared_tree:
                    raise RuntimeError('Registered shared CPU bank actor created a forbidden CUDA context')
        memory=psutil.virtual_memory()
        if memory.total<AGGREGATE_CAPACITY:
            raise MemoryError('Actual total host RAM cannot admit the complete704GiB group')
        if memory.available+group_rss<AGGREGATE_CAPACITY:
            raise RAMAdmissionUnavailable(host_total_bytes=memory.total,available_bytes=memory.available,
                group_resident_bytes=group_rss,required_bytes=AGGREGATE_CAPACITY)
        free=shutil.disk_usage(self.request['native_root']).free
        if free<resource.STORAGE_FLOOR:
            raise OSError('Full native tmpfs output has less than its admitted10GiB free floor')
        return dict(own_process_tree_RSS_bytes=own_rss,own_process_tree_RSS_limit_bytes=NATIVE_RSS_LIMIT,
            aggregate_native_and_cache_capacity_bytes=AGGREGATE_CAPACITY,
            canonical_cache_reserve_bytes=32*GIB,host_RAM_bytes=memory.total,
            host_available_RAM_bytes=memory.available,native_free_bytes=free,
            native_free_floor_bytes=resource.STORAGE_FLOOR,actual_controller_cgroup=observed,
            actual_stage_cgroup=current,group_process_tree_RSS_bytes=group_rss,
            candidate_subpool_RSS_bytes=candidate_rss,candidate_subpool_RSS_limit_bytes=CANDIDATE_RSS_LIMIT,
            group_process_tree_RSS_limit_bytes=GROUP_RSS_LIMIT,group_RSS_by_GPU_bytes=rss_by_gpu,
            group_RSS_deduplicated_PIDs=sorted(group_pids),group_controller_observations=observations,
            group_registry_path=str(self.registry_path),group_registry_sha256=self.registry_sha256,
            RAM_admission_formula='actual_available_RAM + deduplicated_all_registered_group_tree_RSS >= 704GiB',
            scientific_scale_preserved=True,processes_signaled=False,PBS_GPU_owned_reservation=False,
            PBS_CPU_owned_reservation=False,PBS_job_role='host_lifetime_only',
            own_GPU=self.request['gpu'],own_CPU_affinity=self.request['cpu_affinity'])


SharedGroupLease=BasicLease
__all__=['BasicLease','SharedGroupLease','RAMAdmissionUnavailable','FORMAT','GUARD_FILE','AGGREGATE_CAPACITY',
         'GROUP_RSS_LIMIT','CANDIDATE_RSS_LIMIT','NATIVE_RSS_LIMIT','GIB']
