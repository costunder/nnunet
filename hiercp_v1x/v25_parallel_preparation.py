"""Execution-only four-process dispatch for the frozen complete v2.5 inputs.

This adapter never changes the frozen builder, cache keys, RNG, graph schema,
or required population. One coordinator publishes the original complete index
only after four disjoint patient shards have verified every canonical record.
Each worker retains the original four row threads and 64-GiB RSS guard. The
256-GiB worker plus 32-GiB cache admission is a host execution budget, not proof
of a scheduler reservation; the launcher must supply its resource admission.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
import gzip
import hashlib
import io
import json
import multiprocessing
import os
from pathlib import Path
import time
import traceback
import types
from concurrent.futures import ThreadPoolExecutor

import psutil
import torch

from .contracts import canonical_hash
from . import v25_factory as factory
from .v24_inputs import query_inputs, materialize_pair, tensor_digest
from hiercp_v22.contracts import sha

FORMAT = 'v25_four_process_exact_canonical_execution_v1'
SHARD_FORMAT = 'v25_exact_canonical_patient_shard_v1'
WORKER_COUNT = 4
WORKER_RSS_BYTES = 64 * 2**30
CACHE_BYTES = 32 * 2**30
HOST_BYTES = WORKER_COUNT * WORKER_RSS_BYTES + CACHE_BYTES
_FROZEN_PREPARE_LOCAL = factory.V25NativeInputs.prepare_local


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def split_cases(cases):
    """Each complete patient has one writer; no row or patient is omitted."""
    cases = list(cases)
    if not cases or len(cases) != len(set(cases)):
        raise ValueError('A nonempty unique complete patient population is required')
    return [cases[index::WORKER_COUNT] for index in range(WORKER_COUNT)]


def index_header(inputs):
    return dict(format=factory.CANONICAL_FORMAT,
        recipient_GT_used_in_forward=False, debug=False,
        query_rows_sha256=canonical_hash([query_inputs([row])[0] for row in inputs.inventory['records']]),
        config_sha256=inputs.config_sha256, source_sha256=inputs.source_sha256,
        prototype_bank_sha256=inputs.bank_sha256,
        context_contract_sha256=inputs.context_contract['contract_sha256'])


@contextmanager
def publication_lock(path):
    """Process lock only around original GraphWriter publication, on Linux."""
    if os.name != 'posix':
        raise RuntimeError('Four-process shared canonical publication requires POSIX flock')
    import fcntl
    with Path(path).open('a+b') as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _existing_record(root, relative, row):
    from hiercp_v22.storage import load_record
    path = Path(root)/relative
    record = load_record(root, relative)
    if query_inputs([dict(row,id=record['observation_id'],center=record['center'])])[0] != query_inputs([row])[0]:
        raise ValueError('Partial owned canonical query differs; file preserved: '+str(path))
    if tensor_digest({key:record[key] for key in
            ('source_patch','target_patch','source_local','target_local')}) != record['tensor_sha256']:
        raise ValueError('Partial owned canonical tensor proof differs; file preserved: '+str(path))
    with gzip.open(path,'rb') as stream:
        partial = torch.load(io.BytesIO(stream.read()),map_location='cpu',weights_only=False)
    return record, dict(path=relative,sha256=sha(path),shared_source=partial['shared_source'])


def prepare_cases(inputs, cases, *, lock_path, cancellation=None, progress=None):
    """Original complete case/row work; called independently in each shard."""
    from hiercp_v22.storage import GraphWriter
    root = inputs.root/'local'
    root.mkdir(exist_ok=True)
    writer = GraphWriter(root,minimum_free_bytes=inputs.runtime['minimum_free_disk_gib']*2**30)
    stored = []
    completed = 0
    def check_cancel():
        if cancellation is not None and cancellation.is_set():
            raise RuntimeError('Parallel execution admission failed; partial canonical files preserved')
    with ThreadPoolExecutor(max_workers=inputs.runtime['workers']) as pool:
        for case in cases:
            check_cancel()
            plan = inputs.population.case(case,128)
            rows = list(plan.query_rows)
            binding = inputs.input_binding(plan)
            # Bind the case and immutable binding as arguments, never through
            # a loop variable shared by asynchronously dispatched patients.
            def write(row, binding=binding, case=case):
                check_cancel()
                key = canonical_hash(dict(query=query_inputs([row])[0],inputs=binding))
                relative = 'records/'+key+'.pt.gz'
                if (root/relative).exists():
                    record, entry = _existing_record(root,relative,row)
                else:
                    record = inputs._local(row)
                    check_cancel()
                    source_key = (row['donor_case_id'],row['donor_component'],tuple(inputs.raw[case]['spacing']))
                    with publication_lock(lock_path):
                        # A duplicate partition is an error, not an overwrite.
                        entry = writer.write(relative,record,source_key)
                views,_,_ = materialize_pair(record,epoch=0)
                return dict(id=row['id'],query=query_inputs([row])[0],tensor_sha256=record['tensor_sha256'],
                    sampled_two_view_nodes=sum(sum(store.num_nodes for store in graph.node_stores) for graph in views),
                    sampled_two_view_edges=sum(sum(store.num_edges for store in graph.edge_stores) for graph in views),**entry)
            entries = list(pool.map(write,rows))
            stored.extend(entries)
            completed += 1
            inputs._rss()
            if progress is not None:
                progress(case,completed,len(cases),len(stored))
    return stored


def shard_receipt(inputs, cases, records, *, shard, execution_sha256, execution_identity=None):
    required = {row['id'] for case in cases for row in inputs.population.case(case,128).query_rows}
    actual = [row['id'] for row in records]
    if len(actual) != len(set(actual)) or set(actual) != required:
        raise ValueError('Complete canonical shard omitted or duplicated query IDs')
    bindings = {case:copy.deepcopy(inputs._input_bindings[case]) for case in cases}
    raw_files = inputs._binding_header()['raw_files']
    proofs = {}
    for path, proof in inputs._file_proofs.items():
        if path not in raw_files:
            continue
        if factory._stat(path) != proof:
            raise ValueError('SHA-admitted raw file changed before shard publication: '+path)
        proofs[path] = dict(sha256=raw_files[path],stat_identity=list(proof))
    value = dict(format=SHARD_FORMAT,shard=shard,cases=list(cases),
        index_header=index_header(inputs),bindings=bindings,raw_file_proofs=proofs,
        records=sorted(records,key=lambda row:row['id']),complete=True,
        execution_source_sha256=execution_sha256,
        execution_identity=copy.deepcopy(execution_identity),
        original_builder_used=True,physical_row_workers=inputs.runtime['workers'],
        worker_RSS_limit_bytes=inputs.rss_bytes)
    value['content_sha256'] = canonical_hash(value)
    return value


def merge_shards(inputs, receipts, shards, *, execution_sha256, execution_slots=None,
                 worker_PIDs=None, manifest_sha256=None):
    """Validate receipts and admit actual raw proofs before the original publish."""
    if len(receipts) != WORKER_COUNT or len(shards) != WORKER_COUNT:
        raise ValueError('Exactly four complete shard receipts are required')
    if execution_slots is not None and (len(execution_slots) != WORKER_COUNT
            or worker_PIDs is None or len(worker_PIDs) != WORKER_COUNT or manifest_sha256 is None):
        raise ValueError('Four actual child identities and immutable execution manifest SHA required')
    expected_header = index_header(inputs)
    bindings, raw_proofs, records = {}, {}, []
    expected_keys = {'format','shard','cases','index_header','bindings','raw_file_proofs',
        'records','complete','execution_source_sha256','original_builder_used',
        'physical_row_workers','worker_RSS_limit_bytes','execution_identity','content_sha256'}
    for shard, (receipt,cases) in enumerate(zip(receipts,shards)):
        if (set(receipt) != expected_keys or receipt['format'] != SHARD_FORMAT
                or receipt['shard'] != shard or receipt['cases'] != list(cases)
                or receipt['index_header'] != expected_header or receipt['complete'] is not True
                or receipt['execution_source_sha256'] != execution_sha256
                or receipt['original_builder_used'] is not True
                or receipt['physical_row_workers'] != inputs.runtime['workers']
                or receipt['worker_RSS_limit_bytes'] != WORKER_RSS_BYTES
                or receipt['content_sha256'] != canonical_hash({key:value for key,value in receipt.items() if key!='content_sha256'})
                or set(receipt['bindings']) != set(cases)):
            raise ValueError('Closed complete canonical shard identity/proof differs')
        if execution_slots is not None:
            identity = receipt['execution_identity']
            slot = execution_slots[shard]
            if (not isinstance(identity,dict)
                    or set(identity) != {'PID','create_time','command','UID','CPU_affinity',
                        'resource_request_path','resource_request_sha256','execution_manifest_sha256'}
                    or identity['PID'] != worker_PIDs[shard]
                    or identity['UID'] != 41833
                    or type(identity['create_time']) not in (int,float) or identity['create_time'] <= 0
                    or not isinstance(identity['command'],list) or not identity['command']
                    or identity['CPU_affinity'] != slot['CPU_affinity']
                    or identity['resource_request_path'] != slot['resource_request_path']
                    or identity['resource_request_sha256'] != slot['resource_request_sha256']
                    or identity['execution_manifest_sha256'] != manifest_sha256):
                raise ValueError('Actual shard PID/CPU/resource/manifest identity differs')
        required_ids = {row['id'] for case in cases for row in inputs.population.case(case,128).query_rows}
        ids = [row['id'] for row in receipt['records']]
        if len(ids) != len(set(ids)) or set(ids) != required_ids:
            raise ValueError('Canonical shard query set omitted or duplicated records')
        for case,binding in receipt['bindings'].items():
            inputs._validate_binding(case,binding)
            if case in bindings:
                raise ValueError('Canonical patient appears in two shards')
            bindings[case] = copy.deepcopy(binding)
        for path,proof in receipt['raw_file_proofs'].items():
            if path in raw_proofs and raw_proofs[path] != proof:
                raise ValueError('Shared immutable raw file proof differs between shards')
            raw_proofs[path] = proof
        records.extend(receipt['records'])
    expected_ids = [row['id'] for row in inputs.inventory['records']]
    actual_ids = [row['id'] for row in records]
    if len(actual_ids) != len(expected_ids) or len(actual_ids) != len(set(actual_ids)) or set(actual_ids) != set(expected_ids):
        raise ValueError('Full canonical union omitted or duplicated native records')
    header = inputs._binding_header()
    if set(bindings) != set(header['case_ids']) or set(raw_proofs) != set(header['raw_files']):
        raise ValueError('Complete patient bindings and SHA-admitted raw files required')
    # Re-read each real original tensor/file rather than trusting child receipts.
    for entry in records:
        relative = entry['path']
        expected_key = canonical_hash(dict(query=entry['query'],inputs=bindings[entry['query']['case_id']]))
        if relative != 'records/'+expected_key+'.pt.gz':
            raise ValueError('Canonical key is not bound to its actual patient inputs')
        path = inputs.root/'local'/relative
        if sha(path) != entry['sha256']:
            raise ValueError('Canonical compressed file changed before complete index admission')
        record, checked = _existing_record(inputs.root/'local',relative,entry['query'])
        if (record['tensor_sha256'] != entry['tensor_sha256']
                or checked['shared_source'] != entry['shared_source']):
            raise ValueError('Canonical actual tensor/shared-source proof differs')
    for path,checksum in header['raw_files'].items():
        proof = raw_proofs[path]
        if (set(proof) != {'sha256','stat_identity'} or proof['sha256'] != checksum
                or len(proof['stat_identity']) != 5
                or any(type(value) is not int for value in proof['stat_identity'])
                or factory._stat(path) != tuple(proof['stat_identity'])):
            raise ValueError('Actual SHA-admitted raw source changed: '+path)
    inputs._input_bindings = bindings
    inputs._file_proofs.update({path:tuple(proof['stat_identity']) for path,proof in raw_proofs.items()})
    bindings_sha256 = inputs._publish_input_bindings()
    index = inputs.root/'local/index.json'
    factory._publish(index,dict(**expected_header,input_bindings_sha256=bindings_sha256,
        records=sorted(records,key=lambda row:row['id']),complete=True))
    return index


def _worker(payload, cancellation):
    """Spawn entry; CUDA remains disabled and original sources stay untouched."""
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    inputs = None
    execution = Path(payload['execution_root'])
    receipt_path = execution/('shard_%d.json'%payload['shard'])
    try:
        if _sha(__file__) != payload['execution_source_sha256']:
            raise ValueError('Parallel execution source bytes changed')
        from .v25_preparation import install_runtime
        install_runtime(factory)
        psutil.Process().cpu_affinity(payload['CPU_affinity'])
        resource_guard = None
        if payload.get('resource_request_path') is not None:
            from .v25_direct_resources import validate_resource_proof
            proof_path = Path(payload['resource_request_path'])
            if sha(proof_path) != payload['resource_request_sha256']:
                raise ValueError('Assigned direct execution proof bytes changed')
            resource_guard = validate_resource_proof(proof_path,force=True)
            if set(resource_guard.request['CPU_affinity']) != set(payload['CPU_affinity']):
                raise ValueError('Assigned shard CPU affinity differs from its own direct proof')
        torch.set_num_threads(payload['runtime']['torch_threads'])
        inputs = factory.V25NativeInputs(payload['native_experiment'],payload['inventory_path'],
            payload['cache_root'],payload['runtime'])
        if index_header(inputs) != payload['index_header']:
            raise ValueError('Spawned original input identity differs')
        inputs._admit_runtime_files(tuple(inputs._runtime_file_sha256)+(Path(__file__),))
        if resource_guard is not None:
            original_rss = inputs._rss
            def guarded_rss(instance,*args,**kwargs):
                if cancellation.is_set():
                    raise RuntimeError('Parallel admission requested cooperative stop; partial outputs preserved')
                resource_guard.verify()
                return original_rss(*args,**kwargs)
            inputs._rss = types.MethodType(guarded_rss,inputs)
        def progress(case,done,total,records):
            print('V2.5 CANONICAL SHARD%d case=%s patients=%d/%d records=%d'%
                (payload['shard']+1,case,done,total,records),flush=True)
        records = prepare_cases(inputs,payload['cases'],lock_path=execution/'canonical_publication.lock',
            cancellation=cancellation,progress=progress)
        if resource_guard is not None:
            resource_guard.verify(force=True)
        inputs.guard_source()
        actor = psutil.Process()
        execution_identity = dict(PID=actor.pid,create_time=actor.create_time(),command=actor.cmdline(),
            UID=os.getuid(),CPU_affinity=actor.cpu_affinity(),
            resource_request_path=payload['resource_request_path'],
            resource_request_sha256=payload['resource_request_sha256'],
            execution_manifest_sha256=payload['execution_manifest_sha256'])
        factory._publish(receipt_path,shard_receipt(inputs,payload['cases'],records,
            shard=payload['shard'],execution_sha256=payload['execution_source_sha256'],
            execution_identity=execution_identity))
    except Exception as error:
        cancellation.set()
        factory._publish(execution/('shard_%d_failure.json'%payload['shard']),
            dict(error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc(),
                partial_outputs_preserved=True))
        raise
    finally:
        if inputs is not None:
            inputs.close()


def cache_allocated_bytes(root):
    total = 0
    root = Path(root)
    if root.is_symlink():
        raise ValueError('Shared canonical cache cannot be a symlink')
    for path in root.rglob('*'):
        if path.is_symlink():
            raise ValueError('Shared canonical cache contains a symlink')
        if path.is_file():
            value = path.stat()
            total += getattr(value,'st_blocks',(value.st_size+511)//512)*512
    return total


def memory_admission(worker_rss, cache_bytes, *, coordinator_rss=0):
    """Explicit host aggregate admission, never a claim about PBS ownership."""
    if len(worker_rss) != WORKER_COUNT or any(value < 0 for value in worker_rss) or cache_bytes < 0 or coordinator_rss < 0:
        raise ValueError('Four actual worker RSS values and cache allocation are required')
    if any(value > WORKER_RSS_BYTES for value in worker_rss):
        raise MemoryError('A full original worker exceeds64GiB RSS; partial files preserved')
    if cache_bytes > CACHE_BYTES:
        raise MemoryError('Complete shared canonical cache exceeds32GiB host admission')
    used = sum(worker_rss)+coordinator_rss+cache_bytes
    if used > HOST_BYTES:
        raise MemoryError('Parallel host aggregate exceeds288GiB; partial files preserved')
    return dict(worker_RSS_bytes=list(worker_rss),coordinator_RSS_bytes=coordinator_rss,
        cache_allocated_bytes=cache_bytes,total_accounted_bytes=used,
        worker_RSS_limit_bytes=WORKER_RSS_BYTES,cache_limit_bytes=CACHE_BYTES,
        host_aggregate_limit_bytes=HOST_BYTES,scheduler_reservation_proven=False)


def prepare_four_way(inputs, *, native_experiment, inventory_path, execution_root,
                     resource_admission, execution_slots=None, manifest_path=None):
    """Full14102 production dispatch; caller owns original prepare-stage lock."""
    if os.name != 'posix':
        raise RuntimeError('Production parallel preparation requires POSIX spawn/flock')
    if (inputs.runtime['workers'] != 4 or inputs.rss_bytes != WORKER_RSS_BYTES
            or inputs.raw_resident_bytes != 8*2**30):
        raise ValueError('Frozen full four-row-worker/64GiB/8GiB input runtime required')
    if not isinstance(resource_admission,dict) or resource_admission.get('authorized_host_aggregate_limit_bytes') != HOST_BYTES:
        raise ValueError('Explicit288GiB host execution admission required; PBS is not inferred')
    if execution_slots is None or len(execution_slots) != WORKER_COUNT or manifest_path is None:
        raise ValueError('Four independently guarded direct slots and immutable execution manifest required')
    index = inputs.root/'local/index.json'
    if index.exists():
        # The frozen original complete-cache admission is authoritative.
        return _FROZEN_PREPARE_LOCAL(inputs)
    execution = Path(execution_root).resolve()
    if execution.exists():
        raise FileExistsError('Use a fresh execution receipt namespace; prior outputs preserved')
    execution.mkdir(parents=True)
    cases = inputs.population.partition_cases('inner_train')+inputs.population.partition_cases('inner_val')
    shards = split_cases(cases)
    execution_sha256 = _sha(__file__)
    cpu_affinity = psutil.Process().cpu_affinity()
    if len(cpu_affinity) < inputs.runtime['workers']:
        raise ValueError('Actual parallel CPU affinity must admit original four row workers')
    request = dict(format=FORMAT,cache_root=str(inputs.root),native_experiment=str(Path(native_experiment).resolve()),
        inventory_path=str(Path(inventory_path).resolve()),runtime=copy.deepcopy(inputs.runtime),
        index_header=index_header(inputs),shards=shards,execution_source_sha256=execution_sha256,
        worker_processes=4,row_threads_per_process=4,CPU_affinity=cpu_affinity,
        resource_admission=copy.deepcopy(resource_admission),scheduler_reservation_inferred=False,
        scientific_sources_unchanged=True,debug=False)
    if execution_slots is not None:
        if len(execution_slots) != WORKER_COUNT:
            raise ValueError('Exactly four separately admitted execution slots required')
        request['execution_slots'] = copy.deepcopy(execution_slots)
    if manifest_path is not None:
        request['execution_manifest_path'] = str(Path(manifest_path).resolve())
        request['execution_manifest_sha256'] = sha(manifest_path)
    factory._publish(execution/'request.json',request)
    inputs._admit_runtime_files(tuple(inputs._runtime_file_sha256)+(Path(__file__),))
    context = multiprocessing.get_context('spawn')
    cancellation = context.Event()
    processes = []
    guard_failure = None
    last_measurement = 0.
    try:
        for shard,subset in enumerate(shards):
            payload = dict(request,shard=shard,cases=subset,execution_root=str(execution))
            if execution_slots is not None:
                payload.update(execution_slots[shard])
            process = context.Process(target=_worker,args=(payload,cancellation),name='v25-canonical-shard%d'%shard)
            process.start()
            processes.append(process)
        factory._publish(execution/'workers.json',dict(workers=[dict(shard=index,PID=process.pid)
            for index,process in enumerate(processes)],started_at=time.time()))
        while any(process.is_alive() for process in processes):
            now = time.monotonic()
            if now-last_measurement >= 10:
                rss = []
                for process in processes:
                    try:
                        rss.append(psutil.Process(process.pid).memory_info().rss if process.is_alive() else 0)
                    except psutil.NoSuchProcess:
                        rss.append(0)
                try:
                    sample = memory_admission(rss,cache_allocated_bytes(inputs.root),
                        coordinator_rss=psutil.Process().memory_info().rss)
                    print('V2.5 FOUR-WAY CPU RSS_GiB=%s cache_GiB=%.3f'%
                        ([round(value/2**30,3) for value in rss],sample['cache_allocated_bytes']/2**30),flush=True)
                except (MemoryError,ValueError) as error:
                    if guard_failure is None:
                        guard_failure = error
                        cancellation.set()
                last_measurement = now
            if any(process.exitcode not in (None,0) for process in processes):
                cancellation.set()
            # Never terminate/kill a child or session. Admission failure requests
            # cooperative stop at each complete original builder boundary.
            time.sleep(.25)
    except BaseException:
        cancellation.set()
        raise
    finally:
        # Any started child is joined after cooperative stop, preserving its
        # own exact exception, receipt and all previously completed files.
        for process in processes:
            process.join()
    if guard_failure is not None:
        raise guard_failure
    failures = [process.exitcode for process in processes if process.exitcode != 0]
    if failures:
        raise RuntimeError('Canonical shard failed%s; original error receipts and partial files preserved'%failures)
    if _sha(__file__) != execution_sha256:
        raise ValueError('Parallel execution source changed before complete cache admission')
    if manifest_path is not None and sha(manifest_path) != request['execution_manifest_sha256']:
        raise ValueError('Execution manifest changed before complete cache admission')
    if execution_slots is not None:
        for slot in execution_slots:
            if sha(slot['resource_request_path']) != slot['resource_request_sha256']:
                raise ValueError('An assigned execution proof changed before complete cache admission')
    inputs.guard_source()
    receipts = [json.loads((execution/('shard_%d.json'%index)).read_text()) for index in range(WORKER_COUNT)]
    index = merge_shards(inputs,receipts,shards,execution_sha256=execution_sha256,
        execution_slots=execution_slots,worker_PIDs=[process.pid for process in processes],
        manifest_sha256=request['execution_manifest_sha256'])
    factory._publish(execution/'completion.json',dict(format=FORMAT,complete=True,debug=False,
        records=len(inputs.inventory['records']),cases=len(cases),shard_count=4,
        local_index_sha256=sha(index),execution_source_sha256=execution_sha256,
        scientific_sources_unchanged=True,scheduler_reservation_inferred=False))
    return index


def prepare_local_parallel(inputs, manifest_path):
    """Launcher API: four independent direct proofs and CPU sets, one cache."""
    path = Path(manifest_path).resolve(strict=True)
    before = factory._stat(path)
    manifest = json.loads(path.read_text(encoding='utf8'))
    if factory._stat(path) != before:
        raise ValueError('Parallel execution manifest changed during admission')
    required = {'native_experiment','inventory_path','execution_root','group_root','slots'}
    if not required <= set(manifest) or set(manifest['slots']) != set('ABCD'):
        raise ValueError('Four-arm direct execution manifest and native inputs required')
    slots = []
    used_cpus, used_gpus = set(), set()
    for arm in 'ABCD':
        slot = manifest['slots'][arm]
        keys = {'resource_request_path','CPU_affinity','gpu','arm','resource_request_sha256'}
        if not keys <= set(slot) or slot['arm'] != arm:
            raise ValueError('Independent direct execution slot identity differs')
        cpus = slot['CPU_affinity']
        if (not isinstance(cpus,list) or len(cpus) < 4 or len(cpus) != len(set(cpus))
                or any(type(cpu) is not int or cpu < 0 for cpu in cpus)
                or used_cpus.intersection(cpus) or type(slot['gpu']) is not int
                or slot['gpu'] < 0 or slot['gpu'] in used_gpus):
            raise ValueError('Four distinct GPUs and disjoint admitted CPU sets required')
        proof = Path(slot['resource_request_path']).resolve(strict=True)
        if sha(proof) != slot['resource_request_sha256']:
            raise ValueError('Direct slot resource proof differs from execution manifest')
        used_cpus.update(cpus); used_gpus.add(slot['gpu'])
        slots.append({key:copy.deepcopy(slot[key]) for key in keys})
    # The direct resource guard checks each actual child after that child pins
    # its own CPU set. The GPU is reserved for later neural stages; these
    # preparation children keep CUDA_VISIBLE_DEVICES empty throughout.
    admission = dict(authorized_host_aggregate_limit_bytes=HOST_BYTES,
        worker_RSS_limit_bytes=WORKER_RSS_BYTES,shared_cache_limit_bytes=CACHE_BYTES,
        independent_direct_execution_proofs=4,CPU_affinity=sorted(used_cpus),
        resource_manifest_sha256=sha(path),scheduler_reservation_inferred=False)
    return prepare_four_way(inputs,native_experiment=manifest['native_experiment'],
        inventory_path=manifest['inventory_path'],execution_root=manifest['execution_root'],
        resource_admission=admission,execution_slots=slots,manifest_path=path)


def install_parallel_preparation(manifest_path, *, factory_module=factory):
    """Patch only the execution method after installing the frozen CPU adapter."""
    from .v25_preparation import AcceleratedNativeInputs
    if factory_module is not factory or factory_module.V25NativeInputs is not AcceleratedNativeInputs:
        raise ValueError('Install the frozen AcceleratedNativeInputs adapter first')
    cls = factory_module.V25NativeInputs
    original = cls.prepare_local
    if hasattr(original,'__v25_original__'):
        raise ValueError('Parallel preparation execution adapter is already installed')
    def prepare(instance):
        return prepare_local_parallel(instance,manifest_path)
    prepare.__v25_original__ = original
    cls.prepare_local = prepare
    return dict(format=FORMAT,execution_source_sha256=_sha(__file__),workers=4,
        row_threads_per_worker=4,host_aggregate_limit_bytes=HOST_BYTES,
        shared_complete_canonical_cache=True,scientific_sources_unchanged=True,
        scheduler_reservation_inferred=False)
