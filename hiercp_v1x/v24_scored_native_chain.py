"""Append-only CPU controller for an external completed full642 score overlay.

No scorer, model or checkpoint loader is constructed by the waiting controller.
Only fresh native preparation, unchanged B2/B4 clone calibration, and original
B2/full250 training plus BEST/full26 validation may be launched.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import time

FORMAT = 'v24_scored_bank_to_native_chain_request_v1'
STATUS_FORMAT = 'v24_scored_bank_to_native_chain_status_v1'
REGISTRATION_FORMAT = 'v24_scored_bank_to_native_chain_registration_v1'
STORAGE_FORMAT = 'v24_scored_native_storage_ready_v1'
CLAIM_FORMAT = 'v24_external_scored_bank_native_exclusive_claim_v1'
SCORING_FORMAT = 'v24_external_completed_scoring_native_admission_v1'
PROFILE = 'reuse_historical_Basic642_scores_only_and_full131_Blosc2_readonly'
ACTIONS = ('prepare-native', 'calibrate-native', 'train')
AFFINITIES = {4: [0, 1, 2, 3], 5: [42, 43, 44, 45], 6: [32, 33, 34, 35]}
RESERVE = 10 * 2**30 + 512 * 2**20
CAPACITY_ERROR = 'Complete remaining three-arm writes plus10GiB/growth reserve do not fit'
FILES = ('hiercp_v1x/v24_scored_native_chain.py', 'tools/watch_v24_scored_bank_to_nnunet.py',
    'tools/run_v24_matched_basic_cp.py', 'hiercp_v1x/v24_matched_basic_cp_pipeline.py',
    'hiercp_v1x/v24_direct_gpu_access.py', 'hiercp_v1x/v24_matched_basic_cp_runtime.py',
    'hiercp_v1x/v24_native_best_eval_runtime.py', 'hiercp_v1x/v24_readonly_native_storage.py',
    'hiercp_v1x/v24_readonly_static_operator_storage.py', 'hiercp_v1x/v24_nnunet_cp.py')


class PendingMetadata(RuntimeError):
    """A newly published record is visibly incomplete; bounded retry only."""
    def __init__(self, message, *, path=None, observed_stat=None):
        super().__init__(message)
        self.signature = (str(path), tuple(observed_stat or ()), message)


class WaitingForStorage(RuntimeError):
    pass


class WaitingForGPU(RuntimeError):
    pass


class WaitingForAllocation(RuntimeError):
    pass


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 2**20), b''):
            result.update(chunk)
    return result.hexdigest()


def checksum(value):
    return isinstance(value, str) and len(value) == 64 and not set(value) - set('0123456789abcdef')


def absolute(value):
    if not isinstance(value, str) or not Path(value).is_absolute() or '..' in Path(value).parts:
        raise ValueError('Absolute non-traversing path required: ' + repr(value))
    return Path(value)


def owned(path, *, directory=False, missing=False):
    path = absolute(str(path))
    for part in (path, *path.parents):
        if part.is_symlink():
            raise ValueError('Symlink controller/input/output alias refused: ' + str(part))
    if missing and not path.exists():
        parent = path.parent
        while not parent.exists():
            parent = parent.parent
        owned(parent, directory=True)
        return path
    info = path.stat()
    if not hasattr(os, 'getuid') or os.getuid() != 41833 or info.st_uid != 41833:
        raise ValueError('Actual own UID41833 regular input/output required: ' + str(path))
    if not (stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode)):
        raise ValueError('Actual regular input/output required: ' + str(path))
    return path


def identity(path):
    value = Path(path).stat()
    return [value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns]


def read(path):
    path = owned(path)
    before = identity(path)
    raw = path.read_bytes()
    if identity(path) != before:
        raise PendingMetadata('Record changed while reading: ' + str(path), path=path, observed_stat=identity(path))
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise PendingMetadata('Incomplete JSON publication: ' + str(path) + ': ' + str(error), path=path, observed_stat=before) from error
    if not isinstance(value, dict):
        raise ValueError('JSON object required: ' + str(path))
    return value


def publish(path, value, *, replace=False):
    """Atomic new-file publication; only this controller's status is replaced."""
    path = Path(path)
    temporary = path.parent / ('.' + path.name + '.' + str(os.getpid()) + '.' + str(time.time_ns()) + '.tmp')
    with temporary.open('x', encoding='utf8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    try:
        if replace:
            if path.exists():
                owned(path)
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


class FileProofCache:
    """Hash every immutable witness once; check its complete stat every use."""
    def __init__(self):
        self.records = {}

    def verify(self, proof):
        if (not isinstance(proof, dict) or set(proof) != {'path', 'sha256', 'stat'}
                or not checksum(proof['sha256']) or not isinstance(proof['stat'], list)
                or len(proof['stat']) != 5):
            raise ValueError('Exact immutable path/raw SHA/full stat witness required')
        path = owned(proof['path']); before = identity(path)
        key = (str(path), proof['sha256'], tuple(proof['stat']))
        if before != proof['stat']:
            self.records.pop(str(path), None)
            raise ValueError('Immutable input identity changed: ' + str(path))
        if self.records.get(str(path)) != key:
            self.records.pop(str(path), None)
            if sha(path) != proof['sha256'] or identity(path) != before:
                raise ValueError('Immutable input SHA/stat changed: ' + str(path))
            self.records[str(path)] = key
        return path

    def capture(self, path):
        path = owned(path); before = identity(path); digest = sha(path)
        if identity(path) != before:
            raise ValueError('Input changed during admission: ' + str(path))
        proof = dict(path=str(path), sha256=digest, stat=before)
        self.verify(proof)
        return proof

    def check(self):
        for path, (_, digest, info) in list(self.records.items()):
            self.verify(dict(path=path, sha256=digest, stat=list(info)))


def claim_path(request):
    source = absolute(request['source_output'])
    key = hashlib.sha256((str(request['GPU_arm']) + ':' + str(source) + ':' + PROFILE + ':context_depth1').encode()).hexdigest()
    return source.parent / 'scored_native_chain_claims' / (key + '.json')


def build_request(**fields):
    """Pure installer helper: explicit supplied paths/identities, fixed science.

    The installer seals a separate direct stage declaration before adding the
    direct_GPU_access witness. No allocation, file or process is created here.
    """
    request = dict(format=FORMAT, context_depth1=True, storage_profile=PROFILE,
        scoring_RAM_GiB=64, native_RAM_GiB=48, native_only=True, epochs=250,
        physical_batch=2, gradient_accumulation=1, patch_size=[128]*3,
        cp_probability=.5, train_cases=105, validation_cases=26, source_entries=642,
        candidates_per_source=128, eligible_cases=81, evaluation_checkpoint='checkpoint_best.pth',
        minimum_runtime_free_disk_GiB=10.5, debug=False, rescore=False,
        native_predecessors=[], source_claim_witnesses=[], stunet_checkpoint=None,
        stunet_checkpoint_sha256=None, direct_GPU_access=None)
    request.update(copy.deepcopy(fields))
    gpu = request['GPU_arm']; request['CPU_affinity'] = AFFINITIES[gpu]
    root, scoring = Path(request['chain_root']), Path(request['scoring_root'])
    request['native_claim_path'] = str(claim_path(request))
    entry = str(Path(request['code'])/'tools/run_v24_matched_basic_cp.py')
    request['stages'] = []
    for action in ACTIONS:
        tail = ['--bank',str(scoring/'bank/score_overlay.json'),'--output',str(root/'native')] if action=='prepare-native' else ['--native',str(root/'native/native.json')]
        request['stages'].append(dict(name=action.replace('-','_'),action=action,
            command=[request['python'],'-B','-u',entry,action,*tail,'--gpu',str(gpu),
                     '--storage-admission',request['storage_admission']]))
    return validate_declaration(request)


def requests_from_inspection(inspection, *, code, commit, code_files, group, scoring_roots):
    """Read-only installer assembly from exact captured old actor/claim records.

    All actual science/Git/source paths are inspected again. It does not write
    requests, alter a prior claim, acquire a GPU or launch a process.
    """
    group = absolute(str(group)); code = absolute(str(code)); output = {}
    arms = {'gpu'+str(gpu):str(group/('gpu'+str(gpu))) for gpu in AFFINITIES}
    for row in inspection['arms']:
        gpu = row['gpu']; source = owned(row['gnn'],directory=True)
        science = read(source/'request.json')
        source_code = row['direct_request']['source_code'] if gpu in (4,5) else row['pin']['source_code']
        original = owned(source_code,directory=True)
        source_files = copy.deepcopy(science['source'])
        if gpu in (4,5):
            continuation = read(source/'interrupted_gnn_continuation.json')
            source_files.update({name:proof['sha256'] for name,proof in continuation['source']['source_files'].items()})
            execution_code = continuation['execution_code']
            execution_files = {name:proof['sha256'] for name,proof in continuation['execution_files'].items()}
        else:
            execution_code, execution_files = source_code, copy.deepcopy(source_files)
        revision = lambda path:subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()
        scoring_root = absolute(str(scoring_roots[gpu])); actual_scoring = scoring_root/'request.json'
        scoring = read(actual_scoring) if actual_scoring.exists() else None
        worker = row['scoring_registration']['command'][3] if gpu==6 else (
            read(scoring_root/'registration.json')['command'][3] if scoring else str(code/'tools/run_v24_completed_matched_scoring.py'))
        if scoring is None and gpu!=5:
            raise ValueError('GPU4/6 must reuse their existing full642 scoring job')
        args = row.get('direct_request',{}).get('original_args',[])
        arg = lambda key: args[args.index(key)+1] if key in args else None
        inventory = scoring['inventory_path'] if scoring else arg('--inventory')
        cache = scoring['input_cache'] if scoring else arg('--input-cache')
        baseline = scoring['baseline_preprocessed'] if scoring else read(absolute(scoring_roots[4])/'request.json')['baseline_preprocessed']
        bank = scoring['source_bank'] if scoring else read(absolute(scoring_roots[4])/'request.json')['source_bank']
        checkpoint = scoring['stunet_checkpoint'] if scoring else None
        predecessor = [dict(root=item['root'],**{key:copy.deepcopy(item[key]['proof']) for key in ('request','status','registration')})
                       for item in row['predecessors']]
        claims = [copy.deepcopy(item.get('proof',item)) for item in row['claims']]
        allocation = inspection['allocation']
        scheduler = dict(job_id=allocation['PBS_job'],state='R',expected_owner='aicompetition06',expected_node='ece-a6gpu6',
            minimum_remaining_seconds=28800,native_minimum_remaining_seconds=25200,
            expires_at_unix=row['scoring_request']['allocation_expiry_unix']) if gpu==6 else None
        uuid = row['direct_request']['GPU_UUID'] if gpu in (4,5) else row['scoring_request']['resource_at_launch']['GPU'].split(',')[0].strip()
        output[gpu] = build_request(GPU_arm=gpu,GPU_UUID=uuid,chain_root=arms['gpu'+str(gpu)],arm_roots=arms,
            scoring_root=str(scoring_root),source_output=str(source),source_code=source_code,source_commit=revision(original),source_files=source_files,
            source_execution_code=execution_code,source_execution_commit=revision(execution_code),source_execution_files=execution_files,
            code=str(code),commit=commit,code_files=code_files,python=row.get('training_registration',row.get('scoring_registration'))['command'][0],
            inventory=inventory,input_cache=cache,baseline_preprocessed=baseline,historical_basic_bank=bank,
            storage_admission=str(group/'storage_admission.json'),storage_binding=str(group/'storage_binding.json'),
            scoring_worker=worker,scoring_worker_sha256=scoring['worker_sha256'] if scoring else code_files['tools/run_v24_completed_matched_scoring.py'],
            stunet_checkpoint=checkpoint,stunet_checkpoint_sha256=sha(owned(checkpoint)) if checkpoint else None,
            native_predecessors=predecessor,source_claim_witnesses=claims,scheduler=scheduler,PBS_owned_reservation=gpu==6,
            direct_mode='previous_explicit_direct_physical_GPU_number' if gpu in (4,5) else None)
    if set(output)!={4,5,6}:
        raise ValueError('Full three-arm installer metadata required')
    return output


def seal_request(request, *, host_lifetime=None):
    """Write only new owned controller/declaration/proof files, exclusively."""
    request = copy.deepcopy(validate_declaration(request)); root=owned(request['chain_root'],directory=True)
    if request['GPU_arm'] in (4,5):
        from . import v24_direct_gpu_access as direct
        if host_lifetime is None:
            raise ValueError('Exact independent current host lifetime witness required')
        stage=root/'direct_GPU_stage_request.json'
        publish(stage,direct.stage_request_document(request))
        proof=root/'direct_GPU_access_proof.json'
        publish(proof,dict(format=direct.FORMAT,GPU_arm=request['GPU_arm'],GPU_UUID=request['GPU_UUID'],
            CPU_affinity=request['CPU_affinity'],UID=41833,PBS_owned_reservation=False,
            other_PBS_reservations_known_not_owned=True,explicit_existing_direct_execution_requested=True,
            guard_sha256=sha(direct.__file__),request=dict(path=str(stage),sha256=sha(stage)),
            host_container_lifetime=copy.deepcopy(host_lifetime),poll_seconds=30,minimum_initial_remaining_seconds=28800,
            native_minimum_initial_remaining_seconds=25200,expiry_safety_reserve_seconds=180))
        request['direct_GPU_access']=dict(path=str(proof),sha256=sha(proof),stat=identity(proof))
        direct.validate_controller_request(request)
    publish(root/'request.json',request)
    return request


def validate_declaration(request):
    """Pure structural contract used by both controller and fresh stage guards."""
    gpu = request.get('GPU_arm')
    if (request.get('format') != FORMAT or type(gpu) is not int or gpu not in AFFINITIES
            or request.get('CPU_affinity') != AFFINITIES[gpu] or request.get('context_depth1') is not True
            or request.get('storage_profile') != PROFILE or request.get('scoring_RAM_GiB') != 64
            or request.get('native_RAM_GiB') != 48 or request.get('native_only') is not True
            or request.get('epochs') != 250 or request.get('physical_batch') != 2
            or request.get('gradient_accumulation') != 1 or request.get('patch_size') != [128] * 3
            or request.get('cp_probability') != .5 or request.get('train_cases') != 105
            or request.get('validation_cases') != 26 or request.get('source_entries') != 642
            or request.get('candidates_per_source') != 128 or request.get('eligible_cases') != 81
            or request.get('evaluation_checkpoint') != 'checkpoint_best.pth'
            or request.get('minimum_runtime_free_disk_GiB') != 10.5
            or request.get('debug') is not False or request.get('rescore') is not False
            or not isinstance(request.get('GPU_UUID'), str) or not request['GPU_UUID'].startswith('GPU-')):
        raise ValueError('Explicit native-only full642/B2/250/105+26/BEST/reserve contract required')
    for key in ('chain_root', 'scoring_root', 'source_output', 'source_code', 'source_execution_code',
                'code', 'python', 'inventory', 'input_cache', 'baseline_preprocessed',
                'historical_basic_bank', 'storage_admission', 'storage_binding', 'scoring_worker'):
        absolute(request.get(key))
    chain = absolute(request['chain_root']); scoring = absolute(request['scoring_root'])
    for protected in (scoring, absolute(request['source_output']), absolute(request['source_code']),
            absolute(request['source_execution_code']), absolute(request['code']), absolute(request['input_cache']),
            absolute(request['historical_basic_bank']), absolute(request['baseline_preprocessed'])):
        if chain == protected or chain.is_relative_to(protected) or protected.is_relative_to(chain):
            raise ValueError('Private native root must not overlap any previous inputs/code/results')
    arms = request.get('arm_roots')
    if (not isinstance(arms, dict) or set(arms) != {'gpu4', 'gpu5', 'gpu6'}
            or arms['gpu' + str(gpu)] != str(chain)
            or len({str(absolute(path).parent) for path in arms.values()}) != 1
            or len(set(arms.values())) != 3):
        raise ValueError('Exactly three distinct own arm roots in one immutable native group required')
    group = chain.parent
    if (absolute(request['storage_admission']).parent != group or absolute(request['storage_binding']).parent != group
            or request.get('native_claim_path') != str(claim_path(request))):
        raise ValueError('Exact group storage binding and globally derived scored-source claim required')
    for key, commit, files in (('code', 'commit', 'code_files'),
            ('source_code', 'source_commit', 'source_files'),
            ('source_execution_code', 'source_execution_commit', 'source_execution_files')):
        revision = request.get(commit); mapping = request.get(files)
        if (not isinstance(revision, str) or len(revision) != 40 or set(revision) - set('0123456789abcdef')
                or not isinstance(mapping, dict) or not mapping):
            raise ValueError('Full pinned Git commit/source manifest required: ' + key)
        for name, digest in mapping.items():
            if not isinstance(name, str) or Path(name).is_absolute() or '..' in Path(name).parts or not checksum(digest):
                raise ValueError('Exact relative source SHA manifest required')
    if not set(FILES) <= set(request['code_files']) or not checksum(request.get('scoring_worker_sha256')):
        raise ValueError('Actual native controller/guards/runtime/scoring worker must be pinned')
    if gpu in (4, 6):
        absolute(request.get('stunet_checkpoint'))
        if not checksum(request.get('stunet_checkpoint_sha256')):
            raise ValueError('Original official STU checkpoint SHA required')
    elif request.get('stunet_checkpoint') is not None or request.get('stunet_checkpoint_sha256') is not None:
        raise ValueError('Original CNN/GAT arm cannot acquire STU initialization')
    stages = request.get('stages')
    if not isinstance(stages, list) or tuple(row.get('action') for row in stages) != ACTIONS:
        raise ValueError('Only external-score native preparation/calibration/full training stages allowed')
    entry = str(absolute(request['code']) / 'tools/run_v24_matched_basic_cp.py')
    for action, row in zip(ACTIONS, stages):
        tail = ['--bank', str(scoring / 'bank/score_overlay.json'), '--output', str(chain / 'native')] if action == 'prepare-native' else ['--native', str(chain / 'native/native.json')]
        expected = [request['python'], '-B', '-u', entry, action, *tail,
            '--gpu', str(gpu), '--storage-admission', request['storage_admission']]
        if row != dict(name=action.replace('-', '_'), action=action, command=expected):
            raise ValueError('Exact original native CLI/arm/external overlay/no-resume command required: ' + action)
    if not isinstance(request.get('native_predecessors'), list) or not isinstance(request.get('source_claim_witnesses'), list):
        raise ValueError('Explicit append-only original controller/claim preservation declarations required')
    if gpu in (4, 5):
        if (request.get('scheduler') is not None or request.get('PBS_owned_reservation') is not False
                or request.get('direct_mode') != 'previous_explicit_direct_physical_GPU_number'):
            raise ValueError('GPU4/5 require honest fresh existing-direct access, never GPU6 PBS ownership')
    else:
        scheduler = request.get('scheduler')
        if (not isinstance(scheduler, dict) or request.get('direct_GPU_access') is not None
                or request.get('PBS_owned_reservation') is not True or scheduler.get('state') != 'R'
                or scheduler.get('expected_owner') != 'aicompetition06' or scheduler.get('expected_node') != 'ece-a6gpu6'
                or not isinstance(scheduler.get('job_id'), str) or not scheduler['job_id']
                or scheduler.get('minimum_remaining_seconds') != 28800
                or scheduler.get('native_minimum_remaining_seconds') != 25200
                or type(scheduler.get('expires_at_unix')) not in (int, float)
                or not math.isfinite(scheduler['expires_at_unix'])):
            raise ValueError('Actual distinct owned GPU6 PBS lease and original lifetime floors required')
    return request


def process_snapshot(pid, birth, command):
    import psutil
    if (type(pid)is not int or pid<=0 or type(birth)not in (int,float) or not math.isfinite(birth)
            or not isinstance(command,list) or not command or not all(isinstance(arg,str) for arg in command)):
        raise ValueError('Actual registered PID/birth/full command required')
    try:
        process = psutil.Process(pid)
        if process.create_time()!=birth or process.uids().real!=41833:
            raise ValueError('Exact registered PID/birth/full argv/UID changed')
        if process.status()==psutil.STATUS_ZOMBIE:
            return None  # A zombie owns no work; Linux no longer exposes argv.
        if process.cmdline()!=command:
            raise ValueError('Exact registered PID/birth/full argv/UID changed')
        return process
    except psutil.NoSuchProcess:
        return None


def verify_code(request, cache):
    for key, commit, files in (('code', 'commit', 'code_files'), ('source_code', 'source_commit', 'source_files'),
            ('source_execution_code', 'source_execution_commit', 'source_execution_files')):
        root = owned(request[key], directory=True)
        if subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip() != request[commit]:
            raise ValueError('Immutable actual source Git HEAD changed: ' + str(root))
        for name, digest in request[files].items():
            path = owned(root / name)
            cache.verify(dict(path=str(path), sha256=digest, stat=identity(path)))


def preserve_predecessors(request, cache):
    """Preserve every old global-claim lineage byte and refuse a live native actor."""
    source = Path(request['source_output']); stem = hashlib.sha256(str(source).encode()).hexdigest()
    registry = source.parent / 'gnn_native_chain_claims_20261010'
    actual = sorted(str(path) for path in registry.glob(stem + '*.json')) if registry.exists() else []
    supplied = request['source_claim_witnesses']
    if sorted(row.get('path', '') for row in supplied) != actual:
        raise ValueError('Every original source claim/derived lease must remain exactly preserved')
    old_roots = set()
    for witness in supplied:
        value = read(cache.verify(witness))
        for key in ('chain_root', 'previous_chain_root'):
            if isinstance(value.get(key), str):
                old_roots.add(value[key])
    predecessors = request['native_predecessors']
    if {row.get('root') for row in predecessors} != old_roots:
        raise ValueError('Every prior source-controller root needs exact closed/no-native proof')
    for row in predecessors:
        root = owned(row['root'], directory=True)
        if set(row) != {'root', 'request', 'status', 'registration'}:
            raise ValueError('Exact original closed-controller three-file witnesses required')
        values = {}
        for key in ('request', 'status', 'registration'):
            path = cache.verify(row[key])
            if path != root / (key + '.json'):
                raise ValueError('Prior controller metadata path changed')
            values[key] = read(path)
        status, registration = values['status'], values['registration']
        if (values['request'].get('source_output') != request['source_output']
                or status.get('status') not in ('FAILED','WAITING_FOR_FULL_GNN','WAITING_FOR_NATIVE_STORAGE')
                or (root / 'native').exists()
                or status.get('full_native_training_completed') is True
                or process_snapshot(registration['pid'], registration['create_time'], registration['command']) is not None):
            raise ValueError('Prior owned native controller/child/output still active; no duplicate native launch')
        if status.get('child_pid') is not None and process_snapshot(status['child_pid'],
                status['child_create_time'],status['command']) is not None:
            raise ValueError('Actual old native/scoring child still active; no duplicate native launch')


def source_complete(request):
    """Actual full invocation only; numerical BEST admission belongs to scoring."""
    source = owned(request['source_output'], directory=True)
    scientific = read(source / 'request.json')
    config = scientific.get('config', {}); runtime = config.get('v24_runtime', {})
    digest = hashlib.sha256(json.dumps({k:v for k,v in scientific.items() if k!='request_sha256'},
        sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
    if (scientific.get('request_sha256') != digest or scientific.get('physical_GPU') != request['GPU_arm']
            or config.get('epochs') != 40 or config.get('seed') != 42
            or runtime.get('debug') is not False or runtime.get('hidden_subset') is not False
            or runtime.get('all_P_from_epoch_one') is not True or runtime.get('other_P_as_negative') is not False
            or any(request['source_files'].get(name) != digest for name, digest in scientific.get('source', {}).items())
            or not scientific.get('source')):
        raise ValueError('Actual original full40 GT-blind scientific source contract changed')
    if request['GPU_arm'] == 4 and runtime.get('curriculum', {}).get('initial_u') != 128:
        raise ValueError('GPU4 must retain all128U from epoch one')
    path = source / 'training/invocations.jsonl'
    actor_registration = source.parent/'direct_training_registration.json'
    if actor_registration.exists():
        registration = read(actor_registration)
        declaration = read(source.parent/'direct_training_request.json')
        if (registration.get('format')!='v24_existing_direct_GPU_exact_continuation_registration_v1'
                or declaration.get('format')!='v24_existing_direct_GPU_exact_continuation_request_v1'
                or declaration.get('output')!=str(source) or declaration.get('source_code')!=request['source_code']
                or registration.get('output')!=str(source) or registration.get('GPU_arm')!=request['GPU_arm']
                or registration.get('request_sha256')!=sha(source.parent/'direct_training_request.json')):
            raise ValueError('Actual own direct GNN registration/request/source binding changed')
        if process_snapshot(registration['pid'],registration['create_time'],registration['command']) is not None:
            return None
    if not path.exists():
        return None
    rows = []
    for line in owned(path).read_bytes().splitlines(keepends=True):
        if not line.endswith(b'\n'):
            break
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError('Actual invocation JSON object required')
            rows.append(value)
    last = rows[-1] if rows else {}
    if last.get('status') in ('FAILED', 'PAUSED'):
        raise RuntimeError('Actual source GNN '+last['status']+'; no automatic native launch')
    if last.get('status') != 'COMPLETE':
        return None
    tensors = 981 if request['GPU_arm'] == 5 else 537
    if (last.get('full_training') is not True or last.get('debug') is not False or last.get('actual_CUDA') is not True
            or last.get('completed_epochs') != 40 or last.get('optimizer_updates') != 680
            or last.get('connected_parameter_tensors') != tensors or last.get('expected_parameter_tensors') != tensors):
        raise ValueError('Actual full40/680/CUDA/connected complete invocation required')
    return dict(path=str(path), sha256=sha(path), completed_epochs=40, optimizer_updates=680,
        CLI_exit_status_observed=False, actual_CUDA=True, connected_parameter_tensors=tensors)


def scoring_complete(request, cache):
    """Admit the existing sealed complete job and pin without rescoring."""
    root = owned(request['scoring_root'], directory=True, missing=True)
    required = [root / name for name in ('request.json', 'registration.json', 'complete.json')]
    if (root / 'failed.json').exists():
        raise RuntimeError('Actual full642 scorer failed: ' + read(root / 'failed.json').get('error', 'see failed.json'))
    if any(not path.exists() for path in required):
        return None
    declared, registration, done = [read(path) for path in required]
    gpu = request['GPU_arm']
    if (declared.get('format') != 'v24_full642_depth1_current_BEST_scoring_request_v1'
            or declared.get('gpu') != gpu or declared.get('source_entries') != 642
            or declared.get('CP_positions') != 82176 or declared.get('debug') is not False
            or declared.get('full642_scoring') is not True or declared.get('physical_patient_batch') != 4
            or declared.get('physical_candidate_chunk') != (32 if gpu == 5 else 64)
            or declared.get('output') != str(root / 'bank')
            or declared.get('worker_sha256') != request['scoring_worker_sha256']
            or declared.get('source_output', request['source_output']) != request['source_output']
            or declared.get('source_code', request['source_code']) != request['source_code']
            or declared.get('inventory_path') != request['inventory']
            or declared.get('input_cache') != request['input_cache']
            or declared.get('source_bank') != request['historical_basic_bank']
            or declared.get('baseline_preprocessed') != request['baseline_preprocessed']
            or declared.get('stunet_checkpoint') != request['stunet_checkpoint']
            or registration.get('format') != 'v24_full642_depth1_current_BEST_scoring_registration_v1'
            or registration.get('root') != str(root) or registration.get('request_sha256') != sha(root / 'request.json')
            or registration.get('sources') != 642 or registration.get('CP_positions') != 82176
            or registration.get('command') != [request['python'], '-B', '-u', request['scoring_worker'], '--request', str(root / 'request.json')]
            or done.get('status') != 'FULL642_SCORING_COMPLETE_NATIVE_STORAGE_PENDING'
            or done.get('completed_sources') != 642 or done.get('completed_CP_positions') != 82176
            or done.get('debug') is not False or done.get('actual_CUDA') is not True
            or done.get('full_native_training') is not False or done.get('full_native_evaluation') is not False
            or done.get('production_optimizer_updates') != 0):
        raise ValueError('Actual sealed own-arm full642/82176 completed scoring request/registration/receipt required')
    if process_snapshot(registration['pid'], registration['create_time'], registration['command']) is not None:
        return None
    overlay = root / 'bank/score_overlay.json'
    if done.get('overlay') != str(overlay) or not checksum(done.get('overlay_sha256')):
        raise ValueError('Actual full642 overlay completion path/SHA required')
    cache.verify(dict(path=str(overlay), sha256=done['overlay_sha256'], stat=identity(owned(overlay))))
    pin_path = owned(declared['pin_path'])
    from .v24_nnunet_cp import validate_current_pin
    pin = validate_current_pin(read(pin_path))
    if (pin['physical_GPU'] != gpu or pin['source_output'] != request['source_output']
            or pin['source_code'] != request['source_code'] or pin['inventory'] != request['inventory']
            or pin['input_cache'] != request['input_cache'] or pin['stunet_checkpoint'] != request['stunet_checkpoint']):
        raise ValueError('External CP must retain this exact completed GNN/own BEST')
    from .v24_matched_basic_cp_pipeline import validate_profiled_overlay
    binding = validate_profiled_overlay(overlay, full_hash=True, expected_depth1=True)
    if binding['overlay']['pin'] != pin or binding['overlay']['pin_sha256'] != sha(pin_path):
        raise ValueError('Complete score overlay and original own BEST pin differ')
    cache.verify(dict(path=request['scoring_worker'], sha256=request['scoring_worker_sha256'],
        stat=identity(owned(request['scoring_worker']))))
    for key, digest in pin['source_artifacts_sha256'].items():
        path = owned(Path(request['source_output']) / key)
        cache.verify(dict(path=str(path), sha256=digest, stat=identity(path)))
    for proof in (pin['BEST_checkpoint_proof'], pin['final_checkpoint_proof']):
        cache.verify(dict(path=proof['path'], sha256=proof['raw_sha256'], stat=proof['source_stat']))
    return dict(format=SCORING_FORMAT, GPU_arm=gpu, source_output=request['source_output'],
        source_code=request['source_code'], scoring_root=str(root),
        request=cache.capture(required[0]), registration=cache.capture(required[1]),
        complete=cache.capture(required[2]), pin=cache.capture(pin_path), overlay=cache.capture(overlay),
        complete_sources=642, complete_positions=82176, context_depth1=True, rescore=False)


def storage_ready(request, cache):
    binding_path = owned(request['storage_binding'], missing=True)
    if not binding_path.exists():
        raise WaitingForStorage('Full three-arm storage admission/binding has not yet been published')
    binding = read(binding_path)
    if (set(binding) != {'format', 'admission', 'arm_roots', 'required_free_bytes',
                        'minimum_runtime_free_bytes', 'model_data_scale_preserved'}
            or binding.get('format') != STORAGE_FORMAT or binding.get('arm_roots') != request['arm_roots']
            or binding.get('model_data_scale_preserved') is not True
            or binding.get('admission', {}).get('path') != request['storage_admission']):
        raise ValueError('Fresh full three-arm storage-ready binding required')
    path = cache.verify(binding['admission'])
    from .v24_matched_basic_cp_pipeline import verify_admission, check_aggregate_disk
    document = verify_admission(path)
    aggregate = document['aggregate_budget']
    if (document['arm_roots'] != request['arm_roots'] or document['inventory']['path'] != request['inventory']
            or document['baseline']['preprocessed'] != request['baseline_preprocessed']
            or document['historical_basic_bank']['path'] != request['historical_basic_bank']
            or aggregate.get('global_peak_checkpoint_slots') != 7
            or aggregate.get('minimum_runtime_free_bytes') != RESERVE
            or binding['required_free_bytes'] != aggregate['required_free_bytes']
            or binding['minimum_runtime_free_bytes'] != RESERVE
            or aggregate['required_free_bytes'] != aggregate['all_three_new_writable_bytes'] + RESERVE):
        raise ValueError('Exact original global7/fullthree writes plus10GiB+512MiB reserve required, once only')
    try:
        measured = check_aggregate_disk(document, request['chain_root'], preparation=True)
    except OSError as error:
        if str(error) == CAPACITY_ERROR:
            raise WaitingForStorage(str(error)) from error
        raise
    return dict(path=str(path), sha256=binding['admission']['sha256'], stat=binding['admission']['stat'],
        binding=cache.capture(binding_path), measured=measured, required_free_bytes=aggregate['required_free_bytes'],
        minimum_runtime_free_bytes=RESERVE, reserve_included_in_required_free_bytes=True)


def _seconds(value):
    fields = value.split(':')
    if len(fields) != 3 or any(not field.isdigit() for field in fields):
        raise ValueError('Exact owned PBS walltime required')
    hours, minutes, seconds = map(int, fields)
    return hours * 3600 + minutes * 60 + seconds


def scheduler_ready(request, action):
    expected = request['scheduler']
    try:
        text = subprocess.check_output(['/opt/pbs/bin/qstat', '-f', expected['job_id']], text=True, timeout=10)
    except subprocess.CalledProcessError as error:
        raise WaitingForAllocation('Exact owned GPU6 PBS reservation is unavailable') from error
    fields = {}
    for line in text.splitlines():
        if ' = ' in line:
            key, value = line.strip().split(' = ', 1); fields[key] = value
    started = time.mktime(time.strptime(fields.get('stime', ''), '%a %b %d %H:%M:%S %Y'))
    expires = started + _seconds(fields.get('Resource_List.walltime', ''))
    remaining = min(expires - time.time(), _seconds(fields.get('Resource_List.walltime', '')) - _seconds(fields.get('resources_used.walltime', '')))
    floor = 25200 if action == 'train' else 28800
    if (abs(expires - expected['expires_at_unix']) > 1 or fields.get('job_state') != 'R'
            or not fields.get('Job_Owner', '').startswith('aicompetition06@')
            or expected['expected_node'] not in fields.get('exec_host', '')
            or fields.get('Resource_List.ncpus') != '6' or fields.get('Resource_List.ngpus') != '1'
            or fields.get('Resource_List.mem', '').lower() != '128gb' or remaining < floor):
        raise WaitingForAllocation('Exact owned GPU6 CPU/RAM/GPU/lease margin is insufficient')
    return dict(PBS_owned_reservation=True, job_id=expected['job_id'], expires_at_unix=expires,
        remaining_seconds=remaining, required_remaining_seconds=floor)


def resources_ready(request, action):
    import psutil
    if psutil.Process().cpu_affinity() != request['CPU_affinity']:
        raise ValueError('Exactly original four assigned logical CPUs required')
    from . import v24_direct_gpu_access as direct
    row = direct._gpu(request['GPU_arm'])
    if (row['UUID'] != request['GPU_UUID'] or row['total_MiB'] < 40 * 1024
            or row['free_MiB'] < 40 * 1024 or direct._apps(request['GPU_UUID'])):
        raise WaitingForGPU('Exact assigned full GPU is still occupied or lacks original40GiB free')
    if request['GPU_arm'] == 6:
        allocation = scheduler_ready(request, action)
    else:
        admitted = direct.validate_controller_request(request)
        # The CPU controller validates lifetime/identity but never initializes CUDA.
        proof = admitted['proof']; lifetime = proof['host_container_lifetime']
        try:
            actor = direct._exact(lifetime['allocation_process'])
        except direct.DirectGPUAccessUnavailable as error:
            raise WaitingForAllocation(str(error)) from error
        minimum = proof['native_minimum_initial_remaining_seconds'] if action == 'train' else proof['minimum_initial_remaining_seconds']
        if actor.environ().get('PBS_JOBID') != lifetime['PBS_job_id'] or lifetime['expires_at_unix'] - time.time() < minimum:
            raise WaitingForAllocation('Independent host lifetime is insufficient for unchanged native stage')
        allocation = dict(PBS_owned_reservation=False, remaining_seconds=lifetime['expires_at_unix'] - time.time())
    return dict(GPU_UUID=row['UUID'], free_VRAM_MiB=row['free_MiB'], CPU_affinity=request['CPU_affinity'], allocation=allocation)


_STAGE_GUARDS = {}


class OwnedGPU6Guard:
    """Exact owned PBS6/own process/foreign CUDA checks, every30seconds."""
    def __init__(self, request, action, request_path, digest):
        from . import v24_direct_gpu_access as direct
        import psutil
        self.request, self.action, self.path, self.digest = request, action, request_path, digest
        self.actor = direct._identity(psutil.Process()); self.observed = {self.actor['pid']: self.actor}
        self.last = float('-inf'); self.expiry = request['scheduler']['expires_at_unix']
        self.monotonic_expiry = time.monotonic() + self.expiry - time.time()
        self.selected = False
        self(initial=True)

    def __call__(self, *, initial=False):
        from . import v24_direct_gpu_access as direct
        actor = direct._exact(self.actor)
        visibility = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        if (sha(self.path) != self.digest or actor.cpu_affinity() != self.request['CPU_affinity']
                or visibility not in ('', '6', self.request['GPU_UUID']) or self.selected and not visibility):
            raise RuntimeError('Sealed owned GPU6 stage request/CPU/CVD/process identity changed')
        self.selected |= bool(visibility)
        remaining = min(self.expiry - time.time(), self.monotonic_expiry - time.monotonic())
        if remaining < (25200 if initial and self.action == 'train' else 28800 if initial else 180):
            raise RuntimeError('Owned GPU6 lease is insufficient; no new CUDA work')
        if initial or time.monotonic() - self.last >= 30:
            scheduler_ready(self.request, self.action) if initial else scheduler_ready_runtime(self.request)
            row = direct._gpu(6)
            if row['UUID'] != self.request['GPU_UUID'] or row['total_MiB'] < 40960:
                raise RuntimeError('Owned full GPU6 UUID/VRAM changed')
            current = {actor.pid: actor}
            for child in actor.children(recursive=True):
                try:
                    child_identity = direct._identity(child)
                except direct.psutil.NoSuchProcess:
                    continue
                if child_identity['UID'] != 41833 or self.observed.get(child.pid, child_identity) != child_identity:
                    raise RuntimeError('Foreign/reused descendant cannot own GPU6 CUDA work')
                self.observed[child.pid] = child_identity; current[child.pid] = child
            applications = direct._apps(self.request['GPU_UUID'])
            if initial and (applications or row['free_MiB'] < 40960):
                raise WaitingForGPU('Owned GPU6 must initially be idle/full40GiB for this stage')
            for pid in applications:
                if pid not in current:
                    raise RuntimeError('Foreign/unmanaged CUDA process on owned GPU6: ' + str(pid))
                direct._exact(self.observed[pid])
            self.last = time.monotonic()
        return dict(format='v24_owned_GPU6_native_stage_guard_v1', PBS_owned_reservation=True,
            GPU_arm=6, GPU_UUID=self.request['GPU_UUID'], own_process_identity=self.actor,
            remaining_seconds=remaining, foreign_CUDA_poll_seconds=30, signals_sent=False)


def scheduler_ready_runtime(request):
    declared = copy.deepcopy(request)
    # Runtime admission keeps the same actual lease/resource identity but uses
    # the original180-second cooperative reserve rather than a fresh-run floor.
    expected = declared['scheduler']
    text = subprocess.check_output(['/opt/pbs/bin/qstat', '-f', expected['job_id']], text=True, timeout=10)
    fields = dict(line.strip().split(' = ', 1) for line in text.splitlines() if ' = ' in line)
    started = time.mktime(time.strptime(fields.get('stime', ''), '%a %b %d %H:%M:%S %Y'))
    expires = started + _seconds(fields.get('Resource_List.walltime', ''))
    if (abs(expires - expected['expires_at_unix']) > 1 or fields.get('job_state') != 'R'
            or not fields.get('Job_Owner', '').startswith('aicompetition06@')
            or expected['expected_node'] not in fields.get('exec_host', '')
            or fields.get('Resource_List.ncpus') != '6' or fields.get('Resource_List.ngpus') != '1'
            or fields.get('Resource_List.mem', '').lower() != '128gb'
            or min(expires - time.time(), _seconds(fields['Resource_List.walltime']) - _seconds(fields.get('resources_used.walltime', ''))) < 180):
        raise RuntimeError('Exact owned GPU6 lease/resource identity or180-second reserve changed')
    return dict(job_id=expected['job_id'], expires_at_unix=expires)


def early_stage_guard(request, path, digest, action):
    validate_declaration(request)
    if action not in ACTIONS:
        raise ValueError('Native-only controller cannot execute a pin/scoring stage')
    key = (str(path), digest, action)
    previous = _STAGE_GUARDS.get(key)
    if previous is not None:
        previous(); return previous
    if request['GPU_arm'] == 6:
        access = OwnedGPU6Guard(request, action, path, digest)
    else:
        from . import v24_direct_gpu_access as direct
        admitted = direct.validate_controller_request(request)
        access = direct.make_guard(admitted['proof_path'], request['GPU_arm'], request['CPU_affinity'], stage_action=action)
    os.environ['CUDA_VISIBLE_DEVICES'] = request['GPU_UUID']
    binding = read(request['storage_binding']); proof = binding['admission']
    if proof['path'] != request['storage_admission'] or sha(proof['path']) != proof['sha256'] or identity(proof['path']) != proof['stat']:
        raise ValueError('Native stage storage admission changed')
    root = Path(request['chain_root']); binding_stat = identity(request['storage_binding']); admission_stat = proof['stat']
    def guard():
        if (sha(path) != digest or identity(request['storage_binding']) != binding_stat
                or identity(request['storage_admission']) != admission_stat):
            raise ValueError('Sealed native stage request/storage proof changed')
        if shutil.disk_usage(root).free < RESERVE:
            raise WaitingForStorage('Original10GiB+512MiB runtime floor unavailable; no new native CUDA work')
        from .v24_nnunet_cp import require_project_budget
        require_project_budget()
        return access()
    guard(); _STAGE_GUARDS[key] = guard
    return guard


def external_overlay_for_stage(request):
    validate_declaration(request)
    cache = FileProofCache()
    admitted = read(Path(request['chain_root']) / 'admitted_scoring.json')
    if (admitted.get('format') != SCORING_FORMAT or admitted.get('GPU_arm') != request['GPU_arm']
            or admitted.get('source_output') != request['source_output'] or admitted.get('source_code') != request['source_code']
            or admitted.get('scoring_root') != request['scoring_root'] or admitted.get('complete_sources') != 642
            or admitted.get('complete_positions') != 82176 or admitted.get('rescore') is not False):
        raise ValueError('This controller requires its admitted complete external own-score proof')
    for key in ('request', 'registration', 'complete', 'pin', 'overlay'):
        cache.verify(admitted[key])
    path = Path(request['scoring_root']) / 'bank/score_overlay.json'
    if admitted['overlay']['path'] != str(path):
        raise ValueError('External score-overlay path changed')
    return path


def native_stage_proof(request, action):
    """Same original native proof, using the existing external pin/score paths."""
    from . import v24_matched_basic_cp_pipeline as pipeline
    from . import v24_nnunet_cp as original
    root = Path(request['chain_root']); gpu = request['GPU_arm']
    stage = root / ('matched_storage_' + action.replace('-', '_') + '.json')
    value = read(stage); storage = read(request['storage_binding'])['admission']
    if (action not in ACTIONS or value.get('format') != pipeline.STAGE_FORMAT
            or value.get('status') != 'COMPLETE' or value.get('action') != action
            or value.get('storage_profile') != PROFILE or value.get('storage_admission_sha256') != storage['sha256']
            or value.get('cached_inputs_written') is not False or value.get('model_data_scale_preserved') is not True
            or value.get('runtime_sources_sha256') != {name: pipeline.sha(pipeline.ROOT / name) for name in pipeline.FILES}):
        raise ValueError('Actual external-score native child completion receipt required')
    overlay_path = external_overlay_for_stage(request)
    admitted = read(root / 'admitted_scoring.json')
    pin = original.validate_current_pin(read(admitted['pin']['path']))
    overlay = pipeline.validate_profiled_overlay(overlay_path, expected_depth1=True)
    if (pin['physical_GPU'] != gpu or pin['source_output'] != request['source_output']
            or pin['source_code'] != request['source_code'] or overlay['overlay']['pin'] != pin):
        raise ValueError('Native must use its original completed own-arm BEST and unchanged full642 scores')
    path = root / 'native/native.json'
    native, _, _ = pipeline._guard_native(path)
    if (native['overlay'] != str(overlay_path) or native['physical_GPU'] != gpu
            or native.get('epochs') != 250 or native.get('physical_batch') != 2
            or native.get('gradient_accumulation') != 1 or native.get('effective_batch') != 2
            or native.get('full_preprocessed_cases') != 131 or native.get('train_cases') != 105
            or native.get('validation_cases') != 26 or native.get('patch_size') != [128]*3
            or native.get('cp_probability') != .5 or native.get('source_entries') != 642
            or native.get('candidates_per_source') != 128 or native.get('debug') is not False
            or native.get('evaluation_checkpoint') != 'checkpoint_best.pth'):
        raise ValueError('Native output changed its external CP source or physical GPU')
    if action in ('calibrate-native', 'train'):
        pipeline.validate_calibration(path)
    if action == 'train':
        from . import v24_native_best_eval_runtime as best
        fold = Path(native['checkpoint_coordination']['folds']['gpu' + str(gpu)])
        receipt = fold / best.RECEIPT
        verified = best.validate_best_validation_receipt(receipt, fold=fold,
            expected_cases=native['split']['outer_val'], bank_sha256=native['bank_sha256'])
        complete = list((root / 'native').glob('training_complete_*.json')); final = fold / 'checkpoint_final.pth'
        if len(complete) != 1:
            raise ValueError('Exactly one actual full250/BEST26 completion proof required')
        expected = dict(format=pipeline.NATIVE_FORMAT, epochs=250, GPU=gpu, checkpoint=str(final),
            checkpoint_sha256=sha(final), checkpoint_role='full250_training_completion_witness',
            evaluation_checkpoint_name=best.BEST, evaluation_checkpoint_path=verified['checkpoint_path'],
            evaluation_checkpoint_sha256=verified['checkpoint_sha256'], best_validation_receipt=str(receipt),
            best_validation_receipt_sha256=sha(receipt), bank_sha256=native['bank_sha256'])
        if read(complete[0]) != expected:
            raise ValueError('Actual full250 final witness and own BEST/full26 evaluation proof differ')
        path = complete[0]
    return dict(action=action, path=str(path), sha256=sha(path), matched_stage_receipt=str(stage),
        matched_stage_receipt_sha256=sha(stage), physical_GPU=gpu)


class Controller:
    def __init__(self, request_path):
        import psutil
        self.path = owned(request_path); self.root = owned(self.path.parent, directory=True)
        self.digest = sha(self.path); self.request = validate_declaration(read(self.path))
        if self.request['chain_root'] != str(self.root):
            raise ValueError('Actual controller request must live in its sealed native root')
        self.cache = FileProofCache(); self.cache.capture(self.path)
        verify_code(self.request, self.cache); preserve_predecessors(self.request, self.cache)
        for key in ('source_output','input_cache','baseline_preprocessed','historical_basic_bank'):
            owned(self.request[key],directory=True)
        owned(self.request['inventory']); owned(self.request['python']); owned(self.request['scoring_worker'])
        if self.request['GPU_arm'] in (4,5):
            from .v24_direct_gpu_access import validate_controller_request
            validated = validate_controller_request(self.request)
            self.cache.capture(validated['proof_path']); self.cache.capture(validated['stage_request_path'])
        if self.request['stunet_checkpoint'] is not None:
            self.cache.verify(dict(path=self.request['stunet_checkpoint'],sha256=self.request['stunet_checkpoint_sha256'],
                stat=identity(owned(self.request['stunet_checkpoint']))))
        actor = psutil.Process()
        if actor.cpu_affinity() != self.request['CPU_affinity']:
            raise ValueError('Actual controller must inherit exactly the assigned four CPUs')
        self.state = dict(format=STATUS_FORMAT, request_sha256=self.digest, GPU_arm=self.request['GPU_arm'],
            source_output=self.request['source_output'], source_code=self.request['source_code'],
            source_commit=self.request['source_commit'], code=self.request['code'], commit=self.request['commit'],
            supervisor_pid=actor.pid, supervisor_create_time=actor.create_time(), UID=41833,
            actual_CPU_affinity=actor.cpu_affinity(), status='WAITING_FOR_FULL_GNN', stage=None,
            child_pid=None, child_create_time=None, stages_completed=[], stage_proofs=[],
            full_native_training_completed=False, full_native_evaluation_completed=False,
            rescore=False, signals_sent=False, started=time.time())
        self.scoring = None; self.storage = None; self.pending = None
        claim = claim_path(self.request)
        owned(claim.parent, directory=True)
        publish(claim, dict(format=CLAIM_FORMAT, GPU_arm=self.request['GPU_arm'], chain_root=str(self.root),
            source_output=self.request['source_output'], scoring_root=self.request['scoring_root'], request_sha256=self.digest))
        publish(self.root / 'lease.json', dict(supervisor_pid=actor.pid, supervisor_create_time=actor.create_time(),
            command=actor.cmdline(), request_sha256=self.digest, claim=str(claim), claim_sha256=sha(claim)))
        self.cache.capture(claim)
        self.cache.capture(self.root/'lease.json')

    def update(self, **values):
        self.state.update(values, checked_at=time.time())
        publish(self.root / 'status.json', self.state, replace=True)

    def ready(self, action):
        self.cache.check(); preserve_predecessors(self.request, self.cache)
        if self.scoring is None:
            completion = source_complete(self.request)
            if completion is None:
                self.update(status='WAITING_FOR_FULL_GNN', stage=None, child_pid=None, waiting_reason='Actual own full40/680 GNN completion is pending')
                return False
            self.state['gnn_completion'] = completion
            self.scoring = scoring_complete(self.request, self.cache)
            if self.scoring is None:
                self.update(status='WAITING_FOR_FULL642_SCORING', stage=None, child_pid=None, waiting_reason='Actual own full642/82176 scoring completion/process release is pending')
                return False
            publish(self.root / 'admitted_scoring.json', self.scoring)
            self.cache.capture(self.root/'admitted_scoring.json')
        self.storage = storage_ready(self.request, self.cache)
        resource = resources_ready(self.request, action)
        self.update(status='READY_FOR_NATIVE_STAGE', stage=None, child_pid=None, waiting_reason=None, storage=self.storage, resources=resource)
        return True

    def stage(self, stage):
        import psutil
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES='', CUDA_DEVICE_ORDER='PCI_BUS_ID',
            PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
            V24_ARM_RSS_GIB='48', V24_CONTEXT_DEPTH1_CHAIN_REQUEST=str(self.path),
            V24_CONTEXT_DEPTH1_CHAIN_REQUEST_SHA256=self.digest,
            V24_READONLY_STORAGE_ADMISSION_SHA256=self.storage['sha256'])
        log = self.root / (stage['name'] + '.log')
        self.update(status='RUNNING', stage=stage['name'], command=stage['command'], stage_log=str(log), stage_started=time.time())
        violation = None
        with log.open('xb', buffering=0) as stream:
            child = subprocess.Popen(stage['command'], cwd=self.request['code'], env=env,
                stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
            actor = psutil.Process(child.pid)
            process_snapshot(child.pid,actor.create_time(),stage['command'])
            self.update(child_pid=child.pid, child_create_time=actor.create_time())
            while child.poll() is None:
                try:
                    self.cache.check(); preserve_predecessors(self.request, self.cache)
                except (ValueError, OSError, RuntimeError) as error:
                    violation = type(error).__name__ + ': ' + str(error)
                try:
                    rss = actor.memory_info().rss; descendants = actor.children(recursive=True)
                except psutil.NoSuchProcess:
                    if child.poll() is None:
                        raise RuntimeError('Exact own native child disappeared without recorded exit')
                    break
                for descendant in descendants:
                    try:
                        rss += descendant.memory_info().rss
                    except psutil.NoSuchProcess:
                        continue
                if rss > 48 * 2**30:
                    violation = 'Actual own native child tree exceeded original48GiB; no next stage'
                self.update(child_process_tree_RSS_bytes=rss, RAM_budget_bytes=48 * 2**30, resource_violation=violation)
                time.sleep(5)
            result = child.wait()
        if result or violation:
            raise RuntimeError('Native stage ' + stage['name'] + ' failed (' + str(result) + '): ' + str(violation)
                + '; original logs preserved; no process signaled')
        proof = native_stage_proof(self.request, stage['action'])
        self.state['stages_completed'].append(stage['name']); self.state['stage_proofs'].append(proof)
        self.update(child_pid=None, child_create_time=None)
        return proof

    def run(self):
        self.update()
        try:
            for stage in self.request['stages']:
                while True:
                    try:
                        if self.ready(stage['action']):
                            self.pending = None; break
                    except PendingMetadata as error:
                        signature = error.signature; now = time.monotonic()
                        if self.pending is None or self.pending[0] != signature:
                            self.pending = (signature, now)
                        if now - self.pending[1] >= 10:
                            raise ValueError('Stable corrupt metadata remained after visible publication retries: ' + str(error)) from error
                        self.update(status='WAITING_METADATA_PUBLICATION', stage=None, child_pid=None, waiting_reason=str(error))
                    except (WaitingForStorage, WaitingForGPU, WaitingForAllocation) as error:
                        label = ('WAITING_FOR_NATIVE_STORAGE' if isinstance(error, WaitingForStorage) else
                            'WAITING_FOR_ASSIGNED_GPU' if isinstance(error, WaitingForGPU) else 'WAITING_ALLOCATION')
                        self.pending = None
                        self.update(status=label, stage=None, child_pid=None, waiting_reason=str(error),
                            actual_free_disk_bytes=shutil.disk_usage(self.root).free)
                    time.sleep(5)
                proof = self.stage(stage)
            self.update(status='COMPLETE', stage=None, child_pid=None, child_create_time=None, completed=time.time(),
                full_native_training_completed=True, full_native_evaluation_completed=True, completion_proof=proof,
                evaluation_checkpoint='checkpoint_best.pth')
        except Exception as error:
            self.update(status='FAILED', error=type(error).__name__ + ': ' + str(error), failed=time.time())
            raise


def run(request_path):
    sys.dont_write_bytecode = True
    return Controller(request_path).run()
