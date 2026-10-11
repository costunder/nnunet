"""Register only the new owned GPU4/5/6 automatic native continuation.

Standalone target-host helper. Supply PAYLOAD when composing the script, or
--payload JSON --payload-sha RAW_SHA. Required uppercase payload keys are
CODE_ROOT, COMMIT, FILES, INSPECTION, GUARD_BASE64 and IDENTIFIER. Existing
GPU4/6 scorers, old claims, models and experiment results remain unchanged.
Only fresh registration namespaces are written. No process is signalled.
--check performs read-only preflight; actual registration launches CPU waiters
whose existing sealed stage guards admit subsequent native work independently.
"""
import argparse
import base64
import copy
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import psutil


BASE = Path('/var/tmp/aicompetition06_v24_20261010')
GROUP = BASE / 'GPU456_AUTO_NATIVE_20261011_1791730500000000000'
SCORES = {4: BASE / 'GPU4_FULL642_depth1_scoring_20261011_1791730200000000000',
    5: BASE / 'GPU5_FULL642_depth1_scoring_20261012_1791731200000000000',
    6: BASE / 'GPU6_FULL642_depth1_scoring_20261011_1791704180612525600'}
GUARD_SHA = '7d8c7b4d7da52ddf04befe48fb4d539bddbbfabc83f0633f631fc1536db24ffe'


def sha_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def json_value(raw, filename):
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError(str(filename) + ': invalid JSON: ' + str(error)) from error
    if not isinstance(value, dict):
        raise ValueError(str(filename) + ': JSON object required')
    return value


def exact_actor(registration, affinity, *, allow_gone=False):
    """Never infer a live actor from PID alone or replay an uncertain launch."""
    try:
        actor = psutil.Process(registration['pid'])
        if (actor.create_time() != registration['create_time'] or actor.cmdline() != registration['command']
                or actor.uids().real != 41833 or actor.cpu_affinity() != affinity
                or actor.environ().get('CUDA_VISIBLE_DEVICES') != ''):
            raise ValueError('Registered process PID/birth/full argv/UID/CPU/CUDA visibility binding differs')
        if actor.status() == psutil.STATUS_ZOMBIE:
            if allow_gone:
                return None
            raise RuntimeError('Registered actor is a zombie; explicit recovery required')
        return actor
    except psutil.NoSuchProcess:
        if allow_gone:
            return None
        raise RuntimeError('Registered actor is gone; preserve records, explicit recovery required')


def lifetime_from_inspection(inspection, chain):
    allocation = inspection['allocation']
    gpu6 = next(row for row in inspection['arms'] if row['gpu'] == 6)
    expiry = gpu6['scoring_request']['allocation_expiry_unix']
    identity = dict(pid=allocation['pid'], create_time=allocation['birth'],
        command=allocation['command'], UID=41833)
    actor = psutil.Process(identity['pid'])
    if (actor.create_time() != identity['create_time'] or actor.cmdline() != identity['command']
            or actor.uids().real != 41833 or actor.status() == psutil.STATUS_ZOMBIE
            or actor.environ().get('PBS_JOBID') != allocation['PBS_job'] or expiry - time.time() < 28800):
        raise ValueError('Actual independent host process/PBS/lifetime is no longer sufficient')
    declared = dict(scheduler=dict(job_id=allocation['PBS_job'], expires_at_unix=expiry,
        expected_node='ece-a6gpu6', expected_owner='aicompetition06', state='R',
        minimum_remaining_seconds=28800, native_minimum_remaining_seconds=25200))
    chain.scheduler_ready(declared, 'prepare-native')
    return dict(role='host_container_lifetime_only_not_GPU45_PBS_allocation',
        expires_at_unix=expiry, PBS_job_id=allocation['PBS_job'], allocation_process=identity)


def preflight(payload):
    if os.getuid() != 41833 or socket.gethostname().split('.')[0] != 'ece-a6gpu6':
        raise ValueError('Exact own UID41833/ece-a6gpu6 target host required')
    code = Path(payload['CODE_ROOT'])
    if not code.is_absolute() or '..' in code.parts or not isinstance(payload.get('FILES'), dict):
        raise ValueError('Explicit new code root/full immutable manifest required')
    for part in (code, *code.parents):
        if part.is_symlink():
            raise ValueError('Code root alias refused: ' + str(part))
    if (code.stat().st_uid != 41833 or not code.is_dir()
            or subprocess.check_output(['git', '-C', str(code), 'rev-parse', 'HEAD'], text=True).strip() != payload['COMMIT']):
        raise ValueError('Actual fresh code root/commit differs')
    for name, expected in payload['FILES'].items():
        if not isinstance(name, str) or Path(name).is_absolute() or '..' in Path(name).parts:
            raise ValueError('Exact relative immutable code manifest required')
        filename = code / name
        if (any(part.is_symlink() for part in (filename, *filename.parents))
                or filename.stat().st_uid != 41833 or not filename.is_file()
                or sha_bytes(filename.read_bytes()) != expected):
            raise ValueError('Actual immutable code bytes differ: ' + str(filename))
        committed = subprocess.check_output(['git', '-C', str(code), 'show', payload['COMMIT'] + ':' + name])
        if sha_bytes(committed) != expected:
            raise ValueError('Declared code bytes differ from exact committed blob: ' + name)
    guard = base64.b64decode(payload['GUARD_BASE64'], validate=True)
    if sha_bytes(guard) != GUARD_SHA:
        raise ValueError('Exact original v1 scoring guard bytes required')
    if (payload.get('GROUP', str(GROUP)) != str(GROUP)
            or payload.get('SCORING_ROOTS', {str(key): str(value) for key, value in SCORES.items()})
                != {str(key): str(value) for key, value in SCORES.items()}
            or not str(payload['IDENTIFIER']).isdigit()):
        raise ValueError('This task owns only its exact new native/scoring namespaces')
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(code))
    from hiercp_v1x import v24_scored_native_chain as chain
    from tools import watch_v24_gnn_to_matched_scoring as waiter
    required = set(chain.FILES) | {'tools/run_v24_completed_matched_scoring.py',
        'tools/watch_v24_gnn_to_matched_scoring.py'}
    if not required <= set(payload['FILES']):
        raise ValueError('New native controller and actual scorer/waiter closures must be pinned')
    group = chain.owned(GROUP, directory=True)
    binding = chain.read(group / 'storage_binding.json')
    if binding.get('format') != chain.STORAGE_FORMAT:
        raise ValueError('Actual full three-arm storage-ready binding required')
    chain.FileProofCache().verify(binding['admission'])
    requests = chain.requests_from_inspection(payload['INSPECTION'], code=code,
        commit=payload['COMMIT'], code_files=payload['FILES'], group=group,
        scoring_roots={gpu: str(root) for gpu, root in SCORES.items()})
    lifetime = lifetime_from_inspection(payload['INSPECTION'], chain)
    for number, request in requests.items():
        root = chain.owned(request['chain_root'], directory=True)
        if binding.get('arm_roots') != request['arm_roots']:
            raise ValueError('Actual storage group differs from native three-arm declaration')
        chain.verify_code(request, chain.FileProofCache())
        chain.preserve_predecessors(request, chain.FileProofCache())
        chain.storage_ready(request, chain.FileProofCache())
        if request['scoring_root'] != str(SCORES[number]):
            raise ValueError('Actual current source/scorer binding differs')
        claim = Path(request['native_claim_path'])
        if claim.exists():
            previous = chain.read(claim)
            if previous.get('chain_root') != str(root) or previous.get('source_output') != request['source_output']:
                raise FileExistsError('Existing native source claim prevents a second launch: ' + str(claim))
    return chain, waiter, requests, lifetime, guard


def existing_request(chain, expected, lifetime):
    filename = Path(expected['chain_root']) / 'request.json'
    if filename.exists():
        actual = chain.validate_declaration(chain.read(filename))
        plain = copy.deepcopy(actual); plain['direct_GPU_access'] = None
        if plain != expected:
            raise ValueError('Preserved sealed native request differs; no overwrite: ' + str(filename))
        if actual['GPU_arm'] in (4, 5):
            from hiercp_v1x.v24_direct_gpu_access import validate_controller_request
            proof = validate_controller_request(actual)['proof']
            if proof['host_container_lifetime'] != lifetime:
                raise ValueError('Existing direct proof lifetime differs; no overwrite')
        return actual
    return chain.seal_request(expected, host_lifetime=lifetime if expected['GPU_arm'] in (4, 5) else None)


def recover_actor(chain, root, intent, affinity, actor_path, controller_lease):
    """Only exact current actor/own lease recovery; never spawn after intent."""
    if actor_path.exists():
        registration = chain.read(actor_path)
    elif controller_lease is not None and controller_lease.exists():
        lease = chain.read(controller_lease)
        registration = dict(pid=lease['supervisor_pid'], create_time=lease['supervisor_create_time'],
            command=lease['command'], request_sha256=lease['request_sha256'])
    else:
        matches = []
        for actor in psutil.process_iter(['pid', 'create_time', 'cmdline', 'uids']):
            try:
                info = actor.info
                if (info['uids'].real == 41833 and info['cmdline'] == intent['command']
                        and info['create_time'] >= intent['created_at'] - 1):
                    matches.append(dict(pid=actor.pid, create_time=info['create_time'], command=info['cmdline'],
                        request_sha256=intent['request_sha256']))
            except psutil.NoSuchProcess:
                continue
        if len(matches) != 1:
            raise RuntimeError('Uncertain preserved launch outcome; no second spawn. Inspect ' + str(root)
                + ' and exact lease/argv; matched actors=' + str(len(matches)))
        registration = matches[0]
    if registration.get('command') != intent['command'] or registration.get('request_sha256') != intent['request_sha256']:
        raise ValueError('Preserved launch actor/lease differs from sealed launch intent')
    exact_actor(registration, affinity)
    if not actor_path.exists():
        chain.publish(actor_path, registration)
    return registration


def launch(chain, root, request_path, command, affinity, *, native, gpu, code):
    registration_path = root / ('registration.json' if native else 'scoring_wait_registration.json')
    intent_path = root / ('controller_launch_intent.json' if native else 'scoring_wait_launch_intent.json')
    actor_path = root / ('controller_launch_actor.json' if native else 'scoring_wait_launch_actor.json')
    request_sha = chain.sha(request_path)
    if intent_path.exists():
        intent = chain.read(intent_path)
        if intent.get('command') != command or intent.get('request_sha256') != request_sha:
            raise ValueError('Existing launch intent differs; no overwrite or second launch')
        actor = recover_actor(chain, root, intent, affinity, actor_path, root / 'lease.json' if native else None)
    else:
        if registration_path.exists() or actor_path.exists() or native and (root / 'lease.json').exists():
            raise RuntimeError('Unknown existing actor/registration without task launch intent; explicit recovery required')
        intent = dict(command=command, request_sha256=request_sha, created_at=time.time(), GPU_arm=gpu)
        chain.publish(intent_path, intent)
        log = root / ('controller_console.log' if native else 'scoring_wait_console.log')
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', CUDA_DEVICE_ORDER='PCI_BUS_ID',
            PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        def assign_cpu():
            os.sched_setaffinity(0, set(affinity))
        with log.open('xb', buffering=0) as stream:
            child = subprocess.Popen(command, cwd=str(code), env=environment, stdin=subprocess.DEVNULL,
                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True, preexec_fn=assign_cpu)
        process = psutil.Process(child.pid)
        actor = dict(pid=child.pid, create_time=process.create_time(), command=command, request_sha256=request_sha)
        exact_actor(actor, affinity)
        chain.publish(actor_path, actor)
    registration = dict(format=chain.REGISTRATION_FORMAT if native else 'v24_GNN_to_full642_scoring_wait_registration_v1',
        root=str(root), GPU_arm=gpu, pid=actor['pid'], create_time=actor['create_time'], command=command,
        request_sha256=request_sha, CPU_affinity=affinity, CUDA_visible_devices_while_waiting='',
        old_results_written=False, signals_sent=False, registered_at=intent['created_at'])
    if registration_path.exists():
        previous = chain.read(registration_path)
        if previous != registration:
            raise ValueError('Existing exact registration differs; preserve it for explicit recovery')
    else:
        chain.publish(registration_path, registration)
    status_path = root / ('status.json' if native else 'scoring_wait_status.json')
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        exact_actor(actor, affinity)
        if status_path.exists():
            try:
                status = chain.read(status_path)
            except chain.PendingMetadata:
                time.sleep(.25); continue
            if (status.get('request_sha256') != request_sha or status.get('GPU_arm') != gpu
                    or native and (status.get('format') != chain.STATUS_FORMAT
                        or status.get('supervisor_pid') != actor['pid']
                        or status.get('supervisor_create_time') != actor['create_time'])):
                raise ValueError('Actual new controller handshake differs: ' + str(status_path))
            if status.get('status') == 'FAILED':
                raise RuntimeError(str(status_path) + ': ' + str(status.get('error')) + '; logs preserved')
            if not isinstance(status.get('status'), str):
                raise ValueError('Actual new controller status label required: ' + str(status_path))
            return dict(GPU_arm=gpu, root=str(root), pid=actor['pid'], create_time=actor['create_time'],
                command=command, status=status['status'], request_sha256=request_sha,
                status_path=str(status_path), native_controller=native, CUDA_initial_visibility='', signals_sent=False)
        time.sleep(.25)
    raise RuntimeError('New CPU actor has not published status within20s; no replay/signals. Inspect ' + str(root))


def register(payload, *, check=False):
    chain, waiter, requests, lifetime, guard = preflight(payload)
    if check:
        return dict(status='READONLY_PREFLIGHT_PASS', namespaces={key: value['chain_root'] for key, value in requests.items()},
            GPU5_scoring_root=str(SCORES[5]), files_written=False, processes_started=False)
    scoring5 = chain.owned(SCORES[5], directory=True, missing=True)
    if not scoring5.exists():
        scoring5.mkdir(mode=0o700)
    guard_path = scoring5 / 'direct_GPU45_access_guard.py'
    if guard_path.exists():
        if chain.owned(guard_path).read_bytes() != guard:
            raise ValueError('Preserved GPU5 v1 guard differs; no overwrite')
    else:
        with guard_path.open('xb') as stream:
            stream.write(guard); stream.flush(); os.fsync(stream.fileno())
    sealed = {}
    for gpu, request in requests.items():
        registry = Path(request['native_claim_path']).parent
        if not registry.exists():
            registry.mkdir(mode=0o700)
        chain.owned(registry, directory=True)
        sealed[gpu] = existing_request(chain, request, lifetime)
    wait_path = scoring5 / 'scoring_wait_request.json'
    wait_request = waiter.build_request(Path(sealed[5]['chain_root']) / 'request.json',
        host_container_lifetime=lifetime, GPU_UUID=sealed[5]['GPU_UUID'])
    if wait_path.exists():
        if chain.read(wait_path) != wait_request:
            raise ValueError('Preserved scoring wait declaration differs; no overwrite')
    else:
        chain.publish(wait_path, wait_request)
    actors = []
    for gpu, request in sealed.items():
        root = Path(request['chain_root'])
        command = [request['python'], '-B', '-u', str(Path(request['code']) / 'tools/watch_v24_scored_bank_to_nnunet.py'),
            '--request', str(root / 'request.json')]
        actors.append(launch(chain, root, root / 'request.json', command, request['CPU_affinity'],
            native=True, gpu=gpu, code=request['code']))
    request = sealed[5]
    command = [request['python'], '-B', '-u', str(Path(request['code']) / 'tools/watch_v24_gnn_to_matched_scoring.py'),
        '--request', str(wait_path)]
    actors.append(launch(chain, scoring5, wait_path, command, request['CPU_affinity'],
        native=False, gpu=5, code=request['code']))
    return dict(status='ACTUAL_NEW_AUTO_NATIVE_CONTROLLERS_REGISTERED', actors=actors,
        existing_GPU4_6_scorers_modified=False, old_claims_modified=False, model_or_dataset_scale_changed=False,
        signals_sent=False, sessions_modified=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--payload')
    parser.add_argument('--payload-sha')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args(argv)
    if args.payload:
        filename = Path(args.payload)
        if not filename.is_absolute() or '..' in filename.parts or any(part.is_symlink() for part in (filename, *filename.parents)):
            raise ValueError('Absolute non-symlink payload path required')
        if filename.stat().st_uid != 41833 or not filename.is_file():
            raise ValueError('Own regular payload file required')
        raw = filename.read_bytes()
        if sha_bytes(raw) != args.payload_sha:
            raise ValueError('Exact payload raw SHA256 required')
        payload = json_value(raw, filename)
    else:
        payload = globals().get('PAYLOAD')
        if not isinstance(payload, dict):
            parser.error('Supply sealed PAYLOAD or --payload and --payload-sha')
    print(json.dumps(register(payload, check=args.check), ensure_ascii=False, allow_nan=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
