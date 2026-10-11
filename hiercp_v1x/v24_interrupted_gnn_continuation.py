"""Append-only exact-state continuation of an interrupted full GPU4/5 GNN.

The sealed source CLI and checkpoint reader execute their original code. CPU
admission never constructs a model, restores RNG, or initializes CUDA. A fresh
result namespace contains exact latest/BEST copies, checkpoint-aligned active
logs, and an unabridged lineage copy of the original logs. GPU execution requires
an independently owned allocation guard supplied by the launcher.
"""
from __future__ import annotations

import builtins
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import socket
import sys
import time
from types import FunctionType, SimpleNamespace
import uuid

ROOT = Path(__file__).resolve().parents[1]
FORMAT = 'v24_interrupted_exact_saved_GNN_continuation_v1'
RECEIPT = 'interrupted_gnn_continuation.json'
WITNESS_FORMAT = 'v24_interrupted_owned_GNN_source_witness_v1'
NUMERICAL = ('model', 'optimizer', 'scheduler', 'scaler', 'rank_rng', 'shuffle_generator')
EXECUTION_FILES = ('hiercp_v1x/v24_interrupted_gnn_continuation.py',
                   'tools/run_v24_interrupted_gnn_continuation.py')
LOG_NAMES = frozenset(('update_timing.jsonl', 'validation_timing.jsonl', 'curve.jsonl',
                       'invocations.jsonl', 'failures.jsonl'))
PHASES = ('initial_full_validation', 'training', 'train_probe', 'stage_validation',
          'full_validation', 'epoch_completion', 'complete')
LEGACY_GPU5_CLI_SHA256 = 'f61a5da0edfea661b7436b977520c80bc55df7d7ed51c8ef60fb616163c4959f'
WITNESS_KEYS = frozenset(('format', 'GPU_arm', 'uid', 'server_host', 'source_output',
    'source_code', 'source_job', 'status_file', 'pipeline_request_file', 'worker',
    'child', 'checkpoint', 'training_inventory', 'observations', 'detected_ns'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 2**20), b''):
            digest.update(block)
    return digest.hexdigest()


def stat(path):
    value = Path(path).stat()
    return [value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns]


def owned(path, *, directory=False):
    path = Path(path)
    if (not path.is_absolute() or path.is_symlink()
            or path.resolve(strict=True) != path.absolute()
            or not (path.is_dir() if directory else path.is_file())
            or hasattr(os, 'getuid') and path.stat().st_uid != os.getuid()):
        raise ValueError('Exact owned regular nonsymlink path required: ' + str(path))
    return path


def read(path):
    return json.loads(owned(path).read_text(encoding='utf8'))


def file_proof(path):
    path = owned(path); before = stat(path); checksum = sha(path)
    if stat(path) != before:
        raise ValueError('Source changed while hashing: ' + str(path))
    return dict(path=str(path), stat=before, sha256=checksum)


def verify_file(proof):
    if not isinstance(proof, dict) or set(proof) != {'path', 'stat', 'sha256'}:
        raise ValueError('Closed exact file witness required')
    if file_proof(proof['path']) != proof:
        raise ValueError('Preserved source SHA/stat changed: ' + str(proof['path']))
    return Path(proof['path'])


def atomic_bytes(path, raw):
    """Exclusive complete-file publication; remove only this private temporary."""
    path = Path(path); temporary = path.with_name(path.name + '.interrupted_' + uuid.uuid4().hex + '.tmp')
    created = False
    try:
        with temporary.open('xb') as stream:
            created = True; stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if created:
            temporary.unlink()


def atomic_json(path, value):
    atomic_bytes(path, json.dumps(value, indent=2, allow_nan=False).encode('utf8'))


def publish(path, value):
    path = Path(path)
    if path.exists():
        if read(path) != value:
            raise FileExistsError('Existing continuation metadata differs: ' + str(path))
        return
    atomic_json(path, value)


def copy_exact(source, destination):
    proof = file_proof(source); destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Streaming avoids a second checkpoint-sized resident Python bytes object.
    temporary = destination.with_name(destination.name + '.copy_' + uuid.uuid4().hex + '.tmp')
    created = False
    try:
        with Path(proof['path']).open('rb') as reader, temporary.open('xb') as writer:
            created = True; opened = os.fstat(reader.fileno())
            if [opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns, opened.st_ctime_ns] != proof['stat']:
                raise ValueError('Source changed before exact copy')
            shutil.copyfileobj(reader, writer, 8 * 2**20); writer.flush(); os.fsync(writer.fileno())
        verify_file(proof)
        if sha(temporary) != proof['sha256']:
            raise ValueError('Copied checkpoint/file bytes differ')
        os.link(temporary, destination)
    finally:
        if created:
            temporary.unlink()
    return proof


def training_inventory(source):
    folder = owned(Path(source) / 'training', directory=True)
    result = {}
    for path in sorted(folder.iterdir()):
        owned(path)  # Unknown directories/symlinks fail before any copy.
        result[path.name] = file_proof(path)
    return result


def load_science(source_code):
    """Activate only the sealed checkout; this module is loaded under a private name."""
    code = owned(source_code, directory=True)
    for name, module in tuple(sys.modules.items()):
        if module is sys.modules.get(__name__):
            continue
        if name == 'tools' or name.startswith(('tools.', 'hiercp_v1x', 'hiercp_v22')):
            filename = getattr(module, '__file__', None)
            if filename and not Path(filename).resolve().is_relative_to(code):
                raise ValueError('Mixed scientific checkout refused: ' + name)
    sys.dont_write_bytecode = True
    if str(code) not in sys.path:
        sys.path.insert(0, str(code))
    from tools import run_v24_all_p as cli
    from hiercp_v1x import v24_training_continuation as continuation
    if Path(cli.__file__).resolve() != code / 'tools/run_v24_all_p.py':
        raise ValueError('Actual original source CLI required')
    return cli, continuation


def source_names(cli, gpu):
    selector = getattr(cli, 'execution_source_files', None)
    if selector is not None:
        return tuple(selector(gpu))
    # The immutable reviewed GPU5 CLI predates GPU4's additional execution list.
    if gpu == 5 and sha(cli.__file__) == LEGACY_GPU5_CLI_SHA256:
        return tuple(cli.FILES)
    raise ValueError('Explicit original execution source manifest required')


def _context():
    import psutil
    if not hasattr(os, 'getuid') or os.getuid() != 41833 or socket.gethostname().split('.')[0] != 'ece-a6gpu6':
        raise ValueError('Exact original ece-a6gpu6 server and UID41833 required')
    return dict(uid=os.getuid(), host=socket.gethostname().split('.')[0], pid_exists=psutil.pid_exists)


def _process_shape(value):
    if (not isinstance(value, dict) or set(value) != {'pid', 'create_time', 'uid', 'command', 'absent'}
            or type(value['pid']) is not int or value['pid'] <= 0
            or type(value['create_time']) not in (int, float)
            or not math.isfinite(value['create_time']) or value['create_time'] <= 0
            or value['uid'] != 41833 or value['absent'] is not True
            or not isinstance(value['command'], list) or not value['command']
            or any(not isinstance(arg, str) or not arg for arg in value['command'])):
        raise ValueError('Exact absent original UID/PID/birth/full-argv witness required')


def _bind_job(proof, gpu, context):
    job = owned(proof['source_job'], directory=True)
    if Path(proof['status_file']['path']) != job / 'status.json' or Path(proof['pipeline_request_file']['path']) != job / 'request.json':
        raise ValueError('Exact original job metadata paths required')
    status = read(verify_file(proof['status_file'])); request = read(verify_file(proof['pipeline_request_file']))
    if (status.get('status') not in ('RUNNING', 'FAILED', 'PAUSED') or status.get('stage') != 'train'
            or request.get('GPU') != gpu or request.get('production_output') != proof['source_output']
            or status.get('request') != request):
        raise ValueError('Original interrupted train job/request differs')
    for name in ('worker', 'child'):
        row = proof[name]; _process_shape(row)
        if (status.get(name + '_pid') != row['pid'] or status.get(name + '_create_time') != row['create_time']
                or context['pid_exists'](row['pid'])):
            raise ValueError('Original process still exists, was reused, or PID/birth differs')
    if proof['worker']['pid'] == proof['child']['pid'] or proof['worker']['command'][-1] != str(job / 'request.json'):
        raise ValueError('Original distinct worker/child and exact worker request argv required')
    stages = [row for row in request.get('stages', []) if row.get('action', row.get('name')) == 'train']
    if len(stages) != 1 or stages[0].get('command') != proof['child']['command']:
        raise ValueError('Original child full argv differs from the exact original train stage')
    declared_worker = status.get('worker_command')
    declared_child = status.get('child_command')
    if declared_worker is not None and declared_worker != proof['worker']['command']:
        raise ValueError('Original recorded worker argv differs')
    if declared_child is not None and declared_child != proof['child']['command']:
        raise ValueError('Original recorded child argv differs')


def collect_interruption_witness(source_output, source_code, source_job, *, worker, child):
    """Read-only metadata witness. The launcher supplies prior exact live identities."""
    context = _context(); source = owned(source_output, directory=True); code = owned(source_code, directory=True)
    job = owned(source_job, directory=True); status = file_proof(job / 'status.json'); request = file_proof(job / 'request.json')
    proof = dict(format=WITNESS_FORMAT, GPU_arm=read(source / 'request.json')['physical_GPU'],
        uid=context['uid'], server_host=context['host'], source_output=str(source), source_code=str(code),
        source_job=str(job), status_file=status, pipeline_request_file=request, worker=copy.deepcopy(worker),
        child=copy.deepcopy(child), checkpoint=file_proof(source / 'training/checkpoint_latest.pt'),
        training_inventory=training_inventory(source), observations=[], detected_ns=0)
    if proof['GPU_arm'] not in (4, 5):
        raise ValueError('Only unfinished GPU4/5 arms admitted')
    for index in range(2):
        _bind_job(proof, proof['GPU_arm'], context)
        if file_proof(source / 'training/checkpoint_latest.pt') != proof['checkpoint']:
            raise ValueError('Original checkpoint changed between absence observations')
        proof['observations'].append(dict(time_ns=time.time_ns(), worker_absent=True, child_absent=True,
                                          checkpoint=copy.deepcopy(proof['checkpoint'])))
        if index == 0:
            time.sleep(1.01)
    if training_inventory(source) != proof['training_inventory']:
        raise ValueError('Original complete training inventory changed')
    proof['detected_ns'] = time.time_ns()
    return proof


def validate_interruption(path, source, code, latest, gpu):
    context = _context(); attachment = file_proof(path); proof = read(path)
    if (set(proof) != WITNESS_KEYS or proof['format'] != WITNESS_FORMAT
            or proof['GPU_arm'] != gpu or gpu not in (4, 5) or proof['uid'] != context['uid']
            or proof['server_host'] != context['host'] or proof['source_output'] != str(source)
            or proof['source_code'] != str(code) or latest['status'] not in ('RUNNING', 'PAUSED')):
        raise ValueError('Closed exact interrupted full-GNN source witness required')
    _bind_job(proof, gpu, context)
    checkpoint = proof['checkpoint']
    if (checkpoint['path'] != latest['path'] or checkpoint['stat'] != latest['source_stat']
            or checkpoint['sha256'] != latest['raw_sha256']):
        raise ValueError('Original six-state checkpoint and interruption witness differ')
    verify_file(checkpoint)
    observations = proof['observations']
    if (not isinstance(observations, list) or len(observations) != 2
            or any(set(row) != {'time_ns', 'worker_absent', 'child_absent', 'checkpoint'}
                or type(row['time_ns']) is not int or row['time_ns'] <= 0
                or row['worker_absent'] is not True or row['child_absent'] is not True
                or row['checkpoint'] != checkpoint for row in observations)
            or observations[1]['time_ns'] - observations[0]['time_ns'] < 1_000_000_000
            or type(proof['detected_ns']) is not int
            or not observations[1]['time_ns'] <= proof['detected_ns'] <= time.time_ns()):
        raise ValueError('Two stable original checkpoint/PID observations required')
    if training_inventory(source) != proof['training_inventory']:
        raise ValueError('Complete preserved training inventory differs from interruption witness')
    _bind_job(proof, gpu, context); verify_file(attachment)
    return dict(attachment=attachment, witness=proof, original_job_status_preserved=True,
                original_workers_absent=True, no_process_signals=True)


def _memory_selection(source, gpu):
    if gpu == 4:
        return None
    path = source / 'gpu5_performance_continuation.json'
    if not path.exists() and (source / RECEIPT).exists():
        previous = read(source / RECEIPT)
        if previous.get('format') != FORMAT or previous.get('destination') != str(source):
            raise ValueError('Exact prior interrupted GPU5 continuation lineage required')
        inherited = previous.get('source', {}).get('preserved_memory')
        if not isinstance(inherited, dict) or set(inherited) != {'receipt', 'selection'}:
            raise ValueError('Exact preserved GPU5 memory lineage required')
        path = verify_file(inherited['receipt'])
    receipt = read(path); selection = receipt.get('runtime', {}).get('preserved_memory_selection')
    if (receipt.get('format') != 'v24_gpu5_execution_only_exact_continuation_v1'
            or not isinstance(selection, dict) or selection.get('variant') != 'current-control'
            or selection.get('actual_module_name') != 'hiercp_v1x.v24_memory_runtime'
            or selection.get('imported_before_prefetch_input_factory') is not True):
        raise ValueError('Actual previously admitted GPU5 memory/runtime receipt required')
    if sha(owned(selection['actual_module_path'])) != selection['actual_module_sha256']:
        raise ValueError('Previously admitted GPU5 memory source changed')
    for proof in selection['source_provenance']['source_file_proofs']:
        if stat(owned(proof['path'])) != proof['stat'] or sha(proof['path']) != proof['raw_sha256']:
            raise ValueError('Actual GPU5 preserved-memory provenance changed')
    return dict(receipt=file_proof(path), selection=selection)


def _validate_unfinished_latest(latest, gpu):
    # The original loop saves epoch41/training after completing epoch40 before
    # publishing its final complete state. Resume that boundary without updates.
    if (latest['phase'] == 'complete' or latest['epoch'] > 41
            or latest['epoch'] == 41 and latest['phase'] != 'training'
            or latest['optimizer_parameter_tensors'] != (537 if gpu == 4 else 981)):
        raise ValueError('Actual unfinished original full-model optimizer snapshot required')


def inspect_source(source_output, source_code, interruption_proof, *, cli=None, continuation=None):
    """CPU-only original strict six-state, BEST, source and absence admission."""
    if cli is None or continuation is None:
        cli, continuation = load_science(source_code)
    from hiercp_v1x.u_bridge_training import digest
    from hiercp_v1x.v23_training import best_selection_key
    import torch
    source = owned(source_output, directory=True); code = owned(source_code, directory=True)
    request = read(source / 'request.json'); gpu = request.get('physical_GPU')
    if gpu not in (4, 5):
        raise ValueError('Only original interrupted GPU4/5 full GNN admitted')
    declared = {key: value for key, value in request.items() if key != 'request_sha256'}
    names = source_names(cli, gpu)
    if (request.get('format') != 'v24_full_native_GT_free_execution_request_v1'
            or hashlib.sha256(json.dumps(declared, sort_keys=True, separators=(',', ':')).encode()).hexdigest() != request['request_sha256']
            or set(request['source']) != set(names)):
        raise ValueError('Original full execution request/source manifest required')
    files = {name: file_proof(code / name) for name in names}
    if ({name: row['sha256'] for name, row in files.items()} != request['source']
            or Path(cli.ROOT).resolve() != code):
        raise ValueError('Original immutable scientific CLI/source bytes differ')
    owner = read(source / 'training/training_identity.json'); binding = owner['binding']
    if digest(binding) != owner['identity_sha256'] or binding['identity'] != request:
        raise ValueError('Original full training identity/request differs')
    calibration = read(source / 'calibration.json'); chunk = 64 if gpu == 4 else 32
    if (calibration != binding['config']['v24_runtime']['batch_calibration']
            or calibration['request_sha256'] != request['request_sha256'] or calibration['physical_GPU'] != gpu
            or calibration['initial_state_sha256'] != binding['initial_state_sha256']
            or calibration['model_contract'] != binding['model_contract']
            or calibration['selected_physical_patient_batch'] != 4
            or calibration['selected_physical_candidate_batch'] != chunk
            or calibration['world_size'] != 1 or calibration['debug'] is not False
            or calibration['original_model_and_RNG_preserved'] is not True
            or calibration['measured_full_P_U128_backward'] is not True
            or binding['physical_patient_batch'] != 4 or binding['physical_candidate_batch'] != chunk
            or binding['epochs'] != 40 or binding['workers'] != 4 or binding['debug'] is not False
            or len(binding['train_cases']) != 65 or len(binding['val_cases']) != 21
            or len(set(binding['train_cases'] + binding['val_cases'])) != 86):
        raise ValueError('Original full population, measured B4/chunk32|64, workers4 and40epochs required')
    saved, latest = continuation._checkpoint(source / 'training/checkpoint_latest.pt', owner)
    _validate_unfinished_latest(latest, gpu)
    if torch.cuda.is_initialized() or any(value.device.type != 'cpu' or value.is_floating_point() and not bool(torch.isfinite(value).all())
            for value in saved['model'].values() if isinstance(value, torch.Tensor)):
        raise ValueError('Complete finite CPU-only model snapshot required')
    best_proof = None
    if saved['state']['best'] is not None:
        best, best_proof = continuation._checkpoint(source / 'training/checkpoint_best.pt', owner)
        selected = saved['state']['best']; full = best['state']['pending_epoch_completion']['full_validation']
        if (best['state']['best'] != selected or best['state']['epoch'] != selected['epoch']
                or best['state']['updates'] != selected['updates'] or full['phase'] != 'full_validation'
                or full['active_u'] != 128 or list(best_selection_key(full)) != selected['selection_key']):
            raise ValueError('Original actual fixed-full128 BEST must remain exact')
        del best
    elif (source / 'training/checkpoint_best.pt').exists():
        raise ValueError('Unexpected BEST artifact without a saved selected winner')
    interruption = validate_interruption(interruption_proof, source, code, latest, gpu)
    helper = file_proof(code / 'hiercp_v1x/v24_training_continuation.py')
    return dict(source_output=str(source), source_code=str(code), GPU_arm=gpu,
        source_files=files, checkpoint_reader=helper, request=file_proof(source / 'request.json'),
        training_identity=file_proof(source / 'training/training_identity.json'), calibration=file_proof(source / 'calibration.json'),
        training_identity_sha256=owner['identity_sha256'], latest=latest, BEST=best_proof,
        interruption=interruption, preserved_memory=_memory_selection(source, gpu)), saved


def derive_checkpoint_log(raw, name, state):
    """Archive every original byte; active JSONL contains only durable events."""
    if name not in LOG_NAMES:
        raise ValueError('Unknown original training log')
    kept = []; excluded = 0; partial = 0; validation_cursor = 0; overflow_count = 0
    lines = raw.splitlines(keepends=True)
    for index, line in enumerate(lines):
        if not line.endswith(b'\n'):
            if index != len(lines) - 1:
                raise ValueError('Non-final partial JSONL record')
            partial += 1; continue
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError('Original JSONL object required')
        keep = False; epoch = row.get('epoch')
        if name in ('invocations.jsonl', 'failures.jsonl'):
            keep = False
        elif name == 'curve.jsonl':
            if type(epoch) is not int:
                raise ValueError('Original curve epoch required')
            keep = 1 <= epoch < state['epoch']
            if keep and row != state['history'][epoch - 1]:
                raise ValueError('Durable curve differs from exact saved history')
        elif name == 'update_timing.jsonl':
            if type(epoch) is not int:
                raise ValueError('Original update epoch required')
            if row.get('status') == 'OPTIMIZER_UPDATED':
                update = row.get('update')
                if type(update) is not int:
                    raise ValueError('Original optimizer update number required')
                keep = epoch <= state['epoch'] and 0 < update <= state['updates']
            elif row.get('status') == 'AMP_OVERFLOW_RETRY_SAME_INPUT':
                cursor = row.get('cursor')
                if type(cursor) is not int:
                    raise ValueError('Original AMP retry cursor required')
                overflow_count += 1
                keep = (epoch < state['epoch'] or epoch == state['epoch'] and cursor <= state['train_position']) and overflow_count <= state['overflows']
            else:
                raise ValueError('Unknown original update event')
        elif name == 'validation_timing.jsonl':
            phase = row.get('phase'); cases = row.get('case_ids')
            if (type(epoch) is not int or phase not in ('initial_full_validation', 'train_probe', 'stage_validation', 'full_validation')
                    or not isinstance(cases, list) or not cases or len(set(cases)) != len(cases)):
                raise ValueError('Original validation phase/full case_ids batch required')
            if epoch < state['epoch']:
                keep = True
            elif epoch == state['epoch']:
                if phase == state['phase']:
                    start = validation_cursor; validation_cursor += len(cases)
                    keep = validation_cursor <= state['evaluation_position']
                    if keep and cases != [row['case_id'] for row in state['evaluation_rows'][start:validation_cursor]]:
                        raise ValueError('Durable evaluation case order differs')
                else:
                    keep = PHASES.index(phase) < PHASES.index(state['phase'])
        if keep:
            kept.append(line)
        else:
            excluded += 1
    result = b''.join(kept)
    if name == 'curve.jsonl' and [json.loads(line) for line in kept] != state['history']:
        raise ValueError('Complete saved epoch history must be represented exactly')
    return result, dict(boundary_epoch=state['epoch'], boundary_updates=state['updates'],
        boundary_train_position=state['train_position'], boundary_evaluation_position=state['evaluation_position'],
        rows_kept=len(kept), rows_archived_only=excluded, partial_rows_archived=partial,
        bytes=len(result), sha256=hashlib.sha256(result).hexdigest())


def _report_durable(name, raw, state):
    match = re.fullmatch(r'(initial_full_validation|train_probe|stage_validation|full_validation)_epoch_(\d{3})\.json', name)
    if match is None:
        raise ValueError('Unknown original training JSON artifact: ' + name)
    phase, token = match.groups(); epoch = int(token)
    keep = ((phase == 'initial_full_validation' and epoch == 0 and 'initial_validation' in state)
            or 1 <= epoch < state['epoch']
            or epoch == state['epoch'] and PHASES.index(phase) < PHASES.index(state['phase']))
    if keep:
        row = json.loads(raw)
        if row.get('phase') != phase or row.get('epoch') != epoch:
            raise ValueError('Durable full-population report filename/content differs')
        expected = (state.get('initial_validation') if phase == 'initial_full_validation'
            else state['history'][epoch - 1].get(phase) if epoch < state['epoch']
            else state.get('pending_epoch_completion', {}).get(phase, state.get(phase)))
        if expected is None or row != expected:
            raise ValueError('Durable report differs from exact saved history/phase')
    return keep


def execution_identity():
    return dict(execution_code=str(ROOT), execution_files={name: file_proof(ROOT / name) for name in EXECUTION_FILES})


def validate_args(args, source, cli):
    if args.mode != 'train' or args.gpu not in (4, 5) or args.resume is not None:
        raise ValueError('Original GPU4/5 train args; the helper alone selects exact resume')
    config = cli.validate_config(read(args.config), args.gpu, args.stunet_checkpoint)
    if cli.request(args, config) != read(Path(source) / 'request.json'):
        raise ValueError('Original scientific request/config/cache/asset/inventory changed')
    if args.cpu_affinity is None or len(args.cpu_affinity) != 4 or len(set(args.cpu_affinity)) != 4:
        raise ValueError('Four actual assigned CPU IDs must be explicit')
    return config


def prepare(args, *, source_output, source_code, interruption_proof, cli=None, continuation=None):
    if cli is None or continuation is None:
        cli, continuation = load_science(source_code)
    source = owned(source_output, directory=True); code = owned(source_code, directory=True)
    validate_args(args, source, cli)
    proof, saved = inspect_source(source, code, interruption_proof, cli=cli, continuation=continuation)
    output = Path(args.output).absolute()
    if output.exists() or output.is_relative_to(source) or source.is_relative_to(output):
        raise FileExistsError('Fresh disjoint append-only continuation namespace required')
    owned(output.parent, directory=True)
    inventory = proof['interruption']['witness']['training_inventory']
    required_bytes = sum(row['stat'][2] for row in inventory.values() if not row['path'].endswith('.tmp')) + 10 * 2**30
    if shutil.disk_usage(output.parent).free < required_bytes:
        raise OSError('Exact checkpoint/BEST/log copies plus10GiB free reserve do not fit')
    identity = execution_identity(); output.mkdir(); (output / 'training').mkdir()
    (output / 'source_lineage/training').mkdir(parents=True)
    copied = {}; lineage = {}; derived = {}; exclusions = []
    root_inventory = {}
    for path in sorted(source.iterdir()):
        if path.is_file() and (path.name in ('request.json', 'calibration.json', 'preparation.json')
                or path.name.startswith('resources_') and path.suffix == '.json'):
            root_inventory[path.name] = file_proof(path)
            copied[path.name] = copy_exact(path, output / path.name)
    for name, item in inventory.items():
        path = Path(item['path']); verify_file(item)
        if name == 'STOP_AFTER_BATCH' or re.fullmatch(r'checkpoint_(latest|best)\.pt\.[0-9a-f]{32}\.tmp', name):
            exclusions.append(dict(source=item, reason='Original stop marker or uncommitted partial checkpoint preserved in source; never activated'))
            continue
        if name in ('checkpoint_latest.pt', 'checkpoint_best.pt', 'training_identity.json', 'execution_contract.json'):
            copied['training/' + name] = copy_exact(path, output / 'training' / name)
            continue
        if name in LOG_NAMES or name.endswith('.json'):
            relative = 'source_lineage/training/' + name
            lineage[name] = copy_exact(path, output / relative); raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != item['sha256']:
                raise ValueError('Original log/report changed during checkpoint-boundary derivation')
            if name in LOG_NAMES:
                content, record = derive_checkpoint_log(raw, name, saved['state'])
                atomic_bytes(output / 'training' / name, content)
                derived[name] = dict(record, path='training/' + name, lineage_path=relative, source_sha256=item['sha256'])
            elif _report_durable(name, raw, saved['state']):
                copied['training/' + name] = copy_exact(path, output / 'training' / name)
            else:
                exclusions.append(dict(source=item, lineage_path=relative, reason='Report not represented by durable checkpoint; exact original bytes archived'))
            continue
        raise ValueError('Unknown original training artifact; preserve and inspect: ' + name)
    repeated, _ = inspect_source(source, code, interruption_proof, cli=cli, continuation=continuation)
    if repeated != proof:
        raise ValueError('Original complete interrupted snapshot changed during preparation')
    for row in root_inventory.values():
        verify_file(row)
    owner = read(output / 'training/training_identity.json')
    _, latest = continuation._checkpoint(output / 'training/checkpoint_latest.pt', owner)
    for key in ('raw_sha256', 'content_sha256', 'identity_sha256', 'numerical_state_sha256', 'state_sha256'):
        if latest[key] != proof['latest'][key]:
            raise ValueError('Copied exact checkpoint state differs: ' + key)
    cursor = dict(format='v24_checkpoint_aligned_continuation_viewer_cursor_v1', source=proof,
        copied_files={row['path']:dict(bytes=row['bytes'], sha256=row['sha256'], derived_checkpoint_boundary=True)
                      for row in derived.values()}, created_at=time.time())
    atomic_json(output / 'training_continuation.json', cursor)
    document = dict(format=FORMAT, status='PREPARED_EXACT_SAVED_STATE', destination=str(output),
        source=proof, **identity, copied_files=copied, original_root_inventory=root_inventory,
        original_full_logs_and_reports=lineage, derived_logs=derived, archived_only=exclusions,
        viewer_cursor=file_proof(output / 'training_continuation.json'), original_files_modified=False,
        scientific_request_unchanged=True, six_numerical_states_unchanged=True,
        curriculum_history_and_partial_cursors_unchanged=True, production_optimizer_updates_performed=0,
        required_free_bytes=required_bytes, created_at=time.time())
    atomic_json(output / RECEIPT, document)
    return document


def admit(args, *, source_output, source_code, interruption_proof, cli=None, continuation=None):
    if cli is None or continuation is None:
        cli, continuation = load_science(source_code)
    source = owned(source_output, directory=True); validate_args(args, source, cli)
    output = owned(args.output, directory=True); document = read(output / RECEIPT)
    if (document.get('format') != FORMAT or document.get('status') != 'PREPARED_EXACT_SAVED_STATE'
            or document.get('destination') != str(output) or document.get('original_files_modified') is not False
            or document.get('production_optimizer_updates_performed') != 0
            or any(document.get(key) != value for key, value in execution_identity().items())
            or any(document.get(key) is not True for key in ('scientific_request_unchanged',
                'six_numerical_states_unchanged', 'curriculum_history_and_partial_cursors_unchanged'))):
        raise ValueError('Exact prepared original state/execution receipt required')
    proof, _ = inspect_source(source, source_code, interruption_proof, cli=cli, continuation=continuation)
    if proof != document['source']:
        raise ValueError('Original interrupted source changed after preparation')
    for relative, row in document['copied_files'].items():
        target = output / relative
        if not target.resolve(strict=True).is_relative_to(output) or sha(owned(target)) != row['sha256']:
            raise ValueError('Prepared canonical checkpoint/report bytes changed')
        verify_file(row)
    for name, row in document['derived_logs'].items():
        if sha(owned(output / row['path'])) != row['sha256'] or sha(owned(output / row['lineage_path'])) != row['source_sha256']:
            raise ValueError('Prepared checkpoint-aligned active/lineage log changed')
    for name, row in document['original_full_logs_and_reports'].items():
        verify_file(row)
        if sha(owned(output / 'source_lineage/training' / name)) != row['sha256']:
            raise ValueError('Prepared complete original report/log lineage changed')
    for row in document['original_root_inventory'].values():
        verify_file(row)
    verify_file(document['viewer_cursor'])
    _, latest = continuation._checkpoint(output / 'training/checkpoint_latest.pt', read(output / 'training/training_identity.json'))
    for key in ('raw_sha256', 'content_sha256', 'identity_sha256', 'numerical_state_sha256', 'state_sha256'):
        if latest[key] != proof['latest'][key]:
            raise ValueError('Prepared six-state/cursor checkpoint changed: ' + key)
    return document


def clone_main(original, build_hook, publisher):
    """Identical original CLI code, private factory import and atomic publisher."""
    namespace = dict(original.__globals__); namespace['publish'] = publisher
    original_builtins = namespace.get('__builtins__', builtins)
    private = dict(original_builtins if isinstance(original_builtins, dict) else vars(original_builtins))
    importer = builtins.__import__
    def importing(name, globals=None, locals=None, fromlist=(), level=0):
        module = importer(name, globals, locals, fromlist, level)
        if name == 'hiercp_v1x.v24_factory' and level == 0 and tuple(fromlist) == ('V24NativeInputs', 'build_runtime'):
            return SimpleNamespace(V24NativeInputs=module.V24NativeInputs, build_runtime=build_hook(module.build_runtime))
        return module
    private['__import__'] = importing; namespace['__builtins__'] = private
    result = FunctionType(original.__code__, namespace, original.__name__, original.__defaults__, original.__closure__)
    result.__kwdefaults__ = copy.deepcopy(original.__kwdefaults__)
    return result


def load_allocation_guard(module_path, module_sha256, proof_path, proof_sha256, gpu, CPU_affinity):
    if (gpu not in (4, 5) or not isinstance(CPU_affinity, list) or len(CPU_affinity) != 4
            or len(set(CPU_affinity)) != 4 or any(type(value) is not int or value < 0 for value in CPU_affinity)):
        raise ValueError('Explicit GPU4/5 and four assigned CPU IDs required')
    module_proof = file_proof(module_path); allocation_proof = file_proof(proof_path)
    if module_proof['sha256'] != module_sha256 or allocation_proof['sha256'] != proof_sha256:
        raise ValueError('Owned allocation guard/proof source SHA differs')
    spec = importlib.util.spec_from_file_location('_v24_interrupted_allocation_guard', module_proof['path'])
    if spec is None or spec.loader is None:
        raise ValueError('Actual launcher allocation guard import required')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    guard = module.make_guard(Path(allocation_proof['path']), gpu, list(CPU_affinity))
    if not callable(guard):
        raise ValueError('Actual live allocation guard callable required')
    def guarded():
        verify_file(module_proof); verify_file(allocation_proof); guard()
    guarded()
    return guarded, dict(module=module_proof, allocation=allocation_proof)


def train(args, *, source_output, source_code, interruption_proof, allocation_guard,
          allocation_binding, cli=None, continuation=None):
    if not callable(allocation_guard) or not isinstance(allocation_binding, dict):
        raise ValueError('Explicit actual scheduler allocation guard/binding required')
    allocation_guard()
    if cli is None or continuation is None:
        cli, continuation = load_science(source_code)
    document = admit(args, source_output=source_output, source_code=source_code,
        interruption_proof=interruption_proof, cli=cli, continuation=continuation)
    import psutil
    process = psutil.Process()
    if (len(args.cpu_affinity) != 4 or not set(args.cpu_affinity) <= set(process.cpu_affinity())):
        raise ValueError('Actual allocation does not contain four admitted CPU IDs')
    atomic_json(Path(args.output) / 'INTERRUPTED_CONTINUATION_TRAIN_CLAIM.json', dict(format=FORMAT,
        pid=process.pid, create_time=process.create_time(), uid=os.getuid(), command=process.cmdline(),
        source_checkpoint=document['source']['latest']['raw_sha256'], allocation=allocation_binding,
        CPU_affinity=args.cpu_affinity, created_at=time.time()))
    selected = document['source']['preserved_memory']
    if selected is not None:
        from tools.v24_memory_reclamation_gate import route_memory_runtime
        routed = route_memory_runtime(Path(selected['selection']['source_provenance']['source_code']), 'current-control')
        if routed != selected['selection']:
            raise ValueError('Exact previously probed GPU5 memory routing differs')
    from hiercp_v1x import v24_factory, v24_memory_runtime as memory, v24_hash_runtime as hashing
    from hiercp_v1x import v24_prefetch_runtime as prefetch, v24_input_runtime as inputs, v24_training as training
    from hiercp_v1x.v24_preparation_runtime import install_runtime
    from hiercp_v1x.u_bridge_training import capture_rng, digest
    from hiercp_v1x.historical_evaluation import ResourceBudget
    install_runtime(v24_factory)
    installed = dict(memory=memory.install_memory_runtime(v24_factory), hash=hashing.install(),
        prefetch=prefetch.install_runtime(memory), inputs=inputs.install_runtime(pin_final_outputs=True))
    calls = []
    def hook(build):
        def bounded(*positional, **keywords):
            allocation_guard(); result = build(*positional, **keywords)
            net, scorer, population, config, contract, close = result
            try:
                before_model, before_rng = digest(net.state_dict()), digest(capture_rng())
                binding = dict(memory=memory.bind_memory_runtime(scorer), prefetch=prefetch.bind(scorer), inputs=inputs.bind(scorer))
                if digest(net.state_dict()) != before_model or digest(capture_rng()) != before_rng:
                    raise ValueError('Continuation adapters changed original initialized model/RNG')
                if before_model != read(Path(args.output) / 'calibration.json')['initial_state_sha256']:
                    raise ValueError('Original initialized full model/calibration differs')
                raw = scorer.geometry.memory_guard.__self__
                guards = list(document['source']['source_files'].values()) + list(document['execution_files'].values())
                guards.append(document['source']['checkpoint_reader'])
                guards.extend((file_proof(memory.__file__), file_proof(hashing.__file__), file_proof(prefetch.__file__), file_proof(inputs.__file__)))
                guards.extend(allocation_binding.values())
                for row in guards:
                    path = verify_file(row); witness = v24_factory._stat(path)
                    raw._input_file_proofs[str(path)] = witness; raw._file_proofs[str(path)] = witness
                raw.guard_source(); allocation_guard()
                record = dict(format=FORMAT, original_CLI_code_preserved=True, original_strict_resume=True,
                    source_checkpoint=document['source']['latest'], original_scientific_identity_unchanged=True,
                    initial_model_and_RNG_preserved=True, installed=installed, bound=binding,
                    preserved_memory=selected, allocation=allocation_binding, execution=execution_identity())
                atomic_json(Path(args.output) / ('interrupted_continuation_binding_' + str(time.time_ns()) + '.json'), record)
                calls.append(record); return result
            except BaseException:
                close(); raise
        return bounded
    cloned = clone_main(cli.main, hook, publish)
    if cloned.__code__ is not cli.main.__code__:
        raise ValueError('Actual original full CLI code required')
    original_writer = training._write_new; original_check = ResourceBudget.check
    def checked(budget):
        allocation_guard(); return original_check(budget)
    training._write_new = atomic_json; ResourceBudget.check = checked
    atomic_json(Path(args.output) / ('interrupted_continuation_execution_' + str(time.time_ns()) + '.json'),
        dict(format=FORMAT, original_CLI_code_preserved=True, original_strict_resume=True,
            six_saved_states_to_restore=document['source']['latest']['numerical_state_sha256'],
            resume_checkpoint=str(Path(args.output) / 'training/checkpoint_latest.pt'),
            durable_optimizer_updates=document['source']['latest']['updates'], allocation=allocation_binding,
            GPU4_fresh_runner_resume_gate_bypassed_by_original_CLI_only=args.gpu == 4, created_at=time.time()))
    try:
        allocation_guard()
        result = cloned(continuation._argv(args, Path(args.output) / 'training/checkpoint_latest.pt'))
        if len(calls) != 1:
            raise ValueError('Exactly one original full-model runtime build required')
        return result
    finally:
        training._write_new = original_writer; ResourceBudget.check = original_check
