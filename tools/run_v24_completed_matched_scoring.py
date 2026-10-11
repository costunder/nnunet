"""Full642 depth1 frozen scoring after an actual completed GPU4/5 GNN.

This separate worker admits its own full40/680 BEST on CPU before CUDA. It
uses the original scoring function and the checksum-pinned existing direct
GPU guard. Native training is a separate, storage-dependent stage.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import time

FORMAT = 'v24_full642_depth1_current_BEST_scoring_request_v1'
STATUS_FORMAT = 'v24_full642_depth1_current_BEST_scoring_status_v1'
BINDING_FORMAT = 'v24_completed_matched_scoring_direct_access_binding_v1'
GUARD_FORMAT = 'v24_existing_direct_GPU_availability_guard_v1'
GUARD_SHA256 = '7d8c7b4d7da52ddf04befe48fb4d539bddbbfabc83f0633f631fc1536db24ffe'
AFFINITIES = {4: [0, 1, 2, 3], 5: [42, 43, 44, 45]}
REQUIRED_CODE_FILES = {
    'tools/run_v24_completed_matched_scoring.py', 'tools/local_cnn_device.py',
    'config/server_gpu_allocations.json', 'tools/run_v24_all_p.py',
    'hiercp_v1x/v24_nnunet_cp.py', 'hiercp_v1x/v24_matched_basic_cp_scoring.py',
    'hiercp_v1x/v24_context_depth1.py', 'hiercp_v1x/v24_factory.py',
    'hiercp_v1x/v24_training.py', 'hiercp_v1x/v24_targets.py',
    'hiercp_v1x/v24_provider.py', 'hiercp_v1x/v24_inputs.py',
    'hiercp_v1x/v24_model.py', 'hiercp_v1x/v24_geometry.py',
    'hiercp_v1x/v24_stunet.py', 'hiercp_v1x/v24_preparation_runtime.py',
    'hiercp_v1x/v24_memory_runtime.py', 'hiercp_v1x/v24_prefetch_runtime.py',
    'hiercp_v1x/v24_input_runtime.py', 'hiercp_v1x/v24_hash_runtime.py',
    'hiercp_v1x/v24_training_continuation.py',
    'hiercp_v1x/v24_interrupted_gnn_continuation.py',
    'hiercp_v1x/v23_training.py', 'hiercp_v1x/u_bridge_training.py',
    'hiercp_v1x/transition_v1_local.py', 'hiercp_v1x/transition_v1_empty_context.py',
    'hiercp_v1x/v23_data.py', 'hiercp_v22/storage.py',
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 2**20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def identity(path):
    value = Path(path).stat()
    return [value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns]


def read(path):
    value = json.loads(Path(path).read_text(encoding='utf8'))
    if not isinstance(value, dict):
        raise ValueError('JSON object required: ' + str(path))
    return value


def checksum(value):
    return isinstance(value, str) and len(value) == 64 and set(value) <= set('0123456789abcdef')


def absolute(value):
    if not isinstance(value, str):
        raise ValueError('Absolute path string required')
    path = Path(value)
    if not path.is_absolute() or '..' in path.parts:
        raise ValueError('Absolute non-traversing path required: ' + value)
    return path


def owned(path, *, directory=False):
    path = absolute(str(path))
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError('Symlink input/output alias refused: ' + str(path))
    info = path.stat()
    if not hasattr(os, 'getuid') or os.getuid() != 41833 or info.st_uid != 41833:
        raise ValueError('Actual own Linux UID41833 file/directory required: ' + str(path))
    if not (stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode)):
        raise ValueError('Own regular file/directory required: ' + str(path))
    return path


def publish(path, value):
    """Publish complete bytes exclusively; never replace an old result."""
    path = Path(path)
    descriptor, temporary = tempfile.mkstemp(prefix='.' + path.name + '.', dir=path.parent)
    temporary = Path(temporary)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf8') as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink()


def validate_request(request, request_path, worker_path):
    """CPU-only structural admission; no scientific imports or CUDA calls."""
    root = absolute(str(request_path)).parent
    gpu = request.get('gpu')
    if (request.get('format') != FORMAT or type(gpu) is not int or gpu not in AFFINITIES
            or request.get('debug') is not False or request.get('full642_scoring') is not True
            or request.get('source_entries') != 642 or request.get('CP_positions') != 82176
            or request.get('physical_patient_batch') != 4
            or request.get('physical_candidate_chunk') != (64 if gpu == 4 else 32)
            or request.get('CPU_affinity') != AFFINITIES[gpu]):
        raise ValueError('Explicit full642/82176 frozen GPU4/5 B4 and original measured chunk/CPU contract required')
    for field in ('source_output', 'source_code', 'inventory_path', 'input_cache',
                  'baseline_preprocessed', 'source_bank', 'code'):
        absolute(request.get(field))
    targets = dict(pin_path=root / 'own_best_pin.json', output=root / 'bank',
        direct_guard_module=root / 'direct_GPU45_access_guard.py',
        direct_access_proof=root / 'direct_GPU_access_proof.json',
        direct_access_binding=root / 'direct_access_binding.json')
    if any(absolute(request.get(field)) != target for field, target in targets.items()):
        raise ValueError('Fresh own-root pin/bank/direct proof paths required')
    protected = [absolute(request[name]) for name in
        ('source_output', 'source_code', 'input_cache', 'baseline_preprocessed', 'source_bank', 'code')]
    if any(root == path or root.is_relative_to(path) or path.is_relative_to(root) for path in protected):
        raise ValueError('Scoring namespace must be separate from all original results/inputs/code')
    if gpu == 4:
        absolute(request.get('stunet_checkpoint'))
    elif request.get('stunet_checkpoint') is not None:
        raise ValueError('Original GPU5 CNN/GAT must not receive a STU checkpoint')
    if (not isinstance(request.get('commit'), str) or len(request['commit']) != 40
            or set(request['commit']) - set('0123456789abcdef')
            or not checksum(request.get('worker_sha256'))
            or request.get('direct_guard_sha256') != GUARD_SHA256):
        raise ValueError('Exact Git commit/worker/guard SHA required')
    files = request.get('code_files')
    if not isinstance(files, dict) or not REQUIRED_CODE_FILES <= set(files):
        raise ValueError('Complete scoring/scientific source manifest required')
    for name, digest in files.items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts or not checksum(digest):
            raise ValueError('Relative source filename and exact SHA required')
    if absolute(str(worker_path)) != absolute(request['code']) / 'tools/run_v24_completed_matched_scoring.py':
        raise ValueError('Worker must execute from its sealed immutable checkout')
    preserved = request.get('preserved_source_hashes')
    required = {str(absolute(request['source_output']) / name) for name in
        ('request.json', 'calibration.json', 'training/training_identity.json',
         'training/invocations.jsonl', 'training/checkpoint_latest.pt', 'training/checkpoint_best.pt')}
    required.add(str(absolute(request['source_bank']) / 'index.json'))
    if (not isinstance(preserved, dict) or not required <= set(preserved)
            or any(not checksum(digest) for digest in preserved.values())):
        raise ValueError('Original request/identity/full invocation/latest/BEST/source-bank hashes required')
    for path in preserved:
        absolute(path)
    if not isinstance(request.get('native_pending_reason'), str) or not request['native_pending_reason'].strip():
        raise ValueError('Explicit honest native-storage pending reason required')
    return root


def verify_file(path, digest):
    path = owned(path)
    before = identity(path)
    if sha(path) != digest or identity(path) != before:
        raise ValueError('Immutable source SHA/stat changed: ' + str(path))
    return str(path), before, digest


def verify_binding(request, request_path):
    binding = read(owned(request['direct_access_binding']))
    if (set(binding) != {'format', 'GPU_arm', 'UID', 'CPU_affinity', 'request', 'proof', 'guard'}
            or binding.get('format') != BINDING_FORMAT or binding.get('GPU_arm') != request['gpu']
            or binding.get('UID') != 41833 or binding.get('CPU_affinity') != request['CPU_affinity']):
        raise ValueError('Sealed fresh scoring-only direct access binding required')
    expected = dict(request=str(request_path), proof=request['direct_access_proof'], guard=request['direct_guard_module'])
    records = []
    for key, path in expected.items():
        witness = binding[key]
        if (not isinstance(witness, dict) or set(witness) != {'path', 'sha256', 'stat'}
                or witness['path'] != path or not checksum(witness['sha256'])
                or not isinstance(witness['stat'], list) or len(witness['stat']) != 5):
            raise ValueError('Exact scoring request/proof/guard file witness required')
        record = verify_file(path, witness['sha256'])
        if record[1] != witness['stat']:
            raise ValueError('Direct binding file identity changed: ' + path)
        records.append(record)
    proof = read(request['direct_access_proof'])
    if (proof.get('format') != GUARD_FORMAT or proof.get('GPU_arm') != request['gpu']
            or proof.get('CPU_affinity') != request['CPU_affinity'] or proof.get('UID') != 41833
            or proof.get('request') != {key: binding['request'][key] for key in ('path', 'sha256')}
            or proof.get('guard_sha256') != request['direct_guard_sha256']
            or binding['guard']['sha256'] != request['direct_guard_sha256']):
        raise ValueError('Fresh v1 proof must bind this scoring request and guard, never the old GNN request')
    records.append(verify_file(request['direct_access_binding'], sha(request['direct_access_binding'])))
    return records


def verify_inputs(request, request_path):
    code = owned(request['code'], directory=True)
    owned(request_path.parent, directory=True)
    if subprocess.check_output(['git', '-C', str(code), 'rev-parse', 'HEAD'], text=True).strip() != request['commit']:
        raise ValueError('Immutable scoring checkout HEAD differs')
    if sha(__file__) != request['worker_sha256']:
        raise ValueError('Actual scoring worker SHA differs')
    tracked = set(subprocess.check_output(['git', '-C', str(code), 'ls-files', '-z'], text=True).split('\0'))
    if not set(request['code_files']) <= tracked:
        raise ValueError('Scoring manifest contains untracked implementation files')
    records = [verify_file(code / name, digest) for name, digest in request['code_files'].items()]
    records.extend(verify_file(path, digest) for path, digest in request['preserved_source_hashes'].items())
    records.extend(verify_binding(request, request_path))
    for name in ('own_best_pin.json', 'bank', 'started.json', 'failed.json', 'complete.json'):
        if (request_path.parent / name).exists():
            raise FileExistsError('Fresh scoring namespace required: ' + name)
    return records


def verify_unchanged(records, *, full_hash=False):
    for path, before, digest in records:
        owned(path)
        if identity(path) != before or full_hash and sha(path) != digest or identity(path) != before:
            raise ValueError('Protected original source/code/direct witness changed: ' + path)


def load_guard(request):
    spec = importlib.util.spec_from_file_location('_v24_full642_fresh_direct_guard', request['direct_guard_module'])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.make_guard(request['direct_access_proof'], request['gpu'], request['CPU_affinity'])


def load_scientific_api(code):
    sys.path.insert(0, code)
    from hiercp_v1x import v24_nnunet_cp as pin_api
    from hiercp_v1x import v24_matched_basic_cp_scoring as score_api
    for module, name in ((pin_api, 'v24_nnunet_cp.py'), (score_api, 'v24_matched_basic_cp_scoring.py')):
        if Path(module.__file__).resolve() != Path(code) / 'hiercp_v1x' / name:
            raise ValueError('Scientific module must come from the sealed scoring checkout')
    return pin_api.pin_completed_current, score_api.score_current_matched_basic_bank


def cuda_metrics():
    import torch
    if not torch.cuda.is_initialized():
        raise RuntimeError('Full642 scorer returned without actual CUDA initialization')
    return dict(actual_CUDA=True, peak_CUDA_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_CUDA_reserved_bytes=torch.cuda.max_memory_reserved())


def run(request_path):
    sys.dont_write_bytecode = True
    request_path = owned(Path(request_path))
    root = owned(request_path.parent, directory=True)
    request = read(request_path)
    state = dict(format=STATUS_FORMAT, gpu=request.get('gpu'), debug=False, full642_scoring=True,
        full_native_training=False, full_native_evaluation=False, production_optimizer_updates=0,
        source_files_written=False, signals_sent=False, nnunet_evaluation_checkpoint='checkpoint_best.pth',
        request_sha256=sha(request_path), code=request.get('code'), commit=request.get('commit'))
    try:
        validate_request(request, request_path, Path(__file__).resolve())
        records = verify_inputs(request, request_path)
        access = load_guard(request)
        def guard():
            verify_unchanged(records)
            return access()
        guard()
        print(json.dumps(dict(phase='CPU_PIN_PENDING', GPU=request['gpu'], debug=False,
            full_native_training=False, source_output=request['source_output'])), flush=True)
        pin_completed, score = load_scientific_api(request['code'])
        pin_completed(source_output=request['source_output'], source_code=request['source_code'],
            inventory_path=request['inventory_path'], input_cache=request['input_cache'],
            output=request['pin_path'], gpu=request['gpu'], stunet_checkpoint=request['stunet_checkpoint'])
        pin = read(request['pin_path'])
        if (pin.get('format') != 'v24_completed_single_arm_GT_blind_best_v1'
                or pin.get('physical_GPU') != request['gpu'] or pin.get('completed_epochs') != 40
                or pin.get('optimizer_updates') != 680 or pin.get('full_training') is not True
                or pin.get('debug') is not False or pin.get('source_output') != request['source_output']
                or pin.get('source_code') != request['source_code']):
            raise ValueError('Actual own completed full40/680 BEST pin required')
        state.update(status='FULL642_SCORING_RUNNING', started_at=time.time(), phase='CP_SCORING',
            pin_path=request['pin_path'], pin_sha256=sha(request['pin_path']),
            selected=pin['selected'], completed_GNN_epochs=40, completed_GNN_optimizer_updates=680,
            direct_access=guard())
        publish(root / 'started.json', state)
        print(json.dumps(state), flush=True)
        names = ('pin_path', 'inventory_path', 'baseline_preprocessed', 'input_cache',
                 'source_bank', 'output', 'gpu', 'stunet_checkpoint')
        path = score(**{name: request[name] for name in names}, context_depth1=True, deadline_guard=guard)
        overlay = read(path)
        if (Path(path) != root / 'bank/score_overlay.json'
                or overlay.get('format') != 'v24_own_BEST_scores_on_unchanged_Basic_81case642source128positions_v1'
                or overlay.get('complete') is not True or overlay.get('debug') is not False
                or overlay.get('complete_sources') != 642 or overlay.get('complete_positions') != 82176
                or overlay.get('complete_cases') != 81 or overlay.get('pin') != pin
                or overlay.get('full_P_context') is not True or overlay.get('recipient_GT_used_in_forward') is not False
                or overlay.get('production_optimizer_updates') != 0 or overlay.get('model_and_RNG_unchanged') is not True
                or overlay.get('original_non_score_payloads_preserved') is not True
                or overlay.get('original_source_schedule_and_paste_contract_preserved') is not True
                or overlay.get('inference_profile', {}).get('format') != 'v24_matched_context_liver_depth_gt1mm_inference_v1'):
            raise ValueError('Actual unchanged Basic full642 own-BEST score overlay required')
        verify_unchanged(records, full_hash=True)
        state.update(status='FULL642_SCORING_COMPLETE_NATIVE_STORAGE_PENDING', completed_at=time.time(),
            overlay=str(path), overlay_sha256=sha(path), completed_sources=642, completed_CP_positions=82176,
            native_pending_reason=request['native_pending_reason'], direct_access=guard(), **cuda_metrics())
        publish(root / 'complete.json', state)
        print(json.dumps(state), flush=True)
        return root / 'complete.json'
    except Exception as error:
        state.update(status='FULL642_SCORING_FAILED', failed_at=time.time(),
            error=type(error).__name__ + ': ' + str(error))
        publish(root / 'failed.json', state)
        print(json.dumps(state), flush=True)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    args = parser.parse_args(argv)
    run(args.request)


if __name__ == '__main__':
    main()
