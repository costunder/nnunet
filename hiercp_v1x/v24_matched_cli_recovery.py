"""CPU-only admission for one append-only repair of the matched first-pin CLI.

This module never launches or signals a process and never changes an old claim,
lease, result, or checkout. The caller publishes a fresh request and exclusive
lease only after the exact task-owned previous controller has stopped.
"""
from __future__ import annotations

import math
from pathlib import Path

FORMAT = 'v24_matched_Basic642_first_pin_CLI_supersession_proof_v1'
LEASE = 'v24_matched_Basic642_first_pin_CLI_supersession_lease_v1'
HELPER = 'hiercp_v1x/v24_matched_cli_recovery.py'
PIN_ERROR = 'ValueError: Actual immutable completed v24 request/source/ownership required'
CONTROLLER_ERROR = ('RuntimeError: Stage pin_current_gnn failed (1): None; logs preserved; '
                    'no process was signaled and no downstream stage was launched')
COMMON_FILES = {'request.json', 'status.json', 'launch.lease.json', 'registration.json', 'watcher.log'}
COMMON_PROOF = {
    'format', 'mode', 'previous_chain_root', 'chain_root', 'source_output', 'source_claim_sha256',
    'previous_files_sha256', 'previous_supervisor_pid', 'previous_supervisor_create_time',
    'previous_supervisor_command', 'old_waiter_closed', 'old_waiter_children',
    'old_waiter_cuda_visible_devices', 'original_GNN_processes_signaled', 'previous_results_written',
    'evaluation_checkpoint', 'full_training_preserved', 'matched_historical_Basic_sources',
    'matched_historical_Basic_candidates_per_source', 'source_payloads_written',
}
CHANGED_REQUEST_FIELDS = {
    'chain_root', 'code', 'commit', 'code_files', 'stages', 'storage_admission',
    'storage_admission_sha256', 'storage_admission_stat', 'min_free_disk_GiB', 'matched_cli_handoff',
}


def _positive_identity(pid, birth):
    return (type(pid) is int and pid > 0 and type(birth) in (int, float)
            and math.isfinite(birth) and birth > 0)


def _finished(chain, pid, birth, command, role):
    witness = chain.process_witness(pid, birth, command)
    if witness is not None and witness['status'] != 'zombie':
        raise RuntimeError('Previous exact own ' + role + ' remains live; CLI repair refused')


def validate(request):
    """Return immutable proof/checksum after recursively proving every old lease."""
    # Deferred import prevents a module cycle and keeps the waiting path CPU-only.
    from tools import watch_v24_gnn_to_nnunet as chain

    handoff = request.get('matched_cli_handoff')
    if (request.get('GPU_arm') not in (4, 5, 6)
            or request.get('storage_profile') != chain.MATCHED_BASIC_PROFILE
            or request.get('cp_parity_handoff') is None or request.get('recovery') is not None
            or not isinstance(handoff, dict) or set(handoff) != {'proof', 'proof_sha256', 'lease'}
            or HELPER not in request.get('code_files', {})):
        raise ValueError('Explicit SHA-pinned own GPU4/5/6 matched first-pin CLI handoff required')
    claim = chain.owned(request['source_claim_path'])
    proof_path = chain.owned(handoff['proof'])
    lease = claim.with_name(claim.stem + '.matched_cli_handoff_lease.json')
    if (not chain._checksum(handoff['proof_sha256']) or chain.sha(proof_path) != handoff['proof_sha256']
            or proof_path.parent != claim.parent
            or not proof_path.name.startswith(claim.stem + '.matched_cli_handoff_proof_')
            or proof_path.suffix != '.json' or handoff['lease'] != str(lease)):
        raise ValueError('First-pin CLI repair requires one SHA proof and globally derived exclusive lease')
    chain.owned(lease, must_exist=False)
    proof = chain.read(proof_path)
    mode = proof.get('mode')
    failed = mode == 'failed_first_pin'
    expected_fields = COMMON_PROOF | ({'previous_child_pid', 'previous_child_create_time',
                                      'previous_child_command'} if failed else set())
    expected_files = COMMON_FILES | ({'pin_current_gnn.log'} if failed else set())
    previous_root = chain.owned(proof['previous_chain_root'], directory=True)
    root = Path(request['chain_root'])
    if (set(proof) != expected_fields or mode not in ('closed_waiter', 'failed_first_pin')
            or proof.get('format') != FORMAT or proof.get('chain_root') != str(root)
            or proof.get('source_output') != request['source_output']
            or root == previous_root or root in previous_root.parents or previous_root in root.parents
            or set(proof.get('previous_files_sha256', {})) != expected_files
            or not chain._checksum(proof.get('source_claim_sha256'))
            or proof.get('old_waiter_closed') is not True or proof.get('old_waiter_children') != []
            or proof.get('old_waiter_cuda_visible_devices') != ''
            or proof.get('original_GNN_processes_signaled') is not False
            or proof.get('previous_results_written') is not False
            or proof.get('evaluation_checkpoint') != 'checkpoint_best.pth'
            or proof.get('full_training_preserved') is not True
            or type(proof.get('matched_historical_Basic_sources')) is not int
            or proof['matched_historical_Basic_sources'] != 642
            or type(proof.get('matched_historical_Basic_candidates_per_source')) is not int
            or proof['matched_historical_Basic_candidates_per_source'] != 128
            or proof.get('source_payloads_written') is not False):
        raise ValueError('Exact stopped first-pin CLI repair proof/full642/BEST contract required')
    for name, checksum in proof['previous_files_sha256'].items():
        if not chain._checksum(checksum) or chain.sha(previous_root / name) != checksum:
            raise ValueError('Previous matched controller artifact changed: ' + name)
    previous = chain.read(previous_root / 'request.json')
    if (previous.get('chain_root') != str(previous_root)
            or previous.get('matched_cli_handoff') is not None
            or previous.get('storage_profile') != chain.MATCHED_BASIC_PROFILE
            or previous.get('cp_parity_handoff') is None or previous.get('recovery') is not None):
        raise ValueError('CLI repair requires original matched controller; nested repair/replay refused')
    original = {name: value for name, value in previous.items() if name not in CHANGED_REQUEST_FIELDS}
    new = {name: value for name, value in request.items() if name not in CHANGED_REQUEST_FIELDS}
    if (original != new or previous.get('code') == request.get('code')
            or previous.get('commit') == request.get('commit')):
        raise ValueError('CLI repair changed original science/assets/PBS/old lease lineage or replayed its code')
    chain.validate_request(previous, fresh=False)
    chain.verify_code(previous)
    checksum = proof['previous_files_sha256']['request.json']
    if chain.verify_source_claim(previous, checksum) != proof['source_claim_sha256']:
        raise ValueError('Original matched source claim checksum changed during CLI repair')
    status, registration, old_lease = (chain.read(previous_root / name) for name in
                                      ('status.json', 'registration.json', 'launch.lease.json'))
    pid, birth, command = (proof.get('previous_supervisor_' + name)
                          for name in ('pid', 'create_time', 'command'))
    expected_command = [previous['python'], '-B', '-u',
                        str(Path(previous['code']) / 'tools/watch_v24_gnn_to_nnunet.py'),
                        '--request', str(previous_root / 'request.json')]
    if (not _positive_identity(pid, birth) or command != expected_command
            or registration.get('GPU_arm') != request['GPU_arm']
            or registration.get('root') != str(previous_root)
            or registration.get('pid') != pid or registration.get('create_time') != birth
            or registration.get('command') != command or registration.get('request_sha256') != checksum
            or registration.get('CPU_affinity') != request['CPU_affinity']
            or registration.get('CUDA_visible_devices_while_waiting') != ''
            or status.get('GPU_arm') != request['GPU_arm'] or status.get('request_sha256') != checksum
            or status.get('source_job') != request['source_job']
            or status.get('supervisor_pid') != pid or status.get('supervisor_create_time') != birth
            or status.get('actual_CPU_affinity') != request['CPU_affinity']
            or status.get('source_claim_path') != request['source_claim_path']
            or status.get('source_claim_sha256') != proof['source_claim_sha256']
            or status.get('signals_sent') is not False or status.get('stages_completed') != []
            or status.get('stage_proofs') != []
            or old_lease.get('supervisor_pid') != pid or old_lease.get('request_sha256') != checksum):
        raise ValueError('Previous matched controller is not its exact registered own supervisor/source claim')
    _finished(chain, pid, birth, command, 'matched controller')
    if failed:
        child_pid, child_birth, child_command = (proof.get('previous_child_' + name)
                                                for name in ('pid', 'create_time', 'command'))
        log = previous_root / 'pin_current_gnn.log'
        if (status.get('status') != 'FAILED' or status.get('stage') != 'pin_current_gnn'
                or status.get('error') != CONTROLLER_ERROR
                or status.get('command') != previous['stages'][0]['command']
                or not _positive_identity(child_pid, child_birth)
                or status.get('child_pid') != child_pid or status.get('child_create_time') != child_birth
                or child_command != previous['stages'][0]['command']
                or status.get('stage_log') != str(log)):
            raise ValueError('Only the exact first-pin CLI failure before any stage proof may be recovered')
        _finished(chain, child_pid, child_birth, child_command, 'first-pin child')
        text = chain.owned(log).read_text(encoding='utf8')
        if ('Traceback (most recent call last):' not in text
                or not text.rstrip().endswith(PIN_ERROR)
                or str(Path(previous['code']) / 'hiercp_v1x/v24_nnunet_cp.py') not in text):
            raise ValueError('Exact preserved original immutable-request first-pin CLI traceback required')
        completion = chain.inspect_source(previous)
        if completion is None or status.get('gnn_completion') != completion:
            raise ValueError('Original full40/680 completed own GNN receipt must equal failed controller witness')
    elif (status.get('status') != 'WAITING_FOR_FULL_GNN' or status.get('stage') is not None
            or status.get('child_pid') is not None or status.get('child_create_time') is not None
            or status.get('gnn_completion') is not None):
        raise ValueError('Closed matched waiter must have never launched a GPU/native child')
    for path in previous_root.iterdir():
        if path.name not in expected_files:
            raise ValueError('Previous matched controller contains a pin/bank/native/stage artifact; no replay')
        chain.owned(path)
    return proof, checksum


def verify_source_claim(request, request_checksum):
    """Bind the sole new lease to this exact new request and its preserved source."""
    from tools import watch_v24_gnn_to_nnunet as chain
    proof, _ = validate(request)
    handoff = request['matched_cli_handoff']
    expected = dict(format=LEASE, GPU_arm=request['GPU_arm'], source_output=request['source_output'],
                    source_claim_path=request['source_claim_path'], source_claim_sha256=proof['source_claim_sha256'],
                    previous_chain_root=proof['previous_chain_root'], chain_root=request['chain_root'],
                    request_sha256=request_checksum, proof=handoff['proof'], proof_sha256=handoff['proof_sha256'])
    if chain.read(handoff['lease']) != expected:
        raise ValueError('Exclusive matched first-pin CLI lease belongs to another request/root; no replay')
    return proof['source_claim_sha256']
