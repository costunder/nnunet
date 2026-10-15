"""Hold approved waiting arms until a fresh cohort transfers their GPU slot.

This CPU-only gate never dispatches work, edits a registry, or signals a process.
An upstream COMPLETE state is evidence for the separate transfer coordinator;
it is deliberately not permission to start on the original waiting GPU.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import re
import time

FILE = 'hiercp_v1x/basic_cp_GPU_queue_dependencies.py'
FIELD = 'GPU_queue_deferred_transfer'
FORMAT = 'basic_cp_approved_waiting_GPU_dependencies_v1'
QUEUES = [
    dict(arm='multi5', waiting_physical_GPU=0, target_physical_GPU=5,
         upstream_arm='candidates2048'),
    dict(arm='candidates1024', waiting_physical_GPU=4, target_physical_GPU=6,
         upstream_arm='multi7'),
]


def receipt_value():
    return dict(format=FORMAT, queues=copy.deepcopy(QUEUES))


def deferred_rule(request, admit):
    """Admit only the human-approved two dependencies, bound to the fresh group."""
    if FIELD not in request:
        return None
    reference = request[FIELD]
    if (not isinstance(reference, dict) or set(reference) != {'path', 'sha256'}
            or not isinstance(reference.get('path'), str)
            or not Path(reference['path']).is_absolute()
            or '..' in Path(reference['path']).parts
            or Path(reference['path']).parent != Path(request['root']).parent
            or not isinstance(reference.get('sha256'), str)
            or re.fullmatch('[0-9a-f]{64}', reference['sha256']) is None
            or not isinstance(request.get('code_sha256', {}).get(FILE), str)
            or re.fullmatch('[0-9a-f]{64}', request['code_sha256'][FILE]) is None):
        raise ValueError('Deferred GPU queue requires its exact same-group receipt and code SHA')
    admit(reference['path'], reference['sha256'])
    if json.loads(Path(reference['path']).read_bytes()) != receipt_value():
        raise ValueError('Only the two explicitly approved waiting GPU dependencies are allowed')
    matches = [rule for rule in QUEUES if rule['arm'] == request.get('arm')
               and type(request.get('gpu')) is int
               and rule['waiting_physical_GPU'] == request['gpu']]
    if len(matches) != 1:
        raise ValueError('A deferred dependency may only hold its original incomplete arm/GPU')
    return copy.deepcopy(matches[0])


def reject_native_dispatch(request):
    """No direct native CLI or model Popen may bypass a pending transfer gate."""
    if FIELD in request:
        raise RuntimeError('Deferred GPU queue must transfer through a fresh cohort before native execution')


def upstream_complete(state):
    return (state.get('status') == 'COMPLETE' and state.get('stage') == 'complete'
        and type(state.get('completed_epochs')) is int and state['completed_epochs'] == 250
        and type(state.get('evaluated_cases')) is int and state['evaluated_cases'] == 26
        and state.get('full_native_training') is True
        and state.get('full_native_evaluation') is True
        and state.get('historical_metrics_complete') is True
        and isinstance(state.get('reports'), dict) and bool(state['reports'])
        and isinstance(state.get('archive'), dict) and bool(state['archive']))


def wait_for_transfer(request, lease, publish_status, *, completed_epochs, sleep=time.sleep):
    """Remain CPU-only even after upstream finishes; only a new request releases it."""
    if FIELD not in request:
        return
    rule = deferred_rule(request, lease._admit)
    if type(completed_epochs) is not int or not 0 < completed_epochs < 250:
        raise ValueError('Deferred continuation requires an actual incomplete completed checkpoint epoch')
    if lease.self_identity != request['controller_identity']:
        raise RuntimeError('Only the exact waiting controller may observe its GPU transfer queue')
    view = getattr(lease, '_group_view', lease)
    upstream = view.registered_requests[rule['target_physical_GPU']]
    if upstream.get('arm') != rule['upstream_arm']:
        raise ValueError('Pending GPU dependency must reference its actual current upstream arm')
    original_reference = copy.deepcopy(request[FIELD])
    while True:
        if request.get(FIELD) != original_reference:
            raise RuntimeError('Deferred request changed in memory; only fresh-cohort transfer may release it')
        lease.observe_waiting()
        state_path = Path(upstream['root'])/'status.json'
        state = json.loads(state_path.read_bytes()) if state_path.exists() else {}
        publish_status(request, 'waiting_GPU_queue_transfer', 'WAITING',
            completed_epochs=completed_epochs,
            reason='Approved successor waits for upstream full250/BEST26/results and fresh-cohort transfer',
            queued_physical_GPU=rule['target_physical_GPU'],
            upstream_arm=rule['upstream_arm'], upstream_root=upstream['root'],
            upstream_status=state.get('status'), upstream_stage=state.get('stage'),
            upstream_reported_complete=upstream_complete(state),
            original_GPU_autostart_disabled=True, CUDA_dispatch_started=False)
        sleep(5)
