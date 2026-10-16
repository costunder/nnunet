"""Run six exact continuations from an actually owned same-parent PBS job.

This entry point changes no scheduler/driver/session state. It admits the
unmodified scheduler mask, prepares exact checkpoint bindings, then invokes
the canonical runner. A mixed-parent allocation fails before model loading.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import psutil
from hiercp_v1x.basic_cp_single_parent_resources import (
    ARMS, FORMAT, GIB, UID, _memory_cgroup, digest, identity, nvml_devices,
    pin, require, validate_allocation, validate_capacity,
)
from tools.run_basic_cp_single_parent import read, seal, publish, continuation_arms

INPUT_FORMAT = 'basic_cp_single_parent_staged_inputs_v1'


def duration(value):
    parts = str(value).split(':')
    require(len(parts) == 3 and all(x.isdigit() for x in parts),
            'Actual PBS HH:MM:SS duration required')
    hours, minutes, seconds = map(int, parts)
    require(minutes < 60 and seconds < 60, 'Malformed PBS duration')
    return hours * 3600 + minutes * 60 + seconds


def lifetime(record, raw, query_started):
    wall = duration(record['Resource_List']['walltime'])
    elapsed = duration(record['resources_used']['walltime'])
    require(0 <= elapsed < wall - 180, 'Actual allocation has insufficient lifetime')
    return dict(node_boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                query_started_monotonic=query_started, walltime_seconds=wall,
                elapsed_seconds=elapsed, remaining_seconds=wall - elapsed,
                raw_PBS_response_sha256=hashlib.sha256(raw.encode()).hexdigest())


def actual_job(job_id):
    warnings = []
    for attempt in range(3):
        started = time.monotonic()
        try:
            raw = subprocess.check_output(
                ['/opt/pbs/bin/qstat', '-f', '-F', 'json', job_id],
                text=True, timeout=22)
        except subprocess.TimeoutExpired:
            warnings.append(dict(attempt=attempt + 1, error='TimeoutExpired'))
            continue
        document = json.loads(raw)
        require(set(document['Jobs']) == {job_id}, 'Exactly the current PBS job required')
        record = dict(document['Jobs'][job_id], job_id=job_id)
        require(record.get('queue') == 'ssai_agpu', 'Actual agpu16 owning queue must be ssai_agpu')
        return record, raw, lifetime(record, raw, started), warnings
    raise RuntimeError('Initial actual PBS allocation proof unavailable: ' + json.dumps(warnings))


def parent_proof(parent_UUID):
    """NVML observes physical capacity; this does not initialize a CUDA context."""
    lib = ctypes.CDLL('libnvidia-ml.so.1')
    require(lib.nvmlInit_v2() == 0, 'NVML initialization failed')
    handle = ctypes.c_void_p()
    require(lib.nvmlDeviceGetHandleByUUID(parent_UUID.encode(), ctypes.byref(handle)) == 0,
            'Actual parent NVML handle unavailable')
    name = ctypes.create_string_buffer(128)
    class Memory(ctypes.Structure):
        _fields_ = [('total', ctypes.c_ulonglong), ('free', ctypes.c_ulonglong), ('used', ctypes.c_ulonglong)]
    memory = Memory()
    require(lib.nvmlDeviceGetName(handle, name, len(name)) == 0
            and lib.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(memory)) == 0,
            'Actual parent model/capacity query failed')
    model = name.value.decode()
    require('A100' in model and '80GB' in model and 77 * GIB <= memory.total <= 81 * GIB,
            'Actual same physical A100 80GB parent required')
    return dict(UUID=parent_UUID, model=model, total_bytes=memory.total)


def admit_inputs(value):
    arms=continuation_arms(value)
    require(value.get('format') == INPUT_FORMAT,
            'Staged unchanged production inputs required; completed experiments must stay retained')
    require(value.get('canonical_code') == str(ROOT), 'One canonical code checkout required')
    for key in ('source_snapshot', 'bindings', 'VRAM_measurement_evidence'):
        pin(value[key])
    require(set(read(pin(value['bindings']))) == set(arms), 'Exact unfinished input bindings required')
    require(Path(value['output_root']) == Path('/dev/shm/aicompetition06_basic_cp/experiments'),
            'Use the fixed experiment output root; no GPU-specific code trees')
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--inputs-sha256', required=True)
    args = parser.parse_args()
    require(os.getuid() == UID and socket.gethostname().split('.')[0] == 'ece-agpu16',
            'Actual intended account and execution node required')
    inputs = admit_inputs(read(pin(dict(path=str(args.inputs), sha256=args.inputs_sha256), owner=True)))
    require(continuation_arms(inputs)==ARMS,'PBS six-slot bootstrap requires all six unfinished arms; use direct entry for partial resume')
    job_id = os.environ.get('PBS_JOBID', '')
    require(re.fullmatch(r'\d+\.ECE-util1', job_id), 'Actual current PBS job identity required')
    control = Path('/dev/shm/aicompetition06_basic_cp/control')
    control.mkdir(mode=0o700, parents=True, exist_ok=True)
    require(control.resolve() == control and control.stat().st_uid == UID, 'Own fixed control directory required')
    outcome = control / ('bootstrap_' + job_id + '.json')
    require(not outcome.exists(), 'Previous bootstrap outcome requires inspection; no duplicate dispatch')
    try:
        record, raw, clock, warnings = actual_job(job_id)
        actor = psutil.Process()
        memory = _memory_cgroup(actor.pid)
        cpus = actor.cpu_affinity()
        require(len(cpus) == 48 and job_id in memory['path'], 'Actual 48-core own allocation cgroup required')
        require(475 * GIB <= memory['limit'] <= 476 * GIB, 'Exact 476GiB job RAM ceiling required')
        tokens = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
        require(len(tokens) == len(set(tokens)) == 6 and all(x.startswith('MIG-') for x in tokens),
                'Six unmodified actual PBS-assigned MIG UUIDs required')
        devices = nvml_devices(tokens)
        parents = {row['parent_UUID'] for row in devices.values()}
        require(len(parents) == 1, 'PBS assigned different physical parents; no model will be loaded')
        parent = parents.pop()
        physical = parent_proof(parent)
        require(all(not row['compute_actors'] for row in devices.values()),
                'Allocated execution devices have an unknown existing CUDA actor')
        debug = read(pin(inputs['VRAM_measurement_evidence']))
        debug = debug.get('DEBUG_result.json', debug)
        require(debug.get('accepted') is True and debug.get('physical_batch') == 2
                and debug.get('parameters') == 102350575 and debug.get('patch_size') == [128] * 3,
                'Accepted actual full-model B2 CUDA measurement required')
        peak = max(row['peak_reserved_bytes'] for row in debug['measurements'])
        allocation = dict(format=FORMAT, UID=UID, owner='aicompetition06', node='ece-agpu16',
            mode='six_MIG_same_parent', job_id=job_id, holder_identity=identity(actor), CPU_affinity=cpus,
            physical_parent_GPU_UUID=parent, physical_parent_index=next(iter(devices.values()))['index'],
            execution_UUID_by_arm=dict(zip(ARMS, tokens)),
            VRAM_reservation_bytes_by_arm={arm: peak for arm in ARMS}, VRAM_floor_bytes=256 * 2**20,
            VRAM_measurement_evidence=inputs['VRAM_measurement_evidence'], PBS_lifetime=clock,
            expires_at_unix=time.time() + clock['remaining_seconds'],
            actual_PBS_response_sha256=hashlib.sha256(raw.encode()).hexdigest(),
            physical_parent_observation=physical, execution_device_observations=devices,
            PBS_query_warnings=warnings)
        validate_allocation(allocation, record,
            dict(identity=identity(actor), environment=actor.environ(), CPU_affinity=cpus,
                 memory_limit_bytes=memory['limit']), devices,
            now_monotonic=time.monotonic(), node_boot_id=clock['node_boot_id'],
            query_started_monotonic=clock['query_started_monotonic'])
        validate_capacity(allocation, devices, native_RSS={arm: 0 for arm in ARMS},
                          auxiliary_RSS=actor.memory_info().rss, cgroup_usage_bytes=memory['usage'])
        allocation_path = control / ('allocation_' + job_id + '.json')
        seal(allocation_path, allocation)
        seal(control / ('actual_PBS_' + job_id + '.json'), dict(raw=raw, record=record, lifetime=clock))
        command = [sys.executable, '-B', '-u', str(ROOT / 'tools/prepare_basic_cp_single_parent.py')]
        for name in ('source_snapshot', 'bindings'):
            flag = name.replace('_', '-')
            command += ['--' + flag, inputs[name]['path'], '--' + flag + '-sha256', inputs[name]['sha256']]
        command += ['--allocation', str(allocation_path), '--allocation-sha256', digest(allocation_path),
            '--root', inputs['output_root'],
            '--shared-preprocessed-dataset-directory', inputs['shared_preprocessed_dataset_directory'],
            '--evaluator-root', inputs['evaluator_root'], '--project', inputs['project']]
        subprocess.run(command, check=True, cwd=ROOT, env=dict(os.environ, PYTHONPATH=str(ROOT)))
        cohort = Path(inputs['output_root']) / 'cohort.json'
        subprocess.run([sys.executable, '-B', '-u', str(ROOT / 'tools/run_basic_cp_single_parent.py'),
            '--cohort', str(cohort), '--cohort-sha256', digest(cohort), '--execute'],
            check=True, cwd=ROOT, env=dict(os.environ, PYTHONPATH=str(ROOT)))
        seal(outcome, dict(state='ALL_SIX_PIPELINES_COMPLETE', job_id=job_id, retained_complete_arms=['blend50'],
                           observed_unix=time.time()))
    except BaseException as error:
        publish(outcome, dict(state='FAILED_NO_FOREIGN_PROCESS_SIGNALED', job_id=job_id,
                              error=type(error).__name__ + ': ' + str(error), observed_unix=time.time()))
        raise


if __name__ == '__main__':
    main()
