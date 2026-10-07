"""Actual CT/CUDA DEBUG: exact completed-arm resume through the new launcher.

This is not new training, a four-GPU server run, or a speed/quality benchmark.
Only an isolated copy of existing actual-CT DEBUG state is written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2), b''):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def verify(a):
    reference = a.reference.resolve(strict=True)
    output = a.output.resolve()
    if output.exists() or output.is_relative_to(reference) or reference.is_relative_to(output):
        raise FileExistsError('A new disjoint DEBUG output is required; originals preserved')
    from hiercp_v1x.u_bridge_experiment import FILES, FORMAT, digest
    manifest = read(reference/'experiment.json')
    if (manifest['format'] != FORMAT or manifest['debug'] is not True or manifest['epochs'] != 2
            or manifest['workers'] != 4 or manifest['explicit_batch_candidates'] != [2]
            or manifest['sha256'] != digest({k:v for k,v in manifest.items() if k != 'sha256'})
            or manifest['helpers'] != {name:sha(ROOT/name) for name in FILES}):
        raise ValueError('The preserved actual-CT DEBUG2/batch2/worker4 fixture is required')
    original_files = [reference/'experiment.json', reference/'initial.pt', reference/'calibration.json',
                      *(reference/arm/'checkpoint_latest.pt' for arm in ('selected','native'))]
    original_sha = {str(p):sha(p) for p in original_files}
    output.mkdir(parents=True)
    historical = output/'historical'
    historical.mkdir()
    for name in ('experiment.json','initial.pt','calibration.json'):
        shutil.copy2(reference/name, historical/name)
    for path in reference.glob('calibration_*.json'):
        shutil.copy2(path, historical/path.name)
    for arm in ('selected','native'):
        shutil.copytree(reference/arm, historical/arm, ignore=shutil.ignore_patterns('STOP_AFTER_BATCH'))
    results = []
    for arm in ('selected','native','selected'):
        target = output/arm
        command = [sys.executable, '-B', '-u', str(ROOT/'tools/run_comparison_arm.py'),
                   '--owned-child', '--gpu', str(a.gpu), '--arm', arm,
                   '--experiments-dir', str(output), '--experiment', str(target),
                   '--source-experiment', str(historical), '--inventory', str(a.inventory.resolve()),
                   '--cache-sources', manifest['prepared_data_root']]
        for name in ('fixture','config','source','bank'):
            command.extend(('--debug-'+name,str(getattr(a,name).resolve())))
        number = len(results)+1
        print(f'ACTUAL CT CUDA DEBUG | invocation={number} | arm={arm} | completed-checkpoint resume', flush=True)
        with (output/f'{number}_{arm}.log').open('x',encoding='utf8') as log:
            completed = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
        if completed.returncode:
            raise RuntimeError(f'DEBUG launcher failed({completed.returncode}); preserved log: {output}/{number}_{arm}.log')
        invocation = max((target/'invocations').glob(arm+'_*.json'), key=lambda p:p.stat().st_mtime_ns)
        receipt = read(invocation)['result']
        if (receipt['resume']['invocation_optimizer_updates'] != 0
                or not receipt['resume'].get('completed_checkpoint_no_repeat_verified')):
            raise ValueError('Completed checkpoint repeated training or changed neural state')
        results.append(dict(arm=arm, invocation=number, returncode=0,
                            actual_CUDA=receipt['actual_CUDA'], additional_optimizer_updates=0,
                            exact_completed_resume=True, checkpoint=str(target/arm/'checkpoint_latest.pt'),
                            data_root=str(target/'data'), historical_source_lock_active=number==3,
                            full_parameters=10434532, debug=True))
        if number == 2:
            (historical/'.pipeline.lock').write_text(json.dumps(dict(host=socket.gethostname(),
                pid=os.getpid(), token='DEBUG_owned_historical_lock')),encoding='utf8')
    current_sha = {str(p):sha(p) for p in original_files}
    if current_sha != original_sha:
        raise RuntimeError('Actual-CT reference checkpoint/configuration changed')
    from hiercp_v1x.comparison_experiment import FILES as COMPARISON_FILES
    if {name:sha(ROOT/name) for name in FILES} != manifest['helpers']:
        raise RuntimeError('Frozen original controller or engine changed')
    report = dict(format='independent_arm_actual_CUDA_DEBUG_v1', scope='completed checkpoint resume only',
                  platform=sys.platform, original_reference_preserved=True,
                  reference_sha256=original_sha, invocations=results,
                  source_helpers={name:sha(ROOT/name) for name in COMPARISON_FILES},
                  supervisor_linux_signals_executed=False, four_GPU_server_run=False,
                  additional_training_updates=0, quality_verified=False, server_speedup_measured=False)
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8')
    print(f'DEBUG REPORT: {output}/report.json',flush=True)
    return report


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu',type=int,required=True)
    for name in ('reference','output','inventory','fixture','config','source','bank'):
        p.add_argument('--'+name,type=Path,required=True)
    verify(p.parse_args())
