"""CPU-only GPU5 waiter: actual full GNN completion -> one full642 score job.

The existing completed scorer performs numerical full40/680 own-BEST admission
before CUDA. This waiter never reads a model, initializes CUDA or rescales data.
"""
from __future__ import annotations
import argparse
import copy
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
from hiercp_v1x import v24_scored_native_chain as chain
from tools import run_v24_completed_matched_scoring as scoring

FORMAT='v24_GNN_to_full642_scoring_wait_request_v1'
STATUS_FORMAT='v24_GNN_to_full642_scoring_wait_status_v1'


def build_request(native_request_path, *, host_container_lifetime, GPU_UUID):
    """Read-only installer helper; preserves the newly sealed native request."""
    path=chain.owned(native_request_path);native=chain.validate_declaration(chain.read(path))
    if native['GPU_arm']!=5 or native['GPU_UUID']!=GPU_UUID:
        raise ValueError('This CPU waiter launches only the original GPU5 CNN/GAT scoring')
    root=Path(native['scoring_root'])
    template=dict(format=scoring.FORMAT,gpu=5,debug=False,full642_scoring=True,
        source_entries=642,CP_positions=82176,physical_patient_batch=4,physical_candidate_chunk=32,
        CPU_affinity=chain.AFFINITIES[5],source_output=native['source_output'],source_code=native['source_code'],
        inventory_path=native['inventory'],input_cache=native['input_cache'],baseline_preprocessed=native['baseline_preprocessed'],
        source_bank=native['historical_basic_bank'],stunet_checkpoint=None,pin_path=str(root/'own_best_pin.json'),output=str(root/'bank'),
        code=native['code'],commit=native['commit'],code_files=native['code_files'],worker_sha256=native['scoring_worker_sha256'],
        direct_guard_module=str(root/'direct_GPU45_access_guard.py'),direct_guard_sha256=scoring.GUARD_SHA256,
        direct_access_proof=str(root/'direct_GPU_access_proof.json'),direct_access_binding=str(root/'direct_access_binding.json'),
        native_pending_reason='후속 nnU-Net controller가 전체 CP 완료 및 실제 storage/GPU admission을 기다립니다.',
        nnunet_evaluation_checkpoint='checkpoint_best.pth')
    return dict(format=FORMAT,root=str(root),GPU_arm=5,CPU_affinity=chain.AFFINITIES[5],GPU_UUID=GPU_UUID,
        native_request=chain.FileProofCache().capture(path),scoring_template=template,
        host_container_lifetime=copy.deepcopy(host_container_lifetime))


def prepare_scoring(wait_request, native):
    """After source exit/full completion, publish only new sealed score inputs."""
    root=Path(wait_request['root']);template=copy.deepcopy(wait_request['scoring_template'])
    if root!=Path(native['scoring_root']) or template['source_output']!=native['source_output']:
        raise ValueError('Exact native/source/scoring destination binding required')
    source=Path(native['source_output'])
    paths=[source/name for name in ('request.json','calibration.json','training/training_identity.json',
        'training/invocations.jsonl','training/checkpoint_latest.pt','training/checkpoint_best.pt')]
    paths.extend([Path(native['historical_basic_bank'])/'index.json',source.parent/'direct_training_request.json',
        source.parent/'direct_training_registration.json'])
    template['preserved_source_hashes']={str(path):chain.sha(chain.owned(path)) for path in paths}
    path=root/'request.json'
    scoring.validate_request(template,path,Path(native['scoring_worker']))
    if chain.sha(chain.owned(template['direct_guard_module']))!=scoring.GUARD_SHA256:
        raise ValueError('Exact original v1 scoring guard bytes required')
    chain.publish(path,template)
    proof=root/'direct_GPU_access_proof.json'
    chain.publish(proof,dict(format=scoring.GUARD_FORMAT,GPU_arm=5,GPU_UUID=wait_request['GPU_UUID'],
        CPU_affinity=chain.AFFINITIES[5],UID=41833,PBS_owned_reservation=False,
        other_PBS_reservations_known_not_owned=True,explicit_existing_direct_execution_requested=True,
        guard_sha256=scoring.GUARD_SHA256,request=dict(path=str(path),sha256=chain.sha(path)),
        host_container_lifetime=wait_request['host_container_lifetime'],poll_seconds=30,
        minimum_initial_remaining_seconds=28800,expiry_safety_reserve_seconds=180))
    capture=chain.FileProofCache().capture
    chain.publish(root/'direct_access_binding.json',dict(format=scoring.BINDING_FORMAT,GPU_arm=5,UID=41833,
        CPU_affinity=chain.AFFINITIES[5],request=capture(path),proof=capture(proof),guard=capture(template['direct_guard_module'])))
    scoring.verify_binding(template,path)
    return template,path


def run(request_path):
    import psutil
    path=chain.owned(request_path);wait=chain.read(path);root=chain.owned(path.parent,directory=True)
    if (wait.get('format')!=FORMAT or wait.get('root')!=str(root) or wait.get('GPU_arm')!=5
            or wait.get('CPU_affinity')!=chain.AFFINITIES[5] or psutil.Process().cpu_affinity()!=chain.AFFINITIES[5]):
        raise ValueError('Exact own GPU5 CPU waiter/root/assignment required')
    cache=chain.FileProofCache();cache.capture(path)
    native_path=cache.verify(wait['native_request']);native=chain.validate_declaration(chain.read(native_path))
    if (native['GPU_arm']!=5 or native['scoring_root']!=str(root) or wait['GPU_UUID']!=native['GPU_UUID']
            or Path(native['code'])!=ROOT):
        raise ValueError('Pinned same-checkout GPU5 native/scoring request required')
    chain.verify_code(native,cache)
    state=dict(format=STATUS_FORMAT,GPU_arm=5,request_sha256=chain.sha(path),status='WAITING_FOR_FULL_GNN',
        child_pid=None,signals_sent=False,full_native_training=False,full_native_evaluation=False)
    def update(**values):
        state.update(values,checked_at=time.time());chain.publish(root/'scoring_wait_status.json',state,replace=True)
    try:
        pending=None
        while True:
            cache.check()
            try:
                complete=chain.source_complete(native)
            except chain.PendingMetadata as error:
                now=time.monotonic()
                if pending is None or pending[0]!=error.signature:pending=(error.signature,now)
                if now-pending[1]>=10:raise ValueError('Stable source metadata corruption: '+str(error)) from error
                update(status='WAITING_METADATA_PUBLICATION',reason=str(error));time.sleep(5);continue
            pending=None
            if complete is not None:break
            update(status='WAITING_FOR_FULL_GNN');time.sleep(5)
        template,score_path=prepare_scoring(wait,native)
        spec=importlib.util.spec_from_file_location('_v24_GPU5_completed_scoring_direct_guard',template['direct_guard_module'])
        guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)
        while True:
            cache.check()
            try:
                guard.make_guard(template['direct_access_proof'],5,chain.AFFINITIES[5])
                break
            except guard.DirectGPUAccessUnavailable as error:
                update(status='WAITING_FOR_ASSIGNED_GPU_OR_HOST_LIFETIME',reason=str(error));time.sleep(5)
        command=[native['python'],'-B','-u',native['scoring_worker'],'--request',str(score_path)]
        chain.publish(root/'scoring_launch.lease.json',dict(command=command,request_sha256=chain.sha(score_path),created_at=time.time()))
        with (root/'console.log').open('xb',buffering=0) as log:
            child=subprocess.Popen(command,cwd=native['code'],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                env=dict(os.environ,CUDA_VISIBLE_DEVICES='',CUDA_DEVICE_ORDER='PCI_BUS_ID',PYTHONDONTWRITEBYTECODE='1',V24_ARM_RSS_GIB='64'))
            actor=psutil.Process(child.pid);birth=actor.create_time();chain.process_snapshot(child.pid,birth,command)
            chain.publish(root/'registration.json',dict(format='v24_full642_depth1_current_BEST_scoring_registration_v1',
                pid=child.pid,create_time=birth,command=command,root=str(root),code=native['code'],commit=native['commit'],
                worker_sha256=template['worker_sha256'],request_sha256=chain.sha(score_path),full642_scoring=True,
                sources=642,CP_positions=82176,physical_patient_batch=4,physical_candidate_chunk=32,
                old_results_written=False,signals_sent=False,registered_at=time.time()))
            update(status='FULL642_SCORING_RUNNING',child_pid=child.pid,child_create_time=birth,command=command)
            result=child.wait()
        if result:raise RuntimeError('Actual full642 scorer exited '+str(result)+'; logs preserved, no signal sent')
        if chain.scoring_complete(native,cache) is None:raise RuntimeError('Scorer returned without actual full642 completion proof')
        update(status='FULL642_SCORING_COMPLETE',child_pid=None,completed=time.time())
    except Exception as error:
        update(status='FAILED',error=type(error).__name__+': '+str(error));raise


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--request',type=Path,required=True)
    run(parser.parse_args(argv).request)


if __name__=='__main__':main()
