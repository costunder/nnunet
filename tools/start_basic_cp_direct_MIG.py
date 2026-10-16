"""Resume unfinished full experiments on explicit idle MIG UUIDs, preserving SSH.

This is the previously authorized direct-device execution mode. PBS provides
only a separately labelled host lifetime. No queue, session, driver, cgroup,
device permission or foreign process is changed by this entry point.
"""
from __future__ import annotations
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import psutil
from hiercp_v1x.basic_cp_single_parent_resources import (
    ARMS,FORMAT,GIB,UID,_memory_cgroup,digest,identity,pin,require,
    nvml_devices,validate_allocation,validate_capacity,cgroup_identity,observe_holder,
)
from tools.run_basic_cp_single_parent import read,seal,publish,continuation_arms
from tools.start_basic_cp_single_parent_pbs import admit_inputs,actual_job,parent_proof


def parent_UUID(index):
    lib=ctypes.CDLL('libnvidia-ml.so.1');require(lib.nvmlInit_v2()==0,'NVML initialization failed')
    handle=ctypes.c_void_p();buffer=ctypes.create_string_buffer(128)
    require(lib.nvmlDeviceGetHandleByIndex_v2(index,ctypes.byref(handle))==0
        and lib.nvmlDeviceGetUUID(handle,buffer,len(buffer))==0,'Actual physical parent unavailable')
    return buffer.value.decode()


def probe_execution_device(token):
    """Real CUDA context/identity DEBUG; no kernel, model or production output."""
    code="""import ctypes,json,uuid
lib=ctypes.CDLL('libcuda.so.1');dev=ctypes.c_int();value=(ctypes.c_ubyte*16)();mem=ctypes.c_size_t();ctx=ctypes.c_void_p()
def check(code,stage):
 if code:raise RuntimeError(stage+' CUDA returncode='+str(code))
check(lib.cuInit(0),'cuInit');check(lib.cuDeviceGet(ctypes.byref(dev),0),'cuDeviceGet')
check(lib.cuDeviceGetUuid_v2(ctypes.byref(value),dev),'cuDeviceGetUuid_v2')
check(lib.cuDeviceTotalMem_v2(ctypes.byref(mem),dev),'cuDeviceTotalMem_v2')
check(lib.cuDevicePrimaryCtxRetain(ctypes.byref(ctx),dev),'cuDevicePrimaryCtxRetain')
check(lib.cuDevicePrimaryCtxRelease_v2(dev),'cuDevicePrimaryCtxRelease_v2')
print(json.dumps(dict(DEBUG=True,UUID='MIG-'+str(uuid.UUID(bytes=bytes(value))),total_bytes=mem.value,context_access=True,kernels_started=False,production_training_started=False)))
"""
    result=subprocess.run([sys.executable,'-B','-c',code],capture_output=True,text=True,timeout=30,
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=token,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1'))
    require(result.returncode==0,'Direct selected-device access failed without changing permissions: '+result.stderr[-3000:])
    proof=json.loads(result.stdout);require(proof['UUID']==token and proof['context_access']is True,
        'Actual direct CUDA device differs from the selected MIG UUID')
    return proof


def prepare_and_run(inputs,allocation_path):
    command=[sys.executable,'-B','-u',str(ROOT/'tools/prepare_basic_cp_single_parent.py')]
    for name in('source_snapshot','bindings'):
        flag=name.replace('_','-')
        command+=['--'+flag,inputs[name]['path'],'--'+flag+'-sha256',inputs[name]['sha256']]
    command+=['--allocation',str(allocation_path),'--allocation-sha256',digest(allocation_path),
        '--continuation-inputs',inputs['_inputs_reference']['path'],
        '--continuation-inputs-sha256',inputs['_inputs_reference']['sha256'],
        '--root',inputs['output_root'],'--shared-preprocessed-dataset-directory',inputs['shared_preprocessed_dataset_directory'],
        '--evaluator-root',inputs['evaluator_root'],'--project',inputs['project']]
    subprocess.run(command,check=True,cwd=ROOT,env=dict(os.environ,PYTHONPATH=str(ROOT)))
    cohort=Path(inputs['output_root'])/'cohort.json'
    subprocess.run([sys.executable,'-B','-u',str(ROOT/'tools/run_basic_cp_single_parent.py'),
        '--cohort',str(cohort),'--cohort-sha256',digest(cohort),'--execute'],check=True,cwd=ROOT,
        env=dict(os.environ,PYTHONPATH=str(ROOT)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',type=Path,required=True);parser.add_argument('--inputs-sha256',required=True)
    parser.add_argument('--host-lifetime-proof',type=Path,required=True);parser.add_argument('--host-lifetime-proof-sha256',required=True)
    parser.add_argument('--parent-index',type=int,required=True);parser.add_argument('--execute',action='store_true')
    args=parser.parse_args();require(args.execute,'Explicit direct production continuation required')
    require(os.getuid()==UID and socket.gethostname().split('.')[0]=='ece-agpu16','Actual intended owner/node required')
    inputs=admit_inputs(read(pin(dict(path=str(args.inputs),sha256=args.inputs_sha256),owner=True)))
    inputs=dict(inputs,_inputs_reference=dict(path=str(args.inputs),sha256=args.inputs_sha256))
    arms=continuation_arms(inputs)
    host=read(pin(dict(path=str(args.host_lifetime_proof),sha256=args.host_lifetime_proof_sha256),owner=True))
    require(host.get('format')=='basic_cp_current_host_lifetime_direct_observation_v1'
        and host.get('role')=='host_lifetime_only'and host.get('UID')==UID
        and host.get('PBS_GPU_owned_reservation')is False,
        'Exact separately labelled actual host lifetime observation required')
    host_actor=host['host_holder_identity'];job_id=host['PBS_job_id']
    from hiercp_v1x.basic_cp_single_parent_resources import exact_actor,DIRECT_MODE
    original=exact_actor(host_actor)
    require(original.environ().get('PBS_JOBID')==job_id,'Original separate host lifetime process changed')
    control=Path('/dev/shm/aicompetition06_basic_cp/control');control.mkdir(mode=0o700,parents=True,exist_ok=True)
    outcome=control/('direct_'+job_id+'_outcome.json')
    require(not outcome.exists(),'Existing direct execution outcome requires inspection; no duplicate dispatch')
    try:
        from hiercp_v1x.basic_cp_direct_MIG_access import capture_direct_resources,select_empty_MIGs
        actor=psutil.Process();resources=capture_direct_resources({a:0 for a in arms},{actor.pid},
            required_slots=len(arms),expected_arms=len(arms))
        cpus=[cpu for group in resources['selected_CPU_groups']for cpu in sorted(group)]
        require(len(cpus)==len(set(cpus))==8*len(arms),'Exactly eight available CPU cores per unfinished experiment required')
        actor.cpu_affinity(cpus)
        parent=parent_UUID(args.parent_index);physical=parent_proof(parent)
        debug=read(pin(inputs['VRAM_measurement_evidence']));debug=debug.get('DEBUG_result.json',debug)
        require(debug.get('accepted')is True and debug.get('physical_batch')==2
            and debug.get('parameters')==102350575,'Actual admitted full-model B2 evidence required')
        peak=max(row['peak_reserved_bytes']for row in debug['measurements']);floor=256*2**20
        excluded=inputs.get('protected_execution_UUIDs',[])
        require(isinstance(excluded,list)and len(set(excluded))==len(excluded)
            and all(isinstance(token,str)and token.startswith('MIG-')for token in excluded),
            'Protected execution devices must be exact distinct MIG UUIDs')
        selection=select_empty_MIGs(parent,required=len(arms),min_memory_bytes=peak+floor,exclude_UUIDs=excluded)
        tokens=[row['MIG_UUID']for row in selection['selected']]
        require(len(set(tokens))==len(arms),'One independently selected idle MIG UUID per unfinished experiment required')
        probes=[probe_execution_device(token)for token in tokens]
        devices=nvml_devices(tokens)
        require(all(not row['compute_actors']for row in devices.values()),'Selected MIG acquired an unmanaged CUDA process')
        require(all(proof['total_bytes']==devices[proof['UUID']]['total_bytes']for proof in probes),
            'Actual CUDA/NVML memory observations differ')
        record,raw,clock,warnings=actual_job(job_id)
        memory=_memory_cgroup(actor.pid)
        allocation=dict(format=FORMAT,UID=UID,owner='aicompetition06',node='ece-agpu16',mode=DIRECT_MODE,
            job_id=job_id,holder_identity=identity(actor),host_allocation_process=host_actor,
            PBS_GPU_owned_reservation=False,PBS_CPU_owned_reservation=False,PBS_job_role='host_lifetime_only',
            explicit_existing_direct_execution_requested=True,CPU_affinity=cpus,
            direct_launcher_cgroup=cgroup_identity(memory),direct_resource_observation=resources,
            physical_parent_GPU_UUID=parent,physical_parent_index=args.parent_index,
            active_arms=list(arms),execution_UUID_by_arm=dict(zip(arms,tokens)),VRAM_reservation_bytes_by_arm={arm:peak for arm in arms},
            VRAM_floor_bytes=floor,VRAM_measurement_evidence=inputs['VRAM_measurement_evidence'],PBS_lifetime=clock,
            expires_at_unix=time.time()+clock['remaining_seconds'],actual_PBS_response_sha256=hashlib.sha256(raw.encode()).hexdigest(),
            physical_parent_observation=physical,execution_device_observations=devices,
            direct_device_selection=selection,actual_direct_CUDA_identity_DEBUG=probes,PBS_query_warnings=warnings)
        # The direct launcher is the process owner; the original PBS shell is
        # independently checked for lifetime only and never reconfigured.
        _,_,observation=observe_holder(allocation)
        validate_allocation(allocation,record,observation,devices,now_monotonic=time.monotonic(),
            node_boot_id=clock['node_boot_id'],query_started_monotonic=clock['query_started_monotonic'])
        validate_capacity(allocation,devices,native_RSS={a:0 for a in arms},
            auxiliary_RSS=resources['cohort_auxiliary_RSS_bytes'],
            cgroup_usage_bytes=resources['current_uid_RSS_bytes']+resources['own_tmpfs_allocated_bytes'])
        path=control/('allocation_direct_'+job_id+'.json');seal(path,allocation)
        seal(control/('direct_access_ready_'+job_id+'.json'),dict(allocation=str(path),sha256=digest(path),
            actual_direct_CUDA_identity_DEBUG=probes,full_training_started=False,user_connection_changed=False))
        prepare_and_run(inputs,path)
        seal(outcome,dict(state='ALL_UNFINISHED_PIPELINES_COMPLETE',active_arms=list(arms),
            retained_complete_arms=inputs['retained_complete_arms'],observed_unix=time.time()))
    except BaseException as error:
        publish(outcome,dict(state='FAILED_NO_FOREIGN_PROCESS_SIGNALED',error=type(error).__name__+': '+str(error),
            observed_unix=time.time(),user_connection_changed=False))
        raise


if __name__=='__main__':main()
