"""Continue exact v24 numerical state with measured CPU execution improvements.

Scientific source bytes, model operations, sampling, dataset and batch sizes
remain unchanged. The original checkpoint-copy and strict resume contract is
used, with fresh execution receipts for the exact-hash/prefetch adapters.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.dont_write_bytecode=True
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
RECEIPT='performance_continuation.json'
INTERRUPTION_FORMAT='v24_verified_interrupted_source_v1'
_INTERRUPTION_KEYS={'format','source_output','source_code','GPU','uid','job','worker','child',
    'checkpoint','snapshots','server_host','original_job_status','original_stage','detected_ns'}


def compare_probes(baseline,optimized,checkpoint_sha256):
    if (baseline.get('variant')!='baseline' or optimized.get('variant')!='optimized'
            or any(row.get('debug') is not True or row.get('production_optimizer_updates')!=0
                   or row.get('active_U')!=128 or row.get('all_P_included') is not True
                   or row.get('physical_patient_batch')!=4 or row.get('diagnostic_batches')!=1
                   for row in (baseline,optimized))):
        raise ValueError('Actual complete full128 diagnostic proof required')
    keys=('original_training_identity_sha256','initial_model_sha256','initial_RNG_sha256',
          'output_sha256','loss','gradient_sha256','after_forward_model_sha256','after_RNG_sha256',
          'case_ids','physical_candidate_chunk','debug_numerics','ordered_CPU_input_sha256')
    if any(baseline.get(key)!=optimized.get(key) for key in keys):
        raise ValueError('Actual native output/loss/gradient/model/RNG parity differs')
    for row in (baseline,optimized):
        policy=row.get('debug_numerics',{})
        flags=policy.get('actual_forward_flags',{})
        proof=row.get('input_tensor_proof',{})
        if (policy.get('debug_deterministic_numerics') is not True
                or flags.get('deterministic_algorithms') is not True
                or flags.get('deterministic_warn_only') is not False
                or flags.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8'
                or policy.get('production_config_modified') is not False
                or policy.get('unsupported_operator_fallback') is not False
                or proof.get('complete') is not True or proof.get('every_actual_value_hashed') is not True
                or proof.get('both_sampled_views') is not True or proof.get('ordered_records')!=524
                or proof.get('observed_native_chunks')!=9 or proof.get('expected_native_chunks')!=9
                or proof.get('observed_upper_graphs')!=4 or proof.get('expected_upper_graphs')!=4
                or len(proof.get('content',{}).get('local_chunks',[]))!=9
                or len(proof.get('content',{}).get('upper_graphs',[]))!=4
                or row.get('ordered_CPU_input_sha256')!=proof.get('content_sha256')
                or not isinstance(proof.get('content_sha256'),str)
                or len(proof['content_sha256'])!=64):
            raise ValueError('Strict DEBUG kernels and complete actual CPU tensor proof required')
    if any(row['source_checkpoint']['raw_sha256']!=checkpoint_sha256 for row in (baseline,optimized)):
        raise ValueError('Native diagnostic belongs to a different exact checkpoint')
    if any(row['gradient']['missing'] or not row['gradient']['finite'] for row in (baseline,optimized)):
        raise ValueError('Native gradients must be connected and finite')
    return dict(status='ACTUAL_FULL128_OUTPUT_LOSS_GRADIENT_MODEL_RNG_EXACT_PASS',
        checkpoint_sha256=checkpoint_sha256,baseline_seconds=baseline['seconds'],
        optimized_seconds=optimized['seconds'],debug_single_batch_timing=True,
        ordered_CPU_input_sha256=baseline['ordered_CPU_input_sha256'],
        explicit_DEBUG_deterministic_kernels=True,production_numerical_policy_changed=False,
        full_training_completion_claimed=False)


def _sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*2**20),b''):value.update(block)
    return value.hexdigest()


def _interruption_context():
    import psutil
    if not hasattr(os,'getuid'):
        raise RuntimeError('Interrupted-source admission requires actual server UID evidence')
    return dict(uid=os.getuid(),hostname=socket.gethostname(),pid_exists=psutil.pid_exists,
        owner_uid=lambda path:Path(path).stat().st_uid)


def _owned_regular(path,context,*,directory=False):
    path=Path(path)
    if (not path.is_absolute() or path.is_symlink()
            or not (path.is_dir() if directory else path.is_file())
            or context['owner_uid'](path)!=context['uid']):
        raise ValueError('Exact regular server-owned interruption path required: '+str(path))
    resolved=path.resolve(strict=True)
    if resolved!=path.absolute():
        raise ValueError('Interruption proof path contains an unadmitted symlink')
    return resolved


def _checkpoint_witness(path):
    before=Path(path).stat()
    value=dict(path=str(Path(path).resolve(strict=True)),sha256=_sha(path),
        size=before.st_size,mtime_ns=before.st_mtime_ns)
    after=Path(path).stat()
    if (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)!=(
            after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns):
        raise ValueError('Interrupted source checkpoint changed during witness verification')
    return value


def validate_source_interruption(proof_path,source_output,source_code,latest,*,gpu):
    """Read-only proof for a RUNNING snapshot whose exact owned workers vanished.

    This never rewrites saved status. The unchanged original inspector/copy
    remains responsible for numerical states, full cursor/history and actual
    BEST validation. PID reuse is conservatively rejected, not signalled.
    """
    context=_interruption_context()
    if context['uid']!=41833 or context['hostname']!='ece-a6gpu6':
        raise ValueError('Exact original server and UID41833 required')
    path=_owned_regular(proof_path,context);proof_stat=path.stat();raw_sha=_sha(path)
    proof=json.loads(path.read_text())
    if set(proof)!=_INTERRUPTION_KEYS or proof['format']!=INTERRUPTION_FORMAT:
        raise ValueError('Closed interrupted-source proof schema required')
    source=_owned_regular(source_output,context,directory=True)
    code=_owned_regular(source_code,context,directory=True)
    job=_owned_regular(proof['job'],context,directory=True)
    if (proof['source_output']!=str(source) or proof['source_code']!=str(code)
            or type(proof['GPU'])is not int or proof['GPU']!=gpu or gpu not in (5,6)
            or proof['uid']!=41833 or proof['server_host']!=context['hostname']
            or proof['original_job_status']!='RUNNING' or proof['original_stage']!='train'
            or latest.get('status')!='RUNNING'):
        raise ValueError('Exact interrupted RUNNING source/arm/status required')
    checkpoint=_owned_regular(source/'training/checkpoint_latest.pt',context)
    expected_path=str(checkpoint)
    claim=proof['checkpoint']
    if (not isinstance(claim,dict) or set(claim)!={'path','sha256','size','mtime_ns'}
            or claim['path']!=expected_path or type(claim['size'])is not int or claim['size']<=0
            or type(claim['mtime_ns'])is not int or claim['mtime_ns']<=0
            or not isinstance(claim['sha256'],str) or len(claim['sha256'])!=64
            or any(char not in '0123456789abcdef' for char in claim['sha256'])):
        raise ValueError('Exact immutable interrupted checkpoint witness required')
    if (latest.get('path')!=expected_path or latest.get('raw_sha256')!=claim['sha256']
            or list(latest.get('source_stat',[]))[2:4]!=[claim['size'],claim['mtime_ns']]):
        raise ValueError('Original numerical inspector and interruption checkpoint differ')
    for name in ('worker','child'):
        process=proof[name]
        if (not isinstance(process,dict) or set(process)!={'pid','create_time','absent'}
                or type(process['pid'])is not int or process['pid']<=0
                or type(process['create_time'])not in (int,float)
                or not 0<process['create_time']<float('inf') or process['absent']is not True):
            raise ValueError('Exact absent original worker identity required')
    if proof['worker']['pid']==proof['child']['pid']:
        raise ValueError('Distinct original wrapper and training child required')
    snapshots=proof['snapshots']
    if (not isinstance(snapshots,list) or len(snapshots)!=2
            or any(not isinstance(row,dict) or set(row)!={'time_ns','checkpoint','worker_absent','child_absent'}
                or type(row['time_ns'])is not int or row['time_ns']<=0
                or row['checkpoint']!=claim or row['worker_absent']is not True or row['child_absent']is not True
                for row in snapshots)
            or snapshots[1]['time_ns']-snapshots[0]['time_ns']<1_000_000_000
            or type(proof['detected_ns'])is not int
            or not snapshots[1]['time_ns']<=proof['detected_ns']<=time.time_ns()):
        raise ValueError('Two stable checkpoint/PID snapshots at least one second apart required')
    status_path=_owned_regular(job/'status.json',context);status_sha=_sha(status_path)
    status=json.loads(status_path.read_text());request=status.get('request',{})
    if (status.get('status')!='RUNNING' or status.get('stage')!='train'
            or request.get('GPU')!=gpu or request.get('production_output')!=str(source)
            or request.get('code')!=str(code)
            or any(status.get(name+'_pid')!=proof[name]['pid']
                or status.get(name+'_create_time')!=proof[name]['create_time'] for name in ('worker','child'))):
        raise ValueError('Actual original job/launch identities differ from interruption proof')
    def absent():
        if any(context['pid_exists'](proof[name]['pid']) for name in ('worker','child')):
            raise ValueError('Original PID still exists or was reused; interruption is not admitted')
    absent()
    if _checkpoint_witness(checkpoint)!=claim:
        raise ValueError('Interrupted checkpoint SHA/size/mtime changed')
    absent()
    final_stat=path.stat()
    if ((proof_stat.st_dev,proof_stat.st_ino,proof_stat.st_size,proof_stat.st_mtime_ns,proof_stat.st_ctime_ns)!=(
            final_stat.st_dev,final_stat.st_ino,final_stat.st_size,final_stat.st_mtime_ns,final_stat.st_ctime_ns)
            or _sha(path)!=raw_sha or _sha(status_path)!=status_sha):
        raise ValueError('Interrupted proof or original job metadata changed during admission')
    return dict(path=str(path),raw_sha256=raw_sha,format=INTERRUPTION_FORMAT,
        source_checkpoint_sha256=claim['sha256'],original_status_preserved='RUNNING',
        actual_worker_and_child_absent=True,two_stable_snapshots_verified=True,
        old_status_or_checkpoint_modified=False)


def _source_admission(latest,source_output,source_code,gpu,proof_path=None):
    status=latest.get('status')
    if status=='PAUSED':
        if proof_path is not None:
            raise ValueError('Interruption proof cannot redefine an already PAUSED source')
        return dict(status='PAUSED',original_checkpoint_status='PAUSED',interruption=None)
    if status!='RUNNING' or proof_path is None:
        raise ValueError('Actual PAUSED boundary or explicit verified INTERRUPTED RUNNING source required')
    return dict(status='INTERRUPTED_RUNNING',original_checkpoint_status='RUNNING',
        interruption=validate_source_interruption(proof_path,source_output,source_code,latest,gpu=gpu))


def _publish(path,value):
    with Path(path).open('x',encoding='utf8') as stream:
        json.dump(value,stream,indent=2,allow_nan=False)


def runtime_identity():
    from hiercp_v1x import v24_hash_runtime,v24_prefetch_runtime,v24_input_runtime
    files=(Path(__file__).resolve(),Path(v24_hash_runtime.__file__).resolve(),
           Path(v24_prefetch_runtime.__file__).resolve(),Path(v24_input_runtime.__file__).resolve())
    return dict(files_sha256={str(path):_sha(path) for path in files},
        hash_contract=v24_hash_runtime.runtime_contract(),
        prefetch_contract=v24_prefetch_runtime.runtime_contract(),
        input_contract=v24_input_runtime.runtime_contract(),
        model_sampling_loss_and_batch_unchanged=True,original_strict_resume=True)


def prepare(args,source_output,source_code,source_interruption_proof=None):
    from hiercp_v1x.v24_training_continuation import prepare_continuation,inspect_continuation
    # Admission of a stable batch boundary is independently recorded below;
    # original prepare performs actual model/optimizer/RNG/cursor/BEST checks.
    old=inspect_continuation(source_output,source_code)
    admission=_source_admission(old['latest'],source_output,source_code,args.gpu,source_interruption_proof)
    proof=prepare_continuation(args,source_output=source_output,source_code=source_code)
    value=json.loads(proof.read_text())
    if value['source']!=old or _source_admission(value['source']['latest'],
            source_output,source_code,args.gpu,source_interruption_proof)!=admission:
        raise ValueError('Original interrupted/paused source changed during exact preparation')
    document=dict(format='v24_performance_exact_state_continuation_v1',
        original_continuation_receipt_sha256=_sha(proof),runtime=runtime_identity(),
        source=value['source'],source_admission=admission,destination=str(args.output.resolve()),
        CPU_RAM_GPU_limits_changed=False,model_or_dataset_reduced=False,
        physical_or_effective_batch_changed=False,production_optimizer_updates=0,
        created_at=time.time())
    _publish(args.output/RECEIPT,document)
    return document


def admit(args,source_output,source_code,source_interruption_proof=None):
    from hiercp_v1x.v24_training_continuation import admit_continuation
    original,value=admit_continuation(args)
    document=json.loads((args.output/RECEIPT).read_text())
    if (document['format']!='v24_performance_exact_state_continuation_v1'
            or document['runtime']!=runtime_identity()
            or document['original_continuation_receipt_sha256']!=_sha(original)
            or document['destination']!=str(args.output.resolve())
            or document['source']!=value['source']
            or document['source']['source_output']!=str(Path(source_output).resolve(strict=True))
            or document['source']['source_code']!=str(Path(source_code).resolve(strict=True))):
        raise ValueError('Exact source/runtime/numerical continuation proof differs')
    admission=document.get('source_admission',{})
    stored=admission.get('interruption')
    if stored is not None:
        if not isinstance(stored,dict) or not isinstance(stored.get('path'),str):
            raise ValueError('Immutable original interruption proof attachment required')
        if (source_interruption_proof is not None and str(Path(source_interruption_proof).resolve(strict=True))!=stored['path']):
            raise ValueError('Explicit interruption proof differs from prepared attachment')
        source_interruption_proof=Path(stored['path'])
    checked=_source_admission(document['source']['latest'],source_output,source_code,
        args.gpu,source_interruption_proof)
    if checked!=admission:
        raise ValueError('Prepared interrupted/paused admission or proof bytes changed')
    return original,document


def train(args,source_output,source_code,probe_baseline,probe_optimized,source_interruption_proof=None):
    from hiercp_v1x import v24_hash_runtime as hashes
    from hiercp_v1x import v24_prefetch_runtime as prefetch
    from hiercp_v1x.v24_training_continuation import _clone_main,_argv
    from tools import run_v24_all_p as cli
    original,document=admit(args,source_output,source_code,source_interruption_proof)
    if probe_baseline is None or probe_optimized is None:
        raise ValueError('Explicit original/optimized native full128 probe receipts required before train')
    comparison=compare_probes(json.loads(Path(probe_baseline).read_text()),
        json.loads(Path(probe_optimized).read_text()),document['source']['latest']['raw_sha256'])
    _publish(args.output/'performance_native_probe_admission.json',dict(comparison=comparison,
        baseline_file_sha256=_sha(probe_baseline),optimized_file_sha256=_sha(probe_optimized)))
    installed_hash=hashes.install()
    from hiercp_v1x import v24_factory,v24_memory_runtime as memory
    from hiercp_v1x.v24_preparation_runtime import install_runtime
    install_runtime(v24_factory)
    memory.install_memory_runtime(v24_factory)
    installed_prefetch=prefetch.install_runtime(memory)
    from hiercp_v1x import v24_input_runtime as input_runtime
    installed_input=input_runtime.install_runtime(pin_final_outputs=True)
    calls=[]
    def bind(build):
        def bounded_build(*positional,**keywords):
            result=build(*positional,**keywords)
            net,scorer,population,config,model_contract,close=result
            try:
                from hiercp_v1x.u_bridge_training import capture_rng,digest
                before_model=digest(net.state_dict());before_rng=digest(capture_rng())
                bound_memory=memory.bind_memory_runtime(scorer)
                bound_prefetch=prefetch.bind(scorer)
                bound_input=input_runtime.bind(scorer)
                inputs=scorer.geometry.memory_guard.__self__
                for path in document['runtime']['files_sha256']:
                    item=Path(path);proof=inputs.geometry._file_identity(item)
                    if _sha(item)!=document['runtime']['files_sha256'][path]:
                        raise ValueError('Performance runtime source changed during binding')
                    inputs._input_file_proofs[str(item)]=proof
                    inputs._file_proofs[str(item)]=proof
                inputs.guard_source()
                if digest(net.state_dict())!=before_model or digest(capture_rng())!=before_rng:
                    raise ValueError('CPU performance binding changed actual model or RNG')
                row=dict(hash_runtime=installed_hash,prefetch_runtime=installed_prefetch,
                    input_runtime=installed_input,input_binding=bound_input,
                    memory_binding=bound_memory,prefetch_binding=bound_prefetch,
                    model_and_RNG_unchanged=True,model_contract=model_contract,
                    original_training_identity_unchanged=True)
                calls.append(row)
                _publish(args.output/'performance_binding.json',row)
                return result
            except BaseException:
                close()
                raise
        return bounded_build
    clone=_clone_main(cli.main,bind)
    if clone.__code__ is not cli.main.__code__:
        raise ValueError('Original full training CLI code must remain identical')
    _publish(args.output/('performance_execution_'+str(time.time_ns())+'.json'),dict(
        continuation_receipt_sha256=_sha(args.output/RECEIPT),runtime=document['runtime'],
        original_CLI_code_preserved=True,started_at=time.time()))
    result=clone(_argv(args,args.output/'training/checkpoint_latest.pt'))
    if len(calls)!=1:raise ValueError('Exactly one original full-model runtime build required')
    _publish(args.output/('performance_invocation_'+str(time.time_ns())+'.json'),dict(
        original_CLI_returned=True,build=calls,full_training_completion_claimed=False,
        completed_at=time.time()))
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,add_help=False)
    parser.add_argument('--action',choices=('prepare','train'),required=True)
    parser.add_argument('--source-output',type=Path,required=True)
    parser.add_argument('--source-code',type=Path,required=True)
    parser.add_argument('--probe-baseline',type=Path)
    parser.add_argument('--probe-optimized',type=Path)
    parser.add_argument('--source-interruption-proof',type=Path,
        help='Owned immutable proof admitting the original RUNNING checkpoint after both original workers vanished')
    options,remaining=parser.parse_known_args(argv)
    from tools.run_v24_all_p import parse
    args=parse(remaining)
    if args.mode!='train':raise ValueError('Original full train execution required')
    os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES']='' if options.action=='prepare' else str(args.gpu)
    if options.action=='prepare':result=prepare(args,options.source_output,options.source_code,options.source_interruption_proof)
    else:result=train(args,options.source_output,options.source_code,options.probe_baseline,
        options.probe_optimized,options.source_interruption_proof)
    if options.action=='prepare':print(json.dumps(result),flush=True)
    return result


if __name__=='__main__':main()
