"""Resume the exact durable GPU4 state using its immutable original science code."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import uuid

ROOT=Path(__file__).resolve().parents[1]
SOURCE_COMMIT='2c8d594552ca115a4411a89a055121ab23ba268f'
FORMAT='v24_GPU4_exact_saved_state_continuation_v1'
RECEIPT='GPU4_exact_continuation.json'
EXECUTION_FILES=('tools/run_v24_gpu4_exact_continuation.py','tools/run_v24_gpu4_pipeline.py')
LOG_NAMES=frozenset(('update_timing.jsonl','validation_timing.jsonl','curve.jsonl','invocations.jsonl','failures.jsonl'))
PHASES=('initial_full_validation','training','train_probe','stage_validation','full_validation','epoch_completion','complete')


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*2**20),b''):digest.update(block)
    return digest.hexdigest()


def stat(path):
    value=Path(path).stat()
    return [value.st_dev,value.st_ino,value.st_size,value.st_mtime_ns,value.st_ctime_ns]


def owned(path,*,directory=False):
    path=Path(path)
    if (not path.is_absolute() or path.is_symlink() or path.resolve(strict=True)!=path.absolute()
            or not (path.is_dir() if directory else path.is_file()) or path.stat().st_uid!=41833):
        raise ValueError('Exact regular UID41833 continuation path required: '+str(path))
    return path


def read(path):return json.loads(owned(path).read_text(encoding='utf8'))


def atomic_bytes(path,content):
    """Publish a complete new file without replacing any existing result."""
    path=Path(path);temporary=path.with_name(path.name+'.GPU4_exact_'+uuid.uuid4().hex+'.tmp')
    created=False
    try:
        with temporary.open('xb') as stream:
            created=True;stream.write(content);stream.flush();os.fsync(stream.fileno())
        os.link(temporary,path)
    finally:
        if created:temporary.unlink()


def atomic_json(path,value):
    atomic_bytes(path,json.dumps(value,indent=2,allow_nan=False).encode('utf8'))


def copy_exact(source,destination):
    source=owned(source);before=stat(source);destination=Path(destination)
    destination.parent.mkdir(parents=True,exist_ok=True)
    temporary=destination.with_name(destination.name+'.GPU4_exact_'+uuid.uuid4().hex+'.tmp')
    created=False;checksum=hashlib.sha256()
    try:
        with source.open('rb') as reader,temporary.open('xb') as writer:
            created=True
            opened=os.fstat(reader.fileno())
            if [opened.st_dev,opened.st_ino,opened.st_size,opened.st_mtime_ns,opened.st_ctime_ns]!=before:
                raise ValueError('Original checkpoint/file changed before copy')
            for block in iter(lambda:reader.read(8*2**20),b''):
                checksum.update(block);writer.write(block)
            writer.flush();os.fsync(writer.fileno())
        if stat(source)!=before or sha(temporary)!=checksum.hexdigest():
            raise ValueError('Original checkpoint/file changed during exact copy')
        os.link(temporary,destination)
    finally:
        if created:temporary.unlink()
    return dict(source=str(source),source_stat=before,sha256=checksum.hexdigest(),bytes=before[2])


def derive_checkpoint_log(raw,name,state):
    """Retain only events represented by the durable checkpoint, preserving bytes."""
    if name not in LOG_NAMES:raise ValueError('Unknown original training log: '+name)
    lines=raw.splitlines(keepends=True);kept=[];excluded=0;partial=0;position=0
    for index,line in enumerate(lines):
        if not line.endswith(b'\n'):
            if index!=len(lines)-1:raise ValueError('Non-final partial original JSONL record')
            partial+=1;continue
        if not line.strip():kept.append(line);continue
        row=json.loads(line)
        if not isinstance(row,dict):raise ValueError('Original JSONL object required')
        keep=False;epoch=row.get('epoch')
        if name in ('failures.jsonl','invocations.jsonl'):
            keep=False
        elif name=='curve.jsonl':
            keep=type(epoch)is int and epoch<state['epoch']
        elif name=='update_timing.jsonl':
            if type(epoch)is not int:raise ValueError('Original update epoch required')
            if row.get('status')=='OPTIMIZER_UPDATED':
                update=row.get('update')
                if type(update)is not int:raise ValueError('Original durable update number required')
                keep=epoch<=state['epoch'] and update<=state['updates']
            elif row.get('status')=='AMP_OVERFLOW_RETRY_SAME_INPUT':
                cursor=row.get('cursor')
                if type(cursor)is not int:raise ValueError('Original retry patient cursor required')
                keep=epoch<state['epoch'] or epoch==state['epoch'] and cursor<state['train_position']
            else:raise ValueError('Unknown original update event')
        elif name=='validation_timing.jsonl':
            phase=row.get('phase')
            if type(epoch)is not int or phase not in PHASES:raise ValueError('Original validation epoch/phase required')
            if epoch<state['epoch']:keep=True
            elif epoch==state['epoch']:
                before=PHASES.index(phase)<PHASES.index(state['phase'])
                if phase==state['phase']:
                    patients=row.get('patients')
                    if type(patients)is not int or patients<1:raise ValueError('Original validation batch count required')
                    position+=patients;keep=position<=state['evaluation_position']
                else:keep=before
        if keep:kept.append(line)
        else:excluded+=1
    result=b''.join(kept)
    return result,dict(boundary_epoch=state['epoch'],boundary_updates=state['updates'],
        boundary_train_position=state['train_position'],boundary_evaluation_position=state['evaluation_position'],
        rows_kept=sum(bool(line.strip()) for line in kept),rows_excluded=excluded,partial_rows_archived=partial,
        sha256=hashlib.sha256(result).hexdigest(),bytes=len(result))


def execution_identity():
    commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    if len(commit)!=40 or any(c not in '0123456789abcdef' for c in commit):raise ValueError('Committed overlay code required')
    return dict(execution_code=str(owned(ROOT,directory=True)),execution_commit=commit,
        execution_files_sha256={name:sha(owned(ROOT/name)) for name in EXECUTION_FILES})


def load_science(code):
    """Refuse mixed checkouts; import only the original sealed scientific modules."""
    code=owned(code,directory=True)
    commit=subprocess.check_output(['git','-C',str(code),'rev-parse','HEAD'],text=True).strip()
    if commit!=SOURCE_COMMIT:raise ValueError('Exact original GPU4 science commit required')
    for name,module in tuple(sys.modules.items()):
        if name==__name__:continue
        if name=='tools' or name.startswith(('tools.','hiercp_v1x','hiercp_v22')):
            filename=getattr(module,'__file__',None)
            if filename and not Path(filename).resolve().is_relative_to(code):
                raise ValueError('Mixed scientific module checkout refused: '+name)
    sys.path.insert(0,str(code));sys.dont_write_bytecode=True
    from tools import run_v24_all_p as cli,run_v24_gpu4_all_u as original
    from hiercp_v1x import v24_training_continuation as continuation
    if Path(cli.__file__).resolve()!=code/'tools/run_v24_all_p.py':raise ValueError('Original scientific CLI not loaded')
    return cli,original,continuation


def failed_job(job,source,code):
    import psutil
    job=owned(job,directory=True);status_path=owned(job/'status.json');before=stat(status_path)
    status=read(status_path);request=status['request']
    if (status.get('status')!='FAILED' or status.get('stage')!='train' or request.get('GPU')!=4
            or request.get('code')!=str(code) or request.get('production_output')!=str(source)):
        raise ValueError('Exact failed original GPU4 train job required')
    processes={}
    for name in ('worker','child'):
        pid=status.get(name+'_pid');created=status.get(name+'_create_time')
        if type(pid)is not int or type(created)not in (int,float) or psutil.pid_exists(pid):
            raise ValueError('Original exact worker/child must be absent; never signal or relaunch alongside it')
        processes[name]=dict(pid=pid,create_time=created,absent=True)
    with owned(job/'train.log').open('rb') as stream:
        stream.seek(0,2);stream.seek(max(0,stream.tell()-32768));tail=stream.read()
    if b'[Errno 122] Disk quota exceeded' not in tail:raise ValueError('Explicit original quota failure evidence required')
    result=dict(path=str(job),status='FAILED',stage='train',status_sha256=sha(status_path),
        status_stat=before,request_sha256=sha(owned(job/'request.json')),processes=processes,
        quota_failure_verified=True)
    if stat(status_path)!=before:raise ValueError('Original failed job changed during inspection')
    return result


def source_snapshot(source,code,job,cli,continuation):
    import torch
    from hiercp_v1x.u_bridge_training import digest
    source=owned(source,directory=True);job_proof=failed_job(job,source,code)
    request=read(source/'request.json');declared={k:v for k,v in request.items() if k!='request_sha256'}
    if (request.get('format')!='v24_full_native_GT_free_execution_request_v1'
            or hashlib.sha256(json.dumps(declared,sort_keys=True,separators=(',',':')).encode()).hexdigest()!=request['request_sha256']
            or set(request['source'])!=set(cli.execution_source_files(4)) or request['physical_GPU']!=4):
        raise ValueError('Exact full GPU4 scientific execution request required')
    files={name:sha(owned(code/name)) for name in cli.execution_source_files(4)}
    if files!=request['source']:raise ValueError('Original immutable science source hashes changed')
    owner=read(source/'training/training_identity.json');binding=owner['binding']
    if digest(binding)!=owner['identity_sha256'] or binding['identity']!=request:
        raise ValueError('Original complete training identity differs')
    calibration=read(source/'calibration.json')
    if (calibration!=binding['config']['v24_runtime']['batch_calibration']
            or calibration['request_sha256']!=request['request_sha256']
            or calibration['physical_GPU']!=4 or calibration['world_size']!=1
            or calibration['debug']is not False or calibration['original_model_and_RNG_preserved']is not True
            or calibration['measured_full_P_U128_backward']is not True
            or calibration['initial_state_sha256']!=binding['initial_state_sha256']
            or calibration['selected_physical_patient_batch']!=4 or calibration['selected_physical_candidate_batch']!=64
            or calibration['model_contract']!=binding['model_contract']):
        raise ValueError('Original unchanged full128 B4/chunk64 calibration required')
    config=request['config'];cli.validate_config(config,4,Path(binding['model_contract']['STU_Net']['checkpoint_path']))
    if (binding['physical_patient_batch']!=4 or binding['physical_candidate_batch']!=64 or binding['epochs']!=40
            or len(binding['train_cases'])!=65 or len(binding['val_cases'])!=21
            or binding['model_contract']['encoder']!='official_pretrained_STU_Net_S'
            or binding['model_contract']['encoder_fine_tuned']is not True
            or binding['model_contract']['STU_Net']['sha256']!='f440f401bf4cac1d1d6f7c4635542ef057193a5f96c0b3c661a6179afeab7be6'
            or binding['model_contract']['parameters']!=13331792 or binding['model_contract']['trainable_parameters']!=13331792):
        raise ValueError('Original full model/population/B4/chunk64/40epoch680update contract required')
    saved,latest=continuation._checkpoint(source/'training/checkpoint_latest.pt',owner)
    if (latest['epoch']!=1 or latest['phase']!='training' or latest['updates']!=1 or latest['active_u']!=128
            or latest['train_position']!=4 or latest['optimizer_parameter_tensors']!=537 or latest['best']is not None):
        raise ValueError('Exact original durable epoch1/update1/U128/fourpatient checkpoint required')
    floating=[value for value in saved['model'].values() if isinstance(value,torch.Tensor) and value.is_floating_point()]
    if (len(floating)!=537 or sum(value.numel() for value in floating)!=13331792
            or any(value.device.type!='cpu' or not bool(torch.isfinite(value).all()) for value in floating)
            or torch.cuda.is_initialized()):raise ValueError('Complete finite CPU-only original model snapshot required')
    if (source/'training/checkpoint_best.pt').exists():raise ValueError('Unexpected BEST before first complete epoch')
    proof=dict(source_output=str(source),source_code=str(code),source_commit=SOURCE_COMMIT,
        request_sha256=request['request_sha256'],request_raw_sha256=sha(source/'request.json'),
        training_identity_sha256=owner['identity_sha256'],training_identity_raw_sha256=sha(source/'training/training_identity.json'),
        calibration_raw_sha256=sha(source/'calibration.json'),source_files_sha256=files,latest=latest,BEST=None,source_job=job_proof)
    return proof,saved


def validate_args(args,source,cli):
    if args.mode!='train' or args.gpu!=4 or args.resume is not None:
        raise ValueError('Exact original GPU4 train arguments without externally supplied resume required')
    config=json.loads(args.config.read_text(encoding='utf8'));cli.validate_config(config,4,args.stunet_checkpoint)
    if cli.request(args,config)!=read(source/'request.json'):
        raise ValueError('Scientific request/config/cache/checkpoint/seed changed; continuation refused')
    return config


def prepare(args,source,code,job,cli,continuation):
    proof,saved=source_snapshot(source,code,job,cli,continuation);validate_args(args,source,cli)
    output=Path(args.output).absolute()
    if output.exists() or output.is_relative_to(source) or source.is_relative_to(output):
        raise FileExistsError('Fresh disjoint GPU4 continuation destination required; old results preserved')
    owned(output.parent,directory=True)
    if shutil.disk_usage(output.parent).free<10*2**30:raise OSError('Continuation filesystem requires original10GiB free reserve')
    identity=execution_identity();output.mkdir();(output/'training').mkdir();(output/'source_lineage/training').mkdir(parents=True)
    copied={};derived={};lineage={};excluded=[]
    for path in sorted(source.iterdir()):
        if path.is_file() and (path.name in ('request.json','calibration.json','preparation.json')
                or path.name.startswith('resources_') and path.suffix=='.json'):
            copied[path.name]=copy_exact(path,output/path.name)
    for path in sorted((source/'training').iterdir()):
        if path.name=='STOP_AFTER_BATCH':continue
        owned(path)
        if path.name.startswith('checkpoint_latest.pt.') and path.suffix=='.tmp':
            excluded.append(dict(path=str(path),stat=stat(path),sha256=sha(path),reason='Uncommitted partial checkpoint retained at original source'))
            continue
        if path.name in LOG_NAMES:
            relative='source_lineage/training/'+path.name
            lineage[path.name]=copy_exact(path,output/relative)
            raw=owned(path).read_bytes();active,record=derive_checkpoint_log(raw,path.name,saved['state'])
            if hashlib.sha256(raw).hexdigest()!=lineage[path.name]['sha256']:raise ValueError('Original log changed after lineage copy')
            atomic_bytes(output/'training'/path.name,active)
            derived[path.name]=dict(record,source_sha256=lineage[path.name]['sha256'],path='training/'+path.name,lineage_path=relative)
        elif path.suffix=='.json' or path.name=='checkpoint_latest.pt':
            copied['training/'+path.name]=copy_exact(path,output/'training'/path.name)
        else:raise ValueError('Unknown original training artifact; preserve and inspect: '+str(path))
    repeated,_=source_snapshot(source,code,job,cli,continuation)
    if repeated!=proof:raise ValueError('Original failed snapshot changed during exact preparation')
    for relative,row in copied.items():
        if sha(owned(output/relative))!=row['sha256']:raise ValueError('Copied canonical bytes differ')
    fresh,_=continuation._checkpoint(output/'training/checkpoint_latest.pt',read(output/'training/training_identity.json'))
    if fresh['content_sha256']!=saved['content_sha256']:raise ValueError('Exact copied checkpoint content differs')
    created_at=time.time()
    viewer_cursor=dict(format='v24_GPU4_exact_saved_state_viewer_cursor_v1',source=proof,
        copied_files={row['path']:dict(bytes=row['bytes'],sha256=row['sha256'],
            derived_checkpoint_boundary=True) for row in derived.values()},created_at=created_at)
    atomic_json(output/'training_continuation.json',viewer_cursor)
    document=dict(format=FORMAT,status='PREPARED_EXACT_SAVED_STATE',source=proof,destination=str(output),
        source_code=str(code),source_commit=SOURCE_COMMIT,**identity,copied_files=copied,derived_logs=derived,
        original_full_logs=lineage,excluded_partial_files=excluded,
        scientific_request_unchanged=True,checkpoint_identity_and_content_unchanged=True,
        six_numerical_states_unchanged=True,curriculum_history_and_partial_cursors_unchanged=True,
        viewer_cursor_sha256=sha(output/'training_continuation.json'),
        original_files_modified=False,production_optimizer_updates_performed=0,created_at=created_at)
    atomic_json(output/RECEIPT,document)
    return document


def admit(args,source,code,job,cli,continuation):
    validate_args(args,source,cli);output=owned(args.output,directory=True);document=read(output/RECEIPT)
    expected=execution_identity()
    if (document.get('format')!=FORMAT or document.get('status')!='PREPARED_EXACT_SAVED_STATE'
            or document.get('destination')!=str(output) or document.get('source_code')!=str(code)
            or document.get('source_commit')!=SOURCE_COMMIT or any(document.get(k)!=v for k,v in expected.items())
            or document.get('original_files_modified')is not False or document.get('production_optimizer_updates_performed')!=0):
        raise ValueError('Exact prepared original-state/overlay receipt required')
    for key in ('scientific_request_unchanged','checkpoint_identity_and_content_unchanged',
                'six_numerical_states_unchanged','curriculum_history_and_partial_cursors_unchanged'):
        if document.get(key)is not True:raise ValueError('Missing explicit exact-state proof: '+key)
    proof,_=source_snapshot(source,code,job,cli,continuation)
    if document['source']!=proof:raise ValueError('Original failed source no longer matches prepared snapshot')
    for relative,row in document['copied_files'].items():
        if not (output/relative).resolve().is_relative_to(output) or sha(owned(output/relative))!=row['sha256']:
            raise ValueError('Prepared exact canonical file changed: '+relative)
        if stat(owned(Path(row['source'])))!=row['source_stat'] or sha(Path(row['source']))!=row['sha256']:
            raise ValueError('Preserved original canonical file changed: '+relative)
    for name,row in document['derived_logs'].items():
        if sha(owned(output/row['path']))!=row['sha256'] or sha(owned(output/row['lineage_path']))!=row['source_sha256']:
            raise ValueError('Prepared checkpoint-boundary/lineage log changed: '+name)
        original_log=document['original_full_logs'][name]
        if (stat(owned(Path(original_log['source'])))!=original_log['source_stat']
                or sha(Path(original_log['source']))!=original_log['sha256']):
            raise ValueError('Preserved original full lineage log changed: '+name)
    if sha(owned(output/'training_continuation.json'))!=document['viewer_cursor_sha256']:
        raise ValueError('Prepared truthful viewer continuation cursor changed')
    _,latest=continuation._checkpoint(output/'training/checkpoint_latest.pt',read(output/'training/training_identity.json'))
    for key in ('raw_sha256','content_sha256','identity_sha256','numerical_state_sha256','state_sha256'):
        if latest[key]!=proof['latest'][key]:raise ValueError('Prepared saved state differs: '+key)
    return document


def train(args,source,code,job,cli,original,continuation):
    document=admit(args,source,code,job,cli,continuation)
    from hiercp_v1x import v24_factory,v24_memory_runtime as memory,v24_hash_runtime as hashing
    from hiercp_v1x import v24_prefetch_runtime as prefetch,v24_input_runtime as inputs,v24_training as training
    from hiercp_v1x.v24_preparation_runtime import install_runtime
    from hiercp_v1x.u_bridge_training import capture_rng,digest
    install_runtime(v24_factory)
    installed=dict(memory=memory.install_memory_runtime(v24_factory),hash=hashing.install(),
        prefetch=prefetch.install_runtime(memory),inputs=inputs.install_runtime(pin_final_outputs=True))
    calls=[]
    def hook(build):
        def bounded(*positional,**keywords):
            result=build(*positional,**keywords)
            net,scorer,population,config,contract,close=result
            try:
                before_model,before_rng=digest(net.state_dict()),digest(capture_rng())
                binding=dict(memory=memory.bind_memory_runtime(scorer),prefetch=prefetch.bind(scorer),inputs=inputs.bind(scorer))
                if digest(net.state_dict())!=before_model or digest(capture_rng())!=before_rng:
                    raise ValueError('Continuation adapters changed original initialized model/RNG')
                if before_model!=read(args.output/'calibration.json')['initial_state_sha256']:
                    raise ValueError('Original official model initialization/calibration no longer matches')
                raw=scorer.geometry.memory_guard.__self__
                guarded={str(code/name):checksum for name,checksum in document['source']['source_files_sha256'].items()}
                guarded.update({str(ROOT/name):checksum for name,checksum in document['execution_files_sha256'].items()})
                for filename,checksum in guarded.items():
                    path=owned(Path(filename))
                    if sha(path)!=checksum:raise ValueError('Continuation implementation source changed')
                    proof=v24_factory._stat(path);raw._input_file_proofs[str(path)]=proof;raw._file_proofs[str(path)]=proof
                raw.guard_source()
                record=dict(format=FORMAT,source_latest=document['source']['latest'],initial_model_preserved=True,
                    original_strict_resume=True,source_scientific_identity_unchanged=True,installed=installed,bound=binding,
                    execution=execution_identity(),created_at=time.time())
                atomic_json(args.output/('GPU4_continuation_binding_'+str(time.time_ns())+'.json'),record)
                calls.append(record);return result
            except BaseException:
                close();raise
        return bounded
    cloned=original.clone_main(cli.main,hook,original.atomic_publish)
    original_writer=training._write_new
    argv=continuation._argv(args,args.output/'training/checkpoint_latest.pt')
    atomic_json(args.output/('GPU4_continuation_execution_'+str(time.time_ns())+'.json'),
        dict(format=FORMAT,original_CLI_code_preserved=cloned.__code__ is cli.main.__code__,
            six_saved_states_to_restore=document['source']['latest']['numerical_state_sha256'],
            resume_checkpoint=str(args.output/'training/checkpoint_latest.pt'),
            durable_optimizer_updates=document['source']['latest']['updates'],created_at=time.time()))
    training._write_new=original.atomic_new_json
    try:
        result=cloned(argv)
        if len(calls)!=1:raise ValueError('Exactly one original full-model runtime build required')
        return result
    finally:training._write_new=original_writer


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,add_help=False)
    parser.add_argument('--action',choices=('prepare','train'),required=True)
    parser.add_argument('--source-output',type=Path,required=True)
    parser.add_argument('--source-code',type=Path,required=True)
    parser.add_argument('--source-job',type=Path,required=True)
    options,remaining=parser.parse_known_args(argv)
    if not hasattr(os,'getuid') or os.getuid()!=41833 or socket.gethostname()!='ece-a6gpu6':
        raise ValueError('Exact original GPU4 server/UID required')
    os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID';os.environ['CUDA_VISIBLE_DEVICES']='' if options.action=='prepare' else '4'
    cli,original,continuation=load_science(options.source_code)
    args=cli.parse(remaining)
    if options.action=='prepare':result=prepare(args,options.source_output,options.source_code,options.source_job,cli,continuation)
    else:result=train(args,options.source_output,options.source_code,options.source_job,cli,original,continuation)
    if options.action=='prepare':print(json.dumps(result,allow_nan=False))
    return result


if __name__=='__main__':main()
