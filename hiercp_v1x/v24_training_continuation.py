"""Exact-byte v24 continuation into a fresh owned result namespace.

Original scientific sources, requests, calibration, latest/BEST and all saved
numerical/curriculum/cursor state stay unchanged. An independently bound memory
runtime is installed only around the original CLI's actual runtime constructor.
"""
from __future__ import annotations

import builtins
import copy
import hashlib
import json
import os
from pathlib import Path
import time
from types import FunctionType, SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
FORMAT='v24_exact_byte_training_continuation_v1'
RECEIPT='training_continuation.json'
NUMERICAL=('model','optimizer','scheduler','scaler','rank_rng','shuffle_generator')


def _sha(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*2**20),b''):result.update(block)
    return result.hexdigest()


def _stat(path):
    value=Path(path).stat()
    return (value.st_dev,value.st_ino,value.st_size,value.st_mtime_ns,value.st_ctime_ns)


def _read(path):return json.loads(Path(path).read_text(encoding='utf8'))


def _regular(path):
    path=Path(path)
    if path.is_symlink() or not path.is_file():raise ValueError('Exact regular owned file required: '+str(path))
    return path.resolve(strict=True)


def _publish(path,value):
    with Path(path).open('x',encoding='utf8') as stream:json.dump(value,stream,indent=2,allow_nan=False)


def _copy_exact(source,destination):
    """Copy a stable open snapshot, refusing existing destinations and mutations."""
    source=_regular(source);before=_stat(source);checksum=hashlib.sha256()
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    with source.open('rb') as reader,destination.open('xb') as writer:
        opened=os.fstat(reader.fileno())
        if (opened.st_dev,opened.st_ino,opened.st_size,opened.st_mtime_ns,opened.st_ctime_ns)!=before:
            raise ValueError('Source changed before exact snapshot copy')
        for block in iter(lambda:reader.read(8*2**20),b''):
            checksum.update(block);writer.write(block)
        writer.flush();os.fsync(writer.fileno())
    if _stat(source)!=before or _sha(destination)!=checksum.hexdigest():
        raise ValueError('Exact-byte continuation copy/source snapshot changed')
    return dict(source=str(source),source_stat=list(before),sha256=checksum.hexdigest(),bytes=before[2])


def _helpers():
    from .u_bridge_training import digest
    return digest


def _checkpoint(path,owner):
    import torch
    digest=_helpers();path=_regular(path);before=_stat(path)
    saved=torch.load(path,map_location='cpu',weights_only=False)
    required={'format','identity_sha256',*NUMERICAL,'state','content_sha256'}
    if (set(saved)!=required or saved['format']!='v24_GT_blind_all_P_patient_balanced_training_v1'
            or saved['identity_sha256']!=owner['identity_sha256']
            or digest({key:value for key,value in saved.items() if key!='content_sha256'})!=saved['content_sha256']):
        raise ValueError('Original exact v24 checkpoint content/identity required')
    state=saved['state'];binding=owner['binding'];history=state['history']
    if (state['epoch']!=len(history)+1 or [row['epoch'] for row in history]!=list(range(1,state['epoch']))
            or saved['scheduler']['last_epoch']!=len(history) or len(saved['rank_rng'])!=1
            or state['updates']<0 or state['attempts']<state['updates']
            or state['curriculum']['policy']!=binding['config']['v24_runtime']['curriculum']):
        raise ValueError('Original singleton history/scheduler/curriculum/counters differ')
    train=binding['train_cases'];val=binding['val_cases'];order=state['train_order']
    if order is not None and (len(order)!=len(train) or set(order)!=set(train)):
        raise ValueError('Original full train order differs')
    if (state['train_position']!=len(state['train_rows'])
            or order is not None and [row['case_id'] for row in state['train_rows']]!=order[:state['train_position']]):
        raise ValueError('Original train cursor/rows differ')
    phases=('training','epoch_completion','complete','initial_full_validation','train_probe','stage_validation','full_validation')
    if state['phase'] not in phases:raise ValueError('Original checkpoint phase is unknown')
    cases=train if state['phase']=='train_probe' else val if state['phase'] in phases[3:] else []
    if (state['evaluation_position']!=len(state['evaluation_rows'])
            or [row['case_id'] for row in state['evaluation_rows']]!=cases[:state['evaluation_position']]):
        raise ValueError('Original evaluation cursor/rows differ')
    steps=[float(row['step']) for row in saved['optimizer']['state'].values()]
    if state['updates'] and (not steps or any(step!=state['updates'] for step in steps)):
        raise ValueError('Original AdamW update history differs')
    from .v24_training import _validate_progress
    _validate_progress(state,train,val,binding['physical_patient_batch'],binding['epochs'],
        SimpleNamespace(last_epoch=saved['scheduler']['last_epoch']))
    proof=dict(path=str(path),source_stat=list(before),raw_sha256=_sha(path),
        content_sha256=saved['content_sha256'],identity_sha256=saved['identity_sha256'],
        numerical_state_sha256={key:digest(saved[key]) for key in NUMERICAL},
        state_sha256=digest(state),epoch=state['epoch'],phase=state['phase'],status=state['status'],
        updates=state['updates'],history_epochs=[row['epoch'] for row in history],
        curriculum_sha256=digest(state['curriculum']),active_u=state['curriculum']['active_u'],
        train_position=state['train_position'],evaluation_position=state['evaluation_position'],
        best=copy.deepcopy(state['best']),optimizer_parameter_tensors=len(steps))
    if _stat(path)!=before:raise ValueError('Original checkpoint changed during admission')
    return saved,proof


def inspect_continuation(source_output,source_code):
    """Read-only validation; no model constructor, RNG restore or result writes."""
    from tools import run_v24_all_p as cli
    digest=_helpers();output=Path(source_output).resolve(strict=True);code=Path(source_code).resolve(strict=True)
    request_path=_regular(output/'request.json');request=_read(request_path)
    declared={key:value for key,value in request.items() if key!='request_sha256'}
    if (request.get('format')!='v24_full_native_GT_free_execution_request_v1'
            or hashlib.sha256(json.dumps(declared,sort_keys=True,separators=(',',':')).encode()).hexdigest()!=request['request_sha256']
            or set(request['source'])!=set(cli.FILES)):
        raise ValueError('Original full scientific request/source manifest required')
    sources={name:_sha(_regular(code/name)) for name in cli.FILES}
    if sources!=request['source'] or {name:_sha(_regular(cli.ROOT/name)) for name in cli.FILES}!=sources:
        raise ValueError('All original CLI/scientific source bytes must remain identical; no fallback')
    owner_path=_regular(output/'training/training_identity.json');owner=_read(owner_path)
    if digest(owner['binding'])!=owner['identity_sha256'] or owner['binding']['identity']!=request:
        raise ValueError('Original full training identity/request differs')
    calibration_path=_regular(output/'calibration.json');calibration=_read(calibration_path)
    if (calibration['request_sha256']!=request['request_sha256']
            or calibration['physical_GPU']!=request['physical_GPU']
            or calibration!=owner['binding']['config']['v24_runtime']['batch_calibration']
            or calibration['initial_state_sha256']!=owner['binding']['initial_state_sha256']
            or calibration['model_contract']!=owner['binding']['model_contract']):
        raise ValueError('Original actual calibration/model/initial-state binding differs')
    latest,latest_proof=_checkpoint(output/'training/checkpoint_latest.pt',owner)
    if latest['state']['phase']=='complete' or latest['state']['epoch']>40:
        raise ValueError('Completed full40 run cannot be presented as unfinished continuation')
    best_proof=None
    if latest['state']['best'] is not None:
        best,best_proof=_checkpoint(output/'training/checkpoint_best.pt',owner)
        selected=latest['state']['best'];full=best['state']['pending_epoch_completion']['full_validation']
        from .v23_training import best_selection_key
        if (best['state']['best']!=selected or best['state']['epoch']!=selected['epoch']
                or best['state']['updates']!=selected['updates'] or full['phase']!='full_validation' or full['active_u']!=128
                or list(best_selection_key(full))!=selected['selection_key']):
            raise ValueError('Actual full128 BEST checkpoint/state must be preserved')
        del best
    elif (output/'training/checkpoint_best.pt').exists():
        raise ValueError('Unexpected BEST artifact for an undefined saved winner')
    del latest
    return dict(source_output=str(output),source_code=str(code),source_files_sha256=sources,
        request_sha256=request['request_sha256'],request_raw_sha256=_sha(request_path),
        training_identity_sha256=owner['identity_sha256'],training_identity_raw_sha256=_sha(owner_path),
        calibration_raw_sha256=_sha(calibration_path),latest=latest_proof,BEST=best_proof)


def _argv(args,resume):
    values=['--mode','train','--config',str(args.config),'--native-experiment',str(args.native_experiment),
        '--inventory',str(args.inventory),'--input-cache',str(args.input_cache),'--output',str(args.output),
        '--gpu',str(args.gpu),'--resume',str(resume)]
    if args.stunet_checkpoint is not None:values.extend(('--stunet-checkpoint',str(args.stunet_checkpoint)))
    if args.cpu_affinity is not None:values.extend(('--cpu-affinity',','.join(map(str,args.cpu_affinity))))
    return values


def _runtime_proof():
    from . import v24_memory_runtime as memory
    tool=ROOT/'tools/run_v24_training_continuation.py'
    return dict(module_path=str(Path(__file__).resolve()),module_sha256=_sha(__file__),
        wrapper_CLI_path=str(tool),wrapper_CLI_sha256=_sha(_regular(tool)),
        memory_module_path=str(Path(memory.__file__).resolve()),memory_module_sha256=_sha(memory.__file__),
        memory_runtime_contract=memory.memory_runtime_contract())


def prepare_continuation(args,*,source_output,source_code):
    from tools import run_v24_all_p as cli
    old=inspect_continuation(source_output,source_code);source=Path(old['source_output'])
    config=_read(args.config);cli.validate_config(config,args.gpu,args.stunet_checkpoint)
    if cli.request(args,config)!=_read(source/'request.json'):
        raise ValueError('Original request identity must be identical including config/cache/inventory/pretrained bytes')
    output=Path(args.output).resolve()
    if output.exists() or output.is_relative_to(source) or source.is_relative_to(output):
        raise FileExistsError('Fresh disjoint continuation output required; original results preserved')
    runtime=_runtime_proof();output.mkdir(parents=True)
    copied={}
    for path in sorted(source.iterdir()):
        if path.is_file() and (path.name in ('request.json','calibration.json','preparation.json')
                or path.name.startswith('resources_') and path.suffix=='.json'):
            copied[path.name]=_copy_exact(path,output/path.name)
    training=source/'training'
    for path in sorted(training.iterdir()):
        if path.name=='STOP_AFTER_BATCH':continue
        if not path.is_file() or path.suffix not in ('.json','.jsonl','.pt'):
            raise ValueError('Unrecognized original training artifact; preserve it and inspect explicitly: '+str(path))
        if path.suffix=='.pt' and path.name not in ('checkpoint_latest.pt','checkpoint_best.pt'):
            raise ValueError('Unrecognized original checkpoint artifact: '+str(path))
        relative='training/'+path.name;copied[relative]=_copy_exact(path,output/relative)
    if inspect_continuation(source_output,source_code)!=old:
        raise ValueError('Original source/results changed during fresh continuation preparation')
    fresh=inspect_continuation(output,source_code)
    for key in ('raw_sha256','content_sha256','identity_sha256','numerical_state_sha256','state_sha256'):
        if fresh['latest'][key]!=old['latest'][key]:raise ValueError('Latest continuation state was not copied exactly')
    if fresh['BEST'] is not None and any(fresh['BEST'][key]!=old['BEST'][key] for key in
            ('raw_sha256','content_sha256','numerical_state_sha256','state_sha256')):
        raise ValueError('Actual BEST continuation state was not copied exactly')
    receipt=output/RECEIPT
    _publish(receipt,dict(format=FORMAT,status='PREPARED_EXACT_BYTES',source=old,runtime=runtime,
        destination=str(output),copied_files=copied,scientific_request_unchanged=True,
        checkpoint_identity_and_content_unchanged=True,six_numerical_states_unchanged=True,
        curriculum_history_and_partial_cursors_unchanged=True,true_BEST_preserved=True,
        old_STOP_AFTER_BATCH_excluded=True,original_files_modified=False,
        production_optimizer_updates_performed=0,created_at=time.time()))
    return receipt


def admit_continuation(args):
    from tools import run_v24_all_p as cli
    output=Path(args.output).resolve(strict=True);path=_regular(output/RECEIPT);value=_read(path)
    if (value.get('format')!=FORMAT or value.get('status')!='PREPARED_EXACT_BYTES'
            or value.get('runtime')!=_runtime_proof() or value.get('destination')!=str(output)
            or value.get('original_files_modified')is not False
            or value.get('production_optimizer_updates_performed')!=0):
        raise ValueError('Actual exact-byte continuation/runtime receipt required')
    for key in ('scientific_request_unchanged','checkpoint_identity_and_content_unchanged','six_numerical_states_unchanged',
            'curriculum_history_and_partial_cursors_unchanged','true_BEST_preserved','old_STOP_AFTER_BATCH_excluded'):
        if value.get(key)is not True:raise ValueError('Explicit continuation proof missing: '+key)
    if cli.request(args,_read(args.config))!=_read(output/'request.json'):
        raise ValueError('Current CLI arguments redefine the exact scientific request')
    for relative,row in value['copied_files'].items():
        if not _regular(output/relative).is_relative_to(output) or _sha(output/relative)!=row['sha256']:
            raise ValueError('Prepared original continuation bytes changed: '+relative)
        if _stat(row['source'])!=tuple(row['source_stat']) or _sha(row['source'])!=row['sha256']:
            raise ValueError('Original preserved source snapshot changed: '+relative)
    fresh=inspect_continuation(output,value['source']['source_code'])
    for key in ('raw_sha256','content_sha256','identity_sha256','numerical_state_sha256','state_sha256'):
        if fresh['latest'][key]!=value['source']['latest'][key]:raise ValueError('Saved continuation latest state differs')
    return path,value


def _clone_main(original,build_hook):
    """Intercept exactly the existing factory from-import, preserving main code."""
    importer=builtins.__import__
    namespace=dict(original.__globals__)
    original_builtins=namespace.get('__builtins__',builtins)
    private_builtins=dict(original_builtins if isinstance(original_builtins,dict) else vars(original_builtins))
    def importing(name,globals=None,locals=None,fromlist=(),level=0):
        module=importer(name,globals,locals,fromlist,level)
        if name=='hiercp_v1x.v24_factory' and level==0 and tuple(fromlist)==('V24NativeInputs','build_runtime'):
            return SimpleNamespace(V24NativeInputs=module.V24NativeInputs,build_runtime=build_hook(module.build_runtime))
        return module
    private_builtins['__import__']=importing;namespace['__builtins__']=private_builtins
    clone=FunctionType(original.__code__,namespace,original.__name__,original.__defaults__,original.__closure__)
    clone.__kwdefaults__=copy.deepcopy(original.__kwdefaults__)
    return clone


def run_continuation(args):
    from tools import run_v24_all_p as cli
    from . import v24_factory,v24_memory_runtime as memory
    from .v24_preparation_runtime import install_runtime
    receipt,value=admit_continuation(args)
    install_runtime(v24_factory)
    installed=memory.install_memory_runtime(v24_factory)
    calls=[]
    def bind(original):
        def build(*positional,**keywords):
            result=original(*positional,**keywords)
            net,scorer,population,config,model_contract,close=result
            try:
                from .u_bridge_training import capture_rng,digest
                before_model=digest(net.state_dict());before_rng=digest(capture_rng())
                bound=memory.bind_memory_runtime(scorer)
                if digest(net.state_dict())!=before_model or digest(capture_rng())!=before_rng:
                    raise ValueError('Memory runtime changed actual model/RNG during original build')
                calls.append(dict(installed=installed,bound=bound,model_and_RNG_unchanged=True))
                return result
            except BaseException:
                # Original main enters its finally only after build returns.
                # A rejected binding must still release its own CPU workers.
                close()
                raise
        return build
    clone=_clone_main(cli.main,bind)
    if clone.__code__ is not cli.main.__code__:raise ValueError('Original CLI main code must stay unchanged')
    launch=Path(args.output)/('continuation_execution_'+str(time.time_ns())+'.json')
    _publish(launch,dict(format=FORMAT,status='ADMITTED_BEFORE_ORIGINAL_CLI',continuation_receipt=str(receipt),
        continuation_receipt_sha256=_sha(receipt),runtime=value['runtime'],original_CLI_code_preserved=True,
        original_strict_resume_used=True,identity_rewritten=False,created_at=time.time()))
    result=clone(_argv(args,Path(args.output)/'training/checkpoint_latest.pt'))
    if len(calls)!=1:raise ValueError('Exactly one original full-model runtime build/memory binding required')
    _publish(Path(args.output)/('continuation_invocation_'+str(time.time_ns())+'.json'),
        dict(format=FORMAT,status='ORIGINAL_CLI_RETURNED',build=calls,original_CLI_code_preserved=True,
            full_training_completion_claimed=False,continuation_receipt_sha256=_sha(receipt),completed_at=time.time()))
    return result


def main(args,*,source_output,source_code,action):
    if action not in ('prepare','train'):raise ValueError('Explicit continuation prepare/train action required')
    os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES']='' if action=='prepare' else str(args.gpu)
    if action=='prepare':return prepare_continuation(args,source_output=source_output,source_code=source_code)
    path,value=admit_continuation(args)
    if (str(Path(source_output).resolve(strict=True))!=value['source']['source_output']
            or str(Path(source_code).resolve(strict=True))!=value['source']['source_code']):
        raise ValueError('Explicit original continuation source paths differ')
    return run_continuation(args)
