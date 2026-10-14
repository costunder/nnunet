"""LOCAL PREPARED: unchanged800GiB recovery with one current seven-arm ledger.

Parent alone executes after exact old native/controller/source-worker release.
No process signals, checkpoint copies, code hardlinks, or old metadata writes.
"""
from pathlib import Path
import argparse
import copy
import gc
import hashlib
import json
import os
import re
import socket
import subprocess
import sys
import time
import psutil

UID=41833
BASE=Path('/dev/shm/aicompetition06_basic_cp_ablation_20261013')
ARMS={0:'multi5',1:'multi7',2:'multi10',3:'candidates512',4:'candidates1024',5:'candidates2048',6:'blend50'}
SOURCE_UPDATE_KEYS=frozenset(('previous_partial_banks','previous_producer_request','previous_producer_request_sha256',
    'interrupted_candidate_recovery_snapshot','interrupted_candidate_recovery_snapshot_sha256'))
SCIENCE_KEYS=('arm','gpu','gpu_uuid','copies_per_CP_event','candidates_per_source','bank','bank_sha256',
    'native_root','archive_root','readonly_admission','readonly_admission_sha256','candidate_binding','candidate_binding_sha256',
    'primary_joint_resampling_policy','cpu_affinity','RSS_limit_bytes','runtime_reserve_bytes',
    'shared_bank_workers','shared_bank_CPU_affinity','shared_bank_RSS_limit_GiB',
    'shared_resource_guard_version','group_observation_revision','shared_resource_observation_revision',
    'RAM_admission_pause_revision','GPU_admission_retry_version','interrupted_candidate_recovery_revision','prepared256_witness')
ALLOWED_CODE_CHANGES=frozenset(('hiercp_v1x/basic_cp_shared_bank_resources_v6.py','tools/persist_basic_cp_seven.py'))


def require(condition,message):
    if not condition:raise ValueError(message)


def read(path):return json.loads(Path(path).read_bytes())


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb')as stream:
        first=os.fstat(stream.fileno())
        for block in iter(lambda:stream.read(8*2**20),b''):digest.update(block)
        final=os.fstat(stream.fileno())
    keys=('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')
    require(all(getattr(first,k)==getattr(final,k)for k in keys),'Evidence changed during held-file SHA read')
    return digest.hexdigest()


def owned(path,directory=False):
    path=Path(path)
    require(path.is_absolute()and '..'not in path.parts and path.resolve(strict=True)==path
        and not any(p.is_symlink()for p in(path,*path.parents)),'Absolute own non-symlink evidence required: '+str(path))
    require(path.stat().st_uid==UID and(path.is_dir()if directory else path.is_file()),'Evidence UID/type differs: '+str(path))
    return path


def seal(path,value):
    path=Path(path);temporary=path.with_name('.'+path.name+'.exclusive_'+str(time.time_ns())+'.tmp')
    try:
        with temporary.open('x',encoding='utf8')as stream:
            json.dump(value,stream,indent=2,ensure_ascii=False,allow_nan=False);stream.flush();os.fsync(stream.fileno())
        os.link(temporary,path,follow_symlinks=False);temporary.unlink()
        descriptor=os.open(path.parent,os.O_RDONLY)
        try:os.fsync(descriptor)
        finally:os.close(descriptor)
    finally:
        if temporary.exists():temporary.unlink() # This call's exclusive unpublished metadata only.


def copy_bytes(source,target):
    data=Path(source).read_bytes()
    with Path(target).open('xb')as stream:stream.write(data);stream.flush();os.fsync(stream.fileno())
    require(sha(source)==sha(target),'Small evidence copy changed exact bytes')
    return dict(source=str(source),target=str(target),sha256=sha(target),bytes=len(data))


def identity(process):
    return dict(pid=process.pid,create_time=process.create_time(),UID=process.uids().real,command=process.cmdline())


def extract_identities(value):
    result=[]
    if isinstance(value,dict):
        birth=value.get('create_time',value.get('birth'))
        if type(value.get('pid'))is int and type(birth)in(int,float)and value.get('UID')==UID and isinstance(value.get('command'),list)and value['command']:
            result.append(dict(pid=value['pid'],create_time=birth,UID=UID,command=value['command']))
        for child in value.values():result.extend(extract_identities(child))
    elif isinstance(value,list):
        for child in value:result.extend(extract_identities(child))
    return result


def released(expected):
    try:process=psutil.Process(expected['pid'])
    except psutil.NoSuchProcess:return dict(identity=expected,status='NoSuchProcess')
    require(process.create_time()==expected['create_time']and process.uids().real==expected['UID']==UID,
        'Old actor PID/birth/UID changed')
    require(process.status()in(psutil.STATUS_ZOMBIE,psutil.STATUS_DEAD),'Old exact actor is still live; no signal sent')
    for child in process.children(recursive=True):
        try:require(child.status()in(psutil.STATUS_ZOMBIE,psutil.STATUS_DEAD),'Old terminal actor still owns a live descendant')
        except psutil.NoSuchProcess:continue
    return dict(identity=expected,status=process.status())


def validate_model(model,native):
    require(model['trainer']==native['trainer']and model['epochs']==250
        and model['physical_batch']==model['effective_batch']==2 and model['gradient_accumulation']==1
        and model['patch_size']==[128]*3 and model['workers']==4 and model['debug']is False and model['subset']is False
        and model['pretrained_segmentation_used']is False
        and sorted(model['actual_train_cases'])==sorted(native['split']['outer_train'])
        and sorted(model['actual_validation_cases'])==sorted(native['split']['outer_val'])
        and len(set(model['actual_train_cases']))==105 and len(set(model['actual_validation_cases']))==26,
        'Actual full model/B2/250/128cube/4workers/105train/26validation differs')


def apply_source_updates(request,document):
    require(isinstance(document,dict)and document.get('format')=='basic_cp_current_candidate_source_recovery_updates_v1'
        and isinstance(document.get('request_updates'),dict)and set(document['request_updates'])==SOURCE_UPDATE_KEYS,
        'Exactly five reviewed current-source recovery path/SHA updates required')
    result=copy.deepcopy(request);result.update(document['request_updates'])
    for name in SCIENCE_KEYS:require(result.get(name)==request.get(name),'Source update changed scientific/resource value: '+name)
    return result


def retarget(request,gpu,root,registry,code_export,banks,producer):
    result=copy.deepcopy(request)
    result.update(root=str(root),shared_resource_registry=str(registry),code_sha256=code_export['code_sha256'],commit=code_export['commit'])
    result.pop('PBS_shared_cache_root',None)
    if 'shared_banks'in result:result['shared_banks']=str(banks)
    if 'producer_root'in result:result['producer_root']=str(producer)
    for name in SCIENCE_KEYS:require(result.get(name)==request.get(name),'Retarget changed scientific/native/resource value: '+name)
    require(result['gpu']==gpu and result['arm']==ARMS[gpu],'Exact seven arm/GPU identity differs')
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--execute',action='store_true')
    parser.add_argument('--new-group',type=Path,required=True)
    for name in('current-manifest','release-proof','source-updates','code-export'):
        parser.add_argument('--'+name,type=Path,required=True);parser.add_argument('--'+name+'-sha256',required=True)
    args=parser.parse_args();require(args.execute,'Prepared G5 restart requires explicit parent --execute')
    require(os.getuid()==UID and socket.gethostname().split('.')[0]=='ece-a6gpu6','Exact own UID and node required')
    group=args.new_group
    require(group.is_absolute()and group.parent==BASE and group.name.startswith('Basic_CP_')and '..'not in group.parts,
        'Fresh own G5 namespace must be one direct Basic_CP_ child of the original BASE')
    if group.exists():owned(group,True)
    require(not(group/'launch.json').exists()and not(group/'shared_resource_registry.json').exists()
        and not(group/'banks').exists()and all(not(group/('GPU'+str(g)+'_'+ARMS[g])).exists()for g in range(7)),
        'Fresh seven roots/common registry/absent bank output required')
    input_files={}
    for name in('current_manifest','release_proof','source_updates','code_export'):
        path=owned(getattr(args,name));digest=getattr(args,name+'_sha256')
        require(re.fullmatch('[0-9a-f]{64}',digest)and sha(path)==digest,'Reviewed exact input SHA differs: '+name)
        input_files[str(path)]=digest
    manifest=read(args.current_manifest);release_document=read(args.release_proof);source_document=read(args.source_updates);export=read(args.code_export)
    require(set(manifest['registration'])==set(map(str,range(7))),'Current exact seven registered rows required')
    require(release_document.get('all_old_native_and_DA_terminal')is True
        or release_document.get('all_old_pipeline_actors_terminal')is True,'Reviewed actual old pipeline release receipt required')
    code=owned(export['code'],True)
    require(code==BASE/'Basic_CP_GPU5_source_recovery_1791966900100000000'/'code'
        and re.fullmatch('[0-9a-f]{40}',export['commit']),'Existing exact fd296 G3 export required; no new code tree')
    for name,digest in export['code_sha256'].items():require(sha(owned(code/name))==digest,'Existing immutable reviewed code SHA changed')
    captured=extract_identities(release_document);actors={(a['pid'],a['create_time']):a for a in captured}
    requests={};native_old={};old_metadata={};known_stage_actors=[]
    for gpu in range(7):
        row=manifest['registration'][str(gpu)];root=owned(row['root'],True);path=owned(root/'request.json');request=read(path)
        require(sha(path)==row['request_sha256']and request['controller_identity']==row['controller_identity']
            and request['gpu']==gpu and request['arm']==ARMS[gpu],'Exact source request/controller/arm differs')
        require(set(request['code_sha256'])==set(export['code_sha256'])and
            {name for name in request['code_sha256']if request['code_sha256'][name]!=export['code_sha256'][name]}<=ALLOWED_CODE_CHANGES,
            'Only already reviewed operational Shared6/helper deltas may enter the common code cohort')
        require(request['shared_resource_guard_version']==3 and request['group_observation_revision']==5
            and request['shared_resource_observation_revision']==6 and request['RAM_admission_pause_revision']==1
            and request['GPU_admission_retry_version']==1 and request['RSS_limit_bytes']==96*2**30,
            'Unchanged96/384/768/800 guards, RAM pause and native GPU queue required')
        require('dynamic_RAM_revision'not in request and 'RAM_reservation_policy_revision'not in request,
            'Unapproved dynamic RAM policy must not enter the unchanged800 generation')
        requests[gpu]=request;old_metadata[str(path)]=row['request_sha256']
        actor=row['controller_identity'];actors[(actor['pid'],actor['create_time'])]=actor
        for receipt in root.glob('*_actor.json'):
            owned(receipt)
            for actor in extract_identities(read(receipt)):
                require(actors.get((actor['pid'],actor['create_time']))==actor,
                    'Release receipt must capture the exact previous native/source actor: '+str(actor['pid']))
                known_stage_actors.append(actor)
        native_path=root/'native.json'
        if native_path.exists():native_old[gpu]=read(owned(native_path));old_metadata[str(native_path)]=sha(native_path)
    require(set(native_old)=={0,1,2,3,4,6},'Six actual existing natives and pre-native source5 required')
    release=[released(actor)for actor in actors.values()]
    needles={str(Path(row['root'])/'request.json')for row in manifest['registration'].values()}|{
        str(Path(row['root'])/'native.json')for row in manifest['registration'].values()}
    for process in psutil.process_iter():
        try:
            if process.pid==os.getpid()or process.uids().real!=UID:continue
            if any(argument in needles for argument in process.cmdline()):
                require(process.status()in(psutil.STATUS_ZOMBIE,psutil.STATUS_DEAD),'Old current-generation request/native-bound actor remains live')
        except psutil.NoSuchProcess:continue
    source_request=apply_source_updates(requests[5],source_document)
    predecessor=owned(source_request['previous_producer_request']);snapshot=owned(source_request['interrupted_candidate_recovery_snapshot'])
    require(sha(predecessor)==source_request['previous_producer_request_sha256']and
        sha(snapshot)==source_request['interrupted_candidate_recovery_snapshot_sha256'],'Exact current source predecessor/recovery snapshot changed')
    source_snapshot=read(snapshot)
    require(source_snapshot['counts']['512']['VALID']==642 and source_snapshot['counts']['1024']['VALID']==642,
        'All642 sources in both completed prefixes must remain preserved')
    require(owned(source_request['previous_partial_banks'],True)==Path(read(predecessor)['shared_banks']),
        'Current interrupted bank namespace must bind exact predecessor request')
    lifetime=requests[0]['host_lifetime'];witness=read(owned(lifetime['path']))
    require(all(q['host_lifetime']==lifetime for q in requests.values())and sha(lifetime['path'])==lifetime['sha256'],
        'Exact same truthful current owned host witness required for all7')
    job=json.loads(subprocess.check_output(['/opt/pbs/bin/qstat','-f','-F','json',witness['job_id']],text=True,timeout=20))['Jobs'][witness['job_id']]
    require(job['job_state']=='R'and job['Job_Owner'].split('@')[0]=='aicompetition06'
        and all(host.split('/')[0]=='ece-a6gpu6'for host in job['exec_host'].split('+'))
        and time.time()<lifetime['expires_at_unix']-180,'Actual same-host owned R lifetime with180s reserve required')
    sys.path.insert(0,str(code))
    from hiercp_v1x.basic_cp_seven_resources import CPUS,capture_cgroup_admission,validate_request
    from hiercp_v1x.basic_cp_verified_continuation import _state,_completed_prefix
    from hiercp_v1x.basic_cp_completed_epoch_checkpoint import _LOGGER_FIELDS
    from hiercp_v1x.basic_cp_ablation_resume import fold
    import torch
    registry=group/'shared_resource_registry.json';banks=group/'banks';producer=group/'GPU5_candidates2048'
    pending=[];checkpoint_proofs={}
    for gpu in range(7):
        root=group/('GPU'+str(gpu)+'_'+ARMS[gpu]);original=source_request if gpu==5 else requests[gpu]
        request=retarget(original,gpu,root,registry,export,banks,producer);native=None;proof=None;prefix=None
        if gpu!=5:
            old=native_old[gpu];oldroot=Path(requests[gpu]['root']);model=read(owned(oldroot/'actual_model.json'));validate_model(model,old)
            goal=owned(fold(old),True);latest=owned(goal/'checkpoint_latest.pth');best=owned(goal/'checkpoint_best.pth')
            latest_sha=sha(latest);best_sha=sha(best)
            saved=torch.load(latest,map_location='cpu',weights_only=False);epoch=saved.get('current_epoch')
            require(type(epoch)is int and 0<epoch<250 and _LOGGER_FIELDS<=set(saved['logging']),
                'Actual incomplete whole native latest with all8 logger arrays required')
            state=_state(saved,epoch,old['trainer'])
            best_saved=torch.load(best,map_location='cpu',weights_only=False);best_epoch=best_saved.get('current_epoch')
            require(type(best_epoch)is int and 0<best_epoch<=epoch and _LOGGER_FIELDS<=set(best_saved['logging']),
                'Actual whole BEST inside completed native prefix with all8logger arrays required')
            best_state=_state(best_saved,best_epoch,old['trainer'])
            require(set(saved['network_weights'])==set(best_saved['network_weights'])and all(
                saved['network_weights'][key].shape==best_saved['network_weights'][key].shape
                and saved['network_weights'][key].dtype==best_saved['network_weights'][key].dtype for key in saved['network_weights']),
                'Actual latest/BEST full network architecture differs')
            del saved,best_saved;gc.collect()
            require(sha(latest)==latest_sha and sha(best)==best_sha,'Whole native checkpoint changed during CPU validation')
            full_log=owned(oldroot/'train.log').read_bytes();prefix,prefix_record=_completed_prefix(full_log,epoch)
            continuation=dict(root=str(oldroot),request_sha256=sha(oldroot/'request.json'),native_sha256=sha(oldroot/'native.json'),
                preflight_sha256=sha(owned(oldroot/'preflight.json')),controller_identity=requests[gpu]['controller_identity'],
                checkpoint_path=str(latest),checkpoint_sha256=latest_sha,best_sha256=best_sha,completed_epochs=epoch)
            request.update(verified_native_continuation=continuation);request.pop('completed_native_resume',None)
            native=copy.deepcopy(old);derivative={k:old[k]for k in('bank','bank_sha256','source_bank','source_bank_sha256')if k in old}
            native.update(request);native.update(derivative);native.pop('completed_native_resume',None);native.pop('PBS_shared_cache_root',None)
            require(all(native.get(k)==old.get(k)for k in('native_root','archive_root','private_runtime','private_pre','private_data_folder',
                'trainer','baseline','split','bank','bank_sha256')),'Original mutable native/runtime/split/bank paths changed')
            proof=dict(format='basic_cp_reused_whole_native_common_cohort_v1',complete_epochs=epoch,completed_epochs=epoch,
                native_goal_epochs=250,remaining_epochs=250-epoch,latest=dict(path=str(latest),sha256=latest_sha,state=state),
                BEST=dict(path=str(best),sha256=best_sha,state=best_state),source_root=str(oldroot),
                continuation=continuation,training_prefix=prefix_record,full_history_sha256=sha(oldroot/'train.log'),
                checkpoint_bytes_copied=False,optimizer_and_scaler_preserved=True,partial_checkpoint_used=False)
            checkpoint_proofs[str(gpu)]=proof
        validate_request(request);pending.append((gpu,root,request,native,proof,prefix))
    require(all(sha(path)==digest for path,digest in{**input_files,**old_metadata}.items()),'Reviewed old immutable metadata changed during prepare')
    for actor in actors.values():released(actor)
    group.mkdir(mode=0o700,exist_ok=True)
    evidence=[]
    for gpu,root,request,native,proof,prefix in pending:
        root.mkdir(mode=0o700);oldroot=Path(requests[gpu]['root'])
        if gpu==5:
            # Source has no native yet; keep full future paths declared in the
            # request and only ensure its own storage admission directory exists.
            owned(request['native_root'],True)
        for name in('request.json','native.json','status.json','controller.log','candidate_bank.log'):
            source=oldroot/name
            if source.exists():evidence.append(copy_bytes(owned(source),root/('previous_exact_'+name)))
        if native is not None:
            for name in('preflight.json','actual_model.json','calibration.json'):evidence.append(copy_bytes(owned(oldroot/name),root/name))
            evidence.append(copy_bytes(owned(oldroot/'train.log'),root/'train_full_history_from_previous.log'))
            with(root/'train.log').open('xb')as stream:stream.write(prefix);stream.flush();os.fsync(stream.fileno())
            proof=dict(proof,clean_training_prefix_sha256=sha(root/'train.log'))
            seal(root/'resume.json',proof);seal(root/'own_checkpoint_resume.json',proof)
        seal(root/'pending_request.json',request)
        if native is not None:seal(root/'pending_native.json',native)
    seal(group/'prelaunch_whole_state_proof.json',dict(source_files_sha256={**input_files,**old_metadata},
        released_actors=release,checkpoint_proofs=checkpoint_proofs,evidence=evidence,
        fixed_aggregate_capacity_bytes=800*2**30,old_immutable_metadata_written=False,
        code_links_created=False,checkpoint_bytes_copied=False,bank_output_precreated=False,processes_signaled=False))
    rows={}
    for gpu,root,request,native,proof,prefix in pending:
        command=[sys.executable,'-B','-u',str(code/'tools/run_basic_cp_expanded.py'),'--request',str(root/'request.json')]
        environment=dict(os.environ,PYTHONPATH=str(code),PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES=str(gpu),
            OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        with(root/'controller.log').open('xb',buffering=0)as stream:
            child=subprocess.Popen(command,cwd=code,env=environment,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        process=psutil.Process(child.pid);process.cpu_affinity(CPUS[gpu]);actor=identity(process)
        require(actor['UID']==UID and actor['command']==command,'New exact controller execution identity differs')
        request.update(controller_identity=actor,controller_cgroup_admission=capture_cgroup_admission(process.pid))
        if native is not None:native.update({k:v for k,v in request.items()if k not in('bank','bank_sha256')});seal(root/'native.json',native)
        seal(root/'request.json',request)
        rows[str(gpu)]=dict(root=str(root),arm=ARMS[gpu],controller_identity=actor,request_sha256=sha(root/'request.json'))
    seal(registry,dict(format='basic_cp_shared_resource_registry_v1',registration=rows,allowed_previous_resource_registries={}))
    fresh=dict(manifest,group=str(group),code=str(code),commit=export['commit'],code_sha256=export['code_sha256'],registration=rows,
        shared_resource_registry=str(registry),previous_group=manifest.get('group'),shared_banks=str(banks),producer_root=str(producer),
        shared_candidate_group=str(group),host_lifetime=lifetime,retained_live_GPUs=[],newly_created_controllers=list(range(7)),
        resumed_native_GPUs=[0,1,2,3,4,6],resumed_producer_GPU=5,controller_processes_started=True,native_training_started=False,
        controller_code_exports={str(code):dict(path=str(args.code_export),sha256=args.code_export_sha256)},
        fixed_aggregate_capacity_bytes=800*2**30,one_current_common_registry=True,
        old_immutable_metadata_written=False,checkpoint_bytes_copied=False,code_links_created=False,processes_signaled=False)
    seal(group/'launch.json',fresh)
    require(all(sha(path)==digest for path,digest in{**input_files,**old_metadata}.items()),'Old immutable metadata changed while new common ledger published')
    print(json.dumps(dict(manifest=str(group/'launch.json'),commit=export['commit'],registration=rows,
        completed_epochs={gpu:proof['completed_epochs']for gpu,proof in checkpoint_proofs.items()},
        fixed_aggregate_GiB=800,one_current_common_registry=True,checkpoint_bytes_copied=False,processes_signaled=False)),flush=True)


if __name__=='__main__':main()
