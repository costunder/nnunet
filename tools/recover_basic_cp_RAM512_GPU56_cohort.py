"""Fresh logical cohort on physical GPU5/6 after exact own actor release.

Scientific native checkpoints, source banks and folds are preserved in place.
Only fresh operational metadata and exact own controller children are created.
Completed blend50 remains complete; six unfinished full experiments share GPU5/6.
"""
import argparse
import ast
import copy
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time

import psutil

UID=41833
BASE=Path('/dev/shm/aicompetition06_basic_cp_ablation_20261013')
# Set only after exact current/target receipts have been validated in main().
OLDGROUP=None
GROUP=None
OLD_ARMS={}
OLD_GPU_FOR_NEW={}
ARMS={}
SHARING_DATA=None
SHARING_FIELD='GPU56_sharing_receipt'
INITIAL_ARMS={0:'multi5',1:'blend50',2:'multi10',3:'candidates512',4:'candidates1024',5:'candidates2048',6:'multi7'}
EXECUTION_GPUS={0:5,1:None,2:6,3:5,4:6,5:5,6:6}
PLACEMENT_OPERATION='resume_multi7_GPU1_on_idle_GPU6'
QUEUE_PLACEMENT_OPERATION='queue_multi5_after_candidates2048_and_candidates1024_after_multi7'


def placement_mapping(current_arms,target_arms):
    """Logical experiment identity remains unchanged when CUDA devices are shared."""
    require(current_arms==target_arms==INITIAL_ARMS,
        'GPU56 sharing preserves all seven existing logical experiment slots and arms')
    return {gpu:gpu for gpu in range(7)}


def validate_complete_retained(states):
    require(states.get('1',{}).get('state')=='FULL_COMPLETE'
        and states['1'].get('arm')=='blend50',
        'Existing blend50 full250/BEST26/metrics/archive must remain complete without retraining')
    require(all(states.get(str(g),{}).get('state')=='TRAINING'for g in EXECUTION_GPUS if g!=1),
        'This migration admits exactly the six measured unfinished native experiments')


FIELDS=dict(memory_budget_revision=512,aggregate_memory_cap_GiB=512,
    effective_native_RSS_limit_GiB=32,effective_shared_bank_RSS_limit_GiB=320,
    owned_tmpfs_envelope_GiB=144,post_source_owned_tmpfs_envelope_GiB=272,auxiliary_RAM_envelope_GiB=16)
NEW_RUNTIME_FILES={'hiercp_v1x/basic_cp_512_claims.py','hiercp_v1x/basic_cp_512_resources.py',
    'hiercp_v1x/basic_cp_large_bank_RAM512.py','tools/run_basic_cp_RAM512.py','hiercp_v1x/basic_cp_gpu_placement.py',
    'hiercp_v1x/basic_cp_GPU_queue_dependencies.py','hiercp_v1x/basic_cp_ablation_resources.py'}
NEW_OPERATIONAL_FILES=NEW_RUNTIME_FILES|{'tools/watch_basic_cp_RAM512.py','tools/recover_basic_cp_common_cohort.py',
    'tools/recover_basic_cp_RAM512_cohort.py','tools/rebind_basic_cp_RAM512_cache.py','tools/watch_basic_cp_admission.py',
    'tools/release_basic_cp_RAM512_native_at_epoch.py','tools/release_basic_cp_RAM512_source.py',
    'tools/preserve_basic_cp_RAM512_partial_validation.py',
    'tools/adopt_basic_cp_RAM512_cache_transfer.py','tools/hold_basic_cp_RAM512_transfer.py',
    'tools/watch_basic_cp_RAM512_successor_queue.py','tools/recover_basic_cp_RAM512_successor_cohort.py',
    'tools/basic_cp_RAM512_completed_log_adapter.py','tools/recover_basic_cp_RAM512_GPU56_cohort.py',
    'hiercp_v1x/basic_cp_GPU56_sharing.py'}
OPERATIONAL_CODE_CHANGES={'tools/persist_basic_cp_seven.py','hiercp_v1x/basic_cp_shared_bank_resources_v6.py',
    'tools/recover_basic_cp_common_cohort.py','tools/watch_basic_cp_admission.py',
    'hiercp_v1x/basic_cp_512_claims.py','hiercp_v1x/basic_cp_512_resources.py','hiercp_v1x/basic_cp_group_resources.py',
    'hiercp_v1x/basic_cp_gpu_placement.py','tools/run_basic_cp_RAM512.py','tools/watch_basic_cp_RAM512.py','hiercp_v1x/basic_cp_ablation_resources.py',
    'hiercp_v1x/basic_cp_GPU56_sharing.py'}
NEW_RUNTIME_FILES.add('hiercp_v1x/basic_cp_GPU56_sharing.py')


def require(condition,message):
    if not condition:raise ValueError(message)


def read(path):return json.loads(Path(path).read_bytes())


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb')as stream:
        initial=os.fstat(stream.fileno())
        for block in iter(lambda:stream.read(8*2**20),b''):digest.update(block)
        final=os.fstat(stream.fileno())
    require(all(getattr(initial,k)==getattr(final,k)for k in('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')),
        'Held source evidence changed while hashing')
    return digest.hexdigest()


def original_code(request,manifest):
    command=request['controller_identity']['command']
    require(len(command)==6 and command[-2:]==['--request',str(Path(request['root'])/'request.json')],
        'Exact original controller code/root command required')
    code=Path(command[3]).parents[1]
    descriptor=manifest.get('controller_code_exports',{}).get(str(code))
    require(isinstance(descriptor,dict)and set(descriptor)=={'path','sha256'}and sha(descriptor['path'])==descriptor['sha256'],
        'Actual original controller code export must remain explicitly SHA named')
    export=read(descriptor['path'])
    require(export.get('code')==str(code)and export.get('commit')==request['commit']
        and export.get('code_sha256')==request['code_sha256'],'Original immutable code proof/request differs')
    return code,copy.deepcopy(descriptor)


def audit_code_delta(request,oldcode,export):
    old=request['code_sha256'];new=export['code_sha256'];newcode=Path(export['code'])
    require(set(old)<=set(new)and set(new)-set(old)<=NEW_OPERATIONAL_FILES,
        'Only the explicit RAM512 runtime modules and same-UI viewer may be added')
    require(NEW_RUNTIME_FILES<=set(new),'Complete new RAM512 runtime code required')
    eol={}
    for name,digest in old.items():
        require(sha(oldcode/name)==digest and sha(newcode/name)==new[name],'Actual old/new export code SHA differs')
        if digest==new[name]or name in OPERATIONAL_CODE_CHANGES:continue
        before=(oldcode/name).read_bytes();after=(newcode/name).read_bytes()
        require(before.replace(b'\r\n',b'\n')==after.replace(b'\r\n',b'\n')
            and ast.dump(ast.parse(before),include_attributes=False)==ast.dump(ast.parse(after),include_attributes=False),
            'Scientific/model/data source must remain byte-exact except proven EOL: '+name)
        eol[name]=dict(old_sha256=digest,new_sha256=new[name],normalized_bytes_and_AST_equal=True)
    return eol


def classification(*,latest_epoch,final_epoch,best26,metrics,complete_archive):
    """Explicit full250 roles; never call a completed BEST a latest checkpoint."""
    require(type(latest_epoch)is int and 0<latest_epoch<=250,'Whole actual latest/final completed epoch required')
    require(final_epoch is None or(type(final_epoch)is int and final_epoch==250),'Actual FINAL must prove full250')
    require(all(type(v)is bool for v in(best26,metrics,complete_archive)),'Actual stage proof flags required')
    if final_epoch is None:
        require(latest_epoch<250 and not best26 and not metrics and not complete_archive,
            'Incomplete latest cannot claim full evaluation or final completion')
        return 'TRAINING'
    require(not metrics or best26,'Historical metrics cannot precede actual BEST26')
    require(not complete_archive or(metrics and best26),'Full archive cannot precede actual250/BEST26/metrics')
    if complete_archive:return 'FULL_COMPLETE'
    return 'BEST26_READY'if best26 else'FINAL250_PENDING_BEST26'


def retarget(request,gpu,physical_request,root,registry,export,banks,producer,pending_reference,placement_reference,sharing_reference):
    result=copy.deepcopy(request)
    result.update(FIELDS,root=str(root),gpu=gpu,gpu_uuid=physical_request['gpu_uuid'],cpu_affinity=physical_request['cpu_affinity'],
        GPU_memory_total_MiB=physical_request['GPU_memory_total_MiB'],
        GPU_placement_receipt=copy.deepcopy(placement_reference),shared_resource_registry=str(registry),
        RAM512_claim_ledger=str(root.parent/'RAM512_allocation_ledger.json'),
        RAM512_source_pending_receipt=copy.deepcopy(pending_reference),owned_tmpfs_root=str(BASE),
        code_sha256=export['code_sha256'],commit=export['commit'])
    result.pop('PBS_shared_cache_root',None)
    result.pop('GPU_queue_deferred_transfer',None)
    require(isinstance(SHARING_DATA,dict),'Independently verified GPU56 sharing receipt required')
    physical=EXECUTION_GPUS[gpu]
    result.update(GPU56_sharing_receipt=copy.deepcopy(sharing_reference),
        execution_physical_GPU=physical,
        execution_GPU_UUID=None if physical is None else SHARING_DATA['execution_GPU_UUIDs'][str(physical)])
    if'shared_banks'in result:result['shared_banks']=str(banks)
    if'producer_root'in result:result['producer_root']=str(producer)
    require(result['gpu']==gpu and result['arm']==ARMS[gpu],'Same actual scientific arm/GPU must remain')
    return result



def validate_full_source_selection(request,document,base,*,verifier=None):
    """Re-admit every original sealed archive; no candidate actor or copy."""
    require(document.get('format')=='basic_cp_RAM512_full_source_selection_v1'
        and document.get('candidate_counts')=={'512':642,'1024':642,'2048':642},
        'All three actual full642 source prefixes required')
    require(all(document.get(key)is False for key in('scientific_scale_changed','candidate_generation_executed',
        'source_files_written','processes_signaled')),'Read-only full-source selection required')
    banks=base.owned(document['shared_banks'],True)
    require(str(banks)==request['shared_banks'] and banks.parent==Path(request['root']).parent,
        'Reuse exact original source request bank root without relocation')
    expected_source=Path(request['root'])/'request.json'
    require(document['source_request']==dict(path=str(expected_source),sha256=sha(base.owned(expected_source))),
        'Full banks must remain bound to exact released source request')
    require(document['binding']==dict(path=request['candidate_binding'],sha256=request['candidate_binding_sha256']),
        'Original source binding path/SHA unchanged')
    require(sha(base.owned(document['binding']['path']))==document['binding']['sha256'],
        'Actual original candidate binding changed')
    for name in('source_manifest','source_actor_release'):
        reference=document[name];require(isinstance(reference,dict)and set(reference)=={'path','sha256'}
            and sha(base.owned(reference['path']))==reference['sha256'],'Exact source provenance differs: '+name)
    source_manifest=read(document['source_manifest']['path'])
    row=source_manifest['registration']['5']
    require(row['root']==request['root']and row['request_sha256']==document['source_request']['sha256']
        and row['controller_identity']==request['controller_identity'],'Original source registration differs')
    source_actors=base.extract_identities(read(document['source_actor_release']['path']))
    require(bool(source_actors),'Nonempty actual original source release cohort required')
    for actor in source_actors:base.released(actor)
    files=document.get('original_files_sha256');require(isinstance(files,dict)and files,'All source metadata SHA pins required')
    require(str(banks/'complete.json')in files,'Actual global full-source completion receipt must be pinned')
    for path,digest in files.items():require(sha(base.owned(path))==digest,'Actual selected source evidence changed: '+path)
    complete=read(banks/'complete.json')
    require(complete.get('complete')is True and complete.get('full_sources')==642
        and complete.get('CUDA_initialized')is False and set(complete.get('prefixes',{}))=={'512','1024','2048'},
        'Actual completed CPU-only shared source bank required')
    require(set(document['witnesses'])==set(document['full_admissions'])=={'512','1024','2048'},
        'All exact prefix witnesses and full admissions required')
    if verifier is None:
        from hiercp_v1x.basic_cp_large_bank import verify_prepared_bank
        verifier=verify_prepared_bank
    admitted={}
    for count in(512,1024,2048):
        witness=document['witnesses'][str(count)]
        require(witness['index']==str(banks/str(count)/'index.json'),'Exact already-sealed family index required')
        actual=verifier(witness,Path(request['candidate_binding']),request['candidate_binding_sha256'],candidate_count=count)
        require(actual==document['full_admissions'][str(count)]and actual.get('complete')is True
            and actual.get('source_entries')==642 and actual.get('candidate_count')==count
            and actual.get('all642_entry_sha_verified')is True and actual.get('bank_regenerated')is False,
            'Independent original all642 source admission differs')
        admitted[str(count)]=actual
    require(all(sha(base.owned(path))==digest for path,digest in files.items()),'Released source changed during full admission')
    return admitted


def _load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module);return module


def main():
    global OLDGROUP,GROUP,OLD_ARMS,OLD_GPU_FOR_NEW,ARMS,SHARING_DATA
    parser=argparse.ArgumentParser();parser.add_argument('--execute',action='store_true')
    parser.add_argument('--new-group',type=Path,required=True)
    for name in('current-manifest','release-proof','full-source-proof','code-export','placement-receipt','sharing-receipt'):
        parser.add_argument('--'+name,type=Path,required=True);parser.add_argument('--'+name+'-sha256',required=True)
    args=parser.parse_args();require(args.execute,'Prepared migration requires parent explicit --execute')
    require(os.getuid()==UID and socket.gethostname().split('.')[0]=='ece-a6gpu6','Exact owned actual node/UID required')
    GROUP=args.new_group
    require(GROUP.is_absolute()and GROUP.parent==BASE and GROUP.name.startswith('Basic_CP_RAM512_')
        and re.fullmatch(r'Basic_CP_RAM512_[A-Za-z0-9_]+',GROUP.name),
        'Explicit fresh owned direct-child RAM512 metadata group required')
    inputs={}
    for name in('current_manifest','release_proof','full_source_proof','code_export','placement_receipt','sharing_receipt'):
        path=getattr(args,name);digest=getattr(args,name+'_sha256')
        require(re.fullmatch('[0-9a-f]{64}',digest)and sha(path)==digest,'Exact reviewed input SHA differs: '+name)
        inputs[str(path)]=digest
    manifest,release,source,export=[read(getattr(args,n))for n in('current_manifest','release_proof','full_source_proof','code_export')]
    OLDGROUP=Path(manifest['group'])
    require(args.current_manifest==OLDGROUP/'launch.json'and OLDGROUP.parent==BASE and OLDGROUP!=GROUP,
        'Exact current actual cohort must differ from fresh destination')
    require(set(manifest['registration'])==set(map(str,range(7))),'Current all-seven registration required')
    OLD_ARMS={gpu:manifest['registration'][str(gpu)]['arm']for gpu in range(7)}
    placement=read(args.placement_receipt)
    require(args.placement_receipt.parent==GROUP and set(placement)=={'format','operation','physical_GPU_arms'}
        and placement['format']=='basic_cp_approved_physical_GPU_placement_v1'
        and placement['operation']in(PLACEMENT_OPERATION,QUEUE_PLACEMENT_OPERATION),
        'Exact unchanged logical placement receipt required')
    require(set(placement['physical_GPU_arms'])==set(map(str,range(7))),'All seven target physical bindings required')
    ARMS={int(gpu):arm for gpu,arm in placement['physical_GPU_arms'].items()}
    OLD_GPU_FOR_NEW=placement_mapping(OLD_ARMS,ARMS)
    require(args.sharing_receipt.parent==GROUP,'Fresh same-group physical GPU56 sharing receipt required')
    SHARING_DATA=read(args.sharing_receipt)
    sharing_reference=dict(path=str(args.sharing_receipt),sha256=args.sharing_receipt_sha256)
    placement_reference=dict(path=str(args.placement_receipt),sha256=args.placement_receipt_sha256)
    code=Path(export['code'])
    require(code==GROUP/'code'and re.fullmatch('[0-9a-f]{40}',export['commit']),'Fresh exact RAM512 immutable code export required')
    require(NEW_RUNTIME_FILES<=set(export['code_sha256']),'All four new approved RAM512 modules required')
    for name,digest in export['code_sha256'].items():require(sha(code/name)==digest,'Immutable fresh code SHA differs')
    sys.path.insert(0,str(code))
    base=_load_module('_RAM512_frozen_common_builder',code/'tools/recover_basic_cp_common_cohort.py')
    from hiercp_v1x.basic_cp_seven_resources import CPUS,capture_cgroup_admission,validate_request
    from hiercp_v1x.basic_cp_verified_continuation import _state,_completed_prefix
    from hiercp_v1x.basic_cp_ablation_resume import fold
    from hiercp_v1x.basic_cp_512_claims import Ledger
    from hiercp_v1x.basic_cp_512_resources import ProcessResolver
    from hiercp_v1x.basic_cp_GPU56_sharing import validate_sharing_request
    from tools import run_basic_cp_RAM512 as runtime
    import torch
    base.owned(GROUP,True)
    require(not(GROUP/'launch.json').exists()and not(GROUP/'shared_resource_registry.json').exists()
        and not(GROUP/'banks').exists()and not(GROUP/'RAM512_allocation_ledger.json').exists()
        and all(not(GROUP/('GPU'+str(g)+'_'+ARMS[g])).exists()for g in range(7)),
        'Only fresh new metadata roots/ledger/bank output may be published')
    require(set(manifest['registration'])==set(map(str,range(7)))and(
        release.get('all_old_pipeline_actors_terminal')is True or release.get('all_old_native_and_DA_terminal')is True),
        'All seven original rows and actual full old actor release proof required')
    actors={}
    for actor in base.extract_identities(release):actors[(actor['pid'],actor['create_time'])]=actor
    requests={};natives={};old_metadata={};oldcodes={};code_eol={}
    for gpu in range(7):
        row=manifest['registration'][str(gpu)];root=base.owned(row['root'],True)
        request_path=base.owned(root/'request.json');request=read(request_path)
        require(sha(request_path)==row['request_sha256']and request['controller_identity']==row['controller_identity']
            and request['gpu']==gpu and request['arm']==OLD_ARMS[gpu],'Old actual controller/request/arm/GPU differs')
        oldcode,proof=original_code(request,manifest);base.owned(oldcode,True)
        oldcodes[gpu]=(oldcode,proof);code_eol[str(gpu)]=audit_code_delta(request,oldcode,export)
        requests[gpu]=request;old_metadata[str(request_path)]=row['request_sha256']
        control=row['controller_identity'];actors[(control['pid'],control['create_time'])]=control
        for receipt in root.glob('*_actor.json'):
            for actor in base.extract_identities(read(base.owned(receipt))):
                require(actors.get((actor['pid'],actor['create_time']))==actor,'Release receipt lacks exact own stage actor')
        if(root/'native.json').exists():
            natives[gpu]=read(base.owned(root/'native.json'));old_metadata[str(root/'native.json')]=sha(root/'native.json')
    require(set(natives)==set(range(7)),'All seven released existing native experiments required')
    released=[base.released(actor)for actor in actors.values()]
    needles={str(Path(q['root'])/name)for q in requests.values()for name in('request.json','native.json')}
    for process in psutil.process_iter():
        try:
            if process.pid==os.getpid()or process.uids().real!=UID:continue
            if any(arg in needles for arg in process.cmdline()):
                require(process.status()in(psutil.STATUS_ZOMBIE,psutil.STATUS_DEAD),'Exact old request/native actor still live')
        except psutil.NoSuchProcess:continue
    source_reference=source['source_request']
    require(isinstance(source_reference,dict)and set(source_reference)=={'path','sha256'}and sha(base.owned(source_reference['path']))==source_reference['sha256'],'Pinned original G7 released CPU source request required')
    source_request=read(source_reference['path'])
    source_admission=validate_full_source_selection(source_request,source,base)
    banks=Path(source['shared_banks'])
    counts=copy.deepcopy(source['candidate_counts']);initial_pending=False
    pending_path=GROUP/'source_pending_bootstrap.json'
    pending_value=dict(format='basic_cp_RAM512_sealed_source_pending_v1',source_initially_pending=False,
        candidate_counts=counts,full_source_selection=dict(path=str(args.full_source_proof),sha256=args.full_source_proof_sha256),
        full_source_admissions=source_admission,all_old_source_actors_released=True,actual_old_actors=released,
        workers=24,source_reserved_bytes=0,CPU_source_actor_created=False,source_files_written=False,
        candidate_generation_executed=False,scientific_scale_preserved=True)
    pending_bytes=(json.dumps(pending_value,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode()
    pending_reference=dict(path=str(pending_path),sha256=hashlib.sha256(pending_bytes).hexdigest())
    lifetime=requests[0]['host_lifetime'];witness=read(base.owned(lifetime['path']))
    require(all(q['host_lifetime']==lifetime for q in requests.values())and sha(lifetime['path'])==lifetime['sha256'],
        'All seven require same actual truthful current owned host lifetime')
    job=json.loads(subprocess.check_output(['/opt/pbs/bin/qstat','-f','-F','json',witness['job_id']],text=True,timeout=20))['Jobs'][witness['job_id']]
    require(job['job_state']=='R'and job['Job_Owner'].split('@')[0]=='aicompetition06'
        and all(h.split('/')[0]=='ece-a6gpu6'for h in job['exec_host'].split('+'))
        and time.time()<lifetime['expires_at_unix']-180,'Actual owned same-host R lifetime/180second reserve required')
    registry=GROUP/'shared_resource_registry.json';producer=GROUP/next('GPU'+str(g)+'_'+a for g,a in ARMS.items()if a=='candidates2048')
    prepared=[];states={};checkpoint_files={}
    for gpu in range(7):
        oldgpu=OLD_GPU_FOR_NEW[gpu]
        root=GROUP/('GPU'+str(gpu)+'_'+ARMS[gpu]);oldq=requests[oldgpu]
        request=retarget(oldq,gpu,requests[gpu],root,registry,export,banks,producer,pending_reference,placement_reference,sharing_reference)
        for name in base.SCIENCE_KEYS:
            if name not in('gpu','gpu_uuid','cpu_affinity','GPU_queue_deferred_transfer'):require(request.get(name)==oldq.get(name),'Original science/resource legacy field changed: '+name)
        require(request['gpu_uuid']==requests[gpu]['gpu_uuid']and request['cpu_affinity']==CPUS[gpu],'Exact physical GPU UUID and CPU affinity required')
        native=None;state=None;prefix=None
        if gpu in range(7):
            old=natives[oldgpu];oldroot=Path(requests[oldgpu]['root']);base.validate_model(read(base.owned(oldroot/'actual_model.json')),old)
            goal=base.owned(fold(old),True);latest=goal/'checkpoint_latest.pth';final=goal/'checkpoint_final.pth';best=base.owned(goal/'checkpoint_best.pth')
            selected=base.owned(final if final.exists()else latest);selected_sha=sha(selected);best_sha=sha(best)
            saved=torch.load(selected,map_location='cpu',weights_only=False);epoch=saved.get('current_epoch')
            require(type(epoch)is int and 0<epoch<=250,'Original whole native completed epoch required')
            selected_state=_state(saved,epoch,old['trainer']);del saved;gc.collect()
            best_saved=torch.load(best,map_location='cpu',weights_only=False);best_epoch=best_saved.get('current_epoch')
            require(type(best_epoch)is int and 0<best_epoch<=epoch,'Original own BEST epoch must lie in whole prefix')
            best_state=_state(best_saved,best_epoch,old['trainer']);del best_saved;gc.collect()
            require(sha(selected)==selected_sha and sha(best)==best_sha,'Released original checkpoint bytes changed during decode')
            checkpoint_files[str(selected)]=selected_sha;checkpoint_files[str(best)]=best_sha
            training=oldroot/'training_and_BEST26_complete.json';best_receipt=goal/'best_validation_complete.json'
            best26=training.exists();metrics=oldroot/'evaluation/completion.json';status_path=oldroot/'status.json'
            old_status=read(status_path)if status_path.exists()else{}
            full_complete=old_status.get('status')=='COMPLETE'
            metrics_complete=False;files={str(selected):selected_sha,str(best):best_sha}
            validator=None
            if best26:
                require(epoch==250 and final.exists()and best_receipt.exists(),'BEST26 receipt requires real FINAL250')
                if old.get('RAM512_completed_best_validator')is not None:
                    validator=copy.deepcopy(old['RAM512_completed_best_validator'])
                    prior=runtime._prior_validator(old)
                    require(prior is not None and validator['receipt']==dict(path=str(best_receipt),sha256=sha(base.owned(best_receipt))),
                        'Original already-admitted BEST26 validator/fold/inode receipt must remain')
                    files[validator['export']['path']]=validator['export']['sha256']
                else:
                    oldcode,codeproof=oldcodes[oldgpu]
                    validator=dict(code=str(oldcode),export=codeproof,
                        receipt=dict(path=str(best_receipt),sha256=sha(base.owned(best_receipt))))
                    prior=_load_module('hiercp_v1x._RAM512_builder_priorBEST_'+str(gpu),oldcode/'hiercp_v1x/v24_native_best_eval_runtime.py')
                request['RAM512_completed_best_validator']=validator
                files[str(training)]=sha(base.owned(training));files[str(best_receipt)]=sha(best_receipt)
                prior.validate_best_validation_receipt(best_receipt,fold=goal,expected_cases=old['split']['outer_val'],bank_sha256=old['bank_sha256'])
                completed=read(training)
                require(completed['epochs']==250 and completed['best_sha256']==best_sha
                    and completed['final_training_witness_sha256']==selected_sha
                    and sorted(completed['validation_cases'])==sorted(old['split']['outer_val']),
                    'Original complete250/BEST26 proof differs from unchanged checkpoints/case set')
                if metrics.exists():
                    completion=read(base.owned(metrics));require(completion.get('complete')is True and completion.get('actual_data_evaluation')is True,
                        'Partial historical metrics requires fresh dedicated reviewed output')
                    from tools import render_nnunet_result_schema_v1 as renderer
                    online=renderer._online({'directory':str(oldroot/'evaluation'),'experiment':old['arm'],'status':'VERIFIED_COMPLETE'},read(renderer.SCHEMA_PATH))
                    require(online['status']=='VERIFIED_COMPLETE','All real historical evaluator output hashes required')
                    six=read(base.owned(oldroot/'evaluation/six_metric_comparison.json'))
                    require(six.get('historical_basic_reproduced')is True and six.get('patient_count')==26
                        and six.get('checkpoint')=='BEST'and Path(six['prediction_override'])==goal/'validation',
                        'All original26 BEST historical metrics and Basic CP comparison must be complete')
                    metrics_complete=True;files[str(metrics)]=sha(metrics)
            else:
                require(not best_receipt.exists(),'Completed BEST receipt without full250 linkage requires explicit repair')
                if epoch==250 and(goal/'validation').exists():
                    require(not any((goal/'validation').iterdir()),'Partial BEST predictions preserved; interrupted validation needs explicit fresh-output repair')
            mode=classification(latest_epoch=epoch,final_epoch=epoch if final.exists()else None,
                best26=best26,metrics=metrics_complete,complete_archive=full_complete)
            archive_relocation=None
            if not full_complete and Path(old['archive_root']).exists():
                old_archive=base.owned(old['archive_root'],True)
                new_archive=Path(old['native_root'])/('completed_native_archive_'+GROUP.name)
                require(not new_archive.exists(),'Fresh preserved-partial archive destination already exists')
                request['archive_root']=str(new_archive)
                archive_relocation=dict(previous_preserved=str(old_archive),fresh_independent_destination=str(new_archive),
                    reason='Existing partial archive preserved; no result overwritten',scientific_scale_changed=False)
            if full_complete:
                require(old_status.get('completed_epochs')==250 and old_status.get('evaluated_cases')==26
                    and old_status.get('historical_metrics_complete')is True,'Actual full COMPLETE status required')
                archive=old_status['archive'];archive_path=base.owned(archive['path'])
                require(sha(archive_path)==archive['sha256']and read(archive_path).get('complete')is True,
                    'Actual unchanged full independent archive receipt required')
                files[str(status_path)]=sha(status_path);files[str(archive_path)]=sha(archive_path)
                for report in old_status['reports'].values():
                    require(sha(base.owned(report['path']))==report['sha256'],'Actual completed result report SHA differs')
                    files[report['path']]=report['sha256']
            native=copy.deepcopy(old);derivative={k:old[k]for k in('bank','bank_sha256','source_bank','source_bank_sha256')if k in old}
            native.update(request);native.update(derivative);native.pop('PBS_shared_cache_root',None)
            native.pop('GPU_queue_deferred_transfer',None)
            if request.get('GPU_queue_deferred_transfer')is not None:
                native['GPU_queue_deferred_transfer']=copy.deepcopy(request['GPU_queue_deferred_transfer'])
            if validator is not None:native['RAM512_completed_best_validator']=validator
            require(all(native.get(k)==old.get(k)for k in('native_root','private_runtime','private_pre','private_data_folder',
                'trainer','baseline','split','bank','bank_sha256')),'Original actual native/runtime/model/split/bank path changed')
            require(native['archive_root']==old['archive_root']or archive_relocation is not None,
                'Only an explicit fresh independent archive IO path may differ')
            state=dict(format='basic_cp_RAM512_migration_state_v1',gpu=gpu,arm=ARMS[gpu],native_root=old['native_root'],trainer=old['trainer'],
                state=mode,completed_epochs=epoch,evaluated_cases=26 if best26 else 0,historical_metrics_complete=metrics_complete,
                source_root=str(oldroot),source_files_sha256=files,selected_checkpoint=dict(path=str(selected),sha256=selected_sha,state=selected_state),
                original_BEST=dict(path=str(best),sha256=best_sha,state=best_state),checkpoint_bytes_copied=False,old_results_written=False)
            if archive_relocation is not None:state['archive_IO_relocation']=archive_relocation
            if full_complete:state['completed_result']=dict(reports=old_status['reports'],archive=old_status['archive'])
            state_path=root/'RAM512_migration_state.json';encoded=(json.dumps(state,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode()
            reference=dict(path=str(state_path),sha256=hashlib.sha256(encoded).hexdigest())
            request['RAM512_migration_state']=reference;native['RAM512_migration_state']=reference
            full_log=base.owned(oldroot/'train.log').read_bytes();prefix,prefix_record=_completed_prefix(full_log,epoch)
            state['training_prefix']=prefix_record
            # Recompute after adding prefix provenance; this is the actual sealed body.
            encoded=(json.dumps(state,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode()
            reference['sha256']=hashlib.sha256(encoded).hexdigest();states[str(gpu)]=state
        validate_request(request);validate_sharing_request(request);prepared.append((gpu,root,request,native,state,prefix))
    validate_complete_retained(states)
    require(all(sha(path)==digest for path,digest in{**inputs,**old_metadata,**checkpoint_files}.items()),'Old source metadata/checkpoints changed before new metadata prepare')
    for actor in actors.values():base.released(actor)
    with pending_path.open('xb')as stream:stream.write(pending_bytes);stream.flush();os.fsync(stream.fileno())
    require(sha(pending_path)==pending_reference['sha256'],'Actual pending-source bootstrap SHA differs')
    evidence=[]
    for gpu,root,request,native,state,prefix in prepared:
        root.mkdir(mode=0o700);oldroot=Path(requests[OLD_GPU_FOR_NEW[gpu]]['root'])
        for name in('request.json','native.json','status.json','controller.log','candidate_bank.log'):
            if(oldroot/name).exists():evidence.append(base.copy_bytes(base.owned(oldroot/name),root/('previous_exact_'+name)))
        if native is not None:
            for name in('preflight.json','actual_model.json','calibration.json'):
                evidence.append(base.copy_bytes(base.owned(oldroot/name),root/name))
            evidence.append(base.copy_bytes(base.owned(oldroot/'train.log'),root/'train_full_history_from_previous.log'))
            if(oldroot/'train_full_history_from_previous.log').exists():
                evidence.append(base.copy_bytes(base.owned(oldroot/'train_full_history_from_previous.log'),root/'train_original_full_history_before_G6.log'))
            with(root/'train.log').open('xb')as stream:stream.write(prefix);stream.flush();os.fsync(stream.fileno())
            with(root/'RAM512_migration_state.json').open('xb')as stream:
                stream.write((json.dumps(state,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode());stream.flush();os.fsync(stream.fileno())
            require(sha(root/'RAM512_migration_state.json')==request['RAM512_migration_state']['sha256'],'Sealed migration role SHA differs')
            if state['state']in('BEST26_READY','FULL_COMPLETE'):
                evidence.append(base.copy_bytes(base.owned(oldroot/'training_and_BEST26_complete.json'),root/'training_and_BEST26_complete.json'))
                if(oldroot/'result_evidence.json').exists():
                    name='result_evidence.json'if state['state']=='FULL_COMPLETE'else'previous_exact_result_evidence.json'
                    evidence.append(base.copy_bytes(base.owned(oldroot/'result_evidence.json'),root/name))
                if state['historical_metrics_complete']:
                    import shutil
                    shutil.copytree(oldroot/'evaluation',root/'evaluation',copy_function=shutil.copy2)
                    for source_file in(oldroot/'evaluation').rglob('*'):
                        if source_file.is_file():require(sha(base.owned(source_file))==sha(root/'evaluation'/source_file.relative_to(oldroot/'evaluation')),'Historical metrics independent evidence copy differs')
        base.seal(root/'pending_request.json',request)
        if native is not None:base.seal(root/'pending_native.json',native)
    base.seal(GROUP/'prelaunch_whole_state_proof.json',dict(source_files_sha256={**inputs,**old_metadata,**checkpoint_files},
        released_actors=released,checkpoint_proofs=states,evidence=evidence,code_EOL_proofs=code_eol,
        effective_total_owned_RAM_cap_bytes=512*2**30,source_pending_bootstrap=pending_reference,
        old_immutable_metadata_written=False,code_links_created=False,checkpoint_bytes_copied=False,processes_signaled=False))
    rows={}
    for gpu,root,request,native,state,prefix in prepared:
        command=[sys.executable,'-B','-u',str(code/'tools/run_basic_cp_RAM512.py'),'--request',str(root/'request.json')]
        environment=dict(os.environ,PYTHONPATH=str(code),PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES=request['execution_GPU_UUID']or'',
            OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        with(root/'controller.log').open('xb',buffering=0)as stream:
            child=subprocess.Popen(command,cwd=code,env=environment,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        process=psutil.Process(child.pid);process.cpu_affinity(CPUS[gpu]);actor=base.identity(process)
        require(actor['UID']==UID and actor['command']==command,'New exact own controller command differs')
        # Retain an actual actor immediately, including if later request/cgroup publication fails.
        base.seal(root/'controller_spawned_actor.json',actor)
        request.update(controller_identity=actor,controller_cgroup_admission=capture_cgroup_admission(process.pid))
        if native is not None:
            native.update({k:v for k,v in request.items()if k not in('bank','bank_sha256')});base.seal(root/'native.json',native)
        base.seal(root/'request.json',request)
        rows[str(gpu)]=dict(root=str(root),arm=ARMS[gpu],controller_identity=actor,request_sha256=sha(root/'request.json'))
    base.seal(registry,dict(format='basic_cp_shared_resource_registry_v1',registration=rows,allowed_previous_resource_registries={}))
    # Entire source bank is sealed and released. Bootstrap creates NO CPU source
    # claim; all native allocations retain strict real GPU/RAM admission.
    sample=read(GROUP/('GPU0_'+ARMS[0])/'request.json')
    binding=dict(registry_path=str(registry),registry_sha256=sha(registry),request_sha256={g:r['request_sha256']for g,r in rows.items()},
        host_lifetime_sha256=lifetime['sha256'],expires_at_unix=lifetime['expires_at_unix'],owned_tmpfs_root=str(BASE),source_initially_pending=False)
    Ledger(sample,binding,ProcessResolver(None)).bootstrap(source_pending_receipt=pending_reference)
    fresh=dict(manifest,group=str(GROUP),code=str(code),commit=export['commit'],code_sha256=export['code_sha256'],registration=rows,
        shared_resource_registry=str(registry),previous_group=manifest['group'],shared_banks=str(banks),producer_root=str(producer),
        shared_candidate_group=str(banks.parent),host_lifetime=lifetime,retained_live_GPUs=[],newly_created_controllers=list(range(7)),GPU_placement_receipt=placement_reference,GPU56_sharing_receipt=sharing_reference,execution_physical_GPUs=copy.deepcopy(SHARING_DATA['execution_physical_GPUs']),physical_GPU_arms={str(gpu):arm for gpu,arm in ARMS.items()},
        resumed_native_GPUs=list(range(7)),resumed_producer_GPU=None,source_banks_reused_without_producer=True,controller_processes_started=True,native_training_started=False,
        controller_code_exports={str(code):dict(path=str(args.code_export),sha256=args.code_export_sha256)},
        effective_total_owned_RAM_cap_bytes=512*2**30,one_current_common_registry=True,
        RAM512_source_pending_receipt=pending_reference,RAM512_claim_ledger=str(GROUP/'RAM512_allocation_ledger.json'),
        old_immutable_metadata_written=False,checkpoint_bytes_copied=False,code_links_created=False,processes_signaled=False)
    fresh.pop('GPU_queue_deferred_transfer',None)
    base.seal(GROUP/'launch.json',fresh)
    require(all(sha(path)==digest for path,digest in{**inputs,**old_metadata,**checkpoint_files}.items()),'Original released data/results/checkpoint bytes changed during new metadata launch')
    print(json.dumps(dict(manifest=str(GROUP/'launch.json'),commit=export['commit'],effective_total_RAM_GiB=512,
        completed_epochs={g:s['completed_epochs']for g,s in states.items()},completion_modes={g:s['state']for g,s in states.items()},
        source_pending=False,source_reserved_GiB=0,one_current_common_registry=True,processes_signaled=False,
        physical_GPU_arms={str(gpu):arm for gpu,arm in ARMS.items()},GPU_placement_receipt=placement_reference,GPU56_sharing_receipt=sharing_reference,
        execution_physical_GPUs=copy.deepcopy(SHARING_DATA['execution_physical_GPUs']),
        original_checkpoints_written=False,CUDA_training_started_claimed=False)),flush=True)


if __name__=='__main__':
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from tools.basic_cp_RAM512_completed_log_adapter import private_builder_main
    adapter=Path(__file__).with_name('basic_cp_RAM512_completed_log_adapter.py')
    private_builder_main(sys.modules[__name__],adapter_path=adapter,adapter_sha256=sha(adapter))()
