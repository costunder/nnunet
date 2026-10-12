"""Independent copies and exact whole-arm recovery after observed natural exits.

No hardlinks, inode mutation, source guard weakening, signals or data uploads.
The four original FAILED monitors remain immutable; fresh monitored actors use
their own full checkpoints, complete shared input cache and full scientific plan.
"""
from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
import time

PATHS = {
    'hiercp_v1x/v25_terminal_descendant_resources.py',
    'hiercp_v1x/v25_all_arm_recovery_resources.py',
    'hiercp_v1x/v25_forward_memory_runtime.py',
    'tools/run_v25_execution.py',
    'tools/run_v25_all_arm_recovery_execution.py',
    'tools/run_v25_all_arm_recovery_continuation.py',
    'tools/watch_v25_all_arm_recovery.py',
    'tools/deploy_v25_all_arm_recovery.py',
    'tests/test_v25_terminal_descendant_resources.py',
    'tests/test_v25_terminal_empty_cmdline_race.py',
    'tests/test_v25_all_arm_recovery_execution.py',
    'tests/test_v25_forward_memory_runtime.py',
    'tests/test_v25_all_arm_recovery.py',
    'tests/test_v25_all_arm_recovery_viewer.py',
}
REQUIRED = {
    'hiercp_v1x/v25_terminal_descendant_resources.py',
    'hiercp_v1x/v25_all_arm_recovery_resources.py',
    'hiercp_v1x/v25_forward_memory_runtime.py',
    'tools/run_v25_execution.py',
    'tools/run_v25_all_arm_recovery_execution.py',
    'tools/run_v25_all_arm_recovery_continuation.py',
    'tools/watch_v25_all_arm_recovery.py',
    'tools/deploy_v25_all_arm_recovery.py',
}

REMOTE_SCRIPT = r'''
import ast,base64,hashlib,io,json,os,pathlib,psutil,shutil,stat,subprocess,sys,tarfile,time
group=pathlib.Path('/var/tmp/aicompetition06_v25_20261012/run_1791784719489478600')
oldcode=group/'code_U128_reuse_8ff98768793b';UID=41833
manifest=MANIFEST_DATA;phase=PHASE_DATA;allowed=ALLOWED_DATA;required=REQUIRED_DATA
suffix='_ALL_recovery_'+manifest['commit'][:12]
code=group/('code_ALL_recovery_'+manifest['commit'][:12])
stagepath=group/('ALL_recovery_code_deployed'+suffix+'.json')
intentpath=group/('ALL_recovery_controller_intent'+suffix+'.json')
launchpath=group/('ALL_recovery_controller_launch'+suffix+'.json')
sealpath=group/('ALL_recovery_continuation_seal'+suffix+'.json')
proofpaths={arm:group/(arm+'_direct_execution'+suffix+'.json')for arm in 'ABCD'}
cache=pathlib.Path('/dev/shm/aicompetition06_v25_20261012/run_1791784719489478600/shared_context_cache')
oldlaunchpath=group/'U128_reuse_controller_launch_U128_reuse_8ff98768793b.json'
oldsealpath=group/'U128_reuse_continuation_seal_U128_reuse_8ff98768793b.json'
OLD_COMMIT='8ff98768793b692cfbb0f3d180fbd4cb1b2609b1'
OLD_CONTROLLER=(2510561,1791811611.99)
PEER_CONTROLLER=(2553369,1791817518.53)
PEER_COMMIT='2db93521a9462b2209a8f25c1c84e0bce5ba1855'
peer_code=group/'code_ABD_recovery_2db93521a946'
peerlaunchpath=group/'ABD_recovery_controller_launch_ABD_recovery_2db93521a946.json'
peersealpath=group/'ABD_recovery_continuation_seal_ABD_recovery_2db93521a946.json'
LIVE_GPU0=(2460965,1791804951.51)
GPU0_ROOT=pathlib.Path('/var/tmp/aicompetition06_v24_20261010/GPU4_NNUNET_GPU0_20261012_1791804509808819200_canonical_1')
GPU0_REQUEST_SHA='ad641ae1bc1a362eeef405d9987b6eec67fcb17e48aa61357ef12a814acd028d'
GPU0_STORAGE_SHA='c724450584c757e70b92ca90117c15c8f690a64e1dd3f6e5f015abd09f67d88f'
MAPPING={'A':(6,list(range(32,38))),'B':(5,list(range(42,48))),'C':(1,list(range(16,22))),'D':(4,list(range(6)))}
FAILED_ACTORS={'A':(2553653,1791817545.92),'B':(2553654,1791817545.94),'C':(2510833,1791811642.45),'D':(2553655,1791817545.97)}

def sha(path):
 result=hashlib.sha256()
 with pathlib.Path(path).open('rb')as stream:
  for block in iter(lambda:stream.read(2**20),b''):result.update(block)
 return result.hexdigest()
def regular(path,directory=False):
 path=pathlib.Path(path)
 if not path.is_absolute()or '..'in path.parts or any(p.is_symlink()for p in (path,*path.parents)):
  raise ValueError('Absolute own non-symlink namespace required')
 info=path.lstat()
 if info.st_uid!=UID or not(stat.S_ISDIR(info.st_mode)if directory else stat.S_ISREG(info.st_mode)):
  raise ValueError('Exact owned regular file/directory required: '+str(path))
 return path
def publish(path,value):
 with path.open('x')as stream:
  json.dump(value,stream,indent=2,allow_nan=False);stream.flush();os.fsync(stream.fileno())
def same(path,digest):
 if sha(regular(path))!=digest:raise ValueError('Immutable file changed: '+str(path))
def metadata(path):
 info=regular(path).stat()
 return{key:getattr(info,key)for key in('st_dev','st_ino','st_mode','st_uid','st_gid','st_size','st_mtime_ns','st_ctime_ns','st_nlink')}
def copy_immutable(source,target,digest):
 before=metadata(source)
 with regular(source).open('rb')as stream,target.open('xb')as output:
  shutil.copyfileobj(stream,output,2**20);output.flush();os.fsync(output.fileno())
 shutil.copystat(source,target,follow_symlinks=False)
 same(source,digest);same(target,digest)
 if metadata(source)!=before:raise RuntimeError('Original source inode metadata changed during independent copy')
 if(source.stat().st_dev,source.stat().st_ino)==(target.stat().st_dev,target.stat().st_ino):
  raise RuntimeError('Independent new code inode required; original hardlinks forbidden')
 return before
def execution_profile(relative,explicit):
 relative=pathlib.PurePosixPath(relative)
 if str(relative)in explicit:return True
 # Only inactive repository documentation, bytecode and historical source
 # archives are excluded. All actual package/tool/config code is preserved.
 # v1's archive is a real runtime input and is ALWAYS copied in its entirety.
 if relative.parts[:2]==('versions','v1'):return True
 if '__pycache__'in relative.parts or relative.suffix=='.pyc':return False
 if relative.parts[0]in('docs','.git','outputs','tmp','DEBUG_owned_D'):return False
 if relative.parts[0]=='versions'and('history'in relative.parts or relative.suffix=='.zip'):return False
 return True
def actor(process):
 if process.status()==psutil.STATUS_ZOMBIE:raise RuntimeError('Live actor became zombie; no signal')
 result=dict(pid=process.pid,create_time=process.create_time(),command=process.cmdline(),UID=process.uids().real)
 if not result['command']or result['UID']!=UID:raise RuntimeError('Exact live own PID/birth/argv/UID required')
 return result
def exact(identity):
 process=psutil.Process(identity['pid'])
 if actor(process)!=identity:raise RuntimeError('PID/birth/UID/argv differs')
 return process
def released(identity):
 if set(identity)!={'pid','create_time','command','UID'}or identity['UID']!=UID:
  raise ValueError('Exact known own previous actor required')
 try:process=psutil.Process(identity['pid'])
 except psutil.NoSuchProcess:return
 if process.create_time()!=identity['create_time']or process.uids().real!=UID:
  raise RuntimeError('Previous PID reused; release not assumed')
 raise RuntimeError('Previous own actor still exists; no signal or restart')
def argument(command,flag):
 if command.count(flag)!=1 or command.index(flag)+1>=len(command):raise ValueError('Exact argument required: '+flag)
 return command[command.index(flag)+1]
def old_launch():
 value=json.loads(regular(oldlaunchpath).read_bytes())
 identity=dict(pid=value['PID'],create_time=value['create_time'],command=value['command'],UID=value['UID'])
 if (value['private_code']!=str(oldcode)or value['commit']!=OLD_COMMIT or value['seal']!=str(oldsealpath)
     or(identity['pid'],identity['create_time'])!=OLD_CONTROLLER):raise ValueError('Exact current U128 reuse controller required')
 released(identity)
 previous=json.loads(regular(oldsealpath).read_bytes())
 if previous['controller_identity']!=identity or previous['private_code']!=str(oldcode):raise ValueError('Exact old controller/seal binding required')
 return value,identity,previous
def protected_actors():
 old,controller,previous=old_launch()
 launch=json.loads(regular(peerlaunchpath).read_bytes());peer=json.loads(regular(peersealpath).read_bytes())
 identity=dict(pid=launch['PID'],create_time=launch['create_time'],command=launch['command'],UID=launch['UID'])
 if((identity['pid'],identity['create_time'])!=PEER_CONTROLLER or launch['commit']!=PEER_COMMIT
     or launch['private_code']!=str(peer_code)or launch['seal']!=str(peersealpath)
     or peer['controller_identity']!=identity or peer['private_code']!=str(peer_code)):
  raise RuntimeError('Exact previous ABD controller and immutable seal required')
 released(identity)
 gpu0=actor(psutil.Process(LIVE_GPU0[0]))
 if(gpu0['pid'],gpu0['create_time'])!=LIVE_GPU0 or not any('nnUNet' in token for token in gpu0['command']):
  raise RuntimeError('Exact live GPU0 nnU-Net actor required')
 return dict(previous_u128_controller=controller,previous_ABD_controller=identity,GPU0=gpu0)
def scientific(base):
 rows={}
 for arm in 'ABCD':
  request=json.loads(regular(group/arm/'request.json').read_bytes())
  if len(request['source_sha256'])!=23:raise ValueError('All 23 frozen scientific sources required')
  for name,digest in request['source_sha256'].items():same(base/name,digest)
  rows[arm]=request
 if any(rows[a]['source_sha256']!=rows['A']['source_sha256']for a in 'BCD'):
  raise ValueError('Frozen four-arm source contracts differ')
 return rows
def immutable_inputs(previous):
 paths=[cache/'local/index.json',cache/'input_bindings.json',group/'shared_preparation.json',
  group/'native_storage_admission.json',oldlaunchpath,oldsealpath,peerlaunchpath,peersealpath,group/'view_v25.py',
  GPU0_ROOT/'request.json',GPU0_ROOT/'storage_admission.json']
 for arm in 'ABCD':
  paths.append(pathlib.Path(previous['slots'][arm]['direct_proof']['path']))
  paths.extend(group/arm/name for name in('request.json','preparation.json','calibration.json','training/training_identity.json','monitor/status.json'))
 return{str(path):sha(regular(path))for path in paths}
def validate_archive():
 data=base64.b64decode(PAYLOAD_DATA)
 if hashlib.sha256(data).hexdigest()!=manifest['sha256']:raise ValueError('Reviewed archive changed')
 result=[];seen=set()
 with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz')as bundle:
  for member in bundle.getmembers():
   if member.isdir():continue
   relative=pathlib.PurePosixPath(member.name)
   if(not member.isfile()or relative.is_absolute()or '..'in relative.parts or member.name in seen or member.name not in allowed):
    raise ValueError('Only exact reviewed code overlay paths admitted')
   seen.add(member.name);proof=manifest['proof'][member.name];old=peer_code/member.name
   if(sha(regular(old))if old.exists()else None)!=proof['previous_sha256']:
    raise ValueError('Current U128 reuse file SHA differs: '+member.name)
   body=bundle.extractfile(member).read()
   if hashlib.sha256(body).hexdigest()!=proof['new_sha256']:raise ValueError('New committed file SHA differs: '+member.name)
   if relative.suffix=='.py':ast.parse(body.decode('utf8'),filename=member.name)
   result.append((member.name,body))
 if seen!=set(manifest['paths'])or not required<=seen:raise ValueError('Complete reviewed all-arm recovery overlay required')
 return result
def terminal_arm(arm,controller):
 status=json.loads(regular(group/arm/'monitor/status.json').read_bytes())
 identity=dict(pid=status.get('child_pid'),create_time=status.get('child_create_time'),command=status.get('child_command'),UID=status.get('UID'))
 previous_controller=controller if arm=='C'else protected_actors()['previous_ABD_controller']
 previous_code=oldcode if arm=='C'else peer_code
 previous_seal=oldsealpath if arm=='C'else peersealpath
 if(status.get('status')!='FAILED'or status.get('stage')!='train'or status.get('controller_pid')!=previous_controller['pid']
     or(identity['pid'],identity['create_time'])!=FAILED_ACTORS[arm]):
  raise RuntimeError('Exact own naturally FAILED '+arm+' training actor required')
 gpu,cpus=MAPPING[arm];command=identity['command']
 entry='run_v25_u128_reuse_execution.py'if arm=='C'else'run_v25_ad_execution.py'
 if(command[:4]!=['/home/aicompetition06/.conda/envs/nnunet/bin/python3.10','-B','-u',str(previous_code/'tools'/entry)]
     or any(argument(command,name)!=expected for name,expected in(
      ('--arm',arm),('--gpu',str(gpu)),('--mode','train'),('--output',str(group/arm)),
      ('--continuation-seal',str(previous_seal))))):raise ValueError('Exact old own FAILED stage binding required')
 released(identity)
 log=regular(status['stage_log'])
 if not log.is_relative_to(group/arm/'monitor'):raise ValueError('Actual own failure log namespace required')
 text=log.read_text()
 if 'Traceback (most recent call last)'not in text:raise ValueError('Actual own training failure traceback required')
 return status,identity,dict(path=str(log),sha256=sha(log))
def validate_checkpoint_binding(binding,arm):
 policy=binding['config']['v24_runtime']['curriculum'];initial={'A':8,'B':16,'C':32,'D':128}[arm]
 if(binding.get('epochs')!=40 or binding.get('workers')!=4 or binding.get('debug')is not False
     or binding.get('physical_patient_batch')!=4
     or len(binding.get('train_cases',[]))!=65 or len(binding.get('val_cases',[]))!=21
     or policy['total_epochs']!=40 or policy['total_u']!=128
     or policy['initial_u']!=initial or policy['increment_u']!=(None if arm=='D'else initial)):
  raise ValueError('Original arm full40/128U/65+21/physicalB4/worker4 curriculum contract required')
def checkpoint(arm):
 import torch
 from hiercp_v1x.u_bridge_training import digest
 from hiercp_v1x.v25_training import validate_curriculum_progress
 path=regular(group/arm/'training/checkpoint_latest.pt');before=path.stat()
 saved=torch.load(path,map_location='cpu',weights_only=False);after=path.stat()
 if(before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)!=(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns):
  raise RuntimeError('Failed arm latest checkpoint changed during CPU inspection')
 own=json.loads(regular(group/arm/'training/training_identity.json').read_bytes());state=saved['state']
 if(saved.get('format')!='v25_all_P_patient_balanced_shared_context_training_v1'
     or saved['identity_sha256']!=own['identity_sha256']or digest(own['binding'])!=own['identity_sha256']
     or saved['content_sha256']!=digest({key:value for key,value in saved.items()if key!='content_sha256'})
     or len(saved['rank_rng'])!=1 or 'shuffle_generator'not in saved
     or not all(key in saved for key in('model','optimizer','scheduler','scaler'))or state['status']!='RUNNING'):
  raise ValueError('Exact own model/optimizer/scheduler/scaler/RNG RUNNING checkpoint required')
 binding=own['binding']
 validate_checkpoint_binding(binding,arm)
 validate_curriculum_progress(state,policy=binding['config']['v24_runtime']['curriculum'],train_cases=binding['train_cases'],
  val_cases=binding['val_cases'],physical_patient_batch=binding['physical_patient_batch'])
 if binding['physical_patient_batch']!=4:raise ValueError('Original physical patient B4 required')
 result=dict(path=str(path),sha256=sha(path),identity_sha256=saved['identity_sha256'],content_sha256=saved['content_sha256'],
  epoch=state['epoch'],updates=state['updates'],phase=state['phase'],train_position=state['train_position'],
  evaluation_position=state['evaluation_position'],active_u=state['curriculum']['active_u'],status=state['status'])
 del saved;return result
def gpu0_storage_contract():
 request_path=regular(GPU0_ROOT/'request.json');same(request_path,GPU0_REQUEST_SHA)
 request=json.loads(request_path.read_bytes());proof=request['storage_admission']
 if(request.get('source_arm')!=4 or request.get('execution_physical_gpu')!=0
     or request.get('metadata_root')!=str(GPU0_ROOT)
     or proof!=dict(path=str(GPU0_ROOT/'storage_admission.json'),sha256=GPU0_STORAGE_SHA)):
  raise ValueError('Exact original running GPU0 storage request required')
 path=regular(proof['path']);same(path,proof['sha256']);document=json.loads(path.read_bytes())
 binding=document['gpu0_storage_rebinding'];archive=binding['persistent_archive_bytes']
 canonical=binding['canonical_tmpfs_reserve_bytes'];tmpfs=document['aggregate_budget']['required_free_bytes']
 if archive!=2**30 or canonical!=32*2**30 or tmpfs!=13862461217:
  raise ValueError('Original exact GPU0 persistent1GiB/canonical32GiB/native tmpfs admission required')
 return dict(persistent_archive_bytes=archive,canonical_tmpfs_reserve_bytes=canonical,GPU0_native_tmpfs_required_bytes=tmpfs,
  source_proofs={str(request_path):GPU0_REQUEST_SHA,str(path):GPU0_STORAGE_SHA},original_GPU0_admission_rewritten=False)
def extra_storage_admission(disk,backup_bytes,monitor_bytes,gpu0):
 if(type(backup_bytes)is not int or backup_bytes<=0 or type(disk.get('actual_free_bytes'))is not int
     or type(disk.get('required_free_bytes'))is not int or disk['actual_free_bytes']<0 or disk['required_free_bytes']<0
     or type(monitor_bytes)is not int or monitor_bytes<=0 or gpu0['persistent_archive_bytes']!=2**30):
  raise ValueError('Measured disk/complete checkpoint sizes required')
 if disk['required_free_bytes']<3*2**30:raise ValueError('Original 2GiB reserve and 1GiB growth must remain fully admitted')
 # The existing GPU0 archive is outside the four V25 arms and must be reserved
 # separately. Checkpoint backups go to admitted tmpfs, not this disk budget.
 needed=disk['required_free_bytes']+gpu0['persistent_archive_bytes']+monitor_bytes
 if disk['actual_free_bytes']<needed:raise OSError('Original V25/GPU0 persistent budgets plus actual failed monitors do not fit')
 return dict(original_admission=disk,checkpoint_backup_bytes=backup_bytes,checkpoint_backup_storage='tmpfs',
  failed_monitor_archive_bytes=monitor_bytes,additional_reserve_bytes=gpu0['persistent_archive_bytes'],
  existing_GPU0_persistent_archive_bytes=gpu0['persistent_archive_bytes'],GPU0_storage_contract=gpu0,
  original_2GiB_reserve_1GiB_growth_preserved=True,required_free_bytes=needed,actual_free_bytes=disk['actual_free_bytes'],
  original_storage_admission_rewritten=False)
def tmpfs_backup_admission(backup_bytes,tmpfs_free,available_RAM,gpu0):
 if(type(backup_bytes)is not int or backup_bytes<=0 or type(tmpfs_free)is not int or tmpfs_free<0
     or type(available_RAM)is not int or available_RAM<0 or gpu0['canonical_tmpfs_reserve_bytes']!=32*2**30
     or gpu0['GPU0_native_tmpfs_required_bytes']!=13862461217):raise ValueError('Exact measured tmpfs/RAM admission required')
 needed=gpu0['canonical_tmpfs_reserve_bytes']+gpu0['GPU0_native_tmpfs_required_bytes']+backup_bytes
 memory=416*2**30+backup_bytes
 if tmpfs_free<needed:raise OSError('Full shared canonical/GPU0 native tmpfs plus complete recovery backups do not fit')
 if available_RAM<memory:raise MemoryError('Full original416GiB RAM plus complete recovery backups do not fit')
 return dict(format='v25_tmpfs_recovery_backup_admission_v1',checkpoint_backup_bytes=backup_bytes,
  required_tmpfs_free_bytes=needed,actual_tmpfs_free_bytes=tmpfs_free,required_available_RAM_bytes=memory,
  actual_available_RAM_bytes=available_RAM,tmpfs_recovery_backup_not_persistent=True,
  original_primary_checkpoints_remain_on_disk=True,server_reboot_can_remove_backup=True,
  original_canonical_and_GPU0_tmpfs_budgets_preserved=True)
def stage_capacity(disk,code_bytes,overlay_bytes,gpu0):
 if any(type(value)is not int or value<0 for value in(code_bytes,overlay_bytes)):
  raise ValueError('Exact measured full execution code and overlay sizes required')
 # Separate deployment metadata room; original 2GiB reserve/1GiB growth and
 # the existing external GPU0 archive admission remain completely intact.
 metadata_reserve=2**20
 needed=disk['required_free_bytes']+gpu0['persistent_archive_bytes']+code_bytes+overlay_bytes+metadata_reserve
 if disk['actual_free_bytes']<needed:raise OSError('Independent complete execution code does not fit outside original native/GPU0 budgets')
 return dict(original_admission=disk,complete_execution_code_bytes=code_bytes,overlay_peak_bytes=overlay_bytes,
  deployment_metadata_reserve_bytes=metadata_reserve,required_free_bytes=needed,actual_free_bytes=disk['actual_free_bytes'],
  no_original_inode_linked_or_modified=True,original_native_and_GPU0_budgets_preserved=True)

if os.getuid()!=UID:raise ValueError('Exact account required')
regular(group,True);regular(oldcode,True);regular(peer_code,True);regular(cache,True)
requests=scientific(oldcode)
if set(manifest['paths']).intersection(requests['A']['source_sha256']):raise ValueError('Frozen23 scientific sources cannot change')
if phase=='stage':
 if stagepath.exists()or code.exists():raise FileExistsError('Staged namespace exists; inspect prior receipt instead of retry')
 old,controller,previous=old_launch();protected=protected_actors();admitted=validate_archive();preserved=immutable_inputs(previous)
 peer_previous=json.loads(regular(peersealpath).read_bytes())
 explicit=set(requests['A']['source_sha256'])|set(manifest['paths'])
 for contract,base in((previous,oldcode),(peer_previous,peer_code)):
  explicit.update(str(pathlib.Path(path).relative_to(base))for path in contract['runtime_files'])
 source_files=[];omitted=[]
 for folder,dirs,files in os.walk(peer_code,followlinks=False):
  folder=regular(pathlib.Path(folder),True)
  for name in dirs:regular(folder/name,True)
  for name in files:
   path=regular(folder/name);relative=path.relative_to(peer_code)
   if execution_profile(relative.as_posix(),explicit):source_files.append((path,relative,sha(path)))
   else:omitted.append(dict(path=str(relative),bytes=path.stat().st_size))
 selected={relative.as_posix()for source,relative,digest in source_files}
 original_required=explicit-set(manifest['paths'])
 if not original_required<=selected:raise ValueError('All original scientific/runtime source and v1 archive dependencies required')
 if 'versions/v1/pipeline_v1_source.zip'not in selected:raise ValueError('Whole v1 source archive is a required runtime input')
 sys.path.insert(0,str(peer_code))
 from hiercp_v1x.v25_native_storage import verify_admission,check_aggregate_disk
 verify_admission(group/'native_storage_admission.json',full_hash=False)
 capacity=stage_capacity(check_aggregate_disk(group/'native_storage_admission.json',group/'C'),
  sum(path.stat().st_size for path,relative,digest in source_files),sum(len(body)for name,body in admitted),gpu0_storage_contract())
 code.mkdir(exist_ok=False)
 oldfiles={};source_metadata={}
 for source,relative,digest in source_files:
  target=code/relative;target.parent.mkdir(parents=True,exist_ok=True)
  source_metadata[str(source)]=copy_immutable(source,target,digest);oldfiles[str(source)]=digest
 for name,body in admitted:
  target=code/name;target.parent.mkdir(parents=True,exist_ok=True);temporary=target.with_name(target.name+'.ALL_overlay.tmp')
  with temporary.open('xb')as stream:stream.write(body);stream.flush();os.fsync(stream.fileno())
  os.replace(temporary,target)
 scientific(code)
 for path,digest in preserved.items():same(path,digest)
 for path,digest in oldfiles.items():same(path,digest)
 for path,before in source_metadata.items():
  if metadata(path)!=before:raise RuntimeError('Original source metadata changed after code overlay')
 tests=[name[:-3].replace('/','.')for name in manifest['paths']if name.startswith('tests/')and name.endswith('.py')]
 if not tests:raise ValueError('Reviewed runtime regression tests required')
 env=os.environ.copy();env.update(PYTHONPATH=str(code),PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
 checked=subprocess.run([sys.executable,'-B','-m','unittest',*tests],cwd=code,env=env,
  stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=240)
 testlog=group/('ALL_recovery_CPU_tests'+suffix+'.log')
 with testlog.open('x')as stream:stream.write(checked.stdout);stream.flush();os.fsync(stream.fileno())
 if checked.returncode:raise RuntimeError('Private Linux CPU tests failed; training not started: '+str(testlog))
 renewed=protected_actors()
 if renewed!=protected:raise RuntimeError('Protected old controller/C/GPU0 identity changed during stage')
 result=dict(format='v25_ALL_terminal_recovery_private_stage_v1',phase='stage',commit=manifest['commit'],archive_sha256=manifest['sha256'],
  private_code=str(code),previous_code=str(oldcode),preserved_files=preserved,unchanged_old_code_files=oldfiles,protected_actors=protected,
  CPU_regression_tests=dict(command=checked.args,returncode=checked.returncode,log=str(testlog),sha256=sha(testlog)),
  source_metadata=source_metadata,execution_code_profile=dict(complete_runtime_code=True,whole_v1_source_archive_preserved=True,
   omitted_inactive_documentation_and_history_archives=omitted,model_graph_dataset_or_epochs_changed=False),stage_capacity=capacity,
  frozen23_sources_unchanged=True,existing_code_files_modified=False,signals_sent=False,training_started=False,
  hardlinks_created=False,original_inode_metadata_preserved=True)
 publish(stagepath,result);print(json.dumps(result))
elif phase=='start':
 if any(path.exists()for path in(intentpath,launchpath,sealpath,*proofpaths.values())):
  raise FileExistsError('all-arm launch intent exists; inspect exact actor instead of retry')
 if(group/'view_v25_current.py').exists():raise FileExistsError('Existing current viewer must be preserved; inspect before launch')
 staged=json.loads(regular(stagepath).read_bytes())
 if(staged['commit']!=manifest['commit']or staged['archive_sha256']!=manifest['sha256']
     or staged['CPU_regression_tests']['returncode']!=0):raise ValueError('Exact reviewed CPU-tested staged code required')
 same(staged['CPU_regression_tests']['log'],staged['CPU_regression_tests']['sha256'])
 for path,digest in staged['preserved_files'].items():same(path,digest)
 for path,digest in staged['unchanged_old_code_files'].items():same(path,digest)
 for path,before in staged['source_metadata'].items():
  if metadata(path)!=before:raise RuntimeError('Original source metadata changed since independent stage')
 for name,proof in manifest['proof'].items():same(code/name,proof['new_sha256'])
 scientific(code);old,previous_controller,previous=old_launch();protected=protected_actors()
 if protected!=staged['protected_actors']:raise RuntimeError('Protected controller/C/GPU0 identity changed after stage')
 statuses={};actors={};logs={}
 for arm in 'ABCD':statuses[arm],actors[arm],logs[arm]=terminal_arm(arm,previous_controller)
 for arm in 'ABCD':
  if(group/arm/'training/STOP_AFTER_BATCH').exists():raise RuntimeError('Existing own stop marker requires independent inspection; no silent archive')
 sys.path.insert(0,str(code))
 from hiercp_v1x.v25_direct_resources import capture_cgroup_admission,_pbs_start_time,validate_direct_document
 from hiercp_v1x.v25_native_storage import verify_admission,check_aggregate_disk
 from hiercp_v1x import v25_all_arm_recovery_resources as resources
 verify_admission(group/'native_storage_admission.json',full_hash=False)
 checkpoints={arm:checkpoint(arm)for arm in 'ABCD'}
 disk=check_aggregate_disk(group/'native_storage_admission.json',group/'B');gpu0storage=gpu0_storage_contract()
 backup_bytes=sum(regular(checkpoints[arm]['path']).stat().st_size for arm in 'ABCD')
 monitor_bytes=sum(regular(group/arm/'monitor/status.json').stat().st_size for arm in 'ABCD')
 storage=extra_storage_admission(disk,backup_bytes,monitor_bytes,gpu0storage)
 tmpfs_admission=tmpfs_backup_admission(backup_bytes,shutil.disk_usage(cache).free,psutil.virtual_memory().available,gpu0storage)
 oldproofs={}
 for arm in 'ABCD':
  proof=previous['slots'][arm]['direct_proof'];same(proof['path'],proof['sha256'])
  oldproofs[arm]=validate_direct_document(json.loads(regular(proof['path']).read_bytes()))
 aggregate=oldproofs['A']['aggregate_CPU_affinity']
 if not set(aggregate)<=set(psutil.Process().cpu_affinity()):raise RuntimeError('Actual allowed CPUs do not admit original 24-core set')
 for arm,(gpu,cpus)in MAPPING.items():
  row=oldproofs[arm]
  if(row['controller_identity']!=previous_controller or row['gpu']!=gpu or row['CPU_affinity']!=cpus
      or row['aggregate_CPU_affinity']!=aggregate):raise ValueError('Original exact per-arm GPU/sixCPU proof required')
  values=subprocess.check_output(['nvidia-smi','--id='+str(gpu),'--query-gpu=uuid,memory.free',
   '--format=csv,noheader,nounits'],text=True,timeout=20).strip().split(',')
  if values[0].strip()!=row['GPU_UUID']or float(values[1])<40*1024:
   raise RuntimeError('Exact idle own physical GPU UUID/free40GiB required')
 apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader,nounits'],text=True,timeout=20)
 wanted={oldproofs[arm]['GPU_UUID']for arm in 'ABCD'}
 if any(line.strip()and line.split(',')[-1].strip()in wanted for line in apps.splitlines()):
  raise RuntimeError('Requested GPU1/4/5/6 still has a live CUDA actor; preserve it')
 owned=json.loads(regular(group/'allocation_request.json').read_bytes())
 jobs=json.loads(subprocess.check_output(['/opt/pbs/bin/qstat','-f','-F','json',owned['job_id']],text=True,timeout=20))['Jobs']
 if list(jobs)!=[owned['job_id']]:raise RuntimeError('Exact owned host lifetime unavailable')
 job=next(iter(jobs.values()));h,m,s=map(int,job['Resource_List']['walltime'].split(':'))
 expiry=_pbs_start_time(job['stime'])+h*3600+m*60+s
 if(job.get('job_state')!='R'or job.get('Job_Owner','').split('@')[0]!='aicompetition06'
     or abs(expiry-owned['expires_at_unix'])>1 or time.time()>=expiry-180):raise RuntimeError('Current owned host lifetime failed')
 cgroup=capture_cgroup_admission(os.getpid())
 if cgroup!=oldproofs['C']['cgroup_admission']:raise RuntimeError('Actual new controller cgroup differs from protected C admission')
 if any(row['limit_bytes']is not None and row['limit_bytes']<416*2**30 for row in cgroup['memory_limits']):
  raise MemoryError('Unchanged 416GiB cgroup admission failed')
 if psutil.virtual_memory().available<416*2**30:raise MemoryError('Actual available RAM cannot admit unchanged full416GiB')
 command=list(old['command']);entry=str(oldcode/'tools/run_v25_u128_reuse_continuation.py')
 if command.count(entry)!=1:raise ValueError('Exact current controller entry required')
 command[command.index(entry)]=str(code/'tools/run_v25_all_arm_recovery_continuation.py')
 if argument(command,'--group')!=str(group)or argument(command,'--input-cache')!=str(cache):
  raise ValueError('Original full shared cache/group bindings required')
 command[command.index('--execution-manifest')+1]=str(sealpath)
 archive=group/'code_revisions'/('ALL_recovery_before_launch_'+manifest['commit'][:12])
 archive.mkdir(parents=True,exist_ok=False);monitors={};backups={}
 backup_parent=cache.parent/'recovery_checkpoint_backups'
 if backup_parent.exists():regular(backup_parent,True)
 else:backup_parent.mkdir(exist_ok=False)
 backup_root=backup_parent/('ALL_'+manifest['commit'][:12]);backup_root.mkdir(exist_ok=False)
 regular(backup_root,True)
 for arm in 'ABCD':
  monitor=group/arm/'monitor/status.json';current_status,current_actor,current_log=terminal_arm(arm,previous_controller)
  if current_status!=statuses[arm]or current_actor!=actors[arm]or current_log!=logs[arm]:
   raise RuntimeError('Own failed status/log/actor changed before backup')
  copied=archive/(arm+'_failed_status.json')
  with copied.open('xb')as stream:stream.write(regular(monitor).read_bytes());stream.flush();os.fsync(stream.fileno())
  same(copied,sha(monitor));monitors[arm]=dict(path=str(copied),sha256=sha(copied))
  source=regular(checkpoints[arm]['path']);target=backup_root/(arm+'_checkpoint_latest.pt')
  with source.open('rb')as stream,target.open('xb')as output:
   for block in iter(lambda:stream.read(2**20),b''):output.write(block)
   output.flush();os.fsync(output.fileno())
  same(source,checkpoints[arm]['sha256']);same(target,checkpoints[arm]['sha256'])
  backups[arm]=dict(path=str(target),sha256=checkpoints[arm]['sha256'])
 if protected_actors()!=protected or gpu0_storage_contract()!=gpu0storage:
  raise RuntimeError('Exact protected live actors/GPU0 storage contract changed before intent')
 for arm in 'ABCD':released(actors[arm]);same(checkpoints[arm]['path'],checkpoints[arm]['sha256'])
 # Admit the complete previous scientific provenance and exact failure types
 # BEFORE a detached controller is created. The current inspection actor is
 # used only as a distinct read-only identity in this preflight declaration.
 preflight=dict(group_root=str(group),private_code=str(code),controller_identity=actor(psutil.Process()),
  previous_seals={'u128':dict(path=str(oldsealpath),sha256=sha(oldsealpath)),
   'ABD':dict(path=str(peersealpath),sha256=sha(peersealpath))},
  source_scientific23={str(code/name):digest for name,digest in requests['A']['source_sha256'].items()},
  slots={arm:dict(arm=arm,gpu=MAPPING[arm][0],previous_actor=actors[arm],
   failed_monitor=monitors[arm],failure_log=logs[arm])for arm in 'ABCD'})
 previous_release_admission=resources.verify_initial_release(preflight)
 intent=dict(format='v25_ALL_terminal_recovery_launch_intent_v1',command=command,private_code=str(code),commit=manifest['commit'],
  checkpoints=checkpoints,previous_actors=actors,protected_actors=protected,checkpoint_backups=backups,
  storage_admission=storage,tmpfs_backup_admission=tmpfs_admission,selected_arms=['A','B','C','D'],created_at=time.time(),signals_sent=False)
 publish(intentpath,intent)
 env=os.environ.copy();env.update(PYTHONPATH=str(code),PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
 log=group/('ALL_recovery_pipeline_controller'+suffix+'.log')
 with log.open('xb')as stream:
  child=subprocess.Popen(command,cwd=code,env=env,stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
   start_new_session=True,preexec_fn=lambda:os.sched_setaffinity(0,aggregate))
 process=psutil.Process(child.pid);deadline=time.monotonic()+5
 while process.cmdline()!=command and time.monotonic()<deadline:time.sleep(.05)
 identity=actor(process)
 if identity['command']!=command:raise RuntimeError('Exact new all-arm controller not observed; inspect durable intent')
 newcgroup=capture_cgroup_admission(process.pid)
 if newcgroup!=cgroup:raise RuntimeError('New exact all-arm controller cgroup changed; inspect durable intent')
 for arm in 'ABCD':
  value=dict(oldproofs[arm],controller_identity=identity,resource_guard_sha256=sha(code/'hiercp_v1x/v25_direct_resources.py'),cgroup_admission=newcgroup)
  validate_direct_document(value);publish(proofpaths[arm],value)
 peer_previous=json.loads(regular(peersealpath).read_bytes())
 runtime_names={str(pathlib.Path(path).relative_to(peer_code))for path in peer_previous['runtime_files']}
 runtime_names.update(('hiercp_v1x/v25_all_arm_recovery_resources.py','hiercp_v1x/v25_forward_memory_runtime.py',
  'tools/run_v25_all_arm_recovery_execution.py','tools/run_v25_all_arm_recovery_continuation.py','tools/watch_v25_all_arm_recovery.py'))
 preserved=dict(staged['preserved_files'])
 preserved.update({str(code/name):digest for name,digest in requests['A']['source_sha256'].items()})
 for arm in 'ABCD':
  for proof in(monitors[arm],logs[arm],backups[arm]):preserved[proof['path']]=proof['sha256']
 slots={arm:dict(arm=arm,gpu=MAPPING[arm][0],CPU_affinity=MAPPING[arm][1],
  direct_proof=dict(path=str(proofpaths[arm]),sha256=sha(proofpaths[arm])),checkpoint=checkpoints[arm],
  previous_actor=actors[arm],failed_monitor=monitors[arm],failure_log=logs[arm],
  monitor_path=str(group/arm/'monitor'/('ALL_recovery_'+manifest['commit'][:12])/'status.json'))for arm in 'ABCD'}
 seal=dict(format=resources.FORMAT,selected_arms=['A','B','C','D'],group_root=str(group),private_code=str(code),
  controller_identity=identity,previous_seals={'u128':dict(path=str(oldsealpath),sha256=sha(oldsealpath)),
   'ABD':dict(path=str(peersealpath),sha256=sha(peersealpath))},slots=slots,
  preserved_files=preserved,runtime_files={str(code/name):sha(regular(code/name))for name in sorted(runtime_names)},
  source_scientific23={str(code/name):digest for name,digest in requests['A']['source_sha256'].items()},created_at=time.time(),
  scientific_configuration_unchanged=True,no_process_signaled=True)
 resources.validate_seal(seal);resources.verify_initial_release(seal)
 publish(sealpath,seal)
 viewer=group/'view_v25_current.py'
 viewer_body=("from pathlib import Path\nimport sys\n"
  "GROUP=Path("+repr(str(group))+")\nCODE=Path("+repr(str(code))+")\n"
  "SEAL=Path("+repr(str(sealpath))+")\nSHA="+repr(sha(sealpath))+"\n"
  "sys.path.insert(0,str(CODE))\nfrom tools.watch_v25_all_arm_recovery import main\n"
  "main(['--group',str(GROUP),'--recovery-seal',str(SEAL),'--recovery-seal-sha256',SHA,*sys.argv[1:]])\n")
 ast.parse(viewer_body)
 with viewer.open('x')as stream:stream.write(viewer_body);stream.flush();os.fsync(stream.fileno())
 result=dict(phase='start',PID=identity['pid'],create_time=identity['create_time'],UID=UID,command=command,log=str(log),private_code=str(code),
  seal=str(sealpath),proofs={arm:str(proofpaths[arm])for arm in 'ABCD'},commit=manifest['commit'],checkpoints=checkpoints,
  GPUs=[6,5,1,4],CPU_affinity={arm:MAPPING[arm][1]for arm in 'ABCD'},protected_actors=protected,checkpoint_backups=backups,
  storage_admission=storage,tmpfs_backup_admission=tmpfs_admission,full40_GNN=True,full680_updates=True,physical_patient_batch=4,full250_native=True,own_BEST26=True,
  viewer=dict(path=str(viewer),sha256=sha(viewer)),all_original_FAILED_monitors_preserved=True,workers=4,
  hardlinks_created=False,original_inode_metadata_preserved=True,
  previous_release_admission=previous_release_admission,
  frozen23_sources_unchanged=True,scientific_configuration_unchanged=True,signals_sent=False,debug=False)
 publish(launchpath,result);print(json.dumps(result))
else:raise ValueError('Explicit stage/start required')
'''


def build_remote_script(manifest, payload, phase):
    if phase not in ('stage', 'start'):
        raise ValueError('Explicit stage/start required')
    if not re.fullmatch('[0-9a-f]{40}', manifest.get('commit', '')):
        raise ValueError('Exact reviewed Git commit required')
    if hashlib.sha256(payload).hexdigest() != manifest.get('sha256'):
        raise ValueError('Reviewed archive bytes differ')
    paths = manifest.get('paths', [])
    proof = manifest.get('proof', {})
    if len(paths) != len(set(paths)) or set(paths) != set(proof) or not REQUIRED <= set(paths) <= PATHS:
        raise ValueError('Exact reviewed all-arm execution overlay required')
    for row in proof.values():
        if set(row) != {'previous_sha256', 'new_sha256'} or not re.fullmatch('[0-9a-f]{64}', row['new_sha256']):
            raise ValueError('Exact before/new SHA proofs required')
        if row['previous_sha256'] is not None and not re.fullmatch('[0-9a-f]{64}', row['previous_sha256']):
            raise ValueError('Exact original SHA or explicit absent file required')
    return REMOTE_SCRIPT.replace('MANIFEST_DATA', repr(manifest)).replace('PHASE_DATA', repr(phase)).replace(
        'ALLOWED_DATA', repr(PATHS)).replace('REQUIRED_DATA', repr(REQUIRED)).replace(
        'PAYLOAD_DATA', repr(base64.b64encode(payload).decode()))


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if len(arguments) != 2:
        raise ValueError('Provide exact reviewed manifest and stage/start')
    manifest = json.loads(Path(arguments[0]).resolve(strict=True).read_text(encoding='utf8'))
    script = build_remote_script(manifest, Path(manifest['archive']).read_bytes(), arguments[1])
    here = Path('D:/AI project/nnunet/outputs/v24_gpu6_recovery_20261011')
    spec = importlib.util.spec_from_file_location('secure_remote', here/'remote_once.py')
    remote = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(remote)
    value = json.loads(remote.run(script, 300))
    path = here/('V25_ALL_RECOVERY_'+arguments[1].upper()+'_'+str(time.time_ns())+'.json')
    with path.open('x', encoding='utf8') as stream:
        json.dump(value, stream, indent=2)
    keys = ('phase', 'commit', 'PID', 'create_time', 'private_code', 'checkpoints', 'GPUs', 'CPU_affinity', 'signals_sent', 'CPU_regression_tests')
    print(json.dumps(dict(receipt=str(path), observed={key: value[key] for key in keys if key in value}), ensure_ascii=False))


if __name__ == '__main__':
    main()
