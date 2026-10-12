"""Reviewed U128 validation-reuse private stage/start; no signals and no live source mutation.

Building the script is local and side-effect free. The root explicitly runs
    stage, records exact own terminal failures or batch pauses, verifies release, then start.
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
    'hiercp_v1x/v25_u128_validation_reuse.py',
    'hiercp_v1x/v25_u128_reuse_resources.py',
    'tools/run_v25_u128_reuse_execution.py', 'tools/run_v25_u128_reuse_continuation.py',
    'tools/deploy_v25_u128_reuse.py',
    'tests/test_v25_u128_validation_reuse.py',
    'tests/test_v25_u128_reuse_execution.py',
    'tests/test_v25_u128_reuse_deployment.py',
}
REQUIRED = PATHS
REMOTE_SCRIPT = r'''
import ast,base64,hashlib,io,json,os,pathlib,psutil,stat,subprocess,sys,tarfile,time
group=pathlib.Path('/var/tmp/aicompetition06_v25_20261012/run_1791784719489478600')
oldcode=group/'code_RECOVERY_provider_ab23c1622c56';UID=41833
manifest=MANIFEST_DATA;phase=PHASE_DATA;allowed=ALLOWED_DATA;required=REQUIRED_DATA
code=group/('code_U128_reuse_'+manifest['commit'][:12]);suffix='_U128_reuse_'+manifest['commit'][:12]
stagepath=group/('U128_reuse_code_deployed'+suffix+'.json');intentpath=group/('U128_reuse_controller_intent'+suffix+'.json')
globalintentpath=group/'U128_validation_reuse_controller_intent.json'
launchpath=group/('U128_reuse_controller_launch'+suffix+'.json');sealpath=group/('U128_reuse_continuation_seal'+suffix+'.json')
proofpaths={arm:group/(arm+'_direct_execution'+suffix+'.json') for arm in 'ABCD'}
cache=pathlib.Path('/dev/shm/aicompetition06_v25_20261012/run_1791784719489478600/shared_context_cache')
oldlaunchpath=group/'FINAL_controller_launch_RECOVERY_provider_ab23c1622c56.json'
oldsealpath=group/'FINAL_continuation_seal_RECOVERY_provider_ab23c1622c56.json'
OLD_COMMIT='ab23c1622c566e9865a29ca4a5b0ad3e4232c54f'
PREVIOUS_TRAINING={'A':(2489145,1791808660.16),'B':(2489075,1791808659.79),
 'C':(2489077,1791808659.81),'D':(2489076,1791808659.79)}

def sha(path):
 d=hashlib.sha256()
 with pathlib.Path(path).open('rb') as stream:
  for block in iter(lambda:stream.read(2**20),b''):d.update(block)
 return d.hexdigest()
def regular(path,directory=False):
 path=pathlib.Path(path)
 if not path.is_absolute() or '..'in path.parts or any(p.is_symlink() for p in (path,*path.parents)):
  raise ValueError('Absolute own non-symlink namespace required')
 s=path.lstat()
 if s.st_uid!=UID or not(stat.S_ISDIR(s.st_mode) if directory else stat.S_ISREG(s.st_mode)):
  raise ValueError('Exact owned regular file/directory required: '+str(path))
 return path
def publish(path,value):
 with path.open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.flush();os.fsync(stream.fileno())
def same(path,digest):
 if sha(regular(path))!=digest:raise ValueError('Immutable file changed: '+str(path))
def actor(p):
 if p.status()==psutil.STATUS_ZOMBIE:raise RuntimeError('Live actor is zombie')
 return dict(pid=p.pid,create_time=p.create_time(),command=p.cmdline(),UID=p.uids().real)
def exact(identity):
 p=psutil.Process(identity['pid'])
 if actor(p)!=identity:raise RuntimeError('PID/birth/UID/argv differs')
 return p
def released(identity):
 if set(identity)!={'pid','create_time','command','UID'} or identity['UID']!=UID:
  raise ValueError('Exact known own previous actor required')
 try:p=psutil.Process(identity['pid'])
 except psutil.NoSuchProcess:return
 if p.create_time()!=identity['create_time'] or p.uids().real!=UID:
  raise RuntimeError('Previous PID reused; release not assumed')
 if p.status()==psutil.STATUS_ZOMBIE:raise RuntimeError('Known own previous actor awaits reaping; inspect later, no signal')
 raise RuntimeError('Own previous actor still active; no signal or restart')
def old_launch(*,active):
 old=json.loads(regular(oldlaunchpath).read_bytes())
 if old['private_code']!=str(oldcode) or old['commit']!=OLD_COMMIT or old['seal']!=str(oldsealpath):
  raise ValueError('Exact current ab23 recovery launch required')
 identity=dict(pid=old['PID'],create_time=old['create_time'],command=old['command'],UID=old['UID'])
 if identity['pid']!=2488819 or identity['create_time']!=1791808638.21:raise ValueError('Exact current recovery controller required')
 if active is None:
  try:exact(identity)
  except psutil.NoSuchProcess:released(identity)
 elif active:exact(identity)
 else:released(identity)
 current=json.loads(regular(oldsealpath).read_bytes())
 if current['controller_identity']!=identity or current['private_code']!=str(oldcode):raise ValueError('Exact original seal/controller binding required')
 return old,identity
def scientific(base):
 rows={}
 for arm in 'ABCD':
  request=json.loads(regular(group/arm/'request.json').read_bytes())
  if len(request['source_sha256'])!=23:raise ValueError('All23 frozen sources required')
  for name,digest in request['source_sha256'].items():same(base/name,digest)
  rows[arm]=request
 if any(rows[arm]['source_sha256']!=rows['B']['source_sha256'] for arm in 'ACD'):raise ValueError('Frozen four-arm source contracts differ')
 return rows
def inputs():
 names=[cache/'local/index.json',cache/'input_bindings.json',group/'shared_preparation.json',group/'native_storage_admission.json',oldlaunchpath,oldsealpath]
 current=json.loads(regular(oldsealpath).read_bytes())
 for arm in 'ABCD':names.append(pathlib.Path(current['slots'][arm]['direct_proof']['path']))
 for arm in 'ABCD':names += [group/arm/name for name in ('request.json','preparation.json','calibration.json','training/training_identity.json')]
 # Mutable latest checkpoints are never in the runtime immutable table.
 return {str(p):sha(regular(p)) for p in names}
def stage_binding(command,arm,mode):
 if command[:4]!=['/home/aicompetition06/.conda/envs/nnunet/bin/python3.10','-B','-u',str(oldcode/'tools/run_v25_recovery_execution.py')]:
  raise ValueError('Exact current sealed wrapper entry required')
 previous=json.loads(regular(oldsealpath).read_bytes())
 expected={'--arm':arm,'--mode':mode,'--output':str(group/arm),'--continuation-seal':str(oldsealpath),
  '--allocation-request':previous['slots'][arm]['direct_proof']['path']}
 for flag,value in expected.items():
  if command.count(flag)!=1 or command.index(flag)+1>=len(command) or command[command.index(flag)+1]!=value:
   raise ValueError('Exact previous per-arm stage binding required: '+flag)

def terminal_arm(arm):
 if arm not in 'ABCD':raise ValueError('Exact prior arm required')
 status=json.loads(regular(group/arm/'monitor/status.json').read_bytes())
 if status.get('status')!='FAILED':raise RuntimeError('Exact prior '+arm+' actor has not naturally reached terminal failure')
 identity=dict(pid=status['child_pid'],create_time=status['child_create_time'],command=status['child_command'],UID=status['UID'])
 stage=status.get('stage')
 if stage not in ('train','scoring') or status.get('controller_pid')!=2488819:
  raise ValueError('Only exact known current '+arm+' execution child may be declared released')
 stage_binding(identity['command'],arm,'train'if stage=='train'else'score-cp')
 released(identity);return status,identity

def registration_base(arm,value):
 pid,birth=PREVIOUS_TRAINING[arm]
 if (value['arm']!=arm or value['full40_GNN_preserved']is not True or value['no_signal_sent']is not True
     or value['child']['pid']!=pid or value['child']['create_time']!=birth):
  raise ValueError('Exact original training actor and complete40 contract required')
 old,expected=old_launch(active=False)
 if value['controller']!=expected:raise ValueError('Registration controller differs from exact current launch')
 stage_binding(value['child']['command'],arm,'train')
 released(value['child']);released(value['controller'])

def pause_release(arm):
 if arm not in 'ABCD':raise ValueError('Only four exact owned pause markers admitted')
 record=group/(arm+'_pause_for_u128_reuse_controller_20261012.json');current=group/arm/'training/STOP_AFTER_BATCH'
 value=json.loads(regular(record).read_bytes());content=regular(current).read_bytes()
 pid,birth=PREVIOUS_TRAINING[arm];token='U128_REUSE_'+arm+'_20261012_'+str(pid)
 if (set(value)!={'format','arm','marker','child','controller','created_at','reason','full40_GNN_preserved','no_signal_sent','token'}
     or value['format']!='v25_owned_validation_reuse_pause_v1' or value['arm']!=arm or value['marker']!=str(current)
     or value['token']!=token or value['full40_GNN_preserved']is not True or value['no_signal_sent']is not True
     or json.loads(content)!=value or value['child']['pid']!=pid or value['child']['create_time']!=birth):
  raise ValueError('Only exact task-created '+arm+' pause marker may be archived')
 registration_base(arm,value)
 intent=json.loads(regular(group/'all4_pause_for_u128_reuse_20261012.json').read_bytes())
 if intent.get('signals_sent')is not False or intent['records'][arm]!=value:
  raise ValueError('Exact root registration intent differs')
 return value,content,current,record

def registration_release(arm,status,terminal_actor):
 if arm not in 'ABCD':raise ValueError('Exact four-arm registration required')
 record=group/(arm+'_pause_for_u128_reuse_controller_20261012.json')
 separate=group/(arm+'_terminal_for_u128_reuse_controller_20261012.json')
 if separate.exists():
  if arm!='D':raise ValueError('Only root-owned D post-request terminal registration admitted')
  record=separate
 content=regular(record).read_bytes();value=json.loads(content)
 base={'format','arm','marker','child','controller','created_at','reason','full40_GNN_preserved','no_signal_sent','token'}
 registration_base(arm,value)
 marker=group/arm/'training/STOP_AFTER_BATCH';pause=None;failure_log=None
 if value['format']=='v25_owned_validation_reuse_pause_v1':
  if set(value)!=base:raise ValueError('Exact owned pause registration schema required')
  pause=pause_release(arm);mode='batch_paused'
  if status['stage']=='train' and terminal_actor!=value['child']:
   raise ValueError('Terminal train actor differs from exact registered pause child')
 elif value['format']=='v25_owned_validation_reuse_terminal_v1':
  if (set(value)!=base|{'status_path','status_sha256','failure_log','failure_log_sha256'}
      or value['marker']is not None or value['token']is not None or status['stage']!='train'
      or terminal_actor!=value['child']):
   raise ValueError('Exact original naturally FAILED training actor required')
  if value['status_path']!=str(group/arm/'monitor/status.json') or value['failure_log']!=status['stage_log']:
   raise ValueError('Exact original failed status/log paths required')
  same(value['status_path'],value['status_sha256']);same(value['failure_log'],value['failure_log_sha256'])
  log=regular(value['failure_log'])
  if not log.is_relative_to(group/arm) or 'RuntimeError: Recorded stage descendant is outside exact controller tree'not in log.read_text():
   raise ValueError('Exact observed owned stage-tree failure required')
  failure_log=dict(path=str(log),sha256=value['failure_log_sha256']);mode='natural_failed'
  if record==separate:
   pause=pause_release(arm)
  else:
   if marker.exists():raise ValueError('Natural failure has an unregistered pause marker')
   intent=json.loads(regular(group/'all4_pause_for_u128_reuse_20261012.json').read_bytes())
   if intent.get('signals_sent')is not False or intent['records'][arm]!=value:
    raise ValueError('Exact root terminal registration intent differs')
 else:raise ValueError('Only exact owned batch-pause or natural-failure registration admitted')
 return dict(mode=mode,record=record,value=value,content=content,pause=pause,failure_log=failure_log,
  status_sha256=sha(group/arm/'monitor/status.json'))

def checkpoint_status(state,mode):
 expected={'batch_paused':'PAUSED','natural_failed':'RUNNING'}
 if mode not in expected or state['status']!=expected[mode]:
  raise ValueError('Checkpoint raw status must match exact owned recovery evidence; no rewrite or rewind')

def checkpoint(arm,mode):
 import torch
 from hiercp_v1x.u_bridge_training import digest
 from hiercp_v1x.v25_training import validate_curriculum_progress
 path=regular(group/arm/'training/checkpoint_latest.pt');before=path.stat()
 saved=torch.load(path,map_location='cpu',weights_only=False);after=path.stat()
 if (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)!=(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns):
  raise RuntimeError('Checkpoint changed while inspected')
 own=json.loads(regular(group/arm/'training/training_identity.json').read_bytes());state=saved['state']
 if (saved.get('format')!='v25_all_P_patient_balanced_shared_context_training_v1'
     or saved['identity_sha256']!=own['identity_sha256'] or digest(own['binding'])!=own['identity_sha256']
     or saved['content_sha256']!=digest({k:v for k,v in saved.items() if k!='content_sha256'})
     or len(saved['rank_rng'])!=1 or 'shuffle_generator'not in saved
     or not all(k in saved for k in ('model','optimizer','scheduler','scaler'))):
  raise ValueError('Exact own model/optimizer/scheduler/scaler/RNG checkpoint required')
 checkpoint_status(state,mode)
 binding=own['binding']
 validate_curriculum_progress(state,policy=binding['config']['v24_runtime']['curriculum'],train_cases=binding['train_cases'],val_cases=binding['val_cases'],physical_patient_batch=binding['physical_patient_batch'])
 result=dict(path=str(path),sha256=sha(path),identity_sha256=saved['identity_sha256'],content_sha256=saved['content_sha256'],
  epoch=state['epoch'],updates=state['updates'],phase=state['phase'],train_position=state['train_position'],
  evaluation_position=state['evaluation_position'],active_u=state['curriculum']['active_u'],status=state['status'])
 del saved;return result
def extra_archive_admission(disk,checkpoint_bytes):
 if (type(checkpoint_bytes)is not int or checkpoint_bytes<=0
     or type(disk.get('actual_free_bytes'))is not int or type(disk.get('required_free_bytes'))is not int
     or disk['actual_free_bytes']<0 or disk['required_free_bytes']<0):
  raise ValueError('Measured complete checkpoint bytes and original disk requirement required')
 required=disk['required_free_bytes']+checkpoint_bytes+2**30
 if disk['actual_free_bytes']<required:raise OSError('Original v2.5 remaining writes plus four paused-checkpoint backups and GPU0 BEST26 archive do not fit')
 return dict(original_v25_remaining_admission=disk,checkpoint_backup_bytes=checkpoint_bytes,
  GPU0_persistent_BEST26_reservation_bytes=2**30,required_free_bytes=required,
  actual_free_bytes=disk['actual_free_bytes'],original_storage_admission_rewritten=False)
def validate_archive():
 payload=base64.b64decode(PAYLOAD_DATA)
 if hashlib.sha256(payload).hexdigest()!=manifest['sha256']:raise ValueError('Reviewed archive changed')
 result=[];seen=set()
 with tarfile.open(fileobj=io.BytesIO(payload),mode='r:gz') as bundle:
  for member in bundle.getmembers():
   if member.isdir():continue
   relative=pathlib.PurePosixPath(member.name)
   if not member.isfile() or relative.is_absolute() or '..'in relative.parts or member.name in seen or member.name not in allowed:
    raise ValueError('Only exact reviewed code overlay paths are allowed')
   seen.add(member.name);proof=manifest['proof'][member.name];old=oldcode/member.name
   if (sha(regular(old)) if old.exists() else None)!=proof['previous_sha256']:raise ValueError('Current ab23 file SHA differs: '+member.name)
   body=bundle.extractfile(member).read()
   if hashlib.sha256(body).hexdigest()!=proof['new_sha256']:raise ValueError('New committed file SHA differs')
   if relative.suffix=='.py':ast.parse(body.decode('utf8'),filename=member.name)
   result.append((member.name,body))
 if seen!=set(manifest['paths']) or not required<=seen:raise ValueError('Complete reviewed four-arm overlay required')
 return result

if os.getuid()!=UID:raise ValueError('Exact account required')
regular(group,True);regular(oldcode,True);regular(cache,True);requests=scientific(oldcode)
if set(manifest['paths']).intersection(requests['B']['source_sha256']):raise ValueError('Frozen23 cannot change')
if phase=='stage':
 if stagepath.exists() or code.exists():raise FileExistsError('Fresh staged namespace required; inspect before retry')
 old,previous_controller=old_launch(active=None);admitted=validate_archive();preserved=inputs()
 # Staging never reads or seals the concurrently updated four latest checkpoints.
 source_files=[];directories=[]
 for folder,dirs,files in os.walk(oldcode,followlinks=False):
  folder=regular(pathlib.Path(folder),True);directories.append(folder.relative_to(oldcode))
  for name in dirs:regular(folder/name,True)
  for name in files:
   p=regular(folder/name);source_files.append((p,p.relative_to(oldcode),sha(p)))
 code.mkdir(exist_ok=False)
 for relative in directories:
  if relative.parts:(code/relative).mkdir(parents=True,exist_ok=True)
 oldfiles={}
 for p,relative,digest in source_files:
  target=code/relative
  with p.open('rb') as stream,target.open('xb') as out:
   for block in iter(lambda:stream.read(2**20),b''):out.write(block)
  same(p,digest);same(target,digest);oldfiles[str(p)]=digest
 for name,body in admitted:
  target=code/name;target.parent.mkdir(parents=True,exist_ok=True);temporary=target.with_name(target.name+'.U128_reuse.tmp')
  with temporary.open('xb') as stream:stream.write(body);stream.flush();os.fsync(stream.fileno())
  os.replace(temporary,target)
 scientific(code)
 for path,digest in preserved.items():same(path,digest)
 for path,digest in oldfiles.items():same(path,digest)
 result=dict(format='v25_U128_validation_reuse_private_execution_stage_v1',phase='stage',commit=manifest['commit'],archive_sha256=manifest['sha256'],
  private_code=str(code),previous_code=str(oldcode),preserved_files=preserved,unchanged_old_code_files=oldfiles,
  original_controller_identity=previous_controller,
  frozen23_sources_unchanged=True,existing_code_files_modified=False,current_four_and_GPU0_processes_modified=False,
  all_four_live_latest_not_sealed=True,signals_sent=False,training_started=False)
 publish(stagepath,result);print(json.dumps(result))
else:
 if any(p.exists() for p in (globalintentpath,intentpath,launchpath,sealpath,*proofpaths.values())):
  raise FileExistsError('Recovery launch intent exists; inspect exact actor instead of retry')
 staged=json.loads(regular(stagepath).read_bytes())
 if staged['commit']!=manifest['commit'] or staged['archive_sha256']!=manifest['sha256']:raise ValueError('Reviewed staged commit differs')
 for path,digest in staged['preserved_files'].items():same(path,digest)
 for path,digest in staged['unchanged_old_code_files'].items():same(path,digest)
 for name,proof in manifest['proof'].items():same(code/name,proof['new_sha256'])
 scientific(code);old,previous_controller=old_launch(active=False)
 statuses={};actors={}
 for arm in 'ABCD':statuses[arm],actors[arm]=terminal_arm(arm)
 registrations={arm:registration_release(arm,statuses[arm],actors[arm])for arm in 'ABCD'}
 sys.path.insert(0,str(code))
 from hiercp_v1x.v25_direct_resources import capture_cgroup_admission,_pbs_start_time,validate_direct_document
 from hiercp_v1x.v25_native_storage import verify_admission,check_aggregate_disk
 verify_admission(group/'native_storage_admission.json',full_hash=False)
 checkpoints={arm:checkpoint(arm,registrations[arm]['mode']) for arm in 'ABCD'}
 disk=check_aggregate_disk(group/'native_storage_admission.json',group/'B')
 checkpoint_bytes=sum(regular(checkpoints[arm]['path']).stat().st_size for arm in 'ABCD')
 extra_storage=extra_archive_admission(disk,checkpoint_bytes)
 oldmanifest=json.loads(regular(oldsealpath).read_bytes());oldproofs={}
 for arm in 'ABCD':
  proof=oldmanifest['slots'][arm]['direct_proof'];same(proof['path'],proof['sha256'])
  oldproofs[arm]=validate_direct_document(json.loads(regular(proof['path']).read_bytes()))
 aggregate=oldproofs['B']['aggregate_CPU_affinity']
 if not set(aggregate)<=set(psutil.Process().cpu_affinity()):raise RuntimeError('Actual allowed CPUs do not admit original24-core set')
 for arm,gpu,cpus in (('A',6,list(range(32,38))),('B',5,list(range(42,48))),('C',1,list(range(16,22))),('D',4,list(range(6)))):
  row=oldproofs[arm]
  if row['controller_identity']!=previous_controller or row['gpu']!=gpu or row['CPU_affinity']!=cpus or row['aggregate_CPU_affinity']!=aggregate:
   raise ValueError('Original exact four distinct physical GPU/sixCPU proofs required')
  actual=subprocess.check_output(['nvidia-smi','--id='+str(gpu),'--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True).strip().split(',')
  if actual[0].strip()!=row['GPU_UUID'] or float(actual[1])<40*1024:raise RuntimeError('Exact idle physical GPU UUID/free40GiB required')
 apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader,nounits'],text=True)
 if any(line.strip() and line.split(',')[-1].strip() in {oldproofs[a]['GPU_UUID'] for a in 'ABCD'} for line in apps.splitlines()):
  raise RuntimeError('One requested GPU1/4/5/6 live CUDA actor exists; preserve it')
 owned=json.loads(regular(group/'allocation_request.json').read_bytes())
 jobs=json.loads(subprocess.check_output(['/opt/pbs/bin/qstat','-f','-F','json',owned['job_id']],text=True,timeout=20))['Jobs']
 if list(jobs)!=[owned['job_id']]:raise RuntimeError('Exact owned host lifetime unavailable')
 job=next(iter(jobs.values()));h,m,s=map(int,job['Resource_List']['walltime'].split(':'));expiry=_pbs_start_time(job['stime'])+h*3600+m*60+s
 if job.get('job_state')!='R' or job.get('Job_Owner','').split('@')[0]!='aicompetition06' or abs(expiry-owned['expires_at_unix'])>1 or time.time()>=expiry-180:
  raise RuntimeError('Current owned host lifetime failed')
 cgroup=capture_cgroup_admission(os.getpid())
 if any(row['limit_bytes']is not None and row['limit_bytes']<416*2**30 for row in cgroup['memory_limits']):raise MemoryError('Unchanged416GiB cgroup admission failed')
 if psutil.virtual_memory().available<416*2**30:raise MemoryError('Actual available RAM cannot admit unchanged full416GiB')
 command=list(old['command']);entry=str(oldcode/'tools/run_v25_recovery_continuation.py')
 if command.count(entry)!=1:raise ValueError('Exact old controller entry required')
 command[command.index(entry)]=str(code/'tools/run_v25_u128_reuse_continuation.py')
 command[command.index('--execution-manifest')+1]=str(sealpath)
 archive=group/'code_revisions'/('U128_reuse_before_launch_'+manifest['commit'][:12]);archive.mkdir(parents=True,exist_ok=False)
 for arm in 'ABCD':
  with (archive/(arm+'_status.json')).open('xb') as stream:stream.write(regular(group/arm/'monitor/status.json').read_bytes())
  same(archive/(arm+'_status.json'),registrations[arm]['status_sha256'])
  source=regular(group/arm/'training/checkpoint_latest.pt');target=archive/(arm+'_checkpoint_latest.pt')
  with source.open('rb') as stream,target.open('xb') as out:
   for block in iter(lambda:stream.read(2**20),b''):out.write(block)
  same(source,checkpoints[arm]['sha256']);same(target,checkpoints[arm]['sha256'])
 # Recheck all exact old actors and immutable pause records before intent.
 old_launch(active=False)
 for identity in actors.values():released(identity)
 archived_markers={}
 for arm in 'ABCD':
  renewed=registration_release(arm,statuses[arm],actors[arm])
  if renewed!=registrations[arm]:raise ValueError('Owned recovery registration changed before archive')
  if renewed['pause']is None:continue
  value,content,current,record=renewed['pause'];target=archive/(arm+'_STOP_AFTER_BATCH.json')
  if target.exists():raise FileExistsError('Fresh owned pause-marker archive required')
  archived_markers[arm]=dict(path=str(target),sha256=sha(current))
 intent=dict(format='v25_four_arm_U128_validation_reuse_launch_intent_v1',command=command,private_code=str(code),commit=manifest['commit'],
  checkpoints=checkpoints,previous_actors=actors,released_controller=previous_controller,
  recovery_modes={arm:registrations[arm]['mode']for arm in 'ABCD'},
  archived_pause_markers=archived_markers,extra_storage_admission=extra_storage,created_at=time.time(),signals_sent=False)
 publish(globalintentpath,intent);publish(intentpath,intent)
 for arm in archived_markers:
  value,content,current,record=registrations[arm]['pause'];target=pathlib.Path(archived_markers[arm]['path'])
  if regular(current).read_bytes()!=content:raise ValueError('Exact owned pause bytes changed before archive rename')
  os.rename(current,target);same(target,archived_markers[arm]['sha256'])
 env=os.environ.copy();env.update(PYTHONPATH=str(code),PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
 log=group/('U128_reuse_pipeline_controller'+suffix+'.log')
 with log.open('xb') as stream:
  child=subprocess.Popen(command,cwd=code,env=env,stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
   start_new_session=True,preexec_fn=lambda:os.sched_setaffinity(0,aggregate))
 p=psutil.Process(child.pid);deadline=time.monotonic()+5
 while p.cmdline()!=command and time.monotonic()<deadline:time.sleep(.05)
 identity=actor(p)
 if identity['command']!=command or identity['UID']!=UID:raise RuntimeError('New exact four-arm controller not observed; inspect durable intent')
 for arm in 'ABCD':
  value=dict(oldproofs[arm],controller_identity=identity,resource_guard_sha256=sha(code/'hiercp_v1x/v25_direct_resources.py'),
   cgroup_admission=capture_cgroup_admission(p.pid));publish(proofpaths[arm],value)
 names=('tools/run_v25_u128_reuse_execution.py','tools/run_v25_u128_reuse_continuation.py','hiercp_v1x/v25_u128_reuse_resources.py',
  'hiercp_v1x/v25_u128_validation_reuse.py',
  'hiercp_v1x/v25_provider_memory_runtime.py','hiercp_v1x/host_memory.py','tools/run_v25_execution.py',
  'hiercp_v1x/v25_upper_memory_runtime.py','hiercp_v1x/v25_terminal_descendant_resources.py','hiercp_v1x/v25_direct_resources.py',
  'hiercp_v1x/v25_prefix_geometry_reuse.py','hiercp_v1x/v24_upper_reuse.py','tools/run_v25_downstream.py',
  'hiercp_v1x/v25_native_cp.py','hiercp_v1x/v25_downstream.py')
 preserved=dict(staged['preserved_files']);preserved.update({str(code/name):digest for name,digest in requests['B']['source_sha256'].items()})
 preserved[str(group/'all4_pause_for_u128_reuse_20261012.json')]=sha(group/'all4_pause_for_u128_reuse_20261012.json')
 recoveries={}
 for arm in 'ABCD':
  row=registrations[arm];registration=dict(path=str(row['record']),sha256=sha(row['record']))
  monitor=dict(path=str(archive/(arm+'_status.json')),sha256=row['status_sha256'])
  recovery=dict(mode=row['mode'],registration=registration,monitor=monitor,failure_log=row['failure_log'],
   previous_controller_identity=previous_controller)
  for proof in (registration,monitor,row['failure_log']):
   if proof is not None:preserved[proof['path']]=proof['sha256']
  if row['pause']is not None:
   record=row['pause'][3];preserved[str(record)]=sha(record)
   marker=archived_markers[arm];preserved[marker['path']]=marker['sha256']
  recoveries[arm]=recovery
 seal=dict(format='v25_exact_four_arm_u128_validation_reuse_v1',group_root=str(group),private_code=str(code),previous_code=str(oldcode),
  controller_identity=identity,slots={arm:dict(arm=arm,gpu=oldproofs[arm]['gpu'],CPU_affinity=oldproofs[arm]['CPU_affinity'],
   direct_proof=dict(path=str(proofpaths[arm]),sha256=sha(proofpaths[arm])),checkpoint=checkpoints[arm],
   previous_actor=actors[arm],recovery=recoveries[arm]) for arm in 'ABCD'},
  preserved_files=preserved,runtime_files={str(code/name):sha(code/name) for name in names},created_at=time.time(),
  scientific_configuration_unchanged=True,no_process_signaled=True)
 from hiercp_v1x.v25_u128_reuse_resources import validate_seal
 validate_seal(seal);publish(sealpath,seal)
 result=dict(phase='start',PID=p.pid,create_time=p.create_time(),UID=UID,command=command,log=str(log),private_code=str(code),
  seal=str(sealpath),proofs={arm:str(proofpaths[arm]) for arm in 'ABCD'},commit=manifest['commit'],checkpoints=checkpoints,
  GPUs=[6,5,1,4],CPU_affinity={arm:oldproofs[arm]['CPU_affinity'] for arm in 'ABCD'},old_controller_released=previous_controller,archived_pause_markers=archived_markers,checkpoint_backups=str(archive),
  extra_storage_admission=extra_storage,
  recovery_modes={arm:registrations[arm]['mode']for arm in 'ABCD'},
  full40_GNN=True,full250_native=True,own_BEST26=True,signals_sent=False,debug=False)
 publish(launchpath,result);print(json.dumps(result))
'''


def build_remote_script(manifest, payload, phase):
    if phase not in ('stage','start'):
        raise ValueError('Explicit stage/start required')
    if not re.fullmatch('[0-9a-f]{40}', manifest.get('commit','')):
        raise ValueError('Exact reviewed Git commit required')
    if hashlib.sha256(payload).hexdigest()!=manifest.get('sha256'):
        raise ValueError('Reviewed archive bytes differ')
    paths=manifest.get('paths',[]);proof=manifest.get('proof',{})
    if len(paths)!=len(set(paths)) or set(paths)!=set(proof) or not REQUIRED<=set(paths)<=PATHS:
        raise ValueError('Exact reviewed seven/eight-file U128-reuse overlay required')
    for row in proof.values():
        if set(row)!={'previous_sha256','new_sha256'} or not re.fullmatch('[0-9a-f]{64}',row['new_sha256']):
            raise ValueError('Exact before/new SHA proofs required')
        if row['previous_sha256']is not None and not re.fullmatch('[0-9a-f]{64}',row['previous_sha256']):
            raise ValueError('Exact original SHA or explicit missing file required')
    return REMOTE_SCRIPT.replace('MANIFEST_DATA',repr(manifest)).replace('PHASE_DATA',repr(phase)).replace(
        'ALLOWED_DATA',repr(PATHS)).replace('REQUIRED_DATA',repr(REQUIRED)).replace('PAYLOAD_DATA',repr(base64.b64encode(payload).decode()))


def main(argv=None):
    arguments=list(sys.argv[1:] if argv is None else argv)
    if len(arguments)!=2:
        raise ValueError('Provide exact reviewed manifest and stage/start')
    manifest=json.loads(Path(arguments[0]).resolve(strict=True).read_text(encoding='utf8'))
    script=build_remote_script(manifest,Path(manifest['archive']).read_bytes(),arguments[1])
    here=Path('D:/AI project/nnunet/outputs/v24_gpu6_recovery_20261011')
    spec=importlib.util.spec_from_file_location('secure_remote',here/'remote_once.py')
    remote=importlib.util.module_from_spec(spec);spec.loader.exec_module(remote)
    value=json.loads(remote.run(script,90));path=here/('V25_U128_REUSE_'+arguments[1].upper()+'_'+str(time.time_ns())+'.json')
    with path.open('x',encoding='utf8') as stream:json.dump(value,stream,indent=2)
    print(json.dumps(dict(receipt=str(path),observed={key:value[key] for key in (
        'phase','commit','PID','create_time','private_code','checkpoints','GPUs','CPU_affinity','signals_sent') if key in value}),ensure_ascii=False))


if __name__=='__main__':main()
