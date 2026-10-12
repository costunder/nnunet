"""Reviewed private BD stage/start; no signals and no live source mutation.

Building the script is local and side-effect free. The root explicitly runs
stage, requests its own D batch-boundary pause, verifies release, then start.
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
    'hiercp_v1x/v25_provider_memory_runtime.py',
    'hiercp_v1x/v25_standalone_bd_resources.py',
    'tools/run_v25_bd_execution.py', 'tools/run_v25_bd_continuation.py',
    'tools/deploy_v25_bd_provider_recovery.py',
    'tests/test_v25_provider_memory_runtime.py',
    'tests/test_v25_standalone_bd_execution.py',
    'tests/test_v25_bd_deployment.py',
}
REQUIRED = PATHS - {'tests/test_v25_provider_memory_runtime.py'}
REMOTE_SCRIPT = r'''
import ast,base64,hashlib,io,json,os,pathlib,psutil,stat,subprocess,sys,tarfile,time
group=pathlib.Path('/var/tmp/aicompetition06_v25_20261012/run_1791784719489478600')
oldcode=group/'code_shared_reuse_b84c0b6bd0b6';UID=41833
manifest=MANIFEST_DATA;phase=PHASE_DATA;allowed=ALLOWED_DATA;required=REQUIRED_DATA
code=group/('code_BD_provider_'+manifest['commit'][:12]);suffix='_BD_provider_'+manifest['commit'][:12]
stagepath=group/('BD_code_deployed'+suffix+'.json');intentpath=group/('BD_controller_intent'+suffix+'.json')
globalintentpath=group/'BD_standalone_controller_intent.json'
launchpath=group/('BD_controller_launch'+suffix+'.json');sealpath=group/('BD_continuation_seal'+suffix+'.json')
proofpaths={arm:group/(arm+'_direct_execution'+suffix+'.json') for arm in 'BD'}
cache=pathlib.Path('/dev/shm/aicompetition06_v25_20261012/run_1791784719489478600/shared_context_cache')
oldlaunchpath=group/'parallel_controller_launch_shared_reuse_1.json'
Dsealpath=group/'D_continuation_seal_D_provider_61a97a736994.json'
pausepath=group/'D_pause_for_BD_controller_20261012.json';marker=group/'D/training/STOP_AFTER_BATCH'
B_SHA='42bb39ded6043a42a6b7b48553216466360db00e9de0cea5a5ac03d8c4e844b3'
TOKEN='GPU4_D_BD_CONTROLLER_20261012_2467341'

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
def old_launch():
 old=json.loads(regular(oldlaunchpath).read_bytes())
 if old['private_code']!=str(oldcode) or old['code_commit']!='b84c0b6bd0b6912c2e9c62cf32a7467c449d4d32':
  raise ValueError('Exact original b84 A/C launch required')
 identity=dict(pid=old['PID'],create_time=old['create_time'],command=old['command'],UID=old['UID'])
 if identity['pid']!=2443117 or identity['create_time']!=1791802599.57:raise ValueError('Exact original A/C controller required')
 exact(identity);return old,identity
def scientific(base):
 rows={}
 for arm in 'BD':
  request=json.loads(regular(group/arm/'request.json').read_bytes())
  if len(request['source_sha256'])!=23:raise ValueError('All23 frozen sources required')
  for name,digest in request['source_sha256'].items():same(base/name,digest)
  rows[arm]=request
 if rows['B']['source_sha256']!=rows['D']['source_sha256']:raise ValueError('Frozen BD source contracts differ')
 return rows
def inputs():
 names=[cache/'local/index.json',cache/'input_bindings.json',group/'shared_preparation.json',group/'native_storage_admission.json']
 for arm in 'BD':names += [group/arm/name for name in ('request.json','preparation.json','calibration.json','training/training_identity.json')]
 # Mutable latest checkpoints are never in the runtime immutable table.
 return {str(p):sha(regular(p)) for p in names}
def failed_B():
 status=json.loads(regular(group/'B/monitor/status.json').read_bytes())
 if status.get('status')!='FAILED' or status.get('stage')!='train':raise ValueError('Observed failed B train required')
 identity=dict(pid=status['child_pid'],create_time=status['child_create_time'],command=status['child_command'],UID=status['UID'])
 if str(oldcode/'tools/run_v25_execution.py')not in identity['command'] or '--arm'not in identity['command']:
  raise ValueError('Exact original B execution child required')
 if identity['command'][identity['command'].index('--arm')+1]!='B':raise ValueError('Wrong failed arm')
 released(identity);same(group/'B/training/checkpoint_latest.pt',B_SHA)
 return status,identity
def failed_D():
 status=json.loads(regular(group/'D/monitor/status.json').read_bytes())
 if status.get('status')!='FAILED':raise RuntimeError('Own D controller has not reached terminal failure after pause')
 identity=dict(pid=status['child_pid'],create_time=status['child_create_time'],command=status['child_command'],UID=status['UID'])
 previous=json.loads(regular(Dsealpath).read_bytes())
 if str(pathlib.Path(previous['private_code'])/'tools/run_v25_d_execution.py')not in identity['command']:
  raise ValueError('Only the exact old D execution child may be declared released')
 if '--arm'not in identity['command'] or identity['command'][identity['command'].index('--arm')+1]!='D':
  raise ValueError('Wrong old continuation arm')
 released(identity);return status,identity
def pause_release():
 value=json.loads(regular(pausepath).read_bytes());content=regular(marker).read_bytes()
 if (set(value)!={'format','marker','child','controller','created_at','reason','full40_GNN_preserved','no_signal_sent','token'}
     or value['format']!='v25_owned_D_checkpoint_pause_v1' or value['marker']!=str(marker)
     or value['token']!=TOKEN or value['full40_GNN_preserved']is not True or value['no_signal_sent']is not True
     or json.loads(content)!=value or value['child']['pid']!=2467341 or value['child']['create_time']!=1791805734.47
     or value['controller']['pid']!=2467215 or value['controller']['create_time']!=1791805714.84):
  raise ValueError('Only exact task-created D pause marker may be archived')
 previous=json.loads(regular(Dsealpath).read_bytes())
 if value['controller']!=previous['controller_identity']:raise ValueError('D pause controller differs from original seal')
 if str(pathlib.Path(previous['private_code'])/'tools/run_v25_d_execution.py')not in value['child']['command']:
  raise ValueError('Only known own D training child may be released')
 released(value['child']);released(value['controller'])
 return value,content
def checkpoint(arm):
 import torch
 from hiercp_v1x.u_bridge_training import digest
 path=regular(group/arm/'training/checkpoint_latest.pt');before=path.stat()
 saved=torch.load(path,map_location='cpu',weights_only=False);after=path.stat()
 if (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)!=(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns):
  raise RuntimeError('Checkpoint changed while inspected')
 own=json.loads(regular(group/arm/'training/training_identity.json').read_bytes());state=saved['state']
 if (saved.get('format')!='v25_all_P_patient_balanced_shared_context_training_v1'
     or saved['identity_sha256']!=own['identity_sha256']
     or saved['content_sha256']!=digest({k:v for k,v in saved.items() if k!='content_sha256'})
     or len(saved['rank_rng'])!=1 or 'shuffle_generator'not in saved
     or not all(k in saved for k in ('model','optimizer','scheduler','scaler'))):
  raise ValueError('Exact own model/optimizer/scheduler/scaler/RNG checkpoint required')
 if arm=='B' and (state['epoch'],state['updates'],state['phase'],state['train_position'],state['evaluation_position'],state['curriculum']['active_u'])!=(2,34,'full_validation',65,18,16):
  raise ValueError('Original B exact validation cursor required')
 if arm=='D' and (state['status']!='PAUSED' or state['curriculum']['active_u']!=128):
  raise ValueError('D must retain its latest actual PAUSED full128 cursor')
 result=dict(path=str(path),sha256=sha(path),identity_sha256=saved['identity_sha256'],content_sha256=saved['content_sha256'],
  epoch=state['epoch'],updates=state['updates'],phase=state['phase'],train_position=state['train_position'],
  evaluation_position=state['evaluation_position'],active_u=state['curriculum']['active_u'],status=state['status'])
 if arm=='B' and result['sha256']!=B_SHA:raise ValueError('Original B latest raw SHA changed')
 del saved;return result
def extra_archive_admission(disk,checkpoint_bytes):
 if (type(checkpoint_bytes)is not int or checkpoint_bytes<=0
     or type(disk.get('actual_free_bytes'))is not int or type(disk.get('required_free_bytes'))is not int
     or disk['actual_free_bytes']<0 or disk['required_free_bytes']<0):
  raise ValueError('Measured complete checkpoint bytes and original disk requirement required')
 required=disk['required_free_bytes']+checkpoint_bytes+2**30
 if disk['actual_free_bytes']<required:raise OSError('Original v2.5 remaining writes plus BD backups and GPU0 BEST26 archive do not fit')
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
   if (sha(regular(old)) if old.exists() else None)!=proof['previous_sha256']:raise ValueError('Old b84 file SHA differs: '+member.name)
   body=bundle.extractfile(member).read()
   if hashlib.sha256(body).hexdigest()!=proof['new_sha256']:raise ValueError('New committed file SHA differs')
   if relative.suffix=='.py':ast.parse(body.decode('utf8'),filename=member.name)
   result.append((member.name,body))
 if seen!=set(manifest['paths']) or not required<=seen:raise ValueError('Complete reviewed BD overlay required')
 return result

if os.getuid()!=UID:raise ValueError('Exact account required')
regular(group,True);regular(oldcode,True);regular(cache,True);requests=scientific(oldcode)
if set(manifest['paths']).intersection(requests['B']['source_sha256']):raise ValueError('Frozen23 cannot change')
if phase=='stage':
 if stagepath.exists() or code.exists():raise FileExistsError('Fresh staged namespace required; inspect before retry')
 old,peer=old_launch();Bstatus,Bactor=failed_B();admitted=validate_archive();preserved=inputs()
 # Staging never reads or seals the concurrently updated D latest checkpoint.
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
  target=code/name;target.parent.mkdir(parents=True,exist_ok=True);temporary=target.with_name(target.name+'.BD_provider.tmp')
  with temporary.open('xb') as stream:stream.write(body);stream.flush();os.fsync(stream.fileno())
  os.replace(temporary,target)
 scientific(code)
 for path,digest in preserved.items():same(path,digest)
 for path,digest in oldfiles.items():same(path,digest)
 result=dict(format='v25_BD_private_execution_stage_v1',phase='stage',commit=manifest['commit'],archive_sha256=manifest['sha256'],
  private_code=str(code),previous_code=str(oldcode),preserved_files=preserved,unchanged_old_code_files=oldfiles,
  B_failed_status_sha256=sha(group/'B/monitor/status.json'),B_checkpoint_sha256=B_SHA,
  frozen23_sources_unchanged=True,existing_code_files_modified=False,AC_D_GPU0_processes_modified=False,
  D_live_latest_not_sealed=True,signals_sent=False,training_started=False)
 publish(stagepath,result);print(json.dumps(result))
else:
 if any(p.exists() for p in (globalintentpath,intentpath,launchpath,sealpath,*proofpaths.values())):
  raise FileExistsError('BD launch intent exists; inspect exact actor instead of retry')
 staged=json.loads(regular(stagepath).read_bytes())
 if staged['commit']!=manifest['commit'] or staged['archive_sha256']!=manifest['sha256']:raise ValueError('Reviewed staged commit differs')
 for path,digest in staged['preserved_files'].items():same(path,digest)
 for path,digest in staged['unchanged_old_code_files'].items():same(path,digest)
 for name,proof in manifest['proof'].items():same(code/name,proof['new_sha256'])
 scientific(code);old,peer=old_launch();Bstatus,Bactor=failed_B();pause,pause_bytes=pause_release();Dstatus,Dactor=failed_D()
 same(group/'B/monitor/status.json',staged['B_failed_status_sha256'])
 sys.path.insert(0,str(code))
 from hiercp_v1x.v25_direct_resources import capture_cgroup_admission,_pbs_start_time,validate_direct_document
 from hiercp_v1x.v25_native_storage import verify_admission,check_aggregate_disk
 verify_admission(group/'native_storage_admission.json',full_hash=False)
 checkpoints={arm:checkpoint(arm) for arm in 'BD'}
 disk=check_aggregate_disk(group/'native_storage_admission.json',group/'B')
 checkpoint_bytes=sum(regular(checkpoints[arm]['path']).stat().st_size for arm in 'BD')
 extra_storage=extra_archive_admission(disk,checkpoint_bytes)
 oldmanifestpath=regular(old['manifest']);oldmanifest=json.loads(oldmanifestpath.read_bytes())
 oldproofs={arm:validate_direct_document(json.loads(regular(oldmanifest['slots'][arm]['resource_request_path']).read_bytes())) for arm in 'BD'}
 aggregate=oldproofs['B']['aggregate_CPU_affinity']
 if not set(aggregate)<=set(psutil.Process().cpu_affinity()):raise RuntimeError('Actual allowed CPUs do not admit original24-core set')
 for arm,gpu,cpus in (('B',5,list(range(42,48))),('D',4,list(range(6)))):
  row=oldproofs[arm]
  if row['controller_identity']!=peer or row['gpu']!=gpu or row['CPU_affinity']!=cpus or row['aggregate_CPU_affinity']!=aggregate:
   raise ValueError('Original exact BD distinct physical GPU/sixCPU proof required')
  actual=subprocess.check_output(['nvidia-smi','--id='+str(gpu),'--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True).strip().split(',')
  if actual[0].strip()!=row['GPU_UUID'] or float(actual[1])<40*1024:raise RuntimeError('Exact idle physical GPU UUID/free40GiB required')
 apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader,nounits'],text=True)
 if any(line.strip() and line.split(',')[-1].strip() in {oldproofs[a]['GPU_UUID'] for a in 'BD'} for line in apps.splitlines()):
  raise RuntimeError('GPU4/5 live CUDA actor exists; preserve it')
 owned=json.loads(regular(group/'allocation_request.json').read_bytes())
 jobs=json.loads(subprocess.check_output(['/opt/pbs/bin/qstat','-f','-F','json',owned['job_id']],text=True,timeout=20))['Jobs']
 if list(jobs)!=[owned['job_id']]:raise RuntimeError('Exact owned host lifetime unavailable')
 job=next(iter(jobs.values()));h,m,s=map(int,job['Resource_List']['walltime'].split(':'));expiry=_pbs_start_time(job['stime'])+h*3600+m*60+s
 if job.get('job_state')!='R' or job.get('Job_Owner','').split('@')[0]!='aicompetition06' or abs(expiry-owned['expires_at_unix'])>1 or time.time()>=expiry-180:
  raise RuntimeError('Current owned host lifetime failed')
 cgroup=capture_cgroup_admission(os.getpid())
 if any(row['limit_bytes']is not None and row['limit_bytes']<416*2**30 for row in cgroup['memory_limits']):raise MemoryError('Unchanged416GiB cgroup admission failed')
 peerprocess=exact(peer);rss=0
 for p in [peerprocess,*peerprocess.children(recursive=True)]:
  try:
   if p.status()!=psutil.STATUS_ZOMBIE:rss+=p.memory_info().rss
  except psutil.NoSuchProcess:continue
 if psutil.virtual_memory().available+rss<416*2**30:raise MemoryError('Actual available RAM cannot admit full416GiB')
 command=list(old['command']);entry=str(oldcode/'tools/run_v25_parallel_pipeline.py')
 if command.count(entry)!=1:raise ValueError('Exact old controller entry required')
 command[command.index(entry)]=str(code/'tools/run_v25_bd_continuation.py')
 command[command.index('--execution-manifest')+1]=str(sealpath)
 archive=group/'code_revisions'/('BD_provider_before_launch_'+manifest['commit'][:12]);archive.mkdir(parents=True,exist_ok=False)
 for arm in 'BD':
  with (archive/(arm+'_status.json')).open('xb') as stream:stream.write(regular(group/arm/'monitor/status.json').read_bytes())
  source=regular(group/arm/'training/checkpoint_latest.pt');target=archive/(arm+'_checkpoint_latest.pt')
  with source.open('rb') as stream,target.open('xb') as out:
   for block in iter(lambda:stream.read(2**20),b''):out.write(block)
  same(source,checkpoints[arm]['sha256']);same(target,checkpoints[arm]['sha256'])
 # Recheck release and immutable pause bytes immediately before this one rename.
 pause2,bytes2=pause_release()
 if pause2!=pause or bytes2!=pause_bytes:raise ValueError('Own D pause marker changed before archive')
 archived_marker=archive/'D_STOP_AFTER_BATCH.json'
 if archived_marker.exists():raise FileExistsError('Fresh marker archive required')
 intent=dict(format='v25_BD_standalone_launch_intent_v1',command=command,private_code=str(code),commit=manifest['commit'],
  checkpoints=checkpoints,previous_B_actor=Bactor,previous_D_actor=Dactor,released_D_training_actor=pause['child'],released_D_controller=pause['controller'],
  archived_pause_marker=dict(path=str(archived_marker),sha256=sha(marker)),extra_storage_admission=extra_storage,
  AC_controller=peer,created_at=time.time(),signals_sent=False)
 # A group-wide exclusive intent also prevents two reviewed private commits
 # from starting concurrently before their monitors or CUDA actors appear.
 publish(globalintentpath,intent);publish(intentpath,intent)
 os.rename(marker,archived_marker);same(archived_marker,hashlib.sha256(pause_bytes).hexdigest())
 env=os.environ.copy();env.update(PYTHONPATH=str(code),PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
 log=group/('BD_pipeline_controller'+suffix+'.log')
 with log.open('xb') as stream:
  child=subprocess.Popen(command,cwd=code,env=env,stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
   start_new_session=True,preexec_fn=lambda:os.sched_setaffinity(0,aggregate))
 p=psutil.Process(child.pid);deadline=time.monotonic()+5
 while p.cmdline()!=command and time.monotonic()<deadline:time.sleep(.05)
 identity=actor(p)
 if identity['command']!=command or identity['UID']!=UID:raise RuntimeError('New exact BD controller not observed; inspect durable intent')
 for arm in 'BD':
  value=dict(oldproofs[arm],controller_identity=identity,resource_guard_sha256=sha(code/'hiercp_v1x/v25_direct_resources.py'),
   cgroup_admission=capture_cgroup_admission(p.pid));publish(proofpaths[arm],value)
 names=('tools/run_v25_bd_execution.py','tools/run_v25_bd_continuation.py','hiercp_v1x/v25_standalone_bd_resources.py',
  'hiercp_v1x/v25_provider_memory_runtime.py','hiercp_v1x/host_memory.py','tools/run_v25_execution.py',
  'hiercp_v1x/v25_upper_memory_runtime.py','hiercp_v1x/v25_terminal_descendant_resources.py','hiercp_v1x/v25_direct_resources.py',
  'hiercp_v1x/v25_prefix_geometry_reuse.py','hiercp_v1x/v24_upper_reuse.py','tools/run_v25_downstream.py',
  'hiercp_v1x/v25_native_cp.py','hiercp_v1x/v25_downstream.py')
 preserved=dict(staged['preserved_files']);preserved.update({str(code/name):digest for name,digest in requests['B']['source_sha256'].items()})
 preserved.update({str(pausepath):sha(pausepath),str(archived_marker):sha(archived_marker)})
 seal=dict(format='v25_exact_BD_standalone_continuation_v1',group_root=str(group),private_code=str(code),previous_code=str(oldcode),
  controller_identity=identity,slots={arm:dict(arm=arm,gpu=oldproofs[arm]['gpu'],CPU_affinity=oldproofs[arm]['CPU_affinity'],
   direct_proof=dict(path=str(proofpaths[arm]),sha256=sha(proofpaths[arm])),checkpoint=checkpoints[arm],
   previous_actor=Bactor if arm=='B' else Dactor) for arm in 'BD'},
  sibling_manifest=dict(path=str(oldmanifestpath),sha256=sha(oldmanifestpath)),sibling_controller_identity=peer,
  sibling_slots={arm:dict(path=oldmanifest['slots'][arm]['resource_request_path'],sha256=oldmanifest['slots'][arm]['resource_request_sha256']) for arm in 'AC'},
  preserved_files=preserved,runtime_files={str(code/name):sha(code/name) for name in names},created_at=time.time(),
  scientific_configuration_unchanged=True,no_process_signaled=True)
 from hiercp_v1x.v25_standalone_bd_resources import validate_seal
 validate_seal(seal);publish(sealpath,seal)
 result=dict(phase='start',PID=p.pid,create_time=p.create_time(),UID=UID,command=command,log=str(log),private_code=str(code),
  seal=str(sealpath),proofs={arm:str(proofpaths[arm]) for arm in 'BD'},commit=manifest['commit'],checkpoints=checkpoints,
  GPUs=[5,4],CPU_affinity={arm:oldproofs[arm]['CPU_affinity'] for arm in 'BD'},AC_controller_preserved=peer,
  old_D_controller_released=pause['controller'],archived_pause_marker=str(archived_marker),checkpoint_backups=str(archive),
  extra_storage_admission=extra_storage,
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
        raise ValueError('Exact reviewed seven/eight-file BD overlay required')
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
    value=json.loads(remote.run(script,90));path=here/('V25_BD_PROVIDER_'+arguments[1].upper()+'_'+str(time.time_ns())+'.json')
    with path.open('x',encoding='utf8') as stream:json.dump(value,stream,indent=2)
    print(json.dumps(dict(receipt=str(path),observed={key:value[key] for key in (
        'phase','commit','PID','create_time','private_code','checkpoints','GPUs','CPU_affinity','signals_sent') if key in value}),ensure_ascii=False))


if __name__=='__main__':main()
