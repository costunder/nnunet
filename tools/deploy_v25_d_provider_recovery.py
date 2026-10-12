"""Explicit private stage/start for one failed D; ABC remain untouched.

Import/build_remote_script have no server side effects. The root agent must
explicitly run stage or start with the exact reviewed Git archive manifest.
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

REMOTE_SCRIPT=r'''
import ast,base64,hashlib,io,json,os,pathlib,psutil,re,stat,subprocess,sys,tarfile,time
group=pathlib.Path('/var/tmp/aicompetition06_v25_20261012/run_1791784719489478600')
oldcode=group/'code_shared_reuse_b84c0b6bd0b6'
manifest=MANIFEST_DATA;phase=PHASE_DATA;UID=41833
code=group/('code_D_provider_'+manifest['commit'][:12]);suffix='_D_provider_'+manifest['commit'][:12]
stagepath=group/('D_code_deployed'+suffix+'.json')
intentpath=group/('D_controller_intent'+suffix+'.json');launchpath=group/('D_controller_launch'+suffix+'.json')
sealpath=group/('D_continuation_seal'+suffix+'.json');proofpath=group/('D_direct_execution'+suffix+'.json')
cache=pathlib.Path('/dev/shm/aicompetition06_v25_20261012/run_1791784719489478600/shared_context_cache')
oldlaunchpath=group/'parallel_controller_launch_shared_reuse_1.json'
expected_latest='7d13468af9db5340fdf14663c8a4014f56bd60cf8efd1c66762cdf66b3c7c09d'
required={'hiercp_v1x/v25_provider_memory_runtime.py','hiercp_v1x/v25_standalone_d_resources.py',
 'tools/run_v25_d_execution.py','tools/run_v25_d_continuation.py'}

def sha(path):
 d=hashlib.sha256()
 with pathlib.Path(path).open('rb') as stream:
  for block in iter(lambda:stream.read(2**20),b''):d.update(block)
 return d.hexdigest()
def regular(path,directory=False):
 path=pathlib.Path(path)
 if not path.is_absolute() or '..'in path.parts or any(p.is_symlink() for p in (path,*path.parents)):
  raise ValueError('Absolute owned non-symlink namespace required')
 s=path.lstat()
 if s.st_uid!=UID or not(stat.S_ISDIR(s.st_mode) if directory else stat.S_ISREG(s.st_mode)):
  raise ValueError('Exact own regular file/directory required: '+str(path))
 return path
def publish(path,value):
 with path.open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.flush();os.fsync(stream.fileno())
def same(path,digest):
 if sha(regular(path))!=digest:raise ValueError('Immutable source/input changed: '+str(path))
def actor(process):
 if process.status()==psutil.STATUS_ZOMBIE:raise RuntimeError('Live actor cannot be zombie')
 return dict(pid=process.pid,create_time=process.create_time(),command=process.cmdline(),UID=process.uids().real)
def exact(identity):
 p=psutil.Process(identity['pid'])
 if actor(p)!=identity:raise RuntimeError('Controller PID/birth/argv/UID differs')
 return p
def old_launch():
 value=json.loads(regular(oldlaunchpath).read_bytes())
 if value['private_code']!=str(oldcode) or value['code_commit']!='b84c0b6bd0b6912c2e9c62cf32a7467c449d4d32':
  raise ValueError('Exact live b84 shared-reuse controller required')
 identity=dict(pid=value['PID'],create_time=value['create_time'],command=value['command'],UID=value['UID'])
 exact(identity);return value,identity
def failure():
 status=json.loads(regular(group/'D/monitor/status.json').read_bytes())
 if status.get('status')!='FAILED' or status.get('stage')!='train' or status.get('child_pid')!=2443277:
  raise ValueError('Only the inspected failed D train actor may be recovered')
 identity=dict(pid=status['child_pid'],create_time=status['child_create_time'],command=status['child_command'],UID=status['UID'])
 try:p=psutil.Process(identity['pid'])
 except psutil.NoSuchProcess:return status,identity
 if p.create_time()!=identity['create_time'] or p.uids().real!=UID:
  raise RuntimeError('Failed D PID was reused; no release assumption')
 if p.status()!=psutil.STATUS_ZOMBIE:raise RuntimeError('Original D actor remains live; no restart or signal')
 return status,identity
def scientific(base):
 request=json.loads(regular(group/'D/request.json').read_bytes())
 if len(request['source_sha256'])!=23:raise ValueError('Exact23 frozen source contract required')
 for name,digest in request['source_sha256'].items():same(base/name,digest)
 return request
def inputs():
 names=[cache/'local/index.json',cache/'input_bindings.json',group/'shared_preparation.json',
  group/'native_storage_admission.json',group/'D/request.json',group/'D/preparation.json',
  group/'D/calibration.json',group/'D/training/training_identity.json']
 return {str(p):sha(regular(p)) for p in names}
def validate_archive():
 payload=base64.b64decode(PAYLOAD_DATA)
 if hashlib.sha256(payload).hexdigest()!=manifest['sha256']:raise ValueError('Reviewed archive differs')
 result=[];seen=set()
 with tarfile.open(fileobj=io.BytesIO(payload),mode='r:gz') as bundle:
  for member in bundle.getmembers():
   if member.isdir():continue
   relative=pathlib.PurePosixPath(member.name)
   if (not member.isfile() or relative.is_absolute() or '..'in relative.parts or member.name in seen
       or relative.parts[0]=='.git' or member.name not in manifest['proof']):raise ValueError('Unreviewed archive path')
   seen.add(member.name);value=manifest['proof'][member.name];old=oldcode/member.name
   previous=sha(regular(old)) if old.exists() else None
   if previous!=value['previous_sha256']:raise ValueError('Exact b84 previous file differs: '+member.name)
   body=bundle.extractfile(member).read()
   if hashlib.sha256(body).hexdigest()!=value['new_sha256']:raise ValueError('Reviewed new file SHA differs')
   if relative.suffix=='.py':ast.parse(body.decode('utf8'),filename=member.name)
   result.append((member.name,body))
 if seen!=set(manifest['paths']) or not required<=seen:raise ValueError('Complete D recovery sources required')
 return result

if os.getuid()!=UID:raise ValueError('Exact account required')
regular(group,True);regular(oldcode,True);regular(cache,True)
request=scientific(oldcode)
if set(manifest['paths']).intersection(request['source_sha256']):raise ValueError('Frozen23 sources cannot be changed')

if phase=='stage':
 if stagepath.exists() or code.exists():raise FileExistsError('Private D stage exists; inspect before retry')
 old,peer=old_launch();status,released=failure();admitted=validate_archive();preserved=inputs()
 same(group/'D/training/checkpoint_latest.pt',expected_latest)
 source_files=[];source_dirs=[]
 for folder,dirs,files in os.walk(oldcode,followlinks=False):
  folder=regular(pathlib.Path(folder),True);source_dirs.append(folder.relative_to(oldcode))
  for name in dirs:regular(folder/name,True)
  for name in files:
   p=regular(folder/name);source_files.append((p,p.relative_to(oldcode),sha(p)))
 code.mkdir(exist_ok=False)
 for relative in source_dirs:
  if relative.parts:(code/relative).mkdir(parents=True,exist_ok=True)
 oldfiles={}
 for p,relative,digest in source_files:
  target=code/relative
  with p.open('rb') as stream,target.open('xb') as out:
   for block in iter(lambda:stream.read(2**20),b''):out.write(block)
  same(p,digest);same(target,digest);oldfiles[str(p)]=digest
 for name,body in admitted:
  target=code/name;target.parent.mkdir(parents=True,exist_ok=True)
  temporary=target.with_name(target.name+'.D_provider.tmp')
  with temporary.open('xb') as stream:stream.write(body);stream.flush();os.fsync(stream.fileno())
  os.replace(temporary,target)
 scientific(code)
 for path,digest in preserved.items():same(path,digest)
 for path,digest in oldfiles.items():same(path,digest)
 result=dict(format='v25_D_private_execution_stage_v1',phase='stage',commit=manifest['commit'],
  archive_sha256=manifest['sha256'],private_code=str(code),previous_code=str(oldcode),
  preserved_files=preserved,unchanged_old_code_files=oldfiles,
  D_failed_status_sha256=sha(group/'D/monitor/status.json'),checkpoint_sha256=expected_latest,
  frozen23_sources_unchanged=True,existing_code_files_modified=False,ABC_processes_modified=False,
  signals_sent=False,training_started=False)
 publish(stagepath,result);print(json.dumps(result))
else:
 if any(p.exists() for p in (intentpath,launchpath,sealpath,proofpath)):
  raise FileExistsError('D launch registration/intent exists; inspect exact actor instead of retry')
 staged=json.loads(regular(stagepath).read_bytes())
 if staged['commit']!=manifest['commit'] or staged['archive_sha256']!=manifest['sha256']:
  raise ValueError('Reviewed staged commit differs')
 for path,digest in staged['preserved_files'].items():same(path,digest)
 for path,digest in staged['unchanged_old_code_files'].items():same(path,digest)
 for name,proof in manifest['proof'].items():same(code/name,proof['new_sha256'])
 scientific(code);old,peer=old_launch();status,released=failure()
 same(group/'D/monitor/status.json',staged['D_failed_status_sha256'])
 same(group/'D/training/checkpoint_latest.pt',expected_latest)
 sys.path.insert(0,str(code))
 from hiercp_v1x.v25_direct_resources import capture_cgroup_admission,DIRECT_FORMAT,_pbs_start_time
 from hiercp_v1x.v25_native_storage import verify_admission
 verify_admission(group/'native_storage_admission.json',full_hash=True)
 import torch
 from hiercp_v1x.u_bridge_training import digest
 saved=torch.load(group/'D/training/checkpoint_latest.pt',map_location='cpu',weights_only=False)
 ownership=json.loads((group/'D/training/training_identity.json').read_bytes())
 if (saved['identity_sha256']!=ownership['identity_sha256']
     or saved['content_sha256']!=digest({k:v for k,v in saved.items() if k!='content_sha256'})
     or (saved['state']['epoch'],saved['state']['updates'],saved['state']['train_position'])!=(2,20,12)
     or saved['state']['phase']!='training' or len(saved['rank_rng'])!=1):
  raise ValueError('Exact original model/optimizer/RNG epoch2/update20/position12 checkpoint required')
 checkpoint=dict(path=str(group/'D/training/checkpoint_latest.pt'),sha256=expected_latest,
  identity_sha256=saved['identity_sha256'],content_sha256=saved['content_sha256'],epoch=2,updates=20,train_position=12)
 del saved;del torch
 original_manifest_path=regular(old['manifest']);original_manifest=json.loads(original_manifest_path.read_bytes())
 originalD_path=regular(original_manifest['slots']['D']['resource_request_path'])
 originalD=json.loads(originalD_path.read_bytes());aggregate=originalD['aggregate_CPU_affinity']
 if originalD['controller_identity']!=peer or originalD['gpu']!=4 or originalD['CPU_affinity']!=list(range(6)):
  raise ValueError('Original exact D CPU/GPU binding differs')
 allowed=psutil.Process().cpu_affinity()
 if not set(aggregate)<=set(allowed):raise RuntimeError('Actual allowed CPUs cannot admit original24-core declaration')
 actual=subprocess.check_output(['nvidia-smi','--id=4','--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True).strip().split(',')
 if actual[0].strip()!=originalD['GPU_UUID'] or float(actual[1])<40*1024:raise RuntimeError('Exact GPU4 UUID/free40GiB admission failed')
 apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader,nounits'],text=True)
 if any(line.strip() and line.split(',')[-1].strip()==originalD['GPU_UUID'] for line in apps.splitlines()):
  raise RuntimeError('GPU4 has a live CUDA actor; preserve it')
 ownedpath=regular(group/'allocation_request.json');owned=json.loads(ownedpath.read_bytes())
 jobs=json.loads(subprocess.check_output(['/opt/pbs/bin/qstat','-f','-F','json',owned['job_id']],text=True,timeout=20))['Jobs']
 if list(jobs)!=[owned['job_id']]:raise RuntimeError('Exact host lifetime unavailable')
 job=next(iter(jobs.values()));h,m,s=map(int,job['Resource_List']['walltime'].split(':'))
 expiry=_pbs_start_time(job['stime'])+h*3600+m*60+s
 if (job.get('job_state')!='R' or job.get('Job_Owner','').split('@')[0]!='aicompetition06'
     or abs(expiry-owned['expires_at_unix'])>1 or time.time()>=expiry-180):raise RuntimeError('Current exact host lifetime failed')
 cgroup=capture_cgroup_admission(os.getpid())
 if any(row['limit_bytes']is not None and row['limit_bytes']<416*2**30 for row in cgroup['memory_limits']):
  raise MemoryError('Actual cgroup cannot admit unchanged full416GiB')
 peerprocess=exact(peer);rss=sum(p.memory_info().rss for p in [peerprocess,*peerprocess.children(recursive=True)] if p.status()!=psutil.STATUS_ZOMBIE)
 if psutil.virtual_memory().available+rss<416*2**30:raise MemoryError('Actual available RAM cannot admit original four-arm budget')
 command=list(old['command']);entry=str(oldcode/'tools/run_v25_parallel_pipeline.py')
 if command.count(entry)!=1:raise ValueError('Exact old controller entry required')
 command[command.index(entry)]=str(code/'tools/run_v25_d_continuation.py')
 command[command.index('--execution-manifest')+1]=str(sealpath)
 archive=group/'code_revisions'/('D_provider_before_launch_'+manifest['commit'][:12]);archive.mkdir(parents=True,exist_ok=False)
 with (archive/'D_status.json').open('xb') as stream:stream.write((group/'D/monitor/status.json').read_bytes())
 publish(intentpath,dict(format='v25_D_standalone_launch_intent_v1',command=command,private_code=str(code),
  commit=manifest['commit'],checkpoint=checkpoint,previous_D_actor=released,ABC_controller=peer,created_at=time.time(),signals_sent=False))
 env=os.environ.copy();env.update(PYTHONPATH=str(code),PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
 log=group/('D_pipeline_controller'+suffix+'.log')
 with log.open('xb') as stream:
  child=subprocess.Popen(command,cwd=code,env=env,stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
   start_new_session=True,preexec_fn=lambda:os.sched_setaffinity(0,aggregate))
 p=psutil.Process(child.pid);deadline=time.monotonic()+5
 while p.cmdline()!=command and time.monotonic()<deadline:time.sleep(.05)
 identity=actor(p)
 if identity['command']!=command or identity['UID']!=UID:raise RuntimeError('New exact D controller not observed; inspect durable intent')
 value=dict(originalD,controller_identity=identity,resource_guard_sha256=sha(code/'hiercp_v1x/v25_direct_resources.py'),
  cgroup_admission=capture_cgroup_admission(p.pid));publish(proofpath,value)
 names=('tools/run_v25_d_execution.py','tools/run_v25_d_continuation.py','hiercp_v1x/v25_standalone_d_resources.py',
  'hiercp_v1x/v25_provider_memory_runtime.py','hiercp_v1x/host_memory.py','tools/run_v25_execution.py',
  'hiercp_v1x/v25_upper_memory_runtime.py','hiercp_v1x/v25_terminal_descendant_resources.py',
  'hiercp_v1x/v25_direct_resources.py','hiercp_v1x/v25_prefix_geometry_reuse.py','hiercp_v1x/v24_upper_reuse.py',
  'tools/run_v25_downstream.py','hiercp_v1x/v25_native_cp.py','hiercp_v1x/v25_downstream.py')
 preserved=dict(staged['preserved_files'])
 preserved.update({str(code/name):checksum for name,checksum in request['source_sha256'].items()})
 seal=dict(format='v25_exact_D_standalone_continuation_v1',group_root=str(group),private_code=str(code),previous_code=str(oldcode),
  controller_identity=identity,direct_proof=dict(path=str(proofpath),sha256=sha(proofpath)),
  sibling_manifest=dict(path=str(original_manifest_path),sha256=sha(original_manifest_path)),sibling_controller_identity=peer,
  sibling_slots={arm:dict(path=original_manifest['slots'][arm]['resource_request_path'],
   sha256=original_manifest['slots'][arm]['resource_request_sha256']) for arm in 'ABC'},
  preserved_files=preserved,runtime_files={str(code/name):sha(code/name) for name in names},checkpoint=checkpoint,
  previous_D_actor=released,created_at=time.time(),scientific_configuration_unchanged=True,no_process_signaled=True)
 from hiercp_v1x.v25_standalone_d_resources import validate_seal
 validate_seal(seal);publish(sealpath,seal)
 result=dict(phase='start',PID=p.pid,create_time=p.create_time(),UID=UID,command=command,log=str(log),
  private_code=str(code),seal=str(sealpath),proof=str(proofpath),commit=manifest['commit'],checkpoint=checkpoint,
  GPU=4,CPU_affinity=list(range(6)),ABC_controller_preserved=peer,full40_GNN=True,full250_native=True,
  own_BEST26=True,signals_sent=False,debug=False)
 publish(launchpath,result);print(json.dumps(result))
'''


def build_remote_script(manifest,payload,phase):
    if phase not in ('stage','start'):raise ValueError('Explicit stage/start required')
    if not re.fullmatch('[0-9a-f]{40}',manifest['commit']):raise ValueError('Reviewed Git commit required')
    if hashlib.sha256(payload).hexdigest()!=manifest['sha256']:raise ValueError('Reviewed archive bytes differ')
    if len(manifest['paths'])!=len(set(manifest['paths'])) or set(manifest['paths'])!=set(manifest['proof']):
        raise ValueError('Exact unique reviewed file manifest required')
    return REMOTE_SCRIPT.replace('MANIFEST_DATA',repr(manifest)).replace('PHASE_DATA',repr(phase)).replace(
        'PAYLOAD_DATA',repr(base64.b64encode(payload).decode()))


def main(argv=None):
    arguments=list(sys.argv[1:] if argv is None else argv)
    if len(arguments)!=2:raise ValueError('Provide exact reviewed commit manifest and stage/start')
    manifest=json.loads(Path(arguments[0]).resolve(strict=True).read_text(encoding='utf8'))
    script=build_remote_script(manifest,Path(manifest['archive']).read_bytes(),arguments[1])
    here=Path('D:/AI project/nnunet/outputs/v24_gpu6_recovery_20261011')
    spec=importlib.util.spec_from_file_location('secure_remote',here/'remote_once.py')
    remote=importlib.util.module_from_spec(spec);spec.loader.exec_module(remote)
    value=json.loads(remote.run(script,90));path=here/('V25_D_PROVIDER_'+arguments[1].upper()+'_'+str(time.time_ns())+'.json')
    with path.open('x',encoding='utf8') as stream:json.dump(value,stream,indent=2)
    keys=('phase','commit','PID','create_time','private_code','checkpoint','GPU','CPU_affinity',
        'ABC_processes_modified','existing_code_files_modified','frozen23_sources_unchanged','signals_sent')
    print(json.dumps(dict(receipt=str(path),observed={key:value[key] for key in keys if key in value}),ensure_ascii=False))


if __name__=='__main__':main()
