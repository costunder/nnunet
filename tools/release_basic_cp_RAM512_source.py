"""LOCAL PREPARED only: exact CPU-source capture, approved release, full reuse.

No action occurs on import. Capture reads existing state; release requires the
exact reviewed capture SHA plus explicit producer/worker approvals. Every old
candidate file is retained. This file is not deployed or executed by its author.
"""
from pathlib import Path
import argparse,hashlib,json,os,signal,subprocess,sys,time
import psutil

UID=41833
BASE=Path('/dev/shm/aicompetition06_basic_cp_ablation_20261013')
G3=BASE/'Basic_CP_common_ledger_recovery_1791972000100000000'
COMMIT='fd296b69b09bcb2eb789d30dea92ccb23b162857'
FORMAT='basic_cp_exact_G3_CPU_source_capture_v1'
UPDATES_FORMAT='basic_cp_current_candidate_source_recovery_updates_v1'

def require(value,message):
 if not value:raise ValueError(message)

def read(path):return json.loads(Path(path).read_bytes())

def sha(path):
 with Path(path).open('rb')as stream:
  before=os.fstat(stream.fileno());digest=hashlib.sha256()
  for block in iter(lambda:stream.read(8*2**20),b''):digest.update(block)
  after=os.fstat(stream.fileno())
 require((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns),'Held evidence changed')
 return digest.hexdigest()

def owned(path):
 path=Path(path)
 require(path.is_absolute()and path.resolve(strict=True)==path and path.is_file()and not any(p.is_symlink()for p in(path,*path.parents))and path.stat().st_uid==UID,'Exact own regular evidence required')
 return path

def write_new(path,value):
 path=Path(path)
 require(path.is_absolute()and BASE in path.parents and G3 not in path.parents and path.parent.is_dir()and not any(p.is_symlink()for p in(path,*path.parents))and path.parent.stat().st_uid==UID,'New own operation output outside original G3 required')
 with path.open('x',encoding='utf8')as stream:
  json.dump(value,stream,ensure_ascii=False,allow_nan=False);stream.flush();os.fsync(stream.fileno())

def ident(process,role):
 return dict(pid=process.pid,birth=process.create_time(),UID=process.uids().real,command=process.cmdline(),PPID=process.ppid(),CPU=process.cpu_affinity(),role=role)

def same_identity(left,right):
 return left['pid']==right['pid']and left.get('birth',left.get('create_time'))==right.get('birth',right.get('create_time'))and left['UID']==right['UID']and left['command']==right['command']

def observation(row,*,strict_command=False):
 try:
  process=psutil.Process(row['pid']);actual=ident(process,row['role']);state=process.status()
 except psutil.NoSuchProcess:return dict(released=True,reason='NoSuchProcess',expected=row)
 if actual['birth']!=row['birth']:return dict(released=True,reason='Original birth released',expected=row)
 require(actual['UID']==row['UID']==UID,'Captured source identity UID changed')
 if strict_command:
  require(actual['command']==row['command'],'Exact captured source command changed before signal')
  require(actual['CPU']==row['CPU'],'Captured source64CPU affinity changed before signal')
 return dict(released=state in(psutil.STATUS_ZOMBIE,psutil.STATUS_DEAD),actual=actual,status=state,command_unchanged=actual['command']==row['command'],expected=row)

def no_source_CUDA(rows):
 output=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader,nounits'],text=True,timeout=5)
 pids=set()
 for line in output.splitlines():
  if not line.strip():continue
  cells=[cell.strip()for cell in line.split(',')]
  require(len(cells)==2 and cells[0].isdigit(),'Exact CUDA process evidence required');pids.add(int(cells[0]))
 actual_source_pids={row['pid']for row in rows if not observation(row)['released']}
 require(not pids.intersection(actual_source_pids),'CPU-source actor has forbidden CUDA ownership')
 return sorted(pids)

def capture():
 require(os.getuid()==UID,'Own account required')
 manifest_path=owned(G3/'launch.json');manifest=read(manifest_path);registration=manifest['registration']['5']
 request_path=owned(Path(registration['root'])/'request.json');request=read(request_path)
 require(manifest['commit']==request['commit']==COMMIT and sha(request_path)==registration['request_sha256'],'Exact fd296 source request required')
 require(request['gpu']==5 and request['arm']=='candidates2048'and request['candidates_per_source']==2048 and request['shared_bank_workers']==24 and request['shared_bank_RSS_limit_GiB']==384,'Original full642/2048/24worker/384GiB source required')
 require(request['controller_identity']==registration['controller_identity']and request['shared_banks']==str(G3/'banks'),'Source row and bank root differ')
 producer_path=owned(Path(request['root'])/'candidate_bank_actor.json');producer_receipt=read(producer_path)
 controller=ident(psutil.Process(registration['controller_identity']['pid']),'controller');producer=ident(psutil.Process(producer_receipt['pid']),'producer')
 require(same_identity(controller,registration['controller_identity'])and same_identity(producer,producer_receipt)and producer['PPID']==controller['pid'],'Exact producer/controller ancestry required')
 require(controller['pid']==4149273 and controller['birth']==1791973256.08 and producer['pid']==4155522 and producer['birth']==1791974178.59,'Reviewed G3 CPU actors differ')
 children=[]
 for process in psutil.Process(producer['pid']).children(recursive=True):
  row=ident(process,'unclassified')
  row['role']='worker'if '--multiprocessing-fork'in row['command']else'tracker'if any('resource_tracker'in arg for arg in row['command'])else'unclassified'
  children.append(row)
 workers=[row for row in children if row['role']=='worker'];trackers=[row for row in children if row['role']=='tracker']
 require(len(workers)==24 and len(trackers)==1 and len(children)==25,'Exactly24 source workers and one tracker required')
 rows=[controller,producer,*workers,*trackers]
 require(len({(row['pid'],row['birth'])for row in rows})==27 and all(row['UID']==UID and row['CPU']==request['shared_bank_CPU_affinity']for row in rows),'Exact own64CPU source cohort required')
 no_source_CUDA(rows)
 files={str(path):sha(path)for path in(manifest_path,request_path,producer_path,owned(request['candidate_binding']),owned(request['prepared256_witness']))}
 return dict(format=FORMAT,group=str(G3),commit=COMMIT,source_request=str(request_path),source_request_sha256=files[str(request_path)],files_sha256=files,actors=rows,workers=workers,created_at=time.time(),processes_signaled=False,source_files_written=False)

def admit_capture(path,digest):
 path=owned(path);require(sha(path)==digest,'Reviewed capture SHA changed');value=read(path)
 require(value['format']==FORMAT and value['group']==str(G3)and value['commit']==COMMIT and len(value['actors'])==27 and len(value['workers'])==24,'Exact full source capture required')
 require(all(row['UID']==UID for row in value['actors'])and len({(row['pid'],row['birth'])for row in value['actors']})==27,'Unique owned identities required')
 require(all(sha(owned(path))==digest for path,digest in value['files_sha256'].items()),'Original pinned source metadata changed')
 return value

def signal_single(row,number):
 state=observation(row,strict_command=True)
 if state['released']:return dict(pid=row['pid'],signal_sent=False,reason='Already released original identity')
 require(row['role']in('producer','worker'),'Only explicitly approved own CPU source producer/worker may be signaled')
 try:os.kill(row['pid'],number)
 except ProcessLookupError:return dict(pid=row['pid'],signal_sent=False,reason='Naturally released before individual signal')
 return dict(pid=row['pid'],birth=row['birth'],signal_sent=True,signal=int(number))

def wait_released(rows,seconds):
 deadline=time.monotonic()+seconds
 while True:
  states=[observation(row)for row in rows]
  if all(state['released']for state in states):return True,states
  if time.monotonic()>=deadline:return False,states
  time.sleep(.25)

def release(value,output,*,approved_producer,approved_workers,grace_seconds=5):
 producer=next(row for row in value['actors']if row['role']=='producer')
 require(approved_producer==producer['pid']and approved_workers is True,'Reported exact producer and24 worker targets require explicit release approval')
 require(0<=grace_seconds<=55,'Bounded monitored phase required')
 no_source_CUDA(value['actors'])
 # The old ProcessPool catches Exception, not KeyboardInterrupt; its context
 # waits all queued work. Capture that reason explicitly instead of hiding it.
 intent=dict(format='basic_cp_exact_CPU_source_common_ledger_release_v1',reason='INTERRUPTED_FOR_COMMON_LEDGER_REBIND',capture=value,producer_SIGINT_target=producer,conditional_individual_worker_SIGTERM_targets=value['workers'],no_tracker_controller_GPU_or_group_signals=True,created_at=time.time())
 write_new(Path(output).with_name(Path(output).name+'.intent.json'),intent)
 sent=[signal_single(producer,signal.SIGINT)];released,states=wait_released(value['actors'],grace_seconds)
 if not released:
  no_source_CUDA(value['actors'])
  for worker in value['workers']:sent.append(signal_single(worker,signal.SIGTERM))
  released,states=wait_released(value['actors'],55)
 proof=dict(intent,status='RELEASED'if released else'BLOCKED',actors_after=states,individual_signals=sent,completed_at=time.time(),all_source_released=released,old_source_files_written=False)
 write_new(output,proof);return proof

def prepare_snapshot_updates(value,snapshot_path,updates_path):
 released,states=wait_released(value['actors'],0)
 require(released,'Every captured source actor must naturally release before snapshot')
 # The unchanged strict prior-admission additionally requires controller and
 # producer to be fully reaped; it intentionally does not accept zombies.
 for row in value['actors']:
  if row['role']in('controller','producer'):
   try:process=psutil.Process(row['pid'])
   except psutil.NoSuchProcess:continue
   require(process.create_time()!=row['birth'],'Source producer/controller still needs natural reaping')
 no_source_CUDA(value['actors']);request=read(value['source_request']);code=Path(request['controller_identity']['command'][3]).parents[1]
 for relative,digest in request['code_sha256'].items():require(sha(owned(code/relative))==digest,'Exact fd296 code export changed')
 sys.path.insert(0,str(code))
 from hiercp_v1x.basic_cp_large_bank_recovery import snapshot_interrupted_outputs
 snapshot=snapshot_interrupted_outputs(request['candidate_binding'],request['candidate_binding_sha256'],value['source_request'],value['source_request_sha256'],read(request['prepared256_witness']),snapshot_path,released_actors=value['workers'])
 require(snapshot['counts']['512']['VALID']==642 and snapshot['counts']['1024']['VALID']==642,'Full sealed512/1024 must remain entirely valid')
 updates=dict(previous_partial_banks=request['shared_banks'],previous_producer_request=value['source_request'],previous_producer_request_sha256=value['source_request_sha256'],interrupted_candidate_recovery_snapshot=str(snapshot_path),interrupted_candidate_recovery_snapshot_sha256=sha(snapshot_path))
 result=dict(format=UPDATES_FORMAT,request_updates=updates,counts=snapshot['counts'],all642sources_retained=True,original_source_arrays_geometry_workers_and_RSS_preserved=True,source_actor_release=states,source_files_written=False,created_at=time.time())
 write_new(updates_path,result);return result

def main():
 parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('capture','release-approved','snapshot'))
 parser.add_argument('--output',type=Path,required=True);parser.add_argument('--capture',type=Path);parser.add_argument('--capture-sha256');parser.add_argument('--snapshot-output',type=Path)
 parser.add_argument('--approved-producer',type=int);parser.add_argument('--approved-captured-workers',action='store_true');parser.add_argument('--grace-seconds',type=float,default=5)
 args=parser.parse_args();os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
 if args.stage=='capture':result=capture();write_new(args.output,result)
 else:
  value=admit_capture(args.capture,args.capture_sha256)
  if args.stage=='release-approved':result=release(value,args.output,approved_producer=args.approved_producer,approved_workers=args.approved_captured_workers,grace_seconds=args.grace_seconds)
  else:require(args.snapshot_output is not None,'Explicit fresh snapshot destination required');result=prepare_snapshot_updates(value,args.snapshot_output,args.output)
 print(json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
