"""PREPARED ONLY: adopt own closed rescue cache for the admitted RAM512 cohort.

No process signals or launches. All source experiment files stay read-only.
Archive previous derived metadata/helper bytes before rebinding this one cache;
preserve every CURRENT.pth, cached candidate NPZ and original runtime byte.
"""
from pathlib import Path
import argparse, fcntl, gc, hashlib, importlib.util, json, os, shutil, socket, stat, sys, time
os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
import psutil, torch

P=Path; UID=41833; GIB=2**30
BASE=P('/dev/shm/aicompetition06_basic_cp_ablation_20261013')
OLD=BASE/'Basic_CP_common_ledger_recovery_1791972000100000000'
NEW=BASE/'Basic_CP_RAM512_common_1791977800100000000'
CACHE=P('/var/tmp/aicompetition06_basic_cp_seven_resume_1791934200100000000')
HELPER='tools/persist_basic_cp_seven.py'
FLAGS=dict(memory_budget_revision=512, aggregate_memory_cap_GiB=512,
 effective_native_RSS_limit_GiB=32, effective_shared_bank_RSS_limit_GiB=320,
 owned_tmpfs_envelope_GiB=144, post_source_owned_tmpfs_envelope_GiB=272,
 auxiliary_RAM_envelope_GiB=16)

def require(ok, message):
 if not ok: raise ValueError(message)
def read(path):
 value=json.loads(P(path).read_bytes()); require(isinstance(value,dict),'JSON mapping required'); return value
def exact(path, directory=False):
 path=P(path).absolute(); row=path.stat()
 require(path.resolve(strict=True)==path and not any(p.is_symlink() for p in(path,*path.parents)), 'Exact own nonsymlink evidence required: '+str(path))
 require(row.st_uid==UID and(stat.S_ISDIR(row.st_mode) if directory else stat.S_ISREG(row.st_mode)), 'UID41833 regular evidence required')
 return path
def inode(row): return(row.st_dev,row.st_ino,row.st_size,row.st_mtime_ns)
def digest(path):
 with exact(path).open('rb') as stream:
  before=os.fstat(stream.fileno()); h=hashlib.sha256()
  for block in iter(lambda:stream.read(8*2**20),b''): h.update(block)
  require(inode(before)==inode(os.fstat(stream.fileno())),'Held evidence changed during SHA')
 return h.hexdigest()
def pin(path, checksum):
 require(type(checksum)is str and len(checksum)==64 and all(c in'0123456789abcdef'for c in checksum),'Literal full lowercase SHA256 required')
 path=exact(path); require(digest(path)==checksum,'Pinned admission changed: '+str(path)); return path
def own_cache(path):
 path=P(path).absolute()
 require(path==CACHE or CACHE in path.parents,'Operation escapes own fixed cache')
 require(not any(p.is_symlink()for p in(path,*path.parents)),'Cache symlink forbidden'); return path
def sync(path):
 fd=os.open(path,os.O_RDONLY)
 try: os.fsync(fd)
 finally: os.close(fd)
def copy_new(source,target):
 source=exact(source); target=own_cache(target); target.parent.mkdir(parents=True,exist_ok=True); checksum=digest(source)
 with source.open('rb')as inp,target.open('xb')as out:
  before=os.fstat(inp.fileno()); shutil.copyfileobj(inp,out,8*2**20); out.flush(); os.fsync(out.fileno())
  require(inode(before)==inode(os.fstat(inp.fileno())),'Archive source changed')
 require(digest(target)==checksum==digest(source),'History copy must be independent byte-exact'); sync(target.parent)
def publish(path,value):
 path=own_cache(path); temp=path.with_name('.RAM512_adopt_'+path.name+'_'+str(time.time_ns()))
 with temp.open('x',encoding='utf8')as out: json.dump(value,out,indent=2,allow_nan=False);out.flush();os.fsync(out.fileno())
 os.replace(temp,path);sync(path.parent)
def released(actor):
 require(actor.get('UID')==UID and type(actor.get('pid'))is int and actor['pid']>0,'Exact own actor required')
 birth=actor.get('birth',actor.get('create_time')); require(type(birth)in(int,float)and birth>0,'Exact actor birth required')
 try: process=psutil.Process(actor['pid'])
 except psutil.NoSuchProcess: return dict(state='gone',reason='NoSuchProcess')
 if process.create_time()!=birth: return dict(state='gone',reason='original_birth_replaced',replacement_not_admitted=True)
 require(process.uids().real==UID,'Original actor UID changed')
 if process.status()in(psutil.STATUS_ZOMBIE,psutil.STATUS_DEAD): return dict(state='terminal',reason=process.status())
 require(process.cmdline()==actor['command'],'Original live actor argv changed')
 raise RuntimeError('Exact old cache/cohort actor is still live: '+str(actor['pid']))
def old_helper_command(command,code):
 require(command[:4]==['/home/aicompetition06/.conda/envs/nnunet/bin/python3.10','-B','-u',str(code/HELPER)],'Exact old CPU helper command prefix required')
 parser=argparse.ArgumentParser(add_help=False)
 parser.add_argument('--manifest');parser.add_argument('--output');parser.add_argument('--unfinished-checkpoint')
 parser.add_argument('--resume-cache',action='store_true');parser.add_argument('--interval',type=int,default=60)
 args=parser.parse_args(command[4:])
 require(args.manifest==str(OLD/'launch.json') and args.output==str(CACHE)
  and args.unfinished_checkpoint=='best' and args.resume_cache and 0<args.interval<=60,'Exact admitted previous writer contract required')
def capacity(incoming=0):
 used=sum(exact(p).stat().st_size for p in CACHE.rglob('*')if p.is_file())
 require(used+incoming<=8*GIB and shutil.disk_usage(CACHE).free-incoming>=GIB,'Cache transition must retain8GiB and1GiB free floor')
 require(psutil.Process().memory_info().rss<=4*GIB and not torch.cuda.is_initialized(),'CPU-only helper RSS4GiB bound required')
 return used

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 for name in('old-manifest-sha256','old-helper-actor-sha256','old-helper-sha256',
             'new-manifest-sha256','new-code-export-sha256','new-helper-sha256','new-commit'):
  parser.add_argument('--'+name,required=True)
 args=parser.parse_args(); lock=None
 try:
  require(os.getuid()==UID and socket.gethostname().split('.')[0]=='ece-a6gpu6','Exact own host/UID required')
  exact(CACHE,True); old=read(pin(OLD/'launch.json',args.old_manifest_sha256)); new=read(pin(NEW/'launch.json',args.new_manifest_sha256))
  require(old['group']==str(OLD) and new['group']==str(NEW),'Exact G5→RAM512 G6 namespaces required')
  require(type(args.new_commit)is str and len(args.new_commit)==40 and all(c in'0123456789abcdef'for c in args.new_commit)
   and new['commit']==args.new_commit,'Literal admitted new immutable Git commit required')
  old_code=exact(old['code'],True); code=exact(new['code'],True)
  require(code==NEW/'code','Fresh G6 immutable export path required')
  export=read(pin(NEW/'code_export.json',args.new_code_export_sha256))
  require(export.get('code')==str(code) and export.get('commit')==args.new_commit and isinstance(export.get('code_sha256'),dict),'Exact new code export required')
  for relative,checksum in export['code_sha256'].items():
   path=P(relative);require(not path.is_absolute()and'..'not in path.parts,'Export relative path traversal forbidden');pin(code/path,checksum)
  helper=pin(code/HELPER,args.new_helper_sha256)
  require(export['code_sha256'].get(HELPER)==args.new_helper_sha256,'New helper bytes must be explicitly export-pinned')
  identity=read(exact(CACHE/'identity.json')); previous_adoption=read(exact(CACHE/'adoption_RAM_pause.json'))
  require(digest(CACHE/'source_manifest.json')==args.old_manifest_sha256==previous_adoption['new_manifest_sha256'],'Actual old cache manifest/adoption differs')
  require(identity['uid']==UID and identity['node']=='ece-a6gpu6' and identity['commit']==old['commit']
   and identity['helper_source_sha256']==args.old_helper_sha256 and identity['unfinished_checkpoint_policy']=='best','Actual previous writer/cache identity differs')
  pin(CACHE/'persistence_helper.py',args.old_helper_sha256);pin(old_code/HELPER,args.old_helper_sha256)
  actor=read(pin(OLD/'checkpoint_preserver_actor.json',args.old_helper_actor_sha256))
  require(actor['pid']==4167493,'Only exact already released G5 helper4167493 admitted')
  old_helper_command(actor['command'],old_code);helper_release=released(actor)
  require(identity['pid']==actor['pid'] and identity['birth']==actor.get('birth',actor.get('create_time'))
   and identity['command']==actor['command'],'Old cache writer/actor relationship differs')
  lock=os.open(exact(CACHE/'writer.lock'),os.O_RDWR);fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  process=psutil.Process();require({82,83}.issubset(process.cpu_affinity()),'Helper CPU82/83 unavailable');process.cpu_affinity([82,83]);torch.set_num_threads(1)
  spec=importlib.util.spec_from_file_location('reviewed_RAM512_adoption_helper',helper);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
  admitted,requests=module.admit(NEW/'launch.json');require(admitted==new,'Full new helper admission differs')
  old_requests={gpu:read(pin(P(row['root'])/'request.json',row['request_sha256']))for gpu,row in old['registration'].items()}
  require(set(requests)==set(new['registration'])==set(old['registration'])==set(map(str,range(7))),'All seven exact arms required')
  science=('arm','gpu','gpu_uuid','copies_per_CP_event','candidates_per_source','bank','bank_sha256',
   'readonly_admission','readonly_admission_sha256','candidate_binding','candidate_binding_sha256',
   'prepared256_witness','blend_mode','blend_alpha','donor_fraction','online_seed','primary_joint_resampling_policy',
   'RSS_limit_bytes','cpu_affinity','shared_bank_workers','shared_bank_CPU_affinity','shared_bank_RSS_limit_GiB','baseline','split')
  controller_release={};native_refs={}
  for gpu in map(str,range(7)):
   a=old_requests[gpu];b=requests[gpu]
   require(all(a.get(key)==b.get(key)for key in science) and P(a['root']).name==P(b['root']).name,'Scientific contract changed: GPU'+gpu)
   controller_release[gpu]=released(old['registration'][gpu]['controller_identity'])
   require(all(type(b.get(key))is int and b[key]==value for key,value in FLAGS.items())
    and b['owned_tmpfs_root']==str(BASE) and b['shared_resource_registry']==str(NEW/'shared_resource_registry.json')
    and b['RAM512_claim_ledger']==requests['6']['RAM512_claim_ledger'],'Exact common RAM512 operational contract required')
   require(b['code_sha256']==export['code_sha256'],'New full controller pins differ from export')
   prior_path=P(a['root'])/'native.json';fresh_path=P(b['root'])/'native.json'
   if prior_path.exists():
    prior=read(exact(prior_path));fresh=read(exact(fresh_path));native_refs[gpu]=prior
    for key in('native_root','private_runtime','private_pre','private_data_folder','trainer','baseline','split','bank','bank_sha256'):
     require(prior.get(key)==fresh.get(key),'Original native model/dataset/checkpoint path changed: '+gpu+'/'+key)
  currents={};pending={};protected={}
  for gpu in map(str,range(7)):
   folder=CACHE/P(old_requests[gpu]['root']).name;path=folder/'CURRENT.pth'
   if not path.exists():
    require(gpu=='5','Existing completed whole CURRENT is missing: GPU'+gpu);continue
   native=native_refs[gpu];model=read(exact(P(old_requests[gpu]['root'])/'actual_model.json'));metadata=read(exact(folder/'CURRENT.json'))
   with exact(path).open('rb')as stream:
    before=os.fstat(stream.fileno());value=torch.load(stream,map_location='cpu',weights_only=False)
    try:info=module.checkpoint_info(value,native['trainer'])
    finally:del value;gc.collect()
    require(inode(before)==inode(os.fstat(stream.fileno())),'CURRENT inode changed during full CPU decode')
   checksum=digest(path)
   require(checksum==metadata['sha256'] and info['checkpoint_completed_epoch']==metadata['checkpoint_completed_epoch']
    and info['logger_lengths']==metadata['logger_lengths'] and info['optimizer_states']==metadata['optimizer_states']
    and info['network_tensor_elements']>=model['parameters'],'Actual whole CURRENT/model metadata differs')
   require(metadata['scaler_present']is True,'Whole CURRENT GradScaler evidence required')
   if metadata['role']=='OWN_BEST_TRAINING_STATE':
    require(metadata['native_evaluation_complete']is False and metadata['complete_latest_backup']is False
     and metadata['saved_resume_epoch']==info['checkpoint_completed_epoch']
     and metadata['latest_on_tmpfs_N']>=metadata['saved_resume_epoch']>0,'Full unfinished BEST role differs')
   elif metadata['role']=='BEST_EVALUATED26':
    require(metadata['checkpoint_is_training_resume']is False and metadata['native_evaluation_complete']is True
     and metadata['training_completed_epochs']==250 and metadata['evaluated_cases']==26,'Completed BEST26 cannot become resume250')
   else:raise ValueError('Unexpected prior cache role: '+metadata['role'])
   if(folder/'PENDING.json').exists():
    row=read(exact(folder/'PENDING.json'));require(row['sha256']==checksum and row['copied_at_unix']==metadata['copied_at_unix'],'Unresolved pending publication must be inspected before adoption');pending[gpu]=row['sha256']
   require(not list(folder.glob('.CURRENT.pth_*')),'Incomplete CURRENT temporary prevents adoption')
   currents[gpu]=dict(info,sha256=checksum,role=metadata['role']);capacity()
  for path in CACHE.rglob('*'):
   if path.is_file() and(path.suffix in('.pth','.npz') or 'runtime'in path.relative_to(CACHE).parts):protected[str(path.relative_to(CACHE))]=digest(path)
  sources={}
  def archive_source(path,keep=False):
   path=exact(path);require(CACHE in path.parents,'Archive only own operational cache metadata')
   key=str(path.relative_to(CACHE));require(key not in sources,'Duplicate own metadata history target');sources[key]=dict(path=path,sha256=digest(path),keep=keep)
  for path in CACHE.iterdir():
   if path.is_file() and(path.suffix=='.json' or path.name=='persistence_helper.py'):archive_source(path)
  for gpu in map(str,range(7)):
   folder=CACHE/P(old_requests[gpu]['root']).name
   if not folder.exists():continue
   for path in folder.glob('*.json'):archive_source(path,keep=path.name in('CURRENT.json','PENDING.json'))
   for path in(folder/'artifacts').rglob('*'):
    if path.is_file():require(path.suffix in('.json','.csv','.md','.log','.txt'),'Unexpected protected binary artifact');archive_source(path)
  for path in(CACHE/'banks').rglob('*.json'):archive_source(path)
  incoming=sum(row['path'].stat().st_size for row in sources.values())+helper.stat().st_size+(OLD/'launch.json').stat().st_size+(NEW/'launch.json').stat().st_size+2**20
  capacity(incoming)
  history=CACHE/'history'/('RAM512_'+args.new_manifest_sha256)
  require(not history.exists(),'This RAM512 adoption already attempted; inspect preserved outcome');history.mkdir(mode=0o700)
  copy_new(OLD/'launch.json',history/'manifests/old.json');copy_new(NEW/'launch.json',history/'manifests/new.json')
  copy_new(OLD/'checkpoint_preserver_actor.json',history/'released_old_helper_actor.json');copy_new(NEW/'code_export.json',history/'new_code_export.json')
  # All metadata must be archived and verified before the first operational removal.
  for relative,row in sources.items():copy_new(row['path'],history/'cache_metadata'/relative)
  for relative,row in sources.items():
   require(digest(row['path'])==row['sha256']==digest(history/'cache_metadata'/relative),'Archived metadata changed before operational adoption')
  for row in sources.values():
   if not row['keep']:own_cache(row['path']).unlink();sync(row['path'].parent)
  copy_new(NEW/'launch.json',CACHE/'source_manifest.json');copy_new(helper,CACHE/'persistence_helper.py')
  for relative,checksum in protected.items():require(digest(CACHE/relative)==checksum,'Protected CURRENT/NPZ/runtime bytes changed during adoption')
  publish(CACHE/'identity.json',dict(identity,status='ADOPTED_WAITING_FOR_RAM512_HELPER',commit=args.new_commit,
   helper_source_sha256=args.new_helper_sha256,adopted_from_manifest_sha256=args.old_manifest_sha256,
   adopted_to_manifest_sha256=args.new_manifest_sha256,adoption_history=str(history),previous_helper_released=True))
  receipt=dict(format='basic_cp_RAM512_operational_cache_adoption_v1',old_manifest_sha256=args.old_manifest_sha256,
   new_manifest_sha256=args.new_manifest_sha256,old_helper_source_sha256=args.old_helper_sha256,
   new_helper_source_sha256=args.new_helper_sha256,new_commit=args.new_commit,new_code_export_sha256=args.new_code_export_sha256,
   helper_change_is_operational=True,actual_CPU_decoded_CURRENT_states=currents,completed_pending_publications=pending,
   protected_bytes_sha256=protected,archived_metadata_sha256={k:v['sha256']for k,v in sources.items()},
   previous_helper_release=helper_release,previous_controller_release=controller_release,
   preserved_CURRENT_count=len(currents),preserved_NPZ_count=sum(k.endswith('.npz')for k in protected),
   new_common_ledger_GPUs=list(range(7)),all_original_experiment_files_unchanged=True,processes_signaled=False,
   history=str(history),candidate_arrays_unchanged=True,old_metadata_archived_before_replacement=True)
  publish(CACHE/'adoption_RAM_pause.json',receipt);capacity();print(json.dumps(receipt),flush=True)
 finally:
  if lock is not None:
   try:fcntl.flock(lock,fcntl.LOCK_UN)
   finally:os.close(lock)

if __name__=='__main__':main()
