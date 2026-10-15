"""Adopt one closed RAM512 cache for an admitted whole-arm GPU permutation.

No process signals or launches. All source experiment files stay read-only.
Archive previous derived metadata/helper bytes before rebinding this one cache;
preserve every CURRENT.pth, cached candidate NPZ and original runtime byte.
"""
from pathlib import Path
import argparse, gc, hashlib, importlib.util, json, os, re, shutil, socket, stat, sys, time
os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
import psutil

P=Path; UID=41833; GIB=2**30
BASE=P('/dev/shm/aicompetition06_basic_cp_ablation_20261013')
CACHE=P('/var/tmp/aicompetition06_basic_cp_seven_resume_1791934200100000000')
HELPER='tools/persist_basic_cp_seven.py'
ARMS={'multi5','multi7','multi10','candidates512','candidates1024','candidates2048','blend50'}
OPERATIONAL_FIELDS={'root','gpu','gpu_uuid','GPU_memory_total_MiB','cpu_affinity','commit','code_sha256',
 'controller_identity','controller_cgroup_admission','shared_resource_registry','RAM512_claim_ledger',
 'RAM512_source_pending_receipt','shared_banks','producer_root','PBS_shared_cache_root',
 'GPU_placement_receipt','GPU_queue_deferred_transfer','RAM512_migration_state',
 'RAM512_completed_best_validator','archive_root','GPU56_sharing_receipt',
 'execution_physical_GPU','execution_GPU_UUID','host_lifetime'}
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
def old_helper_command(command,code,manifest):
 require(command[:4]==['/home/aicompetition06/.conda/envs/nnunet/bin/python3.10','-B','-u',str(code/HELPER)],'Exact old CPU helper command prefix required')
 parser=argparse.ArgumentParser(add_help=False)
 parser.add_argument('--manifest');parser.add_argument('--output');parser.add_argument('--unfinished-checkpoint')
 parser.add_argument('--resume-cache',action='store_true');parser.add_argument('--interval',type=int,default=60)
 args=parser.parse_args(command[4:])
 require(args.manifest==str(manifest) and args.output==str(CACHE)
  and args.unfinished_checkpoint=='best' and args.resume_cache and 0<args.interval<=60,'Exact admitted previous writer contract required')
def capacity(incoming=0):
 import torch
 used=sum(exact(p).stat().st_size for p in CACHE.rglob('*')if p.is_file())
 require(used+incoming<=8*GIB and shutil.disk_usage(CACHE).free-incoming>=GIB,'Cache transition must retain8GiB and1GiB free floor')
 require(psutil.Process().memory_info().rss<=4*GIB and not torch.cuda.is_initialized(),'CPU-only helper RSS4GiB bound required')
 return used

def admitted_group(manifest_path):
 path=exact(manifest_path);group=exact(path.parent,True)
 require(path.name=='launch.json'and group.parent==BASE and group.name.startswith('Basic_CP_'),
  'Only an exact own Basic_CP cohort launch namespace may be transferred')
 return group

def arm_mapping(old_requests,new_requests):
 """Return new physical slot -> previous physical slot, without fuzzy matching."""
 expected=set(map(str,range(7)))
 require(set(old_requests)==set(new_requests)==expected,'Exactly seven physical GPU slots required')
 by_arm={}
 for gpu,q in old_requests.items():
  require(type(q.get('gpu'))is int and q['gpu']==int(gpu)and q.get('arm')in ARMS and q['arm']not in by_arm,
   'Old cohort must contain each original scientific arm exactly once')
  by_arm[q['arm']]=gpu
 require(set(by_arm)==ARMS,'Full seven-arm old cohort required')
 result={};seen=set()
 for gpu,q in new_requests.items():
  require(type(q.get('gpu'))is int and q['gpu']==int(gpu)and q.get('arm')in by_arm and q['arm']not in seen,
   'New cohort must contain each original scientific arm exactly once')
  seen.add(q['arm']);result[gpu]=by_arm[q['arm']]
 require(seen==ARMS,'Full seven-arm new cohort required')
 return result

def validate_lifetime_transfer(old_requests,new_requests,*,helper=None):
 """Only a newly proved own running PBS lifetime may replace the old witness."""
 old_ref=old_requests['0'].get('host_lifetime');new_ref=new_requests['0'].get('host_lifetime')
 require(all(q.get('host_lifetime')==old_ref for q in old_requests.values())
  and all(q.get('host_lifetime')==new_ref for q in new_requests.values()),
  'Every old/new cohort arm must share its one exact host lifetime')
 if old_ref==new_ref:return None
 for ref in(old_ref,new_ref):
  require(isinstance(ref,dict)and set(ref)=={'path','sha256','expires_at_unix'},
   'Actual pinned previous and new host lifetime references required')
 old_value=read(pin(old_ref['path'],old_ref['sha256']))
 new_value=read(pin(new_ref['path'],new_ref['sha256']))
 for ref,value in((old_ref,old_value),(new_ref,new_value)):
  require(value.get('format')=='v25_owned_GPU_allocation_v1'and value.get('owner')=='aicompetition06'
   and value.get('UID')==UID and value.get('node')=='ece-a6gpu6'
   and isinstance(value.get('job_id'),str)and re.fullmatch(r'\d+\.ECE-util1',value['job_id'])is not None
   and type(ref['expires_at_unix'])in(int,float)and value.get('expires_at_unix')==ref['expires_at_unix'],
   'Only exact own same-host previous/new lifetime provenance may transfer')
 require(new_ref['expires_at_unix']>old_ref['expires_at_unix']and new_value.get('job_state')=='R',
  'New admitted running lifetime must extend the old actual expiry')
 if helper is None:
  from tools import persist_basic_cp_seven as helper
 from hiercp_v1x import basic_cp_ablation_resources as native_resources
 actual=native_resources._pbs_query(['/opt/pbs/bin/qstat','-f','-F','json',new_value['job_id']],
  timeout=native_resources.PBS_QUERY_TIMEOUT_SECONDS)
 observation=helper.validate_host_lifetime(new_ref,new_value,actual)
 return dict(old_reference=old_ref,new_reference=new_ref,old_job_id=old_value['job_id'],
  new_job_id=new_value['job_id'],actual_new_PBS_observation=observation,
  actual_PBS_response_sha256=hashlib.sha256(actual.encode('utf8')).hexdigest(),
  validated_at_unix=time.time(),old_lifetime_extended_by_fabrication=False,
  new_running_job_independently_queried=True)

def validate_request_transfer(old_requests,new_requests,mapping,*,helper=None):
 """Every unlisted change is fatal; only placement/queue/provenance may vary."""
 require(mapping==arm_mapping(old_requests,new_requests),'Exact derived whole-arm mapping required')
 lifetime_transfer=validate_lifetime_transfer(old_requests,new_requests,helper=helper)
 for gpu,oldgpu in mapping.items():
  a=old_requests[oldgpu];b=new_requests[gpu];physical=old_requests[gpu]
  if 'GPU56_sharing_receipt' in b:
   from hiercp_v1x.basic_cp_GPU56_sharing import validate_sharing_request
   validate_sharing_request(b)
  require(all(a.get(k)==b.get(k)for k in set(a)|set(b)if k not in OPERATIONAL_FIELDS),
   'Scientific or unrecognized immutable request field changed: GPU'+gpu)
  require(b['gpu_uuid']==physical['gpu_uuid']and b['cpu_affinity']==physical['cpu_affinity']
   and b['GPU_memory_total_MiB']==physical['GPU_memory_total_MiB'],
   'New physical GPU UUID/CPU/VRAM must match that exact original physical slot')
  require(P(a['root']).name=='GPU'+oldgpu+'_'+a['arm']and P(b['root']).name=='GPU'+gpu+'_'+a['arm'],
   'Exact original/new scientific-arm cache folder names required')
 return lifetime_transfer

def cache_renames(old_requests,new_requests,mapping):
 renames={}
 for gpu,oldgpu in mapping.items():
  before=P(old_requests[oldgpu]['root']).name;after=P(new_requests[gpu]['root']).name
  if before!=after:renames[before]=after
 require(len(set(renames.values()))==len(renames),'Cache destination names must be unique')
 require(not(set(renames)&set(renames.values())),'Scientific-arm folder names may not create a rename cycle')
 return renames

def relocated_relative(relative,renames):
 p=P(relative);require(not p.is_absolute()and'..'not in p.parts,'Protected relative cache path required')
 return str(P(renames.get(p.parts[0],p.parts[0]),*p.parts[1:]))

def relocate_cache_folders(renames):
 pairs=[]
 for before,after in renames.items():
  source=exact(own_cache(CACHE/before),True);target=own_cache(CACHE/after)
  require(not target.exists(),'Transfer target already exists; unknown outcome forbids replay')
  pairs.append((source,target))
 for source,target in pairs:
  before=source.stat();os.rename(source,target);sync(CACHE);after=exact(target,True).stat()
  require((before.st_dev,before.st_ino)==(after.st_dev,after.st_ino),'Directory transfer must preserve original storage inode')

def whole_unfinished_best_metadata(metadata,info):
 saved=metadata.get('saved_resume_epoch');latest=metadata.get('latest_on_tmpfs_N')
 require(type(saved)is int and 0<saved<=250 and saved==info['checkpoint_completed_epoch'],
  'Exact whole CURRENT saved resume epoch required')
 require('latest_on_tmpfs_N'in metadata and'epochs_may_repeat_after_tmpfs_loss'in metadata,
  'Explicit known/unknown latest and replay provenance required')
 replay=metadata['epochs_may_repeat_after_tmpfs_loss']
 if latest is None:require(replay is None,'Unknown latest must preserve unknown replay count')
 else:require(type(latest)is int and saved<=latest<=250 and type(replay)is int and replay==latest-saved,
  'Known latest/replay must match exact saved epoch')
 return True

def old_actor_release(manifest,requests):
 """Read only captured own actors; never infer release from a missing status."""
 actors={}
 for gpu,row in manifest['registration'].items():
  actors['controller:'+gpu]=row['controller_identity']
  for path in P(requests[gpu]['root']).glob('*_actor.json'):
   actor=read(exact(path))
   require(actor.get('UID')==UID and isinstance(actor.get('command'),list),
    'Every prior native actor receipt must retain its exact own identity')
   actors['native:'+gpu+':'+path.name]=actor
 ledger=P(manifest['group'])/'RAM512_allocation_ledger.json'
 if ledger.exists():
  for token,claim in read(exact(ledger)).get('claims',{}).items():
   for actor in claim.get('captured',[]):actors['claim:'+token+':'+str(actor['pid'])]=actor
 return {key:released(actor)for key,actor in actors.items()}

def main(argv=None):
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--current-manifest',type=P,required=True)
 parser.add_argument('--new-manifest',type=P,required=True)
 parser.add_argument('--new-code-export',type=P,required=True)
 for name in('old-manifest-sha256','old-helper-actor-sha256','old-helper-sha256',
             'new-manifest-sha256','new-code-export-sha256','new-helper-sha256','new-commit'):
  parser.add_argument('--'+name,required=True)
 args=parser.parse_args(argv);lock=None
 try:
  require(os.getuid()==UID and socket.gethostname().split('.')[0]=='ece-a6gpu6','Exact own host/UID required')
  import fcntl,torch
  OLD=admitted_group(args.current_manifest);NEW=admitted_group(args.new_manifest)
  require(OLD!=NEW and args.new_code_export==NEW/'code_export.json','Fresh exact own transfer group/export required')
  exact(CACHE,True);old=read(pin(args.current_manifest,args.old_manifest_sha256));new=read(pin(args.new_manifest,args.new_manifest_sha256))
  require(old['group']==str(OLD)and new['group']==str(NEW),'Exact old/new cohort namespaces required')
  require(type(args.new_commit)is str and len(args.new_commit)==40 and all(c in'0123456789abcdef'for c in args.new_commit)
   and new['commit']==args.new_commit,'Literal admitted new immutable Git commit required')
  old_code=exact(old['code'],True); code=exact(new['code'],True)
  require(code==NEW/'code','Fresh immutable transfer export path required')
  export=read(pin(args.new_code_export,args.new_code_export_sha256))
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
  old_helper_command(actor['command'],old_code,args.current_manifest);helper_release=released(actor)
  require(identity['pid']==actor['pid'] and identity['birth']==actor.get('birth',actor.get('create_time'))
   and identity['command']==actor['command'],'Old cache writer/actor relationship differs')
  lock=os.open(exact(CACHE/'writer.lock'),os.O_RDWR);fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  process=psutil.Process();require({82,83}.issubset(process.cpu_affinity()),'Helper CPU82/83 unavailable');process.cpu_affinity([82,83]);torch.set_num_threads(1)
  spec=importlib.util.spec_from_file_location('reviewed_RAM512_adoption_helper',helper);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
  admitted,requests=module.admit(NEW/'launch.json');require(admitted==new,'Full new helper admission differs')
  old_requests={gpu:read(pin(P(row['root'])/'request.json',row['request_sha256']))for gpu,row in old['registration'].items()}
  require(set(requests)==set(new['registration'])==set(old['registration'])==set(map(str,range(7))),'All seven exact arms required')
  require(all(q['root']==old['registration'][gpu]['root']and q['uid']==UID
   and q['controller_identity']==old['registration'][gpu]['controller_identity']for gpu,q in old_requests.items()),
   'Original request/root/own UID/controller registration differs')
  mapping=arm_mapping(old_requests,requests)
  lifetime_transfer=validate_request_transfer(old_requests,requests,mapping,helper=module)
  renames=cache_renames(old_requests,requests,mapping)
  for before,after in renames.items():
   exact(own_cache(CACHE/before),True);require(not own_cache(CACHE/after).exists(),'Unknown prior cache-folder transfer outcome forbids replay')
  controller_release=old_actor_release(old,old_requests);native_refs={}
  for gpu in map(str,range(7)):
   a=old_requests[mapping[gpu]];b=requests[gpu]
   require(all(type(b.get(key))is int and b[key]==value for key,value in FLAGS.items())
    and b['owned_tmpfs_root']==str(BASE) and b['shared_resource_registry']==str(NEW/'shared_resource_registry.json')
    and b['RAM512_claim_ledger']==requests['6']['RAM512_claim_ledger'],'Exact common RAM512 operational contract required')
   require(b['code_sha256']==export['code_sha256'],'New full controller pins differ from export')
   prior_path=P(a['root'])/'native.json';fresh_path=P(b['root'])/'native.json'
   if prior_path.exists():
    prior=read(exact(prior_path));fresh=read(exact(fresh_path));native_refs[gpu]=prior
    for key in('native_root','private_runtime','private_pre','private_data_folder','trainer','baseline','split','bank','bank_sha256'):
     require(prior.get(key)==fresh.get(key),'Original native model/dataset/checkpoint path changed: '+gpu+'/'+key)
  require(set(native_refs)==set(map(str,range(7))),'All seven original full native states required before transfer')
  currents={};pending={};protected={}
  for gpu in map(str,range(7)):
   previous_request=old_requests[mapping[gpu]];folder=CACHE/P(previous_request['root']).name;path=folder/'CURRENT.pth'
   require(path.exists(),'Existing whole CURRENT is missing: GPU'+gpu+' arm '+previous_request['arm'])
   native=native_refs[gpu];model=read(exact(P(previous_request['root'])/'actual_model.json'));metadata=read(exact(folder/'CURRENT.json'))
   require(model['epochs']==250 and model['physical_batch']==model['effective_batch']==2
    and len(model['actual_train_cases'])==105 and len(model['actual_validation_cases'])==26,
    'Original full250/B2/105/26 native model evidence required')
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
     and whole_unfinished_best_metadata(metadata,info),'Full unfinished BEST role differs')
   elif metadata['role']=='BEST_EVALUATED26':
    require(metadata['checkpoint_is_training_resume']is False and metadata['native_evaluation_complete']is True
     and metadata['training_completed_epochs']==250 and metadata['evaluated_cases']==26,'Completed BEST26 cannot become resume250')
   else:raise ValueError('Unexpected prior cache role: '+metadata['role'])
   if(folder/'PENDING.json').exists():
    row=read(exact(folder/'PENDING.json'));require(row['sha256']==checksum and row['copied_at_unix']==metadata['copied_at_unix'],'Unresolved pending publication must be inspected before adoption');pending[gpu]=row['sha256']
   require(not list(folder.glob('.CURRENT.pth_*')),'Incomplete CURRENT temporary prevents adoption')
   currents[gpu]=dict(info,sha256=checksum,role=metadata['role']);capacity()
  for path in CACHE.rglob('*'):
   if path.is_file() and(path.suffix in('.pth','.npz')or path.name in('CURRENT.json','PENDING.json')or 'runtime'in path.relative_to(CACHE).parts):protected[str(path.relative_to(CACHE))]=digest(path)
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
  relocate_cache_folders(renames)
  protected={relocated_relative(relative,renames):checksum for relative,checksum in protected.items()}
  copy_new(NEW/'launch.json',CACHE/'source_manifest.json');copy_new(helper,CACHE/'persistence_helper.py')
  lifetime_ref=requests['0']['host_lifetime']
  copy_new(pin(lifetime_ref['path'],lifetime_ref['sha256']),CACHE/'host_lifetime.json')
  for relative,checksum in protected.items():require(digest(CACHE/relative)==checksum,'Protected CURRENT/NPZ/runtime bytes changed during adoption')
  publish(CACHE/'identity.json',dict(identity,status='ADOPTED_WAITING_FOR_RAM512_HELPER',commit=args.new_commit,
   helper_source_sha256=args.new_helper_sha256,adopted_from_manifest_sha256=args.old_manifest_sha256,
   adopted_to_manifest_sha256=args.new_manifest_sha256,adoption_history=str(history),previous_helper_released=True,
   host_lifetime=lifetime_ref,host_lifetime_transfer=lifetime_transfer))
  receipt=dict(format='basic_cp_RAM512_operational_cache_adoption_v1',old_manifest_sha256=args.old_manifest_sha256,
   new_manifest_sha256=args.new_manifest_sha256,old_helper_source_sha256=args.old_helper_sha256,
   new_helper_source_sha256=args.new_helper_sha256,new_commit=args.new_commit,new_code_export_sha256=args.new_code_export_sha256,
   helper_change_is_operational=True,actual_CPU_decoded_CURRENT_states=currents,completed_pending_publications=pending,
   protected_bytes_sha256=protected,archived_metadata_sha256={k:v['sha256']for k,v in sources.items()},
   previous_helper_release=helper_release,previous_controller_release=controller_release,
   preserved_CURRENT_count=len(currents),preserved_NPZ_count=sum(k.endswith('.npz')for k in protected),
   new_common_ledger_GPUs=list(range(7)),all_original_experiment_files_unchanged=True,processes_signaled=False,
   history=str(history),candidate_arrays_unchanged=True,old_metadata_archived_before_replacement=True,
   approved_physical_transfer=mapping,cache_directory_renames=renames,checkpoint_NPZ_copies_during_transfer=0,
   host_lifetime_transfer=lifetime_transfer,new_host_lifetime_sha256=lifetime_ref['sha256'])
  publish(CACHE/'adoption_RAM_pause.json',receipt);capacity()
  print(json.dumps(dict(format=receipt['format'],old_manifest_sha256=args.old_manifest_sha256,
   new_manifest_sha256=args.new_manifest_sha256,approved_physical_transfer=mapping,
   preserved_CURRENT_count=len(currents),preserved_NPZ_count=receipt['preserved_NPZ_count'],
   protected_file_count=len(protected),history=str(history),checkpoint_NPZ_copies_during_transfer=0,
   all_original_experiment_files_unchanged=True,processes_signaled=False)),flush=True)
  return receipt
 finally:
  if lock is not None:
   try:fcntl.flock(lock,fcntl.LOCK_UN)
   finally:os.close(lock)

if __name__=='__main__':main()
