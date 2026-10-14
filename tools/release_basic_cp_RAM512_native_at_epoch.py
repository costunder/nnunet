"""PREPARED ONLY; one exact own trainer SIGINT at a verified early next epoch.

Default inspection never signals. No controller, parent/session, group, other
user, or DA child is signaled. Full native250/B2/105/26 settings stay unchanged.
"""
from pathlib import Path
import argparse,datetime,gc,hashlib,json,math,os,re,signal,socket,stat,time
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import psutil
P=Path;UID=41833
BASE=P('/dev/shm/aicompetition06_basic_cp_ablation_20261013')
GROUP=BASE/'Basic_CP_common_ledger_recovery_1791972000100000000'
EXPECTED_TRAINERS={0:(4155489,1791974177.79),2:(4155527,1791974178.71),3:(4155921,1791974199.12),4:(4165223,1791974388.86)}
ARMS={0:'multi5',2:'multi10',3:'candidates512',4:'candidates1024'}
LOGGERS={'mean_fg_dice','ema_fg_dice','dice_per_class_or_region','train_losses','val_losses','lrs','epoch_start_timestamps','epoch_end_timestamps'}
read=lambda p:json.loads(P(p).read_bytes())
def require(ok,message):
 if not ok:raise ValueError(message)
def exact_file(path,directory=False):
 path=P(path).absolute();s=path.stat()
 require(path.resolve(strict=True)==path and not any(p.is_symlink()for p in(path,*path.parents)),'Exact non-symlink evidence required')
 require(s.st_uid==UID and(stat.S_ISDIR(s.st_mode)if directory else stat.S_ISREG(s.st_mode)),'Own UID regular evidence required');return path
def sha(path):
 h=hashlib.sha256()
 with exact_file(path).open('rb')as f:
  for b in iter(lambda:f.read(8*2**20),b''):h.update(b)
 return h.hexdigest()
def inode(s):return(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)
def birth(a):return a.get('birth',a.get('create_time'))
def process_identity(p):return dict(pid=p.pid,birth=p.create_time(),UID=p.uids().real,command=p.cmdline())
def exact_actor(a):
 p=psutil.Process(a['pid']);actual=process_identity(p)
 require(actual==dict(pid=a['pid'],birth=birth(a),UID=a['UID'],command=a['command'])and a['UID']==UID,'Exact actor PID/birth/UID/argv differs')
 require(p.status()not in(psutil.STATUS_ZOMBIE,psutil.STATUS_DEAD),'Exact actor is naturally terminal');return p
def released(a):
 try:p=psutil.Process(a['pid'])
 except psutil.NoSuchProcess:return True
 if p.create_time()!=birth(a):return True
 require(p.uids().real==a['UID']==UID,'Captured original actor became foreign')
 return p.status()in(psutil.STATUS_ZOMBIE,psutil.STATUS_DEAD)
def capture_children(p,known):
 for child in p.children(recursive=True):
  try:
   metadata=dict(pid=child.pid,birth=child.create_time(),UID=child.uids().real)
   require(metadata['UID']==UID,'Own DA/descendant UID required')
   key=(metadata['pid'],metadata['birth']);previous=known.get(key)
   def retain_known_terminal():
    require(previous is not None and all(previous[k]==metadata[k]for k in('pid','birth','UID')),'Terminal descendant lacks the exact previously captured identity')
   if child.status()in(psutil.STATUS_ZOMBIE,psutil.STATUS_DEAD):
    retain_known_terminal();continue # Keep the original admitted argv; never invent an empty one.
   command=child.cmdline()
   if not command and child.status()in(psutil.STATUS_ZOMBIE,psutil.STATUS_DEAD):
    retain_known_terminal();continue # Natural terminal transition between status and argv reads.
   require(command,'Own nonempty live DA/descendant identity required')
   a=dict(metadata,command=command)
   require(previous is None or previous==a,'Captured descendant identity changed')
   known[key]=a
  except psutil.NoSuchProcess:continue # Exact snapshot disappeared naturally; existing capture is retained.
def seal(path,value):
 path=P(path);temp=path.with_name('.'+path.name+'.exclusive_'+str(time.time_ns()))
 try:
  with temp.open('x')as f:json.dump(value,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
  os.link(temp,path,follow_symlinks=False);temp.unlink()
  fd=os.open(path.parent,os.O_RDONLY)
  try:os.fsync(fd)
  finally:os.close(fd)
 finally:
  if temp.exists():temp.unlink() # This call's exclusive unpublished temporary only.
def fold(native):
 return P(native['native_root'])/'nnUNet_results'/native['baseline']['dataset_name']/(native['trainer']+'__'+native['baseline']['plans_name']+'__3d_fullres')/'fold_0'
def training_window(root,goal):
 text=exact_file(root/'train.log').read_text(errors='replace')
 matches=list(re.finditer(r'(?m)^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:\.\d+)?):\s*Epoch (\d+)\s*$',text))
 if not matches:return None
 header=matches[-1];epoch=int(header.group(2));tail=text[header.end():]
 age=time.time()-datetime.datetime.fromisoformat(header.group(1)).timestamp()
 if not(0<=age<=10)or'Current learning rate:'not in tail:return None
 if re.search(r'\[OnlineCP\]\s+epoch='+str(epoch)+r'\b',tail)or any(t in tail for t in('train_loss ','val_loss ','Epoch time:','New best','checkpoint')):return None
 status=read(exact_file(root/'status.json'))
 if status.get('completed_epochs')!=epoch or status.get('stage')!='train':return None
 if list(goal.glob('whole_epoch_*.pth')):return None
 latest=exact_file(goal/'checkpoint_latest.pth');best=exact_file(goal/'checkpoint_best.pth')
 return dict(epoch=epoch,epoch_header=header.group(0),header_age_seconds=age,status_completed_epochs=epoch,
             latest_inode=list(inode(latest.stat())),BEST_inode=list(inode(best.stat())),log_prefix_sha256=hashlib.sha256(text.encode()).hexdigest(),no_current_epoch_CP_or_loss=True,no_whole_epoch_save_temporary=True,observed_unix=time.time())
def checkpoint_info(path,native,parameters):
 import torch
 with exact_file(path).open('rb')as f:
  before=os.fstat(f.fileno());value=torch.load(f,map_location='cpu',weights_only=False)
  try:
   epoch=value.get('current_epoch');logs=value.get('logging',{});opt=value.get('optimizer_state',{});scaler=value.get('grad_scaler_state',{});network=value.get('network_weights',{})
   require(type(epoch)is int and 0<epoch<=250 and set(logs)==LOGGERS and all(len(v)==epoch for v in logs.values()),'Whole eight native logger arrays must match the completed epoch')
   require(all(math.isfinite(float(a))and math.isfinite(float(b))and b>=a for a,b in zip(logs['epoch_start_timestamps'],logs['epoch_end_timestamps'])),'Finite ordered completed-epoch timestamps required')
   require(value.get('trainer_name')==native['trainer']and value.get('init_args')and '_best_ema'in value and 'inference_allowed_mirroring_axes'in value,'Full original native trainer/init/inference state required')
   require(network and all(torch.is_tensor(v)for v in network.values())and sum(v.numel()for v in network.values())>=parameters,'Full original native model state required')
   require(opt.get('state')and opt.get('param_groups')and {'scale','growth_factor','backoff_factor','growth_interval','_growth_tracker'}<=set(scaler),'Full original optimizer/scaler state required')
   f.seek(0);h=hashlib.sha256()
   for b in iter(lambda:f.read(8*2**20),b''):h.update(b)
   require(inode(before)==inode(os.fstat(f.fileno())),'Held checkpoint inode changed during decode/hash')
   return dict(path=str(path),completed_epoch=epoch,sha256=h.hexdigest(),logger_lengths={k:len(v)for k,v in logs.items()},optimizer_states=len(opt['state']),optimizer_param_groups=len(opt['param_groups']),scaler_present=True,network_tensor_elements=sum(v.numel()for v in network.values()),whole_completed_epoch=True,source_inode=list(inode(before)))
  finally:del value;gc.collect()

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--gpu',type=int,choices=tuple(EXPECTED_TRAINERS),required=True);parser.add_argument('--execute',action='store_true');parser.add_argument('--inspect',action='store_true');parser.add_argument('--manifest-sha256');parser.add_argument('--output-proof',type=P);parser.add_argument('--boundary-timeout',type=float,default=240)
 args=parser.parse_args();require(os.getuid()==UID and socket.gethostname().split('.')[0]=='ece-a6gpu6','Own exact host/UID required')
 require(not(args.execute and args.inspect)and math.isfinite(args.boundary_timeout)and args.boundary_timeout>0,'Choose one mode and a positive finite boundary observation timeout')
 manifest_path=exact_file(GROUP/'launch.json');manifest=read(manifest_path);manifest_sha=sha(manifest_path);row=manifest['registration'][str(args.gpu)];root=exact_file(row['root'],True)
 request=read(exact_file(root/'request.json'));native=read(exact_file(root/'native.json'));trainer=read(exact_file(root/'train_actor.json'));controller=row['controller_identity']
 require(sha(root/'request.json')==row['request_sha256']and request['controller_identity']==controller and request['gpu']==native['gpu']==args.gpu and request['arm']==native['arm']==ARMS[args.gpu],'Exact registered original native6 contract required')
 require((trainer['pid'],birth(trainer))==EXPECTED_TRAINERS[args.gpu] and trainer['UID']==UID and trainer['command'][4:]==['--native',str(root/'native.json'),'--stage','train'],'Only the reported exact original train child may be signaled')
 parent=exact_actor(controller);child=exact_actor(trainer);require(child.ppid()==parent.pid,'Exact native6/controller lineage differs')
 model=read(exact_file(root/'actual_model.json'));require(model['epochs']==250 and model['physical_batch']==model['effective_batch']==2 and len(model['actual_train_cases'])==105 and len(model['actual_validation_cases'])==26,'Original full250/B2/105/26 model contract required')
 known={};capture_children(child,known);goal=fold(native);state=read(exact_file(root/'status.json'));baseline=state.get('completed_epochs')
 require(type(baseline)is int and 0<baseline<249,'Actual current completed native epoch required; naturally complete runs do not need this signal')
 inspection=dict(manifest_sha256=manifest_sha,trainer=trainer,controller=controller,baseline_completed_epochs=baseline,captured_DA=list(known.values()),captured_DA_count=len(known),signals_sent=False)
 if not args.execute:print(json.dumps(inspection),flush=True);return
 require(args.manifest_sha256==manifest_sha and args.output_proof is not None,'Parent-reviewed exact manifest SHA and fresh proof path required before signaling')
 output=args.output_proof.absolute();exact_file(output.parent,True)
 require(output.parent.parent==BASE and output.parent.name.startswith('Basic_CP_')and not output.exists(),'Fresh own G5 operational proof namespace required')
 intent=output.with_name(output.name+'.intent.json');require(not intent.exists(),'Unknown prior signal outcome must be inspected; never replay')
 deadline=time.monotonic()+args.boundary_timeout;window=None
 while time.monotonic()<deadline:
  exact_actor(controller);capture_children(exact_actor(trainer),known);observed=training_window(root,goal)
  if observed is not None and observed['epoch']>=baseline+1:window=observed;break
  time.sleep(.25)
 require(window is not None,'No completed next epoch plus safe early training window; no signal sent')
 capture_children(exact_actor(trainer),known)
 seal(intent,dict(inspection,captured_DA=list(known.values()),selected_training_window=window,planned_signal='SIGINT',only_single_own_trainer_signaled=True,source_results_overwritten=False))
 exact_actor(controller);p=exact_actor(trainer);fresh=training_window(root,goal)
 require(fresh is not None and fresh['epoch']==window['epoch']and fresh['latest_inode']==window['latest_inode']and fresh['BEST_inode']==window['BEST_inode']and p.ppid()==controller['pid'],'Safe early training window changed before signal; no signal sent')
 print(json.dumps(dict(about_to_signal='SIGINT',pid=trainer['pid'],birth=birth(trainer),UID=UID,command=trainer['command'],after_completed_epoch=fresh['epoch'],no_parent_session_or_DA_signal=True)),flush=True)
 p.send_signal(signal.SIGINT) # Exactly this task's individually verified native child only.
 signal_receipt=intent.with_name(intent.name+'.single_signal.json');seal(signal_receipt,dict(trainer=trainer,signal='SIGINT',sent_unix=time.time(),selected_training_window=fresh,process_groups_signaled=False,signals_may_not_be_repeated=True))
 deadline=time.monotonic()+120
 while time.monotonic()<deadline:
  if not released(trainer):
   try:
    terminating=psutil.Process(trainer['pid'])
    if terminating.create_time()==birth(trainer):
     require(terminating.uids().real==UID,'Terminating native trainer UID changed')
     capture_children(terminating,known)
   except psutil.NoSuchProcess:pass # Exact already-captured native actor naturally vanished.
  live=[a for a in[trainer,controller,*known.values()]if not released(a)]
  if not live:break
  time.sleep(.25)
 require(not live,'Exact native/controller/DA still naturally releasing; do not repeat any signal')
 import torch
 torch.set_num_threads(1);latest=checkpoint_info(goal/'checkpoint_latest.pth',native,model['parameters']);best=checkpoint_info(goal/'checkpoint_best.pth',native,model['parameters'])
 require(latest['completed_epoch']>=window['epoch']>=baseline+1 and best['completed_epoch']<=latest['completed_epoch'],'Whole saved native progress or BEST epoch differs')
 log=root/'train.log';log_sha=sha(log)
 proof=dict(format='basic_cp_exact_RAM512_native_next_epoch_boundary_release_v1',manifest_sha256=manifest_sha,request_sha256=row['request_sha256'],trainer=trainer,controller=controller,captured_DA=list(known.values()),captured_DA_count=len(known),all_captured_actors_released=True,baseline_completed_epochs=baseline,signal_after_completed_epoch=window['epoch'],latest=latest,BEST=best,train_log=dict(path=str(log),sha256=log_sha,selected_window=window),single_signal_receipt=str(signal_receipt),temporary_migration_stop=True,final_training_goal_epochs=250,full_training_completed=False,original_scientific_contract_unchanged=True,original_results_overwritten=False,created_unix=time.time())
 seal(output,proof);print(json.dumps(proof),flush=True)

if __name__=='__main__':main()
