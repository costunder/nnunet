"""Prepare/register the committed additive GPU5 early-stop automatic chain."""
import argparse,ast,base64,hashlib,json,subprocess,time
from pathlib import Path
from remote_once import run,HERE,ROOT

REMOTE=r'''
import base64,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
import psutil
P=PAYLOAD
base=Path('/var/tmp/aicompetition06_v24_20261010')
release=base/'GPU5_EARLY_STOP_E27_20261012_1791740800000000000'
anchor=Path('/dev/shm/aicompetition06_v24_gpu6_DEBUG_20261011/code_bda74f1e1be6')
code=anchor.parent/('code_'+P['commit'][:12])
native=base/'GPU456_AUTO_NATIVE_20261011_1791730500000000000/gpu5'
score_waiter=base/'GPU5_FULL642_depth1_scoring_20261012_1791731200000000000'
assert os.getuid()==41833 and release.stat().st_uid==41833
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def publish(p,v):
 with p.open('x',encoding='utf8') as stream:json.dump(v,stream,ensure_ascii=False,indent=2);stream.flush();os.fsync(stream.fileno())
def proof(p):
 p=Path(p);s=p.stat()
 assert s.st_uid==41833 and not p.is_symlink()
 return dict(path=str(p),sha256=sha(p),stat=[s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns])
bundle=release/('early_stop_'+P['commit'][:12]+'.bundle');raw=base64.b64decode(P['bundle'],validate=True)
assert hashlib.sha256(raw).hexdigest()==P['bundle_sha256']
if bundle.exists():assert bundle.read_bytes()==raw
else:
 with bundle.open('xb') as stream:stream.write(raw)
if not code.exists():
 assert shutil.disk_usage(anchor.parent).free>100*2**20
 subprocess.run(['git','clone','--shared','--no-checkout',str(anchor),str(code)],check=True,capture_output=True,text=True,timeout=30)
 subprocess.run(['git','-C',str(code),'fetch',str(bundle),'HEAD'],check=True,capture_output=True,text=True,timeout=30)
 subprocess.run(['git','-C',str(code),'checkout','--detach','FETCH_HEAD'],check=True,capture_output=True,text=True,timeout=30)
assert subprocess.check_output(['git','-C',str(code),'rev-parse','HEAD'],text=True).strip()==P['commit']
assert all(sha(code/name)==digest for name,digest in P['files'].items())
test_log=release/('Linux_CPU_tests_'+P['commit'][:12]+'.log')
if not test_log.exists():
 result=subprocess.run(['/home/aicompetition06/.conda/envs/nnunet/bin/python3.10','-B','-m','unittest',
  'tests.test_v24_early_stop_gnn','tests.test_v24_early_stop_downstream','tests.test_v24_early_stop_viewer','-q'],cwd=code,
  capture_output=True,text=True,timeout=180)
 with test_log.open('x') as stream:stream.write(result.stdout+'\n'+result.stderr)
 if result.returncode:raise RuntimeError('Fresh immutable Linux CPU tests failed: '+test_log.read_text())
else:assert '\nOK\n' in test_log.read_text()
sys.path.insert(0,str(code))
from hiercp_v1x import v24_early_stop_downstream as early
from hiercp_v1x import v24_scored_native_chain as chain
controller=release/'controller';score=release/'scoring'
assert not controller.exists() and not score.exists(),'Unknown previous outcome; preserve and inspect existing fresh roots'
actors=[]
for root,prefix in ((native,''),(score_waiter,'scoring_wait_')):
 reg=json.loads((root/(prefix+'registration.json')).read_text())
 p=psutil.Process(reg['pid'])
 command=reg['command'];birth=reg['create_time']
 assert p.create_time()==birth and p.cmdline()==command and p.uids().real==41833
 state=json.loads((root/(prefix+'status.json')).read_text())
 assert state.get('child_pid') is None and state['status'] in ('WAITING_FOR_FULL_GNN','WAITING_FOR_FULL_GNN_BEST','WAITING_FOR_NATIVE_STORAGE','WAITING_FOR_FULL642_SCORING')
 actors.append(dict(pid=p.pid,create_time=birth,command=command,UID=41833))
auth=json.loads((release/'authorization.json').read_text())
actors.append(dict(**auth['original_actor'],UID=41833))
protected=[]
for root in (native,score_waiter):
 for name in ('request.json','registration.json','lease.json','scoring_wait_request.json','scoring_wait_registration.json'):
  path=root/name
  if path.exists():protected.append(proof(path))
native_request=json.loads((native/'request.json').read_text())
protected.append(proof(native_request['native_claim_path']))
request=early.build_request(native/'request.json',controller_root=controller,scoring_root=score,
 code=code,commit=P['commit'],code_files=P['files'],target_epoch=27,
 authorization_path=release/'authorization.json',release_actors=actors,protected_files=protected)
cache=chain.FileProofCache();early.verify_deployment(request,cache)
storage=chain.storage_ready(request['resource_contract'],cache)
controller.mkdir(mode=0o700);score.mkdir(mode=0o700)
claim=early.claim_path(request);claim.parent.mkdir(mode=0o700,exist_ok=True)
publish(controller/'request.json',request)
command=[request['resource_contract']['python'],'-B','-u',str(code/'tools/run_v24_early_stop_downstream.py'),
 'watch','--request',str(controller/'request.json')]
publish(controller/'launch_intent.json',dict(command=command,request_sha256=sha(controller/'request.json'),code=str(code),commit=P['commit'],at=time.time()))
with (controller/'console.log').open('xb') as log:
 child=subprocess.Popen(command,cwd=code,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
  env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'),start_new_session=True)
 p=psutil.Process(child.pid);p.cpu_affinity([42,43,44,45])
 publish(controller/'registration.json',dict(pid=p.pid,create_time=p.create_time(),command=command,UID=41833,
  request_sha256=sha(controller/'request.json'),code=str(code),commit=P['commit'],CPU_affinity=p.cpu_affinity(),signals_sent=False))
 time.sleep(.7)
 state=json.loads((controller/'status.json').read_text())
 assert child.poll() is None and state['status']=='WAITING_FOR_EARLY_STOP_RELEASE' and state['child_pid'] is None
 result=dict(status='GPU5_EARLY_STOP_CHAIN_REGISTERED_WAITING_OWN_EPOCH27_RELEASE',at=time.time(),commit=P['commit'],code=str(code),code_files=P['files'],
  controller_root=str(controller),scoring_root=str(score),request=proof(controller/'request.json'),registration=proof(controller/'registration.json'),
  state=state,storage=storage,protected_files=protected,old_waiter_release_actors=actors[:-1],
  Linux_CPU_tests=test_log.read_text(),full_GNN_training=False,full_native_training=False,full_native_evaluation=False,signals_sent=False)
 publish(release/'chain_registration_verified.json',result)
 print(json.dumps(result,ensure_ascii=False),flush=True)
'''

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--commit',required=True);parser.add_argument('--bundle',type=Path,required=True)
 parser.add_argument('--checkout',type=Path,default=ROOT)
 args=parser.parse_args()
 export_root=args.checkout.resolve(strict=True)
 names=set()
 for source,constant in [('hiercp_v1x/v24_scored_native_chain.py','FILES'),('hiercp_v1x/v24_matched_basic_cp_pipeline.py','FILES'),
  ('tools/run_v24_completed_matched_scoring.py','REQUIRED_CODE_FILES'),('hiercp_v1x/v24_early_stop_downstream.py','FILES')]:
  tree=ast.parse((export_root/source).read_text(encoding='utf8'))
  names.update(next(ast.literal_eval(node.value) for node in tree.body if isinstance(node,ast.Assign) and any(isinstance(target,ast.Name) and target.id==constant for target in node.targets)))
 names.update(['tools/watch_v24_early_stop_pipeline.py','tools/watch_v24_gpu_pipeline.py','tools/watch_v24_direct_gnn_training.py',
  'tools/watch_v24_matched_cp_scoring.py','tools/watch_v24_training.py','tools/training_monitor_adapter.py','tools/training_monitor_ui.py'])
 files={}
 for name in sorted(names):
  raw=subprocess.check_output(['git','-c','safe.directory='+str(export_root),'show',args.commit+':'+name],cwd=export_root)
  assert raw==(export_root/name).read_bytes(),name
  files[name]=hashlib.sha256(raw).hexdigest()
 raw=args.bundle.read_bytes();payload=dict(commit=args.commit,files=files,bundle=base64.b64encode(raw).decode(),bundle_sha256=hashlib.sha256(raw).hexdigest())
 script='PAYLOAD='+repr(payload)+'\n'+REMOTE
 prepared=HERE/('register_GPU5_early_stop_'+args.commit[:12]+'_PREPARED.py')
 if prepared.exists():assert prepared.read_text(encoding='utf8')==script
 else:prepared.write_text(script,encoding='utf8')
 output=run(script,240);data=json.loads(output)
 receipt=HERE/('GPU5_EARLY_STOP_CHAIN_REGISTERED_'+str(time.time_ns())+'.json')
 receipt.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf8')
 print(receipt);print(json.dumps({k:data[k] for k in ('status','commit','code','controller_root','scoring_root','state','Linux_CPU_tests')},ensure_ascii=False))

if __name__=='__main__':main()
