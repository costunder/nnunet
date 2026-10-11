"""Deploy fixed native code and register explicit existing-CP adoption guards."""
import argparse,ast,base64,hashlib,importlib.util,json,subprocess,time
from pathlib import Path

REMOTE=r'''
import base64,hashlib,json,os,subprocess,sys,time
from pathlib import Path
import psutil
P=PAYLOAD
assert os.getuid()==41833
base=Path('/var/tmp/aicompetition06_v24_20261010')
anchor=Path('/dev/shm/aicompetition06_v24_gpu6_DEBUG_20261011/code_dea813d273fe')
code=anchor.parent/('code_'+P['commit'][:12])
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def create(p,raw):
 with Path(p).open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
raw=base64.b64decode(P['bundle'],validate=True)
assert hashlib.sha256(raw).hexdigest()==P['bundle_sha256']
bundle=base/('GPU45_native_prestart_'+P['commit'][:12]+'.bundle')
if bundle.exists():assert bundle.read_bytes()==raw
else:create(bundle,raw)
if not code.exists():
 subprocess.run(['git','clone','--shared','--no-checkout',str(anchor),str(code)],capture_output=True,text=True,check=True,timeout=30)
 subprocess.run(['git','-C',str(code),'fetch',str(bundle),'HEAD'],capture_output=True,text=True,check=True,timeout=30)
 subprocess.run(['git','-C',str(code),'checkout','--detach','FETCH_HEAD'],capture_output=True,text=True,check=True,timeout=30)
assert subprocess.check_output(['git','-C',str(code),'rev-parse','HEAD'],text=True).strip()==P['commit']
assert all(sha(code/name)==digest for name,digest in P['files'].items())
testlog=base/('GPU45_native_prestart_CPU_'+P['commit'][:12]+'.log')
if not testlog.exists():
 tested=subprocess.run([P['python'],'-B','-m','unittest',*P['tests'],'-q'],cwd=code,capture_output=True,text=True,timeout=150)
 create(testlog,(tested.stdout+'\n'+tested.stderr).encode())
 assert tested.returncode==0,testlog.read_text()
else:assert '\nOK\n' in testlog.read_text()
sys.path.insert(0,str(code))
from hiercp_v1x import v24_native_prestart_guard as guard
results=[]
for arm in P['inspection']['arms']:
 gpu=arm['GPU'];original=Path(arm['root'])/'request.json'
 assert gpu in (4,5) and json.loads(original.read_text())==arm['request']
 resource=arm['request'].get('resource_contract',arm['request'])
 assert not (Path(resource['chain_root'])/'native').exists(),'Existing native preparation requires separate explicit recovery'
 recovery=(original.parent if gpu==4 else original.parent.parent)/('NATIVE_PRESTART_'+P['commit'][:12])
 assert not recovery.exists(),'Unknown prior guard outcome; inspect existing exact lease rather than relaunch'
 declaration=guard.build_request(original,recovery,code,P['commit'],P['files'],arm)
 path=guard.seal_request(declaration)
 assert Path(path)==recovery/'request.json'
 command=[P['python'],'-B','-u',str(code/'tools/run_v24_native_prestart_guard.py'),'watch','--request',str(path)]
 create(recovery/'launch_intent.json',(json.dumps(dict(command=command,request_sha256=sha(path),commit=P['commit'],at=time.time()),indent=2)+'\n').encode())
 affinity=resource['CPU_affinity']
 with (recovery/'console.log').open('xb') as log:
  child=subprocess.Popen(command,cwd=code,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
   env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'),
   start_new_session=True,preexec_fn=lambda:os.sched_setaffinity(0,set(affinity)))
  p=psutil.Process(child.pid)
  assert p.uids().real==41833 and p.cmdline()==command and p.cpu_affinity()==affinity
  reg=dict(pid=p.pid,create_time=p.create_time(),command=command,UID=41833,request_sha256=sha(path),root=str(recovery),GPU_arm=gpu,
   CPU_affinity=affinity,code=str(code),commit=P['commit'])
  create(recovery/'registration.json',(json.dumps(reg,indent=2)+'\n').encode())
  deadline=time.monotonic()+30
  while not (recovery/'status.json').exists() and child.poll() is None and time.monotonic()<deadline:time.sleep(.2)
  assert child.poll() is None,(recovery/'console.log').read_text()
  state=json.loads((recovery/'status.json').read_text()) if (recovery/'status.json').exists() else dict(status='CPU_ADMISSION_PENDING',child_pid=None)
  assert state['status'] in ('CPU_ADMISSION_PENDING','WAITING_FOR_CONTROLLER_HANDOFF','WAITING_FOR_EXISTING_SCORING'),state
  results.append(dict(GPU=gpu,recovery_root=str(recovery),request=str(path),request_sha256=sha(path),registration=reg,state=state,
   predecessor=declaration['predecessor_actor'],scoring_actor=declaration['scoring_actor']))
result=dict(status='GPU45_FIXED_NATIVE_PRESTART_REGISTERED',at=time.time(),code=str(code),commit=P['commit'],code_files=P['files'],arms=results,
 Linux_CPU_tests=testlog.read_text(),scorers_restarted=False,signals_sent=False,full_native_training_complete=False,full_BEST26_evaluation_complete=False)
print(json.dumps(result,ensure_ascii=False),flush=True)
'''

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--checkout',type=Path,required=True);parser.add_argument('--ssh-helper-dir',type=Path,required=True)
 parser.add_argument('--bundle',type=Path,required=True);parser.add_argument('--inspection',type=Path,required=True)
 args=parser.parse_args();checkout=args.checkout.resolve(strict=True);helper=args.ssh_helper_dir.resolve(strict=True)
 git=['git','-c','safe.directory='+checkout.as_posix(),'-C',str(checkout)]
 commit=subprocess.check_output(git+['rev-parse','HEAD'],text=True).strip()
 names=set()
 for filename,constant in [('hiercp_v1x/v24_scored_native_chain.py','FILES'),('hiercp_v1x/v24_matched_basic_cp_pipeline.py','FILES'),
  ('tools/run_v24_completed_matched_scoring.py','REQUIRED_CODE_FILES'),('hiercp_v1x/v24_early_stop_downstream.py','FILES'),
  ('hiercp_v1x/v24_native_prestart_guard.py','FILES')]:
  tree=ast.parse((checkout/filename).read_text(encoding='utf8'))
  names.update(next(ast.literal_eval(node.value) for node in tree.body if isinstance(node,ast.Assign) and any(isinstance(target,ast.Name) and target.id==constant for target in node.targets)))
 tests=['tests.test_v24_matched_basic_cp_runtime','tests.test_v24_native_prestart_guard','tests.test_v24_native_prestart_training','tests.test_v24_native_prestart_viewer','tests.test_v24_early_stop_downstream']
 names.update(name.replace('.','/')+'.py' for name in tests)
 names.add('tests/test_v24_scored_native_chain.py')
 names.update(['tools/watch_v24_native_prestart_guard.py','tools/watch_v24_early_stop_pipeline.py','tools/watch_v24_gpu_pipeline.py',
  'tools/watch_v24_direct_gnn_training.py','tools/watch_v24_matched_cp_scoring.py','tools/watch_v24_training.py',
  'tools/training_monitor_adapter.py','tools/training_monitor_ui.py'])
 files={}
 for name in sorted(names):
  raw=subprocess.check_output(git+['show',commit+':'+name]);assert raw==(checkout/name).read_bytes(),name
  files[name]=hashlib.sha256(raw).hexdigest()
 bundle=args.bundle.read_bytes()
 payload=dict(commit=commit,files=files,bundle=base64.b64encode(bundle).decode(),bundle_sha256=hashlib.sha256(bundle).hexdigest(),
  python='/home/aicompetition06/.conda/envs/nnunet/bin/python3.10',tests=tests,inspection=json.loads(args.inspection.read_text(encoding='utf8')))
 script='PAYLOAD='+repr(payload)+'\n'+REMOTE
 prepared=helper/('deploy_GPU45_native_prestart_'+commit[:12]+'_PREPARED.py')
 if prepared.exists():assert prepared.read_text(encoding='utf8')==script
 else:prepared.write_text(script,encoding='utf8')
 spec=importlib.util.spec_from_file_location('remote_once',helper/'remote_once.py');remote=importlib.util.module_from_spec(spec);spec.loader.exec_module(remote)
 result=json.loads(remote.run(script,240))
 receipt=helper/('GPU45_NATIVE_PRESTART_REGISTERED_'+str(time.time_ns())+'.json')
 receipt.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
 print(receipt);print(json.dumps({k:result[k] for k in ('status','code','commit','arms','Linux_CPU_tests')},ensure_ascii=False))
if __name__=='__main__':main()
