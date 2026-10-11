"""Deploy and launch only an admitted zero-update GPU6 startup retry."""
import argparse,base64,hashlib,importlib.util,json,subprocess,time
from pathlib import Path

FILES=('hiercp_v1x/v24_native_startup_retry.py','tools/run_v24_native_startup_retry.py',
 'hiercp_v1x/v24_matched_basic_cp_runtime.py','tools/watch_v24_native_startup_retry.py',
 'tools/watch_v24_gpu_pipeline.py','tools/watch_v24_direct_gnn_training.py',
 'tools/watch_v24_matched_cp_scoring.py','tools/watch_v24_training.py',
 'tools/training_monitor_adapter.py','tools/training_monitor_ui.py',
 'tests/test_v24_native_startup_retry.py','tests/test_v24_native_startup_retry_viewer.py',
 'tests/test_v24_matched_basic_cp_runtime.py')

REMOTE=r'''
import base64,hashlib,json,os,subprocess,time
from pathlib import Path
import psutil
P=PAYLOAD
assert os.getuid()==41833
base=Path('/var/tmp/aicompetition06_v24_20261010')
oldroot=base/'GPU456_AUTO_NATIVE_20261011_1791730500000000000/gpu6'
recovery=oldroot/('NATIVE_STARTUP_RETRY_'+P['commit'][:12])
anchor=Path('/dev/shm/aicompetition06_v24_gpu6_DEBUG_20261011/code_a350de1794d2')
code=anchor.parent/('code_'+P['commit'][:12])
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def create(path,raw):
 with Path(path).open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
raw=base64.b64decode(P['bundle'],validate=True)
assert hashlib.sha256(raw).hexdigest()==P['bundle_sha256']
bundle=base/('GPU6_startup_retry_'+P['commit'][:12]+'.bundle')
if bundle.exists():assert bundle.read_bytes()==raw
else:create(bundle,raw)
if not code.exists():
 subprocess.run(['git','clone','--shared','--no-checkout',str(anchor),str(code)],capture_output=True,text=True,check=True,timeout=30)
 subprocess.run(['git','-C',str(code),'fetch',str(bundle),'HEAD'],capture_output=True,text=True,check=True,timeout=30)
 subprocess.run(['git','-C',str(code),'checkout','--detach','FETCH_HEAD'],capture_output=True,text=True,check=True,timeout=30)
assert subprocess.check_output(['git','-C',str(code),'rev-parse','HEAD'],text=True).strip()==P['commit']
assert all(sha(code/name)==digest for name,digest in P['files'].items())
testlog=base/('GPU6_startup_retry_CPU_'+P['commit'][:12]+'.log')
if not testlog.exists():
 tested=subprocess.run([P['python'],'-B','-m','unittest','tests.test_v24_matched_basic_cp_runtime',
  'tests.test_v24_native_startup_retry','tests.test_v24_native_startup_retry_viewer','-q'],cwd=code,capture_output=True,text=True,timeout=120)
 create(testlog,(tested.stdout+'\n'+tested.stderr).encode())
 assert tested.returncode==0,tested.stdout+tested.stderr
else:assert '\nOK\n' in testlog.read_text()
assert not recovery.exists(),'Unknown prior retry outcome; inspect exact lease before any second launch'
import sys
sys.path.insert(0,str(code))
from hiercp_v1x import v24_native_startup_retry as retry
request=retry.build_request(oldroot/'request.json',recovery,code,P['commit'],P['files'],inspection=P['inspection'])
path=retry.seal_request(request)
assert Path(path)==recovery/'request.json'
command=[P['python'],'-B','-u',str(code/'tools/run_v24_native_startup_retry.py'),'watch','--request',str(path)]
create(recovery/'launch_intent.json',(json.dumps(dict(command=command,request_sha256=sha(path),commit=P['commit'],at=time.time()),indent=2)+'\n').encode())
affinity=request['original_declaration']['CPU_affinity']
with (recovery/'controller_console.log').open('xb') as stream:
 child=subprocess.Popen(command,cwd=code,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,
  env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'),start_new_session=True,
  preexec_fn=lambda:os.sched_setaffinity(0,set(affinity)))
 actor=psutil.Process(child.pid)
 assert actor.uids().real==41833 and actor.cmdline()==command and actor.cpu_affinity()==affinity
 reg=dict(pid=actor.pid,create_time=actor.create_time(),command=command,UID=41833,request_sha256=sha(path),
  root=str(recovery),GPU_arm=6,CPU_affinity=affinity,commit=P['commit'])
 create(recovery/'registration.json',(json.dumps(reg,indent=2)+'\n').encode())
 deadline=time.monotonic()+30
 while not (recovery/'status.json').exists() and child.poll() is None and time.monotonic()<deadline:
  time.sleep(.25)
 if child.poll() is not None:raise RuntimeError('Retry controller failed; original logs preserved: '+(recovery/'controller_console.log').read_text())
 status=json.loads((recovery/'status.json').read_text()) if (recovery/'status.json').exists() else dict(status='CPU_ADMISSION_PENDING',child_pid=None)
 result=dict(status='GPU6_NATIVE_STARTUP_RETRY_REGISTERED',at=time.time(),code=str(code),commit=P['commit'],
  recovery_root=str(recovery),request=str(path),request_sha256=sha(path),registration=reg,state=status,
  Linux_CPU_tests=testlog.read_text(),code_files=P['files'],jobs_restarted_from_checkpoint=False,signals_sent=False,
  original_CP_642_scores_reused=True,full_training_complete=False,full_BEST26_evaluation_complete=False)
 create(recovery/'registration_verified.json',(json.dumps(result,ensure_ascii=False,indent=2)+'\n').encode())
 print(json.dumps(result,ensure_ascii=False),flush=True)
'''


def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--checkout',type=Path,required=True)
 parser.add_argument('--ssh-helper-dir',type=Path,required=True)
 parser.add_argument('--bundle',type=Path,required=True)
 parser.add_argument('--inspection',type=Path,required=True)
 args=parser.parse_args()
 checkout=args.checkout.resolve(strict=True);helper=args.ssh_helper_dir.resolve(strict=True)
 git=['git','-c','safe.directory='+checkout.as_posix(),'-C',str(checkout)]
 commit=subprocess.check_output(git+['rev-parse','HEAD'],text=True).strip()
 files={}
 for name in FILES:
  raw=subprocess.check_output(git+['show',commit+':'+name]);assert raw==(checkout/name).read_bytes(),name
  files[name]=hashlib.sha256(raw).hexdigest()
 bundle=args.bundle.read_bytes()
 payload=dict(commit=commit,files=files,bundle=base64.b64encode(bundle).decode(),bundle_sha256=hashlib.sha256(bundle).hexdigest(),
  python='/home/aicompetition06/.conda/envs/nnunet/bin/python3.10',inspection=json.loads(args.inspection.read_text(encoding='utf8')))
 script='PAYLOAD='+repr(payload)+'\n'+REMOTE
 prepared=helper/('deploy_GPU6_native_retry_'+commit[:12]+'_PREPARED.py')
 if prepared.exists():assert prepared.read_text(encoding='utf8')==script
 else:prepared.write_text(script,encoding='utf8')
 spec=importlib.util.spec_from_file_location('remote_once',helper/'remote_once.py');remote=importlib.util.module_from_spec(spec);spec.loader.exec_module(remote)
 result=json.loads(remote.run(script,180))
 receipt=helper/('GPU6_NATIVE_STARTUP_RETRY_REGISTERED_'+str(time.time_ns())+'.json')
 receipt.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
 print(receipt);print(json.dumps({k:result[k] for k in ('status','code','commit','recovery_root','request','registration','state','Linux_CPU_tests')},ensure_ascii=False))


if __name__=='__main__':main()
