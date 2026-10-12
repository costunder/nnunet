"""Deploy fixed code or finalize existing GPU6 full250/BEST26 artifacts on CPU.

Only new code/completion namespaces are created. Existing results and failed
records stay sealed. This helper never starts training or sends process signals.
"""
import argparse, base64, hashlib, importlib.util, json, subprocess, time
from pathlib import Path

FILES = ('hiercp_v1x/v24_native_startup_retry.py', 'tools/run_v24_native_startup_retry.py',
 'tools/watch_v24_native_startup_retry.py', 'tools/watch_v24_gpu_pipeline.py',
 'tools/watch_v24_direct_gnn_training.py', 'tools/watch_v24_matched_cp_scoring.py',
 'tools/watch_v24_training.py', 'tools/training_monitor_adapter.py', 'tools/training_monitor_ui.py',
 'hiercp_v1x/v24_matched_basic_cp_runtime.py', 'tests/test_v24_native_startup_retry.py',
 'tests/test_v24_native_startup_retry_viewer.py', 'tests/test_v24_matched_basic_cp_runtime.py',
 'tests/test_v24_native_completion_display.py')

DEPLOY = r'''
import base64,hashlib,json,os,subprocess,time
from pathlib import Path
P=PAYLOAD
assert os.getuid()==41833
base=Path('/var/tmp/aicompetition06_v24_20261010')
anchor=Path('/dev/shm/aicompetition06_v24_gpu6_DEBUG_20261011/code_ec81af9c0f8b')
code=anchor.parent/('code_'+P['commit'][:12])
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def create(path,raw):
 with Path(path).open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
assert subprocess.check_output(['git','-C',str(anchor),'rev-parse','HEAD'],text=True).strip()==P['anchor_commit']
raw=base64.b64decode(P['bundle'],validate=True)
assert hashlib.sha256(raw).hexdigest()==P['bundle_sha256']
bundle=base/('GPU6_existing_completion_'+P['commit'][:12]+'.bundle')
if bundle.exists():assert bundle.read_bytes()==raw
else:create(bundle,raw)
if not code.exists():
 subprocess.run(['git','clone','--shared','--no-checkout',str(anchor),str(code)],capture_output=True,text=True,check=True,timeout=30)
assert not code.is_symlink() and code.stat().st_uid==41833
alternate=code/'.git/objects/info/alternates'
original=alternate.read_bytes();visited=set();stores=[]
def collect(store):
 store=Path(store).resolve(strict=True)
 if store in visited:return
 visited.add(store);nested=store/'info/alternates'
 if nested.exists():
  for line in nested.read_text().splitlines():
   item=Path(line);collect(item if item.is_absolute() else store/item)
 stores.append(str(store))
for line in original.decode().splitlines():
 item=Path(line);collect(item if item.is_absolute() else alternate.parent.parent/item)
flat=('\n'.join(stores)+'\n').encode()
if flat!=original:
 backup=alternate.with_name('alternates.pre_completion_flatten')
 if backup.exists():assert backup.read_bytes()==original
 else:create(backup,original)
 temporary=alternate.with_name('alternates.completion_flatten.tmp');create(temporary,flat);os.replace(temporary,alternate)
current=subprocess.check_output(['git','-C',str(code),'rev-parse','HEAD'],text=True).strip()
if current!=P['commit']:
 assert current==P['anchor_commit'] and [p.name for p in code.iterdir()]==['.git']
 subprocess.run(['git','-C',str(code),'fetch',str(bundle),'HEAD'],capture_output=True,text=True,check=True,timeout=30)
 subprocess.run(['git','-C',str(code),'checkout','--detach','FETCH_HEAD'],capture_output=True,text=True,check=True,timeout=30)
assert subprocess.check_output(['git','-C',str(code),'rev-parse','HEAD'],text=True).strip()==P['commit']
assert all(sha(code/name)==value for name,value in P['files'].items())
log=base/('GPU6_existing_completion_CPU_'+P['commit'][:12]+'.log')
if not log.exists():
 tested=subprocess.run([P['python'],'-B','-m','unittest','tests.test_v24_native_startup_retry',
  'tests.test_v24_native_startup_retry_viewer','tests.test_v24_matched_basic_cp_runtime',
  'tests.test_v24_native_completion_display','-q'],
  cwd=code,capture_output=True,text=True,timeout=150)
 create(log,(tested.stdout+'\n'+tested.stderr).encode())
 assert tested.returncode==0,log.read_text()
else:assert '\nOK\n' in log.read_text()
print(json.dumps(dict(status='GPU6_EXISTING_COMPLETION_CODE_VERIFIED',code=str(code),commit=P['commit'],
 code_files=P['files'],Linux_CPU_tests=log.read_text(),at=time.time(),jobs_started=False,signals_sent=False)))
'''

COMPLETE = r'''
import json,os,sys,time
from pathlib import Path
P=PAYLOAD
assert os.getuid()==41833
os.sched_setaffinity(0,{32,33,34,35})
code=Path('/dev/shm/aicompetition06_v24_gpu6_DEBUG_20261011')/('code_'+P['commit'][:12])
arm=Path('/var/tmp/aicompetition06_v24_20261010/GPU456_AUTO_NATIVE_20261011_1791730500000000000/gpu6')
root=arm/('POST_TRAINING_COMPLETION_'+P['commit'][:12])
assert not root.exists(),'Prior outcome requires read-only inspection; never replay completion'
sys.path.insert(0,str(code))
from hiercp_v1x import v24_native_startup_retry as retry
request=retry.build_completion_request(arm/'NATIVE_STARTUP_RETRY_dea813d273fe/request.json',
 root,code,P['commit'],P['files'])
path=retry.seal_completion_request(request)
state=retry.complete_existing_artifacts(path)
lease=retry.read(root/'lease.json')
registration={key:lease[key] for key in ('pid','create_time','command','UID','request_sha256')}
registration.update(root=str(root),GPU_arm=6,CPU_affinity=[32,33,34,35],commit=P['commit'])
retry.publish(root/'registration.json',registration)
assert state['status']=='COMPLETE' and state['completed_epochs']==250 and state['evaluated_cases']==26
print(json.dumps(dict(status='GPU6_EXISTING_FULL250_BEST26_COMPLETION_VERIFIED',code=str(code),commit=P['commit'],
 recovery_root=str(root),request=str(path),request_sha256=retry.digest(path),registration=registration,state=state,
 code_files=P['files'],at=time.time(),training_rerun=False,evaluation_rerun=False,CUDA_compute=False,signals_sent=False)))
'''

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('action',choices=('deploy','complete'))
 parser.add_argument('--checkout',type=Path,required=True)
 parser.add_argument('--ssh-helper-dir',type=Path,required=True)
 parser.add_argument('--bundle',type=Path)
 args=parser.parse_args();checkout=args.checkout.resolve(strict=True);helper=args.ssh_helper_dir.resolve(strict=True)
 git=['git','-c','safe.directory='+checkout.as_posix(),'-C',str(checkout)]
 commit=subprocess.check_output(git+['rev-parse','HEAD'],text=True).strip();files={}
 for name in FILES:
  raw=subprocess.check_output(git+['show',commit+':'+name]);assert raw==(checkout/name).read_bytes(),name
  files[name]=hashlib.sha256(raw).hexdigest()
 payload=dict(commit=commit,files=files,python='/home/aicompetition06/.conda/envs/nnunet/bin/python3.10')
 if args.action=='deploy':
  raw=args.bundle.read_bytes()
  payload.update(bundle=base64.b64encode(raw).decode(),bundle_sha256=hashlib.sha256(raw).hexdigest(),
   anchor_commit='ec81af9c0f8b47c3fadcc86a57f326ababa4773c')
 script='PAYLOAD='+repr(payload)+'\n'+(DEPLOY if args.action=='deploy' else COMPLETE)
 prepared=helper/('GPU6_existing_completion_'+args.action+'_'+commit[:12]+'_PREPARED.py')
 if prepared.exists():assert prepared.read_text(encoding='utf8')==script
 else:prepared.write_text(script,encoding='utf8')
 spec=importlib.util.spec_from_file_location('remote_once',helper/'remote_once.py')
 remote=importlib.util.module_from_spec(spec);spec.loader.exec_module(remote)
 result=json.loads(remote.run(script,240))
 receipt=helper/('GPU6_EXISTING_COMPLETION_'+args.action.upper()+'_'+str(time.time_ns())+'.json')
 receipt.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
 print(receipt);print(json.dumps({key:result[key] for key in ('status','code','commit')},ensure_ascii=False))
 if args.action=='deploy':print(result['Linux_CPU_tests'])
 else:print(json.dumps({key:result['state'][key] for key in ('status','completed_epochs','evaluated_cases','evaluation_checkpoint')},ensure_ascii=False))

if __name__=='__main__':main()
