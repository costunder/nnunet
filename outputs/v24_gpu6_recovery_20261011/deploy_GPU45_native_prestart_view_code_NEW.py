"""Install and test a separate viewer checkout without changing running code."""
import argparse,base64,hashlib,importlib.util,json,subprocess,time
from pathlib import Path

REMOTE=r'''
import base64,hashlib,json,os,subprocess,time
from pathlib import Path
P=PAYLOAD
assert os.getuid()==41833
anchor=Path(P['controller_code']);code=anchor.parent/('code_'+P['commit'][:12])
base=Path('/var/tmp/aicompetition06_v24_20261010')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def create(p,raw):
 with Path(p).open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
assert subprocess.check_output(['git','-C',str(anchor),'rev-parse','HEAD'],text=True).strip()==P['controller_commit']
raw=base64.b64decode(P['bundle'],validate=True)
assert hashlib.sha256(raw).hexdigest()==P['bundle_sha256']
bundle=base/('GPU45_native_prestart_view_'+P['commit'][:12]+'.bundle')
if bundle.exists():assert bundle.read_bytes()==raw
else:create(bundle,raw)
if not code.exists():
 subprocess.run(['git','clone','--shared','--no-checkout',str(anchor),str(code)],capture_output=True,text=True,check=True,timeout=30)
assert code.parent==anchor.parent and code.name=='code_'+P['commit'][:12] and not code.is_symlink() and code.stat().st_uid==41833
# Flatten only this new clone's alternate list, deepest stores first. Older
# immutable checkouts can form a longer chain than Git's alternate depth limit.
alternate=code/'.git/objects/info/alternates'
original=alternate.read_bytes();visited=set();stores=[]
def collect(store):
 store=Path(store).resolve(strict=True)
 if store in visited:return
 visited.add(store)
 nested=store/'info/alternates'
 if nested.exists():
  for line in nested.read_text().splitlines():
   item=Path(line);collect(item if item.is_absolute() else store/item)
 stores.append(str(store))
for line in original.decode().splitlines():
 item=Path(line);collect(item if item.is_absolute() else alternate.parent.parent/item)
flat=('\n'.join(stores)+'\n').encode()
if flat!=original:
 backup=alternate.with_name('alternates.pre_viewer_flatten')
 if backup.exists():assert backup.read_bytes()==original
 else:create(backup,original)
 temporary=alternate.with_name('alternates.viewer_flatten.tmp');create(temporary,flat);os.replace(temporary,alternate)
current=subprocess.check_output(['git','-C',str(code),'rev-parse','HEAD'],text=True).strip()
if current!=P['commit']:
 assert current==P['controller_commit'] and [p.name for p in code.iterdir()]==['.git'],'Only our own empty no-checkout clone may be completed'
 subprocess.run(['git','-C',str(code),'fetch',str(bundle),'HEAD'],capture_output=True,text=True,check=True,timeout=30)
 subprocess.run(['git','-C',str(code),'checkout','--detach','FETCH_HEAD'],capture_output=True,text=True,check=True,timeout=30)
assert subprocess.check_output(['git','-C',str(code),'rev-parse','HEAD'],text=True).strip()==P['commit']
assert all(sha(code/name)==digest for name,digest in P['code_files'].items())
for name,digest in P['controller_files'].items():
 if name not in P['allowed_viewer_changes']:assert sha(code/name)==digest
log=base/('GPU45_native_prestart_view_CPU_'+P['commit'][:12]+'.log')
if not log.exists():
 tested=subprocess.run([P['python'],'-B','-m','unittest',*P['tests'],'-q'],cwd=code,capture_output=True,text=True,timeout=150)
 create(log,(tested.stdout+'\n'+tested.stderr).encode())
 assert tested.returncode==0,log.read_text()
else:assert '\nOK\n' in log.read_text()
print(json.dumps(dict(status='GPU45_NATIVE_PRESTART_VIEW_CODE_VERIFIED',at=time.time(),code=str(code),commit=P['commit'],
 code_files=P['code_files'],controller_code=str(anchor),controller_commit=P['controller_commit'],Linux_CPU_tests=log.read_text(),
 running_controller_code_changed=False,jobs_started=False,signals_sent=False),ensure_ascii=False),flush=True)
'''

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 for name in ('checkout','registration','bundle','ssh-helper-dir'):parser.add_argument('--'+name,type=Path,required=True)
 args=parser.parse_args();checkout=args.checkout.resolve(strict=True);helper=args.ssh_helper_dir.resolve(strict=True)
 old=json.loads(args.registration.read_text(encoding='utf8'))
 git=['git','-c','safe.directory='+checkout.as_posix(),'-C',str(checkout)]
 commit=subprocess.check_output(git+['rev-parse','HEAD'],text=True).strip()
 allowed=['tools/watch_v24_native_prestart_guard.py','tests/test_v24_native_prestart_viewer.py'];files={}
 for name,digest in old['code_files'].items():
  raw=subprocess.check_output(git+['show',commit+':'+name]);assert raw==(checkout/name).read_bytes(),name
  files[name]=hashlib.sha256(raw).hexdigest()
  if name not in allowed:assert files[name]==digest,name
 tests=['tests.test_v24_matched_basic_cp_runtime','tests.test_v24_native_prestart_guard','tests.test_v24_native_prestart_training',
  'tests.test_v24_native_prestart_viewer','tests.test_v24_early_stop_downstream']
 bundle=args.bundle.read_bytes()
 payload=dict(commit=commit,controller_code=old['code'],controller_commit=old['commit'],controller_files=old['code_files'],
  allowed_viewer_changes=allowed,code_files=files,bundle=base64.b64encode(bundle).decode(),bundle_sha256=hashlib.sha256(bundle).hexdigest(),
  python='/home/aicompetition06/.conda/envs/nnunet/bin/python3.10',tests=tests)
 spec=importlib.util.spec_from_file_location('remote_once',helper/'remote_once.py');remote=importlib.util.module_from_spec(spec);spec.loader.exec_module(remote)
 result=json.loads(remote.run('PAYLOAD='+repr(payload)+'\n'+REMOTE,240))
 receipt=helper/('GPU45_NATIVE_PRESTART_VIEW_CODE_VERIFIED_'+str(time.time_ns())+'.json')
 receipt.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
 print(receipt);print(json.dumps({k:result[k] for k in ('status','code','commit','Linux_CPU_tests')},ensure_ascii=False))
if __name__=='__main__':main()
