"""Preserve current launchers and connect fixed UI2 to GPU4/5 adoption guards."""
import argparse,importlib.util,json,time
from pathlib import Path

REMOTE=r'''
import hashlib,json,os,subprocess,time
from pathlib import Path
P=PAYLOAD
assert os.getuid()==41833
base=Path('/var/tmp/aicompetition06_v24_20261010');code=Path(P.get('viewer_code',P['code']))
files=P.get('viewer_files',P['code_files']);viewer_commit=P.get('viewer_commit',P['commit'])
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def create(p,raw):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
def owned(p):
 assert p.is_file() and not p.is_symlink() and p.stat().st_uid==41833;return p
old=read(base/'UI_COMMON_2_20261012_1791737120382169900/installation_verified.json')
expected={Path(r['path']).name:r['new_sha256'] for r in old['previous_UI_launchers_backed_up_and_retargeted']}
early=read(base/'GPU5_EARLY_STOP_E27_20261012_1791740800000000000/UI_STARTUP_2_89f1714e034a/installation_verified.json')
expected.update({Path(r['path']).name:r['new_sha256'] for r in early['launchers_backed_up_and_retargeted']})
gpu6=base/'GPU456_AUTO_NATIVE_20261011_1791730500000000000/gpu6/NATIVE_STARTUP_RETRY_dea813d273fe'
last=read(gpu6/'retry_viewer_installed.json')
expected.update({Path(r['path']).name:r['new_sha256'] for r in last['launchers_backed_up_and_retargeted']})
aliases={4:('view_gpu4_current.py','view_gpu4_pipeline.py','view_gpu4_auto.py'),5:('view_gpu5.py','view_gpu5_current.py','view_gpu5_auto.py')}
names=('view_training.py',*aliases[4],*aliases[5])
previous={name:owned(base/name).read_bytes() for name in names}
assert all(hashlib.sha256(raw).hexdigest()==expected[name] for name,raw in previous.items())
preserved={name:sha(base/name) for name in ('view_gpu6.py','view_gpu6_current.py','view_gpu6_auto.py')}
assert all(sha(code/name)==digest for name,digest in files.items())
target=base/('GPU45_NATIVE_PRESTART_UI_'+viewer_commit[:12]);assert not target.exists(),'Unknown installation outcome; inspect exact backup rather than replace'
target.mkdir(mode=0o700)
for name,raw in previous.items():create(target/'previous_UI_launchers'/name,raw)
wrappers={}
for row in P['arms']:
 gpu=row['GPU'];request=Path(row['request']);assert sha(request)==row['request_sha256']
 args=['--request',str(request),'--renderer',str(code/'tools/watch_v24_training.py'),'--renderer-sha',files['tools/watch_v24_training.py']]
 body='import hashlib,sys\nfrom pathlib import Path\nCODE='+repr(str(code))+'\nFILES='+repr(files)+'\n'
 body+='for name,expected in FILES.items():\n if hashlib.sha256((Path(CODE)/name).read_bytes()).hexdigest()!=expected:raise ValueError("Pinned native prestart viewer changed: "+name)\n'
 body+='sys.path.insert(0,CODE)\nfrom tools.watch_v24_native_prestart_guard import main\nmain('+repr(args)+'+sys.argv[1:])\n'
 wrapper=target/('view_gpu'+str(gpu)+'.py');compile(body,str(wrapper),'exec');create(wrapper,body.encode());wrappers[gpu]=wrapper
oldcommon=target/'previous_UI_launchers/view_training.py'
dispatch='import argparse,hashlib,runpy,sys\nfrom pathlib import Path\np=argparse.ArgumentParser(add_help=False)\np.add_argument("--gpu",type=int,choices=(4,5,6),required=True)\na,rest=p.parse_known_args()\n'
dispatch+='paths='+repr({gpu:str(path) for gpu,path in wrappers.items()})+'\npaths[6]='+repr(str(oldcommon))+'\n'
dispatch+='hashes='+repr({gpu:sha(path) for gpu,path in wrappers.items()})+'\nhashes[6]='+repr(sha(oldcommon))+'\n'
dispatch+='target=paths[a.gpu]\nif hashlib.sha256(Path(target).read_bytes()).hexdigest()!=hashes[a.gpu]:raise ValueError("Pinned common prestart viewer changed")\nsys.argv=[target]+(["--gpu","6"] if a.gpu==6 else [])+rest\nrunpy.run_path(target,run_name="__main__")\n'
common=target/'view_training.py';compile(dispatch,str(common),'exec');create(common,dispatch.encode())
def readview(path,*args):
 r=subprocess.run([P['python'],'-B',str(path),*args,'--once','--json'],capture_output=True,text=True,timeout=45)
 assert r.returncode==0,r.stderr
 v=json.loads(r.stdout);assert v.get('snapshot_fresh') is True and v['status'] not in ('READ_ERROR','READ_PENDING'),v
 return v
for gpu in (4,5,6):assert readview(common,'--gpu',str(gpu))['GPU']==gpu
changes=[]
for name in names:
 launch=common if name=='view_training.py' else wrappers[4 if name in aliases[4] else 5]
 raw=('import hashlib,runpy\nfrom pathlib import Path\nTARGET='+repr(str(launch))+'\nEXPECTED='+repr(sha(launch))+'\nif hashlib.sha256(Path(TARGET).read_bytes()).hexdigest()!=EXPECTED:raise ValueError("Pinned native prestart/common viewer changed")\nrunpy.run_path(TARGET,run_name="__main__")\n').encode()
 replacement=target/('replacement_'+name);compile(raw,str(base/name),'exec');create(replacement,raw)
 assert sha(owned(base/name))==expected[name];os.replace(replacement,base/name)
 changes.append(dict(path=str(base/name),old_sha256=expected[name],new_sha256=sha(base/name),backup=str(target/'previous_UI_launchers'/name)))
views=[]
for gpu in (4,5,6):
 v=readview(base/'view_training.py','--gpu',str(gpu));assert v['GPU']==gpu
 t=subprocess.run([P['python'],'-B',str(base/'view_training.py'),'--gpu',str(gpu),'--once'],capture_output=True,text=True,timeout=45)
 assert t.returncode==0,t.stderr
 assert t.stdout.startswith('nnU-Net controller:') or (len(t.stdout.splitlines()) in (8,9) and t.stdout.startswith('실험/상태:')),t.stdout
 views.append(dict(GPU=gpu,snapshot=v,text=t.stdout))
for gpu,nameset in aliases.items():
 for name in nameset:assert readview(base/name)['GPU']==gpu
assert all(sha(base/name)==digest for name,digest in preserved.items())
result=dict(status='GPU45_FIXED_NATIVE_PRESTART_UI2_VERIFIED',at=time.time(),commit=viewer_commit,controller_commit=P['commit'],controller_code=P['code'],viewer_code=str(code),target=str(target),views=views,
 launchers_backed_up_and_retargeted=changes,GPU6_aliases_unchanged=True,jobs_started=False,signals_sent=False)
create(target/'installation_verified.json',(json.dumps(result,ensure_ascii=False,indent=2)+'\n').encode())
print(json.dumps(result,ensure_ascii=False),flush=True)
'''

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--registration',type=Path,required=True);parser.add_argument('--ssh-helper-dir',type=Path,required=True)
 parser.add_argument('--viewer-code-proof',type=Path)
 args=parser.parse_args();payload=json.loads(args.registration.read_text(encoding='utf8'));payload['python']='/home/aicompetition06/.conda/envs/nnunet/bin/python3.10'
 if args.viewer_code_proof:
  viewer=json.loads(args.viewer_code_proof.read_text(encoding='utf8'))
  assert viewer['status']=='GPU45_NATIVE_PRESTART_VIEW_CODE_VERIFIED' and viewer['controller_commit']==payload['commit'] and viewer['controller_code']==payload['code']
  payload.update(viewer_code=viewer['code'],viewer_files=viewer['code_files'],viewer_commit=viewer['commit'])
 helper=args.ssh_helper_dir.resolve(strict=True);script='PAYLOAD='+repr(payload)+'\n'+REMOTE
 prepared=helper/('install_GPU45_native_prestart_UI_'+payload.get('viewer_commit',payload['commit'])[:12]+'_PREPARED.py')
 if prepared.exists():assert prepared.read_text(encoding='utf8')==script
 else:prepared.write_text(script,encoding='utf8')
 spec=importlib.util.spec_from_file_location('remote_once',helper/'remote_once.py');remote=importlib.util.module_from_spec(spec);spec.loader.exec_module(remote)
 result=json.loads(remote.run(script,240))
 receipt=helper/('GPU45_NATIVE_PRESTART_VIEWERS_VERIFIED_'+str(time.time_ns())+'.json')
 receipt.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
 print(receipt);print(json.dumps(dict(status=result['status'],views=[dict(GPU=r['GPU'],text=r['text']) for r in result['views']]),ensure_ascii=False))
if __name__=='__main__':main()
