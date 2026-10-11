"""Retarget only own GPU6 display aliases to the admitted retry, with backups."""
import argparse,importlib.util,json,time
from pathlib import Path

REMOTE=r'''
import hashlib,json,os,subprocess,time
from pathlib import Path
P=PAYLOAD
assert os.getuid()==41833
base=Path('/var/tmp/aicompetition06_v24_20261010');root=Path(P['recovery_root']);code=Path(P['code'])
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def owned(p):
 p=Path(p);assert p.is_file() and not p.is_symlink() and p.stat().st_uid==41833;return p
def create(p,raw):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
old=json.loads((base/'UI_COMMON_2_20261012_1791737120382169900/installation_verified.json').read_text())
expected={Path(row['path']).name:row['new_sha256'] for row in old['previous_UI_launchers_backed_up_and_retargeted']}
current_common=base/'GPU5_EARLY_STOP_E27_20261012_1791740800000000000/UI_STARTUP_2_89f1714e034a/installation_verified.json'
receipt=json.loads(current_common.read_text())
expected.update({Path(row['path']).name:row['new_sha256'] for row in receipt['launchers_backed_up_and_retargeted'] if Path(row['path']).name=='view_training.py'})
aliases=('view_training.py','view_gpu6.py','view_gpu6_current.py','view_gpu6_auto.py')
previous={name:owned(base/name).read_bytes() for name in aliases}
assert all(hashlib.sha256(raw).hexdigest()==expected[name] for name,raw in previous.items())
preserved={name:sha(base/name) for name in ('view_gpu4_current.py','view_gpu4_pipeline.py','view_gpu4_auto.py','view_gpu5.py','view_gpu5_current.py','view_gpu5_auto.py')}
request=root/'request.json';assert sha(request)==P['request_sha256']
assert all(sha(code/name)==digest for name,digest in P['code_files'].items())
target=root/'viewers';assert not target.exists(),'Inspect unknown earlier viewer installation rather than overwrite'
target.mkdir(mode=0o700)
files=P['code_files'];args=['--request',str(request),'--renderer',str(code/'tools/watch_v24_training.py'),
 '--renderer-sha',files['tools/watch_v24_training.py']]
body='import hashlib,sys\nfrom pathlib import Path\nCODE='+repr(str(code))+'\nFILES='+repr(files)+'\n'
body+='for name,expected in FILES.items():\n if hashlib.sha256((Path(CODE)/name).read_bytes()).hexdigest()!=expected:raise ValueError("Pinned retry viewer changed: "+name)\n'
body+='sys.path.insert(0,CODE)\nfrom tools.watch_v24_native_startup_retry import main\nmain('+repr(args)+'+sys.argv[1:])\n'
wrapper=target/'view_gpu6_retry.py';compile(body,str(wrapper),'exec');create(wrapper,body.encode())
for name,raw in previous.items():create(target/'previous_UI_launchers'/name,raw)
oldcommon=target/'previous_UI_launchers/view_training.py'
dispatcher='import argparse,hashlib,runpy,sys\nfrom pathlib import Path\np=argparse.ArgumentParser(add_help=False)\np.add_argument("--gpu",type=int,choices=(4,5,6),required=True)\na,rest=p.parse_known_args()\n'
dispatcher+='target='+repr(str(wrapper))+' if a.gpu==6 else '+repr(str(oldcommon))+'\n'
dispatcher+='expected='+repr(sha(wrapper))+' if a.gpu==6 else '+repr(sha(oldcommon))+'\n'
dispatcher+='if hashlib.sha256(Path(target).read_bytes()).hexdigest()!=expected:raise ValueError("Pinned common viewer changed")\n'
dispatcher+='sys.argv=[target]+([] if a.gpu==6 else ["--gpu",str(a.gpu)])+rest\nrunpy.run_path(target,run_name="__main__")\n'
common=target/'view_training.py';compile(dispatcher,str(common),'exec');create(common,dispatcher.encode())
def readview(path,*args):
 result=subprocess.run([P['python'],'-B',str(path),*args,'--once','--json'],capture_output=True,text=True,timeout=40)
 assert result.returncode==0,result.stderr
 value=json.loads(result.stdout);assert value['snapshot_fresh'] is True and value['status'] not in ('READ_ERROR','READ_PENDING'),value
 return value
first=readview(common,'--gpu','6');assert first['GPU']==6
changes=[]
for name in aliases:
 launch=common if name=='view_training.py' else wrapper
 raw=('import hashlib,runpy\nfrom pathlib import Path\nTARGET='+repr(str(launch))+'\nEXPECTED='+repr(sha(launch))+'\nif hashlib.sha256(Path(TARGET).read_bytes()).hexdigest()!=EXPECTED:raise ValueError("Pinned GPU6/common viewer changed")\nrunpy.run_path(TARGET,run_name="__main__")\n').encode()
 replacement=target/('replacement_'+name);compile(raw,str(base/name),'exec');create(replacement,raw)
 assert sha(owned(base/name))==expected[name]
 os.replace(replacement,base/name)
 changes.append(dict(path=str(base/name),old_sha256=expected[name],new_sha256=sha(base/name),backup=str(target/'previous_UI_launchers'/name)))
views=[]
for gpu in (4,5,6):
 value=readview(base/'view_training.py','--gpu',str(gpu));assert value['GPU']==gpu
 text=subprocess.run([P['python'],'-B',str(base/'view_training.py'),'--gpu',str(gpu),'--once'],capture_output=True,text=True,timeout=40)
 assert text.returncode==0,text.stderr
 lines=text.stdout.splitlines()
 if text.stdout.startswith('nnU-Net controller:'):
  assert len(lines)>=3 and lines[1]=='현재 등록된 nnU-Net 학습/BEST 평가 작업 실행 중' and lines[2].startswith('nnU-Net epoch ')
 else:assert len(lines) in (8,9) and lines[0].startswith('실험/상태:'),text.stdout
 views.append(dict(GPU=gpu,snapshot=value,text=text.stdout))
for name in aliases[1:]:assert readview(base/name)['GPU']==6
assert all(sha(base/name)==digest for name,digest in preserved.items())
assert sha(request)==P['request_sha256'] and all(sha(code/name)==digest for name,digest in P['code_files'].items())
installed=dict(status='GPU6_NATIVE_RETRY_COMMON_UI2_INSTALLED_VERIFIED',at=time.time(),commit=P['commit'],views=views,
 launchers_backed_up_and_retargeted=changes,GPU45_aliases_unchanged=True,scientific_files_written=False,jobs_started=False,signals_sent=False)
create(root/'retry_viewer_installed.json',(json.dumps(installed,ensure_ascii=False,indent=2)+'\n').encode())
print(json.dumps(installed,ensure_ascii=False),flush=True)
'''


def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--registration',type=Path,required=True)
 parser.add_argument('--ssh-helper-dir',type=Path,required=True)
 args=parser.parse_args();data=json.loads(args.registration.read_text(encoding='utf8'))
 helper=args.ssh_helper_dir.resolve(strict=True)
 payload={key:data[key] for key in ('code','commit','recovery_root','request_sha256','code_files')}
 payload['python']='/home/aicompetition06/.conda/envs/nnunet/bin/python3.10'
 script='PAYLOAD='+repr(payload)+'\n'+REMOTE
 prepared=helper/('install_GPU6_native_retry_UI_'+payload['commit'][:12]+'_PREPARED.py')
 if prepared.exists():assert prepared.read_text(encoding='utf8')==script
 else:prepared.write_text(script,encoding='utf8')
 spec=importlib.util.spec_from_file_location('remote_once',helper/'remote_once.py');remote=importlib.util.module_from_spec(spec);spec.loader.exec_module(remote)
 result=json.loads(remote.run(script,180))
 receipt=helper/('GPU6_NATIVE_RETRY_VIEWER_INSTALLED_'+str(time.time_ns())+'.json')
 receipt.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
 print(receipt);print(json.dumps(dict(status=result['status'],views=[dict(GPU=row['GPU'],text=row['text']) for row in result['views']]),ensure_ascii=False))


if __name__=='__main__':main()
