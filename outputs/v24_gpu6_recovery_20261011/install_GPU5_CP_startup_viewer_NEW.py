"""Install a fresh, Git-pinned display release without changing running science."""
import argparse
import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import time

UI_FILES = (
    'tools/__init__.py', 'tools/watch_v24_early_stop_pipeline.py',
    'tools/watch_v24_gpu_pipeline.py', 'tools/watch_v24_direct_gnn_training.py',
    'tools/watch_v24_matched_cp_scoring.py', 'tools/watch_v24_training.py',
    'tools/training_monitor_adapter.py', 'tools/training_monitor_ui.py',
    'tests/test_v24_early_stop_viewer.py',
    'tests/test_training_monitor_ui.py', 'tests/test_training_monitor_adapter.py',
)

REMOTE = r'''
import base64,hashlib,json,os,subprocess,time
from pathlib import Path
import psutil
P=PAYLOAD
base=Path('/var/tmp/aicompetition06_v24_20261010')
experiment=base/'GPU5_EARLY_STOP_E27_20261012_1791740800000000000'
assert os.getuid()==41833 and experiment.stat().st_uid==41833
release=experiment/('UI_STARTUP_2_'+P['commit'][:12])
code=release/'code';viewers=release/'viewers'
aliases=('view_training.py','view_gpu5.py','view_gpu5_current.py','view_gpu5_auto.py')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def owned(p):
 p=Path(p);assert p.is_file() and not p.is_symlink() and p.stat().st_uid==41833;return p
def create(p,raw):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
def proof(p):
 p=owned(p);s=p.stat();return dict(path=str(p),sha256=sha(p),stat=[s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns])
def actor(gpu):
 r=json.loads((base/'GPU456_AUTO_NATIVE_20261011_1791730500000000000'/('gpu'+str(gpu))/'registration.json').read_text())
 try:
  p=psutil.Process(r['pid']);assert p.uids().real==41833 and p.create_time()==r['create_time'] and p.cmdline()==r['command']
  live=True
 except psutil.NoSuchProcess:live=False
 return dict(pid=r['pid'],birth=r['create_time'],command=r['command'],alive=live)

previous_install=json.loads((experiment/'early_stop_viewer_installed.json').read_text())
expected={Path(r['path']).name:r['new_sha256'] for r in previous_install['launchers_backed_up_and_retargeted']}
previous={n:owned(base/n).read_bytes() for n in aliases}
assert all(hashlib.sha256(raw).hexdigest()==expected[n] for n,raw in previous.items())
preserved_aliases={n:sha(owned(base/n)) for n in ('view_gpu4_current.py','view_gpu4_pipeline.py','view_gpu4_auto.py','view_gpu6.py','view_gpu6_current.py','view_gpu6_auto.py')}
request_path=experiment/'controller/request.json';request=json.loads(request_path.read_text())
protected=[proof(request_path),proof(experiment/'authorization.json'),proof(experiment/'controller/registration.json'),proof(experiment/'controller/lease.json')]
science_code=Path(request['code'])
protected.extend(proof(science_code/name) for name in request['code_files'])
before={str(g):actor(g) for g in (4,6)}
assert not release.exists(),'Preserve unknown prior outcome and inspect its receipt before retrying'
release.mkdir(mode=0o700)
manifest={}
for name,entry in P['files'].items():
 raw=base64.b64decode(entry['data'],validate=True);assert hashlib.sha256(raw).hexdigest()==entry['sha256']
 compile(raw,str(code/name),'exec');create(code/name,raw);manifest[name]=entry['sha256']
create(release/'manifest.json',(json.dumps(dict(commit=P['commit'],files=manifest),indent=2)+'\n').encode())
test=subprocess.run([P['python'],'-B','-m','unittest','tests.test_v24_early_stop_viewer','tests.test_training_monitor_ui','tests.test_training_monitor_adapter','-q'],cwd=code,capture_output=True,text=True,timeout=90)
create(release/'Linux_CPU_tests.log',(test.stdout+'\n'+test.stderr).encode())
assert test.returncode==0,test.stdout+test.stderr
oldcommon=owned(base/'UI_COMMON_2_20261012_1791737120382169900/viewers/view_training.py');oldcommonsha=sha(oldcommon)
args=['--request',str(request_path),'--guardian-root',str(experiment/'pause_guardian'),'--arm-root',str(base/'GPU45_interrupted_exact_CPU_prepared_20261011_1791705559311834000/gpu5'),
 '--renderer',str(code/'tools/watch_v24_training.py'),'--renderer-sha',manifest['tools/watch_v24_training.py']]
body='import hashlib,sys\nfrom pathlib import Path\nCODE='+repr(str(code))+'\nFILES='+repr(manifest)+'\n'
body+='for name,expected in FILES.items():\n if hashlib.sha256((Path(CODE)/name).read_bytes()).hexdigest()!=expected:raise ValueError("Pinned startup viewer changed: "+name)\n'
body+='sys.path.insert(0,CODE)\nfrom tools.watch_v24_early_stop_pipeline import main\nmain('+repr(args)+'+sys.argv[1:])\n'
wrapper=viewers/'view_gpu5_early_stop.py';compile(body,str(wrapper),'exec');create(wrapper,body.encode())
dispatcher='import argparse,hashlib,runpy,sys\nfrom pathlib import Path\np=argparse.ArgumentParser(add_help=False)\np.add_argument("--gpu",type=int,choices=(4,5,6),required=True)\na,rest=p.parse_known_args()\n'
dispatcher+='target='+repr(str(wrapper))+' if a.gpu==5 else '+repr(str(oldcommon))+'\n'
dispatcher+='expected='+repr(sha(wrapper))+' if a.gpu==5 else '+repr(oldcommonsha)+'\n'
dispatcher+='if hashlib.sha256(Path(target).read_bytes()).hexdigest()!=expected:raise ValueError("Pinned common viewer changed")\n'
dispatcher+='sys.argv=[target]+([] if a.gpu==5 else ["--gpu",str(a.gpu)])+rest\nrunpy.run_path(target,run_name="__main__")\n'
common=viewers/'view_training.py';compile(dispatcher,str(common),'exec');create(common,dispatcher.encode())
def snapshot(path,*args):
 result=subprocess.run([P['python'],'-B',str(path),*args,'--once','--json'],capture_output=True,text=True,timeout=40)
 assert result.returncode==0,result.stderr
 value=json.loads(result.stdout);assert value['snapshot_fresh'] is True and value['status'] not in ('READ_ERROR','READ_PENDING')
 return value
preflight=snapshot(common,'--gpu','5');assert preflight['GPU']==5
changes=[]
for name,raw in previous.items():
 create(viewers/'previous_UI_launchers'/name,raw)
 target=common if name=='view_training.py' else wrapper
 replacement='import hashlib,runpy\nfrom pathlib import Path\nTARGET='+repr(str(target))+'\nEXPECTED='+repr(sha(target))+'\nif hashlib.sha256(Path(TARGET).read_bytes()).hexdigest()!=EXPECTED:raise ValueError("Pinned GPU5/common viewer changed")\nrunpy.run_path(TARGET,run_name="__main__")\n'
 fresh=viewers/'replacements'/name;compile(replacement,str(base/name),'exec');create(fresh,replacement.encode())
 assert sha(owned(base/name))==expected[name]
 os.replace(fresh,base/name)
 changes.append(dict(path=str(base/name),old_sha256=expected[name],new_sha256=sha(base/name),backup=str(viewers/'previous_UI_launchers'/name)))
views=[]
for gpu in (4,5,6):
 value=snapshot(base/'view_training.py','--gpu',str(gpu));assert value['GPU']==gpu
 text=subprocess.run([P['python'],'-B',str(base/'view_training.py'),'--gpu',str(gpu),'--once'],capture_output=True,text=True,timeout=40)
 assert text.returncode==0,text.stderr
 lines=text.stdout.splitlines()
 if text.stdout.startswith('nnU-Net controller:'):
  assert len(lines)>=3 and lines[1]=='현재 등록된 nnU-Net 학습/BEST 평가 작업 실행 중' and lines[2].startswith('nnU-Net epoch ')
 else:
  assert len(lines) in (8,9) and lines[0].startswith('실험/상태:'),text.stdout
 views.append(dict(GPU=gpu,snapshot=value,text=text.stdout))
for name in aliases[1:]:assert snapshot(base/name)['GPU']==5
after={str(g):actor(g) for g in (4,6)}
assert all({k:v for k,v in before[g].items() if k!='alive'}=={k:v for k,v in after[g].items() if k!='alive'} for g in before)
assert all(sha(base/n)==digest for n,digest in preserved_aliases.items()) and sha(oldcommon)==oldcommonsha
assert all(proof(r['path'])==r for r in protected)
receipt=dict(status='GPU5_CP_STARTUP_COMMON_UI2_INSTALLED_VERIFIED',at=time.time(),commit=P['commit'],code=str(code),views=views,CPU_tests=test.stderr,
 launchers_backed_up_and_retargeted=changes,GPU46_aliases_unchanged=True,GPU46_actor_identity_unchanged=True,
 scientific_files_unchanged=True,controller_request_unchanged=True,signals_sent=False,jobs_started=False)
create(release/'installation_verified.json',(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n').encode())
print(json.dumps(receipt,ensure_ascii=False),flush=True)
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkout', type=Path, required=True)
    parser.add_argument('--ssh-helper-dir', type=Path, required=True)
    args = parser.parse_args()
    checkout = args.checkout.resolve(strict=True)
    helper_dir = args.ssh_helper_dir.resolve(strict=True)
    git = ['git', '-c', 'safe.directory=' + checkout.as_posix(), '-C', str(checkout)]
    commit = subprocess.check_output(git + ['rev-parse', 'HEAD'], text=True).strip()
    files = {}
    for name in UI_FILES:
        raw = subprocess.check_output(git + ['show', commit + ':' + name])
        assert (checkout / name).read_bytes() == raw, 'Commit the reviewed UI files before deployment'
        files[name] = dict(sha256=hashlib.sha256(raw).hexdigest(), data=base64.b64encode(raw).decode())
    payload = dict(commit=commit, files=files, python='/home/aicompetition06/.conda/envs/nnunet/bin/python3.10')
    script = 'PAYLOAD=' + repr(payload) + '\n' + REMOTE
    prepared = helper_dir / ('install_GPU5_CP_startup_' + commit[:12] + '_PREPARED.py')
    if prepared.exists():
        assert prepared.read_text(encoding='utf8') == script
    else:
        prepared.write_text(script, encoding='utf8')
    spec = importlib.util.spec_from_file_location('remote_once', helper_dir / 'remote_once.py')
    remote = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(remote)
    result = json.loads(remote.run(script, 180))
    receipt = helper_dir / ('GPU5_CP_STARTUP_VIEWER_INSTALLED_' + str(time.time_ns()) + '.json')
    receipt.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf8')
    print(receipt)
    print(json.dumps(dict(status=result['status'], code=result['code'], CPU_tests=result['CPU_tests'],
        views=[dict(GPU=row['GPU'], text=row['text']) for row in result['views']]), ensure_ascii=False))


if __name__ == '__main__':
    main()
