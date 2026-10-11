"""Retarget only audited own display launchers; preserve common UI layout."""
import json,time
from remote_once import run,HERE

REMOTE=r'''
import hashlib,json,os,runpy,subprocess,time
from pathlib import Path
import psutil
P=PAYLOAD;assert os.getuid()==41833
base=Path('/var/tmp/aicompetition06_v24_20261010');release=Path(P['controller_root']).parent
code=Path(P['code']);target=release/'viewers'
oldrelease=base/'UI_COMMON_2_20261012_1791737120382169900'
original=json.loads((oldrelease/'installation_verified.json').read_text())
oldsha={Path(r['path']).name:r['new_sha256'] for r in original['previous_UI_launchers_backed_up_and_retargeted']}
aliases=('view_training.py','view_gpu5.py','view_gpu5_current.py','view_gpu5_auto.py')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def owned(p):
 p=Path(p);assert p.stat().st_uid==41833 and p.is_file() and not p.is_symlink();return p
def create(p,raw):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('xb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
def publish(p,v):create(p,(json.dumps(v,ensure_ascii=False,indent=2)+'\n').encode())
def actor(gpu):
 r=json.loads((base/'GPU456_AUTO_NATIVE_20261011_1791730500000000000'/('gpu'+str(gpu))/'registration.json').read_text())
 p=psutil.Process(r['pid']);assert p.uids().real==41833 and p.create_time()==r['create_time'] and p.cmdline()==r['command']
 return dict(pid=p.pid,birth=p.create_time(),command=p.cmdline())
before={str(g):actor(g) for g in (4,6)}
preserved={n:sha(base/n) for n in ('view_gpu4_current.py','view_gpu4_pipeline.py','view_gpu4_auto.py','view_gpu6.py','view_gpu6_current.py','view_gpu6_auto.py')}
previous={n:owned(base/n).read_bytes() for n in aliases}
assert all(hashlib.sha256(raw).hexdigest()==oldsha[n] for n,raw in previous.items())
assert not target.exists()
target.mkdir(mode=0o700)
names=('tools/watch_v24_early_stop_pipeline.py','tools/watch_v24_gpu_pipeline.py','tools/watch_v24_direct_gnn_training.py',
 'tools/watch_v24_matched_cp_scoring.py','tools/watch_v24_training.py','tools/training_monitor_adapter.py','tools/training_monitor_ui.py')
files={name:P['code_files'][name] for name in names};assert all(sha(code/name)==digest for name,digest in files.items())
request=Path(P['controller_root'])/'request.json'
args=['--request',str(request),'--guardian-root',str(release/'pause_guardian'),
 '--arm-root',str(base/'GPU45_interrupted_exact_CPU_prepared_20261011_1791705559311834000/gpu5'),
 '--renderer',str(code/'tools/watch_v24_training.py'),'--renderer-sha',files['tools/watch_v24_training.py']]
body='import hashlib,runpy,sys\nfrom pathlib import Path\nCODE='+repr(str(code))+'\nFILES='+repr(files)+'\n'
body+='for name,expected in FILES.items():\n if hashlib.sha256((Path(CODE)/name).read_bytes()).hexdigest()!=expected:raise ValueError("Pinned early viewer changed: "+name)\n'
body+='sys.path.insert(0,CODE)\nfrom tools.watch_v24_early_stop_pipeline import main\nmain('+repr(args)+'+sys.argv[1:])\n'
wrapper=target/'view_gpu5_early_stop.py';compile(body,str(wrapper),'exec');create(wrapper,body.encode())
oldcommon=owned(oldrelease/'viewers/view_training.py');oldcommonsha=sha(oldcommon)
dispatcher='import argparse,hashlib,runpy,sys\nfrom pathlib import Path\np=argparse.ArgumentParser(add_help=False)\np.add_argument("--gpu",type=int,choices=(4,5,6),required=True)\na,rest=p.parse_known_args()\n'
dispatcher+='target='+repr(str(wrapper))+' if a.gpu==5 else '+repr(str(oldcommon))+'\n'
dispatcher+='expected='+repr(sha(wrapper))+' if a.gpu==5 else '+repr(oldcommonsha)+'\n'
dispatcher+='if hashlib.sha256(Path(target).read_bytes()).hexdigest()!=expected:raise ValueError("Pinned common viewer changed")\n'
dispatcher+='sys.argv=[target]+([] if a.gpu==5 else ["--gpu",str(a.gpu)])+rest\nrunpy.run_path(target,run_name="__main__")\n'
common=target/'view_training.py';compile(dispatcher,str(common),'exec');create(common,dispatcher.encode())
checked=subprocess.run([P['python'],'-B',str(common),'--gpu','5','--once','--json'],capture_output=True,text=True,timeout=40)
assert checked.returncode==0,checked.stderr
snapshot=json.loads(checked.stdout)
assert snapshot['GPU']==5 and snapshot['snapshot_fresh'] is True and snapshot['status']!='READ_ERROR'
changes=[]
for name,raw in previous.items():
 create(target/'previous_UI_launchers'/name,raw)
 launch=common if name=='view_training.py' else wrapper
 replacement='import hashlib,runpy\nfrom pathlib import Path\nTARGET='+repr(str(launch))+'\nEXPECTED='+repr(sha(launch))+'\nif hashlib.sha256(Path(TARGET).read_bytes()).hexdigest()!=EXPECTED:raise ValueError("Pinned GPU5/common viewer changed")\nrunpy.run_path(TARGET,run_name="__main__")\n'
 fresh=target/'replacements'/name;compile(replacement,str(base/name),'exec');create(fresh,replacement.encode())
 assert sha(owned(base/name))==oldsha[name]
 os.replace(fresh,base/name)
 changes.append(dict(path=str(base/name),old_sha256=oldsha[name],new_sha256=sha(base/name),backup=str(target/'previous_UI_launchers'/name)))
views=[]
for gpu in (4,5,6):
 result=subprocess.run([P['python'],'-B',str(base/'view_training.py'),'--gpu',str(gpu),'--once','--json'],capture_output=True,text=True,timeout=40)
 assert result.returncode==0,result.stderr
 value=json.loads(result.stdout);assert value['GPU']==gpu and value['snapshot_fresh'] is True and value['status']!='READ_ERROR'
 text=subprocess.run([P['python'],'-B',str(base/'view_training.py'),'--gpu',str(gpu),'--once'],capture_output=True,text=True,timeout=40)
 assert text.returncode==0 and len(text.stdout.splitlines())==(9 if value['active_stage']=='gnn' else 8)
 views.append(dict(GPU=gpu,snapshot=value,text=text.stdout))
for name in aliases[1:]:
 result=subprocess.run([P['python'],'-B',str(base/name),'--once','--json'],capture_output=True,text=True,timeout=40)
 value=json.loads(result.stdout);assert result.returncode==0 and value['GPU']==5 and value['snapshot_fresh'] is True
after={str(g):actor(g) for g in (4,6)}
assert before==after and all(sha(base/n)==digest for n,digest in preserved.items()) and sha(oldcommon)==oldcommonsha
receipt=dict(status='GPU5_EARLY_STOP_COMMON_UI2_INSTALLED_VERIFIED',at=time.time(),commit=P['commit'],views=views,
 preserved_GPU46_actors=after,GPU46_aliases_unchanged=True,old_frozen_UI_code_unchanged=True,
 launchers_backed_up_and_retargeted=changes,scientific_files_written=False,jobs_started=False,signals_sent=False)
publish(release/'early_stop_viewer_installed.json',receipt)
print(json.dumps(receipt,ensure_ascii=False),flush=True)
'''

def main():
 data=json.loads((HERE/'GPU5_EARLY_STOP_CHAIN_REGISTERED_1791741440255409200.json').read_text(encoding='utf8'))
 payload={k:data[k] for k in ('controller_root','code','code_files','commit')}
 payload['python']='/home/aicompetition06/.conda/envs/nnunet/bin/python3.10'
 script='PAYLOAD='+repr(payload)+'\n'+REMOTE
 prepared=HERE/('install_GPU5_early_viewer_'+payload['commit'][:12]+'_PREPARED.py')
 if prepared.exists():assert prepared.read_text(encoding='utf8')==script
 else:prepared.write_text(script,encoding='utf8')
 result=json.loads(run(script,180));receipt=HERE/('GPU5_EARLY_STOP_VIEWER_INSTALLED_'+str(time.time_ns())+'.json')
 receipt.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
 print(receipt);print(json.dumps(dict(status=result['status'],views=[dict(GPU=row['GPU'],text=row['text']) for row in result['views']]),ensure_ascii=False))

if __name__=='__main__':main()
