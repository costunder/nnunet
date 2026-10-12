"""Pin the verified existing-artifact GPU6 viewer behind stable frontdoors.

Old sealed releases remain intact. Frontdoors are backed up byte for byte and
replaced only if their expected hash still matches the inspected installation.
"""
import argparse, importlib.util, json, time
from pathlib import Path

REMOTE=r'''
import hashlib,json,os,subprocess,time
from pathlib import Path
P=PAYLOAD
assert os.getuid()==41833
base=Path('/var/tmp/aicompetition06_v24_20261010');code=Path(P['code'])
root=base/('GPU6_POST_COMPLETION_UI_'+P['commit'][:12])
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def create(path,raw):
 with Path(path).open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
def exact(path,raw):
 if path.exists():assert path.read_bytes()==raw,str(path)
 else:create(path,raw)
assert subprocess.check_output(['git','-C',str(code),'rev-parse','HEAD'],text=True).strip()==P['commit']
assert all(sha(code/name)==value for name,value in P['code_files'].items())
state=json.loads(Path(P['request']).with_name('status.json').read_text())
assert state['status']=='COMPLETE' and state['completed_epochs']==250 and state['evaluated_cases']==26
if not root.exists():root.mkdir(mode=0o700)
assert not root.is_symlink() and root.stat().st_uid==41833
wrapper=('import hashlib,sys\nfrom pathlib import Path\nCODE='+repr(str(code))+'\nFILES='+repr(P['code_files'])+'\n'
 'for name,expected in FILES.items():\n'
 ' if hashlib.sha256((Path(CODE)/name).read_bytes()).hexdigest()!=expected:raise ValueError("Pinned completion viewer changed: "+name)\n'
 'sys.path.insert(0,CODE)\nfrom tools.watch_v24_native_startup_retry import main\n'
 'main('+repr(['--request',str(Path(P['request']).parent.parent/'NATIVE_STARTUP_RETRY_dea813d273fe/request.json'),
 '--completion-request',P['request'],'--renderer',str(code/'tools/watch_v24_training.py'),
 '--renderer-sha',P['code_files']['tools/watch_v24_training.py']])+'+sys.argv[1:])\n').encode()
target=root/'view_gpu6.py';exact(target,wrapper)
observed=subprocess.run([P['python'],'-B',str(target),'--json'],capture_output=True,text=True,timeout=45,check=True)
view=json.loads(observed.stdout)
assert view['status']=='COMPLETE' and view['full_native_training'] is True and view['full_native_evaluation'] is True,view
assert view['native']['completed_epochs']==250 and view['native']['BEST_evaluated_cases']==26
assert not view.get('read_problem'),view
paths={4:str(base/'GPU45_NATIVE_PRESTART_UI_ec81af9c0f8b/view_gpu4.py'),
 5:str(base/'GPU45_NATIVE_PRESTART_UI_ec81af9c0f8b/view_gpu5.py'),6:str(target)}
hashes={4:'d20ea4d8b59d94f735f40bc6659c6f80d3bf15d9f045d335137c1e710e98a8f1',
 5:'8a84adc8ecd3ebdf8897a800ec428f87ed2bfed09c75d983fb4f8eaebb4f80ee',6:sha(target)}
assert all(sha(path)==hashes[gpu] for gpu,path in paths.items())
common=('import argparse,hashlib,runpy,sys\nfrom pathlib import Path\n'
 'p=argparse.ArgumentParser(add_help=False)\np.add_argument("--gpu",type=int,choices=(4,5,6),required=True)\n'
 'a,rest=p.parse_known_args()\npaths='+repr(paths)+'\nhashes='+repr(hashes)+'\ntarget=paths[a.gpu]\n'
 'if hashlib.sha256(Path(target).read_bytes()).hexdigest()!=hashes[a.gpu]:raise ValueError("Pinned common viewer changed")\n'
 'sys.argv=[target]+rest\nrunpy.run_path(target,run_name="__main__")\n').encode()
exact(root/'view_training.py',common)
backups=root/'previous_frontdoors'
if not backups.exists():backups.mkdir(mode=0o700)
plan={name:target for name in ('view_gpu6.py','view_gpu6_current.py','view_gpu6_auto.py')}
plan['view_training.py']=root/'view_training.py'
installed={}
for name,newtarget in plan.items():
 path=base/name
 expected=P['previous_frontdoor_hashes'][name]
 raw=('import hashlib,runpy\nfrom pathlib import Path\nTARGET='+repr(str(newtarget))+'\nEXPECTED='+repr(sha(newtarget))+'\n'
 'if hashlib.sha256(Path(TARGET).read_bytes()).hexdigest()!=EXPECTED:raise ValueError("Pinned GPU6/common completion viewer changed")\n'
 'runpy.run_path(TARGET,run_name="__main__")\n').encode()
 if path.read_bytes()!=raw:
  assert not path.is_symlink() and path.stat().st_uid==41833 and sha(path)==expected,path
  exact(backups/name,path.read_bytes())
  temporary=root/(name+'.frontdoor.tmp');create(temporary,raw);os.replace(temporary,path)
 else:assert (backups/name).exists()
 installed[name]=dict(path=str(path),sha256=sha(path),target=str(newtarget))
checks={}
for command in ([str(base/'view_gpu6_current.py'),'--json'],[str(base/'view_training.py'),'--gpu','6','--json']):
 result=subprocess.run([P['python'],'-B',*command],capture_output=True,text=True,timeout=45,check=True)
 item=json.loads(result.stdout)
 assert item['status']=='COMPLETE' and item['full_native_training'] is True and item['full_native_evaluation'] is True,item
 checks[command[0]]=dict(status=item['status'],completed_epochs=item['native']['completed_epochs'],
  BEST_evaluated_cases=item['native']['BEST_evaluated_cases'],read_problem=item.get('read_problem'))
once=subprocess.run([P['python'],'-B',str(base/'view_gpu6_current.py'),'--once'],capture_output=True,text=True,timeout=45,check=True)
result=dict(status='GPU6_COMPLETION_FRONTDOORS_VERIFIED',at=time.time(),installed=installed,
 code=P['code'],commit=P['commit'],completion_request=P['request'],checks=checks,actual_once_output=once.stdout,
 old_sealed_releases_changed=False,training_rerun=False,evaluation_rerun=False,signals_sent=False)
exact(root/'installation.json',(json.dumps(result,ensure_ascii=False,indent=2)+'\n').encode())
print(json.dumps(result,ensure_ascii=False))
'''

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--completion-receipt',type=Path,required=True)
 parser.add_argument('--ssh-helper-dir',type=Path,required=True)
 parser.add_argument('--viewer-deployment',type=Path,help='Separate fixed viewer checkout; completed producer remains unchanged')
 parser.add_argument('--previous-installation',type=Path,help='Exact inspected prior frontdoor installation receipt')
 args=parser.parse_args();helper=args.ssh_helper_dir.resolve(strict=True)
 receipt=json.loads(args.completion_receipt.read_text(encoding='utf8'))
 assert receipt['status']=='GPU6_EXISTING_FULL250_BEST26_COMPLETION_VERIFIED'
 payload={key:receipt[key] for key in ('code','commit','code_files','request')}
 if args.viewer_deployment:
  observer=json.loads(args.viewer_deployment.read_text(encoding='utf8'))
  assert observer['status']=='GPU6_EXISTING_COMPLETION_CODE_VERIFIED'
  payload.update({key:observer[key] for key in ('code','commit','code_files')})
 names=('view_gpu6.py','view_gpu6_current.py','view_gpu6_auto.py','view_training.py')
 if args.previous_installation:
  previous=json.loads(args.previous_installation.read_text(encoding='utf8'))
  assert previous['status']=='GPU6_COMPLETION_FRONTDOORS_VERIFIED'
  payload['previous_frontdoor_hashes']={name:previous['installed'][name]['sha256'] for name in names}
 else:
  payload['previous_frontdoor_hashes']={name:'c4e73261722dbd7f4fae9a080bfce0d8c745ae56437670299fe5572f0837dd08'
   if name=='view_training.py' else '7c98147e7cc03631fdf53033a8faa1b88397c5da657695551ec2aab825ab8aba' for name in names}
 payload['python']='/home/aicompetition06/.conda/envs/nnunet/bin/python3.10'
 script='PAYLOAD='+repr(payload)+'\n'+REMOTE
 prepared=helper/('GPU6_completion_frontdoors_'+payload['commit'][:12]+'_PREPARED.py')
 if prepared.exists():assert prepared.read_text(encoding='utf8')==script
 else:prepared.write_text(script,encoding='utf8')
 spec=importlib.util.spec_from_file_location('remote_once',helper/'remote_once.py')
 remote=importlib.util.module_from_spec(spec);spec.loader.exec_module(remote)
 result=json.loads(remote.run(script,180))
 saved=helper/('GPU6_COMPLETION_FRONTDOORS_VERIFIED_'+str(time.time_ns())+'.json')
 saved.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
 print(saved);print(json.dumps(result['checks'],ensure_ascii=False));print(result['actual_once_output'])

if __name__=='__main__':main()
