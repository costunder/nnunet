"""Verify exact adoption supervisors, unchanged scorers and fixed native path."""
import argparse,importlib.util,json,time
from pathlib import Path

REMOTE=r'''
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
import psutil
P=PAYLOAD
assert os.getuid()==41833
sys.path.insert(0,P['code'])
from hiercp_v1x import v24_native_prestart_guard as guard
from hiercp_v1x import v24_scored_native_chain as chain
def exact(w):
 try:
  p=psutil.Process(w['pid']);assert p.create_time()==w['create_time'] and p.cmdline()==w['command'] and p.uids().real==41833
  return dict(alive=True,pid=p.pid,create_time=p.create_time(),command=p.cmdline(),UID=p.uids().real,CPU_affinity=p.cpu_affinity(),
   parent_pid=p.ppid(),status=p.status(),RSS_GiB=p.memory_info().rss/2**30,CPU_seconds=sum(p.cpu_times()[:2]),children=[q.pid for q in p.children(recursive=True)])
 except psutil.NoSuchProcess:return dict(alive=False)
rows=[]
for row in P['arms']:
 root=Path(row['recovery_root']);request=guard.validate_request(chain.read(root/'request.json'))
 assert chain.sha(root/'request.json')==row['request_sha256']
 cache=chain.FileProofCache();guard.verify_code(request,cache);chain.verify_code(request['resource_contract'],cache)
 for w in request['protected_files']:cache.verify(w)
 reg=chain.read(root/'registration.json');state=chain.read(root/'status.json')
 supervisor=exact(reg);assert supervisor['alive']
 scorer=exact(request['scoring_actor']);old=exact(request['predecessor_actor'])
 view=subprocess.run([request['python'],'-B',str(Path(P['code'])/'tools/watch_v24_native_prestart_guard.py'),
  '--request',str(root/'request.json'),'--renderer',str(Path(P['code'])/'tools/watch_v24_training.py'),
  '--renderer-sha',P['code_files']['tools/watch_v24_training.py'],'--once','--json'],capture_output=True,text=True,timeout=45)
 assert view.returncode==0,view.stderr
 viewer=json.loads(view.stdout);assert viewer.get('snapshot_fresh') is True and viewer['status'] not in ('READ_ERROR','READ_PENDING'),viewer
 native=Path(request['resource_contract']['chain_root'])/'native'
 rows.append(dict(GPU=row['GPU'],root=str(root),state=state,supervisor=supervisor,scorer=scorer,old_supervisor=old,
  native_exists=native.exists(),fixed_native_stages=request['stages'],old_immutable_metadata_verified=len(request['protected_files']),
  viewer=viewer,console=(root/'console.log').read_bytes()[-4096:].decode('utf8',errors='replace')))
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=10)
print(json.dumps(dict(status='GPU45_FIXED_NATIVE_PRESTART_VERIFIED_READONLY',at=time.time(),arms=rows,GPUs=gpu,
 full_native_training_complete=False,full_BEST26_evaluation_complete=False,signals_sent=False),ensure_ascii=False),flush=True)
'''

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--registration',type=Path,required=True);parser.add_argument('--ssh-helper-dir',type=Path,required=True)
 args=parser.parse_args();payload=json.loads(args.registration.read_text(encoding='utf8'));helper=args.ssh_helper_dir.resolve(strict=True)
 spec=importlib.util.spec_from_file_location('remote_once',helper/'remote_once.py');remote=importlib.util.module_from_spec(spec);spec.loader.exec_module(remote)
 result=json.loads(remote.run('PAYLOAD='+repr(payload)+'\n'+REMOTE,120))
 receipt=helper/('GPU45_NATIVE_PRESTART_ACTUAL_VERIFIED_'+str(time.time_ns())+'.json')
 receipt.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
 print(receipt);print(json.dumps(dict(status=result['status'],arms=[dict(GPU=r['GPU'],root=r['root'],state=r['state']['status'],
  supervisor=r['supervisor'],scorer=r['scorer'],old_supervisor=r['old_supervisor'],native_exists=r['native_exists'],
  source_done=(r['viewer'].get('scoring') or {}).get('source_done'),CP_done=(r['viewer'].get('scoring') or {}).get('CP_done'),
  console=r['console']) for r in result['arms']],GPUs=result['GPUs']),ensure_ascii=False))
if __name__=='__main__':main()
