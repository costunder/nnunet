"""Release only the two verified root-created CPU supervisors; preserve CP."""
import argparse,importlib.util,json,time
from pathlib import Path

REMOTE=r'''
import hashlib,json,os,time
from pathlib import Path
import psutil
P=PAYLOAD
assert os.getuid()==41833
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def exact(w):
 try:
  p=psutil.Process(w['pid'])
  assert p.create_time()==w['create_time'] and p.cmdline()==w['command'] and p.uids().real==41833
  assert p.status()!=psutil.STATUS_ZOMBIE
  return p
 except psutil.NoSuchProcess:return None
def create(p,v):
 with Path(p).open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
events=[]
for row in P['arms']:
 gpu=row['GPU'];root=Path(row['recovery_root']);request=read(root/'request.json')
 assert sha(root/'request.json')==row['request_sha256']
 reg=read(root/'registration.json');new=exact(reg);assert new is not None
 state=read(root/'status.json');assert state['status'] in ('WAITING_FOR_CONTROLLER_HANDOFF','WAITING_FOR_EXISTING_SCORING'),state
 oldw=request['predecessor_actor'];assert oldw==row['predecessor']
 original=Path(request['original_request']['path']);oldroot=original.parent
 witness=request['original_request'];s=original.stat()
 assert sha(original)==witness['sha256'] and [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]==witness['stat']
 oldreg=read(oldroot/'registration.json')
 assert (oldreg['pid'],oldreg['create_time'],oldreg['command'])==(oldw['pid'],oldw['create_time'],oldw['command'])
 old=exact(oldw);assert old is not None,'Unknown prior release: inspect exact own handoff first'
 oldstate=read(oldroot/'status.json')
 cp=exact(request['scoring_actor']);assert cp is not None,'Existing scoring release needs separate current preflight'
 preserved={}
 for name in ('request.json','registration.json','lease.json','own_early_stop_best_pin.json'):
  p=oldroot/name
  if p.exists():
   info=p.stat();preserved[str(p)]=(sha(p),[info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns,info.st_ctime_ns])
 if gpu==4:
  assert oldstate['status']=='WAITING_FOR_FULL642_SCORING' and oldstate.get('stage') is None and oldstate.get('child_pid') is None
  assert not old.children(recursive=True)
 else:
  assert oldstate['status']=='RUNNING' and oldstate['stage']=='score'
  assert oldstate['child_pid']==cp.pid and oldstate['child_create_time']==cp.create_time() and oldstate['command']==cp.cmdline()
  assert cp.ppid()==old.pid and [p.pid for p in old.children(recursive=True)]==[cp.pid]
 assert old.pid not in (os.getpid(),os.getppid(),new.pid,cp.pid)
 assert old.cmdline()==oldw['command'] and old.create_time()==oldw['create_time']
 old.terminate()
 old.wait(timeout=10)
 assert exact(oldw) is None and exact(request['scoring_actor']) is not None
 for filename,(digest,identity) in preserved.items():
  p=Path(filename);info=p.stat();assert sha(p)==digest and [info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns,info.st_ctime_ns]==identity
 event=dict(GPU=gpu,old_supervisor=oldw,signal='SIGTERM_EXACT_OWN_CPU_SUPERVISOR_ONLY',
  scoring_actor_preserved=request['scoring_actor'],old_immutable_metadata_unchanged=list(preserved),at=time.time(),
  CP_restarted=False,SSH_or_login_process_signaled=False)
 create(root/'operator_handoff.json',event);events.append(event)
print(json.dumps(dict(status='GPU45_EXACT_OWN_CPU_SUPERVISOR_HANDOFF_VERIFIED',events=events,CP_restarted=False),ensure_ascii=False),flush=True)
'''

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--registration',type=Path,required=True);parser.add_argument('--ssh-helper-dir',type=Path,required=True)
 args=parser.parse_args();payload=json.loads(args.registration.read_text(encoding='utf8'));helper=args.ssh_helper_dir.resolve(strict=True)
 script='PAYLOAD='+repr(payload)+'\n'+REMOTE
 prepared=helper/('handoff_GPU45_native_prestart_'+payload['commit'][:12]+'_PREPARED.py')
 if prepared.exists():assert prepared.read_text(encoding='utf8')==script
 else:prepared.write_text(script,encoding='utf8')
 spec=importlib.util.spec_from_file_location('remote_once',helper/'remote_once.py');remote=importlib.util.module_from_spec(spec);spec.loader.exec_module(remote)
 result=json.loads(remote.run(script,90))
 receipt=helper/('GPU45_NATIVE_PRESTART_HANDOFF_VERIFIED_'+str(time.time_ns())+'.json')
 receipt.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
 print(receipt);print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
