"""Release only two owned, registered, childless superseded CPU waiters."""
import json,time
from remote_once import run,HERE

REMOTE=r'''
import hashlib,json,os,time
from pathlib import Path
import psutil
P=PAYLOAD;assert os.getuid()==41833
root=Path(P['controller_root']);release=root.parent
request=json.loads((root/'request.json').read_text());state=json.loads((root/'status.json').read_text())
lease=json.loads((root/'lease.json').read_text())
assert hashlib.sha256((root/'request.json').read_bytes()).hexdigest()==P['request']['sha256']==lease['request_sha256']
new=psutil.Process(lease['pid'])
assert new.create_time()==lease['create_time'] and new.cmdline()==lease['command'] and new.uids().real==41833
assert state['status']=='WAITING_FOR_EARLY_STOP_RELEASE' and state['child_pid'] is None
expected=P['old_waiter_release_actors'];assert [r['pid'] for r in expected]==[2001549,2001560]
checks=[]
for witness in request['protected_files']:
 path=Path(witness['path']);s=path.stat()
 assert [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]==witness['stat']
 assert hashlib.sha256(path.read_bytes()).hexdigest()==witness['sha256']
for row in expected:
 p=psutil.Process(row['pid'])
 assert p.create_time()==row['create_time'] and p.cmdline()==row['command'] and p.uids().real==row['UID']==41833
 assert p.children(recursive=True)==[],'A CPU waiter has a child; refuse this release'
 filename='scoring_wait_status.json' if row['pid']==2001560 else 'status.json'
 parent=Path(row['command'][-1]).parent
 status=json.loads((parent/filename).read_text())
 assert status['status']=='WAITING_FOR_FULL_GNN' and status.get('child_pid') is None
 checks.append(dict(actor=row,status_before=status,signal='SIGTERM',reason='User-authorized e27 early-stop successor is registered; full40-only childless CPU waiter is superseded'))
with (release/'superseded_waiter_release_intent.json').open('x') as stream:json.dump(dict(at=time.time(),exact_targets=checks),stream,indent=2)
for row in expected:
 p=psutil.Process(row['pid'])
 assert p.create_time()==row['create_time'] and p.cmdline()==row['command'] and p.uids().real==41833 and p.children(recursive=True)==[]
 p.terminate()
 try:p.wait(timeout=10)
 except psutil.TimeoutExpired:raise RuntimeError('Exact CPU waiter did not stop; no force/broad signal used: '+str(p.pid))
for witness in request['protected_files']:
 path=Path(witness['path']);s=path.stat()
 assert [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]==witness['stat']
 assert hashlib.sha256(path.read_bytes()).hexdigest()==witness['sha256']
result=dict(status='EXACT_GPU5_SUPERSEDED_CPU_WAITERS_RELEASED',at=time.time(),targets=checks,
 only_pids_signaled=[r['pid'] for r in expected],GNN_actor_signaled=False,GPU46_actors_signaled=False,
 existing_requests_registrations_claims_preserved=True,original_results_written=False)
with (release/'superseded_waiter_release_verified.json').open('x') as stream:json.dump(result,stream,indent=2)
print(json.dumps(result),flush=True)
'''

def main():
 data=json.loads((HERE/'GPU5_EARLY_STOP_CHAIN_REGISTERED_1791741440255409200.json').read_text(encoding='utf8'))
 payload={k:data[k] for k in ('controller_root','request','old_waiter_release_actors')}
 result=json.loads(run('PAYLOAD='+repr(payload)+'\n'+REMOTE,60))
 receipt=HERE/('GPU5_EXACT_CPU_WAITERS_RELEASED_'+str(time.time_ns())+'.json')
 receipt.write_text(json.dumps(result,indent=2),encoding='utf8')
 print(receipt);print(json.dumps({k:result[k] for k in ('status','only_pids_signaled','GNN_actor_signaled','GPU46_actors_signaled')}))

if __name__=='__main__':main()
