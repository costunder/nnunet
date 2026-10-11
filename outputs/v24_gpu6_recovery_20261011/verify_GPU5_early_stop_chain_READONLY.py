import hashlib,json,os,subprocess,time
from pathlib import Path
import psutil
base=Path('/var/tmp/aicompetition06_v24_20261010')
root=base/'GPU5_EARLY_STOP_E27_20261012_1791740800000000000';assert os.getuid()==41833
def read(p):return json.loads(Path(p).read_text())
def actor(reg):
 try:
  p=psutil.Process(reg['pid']);assert p.create_time()==reg['create_time'] and p.cmdline()==reg['command'] and p.uids().real==41833
  return dict(pid=p.pid,birth=p.create_time(),status=p.status(),RSS_GiB=p.memory_info().rss/2**30,CPU_affinity=p.cpu_affinity())
 except psutil.NoSuchProcess:return None
request=read(root/'controller/request.json');state=read(root/'controller/status.json')
protected=[]
for proof in request['protected_files']:
 p=Path(proof['path']);s=p.stat()
 assert hashlib.sha256(p.read_bytes()).hexdigest()==proof['sha256']
 assert [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]==proof['stat']
 protected.append(proof['path'])
controller=actor(read(root/'controller/registration.json'));guardian=actor(read(root/'pause_registration.json'))
gnu=actor(read(Path(request['source_output']).parent/'direct_training_registration.json'))
old=[dict(pid=r['pid'],exists=psutil.pid_exists(r['pid'])) for r in request['release_actors'] if r['pid']!=1822677]
result=subprocess.run(['/home/aicompetition06/.conda/envs/nnunet/bin/python3.10','-B',str(base/'view_training.py'),'--gpu','5','--once','--json'],
 capture_output=True,text=True,timeout=40)
assert result.returncode==0,result.stderr
snapshot=json.loads(result.stdout);assert snapshot['snapshot_fresh'] is True and snapshot['GPU']==5
gpu=subprocess.check_output(['nvidia-smi','--id=5','--query-gpu=index,name,uuid,memory.total,memory.used,utilization.gpu,mig.mode.current','--format=csv,noheader,nounits'],text=True,timeout=10).strip()
print(json.dumps(dict(status='GPU5_EARLY_STOP_AUTOMATIC_CHAIN_CURRENT_VERIFIED',at=time.time(),controller=controller,
 guardian=guardian,guardian_state=read(root/'pause_guardian/status.json'),original_GNN=gnu,old_waiters=old,
 chain_state=state,snapshot=snapshot,GPU5_resource=gpu,protected_files_verified=protected,
 signals_sent=False,files_written=False),ensure_ascii=False),flush=True)
