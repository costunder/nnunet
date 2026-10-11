"""Deploy the audited epoch27 observer into a fresh owned namespace."""
import base64,hashlib,json,time
from pathlib import Path
from remote_once import run,HERE

REMOTE=r'''
import base64,hashlib,json,os,socket,subprocess,time
from pathlib import Path
import psutil
P=PAYLOAD
base=Path('/var/tmp/aicompetition06_v24_20261010')
root=base/'GPU5_EARLY_STOP_E27_20261012_1791740800000000000'
output=base/'GPU45_interrupted_exact_CPU_prepared_20261011_1791705559311834000/gpu5/gnn'
source=base/'code_b5581f624448'
assert os.getuid()==41833 and socket.gethostname().split('.')[0]=='ece-a6gpu6'
def proof(p):
 p=Path(p);s=p.stat();raw=p.read_bytes();again=p.stat()
 expected=lambda x:[x.st_dev,x.st_ino,x.st_size,x.st_mtime_ns,x.st_ctime_ns]
 assert not p.is_symlink() and p.stat().st_uid==41833 and expected(s)==expected(again)
 return dict(path=str(p),sha256=hashlib.sha256(raw).hexdigest(),stat=expected(s))
def create(p,value):
 with p.open('xb') as f:f.write(value);f.flush();os.fsync(f.fileno())
 return proof(p)
def publish(p,value):return create(p,(json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode('utf8'))
training=output/'training';reg=json.loads((output.parent/'direct_training_registration.json').read_text())
actor=psutil.Process(reg['pid']);worker=dict(pid=actor.pid,create_time=actor.create_time(),command=actor.cmdline(),uid=actor.uids().real)
assert worker==P['worker'] and worker['uid']==41833
assert worker['pid']==1822677 and worker['create_time']==1791705670.42
rows=[json.loads(line) for line in (training/'curve.jsonl').read_text().splitlines() if line.strip()]
assert [r['epoch'] for r in rows]==list(range(1,27)),'Authorized epoch27 boundary already passed; no sentinel written'
assert not (training/'STOP_AFTER_BATCH').exists()
assert not root.exists(),'Preserve previous deployment; inspect its outcome instead of replacing it'
raw=base64.b64decode(P['guardian_base64'],validate=True)
assert hashlib.sha256(raw).hexdigest()==P['guardian_sha256']
compile(raw,str(root/'pause_after_epoch.py'),'exec')
for name,expected in P['source_files'].items():assert proof(source/name)==expected
root.mkdir(mode=0o700)
script=create(root/'pause_after_epoch.py',raw)
authorization=dict(format='v24_user_authorized_GNN_epoch_boundary_stop_request_v1',GPU_arm=5,
 source_output=str(output),source_code=str(source),target_epoch=27,original_planned_epochs=40,user_authorized=True,
 stop_policy='after_current_epoch_training_and_full128_validation',created_at=time.time(),
 reason='Human explicitly selected: 현재 epoch 완료 후 조기 종료하고 BEST로 전환',
 original_actor={k:worker[k] for k in ('pid','create_time','command')})
auth=publish(root/'authorization.json',authorization)
request=dict(format='v24_gpu5_pause_after_current_epoch_guardian_request_v1',UID=41833,target_epoch=27,
 expected_updates=459,output_root=str(output),worker=worker,guardian_root=str(root/'pause_guardian'),
 source_code=str(source),source_files=P['source_files'],training_identity=proof(training/'training_identity.json'),
 execution_contract=proof(training/'execution_contract.json'),initial_curve=proof(training/'curve.jsonl'),
 initial_history_epochs=list(range(1,27)),authorization=auth)
req=publish(root/'pause_request.json',request)
command=['/home/aicompetition06/.conda/envs/nnunet/bin/python3.10','-B','-u',str(root/'pause_after_epoch.py'),
 '--request',req['path'],'--request-sha256',req['sha256']]
publish(root/'pause_launch_intent.json',dict(command=command,request=req,guardian=script,at=time.time()))
with (root/'pause_console.log').open('xb') as log:
 child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
  cwd=root,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES=''))
 observer=psutil.Process(child.pid);observer.cpu_affinity([42,43,44,45])
 reg=publish(root/'pause_registration.json',dict(pid=child.pid,create_time=observer.create_time(),command=command,uid=41833,
  source_actor=worker,request=req,guardian=script,CPU_affinity=observer.cpu_affinity(),signals_sent=False))
 time.sleep(.5)
 state_path=root/'pause_guardian/status.json'
 if child.poll() is not None or not state_path.exists():
  raise RuntimeError('Observer did not arm; preserve fresh logs: '+(root/'pause_console.log').read_text())
 state=json.loads(state_path.read_text())
 assert state['status']=='WAITING_COMPLETE_EPOCH_CURVE' and state['verified'] is False
 result=dict(status='GPU5_EPOCH27_PAUSE_OBSERVER_ARMED',root=str(root),authorization=auth,request=req,
  registration=reg,guardian_state=state,signals_sent=False,at=time.time())
 publish(root/'pause_arm_verified.json',result)
 print(json.dumps(result,ensure_ascii=False),flush=True)
'''

def main():
 data=json.loads((HERE/'GPU5_EARLY_STOP_INPUTS_1791740412482227300.json').read_text(encoding='utf8'))
 worker={k:data['actors']['1822677'][k] for k in ('pid','create_time','command','uid')}
 raw=(HERE/'pause_gpu5_after_current_epoch_NEW.py').read_bytes()
 payload=dict(worker=worker,source_files=data['source_files'],guardian_sha256=hashlib.sha256(raw).hexdigest(),guardian_base64=base64.b64encode(raw).decode())
 script='PAYLOAD='+repr(payload)+'\n'+REMOTE
 prepared=HERE/('arm_GPU5_early_stop_guardian_'+payload['guardian_sha256'][:12]+'_PREPARED.py')
 if prepared.exists():assert prepared.read_text(encoding='utf8')==script
 else:prepared.write_text(script,encoding='utf8')
 output=run(script,60)
 receipt=HERE/('GPU5_EARLY_STOP_GUARDIAN_ARMED_'+str(time.time_ns())+'.json')
 parsed=json.loads(output);receipt.write_text(json.dumps(parsed,ensure_ascii=False,indent=2),encoding='utf8')
 print(receipt);print(json.dumps({k:parsed[k] for k in ('status','root','authorization','registration')},ensure_ascii=False))

if __name__=='__main__':main()
