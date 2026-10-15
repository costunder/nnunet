"""Two approved, independent checkpoint continuations on physical GPU5/6.

No CUDA, parent/session signalling, experiment reduction or in-place registry
edit. A future handoff uses a fresh seven-arm cohort and retains original data.
"""
from __future__ import annotations
import argparse,copy,gc,hashlib,importlib.util,json,os,shutil,signal,socket,subprocess,sys,time,traceback
from pathlib import Path
import psutil

UID=41833
PY='/home/aicompetition06/.conda/envs/nnunet/bin/python3.10'
BASE=Path('/dev/shm/aicompetition06_basic_cp_ablation_20261013')
CACHE=Path('/var/tmp/aicompetition06_basic_cp_seven_resume_1791934200100000000')
PUBLIC=BASE/'view_basic7_20261014.py'
ROUTES=[dict(arm='multi5',waiting_physical_GPU=0,target_physical_GPU=5,upstream_arm='candidates2048'),dict(arm='candidates1024',waiting_physical_GPU=4,target_physical_GPU=6,upstream_arm='multi7')]

def require(ok,message):
    if not ok:raise ValueError(message)
def read(p):return json.loads(Path(p).read_bytes())
def exact_file(p,directory=False):
    p=Path(p);s=p.stat();require(p.is_absolute()and p.resolve()==p and not any(x.is_symlink()for x in(p,*p.parents))and s.st_uid==UID,'Exact own nonsymlink path required')
    require(p.is_dir()if directory else p.is_file(),'Own regular evidence type changed');return p
def sha(p):
    h=hashlib.sha256()
    with exact_file(p).open('rb')as f:
        before=os.fstat(f.fileno())
        for b in iter(lambda:f.read(8*2**20),b''):h.update(b)
        after=os.fstat(f.fileno())
    require((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns),'Held evidence changed during SHA');return h.hexdigest()
def pin(ref):require(set(ref)=={'path','sha256'}and sha(ref['path'])==ref['sha256'],'Pinned queue evidence changed');return Path(ref['path'])
def publish(p,v):
    p=Path(p);temp=p.with_name('.'+p.name+'.own_'+str(time.time_ns()))
    with temp.open('x')as f:json.dump(v,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
    os.replace(temp,p)
def seal(p,v):
    with Path(p).open('x')as f:json.dump(v,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
def identity(p):return dict(pid=p.pid,create_time=p.create_time(),UID=p.uids().real,command=p.cmdline())
def norm(a):return dict(pid=a['pid'],create_time=a.get('create_time',a.get('birth')),UID=a['UID'],command=a['command'])
def gone(a):
    a=norm(a)
    try:p=psutil.Process(a['pid'])
    except psutil.NoSuchProcess:return True
    if p.create_time()!=a['create_time']:return True
    require(p.uids().real==UID,'Captured original actor UID changed')
    return p.status()in(psutil.STATUS_ZOMBIE,psutil.STATUS_DEAD)
def exact_actor(a):
    p=psutil.Process(a['pid']);require(not gone(a)and identity(p)==norm(a)and a['UID']==UID,'Exact own PID/birth/UID/argv changed');return p
def collect(v,actors):
    if isinstance(v,dict):
        birth=v.get('create_time',v.get('birth'))
        if type(v.get('pid'))is int and type(birth)in(int,float)and v.get('UID')==UID and v.get('command'):
            a=norm(v);key=(a['pid'],a['create_time']);require(key not in actors or actors[key]==a,'Conflicting captured identity');actors[key]=a
        for child in v.values():collect(child,actors)
    elif isinstance(v,list):
        for child in v:collect(child,actors)
def capture(p,actors):
    for child in p.children(recursive=True):
        try:
            if child.status()in(psutil.STATUS_ZOMBIE,psutil.STATUS_DEAD):continue
            a=identity(child);require(a['UID']==UID and a['command'],'Unexpected live empty/foreign child');collect(a,actors)
        except psutil.NoSuchProcess:continue
def fold(n):return Path(n['native_root'])/'nnUNet_results'/n['baseline']['dataset_name']/(n['trainer']+'__'+n['baseline']['plans_name']+'__3d_fullres')/'fold_0'
def complete(s):
    return(s.get('status')=='COMPLETE'and s.get('stage')=='complete'and type(s.get('completed_epochs'))is int and s['completed_epochs']==250 and type(s.get('evaluated_cases'))is int and s['evaluated_cases']==26 and s.get('full_native_training')is True and s.get('full_native_evaluation')is True and s.get('historical_metrics_complete')is True and bool(s.get('reports'))and bool(s.get('archive')))
def pending_routes(manifest):
    arms={int(g):r['arm']for g,r in manifest['registration'].items()}
    return[r for r in ROUTES if arms[r['waiting_physical_GPU']]==r['arm']]
def ready_routes(manifest):
    ready=[]
    for rule in pending_routes(manifest):
        row=manifest['registration'][str(rule['target_physical_GPU'])];require(row['arm']==rule['upstream_arm'],'Actual queue upstream arm changed')
        root=Path(row['root']);s=read(root/'status.json')
        if not complete(s):continue
        if not gone(row['controller_identity']):continue
        if any(not gone(read(p))for p in root.glob('*_actor.json')):continue
        ready.append(rule)
    return ready
def confirm_upstream_evidence(manifest,routes):
    for rule in routes:
        row=manifest['registration'][str(rule['target_physical_GPU'])];root=Path(row['root']);s=read(root/'status.json');require(complete(s),'Upstream full completion changed')
        for ref in s['reports'].values():pin(ref)
        archive=read(pin(s['archive']));require(archive.get('complete')is True and archive.get('independent_copies')is True and archive.get('full_native_training_epochs')==250 and archive.get('full_native_BEST_validation_cases')==26,'Actual upstream full archive required')
        n=read(root/'native.json');proof=read(root/'training_and_BEST26_complete.json');require(proof['epochs']==250 and proof['checkpoint']=='checkpoint_best.pth'and proof['evaluation_checkpoint_is_final']is False and len(set(proof['validation_cases']))==26,'Own BEST26 completion required')
        require(sha(fold(n)/'checkpoint_best.pth')==proof['best_sha256']and sha(fold(n)/'checkpoint_final.pth')==proof['final_training_witness_sha256'],'Original upstream BEST/final bytes changed')
def placements(manifest,routes):
    arms={int(g):r['arm']for g,r in manifest['registration'].items()}
    for r in routes:arms[r['waiting_physical_GPU']],arms[r['target_physical_GPU']]=arms[r['target_physical_GPU']],arms[r['waiting_physical_GPU']]
    return arms
def can_pause_all(manifest):
    for row in manifest['registration'].values():
        root=Path(row['root']);n=read(root/'native.json');s=read(root/'status.json')
        live=not gone(row['controller_identity'])
        if complete(s):
            if live:return False
            continue
        if not complete(s)and(fold(n)/'checkpoint_final.pth').exists():return False
        if live:
            if s.get('status')=='WAITING'and s.get('stage')in('waiting_GPU','waiting_RAM','waiting_GPU_queue_transfer'):continue
            actor_path=root/'train_actor.json'
            if s.get('status')!='RUNNING'or s.get('stage')!='train'or not actor_path.exists()or gone(read(actor_path)):return False
    return True
def publish_code(config,group):
    prior=read(pin(config['canonical_export']));source=Path(prior['code']);code=group/'code';code.mkdir(mode=0o700)
    for name,meta in prior['all_exact_files'].items():
        p=exact_file(source/name);require(p.stat().st_size==meta['size']and sha(p)==meta['sha256'],'Canonical queue code changed');dest=code/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest);require(sha(dest)==meta['sha256'],'Private code copy differs')
    export=dict(prior,code=str(code),created_unix=time.time());seal(group/'code_export.json',export);return export
def wait_gone(actors,seconds=180):
    deadline=time.monotonic()+seconds
    while time.monotonic()<deadline:
        if all(gone(a)for a in actors.values()):return
        time.sleep(.2)
    raise RuntimeError('Captured original actors still releasing; no signals may be repeated')
def pause_cohort(manifest,group,code):
    """Latest whole checkpoints are decoded before individual own SIGINTs."""
    spec=importlib.util.spec_from_file_location('_queue_whole_checkpoint_validator',code/'tools/release_basic_cp_RAM512_native_at_epoch.py');helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    import torch
    torch.set_num_threads(1)
    actors={};collect(manifest['registration'],actors);proofs={};selected=[]
    for gpu,row in manifest['registration'].items():
        root=Path(row['root']);q=read(root/'request.json');require(sha(root/'request.json')==row['request_sha256'],'Original registered request changed')
        for path in root.glob('*_actor.json'):collect(read(path),actors)
        control=row['controller_identity']
        if gone(control):continue
        parent=exact_actor(control);train_path=root/'train_actor.json';train=read(train_path)if train_path.exists()else None
        if train is not None and not gone(train):
            n=read(root/'native.json');require(not(fold(n)/'checkpoint_final.pth').exists(),'Final validation is still completing; postpone handoff')
            p=exact_actor(train);require(p.ppid()==parent.pid and train['command']==[PY,'-B','-u',str(Path(q['controller_identity']['command'][3])),'--native',str(root/'native.json'),'--stage','train'],'Only exact own native trainer may be checkpoint-paused')
            model=read(root/'actual_model.json');require(model['epochs']==250 and model['physical_batch']==model['effective_batch']==2 and len(model['actual_train_cases'])==105 and len(model['actual_validation_cases'])==26,'Full native250/B2/105/26 contract changed')
            capture(p,actors);selected.append((gpu,train,'native',root,n,model))
        else:
            state=read(root/'status.json');require(state.get('status')=='WAITING'and state.get('stage')in('waiting_GPU','waiting_RAM','waiting_GPU_queue_transfer'),'Unexpected active non-native controller phase; postpone handoff')
            capture(parent,actors);selected.append((gpu,control,'waiting',root,None,None))
    seal(group/'pause_targets_reported.json',dict(reason='Fresh common RAM512 registry for approved successor GPU transfer',selected=[dict(gpu=g,actor=a,kind=k)for g,a,k,*_ in selected],checkpoint_proofs=proofs,created_unix=time.time()))
    print(json.dumps({'event':'OWN_PAUSE_TARGETS_REPORTED_BEFORE_SIGNAL','targets':[{'gpu':g,'PID':a['pid'],'command':a['command']}for g,a,*_ in selected],'reason':'Approved queued checkpoint continuation; full250 goals retained'}),flush=True)
    for gpu,a,kind,root,n,model in selected:
        if kind=='native':
            deadline=time.monotonic()+1800;window=None
            while time.monotonic()<deadline:
                if gone(a):
                    require(gone(manifest['registration'][gpu]['controller_identity'])or complete(read(root/'status.json')),'Native actor changed before its controller completed; inspect queue handoff')
                    break
                p=exact_actor(a)
                if(fold(n)/'checkpoint_final.pth').exists():
                    time.sleep(1);continue  # Never interrupt a real final BEST evaluation.
                candidate=helper.training_window(root,fold(n))
                if candidate is not None:
                    latest=helper.checkpoint_info(fold(n)/'checkpoint_latest.pth',n,model['parameters']);best=helper.checkpoint_info(fold(n)/'checkpoint_best.pth',n,model['parameters']);fresh=helper.training_window(root,fold(n))
                    if fresh is not None and latest['completed_epoch']==fresh['epoch']==candidate['epoch']and latest['source_inode']==fresh['latest_inode']and best['source_inode']==fresh['BEST_inode']:
                        require(0<best['completed_epoch']<=latest['completed_epoch']<250,'Whole original native checkpoint state changed');proofs[gpu]=dict(LATEST=latest,BEST=best,early_next_epoch_window=fresh);window=fresh;break
                time.sleep(.25)
            if gone(a):continue
            require(window is not None,'No verified saved-epoch window within1800seconds; no signal for this actor')
            p=exact_actor(a);capture(p,actors);fresh=helper.training_window(root,fold(n));require(fresh is not None and fresh['epoch']==window['epoch']and fresh['latest_inode']==window['latest_inode']and fresh['BEST_inode']==window['BEST_inode'],'Saved-epoch signal window changed; no signal')
        else:
            p=exact_actor(a);capture(p,actors)
            for c in p.children(recursive=True):require(c.uids().real==UID and c.cmdline()[0]in('nvidia-smi','/opt/pbs/bin/qstat'),'Waiting controller acquired a real native child; no signal')
        seal(group/('single_signal_'+gpu+'.intent.json'),dict(actor=a,signal='SIGINT',reason='Approved whole-checkpoint cohort transfer',checkpoint_proof=proofs.get(gpu),time=time.time()));p.send_signal(signal.SIGINT)
    wait_gone(actors)
    oldgroup=Path(manifest['group']);cache=read(oldgroup/'checkpoint_preserver_actor.json');collect(cache,actors)
    if not gone(cache):
        p=exact_actor(cache);require(not p.children(recursive=True),'Unexpected cache helper child');seal(group/'single_cache_signal.intent.json',dict(actor=cache,signal='SIGINT',reason='Adopt original cache for approved queue continuation',time=time.time()));p.send_signal(signal.SIGINT)
    wait_gone(actors)
    ledger=read(oldgroup/'RAM512_allocation_ledger.json');collect(ledger.get('claims',{}),actors);wait_gone(actors)
    receipt=dict(format='basic_cp_RAM512_all_previous_actor_release_v1',all_old_pipeline_actors_terminal=True,actors=list(actors.values()),actual_actor_observations=[dict(actor=a,released=gone(a))for a in actors.values()],whole_checkpoint_proofs=proofs,only_reported_own_actors_signaled=True,original_results_written=False,time=time.time());seal(group/'all_old_pipeline_release.json',receipt)
    del helper;gc.collect();return group/'all_old_pipeline_release.json'
def run_checked(command,*,env,stdin=None,output):
    with Path(output).open('xb')as f:
        result=subprocess.run(command,input=stdin.encode()if stdin is not None else None,stdout=f,stderr=subprocess.STDOUT,env=env)
    require(result.returncode==0,'Queue handoff child failed; inspect '+str(output))
def handoff(config,manifest,routes,group):
    confirm_upstream_evidence(manifest,routes);require(can_pause_all(manifest),'An existing final evaluation/archive is still completing')
    group.mkdir(mode=0o700);export=publish_code(config,group);code=group/'code'
    placement=dict(format='basic_cp_approved_physical_GPU_placement_v1',operation='queue_multi5_after_candidates2048_and_candidates1024_after_multi7',physical_GPU_arms={str(g):a for g,a in placements(manifest,routes).items()});seal(group/'approved_GPU_placement.json',placement)
    plan=read(pin(config['queue_plan']));require(plan==dict(format='basic_cp_approved_waiting_GPU_dependencies_v1',queues=ROUTES),'Exact two approved dependencies required');seal(group/'queue_plan.json',plan)
    release=pause_cohort(manifest,group,code);old=Path(manifest['group']);env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(code))
    args=['--execute','--new-group',str(group),'--current-manifest',str(old/'launch.json'),'--current-manifest-sha256',sha(old/'launch.json'),'--release-proof',str(release),'--release-proof-sha256',sha(release),'--full-source-proof',config['full_source_proof']['path'],'--full-source-proof-sha256',config['full_source_proof']['sha256'],'--code-export',str(group/'code_export.json'),'--code-export-sha256',sha(group/'code_export.json'),'--placement-receipt',str(group/'approved_GPU_placement.json'),'--placement-receipt-sha256',sha(group/'approved_GPU_placement.json'),'--queue-plan',str(group/'queue_plan.json'),'--queue-plan-sha256',sha(group/'queue_plan.json')]
    wrapper='import sys,importlib.util\nfrom pathlib import Path\ncode=Path('+repr(str(code))+')\nsys.path.insert(0,str(code))\ndef load(name,path):\n s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m\nb=load("_fresh_queue_builder",code/"tools/recover_basic_cp_RAM512_successor_cohort.py")\na=load("_fresh_log_adapter",code/"tools/basic_cp_RAM512_completed_log_adapter.py")\np=code/"tools/basic_cp_RAM512_completed_log_adapter.py"\nb.main=a.private_builder_main(b,adapter_path=p,adapter_sha256='+repr(export['code_sha256']['tools/basic_cp_RAM512_completed_log_adapter.py'])+')\nsys.argv=[str(b.__file__),*'+repr(args)+']\nb.main()\n'
    run_checked([PY,'-B','-'],env=env,stdin=wrapper,output=group/'queue_builder.log')
    new=read(group/'launch.json');old_helper=manifest['code_sha256']['tools/persist_basic_cp_seven.py'];new_helper=export['code_sha256']['tools/persist_basic_cp_seven.py']
    command=[PY,'-B','-u',str(code/'tools/adopt_basic_cp_RAM512_cache_transfer.py'),'--current-manifest',str(old/'launch.json'),'--new-manifest',str(group/'launch.json'),'--new-code-export',str(group/'code_export.json'),'--old-manifest-sha256',sha(old/'launch.json'),'--old-helper-actor-sha256',sha(old/'checkpoint_preserver_actor.json'),'--old-helper-sha256',old_helper,'--new-manifest-sha256',sha(group/'launch.json'),'--new-code-export-sha256',sha(group/'code_export.json'),'--new-helper-sha256',new_helper,'--new-commit',new['commit']]
    run_checked(command,env=env,output=group/'queue_cache_adoption.log')
    post='import sys,importlib.util\nfrom pathlib import Path\np=Path('+repr(str(code/'tools/hold_basic_cp_RAM512_transfer.py'))+')\nsys.path.insert(0,str(p.parents[1]))\ns=importlib.util.spec_from_file_location("_fresh_queue_post",p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)\nm.postlaunch(Path('+repr(str(group/'launch.json'))+'),'+repr(sha(group/'launch.json'))+',Path('+repr(str(CACHE))+'),'+repr(sha(PUBLIC))+')\n'
    run_checked([PY,'-B','-'],env=env,stdin=post,output=group/'queue_cache_view.log')
    seal(group/'successor_handoff_complete.json',dict(routes=routes,actual_new_manifest_sha256=sha(group/'launch.json'),full_training_complete_claimed=False,UI_version=2,created_unix=time.time()));return new
def hold_existing(groups):
    while True:
        actors={}
        for g in groups:
            if(g/'launch.json').exists():
                m=read(g/'launch.json');collect(m['registration'],actors)
                for row in m['registration'].values():
                    for p in Path(row['root']).glob('*_actor.json'):collect(read(p),actors)
            if(g/'checkpoint_preserver_actor.json').exists():collect(read(g/'checkpoint_preserver_actor.json'),actors)
            for root in g.glob('GPU[0-6]_*'):
                if(root/'request.json').exists():collect(read(root/'request.json')['controller_identity'],actors)
                for p in root.glob('*_actor.json'):collect(read(p),actors)
        if all(gone(a)for a in actors.values()):return
        time.sleep(10)
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);parser.add_argument('--config-sha256',required=True);args=parser.parse_args()
    os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1');require(os.getuid()==UID and socket.gethostname().split('.')[0]=='ece-a6gpu6'and sys.executable==PY,'Exact own host and nnunet environment required')
    config=read(pin(dict(path=str(args.config),sha256=args.config_sha256)));anchor=exact_file(args.config.parent,True);require(anchor.parent==CACHE and anchor.name=='queue_multi5_on5_candidates1024_on6','Exact own durable queue namespace required')
    import fcntl
    lock=os.open(anchor/'watcher.lock',os.O_RDWR|os.O_CREAT,0o600);fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    process=psutil.Process();require({80,81}<=set(process.cpu_affinity()),'Own shared CPU80/81 available required');process.cpu_affinity([80,81]);state_path=anchor/'status.json';actor=identity(process);seal(anchor/('watcher_actor_'+str(process.pid)+'.json'),dict(actor,CUDA_visible_devices='',CPU=[80,81],RAM_limit_GiB=4))
    current=read(pin(config['initial_manifest']));groups=[Path(current['group'])]
    if state_path.exists():
        prior=read(state_path);require(prior.get('config_sha256')==args.config_sha256 and prior.get('phase')in('WAITING','QUEUE_SAVED_ALLOCATION_ENDED'),'Unknown prior queue handoff outcome needs inspection')
        current=read(pin(prior['current_manifest']));groups.append(Path(current['group']))
    for ref in(config['canonical_export'],config['queue_plan'],config['full_source_proof'],config['selected_pause_proof']):pin(ref)
    state=dict(format='basic_cp_RAM512_successor_queue_status_v1',config_sha256=args.config_sha256,actor=actor,routes=ROUTES,phase='WAITING',current_manifest=dict(path=str(Path(current['group'])/'launch.json'),sha256=sha(Path(current['group'])/'launch.json')),GPU0_GPU4_autostart_disabled=True,current_training_interrupted_at_queue_registration=False)
    publish(state_path,state)
    try:
        while pending_routes(current):
            require(psutil.Process().memory_info().rss<=4*2**30,'Queue watcher exceeded its CPU-only4GiB envelope')
            if time.time()>=current['host_lifetime']['expires_at_unix']-180:
                publish(state_path,dict(state,phase='QUEUE_SAVED_ALLOCATION_ENDED',time=time.time(),reason='Preserved dependencies; no new training beyond admitted host lifetime'));hold_existing(groups);return
            readiness=ready_routes(current)
            if readiness and can_pause_all(current):
                newgroup=BASE/('Basic_CP_RAM512_successors_'+str(time.time_ns()));groups.append(newgroup);publish(state_path,dict(state,phase='HANDOFF',target_group=str(newgroup),ready_routes=readiness,time=time.time(),replay_forbidden=True))
                current=handoff(config,current,readiness,newgroup);state.update(current_manifest=dict(path=str(newgroup/'launch.json'),sha256=sha(newgroup/'launch.json')),phase='WAITING',time=time.time());publish(state_path,state)
            else:
                observations={}
                for route in pending_routes(current):
                    row=current['registration'][str(route['target_physical_GPU'])];s=read(Path(row['root'])/'status.json');observations[route['arm']]={k:s.get(k)for k in('status','stage','completed_epochs','evaluated_cases','error')}
                publish(state_path,dict(state,phase='WAITING',upstream=observations,time=time.time()))
            time.sleep(30)
        publish(state_path,dict(state,phase='ALL_SUCCESSORS_DISPATCHED',time=time.time(),full_training_complete_claimed=False));hold_existing(groups)
    except Exception as error:
        publish(state_path,dict(state,phase='FAILED_REQUIRES_INSPECTION',error=type(error).__name__+': '+str(error),traceback=traceback.format_exc(),time=time.time(),created_groups=[str(g)for g in groups],replay_forbidden=True));hold_existing(groups);raise
if __name__=='__main__':main()
