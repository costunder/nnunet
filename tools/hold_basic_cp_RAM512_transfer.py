"""Resume an adopted own RAM512 cache, retarget the existing UI2 and hold.

Builder and cache adopter must already have succeeded. No signals are sent.
No native results, checkpoint, candidate arrays or prior cache are overwritten.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import stat
import subprocess
import sys
import time
import traceback
import psutil

UID=41833
NNPY='/home/aicompetition06/.conda/envs/nnunet/bin/python3.10'
BASE=Path('/dev/shm/aicompetition06_basic_cp_ablation_20261013')
# Literal existing cache spelling prevents accidental creation of a new cache.
EXISTING_CACHE=Path('/var/tmp/aicompetition06_basic_cp_seven_resume_1791934200100000000')
PUBLIC=BASE/'view_basic7_20261014.py'

def require(ok,message):
    if not ok:raise ValueError(message)

def exact(path,directory=False):
    path=Path(path)
    require(path.is_absolute()and'..'not in path.parts and path.resolve(strict=True)==path and not any(p.is_symlink()for p in(path,*path.parents)),'Exact nonsymlink absolute evidence required')
    info=path.stat();require(info.st_uid==UID and(stat.S_ISDIR(info.st_mode)if directory else stat.S_ISREG(info.st_mode)),'Exact UID41833 regular evidence required')
    return path

def sha(path):
    with exact(path).open('rb')as f:
        before=os.fstat(f.fileno());h=hashlib.sha256()
        for block in iter(lambda:f.read(8*2**20),b''):h.update(block)
        after=os.fstat(f.fileno())
    require((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns),'Held evidence changed while hashing')
    return h.hexdigest()

def read(path):
    result=json.loads(exact(path).read_bytes());require(isinstance(result,dict),'Evidence mapping required');return result

def pin(path,checksum):
    require(isinstance(checksum,str)and re.fullmatch('[a-f0-9]{64}',checksum)is not None,'Literal lowercase full SHA256 required')
    path=exact(path);require(sha(path)==checksum,'Pinned evidence changed: '+str(path));return path

def seal(path,value):
    path=Path(path);require(path.parent.parent==BASE and path.parent.name.startswith('Basic_CP_')and not path.exists(),'Fresh own cohort operational receipt required; never replay')
    with path.open('x',encoding='utf8')as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())

def identity(p):
    return dict(pid=p.pid,create_time=p.create_time(),UID=p.uids().real,command=p.cmdline())

def alive(actor):
    require(actor['UID']==UID and actor['command']and type(actor['pid'])is int,'Exact own captured actor required')
    birth=actor.get('birth',actor.get('create_time'))
    try:
        p=psutil.Process(actor['pid'])
        if p.create_time()!=birth:return False
        require(p.uids().real==UID,'Captured actor UID changed')
        if p.status()in(psutil.STATUS_ZOMBIE,psutil.STATUS_DEAD):return False
        command=p.cmdline()
        if command!=actor['command']:
            if not command:
                from hiercp_v1x.basic_cp_ablation_resources import _kernel_exiting
                if _kernel_exiting(p):return True
            raise RuntimeError('Live captured actor argv changed: '+str(p.pid))
        return True
    except psutil.NoSuchProcess:return False

def viewer_wrapper(group,code):
    return '\n'.join(['import runpy,sys','from pathlib import Path','group=Path('+repr(str(group))+')','code=Path('+repr(str(code))+')',
        'sys.argv=[str(code/"tools/watch_basic_cp_RAM512.py"),"--manifest",str(group/"launch.json"),"--code-root",str(code),*sys.argv[1:]]',
        'runpy.run_path(str(code/"tools/watch_basic_cp_RAM512.py"),run_name="__main__")',''])

def bind_viewer(group,code,old_sha):
    pin(PUBLIC,old_sha);archive=group/'previous_public_viewer_RAM512.py';temporary=group/'new_public_viewer_RAM512.py'
    require(not archive.exists()and not temporary.exists(),'Unknown viewer mutation outcome must be inspected before retry')
    with archive.open('xb')as f:f.write(PUBLIC.read_bytes());f.flush();os.fsync(f.fileno())
    require(sha(archive)==old_sha==sha(PUBLIC),'Old viewer must be archived unchanged before rebinding')
    source=viewer_wrapper(group,code);compile(source,str(PUBLIC),'exec')
    with temporary.open('x',encoding='utf8')as f:f.write(source);f.flush();os.fsync(f.fileno())
    checksum=sha(temporary);os.replace(temporary,PUBLIC);require(sha(PUBLIC)==checksum,'Published viewer differs')
    seal(group/'managed_RAM512_viewer_binding.json',dict(path=str(PUBLIC),previous_sha256=old_sha,previous_archive=str(archive),new_sha256=checksum,
        UI_version=2,native_formatter_changed=False,only_manifest_and_code_target_rebound=True))

def postlaunch(manifest_path,expected_manifest_sha,cache_output,old_viewer_sha):
    require(os.getuid()==UID and socket.gethostname().split('.')[0]=='ece-a6gpu6'and sys.executable==NNPY,'Exact host/NN Python/own UID required')
    manifest_path=Path(manifest_path);cache_output=Path(cache_output)
    group=exact(manifest_path.parent,True)
    require(manifest_path.name=='launch.json'and group.parent==BASE and group.name.startswith('Basic_CP_')and cache_output==EXISTING_CACHE,'Exact own cohort manifest and existing cache required; no new cache creation')
    exact(cache_output,True);manifest=read(pin(manifest_path,expected_manifest_sha));code=exact(manifest['code'],True)
    require(manifest['group']==str(group)and code==group/'code'and set(manifest['registration'])==set(map(str,range(7)))and manifest['effective_total_owned_RAM_cap_bytes']==512*2**30,'Actual common seven-arm RAM512 launch required')
    export=read(group/'code_export.json');require(export['commit']==manifest['commit']and export['code']==str(code),'Actual immutable transfer export/commit required')
    for relative,checksum in export['code_sha256'].items():
        p=Path(relative);require(not p.is_absolute()and'..'not in p.parts,'Export path traversal forbidden');pin(code/p,checksum)
    helper=pin(code/'tools/persist_basic_cp_seven.py',export['code_sha256']['tools/persist_basic_cp_seven.py']);helper_sha=sha(helper)
    adoption=read(cache_output/'adoption_RAM_pause.json');saved=read(cache_output/'identity.json')
    history=exact(adoption['history'],True)
    require(cache_output/'history'in history.parents and adoption['old_manifest_sha256']==sha(history/'manifests/old.json')
        and adoption['new_manifest_sha256']==expected_manifest_sha==sha(cache_output/'source_manifest.json')==sha(history/'manifests/new.json')
        and adoption['new_commit']==manifest['commit']and adoption['new_helper_source_sha256']==helper_sha and adoption['all_original_experiment_files_unchanged']is True
        and set(adoption['approved_physical_transfer'])==set(map(str,range(7)))and set(adoption['approved_physical_transfer'].values())==set(map(str,range(7)))
        and adoption['checkpoint_NPZ_copies_during_transfer']==0,'Actual byte-preserving admitted whole-arm cache transfer proof required')
    require(saved['uid']==UID and saved['node']=='ece-a6gpu6'and saved['commit']==manifest['commit']and saved['helper_source_sha256']==helper_sha,'Adopted cache identity differs')
    pin(cache_output/'persistence_helper.py',helper_sha);pin(PUBLIC,old_viewer_sha)
    seal(group/'managed_RAM512_transfer_postlaunch.intent.json',dict(manifest_sha256=expected_manifest_sha,helper_sha256=helper_sha,cache=str(cache_output),started_unix=time.time(),replay_forbidden=True,processes_signaled=False))
    env=dict(os.environ,PYTHONPATH=str(code),PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    command=[NNPY,'-B','-u',str(helper),'--manifest',str(manifest_path),'--output',str(cache_output),'--unfinished-checkpoint','best','--resume-cache']
    log=group/'checkpoint_preserver.log';require(not(cache_output/'stopped.json').exists(),'Unresolved stopped cache outcome must be inspected')
    with log.open('xb',buffering=0)as f:child=subprocess.Popen(command,cwd=code,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    actual=identity(psutil.Process(child.pid));require(actual['UID']==UID and actual['command']==command,'Actual CPU-only cache actor differs')
    cache_actor=dict(actual,birth=actual['create_time'],output=str(cache_output),log=str(log),CPU=[82,83],CUDA_visible_devices='',RSS_limit_bytes=4*2**30,unfinished_checkpoint_policy='best',helperSourceSHA256=helper_sha,original_results_overwritten=False,one_operational_cache=True)
    seal(group/'checkpoint_preserver_actor.json',cache_actor);time.sleep(1);require(child.poll()is None,'Cache helper exited at startup; retain log and inspect')
    bind_viewer(group,code,old_viewer_sha)
    seal(group/'managed_RAM512_transfer_postlaunch.started.json',dict(manifest_sha256=expected_manifest_sha,cache_actor=cache_actor,UI_version=2,completion_claimed=False,started_unix=time.time()))
    return manifest,cache_actor

def hold(manifest,cache_actor):
    group=exact(manifest['group'],True)
    sys.path.insert(0,manifest['code'])
    actors={key:row['controller_identity']for key,row in manifest['registration'].items()};actors['cache']=cache_actor
    while any(alive(a)for a in actors.values()):time.sleep(30)
    states={key:read(Path(row['root'])/'status.json')for key,row in manifest['registration'].items()}
    complete=all(s.get('status')=='COMPLETE'and s.get('completed_epochs')==250 and s.get('evaluated_cases')==26 and s.get('historical_metrics_complete')is True for s in states.values())
    seal(group/'managed_RAM512_transfer_postlaunch.ended.json',dict(ended_unix=time.time(),all_registered_actors_released=True,all_seven_verified_status_complete=complete,states=states,processes_signaled=False))
    require(complete,'Some cohort arms ended without full250/BEST26/historical metrics; original logs/results preserved')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--execute',action='store_true');parser.add_argument('--manifest',type=Path,required=True);parser.add_argument('--expected-manifest-sha',required=True);parser.add_argument('--cache-output',type=Path,required=True);parser.add_argument('--old-viewer-sha',required=True)
    args=parser.parse_args();require(args.execute,'Parent explicit --execute required for prepared postlaunch helper')
    group=exact(args.manifest.parent,True)
    require(args.manifest.name=='launch.json'and group.parent==BASE and group.name.startswith('Basic_CP_'),'Own exact cohort launch namespace required')
    manifest=None;cache_actor=None;failure=None
    try:manifest,cache_actor=postlaunch(args.manifest,args.expected_manifest_sha,args.cache_output,args.old_viewer_sha)
    except Exception as error:
        failure=dict(error=type(error).__name__+': '+str(error),traceback=traceback.format_exc(),failed_unix=time.time(),processes_signaled=False,original_results_preserved=True,replay_forbidden=True)
        seal(group/'managed_RAM512_transfer_postlaunch.failed.json',failure)
        if (group/'launch.json').exists():manifest=read(group/'launch.json')
        if (group/'checkpoint_preserver_actor.json').exists():cache_actor=read(group/'checkpoint_preserver_actor.json')
    if manifest is not None:
        actors={k:r['controller_identity']for k,r in manifest['registration'].items()}
        if cache_actor is not None:actors['cache']=cache_actor
        sys.path.insert(0,manifest['code'])
        if failure:
            while any(alive(a)for a in actors.values()):time.sleep(30)
        else:hold(manifest,cache_actor)
    if failure:raise RuntimeError(failure['error'])

if __name__=='__main__':main()
