"""One named local-CNN experiment: first run, exact resume, exclusive ownership.

Optional --gpu resolves the physical GPU/MIG before importing PyTorch. No global latest-checkpoint search,
checkpoint migration, detached worker, or overwrite of prior training attempts.
"""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def write_json(path,value):
    temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    with temporary.open('x',encoding='utf8') as f:
        json.dump(value,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
    os.replace(temporary,path)

@contextmanager
def exclusive(root):
    root.mkdir(parents=True,exist_ok=True)
    lock=root/'.experiment.lock'
    try:lock.mkdir()
    except FileExistsError:
        raise RuntimeError(f'Experiment is locked: {lock}. Another run may be active; no automatic lock removal.') from None
    owner=dict(host=socket.gethostname(),pid=os.getpid(),token=uuid.uuid4().hex)
    try:
        write_json(lock/'owner.json',owner)
        yield
    finally:
        if (lock/'owner.json').is_file() and json.loads((lock/'owner.json').read_text())==owner:
            (lock/'owner.json').unlink();lock.rmdir()

def foreground(command):
    # The foreground child receives terminal SIGINT itself. Keep the wrapper
    # alive until that child saves/returns, so it cannot prematurely unlock.
    old=signal.signal(signal.SIGINT,lambda *_:print('Waiting for foreground training to pause/save...',flush=True))
    try:
        child=subprocess.Popen(command,cwd=ROOT)
        code=child.wait()
        if code:raise RuntimeError(f'Training/preparation returned {code}; outputs preserved. Re-run this same experiment to resume its saved checkpoint.')
    finally:signal.signal(signal.SIGINT,old)

def select_checkpoint(root,state):
    if not state['attempts']:return None,None
    current=state['attempts'][-1]
    output=inside_experiment(root,current['output'])
    checkpoint=output/'checkpoint_latest.pt'
    if checkpoint.is_file():return checkpoint,output
    if (output/'training_complete.json').exists() or (output/'checkpoint_timing.jsonl').exists():
        raise RuntimeError('Recorded checkpoint is missing; refusing to restart or silently select older weights')
    previous=current['resume_from']
    if previous:
        checkpoint=inside_experiment(root,previous)
        if not checkpoint.is_file():raise FileNotFoundError('The bound resume checkpoint is missing')
        return checkpoint,checkpoint.parent
    print('No optimization checkpoint yet; initial calibration must restart. Prior outputs are preserved.',flush=True)
    return None,None

def inside_experiment(root,relative):
    path=(root/relative).resolve()
    if not path.is_relative_to(root.resolve()):raise ValueError('Checkpoint/output escapes this experiment')
    return path

def validate_inventory(inventory,request,debug):
    meta=json.loads(inventory.read_text())
    if meta.get('format')!='native_local_cnn_inventory_v1' or meta['original_inventory_sha256']!=request['original_inventory_sha256'] or meta['local_cnn']!=request['local_cnn'] or meta['debug']!=debug:
        raise ValueError('Original inventory/FOV differs from the selected experiment')

def run_experiment(a,request,execute=foreground):
    root=a.experiment.resolve()
    with exclusive(root):
        manifest=root/'experiment.json';inventory=root/'inventory/index.json'
        if manifest.exists():
            state=json.loads(manifest.read_text())
            if state['format']!='local_cnn_experiment_v1' or state['request']!=request:
                raise ValueError('Experiment settings/source/data differ; use its original settings or a distinct experiment path')
        else:
            # Explicit adoption of an old `run` output is by this exact path,
            # never by newest mtime, GPU number, glob or another experiment.
            existing=[p.name for p in root.iterdir() if p.name!='.experiment.lock']
            if existing and (not inventory.is_file() or not (root/'training').is_dir()):
                raise ValueError('Existing directory is not an identifiable local CNN run')
            if existing and not any((root/'training'/name).is_file() for name in ('paused.json','training_complete.json')):
                raise ValueError('Legacy run must show PAUSED or completed before adoption; it may still be running')
            if existing:validate_inventory(inventory,request,a.debug)
            state=dict(format='local_cnn_experiment_v1',request=request,attempts=[])
            if existing:state['attempts'].append(dict(output='training',resume_from=None,adopted=True))
            write_json(manifest,state)
        common=[sys.executable,'-B','-u',str(ROOT/'tools/run_local_cnn.py')]
        fov=['--margin-mm',str(a.margin_mm),'--config',str(a.config.resolve())]
        debug=['--debug'] if a.debug else []
        if not inventory.exists():
            if state['attempts']:raise FileNotFoundError('Experiment inventory missing; do not regenerate resume inputs')
            execute(common+['prepare','--cache',str(a.cache.resolve()),'--output',str(inventory.parent),*fov,*debug])
        validate_inventory(inventory,request,a.debug)
        checkpoint,previous_output=select_checkpoint(root,state)
        if previous_output is not None and (previous_output/'training_complete.json').exists():
            # Also recover interruption between completed optimization and final export.
            from l0_local_cnn.artifact import validate,export
            import torch
            final=previous_output/'checkpoint.pt'
            if final.exists():validate(torch.load(final,map_location='cpu',weights_only=False),allow_debug=a.debug)
            else:export(checkpoint,inventory,final)
            print(f'COMPLETE: {final}; no training restarted',flush=True)
            return final
        output=root/'attempts'/f'{len(state["attempts"])+1:04d}'
        if output.exists():raise FileExistsError('Attempt path already exists; outputs will not be overwritten')
        state['attempts'].append(dict(output=output.relative_to(root).as_posix(),
            resume_from=None if checkpoint is None else checkpoint.relative_to(root).as_posix()))
        write_json(manifest,state)
        print(f'EXPERIMENT: {root}\nMARGIN_MM: {a.margin_mm}\nRESUME: {checkpoint or "NEW (no saved update)"}\nCHECKPOINT: {output / "checkpoint_latest.pt"}',flush=True)
        command=common+['train','--cache',str(inventory),'--output',str(output),*fov,*debug,
            '--workers',str(a.workers),'--cuda-gib',str(a.cuda_gib),'--rss-gib',str(a.rss_gib),
            '--resident-gib',str(a.resident_gib),'--batch-candidates',*map(str,a.batch_candidates),
            '--support-patients',str(a.support_patients),'--device-cache-gib',str(a.device_cache_gib)]
        if checkpoint is not None:command+=['--resume',str(checkpoint)]
        if a.debug_pause_step is not None:command+=['--debug-pause-step',str(a.debug_pause_step)]
        execute(command)
        result=output/'checkpoint_latest.pt'
        if not result.is_file():raise FileNotFoundError('Training returned without a checkpoint; no success recorded')
        if (output/'training_complete.json').is_file():
            final=output/'checkpoint.pt'
            if not final.is_file():raise FileNotFoundError('Completed training has no final artifact')
            print(f'EXPERIMENT: {root}\nCOMPLETE: {final}\nRESUME CHECKPOINT: {result}',flush=True)
            return final
        print(f'EXPERIMENT: {root}\nCHECKPOINT: {result}\nRe-run the same command to resume this experiment.',flush=True)
        return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu',type=int,help='Physical nvidia-smi GPU number; resolve its current single/allocated MIG automatically')
    p.add_argument('--experiment',type=Path,required=True,help='Stable run root; reuse exactly this path to resume')
    p.add_argument('--cache',type=Path,required=True,help='Original observation inventory, unchanged on every invocation')
    p.add_argument('--margin-mm',type=float,required=True)
    p.add_argument('--config',type=Path,default=ROOT/'config/v22_local_cnn.json')
    p.add_argument('--workers',type=int,required=True)
    for name in ('cuda-gib','rss-gib','resident-gib','device-cache-gib'):p.add_argument('--'+name,type=float,required=True)
    p.add_argument('--batch-candidates',type=int,nargs='+',required=True)
    p.add_argument('--support-patients',type=int,required=True)
    p.add_argument('--debug',action='store_true');p.add_argument('--debug-pause-step',type=int)
    a=p.parse_args()
    if a.gpu is not None:
        from tools.local_cnn_device import select
        select(a.gpu)
    from l0_local_cnn.model import validate_config
    from l0_regions.training_data import sha,source_identity
    cfg=json.loads(a.config.read_text());cfg['margin_mm']=a.margin_mm;validate_config(cfg)
    if a.debug_pause_step is not None and not a.debug:raise ValueError('Pause-step is DEBUG only')
    if a.workers<2 or not 0<a.resident_gib<a.rss_gib or not 0<=a.device_cache_gib<a.cuda_gib or a.batch_candidates!=sorted(set(a.batch_candidates)) or min(a.batch_candidates)<2 or a.support_patients<1:
        raise ValueError('Explicit valid parallel/resource/batch settings required')
    settings={k:getattr(a,k) for k in ('workers','cuda_gib','rss_gib','resident_gib','device_cache_gib','batch_candidates','support_patients','debug')}
    request=dict(local_cnn=cfg,original_inventory_sha256=sha(a.cache),source=source_identity(),settings=settings)
    run_experiment(a,request)

if __name__=='__main__':main()
