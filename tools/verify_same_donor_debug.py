"""Actual CT short DEBUG; fresh-process optimization and refresh resume parity."""
import argparse,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from l0_regions.training import hash_state
from l0_regions.training_data import write_new

def main():
    p=argparse.ArgumentParser()
    for k in ('cache','fine-cache','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    common=['train','--debug','--learning-policy','same_donor_live_v1','--cache',str(a.cache),'--fine-cache',str(a.fine_cache),
        '--profile-policy','research-report','--workers','4','--cuda-gib','8','--rss-gib','24','--resident-gib','4',
        '--batch-candidates','2','--support-patients','2','--activation-storage','retained','--execution-pipeline','overlapped',
        '--device-cache-gib','1','--sage-workspace-mib','64']
    def run(name,*extra,refresh_pause=False,baseline_pause=False):
        command=[sys.executable,'-B','-u','tools/run_fixed_regions.py']
        if refresh_pause or baseline_pause:
            code="""from l0_regions.execution_pipeline import CheckpointPipeline
from tools.run_fixed_regions import main
save=CheckpointPipeline.save
def pause_refresh(self,net,optimizer,state,identity,**kw):
    result=save(self,net,optimizer,state,identity,**kw)
    if state['phase']=='refresh_memory' and state['memory_done']==2:
        (self.root/'STOP_AFTER_BATCH').write_text('Explicit DEBUG refresh interruption')
    return result
CheckpointPipeline.save=pause_refresh
main()
"""
            if baseline_pause:
                code=code.replace("state['phase']=='refresh_memory' and state['memory_done']==2",
                    "state['phase']=='optimization' and state['step']==0 and state.get('initial_validation') is not None")
            command=[sys.executable,'-B','-u','-c',code]
        subprocess.run(command+common+['--output',str(a.output/name),*map(str,extra)],cwd=ROOT,check=True)
    run('full');run('paused','--debug-pause-step','1');run('resumed','--resume',a.output/'paused/checkpoint_latest.pt')
    run('refresh_paused',refresh_pause=True);run('refresh_resumed','--resume',a.output/'refresh_paused/checkpoint_latest.pt')
    run('baseline_paused',baseline_pause=True);run('baseline_resumed','--resume',a.output/'baseline_paused/checkpoint_latest.pt')
    def load(name):return torch.load(a.output/name/'checkpoint_latest.pt',map_location='cpu',weights_only=False)
    full=load('full');parity={name:{k:hash_state(full[k])==hash_state(load(name)[k]) for k in ('model','optimizer','state','rng')} for name in ('resumed','refresh_resumed','baseline_resumed')}
    if [r['epoch'] for r in full['state']['validation_history']]!=[0,1]:raise AssertionError('Missing initial/epoch validation')
    stopped=load('refresh_paused')
    if stopped['state']['phase']!='refresh_memory' or stopped['state']['memory_done']!=2:raise AssertionError('Wrong interruption scope')
    if not all(all(v.values()) for v in parity.values()):raise AssertionError(parity)
    from l0_regions.final import load as load_final
    from l0_regions.donor_data import DonorDataset,DonorLoader
    from hiercp_v222.v1_local import support_for_recipient
    model,memory,artifact=load_final(a.output/'full/checkpoint.pt',allow_debug=True)
    ds=DonorDataset(a.cache,'inner_val',True,'research-report',a.fine_cache);DonorLoader(ds,4,4*2**30)
    records=[ds.raw.record(i) for i in range(len(ds))]
    scores=model.region_candidates.scores(model,records,support_for_recipient(memory,ds.rows[0]['patient_group']),batch_size=2,workers=4)
    updates=[json.loads(line) for line in (a.output/'full/update_timing.jsonl').read_text().splitlines()]
    report=dict(debug=True,full_training=False,gpu=torch.cuda.get_device_name(),actual_train=len(full['state']['memory']['record_ids']),actual_validation=len(ds),
        fresh_process_resume_exact=parity,refresh_cursor=stopped['state']['memory_done'],
        final_candidate_scores=scores.tolist(),all_modules_updated=all(all(v>0 for v in u['learning']['probe_max_abs_delta'].values()) for u in updates),
        peak_cuda_bytes=max(u['peak_cuda_bytes'] for u in updates),peak_rss_bytes=max(u['rss_bytes'] for u in updates),
        mean_update_seconds=sum(u['step_seconds'] for u in updates)/len(updates),
        scope='actual DEBUG CT; no full batch32/server admission or accuracy improvement claim')
    write_new(a.output/'report.json',report);print(json.dumps(report,indent=2))

if __name__=='__main__':main()
