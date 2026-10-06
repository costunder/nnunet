"""Actual-CT/CUDA mechanical curriculum check; never production accuracy."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inventory',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    import torch,psutil
    from l0_regions.training import hash_state
    from l0_local_cnn.data import Dataset
    from l0_regions.candidate_curriculum import Curriculum
    meta=json.loads(a.inventory.read_text(encoding='utf8'))
    if meta.get('format')!='native_local_cnn_inventory_v1' or meta.get('debug') is not True:
        raise ValueError('An explicit existing actual-CT DEBUG native inventory is required')
    train_cases=['liver_1','liver_2','liver_49','liver_66'];val_cases=['liver_31']
    meta=copy.deepcopy(meta)
    meta['records']=[r for r in meta['records'] if r['case_id'] in train_cases+val_cases]
    meta['local_cnn']['margin_mm']=10.
    for case in train_cases+val_cases:
        if sum(r['case_id']==case and r['target']==0 for r in meta['records'])!=128:
            raise ValueError('Each DEBUG case must retain all actual128 U')
    index=a.output/'inventory.json';index.write_text(json.dumps(meta,indent=2),encoding='utf8')
    cfg=ROOT/'config/v22_cumulative_u16.json'
    ds=Dataset(index,'inner_train',True)
    before=Curriculum(ds.rows,json.loads(cfg.read_text()),debug=True)
    (a.output/'resources.json').write_text(json.dumps(dict(debug=True,gpu=torch.cuda.get_device_name(),
        total_cuda_bytes=torch.cuda.get_device_properties(0).total_memory,cpu=psutil.cpu_count(),
        affinity=len(psutil.Process().cpu_affinity()),ram_available_bytes=psutil.virtual_memory().available,
        train_cases=train_cases,val_cases=val_cases,full_training=False,full_evaluation=False),indent=2),encoding='utf8')
    common=[sys.executable,'-B','-u','tools/run_local_cnn.py','train','--cache',str(index.resolve()),
        '--margin-mm','10','--curriculum-config',str(cfg.resolve()),'--workers','4',
        '--cuda-gib','12','--rss-gib','32','--resident-gib','8','--batch-candidates','32',
        '--support-patients','2','--device-cache-gib','0','--debug']
    def run(name,extra=()):
        output=a.output/name
        with (a.output/(name+'.log')).open('x',encoding='utf8') as stream:
            subprocess.run(common+['--output',str(output.resolve()),*extra],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
        return torch.load(output/'checkpoint_latest.pt',map_location='cpu',weights_only=False)
    left=run('continuous')
    paused=run('paused',['--debug-pause-step','1'])
    right=run('resumed',['--resume',str((a.output/'paused/checkpoint_latest.pt').resolve())])
    parity={k:hash_state(left[k])==hash_state(right[k]) for k in ('model','optimizer','state','rng')}
    if not all(parity.values()):raise AssertionError(dict(exact_resume=parity))
    final=left['state']['candidate_curriculum'];before.load_state_dict(final)
    if len(final['history'])!=1:raise AssertionError('DEBUG epoch/gate coverage mismatch')
    updates=[json.loads(line) for line in (a.output/'continuous/update_timing.jsonl').read_text().splitlines()]
    if not all(u['learning']['optimizer_step_completed'] and all(v>0 for v in u['learning']['probe_max_abs_delta'].values()) for u in updates):
        raise AssertionError('Every monitored real module must update')
    gate=json.loads((a.output/'continuous/curriculum_epoch_001.json').read_text())
    val=json.loads((a.output/'continuous/epoch_001.json').read_text())
    report=dict(debug=True,actual_CT=True,actual_CUDA=True,fresh_untrained_initialization=True,
        full_training=False,full_evaluation=False,quality_validated=False,exact_resume=parity,
        parameters=sum(v.numel() for v in left['model'].values() if isinstance(v,torch.Tensor)),
        completed_debug_epochs=1,actual_updates=len(updates),train_records=len(ds),
        training_gate=gate['transition'],validation_metrics=val['metrics'],
        validation_scope='one actual held-out CT; all P+128U; not full21 evaluation',
        peak_cuda_bytes=max(u['peak_cuda_bytes'] for u in updates),
        calibration=json.loads((a.output/'continuous/batch_calibration.json').read_text()),
        curriculum_final_state=final,all128_query_curriculum_quality_proven=False)
    (a.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8')
    print('REPORT:',a.output/'report.json',flush=True)


if __name__=='__main__':main()
