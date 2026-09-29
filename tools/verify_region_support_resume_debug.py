"""Fresh-process DEBUG training/resume on an existing cache; no preparation."""
import argparse,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    import torch
    from l0_regions.training import hash_state
    from l0_regions.training_data import sha
    original_hash=sha(a.cache)
    def run(name,*extra):
        subprocess.run([sys.executable,'-u',str(ROOT/'tools/run_fixed_regions.py'),'train',
            '--debug','--profile-policy','research-report','--cache',str(a.cache),
            '--output',str(a.output/name),'--workers','8','--cuda-gib','6','--rss-gib','12',
            '--resident-gib','4','--batch-candidates','8',*map(str,extra)],cwd=ROOT,check=True)
    run('uninterrupted')
    run('paused','--debug-pause-step','1')
    run('resumed','--resume',a.output/'paused/checkpoint_latest.pt')
    left=torch.load(a.output/'uninterrupted/checkpoint_latest.pt',weights_only=False,map_location='cpu')
    right=torch.load(a.output/'resumed/checkpoint_latest.pt',weights_only=False,map_location='cpu')
    checks={name:hash_state(left[name])==hash_state(right[name]) for name in ('model','optimizer','rng','state')}
    assert all(checks.values()),checks
    assert sha(a.cache)==original_hash
    rows=[json.loads(line) for line in (a.output/'uninterrupted/support_timing.jsonl').read_text().splitlines()]
    assert sum(r['physical_batch'] for r in rows)==8 and rows[-1]['completed']==8
    contract=json.loads((a.output/'uninterrupted/execution_contract.json').read_text())
    assert contract['cache_rebuilt'] is False
    result=dict(debug=True,actual_CT=True,status='PASS',full_training=False,cache_rebuilt=False,
        cache_unchanged=True,checks=checks,steps=left['state']['step'],initial_support_records=8,
        calibration_support_passes=1,extra_initial_support_passes=0,
        scope='Actual f07b13f DEBUG train8/val2; full supplied DEBUG schedule, refresh/validation/best/final, fresh-process exact resume')
    (a.output/'report.json').write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
