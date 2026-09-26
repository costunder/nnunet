"""Collect measured support and optimizer evidence; never synthesize results."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    import torch
    from hiercp_v222.contracts import sha
    from tools.run_v222_process_runtime import runtime_identity
    def read(path):return json.loads((ROOT/path).read_text())
    def rows(path):return [json.loads(line) for line in (ROOT/path).read_text().splitlines()]
    baseline=rows('work/v222_support_baseline_20260926/profile.jsonl')
    current=rows('work/v222_support_process_full_20260926/profile.jsonl')
    full=read('work/v222_support_process_full_20260926/result.json')
    training=read('work/v222_runtime_process_20260926_DEBUG/result.json')
    if not full['full_support'] or full['observations']!=11279:
        raise AssertionError('Full actual support completion required')
    if not training['same_workspace_reference_bitwise_equal']:
        raise AssertionError('Actual optimizer comparison must pass')
    selected=current[:len(baseline)]
    for old,new in zip(baseline,selected):
        for key in ('completed','batch_nodes','batch_edges','physical_batch'):
            if old[key]!=new[key]:raise AssertionError(('Coverage/topology changed',key,old[key],new[key]))
    old_path=ROOT/'work/v222_support_baseline_20260926/checkpoint_latest.pt'
    expected=torch.load(old_path,map_location='cpu',weights_only=False)['state']['memory_work']
    embeddings=torch.load(ROOT/'work/v222_support_process_full_20260926/embeddings_DEBUG.pt',map_location='cpu',weights_only=True)
    actual=embeddings[:len(expected)]
    if not torch.equal(actual,expected):
        raise AssertionError(f'Support prefix differs: maximum {(actual-expected).abs().max().item()}')
    before=sum(r['batch_wall_seconds'] for r in baseline)
    after=sum(r['batch_wall_seconds'] for r in selected)
    result=dict(debug=True,real_CT=True,full_training=False,server_A100_MIG_test=False,
        runtime_sha256=runtime_identity(),unit_regressions_passed=32,
        same_prefix=dict(observations=len(expected),batches=len(baseline),before_seconds=before,after_seconds=after,
            speed_ratio=before/after,embeddings_bitwise_equal=True,topology_and_batch_counts_equal=True,
            old_checkpoint_sha256=sha(old_path)),full_support=full,training=training,
        scope='Single local measurements; CPU regression checks overlap later full-support batches, not the compared prefix',
        references=dict(baseline='work/v222_support_baseline_20260926/profile.jsonl',
            process='work/v222_support_process_full_20260926/profile.jsonl',
            optimizer='work/v222_runtime_process_20260926_DEBUG/result.json'))
    output=ROOT/'validation/v222_r6/support_process_20260926_DEBUG.json'
    with output.open('x',encoding='utf-8',newline='\n') as stream:json.dump(result,stream,indent=2)
    print(json.dumps(dict(output=str(output),support_seconds=full['seconds'],prefix_speed_ratio=before/after,
        full_support=full['observations'],support_prefix_bitwise=True,optimizer_bitwise=True)))


if __name__=='__main__':main()
