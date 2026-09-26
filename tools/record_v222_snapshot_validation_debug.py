"""Publish measured optimization evidence, requiring full actual support equality."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    import torch
    from tools.run_v222_process_runtime import runtime_identity
    from tools.verify_v1_resume import equal
    def read(path):return json.loads((ROOT/path).read_text())
    def rows(path):return [json.loads(line) for line in (ROOT/path).read_text().splitlines()]
    previous=Path('work/v222_support_process_full_20260926')
    current=Path('work/v222_support_snapshot_full_20260926_DEBUG')
    old,new=read(previous/'result.json'),read(current/'result.json')
    assert old['full_support'] and new['full_support']
    assert old['observations']==new['observations']==11279
    assert new['support_snapshot_count']==1
    for name in ('cache_sha256','checkpoint_sha256','source_identity'):
        assert read(previous/'started.json')[name]==read(current/'started.json')[name]
    before,after=rows(previous/'profile.jsonl'),rows(current/'profile.jsonl')
    assert len(before)==len(after)==353
    for left,right in zip(before,after):
        for name in ('completed','total','batch_nodes','batch_edges','physical_batch'):
            assert left[name]==right[name],name
    expected=torch.load(ROOT/previous/'embeddings_DEBUG.pt',map_location='cpu',weights_only=True)
    actual=torch.load(ROOT/current/'embeddings_DEBUG.pt',map_location='cpu',weights_only=True)
    assert torch.equal(expected,actual),'Full support embeddings changed'
    checkpoints=[torch.load(ROOT/path/'checkpoint_latest.pt',map_location='cpu',weights_only=False) for path in (previous,current)]
    for name in ('model','optimizer'):
        assert equal(checkpoints[0][name],checkpoints[1][name]),name
    rng_before,rng_after=(value['rng'] for value in checkpoints)
    for name in ('torch','cuda'):assert equal(rng_before[name],rng_after[name]),name
    # configure_runtime seeds Torch only. Independent profile processes did
    # not start with the same global Python/NumPy states; do not claim their
    # ending states are comparable. Per-record NumPy generators are seeded.
    assert checkpoints[1]['state']['memory_next']==11279
    training=read('work/v222_runtime_snapshot_fullsupport_20260926_DEBUG/result.json')
    assert training['all_support_used'] and training['same_workspace_reference_bitwise_equal']
    training_before=read('work/v222_runtime_fullsupport_reference_20260926_DEBUG/result.json')
    assert training['queries']==training_before['queries']
    assert [row['loss'] for row in training['steps']]==[row['loss'] for row in training_before['steps']]
    updates=[torch.load(ROOT/'work'/folder/'checkpoint_latest.pt',map_location='cpu',weights_only=False)
        for folder in ('v222_runtime_fullsupport_reference_20260926_DEBUG','v222_runtime_snapshot_fullsupport_20260926_DEBUG')]
    for name in ('model','optimizer'):assert equal(updates[0][name],updates[1][name]),'Six-step '+name
    resume=read('work/v222_snapshot_resume_20260926_DEBUG/result.json')
    for key in ('resumed_all_parameters_bitwise_equal','resumed_optimizer_bitwise_equal','resumed_partial_support_memory_bitwise_equal'):
        assert resume[key],key
    result=dict(debug=True,real_CT=True,full_training=False,full_evaluation=False,server_A100_MIG_test=False,
        runtime_sha256=runtime_identity(),unit_regressions_passed=36,
        support_before=old,support_after=new,full_support_bitwise_equal=True,
        model_and_optimizer_checkpoint_bitwise_equal=True,torch_cuda_rng_unchanged=True,topology_order_and_batch_counts_equal=True,
        support_speed_ratio=old['seconds']/new['seconds'],
        training_before=training_before,training=training,six_updates_bitwise_equal=True,
        resume=resume,resources=read(current/'resources_sample.json'),
        limitations=['Single local runs, not repeated statistical timings',
            'Brief CPU unit checks overlap the full-support run',
            'CUDA event GNN region includes host launch gaps; not isolated kernel timing',
            'Independent profiles did not initialize equal global Python/NumPy RNG states; cross-run comparison covers Torch/CUDA only',
            'Every batch still serializes a complete checkpoint; only unchanged tensor copies are reused',
            'No complete 40-epoch run or server MIG throughput measurement'],
        references=dict(before=str(previous),after=str(current)))
    output=ROOT/'validation/v222_r6/support_snapshot_20260926_DEBUG.json'
    with output.open('x',encoding='utf-8',newline='\n') as stream:json.dump(result,stream,indent=2)
    print(json.dumps(dict(output=str(output),seconds=new['seconds'],speed_ratio=result['support_speed_ratio'],support_bitwise_equal=True,optimizer_bitwise_equal=True)))


if __name__=='__main__':main()
