"""Short actual CT no-coarsening checks; no full training and no source cache writes."""
import argparse,copy,json,subprocess,sys
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from l0_regions.training import hash_state,make_model
from l0_regions.training_data import Budget,write_new
from l0_regions.fine_graph import FineDataset,FineLoader,transition


def main():
    p=argparse.ArgumentParser()
    for k in ('cache','fine-cache','checkpoint','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    old=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    identity=old['identity']
    if not identity['debug']:raise ValueError('DEBUG checkpoint only')
    limits=identity['resource_limits']
    args=['--debug','--profile-policy',identity['profile_policy'],'--cache',str(a.cache),'--fine-cache',str(a.fine_cache),
          '--workers',str(identity['workers']),'--cuda-gib',str(limits['cuda_bytes']/2**30),'--rss-gib',str(limits['rss_bytes']/2**30),
          '--resident-gib',str(identity['resident_budget_bytes']/2**30),'--batch-candidates',*map(str,identity['candidates']),
          '--activation-storage',identity['activation_storage'],'--support-patients',str(identity['support_training']['patients']),
          '--execution-pipeline','overlapped','--device-cache-gib',str(identity['execution_pipeline']['device_cache_bytes']/2**30),
          '--sage-workspace-mib',str(identity['execution_pipeline']['sage_workspace_bytes']//2**20)]
    # Fail if the training path unexpectedly calls any coarsening primitive.
    runner="from unittest.mock import patch; from tools.run_fixed_regions import main\nwith patch('l0_ezsp.encoder.partition',side_effect=AssertionError('partition called')), patch('l0_ezsp.ops.mass_mean',side_effect=AssertionError('pool called')), patch('l0_regions.preparation.prepare',side_effect=AssertionError('prepare called')): main()"
    def run(name,*extra):
        subprocess.run([sys.executable,'-B','-u','-c',runner,'train',*args,'--output',str(a.output/name),*map(str,extra)],cwd=ROOT,check=True)
    run('uninterrupted')
    run('paused','--debug-pause-step',1)
    run('resumed','--resume',a.output/'paused/checkpoint_latest.pt')
    run('transition','--resume',a.checkpoint,'--resume-without-coarsening','--debug-pause-step',old['state']['step'])
    run('continued','--resume',a.output/'transition/checkpoint_latest.pt')
    def load(name):return torch.load(a.output/name/'checkpoint_latest.pt',map_location='cpu',weights_only=False)
    full,resumed,changed=load('uninterrupted'),load('resumed'),load('transition')
    parity={k:hash_state(full[k])==hash_state(resumed[k]) for k in ('model','optimizer','state','rng')}
    preserved={k:hash_state(old[k])==hash_state(changed[k]) for k in ('optimizer','rng')}
    preserved['all_weight_tensors']=all(torch.equal(v,changed['model'][k]) for k,v in old['model'].items() if torch.is_tensor(v))
    for key in ('epoch','step','batch','next_batch'):preserved[key]=old['state'][key]==changed['state'][key]
    if not all(parity.values()) or not all(preserved.values()):raise AssertionError((parity,preserved))
    torch.set_num_threads(identity['workers'])
    ds=FineDataset(a.cache,'inner_train',True,identity['profile_policy'],a.fine_cache)
    loader=FineLoader(ds,identity['workers'],identity['resident_budget_bytes'])
    cpu=loader.get(list(range(len(ds))));batch=cpu.to('cuda');budget=Budget(**limits)
    net=make_model(ds,budget,True,'retained').eval()
    from l0_sage.encoder import GraphSAGEEncoder
    reference=GraphSAGEEncoder(net.local.core,seed=42).cuda().eval()
    reference.encoder.load_state_dict(net.local.core.state_dict())
    for block in reference.encoder.blocks:
        for conv in block.conv.convs.values():conv.stable_spmm=True
    edges={e:t.clone() for e,t in batch.graph.edge_index_dict.items()}
    with patch('l0_ezsp.encoder.partition',side_effect=AssertionError('partition')),patch('l0_ezsp.ops.mass_mean',side_effect=AssertionError('pool')):
        x=net.local(batch);y=reference(batch)
        torch.testing.assert_close(x,y,rtol=0,atol=0)
        probe=torch.arange(x.numel(),device=x.device).view_as(x)/x.numel()
        ga=torch.autograd.grad((x*probe).sum(),tuple(net.local.parameters()),allow_unused=True)
        gb=torch.autograd.grad((y*probe).sum(),tuple(reference.parameters()),allow_unused=True)
        for left,right in zip(ga,gb):
            if left is None or right is None:
                if left is not None or right is not None:raise AssertionError('Gradient path differs')
            else:torch.testing.assert_close(left,right,rtol=0,atol=0)
    if not all(torch.equal(e,batch.graph.edge_index_dict[k]) for k,e in edges.items()):raise AssertionError('Fine edges altered')
    if loader.get(list(range(len(ds)))) is not cpu:raise AssertionError('RAM fine batch not reused')
    from l0_regions.final import load as load_final
    from hiercp_v222.v1_local import support_for_recipient
    model,memory,artifact=load_final(a.output/'uninterrupted/checkpoint.pt',allow_debug=True)
    val=FineDataset(a.cache,'inner_val',True,identity['profile_policy'],a.fine_cache)
    FineLoader(val,identity['workers'],identity['resident_budget_bytes'],store=loader.store)
    records=[val.raw.record(i) for i in range(len(val))]
    with patch('l0_regions.preparation.prepare',side_effect=AssertionError('CP partition')):
        scores=model.region_candidates.scores(model,records,support_for_recipient(memory,val.rows[0]['patient_group']),batch_size=identity['candidates'][0],workers=identity['workers'])
    if len(scores)!=len(records):raise AssertionError('Candidate coverage')
    report=dict(debug=True,full_training=False,nnunet_training=False,gpu=torch.cuda.get_device_name(),resume_parity=parity,
                graph_transition_preserved=preserved,reference_fine_sage_output_gradient_exact=True,
                merge_and_pool_calls_forbidden=True,original_edges_unchanged=True,
                fine_nodes=int(batch.graph.sampled_counts.sum()),fine_edges=sum(e.shape[1] for e in edges.values()),
                actual_pairs=len(ds),candidate_smoke_count=len(records),candidate_scores=scores.tolist(),
                scope='Actual DEBUG train8/val2, no full server batch32 admission or CP quality claim')
    write_new(a.output/'report.json',report);print(json.dumps(report,indent=2))

if __name__=='__main__':main()
