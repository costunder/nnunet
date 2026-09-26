"""Actual CT DEBUG: save at 64MiB, preserve state, continue at 256MiB.

Compares resumed256 against an in-memory256 continuation of the SAME64MiB
state. Does not claim equivalence to continuing at64MiB or full training.
"""
from pathlib import Path
import json
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
import hiercp.model as local_model
from hiercp_v222.v1_cache import configuration
from hiercp_v222.v1_local import model,collate
from hiercp_v222.v1_execution import Saver,optimizer_step,tree_to,restore_rng
from hiercp_v222.training import configure_runtime
from tools.run_v222_optimized import resume_policy
from tools.verify_v1_resume import equal


def main():
    root=Path(sys.argv[1]);root.mkdir(parents=True,exist_ok=False)
    cfg,base=configuration();configure_runtime(base,42);torch.set_num_threads(8)
    local_model.EDGE_ATTENTION_WORKSPACE_BYTES=64*1024**2
    fixture=torch.load(ROOT/'work/v222_v1_l0_20260924/debug2/actual_graphs_DEBUG.pt',map_location='cpu',weights_only=False)
    cpu=collate([(v,i) for i,v in enumerate(fixture['values'])]).pin_memory()
    net=model(cfg,base).cuda();net.local.dense_batch_size=6;net.eval()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):support=net.local(cpu.cuda()).float()
    owners=torch.tensor([0,0,1,1,2,2],device='cuda');classes=torch.tensor([1,0,1,0,1,0],device='cuda')
    memory=(support,owners,classes)
    with torch.autocast('cuda',dtype=torch.bfloat16):plan=net.fit_support_clusters(*memory)
    net.train();opt=torch.optim.AdamW(net.parameters(),lr=base['training']['lr'],weight_decay=base['training']['weight_decay'],fused=True)
    weights=torch.ones(2,device='cuda')
    optimizer_step(net,opt,cpu,memory,plan,classes,weights,5)
    state=dict(phase='optimization',epoch=0,step=1,next_batch=1,memory=memory,plan=plan,release_unused=True)
    saver=Saver(root,net,opt,dict(debug=True));saver.save(state)
    loaded=torch.load(root/'checkpoint_latest.pt',map_location='cpu',weights_only=False)
    workspace,release=resume_policy(loaded,root/'checkpoint_latest.pt',migrate_workspace=256)
    assert workspace==256 and release is True
    local_model.EDGE_ATTENTION_WORKSPACE_BYTES=workspace*1024**2
    expected,_,_,_=optimizer_step(net,opt,cpu,memory,plan,classes,weights,5,check_gradients=True)
    expected_model=tree_to(net.state_dict(),'cpu');expected_opt=tree_to(opt.state_dict(),'cpu')
    del net,opt,saver;torch.cuda.empty_cache()
    net=model(cfg,base).cuda();net.local.dense_batch_size=6;net.load_state_dict(loaded['model']);net.train()
    opt=torch.optim.AdamW(net.parameters(),lr=base['training']['lr'],weight_decay=base['training']['weight_decay'],fused=True)
    opt.load_state_dict(loaded['optimizer']);restored=tree_to(loaded['state'],'cuda');restore_rng(loaded['rng'])
    actual,_,usage,gradients=optimizer_step(net,opt,cpu,restored['memory'],restored['plan'],classes,weights,5,check_gradients=True)
    assert expected==actual
    assert equal(expected_model,tree_to(net.state_dict(),'cpu'))
    assert equal(expected_opt,tree_to(opt.state_dict(),'cpu'))
    assert restored['step']==1 and restored['next_batch']==1
    result=dict(debug=True,real_CT=True,full_training=False,from_workspace_mib=64,to_workspace_mib=256,
        model_parameters=sum(p.numel() for p in net.parameters()),physical_batch=6,
        purpose='Migration correctness on real regression fixture, not throughput or MIG capacity',
        resumed256_matches_in_memory256_loss_model_optimizer=True,comparison_to_continued64=False,
        saved_cursor_preserved=True,loss=actual['loss'],**gradients,**usage)
    (root/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
