"""Real CT DEBUG: old/new transfer equivalence and exact interrupted optimizer continuation."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import json
import os
from copy import copy
import torch
from hiercp_v222.v1_cache import configuration
from hiercp_v222.v1_local import model,collate,LocalBatch
from hiercp_v222.training import configure_runtime
from hiercp_v222.v1_execution import Saver,tree_to,restore_rng,optimizer_step,encode_memory

def equal(a,b):
    if torch.is_tensor(a):return torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b

def main():
    root=Path(sys.argv[1]);root.mkdir(parents=True,exist_ok=False)
    cfg,base=configuration();configure_runtime(base,42);torch.set_num_threads(8)
    fixture=torch.load(os.environ.get('HIERCP_TEST_FIXTURE',ROOT/'work/v222_v1_l0_20260924/debug2/actual_graphs_DEBUG.pt'),weights_only=False,map_location='cpu')
    cpu=collate([(v,i) for i,v in enumerate(fixture['values'])]).pin_memory()
    shallow=copy(cpu.graph)
    assert all(v.is_pinned() for store in shallow.stores for v in store.values() if torch.is_tensor(v))
    old=LocalBatch(cpu.graph.clone().cuda(),cpu.source_patches.cuda(),cpu.target_patches.cuda(),cpu.source_index.cuda(),cpu.indices.cuda())
    new=cpu.cuda(non_blocking=True)
    assert all(v.device.type=='cpu' for store in cpu.graph.stores for v in store.values() if torch.is_tensor(v))
    assert all(equal(old.graph[k].to_dict(),new.graph[k].to_dict()) for k in cpu.graph.node_types+cpu.graph.edge_types)
    net=model(cfg,base).cuda();net.local.dense_batch_size=6;net.eval()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        a=net.local(old);b=net.local(new)
    assert torch.equal(a,b),'Transfer changed actual L0 output'
    support=b.float();del old,new,a,b
    owners=torch.tensor([0,0,1,1,2,2],device='cuda');classes=torch.tensor([1,0,1,0,1,0],device='cuda')
    net.train();optimizer=torch.optim.AdamW(net.parameters(),lr=base['training']['lr'],weight_decay=base['training']['weight_decay'],fused=True)
    with torch.autocast('cuda',dtype=torch.bfloat16):plan=net.fit_support_clusters(support,owners,classes)
    args=(support,owners,classes);weights=torch.ones(2,device='cuda')
    first,_,_,check=optimizer_step(net,optimizer,cpu,args,plan,classes,weights,5,check_gradients=True)
    state=dict(phase='optimization',epoch=0,step=1,next_batch=1,last_group='debug_real_CT',plan=plan,memory=args)
    saver=Saver(root,net,optimizer,dict(debug=True));receipt=saver.save(state)
    if hasattr(saver,'flush'):saver.flush()
    expected,_,_,_=optimizer_step(net,optimizer,cpu,args,plan,classes,weights,5)
    expected_model=tree_to(net.state_dict(),'cpu');expected_optimizer=tree_to(optimizer.state_dict(),'cpu')
    if hasattr(saver,'close'):saver.close()
    del net,optimizer,saver;torch.cuda.empty_cache()
    loaded=torch.load(root/'checkpoint_latest.pt',map_location='cpu',weights_only=False)
    net=model(cfg,base).cuda();net.local.dense_batch_size=6
    net.load_state_dict(loaded['model']);net.train()
    optimizer=torch.optim.AdamW(net.parameters(),lr=base['training']['lr'],weight_decay=base['training']['weight_decay'],fused=True)
    optimizer.load_state_dict(loaded['optimizer']);saved=tree_to(loaded['state'],'cuda');restore_rng(loaded['rng'])
    actual,_,_,_=optimizer_step(net,optimizer,cpu,saved['memory'],saved['plan'],classes,weights,5)
    assert expected==actual,(expected,actual)
    assert equal(expected_model,tree_to(net.state_dict(),'cpu')),'Resumed parameter update differs'
    assert equal(expected_optimizer,tree_to(optimizer.state_dict(),'cpu')),'Resumed optimizer state differs'
    class RealFixture:
        rows=[r|dict(donor_group='case:liver_1',bounds=dict(edges=int(fixture['values'][i][0].num_edges))) for i,r in enumerate(fixture['rows'])]
        cases=sorted({r['case_id'] for r in rows})
        meta=dict(split=dict(inner_train=cases),identities=dict(cases={r['case_id']:dict(patient_group=r['patient_group']) for r in rows}))
        def __len__(self):return len(self.rows)
        def item(self,i,epoch=0):return fixture['values'][i],i
    data=RealFixture()
    interrupted=dict(phase='initial_memory',epoch=0,step=0,memory_work=None,memory_next=0)
    folder=root/'memory_interrupted';folder.mkdir()
    partial_saver=Saver(folder,net,optimizer,dict(debug=True));partial_saver.stop_requested=lambda:True
    assert not encode_memory(net,data,interrupted,partial_saver,2,2,True)
    checkpoint=torch.load(folder/'checkpoint_latest.pt',map_location='cpu',weights_only=False)
    assert checkpoint['state']['memory_next']==2
    recovered=tree_to(checkpoint['state'],'cuda')
    folder=root/'memory_resumed';folder.mkdir();continued_saver=Saver(folder,net,optimizer,dict(debug=True))
    assert encode_memory(net,data,recovered,continued_saver,2,2,True)
    reference=dict(phase='initial_memory',epoch=0,step=0,memory_work=None,memory_next=0)
    folder=root/'memory_reference';folder.mkdir();reference_saver=Saver(folder,net,optimizer,dict(debug=True))
    assert encode_memory(net,data,reference,reference_saver,2,2,True)
    assert equal(reference['memory'],recovered['memory']),'Resumed support memory differs'
    from hiercp_v222.v1_training import evaluate
    metrics_before=evaluate(net,data,reference['memory'],2,2)
    metrics_after=evaluate(net,data,reference['memory'],2,2,batch_complete=torch.cuda.empty_cache)
    assert metrics_before==metrics_after,'Allocator callback changed evaluation metrics'
    result=dict(debug=True,real_CT=True,full_training=False,transfer_L0_bitwise_equal=True,
        host_graph_pinning_preserved=True,host_graph_stores_preserved=True,
        resumed_loss_bitwise_equal=True,resumed_all_parameters_bitwise_equal=True,resumed_optimizer_bitwise_equal=True,
        resumed_partial_support_memory_bitwise_equal=True,
        validation_allocator_callback_metrics_identical=True,
        checkpoint_bytes=(root/'checkpoint_latest.pt').stat().st_size,checkpoint_seconds=receipt['seconds'],first_loss=first['loss'],resumed_loss=actual['loss'],**check)
    (root/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)
if __name__=='__main__':
    if '--optimized' in sys.argv:
        sys.argv.remove('--optimized')
        from tools.v222_runtime_execution import installed, AsyncSaver, encode_memory
        Saver=AsyncSaver
        with installed(dict(debug=True)):
            main()
    else:main()
