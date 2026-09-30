"""Actual CT short checks: resume parity, native FOV, mask independence and CP."""
import argparse,copy,json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch,nibabel as nib,psutil
from types import SimpleNamespace
from l0_regions.training import hash_state
from l0_local_cnn.data import Dataset,CropStore
from l0_local_cnn.artifact import load

def main():
    p=argparse.ArgumentParser();p.add_argument('--inventory',type=Path,required=True);p.add_argument('--completed',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--resume-evidence',type=Path,help='Reuse already completed exact-resume check, repeat only input/CP/batch profile')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    ds=Dataset(a.inventory,'inner_train',True);m=ds.meta;margin=m['local_cnn']['margin_mm']
    common=[sys.executable,'-B','-u','tools/run_local_cnn.py','train','--cache',str(a.inventory),'--margin-mm',str(margin),
        '--workers','4','--cuda-gib','12','--rss-gib','32','--resident-gib','8','--batch-candidates','2','--support-patients','2','--debug']
    if a.resume_evidence is None:
        subprocess.run(common+['--output',str(a.output/'paused'),'--debug-pause-step','1'],check=True,cwd=ROOT)
        subprocess.run(common+['--output',str(a.output/'resumed'),'--resume',str(a.output/'paused/checkpoint_latest.pt')],check=True,cwd=ROOT)
    resumed=a.resume_evidence or a.output/'resumed'
    left=torch.load(a.completed/'checkpoint_latest.pt',weights_only=False,map_location='cpu');right=torch.load(resumed/'checkpoint_latest.pt',weights_only=False,map_location='cpu')
    parity={k:hash_state(left[k])==hash_state(right[k]) for k in ('model','optimizer','state','rng')}
    if not all(parity.values()):raise AssertionError(parity)
    net,memory,value=load(a.completed/'checkpoint.pt',allow_debug=True)
    rows=[r for r in m['records'] if r['case_id']=='liver_66'];store=CropStore(m,4,8*2**30,32*2**30)
    q=store.batch(rows,list(range(len(rows))));audit=q.audit;gpu=q.to('cuda');net.eval()
    with torch.no_grad():before=net.local(gpu)
    gpu.images=torch.where(gpu.organ,gpu.images,float('nan'));gpu.validate()
    with torch.no_grad():after=net.local(gpu)
    torch.testing.assert_close(before,after,atol=0,rtol=0)
    gpu.images=gpu.images.detach().requires_grad_(True);gpu.validate();net.local(gpu)[:,0].sum().backward()
    if torch.count_nonzero(gpu.images.grad[~gpu.organ]):raise AssertionError('External CT gradient')
    ranges=[]
    for mm in (5,10,20):
        meta=copy.deepcopy(m);meta['local_cnn']['margin_mm']=mm;s=CropStore(meta,4,8*2**30,32*2**30)
        s.raw.cache=store.raw.cache.copy();b=s.batch(rows,list(range(len(rows))))
        ranges.append(dict(margin_mm=mm,shape=list(b.images.shape),crops=b.audit));del b,s
    # Exact original placement and final rank-then-full-mask filter on actual CT.
    from hiercp_v22.data import sources,donor_in_target_spacing
    from hiercp_v222.placement import placement_spec
    from l0_local_cnn.recommendation import recommend
    raw_rows={r['case_id']:r for r in m['raw_records']}
    def case(name):
        raw=store.raw.cache[name];r=raw_rows[name]
        return SimpleNamespace(paths=SimpleNamespace(case_id=name),image=raw['ct'],label=raw['lab'],spacing=raw['spacing'],
            image_affine=nib.load(r['image']).affine,label_affine=nib.load(r['label']).affine)
    target=case('liver_66');donor=case(rows[0]['donor_case_id']);col=sources(donor,m['base']['cache']['source_pad'],m['config']['donor_max_diameter_mm'])
    source=next(s for s,_ in col if s.component_id==rows[0]['donor_component'])
    transformed,_=donor_in_target_spacing(source,donor.spacing,target.spacing)
    placements=[placement_spec(target,transformed,r['center'],donor.paths.case_id) for r in rows]
    net.zero_grad(set_to_none=True)
    selection=recommend(net,memory,value,target,donor,source,placements,batch_size=2,workers=4)
    if len(selection['ranked_candidates'])!=len(rows) or not selection['filter_after_model_scoring']:raise AssertionError('CP scoring incomplete')
    # Direct same-case physical 32/48/64 workload, using 128 real stored centers.
    assignment_path=ROOT/'work/v222_v1_full_training_20260924/cache/pair_assignment.json'
    full=json.loads(assignment_path.read_text());neg=[r for r in full if r['case_id']=='liver_66' and r['target']==0]
    if len(neg)!=128:raise ValueError('Full actual 128-center profile inventory required')
    donor_keys={k:rows[0][k] for k in ('donor_case_id','donor_component','donor_group')}
    fullrows=[r for r in rows if r['target']==1]+[dict(r,**donor_keys,bounds={'edges':i}) for i,r in enumerate(neg)]
    from l0_regions.donor_learning import LiveContext,groups,forward_loss,configuration
    from hiercp_v222.v1_local import support_for_recipient
    timings=[];support=support_for_recipient(memory,rows[0]['patient_group']);net.eval();plan=net.fit_support_clusters(*support)
    for count in (32,48,64):
        clone=copy.deepcopy(net,{id(net.local_crop_store):None}).train()
        data=SimpleNamespace(rows=fullrows);ids=next(groups(data,count));ctx=LiveContext(data,count)
        batch=store.batch([fullrows[i] for i in ids],ids).to('cuda');targets=torch.tensor([fullrows[i]['target'] for i in ids],device='cuda')
        opt=torch.optim.AdamW(clone.parameters(),lr=m['base']['training']['lr'])
        for trial in range(2):
            opt.zero_grad(set_to_none=True);torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();t=time.perf_counter()
            loss,_=forward_loss(clone,batch,support,plan,targets,None,ctx,configuration(),indices=ids)
            torch.cuda.synchronize();f=time.perf_counter();loss.backward();torch.cuda.synchronize();back=time.perf_counter();opt.step();torch.cuda.synchronize();end=time.perf_counter()
            timings.append(dict(physical_batch=len(ids),warmup=trial==0,forward_seconds=f-t,backward_seconds=back-f,update_seconds=end-t,
                peak_cuda_bytes=torch.cuda.max_memory_allocated(),unique_crops=len(batch.images),input_shape=list(batch.images.shape),scope='actual same-case CT plus DEBUG support; excludes IO/checkpoint'))
            if torch.cuda.max_memory_allocated()>12*2**30:raise MemoryError('DEBUG resource budget exceeded')
        del clone,batch,opt,loss
    updates=[json.loads(line) for line in (a.completed/'update_timing.jsonl').read_text().splitlines()]
    report=dict(debug=True,full_training=False,full_evaluation=False,quality_validated=False,actual_CT=True,gpu=torch.cuda.get_device_name(),
        train_records=len(ds),val_records=len(Dataset(a.inventory,'inner_val',True)),resume_exact=parity,
        external_CT_NaN_difference=float((before-after).abs().max()),external_CT_gradient_zero=True,
        native_shapes=audit,range_comparison=ranges,CP_selection=selection,physical_batch_profiles=timings,
        module_updates=[u['learning']['probe_max_abs_delta'] for u in updates],
        parameters=sum(p.numel() for p in net.parameters()),L0_parameters=sum(p.numel() for p in net.local.parameters()),
        rss_bytes=psutil.Process().memory_info().rss)
    (a.output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf8');print('REPORT:',a.output/'report.json')
if __name__=='__main__':main()
