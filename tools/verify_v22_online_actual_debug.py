"""Actual CT, 128 production proposals, real owner/RPC/native loader DEBUG.

Only initialization admits an existing eight-support DEBUG checkpoint and old
native fixture. No production validators are changed. No model/score/graph or
paste function is mocked. One event is not G3/G4 full-scale validation.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--native',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--integrated-segmentation',action='store_true',help='DEBUG one-process GNN/CP/standard augmentation/CUDA train and epoch resume')
    args=parser.parse_args();root=args.output.resolve();root.mkdir(parents=True,exist_ok=False)
    if args.integrated_segmentation:
        n=json.loads(args.native.read_text())
        os.environ.update(nnUNet_preprocessed=str(Path(n['preprocessed']).parent),nnUNet_raw=str(Path(n['raw']).parent),
                          nnUNet_results=str(root/'nnUNet_results'),nnUNet_compile='false',nnUNet_wandb_enabled='false')
    import numpy as np
    import torch
    from hiercp_v222.contracts import read_json,sha,write_new
    from hiercp_v222.v1_cache import configuration
    from tools.v22_online_runtime import admit_cuda_workspace,scoring_runtime,backend_state
    from tools.v22_online_rank_bank import RankedEntryBuilder,entry_name
    from tools.v22_online_rank_adapter import RankedBank,RankedLoader,RankedMaterializationService
    from tools.v22_rank_recommendation import load_checkpoint,recommend
    from custom_trainers.nnUNetTrainer_OnlinePairedCP import OnlineCPBank,nnUNetDataLoaderOnlineCP
    from custom_trainers.onlinecp_raw_bank import RawBankStore
    from hiercp_v22.volumes import VolumeCache
    from hiercp.common import discover_cases
    from hiercp.preparation_runtime import snapshot,Measurement
    from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
    admit_cuda_workspace();torch.set_num_threads(8)
    cfg,base=configuration();gpu_lock=threading.RLock()
    with gpu_lock,scoring_runtime(base,cfg['seed']):network,memory,payload=load_checkpoint(args.checkpoint,allow_debug=True)
    if not payload['debug']:raise ValueError('This harness requires explicitly DEBUG weights')
    native=read_json(args.native);recipient='liver_31';donor='liver_73'
    pool=payload['donor_pool'];di=next(i for i,r in enumerate(pool) if r['case_id']==donor and r['component_id']==1)
    paths={p.case_id:p for p in discover_cases(Path(native['medical'])/'Data') if p.case_id in (recipient,donor)}
    raw={r['case_id']:r for r in payload['raw_records']}
    for c,p in paths.items():
        for kind,path in [('image',p.image_path),('label',p.label_path)]:
            if sha(path)!=raw[c][kind+'_sha256']:raise ValueError('Actual raw content differs from reviewed DEBUG artifact')
    names=[entry_name(recipient,i) for i in range(len(pool))]
    meta=dict(format='hiercp_online_bank_v2',debug=True,scope='explicit real CT initialization fixture only',
        native_sha256=sha(args.native),split=payload['split'],identities=payload['identities'],donor_pool=pool,
        entries_by_case={recipient:names},candidate_count=128,hier_top_k=1,tumor_label=2,liver_label=1,
        cp_probability=.8,intensity_scale_range=[.95,1.05],intensity_shift_range_hu=[-5.,5.],
        paste_contract='onlinecp_raw_target_paste_v1')
    index=root/'index.json';write_new(index,meta)
    owner=RankedEntryBuilder.__new__(RankedEntryBuilder)
    owner.path=index;owner.root=root;owner.meta=meta;owner.native=native
    owner.network=network;owner.memory=memory;owner.payload=payload;owner.cfg=payload['config'];owner.base=payload['base']
    owner.gpu_lock=gpu_lock;owner.paths=paths;owner.volumes=VolumeCache(paths,owner.base,owner.cfg)
    owner.plans=read_json(native['plans']);owner.plan=owner.plans['configurations']['3d_fullres']
    owner.locks={c:threading.RLock() for c in paths};owner.native_lock=threading.Lock();owner.event_lock=threading.Lock()
    owner.store=RawBankStore(root);owner.raw_ready={}
    bank=RankedBank.__new__(RankedBank);OnlineCPBank.__init__(bank,index)
    bank.identities={n:(recipient,i) for i,n in enumerate(names)}
    integrated=None;integration_result=None
    if args.integrated_segmentation:
        from tools.v22_integrated_smoke_debug import IntegratedSmoke
        integrated=IntegratedSmoke(root,native,index,bank,names,di,gpu_lock,args.checkpoint)
    # Observe actual model outputs and repeat under the same reviewed runtime.
    checks={};captured={}
    def checked_recommend(*a,**kw):
        from dataclasses import replace
        records=a[1]
        counts={key:[sum(r[side][key].values()) for r in records for side in ('source_local','target_local')]
                for key in ('counts','edge_counts')}
        checks['branch_graph_statistics']={k:dict(min=int(min(v)),max=int(max(v)),mean=float(np.mean(v)),branches=len(v)) for k,v in counts.items()}
        checks['dense_patch_shapes']={side:list(records[0][side].shape) for side in ('source_patch','target_patch')}
        before=backend_state();first=recommend(*a,**kw)
        print(json.dumps(dict(stage='actual_GNN_128_scored',CUDA_allocated=torch.cuda.memory_allocated())),flush=True)
        repeated=recommend(*a,**kw)
        changed=dict(kw,recipient_case=replace(kw['recipient_case'],label=np.where(kw['recipient_case'].label==2,1,kw['recipient_case'].label)))
        without=recommend(*a,**changed)
        project=lambda r:[(v['index'],v['model_score'],v['raw_rank']) for v in r['ranked_candidates']]
        if first!=repeated or project(first)!=project(without):raise AssertionError('Actual online score/rank repeat or annotation invariance failed')
        checks.update(repeat_exact=True,annotation_score_rank_invariant=True,runtime=before)
        captured.update(placement=kw['placements'][first['selected_index']] if first['selected_index'] is not None else None,
            case=kw['recipient_case'],report=first)
        return first
    measurement=Measurement();start=time.perf_counter();service=None
    try:
        with measurement,patch('tools.v22_online_rank_adapter.RankedEntryBuilder',return_value=owner),patch('tools.v22_rank_recommendation.recommend',checked_recommend):
            service=RankedMaterializationService(index,gpu_lock)
            print(json.dumps(dict(debug=True,stage='actual_owner_RPC_128_candidates',resources=snapshot(),support=len(memory['record_ids']),physical_batch=payload['physical_batch'])),flush=True)
            if integrated is not None:integration_result=integrated.run()
            entry=bank._load(names[di])  # Missing receipt forces authenticated RPC -> actual owner.
            selected=bank.selected_index(entry)
            if selected is None:raise AssertionError('No actual valid paste selected; positive-paste path remains untested')
            directory=Path(native['preprocessed'])/owner.plan['data_identifier']
            dataset=infer_dataset_class(str(directory))(str(directory),[recipient])
            data,seg,_,props=dataset.load_case(recipient)
            from comparison_randomness import PairedLoaderMixin
            from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
            labels_manager=PlansManager(owner.plans).get_label_manager(read_json(Path(native['preprocessed'])/'dataset.json'))
            loader=RankedLoader.__new__(RankedLoader)
            # Use actual base constructors. Only the production full-cohort
            # admission is bypassed for this explicit one-event DEBUG fixture.
            PairedLoaderMixin.__init__(loader,dataset,1,owner.plan['patch_size'],owner.plan['patch_size'],labels_manager,
                bank_path=str(index),policy='hier_argmax',online_seed=42,transforms=None)
            loader.online_bank=bank
            loader._source_entry_names=lambda c:names
            # This is the production bank load path; event is now a verified receipt hit.
            loader._load_selected_source=lambda c,i:bank._load(names[i])
            draws=iter([.1,(di+.5)/len(names),.3,.4,.5]);loader._rng=lambda:SimpleNamespace(random=lambda:next(draws))
            loader.get_indices=lambda:[recipient];loader.get_do_oversample=lambda j:False
            crop={};actual_bbox=loader._raw_candidate_crop_bbox
            def record_bbox(*a):
                lo,hi=actual_bbox(*a);crop.update(lo=lo,hi=hi);return lo,hi
            loader._raw_candidate_crop_bbox=record_bbox
            batch=loader.generate_train_batch()
            if int(batch['online_cp_applied'][0])!=1:raise AssertionError('Actual selected event not pasted into final input')
            # Independent ordinary native preprocessing of raw CT with the exact selected footprint.
            sys.path.insert(0,str(ROOT/'tests'))
            from test_raw_cp_resampling_debug import debug_native
            p=captured['placement'];target=captured['case'];ct=target.image.copy();label=target.label.copy()
            points=tuple(p.coordinates().T);ct[points]=p.image[p.mask]*.99;label[points]=2
            expected,labels,_=debug_native(ct,label,target.spacing,owner.plans)
            lo=np.asarray(crop['lo']);hi=np.asarray(crop['hi']);shape=np.asarray(expected.shape[1:])
            slices=(slice(None),)+tuple(slice(max(0,int(l)),min(int(n),int(h))) for l,h,n in zip(lo,hi,shape))
            pad=[(0,0)]+[(max(0,-int(l)),max(0,int(h-n))) for l,h,n in zip(lo,hi,shape)]
            oracle=np.pad(expected[slices],pad,constant_values=0);gt=np.pad(labels[slices],pad,constant_values=-1)
            actual=np.asarray(batch['data'])[0];actual_gt=np.asarray(batch['target'])[0]
            np.testing.assert_allclose(actual,oracle,atol=1e-5,rtol=1e-5);np.testing.assert_array_equal(actual_gt,gt)
            error=float(np.max(np.abs(actual-oracle)))
            print(json.dumps(dict(stage='actual_native_final_input_passed',max_abs_error=error,selected=selected)),flush=True)
    finally:
        if service is not None:service.close()
        bank._get_raw_store().close();owner.store.close()
    result=dict(debug=True,actual_CT=True,recipient=recipient,donor=donor,component=1,candidates=128,
        support=len(memory['record_ids']),physical_batch=payload['physical_batch'],model_parameters=sum(p.numel() for p in network.parameters()),
        production_init_tested=False,initialization_exception='Existing DEBUG checkpoint and old native metadata explicitly initialized; production continues to reject them',
        actual_owner=True,actual_RPC=True,actual_native_loader=True,actual_raw_native_oracle=True,
        final_native_CT_max_abs_error=error,final_native_seg_exact=True,paste_engine=bank.paste_engine_identity(),
        score_checks=checks,selection=entry['_receipt']['selection'],raw_receipt=read_json(root/f'raw_receipts/{recipient}.json'),
        storage_written_bytes=sum(p.stat().st_size for p in root.rglob('*') if p.is_file()),
        seconds=time.perf_counter()-start,resources=measurement.report,cuda_peak_bytes=torch.cuda.max_memory_allocated(),
        full_support=False,segmentation_optimizer_coexistence=integrated is not None,
        integrated_segmentation=integration_result,full_training=False,full_evaluation=False)
    write_new(root/'result.json',result)
    print(json.dumps(dict(result=str(root/'result.json'),bytes=result['storage_written_bytes'],seconds=result['seconds'])),flush=True)


if __name__=='__main__':main()
