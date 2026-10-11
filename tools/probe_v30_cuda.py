"""Explicit real-CT/CUDA DEBUG: full128 inference and one connectivity update.

Production model, graph, dataset, forty-epoch and physical-batch settings are
unchanged. This isolated diagnostic uses one independent donor/recipient pair,
all its eligible observed P and all128 actual U for inference, then all P+7U
for a single explicitly bounded DEBUG optimizer update. Its sixteen prototypes
are fitted on the real donor's twenty-four regions only: no population-quality
or trained recommendation claim is made.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path, PurePosixPath
import random
import sys
import time
import traceback
from types import SimpleNamespace
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(8*2**20),b""):
            digest.update(block)
    return digest.hexdigest()


def write_new(path,value):
    with Path(path).open("x",encoding="utf8") as stream:
        json.dump(value,stream,indent=2,allow_nan=False)


def stage(name,**values):
    print(json.dumps(dict(stage=name,**values)),flush=True)


def archived_snapshot(output):
    from hiercp_v1x.contracts import V1_ARCHIVE_SHA256
    archive=ROOT/"versions/v1/pipeline_v1_source.zip"
    if sha(archive)!=V1_ARCHIVE_SHA256:
        raise ValueError("Sealed original v1 archive SHA256 differs")
    snapshot=output/"original_source";snapshot.mkdir()
    with ZipFile(archive) as zipped:
        for name in zipped.namelist():
            if not ((name.startswith("hiercp/") and name.endswith(".py")) or name=="config/train.json"):
                continue
            relative=PurePosixPath(name)
            if relative.is_absolute() or ".." in relative.parts or ":" in name or "\\" in name:
                raise ValueError("Unsafe sealed original source path")
            target=snapshot.joinpath(*relative.parts);target.parent.mkdir(parents=True,exist_ok=True)
            with target.open("xb") as stream:
                stream.write(zipped.read(name))
    return snapshot


class DebugCanonicalProvider:
    """Actual complete DEBUG canonical records; persistent parallel CPU views."""
    recipient_GT_used_in_forward=False

    def __init__(self,rows,records,*,workers):
        self.ds=SimpleNamespace(rows=copy.deepcopy(rows))
        self.records=tuple(records)
        self.pool=ThreadPoolExecutor(max_workers=workers,thread_name_prefix="v30_DEBUG_views")
        self.calls=0

    def get(self,indices,*,epoch):
        from hiercp_v1x.v24_inputs import materialize_pair,collate
        payloads=list(self.pool.map(lambda index:materialize_pair(self.records[index],epoch=epoch),indices))
        self.calls+=1
        return collate(list(zip(payloads,indices)))

    def close(self):
        self.pool.shutdown(wait=True)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"work/v3/cuda")
    parser.add_argument("--vessels",type=Path,default=ROOT/"datasets/msd_liver/vessels/index.json")
    parser.add_argument("--checkpoint",type=Path,default=ROOT/"work/pretrained_l0_latency_DEBUG_20261008/assets/small_ep4k.model")
    parser.add_argument("--donor",default="liver_1")
    parser.add_argument("--recipient",default="liver_5")
    args=parser.parse_args(argv)
    if args.donor==args.recipient:
        raise ValueError("DEBUG donor and recipient must be independent real patients")
    args.output.mkdir(parents=True,exist_ok=False)
    report=dict(format="v30_actual_CT_CUDA_DEBUG_v1",debug=True,actual_CT=True,
        full_training=False,full_population_evaluation=False,quality_verified=False,
        production_settings_modified=False,trained_checkpoint_saved=False,donor=args.donor,recipient=args.recipient)
    providers=[];phase="initialize"
    try:
        import nibabel as nib
        import numpy as np
        import psutil
        import torch
        from hiercp_v1x.historical_checkpoint import _activate
        from hiercp_v1x.transition_v1_local import _runtime
        from hiercp_v1x import bounded_scope
        from hiercp_v1x.v25_context import install,identity as context_identity
        from hiercp_v1x.v30_vessels import VesselStore
        from hiercp_v1x.historical_evaluation import ResourceBudget
        if not torch.cuda.is_available():
            raise RuntimeError("Actual CUDA is required; no CPU fallback")
        torch.set_num_threads(4)
        torch.manual_seed(42);np.random.seed(42);random.seed(42)
        torch.backends.cudnn.benchmark=False
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        free,total=torch.cuda.mem_get_info()
        available=psutil.virtual_memory().available
        budget=ResourceBudget(min(int(total*.9),free-2**30),int(available*.8))
        report["resources"]=dict(GPU=torch.cuda.get_device_name(),GPU_count=torch.cuda.device_count(),
            GPU_total_bytes=total,GPU_free_before_bytes=free,MIG="consumer CUDA device; no MIG",
            CPU_logical=psutil.cpu_count(),CPU_physical=psutil.cpu_count(logical=False),
            CPU_affinity=psutil.Process().cpu_affinity(),RAM_available_before_bytes=available,
            CPU_workers=4,torch_threads=4,CUDA_DEBUG_budget_bytes=budget.cuda_bytes,RSS_DEBUG_budget_bytes=budget.rss_bytes)
        write_new(args.output/"started.json",report)
        snapshot=archived_snapshot(args.output);_activate(snapshot,None)
        runtime=_runtime(expected_snapshot_root=snapshot)
        base=json.loads((snapshot/"config/train.json").read_text())
        config=bounded_scope.configure(base["graph"],10.)
        install(runtime,approved_profile=True)
        config=replace(config,context_inner_radius_mm=0.)
        base["graph"]=config.to_dict()
        model_config=copy.deepcopy(base["model"])
        model_config.update(hidden_dim=128,heads=4,patient_layers=2,prototype_layers=2,local_layers=3,dense_batch_size=4)
        report["context"]=context_identity();report["graph_config"]=config.to_dict()
        report["source_sha256"]={name:sha(ROOT/name) for name in
            ("hiercp_v1x/v25_context.py","hiercp_v1x/v30_model.py","hiercp_v1x/v30_vessels.py",
             "hiercp_v1x/v30_scorer.py","tools/probe_v30_cuda.py")}
        phase="full131_vessel_admission";stage(phase)
        store=VesselStore(args.vessels)
        with ThreadPoolExecutor(max_workers=4) as pool:
            bindings=dict(zip(store.case_ids,pool.map(store.binding_from_source,store.case_ids)))
        report["full_vessel_inventory"]=store.preload(recipient_bindings=bindings,workers=4)
        from hiercp.common import CasePaths,load_case,paste_source
        from hiercp_v22.data import sources,candidate_pool,donor_in_target_spacing
        from hiercp_v1x.v24_inputs import recipient_context,prepare_donor,build_local_record,tensor_digest
        from hiercp_v1x.v24_geometry import build_recipient_regions,build_upper_graphs
        from hiercp.prototype import build_prototype_bank
        from hiercp_v1x.v30_model import build_v30_model
        from hiercp_v1x.v30_scorer import V30Scorer
        from hiercp_v1x.v23_training import patient_balanced_objective
        phase="actual_CT_load";stage(phase)
        data=ROOT/"datasets/msd_liver/extracted/Task03_Liver"
        def load(case):
            image=data/"imagesTr"/(case+".nii.gz");label=data/"labelsTr"/(case+".nii.gz")
            if sha(image)!=bindings[case].ct_sha256:
                raise ValueError("Actual DEBUG CT differs from saved vessel CT")
            loaded=load_case(CasePaths(case,image,label))
            context=recipient_context(case,loaded.image,np.isin(loaded.label,(1,2)),loaded.spacing,loaded.image_affine)
            return loaded,context,dict(image_sha256=sha(image),label_sha256=sha(label),shape=list(loaded.shape),
                spacing=loaded.spacing.tolist(),affine=loaded.image_affine.tolist())
        with ThreadPoolExecutor(max_workers=2) as pool:
            (donor,donor_context,donor_proof),(recipient,context,recipient_proof)=list(pool.map(load,(args.donor,args.recipient)))
        report["actual_inputs"]=dict(donor=donor_proof,recipient=recipient_proof)
        collection=sources(donor,2,20.)
        if not len(collection):
            raise ValueError("Chosen real donor has no eligible <=20mm source; no substituted donor")
        source,diameter=collection[0]
        prepared=prepare_donor(donor,source,config=config,seed=42,ct_clip=tuple(base["ct_clip"]))
        report["donor_component"]=dict(id=int(source.component_id),equivalent_diameter_mm=float(diameter),
            native_voxels=int(source.voxel_count),source_context_nodes=prepared.canonical_counts["source_context"])
        positive=sources(recipient,2,20.)
        if not len(positive):
            raise ValueError("Chosen real recipient has no eligible observed P; no fabricated supervision")
        phase="full128_native_candidate_pool";stage(phase)
        target_source,_=donor_in_target_spacing(source,donor.spacing,recipient.spacing)
        pool,candidate_audit=candidate_pool(recipient,target_source,dict(seed=42,candidate_count=128),base,
            context.organ_depth,donor_case_id=args.donor)
        rows=[dict(id=f"{args.recipient}:P:{int(item.component_id)}",case_id=args.recipient,center=list(item.anchor_center),
                   donor_case_id=args.donor,donor_component=int(source.component_id),target=1,component=int(item.component_id))
              for item,_ in positive]
        observed=len(rows)
        rows.extend(dict(id=f"{args.recipient}:U:{index:03d}",case_id=args.recipient,center=list(item.center),
            donor_case_id=args.donor,donor_component=int(source.component_id),target=0,component=None) for index,item in enumerate(pool))
        report["candidate_pool"]=dict(full_U_count=len(pool),all_eligible_observed_P=observed,audit=candidate_audit,
            recipient_GT_in_forward=False,labels_used_only_for_observation_and_original_U_generation=True)
        write_new(args.output/"candidates.json",dict(debug=True,rows=rows))
        phase="real_regions_and_DEBUG_prototypes";stage(phase)
        with ThreadPoolExecutor(max_workers=2) as executor:
            donor_regions,recipient_regions=list(executor.map(lambda c:build_recipient_regions(c,config=config,seed=42,
                ct_clip=tuple(base["ct_clip"])),(donor_context,context)))
        bank=build_prototype_bank([(args.donor,donor_regions.region_features)],config=config,rng=np.random.default_rng(42))
        report["prototype_bank"]=dict(debug=True,training_cases=[args.donor],real_regions=int(donor_regions.num_regions),
            prototypes=int(bank.num_prototypes),fingerprint=bank.fingerprint(),population_quality_claim=False)
        phase="full_actual_canonical_preparation";stage(phase,candidates=len(rows),workers=4)
        completed=0
        def construct(row):
            return build_local_record(context,donor,source,prepared,row,config=config,seed=42,
                ct_clip=tuple(base["ct_clip"]),scope_contract=runtime["scope"]["contract_sha256"])
        records=[];began=time.perf_counter()
        with ThreadPoolExecutor(max_workers=4) as executor:
            for record in executor.map(construct,rows):
                records.append(record);completed+=1
                if completed%16==0 or completed==len(rows):
                    stage(phase,completed=completed,total=len(rows),seconds=time.perf_counter()-began)
        report["canonical_preparation"]=dict(records=len(records),seconds=time.perf_counter()-began,
            all_actual_candidates_retained=True,all_source_contexts_present=all(record["source_local"]["counts"]["source_context"]>0 for record in records),
            source_donor_identity_retained=all(record["source_local"]["v25_context_geometry"]["source_identity"]["case_id"]==args.donor for record in records))
        provider=DebugCanonicalProvider(rows,records,workers=4);providers.append(provider)
        rng_before=(random.getstate(),np.random.get_state(),torch.get_rng_state().clone())
        provider.get([0,observed],epoch=29)
        rng_after=(random.getstate(),np.random.get_state(),torch.get_rng_state())
        if (rng_before[0]!=rng_after[0] or not np.array_equal(rng_before[1][1],rng_after[1][1])
                or not torch.equal(rng_before[2],rng_after[2])):
            raise ValueError("Real DEBUG input loading altered global RNG")
        class Geometry:
            recipient_GT_used_in_forward=False
            def __init__(self):self.cache={}
            def __call__(self,plan,provider):
                key=tuple(plan.record_ids)
                if key not in self.cache:
                    self.cache[key]=build_upper_graphs(context,donor,source,recipient_regions,donor_regions,
                        plan.query_rows,bank,config=config,ct_clip=tuple(base["ct_clip"]),training_case_ids=(args.donor,))
                graph,prototype,audit=self.cache[key]
                return graph.clone(),prototype.clone(),copy.deepcopy(audit)
        def plan(selected):
            selected_rows=[rows[index] for index in selected]
            return SimpleNamespace(case_id=args.recipient,partition="inner_train",
                record_ids=tuple(row["id"] for row in selected_rows),query_rows=selected_rows)
        full=plan(range(len(rows)))
        phase="full_model_initialization";stage(phase)
        net=build_v30_model(model_config,original_snapshot_root=snapshot,stunet_checkpoint=args.checkpoint).cuda().eval()
        report["model"]=copy.deepcopy(net.v30_model_contract)
        report["full_STU_scales"]=len(net.local_encoder.dense_encoder.conv_blocks_context)
        report["input_RNG_free"]=True
        scorer=V30Scorer(net,{"inner_train":provider,"inner_val":provider},Geometry(),vessel_store=store,
            recipient_bindings=bindings,physical_candidate_batch=32,checkpoint_local_chunks=True,amp=True,budget=budget,
            prefetch_cpu_chunks=True,pin_cpu_batches=True)
        trials=[];reference=None
        for chunk in (32,64):
            phase="full128_actual_CUDA_inference";stage(phase,physical_candidate_chunk=chunk,candidates=len(rows))
            scorer.physical_candidate_batch=chunk
            torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();began=time.perf_counter()
            with torch.inference_mode():
                result=scorer([full],epoch=29,training=False)
            torch.cuda.synchronize();elapsed=time.perf_counter()-began
            scores=result.scores[0].detach().float().cpu()
            if not torch.isfinite(scores).all() or scores.shape!=(len(rows),):
                raise ValueError("Full128 real CUDA inference did not retain every finite candidate")
            difference=0. if reference is None else float((scores-reference).abs().max())
            if reference is None:reference=scores
            trials.append(dict(physical_candidate_chunk=chunk,candidates=len(rows),seconds=elapsed,
                candidates_per_second=len(rows)/elapsed,peak_VRAM_bytes=torch.cuda.max_memory_allocated(),
                max_score_difference_from_chunk32=difference,workload=result.workload))
            stage(phase,**{k:v for k,v in trials[-1].items() if k!="workload"})
        report["full128_inference"]=dict(trials=trials,scores=reference.tolist(),full_U=128,observed_P=observed,
            all_eligible_P_retained=True,trained_predictions=False)
        # Actual original hard gates are inherited by the saved 128-candidate
        # objects. Perform one exact native paste on private volume copies.
        best=int(reference[observed:].argmax())
        copied_image=recipient.image.copy();copied_label=recipient.label.copy();occupied=recipient.label==2
        old_label=copied_label[pool[best].slices].copy()
        scale,shift=paste_source(copied_image,copied_label,occupied.copy(),target_source,pool[best],tumor_label=2,
            rng=np.random.default_rng(42),intensity_scale_range=tuple(base["generation"]["intensity_scale_range"]),
            intensity_shift_range=tuple(base["generation"]["intensity_shift_range"]),blend_border=base["generation"]["blend_border"])
        added=int(np.count_nonzero((copied_label[pool[best].slices]==2)&(old_label!=2)))
        if added!=int(target_source.patch_mask.sum()):
            raise ValueError("Native exact DEBUG paste changed the verified donor voxel count")
        report["CP_DEBUG"]=dict(full_128_actual_gate_pool=True,selected_U_index=best,center=list(pool[best].center),
            exact_native_added_tumor_voxels=added,target_spacing_donor_voxels=int(target_source.patch_mask.sum()),
            actual_paste_on_private_volume_copies=True,source_files_written=False,intensity_scale=scale,intensity_shift=shift,
            shared_v25_context_used_for_GNN_scoring=True,trained_CP_quality_claim=False)
        del copied_image,copied_label,occupied,result
        phase="one_actual_CUDA_DEBUG_update";stage(phase,observed_P=observed,U=7)
        training_plan=plan(list(range(observed))+list(range(observed,observed+7)))
        net.train();scorer.physical_candidate_batch=32
        optimizer=torch.optim.AdamW(net.parameters(),lr=1e-4,weight_decay=base["training"]["weight_decay"],fused=True)
        scaler=torch.amp.GradScaler("cuda",init_scale=256.,enabled=True)
        expected={id(p) for p in net.parameters() if p.requires_grad}
        actual=[id(p) for group in optimizer.param_groups for p in group["params"]]
        if len(actual)!=len(expected) or set(actual)!=expected:
            raise ValueError("Every actual trainable parameter must appear once in DEBUG optimizer")
        groups=dict(STU=net.local_encoder.dense_encoder,L0=net.local_encoder,L1=net.patient_encoder,
            L2=net.prototype_encoder,patient_readout=net.patient_readout,population_readout=net.population_readout,
            original_score=net.score_head.original,vessel_encoder=net.score_head.vessel_encoder,
            vessel_readout=net.score_head.vessel_readout,vessel_fusion=net.score_head.vessel_fusion)
        before={name:[p.detach().clone() for p in module.parameters()] for name,module in groups.items()}
        torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();began=time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        result=scorer([training_plan],epoch=1,training=True)
        loss,parts=patient_balanced_objective(result.scores,[tuple(range(observed))],result.consistency)
        scaler.scale(loss).backward();scaler.unscale_(optimizer)
        torch.cuda.synchronize();backward_seconds=time.perf_counter()-began
        missing=[name for name,p in net.named_parameters() if p.requires_grad and p.grad is None]
        nonfinite=[name for name,p in net.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
        if missing or nonfinite:raise ValueError(dict(missing_gradients=missing,nonfinite_gradients=nonfinite))
        norms={name:float(torch.stack([p.grad.float().square().sum() for p in module.parameters()]).sum().sqrt())
               for name,module in groups.items()}
        torch.nn.utils.clip_grad_norm_(net.parameters(),base["training"]["grad_clip"],error_if_nonfinite=True)
        scaler.step(optimizer);scaler.update();torch.cuda.synchronize()
        changed={name:any(not torch.equal(old,p.detach()) for old,p in zip(before[name],module.parameters()))
                 for name,module in groups.items()}
        if not all(changed.values()):raise ValueError(dict(disconnected_optimizer_updates=changed))
        report["DEBUG_optimizer_update"]=dict(updates=1,physical_patients=1,physical_candidate_capacity=32,
            active_candidate_batch=observed+7,all_eligible_P=observed,DEBUG_U=7,gradient_accumulation=1,effective_patients=1,
            loss=float(loss.detach()),ranking=float(parts["ranking"].detach()),consistency=float(parts["consistency"].detach()),
            forward_backward_seconds=backward_seconds,total_update_seconds=time.perf_counter()-began,
            peak_VRAM_bytes=torch.cuda.max_memory_allocated(),RSS_bytes=psutil.Process().memory_info().rss,
            all_optimizer_membership_exact=True,missing_gradients=missing,nonfinite_gradients=nonfinite,
            gradient_norms=norms,module_changed=changed,precision="CUDA AMP float16; DEBUG GradScaler init256",
            workload=result.workload,not_full_training=True)
        if net.score_head._context is not None:raise ValueError("Vessel autograd context retained after DEBUG forward")
        report.update(status="passed",implementation_verified=True,static_UNIT_separate=True,
            smoke_test_complete=True,full_training_executed=False,full_evaluation_executed=False,
            real_data_used=True,remaining="Fresh untrained model; one-patient DEBUG and donor-only DEBUG bank do not establish recommendation accuracy")
        write_new(args.output/"report.json",report);stage("passed",report=str(args.output/"report.json"))
    except Exception as error:
        report.update(status="failed",failed_phase=phase,error=repr(error),traceback=traceback.format_exc())
        write_new(args.output/"report.json",report)
        stage("failed",phase=phase,error=repr(error),report=str(args.output/"report.json"))
        raise
    finally:
        for provider in providers:provider.close()


if __name__=="__main__":
    main()
