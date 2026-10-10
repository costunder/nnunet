"""Score the historical Basic/v1 bank with frozen v2.3 BEST, then train on GPU3.

All 642 source payloads and their 128 ordered positions are preserved. Only
scores change. Same-patient admission is explicit: it changes donor eligibility
for downstream inference, never learned operators or the archived source files.
"""
from __future__ import annotations
import argparse, ast, copy, gc, hashlib, inspect, json, os, shutil, subprocess, sys, time, types
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

BEST_SHA = '4d25477404d0c21b93b9fa03431665f07ad6df9c038e80aff85c7a6fb8f46c25'
MODEL_SHA = '2f9717fce3af6b6494ff55eeb2f8b673f6d1037fc903e6346d95881aa2567cbf'
TRAINER = 'nnUNetTrainer_250epochs_OnlineHierCPExactArgmax'
PLANS = 'nnUNetResEncUNetMPlans'
DATASET = 'Dataset730_LiverOnlineCP_OF0'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(8*2**20),b''):h.update(part)
    return h.hexdigest()

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x')as f:json.dump(value,f,indent=2,allow_nan=False)

def budget():
    import psutil
    p=psutil.Process();rss=p.memory_info().rss+sum(c.memory_info().rss for c in p.children(recursive=True)if c.is_running())
    if len(p.cpu_affinity())!=4 or rss>48*2**30:
        raise MemoryError('GPU3 requires CPU4 and process-tree RAM<=48GiB; no model/data reduction')
    return dict(CPU_affinity=p.cpu_affinity(),RSS_GiB=rss/2**30)

def clone_with_replacements(function,replacements):
    tree=ast.parse(inspect.getsource(function));counts={k:0 for k in replacements}
    patterns={ast.dump(ast.parse(k,mode='eval').body,include_attributes=False):k for k in replacements}
    class Replace(ast.NodeTransformer):
        def visit_Compare(self,node):
            key=patterns.get(ast.dump(node,include_attributes=False))
            if key is not None:
                counts[key]+=1
                return ast.copy_location(ast.parse(replacements[key],mode='eval').body,node)
            return self.generic_visit(node)
    tree=Replace().visit(tree)
    if any(v!=1 for v in counts.values()):raise ValueError('Original admission AST changed: '+str(counts))
    ast.fix_missing_locations(tree);namespace=dict(function.__globals__)
    exec(compile(tree,inspect.getsourcefile(function)+':matched_self_CP','exec'),namespace)
    return namespace[function.__name__]

def matched_builders(local,outer_train,geometry=None):
    from hiercp_v1x.historical_patient_graph import build_external_hierarchy
    upper=clone_with_replacements(build_external_hierarchy,{
        'recipient_case.paths.case_id == donor_case.paths.case_id':'False',
        'str(donor_case.paths.case_id) not in training':'False',
        'str(recipient_case.paths.case_id) in training':'False'})
    if geometry is not None:geometry.bind(upper)
    provenance=clone_with_replacements(local._provenance,{
        "value['donor_group'] == value['recipient_group']":'False'})
    def same_provenance(value,*,donor_files,recipient_files):
        if value['donor_group']!=value['recipient_group'] or donor_files!=recipient_files:
            raise ValueError('Matched historical bank requires the same real source/recipient patient')
        return provenance(value,donor_files=donor_files,recipient_files=recipient_files)
    pair=types.FunctionType(local.pair_record.__code__,dict(local.pair_record.__globals__,_provenance=same_provenance),
        local.pair_record.__name__,local.pair_record.__defaults__,local.pair_record.__closure__)
    pair.__kwdefaults__=local.pair_record.__kwdefaults__
    def same_upper(**kwargs):
        a=kwargs['recipient_case'].paths.case_id;b=kwargs['donor_case'].paths.case_id
        if a!=b or a not in outer_train:raise ValueError('Only actual historical training self-CP is allowed')
        graph,proto,audit=upper(**kwargs)
        audit.update(inference_contract='matched_v1_same_patient_CP',donor_equals_recipient=True)
        return graph,proto,audit
    return pair,same_upper

def compare_payload(before,after):
    import numpy as np
    if set(before)!=set(after) or before['scores'].shape!=(128,) or after['scores'].shape!=(128,):
        raise ValueError('Exact historical entry fields and 128 scores required')
    for key in before:
        if key!='scores' and (before[key].dtype!=after[key].dtype or not np.array_equal(before[key],after[key])):
            raise ValueError('Historical CP payload changed: '+key)
    if not np.isfinite(after['scores']).all():raise ValueError('Finite actual neural scores required')

def score(args):
    import numpy as np,torch
    from torch_geometric.data import Batch
    from types import SimpleNamespace
    from hiercp_v1x.comparison_native_upper_cache import _load_geometry_inputs
    from hiercp_v1x import transition_v1_local as local
    from hiercp_v1x.v23_training import V23Scorer,FIELDS
    from hiercp_v1x.u_bridge_training import digest
    from hiercp_v1x.v23_edge_execution import install_l0_edge_workspace
    request=json.loads(args.gnn_request.read_text());meta=json.loads(Path(request['inventory']).read_text())
    bundle,_,_,_,preserved=_load_geometry_inputs(request['native_experiment'],'native',request['inventory'])
    runtime=local._runtime(expected_snapshot_root=bundle.source,scope_contract=bundle.scope['contract_sha256'])
    if sha(args.checkpoint)!=BEST_SHA:raise ValueError('Only the selected v2.3 BEST is admitted')
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False,mmap=True)
    if digest(saved['model'])!=MODEL_SHA or saved['content_sha256']!=digest({k:v for k,v in saved.items()if k!='content_sha256'}):
        raise ValueError('BEST model/content identity mismatch')
    from tools.local_cnn_device import select
    select(3);torch.set_num_threads(1)
    from hiercp.tensor import configure_runtime
    configure_runtime(deterministic=bundle.config['runtime']['deterministic'],allow_tf32=bundle.config['runtime']['allow_tf32'],
        cudnn_benchmark=bundle.config['runtime']['cudnn_benchmark'])
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError('Real single GPU3 required')
    free,total=torch.cuda.mem_get_info()
    if free<40*2**30:raise MemoryError('GPU3 requires original 40GiB GNN budget')
    torch.cuda.set_per_process_memory_fraction(40*2**30/total)
    net=runtime['model'].HierarchicalPyGPlacementModel(**bundle.config['model'])
    net.load_state_dict(saved['model'],strict=True)
    print('GNN_BEST_LOADED '+json.dumps(dict(path=str(args.checkpoint),sha256=BEST_SHA,
        model_sha256=MODEL_SHA,best=saved['state']['best'],next_epoch=saved['state']['epoch'],selection='full128 validation BEST',frozen=True)),flush=True)
    del saved;gc.collect()
    if sum(p.numel()for p in net.parameters())!=10434532:raise ValueError('Original complete GNN required')
    net.eval().requires_grad_(False).cuda();handle=install_l0_edge_workspace(net,1024*2**20)
    # Use the exact trained local encoding implementation with an explicit real owner.
    encoder=SimpleNamespace(net=net,amp=bundle.config['training']['amp'],checkpoint_local_chunks=False)
    old=json.loads((args.bank/'index.json').read_text());bank_sha=sha(args.bank/'index.json')
    entries=old['entries_by_case'];outer=set(meta['split']['outer_train'])
    if len(entries)!=81 or sum(map(len,entries.values()))!=642 or not set(entries)<=outer:
        raise ValueError('Exact historical 81 recipients /642 sources required')
    from hiercp.common import CasePaths,load_case,stable_case_seed,distance_to_mask_mm,extract_centered_patch,context_stats_for_local_mask
    from hiercp.region import load_or_build_patient_regions,REGION_CACHE_SEED_SALT
    from hiercp.schema import graph_config_from_dict
    from hiercp.curriculum import CandidateSpec
    from hiercp_v22.data import sources,donor_in_target_spacing
    from hiercp_v1x.bounded_scope import configure
    raw={r['case_id']:r for r in meta['raw_records']}
    base=copy.deepcopy(meta['base']);base['graph']=configure(base['graph'],10.).to_dict()
    graphconfig=graph_config_from_dict(bundle.config['graph']);clip=tuple(bundle.config['ct_clip'])
    manifest=json.loads((bundle.baseline/'manifest.json').read_text())
    region_config=graph_config_from_dict(json.loads((Path(manifest['source_experiment'])/'configs/v1.0.json').read_text())['graph'])
    target=args.output/'bank';target.mkdir(parents=True,exist_ok=False)
    write(args.output/'scoring_request.json',dict(checkpoint=str(args.checkpoint),checkpoint_sha256=BEST_SHA,
        model_sha256=MODEL_SHA,model_config=bundle.config['model'],graph_config=bundle.config['graph'],
        source_bank=str(args.bank),source_bank_sha256=bank_sha,entries=642,candidates_per_entry=128,
        total_scored_candidates=642*128,scoring_only=True,training_executed=False,
        graph_admission='same actual patient source and target; original v2.3 neural equations',
        outer_validation_used=False,resources=budget(),debug=False,helper_sha256=sha(__file__),
        geometry_helper_sha256=sha(Path(__file__).with_name('v23_matched_geometry.py')),
        geometry_CPU_workers=4,recipient_geometry_cache='immutable complete patient',distance_evaluation='exact queried EDT voxel'))
    from tqdm import tqdm
    progress=tqdm(total=642,desc='v2.3 BEST CP scoring',unit='source',mininterval=2.)
    finished=[];selected_batch=None;reused={}
    if args.reuse_output is not None:
        previous=json.loads((args.reuse_output/'scoring_request.json').read_text())
        if previous['checkpoint_sha256']!=BEST_SHA or previous['source_bank_sha256']!=bank_sha or previous['model_config']!=bundle.config['model'] or previous['graph_config']!=bundle.config['graph']:
            raise ValueError('Resume source/model/geometry contract differs')
        for receipt in sorted((args.reuse_output/'score_receipts').glob('*.json')):
            row=json.loads(receipt.read_text());relative=row['entry']
            src=args.bank/relative;completed=args.reuse_output/'bank'/relative
            if sha(src)!=row['original_sha256'] or sha(completed)!=row['derived_sha256']:
                raise ValueError('Previously scored source changed')
            with np.load(src,allow_pickle=False)as a,np.load(completed,allow_pickle=False)as b:
                compare_payload({k:a[k]for k in a.files},{k:b[k]for k in b.files})
            dest=target/relative;dest.parent.mkdir(parents=True,exist_ok=True)
            with completed.open('rb')as a,dest.open('xb')as b:shutil.copyfileobj(a,b)
            write(args.output/'score_receipts'/receipt.name,row);reused[relative]=row;finished.append(row)
        selected_batch=json.loads((args.reuse_output/'physical_batch_calibration_DEBUG.json').read_text())['selected']
        shutil.copyfile(args.reuse_output/'physical_batch_calibration_DEBUG.json',args.output/'physical_batch_calibration_DEBUG.json')
        write(args.output/'reused_scores.json',dict(source=str(args.reuse_output),entries=len(reused),verified_non_score_payloads=True))
        progress.update(len(reused))
    with ThreadPoolExecutor(max_workers=4,thread_name_prefix='matched_CP_geometry')as pool, ThreadPoolExecutor(max_workers=1,thread_name_prefix='matched_CP_prefetch')as prefetch:
      for case_id,names in entries.items():
        if all(name in reused for name in names):continue
        info=raw[case_id];case=load_case(CasePaths(case_id,Path(info['image']),Path(info['label'])))
        files=local._case_files(case)
        if files['image_sha256']!=info['image_sha256'] or files['label_sha256']!=info['label_sha256']:
            raise ValueError('Actual historical patient bytes changed')
        cache=bundle.baseline/'shared/regions'
        if not (cache/case_id).is_dir():cache=args.output/'regions'
        regions=load_or_build_patient_regions(case,liver_label=1,tumor_label=2,config=region_config,
            ct_clip=clip,seed=stable_case_seed(42,case_id,REGION_CACHE_SEED_SALT),cache_dir=cache,overwrite=False,mmap=True)
        organ,depth=regions.full_organ_mask,regions.organ_depth
        for arr in (case.image,case.label,case.spacing,case.image_affine,case.label_affine,organ,depth):arr.flags.writeable=False
        local._case_geometry(case,organ,depth)
        collection=sources(case,base['cache']['source_pad'],20.)
        from v23_matched_geometry import PatientGeometry
        geometry=PatientGeometry(case,regions,collection,pool)
        pair,upper=matched_builders(local,outer,geometry)
        source_lookup={component:i for i,(component,_)in enumerate(collection.entries)}
        occupied=distance_to_mask_mm(case.label==2,case.spacing)
        assignments,_=bundle.prototype_bank.assign(regions.region_features,top_k=graphconfig.prototype_top_m,temperature=graphconfig.prototype_temperature)
        for relative in names:
          if relative in reused:continue
          begin=time.perf_counter();src=args.bank/relative;before_sha=sha(src)
          with np.load(src,allow_pickle=False)as z:payload={k:z[k]for k in z.files}
          component=int(payload['source_component'][0]);source,_=collection[source_lookup[component]]
          centers=payload['candidate_raw_centers']
          if centers.shape!=(128,3)or len(np.unique(centers,axis=0))!=128:raise ValueError('Exact unique bank candidates required')
          prepared=local.prepare_donor(case,source,organ,depth,base)
          recipient_source,_=donor_in_target_spacing(source,case.spacing,case.spacing)
          specs=[]
          for center in centers:
            center=tuple(map(int,center));mask=recipient_source.patch_mask
            image=extract_centered_patch(case.image,center,mask.shape,pad_value=clip[0])
            coverage=extract_centered_patch(organ,center,mask.shape,pad_value=False)
            mean,std=context_stats_for_local_mask(image,coverage,mask);region=regions.region_at(center)
            specs.append(CandidateSpec(center,1,0,region,int(assignments[region,0]),float((coverage&mask).sum()/mask.sum()),
                float(depth[center]),float(occupied[center]),mean,std))
          patient,prototype,audit=upper(recipient_case=case,donor_case=case,donor_source=source,recipient_source=recipient_source,
            specs=specs,recipient_regions=regions,donor_regions=regions,bank=bundle.prototype_bank,graph_config=graphconfig,
            ct_clip=clip,training_case_ids=meta['split']['inner_train'],tumor_label=2,debug=False)
          if selected_batch is None:
            print('CALIBRATION_DEBUG_BEGIN '+json.dumps(dict(source=relative,candidates=128,
                graph_preparation_seconds=time.perf_counter()-begin,physical_batches=[32,64,128],resources=budget())),flush=True)
          def materialize(i):
            prov=dict(observation_id=f'matched:{case_id}:{component}:{i}',donor_group=case_id,recipient_group=case_id,
                debug=False,input_inventory_sha256=sha_cached_inventory,assignment_sha256=bank_sha,
                donor_image_sha256=files['image_sha256'],donor_label_sha256=files['label_sha256'],
                recipient_image_sha256=files['image_sha256'],recipient_label_sha256=files['label_sha256'])
            record=pair(case,source,case.spacing,prepared,centers[i],organ,depth,base,donor_id=case_id,provenance=prov)
            return local.materialize_pair(record,epoch=bundle.config['training']['fixed_validation_epoch']),i
          sha_cached_inventory=sha(request['inventory'])
          # Calibrate on the first complete source; every trial retains all128 queries.
          trials=[];full_scores=[];failed_trials=[]
          sizes=(32,64,128)if selected_batch is None else(selected_batch,)
          # Candidate graphs are regenerated per trial to keep RAM bounded; no query is dropped.
          def prepared_chunk(start,batch_size):
            host=time.perf_counter();items=list(pool.map(materialize,range(start,min(128,start+batch_size))))
            cpu=local.collate(items).pin_memory()
            return cpu,time.perf_counter()-host
          def full_trial(batch_size):
            torch.cuda.reset_peak_memory_stats();started=time.perf_counter();parts=[];host_seconds=0.;gpu_seconds=0.
            pending=prefetch.submit(prepared_chunk,0,batch_size)
            for start in range(0,128,batch_size):
              cpu,host_elapsed=pending.result();host_seconds+=host_elapsed;budget()
              if start+batch_size<128:pending=prefetch.submit(prepared_chunk,start+batch_size,batch_size)
              torch.cuda.synchronize();tick=time.perf_counter()
              with torch.no_grad():packed,_=V23Scorer._encode(encoder,cpu,training=False)
              torch.cuda.synchronize();gpu_seconds+=time.perf_counter()-tick;parts.append(packed);del cpu
            fields=dict(zip(FIELDS,torch.cat(parts).split(128,dim=1)))
            upper_batch=SimpleNamespace(patient_batch=Batch.from_data_list([patient]).to('cuda'),
                prototype_batch=Batch.from_data_list([prototype]).to('cuda'),counts=(128,),case_ids=(case_id,))
            with torch.no_grad(),torch.autocast('cuda',enabled=encoder.amp):values=net._score_upper(upper_batch,fields)[0].float().cpu().numpy()
            if values.shape!=(128,)or not np.isfinite(values).all():raise ValueError('Full actual finite score vector required')
            trial=dict(physical_candidate_batch=batch_size,
                peak_VRAM_GiB=torch.cuda.max_memory_allocated()/2**30,seconds=time.perf_counter()-started,
                CPU_input_seconds=host_seconds,GPU_local_seconds=gpu_seconds,resources=budget())
            del fields,parts,upper_batch,packed;gc.collect();torch.cuda.empty_cache()
            return values,trial
          for batch_size in sizes:
            try:
              values,trial=full_trial(batch_size)
            except torch.cuda.OutOfMemoryError as error:
              if selected_batch is not None:raise
              failed_trials.append(dict(physical_candidate_batch=batch_size,error=str(error),
                  peak_VRAM_GiB=torch.cuda.max_memory_allocated()/2**30))
              print('DEBUG_BATCH_CALIBRATION_OOM '+json.dumps(failed_trials[-1]),flush=True)
            else:
              full_scores.append(values);trials.append(trial)
              if selected_batch is None:print('CALIBRATION_DEBUG_TRIAL '+json.dumps(trial),flush=True)
            gc.collect();torch.cuda.empty_cache()
          if selected_batch is None:
            if not trials:raise RuntimeError('No measured physical batch fits; full model/data preserved')
            # Batching may perturb floating point reductions. Require the same chosen placement.
            if len({int(np.argmax(v))for v in full_scores})!=1:raise ValueError('Calibration changed argmax across physical batches')
            selected_batch=min(trials,key=lambda t:t['seconds'])['physical_candidate_batch']
            write(args.output/'physical_batch_calibration_DEBUG.json',dict(debug=True,source=relative,trials=trials,failed_trials=failed_trials,
                selected=selected_batch,complete128_candidates_per_trial=True,model_and_graph_reduced=False))
            values=full_scores[[t['physical_candidate_batch']for t in trials].index(selected_batch)]
          updated=dict(payload,scores=np.asarray(values,dtype=payload['scores'].dtype));compare_payload(payload,updated)
          dest=target/relative;dest.parent.mkdir(parents=True,exist_ok=True)
          with dest.open('xb')as f:np.savez(f,**updated)
          with np.load(dest,allow_pickle=False)as z:compare_payload(payload,{k:z[k]for k in z.files})
          if sha(src)!=before_sha:raise ValueError('Original bank entry changed during scoring')
          row=dict(entry=relative,original_sha256=before_sha,derived_sha256=sha(dest),selected_index=int(np.argmax(values)),
            physical_candidate_batch=selected_batch,seconds=time.perf_counter()-begin,source_component=component,case_id=case_id,
            timing=trials[-1],geometry_cache=geometry.statistics.copy())
          write(args.output/'score_receipts'/(Path(relative).stem+'.json'),row);finished.append(row)
          progress.update();progress.set_postfix(batch=selected_batch)
          print('CP_SOURCE_COMPLETE '+json.dumps(row),flush=True)
          del patient,prototype,prepared,source,payload,updated,full_scores;gc.collect();budget()
        del case,regions,collection,occupied,geometry,pair,upper;gc.collect()
    progress.close();handle.restore()
    if len(finished)!=642 or sha(args.bank/'index.json')!=bank_sha or sha(args.checkpoint)!=BEST_SHA:
        raise ValueError('Full source/model preservation failed')
    if digest(net.state_dict())!=MODEL_SHA:raise ValueError('Frozen GNN weights changed')
    derived=copy.deepcopy(old);derived['checkpoint_sha256']=BEST_SHA
    derived['matched_cp_scoring']=dict(format='v23_best_on_unchanged_v1_bank_v1',source_bank=str(args.bank),source_bank_sha256=bank_sha,
        checkpoint=str(args.checkpoint),model_sha256=MODEL_SHA,only_entry_field_changed='scores',entries=642)
    write(target/'index.json',derived)
    write(args.output/'bank_complete.json',dict(entries=642,candidates=82176,all_non_score_payloads_equal=True,
        source_bank_sha256=bank_sha,bank_index_sha256=sha(target/'index.json'),scores=finished))

def train(args):
    """Run the installed historical trainer in an isolated copied runtime."""
    import importlib.util
    if not(args.output/'bank_complete.json').is_file():raise ValueError('Full642 scored bank required before nnUNet')
    proof=json.loads((args.output/'bank_complete.json').read_text())
    if sha(args.output/'bank/index.json')!=proof['bank_index_sha256']:raise ValueError('Derived bank changed')
    for row in proof['scores']:
        if sha(args.output/'bank'/row['entry'])!=row['derived_sha256']:raise ValueError('Derived payload changed')
    package=Path(next(iter(importlib.util.find_spec('nnunetv2').submodule_search_locations)))
    private=args.output/'runtime/nnunetv2'
    shutil.copytree(package,private,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    # Original preprocessed Blosc2 reader unpacks nothing; no writes to shared data.
    dataset=private/'training/dataloading/nnunet_dataset.py'
    from v24_readonly_preprocessing import prove_readonly_dataset_source
    readonly=prove_readonly_dataset_source(dataset.read_text())
    sys.path.insert(0,str(args.code))
    from hiercp_v1x.v24_nnunet_cp import validate_baseline
    native=json.loads(args.native.read_text());baseline=native['baseline']
    validate_baseline(baseline['preprocessed'],baseline['split'])
    actual=Path(native['root'])/'nnUNet_preprocessed'/DATASET
    validate_baseline(actual,baseline['split'])
    for name,checksum in baseline['source_files_sha256'].items():
        if sha(actual/name)!=checksum:raise ValueError('Historical preprocessing metadata differs: '+name)
    write(args.output/'readonly_preprocessing_admission.json',dict(proof=readonly,path=str(actual),source_sha256=sha(dataset)))
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='GPU-b03006a3-937e-8de4-ab2f-6fbb79d9b8a4',
        nnUNet_raw=str(Path(baseline['preprocessed']).parent.parent/'nnUNet_raw'),
        nnUNet_preprocessed=str(Path(native['root'])/'nnUNet_preprocessed'),
        nnUNet_results=str(args.output/'nnUNet_results'),ONLINE_CP_BANK=str(args.output/'bank/index.json'),
        ONLINE_CP_SEED='42',nnUNet_n_proc_DA='4',nnUNet_def_n_proc='4',nnUNet_compile='false',
        PYTHONPATH=str(private.parent),PYTHONDONTWRITEBYTECODE='1',PYTHONHASHSEED='0',
        MATCHED_CP_OUTPUT=str(args.output),MATCHED_CP_EXPECTED_VAL=json.dumps(baseline['split']['outer_val']))
    entry='import runpy;runpy.run_path('+repr(str(Path(__file__).resolve()))+')["native_entry"]()'
    command=[sys.executable,'-B','-u','-c',entry,'730','3d_fullres','0','-tr',TRAINER,'-p',PLANS,'--val_best']
    write(args.output/'nnunet_started.json',dict(command=command,environment={k:env[k]for k in ('nnUNet_raw','nnUNet_preprocessed','nnUNet_results','ONLINE_CP_BANK','CUDA_VISIBLE_DEVICES')},
        epochs=250,physical_batch=2,patch=[128,128,128],CPU_workers=4,resources=budget(),validation_checkpoint='BEST',
        historical_trainer_sha256=sha(package/'training/nnUNetTrainer/nnUNetTrainer_OnlinePairedCP.py')))
    result=subprocess.run(command,env=env,check=False)
    if result.returncode:raise RuntimeError('Matched CP nnUNet failed; all logs/results preserved')
    write(args.output/'nnunet_complete.json',dict(complete=True,epochs=250,validation_checkpoint='BEST'))

def native_entry():
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
    from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class,nnUNetDatasetBlosc2
    from nnunetv2.run.run_training import run_training_entry
    original_start=nnUNetTrainer.on_train_start
    original_epoch=nnUNetTrainer.on_train_epoch_start
    original_validation=nnUNetTrainer.perform_actual_validation
    original_load=nnUNetTrainer.load_checkpoint
    loaded=[]
    output=Path(os.environ['MATCHED_CP_OUTPUT'])
    def start(self):
        if not self.was_initialized:self.initialize()
        folder=Path(os.environ['nnUNet_preprocessed'])/DATASET/self.configuration_manager.data_identifier
        if infer_dataset_class(str(folder)) is not nnUNetDatasetBlosc2:raise ValueError('Read-only Blosc2 dataset required')
        if self.num_epochs!=250 or self.batch_size!=2:raise ValueError('Historical250 epochs / physical batch2 required')
        train_ids,val_ids=self.do_split()
        if len(train_ids)!=105 or set(val_ids)!=set(json.loads(os.environ['MATCHED_CP_EXPECTED_VAL'])):
            raise ValueError('Historical105/26 patient split required')
        budget();result=original_start(self)
        write(output/'actual_nnunet_configuration.json',dict(model=str(self.network),
            parameters=sum(p.numel()for p in self.network.parameters()),
            trainable_parameters=sum(p.numel()for p in self.network.parameters()if p.requires_grad),
            train_cases=len(train_ids),validation_cases=len(val_ids),epochs=self.num_epochs,
            physical_batch=self.batch_size,gradient_accumulation=1,effective_batch=self.batch_size,
            train_steps_per_epoch=self.num_iterations_per_epoch,validation_steps_per_epoch=self.num_val_iterations_per_epoch,
            resources=budget(),debug=False))
        return result
    def epoch(self):
        budget();return original_epoch(self)
    def load(self,filename):
        result=original_load(self,filename);loaded.append(Path(filename).resolve(strict=True));return result
    def validation(self,*args,**kwargs):
        expected=Path(self.output_folder)/'checkpoint_best.pth'
        if not loaded or loaded[-1]!=expected.resolve(strict=True):raise ValueError('nnUNet BEST must be loaded before full validation')
        write(output/'nnunet_best_loaded_before_prediction.json',dict(path=str(expected),sha256=sha(expected),epoch=self.current_epoch))
        budget();return original_validation(self,*args,**kwargs)
    nnUNetTrainer.on_train_start=start
    nnUNetTrainer.on_train_epoch_start=epoch
    nnUNetTrainer.load_checkpoint=load
    nnUNetTrainer.perform_actual_validation=validation
    return run_training_entry()

def main():
    # Some imported geometry helpers probe CUDA; bind the physical device first.
    os.environ['CUDA_VISIBLE_DEVICES']='GPU-b03006a3-937e-8de4-ab2f-6fbb79d9b8a4'
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--action',choices=('score','train'),required=True)
    p.add_argument('--reuse-output',type=Path)
    for key in ('code','gnn-request','checkpoint','bank','output','native'):p.add_argument('--'+key,type=Path,required=True)
    args=p.parse_args();sys.path.insert(0,str(args.code.resolve(strict=True)));budget()
    if args.action=='score':score(args)
    else:train(args)

if __name__=='__main__':main()
