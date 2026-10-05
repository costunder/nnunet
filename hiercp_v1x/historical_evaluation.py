"""Read-only full native P+128U inference with each historical score path.

The cohort/metrics are common; learned operators and training support labels
are preserved. V1/A retain annotation-derived recipient lesion graphs, so this
is explicitly NOT a blind recommendation-quality comparison.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace
import uuid

from .contracts import canonical_hash

FIELDS = ('tumor','source_context','target_context','source_relation','target_relation',
          'source_c0','source_c1','source_c2','target_c0','target_c1','target_c2','fused')
CACHE_FORMAT = 'historical_whole_validation_original_geometry_v1'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def write_new(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def verify_inventory_request(inventory_path, request):
    """Bind every arm to the parent request, including changes between workers."""
    if sha(inventory_path) != request.get('inventory_sha256'):
        raise ValueError('Native evaluation inventory differs from the bound request')


class ResourceBudget:
    def __init__(self, cuda_bytes, rss_bytes):
        self.cuda_bytes, self.rss_bytes = cuda_bytes, rss_bytes
        if min(cuda_bytes, rss_bytes) <= 0:
            raise ValueError('Explicit positive CUDA/RSS budgets required')
    def check(self):
        import torch, psutil
        if torch.cuda.memory_allocated() > self.cuda_bytes:
            raise MemoryError('Historical evaluation CUDA budget exceeded; no input/model reduction')
        if psutil.Process().memory_info().rss > self.rss_bytes:
            raise MemoryError('Historical evaluation RSS budget exceeded; no cases skipped')
    def __call__(self):
        self.check()


def assert_new_destination(output, preserved):
    output = Path(output).resolve()
    for path in preserved:
        old = Path(path).resolve(strict=True)
        if output == old or output.is_relative_to(old) or old.is_relative_to(output):
            raise ValueError('Evaluation output must be disjoint from every preserved experiment')


def original_geometry_provider(ds, *, root, workers, resident_bytes, rss_bytes,
                               prepared_cache=None, minimum_free_bytes):
    """Prepare validation geometry ONCE, then reuse it across V1/A/B.

    A complete D cache can be read directly. Otherwise only the requested whole
    validation cohort is prepared in a separately named evaluation cache. No
    training graph or D preparation request/index/lock is changed.
    """
    from .transition_v1_data import (OriginalInputProvider, ASSIGNMENT_KEYS,
        _verify_stored_files, _sampled_measurement, _validate_measurement)
    from . import transition_v1_local as local
    from .transition_preparation_storage import GraphWriter
    import torch
    root = Path(root).resolve(); root.mkdir(parents=True, exist_ok=True)
    identity = dict(format=CACHE_FORMAT, native_inventory_sha256=ds.index_sha256,
        scope_contract=ds.scope_contract, local_identity=local.source_identity(),
        base=ds.base, rows=ds.rows, debug=ds.debug, partition='inner_val',
        no_training_preparation=True)
    request = root/'request.json'
    if request.exists():
        if request.is_symlink() or json.loads(request.read_text(encoding='utf8')) != identity:
            raise ValueError('Evaluation geometry cache contract changed; old cache preserved')
    else:
        write_new(request, identity)
    if prepared_cache is not None and json.loads(Path(prepared_cache).read_text(encoding='utf8')).get('format')!=CACHE_FORMAT:
        return OriginalInputProvider(ds, workers=workers, resident_bytes=resident_bytes,
            rss_bytes=rss_bytes, cache_index=prepared_cache), dict(
                preparation_reused=True, source=str(prepared_cache), sha256=sha(prepared_cache),
                read_only=True, validation_records=len(ds), no_D_files_written=True)

    class EvaluationProvider(OriginalInputProvider):
        def _guard(self):
            super()._guard()
            if hasattr(self,'_evaluation_index') and sha(self._evaluation_index)!=self._evaluation_index_sha:
                raise ValueError('Read-only evaluation geometry index changed')
        def _load(self, row):
            stored = self._canonical[row['id']]
            selected = _verify_stored_files(root, stored)
            from hiercp_v22.storage import load_record
            record = load_record(selected, stored['path']); local.validate_record(record)
            if (record['input_provenance']['observation_id'] != row['id']
                    or record['case_id'] != row['case_id'] or record['center'] != row['center']
                    or record['donor_case_id'] != row['donor_case_id']
                    or record['component_id'] != row['donor_component']):
                raise ValueError('Historical evaluation geometry identity differs')
            return record

    if prepared_cache is not None:
        cache_path=Path(prepared_cache).resolve(strict=True)
        cache=json.loads(cache_path.read_text(encoding='utf8'))
        if (cache_path.is_symlink() or cache.get('complete') is not True
                or cache.get('request_sha256')!=canonical_hash(identity)
                or cache.get('validation_records')!=len(ds)):
            raise ValueError('Read-only evaluation geometry cache identity/coverage differs')
        root=cache_path.parent
        if json.loads((root/'request.json').read_text(encoding='utf8'))!=identity:
            raise ValueError('Evaluation geometry request differs')
        stored={r['id']:r for r in cache['records']}
        if len(stored)!=len(cache['records']) or set(stored)!={r['id'] for r in ds.rows}:
            raise ValueError('Evaluation geometry cache missing or duplicating validation rows')
        for row in ds.rows:
            item=stored[row['id']]
            if any(row[k]!=item[k] for k in ASSIGNMENT_KEYS):
                raise ValueError('Evaluation geometry cache changed actual P/U/donor')
            _verify_stored_files(root,item);_validate_measurement(item)
        provider=EvaluationProvider(ds,workers=workers,resident_bytes=resident_bytes,rss_bytes=rss_bytes)
        provider._canonical=stored;provider._evaluation_index=cache_path;provider._evaluation_index_sha=sha(cache_path)
        return provider,dict(preparation_reused=True,source=str(cache_path),sha256=sha(cache_path),
                             read_only=True,validation_records=len(ds),no_D_files_written=True)

    provider = EvaluationProvider(ds, workers=workers, resident_bytes=resident_bytes, rss_bytes=rss_bytes)
    completed = {}; started = time.perf_counter(); measurements=[]
    record_receipts=root/'completed_records'; record_receipts.mkdir(exist_ok=True)
    cases = list(dict.fromkeys(r['case_id'] for r in ds.rows))
    from tqdm import tqdm
    for case in tqdm(cases, desc='shared validation geometry', unit='case'):
        rows = [r for r in ds.rows if r['case_id']==case]
        receipt_path = root/'completed_cases'/(hashlib.sha256(case.encode()).hexdigest()+'.json')
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text(encoding='utf8'))
            if (receipt_path.is_symlink() or receipt['request_sha256'] != canonical_hash(identity)
                    or [r['id'] for r in receipt['records']] != [r['id'] for r in rows]):
                raise ValueError('Completed evaluation geometry receipt changed')
            for expected, stored in zip(rows, receipt['records']):
                if any(expected[key] != stored[key] for key in ASSIGNMENT_KEYS):
                    raise ValueError('Completed evaluation geometry changed original P/U/donor')
                _verify_stored_files(root, stored); _validate_measurement(stored)
            completed.update({r['id']:r for r in receipt['records']})
            continue
        segment='segments/'+uuid.uuid4().hex
        segment_root=root/segment; segment_root.mkdir(parents=True, exist_ok=False)
        writer=GraphWriter(segment_root, minimum_free_bytes=minimum_free_bytes)
        reused={}; start=time.perf_counter()
        for row in rows:
            saved_path=record_receipts/(hashlib.sha256(row['id'].encode()).hexdigest()+'.json')
            if not saved_path.exists():
                continue
            saved=json.loads(saved_path.read_text(encoding='utf8')); stored=saved['record']
            if (saved_path.is_symlink() or saved['request_sha256']!=canonical_hash(identity)
                    or any(row[key]!=stored[key] for key in ASSIGNMENT_KEYS)):
                raise ValueError('Partial evaluation geometry receipt changed')
            _verify_stored_files(root,stored);_validate_measurement(stored);reused[row['id']]=stored
        stored_rows=list(reused.values()); missing=[r for r in rows if r['id'] not in reused]
        for offset in range(0,len(missing),workers):
            selected=missing[offset:offset+workers]
            records=provider._records_for(selected)
            def store(item):
                row, record=item
                measurement=_sampled_measurement(local.materialize_pair(record,epoch=0))
                relative='records/'+hashlib.sha256(row['id'].encode()).hexdigest()+'.pt.gz'
                storage=writer.write(relative,record,source_key=(row['donor_case_id'],row['donor_component']))
                stored={**row,'native_input_bounds':row.get('bounds'),**storage,**measurement,'segment':segment}
                write_new(record_receipts/(hashlib.sha256(row['id'].encode()).hexdigest()+'.json'),
                          dict(request_sha256=canonical_hash(identity),record=stored))
                return stored
            with ThreadPoolExecutor(max_workers=workers) as pool:
                stored_rows.extend(pool.map(store,zip(selected,records)))
            provider._guard()
        by_id={r['id']:r for r in stored_rows};stored_rows=[by_id[r['id']] for r in rows]
        receipt=dict(request_sha256=canonical_hash(identity),records=stored_rows,
                     actual_CT=True,seconds=time.perf_counter()-start)
        write_new(receipt_path,receipt); completed.update({r['id']:r for r in stored_rows})
        measurements.append(dict(case_id=case,records=len(rows),seconds=receipt['seconds']))
    if set(completed) != {r['id'] for r in ds.rows}:
        raise ValueError('Whole validation geometry incomplete; no case skipped')
    provider._canonical=completed
    summary=dict(format=CACHE_FORMAT,complete=True,actual_CT=True,validation_records=len(ds),
        request_sha256=canonical_hash(identity),seconds=time.perf_counter()-started,
        case_preparation=measurements,records=[completed[r['id']] for r in ds.rows],
        no_D_files_written=True,production_D_cache=False)
    index=root/'index.json'
    if index.exists():
        previous=json.loads(index.read_text(encoding='utf8'))
        if previous['request_sha256'] != summary['request_sha256'] or previous['records'] != summary['records']:
            raise ValueError('Complete evaluation geometry index changed')
    else:
        write_new(index,summary)
    return provider,dict(preparation_reused=not measurements,source=str(index),sha256=sha(index),
                         validation_records=len(ds),seconds=summary['seconds'],no_D_files_written=True)


def encode_original_fields(model, batch):
    """One union of 2N genuine sampled graphs; CNN maps shared across views."""
    import torch
    if batch.graph.num_graphs != 2*len(batch):
        raise ValueError('Both genuine original sampled views required')
    source,target=model.local_encoder.encode_dense_maps(
        batch.source_patches,batch.source_index,batch.target_patches)
    owners=batch.graph_observation_index
    fields=model.local_encoder.forward_graph(batch.graph,
        source.index_select(0,owners),target.index_select(0,owners))
    if tuple(fields) != FIELDS:
        raise ValueError('Historical semantic L0 schema changed')
    values=[value.reshape(len(batch),2,128).mean(1) for value in fields.values()]
    if any(value.shape != (len(batch),128) for value in values):
        raise ValueError('Historical L0 lost query/view ordering')
    result=torch.cat(values,dim=1)
    if not bool(torch.isfinite(result).all()):
        raise ValueError('Historical L0 is nonfinite')
    return result


def unpack_fields(packed):
    if packed.ndim != 2 or packed.shape[1] != len(FIELDS)*128:
        raise ValueError('Complete semantic original L0 fields required')
    return dict(zip(FIELDS,packed.split(128,dim=1)))


def upper_graphs(bundle, provider, rows, *, region_output, region_reuse=None):
    """Original upper graph, with donor/recipient frames explicitly separated."""
    import numpy as np
    from hiercp.common import (CasePaths,load_case,stable_case_seed,extract_centered_patch,
                              context_stats_for_local_mask,distance_to_mask_mm)
    from hiercp.region import load_or_build_patient_regions, REGION_CACHE_SEED_SALT
    from hiercp.schema import graph_config_from_dict
    from hiercp.curriculum import CandidateSpec
    from hiercp_v22.data import sources,donor_in_target_spacing
    from .historical_patient_graph import build_external_hierarchy
    if len({(r['case_id'],r['donor_case_id'],r['donor_component']) for r in rows}) != 1:
        raise ValueError('Whole case must preserve its one exact native donor')
    row=rows[0]; raw={r['case_id']:r for r in provider.ds.meta['raw_records']}
    config=graph_config_from_dict(bundle.config['graph']); clip=tuple(bundle.config['ct_clip'])
    region_config=config
    if not bundle.receipt.get('debug',False):
        # The m10 run copied the original region/prototype publication. Its
        # metadata intentionally retains native graph extents, as the original
        # scope worker's install_nonlocal_binding did during training.
        manifest=json.loads((bundle.baseline/'manifest.json').read_text(encoding='utf8'))
        native_path=Path(manifest['source_experiment'])/'configs/v1.0.json'
        native=json.loads(native_path.read_text(encoding='utf8'))
        expected=dict(native['graph'])
        expected.update(adaptive_roi_margin_mm=10.,context_outer_radius_mm=10.)
        if bundle.config['graph']!=expected:
            raise ValueError('Historical region reuse requires the exact trained physical-scope adapter')
        region_config=graph_config_from_dict(native['graph'])
    def case_regions(case_id):
        info=raw[case_id]
        case=load_case(CasePaths(case_id,Path(info['image']),Path(info['label'])))
        if sha(info['image'])!=info['image_sha256'] or sha(info['label'])!=info['label_sha256']:
            raise ValueError('Actual raw CT/annotation changed before upper graph')
        old=bundle.baseline/'shared/regions'
        if region_reuse is not None:
            if bundle.receipt.get('debug') is not True:
                raise ValueError('Explicit test-region reuse is DEBUG only')
            old=Path(region_reuse).resolve(strict=True)
            if not (old/case_id).is_dir():
                raise ValueError('Explicit DEBUG region cache lacks actual case; no replacement')
        cache=old if (old/case_id).is_dir() else Path(region_output)
        regions=load_or_build_patient_regions(case,liver_label=1,tumor_label=2,config=region_config,
            ct_clip=clip,seed=stable_case_seed(42,case_id,REGION_CACHE_SEED_SALT),
            cache_dir=cache,overwrite=False,mmap=True)
        return case,regions
    recipient,regions=case_regions(row['case_id']); donor,donor_regions=case_regions(row['donor_case_id'])
    pool=sources(donor,bundle.config['cache']['source_pad'],provider.ds.meta['config']['donor_max_diameter_mm'])
    choices=[source for source,_ in pool if source.component_id==row['donor_component']]
    if len(choices)!=1:
        raise ValueError('Exact historical external donor component absent')
    source=choices[0]; target_source,_=donor_in_target_spacing(source,donor.spacing,recipient.spacing)
    assignments,_=bundle.prototype_bank.assign(regions.region_features,
        top_k=config.prototype_top_m,temperature=config.prototype_temperature)
    occupied=distance_to_mask_mm(recipient.label==2,recipient.spacing)
    specs=[]
    for item in rows:
        center=tuple(item['center']); mask=target_source.patch_mask
        ct=extract_centered_patch(recipient.image,center,mask.shape,pad_value=clip[0])
        organ=extract_centered_patch(regions.full_organ_mask,center,mask.shape,pad_value=False)
        mean,std=context_stats_for_local_mask(ct,organ,mask)
        coverage=float((organ & mask).sum()/mask.sum()); region=int(regions.region_at(center))
        specs.append(CandidateSpec(center,1,0,region,int(assignments[region,0]),coverage,
            float(regions.organ_depth[center]),float(occupied[center]),mean,std))
    graph,prototype,audit=build_external_hierarchy(recipient_case=recipient,donor_case=donor,
        donor_source=source,recipient_source=target_source,specs=specs,recipient_regions=regions,
        donor_regions=donor_regions,bank=bundle.prototype_bank,graph_config=config,
        ct_clip=clip,training_case_ids=provider.ds.meta['split']['inner_train'],
        debug=bool(bundle.receipt.get('debug',False)))
    return graph,prototype,audit


def calibration_candidates(candidates, size):
    if (not candidates or candidates!=sorted(set(candidates)) or min(candidates)<1
            or max(candidates)>size):
        raise ValueError('Explicit physical L0 batches must fit a complete-case candidate count')


def calibrate(encode, loader, ds, candidates, budget, *, arm):
    """Measure declared full L0 physical batches; never change model or cohort."""
    import torch
    by_case={case:[i for i,r in enumerate(ds.rows) if r['case_id']==case]
             for case in dict.fromkeys(r['case_id'] for r in ds.rows)}
    calibration_candidates(candidates,min(map(len,by_case.values())))
    # Graph arms use the recorded heaviest contiguous blocks; no median-only
    # promise. Final execution retains a hard resource guard on every batch.
    records=getattr(loader,'_canonical',None)
    trials=[]
    for batch in candidates:
        choices=[ids[i:i+batch] for ids in by_case.values() for i in range(0,len(ids)-batch+1,batch)]
        if records:
            ids=max(choices,key=lambda block:sum(records[ds.rows[i]['id']]['sampled_two_view_edges'] for i in block))
        else:
            ids=choices[0]
        cpu=loader.get(ids,epoch=0)
        try:
            with torch.no_grad():
                warm=encode(cpu); del warm
                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); started=time.perf_counter()
                out=encode(cpu); torch.cuda.synchronize(); elapsed=time.perf_counter()-started
                budget(); del out
            trials.append(dict(physical_batch=batch,observations_per_second=batch/elapsed,
                seconds=elapsed,peak_cuda_bytes=torch.cuda.max_memory_allocated(),accepted=True,
                input_shape=list(cpu.target_patches.shape) if arm!='C' else list(cpu.images.shape),
                local_graphs=2*batch if arm!='C' else 0,calibration_record_ids=[ds.rows[i]['id'] for i in ids]))
        except torch.cuda.OutOfMemoryError as error:
            trials.append(dict(physical_batch=batch,accepted=False,error=str(error),failure='CUDA_OOM',
                               model_and_cohort_unchanged=True))
            torch.cuda.empty_cache()
        del cpu
    valid=[r for r in trials if r['accepted']]
    if not valid:
        raise MemoryError('No explicitly requested evaluation batch fits; no hidden batch/model fallback')
    selected=max(valid,key=lambda r:r['observations_per_second'])['physical_batch']
    return selected,dict(arm=arm,unit='candidate observations',trials=trials,selected_batch=selected,
        training_batch_changed=False,L0_only_chunking=True,upper_whole_case=True)


def evaluate_historical(bundle, inventory, ds, loader, *, batch, budget, output,
                        debug=False, c=False, region_output=None, region_reuse=None):
    import torch
    from .transition_evaluation import run_scoring,PreparedL0Batch,SCORING_FORMAT
    lookup={r['id']:i for i,r in enumerate(ds.rows)}; audits=[]; timings=[]
    use_amp=bool(bundle.config['training']['amp']) if not c else bool(bundle.config['training']['amp'])
    model=bundle.net if c else bundle.model
    model.eval()
    def provider(rows):
        cpu=loader.get([lookup[r['id']] for r in rows],epoch=0)
        return PreparedL0Batch(cpu,tuple(r['id'] for r in rows),False)
    def encode(cpu):
        budget(); torch.cuda.synchronize(); started=time.perf_counter(); gpu=cpu.to('cuda')
        with torch.autocast(device_type='cuda',enabled=use_amp):
            out=model.local(gpu) if c else encode_original_fields(model,gpu)
        torch.cuda.synchronize()
        timings.append(dict(stage='L0_with_H2D',records=len(cpu),seconds=time.perf_counter()-started))
        budget(); return out.float()
    def score(features,rows):
        started=time.perf_counter()
        if c:
            values=bundle.score_case(features,rows[0]['case_id'])
        elif bundle.arm=='B':
            fields=unpack_fields(features)
            with torch.autocast(device_type='cuda',enabled=use_amp):
                values=model.half_b.score(fields['fused'],(rows[0]['case_id'],),(len(rows),))[0]
        else:
            from torch_geometric.data import Batch
            graph,prototype,audit=upper_graphs(bundle,loader,rows,
                region_output=Path(region_output) if region_output is not None else Path(output)/'regions',
                region_reuse=region_reuse)
            audits.append(audit)
            upper=SimpleNamespace(patient_batch=Batch.from_data_list([graph]).to('cuda'),
                prototype_batch=Batch.from_data_list([prototype]).to('cuda'),counts=(len(rows),),
                case_ids=(rows[0]['case_id'],))
            with torch.autocast(device_type='cuda',enabled=use_amp):
                values=model._score_upper(upper,unpack_fields(features))[0]
        torch.cuda.synchronize(); budget()
        timings.append(dict(stage='upper_whole_case',case_id=rows[0]['case_id'],records=len(rows),
                            seconds=time.perf_counter()-started))
        return dict(scores=values.float(),contract=dict(format=SCORING_FORMAT,
            scored_record_ids=[r['id'] for r in rows],l0_only_chunking=True,upper_chunking=False,
            upper_execution='single_joint_case',query_GT_in_forward=False,upper_invocations=1,
            class_target_argument_passed=False,
            annotation_derived_recipient_inputs=not c and bundle.arm in ('V1','A')))
    start=time.perf_counter()
    report=run_scoring(inventory,provider,encode,score,l0_batch_size=batch,
        case_ids=list(dict.fromkeys(r['case_id'] for r in ds.rows)) if debug else None,debug=debug)
    report.update(arm='C' if c else bundle.arm,checkpoint=bundle.receipt if c else bundle.checkpoint,
        actual_CUDA=True,raw_CT_execution_verified=True,callback_internal_execution_verified=True,
        training_started=False,optimizer_updates=0,full_evaluation=not debug,
        quality_verified=False,blind_recommendation_quality_verified=False,
        annotation_derived_recipient_graph=not c and bundle.arm in ('V1','A'),
        query_GT_in_forward_interpretation='class target fields excluded; V1/A retain recipient lesion annotations',
        support_policy='original_anchor_curriculum_full_train_bank' if c or bundle.arm=='B' else 'original_train_only_prototype_bank',
        original_upper_score_path_preserved=True,external_donor_adapter=audits,
        elapsed_seconds=time.perf_counter()-start,stage_timings=timings,
        peak_cuda_bytes=torch.cuda.max_memory_allocated(),
        CP_started=False,nnunet_started=False)
    preserved=bundle.receipt['source_artifacts_sha256'] if c else bundle.checkpoint.get('files_preserved',{})
    if any(sha(path)!=digest for path,digest in preserved.items()):
        raise ValueError('Old checkpoint/evidence changed during evaluation')
    report['old_evidence_preserved']=True
    return report,encode
