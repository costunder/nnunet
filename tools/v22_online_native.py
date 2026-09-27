"""Local native preparation with admission for every large persistent payload."""
from pathlib import Path
import time
import numpy as np
from tools.v22_online_storage import check_tree_write,save_online_case,save_online_candidate


def native_case(owner,case_id,raw):
    from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
    from custom_trainers.onlinecp_raw_resampling import prepare_case,baseline_output
    from hiercp_v222.contracts import read_json
    from tools.v22_online_rank_bank import publish_receipt
    from hiercp.preparation_runtime import Measurement
    reserve=int(owner.cfg['minimum_free_gb']*1024**3)
    with owner.native_lock,owner.locks[case_id]:
        receipt=owner.root/f'raw_receipts/{case_id}.json'
        if case_id not in owner.raw_ready:
            if receipt.exists():
                saved=read_json(receipt)
                if saved['native_sha256']!=owner.meta['native_sha256']:raise ValueError('Stale raw case receipt')
            else:
                directory=Path(owner.native['preprocessed'])/owner.plan['data_identifier']
                ds=infer_dataset_class(str(directory))(str(directory),[case_id])
                pre,seg,_,props=ds.load_case(case_id)
                # Preflight before native allocation; exact payload is checked again before writes.
                preflight=check_tree_write(owner.root,{'ct':raw.image,'gt':raw.label},reserve,kind='online_native_preparation')
                measurement=Measurement()
                with measurement:
                    lower_bound=int(np.prod(pre.shape[1:]))*10
                    if measurement.before['available_memory_bytes']<=lower_bound:
                        raise RuntimeError(f'Native persistent-array RAM lower bound exceeds available memory: {lower_bound}')
                    case=prepare_case(raw.image,raw.label,props,owner.plans,configuration_name='3d_fullres',
                        raw_spacing_xyz=raw.spacing,raw_spatial_unit=raw.image_header.get_xyzt_units()[0])
                    case['metadata']['case_id']=str(case_id)
                    actual=baseline_output(case);maximum=0.
                    if actual.shape!=pre.shape or case['baseline_seg'].shape!=seg.shape:raise ValueError('Native baseline shape mismatch')
                    for start in range(0,int(pre.shape[1]),16):
                        sl=(slice(None),slice(start,start+16),slice(None),slice(None))
                        expected=np.asarray(pre[sl]);predicted=actual[sl]
                        if not np.isfinite(expected).all() or not np.isfinite(predicted).all():raise ValueError('Nonfinite native baseline')
                        maximum=max(maximum,float(np.max(np.abs(predicted-expected))))
                        if not np.allclose(predicted,expected,rtol=1e-6,atol=1e-5):raise ValueError(f'Native CT baseline mismatch: {maximum}')
                        if not np.array_equal(case['baseline_seg'][sl],seg[sl]):raise ValueError('Native segmentation baseline mismatch')
                    del actual
                    # Admit the combined runtime + preparation write before publishing either.
                    admission=check_tree_write(owner.root,case,reserve,kind='online_native_combined')
                    ref=f'raw_cases/{case_id}.json';prep_ref=f'raw_preparation/{case_id}.json'
                    digest,runtime_storage=save_online_case(owner.root,ref,case,reserve)
                    prep_sha,prep_storage=save_online_case(owner.root,prep_ref,case['preparation'],reserve)
                    del case,pre,seg
                saved=dict(reference=ref,sha256=digest,native_sha256=owner.meta['native_sha256'],
                    preparation_reference=prep_ref,preparation_sha256=prep_sha,
                    properties={k:np.asarray(props[k]).tolist() for k in ('bbox_used_for_cropping','shape_after_cropping_and_before_resampling')},
                    baseline_ct_max_abs_error=maximum,baseline_seg_exact=True,
                    storage=[preflight,admission,runtime_storage,prep_storage],resources=measurement.report)
                publish_receipt(receipt,saved)
            owner.raw_ready[case_id]=saved
        saved=owner.raw_ready[case_id]
        case=dict(owner.store.load_case(saved['reference'],saved['sha256']))
        case['preparation']=owner.store.load_case(saved['preparation_reference'],saved['preparation_sha256'])
        return case,saved['reference'],saved['sha256'],saved['properties']


def selected_payload(owner,recipient,component,case,digest,placement):
    from custom_trainers.onlinecp_raw_resampling import prepare_candidate
    from custom_trainers.onlinecp_raw_bank import SOURCE_MAPPING_FORMAT
    started=time.perf_counter()
    candidate=prepare_candidate(case,placement.image,placement.mask,placement.anchor,placement.center)
    candidate.update(case_id=str(recipient),source_component=int(component),
        raw_target_center=list(placement.center),case_reference_sha256=digest)
    ref=f'raw_candidates/{recipient}__component_{component:03d}/0000.json'
    result,storage=save_online_candidate(owner.root,ref,candidate,int(owner.cfg['minimum_free_gb']*1024**3))
    count=int(np.count_nonzero(candidate['pasted_support']))
    audit=dict(format=SOURCE_MAPPING_FORMAT,raw_source_voxels=int(np.count_nonzero(placement.mask)),
        candidates=1,native_support_voxels=[count],zero_native_support_candidates=int(count==0),
        extrema_complement_scans=int(candidate['audit']['extrema_complement_scans']),
        seconds=time.perf_counter()-started,training_patch_size=owner.plan['patch_size'],
        full_candidate_support_retained=True,training_crop_may_intersect_partial_support=True,
        source_origin_resampling_used=False,all_selected_raw_candidates_retained=True,storage=storage)
    return ref,result,audit
