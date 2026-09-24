"""Immutable raw/native geometry only; no donor/position preselection."""
from pathlib import Path
from hiercp_v2.contracts import read_json,write_new,sha,safe_new_root,validate_native,verify_v1
from hiercp_v2.parallel import run_jobs
from hiercp_v2.volumes import volume_memory_bound
from hiercp.common import CasePaths,load_case
from custom_trainers.onlinecp_raw_bank import save_case
from tools.online_raw_bank_preparation import prepare_raw_case
from .reference import static_inputs,EXPECTED_SHA
from .runtime import source_identity
from . import FORMAT

def prepare(native_path,output):
    verify_v1();native=validate_native(read_json(native_path))
    if set(native['planning_patient_ids'])!=set(native['split']['outer_train']):raise ValueError('Planning must fit training cases only')
    from hiercp_v2.donors import reject_cross_split_duplicates
    reject_cross_split_duplicates(native['raw_records'],native['split'])
    root=safe_new_root(output);plans=read_json(native['plans']);plan=plans['configurations']['3d_fullres']
    raw={r['case_id']:r for r in native['raw_records']};rows={}
    def one(case_id):
        from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
        row=raw[case_id]
        if any(sha(row[k])!=row[k+'_sha256'] for k in ('image','label')):raise ValueError('Original CT/GT changed')
        loaded=load_case(CasePaths(case_id,Path(row['image']),Path(row['label'])))
        inputs=static_inputs(loaded.image,loaded.label)
        ref=f'reference_inputs/{case_id}.json';digest=save_case(root,ref,inputs)
        folder=Path(native['preprocessed'])/plan['data_identifier']
        dataset=infer_dataset_class(str(folder))(str(folder),[case_id])
        data,seg,_,props=dataset.load_case(case_id)
        case,case_ref,case_sha=prepare_raw_case(root,case_id,loaded.image,loaded.label,props,plans,data,seg,
            configuration_name='3d_fullres',raw_spacing=loaded.spacing,raw_spatial_unit=loaded.image_header.get_xyzt_units()[0],
            minimum_free_bytes=80*1024**3)
        preparation_ref=f'preparation/{case_id}.json';preparation_sha=save_case(root,preparation_ref,case['preparation'])
        return case_id,dict(inputs_ref=ref,inputs_sha=digest,case_ref=case_ref,case_sha=case_sha,
            preparation_ref=preparation_ref,preparation_sha=preparation_sha,raw_identity=row,component_count=inputs['component_count'])
    def commit(result):
        rows[result[0]]=result[1];print({'prepared':len(rows),'total':len(native['split']['outer_train']),'case':result[0]},flush=True)
    # Native arrays plus prefilter temporaries are larger than raw inputs.
    bound=max(volume_memory_bound(raw[c]['image']) for c in native['split']['outer_train'])*4
    run_jobs(native['split']['outer_train'],one,commit,'auto',root/'resources.json',memory_per_job=bound)
    write_new(root/'index.json',dict(format=FORMAT,complete=True,split=native['split'],cases=rows,
        native_path=str(Path(native_path).resolve()),native_sha256=sha(native_path),source_identity=source_identity(),reference_sha256=EXPECTED_SHA,
        policy='same_patient_all_components_uniform_first_valid_4000_one_paste_every_visit',
        distance_semantics='literal_reference_uint8_inversion_known_defect',debug=False))
    return root/'index.json'
