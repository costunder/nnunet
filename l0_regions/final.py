"""Sealed region final artifact consumed by the existing rank/paste bridge."""
import copy
import json
from pathlib import Path
import torch
from hiercp_v222.v1_execution import atomic_torch,tree_to
from hiercp_v222.contracts import validate_identities
from hiercp_v22.donors import validate_pool
from tools.v22_artifacts import validate_memory,tree_hash
from .training import FORMAT as RESUME_FORMAT,hash_state,load_checkpoint
from .training_data import RegionDataset,source_identity,sha,Budget

FORMAT='fixed_region_sage_final_v1'


def validate(value,*,allow_debug=False):
    if value.get('format')!=FORMAT:raise ValueError('Region final format required')
    digest=value.get('content_sha256')
    if digest!=hash_state({k:v for k,v in value.items() if k!='content_sha256'}):raise ValueError('Region final contents changed')
    if value['debug'] not in (True,False) or (value['debug'] and not allow_debug):raise ValueError('DEBUG region artifact forbidden')
    if value['runtime_source']!=source_identity():raise ValueError('Region final runtime source changed')
    from hiercp_v222.v1_cache import configuration,provenance
    cfg,base=configuration()
    if value['config']!=cfg or value['base']!=base or value['source_identity']!=provenance():raise ValueError('Region configuration/core source changed')
    if not value['debug'] and value['region']['admission_failures']:raise ValueError('Rejected region profile forbidden')
    if value['completed_epochs']!=(1 if value['debug'] else cfg['gnn_epochs']):raise ValueError('Incomplete region training')
    if not 1<=value['selected_epoch']<=value['completed_epochs']:raise ValueError('Invalid selected epoch')
    if value['selected_model_sha256']!=hash_state(value['state_dict']) or value['memory_model_sha256']!=value['selected_model_sha256']:
        raise ValueError('Final memory/best model mismatch')
    if value['region']['cnn_sha256']!=tree_hash(value['frozen_cnn']):raise ValueError('Frozen partition CNN changed')
    validate_identities(value['identities'],value['split']);validate_pool(value['donor_pool'],value['split'])
    validate_memory(value['memory'],value['support_records'],value['identities'],value['split'],value['donor_pool'])
    return value


def export(checkpoint,index,output):
    """Only completed region runs; never relabel a GAT or DEBUG artifact."""
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if saved.get('format')!=RESUME_FORMAT:raise ValueError('Region checkpoint required')
    # Validate both byte integrity and the currently installed runtime identity.
    saved=load_checkpoint(checkpoint,dict(saved['identity'],source=source_identity()))
    state=saved['state'];debug=saved['identity']['debug'];ds=RegionDataset(index,'inner_train',debug)
    if saved['identity']['cache_sha256']!=sha(index) or state['phase']!='complete':raise ValueError('Complete matching region run required')
    m=ds.meta;raw_path=Path(m['original_cache'])
    if sha(raw_path)!=m['original_cache_sha256']:raise ValueError('Original observation inventory changed')
    original=json.loads(raw_path.read_text(encoding='utf-8'))
    weights=saved['model'];best=state['best']
    if best['model_sha256']!=hash_state(weights):raise ValueError('Final weights not selected best')
    memory=copy.deepcopy(state['memory']);lookup={r['id']:r for r in ds.rows}
    cases=sorted({r['case_id'] for r in ds.rows})
    memory['case_ids']=cases
    memory['patient_groups']=[original['identities']['cases'][c]['patient_group'] for c in cases]
    memory['owners']=torch.tensor([cases.index(lookup[r]['case_id']) for r in memory['record_ids']])
    value=dict(format=FORMAT,debug=debug,config=m['config'],base=m['base'],runtime_source=source_identity(),
        source_identity=m['source_identity']['core'],state_dict=weights,memory=memory,
        completed_epochs=state['epoch'],selected_epoch=state['selected_epoch'],
        selected_model_sha256=best['model_sha256'],memory_model_sha256=best['model_sha256'],
        physical_batch=state['batch'],workers=saved['identity']['workers'],resource_limits=saved['identity']['resource_limits'],
        candidate_cache_bytes=saved['identity']['resident_budget_bytes'],
        support_records=ds.rows,region={k:m[k] for k in ('profile','view_epoch','cnn_sha256','admission_failures')},
        frozen_cnn=torch.load(ds.root/'frozen_cnn.pt',weights_only=True),
        **{k:original[k] for k in ('identities','split','donor_pool','raw_records')})
    value['content_sha256']=hash_state(value);validate(value,allow_debug=debug)
    output=Path(output)
    if output.exists():raise FileExistsError('Existing final artifact preserved')
    atomic_torch(output,value);return output


def load(path,*,device='cuda',allow_debug=False):
    if str(device)!='cuda':raise ValueError('Measured region scorer requires CUDA')
    value=validate(torch.load(path,map_location='cpu',weights_only=False),allow_debug=allow_debug)
    from hiercp_v222.v1_local import V1LocalEncoder
    from hiercp_v222.model import PromptGraphModel
    from tools.v222_review_contracts import installed
    from .training import TrainEncoder
    # Scoring shares the GPU with segmentation; no process allocator or RNG
    # policy is changed here. Outer scoring_runtime owns the execution context.
    budget=Budget(**value['resource_limits'])
    with installed('stride4'):reference=V1LocalEncoder(value['base'])
    reference.dense_encoder.load_state_dict(value['frozen_cnn'],strict=True)
    contract=value['state_dict']['local._extra_state']
    network=PromptGraphModel(value['config'],value['base'],{},local_encoder=TrainEncoder(reference,
        budget=budget,debug=value['debug'],contract=contract)).cuda()
    network.load_state_dict(value['state_dict'],strict=True);network.eval()
    network.local.dense_batch_size=value['physical_batch']
    network.ranking_identities=value['identities'];network.ranking_split=value['split']
    from .recommendation import CandidateEncoder
    network.region_candidates=CandidateEncoder(value,reference,budget)
    return network,tree_to(value['memory'],'cuda'),value
