"""Local CNN final weights bound to selected model, support and input contract."""
import copy,json
from pathlib import Path
import torch
from l0_regions.training import hash_state,load_checkpoint,make_model
from l0_regions.training_data import source_identity,sha,Budget
from hiercp_v222.v1_execution import atomic_torch,tree_to
from .data import Dataset
FORMAT='native_local_cnn_final_v1'

def validate(v,*,allow_debug=False):
    from hiercp_v222.contracts import validate_identities
    from hiercp_v22.donors import validate_pool
    from tools.v22_artifacts import validate_memory
    from .model import validate_config
    from l0_regions.donor_learning import POLICY,configuration,validate_rows
    if v.get('format')!=FORMAT or hash_state({k:x for k,x in v.items() if k!='content_sha256'})!=v.get('content_sha256'):raise ValueError('Local CNN artifact integrity mismatch')
    if v['debug'] and not allow_debug:raise ValueError('DEBUG weights forbidden in production')
    if v['runtime_source']!=source_identity():raise ValueError('Local CNN runtime changed')
    validate_config(v['local_cnn'])
    if v['learning_policy']!=POLICY or v['ranking_contract']!=configuration():raise ValueError('Ranking objective changed')
    validate_rows(v['support_records'])
    if v['state_dict']['local._extra_state']!=v['local_cnn']:raise ValueError('L0 range/architecture differs from weights')
    digest=hash_state(v['state_dict'])
    if digest!=v['selected_model_sha256'] or digest!=v['memory_model_sha256']:raise ValueError('Best/model/memory binding mismatch')
    if v['completed_epochs']!=(1 if v['debug'] else v['config']['gnn_epochs']) or not 1<=v['selected_epoch']<=v['completed_epochs']:raise ValueError('Incomplete learning')
    validate_identities(v['identities'],v['split']);validate_pool(v['donor_pool'],v['split'])
    validate_memory(v['memory'],v['support_records'],v['identities'],v['split'],v['donor_pool'])
    return v

def export(checkpoint,index,output):
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    saved=load_checkpoint(checkpoint,dict(saved['identity'],source=source_identity()))
    if 'local_cnn' not in saved['identity']:raise ValueError('Not local CNN training')
    s=saved['state'];ds=Dataset(index,'inner_train',saved['identity']['debug']);m=ds.meta
    if s['phase']!='complete' or saved['identity']['cache_sha256']!=sha(index):raise ValueError('Matching completed run required')
    if s['best']['model_sha256']!=hash_state(saved['model']):raise ValueError('Best weights mismatch')
    memory=copy.deepcopy(s['memory']);cases=sorted({r['case_id'] for r in ds.rows});lookup={r['id']:r for r in ds.rows}
    memory['case_ids']=cases;memory['patient_groups']=[m['identities']['cases'][c]['patient_group'] for c in cases]
    memory['owners']=torch.tensor([cases.index(lookup[r]['case_id']) for r in memory['record_ids']])
    v={k:copy.deepcopy(m[k]) for k in ('config','base','split','identities','donor_pool','raw_records','local_cnn','learning_policy','source_identity')}
    v.update(format=FORMAT,debug=m['debug'],runtime_source=source_identity(),state_dict=saved['model'],memory=memory,support_records=ds.rows,
        selected_model_sha256=s['best']['model_sha256'],memory_model_sha256=s['best']['model_sha256'],completed_epochs=s['epoch'],selected_epoch=s['selected_epoch'],
        physical_batch=s['batch'],workers=saved['identity']['workers'],resource_limits=saved['identity']['resource_limits'],
        resident_budget_bytes=saved['identity']['resident_budget_bytes'],quality_validated=False,
        ranking_contract=saved['identity']['ranking'],support_training=saved['identity'].get('support_training'))
    v['content_sha256']=hash_state(v);validate(v,allow_debug=m['debug'])
    if Path(output).exists():raise FileExistsError('Existing result preserved')
    atomic_torch(output,v);return Path(output)

def load(path,*,allow_debug=False,device='cuda'):
    if str(device)!='cuda':raise ValueError('Local CNN scorer requires CUDA')
    v=validate(torch.load(path,map_location='cpu',weights_only=False),allow_debug=allow_debug)
    # Instantiate without reseeding the caller's training RNG/runtime settings.
    from .model import LocalCNN
    from hiercp_v222.model import PromptGraphModel
    budget=Budget(**v['resource_limits'])
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        net=PromptGraphModel(v['config'],v['base'],{},local_encoder=LocalCNN(v['local_cnn'],budget=budget,dropout=v['base']['model']['dropout'])).cuda()
    net.load_state_dict(v['state_dict']);net.eval();net.checkpoint_support=False
    net.ranking_identities=v['identities'];net.ranking_split=v['split']
    return net,tree_to(v['memory'],'cuda'),v
