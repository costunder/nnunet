"""Content and generation admission for exact resume (not an authenticity signature).

Teachers are immutable episode state. Never regenerate a teacher or RNG at resume.
Reference memory may intentionally precede the current optimization weights.
"""
import math
import random
import numpy as np
import torch
from tools.v22_artifacts import tree_hash


def rng_contract(device='cuda'):
    devices=[]
    if device=='cuda':
        for i in range(torch.cuda.device_count()):
            p=torch.cuda.get_device_properties(i)
            devices.append(dict(logical_index=i,name=p.name,total_memory=p.total_memory,
                uuid=str(getattr(p,'uuid','unavailable')),state_shape=list(torch.cuda.get_rng_state(i).shape)))
        if not devices:raise ValueError('CUDA training requires an allocated logical device')
    elif device!='cpu':raise ValueError('Unsupported training RNG device')
    return dict(training_device=device,torch_version=str(torch.__version__),
        torch_state_shape=list(torch.get_rng_state().shape),cuda_device_count=len(devices),devices=devices)


def rng_hash(rng):
    def canonical(x):
        if isinstance(x,np.ndarray):return dict(numpy_dtype=str(x.dtype),shape=list(x.shape),bytes=x.tobytes().hex())
        if isinstance(x,(tuple,list)):return [canonical(v) for v in x]
        if isinstance(x,dict):return {k:canonical(v) for k,v in x.items()}
        return x
    return tree_hash(canonical(rng))


def validate_rng(rng,contract,digest):
    if not isinstance(contract,dict) or set(contract)!={'training_device','torch_version','torch_state_shape','cuda_device_count','devices'}:
        raise ValueError('Complete RNG device contract required')
    n=contract['cuda_device_count'];devices=contract['devices']
    if type(n)!=int or n<0 or not isinstance(devices,list) or len(devices)!=n:
        raise ValueError('CUDA RNG device count mismatch')
    if contract['training_device'] not in ('cpu','cuda') or (contract['training_device']=='cuda')!=(n>0):
        raise ValueError('Training device/RNG declaration mismatch')
    if not isinstance(rng,dict) or set(rng)!={'torch','cuda','numpy','python'} or not isinstance(rng['cuda'],list) or len(rng['cuda'])!=n:
        raise ValueError('Missing or truncated RNG generators')
    def byte_state(t,shape):
        if not torch.is_tensor(t) or t.device.type!='cpu' or t.dtype!=torch.uint8 or t.ndim!=1 or not t.numel() or list(t.shape)!=shape:
            raise ValueError('RNG byte state shape/dtype/device mismatch')
    byte_state(rng['torch'],contract['torch_state_shape'])
    try:torch.Generator(device='cpu').set_state(rng['torch'])
    except RuntimeError as error:raise ValueError('Invalid Torch CPU RNG layout') from error
    for i,(t,spec) in enumerate(zip(rng['cuda'],devices)):
        if spec.get('logical_index')!=i or not spec.get('name') or not isinstance(spec.get('total_memory'),int) or spec['total_memory']<=0:
            raise ValueError('Invalid CUDA logical device mapping')
        byte_state(t,spec.get('state_shape'))
    try:
        ns=rng['numpy']
        if not isinstance(ns,tuple) or len(ns)!=5 or ns[0]!='MT19937' or not isinstance(ns[1],np.ndarray) or ns[1].dtype!=np.uint32 or ns[1].shape!=(624,) or not 0<=ns[2]<=624 or ns[3] not in (0,1) or not math.isfinite(ns[4]):
            raise ValueError('Invalid NumPy RNG layout')
        np.random.RandomState().set_state(ns)
        ps=rng['python']
        if not isinstance(ps,tuple) or len(ps)!=3 or (ps[2] is not None and not math.isfinite(ps[2])):raise ValueError('Invalid Python RNG layout')
        random.Random().setstate(ps)
        if rng_hash(rng)!=digest:raise ValueError('RNG content hash mismatch')
    except (TypeError,IndexError,RuntimeError,OverflowError) as error:
        raise ValueError('Invalid serialized RNG state') from error


def generation(state,run_id,model):
    return dict(run_id=run_id,phase=state['phase'],epoch=state['epoch'],step=state['step'],model_sha256=tree_hash(model))


def start_memory(state,run_id,model):
    current=generation(state,run_id,model)
    if state.get('memory_work') is not None:
        if state.get('memory_work_generation')!=current:raise ValueError('Partial support generation/model mismatch')
    else:state['memory_work_generation']=current


def seal_resume(payload):
    """Called only on an immutable CPU snapshot, after resume admission succeeds."""
    state=payload['state']
    # Hash the already frozen full state_dict: parameters AND persistent buffers.
    # This is the current checkpoint model, not best/reference/episode teacher.
    payload['model_sha256']=tree_hash(payload['model'])
    payload['rng_sha256']=rng_hash(payload['rng'])
    payload['resume_integrity']={key:tree_hash(state.get(key)) for key in
        ('memory','memory_generation','memory_work','memory_work_generation','plan','plan_generation')}


def plan_generation(state,run_id,model,group):
    return dict(run_id=run_id,epoch=state['epoch'],teacher_step=state['step'],query_group=group,
        teacher_model_sha256=tree_hash(model),support_generation=state['memory_generation'].copy(),
        support_sha256=tree_hash(state['memory']))


def validate_plan(plan):
    """Check the algebraic contract without re-fitting the frozen teacher."""
    shapes={k:plan.get(k) for k in ('owners','classes','active','assignment','prototype_classes')}
    if any(not torch.is_tensor(t) or t.dtype!=torch.long or t.ndim!=1 or not t.numel() for t in shapes.values()):
        raise ValueError('Invalid cluster index dtype/shape')
    owners,classes,active,assignment,pc=(shapes[k] for k in shapes)
    n=len(owners);a=len(active);k=len(pc)
    if classes.shape!=owners.shape or (owners<0).any() or ((classes<0)|(classes>1)).any() or ((pc<0)|(pc>1)).any() or set(pc.tolist())!={0,1}:
        raise ValueError('Invalid cluster owner/class mapping')
    expected=(torch.bincount(owners*2+classes)>0).nonzero().flatten()
    if not torch.equal(active,expected) or assignment.shape!=(a,) or ((assignment<0)|(assignment>=k)).any():
        raise ValueError('Invalid active labels/assignment')
    if not torch.equal(active%2,pc[assignment]):raise ValueError('Prototype class does not match assigned observed class')
    specs={'membership':(a,k),'centers':(k,128),'mass':(k,),'log_prior':(k,),
           'alignment_row_weights':(a,),'support_embeddings':(n,128)}
    for name,shape in specs.items():
        t=plan.get(name)
        if not torch.is_tensor(t) or t.dtype!=torch.float32 or t.shape!=shape or not torch.isfinite(t).all():
            raise ValueError('Cluster tensor shape/dtype/finiteness mismatch: '+name)
    membership=torch.nn.functional.one_hot(assignment,k).float();mass=membership.sum(0)
    if not torch.equal(plan['membership'],membership) or not torch.equal(plan['mass'],mass) or (mass<=0).any():
        raise ValueError('Cluster membership/mass mismatch')
    class_mass=torch.zeros(2).scatter_add_(0,pc,mass)
    if not torch.allclose(plan['log_prior'],(mass/class_mass[pc]).log(),atol=1e-6,rtol=1e-6):raise ValueError('Cluster prior mismatch')
    if not torch.allclose(plan['alignment_row_weights'],(1/(2*torch.bincount(active%2,minlength=2).float()))[active%2]):
        raise ValueError('Cluster alignment weights mismatch')
    if not torch.allclose(plan['centers'].norm(dim=-1),torch.ones(k),atol=1e-5,rtol=1e-5):raise ValueError('Cluster centers must be unit directions')


def validate_integrity(value,rows):
    state=value['state'];manifest=value.get('resume_integrity')
    if tree_hash(value['model'])!=value.get('model_sha256'):
        raise ValueError('Current checkpoint model content hash missing or mismatched')
    keys=('memory','memory_generation','memory_work','memory_work_generation','plan','plan_generation')
    if not isinstance(manifest,dict) or set(manifest)!=set(keys):raise ValueError('Missing resume integrity manifest')
    for key in keys:
        if manifest[key]!=tree_hash(state.get(key)):raise ValueError('Resume content mismatch: '+key)
    validate_rng(value['rng'],value.get('rng_contract'),value.get('rng_sha256'))
    def check_generation(g):
        if not isinstance(g,dict) or set(g)!={'run_id','phase','epoch','step','model_sha256'} or g['run_id']!=value['run_id'] or g['phase'] not in ('initial_memory','refresh_memory','final_memory'):
            raise ValueError('Invalid support generation')
        if any(type(g[x])!=int or not 0<=g[x]<=state[x] for x in ('epoch','step')) or not isinstance(g['model_sha256'],str) or len(g['model_sha256'])!=64:
            raise ValueError('Invalid support generation counters/hash')
    if state.get('memory') is not None:check_generation(state.get('memory_generation'))
    if state.get('memory_work') is not None:
        check_generation(state.get('memory_work_generation'))
        if state['memory_work_generation']!=generation(state,value['run_id'],value['model']):raise ValueError('Partial support belongs to another model generation')
    if state['phase']=='final_memory':
        best=value.get('best_snapshot')
        if not best or tree_hash(value['model'])!=best['model_sha256']:raise ValueError('Final-memory current model differs from selected best')
    plan=state.get('plan');g=state.get('plan_generation')
    if plan is None:
        if g is not None:raise ValueError('Plan generation without an episode plan')
        return
    validate_plan(plan)
    from hiercp_v222.v1_training import groups
    from types import SimpleNamespace
    order=list(groups(SimpleNamespace(rows=rows),state['batch'],value['config']['seed'],state['epoch']))
    first=state['next_batch']-1
    while first>0 and rows[order[first-1][0]]['patient_group']==state['last_group']:first-=1
    teacher_step=state['epoch']*len(order)+first
    if not isinstance(g,dict) or g.get('run_id')!=value['run_id'] or g.get('epoch')!=state['epoch'] or g.get('query_group')!=state['last_group'] or g.get('teacher_step')!=teacher_step:
        raise ValueError('Cluster teacher generation/cursor mismatch')
    if not isinstance(g.get('teacher_model_sha256'),str) or len(g['teacher_model_sha256'])!=64:raise ValueError('Missing cluster teacher model binding')
    check_generation(g.get('support_generation'))
    # Validation already has refreshed reference memory; its old plan is historical.
    if state['phase'] in ('optimization','refresh_memory'):
        if g['support_generation']!=state['memory_generation'] or g.get('support_sha256')!=tree_hash(state['memory']):raise ValueError('Cluster support generation mismatch')
        from tools.v222_review_contracts import grouped_support
        embeddings,owners,classes=grouped_support(state['memory'],state['last_group'])
        if any(not torch.equal(plan[k],t) for k,t in zip(('support_embeddings','owners','classes'),(embeddings,owners,classes))):
            raise ValueError('Cluster episode inputs differ from reference support')
