"""Content-bound native SGD continuation state, separate from GNN AdamW."""
import hashlib
import math
import struct
import numpy as np
import torch

FORMAT='online_segmentation_state_v2'
PROTECTED=('network_weights','optimizer_state','grad_scaler_state','logging','_best_ema',
           'current_epoch','online_rng','online_run_contract','trainer_name',
           'inference_allowed_mirroring_axes','online_optimizer_contract','online_best')


def snapshot(value,memo=None):
    if memo is None:memo={}
    if torch.is_tensor(value):
        # Native encoder is also registered under decoder: state_dict can have
        # several names for the same tensor. Preserve aliases in the immutable
        # CPU snapshot instead of copying/saving that storage repeatedly.
        key=(str(value.device),value.untyped_storage().data_ptr(),value.storage_offset(),
             tuple(value.shape),tuple(value.stride()),value.dtype)
        if key not in memo:memo[key]=value.detach().to('cpu',copy=True)
        return memo[key]
    if isinstance(value,np.ndarray):return value.copy()
    if isinstance(value,dict):return {k:snapshot(v,memo) for k,v in value.items()}
    if isinstance(value,list):return [snapshot(v,memo) for v in value]
    if isinstance(value,tuple):return tuple(snapshot(v,memo) for v in value)
    return value


def content_hash(value):
    h=hashlib.sha256()
    def put(v):
        def data(tag,b):h.update(tag);h.update(struct.pack('>Q',len(b)));h.update(b)
        if torch.is_tensor(v):
            if v.device.type!='cpu':raise ValueError('Hash requires immutable CPU snapshot')
            put(('tensor',str(v.dtype),tuple(v.shape)))
            data(b'T',memoryview(v.contiguous().reshape(-1).view(torch.uint8).numpy()))
        elif isinstance(v,np.ndarray):
            if v.dtype.hasobject:raise ValueError('Object array is not continuation state')
            put(('array',v.dtype.str,tuple(v.shape)));data(b'A',memoryview(np.ascontiguousarray(v).view(np.uint8).reshape(-1)))
        elif isinstance(v,dict):
            data(b'D',str(len(v)).encode())
            for k in sorted(v,key=lambda k:(type(k).__name__,str(k))):put(k);put(v[k])
        elif isinstance(v,(list,tuple)):
            data(b'L' if isinstance(v,list) else b'Q',str(len(v)).encode())
            for x in v:put(x)
        elif isinstance(v,np.generic):put(v.item())
        elif v is None:data(b'N',b'')
        elif isinstance(v,bool):data(b'B',str(v).encode())
        elif isinstance(v,int):data(b'I',str(v).encode())
        elif isinstance(v,float):data(b'F',struct.pack('>d',v))
        elif isinstance(v,str):data(b'S',v.encode())
        else:raise ValueError(f'Unsupported continuation value {type(v)}')
    put(value);return h.hexdigest()


def finite(value,path,seen=None):
    if seen is None:seen=set()
    if torch.is_tensor(value):
        if id(value) in seen:return
        seen.add(id(value))
        if (value.is_floating_point() or value.is_complex()) and not torch.isfinite(value).all():raise ValueError(f'Nonfinite {path}')
    elif isinstance(value,dict):
        for k,v in value.items():finite(v,f'{path}.{k}',seen)
    elif isinstance(value,(list,tuple)):
        for i,v in enumerate(value):finite(v,f'{path}.{i}',seen)
    elif isinstance(value,(float,np.floating)) and not math.isfinite(value):raise ValueError(f'Nonfinite {path}')


def optimizer_contract(model,optimizer):
    if type(optimizer) is not torch.optim.SGD:raise ValueError('Native continuation currently admits installed SGD only, not GNN AdamW')
    names={id(p):name for name,p in model.named_parameters()};groups=[]
    state=optimizer.state_dict()
    for live,saved in zip(optimizer.param_groups,state['param_groups']):
        rows=[]
        for p,index in zip(live['params'],saved['params']):
            if id(p) not in names:raise ValueError('Optimizer parameter is outside native network')
            initialized=index in state['state'] and bool(state['state'][index])
            if live['momentum'] and p.grad is not None and not initialized:raise ValueError('Used SGD parameter has no momentum state')
            rows.append(dict(id=index,name=names[id(p)],shape=list(p.shape),dtype=str(p.dtype),
                             requires_grad=p.requires_grad,initialized=initialized))
        groups.append(rows)
    return dict(optimizer='torch.optim.SGD',groups=groups)


def seal(value):
    """Only called on a new save-time snapshot; never to repair a loaded file."""
    value['online_state_integrity']=dict(format=FORMAT,sha256=content_hash({k:value[k] for k in PROTECTED}))
    return value


def validate(value,model=None,optimizer=None):
    integrity=value.get('online_state_integrity',{})
    if integrity.get('format')!=FORMAT or any(k not in value for k in PROTECTED):raise ValueError('Missing native state integrity; old checkpoint cannot be relabeled')
    if integrity.get('sha256')!=content_hash({k:value[k] for k in PROTECTED}):raise ValueError('Native model/optimizer/scaler/state content mismatch')
    for key in ('network_weights','optimizer_state','grad_scaler_state'):finite(value[key],key)
    contract=value['online_optimizer_contract'];state=value['optimizer_state']
    if contract.get('optimizer')!='torch.optim.SGD':raise ValueError('Wrong native optimizer type')
    if len(contract['groups'])!=len(state['param_groups']):raise ValueError('SGD group count mismatch')
    ids=[];names=[]
    for rows,group in zip(contract['groups'],state['param_groups']):
        if [r['id'] for r in rows]!=group['params']:raise ValueError('SGD parameter mapping mismatch')
        for r in rows:
            ids.append(r['id']);names.append(r['name'])
            weight=value['network_weights'].get(r['name'])
            if not torch.is_tensor(weight) or list(weight.shape)!=r['shape'] or str(weight.dtype)!=r['dtype']:raise ValueError('Parameter shape/dtype mismatch')
            buffers=state['state'].get(r['id'],{})
            if bool(buffers)!=r['initialized']:raise ValueError('SGD initialized-state coverage mismatch')
            if group['momentum'] and r['initialized']:
                if set(buffers)!={'momentum_buffer'}:raise ValueError('Missing SGD momentum')
                momentum=buffers['momentum_buffer']
                if momentum.shape!=weight.shape or momentum.dtype!=weight.dtype:raise ValueError('SGD momentum shape/dtype mismatch')
            elif buffers:raise ValueError('Unexpected SGD state without momentum')
    if len(set(ids))!=len(ids) or len(set(names))!=len(names) or set(state['state'])-set(ids):raise ValueError('Duplicate/foreign optimizer parameter')
    scaler=value['grad_scaler_state']
    if scaler is not None:
        if not isinstance(scaler,dict) or set(scaler)!={'scale','growth_factor','backoff_factor','growth_interval','_growth_tracker'}:raise ValueError('Incomplete GradScaler')
        if not scaler['scale']>0 or not scaler['growth_factor']>1 or not 0<scaler['backoff_factor']<1:raise ValueError('Invalid GradScaler factors')
        if type(scaler['growth_interval']) is not int or scaler['growth_interval']<1 or type(scaler['_growth_tracker']) is not int or scaler['_growth_tracker']<0:raise ValueError('Invalid GradScaler counters')
    if model is not None:
        live=model.state_dict();saved=value['network_weights']
        if live.keys()!=saved.keys() or any(live[k].shape!=saved[k].shape or live[k].dtype!=saved[k].dtype for k in live):raise ValueError('Native network schema differs')
        if optimizer is None or type(optimizer) is not torch.optim.SGD:raise ValueError('Native optimizer instance differs')
        mapping={id(p):name for name,p in model.named_parameters()}
        if len(optimizer.param_groups)!=len(contract['groups']):raise ValueError('Live SGD group count differs')
        for group,rows,saved_group in zip(optimizer.param_groups,contract['groups'],state['param_groups']):
            if [mapping[id(p)] for p in group['params']]!=[r['name'] for r in rows]:raise ValueError('Live SGD parameter mapping differs')
            for key,val in group.items():
                if key not in ('params','lr','initial_lr') and saved_group.get(key)!=val:raise ValueError(f'Live SGD {key} differs')
            if any(p.requires_grad!=r['requires_grad'] for p,r in zip(group['params'],rows)):raise ValueError('Native trainability differs')
    best=value['online_best']
    if (not isinstance(best,dict) or type(best.get('epoch')) is not int or not 1<=best['epoch']<=value['current_epoch']
            or not math.isfinite(best['metric']) or best['metric']!=value['_best_ema']):raise ValueError('Invalid completed best state')
    if best['epoch']==value['current_epoch'] and best['weights_sha256']!=content_hash(value['network_weights']):raise ValueError('Current best weights mismatch')
    return value
