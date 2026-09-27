"""Adam mapping and phase-specific invariants for exact ranking resume."""
import math
from types import SimpleNamespace
import torch


def optimizer_contract(network,optimizer):
    names={id(p):name for name,p in network.named_parameters()}
    groups=[];parameters={}
    for group in optimizer.param_groups:
        ordered=[]
        for p in group['params']:
            name=names[id(p)];ordered.append(name)
            parameters[name]=dict(shape=list(p.shape),dtype=str(p.dtype))
        groups.append(dict(names=ordered,options={k:v for k,v in group.items() if k!='params'}))
    return dict(algorithm='AdamW',groups=groups,parameters=parameters)


def validate_adam(optimizer,contract,step,expected_hash):
    from tools.v22_artifacts import tree_hash
    if not isinstance(optimizer,dict) or set(optimizer)!={'state','param_groups'}:
        raise ValueError('Complete Adam state/groups required')
    if type(step)!=int or step<0:raise ValueError('Invalid Adam update count')
    if tree_hash(optimizer)!=expected_hash:raise ValueError('Adam content hash mismatch')
    groups=optimizer['param_groups'];reference=contract['groups']
    if len(groups)!=len(reference):raise ValueError('Adam group count changed')
    expected={};offset=0
    for actual,group in zip(groups,reference):
        ids=list(range(offset,offset+len(group['names'])));offset+=len(ids)
        if actual.get('params')!=ids or {k:v for k,v in actual.items() if k!='params'}!=group['options']:
            raise ValueError('Adam parameter order/options changed')
        expected.update(zip(ids,group['names']))
    states=optimizer['state']
    if step==0:
        if states:raise ValueError('Initial Adam must not contain prior moments')
        return
    if set(states)!=set(expected):raise ValueError('Updated Adam is missing parameter moments')
    for group,reference_group in zip(groups,reference):
        for identifier in group['params']:
            entry=states[identifier];spec=contract['parameters'][expected[identifier]]
            keys={'step','exp_avg','exp_avg_sq'}|({'max_exp_avg_sq'} if group['amsgrad'] else set())
            if set(entry)!=keys:raise ValueError('Adam moment fields differ')
            count=entry['step']
            if not torch.is_tensor(count) or count.numel()!=1 or not torch.isfinite(count).all() or float(count)!=step:
                raise ValueError('Adam per-parameter step disagrees with resume cursor')
            for key in keys-{'step'}:
                t=entry[key]
                if not torch.is_tensor(t) or list(t.shape)!=spec['shape'] or str(t.dtype)!=spec['dtype'] or not torch.isfinite(t).all():
                    raise ValueError('Adam moment shape/dtype/finiteness mismatch')
                if key in ('exp_avg_sq','max_exp_avg_sq') and (t<0).any():raise ValueError('Negative Adam second moment')


def validate_resume_state(value,rows):
    from hiercp_v222.v1_training import groups
    from tools.v22_artifacts import validate_best
    if not rows:raise ValueError('Verified resume dataset required for cursor admission')
    state=value['state'];cfg=value['config'];epoch=state['epoch'];step=state['step'];cursor=state['next_batch']
    phase=state['phase'];total_epochs=cfg['gnn_epochs'];n=len(rows)
    if any(type(x)!=int or x<0 for x in (epoch,step,cursor)) or epoch>total_epochs:raise ValueError('Invalid resume counters')
    batch=state.get('batch');memory_batch=state.get('memory_batch')
    if type(memory_batch)!=int or memory_batch<1 or type(state.get('workers'))!=int or state['workers']<0:
        raise ValueError('Invalid saved loader policy')
    if phase=='initial_memory':
        if epoch or step or cursor or state.get('memory') is not None:raise ValueError('Initial memory has training history')
        order=[]
    else:
        if type(batch)!=int or batch<1:raise ValueError('Missing calibrated physical batch')
        order=list(groups(SimpleNamespace(rows=rows),batch,cfg['seed'],epoch))
        if cursor>len(order):raise ValueError('Batch cursor outside epoch')
        if step!=epoch*len(order)+cursor:raise ValueError('Global update count disagrees with epoch/cursor')
        if state.get('memory') is None:raise ValueError('Training/final phase needs full support memory')
    expected=[i for ids in order[:cursor] for i in ids]
    if state.get('seen')!=expected:raise ValueError('Saved seen order/coverage disagrees with cursor')
    if len(state.get('losses',[]))!=cursor or len(state.get('alignment',[]))!=cursor:
        raise ValueError('Loss history disagrees with cursor')
    if any(not math.isfinite(float(x)) for x in state.get('losses',[])+state.get('alignment',[])):
        raise ValueError('Nonfinite loss history')
    if cursor:
        last=rows[order[cursor-1][0]]['patient_group']
        if state.get('last_group')!=last or state.get('plan') is None:raise ValueError('Missing saved support episode plan')
    elif state.get('last_group') is not None or state.get('plan') is not None:
        raise ValueError('Fresh epoch must not reuse a previous episode plan')
    if phase in ('refresh_memory','validation') and (epoch>=total_epochs or cursor!=len(order)):
        raise ValueError('Refresh/validation requires a fully updated nonterminal epoch')
    if epoch==total_epochs and (phase not in ('optimization','final_memory') or cursor):
        raise ValueError('Invalid terminal phase; only fresh transition or final memory is legal')
    if phase=='final_memory' and epoch!=total_epochs:raise ValueError('Premature final memory')
    done=state.get('memory_next',0);prefix=state.get('memory_work')
    if type(done)!=int or not 0<=done<=n or (done<n and done%memory_batch):raise ValueError('Invalid partial support cursor')
    if prefix is None:
        if done:raise ValueError('Missing partial support prefix')
    elif not torch.is_tensor(prefix) or prefix.shape!=(done,128) or prefix.dtype!=torch.float32 or not torch.isfinite(prefix).all():
        raise ValueError('Invalid partial support content')
    if phase not in ('initial_memory','refresh_memory','final_memory') and (done or prefix is not None):
        raise ValueError('Partial support in non-encoding phase')
    best=value.get('best_snapshot');meta=state.get('best_snapshot_meta')
    if best is None:
        if meta is not None or state.get('best_path') is not None or state.get('best')!=float('inf') or epoch:
            raise ValueError('Missing best snapshot for validated history')
    else:
        if not 1<=best['epoch']<=epoch or state.get('best')!=best['metric'] or meta!={k:best[k] for k in ('epoch','metric','model_sha256')}:
            raise ValueError('Best metric/epoch/meta disagrees with validated history')
        if phase=='final_memory' and state.get('selected_epoch')!=best['epoch']:raise ValueError('Final selected epoch differs from best')
