"""Typed admission for ranking resume, epoch and final artifacts."""
import hashlib
import json
import math
import torch

ARTIFACT_CONTRACT='observed_rank_artifact_v2'


def tree_hash(value):
    h=hashlib.sha256()
    def visit(x):
        if torch.is_tensor(x):
            t=x.detach().cpu().contiguous()
            h.update(str((str(t.dtype),tuple(t.shape))).encode())
            h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(x,dict):
            for key in sorted(x):visit(key);visit(x[key])
        elif isinstance(x,(list,tuple)):
            h.update(str(len(x)).encode())
            for item in x:visit(item)
        else:h.update(json.dumps(x,sort_keys=True,allow_nan=False).encode())
        h.update(b'\0')
    visit(value);return h.hexdigest()


def best_snapshot(network,epoch,metric,identity):
    weights={k:v.detach().cpu().clone() for k,v in network.state_dict().items()}
    return dict(state_dict=weights,epoch=int(epoch),metric=float(metric),
        model_sha256=tree_hash(weights),identity=identity.copy())


def validate_best(best,identity):
    if not isinstance(best,dict) or best.get('identity')!=identity:
        raise ValueError('Best snapshot run identity mismatch')
    if not 1<=best.get('epoch',0)<=identity['config']['gnn_epochs'] or not math.isfinite(best.get('metric',float('nan'))):
        raise ValueError('Invalid best epoch/metric')
    if tree_hash(best['state_dict'])!=best.get('model_sha256'):
        raise ValueError('Best snapshot hash mismatch')
    if any(t.device.type!='cpu' or not torch.isfinite(t).all() for t in best['state_dict'].values()):
        raise ValueError('Best snapshot must contain finite CPU tensors')


def validate_memory(memory,rows,identities,split,donor_pool):
    required={'embeddings','owners','classes','case_ids','patient_groups','donor_groups','record_ids'}
    if not isinstance(memory,dict) or not required<=memory.keys():raise ValueError('Incomplete support schema')
    ids=memory['record_ids'];n=len(ids);lookup={r['id']:r for r in rows}
    if not n or len(set(ids))!=n or len(lookup)!=len(rows) or set(ids)!=set(lookup):
        raise ValueError('Support record coverage/duplicates differ')
    cases=memory['case_ids'];groups=memory['patient_groups'];known=identities['cases'];train=set(split['inner_train'])
    if len(cases)!=len(set(cases)) or set(cases)!={r['case_id'] for r in rows} or not set(cases)<=train:
        raise ValueError('Support recipient partition/coverage mismatch')
    if groups!=[known[c]['patient_group'] for c in cases]:raise ValueError('Support case/group mismatch')
    owners=memory['owners'];labels=memory['classes'];embeddings=memory['embeddings']
    if owners.dtype!=torch.long or labels.dtype!=torch.long or owners.shape!=(n,) or labels.shape!=(n,):
        raise ValueError('Invalid support owner/class tensor')
    if embeddings.shape!=(n,128) or embeddings.dtype!=torch.float32 or not torch.isfinite(embeddings).all():
        raise ValueError('Invalid support embedding shape/dtype/finiteness')
    if ((owners<0)|(owners>=len(cases))).any() or ((labels<0)|(labels>1)).any():raise ValueError('Invalid owner/class value')
    if len(memory['donor_groups'])!=n:raise ValueError('Missing donor metadata')
    pool={(d['case_id'],d['component_id']) for d in donor_pool}
    if not pool or any(c not in train for c,_ in pool):raise ValueError('Held-out donor pool')
    owners=owners.cpu().tolist();labels=labels.cpu().tolist()
    for i,identifier in enumerate(ids):
        row=lookup[identifier];donor=row['donor_case_id'];recipient=row['case_id']
        if donor not in train or (donor,row['donor_component']) not in pool:raise ValueError('Held-out/unknown support donor')
        if cases[owners[i]]!=recipient or labels[i]!=row['target']:raise ValueError('Owner/record/class mismatch')
        group=known[recipient]['patient_group'];donor_group=known[donor]['patient_group']
        if row['patient_group']!=group or row['donor_group']!=donor_group or memory['donor_groups'][i]!=donor_group:
            raise ValueError('Support row identity mismatch')
        if group==donor_group:raise ValueError('Self-patient donor in support')


def validate_artifact(value,kind,*,allow_debug=False,identity=None):
    from hiercp_v222.v1_cache import provenance
    from hiercp_v222.contracts import validate_identities
    from hiercp_v222.placement import GEOMETRY_CONTRACT
    from tools.run_v222_process_runtime import runtime_identity
    from tools.v22_rank_objective import resolve_objective,OBJECTIVE
    from tools.v222_review_contracts import resolve_feature_contract,TASK_CONTRACT
    from hiercp_v222.v1_execution import RESUME_FORMAT
    from hiercp_v222.v1_training import CHECKPOINT_FORMAT
    if kind not in ('resume','epoch','final'):raise ValueError('Unknown artifact type')
    required={'format','debug','config','base','artifact_contract','artifact_kind','geometry_contract',
              'source_identity','training_objective','ranking_contract','feature_coordinates','support_task_contract','run_id'}
    required|=({'model','optimizer','rng','state','execution_policy'} if kind=='resume' else
               {'state_dict','completed_epochs','execution_policy_runtime_sha256'})
    if kind=='final':required|={'selected_epoch','memory','support_records','identities','split','donor_pool','raw_records',
                              'selected_model_sha256','memory_model_sha256','memory_sha256'}
    if not isinstance(value,dict) or not required<=value.keys():
        raise ValueError(f'Incomplete {kind} artifact: {sorted(required-set(value))}')
    if value.get('artifact_contract')!=ARTIFACT_CONTRACT or value.get('artifact_kind')!=kind:
        raise ValueError('Artifact format/type migration required')
    if value.get('format')!=(RESUME_FORMAT if kind=='resume' else CHECKPOINT_FORMAT):raise ValueError('Wrong artifact format')
    if value.get('debug') not in (True,False) or (value['debug'] and not allow_debug):raise ValueError('DEBUG weights are not production weights')
    if resolve_objective(value)!=OBJECTIVE or resolve_feature_contract(value)!='stride4':raise ValueError('Ranking/feature contract mismatch')
    if value.get('geometry_contract')!=GEOMETRY_CONTRACT or value.get('support_task_contract')!=TASK_CONTRACT:
        raise ValueError('Geometry/support contract mismatch')
    if value.get('source_identity')!=provenance():raise ValueError('Core provenance mismatch')
    if not value['debug']:
        from hiercp_v222.v1_cache import configuration
        cfg,base=configuration()
        if value.get('config')!=cfg or value.get('base')!=base:raise ValueError('Production model/epoch configuration changed')
    runtime=(value.get('execution_policy') or {}).get('runtime_sha256') if kind=='resume' else value.get('execution_policy_runtime_sha256')
    if runtime!=runtime_identity():raise ValueError('Runtime provenance mismatch')
    if identity is not None and any(value.get(k)!=v for k,v in identity.items()):raise ValueError('Resume identity mismatch')
    weights=value.get('model' if kind=='resume' else 'state_dict')
    if not isinstance(weights,dict) or not weights or any(not torch.is_tensor(t) or not torch.isfinite(t).all() for t in weights.values()):
        raise ValueError('Finite nonempty model state required')
    if kind=='resume':
        if not {'model','optimizer','rng','state'}<=value.keys():raise ValueError('Incomplete resume state')
        state=value['state']
        if state.get('phase') not in ('initial_memory','optimization','refresh_memory','validation','final_memory'):
            raise ValueError('Invalid resume phase')
        if state.get('epoch',-1)<0 or state.get('step',-1)<0 or state.get('next_batch',-1)<0:raise ValueError('Invalid resume cursor')
        if state['epoch']>value['config']['gnn_epochs']:raise ValueError('Resume epoch exceeds configured training')
        if not isinstance(value['optimizer'],dict) or not {'state','param_groups'}<=value['optimizer'].keys():
            raise ValueError('Complete Adam state required')
        rng=value['rng']
        if not isinstance(rng,dict) or not {'torch','cuda','numpy','python'}<=rng.keys() or rng['torch'].dtype!=torch.uint8:
            raise ValueError('Complete RNG state required')
        if not isinstance(rng['cuda'],list) or any(t.dtype!=torch.uint8 for t in rng['cuda']):raise ValueError('Invalid CUDA RNG state')
        prefix=state.get('memory_work');done=state.get('memory_next',0)
        if not isinstance(done,int) or done<0:raise ValueError('Invalid support cursor')
        if prefix is not None and (prefix.shape!=(done,128) or not torch.isfinite(prefix).all()):
            raise ValueError('Invalid partial support prefix')
        if state.get('best_snapshot_meta') is not None:
            if identity is None:raise ValueError('Best snapshot needs run identity')
            validate_best(value.get('best_snapshot'),identity)
            if state['best_snapshot_meta']!={k:value['best_snapshot'][k] for k in ('epoch','metric','model_sha256')}:
                raise ValueError('Best snapshot cursor mismatch')
        elif state.get('best_path') is not None:raise ValueError('External-only best checkpoint requires explicit migration')
        return value
    epochs=value.get('completed_epochs',0)
    if not isinstance(epochs,int) or not 1<=epochs<=value['config']['gnn_epochs']:raise ValueError('Invalid completed epoch')
    if kind=='epoch':return value
    if epochs!=value['config']['gnn_epochs'] or not 1<=value.get('selected_epoch',0)<=epochs:raise ValueError('Incomplete final artifact')
    validate_identities(value['identities'],value['split'])
    validate_memory(value['memory'],value['support_records'],value['identities'],value['split'],value['donor_pool'])
    if not value['debug']:
        expected={r['case_id']:len(r['positives'])+value['config']['comparison_centers_per_patient']
                  for r in value['raw_records'] if r['case_id'] in value['split']['inner_train']}
        actual={c:sum(r['case_id']==c for r in value['support_records']) for c in value['split']['inner_train']}
        if actual!=expected:raise ValueError('Incomplete production observation coverage')
    if tree_hash(value['state_dict'])!=value.get('selected_model_sha256') or value.get('memory_model_sha256')!=value.get('selected_model_sha256'):
        raise ValueError('Selected model/final memory binding mismatch')
    if any(not torch.isfinite(t).all() for t in value['state_dict'].values()):raise ValueError('Nonfinite selected model')
    if tree_hash(value['memory'])!=value.get('memory_sha256'):raise ValueError('Support content hash mismatch')
    return value
