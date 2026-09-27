"""Content binding for caller-supplied graphs; tumor-only labels are excluded."""
import hashlib
import json
import numpy as np
import torch
from .placement import array_hash,validate_grid

MATERIALIZATION_CONTRACT='paired_epoch_view_seed_content_v1'


def recipient_binding(case):
    validate_grid(case)
    return dict(case_id=case.paths.case_id,image_sha256=array_hash(case.image),
                organ_sha256=array_hash(np.isin(case.label,[1,2])))


def graph_hash(record):
    h=hashlib.sha256()
    def visit(value):
        if torch.is_tensor(value):
            if value.device.type!='cpu':raise ValueError('Canonical graph binding requires CPU tensors')
            value=value.detach().numpy()
        if isinstance(value,np.ndarray):h.update(array_hash(value).encode())
        elif isinstance(value,dict):
            for key in sorted(value,key=repr):visit(key);visit(value[key])
        elif isinstance(value,(list,tuple)):
            h.update(str(len(value)).encode())
            for item in value:visit(item)
        else:h.update(json.dumps(value,sort_keys=True,allow_nan=False).encode())
        h.update(b'\0')
    visit({k:v for k,v in record.items() if k!='content_binding'})
    return h.hexdigest()


def bind_record(record,recipient):
    record['content_binding']=dict(contract=MATERIALIZATION_CONTRACT,
        recipient=recipient_binding(recipient),graph_sha256=graph_hash(record))
    return record


def validate_record(record,binding):
    saved=record.get('content_binding')
    if not isinstance(saved,dict) or saved.get('contract')!=MATERIALIZATION_CONTRACT:
        raise ValueError('Verified graph content/materialization contract required')
    if saved.get('recipient')!=binding:raise ValueError('Graph recipient CT/organ content changed')
    if saved.get('graph_sha256')!=graph_hash(record):raise ValueError('Canonical graph payload changed')
