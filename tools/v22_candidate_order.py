"""One label-independent total order, shared by evaluation and selection."""
import json
import hashlib
import numpy as np

TIE_POLICY='score_desc_geometry_sha256_v1'


def candidate_key(case_id, donor_case_id, component_id, center):
    xyz=np.asarray(center)
    if xyz.shape!=(3,) or not np.issubdtype(xyz.dtype,np.integer):
        raise ValueError('Candidate key requires integer native coordinates')
    value=[str(case_id),str(donor_case_id),int(component_id),xyz.tolist()]
    return hashlib.sha256(json.dumps(value,separators=(',',':')).encode()).hexdigest()


def record_key(row):
    return candidate_key(row['case_id'],row['donor_case_id'],
                         row.get('donor_component',row.get('component_id')),row['center'])


def candidate_order(scores, keys):
    values=np.asarray(scores,dtype=np.float64)
    if values.ndim!=1 or not np.isfinite(values).all():
        raise ValueError('Finite one-dimensional scores required')
    if len(keys)!=len(values) or len(set(keys))!=len(keys) or any(not isinstance(k,str) or not k for k in keys):
        raise ValueError('Unique label-independent candidate keys required')
    return np.lexsort((np.asarray(keys,dtype=str),-values))
