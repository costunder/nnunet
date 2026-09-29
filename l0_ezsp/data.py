"""Original fine view, with admission-only physical metadata kept off-model."""
from copy import copy
from dataclasses import dataclass, fields
from concurrent.futures import ThreadPoolExecutor
import torch
from torch_geometric.data import Batch
from hiercp.common import stable_case_seed
from hiercp.schema import graph_config_from_dict
from hiercp_v22.sample import build_local_view
from hiercp_v22.schema import LOCAL_NODE_TYPES
from hiercp_v222.v1_local import RECORD_FORMAT, LocalBatch

@dataclass
class FineBatch(LocalBatch):
    sidecar: dict

    def to(self, device, non_blocking=True):
        base=super().to(device, non_blocking)
        return FineBatch(**vars(base),sidecar={k:{n:t.to(device,non_blocking=non_blocking) for n,t in v.items()} for k,v in self.sidecar.items()})

    def pin_memory(self):
        base=LocalBatch(copy(self.graph),self.source_patches,self.target_patches,self.source_index,self.indices).pin_memory()
        return FineBatch(**vars(base),sidecar={k:{n:t.pin_memory() for n,t in v.items()} for k,v in self.sidecar.items()})

def materialize(record, *, epoch=0, view=0):
    if record.get('format')!=RECORD_FORMAT or record.get('center_masking') is not False:
        raise ValueError('Unmodified raw CT pair required')
    branches=[]
    for key in ('source_local','target_local'):
        branch=record[key]
        nodes={k:dict(v,canonical_id=torch.arange(len(v['grid']))) for k,v in branch['nodes'].items()}
        branches.append(dict(branch,nodes=nodes))
    identity=f"{record['donor_case_id']}:{record['component_id']}:{record['center']}:{view}:{epoch}"
    seed=stable_case_seed(record['seed'],record['case_id'],identity)
    graph=build_local_view(*branches,graph_config_from_dict(record['graph_config']),seed=seed,
                          allow_empty_target_context=record.get('target_context_observed_absent',False))
    sidecar={}
    for kind in LOCAL_NODE_TYPES:
        node=graph[kind]
        sidecar[kind]={'pos_mm':node.pos_mm.float(),'stable_id':node.canonical_id.long()}
        del node.pos_mm
        del node.canonical_id
    patches=[record[k] for k in ('source_patch','target_patch')]
    if any(p.shape!=(1,48,48,48) or not bool(torch.isfinite(p).all()) for p in patches):
        raise ValueError('Finite original CT-only 48-cubed patches required')
    return graph,*patches,sidecar

def collate(items):
    payloads,indices=zip(*items)
    graphs,sources,targets,sidecars=zip(*payloads)
    unique,lookup,ids=[],{},[]
    for source in sources:
        key=(source.data_ptr(),tuple(source.shape),source.dtype)
        if key not in lookup:lookup[key]=len(unique);unique.append(source)
        ids.append(lookup[key])
    return FineBatch(Batch.from_data_list(list(graphs)),torch.stack(unique),torch.stack(targets),
        torch.tensor(ids),torch.tensor(indices),{k:{n:torch.cat([s[k][n] for s in sidecars]) for n in sidecars[0][k]} for k in LOCAL_NODE_TYPES})

def load_pairs(dataset, indices, *, workers, epoch=0):
    if workers<1 or not indices:raise ValueError('Explicit nonempty DEBUG indices and positive workers required')
    def one(i):return materialize(dataset.record(i),epoch=epoch),i
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return collate(list(pool.map(one,indices)))
