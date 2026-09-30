"""Uncoarsened fixed-view SAGE. Reuse original fine edges, never undo a quotient."""
import copy
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import torch
from .materialization import BoundFineBatch,load_pairs,verify_materialization
from .resident import signature,check_verified
from l0_ezsp.validation import validate_batch
from .encoder import RegionSAGEEncoder

MODE='uncoarsened_fixed_view_sage_v1'


class OriginalFineDataset:
    """Original graph reader with one pinned, graph-neutral model compatibility rule."""
    def __init__(self,index,partition,debug):
        import json
        from hiercp_v222.v1_cache import CACHE_FORMAT,provenance,validate_identities
        from hiercp_v222.placement import GEOMETRY_CONTRACT
        from .execution_upgrade import compatible_core
        self.path=Path(index).resolve();self.root=self.path.parent
        self.meta=json.loads(self.path.read_text(encoding='utf-8'))
        if debug:
            if self.meta.get('format')!='v22_review_repair_DEBUG_cache' or self.meta.get('debug') is not True:
                raise ValueError('Explicit regenerated DEBUG cache required')
            if self.meta.get('geometry_contract')!=GEOMETRY_CONTRACT:raise ValueError('DEBUG geometry changed')
        else:
            if self.meta.get('format')!=CACHE_FORMAT or not self.meta.get('complete') or self.meta.get('debug'):
                raise ValueError('Complete paired production cache required')
            validate_identities(self.meta['identities'],self.meta['split'])
        if not compatible_core(self.meta['source_identity'],provenance()):raise ValueError('Fine graph source changed')
        if partition not in ('inner_train','inner_val'):raise ValueError('Inner partition required')
        allowed=set(self.meta['split'][partition])
        self.rows=[r for r in self.meta['records'] if r['case_id'] in allowed]
        if not self.rows or (not debug and set(r['case_id'] for r in self.rows)!=allowed):raise ValueError('Missing cohort cases')
        self.store=None
    def record(self,i):
        if self.store is None:raise RuntimeError('Bind explicit shared RAM budget through FineLoader first')
        return self.store.record(self.rows[i])[0]
    def __len__(self):return len(self.rows)


@dataclass
class VerifiedFineBatch(BoundFineBatch):
    def to(self,device,non_blocking=True):
        check_verified(self)
        base=BoundFineBatch.to(self,device,non_blocking)
        return verified(base)


def verified(batch):
    validate_batch(batch)
    names=BoundFineBatch.__dataclass_fields__
    result=VerifiedFineBatch(**{k:getattr(batch,k) for k in names})
    result._verified_signature=signature(result)
    return result


class FineDataset:
    def __init__(self,index,partition,debug,profile_policy,original_index):
        from .training_data import RegionDataset,sha
        original=RegionDataset(index,partition,debug,profile_policy)
        self.__dict__.update(original.__dict__)
        self.original_path=Path(original_index).resolve()
        if sha(self.original_path)!=self.meta['original_cache_sha256']:
            raise ValueError('Fine cache differs from region preparation source')
        self.raw=OriginalFineDataset(self.original_path,partition,debug)
        if self.raw.rows!=self.rows:raise ValueError('Fine and region observation order/labels differ')
        self.meta=copy.deepcopy(self.meta);self.entries=copy.deepcopy(self.entries)
        self.meta['graph_representation']=MODE
        self.meta['original_cache']=str(self.original_path)
        if any(e['binding']['view']!={'epoch':self.meta['view_epoch'],'index':0,
               'policy':'fixed materialized view; never overlaid onto epoch-resampled fine nodes'} for e in self.entries):
            raise ValueError('Fixed fine view contract changed')
    def __len__(self):return len(self.rows)


class FineLoader:
    def __init__(self,ds,workers,resident_bytes,store=None):
        from tools.v222_runtime_cache import CanonicalStore
        self.ds=ds;self.workers=workers
        self.store=store or CanonicalStore(ds.raw.root,resident_bytes)
        if self.store.cache.budget!=resident_bytes:raise ValueError('Shared RAM cache budget differs')
        ds.raw.store=self.store;ds.raw.budget=resident_bytes
    def get(self,ids):
        identities=[]
        for i in ids:
            r=self.ds.rows[i]
            identities.append(self.store.path_key(r['path'],r['sha256'])[1])
            ref=r.get('shared_source')
            if ref:identities.append(self.store.path_key(ref['path'],ref['sha256'])[1])
        key=('uncoarsened_bound_batch',str(self.ds.path),self.ds.rows[ids[0]]['id'],tuple(ids),
             self.ds.meta['view_epoch'],tuple(identities))
        def build():
            batch=load_pairs(self.ds.raw,ids,workers=self.workers,epoch=self.ds.meta['view_epoch'],cache_path=self.ds.original_path)
            verify_materialization(batch,[self.ds.entries[i]['binding'] for i in ids])
            return verified(batch)
        value=self.store.cache.get(key,build);check_verified(value)
        return value
    def batches(self,order):
        with ThreadPoolExecutor(max_workers=1) as pool:
            iterator=iter(order);ids=next(iterator,None)
            if ids is None:return
            future=pool.submit(self.get,ids)
            for following in iterator:
                batch=future.result();future=pool.submit(self.get,following);yield batch
            yield future.result()


class FineTrainEncoder(RegionSAGEEncoder):
    def __init__(self,reference,*,budget,debug,contract):
        # Identical parameter names/order/initialization as the region SAGE model.
        super().__init__(reference,seed=42,resource_budget=budget,allow_unvalidated_profile=False,levels=1)
        if contract.get('graph_representation')!=MODE:raise ValueError('Explicit uncoarsened model required')
        self.training_contract=copy.deepcopy(contract)
        for block in self.core.blocks:
            for conv in block.conv.convs.values():conv.stable_spmm=True
    def get_extra_state(self):return copy.deepcopy(self.training_contract)
    def set_extra_state(self,state):
        if state!=self.training_contract:raise ValueError('Fine graph model contract changed')
    def forward(self,batch):
        if not isinstance(batch,VerifiedFineBatch):raise TypeError('Original verified fine graph required')
        check_verified(batch);self.resource_budget.check()
        self.adjacencies[0].prepare(batch.graph)
        # CT features at every original node -> three SAGE blocks -> original
        # role/shell attention readout. No partition, pooling assignment or quotient.
        output=self.core.forward_fields(batch)['fused']
        if output.shape!=(len(batch),128) or not bool(torch.isfinite(output).all()):raise ValueError('Fine embedding')
        self.resource_budget.check();return output


def transition(path,identity,network):
    from .training import FORMAT,hash_state
    from .execution_upgrade import blob_hash
    old=torch.load(path,map_location='cpu',weights_only=False)
    digest=old.pop('content_sha256',None)
    if old.get('format')!=FORMAT or digest!=hash_state(old):raise ValueError('Invalid original checkpoint')
    previous=old['identity']
    excluded={'source','graph_representation','fine_cache_sha256'}
    if {k:v for k,v in previous.items() if k not in excluded}!={k:v for k,v in identity.items() if k not in excluded}:
        raise ValueError('Removing coarsening cannot also change batch/support/loss/resources/config')
    if previous.get('graph_representation') is not None:raise ValueError('Already fine: use ordinary resume')
    if previous['source']['core']!=identity['source']['core']:raise ValueError('Core changed')
    keys=set(previous['source']['runtime'])
    revisions=('3857485','8fadbd0','10287dd')
    revision=None
    for candidate in revisions:
        if candidate!='3857485' and 'l0_regions/learning_monitor.py' in keys:continue
        if candidate=='10287dd' and 'l0_regions/execution_pipeline.py' in keys:continue
        if all(v==blob_hash(candidate,k) for k,v in previous['source']['runtime'].items()):
            revision=candidate;break
    if revision is None:raise ValueError('Unreviewed pre-transition runtime')
    state=old['state'];phase=state['phase']
    if phase not in ('optimization','refresh_memory','validation','initial_memory'):
        raise ValueError('Cannot continue completed/best-restored run as optimization')
    expected=network.state_dict();weights=old['model']
    if weights.keys()!=expected.keys():raise ValueError('Model parameter keys changed')
    contract=expected['local._extra_state'];prior=weights['local._extra_state']
    if {k:v for k,v in contract.items() if k!='graph_representation'}!=prior:raise ValueError('Partition lineage/model initialization changed')
    weights['local._extra_state']=copy.deepcopy(contract)
    receipt=dict(reviewed_revision=revision,previous_identity=previous,epoch=state['epoch'],step=state['step'],
                 next_batch=state['next_batch'],previous_phase=phase,physical_batch=state['batch'],exact_resume=False,
                 previous_best=None if state['best'] is None else {k:v for k,v in state['best'].items() if k!='weights'},
                 preserved='all model tensors, Adam, RNG, epoch, query cursor and physical batch',
                 invalidated='coarse support embeddings, cluster plan and previous best metric; rebuild complete fine memory')
    state.update(memory=None,memory_parts=[],memory_done=0,plan=None,last_group=None,best=None,selected_epoch=None,
                 phase='refresh_memory' if phase in ('refresh_memory','validation') else 'initial_memory',fine_graph_transition=receipt)
    return old,receipt


class FineCandidateEncoder:
    def __init__(self,value,budget):self.value=value;self.budget=budget
    @torch.no_grad()
    def scores(self,network,records,support,*,batch_size,workers):
        from l0_ezsp.data import materialize,collate
        from .materialization import seal_materialization
        from .resident import storage_bytes
        results=[];epoch=self.value['region']['view_epoch']
        with torch.autocast('cuda',enabled=False):
            state=network.prepare_support(*support)
            with ThreadPoolExecutor(max_workers=max(1,workers)) as pool:
                for start in range(0,len(records),batch_size):
                    ids=list(range(start,min(start+batch_size,len(records))))
                    items=list(pool.map(lambda i:(materialize(records[i],epoch=epoch,view=0),i),ids))
                    cpu=verified(seal_materialization(collate(items),[{'candidate_index':i} for i in ids]))
                    if storage_bytes(cpu)>self.value['candidate_cache_bytes']:raise MemoryError('Fine candidate batch RAM budget')
                    logits=network.predict_embeddings(network.local(cpu.to('cuda')),state)['logits'].float()
                    results.append((logits[:,1]-logits[:,0]).cpu())
        result=torch.cat(results)
        if result.shape!=(len(records),) or not bool(torch.isfinite(result).all()):raise ValueError('Incomplete fine candidate scoring')
        return result.numpy()
