"""Prepare each new CP event's fixed hierarchy using the frozen snapshot.

Existing rank_then_filter owns all original-mask eligibility and paste checks.
No fine graph or region bank is persisted per event; native RPC persists only the
existing selection receipt. Repeated delivered events use that receipt.
"""
import copy
from concurrent.futures import ThreadPoolExecutor
import torch
from l0_ezsp.data import materialize,collate as fine_collate
from .materialization import seal_materialization
from .preparation import binding,prepare
from .data import collate
from .training import hash_state


class CandidateEncoder:
    def __init__(self,value,reference,budget):
        self.value=value;self.reference=copy.deepcopy(reference).cuda().eval().requires_grad_(False)
        self.budget=budget
        self.event_key=None;self.event_batches=[]

    @torch.no_grad()
    def scores(self,network,records,support,*,batch_size,workers):
        profile=self.value['region']['profile'];epoch=self.value['region']['view_epoch']
        # Full record hash includes CT, donor footprint, affine/transform and
        # graph geometry. A same-ID but different event cannot reuse topology.
        hashes=[hash_state(r) for r in records];event_hash=hash_state(hashes)
        key=hash_state(dict(records=hashes,batch_size=batch_size,region=self.value['region']))
        def one(i):
            record=records[i];shared=hash_state(record['source_local'])
            row=dict(id=f"event:{event_hash}:{i}",sha256=hashes[i],shared_source=shared,
                center=record['center'],donor_case_id=record['donor_case_id'],donor_component=record['component_id'])
            receipt=dict(dataset_index=i,record_id=row['id'],record_sha256=hashes[i],cache_sha256=event_hash,
                shared_source=shared,center=row['center'],donor_case_id=row['donor_case_id'],
                donor_component=row['donor_component'],view_epoch=epoch,view_index=0)
            fine=materialize(record,epoch=epoch,view=0)
            b=binding(record=row,dataset_index=i,cache_sha256=event_hash,frozen_cnn_sha256=self.value['region']['cnn_sha256'],
                profile=profile,view_epoch=epoch,view_index=0,feature_evidence='checkpoint_partition_quality_unverified')
            return (fine,i),receipt,b
        outputs=[]
        with torch.autocast('cuda',enabled=False):
            state=network.prepare_support(*support)
            if key==self.event_key:
                for cpu in self.event_batches:
                    logits=network.predict_embeddings(network.local(cpu.to('cuda')),state)['logits'].float()
                    outputs.append(logits[:,1]-logits[:,0])
                return torch.cat(outputs).cpu().numpy()
            # New event drops only the previous topology cache, never a sample.
            # Keep one independently verified fixed hierarchy, not repeated
            # stochastic executions of the official partition for the same view.
            self.event_key=None;self.event_batches=[]
            # Explicit workers=0 is supported by the existing recommendation
            # diagnostic; production native runner already requires parallel workers.
            pool=ThreadPoolExecutor(max_workers=workers) if workers else None
            try:
                for start in range(0,len(records),batch_size):
                    ids=list(range(start,min(start+batch_size,len(records))))
                    rows=list(pool.map(one,ids)) if pool else list(map(one,ids))
                    fine=seal_materialization(fine_collate([r[0] for r in rows]),[r[1] for r in rows]).to('cuda')
                    items,_=prepare(fine,self.reference,profile,[r[2] for r in rows],self.budget,
                        allow_unvalidated_profile=self.value['debug'])
                    cpu=collate(items,ids)
                    from .resident import storage_bytes
                    if storage_bytes([*self.event_batches,cpu])>self.value['candidate_cache_bytes']:
                        raise MemoryError('Region candidate event exceeds explicit resident budget')
                    self.event_batches.append(cpu);batch=cpu.to('cuda')
                    logits=network.predict_embeddings(network.local(batch),state)['logits'].float()
                    outputs.append(logits[:,1]-logits[:,0])
                    del fine,items,batch,rows
            finally:
                if pool:pool.shutdown(wait=True)
        result=torch.cat(outputs)
        if result.shape!=(len(records),) or not bool(torch.isfinite(result).all()):raise ValueError('Incomplete region scores')
        self.event_key=key
        return result.cpu().numpy()
