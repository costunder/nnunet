"""Record receipts originate at materialization, never from partition bindings."""
import copy
import hashlib
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from tools.v22_artifacts import tree_hash
from l0_ezsp.data import FineBatch, materialize, collate


def content_hash(batch):
    return tree_hash(dict(graph={str(k):v for k,v in batch.graph.to_dict().items()}, sidecar=batch.sidecar,
        source=batch.source_patches, target=batch.target_patches,
        source_index=batch.source_index, indices=batch.indices))


@dataclass
class BoundFineBatch(FineBatch):
    receipts: list
    materialized_sha256: str

    def to(self, device, non_blocking=True):
        base=FineBatch.to(self,device,non_blocking)
        return BoundFineBatch(**vars(base),receipts=copy.deepcopy(self.receipts),
            materialized_sha256=self.materialized_sha256)


def seal_materialization(batch, receipts):
    """Called at the loader boundary; synthetic fixtures explicitly identify themselves."""
    return BoundFineBatch(**vars(batch),receipts=copy.deepcopy(receipts),materialized_sha256=content_hash(batch))


def load_pairs(dataset, indices, *, workers, epoch, cache_path, view=0):
    if workers<1 or not indices or len(set(indices))!=len(indices):raise ValueError('Exact materialization coverage')
    cache_sha=hashlib.sha256(cache_path.read_bytes()).hexdigest()
    def one(i):
        row=dataset.rows[i];record=dataset.record(i)
        receipt=dict(dataset_index=i,record_id=row['id'],record_sha256=row['sha256'],
            cache_sha256=cache_sha,shared_source=row['shared_source'],center=row['center'],
            donor_case_id=row['donor_case_id'],donor_component=record['component_id'],
            view_epoch=epoch,view_index=view)
        return (materialize(record,epoch=epoch,view=view),i),receipt
    with ThreadPoolExecutor(max_workers=workers) as pool:results=list(pool.map(one,indices))
    return seal_materialization(collate([r[0] for r in results]),[r[1] for r in results])


def verify_materialization(batch, bindings):
    if not isinstance(batch,BoundFineBatch) or len(batch.receipts)!=len(bindings):
        raise ValueError('Independent loader record/view receipts required')
    if content_hash(batch)!=batch.materialized_sha256:raise ValueError('Materialized CT/graph changed')
    if len({r['record_id'] for r in batch.receipts})!=len(bindings):raise ValueError('Duplicate actual record')
    for i,(r,b) in enumerate(zip(batch.receipts,bindings)):
        for key in ('dataset_index','record_id','record_sha256','cache_sha256','shared_source','center','donor_case_id','donor_component'):
            if r[key]!=b[key]:raise ValueError('Materialization identity mismatch: '+key)
        if r['view_epoch']!=b['view']['epoch'] or r['view_index']!=b['view']['index']:
            raise ValueError('Materialized view mismatch')
        if int(batch.indices[i])!=r['dataset_index']:raise ValueError('Materialized dataset index mismatch')
