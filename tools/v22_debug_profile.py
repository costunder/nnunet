"""Explicit review-repair DEBUG, never accepted as a production cache/run."""
from contextlib import contextmanager
import json
from pathlib import Path
from unittest.mock import patch
from tools.v222_runtime_cache import CachedPairDataset,CanonicalStore,resource_snapshot
from hiercp_v222 import v1_cache,v1_execution
from hiercp_v222.placement import GEOMETRY_CONTRACT

_configuration=v1_cache.configuration


def configuration():
    cfg,base=_configuration()
    return dict(cfg,gnn_epochs=1,batch_calibration_repeats=1),base


class RepairDataset(CachedPairDataset):
    debug_profile=True

    def __init__(self,index,partition):
        self.path=Path(index).resolve();self.root=self.path.parent
        self.meta=json.loads(self.path.read_text(encoding='utf-8'))
        if self.meta.get('format')!='v22_review_repair_DEBUG_cache' or self.meta.get('debug') is not True:
            raise ValueError('Explicit regenerated DEBUG cache required')
        if self.meta.get('geometry_contract')!=GEOMETRY_CONTRACT or self.meta['source_identity']!=v1_cache.provenance():
            raise ValueError('DEBUG graph source/geometry changed; regenerate')
        if partition not in ('inner_train','inner_val'):raise ValueError('Inner DEBUG partition required')
        self.rows=[r for r in self.meta['records'] if r['case_id'] in self.meta['split'][partition]]
        if not self.rows:raise ValueError('Empty DEBUG partition')
        key=str(self.path)
        with self.stores_lock:
            if key not in self.stores:self.stores[key]=CanonicalStore(self.root,int(resource_snapshot()['available_memory_bytes']*.20))
            self.store=self.stores[key]
        self.budget=self.store.cache.budget


@contextmanager
def installed():
    with patch.object(v1_cache,'configuration',configuration),patch.object(v1_execution,'configuration',configuration), \
         patch.object(v1_cache,'PairDataset',RepairDataset),patch.object(v1_execution,'PairDataset',RepairDataset):
        yield
