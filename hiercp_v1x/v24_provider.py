"""Fresh recipient-GT-free CPU canonical records and exact sampled-view reuse.

Only CPU geometry/views are cached. Persistent parallel readers, shared RSS
accounting and bounded cache eviction change residency, never data inclusion.
"""
from concurrent.futures import ThreadPoolExecutor, wait
import copy
import json
from pathlib import Path
import threading
import time

import psutil

from .contracts import canonical_hash
from hiercp_v22.contracts import sha
from .transition_v1_data import _ResidentStorageLedger, _StorageIntervalUnion, _AccountedCache
from .v24_inputs import LOCAL_FORMAT, materialize_pair, collate, query_inputs, tensor_digest

FORMAT = 'v24_complete_recipient_GT_free_native_canonical_v1'


class V24InputCoordinator:
    def __init__(self, rss_bytes):
        if type(rss_bytes) is not int or rss_bytes <= 0:
            raise ValueError('Actual shared process RSS budget required')
        self.rss_bytes, self.lock, self.providers = rss_bytes, threading.RLock(), []

    def trim(self, *, strict=True):
        with self.lock:return self._trim_locked(strict=strict)

    def _trim_locked(self, *, strict):
        target = self.rss_bytes-min(self.rss_bytes//64,3*2**30)
        for category in ('view','record'):
            while psutil.Process().memory_info().rss > target:
                changed = False
                for provider in self.providers:
                    changed |= provider.evict(category)
                    if psutil.Process().memory_info().rss <= target: break
                if not changed: break
        if strict and psutil.Process().memory_info().rss > self.rss_bytes:
            raise MemoryError('Full active GT-free input exceeds shared RSS; no graph or data reduction')


def _payload_owner(payload):
    graphs, source, target = payload
    return tuple(graph.to_dict() for graph in graphs), source, target


class V24InputProvider:
    recipient_GT_used_in_forward = False

    def __init__(self, dataset, index, *, workers, resident_bytes, coordinator, fixed_validation_epoch=29):
        if workers < 4 or not 0 < resident_bytes < coordinator.rss_bytes or fixed_validation_epoch != 29:
            raise ValueError('Explicit4+ CPU workers/resident/shared RSS/fixed29 required')
        self.ds, self.workers, self.resident_bytes = dataset,workers,resident_bytes
        self.coordinator, self.rss_bytes, self.fixed_validation_epoch = coordinator,coordinator.rss_bytes,fixed_validation_epoch
        self.index = Path(index).resolve(strict=True); self.root = self.index.parent
        self.index_sha256 = sha(self.index)
        self.meta = json.loads(self.index.read_text(encoding='utf8'))
        if (self.meta.get('format') != FORMAT or self.meta.get('complete') is not True
                or self.meta.get('recipient_GT_used_in_forward') is not False
                or self.meta.get('debug') is not False):
            raise ValueError('Complete fresh GT-free canonical index required; legacy cache refused')
        self.records = {row['id']:row for row in self.meta['records']}
        if len(self.records) != len(self.meta['records']) or set(self.records) != {row['id'] for row in dataset.meta['records']}:
            raise ValueError('Fresh canonical cache must contain every14102 native observation once')
        for row in dataset.rows:
            if self.records[row['id']]['query'] != query_inputs([row])[0]:
                raise ValueError('GT-free canonical actual native query/donor changed')
        self._ledger = _ResidentStorageLedger(); self._ledger._groups['view'] = _StorageIntervalUnion()
        self._records = _AccountedCache(self._ledger,'record',lambda value:value)
        self._views = _AccountedCache(self._ledger,'view',lambda value:_payload_owner(value))
        self._protected_ids,self._protected_views = set(),set()
        self._file_proofs = {str(self.index):self._stat(self.index)}
        self._executor = ThreadPoolExecutor(max_workers=workers,thread_name_prefix='v24_native_CPU')
        self._closed = False
        self._stats = dict(get_calls=0, record_loads=0, sampled_pair_hits=0,sampled_pair_misses=0,
            sampled_pairs_materialized=0,record_load_seconds=0.,materialize_seconds=0.,collate_seconds=0.,
            get_seconds=0.,view_evictions=0,record_evictions=0)
        with self.coordinator.lock: self.coordinator.providers.append(self)

    @staticmethod
    def _stat(path):
        value=Path(path).stat()
        return value.st_dev,value.st_ino,value.st_size,value.st_mtime_ns,value.st_ctime_ns

    def _file(self, relative, checksum):
        path=(self.root/relative).resolve(strict=True)
        if not path.is_relative_to(self.root) or path.is_symlink():
            raise ValueError('Canonical reference escapes fresh input namespace')
        proof=self._stat(path); key=str(path)
        if key not in self._file_proofs:
            if sha(path) != checksum: raise ValueError('New canonical file SHA differs')
            self._file_proofs[key]=proof
        elif self._file_proofs[key] != proof:
            raise ValueError('Admitted fresh canonical file was replaced or changed')
        return path

    def _load(self, row):
        from hiercp_v22.storage import load_record
        entry=self.records[row['id']]
        self._file(entry['path'],entry['sha256'])
        self._file(entry['shared_source']['path'],entry['shared_source']['sha256'])
        value=load_record(self.root,entry['path'])
        if (value.get('format') != LOCAL_FORMAT or value.get('recipient_GT_used_in_forward') is not False
                or value.get('P_U_labels_in_forward') is not False
                or value['observation_id'] != row['id'] or value['tensor_sha256'] != entry['tensor_sha256']
                or query_inputs([dict(id=value['observation_id'],case_id=value['case_id'],center=value['center'],
                    donor_case_id=value['donor_case_id'],donor_component=value['component_id'])])[0] != entry['query']):
            raise ValueError('Actual new GT-free tensor/query contract differs')
        return value

    def _parallel(self, function, items):
        futures=[self._executor.submit(function,item) for item in items]
        try: return [future.result() for future in futures]
        finally:
            for future in futures: future.cancel()
            wait(futures)

    def evict(self, category):
        cache=self._views if category=='view' else self._records
        protected=self._protected_views if category=='view' else self._protected_ids
        key=next((key for key in cache if key not in protected),None)
        if key is None: return False
        cache.pop(key); self._stats[category+'_evictions']+=1
        return True

    def _room(self):
        for category in ('view','record'):
            while self._ledger.bytes('all')>self.resident_bytes and self.evict(category): pass
        if self._ledger.bytes('all')>self.resident_bytes:
            raise MemoryError('Complete active v24 canonical/view chunk exceeds resident budget')
        self.coordinator.trim()

    def get(self, ids, *, epoch=0):
        with self.coordinator.lock:
            if self._closed: raise RuntimeError('Closed v24 CPU provider')
            if (not isinstance(ids,(list,tuple)) or not ids or len(ids)!=len(set(ids))
                    or any(type(index)is not int or not 0<=index<len(self.ds.rows) for index in ids)
                    or type(epoch)is not int or not 0<=epoch<=40):
                raise ValueError('Complete unique actual observation indices and real epoch required')
            if self.ds.partition=='inner_val' and epoch not in (0,29):
                raise ValueError('Validation views must stay fixed29')
            if self._stat(self.index)!=self._file_proofs[str(self.index)]:
                raise ValueError('Complete new canonical index changed')
            began=time.perf_counter(); rows=[self.ds.rows[index] for index in ids]
            for row in rows:
                entry=self.records[row['id']]
                self._file(entry['path'],entry['sha256'])
                self._file(entry['shared_source']['path'],entry['shared_source']['sha256'])
            self._protected_ids={row['id'] for row in rows}
            try:
                missing=[row for row in rows if row['id'] not in self._records]
                started=time.perf_counter(); built=self._parallel(self._load,missing)
                self._stats['record_load_seconds']+=time.perf_counter()-started
                self._stats['record_loads']+=len(missing)
                for row,value in zip(missing,built): self._records[row['id']]=value
                records=[self._records[row['id']] for row in rows]
                for row in rows: self._records.move_to_end(row['id'])
                # Train fixed29 probes can recur, but train epoch views are not
                # memoized: original independent epoch/view seeds always run.
                cache_views=self.ds.partition=='inner_val'
                keys=[(row['id'],epoch,record['tensor_sha256']) for row,record in zip(rows,records)]
                self._protected_views=set(keys)
                payloads=[None]*len(rows); missing=[]
                for position,(key,record) in enumerate(zip(keys,records)):
                    payload=self._views.get(key) if cache_views else None
                    if payload is None:
                        missing.append((position,key,record)); self._stats['sampled_pair_misses']+=1
                    else:
                        payloads[position]=payload; self._views.move_to_end(key)
                        self._stats['sampled_pair_hits']+=1
                started=time.perf_counter()
                built=self._parallel(lambda item:materialize_pair(item[2],epoch=epoch),missing)
                self._stats['materialize_seconds']+=time.perf_counter()-started
                self._stats['sampled_pairs_materialized']+=len(missing)
                for (position,key,_),payload in zip(missing,built):
                    payloads[position]=payload
                    if cache_views: self._views[key]=payload
                self._room()
                private=[((a.clone(),b.clone()),source,target) for (a,b),source,target in payloads] if cache_views else payloads
                started=time.perf_counter(); batch=collate(list(zip(private,ids)))
                self._stats['collate_seconds']+=time.perf_counter()-started
                self.coordinator.trim()
                return batch
            finally:
                self._protected_ids.clear(); self._protected_views.clear()
                self._stats['get_calls']+=1; self._stats['get_seconds']+=time.perf_counter()-began

    def profile(self):
        with self.coordinator.lock:
            return dict(**copy.deepcopy(self._stats),CPU_workers=self.workers,persistent_CPU_workers=True,
                sampled_view_entries=len(self._views),canonical_record_entries=len(self._records),
                sampled_view_resident_bytes=self._ledger.bytes('view'),total_unique_resident_bytes=self._ledger.bytes('all'),
                actual_RSS_bytes=psutil.Process().memory_info().rss,neural_features_cached=False,
                recipient_GT_used_in_forward=False,full_dataset_inclusion=True)

    def close(self):
        with self.coordinator.lock:
            if self._closed:return
            self._closed=True; self._executor.shutdown(wait=True,cancel_futures=True)
            self._views.clear(); self._records.clear(); self.coordinator.providers.remove(self)
