"""Native CT crops only. No graph or partition preparation, no image resizing."""
import copy,json
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import psutil,torch
from l0_exploration.data import RawStore
from hiercp_v22.data import sources
from l0_regions.donor_learning import POLICY,validate_rows
from l0_regions.donor_data import assignment
from l0_regions.training_data import sha,write_new
from .model import LocalBatch,validate_config

FORMAT='native_local_cnn_inventory_v1'

def prepare(index,config,output,*,debug=False):
    from hiercp_v222.contracts import validate_identities
    from hiercp_v22.donors import validate_pool
    m=json.loads(Path(index).read_text(encoding='utf8'));validate_config(config)
    if bool(m.get('debug'))!=debug:raise ValueError('Explicit matching DEBUG/full observation inventory required')
    validate_identities(m['identities'],m['split']);validate_pool(m['donor_pool'],m['split'])
    rows=assignment(m,m['config']['seed']);old={r['id']:r for r in m['records']}
    if len(rows)!=len(old):raise ValueError('Observation coverage mismatch')
    # Preserve the existing schedule's ordering key; not an active CNN edge count.
    for r in rows:r['bounds']=copy.deepcopy(old[r['id']]['bounds'])
    if not debug:
        raw={r['case_id']:r for r in m['raw_records']}
        for case in m['split']['outer_train']:
            rs=[r for r in rows if r['case_id']==case]
            if sum(r['target']==0 for r in rs)!=128 or sum(r['target']==1 for r in rs)!=len(raw[case]['positives']):raise ValueError('Full candidate/observation coverage required')
    out={k:copy.deepcopy(m[k]) for k in ('config','base','split','identities','donor_pool','raw_records')}
    out.update(format=FORMAT,complete=True,debug=debug,records=rows,learning_policy=POLICY,local_cnn=config,
               original_inventory_sha256=sha(index),source_identity=m['source_identity'],preparation='metadata only; native crops on demand; no graphs')
    root=Path(output);root.mkdir(parents=True,exist_ok=False);write_new(root/'index.json',out);return root/'index.json'

class Dataset:
    def __init__(self,index,partition,debug):
        from hiercp_v222.contracts import validate_identities
        self.path=Path(index).resolve();self.root=self.path.parent;self.meta=json.loads(self.path.read_text(encoding='utf8'));m=self.meta
        if m.get('format')!=FORMAT or not m.get('complete') or m['debug']!=debug:raise ValueError('Matching complete local CNN inventory required')
        validate_config(m['local_cnn']);validate_identities(m['identities'],m['split']);validate_rows(m['records'])
        from hiercp_v22.donors import validate_pool
        validate_pool(m['donor_pool'],m['split'])
        if not debug:
            raw={r['case_id']:r for r in m['raw_records']}
            if {r['case_id'] for r in m['records']}!=set(m['split']['outer_train']):raise ValueError('Incomplete cohort')
            for case in m['split']['outer_train']:
                rs=[r for r in m['records'] if r['case_id']==case]
                if sum(r['target']==0 for r in rs)!=128 or sum(r['target']==1 for r in rs)!=len(raw[case]['positives']):raise ValueError('Incomplete full candidates/positives')
        expected=assignment(m,m['config']['seed'])
        for r,e in zip(m['records'],expected):
            if any(r[k]!=e[k] for k in e):raise ValueError('Observation/donor assignment changed')
        allowed=set(m['split'][partition]);self.rows=[r for r in m['records'] if r['case_id'] in allowed]
        self.preparation_revision='native_local_cnn_inventory_v1'
    def __len__(self):return len(self.rows)

class CropStore:
    def __init__(self,meta,workers,resident_bytes,rss_bytes):
        if workers<2 or not 0<resident_bytes<rss_bytes:raise ValueError('Parallel readers and explicit cache/RSS headroom required')
        self.raw=RawStore(meta,workers,rss_bytes);self.meta=meta;self.workers=workers;self.limit=resident_bytes;self.rss=rss_bytes
        self.cache=OrderedDict();self.bytes=0;self.bounds={};self.hits=0;self.misses=0
    def raw_bytes(self):return sum(v[k].nbytes for v in self.raw.cache.values() for k in ('ct','lab','organ'))
    def report(self):return dict(bytes=self.bytes,raw_bytes=self.raw_bytes(),limit_bytes=self.limit,hits=self.hits,misses=self.misses)
    def donor_bounds(self,row):
        key=(row['donor_case_id'],row['donor_component'])
        if key not in self.bounds:
            if key[0] not in self.meta['split']['inner_train'] or row['patient_group']==row['donor_group']:raise ValueError('Nonindependent/held-out donor')
            v=self.raw.cache[key[0]];c=sources(SimpleNamespace(label=v['lab'],image=v['ct'],spacing=v['spacing']),self.meta['base']['cache']['source_pad'],self.meta['config']['donor_max_diameter_mm'])
            hits=[i for i,(component,_) in enumerate(c.entries) if component==key[1]]
            if len(hits)!=1:raise ValueError('Missing donor component')
            source,_=c[hits[0]];anchor=np.asarray(source.anchor_center);pts=np.array(np.where(source.full_mask))
            self.bounds[key]=(anchor,(pts.min(1)-anchor-.5)*v['spacing'],(pts.max(1)-anchor+.5)*v['spacing'])
        return self.bounds[key]
    def batch(self,rows,indices):
        # Loader has a single producer; independent raw volumes are decoded in parallel.
        needed={r[k] for r in rows for k in ('case_id','donor_case_id')}
        for k in list(self.raw.cache):
            if k in needed:self.raw.cache[k]=self.raw.cache.pop(k)
        for k in list(self.raw.cache):
            if k not in needed and (self.raw_bytes()+self.bytes>self.limit*.8 or psutil.Process().memory_info().rss>self.rss*.8):del self.raw.cache[k]
        while self.cache and psutil.Process().memory_info().rss>self.rss*.8:
            _,item=self.cache.popitem(last=False);self.bytes-=item[0].nbytes+item[1].nbytes
        self.raw.preload(rows)
        for k in list(self.raw.cache):
            if k not in needed and self.raw_bytes()+self.bytes>self.limit:del self.raw.cache[k]
        while self.cache and self.raw_bytes()+self.bytes>self.limit:
            _,item=self.cache.popitem(last=False);self.bytes-=item[0].nbytes+item[1].nbytes
        if self.raw_bytes()>self.limit:raise MemoryError('Active raw CT batch exceeds explicit resident budget; no sample/ROI reduction')
        volumes=[];lookup={};dids=[];rids=[];audit=[]
        margin=self.meta['local_cnn']['margin_mm'];low,high=self.meta['base']['ct_clip']
        def add(case,center,offset_lo,offset_hi):
            v=self.raw.cache[case];spacing=v['spacing'];center=np.asarray(center)
            if center.shape!=(3,) or not np.isfinite(center).all() or not np.equal(center,np.round(center)).all():raise ValueError('Native integer center required')
            if ((center<0)|(center>=v['organ'].shape)).any():raise ValueError(f'Crop anchor outside CT: {case}, {center.tolist()}')
            # An occupied-mask bbox midpoint can lie in background for a concave
            # component. It defines placement, not organ membership. Mask the
            # complete crop below; never move the anchor or discard the record.
            lo=np.maximum(center.astype(int)+np.floor((offset_lo-margin)/spacing+.5).astype(int),0)
            hi=np.minimum(center.astype(int)+np.ceil((offset_hi+margin)/spacing+.5).astype(int),v['ct'].shape)
            key=(case,*map(int,lo),*map(int,hi))
            if key in lookup:return lookup[key]
            if key in self.cache:
                item=self.cache.pop(key);self.cache[key]=item;self.hits+=1
            else:
                sl=tuple(slice(a,b) for a,b in zip(lo,hi));mask=v['organ'][sl].copy();ct=v['ct'][sl]
                image=np.where(mask,(np.clip(np.where(mask,ct,0),low,high)-low)/(high-low),0).astype(np.float32)
                if not mask.any() or not np.isfinite(image).all():raise ValueError('Invalid local CT')
                item=(image,mask);size=image.nbytes+mask.nbytes;self.misses+=1
                while self.cache and self.bytes+self.raw_bytes()+size>self.limit:
                    _,old=self.cache.popitem(last=False);self.bytes-=old[0].nbytes+old[1].nbytes
                if size+self.raw_bytes()<=self.limit:self.cache[key]=item;self.bytes+=size
            lookup[key]=len(volumes);volumes.append(item)
            audit.append(dict(case=case,origin=lo.tolist(),shape=list(item[0].shape),spacing=spacing.tolist(),extent_mm=((hi-lo)*spacing).tolist(),anchor_in_organ=bool(v['organ'][tuple(center.astype(int))])))
            return lookup[key]
        for row in rows:
            anchor,lo,hi=self.donor_bounds(row)
            self.validate_observation(row)
            dids.append(add(row['donor_case_id'],anchor,lo,hi));rids.append(add(row['case_id'],row['center'],lo,hi))
        shape=np.max([v[0].shape for v in volumes],0);x=np.zeros((len(volumes),1,*shape),np.float32);m=np.zeros_like(x,dtype=bool)
        for i,(image,mask) in enumerate(volumes):
            sl=(i,0,*[slice(0,n) for n in image.shape]);x[sl]=image;m[sl]=mask
        tensors=[torch.from_numpy(x),torch.from_numpy(m),torch.tensor(dids),torch.tensor(rids),torch.tensor(indices)]
        if torch.cuda.is_available():tensors=[t.pin_memory() for t in tensors]
        batch=LocalBatch(*tensors,audit).validate();self.raw.check();return batch

    def validate_observation(self,row):
        """Keep observation provenance separate from the crop's spatial anchor."""
        if 'target' not in row:return  # Online candidates retain the final full-mask placement filter.
        v=self.raw.cache[row['case_id']];center=np.asarray(row['center'])
        if center.shape!=(3,) or not np.isfinite(center).all() or not np.equal(center,np.round(center)).all():
            raise ValueError('Native integer observation center required')
        if ((center<0)|(center>=v['organ'].shape)).any():raise ValueError('Observation outside CT')
        if row['target']==0:
            if v['lab'][tuple(center.astype(int))]!=1:raise ValueError('Comparison center must be annotated liver')
        elif row['target']==1:
            matches=[p for p in self.raw.raw[row['case_id']]['positives'] if p['component']==row['component'] and np.array_equal(p['center'],center)]
            if len(matches)!=1:raise ValueError('Observed tumor anchor differs from original annotation record')
        else:raise ValueError('Unknown observation target')

class Loader:
    def __init__(self,ds,workers,resident_bytes,rss_limit,store=None):
        self.ds=ds;self.store=store or CropStore(ds.meta,workers,resident_bytes,rss_limit);self.retain_batches=False
    def get(self,ids):return self.store.batch([self.ds.rows[i] for i in ids],ids)
    def release_batches(self):pass  # No collated CPU batches are retained.
    def batches(self,order):
        with ThreadPoolExecutor(max_workers=1) as pool:
            it=iter(order);ids=next(it,None)
            if ids is None:return
            future=pool.submit(self.get,ids)
            for ids in it:
                batch=future.result();future=pool.submit(self.get,ids);yield batch
            yield future.result()
