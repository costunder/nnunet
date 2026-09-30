"""Verified raw single-phase CT; no fine-graph/partition cache prerequisite.

Full mode crops only organ bounding boxes at native spacing. DEBUG local crops
are explicit and never a production data configuration. Padding does not change
physical spacing. Source/recipient images shared by index, not repeated by query.
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
import hashlib
import numpy as np
import nibabel as nib
import torch
from hiercp_v22.data import sources
from .model import ExplorationBatch

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024**2),b''):h.update(chunk)
    return h.hexdigest()

class RawStore:
    def __init__(self,meta,workers,rss_bytes):
        if workers<2:raise ValueError('Explicit parallel CT reader workers >=2 required')
        self.meta=meta;self.workers=workers;self.rss_bytes=rss_bytes;self.raw={r['case_id']:r for r in meta['raw_records']};self.cache={};self.donors={}
    def check(self):
        import psutil
        if psutil.Process().memory_info().rss>self.rss_bytes:raise MemoryError('CT store RSS budget exceeded')
    def load(self,case):
        row=self.raw[case]
        for k in ('image','label'):
            if sha(row[k])!=row[k+'_sha256']:raise ValueError('Raw CT/annotation identity changed: '+case)
        image=nib.load(row['image']);label=nib.load(row['label'])
        if image.shape!=label.shape or not np.allclose(image.affine,label.affine):raise ValueError('CT/organ geometry mismatch')
        if not np.allclose(image.header.get_zooms()[:3],row['spacing']) or not np.isfinite(row['spacing']).all():raise ValueError('Recorded spacing differs from native CT')
        ct=np.asarray(image.dataobj,dtype=np.float32);lab=np.asarray(label.dataobj)
        if not np.isin(lab,[0,1,2]).all():raise ValueError('Unknown annotation values')
        organ=lab>0
        if not organ.any() or not np.isfinite(ct[organ]).all():raise ValueError('Invalid internal CT')
        points=np.array(np.where(organ));lo=points.min(1);hi=points.max(1)+1
        self.check()
        return dict(ct=ct,lab=lab,organ=organ,spacing=np.array(row['spacing'],np.float32),lo=lo,hi=hi)
    def preload(self,rows):
        names=sorted({r[k] for r in rows for k in ('case_id','donor_case_id')}-set(self.cache))
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            for case,volume in zip(names,pool.map(self.load,names)):
                self.cache[case]=volume;self.check()
    def donor_center(self,row):
        case=row['donor_case_id'];component=row['donor_component'];key=(case,component)
        if key not in self.donors:
            if case not in self.meta['split']['inner_train']:raise ValueError('Donor is not inner-train')
            if row['patient_group']==row['donor_group']:raise ValueError('Self-patient donor')
            v=self.cache[case]
            source_list=sources(SimpleNamespace(label=v['lab'],image=v['ct'],spacing=v['spacing']),self.meta['base']['cache']['source_pad'],self.meta['config']['donor_max_diameter_mm'])
            matching=[i for i,(cid,_) in enumerate(source_list.entries) if cid==component]
            if len(matching)!=1:raise ValueError('Original donor component not found')
            source,_=source_list[matching[0]]
            self.donors[key]=np.array(source.anchor_center,np.float32)
        return self.donors[key]
    def batch(self,rows,*,debug_native_patch_size=None):
        if not rows:raise ValueError('Empty records')
        self.preload(rows);volumes=[];lookup={};audit=[];rid=[];did=[];centers=[];donors=[]
        def add(case,center):
            v=self.cache[case]
            if debug_native_patch_size is None:lo=v['lo'];hi=v['hi'];key=(case,'full-organ')
            else:
                if debug_native_patch_size<2:raise ValueError('Explicit DEBUG crop required')
                lo=np.floor(center).astype(int)-debug_native_patch_size//2;hi=lo+debug_native_patch_size;lo=np.maximum(0,lo);hi=np.minimum(v['ct'].shape,hi)
                key=(case,*map(int,lo),*map(int,hi))
            if key not in lookup:
                sl=tuple(slice(int(a),int(b)) for a,b in zip(lo,hi));mask=v['organ'][sl].copy();raw=v['ct'][sl]
                low,high=self.meta['base']['ct_clip'];image=np.where(mask,(np.clip(np.where(mask,raw,0),low,high)-low)/(high-low),0).astype(np.float32)
                volumes.append((image,mask,v['spacing']));lookup[key]=len(volumes)-1
                audit.append(dict(case=case,origin=lo.tolist(),shape=list(image.shape),spacing=v['spacing'].tolist(),mode='full-organ-native' if debug_native_patch_size is None else 'DEBUG-native-crop'))
            return lookup[key],np.asarray(center)-lo
        for row in rows:
            i,c=add(row['case_id'],row['center']);j,d=add(row['donor_case_id'],self.donor_center(row));rid.append(i);did.append(j);centers.append(c);donors.append(d)
        shape=np.max([v[0].shape for v in volumes],0);x=np.zeros((len(volumes),1,*shape),np.float32);m=np.zeros_like(x,dtype=bool)
        for i,(image,mask,_) in enumerate(volumes):
            sl=(i,0,*[slice(0,n) for n in image.shape]);x[sl]=image;m[sl]=mask
        batch=ExplorationBatch(torch.from_numpy(x),torch.from_numpy(m),torch.tensor(np.stack([v[2] for v in volumes])),torch.tensor(rid),torch.tensor(did),torch.tensor(np.stack(centers),dtype=torch.float32),torch.tensor(np.stack(donors),dtype=torch.float32))
        batch.validate();self.check();return batch,audit

def make_network(meta,exploration_config):
    from hiercp_v222.model import PromptGraphModel
    from .model import ExplorationL0
    # Existing L1/L2 constructor and all equations remain untouched.
    if meta['base']['model']['hidden_dim']!=exploration_config['hidden_dim']:raise ValueError('L0/L1 interface mismatch')
    return PromptGraphModel(meta['config'],meta['base'],{},local_encoder=ExplorationL0(exploration_config))
