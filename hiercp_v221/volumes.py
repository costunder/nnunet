"""Shared full-volume/depth cache with atomic two-case RAM admission."""
from collections import OrderedDict
from contextlib import contextmanager
import threading
import nibabel as nib
import numpy as np
from hiercp.common import load_case,organ_depth_mm
from hiercp.preparation_runtime import snapshot

def volume_memory_bound(path):
    # Header-only upper estimate for full CT/GT/component arrays and temporaries.
    return int(np.prod(nib.load(path).shape))*24

class VolumeCache:
    def __init__(self,paths,base,cfg):
        self.paths,self.base,self.cfg=paths,base,cfg
        self.budget=int(snapshot()['available_memory_bytes']*.45)
        self.values=OrderedDict(); self.pins={}; self.reserved={}
        self.condition=threading.Condition()
        self.sizes={c:volume_memory_bound(p.image_path) for c,p in paths.items()}

    def _load(self,c):
        from .data import sources
        case=load_case(self.paths[c]); organ=np.isin(case.label,[1,2])
        return dict(case=case,organ=organ,depth=organ_depth_mm(organ,case.spacing),
                    sources=sources(case,self.base['cache']['source_pad'],self.cfg['donor_max_diameter_mm']))

    @contextmanager
    def pair(self,a,b):
        wanted={a,b}
        if sum(self.sizes[c] for c in wanted)>self.budget:
            raise MemoryError('Full donor/recipient pair exceeds measured cache RAM budget; no data reduced')
        with self.condition:
            while True:
                missing=wanted-set(self.values)
                needed=sum(self.sizes[c] for c in missing)
                for c in list(self.values):
                    if sum(self.reserved.values())+needed<=self.budget:break
                    if c not in wanted and not self.pins.get(c,0):
                        del self.values[c]; del self.reserved[c]
                if sum(self.reserved.values())+needed<=self.budget:break
                self.condition.wait()
            # Load under the lock: the temporary full EDT arrays are not
            # concurrently allocated by every worker on a cold cache miss.
            for c in wanted:
                if c not in self.values:
                    self.values[c]=self._load(c); self.reserved[c]=self.sizes[c]
                self.values.move_to_end(c)
            for c in wanted:self.pins[c]=self.pins.get(c,0)+1
            va,vb=self.values[a],self.values[b]
        try:yield va,vb
        finally:
            with self.condition:
                for c in wanted:self.pins[c]-=1
                self.condition.notify_all()
