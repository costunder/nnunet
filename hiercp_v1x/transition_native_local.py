"""Native raw-CT LocalCNN inputs on the ORIGINAL v1 anchor-ranking curriculum.

This is the dependency-closed C input block, not the earlier fixed48 HALF A.
Only native fused128 is returned by the actual existing LocalCNN. No fabricated
semantic outputs, graph role bridge, comparison erasure/transform/corruption,
or six-readout view consistency is introduced. The caller uses the unchanged
v1 ranking loss and common verified B upper, and records these explicit changes.

The loader reads original source identities, all eight candidate centers and
difficulties, but obtains CNN voxels directly from verified native raw CT.
GT/curriculum metadata is kept on the loader; it is not an argument of LocalCNN.
"""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import json
import math
from pathlib import Path
import importlib.util
import sys
import threading

FORMAT = 'original_v1_curriculum_native_raw_local_C_v1'
NATIVE_CONFIG = dict(architecture='paired_native_local_cnn_v1',margin_mm=10.0,
    channels=[12,24,32],convolutions=[2,3,3],hidden_dim=128,
    readout='organ_masked_mean_each_scale',fusion='donor_target_difference_product',
    input='native_spacing_organ_only',initialization='fresh_seed42',learning_policy='same_donor_live_v1')


def native_transition_spec():
    return dict(format=FORMAT,architecture=copy.deepcopy(NATIVE_CONFIG),
        CNN='actual_l0_local_cnn.model.LocalCNN',input='verified_raw_CT_native_spacing_variable_crop_organ_only',
        crop='actual_original_source_full_mask_bbox_untransformed_plus10mm_same_offset_at_each_original_center',
        normalization='original_native_clip_to_zero_one_inside_actual_organ_zero_outside',
        resampling=False,target_erasure=False,candidate_transforms_applied=False,
        relation_corruption_applied=False,original_six_view_consistency=False,
        auxiliary_role_embeddings=False,output='actual_native_fused128_only',
        source='original_own_case_component_and_anchor_exact_sample_seed',
        candidates='all_original_eight_ordered_centers_positive_index0_and_original_difficulties',
        training_candidate_pool=128,GT_index=0,loss='caller_original_v1_ranking_listwise_margin_ordinal_mining',
        dependency_closed_changed_group=['native_input_and_L0','comparison_transforms_and_corruption',
            'two_sampled_graph_views_and_six_embedding_consistency'],
        common_upper='verified_B_prompt_upper_fixed_in_remaining_crosses',
        no_native_equivalence_claim=True)


def _sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024**2),b''): h.update(chunk)
    return h.hexdigest()


def _json_metadata(value):
    """Copy audit metadata to JSON values without mutating neural inputs."""
    import numpy as np
    import torch
    if isinstance(value,torch.Tensor):
        if value.device.type!='cpu':raise ValueError('Native loader audit metadata must be immutable CPU values')
        return value.detach().tolist()
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,Mapping):return {key:_json_metadata(item) for key,item in value.items()}
    if isinstance(value,(list,tuple)):return [_json_metadata(item) for item in value]
    return copy.deepcopy(value)


def native_bounds(center,offset_lo,offset_hi,spacing,shape,margin_mm=10.0):
    """The actual native LocalCNN CropStore recipe; no interpolation or cap."""
    import numpy as np
    c=np.asarray(center);s=np.asarray(spacing,dtype=np.float64);shape=np.asarray(shape)
    lo=np.asarray(offset_lo,dtype=np.float64);hi=np.asarray(offset_hi,dtype=np.float64)
    if (c.shape!=(3,) or s.shape!=(3,) or shape.shape!=(3,) or lo.shape!=(3,) or hi.shape!=(3,)
            or not np.isfinite(c).all() or not np.equal(c,np.round(c)).all()
            or not np.isfinite(s).all() or (s<=0).any() or (shape<=0).any()
            or ((c<0)|(c>=shape)).any() or not np.isfinite(lo).all() or not np.isfinite(hi).all()
            or (hi<=lo).any() or margin_mm!=10.0):
        raise ValueError('Actual native integer center, complete footprint bounds, positive spacing and10mm required')
    origin=np.maximum(c.astype(int)+np.floor((lo-margin_mm)/s+.5).astype(int),0)
    end=np.minimum(c.astype(int)+np.ceil((hi+margin_mm)/s+.5).astype(int),shape)
    if (end<=origin).any(): raise ValueError('Empty native crop; no altered center or shape fallback')
    return origin,end


def make_native_local(*,checkpointing=False,budget=None,dropout=.1):
    """Actual unchanged LocalCNN constructor; the caller owns seed/RNG/device."""
    LocalCNN,_=_native_types()
    return LocalCNN(copy.deepcopy(NATIVE_CONFIG),checkpointing=checkpointing,budget=budget,dropout=dropout)


def _native_types():
    """Resolve native dependencies from their actual workspace source files.

    The activated original archive includes an older tools package. Its import
    path must not shadow the actual native CNN's current helper dependencies.
    Load only these two explicit helper files; original hiercp resolution stays
    untouched and no old archive file is edited.
    """
    root=Path(__file__).resolve().parents[1]
    for name in ('tools.v22_artifacts','tools.v222_review_contracts'):
        path=root.joinpath(*name.split('.')).with_suffix('.py').resolve(strict=True)
        module=sys.modules.get(name)
        if module is not None:
            if Path(module.__file__).resolve()!=path:
                raise ValueError('An archived helper shadows the actual native CNN dependency: '+name)
            continue
        spec=importlib.util.spec_from_file_location(name,path)
        module=importlib.util.module_from_spec(spec);sys.modules[name]=module
        spec.loader.exec_module(module)
    from l0_local_cnn.model import LocalCNN,LocalBatch
    return LocalCNN,LocalBatch


class NativeCurriculumLoader:
    """Parallel raw decoding/crops, bounded immutable caches and full pair batches.

    ``samples`` is canonical original-v1 dictionaries or verified paths. Optional
    sample_sha256 binds paths to an existing signed index; otherwise constructor
    hashes the existing files and records that snapshot in its provenance.
    ``raw_records`` is an iterable/map of signed CT rows, or actual CasePaths;
    CasePaths are snapshotted once at construction. Existing artifacts are read
    only. ``get`` takes SAMPLE indices, returns LocalBatch with eight rows/sample
    and global row IDs ``sample_index_in_loader*8+candidate_index``.
    """
    def __init__(self,samples,raw_records,base,*,workers,resident_bytes,rss_bytes,
                 debug=False,sample_sha256=None,pin_memory=False):
        if (type(workers) is not int or workers<2 or type(resident_bytes) is not int
                or type(rss_bytes) is not int or not 0<resident_bytes<rss_bytes
                or type(debug) is not bool or type(pin_memory) is not bool):
            raise ValueError('Explicit parallel workers>=2, resident/RSS bounds and DEBUG/pinning flags required')
        if (base['seed']!=42 or base['cache']['total_candidates']!=8
                or base['cache']['candidate_pool_size']!=128 or base['training']['epochs']!=40
                or base['labels']!={'liver':1,'tumor':2}):
            raise ValueError('Original seed42/eight/pool128/labels/production40 contract required')
        if not isinstance(samples,Sequence) or not samples:
            raise ValueError('Complete explicitly selected original canonical samples required')
        import numpy as np
        import nibabel as nib
        import torch
        from l0_exploration.data import RawStore
        self.base=copy.deepcopy(base);self.debug=debug;self.workers=workers
        self.limit=resident_bytes;self.rss=rss_bytes;self.pin_memory=pin_memory
        self.samples=[];self.sample_sources=[];self.records=[];self._sample_rows=[]
        self._raw_cache=OrderedDict();self._crop_cache=OrderedDict();self._geometry={}
        self._lock=threading.Lock();self.hits=0;self.misses=0
        self._raw_bytes=0;self._crop_bytes=0
        identities=set()
        for index,item in enumerate(samples):
            source_record=dict(kind='in_memory_canonical_snapshot',sample_ordinal=index)
            if isinstance(item,(str,Path)):
                path=Path(item).resolve(strict=True);digest=_sha(path)
                if sample_sha256 is not None and sample_sha256.get(str(path))!=digest:
                    raise ValueError('Original signed canonical sample file changed')
                item=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
                source_record=dict(kind='signed_original_cache' if sample_sha256 is not None else 'original_cache_snapshot',
                    path=str(path),sha256=digest,sample_ordinal=index)
            if not isinstance(item,Mapping): raise ValueError('Original canonical sample mapping required')
            case=item.get('case_id');sid=item.get('sample_index');split=item.get('split')
            identity=(case,sid)
            centers=item.get('candidate_centers');difficulty=item.get('difficulties');targets=item.get('target_locals')
            if (not isinstance(case,str) or not case or type(sid) is not int or sid<0
                    or split not in ('train','val') or identity in identities
                    or not isinstance(centers,torch.Tensor) or centers.device.type!='cpu' or centers.shape!=(8,3)
                    or not isinstance(difficulty,torch.Tensor) or difficulty.device.type!='cpu' or difficulty.shape!=(8,)
                    or difficulty.dtype!=torch.long or int(difficulty[0])!=0
                    or not isinstance(targets,(list,tuple)) or len(targets)!=8
                    or type(item.get('source_component')) is not int or item['source_component']<1
                    or len(item.get('corruptions',[]))!=8):
                raise ValueError('Original unique case/sample/split/source and complete positive-first eight curriculum required')
            if not bool(torch.isfinite(centers).all()) or not bool(torch.eq(centers,centers.round()).all()):
                raise ValueError('Original candidate centers must be finite native integer coordinates')
            if bool(((difficulty<0)|(difficulty>3)).any()): raise ValueError('Original difficulty levels0..3 required')
            identities.add(identity)
            # Retain only immutable small metadata, never legacy erased/resized
            # dense tensors, canonical fine graphs or unused learned inputs.
            compact=dict(case_id=case,sample_index=sid,split=split,source_component=item['source_component'],
                candidate_centers=centers.detach().clone(),difficulties=difficulty.detach().clone(),
                corruptions=copy.deepcopy(item['corruptions']),
                transforms=[t['transform'].detach().cpu().clone() for t in targets])
            self.samples.append(compact);self.sample_sources.append(source_record)
            row_ids=[]
            for candidate in range(8):
                transform=compact['transforms'][candidate]
                if transform.shape!=(3,3) or not bool(torch.isfinite(transform).all()):
                    raise ValueError('Original candidate transform provenance malformed')
                row_id=len(self.records);row_ids.append(row_id)
                self.records.append(dict(id=f'{case}:{sid}:{candidate}',row_id=row_id,sample_ordinal=index,
                    case_id=case,donor_case_id=case,source_component=compact['source_component'],
                    source_anchor=compact['candidate_centers'][0].tolist(),center=compact['candidate_centers'][candidate].tolist(),
                    candidate_index=candidate,original_difficulty=int(difficulty[candidate]),
                    original_transform=transform.tolist(),original_corruption=copy.deepcopy(compact['corruptions'][candidate]),
                    transform_applied=False,relation_corruption_applied=False,target_erasure=False,
                    original_GT_index=0,source_own_case=True,split=split))
            self._sample_rows.append(tuple(row_ids))
        raw_items=list(raw_records.values()) if isinstance(raw_records,Mapping) else list(raw_records)
        raw=[];seen=set()
        for value in raw_items:
            if isinstance(value,Mapping): row=copy.deepcopy(dict(value))
            else:
                row=dict(case_id=value.case_id,image=str(value.image),label=str(value.label))
                row.update(image_sha256=_sha(row['image']),label_sha256=_sha(row['label']),identity_snapshot=True)
            if row.get('case_id') in seen: raise ValueError('Duplicate raw case identity')
            seen.add(row.get('case_id'))
            for key in ('image','label'):
                path=Path(row[key]).resolve(strict=True);row[key]=str(path)
                if (not isinstance(row.get(key+'_sha256'),str) or len(row[key+'_sha256'])!=64
                        or _sha(path)!=row[key+'_sha256']):
                    raise ValueError('Signed actual raw CT/annotation hash differs')
            image=nib.load(row['image']);label=nib.load(row['label'])
            if image.shape!=label.shape or not np.allclose(image.affine,label.affine):
                raise ValueError('Actual native image/label geometry differs')
            actual_spacing=np.array(image.header.get_zooms()[:3],np.float32)
            if ('spacing' in row and not np.allclose(row['spacing'],actual_spacing)) or not np.isfinite(actual_spacing).all() or (actual_spacing<=0).any():
                raise ValueError('Actual native spacing differs from raw provenance')
            row['spacing']=actual_spacing.tolist();row['native_shape']=list(image.shape)
            row['native_affine']=image.affine.tolist();raw.append(row)
        if not {s['case_id'] for s in self.samples}<=seen:
            raise ValueError('Every original query needs its own actual raw CT/annotation record')
        self.raw_records={r['case_id']:r for r in raw}
        self.raw=RawStore(dict(raw_records=raw,base=self.base),workers,rss_bytes)
        _native_types()
        from l0_regions.resident import signature
        self._sample_signature=signature(self.samples)
        self.check()

    def __len__(self): return len(self.samples)

    def check(self):
        import psutil
        from l0_regions.resident import signature
        if signature(self.samples)!=self._sample_signature:
            raise ValueError('Bound original curriculum/source metadata mutated after admission')
        if psutil.Process().memory_info().rss>self.rss:
            raise MemoryError('Explicit native curriculum RSS budget exceeded; no graph/input/batch reduction')

    def _evict(self,size):
        while self._crop_cache and self._raw_bytes+self._crop_bytes+size>self.limit:
            _,item=self._crop_cache.popitem(last=False);self._crop_bytes-=item[0].nbytes+item[1].nbytes
        while self._raw_cache and self._raw_bytes+self._crop_bytes+size>self.limit:
            _,item=self._raw_cache.popitem(last=False);self._raw_bytes-=sum(item[k].nbytes for k in ('ct','lab','organ'))

    def _volume(self,case):
        with self._lock:
            if case in self._raw_cache:
                self.hits+=1;volume=self._raw_cache.pop(case);self._raw_cache[case]=volume;return volume
        volume=self.raw.load(case)
        for key in ('ct','lab','organ'): volume[key].flags.writeable=False
        size=sum(volume[k].nbytes for k in ('ct','lab','organ'))
        with self._lock:
            self.misses+=1;self._evict(size)
            previous=self._raw_cache.pop(case,None)
            if previous is not None:
                self._raw_bytes-=sum(previous[k].nbytes for k in ('ct','lab','organ'))
            if size<=self.limit: self._raw_cache[case]=volume;self._raw_bytes+=size
        return volume

    def _source_bounds(self,ordinal,volume):
        import numpy as np
        from hiercp.common import choose_source_tumor,stable_case_seed
        sample=self.samples[ordinal]
        if ordinal not in self._geometry:
            source,_,_=choose_source_tumor(volume['ct'],volume['lab'],tumor_label=2,
                selection=self.base['cache']['source_selection'],pad=self.base['cache']['source_pad'],
                rng=np.random.default_rng(stable_case_seed(42,sample['case_id'],f"sample_{sample['sample_index']}")))
            anchor=np.asarray(source.anchor_center)
            if (source.component_id!=sample['source_component']
                    or not np.array_equal(anchor,sample['candidate_centers'][0].numpy())):
                raise ValueError('Original source component/anchor no longer matches actual annotation/seed')
            pts=np.array(np.where(source.full_mask));spacing=volume['spacing']
            offset_lo=(pts.min(1)-anchor-.5)*spacing;offset_hi=(pts.max(1)-anchor+.5)*spacing
            self._geometry[ordinal]=(anchor,offset_lo,offset_hi,int(source.full_mask.sum()))
        return self._geometry[ordinal]

    def _crop(self,case,volume,center,offset_lo,offset_hi):
        import numpy as np
        origin,end=native_bounds(center,offset_lo,offset_hi,volume['spacing'],volume['ct'].shape)
        key=(case,*map(int,origin),*map(int,end))
        with self._lock:
            cached=self._crop_cache.pop(key,None)
            if cached is not None: self._crop_cache[key]=cached;self.hits+=1
        if cached is None:
            sl=tuple(slice(int(a),int(b)) for a,b in zip(origin,end));mask=volume['organ'][sl].copy()
            low,high=self.base['ct_clip'];ct=volume['ct'][sl]
            image=np.where(mask,(np.clip(np.where(mask,ct,0),low,high)-low)/(high-low),0).astype(np.float32)
            if not mask.any() or not np.isfinite(image).all(): raise ValueError('Invalid actual native organ/CT crop')
            image.flags.writeable=False;mask.flags.writeable=False;cached=(image,mask)
            size=image.nbytes+mask.nbytes
            with self._lock:
                self.misses+=1;self._evict(size)
                if size<=self.limit: self._crop_cache[key]=cached;self._crop_bytes+=size
        row=self.raw_records[case]
        audit=dict(case=case,origin=origin.tolist(),shape=list(cached[0].shape),spacing=volume['spacing'].tolist(),
            extent_mm=((end-origin)*volume['spacing']).tolist(),anchor=list(map(int,center)),
            anchor_in_organ=bool(volume['organ'][tuple(np.asarray(center,dtype=int))]),
            actual_raw_image_sha256=row['image_sha256'],actual_raw_label_sha256=row['label_sha256'],
            native_affine=row['native_affine'],resampled=False,target_erasure=False,margin_mm=10.0)
        return key,cached,audit

    def get(self,sample_indices):
        """Every requested sample's eight rows, one padded native LocalBatch."""
        import numpy as np
        import torch
        _,LocalBatch=_native_types()
        ids=tuple(sample_indices)
        if (not ids or any(type(i) is not int or not 0<=i<len(self) for i in ids) or len(set(ids))!=len(ids)):
            raise ValueError('Unique explicit original sample indices required')
        self.check()
        grouped=OrderedDict()
        for ordinal in ids: grouped.setdefault(self.samples[ordinal]['case_id'],[]).append(ordinal)
        def build_case(item):
            case,ordinals=item;volume=self._volume(case);result={}
            for ordinal in ordinals:
                anchor,lo,hi,footprint=self._source_bounds(ordinal,volume)
                donor=self._crop(case,volume,anchor,lo,hi)
                targets=[self._crop(case,volume,center,lo,hi) for center in self.samples[ordinal]['candidate_centers'].numpy()]
                result[ordinal]=(donor,targets,footprint)
            self.check();return result
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            prepared={key:value for group in pool.map(build_case,grouped.items()) for key,value in group.items()}
        volumes=[];lookup={};audit=[];donor_ids=[];recipient_ids=[];row_ids=[]
        def add(crop):
            key,value,geometry=crop
            if key not in lookup:
                lookup[key]=len(volumes);volumes.append(value);audit.append(geometry)
            return lookup[key]
        for ordinal in ids:
            donor,targets,footprint=prepared[ordinal];did=add(donor)
            for candidate,target in enumerate(targets):
                donor_ids.append(did);recipient_ids.append(add(target));row_ids.append(self._sample_rows[ordinal][candidate])
                self.records[row_ids[-1]]['source_full_mask_voxels']=footprint
        shape=np.max([v[0].shape for v in volumes],axis=0)
        images=np.zeros((len(volumes),1,*map(int,shape)),np.float32);organ=np.zeros_like(images,dtype=bool)
        for index,(image,mask) in enumerate(volumes):
            sl=(index,0,*[slice(0,n) for n in image.shape]);images[sl]=image;organ[sl]=mask
        values=[torch.from_numpy(images),torch.from_numpy(organ),torch.tensor(donor_ids,dtype=torch.long),
            torch.tensor(recipient_ids,dtype=torch.long),torch.tensor(row_ids,dtype=torch.long)]
        if self.pin_memory: values=[v.pin_memory() for v in values]
        batch=LocalBatch(*values,audit).validate();self.check();return batch

    batch=get

    def case_ids(self,sample_indices): return tuple(self.samples[i]['case_id'] for i in sample_indices)
    def counts(self,sample_indices): return tuple(8 for _ in sample_indices)
    def difficulty_list(self,sample_indices,*,device='cpu'):
        return [self.samples[i]['difficulties'].to(device) for i in sample_indices]

    def report(self):
        return _json_metadata(dict(format=FORMAT,debug=self.debug,full_training=False,quality_verified=False,
            sample_count=len(self.samples),candidate_rows=len(self.records),spec=native_transition_spec(),
            sample_sources=copy.deepcopy(self.sample_sources),raw_records=copy.deepcopy(list(self.raw_records.values())),
            records=copy.deepcopy(self.records),resident_raw_bytes=self._raw_bytes,resident_crop_bytes=self._crop_bytes,
            resident_limit_bytes=self.limit,rss_limit_bytes=self.rss,workers=self.workers,
            cache_hits=self.hits,cache_misses=self.misses,pin_memory=self.pin_memory,
            GT_or_difficulty_in_LocalCNN_forward=False))

    def release_batches(self):
        """Compatibility with native loader; no collated batches retained here."""
        return None
