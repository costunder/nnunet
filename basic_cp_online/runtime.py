"""Fresh original-policy CP before ordinary nnU-Net crop/standard augmentations."""
from pathlib import Path
import os,json
import numpy as np
from custom_trainers.onlinecp_raw_bank import RawBankStore
from custom_trainers.onlinecp_raw_resampling import prepare_candidate,apply_candidate
from hiercp_v2.contracts import read_json,sha,validate_split
from . import FORMAT
from .reference import select,REFERENCE,EXPECTED_SHA,reference_digest

def source_identity():
    result={p.name:sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))}
    result['comparison_randomness.py']=sha(Path(__file__).resolve().parents[1]/'comparison_randomness.py')
    return result

def validate_manifest(path):
    meta=read_json(path);validate_split(meta['split'])
    if (meta.get('format')!=FORMAT or not meta.get('complete') or meta['source_identity']!=source_identity()
            or meta['reference_sha256']!=EXPECTED_SHA or reference_digest()!=EXPECTED_SHA):raise ValueError('Current original-reference preparation required')
    if set(meta['cases'])!=set(meta['split']['outer_train']):raise ValueError('Prepared cases contain held-out/missing patients')
    if sha(meta['native_path'])!=meta['native_sha256']:raise ValueError('Native preparation changed')
    return meta

class NativeCT:
    """Exact raw-paste/native CT evaluated lazily only at requested crops."""
    def __init__(self,case,candidate,plan):
        self.case,self.candidate,self.plan=case,candidate,plan
        self.shape=(1,*case['metadata']['preprocessed_shape']);self.dtype=np.dtype('float32');self.ndim=4
    def __getitem__(self,index):
        if not isinstance(index,tuple) or len(index)!=4:raise ValueError('Expected channel and 3 crop slices')
        box=[]
        for s,n in zip(index[1:],self.shape[1:]):
            if not isinstance(s,slice):raise ValueError('Native crop requires slices')
            start,stop,step=s.indices(n)
            if step!=1:raise ValueError('Native crop does not support strides')
            box.append([start,stop])
        if any(b<=a for a,b in box):return np.zeros((1,*[max(0,b-a) for a,b in box]),np.float32)[index[0]]
        return apply_candidate(self.case,self.candidate,box,scale=self.plan['scale'],shift_hu=self.plan['shift_hu'])['data'][index[0]]

class Dataset:
    def __init__(self,dataset,manifest,*,cp_probability=1.0):
        if cp_probability not in (1.0,0.8):raise ValueError('Use original CP=1.0 or the approved CP80 comparison')
        self.cp_probability=float(cp_probability)
        self.dataset=dataset;self.identifiers=dataset.identifiers;self.path=Path(manifest).resolve()
        self.meta=validate_manifest(self.path)
        if set(self.identifiers)!=set(self.meta['split']['outer_train']):raise ValueError('CP may wrap exactly the training dataset only')
        self.store=None;self.pid=None;self.last_audit=None
        from comparison_randomness import master_seed
        self.cp_rng=np.random.default_rng(master_seed())
        self.cp_schedule_rng=np.random.default_rng(master_seed())
    def __len__(self):return len(self.identifiers)
    def set_comparison_batch(self,seed):
        self.cp_rng=np.random.default_rng(seed)
    def set_cp_schedule(self,seed):
        self.cp_schedule_rng=np.random.default_rng(seed)
    def record_audit(self,case_id,plan,seed,gate_u):
        self.last_audit={k:plan[k] for k in ('status','component','proposals','coverage') if k in plan}
        self.last_audit.update(case_id=case_id,donor_case_id=case_id,seed=seed,
            cp_probability=self.cp_probability,gate_u=gate_u,attempted=gate_u<self.cp_probability,
            variant='original' if self.cp_probability==1.0 else 'basic_cp80_comparison')
        folder=self.path.parent/'online_audit';folder.mkdir(exist_ok=True)
        with (folder/f'worker_{os.getpid()}.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(self.last_audit)+'\n')
    def load_case(self,case_id):
        if case_id not in self.meta['cases']:raise ValueError('Held-out CP request rejected')
        data,seg,previous,properties=self.dataset.load_case(case_id)
        if previous is not None:raise ValueError('Original Basic CP requires non-cascade training')
        # Match the frozen v2 loader's five draws/event. Source/placement remain
        # the original Basic policy and use a separate RNG; the first draw is
        # the common gate even if another policy fails to place a lesion.
        gate_u=float(self.cp_schedule_rng.random(5)[0])
        seed=int(self.cp_rng.integers(0,2**32,dtype=np.uint64))
        if gate_u>=self.cp_probability:
            self.record_audit(case_id,dict(status='skipped_probability'),seed,gate_u)
            return data,seg,previous,properties
        if self.pid!=os.getpid():
            self.store=RawBankStore(self.path.parent);self.pid=os.getpid()
        row=self.meta['cases'][case_id];inputs=self.store.load_case(row['inputs_ref'],row['inputs_sha'])
        plan=select(inputs,seed)
        self.record_audit(case_id,plan,seed,gate_u)
        if plan['status']!='placed':return data,seg,previous,properties
        case=dict(self.store.load_case(row['case_ref'],row['case_sha']))
        case['preparation']=self.store.load_case(row['preparation_ref'],row['preparation_sha'])
        candidate=prepare_candidate(case,plan['source_ct'],plan['source_mask'],plan['anchor'],plan['center'])
        # Ordinary crop/foreground sampling from the COMPLETE augmented GT;
        # never force a crop around the synthetic lesion.
        augmented_seg=case['baseline_seg'].copy()
        box=tuple(slice(int(a),int(b)) for a,b in candidate['output_bbox'])
        augmented_seg[(slice(None),*box)]=candidate['seg_patch']
        from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
        updated=dict(properties)
        updated['class_locations']=DefaultPreprocessor._sample_foreground_locations(augmented_seg,[1,2],verbose=False)
        return NativeCT(case,candidate,plan),augmented_seg,previous,updated
