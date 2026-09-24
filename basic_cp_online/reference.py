"""Literal reference policy; the known uint8 distance defect is NOT silently fixed."""
import ast
import hashlib
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi

REFERENCE=Path(__file__).resolve().parents[1]/'reference/medical_data_aug/3d_copy_paste_tumor.py.txt'
EXPECTED_SHA='986fd6afb95b94c70614ac02d2b9ced776cbd680c5e70d4dd11e20148383bd70'

def reference_digest():
    return hashlib.sha256(REFERENCE.read_bytes().replace(b'\r\n',b'\n')).hexdigest()

def reference_functions():
    raw=REFERENCE.read_bytes().replace(b'\r\n',b'\n')
    if hashlib.sha256(raw).hexdigest()!=EXPECTED_SHA:raise ValueError('Original Basic CP source changed')
    tree=ast.parse(raw.decode('utf-8'))
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('bbox_of_mask','try_random_centers_shuffled')]
    namespace={'np':np}
    exec(compile(ast.Module(body=functions,type_ignores=[]),str(REFERENCE),'exec'),namespace)
    return namespace

HELPERS=reference_functions()

def static_inputs(image,label):
    if image.ndim!=3 or image.shape!=label.shape or not np.isfinite(image).all():raise ValueError('Finite paired 3-D CT/GT required')
    if not np.isin(label,[0,1,2]).all():raise ValueError('Expected 0/1/2 labels')
    components,count=ndi.label(label==2)
    forbidden=ndi.binary_dilation(label==2,structure=ndi.generate_binary_structure(3,1),iterations=2)
    return dict(image=image,label=label,components=components,component_count=int(count),forbidden=forbidden)

def select(inputs,seed):
    """Every visit: one uniformly drawn OWN component; no size filter or CP gate."""
    image,label=inputs['image'],inputs['label']
    if not np.any(label==1):return dict(status='no_liver')
    count=inputs['component_count']
    if not count:return dict(status='no_tumor')
    component=int(np.random.default_rng(seed).integers(1,count+1))
    mask=inputs['components']==component
    slices=HELPERS['bbox_of_mask'](mask,pad=2)
    source_ct=image[slices].astype(np.float32);source_mask=mask[slices].astype(bool)
    rng=np.random.default_rng(seed);proposals=0
    for center in HELPERS['try_random_centers_shuffled'](label==1,source_mask.shape,rng,max_tries=4000):
        proposals+=1;anchor=np.asarray(source_mask.shape)//2;lo=np.asarray(center)-anchor
        box=tuple(slice(int(a),int(a+b)) for a,b in zip(lo,source_mask.shape))
        if np.any(source_mask & inputs['forbidden'][box]):continue
        coverage=float(np.sum(source_mask & (label[box]==1))/(source_mask.sum()+1e-6))
        if coverage<.85:continue
        # ~uint8(0/1) is nonzero EVERYWHERE. SciPy's original EDT therefore
        # measures from (-1,0,0). Evaluate the identical queried value without
        # allocating a full EDT. The reference defect remains explicit.
        x,y,z=map(int,center)
        if np.sqrt((x+1)**2+y*y+z*z)<12:continue
        return dict(status='placed',component=component,source_ct=source_ct,source_mask=source_mask,
                    anchor=anchor,center=np.asarray(center),source_slices=slices,target_slices=box,
                    scale=float(rng.uniform(.95,1.05)),shift_hu=float(rng.uniform(-5.,5.)),proposals=proposals,coverage=coverage)
    return dict(status='no_valid_location',component=component,proposals=proposals)

def apply_raw(image,label,plan):
    output=image.copy();target=label.copy()
    if plan['status']=='placed':
        box=plan['target_slices'];mask=plan['source_mask'];alpha=mask.astype(np.float32)
        output[box]=(1-alpha)*output[box]+alpha*(plan['source_ct']*plan['scale']+plan['shift_hu'])
        target[box][mask]=2
    return output,target
