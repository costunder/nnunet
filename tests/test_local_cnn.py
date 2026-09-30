"""Synthetic unit checks, not CT accuracy evidence."""
import copy,json
from pathlib import Path
import unittest,torch
import numpy as np
from types import SimpleNamespace
from unittest.mock import patch
from torch.nn import functional as F
from l0_local_cnn.model import LocalCNN,LocalBatch,validate_config

def cfg(m=10):
    c=json.loads(Path('config/v22_local_cnn.json').read_text());c['margin_mm']=m;return c
def batch():
    torch.manual_seed(7);x=torch.randn(3,1,12,12,12);m=torch.ones_like(x,dtype=torch.bool);m[:,:,:2]=False
    return LocalBatch(x,m,torch.tensor([0,0]),torch.tensor([1,2]),torch.tensor([0,1]),[]).validate()
def test_external_nan_and_gradient():
    net=LocalCNN(cfg(),dropout=0).eval();b=batch();b.images.requires_grad_(True);b.validate();y=net(b);y.square().sum().backward()
    assert torch.count_nonzero(b.images.grad[~b.organ])==0
    assert b.images.grad[b.organ].abs().sum()>0
    other=LocalBatch(torch.where(b.organ,b.images.detach(),float('nan')),b.organ,b.donor,b.recipient,b.indices,[]).validate()
    torch.testing.assert_close(y,net(other),atol=0,rtol=0)
def test_right_padding_invariance():
    net=LocalCNN(cfg(),dropout=0).eval();b=batch()
    p=LocalBatch(F.pad(b.images,(0,4,0,4,0,4)),F.pad(b.organ,(0,4,0,4,0,4)),b.donor,b.recipient,b.indices,[]).validate()
    torch.testing.assert_close(net(b),net(p),atol=1e-6,rtol=1e-5)
def test_shared_donor_gradient():
    net=LocalCNN(cfg(),dropout=0);b=batch();b.images.requires_grad_(True);b.validate();y=net(b);y[:,0].sum().backward()
    assert all(b.images.grad[i].abs().sum()>0 for i in range(3))
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
def test_mutation_rejected():
    net=LocalCNN(cfg());b=batch();b.images[0,0,3,3,3]+=1
    with unittest.TestCase().assertRaisesRegex(ValueError,'mutated'):net(b)
def test_bad_margin_rejected():
    for m in (0,-1,float('nan'),float('inf')):
        with unittest.TestCase().assertRaises(ValueError):validate_config(cfg(m))
def test_explicit_margin_preserved():
    for m in (5,10,20,30):
        net=LocalCNN(cfg(m));assert net.get_extra_state()['margin_mm']==m
def test_wrong_range_checkpoint_rejected():
    a=LocalCNN(cfg(10));b=LocalCNN(cfg(20))
    with unittest.TestCase().assertRaisesRegex(ValueError,'mismatch'):b.load_state_dict(a.state_dict())
def test_no_graph_parameters():
    net=LocalCNN(cfg());assert not any('sage' in n or 'gat' in n or 'offset' in n for n,_ in net.named_parameters())

def concave_store():
    from l0_local_cnn.data import CropStore
    from hiercp_v22.data import sources
    label=np.zeros((12,12,12),np.uint8)
    label[2:9,2:9,2:9]=1;label[4:7,4:7,4:7]=0
    label[3:8,3,5]=2;label[3:8,7,5]=2;label[3,3:8,5]=2
    label[9,9,5]=1
    image=np.where(label>0,70.,np.nan).astype(np.float32)
    source,_=sources(SimpleNamespace(label=label,image=image,spacing=np.ones(3)),0,20)[0]
    center=list(source.anchor_center)
    assert label[tuple(center)]==0
    raw=[dict(case_id=c,positives=[dict(component=1,center=center)]) for c in ('donor','recipient')]
    meta=dict(raw_records=raw,split={'inner_train':['donor']},base={'cache':{'source_pad':0},'ct_clip':[-200,250]},
              config={'donor_max_diameter_mm':20},local_cnn=cfg(2))
    store=CropStore(meta,2,2**30,16*2**30)
    store.raw.cache={c:dict(ct=image.copy(),lab=label.copy(),organ=label>0,spacing=np.ones(3)) for c in ('donor','recipient')}
    row=dict(id='observed',case_id='recipient',patient_group='recipient',donor_case_id='donor',donor_group='donor',donor_component=1,component=1,center=center,target=1)
    return store,row

def test_concave_background_anchors_preserved_and_masked():
    store,row=concave_store();original=copy.deepcopy(row)
    b=store.batch([row],[0]);assert row==original
    assert len(b)==1 and all(not a['anchor_in_organ'] for a in b.audit)
    assert torch.isfinite(b.images).all() and torch.count_nonzero(b.images[~b.organ])==0
    b.images.requires_grad_(True);LocalCNN(cfg(2),dropout=0)(b).square().sum().backward()
    assert torch.count_nonzero(b.images.grad[~b.organ])==0
    assert b.images.grad[b.organ].abs().sum()>0

def test_bad_observations_still_rejected():
    store,row=concave_store()
    for changes,message in (({'target':0},'annotated liver'),({'component':99},'annotation record'),({'center':[-1,5,5]},'outside CT'),({'center':[5.5,5,5]},'integer'),({'target':2},'Unknown')):
        with unittest.TestCase().assertRaisesRegex(ValueError,message):store.batch([dict(row,**changes)],[0])
    # A valid comparison and an online candidate still use the same crop geometry.
    store.batch([dict(row,target=0,component=None,center=[9,9,5])],[0])
    online={k:v for k,v in row.items() if k not in ('target','component')}
    assert len(store.batch([online],[0]))==1

def test_empty_mask_and_bad_donor_anchor_still_rejected():
    store,row=concave_store();store.raw.cache['recipient']['organ'][:]=False
    with unittest.TestCase().assertRaisesRegex(ValueError,'Invalid local CT'):store.batch([row],[0])
    store,row=concave_store();anchor,lo,hi=store.donor_bounds(row)
    with patch.object(store,'donor_bounds',return_value=(np.array([-1,5,5]),lo,hi)):
        with unittest.TestCase().assertRaisesRegex(ValueError,'outside CT'):store.batch([row],[0])

class Checks(unittest.TestCase):
    pass
for name,fn in list(globals().items()):
    if name.startswith('test_'):setattr(Checks,name,lambda self,f=fn:f())
del name,fn
if __name__=='__main__':unittest.main()
