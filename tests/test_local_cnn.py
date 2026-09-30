"""Synthetic unit checks, not CT accuracy evidence."""
import copy,json
from pathlib import Path
import unittest,torch
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

class Checks(unittest.TestCase):
    pass
for name,fn in list(globals().items()):
    if name.startswith('test_'):setattr(Checks,name,lambda self,f=fn:f())
del name,fn
if __name__=='__main__':unittest.main()
