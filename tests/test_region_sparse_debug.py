"""Full-edge fixed-order kernel parity; small synthetic unit fixtures only."""
import unittest
import torch
from l0_regions.sparse import stable_spmm,segmented_mm
from l0_sage.encoder import mean_adjacency


class StableSparse(unittest.TestCase):
    def check_device(self,device):
        edge=torch.tensor([[0,1,1,3,2,0,1],[0,0,0,2,3,3,3]],device=device)
        a,_=mean_adjacency(edge,4,5);at=a.transpose(0,1).to_sparse_csr()
        x=torch.arange(28,device=device,dtype=torch.float32).reshape(4,7).requires_grad_()
        y=stable_spmm(a,at,x);want=a.to_dense()@x
        torch.testing.assert_close(y,want)
        grad=torch.arange(35,device=device,dtype=torch.float32).reshape(5,7)
        actual=torch.autograd.grad(y,x,grad)[0];expected=a.to_dense().T@grad
        torch.testing.assert_close(actual,expected)
        self.assertTrue(torch.equal(y,stable_spmm(a,at,x)))
        self.assertTrue(torch.equal(actual,torch.autograd.grad(stable_spmm(a,at,x),x,grad)[0]))
        self.assertTrue(torch.equal(y,segmented_mm(a,x,workspace_bytes=28)))
    def test_cpu(self):self.check_device('cpu')
    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_cuda(self):self.check_device('cuda')


if __name__=='__main__':unittest.main(verbosity=2)
