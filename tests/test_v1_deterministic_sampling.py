"""Synthetic numerical unit tests only, not medical prediction results."""
import unittest
import torch
import torch.nn.functional as F
from hiercp_v222.deterministic_sampling import sample_nodes

class InterpolationTests(unittest.TestCase):
    def test_cpu_forward_and_gradient_match(self):
        torch.manual_seed(42)
        a=torch.randn(2,4,5,6,7,requires_grad=True)
        b=a.detach().clone().requires_grad_()
        grid=torch.cat([torch.rand(17,3)*2.8-1.4,torch.tensor([[1.,1.,1.],[-1.,-1.,-1.]])])
        owners=torch.tensor([0]*9+[1]*10)
        reference=torch.cat([F.grid_sample(a[i:i+1],grid[owners==i][None,:,None,None,:],
            align_corners=True,padding_mode='border')[:, :, :, 0,0].permute(0,2,1).squeeze(0) for i in range(2)])
        result=sample_nodes(b,grid,owners)
        torch.testing.assert_close(result,reference,atol=2e-6,rtol=2e-6)
        reference.square().sum().backward();result.square().sum().backward()
        torch.testing.assert_close(b.grad,a.grad,atol=5e-6,rtol=5e-6)
    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required for deterministic backward test')
    def test_cuda_strict_determinism(self):
        old=torch.are_deterministic_algorithms_enabled()
        try:
            torch.use_deterministic_algorithms(True)
            torch.manual_seed(42)
            a=torch.randn(2,32,12,12,12,device='cuda',requires_grad=True)
            grid=torch.rand(3000,3,device='cuda')*2-1
            owners=torch.arange(2,device='cuda').repeat_interleave(1500)
            out=sample_nodes(a,grid,owners);out.square().sum().backward();first=a.grad.clone()
            a.grad=None;sample_nodes(a,grid,owners).square().sum().backward()
            self.assertTrue(torch.equal(first,a.grad))
        finally:torch.use_deterministic_algorithms(old)

if __name__=='__main__':unittest.main()
