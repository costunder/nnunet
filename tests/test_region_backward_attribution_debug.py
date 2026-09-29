import unittest
from unittest.mock import patch
import torch
from tools.region_backward_attribution import Attribution,tensor_leaves

class AttributionChecks(unittest.TestCase):
    def test_nested_labels_do_not_relabel_child(self):
        x=torch.tensor([2.],requires_grad=True);a=Attribution()
        child=a.tag(x.square(),'CNN');outer=a.tag(child.sin().sum(),'L0_other')
        self.assertEqual(a.labels[child.grad_fn],'CNN')
        self.assertEqual(a.labels[outer.grad_fn],'L0_other')
        self.assertIn('parameter_accumulation',a.labels.values())
        before=dict(a.labels);a.tag(outer,'loss')
        self.assertEqual(before,a.labels)

    def test_shared_graph_claimed_once(self):
        x=torch.tensor([2.],requires_grad=True);a=Attribution()
        shared=a.tag(x.square(),'CNN')
        left=a.tag(shared*3,'left');right=a.tag(shared*4,'right')
        loss=a.tag((left+right).sum(),'loss')
        self.assertEqual(a.labels[shared.grad_fn],'CNN')
        loss.backward();self.assertEqual(x.grad.item(),28.)

    def test_nested_tensor_leaves(self):
        x=torch.ones(1);y=torch.ones(2)
        self.assertEqual([id(t) for t in tensor_leaves({'a':(x,[y]),'b':None})],[id(x),id(y)])

    def test_event_hooks_preserve_gradient_and_cleanup(self):
        class Event:
            def __init__(self,**kwargs):self.recorded=False
            def record(self):self.recorded=True
            def elapsed_time(self,other):
                assert self.recorded and other.recorded
                return 1.
        x=torch.tensor([2.],requires_grad=True);a=Attribution();loss=a.tag(x.square().sum(),'CNN')
        with patch('torch.cuda.Event',Event),patch('torch.cuda.synchronize'):
            a.attach(loss);loss.backward();report=a.report()
        self.assertEqual(x.grad.item(),4.)
        self.assertEqual(report['registered_nodes'],report['executed_nodes'])
        a.close();self.assertFalse(a.handles);self.assertFalse(a.labels)

if __name__=='__main__':unittest.main()
