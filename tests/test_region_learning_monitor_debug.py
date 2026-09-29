import unittest
import torch
from torch import nn
from l0_regions.learning_monitor import LearningMonitor,validation_line


def network():
    net=nn.Module();net.local=nn.Module();net.local.core=nn.Module()
    net.local.core.dense_encoder=nn.Linear(4,4)
    net.local.core.blocks=nn.ModuleList([nn.Module() for _ in range(3)])
    for block in net.local.core.blocks:block.conv=nn.Linear(4,4)
    net.l1=nn.Linear(4,4);net.l2=nn.Linear(4,4)
    return net


class MonitorTests(unittest.TestCase):
    def test_probe_observes_update_without_changing_rng_or_parameters(self):
        for device in (['cpu','cuda'] if torch.cuda.is_available() else ['cpu']):
            net=network().to(device);opt=torch.optim.SGD(net.parameters(),lr=.1)
            rng=torch.get_rng_state().clone();monitor=LearningMonitor(net)
            before=monitor.before_step()
            loss=sum(p.square().sum() for p in net.parameters());loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(net.parameters(),100)
            opt.step();expected=[p.detach().clone() for p in net.parameters()]
            terms={k:loss.detach() for k in ('ranking_loss','observation_auxiliary_loss','alignment_loss')}
            health=monitor.after_step(loss,terms,norm,before,.1)
            self.assertTrue(all(v>0 for v in health['probe_max_abs_delta'].values()))
            self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            self.assertTrue(all(torch.equal(p,e) for p,e in zip(net.parameters(),expected)))
            unchanged=monitor.after_step(loss,terms,norm,monitor.before_step(),0.)
            self.assertTrue(all(v==0 for v in unchanged['probe_max_abs_delta'].values()))
            self.assertEqual(monitor.postfix(unchanged,2)['probe'],'0/6')

    def test_validation_is_ranking_not_segmentation(self):
        row=validation_line(2,dict(ranking_pairwise_loss=.5,ranking_mrr=.3,ranking_recall_at_5=.7),.5,True)
        self.assertIn('NEW_BEST',row);self.assertIn('R@5=0.700',row)
        self.assertIn('not segmentation Dice',row)

if __name__=='__main__':unittest.main()
