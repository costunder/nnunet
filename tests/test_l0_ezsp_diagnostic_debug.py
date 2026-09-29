"""Diagnostic policy tests, separate from preserved original nine tests."""
import copy,unittest
import torch
from tests.test_l0_ezsp_debug import synthetic
from l0_ezsp.data import collate
from l0_ezsp.config import load_profile,fingerprint
from l0_ezsp.encoder import EZSPEncoder
from l0_ezsp.diagnostic import DiagnosticEZSPEncoder,ResourceBudget,DiagnosticResourceLimit
from l0_ezsp.partition import CoarseningConstraintError,partition,Region
from hiercp_v222.v1_cache import configuration
from hiercp_v22.schema import LOCAL_NODE_TYPES,LOCAL_EDGE_TYPES

@unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
class DiagnosticTests(unittest.TestCase):
    def setUp(self):
        _,self.base=configuration();self.profile=load_profile(reg_scale1=.1,reg_scale2=.1)
        self.budget=ResourceBudget(6*2**30,12*2**30,60)

    def test_profile_excess_does_not_change_partition_or_limits(self):
        torch.manual_seed(8);strict=EZSPEncoder(self.base,self.profile).cuda().eval()
        torch.manual_seed(8);diag=DiagnosticEZSPEncoder(self.base,self.profile,resource_budget=self.budget).cuda().eval()
        batch=collate([(synthetic(),0)]).to('cuda');batch.sidecar['tumor_surface']['pos_mm']*=10000
        before=fingerprint(self.profile)
        with torch.no_grad():
            with self.assertRaises(CoarseningConstraintError):strict(batch)
            value=diag(batch)
        self.assertTrue(torch.isfinite(value).all());self.assertEqual(before,fingerprint(diag.profile))
        for role in LOCAL_NODE_TYPES:
            torch.testing.assert_close(strict.last_topology[1]['parents'][role],diag.last_topology[1]['parents'][role])
        self.assertTrue(diag.last_audit['scale1']['profile_exceeded'])
        self.assertFalse(diag.last_audit['scale2']['production_admitted'])
        with self.assertRaises(RuntimeError):diag.state_dict()

    def test_integrity_checks_are_not_bypassed(self):
        net=DiagnosticEZSPEncoder(self.base,self.profile,resource_budget=self.budget).cuda()
        original=collate([(synthetic(),0)]).to('cuda')
        for kind in ('ct','grid','pos','coverage','edge','shell','role','missing_manifest'):
            batch=copy.deepcopy(original)
            if kind=='ct':batch.target_patches[0,0,0,0,0]=float('nan')
            if kind=='grid':batch.graph['source_context'].grid[0,0]=float('nan')
            if kind=='pos':batch.sidecar['source_context']['pos_mm'][0,0]=float('inf')
            if kind=='coverage':batch.sidecar['source_context']['stable_id'][1]=0
            if kind=='edge':batch.graph[LOCAL_EDGE_TYPES[0]].edge_index[0,0]=-1
            if kind=='shell':batch.graph['source_context'].shell_id[0]=3
            if kind=='role':del batch.graph['tumor_surface']
            if kind=='missing_manifest':del batch.graph.sampled_counts
            with self.subTest(kind=kind),self.assertRaises((ValueError,CoarseningConstraintError)):net(batch)

    def test_resource_budget_blocks(self):
        budget=ResourceBudget(1,1,60)
        net=DiagnosticEZSPEncoder(self.base,self.profile,resource_budget=budget).cuda()
        with self.assertRaises(DiagnosticResourceLimit):net(collate([(synthetic(),0)]).to('cuda'))

    def test_group_excess_counts_weight_fine_nodes(self):
        x=torch.ones(3,32,device='cuda');owner=torch.zeros(3,dtype=torch.long,device='cuda')
        pos=torch.tensor([[0.,0.,0.],[20.,0.,0.],[40.,0.,0.]],device='cuda')
        r=Region(owner,owner-1,torch.tensor([2.,3.,5.],device='cuda'),pos,pos,torch.arange(3,device='cuda'),
            torch.tensor([[0,1],[1,2]],device='cuda'),torch.ones(2,device='cuda'))
        result=partition(x,r,role='tumor_surface',level=1,cfg=self.profile,diagnostics=True)
        d=result.stats['group_diagnostics'][0]
        self.assertEqual(d['clusters'],1);self.assertEqual(d['bbox_excess']['clusters'],1)
        self.assertEqual(d['bbox_excess']['fine_nodes'],10);self.assertEqual(d['bbox_excess']['fine_node_fraction'],1)
        self.assertEqual(d['connected_components_before'],1);self.assertEqual(d['connected_components_after'],1)
        self.assertEqual(d['termination_condition'],'no_remaining_adjacency')

if __name__=='__main__':unittest.main(verbosity=2)
