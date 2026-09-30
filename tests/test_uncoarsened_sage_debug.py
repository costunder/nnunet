"""DEBUG integrity checks; synthetic inputs are never research performance results."""
import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import torch
from l0_regions.fine_graph import OriginalFineDataset,verified,transition,MODE
from l0_regions.materialization import seal_materialization
from l0_regions.resident import check_verified
from l0_ezsp.data import collate
from tests.test_l0_ezsp_debug import synthetic
from l0_regions.execution_upgrade import MODEL,OLD_MODEL,REVIEWED_MODEL
from hiercp_v222.placement import GEOMETRY_CONTRACT


class Integrity(unittest.TestCase):
    def reader(self,previous,current,*,geometry=GEOMETRY_CONTRACT,part='inner_train'):
        meta=dict(format='v22_review_repair_DEBUG_cache',debug=True,geometry_contract=geometry,
                  source_identity=previous,split=dict(inner_train=['a'],inner_val=['b']),records=[dict(case_id='a')])
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'index.json';path.write_text(json.dumps(meta))
            with patch('hiercp_v222.v1_cache.provenance',return_value=current):return OriginalFineDataset(path,part,True)
    def test_only_reviewed_model_hash_change_allowed(self):
        before={MODEL:OLD_MODEL,'graph.py':'same'};after={MODEL:REVIEWED_MODEL,'graph.py':'same'}
        self.assertEqual(len(self.reader(before,after)),1)
        with self.assertRaisesRegex(ValueError,'source changed'):self.reader(before,dict(after,**{'graph.py':'changed'}))
    def test_geometry_and_outer_partition_rejected(self):
        with self.assertRaisesRegex(ValueError,'geometry'):self.reader({}, {},geometry='changed')
        with self.assertRaisesRegex(ValueError,'Inner partition'):self.reader({}, {},part='outer_test')
    def test_reader_requires_explicit_ram_budget(self):
        ds=self.reader({}, {})
        with self.assertRaisesRegex(RuntimeError,'RAM budget'):ds.record(0)
    def batch(self):return seal_materialization(collate([(synthetic(),0),(synthetic(),1)]),[{'debug':0},{'debug':1}])
    def test_verified_tensor_mutation_rejected(self):
        b=verified(self.batch());check_verified(b)
        b.target_patches[0,0,0,0,0]+=1
        with self.assertRaises(ValueError):check_verified(b)
    def test_invalid_edge_and_coverage_not_bypassed(self):
        for mode in ('edge','coverage','finite'):
            b=self.batch()
            if mode=='edge':b.graph[b.graph.edge_types[0]].edge_index[0,0]=-1
            elif mode=='coverage':del b.graph.sampled_counts
            else:b.target_patches[0,0,0,0,0]=float('nan')
            with self.assertRaises(ValueError):verified(b)
    def test_original_topology_not_quotiented(self):
        b=self.batch();before={r:e.clone() for r,e in b.graph.edge_index_dict.items()}
        v=verified(b)
        for r,e in before.items():self.assertTrue(torch.equal(e,v.graph[r].edge_index))
    def test_transition_rejects_unrelated_changes_and_repeated_transition(self):
        from l0_regions.training import FORMAT,hash_state
        original=dict(format=FORMAT,identity=dict(workers=16,source={},graph_representation=MODE))
        original['content_sha256']=hash_state(original)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'checkpoint.pt';torch.save(original,path)
            with self.assertRaisesRegex(ValueError,'cannot also change'):transition(path,dict(workers=8,source={}),None)
            with self.assertRaisesRegex(ValueError,'Already fine'):transition(path,dict(workers=16,source={}),None)
    def test_transition_rejects_bad_checkpoint_digest(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'checkpoint.pt';torch.save(dict(format='bad',content_sha256='bad'),path)
            with self.assertRaisesRegex(ValueError,'Invalid original checkpoint'):transition(path,{},None)
    def test_launcher_preserves_saved_settings(self):
        from tools.run_uncoarsened_sage import arguments
        identity=dict(debug=False,resource_limits=dict(cuda_bytes=40*2**30,rss_bytes=192*2**30),
            resident_budget_bytes=128*2**30,workers=16,candidates=[32,48,64],profile_policy='research-report',
            activation_storage='retained',support_training=dict(patients=16),
            execution_pipeline=dict(mode='overlapped',device_cache_bytes=8*2**30,sage_workspace_bytes=512*2**20))
        def call():return arguments(dict(identity=identity),cache='region',fine_cache='original',resume='latest',output='new')
        args=call()
        for k,v in (('--workers','16'),('--support-patients','16'),('--cuda-gib','40.0'),('--sage-workspace-mib','512')):
            self.assertEqual(args[args.index(k)+1],v)
        self.assertEqual(args[args.index('--batch-candidates')+1:args.index('--batch-candidates')+4],['32','48','64'])
        self.assertIn('--resume-without-coarsening',args)
        identity['graph_representation']=MODE
        self.assertNotIn('--resume-without-coarsening',call())


if __name__=='__main__':unittest.main(verbosity=2)
