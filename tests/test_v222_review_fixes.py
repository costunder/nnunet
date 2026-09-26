"""Independent coordinate and ownership counterexamples; no medical scores."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock,patch
import torch
from tools.v222_review_contracts import (cnn_lattice,input_grid_to_feature_grid,
    grouped_support,resolve_feature_contract,validate_group_resume,_original_support)
from hiercp_v222.deterministic_sampling import sample_nodes
from tools.v222_resume_guard import assert_source_runs_idle,output_argument

ROOT=Path(__file__).resolve().parents[1]


class CoordinateTests(unittest.TestCase):
    def test_nominal_ct_voxel_locations_not_just_grid_sample_equivalence(self):
        axis=torch.arange(12,dtype=torch.float32)*4
        z,y,x=torch.meshgrid(axis,axis,axis,indexing='ij')
        features=torch.stack((x,y,z))[None].repeat(2,1,1,1,1).requires_grad_()
        positions=torch.tensor([[16.,16.,16.],[23.5,23.5,23.5],[40.,40.,40.],[47.,47.,47.],[8.,20.,32.]])
        grid=positions/47*2-1
        owners=torch.tensor([0,0,0,0,1])
        legacy=sample_nodes(features,grid,owners)
        corrected=sample_nodes(features,input_grid_to_feature_grid(grid),owners)
        torch.testing.assert_close(legacy[1],torch.full((3,),22.),atol=1e-5,rtol=0)
        torch.testing.assert_close(corrected,positions.clamp(0,44),atol=1e-5,rtol=0)
        corrected.sum().backward()
        self.assertTrue(torch.isfinite(features.grad).all())
        self.assertAlmostEqual(float(features.grad.sum()),15.,places=5)

    def test_actual_cnn_stride_padding_lattice(self):
        from hiercp.model import PatchFeatureEncoder3D
        with torch.device('meta'):encoder=PatchFeatureEncoder3D(1,12,32)
        self.assertEqual(cnn_lattice(encoder),((12,12,12),(4,4,4),(0.,0.,0.)))
        encoder.down1[0].padding=(0,0,0)
        with self.assertRaisesRegex(ValueError,'lattice changed'):cnn_lattice(encoder)

    def test_coordinate_contract_cannot_silently_change_on_resume(self):
        self.assertEqual(resolve_feature_contract(),'stride4')
        self.assertEqual(resolve_feature_contract({}),'legacy')
        saved=dict(execution_policy=dict(feature_coordinates='stride4'))
        self.assertEqual(resolve_feature_contract(saved),'stride4')
        self.assertEqual(resolve_feature_contract(dict(feature_coordinates='stride4')),'stride4')
        with self.assertRaisesRegex(ValueError,'separate training'):resolve_feature_contract({},'stride4')
        with self.assertRaisesRegex(ValueError,'separate training'):resolve_feature_contract(saved,'legacy')

    def test_model_retains_corrected_sampler_after_install_scope(self):
        from tools.v222_review_contracts import installed
        from hiercp_v222.v1_cache import configuration
        from hiercp_v222.v1_local import V1LocalEncoder
        _,base=configuration()
        with torch.device('meta'),installed('stride4'):encoder=V1LocalEncoder(base)
        x=(torch.arange(12).float()*4)[None,None,None,None,:].expand(1,1,12,12,12)
        result=encoder._sample_dense_features(x,torch.zeros(1,3),torch.zeros(1,dtype=torch.long))
        self.assertAlmostEqual(result.item(),23.5,places=5)


class GroupTests(unittest.TestCase):
    def memory(self,groups):
        n=len(groups)*2
        return dict(patient_groups=groups,owners=torch.arange(len(groups)).repeat_interleave(2),
            embeddings=torch.arange(n*128).reshape(n,128).float(),classes=torch.tensor([0,1]*len(groups)),
            donor_groups=['D']*n)

    def test_two_scans_of_one_other_patient_are_not_two_tasks(self):
        memory=self.memory(['A','A','B'])
        self.assertEqual(len(_original_support(memory,'B')[1].unique()),2)
        with self.assertRaisesRegex(ValueError,'distinct patient'):grouped_support(memory,'B')

    def test_merge_same_patient_and_exclude_donor_identity(self):
        memory=self.memory(['A','A','B','C']);memory['donor_groups'][0]='C'
        values,owners,classes=grouped_support(memory,'C')
        self.assertEqual(owners.tolist(),[0,0,0,1,1])
        self.assertTrue(torch.equal(values,memory['embeddings'][1:6]))
        self.assertTrue(torch.equal(classes,memory['classes'][1:6]))

    def test_current_one_case_one_group_path_is_bitwise_preserved(self):
        memory=self.memory(['A','B','C'])
        for actual,expected in zip(grouped_support(memory,'C'),_original_support(memory,'C')):
            self.assertTrue(torch.equal(actual,expected))

    def test_legacy_multiscan_plan_cannot_be_relabelled(self):
        meta=dict(split=dict(inner_train=['A1','A2']),identities=dict(cases={'A1':dict(patient_group='A'),'A2':dict(patient_group='A')}))
        with self.assertRaisesRegex(ValueError,'multi-case'):validate_group_resume({},meta)


class ProcessOwnershipTests(unittest.TestCase):
    def test_new_entrypoint_is_detected(self):
        self.assertEqual(output_argument(['python','tools/run_v222_process_runtime.py','--output=/old/training']),'/old/training')

    def test_exact_parent_worker_allowed_but_source_parent_not_exempt(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'work') as tmp:
            run=Path(tmp);(run/'worker.json').write_text(json.dumps(dict(pid=123,created_at=456)))
            parent=Mock(pid=123);parent.create_time.return_value=456;parent.status.return_value='running'
            parent.cmdline.return_value=['python','tools/run_v222_server.py','--worker','--output',str(run)]
            own=Mock(pid=os.getpid());own.parents.return_value=[parent];own.username.return_value='test'
            factory=lambda pid:parent if pid==123 else own
            with patch('tools.v222_resume_guard.psutil.Process',side_effect=factory),patch('tools.v222_resume_guard.psutil.process_iter',return_value=[]):
                self.assertTrue(assert_source_runs_idle(run/'paired_cache/index.json',output=run/'training')['source_runs_idle'])
                with self.assertRaisesRegex(RuntimeError,'source worker'):
                    assert_source_runs_idle(run/'paired_cache/index.json',output=run/'other/training')
                own.parents.return_value=[]
                with self.assertRaisesRegex(RuntimeError,'source worker'):
                    assert_source_runs_idle(run/'paired_cache/index.json',output=run/'training')

    def test_actual_parent_child_source_cache_admission(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'work') as tmp:
            root=Path(tmp);helper=root/'run_v222_server.py';run=root/'run';run.mkdir()
            child=(f'import sys;sys.path.insert(0,{str(ROOT)!r});'
                'from tools.v222_resume_guard import assert_source_runs_idle;'
                f'print(assert_source_runs_idle({str(run/"paired_cache/index.json")!r},output={str(run/"training")!r}))')
            helper.write_text('import os,json,subprocess,sys,psutil\nfrom pathlib import Path\n'
                f'Path({str(run/"worker.json")!r}).write_text(json.dumps(dict(pid=os.getpid(),created_at=psutil.Process().create_time())))\n'
                f'result=subprocess.run([sys.executable,"-c",{child!r}],capture_output=True,text=True)\n'
                'print(result.stdout);print(result.stderr,file=sys.stderr)\n'
                'assert result.returncode==0\n')
            result=subprocess.run([sys.executable,str(helper),'--worker','--output',str(run)],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn('source_runs_idle',result.stdout)


class CalibrationTests(unittest.TestCase):
    def test_worker_layout_and_production_prefetch_path(self):
        from tools import v222_process_loader as producer
        self.assertEqual([producer.producer_layout(w) for w in (0,2,4,8)],[(1,0),(2,1),(4,1),(4,2)])
        events=[]
        class Loader:
            def __init__(self,dataset,workers):
                self.depth,self.decode_width=producer.producer_layout(workers)
            def batches(self,groups,epoch):
                events.append(len(groups))
                for ids in groups:
                    self.assert_ids=ids
                    yield Mock(graph=Mock(num_nodes=17,num_edges=51))
            def make(self,*args):raise AssertionError('Calibration must use production stream')
            def close(self):pass
        with tempfile.TemporaryDirectory(dir=ROOT/'work') as tmp,patch.object(producer,'ProcessPairLoader',Loader),patch.object(producer,'close_producers'):
            Loader.producer_count=4
            selected=producer.calibrate_workers(Mock(),[0,1],Path(tmp))
            self.assertIn(selected,(0,2,4,8))
            self.assertEqual(events,[4]*8)
            receipt=json.loads((Path(tmp)/'loader_calibration.json').read_text())
            self.assertFalse(receipt['gpu_overlap_measured'])


if __name__=='__main__':unittest.main()
