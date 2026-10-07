"""Isolated actual-archive CPU constructor/checkpoint UNIT; no training/CT/GPU.

Untrained UNIT serialization lives in a new work directory. This tests the
production entry identity wrapper without entering its CUDA training main.
"""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]


SCRIPT = r'''
import copy,json,pickle,sys,unittest,uuid
from pathlib import Path
import torch
from hiercp_v1x.scope_probe_support import activate_original,state_digest,_sha
from hiercp_v1x.contracts import canonical_hash
from hiercp_v1x import bounded_scope
from hiercp_v1x.half_a_entry import install_identity
from hiercp_v1x.half_a_model import HalfALocalCNN,MARKER_KEY,half_a_identity
from hiercp_v1x.half_a_training import MARKER
from tools.verify_v1_half_a_learning import half_a_state_digest

from tests.artifacts import unit_artifact_root
root=Path.cwd()
source=root/'work/v14_matched_learning_DEBUG_20261004_r3/source/v1.0'
proof=activate_original(source)
from hiercp import contracts,model,tensor,pipeline
config=json.loads((source/'config/train.json').read_text(encoding='utf8'))
scope=bounded_scope.install(10,expected_snapshot_root=source)
torch.manual_seed(42)
native=model.HierarchicalPyGPlacementModel(**config['model'])
native_state=copy.deepcopy(native.state_dict())
native_upper={name:value for name,value in native_state.items() if not name.startswith('local_encoder.')}
native_rng=torch.get_rng_state().clone()
expected_native=json.loads((root/'work/v14_matched_learning_DEBUG_20261004_r3/report.json').read_text(encoding='utf8'))['initial_neural_sha256']
assert state_digest(native_state)==expected_native
assert sum(p.numel() for p in native.parameters())==10434532
assert not torch.cuda.is_initialized()
out=unit_artifact_root()/('half_a_entry_archived_CPU_UNIT_'+uuid.uuid4().hex)
output=out/'results/half_A'
output.mkdir(parents=True)
receipt={'experiment':str(out),'contract_sha256':canonical_hash(dict(UNIT=True,
    half_a=half_a_identity()['contract_sha256'],scope=scope['contract_sha256']))}
install_identity(receipt,scope)
torch.manual_seed(42)
net=model.HierarchicalPyGPlacementModel(**config['model'])
own_state=copy.deepcopy(net.state_dict())
payload={'UNIT':True,'untrained':True,'full_training':False,'production_ready':False,
    'architecture_version':net.architecture_version,'geometry_contract':contracts.GEOMETRY_CONTRACT,
    'graph_config':config['graph'],'model_kwargs':config['model'],'state_dict':own_state}


class ActualArchiveEntryIdentity(unittest.TestCase):
    def unit_completed_pair(self):
        """Tiny synthetic UNIT evidence for the actual archived loader guard.

        Completion fields exercise metadata validation only: these files contain
        untrained marker tensors, never trained model weights or medical data.
        Full model serialization/reload is covered separately below.
        """
        state={name:copy.deepcopy(own_state[name]) for name in
            (MARKER,MARKER_KEY,'v1x_bounded_scope_digest','local_encoder._extra_state')}
        common={'UNIT':True,'untrained':True,'full_training':False,'production_ready':False,
            'architecture_version':net.architecture_version,'state_dict':state,
            'model_kwargs':copy.deepcopy(config['model']),
            'graph_config':copy.deepcopy(config['graph']),
            'geometry_contract':contracts.GEOMETRY_CONTRACT,
            'ct_clip':tuple(float(v) for v in config['ct_clip']),
            'cache_publication':{'UNIT_metadata_only':True,'config_sha256':'11'*32,
                'index_sha256':'22'*32,'complete_sha256':'33'*32},
            'training_signature':{'format':'hiercp_training_signature_v1',
                'UNIT_metadata_only':True,'seed':42,'target_epochs':40},
            'validation_policy':{'format':'hiercp_fixed_validation_v1',
                'epoch':config['training']['fixed_validation_epoch'],
                'checkpoint_order':['mrr','acc','margin','-ranking','-consistency'],
                'metric_precision':config['training']['checkpoint_metric_precision']},
            'preflight_calibration':{'UNIT_metadata_only':True,
                'format':'hiercp_preflight_calibration_v2','selected_batch_size':16,
                'selected_num_workers':8},
            'target_epochs':40,'training_complete':True}
        last={**copy.deepcopy(common),'format':'hiercp_training_state_v1',
            'epoch':40,'best_epoch':11,'best_checkpoint':str(output/'checkpoint_best.pt')}
        best={**copy.deepcopy(common),'epoch':11,'completed_epoch':40}
        # Best and last weights need not be equal; only identity and publication
        # metadata must match. This UNIT tensor is deliberately absent in last.
        best['state_dict']['UNIT_best_tensor']=torch.tensor([1.])
        return output/('completed_last_UNIT_'+uuid.uuid4().hex+'.pt'),last,best

    def save_unit(self,path,value):
        # Every target is in this invocation's newly created UNIT directory.
        with path.open('wb') as stream:torch.save(value,stream)

    def test_constructor_keeps_archived_upper_weights_and_rng(self):
        self.assertEqual(net.architecture_version,contracts.ARCHITECTURE_VERSION
            +'|bounded_scope_'+scope['contract_sha256']+'|half_A_'+receipt['contract_sha256'])
        self.assertIsInstance(net.local_encoder,HalfALocalCNN)
        self.assertTrue(torch.equal(torch.get_rng_state(),native_rng))
        for name,value in native_upper.items():
            self.assertTrue(torch.equal(own_state[name],value),name)
        self.assertEqual(state_digest({name:own_state[name] for name in native_upper}),state_digest(native_upper))
        self.assertFalse(torch.cuda.is_initialized())

    def test_replacement_has_only_live_declared_parameter_owners(self):
        parameters=dict(net.named_parameters())
        local={name.split('.')[1] for name in parameters if name.startswith('local_encoder.')}
        self.assertEqual(local,{'cnn','project','fuse'})
        self.assertTrue(all(p.device.type=='cpu' and p.requires_grad for p in parameters.values()))
        self.assertEqual({id(p) for p in parameters.values()},{id(p) for p in net.trainable_parameters()})
        for owner in ('patient_encoder','prototype_encoder','patient_readout','population_readout','score_head'):
            old={name:value for name,value in native.named_parameters() if name.startswith(owner+'.')}
            new={name:value for name,value in parameters.items() if name.startswith(owner+'.')}
            self.assertEqual(set(old),set(new))
            self.assertEqual(sum(p.numel() for p in old.values()),sum(p.numel() for p in new.values()))
        # Structural ownership is checked here. Actual gradient connectivity
        # remains the separate real-CT/CUDA DEBUG tool's responsibility.

    def test_own_checkpoint_metadata_and_actual_serialized_state_reload(self):
        contracts.require_current_checkpoint(payload)
        path=output/'own_initial_UNTRAINED_UNIT.pt'
        with path.open('xb') as stream:torch.save(payload,stream)
        loaded=tensor.torch_load_compat(path,map_location='cpu')
        loaded_from_public_loader=tensor.load_checkpoint(path,torch.device('cpu'))
        self.assertEqual(loaded_from_public_loader['architecture_version'],net.architecture_version)
        contracts.require_current_checkpoint(loaded)
        torch.manual_seed(917)
        resumed=model.HierarchicalPyGPlacementModel(**config['model'])
        result=resumed.load_state_dict(loaded['state_dict'])
        self.assertEqual(result.missing_keys,[])
        self.assertEqual(result.unexpected_keys,[])
        self.assertEqual(half_a_state_digest(resumed.state_dict()),half_a_state_digest(own_state))
        self.assertFalse(torch.cuda.is_initialized())

    def test_legacy_archived_state_rejected_even_with_permissive_load(self):
        legacy={**payload,'architecture_version':contracts.ARCHITECTURE_VERSION,'state_dict':native_state}
        with self.assertRaises(ValueError):contracts.require_current_checkpoint(legacy)
        with self.assertRaises((ValueError,RuntimeError)):net.load_state_dict(native_state,strict=False)

    def test_all_three_markers_rejected_by_preload_metadata_check(self):
        for marker in (MARKER,'v1x_bounded_scope_digest',MARKER_KEY):
            for alteration in ('missing','changed'):
                with self.subTest(marker=marker,alteration=alteration):
                    changed={**payload,'state_dict':dict(own_state)}
                    if alteration=='missing':del changed['state_dict'][marker]
                    else:
                        value=changed['state_dict'][marker].clone();value[0]^=1
                        changed['state_dict'][marker]=value
                    with self.assertRaises((ValueError,RuntimeError)):
                        contracts.require_current_checkpoint(changed)

    def test_completed_skip_deserialization_rejects_wrong_local_contract(self):
        changed={**payload,'state_dict':dict(own_state)}
        extra=dict(changed['state_dict']['local_encoder._extra_state'])
        extra['native_spacing_v22_equivalence']=True
        changed['state_dict']['local_encoder._extra_state']=extra
        path=output/'wrong_local_contract_UNTRAINED_UNIT.pt'
        with path.open('xb') as stream:torch.save(changed,stream)
        with self.assertRaises((ValueError,RuntimeError)):
            tensor.torch_load_compat(path,map_location='cpu')
        with self.assertRaises((ValueError,RuntimeError)):
            net.load_state_dict(changed['state_dict'],strict=False)

    def test_completed_last_checks_valid_best_using_actual_archived_loader(self):
        path,last,best=self.unit_completed_pair()
        self.save_unit(path,last)
        self.save_unit(output/'checkpoint_best.pt',best)
        loaded=tensor.torch_load_compat(path,map_location='cpu')
        self.assertEqual(loaded['epoch'],40)
        self.assertEqual(loaded['best_epoch'],11)
        self.assertTrue(loaded['UNIT'])
        self.assertNotIn('UNIT_best_tensor',loaded['state_dict'])
        self.assertFalse(torch.cuda.is_initialized())

    def test_completed_last_rejects_missing_or_corrupt_best(self):
        for corruption in ('missing','truncated'):
            with self.subTest(corruption=corruption):
                path,last,best=self.unit_completed_pair()
                self.save_unit(path,last)
                best_path=output/'checkpoint_best.pt'
                if best_path.exists():best_path.unlink()
                if corruption=='truncated':best_path.write_bytes(b'UNIT truncated checkpoint')
                with self.assertRaises((FileNotFoundError,RuntimeError,EOFError,pickle.UnpicklingError)):
                    tensor.torch_load_compat(path,map_location='cpu')

    def test_completed_last_rejects_best_with_wrong_actual_markers(self):
        for marker in (MARKER,'v1x_bounded_scope_digest',MARKER_KEY):
            for alteration in ('missing','changed'):
                with self.subTest(marker=marker,alteration=alteration):
                    path,last,best=self.unit_completed_pair()
                    if alteration=='missing':del best['state_dict'][marker]
                    else:best['state_dict'][marker][0]^=1
                    self.save_unit(path,last)
                    self.save_unit(output/'checkpoint_best.pt',best)
                    with self.assertRaises((ValueError,RuntimeError)):
                        tensor.torch_load_compat(path,map_location='cpu')

    def test_completed_last_rejects_best_completion_or_publication_mismatch(self):
        wrong={'architecture_version':'UNIT_wrong_architecture','training_complete':False,
            'completed_epoch':39,'target_epochs':39,'epoch':12}
        for key in ('model_kwargs','graph_config','geometry_contract','ct_clip',
                    'cache_publication','training_signature','validation_policy','preflight_calibration'):
            wrong[key]={'UNIT_changed_metadata':True}
        for key,value in wrong.items():
            with self.subTest(key=key,alteration='changed'):
                path,last,best=self.unit_completed_pair()
                best[key]=value
                self.save_unit(path,last)
                self.save_unit(output/'checkpoint_best.pt',best)
                with self.assertRaises((ValueError,RuntimeError)):
                    tensor.torch_load_compat(path,map_location='cpu')
        for key in ('model_kwargs','graph_config','geometry_contract','ct_clip',
                    'cache_publication','training_signature','validation_policy','preflight_calibration'):
            with self.subTest(key=key,alteration='missing_in_both'):
                path,last,best=self.unit_completed_pair()
                del last[key],best[key]
                self.save_unit(path,last)
                self.save_unit(output/'checkpoint_best.pt',best)
                with self.assertRaises((ValueError,RuntimeError)):
                    tensor.torch_load_compat(path,map_location='cpu')
        for key in ('epoch','target_epochs'):
            with self.subTest(last_key=key):
                path,last,best=self.unit_completed_pair()
                last[key]=39
                self.save_unit(path,last)
                self.save_unit(output/'checkpoint_best.pt',best)
                with self.assertRaises((ValueError,RuntimeError)):
                    tensor.torch_load_compat(path,map_location='cpu')
        # Original run_train also skips when epoch>=40, even with a false
        # completion flag. Such a state must fail strict completed identity;
        # truthy non-booleans must not bypass the same deserialization guard.
        for flag in (False,1,'UNIT_truthy_completion'):
            with self.subTest(last_completion_flag=flag):
                path,last,best=self.unit_completed_pair()
                last['training_complete']=flag
                self.save_unit(path,last)
                self.save_unit(output/'checkpoint_best.pt',best)
                with self.assertRaises((ValueError,RuntimeError)):
                    tensor.torch_load_compat(path,map_location='cpu')

    def test_incomplete_last_can_resume_without_completed_best(self):
        path,last,best=self.unit_completed_pair()
        last.update(epoch=9,training_complete=False)
        self.save_unit(path,last)
        best_path=output/'checkpoint_best.pt'
        if best_path.exists():best_path.unlink()
        loaded=tensor.torch_load_compat(path,map_location='cpu')
        self.assertEqual(loaded['epoch'],9)
        self.assertIs(loaded['training_complete'],False)
        self.assertFalse(torch.cuda.is_initialized())


suite=unittest.defaultTestLoader.loadTestsFromTestCase(ActualArchiveEntryIdentity)
result=unittest.TextTestRunner(verbosity=2).run(suite)
assert not torch.cuda.is_initialized()
audit=dict(UNIT=True,archived_constructor=True,CPU_only=True,CT_run=False,
    GPU_run=False,training_run=False,source=proof['source'],output=str(out),
    archive_sha256=proof['archive_sha256'],
    source_files_preserved=all(_sha(source/name)==identity for name,identity in proof['verified_files'].items()),
    original_native_initial_state_sha256=expected_native,common_upper_initial_sha256=state_digest(native_upper),
    new_parameters=sum(p.numel() for p in net.parameters()),
    structural_optimizer_parameter_interface_verified=True,gradient_connectivity_tested=False,
    checks=result.testsRun,failures=len(result.failures),errors=len(result.errors))
with (out/'audit.json').open('x',encoding='utf8') as stream:json.dump(audit,stream,indent=2,allow_nan=False)
print(json.dumps(audit))
if not result.wasSuccessful():raise AssertionError('Archived Half-A entry identity UNIT failed')
'''


class ArchivedHalfAEntryUnit(unittest.TestCase):
    def test_actual_archive_constructor_checkpoint_identity_and_reload(self):
        result = subprocess.run([sys.executable, '-B', '-u', '-c', SCRIPT], cwd=ROOT,
            env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES=''),
            capture_output=True, text=True, timeout=90)
        self.assertEqual(result.returncode, 0, result.stdout+'\n'+result.stderr)


if __name__ == '__main__':
    unittest.main()
