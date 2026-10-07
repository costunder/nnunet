"""Actual archived constructor/loader regression; metadata only, no learning claim."""
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = r'''
import copy, json, uuid
from pathlib import Path
import torch
from hiercp_v1x.scope_probe_support import activate_original, state_digest
from hiercp_v1x import bounded_scope
from hiercp_v1x.half_b_entry import install_identity
from hiercp_v1x.half_b_training import MARKER
from hiercp_v1x.half_b_model import MARKER_KEY
from tests.artifacts import unit_artifact_root
root=Path.cwd()
source=root/'work/v14_matched_learning_DEBUG_20261004_r3/source/v1.0'
activate_original(source)
scope=bounded_scope.install(10, expected_snapshot_root=source)
from hiercp import model, contracts, tensor
cfg=json.loads((source/'config/train.json').read_text())
torch.manual_seed(42)
native=model.HierarchicalPyGPlacementModel(**cfg['model'])
before={k:v.clone() for k,v in native.state_dict().items() if k.startswith('local_encoder.')}
native_rng=torch.get_rng_state().clone()
folder=unit_artifact_root()/('half_b_entry_CPU_UNIT_'+uuid.uuid4().hex)
output=folder/'results/half_B'
output.mkdir(parents=True)
receipt=dict(experiment=str(folder),contract_sha256='bc'*32)
install_identity(receipt,scope)
torch.manual_seed(42)
net=model.HierarchicalPyGPlacementModel(**cfg['model'])
state=copy.deepcopy(net.state_dict())
assert torch.equal(torch.get_rng_state(),native_rng)
assert all(torch.equal(state[k],v) for k,v in before.items())
assert not any(k.startswith(('patient_encoder.','prototype_encoder.','score_head.','patient_readout.','population_readout.')) for k in state)
assert net.architecture_version.endswith('|half_B_'+receipt['contract_sha256'])
net.load_state_dict(state)
for key in (MARKER,MARKER_KEY,'v1x_bounded_scope_digest'):
    bad=copy.deepcopy(state);bad[key]=bad[key].clone();bad[key][0]^=1
    try:net.load_state_dict(bad)
    except (ValueError,RuntimeError):pass
    else:raise AssertionError('Actual wrong identity accepted: '+key)
bad=copy.deepcopy(state);bad['half_b._extra_state']['support_policy']='wrong'
try:net.load_state_dict(bad)
except (ValueError,RuntimeError):pass
else:raise AssertionError('Wrong support recipe accepted')
common=dict(UNIT_metadata_only=True,untrained=True,production_ready=False,
 architecture_version=net.architecture_version,state_dict=state,
 model_kwargs=cfg['model'],graph_config=cfg['graph'],geometry_contract=contracts.GEOMETRY_CONTRACT,
 ct_clip=tuple(cfg['ct_clip']),cache_publication={'UNIT':True},training_signature={'UNIT':True},
 validation_policy={'UNIT':True},preflight_calibration={'UNIT':True},target_epochs=40,training_complete=True)
last={**common,'format':'hiercp_training_state_v1','epoch':40,'best_epoch':11}
best={**common,'epoch':11,'completed_epoch':40}
torch.save(best,output/'checkpoint_best.pt')
torch.save(last,output/'checkpoint_best.last.pt')
tensor.torch_load_compat(output/'checkpoint_best.last.pt',map_location='cpu')
best['architecture_version']='foreign_A'
torch.save(best,output/'checkpoint_best.pt')
try:tensor.torch_load_compat(output/'checkpoint_best.last.pt',map_location='cpu')
except ValueError:pass
else:raise AssertionError('Completed last accepted foreign best')
assert not torch.cuda.is_initialized()
print(json.dumps(dict(status='PASS',actual_archived_source=True,
 untrained=True,full_training=False,actual_CUDA=False,
 L0_initial_hash=state_digest(before),checks=9)))
'''


class ActualArchiveEntry(unittest.TestCase):
    def test_actual_constructor_and_checkpoint_guards(self):
        result = subprocess.run([sys.executable, '-B', '-c', SCRIPT], cwd=ROOT,
            capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('"status": "PASS"', result.stdout)


if __name__ == '__main__':
    unittest.main()
