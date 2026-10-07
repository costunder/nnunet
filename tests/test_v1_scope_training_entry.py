"""Full-scope orchestration/worker/checkpoint units; no CT or GPU claim."""
from __future__ import annotations

import ast
import copy
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import unittest
from unittest.mock import patch
import uuid

from hiercp_v1x import scope_training_entry as entry

ROOT = Path(__file__).resolve().parents[1]


def request():
    experiment = ROOT/'work/scope_training_entry_UNIT_bounded'
    native = ROOT/'work/scope_training_entry_UNIT_native'
    return dict(format=entry.FORMAT, source=str(experiment/'source/v1.0'),
        config=str(experiment/'configs/v1.0.json'), experiment=str(experiment),
        phase='train', source_experiment=str(native), native_cache=str(native/'shared/cache'),
        prototype_bank=str(experiment/'shared/prototype_bank.pt'), regions=str(experiment/'shared/regions'),
        cache=str(experiment/'shared/cache'), split=str(experiment/'shared/split.json'),
        medical_root=str(ROOT/'UNIT_MEDICAL'), margin_mm=10, workers=1,
        cuda_gib=40., rss_gib=192., gpu=3)


class RequestUnits(unittest.TestCase):
    def test_explicit_full_request_preserves_scope_device_and_budgets(self):
        value = request()
        self.assertEqual(entry.validate_request(value), value)

    def test_required_fields_are_not_filled_by_defaults(self):
        for key in request():
            if key == 'format':
                continue
            value = request(); del value[key]
            with self.subTest(key=key), self.assertRaises(ValueError):
                entry.validate_request(value)

    def test_invalid_scope_serial_worker_gpu_budget_and_phase_rejected(self):
        for key, bad in [('margin_mm', True), ('margin_mm', 5), ('workers', 0),
                ('gpu', -1), ('cuda_gib', float('nan')), ('rss_gib', 0), ('phase', 'generate')]:
            value = request(); value[key] = bad
            with self.subTest(key=key, bad=bad), self.assertRaises(ValueError):
                entry.validate_request(value)

    def test_source_results_and_caches_cannot_be_reused_as_outputs(self):
        for key in ('source', 'config', 'prototype_bank', 'regions', 'cache', 'split'):
            value = request(); value[key] = str(ROOT/'UNIT_ESCAPED')
            with self.subTest(key=key), self.assertRaises(ValueError):
                entry.validate_request(value)
        value = request(); value['experiment'] = value['source_experiment']
        with self.assertRaises(ValueError): entry.validate_request(value)
        value = request(); value['native_cache'] = value['cache']
        with self.assertRaises(ValueError): entry.validate_request(value)

    def test_scope_worker_initializer_is_picklable(self):
        original = entry.ScopeWorkerInitializer('/UNIT/source', 10., 'a'*64, True)
        self.assertEqual(pickle.loads(pickle.dumps(original)), original)

    def test_loader_kwargs_keeps_native_parameters_and_adds_parallel_worker_hook(self):
        from types import SimpleNamespace
        original = lambda **kw: dict(num_workers=kw['workers'], pin_memory=kw['pin_memory'],
            prefetch_factor=kw['prefetch_factor'], persistent_workers=kw['persistent_workers'])
        pipeline = SimpleNamespace(_loader_kwargs=original)
        hook = entry.ScopeWorkerInitializer('/UNIT/source', 10., 'a'*64)
        entry.install_loader_hook(pipeline, hook)
        options = dict(workers=16, pin_memory=True, prefetch_factor=3, persistent_workers=True)
        result = pipeline._loader_kwargs(**options)
        self.assertEqual(result['worker_init_fn'], hook)
        self.assertEqual({k:v for k,v in result.items() if k != 'worker_init_fn'}, original(**options))
        options['workers'] = 0
        self.assertNotIn('worker_init_fn', pipeline._loader_kwargs(**options))


SCRIPT = r'''
import hashlib,json,os,random,sys,uuid,zipfile
from pathlib import Path
import numpy as np
import torch
from hiercp_v1x import bounded_scope,scope_training_entry as entry
from hiercp_v1x.scope_probe_support import activate_original
from tests.artifacts import unit_artifact_root
root=Path.cwd()
out=unit_artifact_root()/('scope_training_entry_UNIT_'+uuid.uuid4().hex)
source=out/'source'
source.mkdir(parents=True)
with zipfile.ZipFile(root/'versions/v1/pipeline_v1_source.zip') as bundle:
    bundle.extractall(source)
activate_original(source)
receipt=bounded_scope.install(10,expected_snapshot_root=source)
entry.install_checkpoint_binding(receipt)
from hiercp import contracts,model,pipeline
config=json.loads((source/'config/train.json').read_text())
net=model.HierarchicalPyGPlacementModel(**config['model'])
assert sum(p.numel() for p in net.parameters())==10434532
checkpoint={'architecture_version':net.architecture_version,'geometry_contract':contracts.GEOMETRY_CONTRACT,
            'graph_config':config['graph'],'state_dict':net.state_dict()}
contracts.require_current_checkpoint(checkpoint)
for kind in ['architecture','missing','marker']:
    changed={**checkpoint,'state_dict':dict(checkpoint['state_dict'])}
    if kind=='architecture':changed['architecture_version']=contracts.ARCHITECTURE_VERSION
    elif kind=='missing':del changed['state_dict'][entry.MARKER]
    else:
        changed['state_dict'][entry.MARKER]=changed['state_dict'][entry.MARKER].clone()
        changed['state_dict'][entry.MARKER][0]^=1
    try:contracts.require_current_checkpoint(changed)
    except ValueError:pass
    else:raise AssertionError('Unbound checkpoint accepted:'+kind)
    if kind!='architecture':
        try:net.load_state_dict(changed['state_dict'])
        except ValueError:pass
        else:raise AssertionError('Unbound direct state load accepted:'+kind)
initializer=entry.ScopeWorkerInitializer(str(source),10,receipt['contract_sha256'],True)
random.seed(19);np.random.seed(19);torch.manual_seed(19)
before=(random.getstate(),np.random.get_state(),torch.get_rng_state())
initializer(0)
assert random.getstate()==before[0]
assert np.array_equal(np.random.get_state()[1],before[1][1])
assert torch.equal(torch.get_rng_state(),before[2])
req={'experiment':str(out),'scope_UNIT':True}
session=entry._install_training_observation(pipeline,out,req,receipt)
try:
    entry.install_loader_hook(pipeline,initializer)
    from unittest.mock import patch
    captured=[]
    def intercept(args):
        captured.append(args)
    # Reaching the original guarded invocation verifies the complete AST
    # overlay compiles; it does not run a dataset or optimizer.
    with patch.object(pipeline,'_guard_reduction_overrides',side_effect=RuntimeError('UNIT stop before data')):
        try:pipeline.run_train(type('Args',(),{})())
        except RuntimeError as error:assert str(error)=='UNIT stop before data'
        else:raise AssertionError('Full training entered without explicit data')
finally:session.close()
print(json.dumps({'UNIT':True,'CT_run':False,'GPU_run':False,'checks':8}))
'''


class ArchivedWorkerCheckpointUnits(unittest.TestCase):
    def test_exact_archive_checkpoint_binding_worker_rng_and_full_epoch_overlay(self):
        result = subprocess.run([sys.executable, '-u', '-c', SCRIPT], cwd=ROOT,
            env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'), capture_output=True, text=True, timeout=90)
        self.assertEqual(result.returncode, 0, result.stdout + '\n' + result.stderr)
        self.assertTrue(json.loads(result.stdout.strip().splitlines()[-1])['UNIT'])

    def test_worker_does_not_initialize_or_reseed_cuda(self):
        source = Path(entry.__file__).read_text(encoding='utf8')
        tree = ast.parse(source)
        worker = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'ScopeWorkerInitializer')
        self.assertNotIn('cuda', ast.unparse(worker))

    def test_pipeline_invocation_has_no_epoch_batch_worker_subset_overrides(self):
        source = Path(entry.__file__).read_text(encoding='utf8')
        for flag in ('--epochs', '--batch-size', '--num-workers', '--max-cases', '--overwrite'):
            self.assertNotIn(flag, source)
        self.assertIn("'paired_native=False'".replace("'", ''), source.replace(' ', ''))


if __name__ == '__main__':
    unittest.main()
