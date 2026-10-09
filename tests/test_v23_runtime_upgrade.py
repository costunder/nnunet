"""UNIT: exact paused cursors and true BEST across a performance-only upgrade."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import torch

from tests.test_v23_training import _handoff_unit_fixture
from hiercp_v1x.contracts import canonical_hash
from hiercp_v1x.u_bridge_training import digest
from hiercp_v1x.v23_training import handoff_single_gpu_checkpoint, _file_sha256
from hiercp_v1x.v23_runtime_upgrade import upgrade_single_gpu_checkpoint, NUMERICAL_KEYS


def fixture(directory, phase='training', cursor=4):
    population, checkpoint, identity, best, proof, request, calibration, config, _, _ = _handoff_unit_fixture(directory)
    source = directory / 'singleton'
    handoff_single_gpu_checkpoint(checkpoint, identity, request, config, calibration, source,
        source_proof=proof, population=population, source_best_checkpoint=best, debug=True)
    checkpoint, identity, best = (source / name for name in
        ('checkpoint_latest.pt', 'training_identity.json', 'checkpoint_best.pt'))
    saved = torch.load(checkpoint, weights_only=False)
    state = saved['state']; state['status'] = 'PAUSED'; state['phase'] = phase
    if phase == 'training':
        state['train_position'] = cursor
        state['train_rows'] = [dict(case_id=case) for case in state['train_order'][:cursor]]
        state['updates'] += 1; state['attempts'] += 1
        saved['optimizer']['state'][0]['step'] += 1
    else:
        state['train_position'] = len(population.train)
        state['train_rows'] = [dict(case_id=case) for case in state['train_order']]
        state['evaluation_position'] = cursor
        state['evaluation_rows'] = [dict(case_id=case) for case in population.val[:cursor]]
        state['updates'] += 17; state['attempts'] += 17
        saved['optimizer']['state'][0]['step'] += 17
    saved.pop('content_sha256'); saved['content_sha256'] = digest(saved); torch.save(saved, checkpoint)
    new = dict(format='UNIT', source={'unit_source':'7'*64}, gpus=[5])
    new['request_sha256'] = canonical_hash(new)
    calibration = dict(calibration, request_sha256=new['request_sha256'])
    config = copy.deepcopy(config)
    config['v23_runtime'].update(prefetch_cpu_chunks=True, pin_cpu_batches=True,
        memoize_cpu_geometry=True, cache_stage_admission=True, geometry_resident_gib_per_rank=128,
        memoize_fixed_views=True, persistent_cpu_workers=True)
    proof = dict(proof, source_commit='b'*40, source_request=request,
        source_files_sha256=request['source'], source_checkpoint_file_sha256=_file_sha256(checkpoint),
        source_identity_file_sha256=_file_sha256(identity), source_best_checkpoint_file_sha256=_file_sha256(best),
        clean_pause_verified=True, new_runtime_full128_backward_verified=True,
        numerical_equivalence_receipt_sha256='8'*64)
    return population, checkpoint, identity, best, proof, new, calibration, config, saved


class RuntimeUpgradeUnit(unittest.TestCase):
    def upgrade(self, directory, values, **changes):
        population, checkpoint, identity, best, proof, request, calibration, config, saved = values
        args = dict(source_proof=proof, population=population, source_best_checkpoint=best, debug=True)
        args.update(changes)
        receipt = upgrade_single_gpu_checkpoint(checkpoint, identity, request, config, calibration,
            directory/'upgraded', **args)
        return receipt, torch.load(directory/'upgraded/checkpoint_latest.pt', weights_only=False)

    def test_partial_training_and_both_pending_validation_cursors_are_preserved(self):
        for phase, cursor in (('training',4), ('full_validation',20), ('full_validation',21)):
            with self.subTest(phase=phase, cursor=cursor), tempfile.TemporaryDirectory(prefix='v23_upgrade_UNIT_', dir=Path.cwd()) as name:
                directory=Path(name); values=fixture(directory,phase,cursor)
                receipt, result=self.upgrade(directory, values); source=values[-1]
                for key in NUMERICAL_KEYS: self.assertEqual(digest(source[key]), digest(result[key]))
                progress=lambda state: digest({k:v for k,v in state.items() if k!='handoff'})
                self.assertEqual(progress(source['state']),progress(result['state']))
                self.assertTrue(receipt['no_new_optimizer_update'])
                old_best=torch.load(values[3],weights_only=False)
                new_best=torch.load(directory/'upgraded/checkpoint_best.pt',weights_only=False)
                for key in NUMERICAL_KEYS: self.assertEqual(digest(old_best[key]),digest(new_best[key]))
                self.assertNotEqual(digest(result['model']),digest(new_best['model']))
                self.assertEqual(source['state']['updates']+receipt['remaining_planned_updates'],603)
                with self.assertRaises(FileExistsError): self.upgrade(directory,values)

    def test_core_policy_or_GPU_change_and_missing_best_are_rejected(self):
        for change in ('model','policy','gpu','best'):
            with self.subTest(change=change), tempfile.TemporaryDirectory(prefix='v23_upgrade_reject_UNIT_',dir=Path.cwd()) as name:
                directory=Path(name); values=list(fixture(directory))
                if change=='model': values[7]['model']['UNIT']=False
                if change=='policy': values[7]['v23_runtime']['target_selection_policy']='native_prefix'
                if change=='gpu':
                    values[5]['gpus']=[6]
                    values[5]['request_sha256']=canonical_hash({k:v for k,v in values[5].items() if k!='request_sha256'})
                with self.assertRaises(ValueError):
                    self.upgrade(directory,values,**({'source_best_checkpoint':None} if change=='best' else {}))
                self.assertFalse((directory/'upgraded').exists())

    def test_source_file_tamper_and_running_checkpoint_are_rejected(self):
        for running in (False,True):
            with self.subTest(running=running), tempfile.TemporaryDirectory(prefix='v23_upgrade_source_UNIT_',dir=Path.cwd()) as name:
                directory=Path(name); values=list(fixture(directory)); checkpoint=values[1]
                if running:
                    saved=values[-1]; saved['state']['status']='RUNNING'
                    saved.pop('content_sha256');saved['content_sha256']=digest(saved);torch.save(saved,checkpoint)
                    values[4]['source_checkpoint_file_sha256']=_file_sha256(checkpoint)
                else:
                    with checkpoint.open('ab') as stream: stream.write(b'changed')
                with self.assertRaises(ValueError): self.upgrade(directory,values)
                self.assertFalse((directory/'upgraded').exists())




"""DEBUG UNIT only: actual pinned original convolutions and overlay ownership.

Synthetic graph tensors exercise operator/chunk boundaries, not native CT,
the full training loss, all1085 native gradients, production throughput or a
release numerical gate. The separate full4/P128/two-view paired gate is required.
"""

import hashlib
import importlib.util
from pathlib import Path, PurePosixPath
import sys
import uuid
import unittest
from unittest.mock import patch
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
from hiercp_v1x import v23_edge_execution as adapter


def _verify_reusable_debug_source(source, archive):
    """Admit already imported sealed bytes; never remove/rebind sys.modules."""
    source = Path(source).resolve(strict=True)
    verified = {}
    with ZipFile(archive) as bundle:
        for name in bundle.namelist():
            if not ((name.startswith('hiercp/') and name.endswith('.py')) or name == 'config/train.json'):
                continue
            relative = PurePosixPath(name)
            if relative.is_absolute() or '..' in relative.parts or '\\' in name or ':' in name:
                raise ValueError('Unsafe pinned archive path')
            target = source.joinpath(*relative.parts)
            expected = hashlib.sha256(bundle.read(name)).hexdigest()
            if (target.is_symlink() or not target.is_file()
                    or not target.resolve(strict=True).is_relative_to(source)
                    or hashlib.sha256(target.read_bytes()).hexdigest() != expected):
                raise ValueError('Already imported DEBUG source differs from sealed archive: ' + name)
            verified[name] = expected
    expected_core = {name for name in verified if name.startswith('hiercp/')}
    actual_core = {path.relative_to(source).as_posix() for path in (source / 'hiercp').rglob('*.py')}
    if len(verified) < 10 or actual_core != expected_core:
        raise ValueError('Incomplete or extra already imported DEBUG source modules')
    for name, module in tuple(sys.modules.items()):
        if name != 'hiercp' and not name.startswith('hiercp.'):
            continue
        filename = getattr(module, '__file__', None)
        if filename is None:
            raise ValueError('Already imported DEBUG module has no verified source: ' + name)
        actual = Path(filename).resolve(strict=True)
        if not actual.is_relative_to(source) or actual.relative_to(source).as_posix() not in expected_core:
            raise ValueError('Another original implementation is already imported: ' + name)
    return dict(source=str(source), archive_sha256=adapter.ARCHIVE_SHA256,
                verified_files=verified, already_imported_sealed_source=True)


class OriginalL0ExecutionDebug(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(min(4, cls.old_threads))
        archive = ROOT / 'versions/v1/pipeline_v1_source.zip'
        if hashlib.sha256(archive.read_bytes()).hexdigest() != adapter.ARCHIVE_SHA256:
            raise ValueError('DEBUG must use exact sealed original archive')
        from hiercp_v1x.historical_checkpoint import _activate
        if 'hiercp' in sys.modules:
            # Earlier geometry UNITs import the unchanged original in this
            # combined process. Reuse only after verifying all archive bytes,
            # loaded module ownership and the exact explicit10mm scope.
            filename = getattr(sys.modules['hiercp'], '__file__', None)
            if filename is None:
                raise ValueError('Already imported original package has no verified source')
            snapshot = Path(filename).resolve(strict=True).parents[1]
            sealed = _verify_reusable_debug_source(snapshot, archive)
            from hiercp_v1x import bounded_scope
            actual = bounded_scope._ACTIVE
            if actual is None:
                actual = bounded_scope.install(10, expected_snapshot_root=snapshot)
            if (actual.get('margin_mm') != 10
                    or actual.get('source_archive_sha256') != adapter.ARCHIVE_SHA256
                    or actual.get('original_module_sha256') != {
                        name: value for name, value in sealed['verified_files'].items()
                        if name.startswith('hiercp/')}):
                raise ValueError('Already activated DEBUG original scope/source differs')
            cls.activation, cls.scope = _activate(snapshot, actual)
            cls.activation['sealed_source_verification'] = sealed
        else:
            # A fresh standalone test process retains its owned source snapshot
            # as DEBUG provenance; no existing source files are overwritten.
            snapshot = ROOT / 'outputs' / ('v23_edge_original_DEBUG_' + uuid.uuid4().hex)
            snapshot.mkdir(parents=True)
            with ZipFile(archive) as bundle:
                for name in bundle.namelist():
                    if not ((name.startswith('hiercp/') and name.endswith('.py')) or name == 'config/train.json'):
                        continue
                    relative = PurePosixPath(name)
                    if relative.is_absolute() or '..' in relative.parts or '\\' in name or ':' in name:
                        raise ValueError('Unsafe pinned archive path')
                    target = snapshot.joinpath(*relative.parts)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(bundle.read(name))
            cls.activation, cls.scope = _activate(snapshot, None)
        cls.snapshot = snapshot
        import hiercp.model as original
        cls.original = original
        cls.net = original.HierarchicalPyGPlacementModel(hidden_dim=128, heads=4,
            local_layers=3, patient_layers=2, prototype_layers=2, dropout=.1,
            dense_base_channels=12, dense_feature_dim=32, dense_batch_size=4,
            channels_last_3d=True, checkpoint_local_blocks=True,
            checkpoint_dense_encoder=True, ablation_mode='full')
        cls.relations = tuple(original.LOCAL_EDGE_TYPES)
        cls.convs = tuple(conv for block in cls.net.local_encoder.blocks for conv in block.conv.convs.values())

    @classmethod
    def tearDownClass(cls):
        cls.torch.set_num_threads(cls.old_threads)

    def tearDown(self):
        # Every test owns/restores its overlay. A residual mutation is a failure,
        # not a fallback to another operator/source.
        self.assertEqual(self.original.EDGE_ATTENTION_WORKSPACE_BYTES, 64 * 1024**2)
        for conv in self.convs:
            self.assertNotIn('_edge_chunk_size', vars(conv))

    def test_instance_scope_state_bytes_and_restoration(self):
        torch = self.torch
        before = {name: value.detach().clone() for name, value in self.net.state_dict().items()}
        modules = tuple((name, id(value), type(value)) for name, value in self.net.named_modules())
        upper = [(name, value, value._edge_chunk_size.__func__) for name, value in self.net.named_modules()
            if type(value) is self.original.CompatibilityGatedGATv2Conv and not name.startswith('local_encoder.')]
        with adapter.install_l0_edge_workspace(self.net, 256 * 1024**2) as handle:
            receipt = handle.receipt()
            self.assertEqual(receipt['relation_convolutions'], 48)
            self.assertTrue(receipt['independent_full_native_calibration_required'])
            self.assertFalse(receipt['adapter_itself_approves_gate'])
            self.assertEqual(receipt['adapter_sha256'], hashlib.sha256(Path(adapter.__file__).read_bytes()).hexdigest())
            for row in receipt['convs']:
                self.assertEqual((row['original_float32_chunk'], row['float32_chunk']), (16384, 65536))
                self.assertEqual((row['original_float64_chunk'], row['float64_chunk']), (8192, 32768))
            for _, conv, method in upper:
                self.assertNotIn('_edge_chunk_size', vars(conv))
                self.assertIs(conv._edge_chunk_size.__func__, method)
                self.assertEqual(conv._edge_chunk_size(torch.float32), 16384)
            self.assertEqual(modules, tuple((name, id(value), type(value)) for name, value in self.net.named_modules()))
            self.assertEqual(set(before), set(self.net.state_dict()))
            for name, value in self.net.state_dict().items():
                self.assertTrue(torch.equal(value, before[name]), name)
        self.assertFalse(handle.restore())
        for conv in self.convs:
            self.assertEqual(conv._edge_chunk_size(torch.float32), 16384)

    def test_1024_mib_and_original_per_dtype_formula(self):
        torch = self.torch
        with adapter.install_l0_edge_workspace(self.net, 1024 * 1024**2) as handle:
            for row in handle.receipt()['convs']:
                self.assertEqual((row['float32_chunk'], row['float64_chunk']), (262144, 131072))
            for conv in self.convs:
                self.assertEqual(conv._edge_chunk_size(torch.float16), 262144)
                self.assertEqual(conv._edge_chunk_size(torch.bfloat16), 262144)

    def test_invalid_workspace_and_source_bindings_fail_before_mutation(self):
        for value in (-1, 0, 63 * 1024**2, True, 256.0 * 1024**2):
            with self.assertRaisesRegex(ValueError, 'integer workspace'):
                adapter.install_l0_edge_workspace(self.net, value)
        with self.assertRaisesRegex(ValueError, 'pinned original source/archive'):
            adapter.install_l0_edge_workspace(self.net, 256 * 1024**2, expected_model_source_sha256='0' * 64)
        with self.assertRaisesRegex(ValueError, 'pinned original source/archive'):
            adapter.install_l0_edge_workspace(self.net, 256 * 1024**2, expected_archive_sha256='0' * 64)
        with patch.object(adapter, '_file_sha', return_value='0' * 64):
            with self.assertRaisesRegex(ValueError, 'source byte SHA256'):
                adapter.install_l0_edge_workspace(self.net, 256 * 1024**2)

    def test_duplicate_install_and_external_method_change_fail_explicitly(self):
        handle = adapter.install_l0_edge_workspace(self.net, 256 * 1024**2)
        try:
            with self.assertRaisesRegex(ValueError, 'existing instance execution override'):
                adapter.install_l0_edge_workspace(self.net, 1024 * 1024**2)
            conv = self.convs[0]
            installed = conv._edge_chunk_size
            conv._edge_chunk_size = lambda dtype: 7
            try:
                with self.assertRaisesRegex(ValueError, 'ownership changed'):
                    handle.guard()
                with self.assertRaisesRegex(ValueError, 'ownership changed'):
                    handle.restore()
                self.assertIn('_edge_chunk_size', vars(self.convs[-1]))
            finally:
                conv._edge_chunk_size = installed
        finally:
            handle.restore()

    def test_upper_override_and_global_change_are_not_silently_admitted(self):
        torch = self.torch
        upper = next(value for name, value in self.net.named_modules()
            if type(value) is self.original.CompatibilityGatedGATv2Conv and name.startswith('patient_encoder.'))
        upper._edge_chunk_size = lambda dtype: 7
        try:
            with self.assertRaisesRegex(ValueError, 'Upper-level execution overrides'):
                adapter.install_l0_edge_workspace(self.net, 256 * 1024**2)
        finally:
            del upper._edge_chunk_size
        with patch.object(self.original, 'EDGE_ATTENTION_WORKSPACE_BYTES', 256 * 1024**2):
            with self.assertRaisesRegex(ValueError, 'Original global64MiB'):
                adapter.install_l0_edge_workspace(self.net, 256 * 1024**2)
        with adapter.install_l0_edge_workspace(self.net, 256 * 1024**2) as handle:
            with patch.object(self.original, 'EDGE_ATTENTION_WORKSPACE_BYTES', 256 * 1024**2):
                with self.assertRaisesRegex(ValueError, 'Original global64MiB'):
                    handle.guard()
                with self.assertRaisesRegex(ValueError, 'Original global64MiB'):
                    self.convs[0]._edge_chunk_size(torch.float32)

    def test_all_three_blocks_all48_convs_named_gradients_debug_cpu(self):
        # Every original relation/block's parameter gradients are checked.
        # Larger boundary graphs below check actual streaming with dropout.
        for conv in self.convs:
            self._paired_conv(conv, self.torch.device('cpu'), self.torch.float32, 29)

    def test_original_stream_boundary_dropout_and_exact_next_rng_cpu(self):
        for block in self.net.local_encoder.blocks:
            conv = next(iter(block.conv.convs.values()))
            self._paired_conv(conv, self.torch.device('cpu'), self.torch.float32, 16385)
        self._paired_conv(self.convs[0], self.torch.device('cpu'), self.torch.float64, 8193)

    @unittest.skipUnless(__import__('torch').cuda.is_available(), 'Actual CUDA DEBUG test requires a CUDA device')
    def test_original_stream_boundary_dropout_and_exact_next_rng_cuda(self):
        torch = self.torch
        conv = self.convs[0]
        old_device = next(conv.parameters()).device
        try:
            conv.to('cuda')
            self._paired_conv(conv, torch.device('cuda'), torch.float32, 16385)
        finally:
            conv.to(old_device)

    def _paired_conv(self, conv, device, dtype, edge_count):
        torch = self.torch
        import copy
        baseline = copy.deepcopy(conv).to(device=device, dtype=dtype).train()
        # Real original operator state is copied to the modified original model
        # instance; no test replacement operator is used.
        original_state = {name: value.detach().clone() for name, value in conv.state_dict().items()}
        old_device, old_dtype = next(conv.parameters()).device, next(conv.parameters()).dtype
        conv.to(device=device, dtype=dtype).train()
        conv.load_state_dict(baseline.state_dict())
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if device.type == 'cuda' else None
        try:
            generator = torch.Generator(device=device).manual_seed(1709 + edge_count)
            source = torch.randn(37, 128, generator=generator, device=device, dtype=dtype)
            destination = torch.randn(31, 128, generator=generator, device=device, dtype=dtype)
            edges = torch.stack((torch.arange(edge_count, device=device) % 37,
                (torch.arange(edge_count, device=device) * 7 + 3) % 31))
            attrs = torch.randn(edge_count, conv.edge_dim, generator=generator, device=device, dtype=dtype)
            coefficients = torch.randn(31, 128, generator=generator, device=device, dtype=dtype)

            def run(operator):
                operator.zero_grad(set_to_none=True)
                left = source.detach().clone().requires_grad_(True)
                right = destination.detach().clone().requires_grad_(True)
                attributes = attrs.detach().clone().requires_grad_(True)
                output, (actual_edges, attention) = operator((left, right), edges, attributes,
                    return_attention_weights=True)
                loss = (output * coefficients).mean() + output.square().mean()
                loss.backward()
                grads = {name: value.grad.detach().clone() if value.grad is not None else None
                    for name, value in operator.named_parameters()}
                if any(value is None or not torch.isfinite(value).all() for value in grads.values()):
                    raise AssertionError('Every actual original conv named parameter gradient must be finite/present')
                return dict(output=output.detach(), attention=attention.detach(), edges=actual_edges.detach(),
                    loss=loss.detach(), grads=grads, inputs=(left.grad, right.grad, attributes.grad),
                    next_cpu=torch.rand(17), next_cuda=torch.rand(17, device=device) if device.type == 'cuda' else None)

            first = run(baseline)
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)
            with adapter.install_l0_edge_workspace(self.net, 256 * 1024**2):
                second = run(conv)
            for name in ('output', 'attention', 'loss'):
                torch.testing.assert_close(first[name], second[name], rtol=1e-5, atol=1e-6)
            self.assertTrue(torch.equal(first['edges'], second['edges']))
            # Global dropout mask and shape are unchanged across chunk boundaries.
            self.assertTrue(torch.equal(first['attention'] == 0, second['attention'] == 0))
            self.assertEqual(set(first['grads']), set(second['grads']))
            for name in first['grads']:
                torch.testing.assert_close(first['grads'][name], second['grads'][name], rtol=1e-5, atol=1e-6,
                    msg='Actual named original conv gradient: ' + name)
            for a, b in zip(first['inputs'], second['inputs']):
                torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
            self.assertTrue(torch.equal(first['next_cpu'], second['next_cpu']))
            if device.type == 'cuda':
                self.assertTrue(torch.equal(first['next_cuda'], second['next_cuda']))
        finally:
            conv.to(device=old_device, dtype=old_dtype)
            conv.load_state_dict(original_state)
            conv.zero_grad(set_to_none=True)
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)




"""DEBUG integration fragment: import the actual future runner/adapter paths.

Append this class after the eight OriginalL0ExecutionDebug operator tests in
the existing runtime-upgrade suite. Hook transport tests use scalar inputs only
to test PyTorch hook ownership; they do not claim native model/CT validation.
"""
from pathlib import Path
import copy
import hashlib
import inspect
import json
import unittest
import uuid
from unittest.mock import patch

from tools import run_v23_all_p as runner
from hiercp_v1x import v23_edge_execution as adapter


class RunnerEdgeConfigDebug(unittest.TestCase):
    def runtime(self, mode=256):
        return {'v23_runtime': {'l0_edge_workspace_mib': mode,
            'l0_edge_adapter_sha256': runner.EDGE_ADAPTER_SHA256}}

    def source(self):
        return {**runner.EDGE_FEEDING_SHA256,
            'hiercp_v1x/v23_edge_execution.py': runner.EDGE_ADAPTER_SHA256}

    def test_explicit_measured_integer_settings_and_exact_source(self):
        for mode in (64, 256, 1024):
            self.assertEqual(runner.validate_v23_edge_runtime_config(self.runtime(mode), self.source()), mode)
        for mode in (None, True, 0, 63, 128, 256.0, '256'):
            with self.assertRaises(ValueError):
                runner.validate_v23_edge_runtime_config(self.runtime(mode), self.source())
        value = self.runtime(); value['v23_runtime'].pop('l0_edge_workspace_mib')
        with self.assertRaises(ValueError):
            runner.validate_v23_edge_runtime_config(value, self.source())
        for path in self.source():
            source = self.source(); source[path] = '0' * 64
            with self.assertRaises(ValueError):
                runner.validate_v23_edge_runtime_config(self.runtime(), source)
        value = self.runtime(); value['v23_runtime']['l0_edge_adapter_sha256'] = '0' * 64
        with self.assertRaises(ValueError):
            runner.validate_v23_edge_runtime_config(value, self.source())

    def test_same_reusable_installer_signature_without_certificate_cycle(self):
        self.assertEqual(list(inspect.signature(runner.install_v23_l0_execution).parameters),
            ['model', 'scorer', 'config'])
        source = inspect.getsource(runner.install_v23_l0_execution)
        self.assertNotIn('require_engineering_calibration', source)
        self.assertNotIn('validate_engineering_certificate', source)
        self.assertNotIn('batch_calibration', source)

    def test_legacy_UNIT_calibration_without_keys_and_partial_new_settings(self):
        request = {'request_sha256': 'DEBUG_REQUEST', 'gpus': [5]}
        calibration = dict(world_size=1, request_sha256='DEBUG_REQUEST', physical_GPUs=[5],
            debug=False, measured_full_P_U128_backward=True, original_model_and_RNG_preserved=True,
            initial_state_sha256='DEBUG_INITIAL',
            GPU_reports=[{'physical_GPU': 5, 'initial_state_sha256': 'DEBUG_INITIAL'}])
        runner.validate_single_gpu_calibration(calibration, request)
        for key, value in self.runtime()['v23_runtime'].items():
            partial = {**request, 'config': {'v23_runtime': {key: value}}}
            with self.assertRaisesRegex(ValueError, 'Both explicit edge execution keys'):
                runner.validate_single_gpu_calibration(calibration, partial)
        calibration['world_size'] = 3
        with self.assertRaises(ValueError):
            runner.validate_single_gpu_calibration(calibration, request)

    def fixture(self):
        import torch
        # The preceding eight operator tests already activate the exact sealed
        # original. Construction is real, but this test does not run its network.
        from hiercp.model import HierarchicalPyGPlacementModel
        net = HierarchicalPyGPlacementModel(hidden_dim=128, heads=4,
            local_layers=3, patient_layers=2, prototype_layers=2, dropout=.1,
            dense_base_channels=12, dense_feature_dim=32, dense_batch_size=4,
            channels_last_3d=True, checkpoint_local_blocks=True,
            checkpoint_dense_encoder=True, ablation_mode='full')
        class MetadataScorer(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.net = net; self.providers = {}
            def forward(self, value):
                return value * 2
        # All files are new owned DEBUG fixtures. Actual source bytes are copied,
        # so the same future production source guard is exercised without mocks.
        root = runner.ROOT / 'outputs' / ('v23_edge_runner_DEBUG_' + uuid.uuid4().hex)
        root.mkdir(parents=True)
        for relative in runner.EDGE_FEEDING_SHA256:
            path = root / relative; path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((runner.ROOT / relative).read_bytes())
        path = root / 'hiercp_v1x/v23_edge_execution.py'
        path.write_bytes(Path(adapter.__file__).read_bytes())
        return root, net, MetadataScorer()

    def test_actual_48_instance_hook_returns_none_keeps_state_and_restores(self):
        import torch
        root, net, scorer = self.fixture()
        state = {name: value.detach().clone() for name, value in net.state_dict().items()}
        with patch.object(runner, 'ROOT', root):
            handle, hook = runner.install_v23_l0_execution(net, scorer, self.runtime())
            scorer._l0_edge_handle, scorer._l0_edge_guard_hook = handle, hook
            try:
                self.assertEqual(handle.receipt()['relation_convolutions'], 48)
                self.assertEqual(handle._v23_forward_guard_calls, 0)
                actual = scorer(torch.tensor(3.0))
                self.assertEqual(actual.item(), 6.0)
                self.assertEqual(handle._v23_forward_guard_calls, 1)
                before = next(iter(scorer._forward_pre_hooks.values()))
                self.assertIsNone(before(scorer, (torch.tensor(4.0),)))
                self.assertEqual(handle._v23_forward_guard_calls, 2)
                self.assertEqual(set(state), set(net.state_dict()))
                for name, value in net.state_dict().items():
                    self.assertTrue(torch.equal(value, state[name]), name)
            finally:
                runner.cleanup_v23_l0_execution(scorer)
        self.assertEqual(len(scorer._forward_pre_hooks), 0)
        self.assertFalse(handle.restore())

    def test_running_source_change_fails_before_forward_and_cleanup_still_restores(self):
        import torch
        root, net, scorer = self.fixture()
        with patch.object(runner, 'ROOT', root):
            handle, hook = runner.install_v23_l0_execution(net, scorer, self.runtime())
            scorer._l0_edge_handle, scorer._l0_edge_guard_hook = handle, hook
            try:
                path = root / next(iter(runner.EDGE_FEEDING_SHA256))
                with path.open('ab') as stream:
                    stream.write(b'\n# owned DEBUG source mutation\n')
                with self.assertRaisesRegex(ValueError, 'source file changed'):
                    scorer(torch.tensor(3.0))
                self.assertEqual(handle._v23_forward_guard_calls, 0)
            finally:
                runner.cleanup_v23_l0_execution(scorer)
        self.assertFalse(handle.restore())

    def test_wrong_scorer_and_hook_registration_failure_do_not_leave_overlay(self):
        import torch
        root, net, scorer = self.fixture()
        with patch.object(runner, 'ROOT', root):
            with patch.object(scorer, 'net', torch.nn.Identity()):
                with self.assertRaisesRegex(ValueError, 'same original model'):
                    runner.install_v23_l0_execution(net, scorer, self.runtime())
            with patch.object(scorer, 'register_forward_pre_hook', side_effect=RuntimeError('DEBUG register failure')):
                with self.assertRaisesRegex(RuntimeError, 'DEBUG register failure'):
                    runner.install_v23_l0_execution(net, scorer, self.runtime())
        for block in net.local_encoder.blocks:
            for conv in block.conv.convs.values():
                self.assertNotIn('_edge_chunk_size', vars(conv))

    def test_cleanup_closes_all_providers_and_restores_even_when_one_close_fails(self):
        root, net, scorer = self.fixture()
        closed = []
        class Provider:
            def __init__(self, key, fail=False): self.key, self.fail = key, fail
            def close(self):
                closed.append(self.key)
                if self.fail: raise RuntimeError('DEBUG close failure')
        scorer.providers = {'train': Provider('train', True), 'val': Provider('val')}
        with patch.object(runner, 'ROOT', root):
            handle, hook = runner.install_v23_l0_execution(net, scorer, self.runtime())
            scorer._l0_edge_handle, scorer._l0_edge_guard_hook = handle, hook
            with self.assertRaisesRegex(RuntimeError, 'DEBUG close failure'):
                runner.cleanup_v23_l0_execution(scorer)
        self.assertEqual(closed, ['train', 'val'])
        self.assertFalse(handle.restore())
        self.assertEqual(len(scorer._forward_pre_hooks), 0)


if __name__=='__main__': unittest.main()
