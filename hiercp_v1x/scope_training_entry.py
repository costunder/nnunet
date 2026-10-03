"""Full archived v1 learning under one explicit physical ROI contract.

The exact archived files stay unchanged. The runtime installs bounded geometry,
worker hooks, checkpoint identity and observational epoch logging; native
curriculum, data coverage, model, optimizer and epoch schedule remain owned by
the archived pipeline. This module runs only the explicitly requested phase.
"""
from __future__ import annotations

import argparse
import ast
from collections.abc import Mapping
import copy
from dataclasses import dataclass
import csv
import functools
import hashlib
import importlib
import inspect
import json
import math
from pathlib import Path, PurePosixPath
import random
import sys
import threading
from types import FunctionType

ROOT = Path(__file__).resolve().parents[1]
FORMAT = 'hiercp_v1_bounded_scope_training_request_v1'
MARKER = 'v1x_bounded_scope_digest'
_CHECKPOINT_ACTIVE = None


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def validate_request(value):
    required = {'source', 'config', 'experiment', 'phase', 'source_experiment',
        'native_cache', 'prototype_bank', 'regions', 'cache', 'split',
        'medical_root', 'margin_mm', 'workers', 'cuda_gib', 'rss_gib', 'gpu'}
    if not isinstance(value, dict) or not required <= set(value):
        raise ValueError('Full scope worker request is missing explicit fields')
    if value.get('format', FORMAT) != FORMAT or value['phase'] not in ('prepare', 'train'):
        raise ValueError('Only explicitly bound full prepare/train phases are supported')
    if value['margin_mm'] not in (10, 20, 30) or isinstance(value['margin_mm'], bool):
        raise ValueError('Explicit physical margin must be 10, 20 or 30 mm')
    if type(value['workers']) is not int or value['workers'] < 1:
        raise ValueError('Parallel case preparation owns worker selection; replay workers must be positive')
    if type(value['gpu']) is not int or value['gpu'] < 0:
        raise ValueError('An explicit physical GPU number is required')
    for key in ('cuda_gib', 'rss_gib'):
        number = value[key]
        if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number) or number <= 0:
            raise ValueError('Explicit finite positive GPU/RSS budgets are required')
    for key in required - {'phase', 'margin_mm', 'workers', 'cuda_gib', 'rss_gib', 'gpu', 'native_cache'}:
        if not isinstance(value[key], str) or not Path(value[key]).is_absolute():
            raise ValueError(f'{key} must be an absolute bound path')
    if value['native_cache'] is not None and (not isinstance(value['native_cache'], str)
            or not Path(value['native_cache']).is_absolute()):
        raise ValueError('Native replay cache must be an explicit absolute path or null')
    experiment = Path(value['experiment']).resolve()
    source_experiment = Path(value['source_experiment']).resolve()
    if (experiment == source_experiment or experiment.is_relative_to(source_experiment)
            or source_experiment.is_relative_to(experiment)):
        raise ValueError('Bounded experiment must be disjoint from the preserved native experiment')
    for key in ('source', 'config', 'prototype_bank', 'regions', 'cache', 'split'):
        if not Path(value[key]).resolve().is_relative_to(experiment):
            raise ValueError(f'{key} escapes the bounded experiment')
    if value['native_cache'] is not None:
        native = Path(value['native_cache']).resolve()
        if native != source_experiment / 'shared/cache' or native == Path(value['cache']).resolve():
            raise ValueError('Native cache identity is not the preserved preparation owner')
    return copy.deepcopy(value)


class ResourceBudget:
    def __init__(self, cuda_gib, rss_gib):
        self.cuda_bytes = int(cuda_gib * 2**30)
        self.rss_bytes = int(rss_gib * 2**30)

    def __call__(self):
        import psutil
        import torch
        if psutil.Process().memory_info().rss > self.rss_bytes:
            raise MemoryError('Explicit bounded-scope process RSS budget exceeded; no smaller fallback')
        if torch.cuda.is_initialized() and torch.cuda.memory_allocated() > self.cuda_bytes:
            raise MemoryError('Explicit bounded-scope CUDA budget exceeded; no smaller fallback')


def install_model_budget(budget):
    from hiercp import model
    original = model.HierarchicalPyGPlacementModel.forward

    @functools.wraps(original)
    def forward(self, *args, **kwargs):
        budget()
        result = original(self, *args, **kwargs)
        budget()
        return result

    model.HierarchicalPyGPlacementModel.forward = forward


def install_checkpoint_binding(receipt):
    """Require the scope digest in metadata AND the actual model state."""
    global _CHECKPOINT_ACTIVE
    if _CHECKPOINT_ACTIVE is not None:
        if _CHECKPOINT_ACTIVE != receipt['contract_sha256']:
            raise ValueError('Another scope checkpoint binding is already active')
        return
    import torch
    from hiercp import contracts, model
    identity = receipt['contract_sha256']
    original_version = contracts.ARCHITECTURE_VERSION
    expected_architecture = original_version + '|bounded_scope_' + identity
    if model.HierarchicalPyGPlacementModel.architecture_version != expected_architecture:
        raise ValueError('Model architecture is not bound to this physical scope')
    marker = torch.tensor(list(bytes.fromhex(identity)), dtype=torch.uint8)
    constructor = model.HierarchicalPyGPlacementModel.__init__

    @functools.wraps(constructor)
    def initialized(self, *args, **kwargs):
        constructor(self, *args, **kwargs)
        self.register_buffer(MARKER, marker.clone())

    model.HierarchicalPyGPlacementModel.__init__ = initialized
    original_load = model.HierarchicalPyGPlacementModel.load_state_dict

    @functools.wraps(original_load)
    def load(self, state, *args, **kwargs):
        actual = state.get(MARKER) if isinstance(state, Mapping) else None
        if (not isinstance(actual, torch.Tensor) or actual.dtype != torch.uint8
                or actual.shape != marker.shape or not torch.equal(actual.detach().cpu(), marker)):
            raise ValueError('Actual model weights belong to native or another physical scope')
        return original_load(self, state, *args, **kwargs)

    model.HierarchicalPyGPlacementModel.load_state_dict = load
    original = contracts.require_current_checkpoint

    @functools.wraps(original)
    def require(payload):
        if not isinstance(payload, Mapping) or payload.get('architecture_version') != expected_architecture:
            raise ValueError('Native or other-margin checkpoint cannot resume bounded learning')
        state = payload.get('state_dict')
        actual = state.get(MARKER) if isinstance(state, Mapping) else None
        if (not isinstance(actual, torch.Tensor) or actual.dtype != torch.uint8
                or actual.shape != marker.shape or not torch.equal(actual.detach().cpu(), marker)):
            raise ValueError('Checkpoint actual model state lacks the exact physical scope digest')
        checked = dict(payload)
        checked['architecture_version'] = original_version
        original(checked)

    for module in tuple(sys.modules.values()):
        if module is None or not getattr(module, '__name__', '').startswith('hiercp.'):
            continue
        if getattr(module, 'require_current_checkpoint', None) is original:
            module.require_current_checkpoint = require
    contracts.require_current_checkpoint = require
    _CHECKPOINT_ACTIVE = identity


@dataclass(frozen=True)
class ScopeWorkerInitializer:
    source: str
    margin_mm: float
    contract_sha256: str
    telemetry: bool = False

    def __call__(self, worker_id):
        """Install scope in fork/spawn workers without altering their seeded RNG."""
        import numpy as np
        import torch
        from hiercp_v1x import bounded_scope
        python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
        try:
            source = Path(self.source).resolve(strict=True)
            loaded = [module for name, module in sys.modules.items()
                      if name == 'hiercp' or name.startswith('hiercp.')]
            if loaded:
                # PyG generates propagate modules in the system temp directory
                # for verified original MessagePassing classes. Bind those to
                # the actual class method; they are not source-file imports.
                original_model = sys.modules.get('hiercp.model')
                generated = set()
                if original_model is not None:
                    for value in vars(original_model).values():
                        if isinstance(value, type) and value.__module__ == 'hiercp.model':
                            for method in ('propagate', 'edge_updater'):
                                operation = getattr(value, method, None)
                                name = getattr(operation, '__module__', '')
                                if name.startswith('hiercp.model_') and name.endswith('_' + method):
                                    generated.add(name)
                for module in loaded:
                    path = getattr(module, '__file__', None)
                    if path is not None and not Path(path).resolve().is_relative_to(source) and module.__name__ not in generated:
                        raise ValueError(f'DataLoader worker imported another native source snapshot: {module.__name__}: {path}')
            else:
                from hiercp_v1x.scope_probe_support import activate_original
                activate_original(source)
            sys.path.insert(0, str(source))
            receipt = bounded_scope._ACTIVE
            if receipt is None:
                receipt = bounded_scope.install(self.margin_mm, expected_snapshot_root=source)
            if receipt['contract_sha256'] != self.contract_sha256:
                raise ValueError('DataLoader worker scope differs from its parent')
            if self.telemetry:
                from hiercp_v1x.epoch_telemetry import prepare_sampling_capture
                prepare_sampling_capture()
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.set_rng_state(torch_state)


def install_loader_hook(pipeline, initializer):
    original = pipeline._loader_kwargs

    @functools.wraps(original)
    def options(*args, **kwargs):
        result = original(*args, **kwargs)
        if result['num_workers'] > 0:
            if 'worker_init_fn' in result:
                raise ValueError('Another worker initializer is already configured')
            result['worker_init_fn'] = initializer
        return result

    pipeline._loader_kwargs = options


def install_nonlocal_binding(request, config):
    """Reuse copied native region/prototype bytes with their native metadata."""
    from hiercp import cache, region, schema
    native_path = Path(request.get('native_config', str(Path(request['source_experiment'])/'configs/v1.0.json')))
    native_config = json.loads(native_path.read_text(encoding='utf8'))
    for key in ('model', 'training', 'cache', 'labels', 'ct_clip', 'seed'):
        if config[key] != native_config[key]:
            raise ValueError(f'Physical scope learning changed the native {key} contract')
    expected = dict(native_config['graph'])
    expected.update(adaptive_roi_margin_mm=float(request['margin_mm']),
                    context_outer_radius_mm=float(request['margin_mm']))
    if config['graph'] != expected:
        raise ValueError('Only the two declared physical ROI graph fields may change')
    native_graph = schema.graph_config_from_dict(native_config['graph'])
    original = region.load_or_build_patient_regions

    @functools.wraps(original)
    def regions(case, **kwargs):
        if kwargs['config'].to_dict() != config['graph']:
            raise ValueError('Nonlocal region request uses another graph contract')
        if Path(kwargs.get('cache_dir', '')).resolve() != Path(request['regions']).resolve():
            raise ValueError('Nonlocal region cache escapes the copied experiment')
        delegated = dict(kwargs, config=native_graph)
        return original(case, **delegated)

    for module in tuple(sys.modules.values()):
        if module is not None and getattr(module, '__name__', '').startswith('hiercp.'):
            if getattr(module, 'load_or_build_patient_regions', None) is original:
                module.load_or_build_patient_regions = regions
    region.load_or_build_patient_regions = regions
    bank_path = Path(request['prototype_bank'])
    native_bank = Path(request['source_experiment'])/'shared/prototype_bank.pt'
    metadata_path, manifest_path = bank_path.parent/'metadata.json', bank_path.parent/'manifest.csv'
    native_metadata, native_manifest = native_bank.parent/'metadata.json', native_bank.parent/'manifest.csv'
    for copied, native in ((bank_path, native_bank), (metadata_path, native_metadata), (manifest_path, native_manifest)):
        if copied.is_symlink() or sha(copied) != sha(native):
            raise ValueError('Copied population bank publication differs from native bytes')
    metadata = json.loads(metadata_path.read_text(encoding='utf8'))
    if metadata.get('state') != 'ready':
        raise ValueError('Native prototype publication is incomplete; it cannot be regenerated in a scope experiment')
    split = json.loads(Path(request['split']).read_text(encoding='utf8'))
    if len(split['train']) != 84 or len(split['val']) != 21:
        raise ValueError('Full physical-scope learning requires the unchanged 84/21 patients')
    if (metadata.get('prototype_sha256') != sha(bank_path)
            or metadata.get('manifest_sha256') != sha(manifest_path)
            or set(metadata.get('training_cases', ())) != set(split['train'])
            or not region.graph_config_budget_compatible(metadata.get('graph_config'), native_graph.to_dict())):
        raise ValueError('Native prototype publication SHA/cohort/graph contract changed')
    sources = metadata.get('source_cases')
    if not isinstance(sources, list) or {row.get('case_id') for row in sources} != set(split['train']):
        raise ValueError('Native prototype source provenance is incomplete')
    cache._validate_prototype_manifest(manifest_path, sources)
    from hiercp.prototype import PrototypeBank
    bank = PrototypeBank.load(bank_path)
    if (set(bank.training_case_ids) != set(split['train'])
            or bank.fingerprint() != metadata.get('prototype_fingerprint')):
        raise ValueError('Native population contents disagree with the ready publication')
    return bank


class NativeReplay:
    """Read verified old successes; never reinterpret failed records as success."""
    def __init__(self, request, config, adapter, budget):
        self.request, self.config, self.adapter, self.budget = request, config, adapter, budget
        self.rows, self.lock = {}, threading.Lock()
        self.path = Path(request['experiment']) / 'scope_replay.jsonl'
        self.stream = self.path.open('a', encoding='utf8', newline='\n')
        self.raw_verified = {}
        native = request['native_cache']
        self.cache = Path(native) if native is not None else None
        if self.cache is not None:
            metadata = json.loads((self.cache / 'config.json').read_text(encoding='utf8'))
            self.fingerprint = metadata['config_fingerprint']
            with (self.cache / 'manifest.csv').open(encoding='utf8', newline='') as stream:
                for row in csv.DictReader(stream):
                    if row.get('status') != 'ok' or row.get('sample_index') in ('', None):
                        continue
                    key = (row['case_id'], int(row['sample_index']))
                    if key in self.rows:
                        raise ValueError('Duplicate native sample manifest key')
                    self.rows[key] = row

    def close(self):
        self.stream.close()

    def record(self, value):
        with self.lock:
            self.stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + '\n')
            self.stream.flush()

    def install(self):
        from hiercp import cache
        original = cache.build_training_sample

        @functools.wraps(original)
        def build(case, bank, regions, **kwargs):
            self.budget()
            row = self.rows.get((case.paths.case_id, kwargs['sample_index']))
            if row is None:
                # This is the original builder and its original failure policy,
                # never a successful-native fallback or a skipped sample.
                sample = original(case, bank, regions, **kwargs)
                self.record(dict(case_id=case.paths.case_id, sample_index=kwargs['sample_index'],
                    paired_native=False, reason='verified native sample unavailable; original curriculum preparation'))
                return sample
            sample = self.load(case, bank, kwargs, row)
            from hiercp_v1x.scope_learning_inputs import rebuild_scope
            raw = dict(case_id=case.paths.case_id, image=str(case.paths.image_path),
                label=str(case.paths.label_path), image_sha256=row['source_image_sha256'],
                label_sha256=row['source_label_sha256'])
            changed, diagnostics = rebuild_scope([sample], {'source_records': [raw]}, self.config,
                self.adapter, self.request['margin_mm'], self.request['workers'], self.budget,
                raw_cases={case.paths.case_id: (case, regions.full_organ_mask, regions.organ_depth)},
                profile_payload=False)
            self.record(dict(case_id=case.paths.case_id, sample_index=kwargs['sample_index'],
                paired_native=True, native_sample_sha256=row['artifact_sha256'],
                supervision_sha256=diagnostics['supervision_sha256'],
                bounded_preparation_seconds=diagnostics['bounded_canonical_preparation_seconds']))
            return changed[0]

        cache.build_training_sample = build

    def load(self, case, bank, kwargs, row):
        import torch
        from hiercp import cache
        relative = PurePosixPath(row['path'])
        path = self.cache / row['path']
        if (relative.is_absolute() or len(relative.parts) != 1 or '\\' in row['path']
                or ':' in row['path'] or path.is_symlink() or path.resolve().parent != self.cache.resolve()):
            raise ValueError('Native sample artifact path escaped its preparation owner')
        if (row['candidates'] != '8' or row['split'] != kwargs['split_name']
                or row['config_fingerprint'] != self.fingerprint
                or path.stat().st_size != int(row['file_size']) or sha(path) != row['artifact_sha256']):
            raise ValueError('Native sample bytes/size/full curriculum/config/split changed')
        with self.lock:
            signatures = tuple(row['source_' + key + '_sha256'] for key in ('image', 'label'))
            if case.paths.case_id in self.raw_verified and self.raw_verified[case.paths.case_id] != signatures:
                raise ValueError('Native rows disagree on the raw CT/label provenance')
            if case.paths.case_id not in self.raw_verified:
                for key in ('image', 'label'):
                    if sha(getattr(case.paths, key + '_path')) != row['source_' + key + '_sha256']:
                        raise ValueError('Raw CT/label differs from the original native sample')
                self.raw_verified[case.paths.case_id] = signatures
        sample = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
        graph = self.adapter.configure(sample['graph_config'], self.request['margin_mm']).to_dict()
        if (sample.get('format') != cache.CACHE_FORMAT
                or sample.get('config_fingerprint') != self.fingerprint
                or tuple(sample['ct_clip']) != tuple(self.config['ct_clip'])
                or graph != kwargs['graph_config'].to_dict() or sample['case_id'] != case.paths.case_id
                or sample['sample_index'] != kwargs['sample_index'] or sample['split'] != kwargs['split_name']
                or len(sample['target_locals']) != 8 or sample['prototype_fingerprint'] != bank.fingerprint()):
            raise ValueError('Native replay sample graph/case/index/split/prototype identity changed')
        self.budget()
        return sample


def _install_training_observation(pipeline, output, request, receipt):
    """Checked telemetry plus a pre-training full validation; no optimizer step."""
    from hiercp_v1x import epoch_telemetry
    source = inspect.getsource(pipeline.run_train)
    rewritten, identity = epoch_telemetry.overlay_source(source)
    # The loader kwargs supplies our picklable scope worker for calibration AND
    # training. Remove only the telemetry overlay's duplicate worker keyword.
    count = rewritten.count('worker_init_fn=_v1x_epoch_telemetry.worker_init,')
    if count != 2:
        raise ValueError('Pinned telemetry worker insertion sites changed')
    rewritten = rewritten.replace('        worker_init_fn=_v1x_epoch_telemetry.worker_init,\n', '')
    marker = '    for epoch in range(start_epoch, epochs + 1):\n'
    if rewritten.count(marker) != 1:
        raise ValueError('Pinned native full epoch loop changed')
    initial = '''    if start_epoch == 1:
        _v1x_initial_rng = capture_rng_state()
        _v1x_initial_worker_rng = val_worker_generator.get_state()
        try:
            with torch.no_grad():
                _v1x_initial_metrics = epoch_pass(val_epoch_loader, fixed_validation_epoch, False)
            _v1x_scope_initial_validation(_v1x_initial_metrics)
        finally:
            restore_rng_state(_v1x_initial_rng)
            val_worker_generator.set_state(_v1x_initial_worker_rng)
'''
    rewritten = rewritten.replace(marker, initial + marker)
    identity['instrumented_run_train_sha256'] = hashlib.sha256(rewritten.encode('utf8')).hexdigest()
    identity['initial_full_validation_inserted'] = True
    identity['scope_worker_hook_via_loader_kwargs'] = True
    tree = ast.parse('from __future__ import annotations\n' + rewritten)
    namespace = dict(pipeline.__dict__)
    exec(compile(tree, str(pipeline.__file__) + '::<scope-learning-observation>', 'exec'), namespace)
    original = pipeline.run_train
    installed = FunctionType(namespace['run_train'].__code__, pipeline.__dict__, original.__name__,
        original.__defaults__, original.__closure__)
    functools.update_wrapper(installed, original)
    installed.__kwdefaults__ = original.__kwdefaults__
    session = epoch_telemetry.EpochTelemetry(output, dict(request=request,
        scope_contract=receipt, full_training=True, quality_verified=False), identity)
    pipeline.__dict__['_v1x_epoch_telemetry'] = session

    def initial_validation(metrics):
        path = output / 'initial_validation.json'
        payload = dict(format='hiercp_v1_bounded_initial_validation_v1', epoch=0,
            metrics=metrics, scope_contract_sha256=receipt['contract_sha256'],
            optimizer_steps=0, checkpoint_created=False, full_validation=True)
        if path.exists():
            if json.loads(path.read_text(encoding='utf8'))['scope_contract_sha256'] != receipt['contract_sha256']:
                raise ValueError('Existing initial validation belongs to another scope')
            print('Initial validation already preserved; original metrics retained', flush=True)
        else:
            with path.open('x', encoding='utf8') as stream:
                json.dump(payload, stream, indent=2, allow_nan=False)
        print(f"Initial validation | MRR={metrics['mrr']:.6f} top1={metrics['acc']:.6f} margin={metrics['margin']:.6f}", flush=True)

    pipeline.__dict__['_v1x_scope_initial_validation'] = initial_validation
    pipeline.run_train = installed
    return session


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', required=True)
    args = parser.parse_args(argv)
    request = validate_request(json.loads(Path(args.request).read_text(encoding='utf8')))
    from tools.run_v1_bounded_training import verify_bound_experiment, request_for
    manifest = verify_bound_experiment(request['experiment'])
    expected = request_for(request['experiment'], manifest, request['phase'])
    if {key: value for key, value in request.items() if key != 'format'} != expected:
        raise ValueError('Worker request differs from the actual frozen full experiment manifest')
    # Current adapters must resolve before putting the exact archive first.
    from hiercp_v1x import bounded_scope
    from hiercp_v1x.scope_probe_support import activate_original
    from hiercp_v1x import scope_learning_inputs
    from hiercp_v1x.sampling_entry import preserve_worker_calibration_rng
    from tools.local_cnn_device import select
    select(request['gpu'])
    activate_original(Path(request['source']))
    receipt = bounded_scope.install(request['margin_mm'], expected_snapshot_root=request['source'])
    install_checkpoint_binding(receipt)
    config = json.loads(Path(request['config']).read_text(encoding='utf8'))
    if config['graph'] != bounded_scope.configure(config['graph'], request['margin_mm']).to_dict():
        raise ValueError('Effective graph config differs from the explicit physical scope')
    if config['training']['epochs'] != 40 or config['seed'] != 42 or config['cache']['total_candidates'] != 8 or config['cache']['candidate_pool_size'] != 128:
        raise ValueError('Full native 40-epoch/seed42/eight-candidate/pool128 learning contract changed')
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Actual CUDA is required; no CPU training fallback')
    total = torch.cuda.get_device_properties(0).total_memory
    if int(request['cuda_gib'] * 2**30) > total:
        raise ValueError('Explicit CUDA budget exceeds the selected device')
    torch.cuda.set_per_process_memory_fraction(request['cuda_gib'] * 2**30 / total, 0)
    budget = ResourceBudget(request['cuda_gib'], request['rss_gib'])
    install_model_budget(budget)
    install_nonlocal_binding(request, config)
    pipeline = importlib.import_module('hiercp.pipeline')
    session, replay = None, None
    try:
        if request['phase'] == 'prepare':
            replay = NativeReplay(request, config, bounded_scope, budget)
            replay.install()
            command = ['prepare', '--config', request['config'], '--data-dir', str(Path(request['medical_root'])/'Data'),
                '--split-file', request['split'], '--region-cache-dir', request['regions'],
                '--prototype-bank', request['prototype_bank'], '--cache-dir', request['cache']]
        else:
            preserve_worker_calibration_rng(pipeline)
            output = Path(request['experiment'])/'results/v1.0'
            session = _install_training_observation(pipeline, output, request, receipt)
            initializer = ScopeWorkerInitializer(request['source'], request['margin_mm'], receipt['contract_sha256'], True)
            install_loader_hook(pipeline, initializer)
            command = ['train', '--config', request['config'], '--cache-dir', request['cache'],
                '--prototype-bank', request['prototype_bank'], '--checkpoint', str(output/'checkpoint_best.pt'), '--device', 'cuda']
        previous = sys.argv
        try:
            sys.argv = ['hiercp.pipeline', *command]
            budget()
            result = pipeline.main()
            budget()
            return result
        finally:
            sys.argv = previous
    finally:
        if session is not None:
            session.close()
        if replay is not None:
            replay.close()


if __name__ == '__main__':
    main()
