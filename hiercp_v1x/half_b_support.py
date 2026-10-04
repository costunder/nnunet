"""Full signed v1 training support for the independently bound upper half B.

Support uses both original fixed epoch-0 local views. Only the local encoder
runs during a refresh; no upper forward, loss, optimizer, query sampler or
validation record contributes to the bank. The original query curriculum,
two-view schedule and complete 40-epoch loop remain in the archived source.
"""
from __future__ import annotations

import ast
from collections import Counter
import functools
import hashlib
import inspect
import json
from pathlib import Path
import re
import time
from types import FunctionType
import uuid

from .contracts import canonical_hash
from .experiment import digest, read, write_new


SUPPORT_POLICY = 'fixed_epoch0_two_original_views_eval_no_grad_full_train_initial_then_after_each_train_before_validation'
_MANAGER_GLOBAL = '_v1x_half_b_support_manager'


def signed_training_cohort(files, training_cases, validation_cases, index):
    """Resolve real materialized train samples; never fabricate eligible cases."""
    for cases in (training_cases, validation_cases):
        if (not isinstance(cases, (list, tuple)) or not cases
                or any(not isinstance(case, str) or not case for case in cases)
                or len(set(cases)) != len(cases)):
            raise ValueError('Unique explicit configured train/validation case IDs required')
    if set(training_cases) & set(validation_cases):
        raise ValueError('Training and validation cases overlap')
    if (not isinstance(files, list) or not files
            or any(not isinstance(name, str) for name in files)
            or len(set(files)) != len(files)):
        raise ValueError('Complete unique signed training cache filenames required')
    entries = index.get('entries') if isinstance(index, dict) else None
    if not isinstance(entries, list) or any(not isinstance(row, dict) for row in entries):
        raise ValueError('Actual cache index rows required')
    rows = [row for row in entries if row.get('split') == 'train']
    if (any(not isinstance(row.get('path'), str) for row in rows)
            or sorted(row['path'] for row in rows) != sorted(files)):
        raise ValueError('Signed training files differ from the full materialized index')
    expected = {}
    counts = Counter()
    for row in rows:
        name, case, sample = row['path'], row.get('case_id'), row.get('sample_index')
        match = re.fullmatch(r'([A-Za-z0-9_-]+)__(\d{3})\.pt', name)
        if (match is None or case not in training_cases or case in validation_cases
                or type(sample) is not int or sample < 0
                or name != f'{case}__{sample:03d}.pt'):
            raise ValueError('Training support row has a different case/sample/split binding')
        expected[name] = case
        counts[case] += 1
    cases = [case for case in training_cases if counts[case]]
    if len(cases) < 3:
        raise ValueError('At least three actual training patients required for query exclusion')
    return dict(expected_samples={name: expected[name] for name in files}, case_ids=cases,
        samples=len(files), case_sample_counts={case: counts[case] for case in cases},
        configured_cases=len(training_cases), materialized_cases=len(cases),
        configured_but_not_materialized_cases=[case for case in training_cases if not counts[case]])


def encode_local_support(model, batch, *, device, use_amp):
    """Original L0, both views and original mean; no upper forward or loss.

    This helper also serves the explicitly separate real-data DEBUG tool. The
    caller owns the batch, budget and global RNG lifetime. Every original module
    training flag is restored, including intentionally mixed submodule modes.
    """
    import torch
    device = torch.device(device)
    counts = getattr(batch, 'counts', None)
    if (not isinstance(counts, tuple) or not counts
            or any(type(count) is not int or count != 8 for count in counts)
            or getattr(batch, 'local_batch_view2', None) is None):
        raise ValueError('Complete original eight-candidate/two-view support batch required')
    modes = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        with torch.no_grad(), torch.autocast(device_type=device.type, enabled=bool(use_amp)):
            source = batch.source_patches.to(device, non_blocking=device.type == 'cuda')
            target = batch.target_patches.to(device, non_blocking=device.type == 'cuda')
            owners = torch.repeat_interleave(
                torch.arange(len(counts), device=device, dtype=torch.long),
                torch.tensor(counts, device=device, dtype=torch.long))
            source_map, target_map = model.local_encoder.encode_dense_maps(source, owners, target)
            first = model.local_encoder.forward_graph(
                batch.local_batch.to(device), source_map, target_map)
            second = model.local_encoder.forward_graph(
                batch.local_batch_view2.to(device), source_map, target_map)
            value = model._mean_embeddings(first, second)['fused'].detach().float().contiguous()
        if value.shape != (sum(counts), 128):
            raise ValueError('Original full-width fused128 support output required')
        torch._assert_async(torch.isfinite(value).all(), 'Nonfinite original support embedding')
        return value
    finally:
        for module, training in modes:
            module.training = training


class SupportManager:
    """Own one private sequential loader; query shuffle/worker RNG is untouched."""

    def __init__(self, pipeline, receipt, scope, output, budget):
        self.pipeline, self.receipt, self.scope = pipeline, receipt, scope
        self.output, self.budget = Path(output).resolve(), budget
        self.baseline = Path(receipt['baseline_experiment']).resolve(strict=True)
        self.cache = self.baseline / 'shared/cache'
        self.config = receipt['config']
        self.manifest_sha256 = receipt['contract_sha256']
        if (not isinstance(self.manifest_sha256, str)
                or re.fullmatch(r'[0-9a-f]{64}', self.manifest_sha256) is None):
            raise ValueError('Half-B manifest digest is missing')
        proof = receipt['baseline_proof']
        if scope['contract_sha256'] != proof['neural_baseline']['scope_digest']:
            raise ValueError('Support scope differs from the completed baseline')
        manifest_path, index_path = self.baseline/'manifest.json', self.cache/'index.json'
        for path in (manifest_path, index_path):
            if (path.is_symlink() or not path.is_file()
                    or proof['files'].get(str(path)) != digest(path)):
                raise ValueError('Actual bound support manifest/index changed')
        native, index = read(manifest_path), read(index_path)
        self.training_cases = tuple(native['split']['train'])
        self.validation_cases = tuple(native['split']['val'])
        files = proof['neural_baseline']['training_signature']['train_cache_files']
        self.cohort = signed_training_cohort(files, self.training_cases, self.validation_cases, index)
        self.files = tuple(self.cache/name for name in files)
        self.batch = proof['execution']['physical_batch']
        self.workers = proof['execution']['workers']
        if (type(self.batch) is not int or self.batch < 1
                or type(self.workers) is not int or self.workers < 0
                or self.config['training']['batch_size'] != self.batch
                or self.config['training']['num_workers'] != self.workers
                or self.config['seed'] != 42):
            raise ValueError('Support must use the original measured batch/workers and seed42')
        for path in (manifest_path, index_path):
            if proof['files'].get(str(path)) != digest(path):
                raise ValueError('Support publication changed while it was read')
        self._loader = None
        self._worker_generator = None
        self._bound_model = None
        self._generation = None
        self._closed = False

    def _ensure_loader(self):
        if self._loader is not None:
            return self._loader
        import torch
        from torch.utils.data import DataLoader
        from hiercp.data import HierarchicalCacheDataset, collate_samples
        from .scope_training_entry import ScopeWorkerInitializer
        training = self.config['training']
        options = self.pipeline._loader_kwargs(workers=self.workers,
            pin_memory=bool(training.get('pin_memory', True)),
            prefetch_factor=int(training.get('prefetch_factor', 2)),
            persistent_workers=bool(training.get('persistent_workers', True)))
        initializer = options.get('worker_init_fn')
        if self.workers:
            if initializer is None:
                options['worker_init_fn'] = ScopeWorkerInitializer(
                    self.receipt['source'], 10, self.scope['contract_sha256'], False)
            elif (not isinstance(initializer, ScopeWorkerInitializer)
                    or Path(initializer.source).resolve() != Path(self.receipt['source']).resolve()
                    or initializer.margin_mm != 10
                    or initializer.contract_sha256 != self.scope['contract_sha256']):
                raise ValueError('Private support worker scope initialization is different')
        dataset = HierarchicalCacheDataset(self.files, mmap=True, training=False, seed=42)
        dataset.set_epoch(0)
        self._worker_generator = torch.Generator()
        self._worker_generator.manual_seed(42 + 5003)
        self._loader = DataLoader(dataset, batch_size=self.batch, shuffle=False,
            generator=self._worker_generator, collate_fn=collate_samples, **options)
        return self._loader

    def before_pass(self, model, training_mode, session, *, train_files,
                    batch_size, workers, seed, use_amp, device):
        """Refresh before eval; reconstruct missing bank before a resumed train."""
        import torch
        if self._closed:
            raise ValueError('Support manager is closed')
        if (tuple(Path(path).resolve() for path in train_files) != self.files
                or batch_size != self.batch or workers != self.workers or seed != 42
                or bool(use_amp) != bool(torch.device(device).type == 'cuda'
                                        and self.config['training']['amp'])):
            raise ValueError('Runtime support files/batch/workers/seed/precision changed')
        if type(training_mode) is not bool:
            raise ValueError('Original train/eval pass flag must be boolean')
        active_epoch = session._epoch
        if active_epoch is None:
            active_epoch = 0
        if type(active_epoch) is not int or not 0 <= active_epoch <= 40:
            raise ValueError('Support refresh must belong to the original epoch0..40')
        if training_mode and self._bound_model is model and self._generation is not None:
            return
        epoch = max(0, active_epoch - 1) if training_mode else active_epoch
        self.refresh(model, epoch=epoch, device=device, use_amp=use_amp,
            trigger='missing_bank_before_resumed_train' if training_mode else 'before_validation')

    def refresh(self, model, *, epoch, device, use_amp, trigger):
        import torch
        from hiercp.tensor import capture_rng_state, restore_rng_state, collect_runtime_resources
        from .scope_probe_support import state_digest
        if self._closed or type(epoch) is not int or not 0 <= epoch <= 40:
            raise ValueError('Invalid or closed support refresh')
        if not hasattr(model, 'half_b') or not callable(getattr(model.half_b, 'bind_support', None)):
            raise ValueError('The actual Half-B upper bank binding is required')
        device = torch.device(device)
        rng = capture_rng_state()
        modes = [(module, module.training) for module in model.modules()]
        started = time.perf_counter()
        generation = f'epoch{epoch:02d}_{uuid.uuid4().hex}'
        parts, observed_files, observed_cases = [], [], []
        owner_lookup = {case: index for index, case in enumerate(self.cohort['case_ids'])}
        loaded = 0
        try:
            self.budget()
            model.eval()
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(device)
            loader = self._ensure_loader()
            # Worker startup receives only this private generator. Resetting it
            # also makes rebuilding an interrupted epoch independent of how many
            # earlier refreshes happened, without touching query generators.
            self._worker_generator.manual_seed(42 + 5003)
            last_progress = started
            for batch in loader:
                self.budget()
                count = len(batch.counts)
                names = [path.name for path in self.files[loaded:loaded+count]]
                cases = [self.cohort['expected_samples'][name] for name in names]
                if (count < 1 or len(names) != count or tuple(batch.case_ids) != tuple(cases)
                        or batch.counts != (8,)*count
                        or batch.difficulties.dtype != torch.long
                        or batch.difficulties.shape != (count*8,)):
                    raise ValueError('Private support loader changed signed sample coverage/order')
                # Tiny metadata is still CPU-owned here. These labels are only
                # support evidence; they are never passed to a query forward.
                difficulty = batch.difficulties.reshape(count, 8)
                if (difficulty.device.type != 'cpu' or not bool((difficulty[:, 0] == 0).all())
                        or not bool(((difficulty[:, 1:] >= 1) & (difficulty[:, 1:] <= 3)).all())):
                    raise ValueError('Original source-anchor/curriculum support labels changed')
                parts.append(encode_local_support(model, batch, device=device, use_amp=use_amp))
                observed_files.extend(names)
                observed_cases.extend(cases)
                loaded += count
                self.budget()
                now = time.perf_counter()
                if now - last_progress >= 30:
                    print(json.dumps(dict(stage='half_B_support_refresh', generation=generation,
                        epoch=epoch, fixed_view_epoch=0, loaded_samples=loaded,
                        expected_samples=len(self.files), physical_batch=self.batch,
                        workers=self.workers, training_queries_consumed=0), allow_nan=False), flush=True)
                    last_progress = now
                del batch
            if loaded != len(self.files) or observed_files != [path.name for path in self.files]:
                raise ValueError('Incomplete full signed support memory; no subset fallback')
            embeddings = torch.cat(parts, dim=0).detach().float().contiguous()
            owners = torch.tensor([owner_lookup[case] for case in observed_cases],
                dtype=torch.long, device=device).repeat_interleave(8)
            classes = torch.tensor([1,0,0,0,0,0,0,0],
                dtype=torch.long, device=device).repeat(loaded)
            memory = dict(embeddings=embeddings, owners=owners, classes=classes,
                patient_case_ids=tuple(self.cohort['case_ids']),
                sample_ids=tuple(name for name in observed_files for _ in range(8)),
                candidate_indices=tuple(range(8))*loaded,
                expected_samples=dict(self.cohort['expected_samples']),
                training_case_ids=self.training_cases, validation_case_ids=self.validation_cases,
                manifest_sha256=self.manifest_sha256, generation=generation, epoch=epoch,
                support_policy=SUPPORT_POLICY, fixed_view_epoch=0,
                full_signed_training_cache=True, debug=False)
            model.half_b.bind_support(memory)
            self._bound_model, self._generation = model, generation
            # One complete detached bank transfer for reproducible evidence,
            # outside optimization. No per-query CPU score/feature transfer.
            tensors_sha256 = state_digest(dict(embeddings=embeddings, owners=owners, classes=classes))
            content_sha256 = canonical_hash(dict(tensors_sha256=tensors_sha256,
                sample_ids=memory['sample_ids'], candidate_indices=memory['candidate_indices'],
                patient_case_ids=memory['patient_case_ids'], expected_samples=memory['expected_samples'],
                manifest_sha256=self.manifest_sha256, fixed_view_epoch=0, support_policy=SUPPORT_POLICY))
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            record = dict(format='hiercp_half_B_full_training_support_refresh_v1',
                stage='half_B_support_refreshed', generation=generation, epoch=epoch,
                trigger=trigger, fixed_view_epoch=0, support_policy=SUPPORT_POLICY,
                half_B_contract_sha256=self.manifest_sha256,
                samples=loaded, candidate_embeddings=len(embeddings),
                actual_training_patients=len(self.cohort['case_ids']),
                configured_training_patients=len(self.training_cases),
                configured_but_not_materialized_cases=self.cohort['configured_but_not_materialized_cases'],
                case_sample_counts=self.cohort['case_sample_counts'],
                physical_batch=self.batch, workers=self.workers,
                full_signed_training_cache=True, validation_records_used=0,
                training_queries_consumed=0, optimizer_steps=0, embeddings_detached=True,
                original_two_views=True, model_mode_and_global_rng_restored=True,
                tensor_sha256=tensors_sha256, content_sha256=content_sha256,
                wall_seconds=time.perf_counter()-started,
                resources=collect_runtime_resources(device, storage_path=self.cache),
                cuda_peak_allocated_bytes=(int(torch.cuda.max_memory_allocated(device))
                    if device.type == 'cuda' else None))
            self.budget()
        finally:
            for module, training in modes:
                module.training = training
            restore_rng_state(rng)
        # Publish a complete receipt only after RNG/mode restoration succeeds.
        record['wall_seconds'] = time.perf_counter()-started
        write_new(self.output/f'support_refresh_{generation}.json', record)
        print(json.dumps(record, allow_nan=False), flush=True)

    def close(self):
        """Release this manager's private iterator/bank references only."""
        self._closed = True
        if self._bound_model is not None:
            self._bound_model.half_b.clear_support()
        self._bound_model, self._generation = None, None
        self._loader, self._worker_generator = None, None


def observation_source(source):
    """Checked in-memory telemetry/initial-validation/support overlay."""
    from . import epoch_telemetry
    rewritten, identity = epoch_telemetry.overlay_source(source)
    if rewritten.count('worker_init_fn=_v1x_epoch_telemetry.worker_init,') != 2:
        raise ValueError('Pinned telemetry worker insertion sites changed')
    rewritten = rewritten.replace('        worker_init_fn=_v1x_epoch_telemetry.worker_init,\n', '')
    marker = '    for epoch in range(start_epoch, epochs + 1):\n'
    if rewritten.count(marker) != 1:
        raise ValueError('Pinned complete original epoch loop changed')
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
    rewritten = rewritten.replace(marker, initial+marker)
    anchor = '''    def epoch_pass(loader: DataLoader, epoch: int, training_mode: bool) -> dict[str, Any]:
        model.train(training_mode)
'''
    hook = '''    def epoch_pass(loader: DataLoader, epoch: int, training_mode: bool) -> dict[str, Any]:
        _v1x_half_b_support_manager.before_pass(model, training_mode, _v1x_epoch_telemetry,
            train_files=train_files, batch_size=batch_size, workers=workers,
            seed=seed, use_amp=use_amp, device=device)
        model.train(training_mode)
'''
    if rewritten.count(anchor) != 1:
        raise ValueError('Pinned original epoch-pass mode boundary changed')
    rewritten = rewritten.replace(anchor, hook)
    identity.update(instrumented_run_train_sha256=hashlib.sha256(rewritten.encode('utf8')).hexdigest(),
        initial_full_validation_inserted=True, scope_worker_hook_via_loader_kwargs=True,
        half_B_support_hook=True, half_B_support_policy=SUPPORT_POLICY)
    ast.parse('from __future__ import annotations\n'+rewritten)
    return rewritten, identity


def install_training_observation(pipeline, output, request, scope_receipt, manager):
    """Install only on the exact original archive; return its telemetry session."""
    from . import epoch_telemetry
    if (request != manager.receipt or Path(output).resolve() != manager.output
            or scope_receipt.get('contract_sha256') != manager.scope['contract_sha256']
            or scope_receipt.get('half_B_contract') != request.get('contract_sha256')):
        raise ValueError('Support observation request/scope/output differs from its manager')
    if (hasattr(pipeline, '_v1x_epoch_telemetry') or hasattr(pipeline, _MANAGER_GLOBAL)
            or digest(Path(pipeline.__file__)) != epoch_telemetry.PIPELINE_SHA256):
        raise ValueError('Original pipeline differs or already has a training observation')
    source = inspect.getsource(pipeline.run_train)
    rewritten, identity = observation_source(source)
    namespace = dict(pipeline.__dict__)
    tree = ast.parse('from __future__ import annotations\n'+rewritten)
    exec(compile(tree, str(pipeline.__file__)+'::<half-B-support-observation>', 'exec'), namespace)
    original = pipeline.run_train
    installed = FunctionType(namespace['run_train'].__code__, pipeline.__dict__, original.__name__,
        original.__defaults__, original.__closure__)
    functools.update_wrapper(installed, original)
    installed.__kwdefaults__ = original.__kwdefaults__
    session = epoch_telemetry.EpochTelemetry(output, dict(request=request,
        scope_contract=scope_receipt, full_training=True, quality_verified=False), identity)
    pipeline.__dict__['_v1x_epoch_telemetry'] = session
    pipeline.__dict__[_MANAGER_GLOBAL] = manager

    def initial_validation(metrics):
        path = Path(output)/'initial_validation.json'
        payload = dict(format='hiercp_v1_bounded_initial_validation_v1', epoch=0,
            metrics=metrics, scope_contract_sha256=scope_receipt['contract_sha256'],
            half_B_contract_sha256=request['contract_sha256'], support_policy=SUPPORT_POLICY,
            optimizer_steps=0, checkpoint_created=False, full_validation=True)
        if path.exists():
            existing = read(path)
            if (existing.get('scope_contract_sha256') != scope_receipt['contract_sha256']
                    or existing.get('half_B_contract_sha256') != request['contract_sha256']
                    or existing.get('support_policy') != SUPPORT_POLICY):
                raise ValueError('Existing initial validation belongs to another half/scope/support policy')
            print('Initial validation already preserved; original metrics retained', flush=True)
        else:
            write_new(path, payload)
        print(f"Initial validation | MRR={metrics['mrr']:.6f} top1={metrics['acc']:.6f} margin={metrics['margin']:.6f}", flush=True)

    pipeline.__dict__['_v1x_scope_initial_validation'] = initial_validation
    pipeline.run_train = installed
    return session
