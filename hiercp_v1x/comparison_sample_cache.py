"""Read completed comparison samples without duplicating their local graphs.

Only small hierarchy skeletons are published here. Every dense patch and local
node/edge table remains in its original, immutable canonical_local publication.
Original epoch-dependent view generation still runs on every call.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import json
from pathlib import Path
import re
import threading
import time
import uuid

import numpy as np
import torch

from .preparation_reuse import _publication_guard, _rename_new, _safe
from .u_bridge_data import FORMAT as SOURCE_FORMAT, _hash, _sha


FORMAT = 'comparison_referenced_sample_layout_v1'
_HEX = re.compile(r'[0-9a-f]{64}')
_REFERENCED = {'source_patch', 'target_patches', 'source_local', 'target_locals'}
_SAMPLED = {'local_graphs', 'local_graphs_view2'}


def _canonical(value):
    return json.loads(json.dumps(value, allow_nan=False))


def _stat(path):
    value = path.stat()
    if not path.is_file():
        raise ValueError(f'Sample cache reference is not a file: {path}')
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def _same(first, second):
    """Cold publication proof, including layout and non-tensor graph metadata."""
    if type(first) is not type(second):
        return False
    if torch.is_tensor(first):
        return (first.device == second.device and first.dtype == second.dtype
                and first.shape == second.shape and first.stride() == second.stride()
                and torch.equal(first, second))
    if isinstance(first, np.ndarray):
        return (first.dtype == second.dtype and first.shape == second.shape
                and first.strides == second.strides and np.array_equal(first, second))
    if isinstance(first, dict):
        return first.keys() == second.keys() and all(_same(v, second[k]) for k, v in first.items())
    if isinstance(first, (tuple, list)):
        return len(first) == len(second) and all(_same(a, b) for a, b in zip(first, second))
    if hasattr(first, 'to_dict'):
        return _same(first.to_dict(), second.to_dict())
    return first == second


def _assemble_parts(built):
    if not built:
        raise ValueError('Sample cache requires every original candidate')
    # Exactly the conversions in original build_inference_sample. The local
    # tables are referenced, never cloned or duplicated into the layout cache.
    return dict(source_patch=torch.from_numpy(built[0].source_patch.astype(np.float16)),
        target_patches=torch.from_numpy(np.stack([b.target_patch for b in built]).astype(np.float16)),
        source_local=built[0].source_local, target_locals=[b.target_local for b in built])


class _SampleProxy:
    def __init__(self, original, provider):
        self._original, self._provider = original, provider

    def __getattr__(self, name):
        return getattr(self._original, name)

    def materialize_sample_views(self, sample, **kwargs):
        capture = getattr(self._provider._layout_context, 'capture', None)
        if capture is not None:
            capture(sample)
        return self._original.materialize_sample_views(sample, **kwargs)


class _RuntimeProxy:
    def __init__(self, original, provider):
        self._original = original
        self.sample = _SampleProxy(original.sample, provider)

    def __getattr__(self, name):
        return getattr(self._original, name)


def sample_cache_provider(original):
    """Wrap outside parallel views, inside optional input timing instrumentation."""
    class SampleCacheProvider(original):
        def __init__(self, *args, **kwargs):
            self._layout_context = threading.local()
            self._layout_runtime_lock = threading.RLock()
            self._layout_runtime = None
            self._layout_verify_lock = threading.RLock()
            # These receipts contain only hashes and file-stat tuples, no arrays.
            self._layout_verified_files = {}
            self._layout_verified_raw = set()
            self._layout_stats = dict(original_builds=0, hits=0, disk_reads=0,
                local_reference_reads=0, raw_verifications=0, original_seconds=0., hot_seconds=0.)
            super().__init__(*args, **kwargs)
            self._layout_root = _safe(Path(self.root) / 'sample_layout', directory=True)

        def _runtime(self):
            with self._layout_runtime_lock:
                value = super()._runtime()
                if self._layout_runtime is None or self._layout_runtime._original is not value:
                    self._layout_runtime = _RuntimeProxy(value, self)
                return self._layout_runtime

        def _local_map(self, *args, **kwargs):
            built = super()._local_map(*args, **kwargs)
            if getattr(self._layout_context, 'capture', None) is not None:
                if getattr(self._layout_context, 'built', None) is not None:
                    raise ValueError('Original sample produced local graphs more than once')
                # Retain original references only until the canonical capture;
                # rereading evicted files here could duplicate whole graphs.
                self._layout_context.built = built
            return built

        def _layout_file(self, path, expected=None, *, raw=False):
            # Raw dataset paths retain the original provider's symlink semantics.
            path = Path(path).resolve(strict=True) if raw else _safe(path)
            with _publication_guard(path):
                token = _stat(path)
                with self._layout_verify_lock:
                    previous = self._layout_verified_files.get(str(path))
                if previous is not None and token == previous[0]:
                    actual = previous[1]
                else:
                    actual = _sha(path)
                    if _stat(path) != token:
                        raise ValueError(f'Sample cache input changed while hashing: {path}')
                    required = expected if expected is not None else (previous[1] if previous else None)
                    if required is not None and actual != required:
                        raise ValueError(f'Sample cache input SHA256 differs: {path}')
                    with self._layout_verify_lock:
                        self._layout_verified_files[str(path)] = (token, actual)
                        if raw:
                            self._layout_stats['raw_verifications'] += 1
                if expected is not None and actual != expected:
                    raise ValueError(f'Sample cache input SHA256 differs: {path}')
                return dict(sha256=actual, bytes=token[2])

        def _layout_binding(self, example, center_keys, centers, *, verify=True):
            if verify:
                self._runtime()  # Preserve original snapshot verification before hits.
            scope = self._scope
            if not isinstance(scope, dict) or not scope.get('original_module_sha256'):
                raise ValueError('Sample cache requires verified original source identities')
            raw = self.raw[example['case_id']]
            hashes = [scope['source_archive_sha256'], scope['contract_sha256'],
                      *scope['original_module_sha256'].values(), raw['image_sha256'], raw['label_sha256']]
            if any(not isinstance(v, str) or _HEX.fullmatch(v) is None for v in hashes):
                raise ValueError('Sample cache requires complete SHA256 identities')
            # Readiness uses signed in-memory identities, never CT/tensor reads.
            local_binding = dict(format=SOURCE_FORMAT, source_id=example['id'],
                source_component=example['source_component'], positive_center=example['positive_center'],
                center=centers[0], image_sha256=raw['image_sha256'], label_sha256=raw['label_sha256'],
                config=self.config, prototype_fingerprint=self.bank_fingerprint,
                scope_contract_sha256=scope['contract_sha256'])
            if verify:
                raw_identity = tuple((str(raw[name]), raw[name + '_sha256']) for name in ('image', 'label'))
                with self._layout_verify_lock:
                    verified = raw_identity in self._layout_verified_raw
                if not verified:
                    for name in ('image', 'label'):
                        self._layout_file(raw[name], raw[name + '_sha256'], raw=True)
                    with self._layout_verify_lock:
                        self._layout_verified_raw.add(raw_identity)
                # Prove the cheap identity formula against the original method.
                if _canonical(self._binding(example, centers[0])) != _canonical(local_binding):
                    raise ValueError('Sample layout identity differs from original canonical binding')
            local_bindings = [_canonical(dict(local_binding, center=center)) for center in centers]
            binding = _canonical(dict(format=FORMAT, example=example, center_keys=center_keys,
                centers=centers, local_binding=local_binding, source_scope=scope,
                raw_shape=raw.get('shape'), raw_spacing=raw.get('spacing')))
            return binding, local_bindings

        def cached_batch_ready(self, indices, arm, epoch, *, training, full=False):
            """Cheap durable-layout check for bounded HOT-only batch staging.

            Present invalid publications raise; only unpublished layouts return
            False. Actual sample reads still verify raw and reference content.
            """
            if type(training) is not bool or type(epoch) is not int or epoch < 0:
                raise ValueError('Explicit sample readiness epoch/mode required')
            if self._scope is None:
                return False
            ids = list(indices)
            if not ids or len(set(ids)) != len(ids) or (training and full):
                raise ValueError('Readiness requires the original nonempty distinct batch')
            for index in ids:
                example = self._example(index)
                keys = tuple(self.candidate_keys(index, arm, epoch, full=full))
                centers = self._centers_for(example, keys)
                binding, bindings = self._layout_binding(example, keys, centers, verify=False)
                identity = _hash(binding)
                with self._lock:
                    if ('sample_layout', identity) in self._cache:
                        continue
                directory = _safe(self._layout_root / identity, directory=True)
                if not directory.exists():
                    return False
                self._layout_metadata(directory, binding, bindings)
            return True

        def _layout_local(self, reference, binding):
            key = _hash(binding)
            if reference.get('key') != key:
                raise ValueError('Sample cache local reference order/binding differs')
            path = Path(self.graph_dir) / (key + '.pt')

            def load():
                receipt = self._layout_file(path, reference['sha256'])
                if receipt['bytes'] != reference['bytes']:
                    raise ValueError('Sample cache local reference size differs')
                payload = torch.load(path, map_location='cpu', weights_only=False)
                if not isinstance(payload, dict) or payload.get('binding') != binding:
                    raise ValueError('Original canonical local reference binding differs')
                built = payload.get('built')
                if type(built) is not self._runtime().local.BuiltLocalGraph:
                    raise ValueError('Original canonical local reference type differs')
                with self._lock:
                    self._layout_stats['local_reference_reads'] += 1
                    self.stats['disk_hits'] += 1
                return built

            # Existing preparation-reuse, pressure checks and resident LRU apply.
            return self._get(('local', key), load)

        def _layout_parts(self, references, bindings):
            if len(references) != len(bindings):
                raise ValueError('Sample cache local reference coverage differs')
            def obtain(pair):
                return self._layout_local(*pair)

            shared_map = getattr(self, '_comparison_map', None)
            if shared_map is not None:
                built = list(shared_map(obtain, zip(references, bindings)))
            else:
                with ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix='sample-layout') as pool:
                    built = list(pool.map(obtain, zip(references, bindings)))
            return _assemble_parts(built)

        def _layout_publish(self, directory, binding, bindings, canonical):
            if not _REFERENCED <= canonical.keys() or canonical.keys() & _SAMPLED:
                raise ValueError('Capture requires the original complete canonical sample before views')
            references = []
            for local_binding in bindings:
                key = _hash(local_binding)
                references.append(dict(key=key, **self._layout_file(Path(self.graph_dir) / (key + '.pt'))))
            built = getattr(self._layout_context, 'built', None)
            if built is None or len(built) != len(references):
                raise ValueError('Original candidate map was not captured completely')
            parts = _assemble_parts(built)
            try:
                self._check_budget()
                if any(not _same(canonical[name], parts[name]) for name in _REFERENCED):
                    raise ValueError('Referenced local files cannot reconstruct the original sample exactly')
            finally:
                del parts, built
                self._layout_context.built = None
            skeleton = copy.deepcopy({k: v for k, v in canonical.items() if k not in _REFERENCED})
            self._check_budget()
            directory.parent.mkdir(parents=True, exist_ok=True)
            staging = directory.with_name('.' + directory.name + '.' + uuid.uuid4().hex + '.tmp')
            staging.mkdir(exist_ok=False)
            payload = staging / 'skeleton.pt'
            with payload.open('xb') as stream:
                torch.save(skeleton, stream)
            restored = torch.load(payload, map_location='cpu', weights_only=False)
            if not _same(skeleton, restored):
                raise ValueError('Sample skeleton serialization changed values/layout/metadata')
            del restored
            metadata = dict(format=FORMAT, binding=binding, references=references,
                skeleton_sha256=_sha(payload), skeleton_bytes=payload.stat().st_size,
                local_graphs_and_patches_stored=False)
            metadata['metadata_sha256'] = _hash(metadata)
            with (staging / 'metadata.json').open('x', encoding='utf8') as stream:
                json.dump(metadata, stream, allow_nan=False)
            self._check_budget()
            _rename_new(staging, directory)

        def _layout_metadata(self, directory, binding, bindings):
            if ({p.name for p in directory.iterdir()} != {'metadata.json', 'skeleton.pt'}
                    or not directory.is_dir()):
                raise ValueError('Sample cache publication inventory differs')
            metadata_path = _safe(directory / 'metadata.json')
            raw = metadata_path.read_bytes()
            metadata = json.loads(raw)
            signed = dict(metadata)
            signature = signed.pop('metadata_sha256', None)
            references = metadata.get('references')
            if (signature != _hash(signed) or metadata.get('format') != FORMAT
                    or metadata.get('binding') != binding
                    or metadata.get('local_graphs_and_patches_stored') is not False
                    or not isinstance(references, list) or len(references) != len(bindings)
                    or any(set(ref) != {'key', 'sha256', 'bytes'} or ref['key'] != _hash(local_binding)
                        or not isinstance(ref['sha256'], str) or _HEX.fullmatch(ref['sha256']) is None
                        or type(ref['bytes']) is not int or ref['bytes'] <= 0
                        for ref, local_binding in zip(references, bindings))):
                raise ValueError('Sample cache identity/reference inventory differs')
            return metadata, raw

        def _layout_read(self, directory, binding, bindings):
            metadata, raw = self._layout_metadata(directory, binding, bindings)
            metadata_path = directory / 'metadata.json'
            metadata_sha = hashlib.sha256(raw).hexdigest()
            self._layout_file(metadata_path, metadata_sha)
            receipt = self._layout_file(directory / 'skeleton.pt', metadata['skeleton_sha256'])
            if receipt['bytes'] != metadata['skeleton_bytes']:
                raise ValueError('Sample cache skeleton size differs')
            skeleton = torch.load(directory / 'skeleton.pt', map_location='cpu', weights_only=False)
            if (not isinstance(skeleton, dict) or skeleton.keys() & (_REFERENCED | _SAMPLED)
                    or not {'patient_graph', 'prototype_graph', 'candidate_centers'} <= skeleton.keys()
                    or skeleton['candidate_centers'].tolist() != binding['centers']):
                raise ValueError('Sample cache skeleton fields/ordered candidates differ')
            with self._lock:
                self._layout_stats['disk_reads'] += 1
            return dict(skeleton=skeleton, metadata=metadata, metadata_file_sha256=metadata_sha)

        def sample(self, index, center_keys, epoch, training):
            if type(epoch) is not int or epoch < 0 or type(training) is not bool:
                raise ValueError('Explicit epoch and training/evaluation view mode required')
            if getattr(self._layout_context, 'capture', None) is not None:
                raise RuntimeError('Nested canonical sample capture')
            began = time.perf_counter()
            keys = tuple(center_keys)
            example = self._example(index)
            centers = self._centers_for(example, keys)
            binding, bindings = self._layout_binding(example, keys, centers)
            identity = _hash(binding)
            directory = self._layout_root / identity
            self._check_budget()
            cold_result = []

            def load_or_build():
                _safe(directory, directory=True)
                with _publication_guard(directory):
                    if not directory.exists():
                        captures = []

                        def capture(canonical):
                            if captures:
                                raise ValueError('Original sample materialized more than once')
                            self._layout_publish(directory, binding, bindings, canonical)
                            captures.append(True)

                        self._layout_context.capture = capture
                        self._layout_context.built = None
                        try:
                            result = super(SampleCacheProvider, self).sample(index, keys, epoch, training)
                        finally:
                            self._layout_context.capture = None
                            self._layout_context.built = None
                        if len(captures) != 1:
                            raise ValueError('Original sample bypassed canonical capture')
                        cold_result.append(result)
                        with self._lock:
                            self._layout_stats['original_builds'] += 1
                            self._layout_stats['original_seconds'] += time.perf_counter() - began
                    return self._layout_read(directory, binding, bindings)

            # All publication I/O lives behind the original RAM-LRU miss path.
            cached = self._get(('sample_layout', identity), load_or_build)
            if cold_result:
                return cold_result[0]
            metadata = cached['metadata']
            sample = copy.deepcopy(cached['skeleton'])
            sample.update(self._layout_parts(metadata['references'], bindings))
            result = self._runtime().sample.materialize_sample_views(sample, training=training,
                epoch=epoch, global_seed=self.config['seed'])
            result['bridge_source_id'] = example['id']
            result['bridge_center_keys'] = keys
            self._check_budget()
            with self._lock:
                self._layout_stats['hits'] += 1
                self._layout_stats['hot_seconds'] += time.perf_counter() - began
            return result

        def report(self):
            result = super().report()
            with self._lock:
                result['sample_cache'] = dict(self._layout_stats, format=FORMAT,
                    directory=str(self._layout_root), local_graphs_and_patches_stored=False,
                    resident_storage='original provider LRU and budget',
                    integrity_policy='verify publications on LRU misses; raw identity once per process',
                    epoch_views_rebuilt=True, rotating_candidate_sets_require_first_build=True)
            return result

    SampleCacheProvider.__name__ = 'SampleCache' + original.__name__
    return SampleCacheProvider
