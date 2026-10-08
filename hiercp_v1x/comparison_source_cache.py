"""Persist exact source preparation without storing duplicate whole-CT masks.

The original provider still selects and prepares every cold source. Its own
resident LRU owns reconstructed objects; this adapter retains only counters and
a thread-local factory context. No model, candidate, view seed or source file is
changed. Publications live in a new, independently bound execution cache.
"""
from __future__ import annotations

from dataclasses import fields
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


FORMAT = 'comparison_exact_compact_source_v1'
_HEX = re.compile(r'[0-9a-f]{64}')
_SOURCE_FIELDS = {'component_id', 'patch_mask', 'patch_image', 'patch_slices',
                  'anchor_center', 'centroid', 'voxel_count'}
_PREPARED_FIELDS = {'source_footprint', 'source_patch', 'canonical_nodes',
                    'canonical_edges', 'canonical_counts'}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def _sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            result.update(block)
    return result.hexdigest()


def source_binding(provider, example, case):
    """Use already verified raw/snapshot identity; never relax its checks."""
    provider._runtime()  # Original lazy verification must precede any disk hit.
    scope = provider._scope
    raw = provider.raw[example['case_id']]
    if (case.paths.case_id != example['case_id'] or tuple(case.image.shape) != tuple(case.shape)
            or tuple(case.label.shape) != tuple(case.shape)):
        raise ValueError('Source cache case/image/label identity differs')
    if not isinstance(scope, dict) or not scope.get('original_module_sha256'):
        raise ValueError('Source cache requires verified original module identities')
    identities = [raw['image_sha256'], raw['label_sha256'],
                  scope['source_archive_sha256'], scope['contract_sha256'],
                  *scope['original_module_sha256'].values()]
    if any(not isinstance(value, str) or _HEX.fullmatch(value) is None for value in identities):
        raise ValueError('Source cache requires complete SHA256 source bindings')
    binding = dict(format=FORMAT, image_sha256=raw['image_sha256'], label_sha256=raw['label_sha256'],
        source_archive_sha256=scope['source_archive_sha256'], scope_contract_sha256=scope['contract_sha256'],
        original_module_sha256=scope['original_module_sha256'], config=provider.config,
        example={name: example[name] for name in ('id', 'case_id', 'sample_index',
            'source_component', 'positive_center', 'original_sample_sha256')},
        shape=list(case.shape), spacing=list(map(float, case.spacing)),
        image_dtype=case.image.dtype.str, label_dtype=case.label.dtype.str,
        spacing_dtype=np.asarray(case.spacing).dtype.str)
    # Freeze mutable provider dictionaries and normalize tuples exactly once.
    binding = json.loads(json.dumps(binding, allow_nan=False))
    if (len(binding['shape']) != 3 or any(type(n) is not int or n <= 0 for n in binding['shape'])
            or len(binding['spacing']) != 3 or any(v <= 0 for v in binding['spacing'])):
        raise ValueError('Source cache requires the original 3D shape and positive spacing')
    return binding


def _layout(array):
    return dict(shape=list(array.shape), dtype=array.dtype.str, strides=list(array.strides))


def _scope_identity(binding):
    identity = binding.get('scope_contract_sha256')
    if not isinstance(identity, str) or _HEX.fullmatch(identity) is None:
        raise ValueError('Verified bounded source scope SHA256 required')
    return identity


def _validate_compact(payload, binding):
    if (not isinstance(payload, dict) or set(payload) != {'source', 'prepared', 'numpy_layouts'}
            or set(payload['source']) != _SOURCE_FIELDS
            or set(payload['prepared']) != _PREPARED_FIELDS):
        raise ValueError('Unexpected compact source payload fields')
    source, prepared = payload['source'], payload['prepared']
    if (type(source['component_id']) is not int
            or source['component_id'] != binding['example']['source_component']
            or tuple(source['anchor_center']) != tuple(binding['example']['positive_center'])):
        raise ValueError('Cached source component/anchor differs from original sample')
    slices = source['patch_slices']
    if (not isinstance(slices, tuple) or len(slices) != 3
            or any(not isinstance(s, slice) or s.step is not None
                or type(s.start) is not int or type(s.stop) is not int
                or not 0 <= s.start < s.stop <= size for s, size in zip(slices, binding['shape']))):
        raise ValueError('Cached source patch slices differ from native volume bounds')
    patch_shape = tuple(s.stop - s.start for s in slices)
    arrays = {'patch_mask': source['patch_mask'], 'patch_image': source['patch_image'],
              'source_footprint': prepared['source_footprint'], 'source_patch': prepared['source_patch']}
    if (any(not isinstance(a, np.ndarray) or a.dtype.hasobject or not np.isfinite(a).all()
            for a in arrays.values()) or set(payload['numpy_layouts']) != set(arrays)
            or any(_layout(a) != payload['numpy_layouts'][name] for name, a in arrays.items())):
        raise ValueError('Cached source array values/layout differ')
    if (source['patch_mask'].dtype != np.dtype(bool) or source['patch_mask'].shape != patch_shape
            or source['patch_image'].dtype != np.dtype('float32') or source['patch_image'].shape != patch_shape
            or type(source['voxel_count']) is not int or source['voxel_count'] <= 0
            or np.count_nonzero(source['patch_mask']) != source['voxel_count']
            or prepared['source_footprint'].dtype != np.dtype(bool)
            or not np.array_equal(prepared['source_footprint'], source['patch_mask'])
            or prepared['source_patch'].dtype != np.dtype('float32')
            or prepared['source_patch'].ndim != 4):
        raise ValueError('Cached exact source footprint/patch contract differs')
    if (tuple(s.start + size // 2 for s, size in zip(slices, patch_shape)) != tuple(source['anchor_center'])
            or len(source['centroid']) != 3 or not np.isfinite(source['centroid']).all()
            or any(not 0 <= v < size for v, size in zip(source['centroid'], binding['shape']))):
        raise ValueError('Cached source anchor/centroid geometry differs')
    nodes, edges, counts = (prepared[name] for name in ('canonical_nodes', 'canonical_edges', 'canonical_counts'))
    if not all(isinstance(value, dict) for value in (nodes, edges, counts)) or set(nodes) != set(counts):
        raise ValueError('Cached canonical source node schema differs')
    for name, node in nodes.items():
        if (not isinstance(node, dict) or not node or 'x' not in node
                or type(counts[name]) is not int or counts[name] < 0):
            raise ValueError('Cached canonical source node/count differs')
        for value in node.values():
            if (not torch.is_tensor(value) or value.device.type != 'cpu' or value.ndim < 1
                    or value.shape[0] != counts[name] or not bool(torch.isfinite(value).all())):
                raise ValueError('Cached canonical source tensor differs')
    for relation, value in edges.items():
        if (not isinstance(relation, tuple) or len(relation) != 3
                or relation[0] not in nodes or relation[2] not in nodes
                or not torch.is_tensor(value) or value.device.type != 'cpu'
                or value.dtype not in (torch.int32, torch.int64) or value.ndim != 2 or value.shape[0] != 2):
            raise ValueError('Cached canonical source edge schema differs')
        if value.numel() and (bool((value < 0).any())
                or bool((value[0] >= counts[relation[0]]).any())
                or bool((value[1] >= counts[relation[2]]).any())):
            raise ValueError('Cached canonical source edge index is out of bounds')
    return source, prepared


def compact_source(result, runtime, binding):
    """Prove that the original full mask is represented by the exact patch."""
    if not isinstance(result, tuple) or len(result) != 2:
        raise ValueError('Original source preparation must return its two objects')
    source, prepared = result
    if (type(source) is not runtime.common.SourceTumor or type(prepared) is not runtime.local.PreparedLocalSource
            or {f.name for f in fields(source)} != _SOURCE_FIELDS | {'full_mask'}
            or {f.name for f in fields(prepared)} != _PREPARED_FIELDS):
        raise ValueError('Original source/prepared dataclass schema differs')
    if getattr(prepared, 'v1x_bounded_scope_contract', None) != _scope_identity(binding):
        raise ValueError('Original prepared source bounded scope differs from cache binding')
    payload = dict(source={name: getattr(source, name) for name in _SOURCE_FIELDS},
                   prepared={name: getattr(prepared, name) for name in _PREPARED_FIELDS})
    arrays = dict(patch_mask=source.patch_mask, patch_image=source.patch_image,
                  source_footprint=prepared.source_footprint, source_patch=prepared.source_patch)
    payload['numpy_layouts'] = {name: _layout(array) for name, array in arrays.items()}
    _validate_compact(payload, binding)
    mask = source.full_mask
    expected_strides = (binding['shape'][1] * binding['shape'][2], binding['shape'][2], 1)
    if (not isinstance(mask, np.ndarray) or mask.dtype != np.dtype(bool)
            or mask.shape != tuple(binding['shape']) or mask.strides != expected_strides
            or not np.array_equal(mask[source.patch_slices], source.patch_mask)
            or np.count_nonzero(mask) != source.voxel_count):
        raise ValueError('Original whole-source mask cannot be reconstructed exactly from its patch')
    return payload


def restore_source(payload, runtime, binding, check):
    source, prepared = _validate_compact(payload, binding)
    identity = _scope_identity(binding)
    check()
    full_mask = np.zeros(tuple(binding['shape']), dtype=bool)
    full_mask[source['patch_slices']] = source['patch_mask']
    result = (runtime.common.SourceTumor(full_mask=full_mask, **source),
              runtime.local.PreparedLocalSource(**prepared))
    # This is a runtime attribute installed by bounded_scope.prepare, not a
    # dataclass field. _load has already verified the metadata SHA, the exact
    # active source/scope binding and payload bytes before reaching this point.
    # Recover it from that binding, including for existing v1 publications.
    # Never alter the bounded graph admission check or silently widen the ROI.
    result[1].v1x_bounded_scope_contract = identity
    check()
    return result


def _load(directory, runtime, binding, check):
    metadata_path, payload_path = directory/'metadata.json', directory/'payload.pt'
    _safe(metadata_path); _safe(payload_path)
    metadata = json.loads(metadata_path.read_text(encoding='utf8'))
    signed = dict(metadata); signature = signed.pop('metadata_sha256', None)
    if (signature != _digest(signed) or metadata.get('format') != FORMAT
            or metadata.get('binding') != binding or metadata.get('binding_sha256') != _digest(binding)
            or metadata.get('full_volume_mask_stored') is not False
            or payload_path.stat().st_size != metadata.get('payload_bytes')
            or _sha(payload_path) != metadata.get('payload_sha256')):
        raise ValueError('Source execution cache binding/content identity differs')
    check()
    payload = torch.load(payload_path, map_location='cpu', weights_only=False)
    return restore_source(payload, runtime, binding, check)


def source_cache_provider(original):
    """Intercept only the unchanged source factory under the original RAM LRU."""
    class CachedSourceProvider(original):
        def __init__(self, *args, **kwargs):
            self._source_cache_context = threading.local()
            self.source_cache_stats = dict(original_builds=0, disk_hits=0,
                original_build_seconds=0., disk_read_seconds=0., full_volume_mask_stored=False)
            super().__init__(*args, **kwargs)

        def _source(self, example, case, organ, depth):
            if getattr(self._source_cache_context, 'active', None) is not None:
                raise RuntimeError('Nested original source preparation context')
            binding = source_binding(self, example, case)
            self._source_cache_context.active = (('source', example['id']), binding, self._runtime())
            try:
                return super()._source(example, case, organ, depth)
            finally:
                self._source_cache_context.active = None

        def report(self):
            result = super().report()
            with self._lock:
                result['source_cache_stats'] = dict(self.source_cache_stats,
                    directory=str(Path(self.root)/'source_prepared'),
                    resident_storage='original provider LRU and budget')
            return result

        def _get(self, key, factory):
            context = getattr(self._source_cache_context, 'active', None)
            if context is None or key != context[0]:
                return super()._get(key, factory)
            _, binding, runtime = context

            def disk_factory():
                directory = _safe(Path(self.root)/'source_prepared'/_digest(binding), directory=True)
                with _publication_guard(directory):
                    if directory.exists():
                        began = time.perf_counter()
                        result = _load(directory, runtime, binding, self._check_budget)
                        with self._lock:
                            self.source_cache_stats['disk_hits'] += 1
                            self.source_cache_stats['disk_read_seconds'] += time.perf_counter() - began
                        return result
                    began = time.perf_counter()
                    result = factory()  # Unmodified original source selection and construction.
                    elapsed = time.perf_counter() - began
                    payload = compact_source(result, runtime, binding)
                    self._check_budget()
                    directory.parent.mkdir(parents=True, exist_ok=True)
                    staging = directory.with_name('.'+directory.name+'.'+uuid.uuid4().hex+'.tmp')
                    staging.mkdir()
                    payload_path = staging/'payload.pt'
                    with payload_path.open('xb') as stream:
                        torch.save(payload, stream)
                    metadata = dict(format=FORMAT, binding=binding, binding_sha256=_digest(binding),
                        payload_sha256=_sha(payload_path), payload_bytes=payload_path.stat().st_size,
                        full_volume_mask_stored=False, original_build_seconds=elapsed)
                    metadata['metadata_sha256'] = _digest(metadata)
                    with (staging/'metadata.json').open('x', encoding='utf8') as stream:
                        json.dump(metadata, stream, allow_nan=False)
                    # Verify the compact serialization without allocating another full mask.
                    _validate_compact(torch.load(payload_path, map_location='cpu', weights_only=False), binding)
                    self._check_budget()
                    _rename_new(staging, directory)
                    with self._lock:
                        self.source_cache_stats['original_builds'] += 1
                        self.source_cache_stats['original_build_seconds'] += elapsed
                    return result

            return super()._get(key, disk_factory)

    CachedSourceProvider.__name__ = 'CachedSource'+original.__name__
    return CachedSourceProvider
