"""Exact private memoization of source-invariant full-graph geometry fields.

Every frozen graph, feature, coordinate, edge and rejection gate still executes.
The frozen function bytecodes and original namespaces remain unchanged. Only
centered source footprint fields are memoized by exact mask bytes, ROI shape and
spacing. A byte-bounded LRU evicts derived arrays; it never limits the graph,
candidate population, searched domain or scientific input.
"""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import types

import numpy as np


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _clone(function, replacements):
    _require(isinstance(function,types.FunctionType) and function.__closure__ is None,
             "Exact frozen no-closure function required for private geometry memoization")
    _require(set(replacements).issubset(function.__globals__),
             "Frozen geometry global names changed")
    namespace = dict(function.__globals__)
    namespace.update(replacements)
    clone = types.FunctionType(function.__code__,namespace,function.__name__,
                               function.__defaults__,function.__closure__)
    clone.__kwdefaults__ = function.__kwdefaults__
    _require(clone.__code__ is function.__code__,"Frozen scientific function bytecode changed")
    return clone


def _digest(array):
    value = np.ascontiguousarray(array)
    return hashlib.sha256(memoryview(value).cast('B')).hexdigest()


def _array_bytes(value):
    if isinstance(value,np.ndarray):
        return value.nbytes
    return sum(_array_bytes(item) for item in value)


def _readonly(value):
    if isinstance(value,np.ndarray):
        value.flags.writeable = False
    else:
        for item in value:
            _readonly(item)
    return value


class SourceFieldCache:
    """One immutable source footprint and exact spacing/ROI keyed derived arrays."""
    def __init__(self, footprint, *, center_crop_or_pad, signed_distance,
                 normal_and_curvature, distance_transform_edt, cache_limit_bytes):
        _require(type(cache_limit_bytes) is int and cache_limit_bytes > 0,
                 "Explicit positive derived-geometry cache byte budget required")
        value = np.asarray(footprint,dtype=bool)
        _require(value.ndim == 3 and value.any(),"Actual nonempty3D source footprint required")
        self.footprint = value.copy()
        self.footprint.flags.writeable = False
        self.voxel_count = int(np.count_nonzero(value))
        self.crop = center_crop_or_pad
        self.original_signed = signed_distance
        self.original_normal = normal_and_curvature
        self.original_edt = distance_transform_edt
        self.limit = cache_limit_bytes
        self.expected = {}
        self.records = OrderedDict()
        self.bytes = 0
        self.sdf_objects = {}
        self.diagnostics = {"cache_limit_bytes":self.limit,"cache_bytes":0,"peak_cache_bytes":0,
            "signed_hits":0,"signed_misses":0,"normal_hits":0,"normal_misses":0,
            "outside_hits":0,"outside_misses":0,"evictions":0,
            "non_source_signed_calls":0,"non_source_normal_calls":0,"non_source_edt_calls":0,
            "source_geometry_only":True,"graph_or_candidate_scale_changed":False}

    @staticmethod
    def _spacing(spacing):
        return tuple(float(value) for value in spacing)

    def _expected(self, shape):
        shape = tuple(int(value) for value in shape)
        if shape not in self.expected:
            centered = np.asarray(self.crop(self.footprint.astype(np.uint8),shape,pad_value=0),dtype=bool)
            _require(int(np.count_nonzero(centered)) == self.voxel_count,
                     "Memoization cannot admit a cropped source footprint")
            self.expected[shape] = (_digest(centered),_digest(~centered))
        return self.expected[shape]

    def _source_key(self, mask, spacing, *, inverse=False):
        binary = np.asarray(mask,dtype=bool)
        if binary.ndim != 3:
            return None
        expected_voxels = binary.size-self.voxel_count if inverse else self.voxel_count
        if int(np.count_nonzero(binary)) != expected_voxels:
            return None
        expected = self._expected(binary.shape)[1 if inverse else 0]
        if _digest(binary) != expected:
            return None
        return (tuple(binary.shape),self._spacing(spacing),self._expected(binary.shape)[0])

    def _get(self, key, name):
        record = self.records.get(key)
        if record is not None:
            self.records.move_to_end(key)
            return record.get(name)
        return None

    def _evict(self, key):
        record = self.records.pop(key)
        for name,value in record.items():
            self.bytes -= _array_bytes(value)
            if name == 'signed':
                self.sdf_objects.pop(id(value),None)
        self.diagnostics['evictions'] += 1

    def _store(self, key, name, value):
        size = _array_bytes(value)
        # Oversized scratch results are still computed in full, simply uncached.
        if size > self.limit:
            return value
        needed = self.bytes + size
        while needed > self.limit and self.records:
            oldest = next(iter(self.records))
            self._evict(oldest)
            needed = self.bytes + size
        record = self.records.setdefault(key,{})
        _require(name not in record,"Source geometry cache unexpectedly overwrote a field")
        record[name] = _readonly(value)
        self.records.move_to_end(key)
        self.bytes += size
        if name == 'signed':
            self.sdf_objects[id(value)] = (key,value)
        self.diagnostics.update(cache_bytes=self.bytes,
            peak_cache_bytes=max(self.bytes,self.diagnostics['peak_cache_bytes']))
        _require(self.bytes <= self.limit,"Derived-geometryLRU exceeded its explicit byte budget")
        return value

    def signed_distance(self, mask, spacing):
        key = self._source_key(mask,spacing)
        if key is None:
            self.diagnostics['non_source_signed_calls'] += 1
            return self.original_signed(mask,spacing)
        value = self._get(key,'signed')
        if value is not None:
            self.diagnostics['signed_hits'] += 1
            return value
        self.diagnostics['signed_misses'] += 1
        return self._store(key,'signed',self.original_signed(mask,spacing))

    def normal_and_curvature(self, signed, spacing):
        item = self.sdf_objects.get(id(signed))
        if item is None or item[1] is not signed or item[0][1] != self._spacing(spacing):
            self.diagnostics['non_source_normal_calls'] += 1
            return self.original_normal(signed,spacing)
        key = item[0]
        value = self._get(key,'normal')
        if value is not None:
            self.diagnostics['normal_hits'] += 1
            return value
        self.diagnostics['normal_misses'] += 1
        return self._store(key,'normal',self.original_normal(signed,spacing))

    def distance_transform_edt(self, mask, *, sampling, **kwargs):
        # Other scipy options retain their exact original behavior.
        if kwargs:
            self.diagnostics['non_source_edt_calls'] += 1
            return self.original_edt(mask,sampling=sampling,**kwargs)
        key = self._source_key(mask,sampling,inverse=True)
        if key is None:
            self.diagnostics['non_source_edt_calls'] += 1
            return self.original_edt(mask,sampling=sampling)
        value = self._get(key,'outside')
        if value is not None:
            self.diagnostics['outside_hits'] += 1
            return value
        self.diagnostics['outside_misses'] += 1
        return self._store(key,'outside',self.original_edt(mask,sampling=sampling))


def cached_full_local_graph(original_local, prepared_source, *, cache_limit_bytes):
    """Return(private full builder, source-field cache), with all scientific code intact.

    Use one returned adapter per source. Patient-organ conversions may separately
    be cached as immutableuint8 without changing any organ voxel value.
    """
    namespace = original_local.__globals__
    patch = namespace['_patch_fields']
    payload = patch.__globals__['build_patch_payload']
    payload_globals = payload.__globals__
    crop = payload_globals['center_crop_or_pad']
    signed = payload_globals['signed_distance']
    normal = payload_globals['normal_and_curvature']
    ndi = payload_globals['ndi']
    cache = SourceFieldCache(prepared_source.source_footprint,center_crop_or_pad=crop,
        signed_distance=signed,normal_and_curvature=normal,distance_transform_edt=ndi.distance_transform_edt,
        cache_limit_bytes=cache_limit_bytes)

    class NDIProxy:
        def __getattr__(self,name):
            return getattr(ndi,name)

        def distance_transform_edt(self, mask, *args, **kwargs):
            if args or set(kwargs) != {'sampling'}:
                return ndi.distance_transform_edt(mask,*args,**kwargs)
            return cache.distance_transform_edt(mask,**kwargs)

    private_payload = _clone(payload,{'signed_distance':cache.signed_distance,
        'normal_and_curvature':cache.normal_and_curvature,'ndi':NDIProxy()})
    private_patch = _clone(patch,{'build_patch_payload':private_payload})
    private_local = _clone(original_local,{'_patch_fields':private_patch})
    return private_local,cache


class ProviderGeometryCache:
    """Private worker wrapper: exact source changes discard the previous cache."""
    def __init__(self,provider,cache_limit_bytes):
        self.original_local = provider.local
        self.organ = provider.regions.full_organ_mask
        self.organ_uint8 = np.asarray(self.organ,dtype=np.uint8)
        if self.organ_uint8 is self.organ:
            # Never change write flags on an existing historical input.
            self.organ_uint8 = self.organ_uint8.copy()
        self.organ_uint8.flags.writeable = False
        self.cache_limit_bytes = cache_limit_bytes
        self.prepared = self.builder = self.cache = None
        self.source_changes = 0

    def __call__(self,*args,**kwargs):
        prepared = kwargs.get('prepared_source')
        if prepared is None:
            return self.original_local(*args,**kwargs)
        if prepared is not self.prepared:
            self.prepared = self.builder = self.cache = None
            self.builder,self.cache = cached_full_local_graph(self.original_local,prepared,
                cache_limit_bytes=self.cache_limit_bytes)
            self.prepared = prepared
            self.source_changes += 1
        kwargs = dict(kwargs)
        if kwargs.get('full_organ_mask') is self.organ:
            kwargs['full_organ_mask'] = self.organ_uint8
        return self.builder(*args,**kwargs)

    def reset_source(self):
        """Release the prior source BEFORE constructing the next complete source."""
        self.prepared = self.builder = self.cache = None

    @property
    def diagnostics(self):
        return dict(source_changes=self.source_changes,patient_organ_uint8_cache_bytes=self.organ_uint8.nbytes,
                    source_fields={} if self.cache is None else dict(self.cache.diagnostics))


def release_provider_cache(provider):
    """Call before replacing patient inputs to avoid retaining the previous source."""
    manager = getattr(provider,'_exact_geometry_cache_manager',None)
    if manager is not None:
        _require(provider.local is manager,"Private provider local callable changed unexpectedly")
        provider.local = manager.original_local
        del provider._exact_geometry_cache_manager


def reset_provider_source_cache(provider):
    """Keep patient organ data but release all previous-source derived tensors."""
    manager = getattr(provider,'_exact_geometry_cache_manager',None)
    if manager is not None:
        _require(provider.local is manager,"Private provider local callable changed unexpectedly")
        manager.reset_source()


def optimize_provider(provider,*,cache_limit_bytes):
    """Install exact memoization only on this worker's current patient provider."""
    _require(type(cache_limit_bytes) is int and cache_limit_bytes > 0,
             "Explicit positive source-derived cache byte budget required")
    release_provider_cache(provider)
    manager = ProviderGeometryCache(provider,cache_limit_bytes)
    provider.local = manager
    provider._exact_geometry_cache_manager = manager
    return manager


__all__ = ['SourceFieldCache','cached_full_local_graph','ProviderGeometryCache',
           'optimize_provider','release_provider_cache','reset_provider_source_cache']
