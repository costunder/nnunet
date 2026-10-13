"""CPU DEBUG exact geometry memoization fixtures; no training or final metrics."""
from __future__ import annotations

import types
import unittest

import numpy as np
from scipy import ndimage as ndi

from hiercp_v1x.basic_cp_large_geometry_cache import SourceFieldCache,cached_full_local_graph,optimize_provider,release_provider_cache


def center_crop_or_pad(array,shape,*,pad_value=0):
    target=np.full(shape,pad_value,dtype=array.dtype)
    src=[];dst=[]
    for before,after in zip(array.shape,shape):
        size=min(before,after)
        src.append(slice((before-size)//2,(before-size)//2+size))
        dst.append(slice((after-size)//2,(after-size)//2+size))
    target[tuple(dst)]=array[tuple(src)]
    return target


def signed_distance(mask,spacing):
    binary=np.asarray(mask,dtype=bool)
    outside=ndi.distance_transform_edt(~binary,sampling=spacing).astype(np.float32)
    inside=ndi.distance_transform_edt(binary,sampling=spacing).astype(np.float32)
    outside[binary]=-inside[binary]
    return outside


def normal_and_curvature(signed,spacing):
    gradient=np.gradient(signed.astype(np.float32),*spacing,edge_order=1)
    vector=np.stack(gradient,axis=-1).astype(np.float32)
    magnitude=np.linalg.norm(vector,axis=-1,keepdims=True)
    normal=vector/np.maximum(magnitude,1e-6)
    curvature=np.zeros_like(signed,dtype=np.float32)
    for axis in range(3):curvature+=np.gradient(normal[...,axis],spacing[axis],axis=axis,edge_order=1)
    return normal.astype(np.float32),np.clip(curvature/3,-2,2).astype(np.float32)


def build_patch_payload(footprint,shape,spacing,organ):
    native=center_crop_or_pad(footprint.astype(np.uint8),shape,pad_value=0).astype(bool)
    signed=signed_distance(native,spacing)
    normals=normal_and_curvature(signed,spacing)
    # Patient organ remains independently computed for every candidate.
    liver=signed_distance(organ,spacing)
    outside=ndi.distance_transform_edt(~native,sampling=spacing)
    return {'footprint':native,'signed':signed,'normal':normals,'organ_signed':liver,'outside':outside}


def _patch_fields(footprint,shape,spacing,organ):
    return build_patch_payload(footprint,shape,spacing,organ)


def local_fixture(prepared,shape,spacing,organ):
    return _patch_fields(prepared.source_footprint,shape,spacing,organ)


def provider_local_fixture(shape,spacing,*,full_organ_mask,prepared_source=None):
    return _patch_fields(prepared_source.source_footprint,shape,spacing,full_organ_mask)


class ExactFieldCacheTests(unittest.TestCase):
    def setUp(self):
        self.footprint=np.zeros((3,3,3),dtype=bool)
        self.footprint[1,1,1]=True
        self.spacing=(.7,.7,2.5)
        self.cache=SourceFieldCache(self.footprint,center_crop_or_pad=center_crop_or_pad,
            signed_distance=signed_distance,normal_and_curvature=normal_and_curvature,
            distance_transform_edt=ndi.distance_transform_edt,cache_limit_bytes=2**20)

    def test_signed_normals_outside_are_byte_exact_on_first_and_cached_calls(self):
        for shape in ((7,7,5),(9,9,7),(7,7,5)):
            native=center_crop_or_pad(self.footprint,shape)
            actual=self.cache.signed_distance(native,self.spacing)
            self.assertEqual(actual.tobytes(),signed_distance(native,self.spacing).tobytes())
            normals=self.cache.normal_and_curvature(actual,self.spacing)
            for observed,expected in zip(normals,normal_and_curvature(actual,self.spacing)):
                self.assertEqual(observed.tobytes(),expected.tobytes())
            outside=self.cache.distance_transform_edt(~native,sampling=self.spacing)
            self.assertEqual(outside.tobytes(),ndi.distance_transform_edt(~native,sampling=self.spacing).tobytes())
        self.assertEqual(self.cache.diagnostics['signed_hits'],1)
        self.assertEqual(self.cache.diagnostics['normal_hits'],1)
        self.assertEqual(self.cache.diagnostics['outside_hits'],1)

    def test_other_organ_mask_or_same_voxel_count_different_location_is_not_cached(self):
        native=center_crop_or_pad(self.footprint,(7,7,5))
        shifted=np.roll(native,1,axis=0)
        for mask in (shifted,np.ones_like(native)):
            actual=self.cache.signed_distance(mask,self.spacing)
            self.assertEqual(actual.tobytes(),signed_distance(mask,self.spacing).tobytes())
        self.assertEqual(self.cache.diagnostics['non_source_signed_calls'],2)
        self.assertEqual(self.cache.bytes,0)

    def test_spacing_and_native_roi_shape_are_separate_keys(self):
        for shape,spacing in (((7,7,5),self.spacing),((7,7,5),(1.,1.,1.)),((9,9,7),self.spacing)):
            native=center_crop_or_pad(self.footprint,shape)
            self.cache.signed_distance(native,spacing)
        self.assertEqual(self.cache.diagnostics['signed_misses'],3)
        self.assertEqual(len(self.cache.records),3)

    def test_memory_eviction_changes_recomputation_only(self):
        self.cache.limit=2*7*7*5*4
        first=center_crop_or_pad(self.footprint,(7,7,5))
        second=center_crop_or_pad(self.footprint,(7,7,7))
        for native in (first,second,first):
            actual=self.cache.signed_distance(native,self.spacing)
            self.assertEqual(actual.tobytes(),signed_distance(native,self.spacing).tobytes())
            self.assertLessEqual(self.cache.bytes,self.cache.limit)
        self.assertGreater(self.cache.diagnostics['evictions'],0)

    def test_oversized_full_result_is_computed_without_caching_or_crop(self):
        self.cache.limit=16
        native=center_crop_or_pad(self.footprint,(7,7,5))
        actual=self.cache.signed_distance(native,self.spacing)
        self.assertEqual(actual.shape,native.shape)
        self.assertEqual(actual.tobytes(),signed_distance(native,self.spacing).tobytes())
        self.assertEqual(self.cache.bytes,0)

    def test_options_on_other_distance_calls_are_retained(self):
        native=center_crop_or_pad(self.footprint,(7,7,5))
        actual=self.cache.distance_transform_edt(~native,sampling=self.spacing,return_distances=False,return_indices=True)
        expected=ndi.distance_transform_edt(~native,sampling=self.spacing,return_distances=False,return_indices=True)
        self.assertEqual(actual.tobytes(),expected.tobytes())


class PrivateNamespaceTests(unittest.TestCase):
    def test_full_builder_and_original_globals_bytecode_are_unchanged(self):
        footprint=np.zeros((3,3,3),dtype=bool);footprint[1,1,1]=True
        prepared=types.SimpleNamespace(source_footprint=footprint)
        originals={id(fn):(fn,dict(fn.__globals__)) for fn in
                   (local_fixture,_patch_fields,build_patch_payload,signed_distance,normal_and_curvature)}
        adapted,cache=cached_full_local_graph(local_fixture,prepared,cache_limit_bytes=2**20)
        self.assertIs(adapted.__code__,local_fixture.__code__)
        for shape in ((7,7,5),(9,9,7),(7,7,5)):
            organ=np.ones(shape,dtype=bool);organ[0]=0
            expected=local_fixture(prepared,shape,(.7,.7,2.5),organ)
            observed=adapted(prepared,shape,(.7,.7,2.5),organ)
            for key in expected:
                old,new=expected[key],observed[key]
                if isinstance(old,tuple):
                    for left,right in zip(old,new):self.assertEqual(left.tobytes(),right.tobytes())
                else:self.assertEqual(old.tobytes(),new.tobytes())
        for function,globals_before in originals.values():
            for name,value in globals_before.items():
                self.assertIs(function.__globals__[name],value)
        self.assertEqual(cache.diagnostics['signed_hits'],1)
        self.assertEqual(cache.diagnostics['non_source_signed_calls'],3)

    def test_private_provider_resets_on_source_change_and_releases_on_patient_change(self):
        footprint=np.zeros((3,3,3),dtype=bool);footprint[1,1,1]=True
        prepared=types.SimpleNamespace(source_footprint=footprint)
        organ=np.ones((7,7,5),dtype=bool);organ[0]=0
        original_flags=organ.flags.writeable
        provider=types.SimpleNamespace(local=provider_local_fixture,
                                       regions=types.SimpleNamespace(full_organ_mask=organ))
        manager=optimize_provider(provider,cache_limit_bytes=2**20)
        for source in (prepared,prepared,types.SimpleNamespace(source_footprint=footprint.copy())):
            observed=provider.local(organ.shape,(.7,.7,2.5),full_organ_mask=organ,prepared_source=source)
            expected=provider_local_fixture(organ.shape,(.7,.7,2.5),full_organ_mask=organ,prepared_source=source)
            self.assertEqual(observed['signed'].tobytes(),expected['signed'].tobytes())
            self.assertEqual(observed['organ_signed'].tobytes(),expected['organ_signed'].tobytes())
        self.assertEqual(manager.source_changes,2)
        self.assertEqual(organ.flags.writeable,original_flags)
        self.assertTrue(manager.organ_uint8.dtype==np.uint8)
        release_provider_cache(provider)
        self.assertIs(provider.local,provider_local_fixture)
        self.assertFalse(hasattr(provider,'_exact_geometry_cache_manager'))


if __name__=='__main__':unittest.main()
