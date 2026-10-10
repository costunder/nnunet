"""Explicit execution adapter; the five GT-free scientific files stay intact.

Original regions and their full EDT validator run once before admission.
Independent case builds run outside the global raw/LRU lock. Reuse is limited
to immutable, input-bound region values, sharing the original 8-GiB raw budget.
"""
from collections import OrderedDict
import ast
from dataclasses import dataclass
import hashlib
import inspect
from pathlib import Path
import threading
import time
import types
import weakref

import numpy as np
import psutil

from .contracts import canonical_hash
from .v24_inputs import immutable_array
from . import v24_factory as factory

FORMAT='v24_exact_immutable_region_preparation_runtime_v1'
_ORIGINAL=factory.V24NativeInputs
_ARRAY_NAMES=('full_organ_mask','organ_depth','region_labels','region_features',
              'region_positions','region_edge_index','region_centers_vox')


def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass(frozen=True)
class _AdmittedRegion:
    full_organ_mask: np.ndarray
    organ_depth: np.ndarray
    region_labels: np.ndarray
    region_features: np.ndarray
    region_positions: np.ndarray
    region_edge_index: np.ndarray
    region_centers_vox: np.ndarray
    input_sha256: str

    @property
    def num_regions(self):return int(self.region_features.shape[0])

    def region_at(self,center):
        from hiercp.region import PatientRegionData
        return PatientRegionData.region_at(self,center)


class AcceleratedNativeInputs(_ORIGINAL):
    def __init__(self,*args,**kwargs):
        self._static_regions=OrderedDict();self._region_bytes=0
        self._region_locks={};self._region_lock_guard=threading.Lock()
        self._region_admissions={}
        self._upper_sources=OrderedDict();self._source_bytes=0
        self._region_profile=dict(builds=0,hits=0,original_equation_validations=0,evictions=0,
                                  source_only_donor_builds=0,source_only_donor_hits=0)
        self._upper_runtime=None
        super().__init__(*args,**kwargs)
        if self.runtime['workers']!=4 or self.raw_resident_bytes!=8*2**30 or self.rss_bytes!=64*2**30:
            raise ValueError('Adapter preserves explicit four-worker/8GiB shared raw-region/64GiB RSS budget')
        self._region_workspaces=threading.BoundedSemaphore(self.runtime['workers'])
        # Execution-file admission is separate from the unchanged scientific
        # fingerprint, allowing already verified canonical/upper reuse.
        self._admit_runtime_files((Path(__file__),Path(__file__).resolve().parents[1]/'tools/run_v24_all_p.py'))
        self._install_prepare_progress()

    def _admit_runtime_files(self,paths):
        self._runtime_file_sha256={}
        for path in paths:
            path=Path(path).resolve(strict=True);before=factory._stat(path)
            checksum=_sha(path)
            if factory._stat(path)!=before:raise ValueError('Runtime source changed during admission')
            self._input_file_proofs[str(path)]=before;self._file_proofs[str(path)]=before
            self._runtime_file_sha256[str(path)]=checksum

    def _install_prepare_progress(self):
        cache=self.geometry;original_prepare=cache.prepare;prepare_lock=threading.Lock()
        def prepare(cache,count,target_selection=None):
            with prepare_lock:
                total=len(cache.population.partition_cases('inner_train',ranking_only=True))+len(cache.population.partition_cases('inner_val'))
                original_get=cache.get;counter_lock=threading.Lock();completed=0;started=time.perf_counter()
                def get(cache,plan,provider=None):
                    nonlocal completed
                    result=original_get(plan,provider)
                    with counter_lock:
                        completed+=1
                        print('V24 UPPER PREPARE completed=%d/%d active_U=%d elapsed_seconds=%.3f' %
                              (completed,total,count,time.perf_counter()-started),flush=True)
                    return result
                wrapped=types.MethodType(get,cache);cache.get=wrapped
                try:return original_prepare(count,target_selection=target_selection)
                finally:
                    if cache.get is not wrapped:raise ValueError('Upper preparation callback changed concurrently')
                    cache.get=original_get
        cache.prepare=types.MethodType(prepare,cache)

    def _case_region_lock(self,case):
        with self._region_lock_guard:
            return self._region_locks.setdefault(case,threading.RLock())

    def _region(self,case):
        from .v24_geometry import build_recipient_regions,_validate_regions
        from hiercp.common import stable_case_seed
        from hiercp.region import REGION_CACHE_SEED_SALT
        # Never wait for a case lock while owning the global raw/LRU lock.
        with self._case_region_lock(case):
            with self._lock:
                self.guard_input_files(case)
                if case in self._static_regions:
                    region,size=self._static_regions[case]
                    self._static_regions.move_to_end(case);self._region_profile['hits']+=1
                    self._rss_locked();return region
            raw=self._case(case)
            with self._region_workspaces:
                context=raw['context']
                def original_cached_depth(mask,spacing):
                    if mask is not context.organ_mask or not np.array_equal(spacing,context.spacing):
                        raise ValueError('Cached original EDT requires the same immutable organ/spacing')
                    return context.organ_depth
                class ExactDepthBuilders(dict):
                    def __setitem__(self,key,builder):
                        namespace=dict(builder.__globals__,organ_depth_mm=original_cached_depth)
                        replacement=types.FunctionType(builder.__code__,namespace,builder.__name__,
                            builder.__defaults__,builder.__closure__)
                        replacement.__kwdefaults__=builder.__kwdefaults__
                        super().__setitem__(key,replacement)
                builders=ExactDepthBuilders()
                for key,builder in build_recipient_regions.__globals__['_REGION_BUILDERS'].items():
                    builders[key]=builder
                wrapper=types.FunctionType(build_recipient_regions.__code__,
                    dict(build_recipient_regions.__globals__,_REGION_BUILDERS=builders),
                    build_recipient_regions.__name__,build_recipient_regions.__defaults__,build_recipient_regions.__closure__)
                wrapper.__kwdefaults__=build_recipient_regions.__kwdefaults__
                region=wrapper(context,config=self.graph_config,
                    seed=stable_case_seed(42,case,REGION_CACHE_SEED_SALT),ct_clip=self.ct_clip)
                # The cached depth was itself computed by the original full
                # EDT on this immutable context. Replace exactly that repeated
                # computation; every original validation condition is retained.
                tree=ast.parse(inspect.getsource(_validate_regions));count=0
                expected=ast.parse('organ_depth_mm(recipient.organ_mask, recipient.spacing)',mode='eval').body
                class CachedDepth(ast.NodeTransformer):
                    def visit_Call(self,node):
                        nonlocal count
                        if ast.dump(node)==ast.dump(expected):
                            count+=1;return ast.copy_location(ast.parse('recipient.organ_depth',mode='eval').body,node)
                        return self.generic_visit(node)
                tree=CachedDepth().visit(tree)
                if count!=1:raise ValueError('Exactly one original validation EDT expression required')
                namespace=dict(_validate_regions.__globals__)
                exec(compile(ast.fix_missing_locations(tree),inspect.getsourcefile(_validate_regions)+':cached_depth','exec'),namespace)
                namespace[_validate_regions.__name__](context,region)
                frozen=_AdmittedRegion(**{name:immutable_array(getattr(region,name)) for name in _ARRAY_NAMES},
                    input_sha256=canonical_hash(context.binding()))
                descriptor=tuple((name,id(getattr(frozen,name)),getattr(frozen,name).shape,
                                  getattr(frozen,name).dtype.str) for name in _ARRAY_NAMES)
                size=sum(getattr(frozen,name).nbytes for name in _ARRAY_NAMES)
                with self._lock:
                    self.guard_input_files(case)
                    identity=id(frozen)
                    def release(ref,identity=identity):
                        with self._lock:self._region_admissions.pop(identity,None)
                    self._region_admissions[identity]=(weakref.ref(frozen,release),descriptor)
                    self._static_regions[case]=(frozen,size);self._region_bytes+=size
                    self._region_profile['builds']+=1;self._region_profile['original_equation_validations']+=1
                    self._rss_locked()
                return frozen

    def _validate_admitted_regions(self,recipient,region):
        if not isinstance(region,_AdmittedRegion):
            raise ValueError('Upper runtime requires a fully validated immutable region admission')
        with self._lock:
            proof=self._region_admissions.get(id(region))
            if proof is None or proof[0]() is not region:
                raise ValueError('Unknown or evicted region admission')
            descriptor=tuple((name,id(getattr(region,name)),getattr(region,name).shape,
                              getattr(region,name).dtype.str) for name in _ARRAY_NAMES)
            if descriptor!=proof[1] or any(getattr(region,name).flags.writeable for name in _ARRAY_NAMES):
                raise ValueError('Immutable admitted region fields changed')
            if region.input_sha256!=canonical_hash(recipient.binding()):
                raise ValueError('Region admission belongs to different CT/organ/physical inputs')
            self.guard_input_files(recipient.case_id)

    def _source_only_donor(self,row):
        from hiercp_v22.data import sources
        from .v24_inputs import array_digest
        key=row['donor_case_id'],row['donor_component']
        with self._case_region_lock(('source',key)):
            if key[0] not in self.training_cases or key[0]==row['case_id']:
                raise ValueError('Fixed independent inner-train donor required')
            with self._lock:self.guard_input_files(row['case_id'],key[0])
            raw=self._case(key[0])
            with self._lock:
                if key in self._upper_sources:
                    source,size=self._upper_sources[key];self._upper_sources.move_to_end(key)
                    self._region_profile['source_only_donor_hits']+=1
                    return raw['donor_case'],source
            collection=sources(raw['donor_case'],self.config['cache']['source_pad'],self.inventory['config']['donor_max_diameter_mm'])
            match=[i for i,(component,_) in enumerate(collection.entries) if component==key[1]]
            if len(match)!=1:raise ValueError('Exact native donor component absent')
            source,_=collection[match[0]]
            source.full_mask=immutable_array(source.full_mask)
            source.v24_mask_sha256=array_digest(source.full_mask)
            size=sum(value.nbytes for value in vars(source).values() if isinstance(value,np.ndarray))
            with self._lock:
                self.guard_input_files(row['case_id'],key[0]);self._upper_sources[key]=(source,size);self._source_bytes+=size
                self._region_profile['source_only_donor_builds']+=1;self._rss_locked()
            # Original local preparation remains in the original _donor path.
            # Upper graph construction never consumes that prepared object.
            return raw['donor_case'],source

    def _upper(self,plan,provider=None):
        from .v24_geometry import build_upper_graphs
        if self._upper_runtime is None:
            source=inspect.getsource(build_upper_graphs);tree=ast.parse(source)
            calls=[node for node in ast.walk(tree) if isinstance(node,ast.Call)
                   and isinstance(node.func,ast.Name) and node.func.id=='_validate_regions']
            if len(calls)!=1:raise ValueError('Exactly one original upper region-validation call required')
            # Identical original graph bytecode, fields, candidates and layers.
            # Only the validator resolves to this object's proven immutable data.
            namespace=dict(build_upper_graphs.__globals__,_validate_regions=self._validate_admitted_regions)
            self._upper_runtime=types.FunctionType(build_upper_graphs.__code__,namespace,
                build_upper_graphs.__name__,build_upper_graphs.__defaults__,build_upper_graphs.__closure__)
            self._upper_runtime.__kwdefaults__=build_upper_graphs.__kwdefaults__
        raw=self._case(plan.case_id);donor,source=self._source_only_donor(plan.query_rows[0])
        return self._upper_runtime(raw['context'],donor,source,self._region(plan.case_id),
            self._region(plan.query_rows[0]['donor_case_id']),plan.query_rows,self.bundle.prototype_bank,
            config=self.graph_config,ct_clip=self.ct_clip,training_case_ids=self.training_cases)

    def _rss_locked(self):
        coordinator=getattr(self,'memory_coordinator',None)
        if coordinator is not None:coordinator.trim(strict=False)
        def over():return self._raw_bytes+self._region_bytes+self._source_bytes>self.raw_resident_bytes or psutil.Process().memory_info().rss>self.rss_bytes
        while over() and (self._raw_cache or self._static_regions or self._upper_sources):
            if self._raw_cache:
                case,(_,size)=self._raw_cache.popitem(last=False);self._raw_bytes-=size
                self._regions.pop(case,None)
                for key in [key for key in self._donors if key[0]==case]:self._donors.pop(key)
                for key in [key for key in self._upper_sources if key[0]==case]:
                    self._source_bytes-=self._upper_sources.pop(key)[1]
            elif self._upper_sources:
                key,(source,size)=self._upper_sources.popitem(last=False);self._source_bytes-=size
            else:
                case,(region,size)=self._static_regions.popitem(last=False);self._region_bytes-=size
                # Keep only weak lifetime admission below: active upper builders
                # may still hold an evicted region, and must remain valid.
                self._region_profile['evictions']+=1
        if psutil.Process().memory_info().rss>self.rss_bytes:
            raise MemoryError('Actual complete v24 workspace exceeds process RSS; no subset or fallback')

    def preparation_runtime_receipt(self):
        with self._lock:
            self.guard_source()
            for path,checksum in self._runtime_file_sha256.items():
                if _sha(path)!=checksum:raise ValueError('Admitted runtime source bytes changed')
            return dict(format=FORMAT,workers=4,raw_and_static_region_resident_limit_bytes=self.raw_resident_bytes,
                raw_resident_bytes=self._raw_bytes,static_region_resident_bytes=self._region_bytes,
                upper_source_resident_bytes=self._source_bytes,
                RSS_limit_bytes=self.rss_bytes,regions=len(self._static_regions),profile=dict(self._region_profile),
                original_validation_equations_before_admission=True,validation_reuses_original_immutable_EDT=True,
                unused_local_donor_preparation_for_upper=False,immutable_bytes_backed_arrays=True,
                original_upper_code_object_preserved=True,disk_region_cache=False,
                runtime_source_sha256=_sha(__file__),runtime_files_sha256=dict(self._runtime_file_sha256))

    def close(self):
        with self._lock:
            self._static_regions.clear();self._region_admissions.clear();self._region_bytes=0
            self._upper_sources.clear();self._source_bytes=0
            return super().close()


def install_runtime(factory_module=factory):
    if factory_module is not factory:raise ValueError('Exact actual v24 factory module required')
    if factory_module.V24NativeInputs not in (_ORIGINAL,AcceleratedNativeInputs):
        raise ValueError('Existing factory input constructor has an unrecognized override')
    factory_module.V24NativeInputs=AcceleratedNativeInputs
    root=Path(__file__).resolve().parent
    return dict(format=FORMAT,runtime_source_sha256=_sha(__file__),factory_constructor='AcceleratedNativeInputs',
        unchanged_scientific_files_sha256={name:_sha(root/name) for name in
            ('v24_factory.py','v24_geometry.py','v24_inputs.py','v24_model.py','v24_provider.py')},
        workers=4,raw_and_static_region_resident_limit_bytes=8*2**30,RSS_limit_bytes=64*2**30,
        original_validation_equations_before_admission=True,validation_reuses_original_immutable_EDT=True,
        unused_local_donor_preparation_for_upper=False,original_upper_code_object_preserved=True,
        per_case_inflight_region_locks=True,filesystem_region_cache=False)
