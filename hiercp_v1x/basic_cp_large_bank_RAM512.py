"""Full642 recovery provider under the explicit total512GiB operational guard.

The historical source request remains immutable96/384; only the actual source
process ceiling becomes320. No graph, search, source, candidate, model or worker
count changes. Every scientific function is a private bytecode clone; imported
canonical modules and their mutable runtime state are never patched.
"""
from __future__ import annotations
import copy as _copy
import inspect as _inspect
import types as _types
from pathlib import Path as _Path
from . import basic_cp_large_bank_recovery as _canonical

PROVIDER_FILE='hiercp_v1x/basic_cp_large_bank_RAM512.py'
RESOURCE_FILE='hiercp_v1x/basic_cp_512_resources.py'
EFFECTIVE_SOURCE_GiB=320
FULL_WORKERS=24

# Preserve external helper globals (e.g. futures.wait and original require).
# Only functions using recovery's exact state namespace receive this module's
# private state; nested external scientific scopes are cloned explicitly below.
for _name,_value in vars(_canonical).items():
 if not _name.startswith('__'):globals()[_name]=_value
_PROVIDER=None;_BINDING=None;_ROOT=None;_LEASE=None;_REQUEST_IDENTITY=None
_RECOVERY_REF=None;_SNAPSHOTS={}

def _rebind(function,scope,*,code=None):
 result=_types.FunctionType(function.__code__ if code is None else code,scope,
     function.__name__,function.__defaults__,function.__closure__)
 result.__kwdefaults__=_copy.deepcopy(function.__kwdefaults__)
 return result

for _name,_value in vars(_canonical).items():
 if isinstance(_value,_types.FunctionType)and _value.__globals__ is vars(_canonical):
  globals()[_name]=_rebind(_value,globals())

def _validate_512_request(request):
 expected=dict(memory_budget_revision=512,aggregate_memory_cap_GiB=512,
     effective_native_RSS_limit_GiB=32,effective_shared_bank_RSS_limit_GiB=320,
     owned_tmpfs_envelope_GiB=144,post_source_owned_tmpfs_envelope_GiB=272,
     auxiliary_RAM_envelope_GiB=16)
 require(all(type(request.get(key))is int and request[key]==value for key,value in expected.items()),
     'Explicit reviewed144/320/32/16 total512 operational policy required')
 require({PROVIDER_FILE,RESOURCE_FILE}<=set(request.get('code_sha256',{})),
     'Both exact new512 provider/resource source SHAs required')
 require(all(type(request.get(key))is int and request[key]==value for key,value in
     dict(shared_resource_guard_version=3,shared_bank_RSS_limit_GiB=384,
          RSS_limit_bytes=96*2**30,shared_bank_workers=FULL_WORKERS).items()),
     'Immutable declared96/384 and full24-worker source contract required')
 return request

def _shared_lease_factory(request):
 _validate_512_request(request)
 from .basic_cp_512_resources import SharedBankLease
 return SharedBankLease(request)

# Replace exactly the old resource-class selector, retaining its abort/request
# identity/force/quick-verification statements unchanged. It creates no claim.
_verify_source=_inspect.getsource(_canonical.strict.verify_resources)
_selector="""  if 'RAM_admission_pause_revision'in request:
   from .basic_cp_RAM_admission_pause import SharedBankLease
  elif request.get('shared_resource_observation_revision')==6:
   from .basic_cp_shared_bank_resources_v6 import SharedBankLease
  elif request.get('group_observation_revision')==5:
   from .basic_cp_shared_bank_resources_v5 import SharedBankLease
  elif request.get('shared_resource_guard_version')==3:
   from .basic_cp_shared_bank_resources_v3 import SharedBankLease
  else:
   from .basic_cp_shared_bank_resources import SharedBankLease"""
require(_verify_source.count(_selector)==1,'Exact immutable old shared-lease selector required')
exec(compile(_verify_source.replace(_selector,'  SharedBankLease=_shared_lease_factory'),__file__,'exec'),globals())

# Private atomic-I/O namespace; canonical.base and historical functions intact.
base=_types.SimpleNamespace(**vars(_canonical.base))
base.write_json_exclusive=_atomic_json

class FastFiniteLiverCenters(_canonical.original_base.FiniteLiverCenters):
 sampler=_rebind(_canonical.FastFiniteLiverCenters.sampler,
     dict(_canonical.FastFiniteLiverCenters.sampler.__globals__,verify_resources=verify_resources))

class FastGeometryProvider(_canonical.FastGeometryProvider):
 _full_proposals=_rebind(_canonical.FastGeometryProvider._full_proposals,
     dict(_canonical.FastGeometryProvider._full_proposals.__globals__,
          verify_resources=verify_resources,FiniteLiverCenters=FastFiniteLiverCenters))

_publish_checked_prefix=_rebind(_canonical._publish_checked_prefix,
    dict(_canonical._publish_checked_prefix.__globals__,write_json_exclusive=_checked_write))
_reuse_completed_prefix=_rebind(_canonical._reuse_completed_prefix,
    dict(_canonical._reuse_completed_prefix.__globals__,base=base,
         verify_resources=verify_resources,_reuse_entry=_reuse_entry))

def _source_execution_view(request):
 _validate_512_request(request)
 result=_copy.deepcopy(request)
 result['shared_bank_RSS_limit_GiB']=EFFECTIVE_SOURCE_GiB
 return result

def _build_read_json(path):
 result=base.read_json(path)
 current=_Path(os.environ['BASIC_CP_LEASE_REQUEST']).absolute()
 if _Path(path).absolute()==current:return _source_execution_view(result)
 return result

# Exact same build bytecode; only its single operational384 constant changes.
# A build-only request read overlays the effective ceiling. Guards and prior
# source/binding/snapshot readers continue to read the original actual bytes.
_old_build_code=_canonical._build.__code__
require(sum(type(value)is int and value==384 for value in _old_build_code.co_consts)==1,
        'Exact single operational384GiB build constant required')
_new_build_code=_old_build_code.replace(co_consts=tuple(
    EFFECTIVE_SOURCE_GiB if type(value)is int and value==384 else value
    for value in _old_build_code.co_consts))
_build_base=_types.SimpleNamespace(**vars(base));_build_base.read_json=_build_read_json
_build_scope=dict(globals(),base=_build_base)
_build=_rebind(_canonical._build,_build_scope,code=_new_build_code)

def build_staged_banks(binding,destination,*,recovery_snapshot,recovery_snapshot_sha256,**kwargs):
 global _ROOT,_RECOVERY_REF
 request=base.read_json(_Path(os.environ['BASIC_CP_LEASE_REQUEST']))
 _validate_512_request(request)
 require(kwargs.get('workers')==FULL_WORKERS and kwargs.get('rss_limit_gib')==EFFECTIVE_SOURCE_GiB,
     'Full24 workers and explicit320GiB source operation required')
 _ROOT=_Path(destination).absolute()
 _RECOVERY_REF=(str(_Path(recovery_snapshot).absolute()),recovery_snapshot_sha256)
 _snapshot(*_RECOVERY_REF)
 verify_resources(force=True) # Bind/validate already-published source claim before any pool creation.
 return _build(binding,destination,**kwargs)

__all__=['build_staged_banks','main','PROVIDER_FILE','RESOURCE_FILE','FULL_WORKERS','EFFECTIVE_SOURCE_GiB']

if __name__=='__main__':main()
