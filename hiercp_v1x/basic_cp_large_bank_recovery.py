"""Explicit interrupted-publication recovery in a fresh full642 bank namespace.

Default historical/fast readers remain strict. Only an immutable, SHA-pinned
post-release snapshot may authorize recomputing an individually interrupted
publication. Certified completed bytes changing later remain fatal. All sources,
counts and frozen geometry predicates remain unchanged. New publications are
fsynced and atomically promoted without overwriting any existing output.
"""
from __future__ import annotations
import argparse
import copy
import gc
import inspect
import json
import multiprocessing
import os
from pathlib import Path
import tempfile
import time
import types
import zipfile
from concurrent.futures import FIRST_COMPLETED,ProcessPoolExecutor,wait
import numpy as np
from . import basic_cp_large_bank as original_base
from . import basic_cp_large_bank_fast as strict

FORMAT='basic_cp_exact_interrupted_publication_snapshot_v1'
UID=41833
require=original_base.require
_PROVIDER=None;_BINDING=None;_ROOT=None;_LEASE=None;_REQUEST_IDENTITY=None
_RECOVERY_REF=None;_SNAPSHOTS={}

def _clone(function,**bindings):
 scope=dict(function.__globals__,**bindings)
 result=types.FunctionType(function.__code__,scope,function.__name__,function.__defaults__,function.__closure__)
 result.__kwdefaults__=copy.deepcopy(function.__kwdefaults__)
 return result

def _promote(temporary,target):
 # link+unlink is an atomic no-clobber promotion; unlike replace, an unexpected
 # existing published path cannot be overwritten even during a race.
 if os.name=='nt':
  # Windows rename atomically refuses an existing destination; directoryfsync
  # is unsupported there. Clinical production uses the Linux branch below.
  os.rename(temporary,target);return
 os.link(temporary,target,follow_symlinks=False);temporary.unlink()
 fd=os.open(target.parent,os.O_RDONLY)
 try:os.fsync(fd)
 finally:os.close(fd)

def _atomic_json(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 temporary=path.with_name('.'+path.name+'.atomic_'+str(time.time_ns())+'.tmp')
 try:
  with temporary.open('x',encoding='utf8')as stream:
   json.dump(value,stream,ensure_ascii=False,allow_nan=False);stream.flush();os.fsync(stream.fileno())
  _promote(temporary,path)
 finally:
  if temporary.exists():temporary.unlink() # Only this call's exclusive temporary.

def _atomic_npz(path,value):
 path=Path(path);temporary=path.with_name('.'+path.name+'.atomic_'+str(time.time_ns())+'.tmp')
 try:
  with temporary.open('xb')as stream:
   np.savez(stream,**value);stream.flush();os.fsync(stream.fileno())
  _promote(temporary,path)
 finally:
  if temporary.exists():temporary.unlink()

def _atomic_copy(source,target):
 target=Path(target);target.parent.mkdir(parents=True,exist_ok=True)
 temporary=target.with_name('.'+target.name+'.atomic_'+str(time.time_ns())+'.tmp')
 try:
  with Path(source).open('rb')as incoming,temporary.open('xb')as outgoing:
   for block in iter(lambda:incoming.read(8*2**20),b''):outgoing.write(block)
   outgoing.flush();os.fsync(outgoing.fileno())
  _promote(temporary,target)
 finally:
  if temporary.exists():temporary.unlink()

# A private I/O view, never a mutation of either imported historical module.
base=types.SimpleNamespace(**vars(original_base));base.write_json_exclusive=_atomic_json
verify_resources=types.FunctionType(strict.verify_resources.__code__,globals(),'verify_resources',strict.verify_resources.__defaults__)
verify_resources.__kwdefaults__=copy.deepcopy(strict.verify_resources.__kwdefaults__)

def _asset(path):
 path=Path(path)
 if not path.exists():return None
 require(path.is_absolute()and path.resolve(strict=True)==path and path.is_file()and not any(p.is_symlink()for p in(path,*path.parents)),'Exact regular publication evidence required')
 before=base.stat_identity(path);digest=base.file_sha(path)
 if os.name!='nt':require(path.stat().st_uid==UID,'Publication evidence must belong to the original account')
 require(base.stat_identity(path)==before,'Publication changed during snapshot')
 return dict(path=str(path),stat=before,sha256=digest)

def _unchanged(asset):
 if asset is not None:
  path=Path(asset['path']);require(path.exists()and base.stat_identity(path)==asset['stat']and base.file_sha(path)==asset['sha256'],'Certified interrupted-source publication changed after snapshot')

class InterruptedPublication(ValueError):pass

def _classify(root,count,relative,case,original_path,admitted_path):
 root=Path(root);archive=base.safe_entry(root/str(count),relative)
 receipt=root/str(count)/'entry_receipts'/Path(relative).with_suffix('.json')
 assets=dict(archive=_asset(archive),receipt=_asset(receipt))
 result=dict(entry=relative,case_id=case,candidate_count=count,assets=assets)
 if assets['receipt']is None:
  return dict(result,status='UNPUBLISHED',reason='No complete receipt was published; any archive is preserved but not admitted')
 try:
  try:row=base.read_json(receipt)
  except(json.JSONDecodeError,UnicodeDecodeError)as error:raise InterruptedPublication('Malformed interrupted receipt: '+str(error))from error
  if not isinstance(row,dict)or not {'entry','case_id','new_entry_sha256','original_entry_sha256','proof','admitted_prefix','generation_prefix'}<=set(row):
   raise InterruptedPublication('Incomplete receipt fields')
  require(row['entry']==relative and row['case_id']==case,'Completed receipt changed source identity')
  if assets['archive']is None:raise InterruptedPublication('Receipt exists without its completed archive')
  if row['new_entry_sha256']!=assets['archive']['sha256']:raise InterruptedPublication('Interrupted archive SHA does not match its receipt')
  try:data=base.read_entry(archive)
  except(zipfile.BadZipFile,EOFError,OSError,ValueError)as error:raise InterruptedPublication('Incomplete NPZ publication: '+str(error))from error
  original,admitted=base.read_entry(original_path),base.read_entry(admitted_path)
  require(row['original_entry_sha256']==base.file_sha(original_path),'Immutable original128 source SHA differs')
  proof=base.historical.verify_preservation(original,data,original_count=128,candidate_count=count)
  require(all(row['proof'].get(key)==value for key,value in proof.items()),'Completed publication original128 proof differs')
  n=len(admitted['candidate_centers']);prefix_info=row['admitted_prefix']
  require(prefix_info['path']==str(admitted_path)and prefix_info['sha256']==base.file_sha(admitted_path),'Immutable admitted256 prefix differs')
  require(row['proof'].get('admitted_prefix_count')==n and all(data[key][:n].tobytes()==admitted[key].tobytes()for key in('candidate_centers','candidate_raw_centers')),'Completed admitted256 bytes differ')
  previous={512:None,1024:512,2048:1024}[count]
  current_prefix=admitted_path if previous is None else base.safe_entry(root/str(previous),relative)
  generation=row['generation_prefix'];referenced=Path(generation['path'])
  require(referenced.is_absolute()and referenced.exists()and base.file_sha(referenced)==generation['sha256'],'Immutable generation-prefix witness differs')
  require(current_prefix.exists(),'Completed publication lacks its nested current prefix')
  current,referenced_data=base.read_entry(current_prefix),base.read_entry(referenced)
  length=len(current['candidate_centers'])
  require(generation['count']==length and all(current[key].tobytes()==referenced_data[key].tobytes()==data[key][:length].tobytes()for key in('candidate_centers','candidate_raw_centers')),'Completed current nested prefix bytes differ')
  require(row.get('full_graph_geometry_cache')is True,'Actual frozen full-graph generator receipt required')
  for asset in assets.values():_unchanged(asset)
  return dict(result,status='VALID',reason='Complete owned receipt/NPZ SHA and original128/admitted256/current nested prefixes verified')
 except InterruptedPublication as error:
  for asset in assets.values():_unchanged(asset)
  return dict(result,status='INTERRUPTED',reason=str(error))

def _released(identity):
 import psutil
 try:process=psutil.Process(identity['pid'])
 except psutil.NoSuchProcess:return True
 if process.create_time()!=identity.get('birth',identity.get('create_time')):return True
 require(process.uids().real==identity['UID']==UID,'Original surviving publication actor UID differs')
 return process.status()in(psutil.STATUS_ZOMBIE,psutil.STATUS_DEAD)

def snapshot_interrupted_outputs(binding_path,binding_sha256,prior_request,prior_request_sha256,prepared256_witness,output,*,released_actors):
 """After exact source release, certify every full642×3 publication slot."""
 binding_path,prior_request,output=map(Path,(binding_path,prior_request,output))
 require(os.getuid()==UID and base.file_sha(binding_path)==binding_sha256 and base.file_sha(prior_request)==prior_request_sha256,'Exact owned source/binding admissions required')
 request=base.read_json(prior_request);binding=base.read_json(binding_path)
 require(request['candidate_binding']==str(binding_path)and request['candidate_binding_sha256']==binding_sha256,'Previous producer binding differs')
 root=Path(request['shared_banks']);partial=strict._admit_partial(root,binding,prior_request,prior_request_sha256)
 require(len(released_actors)==24 and len({(i['pid'],i.get('birth',i.get('create_time')))for i in released_actors})==24 and all(i['UID']==UID and '--multiprocessing-fork'in i['command']and _released(i)for i in released_actors),'Exactly all24 captured own source workers must have released')
 original_index=Path(binding['bank_index']);original=base.read_json(original_index)
 base.historical.validate_original_index(original);require(base.file_sha(original_index)==base.historical.BANK_SHA256,'Canonical full original128 source index required')
 if isinstance(prepared256_witness,(str,Path)):prepared256_witness=base.read_json(Path(prepared256_witness))
 admission=base.historical.verify_prepared_bank(prepared256_witness,binding_path,binding_sha256)
 admitted_root=Path(admission['index']).parent
 names={name:case for case,entries in original['entries_by_case'].items()for name in entries}
 require(len(names)==642 and len(original['entries_by_case'])==81,'Full642/81 production source contract required')
 completed={};sealed_assets=[]
 for count in base.COUNTS:
  prefix=root/str(count)
  if(prefix/'complete.json').exists():
   witness=dict(index=str(prefix/'index.json'),index_sha256=base.file_sha(prefix/'index.json'),complete_sha256=base.file_sha(prefix/'complete.json'),manifest_sha256=base.file_sha(prefix/'preservation_manifest.json'),preparation_started_sha256=base.file_sha(prefix/'preparation_started.json'))
   completed[str(count)]=base.verify_prepared_bank(witness,binding_path,binding_sha256,candidate_count=count)
   sealed_assets.extend(_asset(prefix/name)for name in('index.json','complete.json','preservation_manifest.json','preparation_started.json'))
 records={};counts={}
 for count in base.COUNTS:
  rows={relative:_classify(root,count,relative,case,base.safe_entry(original_index.parent,relative),base.safe_entry(admitted_root,relative))for relative,case in sorted(names.items())}
  records[str(count)]=rows;counts[str(count)]={status:sum(row['status']==status for row in rows.values())for status in('VALID','INTERRUPTED','UNPUBLISHED')}
  if str(count)in completed:require(counts[str(count)]['VALID']==642,'A sealed full prefix can never be downgraded to interrupted')
  expected={str((root/str(count)/'entry_receipts'/Path(relative).with_suffix('.json')))for relative in names}
  require(all(str(path)in expected for path in(root/str(count)/'entry_receipts').rglob('*.json')),'Unexpected source identity in old publication receipts')
 result=dict(format=FORMAT,source_root=str(root),prior_request=str(prior_request),prior_request_sha256=prior_request_sha256,binding_path=str(binding_path),binding_sha256=binding_sha256,original_index_sha256=base.historical.BANK_SHA256,full_sources=642,eligible_cases=81,candidate_counts=list(base.COUNTS),partial_admission=partial,prepared256_reuse_admission=admission,completed_prefixes=completed,sealed_metadata_assets=sealed_assets,records=records,counts=counts,released_actors=released_actors,created_at=time.time(),old_outputs_written=False,full_geometry_rules_unchanged=True,recompute_interrupted_full_sources_not_subset=True)
 _atomic_json(output,result);return result

def _snapshot(path,digest):
 path=Path(path);require(base.file_sha(path)==digest,'Recovery snapshot SHA changed')
 key=(str(path),digest)
 if key not in _SNAPSHOTS:
  value=base.read_json(path)
  require(value['format']==FORMAT and value['full_sources']==642 and value['eligible_cases']==81 and value['candidate_counts']==list(base.COUNTS)and value['old_outputs_written']is False,'Exact full-scale interrupted snapshot required')
  require(all(len(value['records'][str(count)])==642 for count in base.COUNTS),'Snapshot must cover every original source at all three counts')
  _SNAPSHOTS[key]=value
 return _SNAPSHOTS[key]

def _admit_partial(root,binding,prior_request,prior_request_sha):
 partial=strict._admit_partial(root,binding,prior_request,prior_request_sha)
 require(_RECOVERY_REF is not None,'Explicit snapshot admission required for recovery builder')
 snapshot=_snapshot(*_RECOVERY_REF)
 for asset in snapshot['sealed_metadata_assets']:_unchanged(asset)
 require(snapshot['source_root']==str(Path(root).absolute())and snapshot['prior_request']==str(Path(prior_request).absolute())and snapshot['prior_request_sha256']==prior_request_sha and snapshot['binding_sha256']==base.file_sha(snapshot['binding_path'])and base.read_json(snapshot['binding_path'])==binding,'Recovery snapshot belongs to a different exact prior producer')
 return dict(partial,recovery_snapshot=str(_RECOVERY_REF[0]),recovery_snapshot_sha256=_RECOVERY_REF[1])

def _reuse_entry(partial,relative,count,original,admitted_prefix,destination):
 if partial is None:return False
 require('recovery_snapshot'in partial and 'recovery_snapshot_sha256'in partial,'No broad recovery fallback without explicit snapshot')
 snapshot=_snapshot(partial['recovery_snapshot'],partial['recovery_snapshot_sha256']);row=snapshot['records'][str(count)][relative]
 for asset in row['assets'].values():_unchanged(asset)
 if row['status']in('INTERRUPTED','UNPUBLISHED'):return False
 require(row['status']=='VALID','Unknown recovery publication classification')
 old_path=Path(row['assets']['archive']['path']);receipt=Path(row['assets']['receipt']['path']);record=base.read_json(receipt);data=base.read_entry(old_path)
 base.historical.verify_preservation(original,data,original_count=128,candidate_count=count)
 n=len(admitted_prefix['candidate_centers']);require(record['proof']['admitted_prefix_count']==n and all(data[key][:n].tobytes()==admitted_prefix[key].tobytes()for key in('candidate_centers','candidate_raw_centers')),'Certified admitted256 prefix changed')
 target=base.safe_entry(Path(destination)/str(count),relative);_atomic_copy(old_path,target)
 require(base.file_sha(target)==record['new_entry_sha256'],'Copied certified source bytes changed')
 record=copy.deepcopy(record);record['partial_reuse']=dict(path=str(old_path),sha256=record['new_entry_sha256'],receipt=str(receipt),receipt_sha256=base.file_sha(receipt))
 _atomic_json(Path(destination)/str(count)/'entry_receipts'/Path(relative).with_suffix('.json'),record)
 return True

class FastFiniteLiverCenters(original_base.FiniteLiverCenters):
 sampler=_clone(original_base.FiniteLiverCenters.sampler,verify_resources=verify_resources)

class FastGeometryProvider(strict.FastGeometryProvider):
 _full_proposals=_clone(original_base.LargeGeometryProvider.proposals,verify_resources=verify_resources,FiniteLiverCenters=FastFiniteLiverCenters)

def _initialize(binding,root):
 global _PROVIDER,_BINDING,_ROOT
 require(_PROVIDER is None,'Recovery worker initializes once');_ROOT=Path(root);_BINDING=copy.deepcopy(binding)
 os.environ.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
 verify_resources(force=True);_PROVIDER=FastGeometryProvider(binding)
 import torch
 torch.set_num_threads(1);require(not torch.cuda.is_initialized(),'Recovery preparation remains CPU-only')

# Retain all existing scientific statements; replace only the exact output I/O.
_stage_source=inspect.getsource(strict._build_stage_patient)
_old_io="with target.open('xb')as stream:np.savez(stream,**result);stream.flush();os.fsync(stream.fileno())"
require(_stage_source.count(_old_io)==1,'Exact frozen stage I/O source required')
exec(compile(_stage_source.replace(_old_io,'_atomic_npz(target,result)'),__file__,'exec'),globals())

def _checked_write(path,value):verify_resources(force=True);return _atomic_json(path,value)
_publish_checked_prefix=_clone(original_base._publish_completed_prefix,write_json_exclusive=_checked_write)
# Retain the full642 SHA/array copy statements and existing build boundary
# force checks. Each source polls the same admitted quick guard, whose normal
# interval is5seconds, instead of rescanning all seven process trees per source.
_prefix_source=inspect.getsource(strict._reuse_completed_prefix)
_prefix_force='verify_resources(force=True)'
require(_prefix_source.count(_prefix_force)==1,'Exact single completed-prefix per-source forced guard required')
_prefix_scope=dict(strict._reuse_completed_prefix.__globals__,base=base,verify_resources=verify_resources,_reuse_entry=_reuse_entry)
exec(compile(_prefix_source.replace(_prefix_force,'verify_resources()'),__file__,'exec'),_prefix_scope)
_reuse_completed_prefix=_prefix_scope['_reuse_completed_prefix']
_build=types.FunctionType(strict.build_staged_banks.__code__,globals(),'_build',strict.build_staged_banks.__defaults__)
_build.__kwdefaults__=copy.deepcopy(strict.build_staged_banks.__kwdefaults__)

def build_staged_banks(binding,destination,*,recovery_snapshot,recovery_snapshot_sha256,**kwargs):
 global _RECOVERY_REF
 _RECOVERY_REF=(str(Path(recovery_snapshot).absolute()),recovery_snapshot_sha256)
 _snapshot(*_RECOVERY_REF)
 return _build(binding,destination,**kwargs)

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 for name in('binding','output','prepared256-witness','reuse-partial-from','reuse-request','recovery-snapshot'):parser.add_argument('--'+name,type=Path,required=True)
 parser.add_argument('--workers',type=int,required=True);parser.add_argument('--rss-limit-gib',type=float,required=True)
 parser.add_argument('--reuse-request-sha256',required=True);parser.add_argument('--recovery-snapshot-sha256',required=True)
 args=parser.parse_args();request=base.read_json(Path(os.environ['BASIC_CP_LEASE_REQUEST']))
 require(type(request.get('interrupted_candidate_recovery_revision'))is int and request['interrupted_candidate_recovery_revision']==1 and request['interrupted_candidate_recovery_snapshot']==str(args.recovery_snapshot)and request['interrupted_candidate_recovery_snapshot_sha256']==args.recovery_snapshot_sha256,'Explicit request-pinned interrupted recovery required')
 build_staged_banks(base.read_json(args.binding),args.output,workers=args.workers,rss_limit_gib=args.rss_limit_gib,prepared256_witness=base.read_json(args.prepared256_witness),binding_path=args.binding,binding_sha256=base.file_sha(args.binding),reuse_partial_from=args.reuse_partial_from,reuse_request=args.reuse_request,reuse_request_sha256=args.reuse_request_sha256,recovery_snapshot=args.recovery_snapshot,recovery_snapshot_sha256=args.recovery_snapshot_sha256)

if __name__=='__main__':main()
