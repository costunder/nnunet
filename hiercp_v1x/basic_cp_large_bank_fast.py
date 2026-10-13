"""Full642 staged candidate preparation with exact cached complete geometry.

All frozen graph nodes, features and edges remain computed. The published512
bank precedes1024 and2048, so an independent native arm can start as soon as
its full population is ready. Already published own partial entries are copied
byte-for-byte into a fresh namespace; no old outputs are overwritten.
"""
from __future__ import annotations
import argparse,copy,json,multiprocessing,os,time,types
from concurrent.futures import FIRST_COMPLETED,ProcessPoolExecutor,wait
from pathlib import Path
import numpy as np
from . import basic_cp_large_bank as base

_PROVIDER=None;_BINDING=None;_ROOT=None;_LEASE=None;_REQUEST_IDENTITY=None
require=base.require

def verify_resources(*,force=False):
 global _LEASE,_REQUEST_IDENTITY
 if _ROOT is not None and (_ROOT/'preparation_abort.json').exists():
  record=base.read_json(_ROOT/'preparation_abort.json')
  require(record.get('format')==base.FORMAT and record.get('stop_remaining')is True,'Own abort marker differs')
  raise base.BankContractError('Own staged candidate preparation stopped: '+record['error'])
 path=Path(os.environ['BASIC_CP_LEASE_REQUEST']).absolute()
 identity=(str(path),base.stat_identity(path))
 if _LEASE is None:
  request=base.read_json(path)
  if request.get('shared_resource_observation_revision')==6:
   from .basic_cp_shared_bank_resources_v6 import SharedBankLease
  elif request.get('group_observation_revision')==5:
   from .basic_cp_shared_bank_resources_v5 import SharedBankLease
  elif request.get('shared_resource_guard_version')==3:
   from .basic_cp_shared_bank_resources_v3 import SharedBankLease
  else:
   from .basic_cp_shared_bank_resources import SharedBankLease
  _LEASE=SharedBankLease(request);_REQUEST_IDENTITY=identity
 require(identity==_REQUEST_IDENTITY,'Exact shared CPU lease request changed')
 return _LEASE.verify(force=True)if force else _LEASE.quick_verify()

def _cloned(function,**bindings):
 result=types.FunctionType(function.__code__,dict(function.__globals__,**bindings),function.__name__,function.__defaults__,function.__closure__)
 result.__kwdefaults__=function.__kwdefaults__
 return result

def _checked_write(path,value):
 verify_resources(force=True)
 return base.write_json_exclusive(path,value)

_publish_checked_prefix=_cloned(base._publish_completed_prefix,write_json_exclusive=_checked_write)

class FastFiniteLiverCenters(base.FiniteLiverCenters):
 sampler=_cloned(base.FiniteLiverCenters.sampler,verify_resources=verify_resources)

class FastGeometryProvider(base.LargeGeometryProvider):
 def load_patient(self,case_id):
  if getattr(self,'_fast_case_id',None)==case_id:
   self.verify_patient_inputs()
   return self.preseg
  from .basic_cp_large_geometry_cache import release_provider_cache
  release_provider_cache(self)
  # The base loader used to keep this old full-volume EDT/forbidden dictionary
  # until AFTER decoding the next patient. Drop every old pool/cache reference
  # before that allocation; all arrays are recomputed unchanged for the new case.
  self.patient_cache=None;self.flat_liver=None;self.patient_pool=None
  import gc
  gc.collect()
  segmentation=super().load_patient(case_id)
  from .basic_cp_large_geometry_cache import optimize_provider
  optimize_provider(self,cache_limit_bytes=512*2**20)
  self._fast_case_id=case_id
  return segmentation
 _full_proposals=_cloned(base.LargeGeometryProvider.proposals,verify_resources=verify_resources,FiniteLiverCenters=FastFiniteLiverCenters)
 def proposals(self,case_id,entry):
  # A generator starts this boundary before source.prepare allocates the next
  # complete canonical source. Patient-wide immutable geometry remains cached.
  from .basic_cp_large_geometry_cache import reset_provider_source_cache
  reset_provider_source_cache(self)
  yield from self._full_proposals(case_id,entry)

def _initialize(binding,root):
 global _PROVIDER,_BINDING,_ROOT
 require(_PROVIDER is None,'Staged worker initializes once')
 os.environ.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
 _ROOT=Path(root);_BINDING=copy.deepcopy(binding);verify_resources(force=True)
 _PROVIDER=FastGeometryProvider(binding)
 import torch
 torch.set_num_threads(1);require(not torch.cuda.is_initialized(),'Shared CPU preparation cannot initialize CUDA')

def _admit_partial(root,binding,prior_request,prior_request_sha):
 root=Path(root).absolute();request=Path(prior_request).absolute()
 require(request.resolve(strict=True)==request and base.file_sha(request)==prior_request_sha,'Exact prior producer request required')
 old=base.read_json(request)
 require(Path(old['shared_banks'])==root and old['arm']=='candidates2048','Prior producer is own exact shared bank actor')
 started=base.read_json(root/'preparation_started.json')
 require(started['format']==base.FORMAT and started['binding']==binding and started['full_sources']==642
         and started['candidate_counts']==list(base.COUNTS),'Prior complete-scale candidate preparation differs')
 import psutil
 for name in ('controller_identity',):
  ident=old[name]
  if psutil.pid_exists(ident['pid']):
   require(psutil.Process(ident['pid']).create_time()!=ident['create_time'],'Prior exact controller must have released')
 actor=Path(old['root'])/'candidate_bank_actor.json'
 if actor.exists():
  ident=base.read_json(actor)
  if psutil.pid_exists(ident['pid']):
   require(psutil.Process(ident['pid']).create_time()!=ident['birth'],'Prior exact producer must have released')
 return dict(root=str(root),request=str(request),request_sha256=prior_request_sha,
             started_sha256=base.file_sha(root/'preparation_started.json'),stopped_without_source_changes=True)

def _reuse_entry(partial,relative,count,original,admitted_prefix,destination):
 if partial is None:return False
 old_root=Path(partial['root'])/str(count);receipt=old_root/'entry_receipts'/Path(relative).with_suffix('.json')
 if not receipt.exists():return False
 old_path=base.safe_entry(old_root,relative);row=base.read_json(receipt)
 require(row['entry']==relative and base.file_sha(old_path)==row['new_entry_sha256'],'Complete own partial receipt bytes required')
 old=base.read_entry(old_path)
 base.historical.verify_preservation(original,old,original_count=128,candidate_count=count)
 n=len(admitted_prefix['candidate_centers'])
 require(row['proof']['admitted_prefix_count']==n and all(old[k][:n].tobytes()==admitted_prefix[k].tobytes()
         for k in ('candidate_centers','candidate_raw_centers')),'Original completed256 prefix changed')
 target=base.safe_entry(destination/str(count),relative);target.parent.mkdir(parents=True,exist_ok=True)
 with target.open('xb')as stream:stream.write(old_path.read_bytes());stream.flush();os.fsync(stream.fileno())
 require(base.file_sha(target)==row['new_entry_sha256'],'Reused candidate NPZ must remain byte-exact')
 row=copy.deepcopy(row);row['partial_reuse']=dict(path=str(old_path),sha256=row['new_entry_sha256'],receipt=str(receipt),receipt_sha256=base.file_sha(receipt))
 new_receipt=destination/str(count)/'entry_receipts'/Path(relative).with_suffix('.json')
 new_receipt.parent.mkdir(parents=True,exist_ok=True);base.write_json_exclusive(new_receipt,row)
 return True

def _reuse_completed_prefix(partial,count,binding,binding_path,binding_sha256,prefix256_root,destination):
 """Copy a proved full642 prefix without loading a patient or rerunning geometry."""
 if partial is None:return None
 previous=Path(partial['root'])/str(count)
 if not(previous/'complete.json').exists():return None
 witness=dict(index=str(previous/'index.json'),index_sha256=base.file_sha(previous/'index.json'),
  complete_sha256=base.file_sha(previous/'complete.json'),manifest_sha256=base.file_sha(previous/'preservation_manifest.json'),
  preparation_started_sha256=base.file_sha(previous/'preparation_started.json'))
 admission=base.verify_prepared_bank(witness,binding_path,binding_sha256,candidate_count=count)
 original_index=Path(binding['bank_index']);original=base.read_json(original_index);copied=[]
 for case,entries in original['entries_by_case'].items():
  for relative in entries:
   verify_resources(force=True)
   entry=base.read_entry(base.safe_entry(original_index.parent,relative))
   prefix=base.read_entry(base.safe_entry(Path(prefix256_root),relative))
   require(_reuse_entry(partial,relative,count,entry,prefix,destination),
           'Completed642 prefix must retain every original per-source receipt')
   copied.append(relative)
 require(len(copied)==len(set(copied))==642,'Exactly all642 previous prefix sources must be copied')
 record=dict(candidate_count=count,source_entries=642,entries=copied,previous_complete_admission=admission,
  copied_entry_bytes_verified=True,patient_CT_loaded=False,preprocessed_geometry_recomputed=False,
  full_graph_recomputed=False,original_inputs_written=False)
 base.write_json_exclusive(destination/str(count)/'complete_prefix_reuse.json',record)
 return record

def _build_stage_patient(binding,case_id,relative_entries,destination,prefix256_root,count,partial):
 require(_PROVIDER is not None and binding==_BINDING,'Exact worker binding required')
 provider=_PROVIDER;root=Path(destination);old_root=Path(binding['bank_index']).parent
 segmentation=provider.load_patient(case_id)
 for relative in relative_entries:
  verify_resources(force=True);started=time.perf_counter()
  original_path=base.safe_entry(old_root,relative);original=base.read_entry(original_path)
  old_stat,old_sha=base.stat_identity(original_path),base.file_sha(original_path)
  admitted_path=original_path if prefix256_root is None else base.safe_entry(Path(prefix256_root),relative)
  admitted=original if prefix256_root is None else base.read_entry(admitted_path)
  admitted_stat,admitted_sha=base.stat_identity(admitted_path),base.file_sha(admitted_path)
  previous={512:None,1024:512,2048:1024}[count]
  prefix_path=admitted_path if previous is None else base.safe_entry(root/str(previous),relative)
  prefix=admitted if previous is None else base.read_entry(prefix_path)
  prefix_sha=base.file_sha(prefix_path)
  require(all(base.historical.preprocessed_valid(c,original['source_mask'],original['anchor_offset'],segmentation,
              liver_label=1,tumor_label=2,minimum_coverage=.85)for c in prefix['candidate_centers']),
          'Full current prefix actual CP geometry required')
  if _reuse_entry(partial,relative,count,original,admitted,root):
   result=base.read_entry(base.safe_entry(root/str(count),relative))
   require(all(result[k][:len(prefix[k])].tobytes()==prefix[k].tobytes()for k in ('candidate_centers','candidate_raw_centers')),
           'Reused partial must preserve current nested prefix')
   provider.verify_patient_inputs();continue
  def publish(actual,result,proof):
   require(actual==count,'Exact staged count required');verify_resources(force=True)
   target=base.safe_entry(root/str(count),relative);target.parent.mkdir(parents=True,exist_ok=True)
   with target.open('xb')as stream:np.savez(stream,**result);stream.flush();os.fsync(stream.fileno())
   base.historical.verify_preservation(original,base.read_entry(target),original_count=128,candidate_count=count)
   require(base.stat_identity(original_path)==old_stat and base.file_sha(original_path)==old_sha
       and base.stat_identity(admitted_path)==admitted_stat and base.file_sha(admitted_path)==admitted_sha
       and base.file_sha(prefix_path)==prefix_sha,'Actual source/admitted/generation prefix changed')
   n=len(admitted['candidate_centers'])
   require(all(result[k][:n].tobytes()==admitted[k].tobytes()for k in ('candidate_centers','candidate_raw_centers')),
           'Original completed256 prefix must remain exact')
   proof.update(admitted_prefix_count=n,all_admitted_prefix_centers_byte_preserved=True)
   row=dict(entry=relative,case_id=case_id,original_stat=old_stat,original_entry_sha256=old_sha,
     new_entry_sha256=base.file_sha(target),proof=proof,admitted_prefix=dict(path=str(admitted_path),sha256=admitted_sha,stat=admitted_stat),
     generation_prefix=dict(path=str(prefix_path),sha256=prefix_sha,count=len(prefix['candidate_centers'])),
     proposal_diagnostics=copy.deepcopy(provider.last_proposal_diagnostics),seconds=time.perf_counter()-started,
     full_graph_geometry_cache=True)
   receipt=root/str(count)/'entry_receipts'/Path(relative).with_suffix('.json');receipt.parent.mkdir(parents=True,exist_ok=True)
   base.write_json_exclusive(receipt,row);provider.verify_patient_inputs()
   print(f'[Basic{count}] {case_id} {relative} full{count} historical128 preserved',flush=True)
  base.collect_prefixes(original,prefix,provider.proposals(case_id,prefix),targets=(count,),publish=publish)
 import torch
 require(not torch.cuda.is_initialized(),'Shared candidate preparation cannot initialize CUDA')
 return dict(case_id=case_id,source_entries=list(relative_entries))

def build_staged_banks(binding,destination,*,workers,rss_limit_gib,prepared256_witness,binding_path,binding_sha256,
                       reuse_partial_from=None,reuse_request=None,reuse_request_sha256=None):
 global _ROOT
 verify_resources(force=True);original_index=Path(binding['bank_index']).resolve(strict=True)
 base.historical.validate_original_index(base.read_json(original_index))
 require(base.file_sha(original_index)==base.historical.BANK_SHA256,'Original index SHA required')
 affinity=base.historical.cpu_affinity();require(type(workers)is int and 2<=workers<=len(affinity),'Measured parallel CPU workers required')
 requested=base.read_json(Path(os.environ['BASIC_CP_LEASE_REQUEST']))
 admitted_rss=384 if requested.get('shared_resource_guard_version')==3 else 288
 require(requested.get('shared_bank_RSS_limit_GiB')==admitted_rss and rss_limit_gib==admitted_rss,'Exact measured shared preparation budget required');rss_limit=int(rss_limit_gib*2**30)
 admission=base.historical.verify_prepared_bank(prepared256_witness,binding_path,binding_sha256)
 prefix256_root=str(Path(admission['index']).parent)
 partial=None
 if reuse_partial_from is not None:partial=_admit_partial(reuse_partial_from,binding,reuse_request,reuse_request_sha256)
 root=Path(destination).resolve();require(root!=original_index.parent and original_index.parent not in root.parents,'Fresh bank namespace required')
 root.mkdir(parents=True,exist_ok=False);_ROOT=root
 started=dict(format=base.FORMAT,binding=copy.deepcopy(binding),original_index_sha256=base.historical.BANK_SHA256,
  full_sources=642,eligible_cases=81,original_candidate_count=128,candidate_counts=list(base.COUNTS),candidate_count=2048,
  additional_candidate_search=base.SEARCH,CUDA_model_or_scoring_executed=False,started_at=time.time(),prepared256_reuse_admission=admission,
  workers=workers,CPU_affinity=affinity,RSS_limit_bytes=rss_limit,full_graph_geometry_cache=True,staged_full_population_order=list(base.COUNTS),
  partial_reuse_admission=partial,source_chunk_size=4,source_chunk_is_scheduling_only=True)
 base.write_json_exclusive(root/'preparation_started.json',started)
 for count in base.COUNTS:
  (root/str(count)/'entry_receipts').mkdir(parents=True);base.write_json_exclusive(root/str(count)/'preparation_started.json',dict(started,candidate_count=count))
 progress=dict(format=base.FORMAT,status='PREPARING',source_total=642,case_total=81,case_completed=0,source_completed=0,
  prefix_completed={str(c):0 for c in base.COUNTS},candidate_count=2048,workers=workers,started_at=started['started_at'],
  updated_at=time.time(),raw_CT_preprocessing_executed=False,GNN_scoring_executed=False,staged_full_population_order=list(base.COUNTS))
 base.historical.publish_progress(root,progress);peak=base.historical.process_tree_rss();completed={}
 try:
  with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),initializer=_initialize,initargs=(binding,str(root)))as pool:
   for count in base.COUNTS:
    progress.update(active_candidate_count=count,case_completed=0)
    reused=_reuse_completed_prefix(partial,count,binding,binding_path,binding_sha256,Path(prefix256_root),root)
    if reused is not None:
     verify_resources(force=True)
     rss=base.historical.process_tree_rss();peak=max(peak,rss)
     receipt=_publish_checked_prefix(root,count,binding,started,peak_rss=peak,workers=workers,affinity=affinity,rss_limit=rss_limit)
     require(receipt is not None,'Copied completed prefix must publish every642 source before proceeding')
     completed[str(count)]=receipt;progress['prefix_completed'][str(count)]=642
     progress.update(case_completed=81,RSS_bytes=rss,peak_RSS_bytes=peak,updated_at=time.time())
     base.historical.publish_progress(root,progress)
     print(f'BASIC_CP_FULL642_PREFIX_BYTE_REUSED {count}',flush=True)
     continue
    cases=base.read_json(original_index)['entries_by_case'];finished_sources={case:set()for case in cases}
    jobs=[(case,entries[start:start+4])for case,entries in sorted(cases.items(),key=lambda row:(-len(row[1]),row[0]))
          for start in range(0,len(entries),4)]
    require(sum(len(entries)for _,entries in jobs)==642,'Scheduling must include all642 sources')
    futures=set(pool.submit(_build_stage_patient,binding,case,entries,str(root),prefix256_root,count,partial)
                for case,entries in jobs)
    try:
     while futures:
      verify_resources(force=True);done,futures=wait(futures,timeout=2,return_when=FIRST_COMPLETED)
      rss=base.historical.process_tree_rss();peak=max(peak,rss);require(rss<=rss_limit,'Measured shared preparation exceeds its admitted budget; no count/source reduction')
      for future in done:
       row=future.result();finished_sources[row['case_id']].update(row['source_entries'])
      progress['case_completed']=sum(len(finished_sources[case])==len(entries)for case,entries in cases.items())
      for target in base.COUNTS:progress['prefix_completed'][str(target)]=sum(1 for _ in(root/str(target)/'entry_receipts').rglob('*.json'))
      progress.update(source_completed=progress['prefix_completed']['2048'],RSS_bytes=rss,peak_RSS_bytes=peak,updated_at=time.time())
      base.historical.publish_progress(root,progress)
    except Exception as error:
     base._failure(root,progress,error)
     for future in futures:future.cancel()
     raise
    verify_resources(force=True)
    receipt=_publish_checked_prefix(root,count,binding,started,peak_rss=peak,workers=workers,affinity=affinity,rss_limit=rss_limit)
    require(receipt is not None,'Every642 source must complete before publishing staged bank');completed[str(count)]=receipt
    print(f'BASIC_CP_FULL642_PREFIX_READY {count}',flush=True)
  result=dict(format=base.FORMAT,complete=True,prefixes=completed,full_sources=642,additional_candidate_search=base.SEARCH,
   peak_RSS_bytes=peak,CUDA_initialized=False,training_executed=False,full_graph_geometry_cache=True)
  base.write_json_exclusive(root/'complete.json',result);progress.update(status='COMPLETE',updated_at=time.time());base.historical.publish_progress(root,progress)
  return result
 except Exception as error:base._failure(root,progress,error);raise

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--binding',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
 parser.add_argument('--workers',type=int,required=True);parser.add_argument('--rss-limit-gib',type=float,required=True)
 parser.add_argument('--prepared256-witness',type=Path,required=True);parser.add_argument('--reuse-partial-from',type=Path)
 parser.add_argument('--reuse-request',type=Path);parser.add_argument('--reuse-request-sha256')
 args=parser.parse_args();binding=base.read_json(args.binding)
 build_staged_banks(binding,args.output,workers=args.workers,rss_limit_gib=args.rss_limit_gib,
  prepared256_witness=base.read_json(args.prepared256_witness),binding_path=args.binding,binding_sha256=base.file_sha(args.binding),
  reuse_partial_from=args.reuse_partial_from,reuse_request=args.reuse_request,reuse_request_sha256=args.reuse_request_sha256)

if __name__=='__main__':main()
