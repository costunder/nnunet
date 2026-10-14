"""PREPARED: exact own finalized native6 release and reversible partial output preservation.

No action on import. Inspection is read-only. Release sends at most one SIGINT
to the exact reported own validation trainer. Preservation requires every
captured trainer/controller/export actor terminal and original FINAL250/BEST
whole-state SHA proof. No checkpoint, original prediction byte, or receipt is
overwritten or removed. New validation will evaluate the full26 again.
"""
import argparse,hashlib,importlib.util,json,os,signal,socket,time
from pathlib import Path
import psutil

BASE=Path('/dev/shm/aicompetition06_basic_cp_ablation_20261013')
OLD=BASE/'Basic_CP_common_ledger_recovery_1791972000100000000'
NEW=BASE/'Basic_CP_RAM512_common_1791977800100000000'
UID=41833
read=lambda p:json.loads(Path(p).read_bytes())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def require(ok,message):
 if not ok:raise ValueError(message)


def module(path):
 spec=importlib.util.spec_from_file_location('_RAM512_final_release_helpers',path)
 value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value


def inventory(api):
 manifest=read(api.exact_file(OLD/'launch.json'));row=manifest['registration']['6'];root=api.exact_file(row['root'],True)
 q=read(api.exact_file(root/'request.json'));native=read(api.exact_file(root/'native.json'))
 trainer=read(api.exact_file(root/'train_actor.json'));controller=row['controller_identity']
 require(sha(root/'request.json')==row['request_sha256']and q['controller_identity']==controller,
         'Exact own registered request/controller required')
 require(q['gpu']==native['gpu']==6 and q['arm']==native['arm']=='blend50','Original full blend50 native6 required')
 require((trainer['pid'],api.birth(trainer),trainer['UID'])==(4155503,1791974178.06,UID),
         'Only reported own original native6 validation child admitted')
 require(trainer['command']==[q['controller_identity']['command'][0],'-B','-u',q['controller_identity']['command'][3],
         '--native',str(root/'native.json'),'--stage','train'],'Original native6 command differs')
 model=read(api.exact_file(root/'actual_model.json'));goal=api.fold(native)
 require(model['epochs']==250 and model['physical_batch']==model['effective_batch']==2
         and len(model['actual_train_cases'])==105 and len(model['actual_validation_cases'])==26
         and model['debug']is False and model['subset']is False,'Original full250/B2/105/26 model required')
 require((goal/'checkpoint_final.pth').exists()and not(goal/'checkpoint_latest.pth').exists()
         and not(goal/'best_validation_complete.json').exists()
         and not(root/'training_and_BEST26_complete.json').exists(),'Only finalized250 with unfinished BEST26 may be released')
 return root,q,native,model,goal,trainer,controller


def preserve_validation(api,proof):
 require(proof['all_captured_actors_released']is True and proof['full_training_completed']is True,
         'Actual full finalized training and released actors required')
 require(all(api.released(a)for a in[proof['trainer'],proof['controller'],*proof['captured_DA']]),
         'No original validation/export actor may survive folder preservation')
 goal=api.exact_file(proof['fold'],True)
 for name,key in(('checkpoint_final.pth','FINAL'),('checkpoint_best.pth','BEST')):
  require(sha(goal/name)==proof[key]['sha256'],'Original whole-state checkpoint changed before preservation')
 require(not(goal/'best_validation_complete.json').exists(),'A completed BEST26 is never relocated')
 source=goal/'validation';destination=goal/'validation_RAM512_preserved_partial_1791977800100000000'
 require(not destination.exists()and not destination.is_symlink(),'Inspect existing preserved output before replay')
 records={}
 if source.exists():
  api.exact_file(source,True)
  for p in source.rglob('*'):
   if p.is_file():records[str(p.relative_to(source))]=dict(sha256=api.sha(p),bytes=p.stat().st_size)
  # Both exact paths remain within the verified existing experiment fold.
  # Rename to a fresh destination preserves every original file byte.
  os.rename(source,destination)
  for relative,record in records.items():
   require(api.sha(destination/relative)==record['sha256'],'Preserved partial prediction changed')
 require(not source.exists(),'Fresh full26 evaluation requires an absent validation destination')
 return dict(partial_output=str(destination)if destination.exists()else None,preserved_files=records,
             original_predictions_overwritten=False,original_checkpoint_inodes_unchanged=True,
             subsequent_evaluation_cases=26,subsequent_checkpoint='BEST',full_training_epochs=250)


def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('mode',choices=('inspect','release','preserve-output'))
 parser.add_argument('--helpers',type=Path,required=True);parser.add_argument('--helpers-sha256',required=True)
 parser.add_argument('--manifest-sha256',required=True);parser.add_argument('--proof',type=Path,required=True)
 args=parser.parse_args();require(os.getuid()==UID and socket.gethostname().split('.')[0]=='ece-a6gpu6','Exact own host/UID required')
 require(sha(args.helpers)==args.helpers_sha256 and sha(OLD/'launch.json')==args.manifest_sha256,'Parent reviewed exact helpers/manifest SHA required')
 api=module(args.helpers);require(args.proof.parent==NEW,'Exact fresh RAM512 operational namespace required')
 if args.mode=='preserve-output':
  proof=read(api.exact_file(args.proof));receipt=preserve_validation(api,proof)
  api.seal(args.proof.with_name('GPU6_preserved_partial_BEST26.json'),receipt);print(json.dumps(receipt),flush=True);return
 root,q,native,model,goal,trainer,controller=inventory(api)
 parent=api.exact_actor(controller);child=api.exact_actor(trainer)
 require(child.ppid()==parent.pid,'Original validation/controller lineage changed')
 known={};api.capture_children(child,known)
 import torch
 torch.set_num_threads(1)
 before_final=api.checkpoint_info(goal/'checkpoint_final.pth',native,model['parameters'])
 before_best=api.checkpoint_info(goal/'checkpoint_best.pth',native,model['parameters'])
 require(before_final['completed_epoch']==250 and before_best['completed_epoch']<=250,
         'Verified full FINAL250 and whole BEST required before any signal')
 info=dict(trainer=trainer,controller=controller,captured_DA=list(known.values()),fold=str(goal),
           manifest_sha256=args.manifest_sha256,full_training_completed=True,training_goal_epochs=250,
           unfinished_BEST26=True,all_original_results_preserved=True,signals_sent=False,
           FINAL_before_signal=before_final,BEST_before_signal=before_best)
 if args.mode=='inspect':print(json.dumps(info),flush=True);return
 require(not args.proof.exists(),'Inspect prior exact signal outcome instead of replay')
 intent=args.proof.with_name(args.proof.name+'.intent.json')
 api.seal(intent,dict(info,planned_signal='SIGINT',only_single_own_native6=True))
 api.exact_actor(controller);p=api.exact_actor(trainer)
 require(p.ppid()==controller['pid']and not(goal/'best_validation_complete.json').exists()
         and api.inode((goal/'checkpoint_final.pth').stat())==tuple(before_final['source_inode'])
         and api.inode((goal/'checkpoint_best.pth').stat())==tuple(before_best['source_inode']),
         'Original validation completed or changed before signal')
 print(json.dumps(dict(about_to_signal='SIGINT',target=trainer,reason='Approved total512 memory cohort migration',
                       checkpoint_final_training_epochs=250,partial_outputs_will_be_preserved=True)),flush=True)
 p.send_signal(signal.SIGINT)
 api.seal(intent.with_name(intent.name+'.single_signal.json'),dict(target=trainer,signal='SIGINT',sent_unix=time.time(),may_not_repeat=True))
 deadline=time.monotonic()+120
 while time.monotonic()<deadline:
  if not api.released(trainer):
   try:
    process=psutil.Process(trainer['pid'])
    if process.create_time()==api.birth(trainer):
     require(process.uids().real==UID,'Terminating validator UID changed')
     api.capture_children(process,known)
   except psutil.NoSuchProcess:pass
  live=[a for a in[trainer,controller,*known.values()]if not api.released(a)]
  if not live:break
  time.sleep(.25)
 require(not live,'Exact own validator/controller/export children still releasing; no signal replay permitted')
 final=api.checkpoint_info(goal/'checkpoint_final.pth',native,model['parameters'])
 best=api.checkpoint_info(goal/'checkpoint_best.pth',native,model['parameters'])
 require(final['completed_epoch']==250 and best['completed_epoch']<=250
         and final['sha256']==before_final['sha256']and best['sha256']==before_best['sha256'],
         'Actual whole finalized250/BEST evidence differs')
 api.seal(args.proof,dict(info,captured_DA=list(known.values()),all_captured_actors_released=True,
                        FINAL=final,BEST=best,single_SIGINT_sent=True,final_training_goal_epochs=250,
                        full_evaluation_completed=False,original_scientific_contract_unchanged=True))
 print(json.dumps(dict(proof=str(args.proof),full_training_completed=True,BEST26_completed=False)),flush=True)


if __name__=='__main__':main()
