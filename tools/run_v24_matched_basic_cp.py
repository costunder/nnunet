"""Own-arm complete GNN -> Basic-matched CP -> full native BEST evaluation."""
from __future__ import annotations
import argparse
from pathlib import Path
import os
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True


def parse(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('pin-current-gnn','prepare-current-bank','prepare-native',
                                          'calibrate-native','train','clone-debug-worker'))
    for name in ('storage-admission','pin','inventory','baseline-preprocessed','historical-basic-bank',
                 'input-cache','output','source-output','source-code','stunet-checkpoint','bank','native'):
        parser.add_argument('--'+name,type=Path)
    parser.add_argument('--gpu',type=int,choices=(4,5,6))
    parser.add_argument('--physical-batch',type=int,choices=(2,4))
    args = parser.parse_args(argv)
    required = {'pin-current-gnn':('source_output','source_code','inventory','input_cache','output'),
        'prepare-current-bank':('pin','inventory','baseline_preprocessed','historical_basic_bank','input_cache','output'),
        'prepare-native':('bank','output'), 'calibrate-native':('native',),'train':('native',),
        'clone-debug-worker':('native','physical_batch','output')}
    if args.action != 'clone-debug-worker':
        required[args.action] += ('gpu','storage_admission')
    elif args.gpu is not None or args.storage_admission is not None:
        parser.error('Clone DEBUG worker derives exact arm/admission from sealed native.json')
    for name in required[args.action]:
        if getattr(args,name) is None:
            parser.error('--'+name.replace('_','-')+' is required')
    if args.action in ('pin-current-gnn','prepare-current-bank') and (args.gpu in (4,6)) != (args.stunet_checkpoint is not None):
        parser.error('GPU4/6 require their original official STU weights; GPU5 uses its original CNN')
    if args.action != 'clone-debug-worker' and args.physical_batch is not None:
        parser.error('Production physical batch is fixed by historical plans; worker option is DEBUG only')
    return args


def main(args):
    from hiercp_v1x import v24_matched_basic_cp_pipeline as pipeline
    if args.action == 'clone-debug-worker':
        return pipeline.calibration_worker(args.native,args.physical_batch,args.output)
    from hiercp_v1x import v24_nnunet_cp as original
    document = pipeline.verify_admission(args.storage_admission)
    checksum = pipeline.sha(args.storage_admission)
    if os.environ.get('V24_READONLY_STORAGE_ADMISSION_SHA256') != checksum:
        raise ValueError('Exact watcher-pinned matched storage admission SHA required')
    chain = Path(document['arm_roots']['gpu'+str(args.gpu)])
    if args.action in ('pin-current-gnn','prepare-current-bank'):
        if str(args.inventory) != document['inventory']['path'] or pipeline.sha(args.inventory) != document['inventory']['sha256']:
            raise ValueError('Exact full native inventory required')
    pipeline.check_aggregate_disk(document,chain)
    original.require_project_budget()
    if args.action == 'pin-current-gnn':
        if args.output != chain/'pin.json':
            raise ValueError('Exact own-arm completed current GNN pin output required')
        result = original.pin_completed_current(source_output=args.source_output,source_code=args.source_code,
            inventory_path=args.inventory,input_cache=args.input_cache,output=args.output,gpu=args.gpu,
            stunet_checkpoint=args.stunet_checkpoint)
    elif args.action == 'prepare-current-bank':
        if (str(args.historical_basic_bank) != document['historical_basic_bank']['path']
                or str(args.baseline_preprocessed) != document['baseline']['preprocessed']
                or args.pin != chain/'pin.json' or args.output != chain/'bank'):
            raise ValueError('Exact immutable historical source bank, full native preprocessing and own GNN pin required')
        from hiercp_v1x.v24_matched_basic_cp_scoring import score_current_matched_basic_bank
        result = score_current_matched_basic_bank(pin_path=args.pin,inventory_path=args.inventory,
            baseline_preprocessed=args.baseline_preprocessed,input_cache=args.input_cache,
            source_bank=args.historical_basic_bank,output=args.output,gpu=args.gpu,
            stunet_checkpoint=args.stunet_checkpoint)
    elif args.action == 'prepare-native':
        if args.bank != chain/'bank/score_overlay.json':
            raise ValueError('Full642 own-arm matched score overlay required')
        result = pipeline.prepare_native(args.bank,args.output,args.storage_admission,gpu=args.gpu)
    else:
        if args.native != chain/'native/native.json':
            raise ValueError('Exact own-arm matched native metadata required')
        result = (pipeline.calibrate_native(args.native,gpu=args.gpu) if args.action == 'calibrate-native'
                  else pipeline.train(args.native,gpu=args.gpu))
    pipeline.verify_admission(args.storage_admission)
    pipeline.check_aggregate_disk(document,chain,preparation=False)
    pipeline.publish(chain/('matched_storage_'+args.action.replace('-','_')+'.json'),dict(
        format=pipeline.STAGE_FORMAT,status='COMPLETE',action=args.action,storage_profile=pipeline.PROFILE,
        storage_admission=str(args.storage_admission),storage_admission_sha256=checksum,
        runtime_sources_sha256={name:pipeline.sha(ROOT/name) for name in pipeline.FILES},
        cached_inputs_written=False,model_data_scale_preserved=True,native_epochs=250,
        native_physical_batch=2,cp_probability=.5,source_entries=642,candidates_per_source=128,
        source_payloads_written=False,evaluation_checkpoint='checkpoint_best.pth',completed_at=time.time()))
    print(result,flush=True)
    return result


if __name__ == '__main__':
    main(parse())
