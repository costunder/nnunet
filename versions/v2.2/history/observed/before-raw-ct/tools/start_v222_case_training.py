"""Immediate full-cohort preparation and 40-epoch GNN training on published cases.

This runner starts the GNN stage. It does not label a GNN checkpoint as a trained
nnU-Net or use unmeasured native worker settings to launch segmentation.
"""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    from hiercp_v222.contracts import (load_config,read_json,write_new,safe_new_root,
        validate_identities,verify_v1,source_identity,CASE_BENCHMARK_IDENTITY)
    from hiercp_v222.data import prepare
    from hiercp_v222.training import train
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    cfg,base=load_config();verify_v1()
    dataset=Path(args.dataset).resolve();root=safe_new_root(args.output)
    split=read_json(dataset/'split.json')
    identities=dict(format=CASE_BENCHMARK_IDENTITY,independence_scope='published_case_only',
        patient_independence_verified=False,annotation_scope='provided_masks_may_omit_lesions',
        authorization='User requested training with the discussed local Basic CP comparison settings on 2026-09-22.',
        cases={c:dict(patient_group='case:'+c,identity_basis='published_case_id_only',annotation_complete=None)
               for c in split['outer_train']+split['outer_val']})
    validate_identities(identities,split)
    write_new(root/'identities.json',identities)
    write_new(root/'requested.json',dict(utc=datetime.now(timezone.utc).isoformat(),config=cfg,
        dataset=str(dataset),source_identity=source_identity(),split=split,identities=identities,
        stage_scope='full_context_preparation_and_40_epoch_GNN',debug=False,subset=False,
        subsequent_segmentation=dict(epochs=250,seed=42,cp_probability=cfg['cp_probability'],
            status='requires_completed_GNN_and_native_throughput_calibration'),
        cp_probability_basis='User-approved paired CP80 comparison; TumorCP uses 0.8, not proven optimal for this dataset'))
    stage='context_preparation'
    try:
        print(json.dumps(dict(stage=stage,full_cases=len(split['outer_train']),debug=False)),flush=True)
        cache=prepare(dataset/'medical',dataset/'split.json',root/'identities.json',root/'context',cfg,base)
        stage='gnn_training'
        print(json.dumps(dict(stage=stage,epochs=cfg['gnn_epochs'],cache=str(cache),debug=False)),flush=True)
        checkpoint=train(cache,root/'gnn',cfg,base)
        write_new(root/'gnn_complete.json',dict(checkpoint=str(checkpoint),completed_epochs=cfg['gnn_epochs'],
            segmentation_training_executed=False))
        print(json.dumps(dict(stage='gnn_complete',checkpoint=str(checkpoint))),flush=True)
    except Exception as error:
        write_new(root/'failed.json',dict(stage=stage,type=type(error).__name__,error=str(error),
            recovery='All partial artifacts are preserved. Diagnose the recorded error; do not reduce the model or dataset.'))
        raise


if __name__=='__main__':main()
