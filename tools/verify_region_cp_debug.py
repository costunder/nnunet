"""Actual CT short CP smoke plus fixed-event reuse and mutation rejection."""
import argparse
import copy
import json
from pathlib import Path
import sys
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('cache','checkpoint','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    import torch
    from l0_regions.recommendation import prepare
    from l0_regions.final import validate
    from tools.v22_rank_recommendation import load_checkpoint
    from tools.verify_v22_review_selection_debug import main as verify
    captured={}
    def load(*args,**kwargs):
        result=load_checkpoint(*args,**kwargs);captured['network']=result[0];return result
    with patch('l0_regions.recommendation.prepare',wraps=prepare) as partitioned,patch('tools.v22_rank_recommendation.load_checkpoint',side_effect=load):
        verify(a.cache,a.checkpoint,a.output)
        calls=partitioned.call_count
    network=captured['network'];owner=network.region_candidates
    if calls!=len(owner.event_batches):raise AssertionError('Repeated same event repartitioned')
    cpu=owner.event_batches[0]
    cpu.target_patches.add_(1)
    try:cpu.to('cuda')
    except ValueError:mutation_rejected=True
    else:raise AssertionError('Changed resident CT accepted')
    payload=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    rejected=[]
    for field in ('state_dict','memory','frozen_cnn'):
        value=copy.deepcopy(payload)
        tensor=value[field]['embeddings'] if field=='memory' else next(v for v in value[field].values() if torch.is_tensor(v))
        tensor.reshape(-1)[0]+=1
        try:validate(value,allow_debug=True)
        except ValueError:rejected.append(field)
        else:raise AssertionError('Changed final artifact accepted: '+field)
    result=dict(debug=True,status='PASS',actual_CT=True,full_training=False,
        event_partition_batches=calls,repeated_event_repartitioned=False,resident_mutation_rejected=mutation_rejected,
        final_artifact_mutations_rejected=rejected,
        scope='3 explicit DEBUG candidates; existing raw CP and complete mask oracle. Native segmentation update not run; cold official repartition repeatability not claimed.')
    with (a.output/'region_contract.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
