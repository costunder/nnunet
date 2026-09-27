"""C01 mutation admission on a real mid-update DEBUG checkpoint; no resealing."""
from pathlib import Path
import copy,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
    import torch
    from tools.v22_artifacts import validate_artifact
    from tools.v22_debug_profile import RepairDataset
    checkpoint,cache,output=map(Path,sys.argv[1:])
    value=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if not value['debug'] or value['state']['phase']!='optimization' or value['state']['step']<1 or value['state']['next_batch']<1:
        raise ValueError('Actual mid-optimization DEBUG checkpoint required')
    keys=('training_objective','ranking_contract','feature_coordinates','artifact_contract','geometry_contract','run_id',
          'support_task_contract','config','base','source_identity','cache_sha256','debug','optimizer_contract','rng_contract')
    identity={k:value[k] for k in keys};rows=RepairDataset(cache,'inner_train').rows
    validate_artifact(value,'resume',allow_debug=True,identity=identity,resume_rows=rows)
    results=[]
    for change in ('finite_parameter','missing_model_hash'):
        bad=copy.deepcopy(value)
        if change=='finite_parameter':next(t for t in bad['model'].values() if t.is_floating_point()).flatten()[0]+=.125
        else:del bad['model_sha256']
        try:validate_artifact(bad,'resume',allow_debug=True,identity=identity,resume_rows=rows)
        except ValueError as error:results.append(dict(mutation=change,rejected=True,error=str(error)))
        else:raise AssertionError('Corrupt mid-update checkpoint admitted')
    result=dict(debug=True,scope='CPU admission of actual GPU-generated checkpoint',step=value['state']['step'],
        next_batch=value['state']['next_batch'],normal_admitted=True,hashes_recomputed_after_mutation=False,rejections=results)
    with output.open('x',encoding='utf-8',newline='\n') as f:json.dump(result,f,indent=2)
    print(json.dumps(result))


if __name__=='__main__':main()
