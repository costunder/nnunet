"""Actual process: uninterrupted DEBUG vs interrupted run, including Adam and RNG."""
from pathlib import Path
import argparse,json,random,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
    import numpy as np
    import torch
    from tools.run_v222_process_runtime import main as run
    from tools.v22_artifacts import tree_hash
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cache',type=Path);parser.add_argument('paused',type=Path)
    parser.add_argument('resumed',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args()
    saved=torch.load(args.paused/'checkpoint_latest.pt',map_location='cpu',weights_only=False)
    if saved.get('debug') is not True:raise ValueError('Explicit DEBUG artifact required')
    # configure_runtime seeds Torch. Match the other serialized generators too;
    # no checkpoint tensor, model setting, row, loss or batch is modified.
    np.random.set_state(saved['rng']['numpy']);random.setstate(saved['rng']['python'])
    argv=sys.argv
    sys.argv=['tools/run_v222_process_runtime.py','--debug-profile','review_repair','--cache',str(args.cache),
        '--calibration',str(args.paused),'--output',str(args.output),'--workspace-mib','256','--release-unused']
    try:run()
    finally:sys.argv=argv
    a=torch.load(args.output/'checkpoint_latest.pt',map_location='cpu',weights_only=False)
    b=torch.load(args.resumed/'checkpoint_latest.pt',map_location='cpu',weights_only=False)
    def same(x,y):
        if torch.is_tensor(x):return torch.equal(x,y)
        if isinstance(x,np.ndarray):return np.array_equal(x,y)
        if isinstance(x,dict):return x.keys()==y.keys() and all(same(x[k],y[k]) for k in x)
        if isinstance(x,(tuple,list)):return len(x)==len(y) and all(same(i,j) for i,j in zip(x,y))
        return x==y
    checks={key:same(a[key],b[key]) for key in ('model','optimizer','rng')}
    checks['final_support']=same(a['state']['memory'],b['state']['memory'])
    if not all(checks.values()):raise AssertionError(checks)
    result=dict(debug=True,actual_CT=True,actual_process=True,uninterrupted_vs_resumed_bitwise=checks,
        updates=a['state']['step'],training_resume_preserved=True,calibration_reuse_executed=True,
        optimizer_sha256=tree_hash(a['optimizer']))
    (args.output/'continuation_result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
