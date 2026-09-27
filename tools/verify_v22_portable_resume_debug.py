"""Resume an actual DEBUG rolling checkpoint while forbidding external best-file reads."""
from pathlib import Path
import argparse,json,shutil,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
    import torch
    from tools.v22_artifacts import tree_hash
    from tools.run_v222_process_runtime import main as run
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cache',type=Path)
    parser.add_argument('rolling',type=Path)
    parser.add_argument('reference',type=Path)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    original_load=torch.load
    saved=original_load(args.rolling,map_location='cpu',weights_only=False)
    if saved.get('debug') is not True or saved['state']['phase']!='final_memory':
        raise ValueError('Completed-epoch DEBUG final-memory rolling checkpoint required')
    forbidden=Path(saved['state']['best_path']).resolve()
    copied=args.output/'checkpoint_latest.pt'
    shutil.copyfile(args.rolling,copied)
    def guarded_load(path,*pos,**kwargs):
        if isinstance(path,(str,Path)) and Path(path).resolve()==forbidden:
            raise AssertionError('Portable resume tried to open the external best epoch file')
        return original_load(path,*pos,**kwargs)
    argv=sys.argv
    torch.load=guarded_load
    sys.argv=['tools/run_v222_process_runtime.py','--debug-profile','review_repair',
        '--cache',str(args.cache),'--resume',str(copied),
        '--output',str(args.output/'resumed'),'--release-unused']
    try:run()
    finally:torch.load=original_load;sys.argv=argv
    result=original_load(args.output/'resumed/checkpoint.pt',map_location='cpu',weights_only=False)
    reference=original_load(args.reference,map_location='cpu',weights_only=False)
    for key in ('state_dict','memory'):
        if tree_hash(result[key])!=tree_hash(reference[key]):raise AssertionError(f'Portable {key} changed')
    if result['run_id']!=reference['run_id']:raise AssertionError('Run identity changed')
    report=dict(debug=True,external_best_reads_forbidden=True,copied_only_rolling_checkpoint=True,
        selected_weights_bitwise_equal=True,final_memory_bitwise_equal=True,run_id_preserved=True,
        cpu_best_snapshot=all(t.device.type=='cpu' for t in saved['best_snapshot']['state_dict'].values()),
        source_checkpoint_sha256=__import__('hashlib').sha256(args.rolling.read_bytes()).hexdigest())
    (args.output/'result.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
