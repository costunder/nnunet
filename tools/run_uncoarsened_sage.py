"""Foreground graph-only transition/resume; inherit exact saved training settings."""
import argparse,json,shlex,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from l0_regions.training import FORMAT,hash_state
from l0_regions.fine_graph import MODE


def arguments(saved,*,cache,fine_cache,resume,output):
    identity=saved['identity'];limits=identity['resource_limits']
    if identity['debug']:raise ValueError('This server launcher requires a production-sized checkpoint; use explicit DEBUG tools locally')
    if identity.get('graph_representation') not in (None,MODE):raise ValueError('Unknown saved graph representation')
    args=['train','--cache',str(cache),'--fine-cache',str(fine_cache),'--resume',str(resume),'--output',str(output),
        '--profile-policy',identity['profile_policy'],'--workers',str(identity['workers']),
        '--cuda-gib',str(limits['cuda_bytes']/2**30),'--rss-gib',str(limits['rss_bytes']/2**30),
        '--resident-gib',str(identity['resident_budget_bytes']/2**30),
        '--batch-candidates',*map(str,identity['candidates']),'--activation-storage',identity['activation_storage']]
    if identity.get('graph_representation') is None:args+=['--resume-without-coarsening']
    if 'support_training' in identity:args+=['--support-patients',str(identity['support_training']['patients'])]
    if 'execution_pipeline' in identity:
        execution=identity['execution_pipeline']
        args+=['--execution-pipeline',execution['mode'],'--device-cache-gib',str(execution['device_cache_bytes']/2**30),
               '--sage-workspace-mib',str(execution['sage_workspace_bytes']//2**20)]
    return args


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('cache','fine-cache','resume','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--print-command',action='store_true',help='Read and validate checkpoint; print inherited command without training')
    a=parser.parse_args()
    saved=torch.load(a.resume,map_location='cpu',weights_only=False)
    if saved.get('format')!=FORMAT or saved.get('content_sha256')!=hash_state({k:v for k,v in saved.items() if k!='content_sha256'}):
        raise ValueError('Invalid checkpoint contents')
    args=arguments(saved,cache=a.cache,fine_cache=a.fine_cache,resume=a.resume,output=a.output)
    print(json.dumps(dict(stage='uncoarsened_launch',epoch_index=saved['state']['epoch'],saved_step=saved['state']['step'],
        next_batch=saved['state']['next_batch'],physical_batch=saved['state']['batch'],
        coarsening_enabled=False,settings='inherited from the selected checkpoint; no batch/support/resource reduction',
        training_started=False)),flush=True)
    print(shlex.join([sys.executable,'-u','tools/run_fixed_regions.py',*args]),flush=True)
    del saved
    if a.print_command:return
    # Same process, so Ctrl+C reaches the existing save-and-pause handler.
    from tools.run_fixed_regions import main as train_main
    sys.argv=['tools/run_fixed_regions.py',*args]
    train_main()


if __name__=='__main__':main()
