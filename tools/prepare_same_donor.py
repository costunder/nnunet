"""Explicit geometry preparation only. Never starts training or rewrites old cache."""
import argparse
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from l0_regions.donor_data import prepare
from l0_regions.training_data import Budget

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('cache','initialization','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--workers',type=int,required=True);p.add_argument('--rss-gib',type=float,required=True)
    p.add_argument('--cuda-gib',type=float,required=True);p.add_argument('--debug',action='store_true')
    a=p.parse_args()
    import torch
    torch.set_num_threads(1)
    result=prepare(a.cache,a.initialization,a.output,workers=a.workers,budget=Budget(int(a.cuda_gib*2**30),int(a.rss_gib*2**30)),debug=a.debug)
    print('RESULT:',result,flush=True)

if __name__=='__main__':main()
