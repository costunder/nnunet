"""Actual-cache DEBUG measurement of the production prefetch calibration path."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
    from tools.v222_process_loader import calibrate_workers
    from tools.v222_runtime_cache import CachedPairDataset
    root=Path(sys.argv[1]);root.mkdir(parents=True,exist_ok=False)
    data=CachedPairDataset(ROOT/'work/v222_v1_recovered2_training_20260924/cache/index_execution_r6_final.json','inner_train')
    # Explicit DEBUG calibration, matching the original 16-logical-CPU probe.
    ids=sorted(range(len(data)),key=lambda i:data.rows[i]['bounds']['edges'],reverse=True)[:16]
    calibrate_workers(data,ids,root)


if __name__=='__main__':main()
