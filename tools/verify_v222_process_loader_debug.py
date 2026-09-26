"""Real-cache CPU IPC equivalence; no GPU training or medical score."""
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path
import sys
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    import torch
    from hiercp_v222.v1_local import LocalBatch
    from tools.v222_runtime_cache import CachedPairDataset, CachedPairLoader
    from tools.v222_process_loader import ProcessPairLoader, close_producers
    torch.set_num_threads(2)
    cache=ROOT/'work/v222_v1_recovered2_training_20260924/cache/index_execution_r6_final.json'
    data=CachedPairDataset(cache,'inner_train')
    ids=[0,1]
    loader=CachedPairLoader(data,2)
    try:
        with patch.object(LocalBatch,'pin_memory',lambda self:self):
            expected=loader.make(ids,3)
        producer=ProcessPairLoader(data,8)
        with patch.object(LocalBatch,'pin_memory',lambda self:self):
            actual=producer.make(ids,3)
            for name in ('source_patches','target_patches','source_index','indices'):
                torch.testing.assert_close(getattr(actual,name),getattr(expected,name),atol=0,rtol=0)
            for kind in expected.graph.node_types+expected.graph.edge_types:
                for key in expected.graph[kind].keys():
                    left,right=expected.graph[kind][key],actual.graph[kind][key]
                    if torch.is_tensor(left):torch.testing.assert_close(left,right,atol=0,rtol=0)
                    elif left != right:raise AssertionError((kind,key))
        original_pool=producer.pool
        producer.close()
        reopened=ProcessPairLoader(data,8)
        assert reopened.pool is original_pool,'Closing a phase discarded the producer cache'
        reopened.close()
    finally:
        loader.close()
        close_producers()
    print(json.dumps(dict(debug=True,actual_records=2,epoch=3,exact_CPU_tensor_match=True,
        full_training=False,shared_IPC=True,CUDA_used=False)))


if __name__=='__main__':main()
