"""DEBUG only: real complete 128 pool through the untrained CNN-free scorer.

Physical batch 4 was measured by verify_v22_debug; calibration is replaced only
inside this diagnostic to isolate full-pool wiring. No final checkpoint is made.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import time
import threading
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
    import torch,numpy as np
    from hiercp_v22.contracts import load_config,write_new,sha,source_identity,safe_new_root
    from hiercp_v22.migration import convert_record
    from hiercp_v2.storage import load_record
    from hiercp_v22.data import collate,materialize
    from hiercp_v22.model import PromptGraphModel,context_descriptor
    from hiercp_v22.scoring import Scorer
    from hiercp_v22 import scoring
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-root',type=Path,required=True)
    parser.add_argument('--event',required=True)
    parser.add_argument('--anchors',nargs=3,required=True,help='Three actual inner-train T record paths')
    parser.add_argument('--split',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();cfg,base=load_config();out=safe_new_root(args.output)
    from hiercp_v22.contracts import read_json
    split=read_json(args.split);torch.set_num_threads(4)
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    files=[f'graphs/{args.event}/{i:04d}.pt.gz' for i in range(128)]
    if len(list((args.cache_root/'graphs'/args.event).glob('*.pt.gz')))!=128:
        raise ValueError('Exactly one complete 128-candidate event required')
    hashes={r:sha(args.cache_root/r) for r in files+args.anchors}
    def read(r):return convert_record(load_record(args.cache_root,r))
    start=time.perf_counter()
    with ThreadPoolExecutor(max_workers=4) as executor:records=list(executor.map(read,files))
    anchors=[read(r) for r in args.anchors];ids=[r['case_id'] for r in anchors]
    if len(set(ids))!=3 or not set(ids)<=set(split['inner_train']):raise ValueError('Three independent inner-train anchors required')
    if any(r.get('evidence')!=1 for r in anchors):raise ValueError('Support must be actual T observations')
    recipient=records[0]['case_id']
    if any(r['case_id']!=recipient or r.get('evidence')!=-1 for r in records):raise ValueError('One recipient U pool required')
    model=PromptGraphModel(cfg,base,patient_ids=ids).cuda().eval()
    batch=collate([(materialize(r),i) for i,r in enumerate(anchors)]).to('cuda')
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):encoded=model.encode_local(batch).float()
    memory=dict(case_ids=ids,embeddings=encoded,owners=torch.arange(3,device='cuda'),
                evidence=torch.ones(3,dtype=torch.long,device='cuda'),descriptors=context_descriptor(batch),
                donor_allowed=torch.ones(3,dtype=torch.bool,device='cuda'))
    scorer=Scorer.__new__(Scorer);scorer.device=torch.device('cuda');scorer.cfg=cfg;scorer.base=base
    scorer.model=model;scorer.memory=memory;scorer.payload={'split':split}
    del batch
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();score_start=time.perf_counter()
    with patch.object(scoring,'calibrate',side_effect=lambda *a,**k:(4,{'scope':'DEBUG batch=4, previously measured; no production override'})), \
         patch.object(scoring,'calibrate_workers',return_value=(2,{'scope':'DEBUG 2 workers; production remains auto'})):
        scores,report=scorer.score_records(records,recipient)
    torch.cuda.synchronize()
    if scores.shape!=(128,) or not np.isfinite(scores).all():raise RuntimeError('Full-pool scoring failed')
    if any(sha(args.cache_root/r)!=digest for r,digest in hashes.items()):raise RuntimeError('Original records changed')
    write_new(out/'verification.json',dict(scope='DEBUG untrained full-width 3-patient support, real complete candidate pool; no medical accuracy',
        source_identity=source_identity(),recipient=recipient,support_patients=ids,candidate_count=128,
        score_min=float(scores.min()),score_max=float(scores.max()),score_std=float(scores.std()),
        selected_argmax=int(np.argmax(scores)),seconds_scoring=time.perf_counter()-score_start,
        seconds_total=time.perf_counter()-start,peak_process_bytes=torch.cuda.max_memory_allocated(),
        physical_graph_batch=4,debug_workers=2,original_sha_unchanged=True,full_training=False,full_evaluation=False,
        native_paste_executed=False,resources=report,record_sha256=hashes))
    print({'verification':str(out/'verification.json'),'candidates':128,'seconds':time.perf_counter()-start},flush=True)


if __name__=='__main__':main()
