"""Explicit fixed-view training cache; existing diagnostic receipts stay intact."""
import copy
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import psutil
import torch
from tqdm import tqdm
from hiercp_v222.v1_cache import configuration, provenance
from hiercp_v222.v1_local import V1LocalEncoder
from l0_ezsp.config import load_profile
from l0_ezsp.identity import load_cnn_only
from .materialization import load_pairs
from .preparation import binding, prepare, fixed_profile
from .data import load, save_new
from tools.v22_artifacts import tree_hash
from tools.v222_review_contracts import installed

FORMAT='fixed_region_training_cache_v1'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_new(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False)

def source_identity(*,preparation=False):
    root=Path(__file__).resolve().parents[1]
    if preparation:
        from .preparation import PREPARATION_FILES
        return dict(core=provenance(),preparation={name:sha(root/name) for name in (*PREPARATION_FILES,'l0_regions/training_data.py')})
    files=[*root.joinpath('l0_regions').glob('*.py'),root/'l0_sage/encoder.py',
           root/'tools/run_fixed_regions.py',root/'tools/v22_rank_objective.py',root/'tools/v222_review_contracts.py']
    return dict(core=provenance(),runtime={p.relative_to(root).as_posix():sha(p) for p in files})

class Budget:
    def __init__(self,cuda_bytes,rss_bytes):
        if min(cuda_bytes,rss_bytes)<=0:raise ValueError('Positive resource budgets required')
        self.cuda_bytes=cuda_bytes;self.rss_bytes=rss_bytes
    def check(self):
        if torch.cuda.memory_allocated()>self.cuda_bytes:raise MemoryError('CUDA allocation budget exceeded')
        if psutil.Process().memory_info().rss>self.rss_bytes:raise MemoryError('Process RSS budget exceeded')

def dataset(index,partition,debug):
    if debug:
        from tools.v22_debug_profile import RepairDataset
        return RepairDataset(index,partition)
    from tools.v222_runtime_cache import CachedPairDataset
    return CachedPairDataset(index,partition)

def reference_from_checkpoint(checkpoint,debug):
    cfg,base=configuration();torch.manual_seed(cfg['seed'])
    with installed('stride4'):net=V1LocalEncoder(base)
    ckpt=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True)
    if ckpt.get('debug') and not debug:raise ValueError('DEBUG partition weights cannot initialize full training')
    if ckpt.get('state',{}).get('step',0)<=0:raise ValueError('An existing optimized CNN snapshot is required')
    load_cnn_only(net,ckpt['model'])
    return net

def prepare_cache(index,checkpoint,output,*,batch,workers,reg1,reg2,view_epoch,budget,debug=False):
    index=Path(index).resolve();output=Path(output).resolve();checkpoint=Path(checkpoint).resolve()
    if min(batch,workers)<1 or view_epoch<0:raise ValueError('Explicit positive preparation settings required')
    reference=reference_from_checkpoint(checkpoint,debug)
    frozen=copy.deepcopy(reference).eval().requires_grad_(False).cuda()
    profile=fixed_profile(load_profile(reg_scale1=reg1,reg_scale2=reg2))
    cfg,base=configuration();cache_hash=sha(index);checkpoint_hash=sha(checkpoint)
    output.mkdir(parents=True,exist_ok=False)
    # Only CNN initialization is copied, never GAT/optimizer/old support.
    with (output/'frozen_cnn.pt').open('xb') as f:torch.save(reference.dense_encoder.state_dict(),f)
    meta=dict(format=FORMAT,debug=debug,source_identity=source_identity(preparation=True),original_cache_sha256=cache_hash,
        original_cache=str(index),partition_checkpoint_sha256=checkpoint_hash,
        cnn_sha256=tree_hash(reference.dense_encoder.state_dict()),cnn_file_sha256=sha(output/'frozen_cnn.pt'),
        profile=profile,view_epoch=view_epoch,config=cfg,base=base,partitions={},admission_failures=0,
        complete=False,full_training_admitted=False)
    write_new(output/'request.json',meta)
    total_bytes=0
    for part in ('inner_train','inner_val'):
        ds=dataset(index,part,debug);rows=[]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for start in tqdm(range(0,len(ds.rows),batch),desc='prepare '+part,unit='batch'):
                budget.check()
                if psutil.disk_usage(str(output)).free<cfg['minimum_free_gb']*2**30:raise OSError('Disk reserve reached; partial files preserved')
                ids=list(range(start,min(start+batch,len(ds.rows))))
                fine=load_pairs(ds,ids,workers=workers,epoch=view_epoch,cache_path=index).to('cuda')
                bindings=[binding(record=dict(ds.rows[i],donor_component=ds.record(i)['component_id']),dataset_index=i,
                    cache_sha256=cache_hash,frozen_cnn_sha256=meta['cnn_sha256'],profile=profile,view_epoch=view_epoch,
                    view_index=0,feature_evidence='checkpoint_partition_quality_unverified') for i in ids]
                # DEBUG may measure rejected profiles, full training may not.
                items,audit=prepare(fine,frozen,profile,bindings,budget,allow_unvalidated_profile=debug)
                paths=[output/f'{part}_{i:06d}.pt' for i in ids]
                total_bytes+=sum(pool.map(lambda pair:save_new(*pair),zip(paths,items)))
                for i,path,item in zip(ids,paths,items):
                    if item['profile_exceeded']:meta['admission_failures']+=1
                    rows.append(dict(row=ds.rows[i],file=path.name,binding=item['binding'],profile_exceeded=item['profile_exceeded'],
                        fine_nodes=sum(len(v['grid']) for v in item['fine'].values()),
                        region_nodes=[sum(len(v['mass']) for v in layer.values()) for layer in item['scales']],
                        region_edges=[sum(e.shape[1] for e in edges.values()) for edges in item['edges']] ))
                write_new(output/f'{part}_{start:06d}_audit.json',audit)
                del fine,items
        meta['partitions'][part]=rows
    if sha(index)!=cache_hash or sha(checkpoint)!=checkpoint_hash:raise ValueError('Preparation inputs changed')
    if meta['source_identity']!=source_identity(preparation=True):raise ValueError('Preparation sources changed')
    if not debug and meta['admission_failures']:raise ValueError('Rejected pair cannot enter full training')
    meta.update(complete=True,full_training_admitted=not debug,serialized_pt_bytes=total_bytes)
    # This admission proves execution/receipt/profile checks, not CP efficacy.
    write_new(output/'index.json',meta)
    return output/'index.json'

class RegionDataset:
    def __init__(self,index,partition,debug):
        self.path=Path(index).resolve();self.root=self.path.parent
        self.meta=json.loads(self.path.read_text(encoding='utf-8'))
        m=self.meta
        if m.get('format')!=FORMAT or m.get('debug')!=debug or not m.get('complete'):raise ValueError('Cache mode/format/completion mismatch')
        if m['source_identity']!=source_identity(preparation=True):raise ValueError('Region preparation source identity changed')
        if not debug and (not m['full_training_admitted'] or m['admission_failures']):raise ValueError('Full training admission failed')
        if sha(self.root/'frozen_cnn.pt')!=m['cnn_file_sha256']:raise ValueError('CNN snapshot file changed')
        self.entries=m['partitions'][partition];self.rows=[v['row'] for v in self.entries]
        if not self.rows or len({r['id'] for r in self.rows})!=len(self.rows):raise ValueError('Missing/duplicate records')
        if any(e['binding']['dataset_index']!=i or e['binding']['record_id']!=e['row']['id'] for i,e in enumerate(self.entries)):
            raise ValueError('Record/index binding mismatch')
    def __len__(self):return len(self.rows)
    def request(self,ids):
        return [self.root/self.entries[i]['file'] for i in ids],[self.entries[i]['binding'] for i in ids],ids
