"""Actual native admission and fresh GT-blind v2.4 runtime construction.

The sealed v1 archive and train-only prototype bank are provenance inputs.
No trained v2.3 weights or annotation-dependent canonical/upper cache is read.
"""
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import copy
import json
from pathlib import Path
import threading
from types import SimpleNamespace

import numpy as np
import psutil
import torch

from .contracts import canonical_hash
from hiercp_v22.contracts import sha
from .v24_inputs import (recipient_context, prepare_donor, build_local_record,
    materialize_pair, query_inputs, array_digest, immutable_array)
from .v24_provider import FORMAT as CANONICAL_FORMAT, V24InputCoordinator, V24InputProvider


def _stat(path):
    value=Path(path).stat()
    return value.st_dev,value.st_ino,value.st_size,value.st_mtime_ns,value.st_ctime_ns


def _publish(path, value):
    path=Path(path)
    if path.exists():
        if json.loads(path.read_text(encoding='utf8')) != value:
            raise FileExistsError('Existing owned v24 metadata differs; no overwrite: '+str(path))
        return
    with path.open('x',encoding='utf8') as stream: json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)


class V24NativeInputs:
    """Annotated donors plus label-less recipient contexts, with file admission."""
    def __init__(self, experiment, inventory, output, runtime):
        from .comparison_native_upper_cache import _load_geometry_inputs
        from .v23_data import V23Population
        from .v24_geometry import V24UpperGeometryCache
        self.bundle,self.inventory,self.cohort,self.signature,self.preserved = _load_geometry_inputs(experiment,'native',inventory)
        from hiercp.schema import graph_config_from_dict
        self.population=V23Population(self.inventory,debug=False)
        self.root=Path(output).resolve(); self.root.mkdir(parents=True,exist_ok=True)
        self.runtime=copy.deepcopy(runtime)
        self.config=copy.deepcopy(self.bundle.config)
        self.graph_config=graph_config_from_dict(self.config['graph'])
        self.ct_clip=tuple(self.config['ct_clip'])
        self.training_cases=set(self.inventory['split']['inner_train'])
        self.raw={row['case_id']:row for row in self.inventory['raw_records']}
        self._file_proofs={str(Path(path).resolve()):_stat(path) for path in self.preserved}
        self._input_file_proofs={str(self.bundle.source/name):_stat(self.bundle.source/name)
            for name in self.signature['source_proof']['verified_files']}
        self._input_file_proofs[self.signature['bank_path']]=_stat(self.signature['bank_path'])
        self._raw_cache,self._donors,self._regions=OrderedDict(),OrderedDict(),OrderedDict()
        self._raw_bytes=0; self._lock=threading.RLock()
        # Raw/region workspace is evictable and shares the actual process RSS
        # ceiling with providers. This is a residency limit, not a data cap.
        self.raw_resident_bytes=runtime['raw_resident_gib']*2**30
        self.rss_bytes=runtime['rss_gib']*2**30
        from . import v24_inputs,v24_geometry,v24_model
        source_paths=[Path(module.__file__) for module in (v24_inputs,v24_geometry,v24_model)]+[
            Path(__file__),Path(__file__).with_name('v24_provider.py')]
        self._input_file_proofs.update({str(path.resolve()):_stat(path) for path in source_paths})
        self.source_sha256=canonical_hash(dict(v24={path.name:sha(path) for path in source_paths},
            archived=self.signature['source_proof']['verified_files']))
        self.config_sha256=canonical_hash(dict(graph=self.config['graph'],ct_clip=self.config['ct_clip'],
            source_pad=self.config['cache']['source_pad']))
        self.bank_sha256=self.signature['bank_sha256']
        self.geometry=V24UpperGeometryCache(self.population,self.root/'upper',self._upper,
            input_binding=self.input_binding,guard_inputs=self.guard_inputs,
            workers=runtime['workers'],resident_bytes=runtime['geometry_resident_gib']*2**30,rss_bytes=self.rss_bytes)

    def guard_source(self):
        """Experiment/supervision provenance, outside the GT-free input cache."""
        for path,proof in self._file_proofs.items():
            if _stat(path)!=proof: raise ValueError('Admitted native/source/prototype provenance changed: '+path)

    def guard_input_files(self, case, donor=None):
        for path,proof in self._input_file_proofs.items():
            if _stat(path)!=proof: raise ValueError('Admitted GT-free implementation/prototype input changed')
        pairs=[(case,'image')]
        if donor is not None: pairs.extend(((donor,'image'),(donor,'label')))
        for identity,kind in pairs:
            path=str(Path(self.raw[identity][kind]).resolve())
            if path in self._file_proofs and _stat(path)!=self._file_proofs[path]:
                raise ValueError('Actual CT or allowed donor annotation changed')

    def _case(self, case):
        from hiercp.common import CasePaths,load_case
        with self._lock:
            self.guard_input_files(case)
            if case in self._raw_cache:
                self._raw_cache.move_to_end(case); return self._raw_cache[case][0]
            row=self.raw[case]
            for kind in ('image','label'):
                path=Path(row[kind]).resolve(strict=True)
                if sha(path)!=row[kind+'_sha256']: raise ValueError('Actual native CT/annotation SHA differs: '+case)
                self._file_proofs[str(path)]=_stat(path)
            loaded=load_case(CasePaths(case,Path(row['image']),Path(row['label'])))
            organ=np.isin(loaded.label,(1,2))
            context=recipient_context(case,loaded.image,organ,loaded.spacing,loaded.image_affine)
            depth=context.organ_depth
            # Donor annotation remains available only in this explicit donor
            # branch; recipient builders receive context, never loaded.label.
            loaded.image=context.image; loaded.spacing=context.spacing; loaded.image_affine=context.image_affine
            loaded.label.flags.writeable=False
            value=dict(context=context,donor_case=loaded,binding=context.binding(),organ_depth=depth)
            size=context.image.nbytes+context.organ_mask.nbytes+loaded.label.nbytes+depth.nbytes
            self._raw_cache[case]=(value,size); self._raw_bytes+=size
            while self._raw_bytes>self.raw_resident_bytes and len(self._raw_cache)>1:
                old,(unused,bytes_)=self._raw_cache.popitem(last=False); self._raw_bytes-=bytes_
                for key in [key for key in self._donors if key[0]==old]: self._donors.pop(key)
                self._regions.pop(old,None)
            self._rss()
            return value

    def _rss(self):
        with self._lock:return self._rss_locked()

    def _rss_locked(self):
        coordinator=getattr(self,'memory_coordinator',None)
        if coordinator is not None:coordinator.trim(strict=False)
        while psutil.Process().memory_info().rss>self.rss_bytes and self._raw_cache:
            case,(_,size)=self._raw_cache.popitem(last=False);self._raw_bytes-=size
            self._regions.pop(case,None)
            for key in [key for key in self._donors if key[0]==case]:self._donors.pop(key)
        if psutil.Process().memory_info().rss>self.rss_bytes:
            raise MemoryError('Actual complete v24 workspace exceeds process RSS; no subset or fallback')

    def _donor(self,row):
        from hiercp_v22.data import sources
        from hiercp.common import stable_case_seed
        key=row['donor_case_id'],row['donor_component']
        with self._lock:
            if key[0] not in self.training_cases or key[0]==row['case_id']:
                raise ValueError('Fixed independent inner-train donor required')
            raw=self._case(key[0])
            if key not in self._donors:
                collection=sources(raw['donor_case'],self.config['cache']['source_pad'],self.inventory['config']['donor_max_diameter_mm'])
                match=[i for i,(component,_) in enumerate(collection.entries) if component==key[1]]
                if len(match)!=1: raise ValueError('Exact native donor component absent')
                source,_=collection[match[0]]
                source.full_mask=immutable_array(source.full_mask)
                source.v24_mask_sha256=array_digest(source.full_mask)
                prepared=prepare_donor(raw['donor_case'],source,config=self.graph_config,
                    seed=self.config['seed'],ct_clip=self.ct_clip)
                self._donors[key]=source,prepared
            source,prepared=self._donors[key]
            return raw['donor_case'],source,prepared

    def _region(self,case):
        from .v24_geometry import build_recipient_regions
        from hiercp.common import stable_case_seed
        from hiercp.region import REGION_CACHE_SEED_SALT
        with self._lock:
            raw=self._case(case)
            if case not in self._regions:
                self._regions[case]=build_recipient_regions(raw['context'],config=self.graph_config,
                    seed=stable_case_seed(42,case,REGION_CACHE_SEED_SALT),ct_clip=self.ct_clip)
            self._rss(); return self._regions[case]

    def input_binding(self,plan):
        raw=self._case(plan.case_id); row=plan.query_rows[0]
        donor,source,_=self._donor(row)
        return dict(recipient=copy.deepcopy(raw['binding']),
            donor=dict(case_id=row['donor_case_id'],component_id=row['donor_component'],
                CT_sha256=self.raw[row['donor_case_id']]['image_sha256'],
                label_sha256=self.raw[row['donor_case_id']]['label_sha256'],
                mask_sha256=source.v24_mask_sha256),
            prototype_bank_sha256=self.bank_sha256,config_sha256=self.config_sha256,source_sha256=self.source_sha256)

    def guard_inputs(self,plan,binding):
        self.guard_input_files(plan.case_id,plan.query_rows[0]['donor_case_id']); self._rss()
        if (binding['source_sha256']!=self.source_sha256 or binding['config_sha256']!=self.config_sha256
                or binding['prototype_bank_sha256']!=self.bank_sha256
                or binding['recipient']['case_id']!=plan.case_id):
            raise ValueError('Actual GT-free source/config/bank/recipient binding changed')

    def _upper(self,plan,provider=None):
        from .v24_geometry import build_upper_graphs
        raw=self._case(plan.case_id); donor,source,_=self._donor(plan.query_rows[0])
        return build_upper_graphs(raw['context'],donor,source,self._region(plan.case_id),
            self._region(plan.query_rows[0]['donor_case_id']),plan.query_rows,self.bundle.prototype_bank,
            config=self.graph_config,ct_clip=self.ct_clip,training_case_ids=self.training_cases)

    def _local(self,row):
        from hiercp.common import stable_case_seed
        raw=self._case(row['case_id']); donor,source,prepared=self._donor(row)
        return build_local_record(raw['context'],donor,source,prepared,row,
            config=self.graph_config,seed=self.config['seed'],
            ct_clip=self.ct_clip,scope_contract=self.bundle.scope['contract_sha256'],
            donor_mask_sha256=source.v24_mask_sha256)

    def prepare_local(self):
        from hiercp_v22.storage import GraphWriter
        root=self.root/'local'; root.mkdir(exist_ok=True); index=root/'index.json'
        expected=dict(format=CANONICAL_FORMAT,recipient_GT_used_in_forward=False,debug=False,
            query_rows_sha256=canonical_hash([query_inputs([row])[0] for row in self.inventory['records']]),
            config_sha256=self.config_sha256,source_sha256=self.source_sha256,
            prototype_bank_sha256=self.bank_sha256)
        if index.exists():
            value=json.loads(index.read_text(encoding='utf8'))
            if any(value.get(key)!=item for key,item in expected.items()) or value.get('complete') is not True:
                raise ValueError('Existing GT-free canonical namespace does not match actual query/source')
            return index
        writer=GraphWriter(root,minimum_free_bytes=self.runtime['minimum_free_disk_gib']*2**30)
        # Serialize each case's preparation, with original independent rows
        # built in the persistent CPU pool. All14102 records are retained.
        stored=[]
        with ThreadPoolExecutor(max_workers=self.runtime['workers']) as pool:
            for case in self.population.partition_cases('inner_train')+self.population.partition_cases('inner_val'):
                plan=self.population.case(case,128)
                rows=list(plan.query_rows)
                binding=self.input_binding(plan)
                def write(row):
                    key=canonical_hash(dict(query=query_inputs([row])[0],inputs=binding)); relative='records/'+key+'.pt.gz'
                    path=root/relative
                    if path.exists():
                        from hiercp_v22.storage import load_record
                        record=load_record(root,relative)
                        if query_inputs([dict(row,id=record['observation_id'],center=record['center'])])[0]!=query_inputs([row])[0]:
                            raise ValueError('Partial owned canonical query differs')
                        # Read-only recovery verifies actual original record,
                        # never silently creates or replaces missing content.
                        from .v24_inputs import tensor_digest
                        if tensor_digest({k:record[k] for k in ('source_patch','target_patch','source_local','target_local')})!=record['tensor_sha256']:
                            raise ValueError('Partial canonical tensor proof differs')
                        import gzip,io
                        with gzip.open(path,'rb') as stream: partial=torch.load(io.BytesIO(stream.read()),map_location='cpu',weights_only=False)
                        entry=dict(path=relative,sha256=sha(path),shared_source=partial['shared_source'])
                    else:
                        record=self._local(row)
                        source_key=(row['donor_case_id'],row['donor_component'],tuple(self.raw[case]['spacing']))
                        entry=writer.write(relative,record,source_key)
                    views,_,_=materialize_pair(record,epoch=0)
                    return dict(id=row['id'],query=query_inputs([row])[0],tensor_sha256=record['tensor_sha256'],
                        sampled_two_view_nodes=sum(sum(store.num_nodes for store in graph.node_stores) for graph in views),
                        sampled_two_view_edges=sum(sum(store.num_edges for store in graph.edge_stores) for graph in views),**entry)
                entries=list(pool.map(write,rows)); stored.extend(entries)
                print(f'v24 GT-free CPU canonical {case}: {len(stored)}/{len(self.inventory["records"])} records',flush=True)
                self._rss()
        if {row['id'] for row in stored}!={row['id'] for row in self.inventory['records']} or len(stored)!=len(self.inventory['records']):
            raise ValueError('Full GT-free canonical preparation omitted or duplicated native records')
        _publish(index,dict(**expected,records=sorted(stored,key=lambda row:row['id']),complete=True))
        return index

    def close(self):
        self._raw_cache.clear(); self._donors.clear(); self._regions.clear()


def build_runtime(args, experiment_config, budget):
    """Actual fresh net/scorer/population/config/close callable; no placeholders."""
    import random
    from .v24_model import build_gt_free_model
    from .v24_training import V24Scorer
    from .u_bridge_training import digest
    runtime=copy.deepcopy(experiment_config['v24_runtime'])
    inputs=V24NativeInputs(args.native_experiment,args.inventory,args.input_cache,runtime)
    config=copy.deepcopy(inputs.config); config['v24_runtime']=runtime
    config['training'].update(epochs=40,lr=1e-4,consistency_weight=.1,gradient_accumulation_steps=1,fixed_validation_epoch=29)
    from hiercp.tensor import configure_runtime
    configure_runtime(**{key:config['runtime'][key] for key in ('deterministic','allow_tf32','cudnn_benchmark')})
    torch.manual_seed(42); np.random.seed(42); random.seed(42)
    net=build_gt_free_model(config['model'],original_snapshot_root=inputs.bundle.source)
    contract=copy.deepcopy(net.v24_model_contract)
    contract.update(trained_v23_weights_loaded=False,initialization_seed=42,optimizer='fresh AdamW',lr=1e-4)
    if experiment_config['encoder']=='official_pretrained_STU_Net_S':
        if args.stunet_checkpoint is None: raise ValueError('Official SHA-bound STU-Net-S checkpoint required; no random substitute')
        from .v24_stunet import install_stunet_l0
        audit=install_stunet_l0(net,args.stunet_checkpoint)
        contract.update(encoder='official_pretrained_STU_Net_S',STU_Net=audit,
            parameters=sum(p.numel() for p in net.parameters()),trainable_parameters=sum(p.numel() for p in net.parameters() if p.requires_grad),
            local_GNN_message_passing=False,encoder_fine_tuned=True,encoder_lr=1e-4)
    elif experiment_config['encoder']=='original_CNN_GAT':
        contract.update(encoder='original_CNN_GAT',local_GNN_message_passing=True)
    else: raise ValueError('Explicit authorized v24 encoder required')
    net.to('cuda'); inputs.guard_source()
    index=Path(args.input_cache)/'local/index.json'
    coordinator=V24InputCoordinator(budget.rss_bytes)
    inputs.memory_coordinator=coordinator
    inputs.geometry.memory_guard=inputs._rss
    providers={partition:V24InputProvider(SimpleNamespace(rows=[row for row in inputs.inventory['records']
        if row['case_id'] in inputs.inventory['split'][partition]],meta=inputs.inventory,partition=partition),
        index,workers=runtime['workers'],resident_bytes=runtime['resident_gib']*2**30,coordinator=coordinator)
        for partition in ('inner_train','inner_val')}
    inputs.geometry.admit(runtime['curriculum']['initial_u']); inputs.geometry.admit(128)
    scorer=V24Scorer(net,providers,inputs.geometry,
        physical_candidate_batch=min(runtime['physical_candidate_batch_candidates']),
        checkpoint_local_chunks=True,amp=bool(config['training']['amp']),budget=budget,
        prefetch_cpu_chunks=True,pin_cpu_batches=True)
    config['v24_runtime']['initial_model_sha256']=digest(net.state_dict())
    def close():
        for provider in providers.values(): provider.close()
        inputs.close()
    return net,scorer,inputs.population,config,contract,close
