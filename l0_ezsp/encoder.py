"""Experimental pinned-EZ-SP L0. No production registration or silent fallback."""
from contextlib import contextmanager
from time import perf_counter
import torch
from hiercp_v222.v1_local import V1LocalEncoder
from hiercp_v22.schema import LOCAL_NODE_TYPES,LOCAL_EDGE_TYPES,SOURCE_LOCAL_NODE_TYPES
from hiercp_v222.deterministic_sampling import sample_nodes
from tools.v222_review_contracts import cnn_lattice,input_grid_to_feature_grid
from .config import validate,fingerprint
from .data import FineBatch
from .partition import Region,partition,CoarseningBudgetError,CoarseningConstraintError
from .ops import undirected,quotient,mass_mean,mass_readout

class EZSPEncoder(V1LocalEncoder):
    def __init__(self,base,profile):
        super().__init__(base)
        self.profile=validate(profile)
        self.profile_sha256=fingerprint(profile)
        self.feature_lattice=cnn_lattice(self.dense_encoder)
        self.timing_enabled=False
        self.last_audit={}
        self.last_topology={}
        self.detailed_diagnostics=False

    def _admission(self,stats,budget,violations):
        if budget or violations:
            error=CoarseningBudgetError if budget or any('role_shell_nodes' in v for v in violations.values()) else CoarseningConstraintError
            raise error('Unvalidated EZ-SP profile admission rejected; no repair/fallback',stats)

    def get_extra_state(self):
        from .identity import identity
        return identity(self.profile)

    def set_extra_state(self,state):
        from .identity import require_same_experiment
        require_same_experiment({'l0_ezsp_identity':state},self.profile)

    def _load_from_state_dict(self,state_dict,prefix,local_metadata,strict,missing_keys,unexpected_keys,error_msgs):
        # nn.Module.load_state_dict on a parent bypasses the child's public loader.
        # Enforce identity here even when callers use strict=False.
        key=prefix+'_extra_state'
        if key not in state_dict:raise ValueError('Legacy L0 weights cannot be an exact EZ-SP resume; use explicit CNN-only initialization')
        self.set_extra_state(state_dict[key])
        return super()._load_from_state_dict(state_dict,prefix,local_metadata,strict,missing_keys,unexpected_keys,error_msgs)

    @contextmanager
    def timed(self,name):
        if self.timing_enabled:torch.cuda.synchronize();start=perf_counter()
        try:yield
        finally:
            if self.timing_enabled:
                torch.cuda.synchronize()
                self.last_audit.setdefault('seconds',{})[name]=perf_counter()-start

    def _coarsen(self,x,regions,edges,level,graph_count):
        with self.timed(f'partition_s{level}'):
            parts={k:partition(x[k],regions[k],role=k,level=level,cfg=self.profile,diagnostics=self.detailed_diagnostics,measure=self.timing_enabled) for k in LOCAL_NODE_TYPES}
        coarse={k:p.region for k,p in parts.items()}
        stats={'roles':{k:p.stats for k,p in parts.items()},'relations':{}}
        if self.detailed_diagnostics:
            for role,p in parts.items():
                groups=p.stats['group_diagnostics'];present={(d['owner'],d['shell']) for d in groups}
                for owner in range(graph_count):
                    for shell in (range(3) if 'context' in role else (-1,)):
                        if (owner,shell) not in present:
                            groups.append(dict(owner=owner,shell=shell,input_nodes=0,clusters=0,
                                connected_components_before=0,connected_components_after=0,cluster_size_histogram={},cluster_size_quantiles=[],
                                bbox_excess=dict(clusters=0,fine_nodes=0,fine_node_fraction=0.),
                                variance_excess=dict(clusters=0,fine_nodes=0,fine_node_fraction=0.),
                                either_excess=dict(clusters=0,fine_nodes=0,fine_node_fraction=0.),termination_condition='empty_existing_shell'))
        self.last_audit[f'scale{level}']=stats
        edge_out={};witnesses={}
        with self.timed(f'quotient_admission_s{level}'):
            edge_counts=torch.zeros(graph_count,device=next(iter(x.values())).device,dtype=torch.long)
            for kind in LOCAL_EDGE_TYPES:
                a,_,b=kind;e=edges[kind]
                if bool((regions[a].owner[e[0]]!=regions[b].owner[e[1]]).any()):
                    raise CoarseningConstraintError('Cross-pair input relation',stats)
                q,count,witness,collapsed=quotient(e,parts[a].parent,parts[b].parent,same_type=a==b)
                edge_out[kind]=q;witnesses[kind]=(count,witness)
                edge_counts+=torch.bincount(coarse[a].owner[q[0]],minlength=graph_count)
                stats['relations']['|'.join(kind)]={'input':e.shape[1],'output':q.shape[1],
                    'collapsed_internal':int(collapsed),'multiplicity_sum':int(count.sum())}
            node_counts=sum((torch.bincount(r.owner,minlength=graph_count) for r in coarse.values()))
            stats['nodes_per_pair']=node_counts.tolist();stats['edges_per_pair']=edge_counts.tolist()
            limits=self.profile['diagnostic_profile_from_original_NOT_VALIDATED'];j=level-1
            budget=bool((node_counts>limits['nodes_total'][j]).any() | (edge_counts>limits['directed_edges_total'][j]).any())
            violations={k:p.violations for k,p in parts.items() if p.violations}
            hard={k:[v for v in values if v not in ('bbox','variance','role_shell_nodes')] for k,values in violations.items()}
            if any(hard.values()):raise CoarseningConstraintError('Structural partition integrity failed',{'hard':hard})
            def owner_mass(r):
                # Weighted CUDA bincount rejects the reviewed deterministic
                # scorer runtime. index_add keeps the exact same coverage sum.
                return r.mass.new_zeros(graph_count).index_add(0,r.owner,r.mass)
            for k,p in parts.items():
                if not torch.equal(owner_mass(regions[k]),owner_mass(p.region)):
                    raise CoarseningConstraintError('Fine node mass coverage lost',stats)
            stats['violations']=violations;stats['total_budget_failed']=budget
            # Retained only until the next forward for inspection, never checkpointed.
            self.last_topology[level]={'parents':{k:p.parent for k,p in parts.items()},'edges':edge_out,'witnesses':witnesses}
            self._admission(stats,budget,violations)
        with self.timed(f'live_aggregate_s{level}'):
            pooled={k:mass_mean(x[k],p.parent,regions[k].mass)[0] for k,p in parts.items()}
        return pooled,coarse,edge_out

    def _readout(self,x,regions,count):
        result={}
        for k,r in regions.items():
            if 'context' not in k:
                result[k]=mass_readout(x[k],r.mass,r.owner,self.pool[k].gate_nn,count)
            else:
                for shell in range(3):
                    key=f'{k}_c{shell}';mask=r.shell==shell
                    values=mass_readout(x[k][mask],r.mass[mask],r.owner[mask],self.context_shell_pool[key].gate_nn,count)
                    present=torch.bincount(r.owner[mask],minlength=count)>0
                    result[key]=torch.where(present[:,None],values,self.empty_context_shell[key].to(values.dtype)[None])
        return result

    def forward_fields(self,batch):
        if not isinstance(batch,FineBatch):raise TypeError('EZ-SP requires separate physical sidecar; no legacy relabel')
        from .validation import validate_batch
        validate_batch(batch)
        graph=batch.graph
        if set(graph.node_types)!=set(LOCAL_NODE_TYPES) or set(graph.edge_types)!=set(LOCAL_EDGE_TYPES):raise ValueError('Unexpected fine schema')
        self.last_audit={'training_ready':False,'profile_sha256':self.profile_sha256};self.last_topology={}
        with self.timed('cnn'):
            source,target=self.encode_dense_maps(batch.source_patches,batch.source_index,batch.target_patches)
        x={};regions={};shape,jump,origin=self.feature_lattice
        with self.timed('sample_and_input_adjacency'):
            for k in LOCAL_NODE_TYPES:
                node=graph[k];meta=batch.sidecar[k]
                if {'x','pos','pos_mm','observed'}.intersection(node.keys()):raise ValueError('Metadata must remain outside learned inputs')
                feature=source if k in SOURCE_LOCAL_NODE_TYPES else target
                if tuple(feature.shape[2:])!=shape:raise ValueError('Unexpected CNN lattice')
                grid=input_grid_to_feature_grid(node.grid,feature_shape=shape,stride=jump,origin=origin)
                x[k]=sample_nodes(feature,grid,node.batch)
                shell=node.shell_id.long() if 'context' in k else torch.full_like(node.batch,-1)
                if 'context' in k and bool(((shell<0)|(shell>2)).any()):raise ValueError('Invalid shell')
                same=next(e for e in LOCAL_EDGE_TYPES if e[0]==k and e[2]==k)
                adj,w=undirected(graph[same].edge_index,node.batch*4+shell+1)
                pos=meta['pos_mm']
                regions[k]=Region(node.batch,shell,torch.ones(len(node.batch),device=pos.device),pos,pos,meta['stable_id'],adj,w)
            if any('edge_attr' in graph[e] for e in LOCAL_EDGE_TYPES):raise ValueError('No learned geometric edge attributes')
        x,r1,e1=self._coarsen(x,regions,graph.edge_index_dict,1,len(batch))
        with self.timed('project_gat2'):
            x={k:self.project[k](v) for k,v in x.items()}
            for block in self.blocks[:2]:x=self._run_local_block(block,x,e1,{e:None for e in e1})
        with self.timed('readout_s1'):read1=self._readout(x,r1,len(batch))
        x,r2,e2=self._coarsen(x,r1,e1,2,len(batch))
        with self.timed('gat1'):x=self._run_local_block(self.blocks[2],x,e2,{e:None for e in e2})
        with self.timed('readout_fuse'):
            read2=self._readout(x,r2,len(batch));pooled={k:(v+read2[k])*.5 for k,v in read1.items()}
            source_shells=[pooled[f'source_context_c{i}'] for i in range(3)]
            target_shells=[pooled[f'target_context_c{i}'] for i in range(3)]
            sc=self.context_shell_fuse['source_context'](torch.cat(source_shells,-1))
            tc=self.context_shell_fuse['target_context'](torch.cat(target_shells,-1))
            tumor=self.tumor_fuse(pooled['tumor_surface'])
            source_context=self.source_context_fuse(self._pair(sc,pooled['source_liver_surface']))
            target_context=self.target_context_fuse(self._pair(tc,pooled['target_liver_surface']))
            sr=self.source_relation(self._pair(tumor,source_shells[0]));tr=self.target_relation(self._pair(tumor,target_shells[0]))
            fused=self.final_fuse(torch.cat((tumor,source_context,target_context,sr,tr,(sr-tr).abs()),-1))
        return {'fused':fused,'tumor':tumor,'source_context':source_context,'target_context':target_context,
            'source_relation':sr,'target_relation':tr,**{f'source_c{i}':v for i,v in enumerate(source_shells)},**{f'target_c{i}':v for i,v in enumerate(target_shells)}}

    def forward(self,batch):return self.forward_fields(batch)['fused']

def build_model(cfg,base,profile):
    """Explicit experimental injection; L1/L2/loss code remains unchanged."""
    from hiercp_v222.model import PromptGraphModel
    return PromptGraphModel(cfg,base,{},local_encoder=EZSPEncoder(base,profile))
