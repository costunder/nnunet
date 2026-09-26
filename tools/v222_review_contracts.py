"""Explicit feature-coordinate and support-task contracts after independent review.

Preserved research/cache modules are not rewritten. Existing normalized input
grids stay valid; only their mapping to the CNN lattice changes in new runs.
Legacy checkpoints must retain their original coordinate interpretation.
"""
from contextlib import contextmanager, ExitStack
from unittest.mock import patch
from types import MethodType
import torch

FEATURE_CONTRACTS=('legacy','stride4')
TASK_CONTRACT='patient_group_v1'


def resolve_feature_contract(saved=None, requested=None):
    if requested is not None and requested not in FEATURE_CONTRACTS:
        raise ValueError('Unknown feature coordinate contract')
    if saved is None:return requested or 'stride4'
    policy=saved.get('execution_policy') or {}
    previous=policy.get('feature_coordinates',saved.get('feature_coordinates','legacy'))
    if 'feature_coordinates' in saved and saved['feature_coordinates']!=previous:
        raise ValueError('Conflicting saved feature coordinate contracts')
    if previous not in FEATURE_CONTRACTS:raise ValueError('Unknown saved feature coordinate contract')
    if requested is not None and requested!=previous:
        raise ValueError('Feature coordinates change model inputs; start a separate training run, not an exact resume')
    return previous


def validate_group_resume(saved, meta):
    """Unique public case groups are unchanged; old multi-scan plans are not."""
    if (saved.get('execution_policy') or {}).get('support_task_contract')==TASK_CONTRACT:return
    groups=[meta['identities']['cases'][case]['patient_group'] for case in meta['split']['inner_train']]
    if len(set(groups))!=len(groups):
        raise ValueError('Legacy multi-case patient support plan cannot be relabelled; start a separate training run')


def cnn_lattice(encoder,input_shape=(48,48,48)):
    """Nominal center origin/jump from the actual sequential Conv3d geometry."""
    shape=list(input_shape);jump=[1,1,1];origin=[0.,0.,0.]
    for layer in encoder.modules():
        if isinstance(layer,torch.nn.Conv3d):
            for axis in range(3):
                k,s,p,d=(getattr(layer,name)[axis] for name in ('kernel_size','stride','padding','dilation'))
                origin[axis]+=((k-1)*d/2-p)*jump[axis]
                jump[axis]*=s
                shape[axis]=(shape[axis]+2*p-d*(k-1)-1)//s+1
    if (tuple(shape),tuple(jump),tuple(origin))!=((12,12,12),(4,4,4),(0.,0.,0.)):
        raise ValueError('CNN lattice changed; review the feature coordinate contract')
    return tuple(shape),tuple(jump),tuple(origin)


def input_grid_to_feature_grid(grid, *, input_shape=(48,48,48), feature_shape=(12,12,12), stride=(4,4,4), origin=(0.,0.,0.)):
    """XYZ normalized input -> input voxel -> feature index -> normalized XYZ.

    Border clamping is performed by the existing sampler. Input sites beyond
    the final nominal center44 use the last feature; they are not extrapolated.
    """
    if grid.ndim!=2 or grid.shape[1]!=3 or not grid.is_floating_point():
        raise ValueError('Floating normalized XYZ coordinates required')
    if min(feature_shape)<=1 or min(stride)<=0:raise ValueError('Nondegenerate feature lattice required')
    vector=lambda values:torch.tensor(tuple(reversed(values)),device=grid.device,dtype=torch.float32)
    position=(grid.float()+1)*.5*(vector(input_shape)-1)
    feature=(position-vector(origin))/vector(stride)
    return 2*feature/(vector(feature_shape)-1)-1


def grouped_support(memory,query_group):
    """A patient with two scans owns one task; both paired identities excluded."""
    groups=memory['patient_groups'];donors=memory.get('donor_groups')
    owners=memory['owners']
    if donors is None or len(donors)!=len(owners):raise ValueError('Per-record donor groups required')
    # Exactly preserve the existing public-case path and its order.
    if len(set(groups))==len(groups):
        return _original_support(memory,query_group)
    unique=sorted(set(groups));group_ids={group:i for i,group in enumerate(unique)}
    case_to_group=torch.tensor([group_ids[g] for g in groups],device=owners.device)
    task=case_to_group[owners]
    keep=torch.tensor([g!=query_group for g in donors],device=owners.device,dtype=torch.bool)
    allowed=torch.tensor([g!=query_group for g in groups],device=owners.device)
    keep=keep & allowed[owners]
    used=torch.unique(task[keep],sorted=True)
    if len(used)<2:raise ValueError('Two other distinct patient groups required after recipient/donor exclusion')
    compact=torch.full((len(unique),),-1,device=owners.device,dtype=torch.long)
    compact[used]=torch.arange(len(used),device=owners.device)
    return memory['embeddings'][keep],compact[task[keep]],memory['classes'][keep]


from hiercp_v222.v1_local import support_for_recipient as _original_support


@contextmanager
def installed(feature_coordinates):
    if feature_coordinates not in FEATURE_CONTRACTS:raise ValueError('Unknown coordinate contract')
    from hiercp_v222 import v1_local,v1_execution,v1_training
    from hiercp_v222.deterministic_sampling import sample_nodes
    original_init=v1_local.V1LocalEncoder.__init__
    def initialize(self,base):
        original_init(self,base)
        self.feature_coordinate_contract=feature_coordinates
        self.feature_lattice=cnn_lattice(self.dense_encoder)
        self._sample_dense_features=MethodType(sample,self)
    def sample(self,feature_map,grid,node_batch):
        shape,jump,origin=self.feature_lattice
        if tuple(feature_map.shape[2:])!=shape:raise ValueError('Feature map and declared lattice differ')
        return sample_nodes(feature_map,input_grid_to_feature_grid(grid,feature_shape=shape,stride=jump,origin=origin),node_batch)
    with ExitStack() as stack:
        for module in (v1_local,v1_execution,v1_training):
            stack.enter_context(patch.object(module,'support_for_recipient',grouped_support))
        if feature_coordinates=='stride4':
            stack.enter_context(patch.object(v1_local.V1LocalEncoder,'__init__',initialize))
        yield
