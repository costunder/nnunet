"""PRODIGY-style episode relation construction, independent of CP geometry.

The caller must supply defined episode classes and support class identities.
Random label embeddings do not generate class identities. Query targets are
deliberately absent from this API. This does not define the medical CP task.
"""
from dataclasses import dataclass
import torch

U, F, T = -1, 0, 1
RELATION_CONTRACT = 'per_data_label_T_relation_F_nonrelation_U_unknown'


@dataclass(frozen=True)
class PromptRelations:
    states: torch.Tensor
    edge_index: torch.Tensor
    edge_features: torch.Tensor
    support_count: int
    query_count: int
    class_count: int


def from_episode_classes(support_labels, *, class_count, query_count):
    """Build all episode edges; class labels are supplied, never inferred from CT.

    Data nodes precede label nodes. Support edges go both ways; query edges
    go label -> data only. Features are [is_support, is_true_relation]. A
    query edge [0,0] is UNKNOWN, not a known false edge [1,0].
    """
    if type(class_count) is not int or class_count < 2:
        raise ValueError('At least two explicitly defined episode classes required')
    if type(query_count) is not int or query_count < 1:
        raise ValueError('Positive query count required')
    if not torch.is_tensor(support_labels) or support_labels.dtype!=torch.long or support_labels.ndim!=1:
        raise ValueError('Support class IDs must be a 1-D int64 tensor')
    if not support_labels.numel() or bool(((support_labels<0)|(support_labels>=class_count)).any()):
        raise ValueError('Support class ID outside the defined episode classes')
    if bool((torch.bincount(support_labels,minlength=class_count)==0).any()):
        raise ValueError('Each episode class requires an actual support example')
    device=support_labels.device; ns=support_labels.numel(); n=ns+query_count
    classes=torch.arange(class_count,device=device)
    known=(support_labels[:,None]==classes[None]).to(torch.int8)
    unknown=torch.full((query_count,class_count),U,device=device,dtype=torch.int8)
    states=torch.cat((known,unknown))
    data=torch.arange(n,device=device).repeat_interleave(class_count)
    labels=classes.repeat(n)+n
    support=data<ns
    feature=torch.stack((support,states.flatten()==T),-1).float()
    # Every label informs support/query; only labeled support informs labels.
    backward=torch.stack((labels,data))
    forward=torch.stack((data[support],labels[support]))
    return PromptRelations(states,torch.cat((backward,forward),1),
                           torch.cat((feature,feature[support])),ns,query_count,class_count)
