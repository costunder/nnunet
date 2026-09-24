"""Support-only, class-conditional prototype assignment (project design, not SwAV).

No query arguments. Assignment is a detached teacher operation once per patient
episode, not a per-query CPU operation. All evidence-bearing patient labels are
retained. See docs/pipeline_v222_cluster_r2.md and REFERENCES.md.
"""
import time
import numpy as np
import torch
from torch.nn import functional as F
from scipy.cluster.hierarchy import linkage, cut_tree
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import silhouette_score


def select_partition(vectors):
    """Cosine average-linkage; best positive silhouette among non-singleton cuts.

    K=1 is an explicit unresolved-mode outcome, not an OOM/error fallback.
    At least two independent patients must support each discovered submode.
    All feasible K are inspected; no user-invisible cluster or sample cap.
    """
    x = np.asarray(vectors, dtype=np.float64)
    if x.ndim != 2 or not len(x) or not np.isfinite(x).all():
        raise ValueError('Finite nonempty patient-label matrix required')
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if (norms <= np.finfo(np.float64).eps).any():
        raise ValueError('Zero patient label cannot define a cosine prototype')
    x = x/norms
    best = np.zeros(len(x), dtype=np.int64); best_score = 0.0
    candidates = []
    distances = np.maximum(pdist(x, metric='cosine'), 0)
    reason = 'insufficient_patients_for_two_non_singleton_modes'
    if len(x) >= 4 and distances.max() > np.finfo(np.float64).eps:
        tree = linkage(distances, method='average')
        matrix = squareform(distances)
        reason = 'no_positive_silhouette_non_singleton_cut'
        for k in range(2, len(x)//2+1):
            labels = cut_tree(tree, n_clusters=k).ravel()
            counts = np.bincount(labels)
            if counts.min() < 2:
                candidates.append(dict(k=k, eligible=False, counts=counts.tolist()))
                continue
            score = float(silhouette_score(matrix, labels, metric='precomputed'))
            candidates.append(dict(k=k, eligible=True, counts=counts.tolist(), silhouette=score))
            if score > best_score:  # Equal scores retain the smaller K.
                best, best_score, reason = labels.copy(), score, 'maximum_positive_silhouette'
    elif len(x) >= 4:
        reason = 'identical_directions_no_resolved_modes'
    return best, dict(patients=len(x), k=int(best.max())+1,
                      silhouette=best_score if best.max() else None,
                      reason=reason, candidates=candidates,
                      mean_direction_variance=float(np.var(x, axis=0).sum()))


def fit_prototypes(local_labels, owners, classes):
    """Detached local L1 labels -> observed-class memberships and fixed teachers."""
    start = time.perf_counter()
    labels = local_labels.detach().float()
    p, c, d = labels.shape
    if c != 2 or owners.shape != classes.shape:
        raise ValueError('Two observation classes and explicit support ownership required')
    counts = torch.bincount(owners*2+classes, minlength=p*2).reshape(p, 2)
    active = (counts > 0).flatten().nonzero().flatten()
    # One intentional transfer per episode; hierarchy operates on patient labels,
    # not the millions of spatial graph edges or individual query patches.
    cpu = labels.reshape(-1,d)[active].cpu().numpy()
    active_cpu = active.cpu().numpy()
    assignments = np.empty(len(active_cpu), dtype=np.int64)
    prototype_classes = []; audits = []; offset = 0
    for cls in (0,1):
        positions = np.flatnonzero(active_cpu % 2 == cls)
        if not len(positions):
            raise ValueError('Both observation classes require real support evidence')
        membership, audit = select_partition(cpu[positions])
        assignments[positions] = membership+offset
        prototype_classes.extend([cls]*audit['k']); offset += audit['k']
        audits.append(dict(observation_class=cls, **audit))
    assignment = torch.as_tensor(assignments, device=labels.device)
    pc = torch.tensor(prototype_classes, device=labels.device)
    membership = F.one_hot(assignment, num_classes=offset).float()
    mass = membership.sum(0)
    with torch.autocast(device_type=labels.device.type, enabled=False):
        centers = membership.T @ F.normalize(labels.reshape(-1,d)[active],dim=-1)/mass[:,None]
    if bool((centers.norm(dim=-1) <= torch.finfo(centers.dtype).eps).any()):
        raise ValueError('Cancelling support vectors cannot define a spherical center')
    centers = F.normalize(centers,dim=-1).detach()
    class_mask = pc[:,None] == torch.arange(2,device=labels.device)[None]
    class_mass = (mass[:,None]*class_mask).sum(0)
    weights = mass / class_mass[pc]
    with torch.autocast(device_type=labels.device.type, enabled=False):
        similarity = centers @ centers.T
    off_diagonal = ~torch.eye(offset, device=labels.device, dtype=torch.bool)
    return dict(active=active, assignment=assignment, membership=membership, centers=centers,
                prototype_classes=pc, mass=mass, log_prior=weights.log(),
                owners=owners.detach().clone(), classes=classes.detach().clone(),
                alignment_row_weights=(1/(2*torch.bincount(active%2,minlength=2).float()))[active%2],
                audit=dict(method='class_conditional_cosine_average_linkage_silhouette',
                           fit_scope='explicit_other_patient_support_only',
                           classes=audits, prototype_count=offset, counts=mass.tolist(),
                           active_patient_labels=len(active_cpu), missing_class_labels_excluded=p*2-len(active_cpu),
                           maximum_distinct_center_cosine=float(similarity[off_diagonal].max()),
                           seconds=time.perf_counter()-start))


def live_prototypes(aligned_labels, plan):
    with torch.autocast(device_type=aligned_labels.device.type, enabled=False):
        live = F.normalize(aligned_labels.flatten(0,1).float()[plan['active']],dim=-1)
        return F.normalize(plan['membership'].T @ live / plan['mass'][:,None],dim=-1)


def alignment_loss(aligned_labels, plan, temperature):
    live = F.normalize(aligned_labels.flatten(0,1).float()[plan['active']],dim=-1)
    with torch.autocast(device_type=live.device.type, enabled=False):
        logits = live @ plan['centers'].T / temperature
    # Competing modes AND the other observed class are negatives. Thus even K=1
    # within each class has a nontrivial two-center discriminative objective.
    losses = F.cross_entropy(logits, plan['assignment'], reduction='none')
    return (losses*plan['alignment_row_weights']).sum()


def prototype_logits(query, aligned_labels, plan, temperature):
    with torch.autocast(device_type=query.device.type, enabled=False):
        scores = F.normalize(query.float(),dim=-1) @ live_prototypes(aligned_labels,plan).T / temperature
    scores = scores+plan['log_prior']
    # Mixture normalized within each class: more clusters alone cannot raise its
    # score, and one patient's label has the same mass regardless of lesion count.
    mask = plan['prototype_classes'][:,None] == torch.arange(2,device=query.device)[None]
    return torch.logsumexp(scores[:,:,None].masked_fill(~mask[None],-torch.inf),dim=1)
