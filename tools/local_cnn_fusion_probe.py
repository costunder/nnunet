"""Fixed-weight fusion counterfactuals with identically re-encoded support/query.

Only detached 128D recipient projections and fused vectors are cached, not CT
volumes or feature maps. The production branch is Fuse(...); no residual lambda
is labelled production-compatible. Each branch refits its own support plan.
"""
import math
import time

import torch

from l0_local_cnn.model import LocalBatch
from l0_regions.resident import check_verified
from tools.local_cnn_l1_probe import candidate_signal, _score_signal


def validate_scales(scales):
    values = [float(x) for x in scales]
    if not values or len(set(values)) != len(values) or any(not math.isfinite(x) or x < 0 for x in values):
        raise ValueError('Distinct explicit finite nonnegative fusion scales required')
    return values


@torch.no_grad()
def fusion_basis(local, batch):
    """One unchanged production CNN pass for r and Fuse(d,r,r-d,r*d)."""
    if not isinstance(batch, LocalBatch):
        raise TypeError('Original native local CT batch required')
    check_verified(batch)
    if any(module.training for module in local.modules()):
        raise ValueError('Read-only fusion diagnosis requires eval mode')
    if not bool((batch.donor == batch.donor[0]).all()):
        raise ValueError('Same-donor physical candidate batch required')
    if local.resource_budget:
        local.resource_budget.check()
    pooled = []
    for x, mask in local.cnn(batch.images, batch.organ):
        count = mask.sum((2, 3, 4))
        if not bool((count > 0).all()):
            raise ValueError('Organ vanished at a scale; no sample skip or zero fallback')
        pooled.append(torch.where(mask, x, 0).sum((2, 3, 4)) / count)
    features = local.project(torch.cat(pooled, 1))
    donor, recipient = features[batch.donor], features[batch.recipient]
    fused = local.fuse(torch.cat((donor, recipient, recipient-donor, recipient*donor), 1))
    _check_basis(recipient, fused)
    if local.resource_budget:
        local.resource_budget.check()
    return recipient.detach(), fused.detach()


def _check_basis(recipient, fused):
    if (recipient.ndim != 2 or recipient.shape != fused.shape or recipient.shape[1] != 128
            or not len(recipient) or recipient.device != fused.device
            or not bool(torch.isfinite(recipient).all() & torch.isfinite(fused).all())):
        raise ValueError('Matching finite nonempty 128D fusion basis required')
    if recipient.requires_grad or fused.requires_grad:
        raise ValueError('Fusion counterfactual basis must be detached')


def support_indices(rows, query_group):
    """Exactly the production full eligible set: exclude query on both sides."""
    indices = [i for i, row in enumerate(rows)
               if query_group not in (row['patient_group'], row['donor_group'])]
    groups = sorted({rows[i]['patient_group'] for i in indices})
    if len(groups) < 2 or {rows[i]['target'] for i in indices} != {0, 1}:
        raise ValueError('Full eligible support needs two other groups and both observed classes')
    return indices


class BasisBank:
    """Explicit record binding, exact coverage; no checkpoint or ready file."""
    def __init__(self, rows):
        if len({r['id'] for r in rows}) != len(rows):
            raise ValueError('Duplicate support record identity')
        self.rows = rows
        self.values = {}

    def add(self, indices, recipient, fused):
        _check_basis(recipient, fused)
        if len(indices) != len(recipient) or len(set(indices)) != len(indices):
            raise ValueError('Distinct exactly bound fusion indices required')
        if any(type(i) is not int or not 0 <= i < len(self.rows) for i in indices):
            raise ValueError('Fusion record index outside original inventory')
        recipient, fused = recipient.cpu(), fused.cpu()
        for j, i in enumerate(indices):
            if i in self.values:
                raise ValueError('Repeated fusion encoding; cached vectors must be reused')
            self.values[i] = (recipient[j].clone(), fused[j].clone())

    def get(self, indices, device):
        if len(set(indices)) != len(indices) or any(i not in self.values for i in indices):
            raise ValueError('Incomplete or duplicated fusion support coverage')
        return tuple(torch.stack([self.values[i][column] for i in indices]).to(device)
                     for column in (0, 1))


def missing_batches(rows, required, bank, batch):
    """Encode only missing rows, grouped by the original case and donor."""
    if type(batch) is not int or batch < 1:
        raise ValueError('Original physical batch required')
    required = sorted(set(required))
    if any(type(i) is not int or not 0 <= i < len(rows) for i in required):
        raise ValueError('Support requirement outside original inventory')
    by_case = {}
    for i in required:
        if i not in bank.values:
            row = rows[i]
            key = (row['case_id'], row['donor_case_id'], row['donor_component'])
            by_case.setdefault(key, []).append(i)
    return [ids[start:start+batch] for ids in by_case.values() for start in range(0, len(ids), batch)]


@torch.no_grad()
def encode_missing(local, loader, bank, required, batch, budget):
    """One pass over the complete union of required support, with progress."""
    from tqdm import tqdm
    order = missing_batches(bank.rows, required, bank, batch)
    total = sum(map(len, order))
    reused = sum(i in bank.values for i in set(required))
    started = time.perf_counter()
    times = dict(loader_wait_seconds=0., transfer_seconds=0., l0_basis_seconds=0., cpu_vector_copy_seconds=0.)
    bar = tqdm(total=total, desc='DIAGNOSTIC fusion support re-encode', unit='record')
    iterator = iter(loader.batches(order))
    for ids in order:
        tick = time.perf_counter(); cpu = next(iterator)
        times['loader_wait_seconds'] += time.perf_counter()-tick
        if cpu.indices.tolist() != ids:
            raise ValueError('Fusion support loader changed record order')
        torch.cuda.synchronize(); tick = time.perf_counter(); query = cpu.to('cuda')
        torch.cuda.synchronize(); times['transfer_seconds'] += time.perf_counter()-tick
        tick = time.perf_counter(); recipient, fused = fusion_basis(local, query)
        torch.cuda.synchronize(); times['l0_basis_seconds'] += time.perf_counter()-tick
        tick = time.perf_counter(); bank.add(ids, recipient, fused)
        times['cpu_vector_copy_seconds'] += time.perf_counter()-tick
        budget.check(); bar.update(len(ids))
        del cpu, query, recipient, fused
    bar.close()
    if any(i not in bank.values for i in required):
        raise ValueError('Full eligible fusion support coverage incomplete')
    return dict(required_unique_support_records=len(set(required)), newly_encoded_records=total,
                reused_query_trace_records=reused, physical_batch=batch,
                smaller_batches='only final remainder within each same-donor case',
                elapsed_seconds=time.perf_counter()-started, **times,
                complete=True, cached_vectors='detached recipient and Fuse output only; 2 x 128D per record',
                cnn_passes='one per missing physical batch, shared by all lambda branches',
                l0_basis_timing_scope='CNN, masked pooling, projection and original fusion; synchronized CUDA wall time',
                support_policy='full eligible inner-train support, query excluded on recipient and donor sides')


@torch.no_grad()
def probe_fusion(net, query_recipient, query_fused, support_recipient, support_fused,
                 owners, classes, *, scales, batch, truth, support_record_ids, query_group):
    """Matched full-support/query basis; all lambda plans freshly fitted."""
    if net.training or any(module.training for module in net.modules()):
        raise ValueError('Read-only fusion sweep requires eval mode')
    _check_basis(query_recipient, query_fused); _check_basis(support_recipient, support_fused)
    values = validate_scales(scales)
    if type(batch) is not int or batch < 1:
        raise ValueError('Original physical query batch required')
    if (owners.dtype != torch.long or classes.dtype != torch.long
            or owners.shape != classes.shape or owners.shape != (len(support_fused),)
            or len(support_record_ids) != len(support_fused)
            or len(set(support_record_ids)) != len(support_record_ids)):
        raise ValueError('Complete original support ownership/class/record bindings required')
    branches = [('production_fuse', None)] + [('recipient_residual', scale) for scale in values]
    reports = []
    for name, scale in branches:
        q = query_fused if scale is None else query_recipient + scale*query_fused
        support = support_fused if scale is None else support_recipient + scale*support_fused
        if not bool(torch.isfinite(q).all() & torch.isfinite(support).all()):
            raise FloatingPointError('Nonfinite fusion counterfactual; no fallback')
        tick = time.perf_counter()
        # Never reuse the saved plan with a changed embedding coordinate system.
        state = net.prepare_support(support, owners, classes)
        if not torch.equal(state['cluster_plan']['support_embeddings'], support):
            raise AssertionError('Fusion plan is not bound to its own branch support')
        logits = torch.cat([net.predict_embeddings(value, state)['logits'] for value in q.split(batch)])
        reports.append(dict(branch=name, fusion_scale=scale, score=_score_signal(logits, truth),
            query_signal=candidate_signal(q), support_signal=candidate_signal(support),
            support_records=len(support), support_patients=int(owners.max())+1,
            current_weights_full_support=True, query_support_formula_identical=True,
            support_plan_refitted=True, seconds=time.perf_counter()-tick,
            timing_scope='support plan fit, L1/L2 support preparation, query scoring and descriptive statistics'))
        del state, logits
    return dict(diagnostic_only=True, formula='r + lambda * Fuse(d,r,r-d,r*d)',
        baseline='production_fuse with freshly re-encoded support; saved epoch memory is not mixed in',
        production_compatible_lambda=None, support_record_ids=support_record_ids, query_group=query_group,
        branches=reports, optimizer_updates=0,
        interpretation='Fixed-weight sensitivity, not trained accuracy; a variance increase alone is not a ranking improvement')
