"""Explicit short frozen-feature control; never a v1 replay or production model.

Only a newly initialized scalar head trains. The caller supplies every retained
candidate in its selected train/validation cases and all optimization settings.
No CNN, original scorer, checkpoint, cache, or production optimizer is changed.
"""
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
import time
import zipfile

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from hiercp_v222.v1_execution import rng_state, restore_rng
from l0_regions.training import hash_state
from tools.v22_candidate_order import TIE_POLICY
from tools.v22_rank_objective import ranking_metrics


def _v1_reference():
    root = Path(__file__).resolve().parents[1] / 'versions' / 'v1'
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    with zipfile.ZipFile(root / 'pipeline_v1_source.zip') as archive:
        model = archive.read('hiercp/model.py')
        config_data = archive.read('config/train.json')
    for name, value in [('hiercp/model.py', model), ('config/train.json', config_data)]:
        if hashlib.sha256(value).hexdigest() != manifest['files'][name]:
            raise ValueError(f'Preserved v1 reference hash differs: {name}')
    config = json.loads(config_data)
    return dict(revision=manifest['revision'], source='versions/v1/pipeline_v1_source.zip:hiercp/model.py:1237',
                model_sha256=manifest['files']['hiercp/model.py'], config=config)


class DirectScalarHead(nn.Sequential):
    """The v1 scalar-head shape family, with diagnostic recipient-only input."""
    def __init__(self, hidden_dim, dropout):
        super().__init__(nn.Linear(128, hidden_dim * 4), nn.LayerNorm(hidden_dim * 4),
                         nn.SiLU(inplace=True), nn.Dropout(dropout),
                         nn.Linear(hidden_dim * 4, hidden_dim * 2), nn.SiLU(inplace=True),
                         nn.Dropout(dropout), nn.Linear(hidden_dim * 2, 1, bias=False))


def _cases(cases, name):
    if not isinstance(cases, list) or not cases:
        raise ValueError(f'{name} requires explicit nonempty selected cases')
    rows, features, truth, keys, case_ids = [], [], [], [], []
    for case in cases:
        case_id, donor_id = case['case_id'], case['donor_case_id']
        if not isinstance(case_id, str) or not case_id or not isinstance(donor_id, str) or not donor_id:
            raise ValueError('Actual case/donor identities required')
        if case_id == donor_id:
            raise ValueError('Self-case donor is forbidden')
        x, y = case['features'], case['truth']
        if not torch.is_tensor(x) or x.ndim != 2 or x.shape[1] != 128 or not len(x):
            raise ValueError('Recipient project features must be nonempty [N,128] tensors')
        if not x.is_floating_point() or x.requires_grad or not bool(torch.isfinite(x).all()):
            raise ValueError('Frozen finite floating recipient features required')
        if not torch.is_tensor(y) or y.shape != (len(x),) or y.dtype != torch.long or not bool(((y == 0) | (y == 1)).all()):
            raise ValueError('Bound binary long observation labels required')
        ids, candidate_keys = case['record_ids'], case['candidate_keys']
        if len(ids) != len(x) or any(not isinstance(value, str) or not value for value in ids):
            raise ValueError('Complete actual record IDs required')
        if len(candidate_keys) != len(x) or any(not isinstance(value, str) or not value for value in candidate_keys):
            raise ValueError('Complete label-independent geometry candidate keys required')
        case_ids.append(case_id)
        for record, target in zip(ids, y.detach().cpu().tolist()):
            rows.append(dict(id=record, case_id=case_id, donor_case_id=donor_id, target=target))
        features.append(x.detach().float())
        truth.extend(y.detach().cpu().tolist())
        keys.extend(candidate_keys)
    if len(set(case_ids)) != len(case_ids):
        raise ValueError('Selected case must appear exactly once')
    if len({x.device for x in features}) != 1:
        raise ValueError('All frozen features must share the same explicit device')
    if len(set(keys)) != len(keys):
        raise ValueError('Geometry candidate keys must be unique')
    if len({row['id'] for row in rows}) != len(rows):
        raise ValueError('Record identities must be unique')
    return SimpleNamespace(rows=rows, features=torch.cat(features),
                           truth=torch.tensor(truth, device=features[0].device), keys=keys, case_ids=case_ids)


def _groups(dataset, physical_batch, seed, epoch):
    """Same P x U tiling rule; preserves the caller's original candidate order."""
    rng = np.random.default_rng(seed + epoch)
    case_order = list(dataset.case_ids)
    rng.shuffle(case_order)
    order = []
    pairs = 0
    for case_id in case_order:
        ids = [i for i, row in enumerate(dataset.rows) if row['case_id'] == case_id]
        pos = [i for i in ids if dataset.rows[i]['target']]
        neg = [i for i in ids if not dataset.rows[i]['target']]
        pairs += len(pos) * len(neg)
        if pos and neg:
            tiles = []
            width = min(len(pos), physical_batch // 2)
            for start in range(0, len(pos), width):
                anchors = pos[start:start + width]
                remaining = physical_batch - len(anchors)
                tiles.extend(anchors + neg[j:j + remaining] for j in range(0, len(neg), remaining))
        else:
            tiles = [ids[i:i + physical_batch] for i in range(0, len(ids), physical_batch)]
        rng.shuffle(tiles)
        order.extend(tiles)
    if not pairs:
        raise ValueError('Direct-head selected cohort needs observed/unobserved ranking pairs')
    if {i for tile in order for i in tile} != set(range(len(dataset.rows))):
        raise ValueError('Direct-head complete schedule lost observations')
    return order, pairs


@torch.no_grad()
def _evaluate(head, dataset, physical_batch, budget):
    head.eval()
    score_parts = []
    for x in dataset.features.split(physical_batch):
        budget.check()
        score_parts.append(head(x).squeeze(-1))
    scores = torch.cat(score_parts)
    if not bool(torch.isfinite(scores).all()):
        raise FloatingPointError('Nonfinite frozen-feature direct-head scores')
    values, targets = scores.detach().cpu(), dataset.truth.detach().cpu()
    metrics, reports = ranking_metrics(values, targets, [r['case_id'] for r in dataset.rows],
                                       candidate_keys=dataset.keys)
    by_case = {row['case_id']: row for row in reports}
    wins = ties = gaps = 0.
    comparisons = 0
    for case_id in dataset.case_ids:
        ids = torch.tensor([i for i, row in enumerate(dataset.rows) if row['case_id'] == case_id])
        case_scores, case_truth = values[ids], targets[ids]
        difference = case_scores[case_truth == 1, None] - case_scores[None, case_truth == 0]
        n = difference.numel()
        by_case[case_id].update(records=len(ids), score_std=float(case_scores.std(unbiased=False)),
            observed=int((case_truth == 1).sum()), unobserved=int((case_truth == 0).sum()),
            comparisons=n, pair_win_rate=float((difference > 0).float().mean()) if n else None,
            exact_tie_rate=float((difference == 0).float().mean()) if n else None,
            positive_minus_unobserved_gap=float(difference.mean()) if n else None)
        if n:
            comparisons += n
            wins += float((difference > 0).sum())
            ties += float((difference == 0).sum())
            gaps += float(difference.sum())
    metrics.update(pair_win_rate=wins / comparisons, exact_tie_rate=ties / comparisons,
                   positive_minus_unobserved_gap=gaps / comparisons,
                   score_std=float(values.std(unbiased=False)), tie_policy=TIE_POLICY)
    return dict(metrics=metrics, cases=reports)


def probe_direct_head(train_cases, validation_cases, *, steps, physical_batch, seed, lr, hidden_dim, budget):
    """Fit a fresh scalar head on detached project-r vectors for explicit steps.

    A case contains features [N,128], truth long [N], record_ids, geometry
    candidate_keys, case_id and donor_case_id. Every supplied row is retained.
    P x U tiles use the production tiling/global pair-mean rule, not a replay of
    its saved schedule or optimizer. The
    caller chooses short diagnostic steps; a partial schedule is reported as such.
    """
    for label, value, minimum in [('steps', steps, 1), ('physical_batch', physical_batch, 2),
                                   ('hidden_dim', hidden_dim, 1)]:
        if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
            raise ValueError(f'Explicit {label} >= {minimum} required')
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError('Explicit nonnegative initialization/schedule seed required')
    if not isinstance(lr, (int, float)) or isinstance(lr, bool) or not math.isfinite(lr) or lr <= 0:
        raise ValueError('Explicit finite positive direct-head learning rate required')
    caller_rng = rng_state()
    input_digest = hash_state(dict(train=train_cases, validation=validation_cases))
    try:
        budget.check()
        train = _cases(train_cases, 'train')
        validation = _cases(validation_cases, 'validation')
        if set(train.case_ids) & set(validation.case_ids):
            raise ValueError('Direct-head train/validation case leakage')
        if {r['id'] for r in train.rows} & {r['id'] for r in validation.rows}:
            raise ValueError('Direct-head train/validation record leakage')
        if set(train.keys) & set(validation.keys):
            raise ValueError('Direct-head train/validation candidate geometry leakage')
        if train.features.device != validation.features.device:
            raise ValueError('Train/validation frozen features require the same explicit device')
        order, total_pairs = _groups(train, physical_batch, seed, 0)
        _groups(validation, physical_batch, seed, 0)  # Require real evaluable pairs.
        context = SimpleNamespace(steps=len(order), pairs=total_pairs)
        reference = _v1_reference()
        config = reference.pop('config')
        torch.manual_seed(seed)
        head = DirectScalarHead(hidden_dim, config['model']['dropout']).to(train.features.device)
        optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=config['training']['weight_decay'])
        initial = [parameter.detach().clone() for parameter in head.parameters()]
        before = dict(train=_evaluate(head, train, physical_batch, budget),
                      validation=_evaluate(head, validation, physical_batch, budget))
        losses, norms, presentations = [], [], []
        seen, epoch, position = set(), 0, 0
        if train.features.is_cuda:
            torch.cuda.synchronize(train.features.device)
        start = time.perf_counter()
        for step in range(steps):
            if position == len(order):
                epoch += 1
                order, pairs = _groups(train, physical_batch, seed, epoch)
                if pairs != context.pairs or len(order) != context.steps:
                    raise ValueError('Direct-head full schedule coverage changed across epochs')
                position = 0
            indices = order[position]
            position += 1
            seen.update(indices)
            presentations.append(len(indices))
            budget.check()
            ids = torch.tensor(indices, device=train.features.device)
            head.train();optimizer.zero_grad(set_to_none=True)
            scores = head(train.features[ids]).squeeze(-1)
            targets = train.truth[ids]
            difference = scores[targets == 1, None] - scores[None, targets == 0]
            loss = F.softplus(-difference).sum() * context.steps / context.pairs
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError('Nonfinite direct-head pairwise objective')
            loss.backward()
            params = list(head.parameters())
            if any(parameter.grad is None for parameter in params):
                raise RuntimeError('Disconnected direct-head parameter gradient')
            norm = torch.nn.utils.clip_grad_norm_(params, config['training']['grad_clip'], error_if_nonfinite=True)
            optimizer.step()
            losses.append(loss.detach())
            norms.append(norm.detach())
            budget.check()
        if train.features.is_cuda:
            torch.cuda.synchronize(train.features.device)
        seconds = time.perf_counter() - start
        after = dict(train=_evaluate(head, train, physical_batch, budget),
                     validation=_evaluate(head, validation, physical_batch, budget))
        delta = torch.cat([(p.detach() - old).reshape(-1) for p, old in zip(head.parameters(), initial)])
        if not bool(torch.isfinite(delta).all()):
            raise FloatingPointError('Nonfinite direct-head parameter update')
        result = dict(diagnostic_only=True, production_optimizer_updates=0, cnn_updates=0,
            checkpoints_written=0, head_optimizer_updates=steps, head_initialized_fresh=True,
            feature_source='frozen recipient project-r; no donor fusion, L1, L2 or support',
            objective='Global selected-cohort observed/unobserved pair mean; rank-only, no CE or alignment',
            full_v1_reproduction=False, v1_scalar_head_reference=reference,
            architecture=dict(input=128, hidden_dim=hidden_dim, widths=[hidden_dim * 4, hidden_dim * 2, 1],
                dropout=config['model']['dropout'], final_bias=False,
                parameter_count=sum(p.numel() for p in head.parameters())),
            optimizer=dict(name='AdamW', lr=lr, weight_decay=config['training']['weight_decay'],
                grad_clip=config['training']['grad_clip'], seed=seed, saved_production_moments_used=False),
            physical_batch=physical_batch, effective_batch=physical_batch, gradient_accumulation_steps=1,
            actual_update_batch_sizes=presentations, before=before, after=after,
            schedule=dict(selected_train_cases=train.case_ids, selected_validation_cases=validation.case_ids,
                train_observations=len(train.rows), validation_observations=len(validation.rows),
                all_supplied_observations_retained=True, training_pairs_per_complete_epoch=context.pairs,
                tiles_per_complete_epoch=context.steps, complete_schedule_epochs=epoch + int(position == len(order)),
                final_epoch_tiles_run=position, unique_train_observations_presented=len(seen),
                order='Caller candidate order within case; case/tile shuffle uses explicit seed+epoch',
                every_within_case_pair_once_per_complete_epoch=True),
            learning_check=dict(all_parameter_gradients_present_and_finite=True,
                gradient_norm_min=float(torch.stack(norms).min()), gradient_norm_max=float(torch.stack(norms).max()),
                head_parameter_delta_norm=float(delta.norm()), loss_first=float(losses[0]), loss_last=float(losses[-1])),
            optimization_seconds=seconds,
            interpretation='Short frozen-feature direct-head control with a fresh head/AdamW and rank-only loss, unlike the original historical full-objective optimizer. Success shows information accessible to this scalar readout; it does not uniquely attribute the original failure to fusion/L1/L2, prove useful CP or final accuracy. Training fit alone is not causal proof; failed short frozen-feature fitting is not proof that CNN features are useless.')
    finally:
        restore_rng(caller_rng)
        if hash_state(dict(train=train_cases, validation=validation_cases)) != input_digest:
            raise AssertionError('Direct-head diagnostic mutated original frozen feature/label inputs')
    result['caller_rng_restored'] = True
    result['frozen_inputs_unchanged'] = True
    return result
