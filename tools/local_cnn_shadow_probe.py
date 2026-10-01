"""Read-only AdamW counterfactuals on one copied, scheduled training update.

All branches start from the same model, optimizer moments and train-mode dropout
realization. The optional fifth branch uses explicit zero derivatives to isolate
saved AdamW history and weight decay. No branch is a training output/checkpoint.
"""
import copy
import time

import torch
from torch.nn import functional as F

from hiercp_v222.v1_execution import rng_state, restore_rng
from l0_regions.donor_learning import configuration, validate_rows
from l0_regions.training import hash_state


BRANCHES = ('ranking', 'observation_ce', 'alignment', 'full')
MODULES = {
    'CNN': ('local.cnn.',),
    'readout_fusion': ('local.project.', 'local.fuse.'),
    'L1': ('l1.', 'label_seed'),
    'L2': ('l2.', 'l2_updates.'),
}


def objective_terms(net, query, support, plan, rows, indices, context, settings):
    """The production same-donor loss, decomposed without changing coefficients."""
    selected = [rows[i] for i in indices]
    if not selected or len({r['case_id'] for r in selected}) != 1:
        raise ValueError('Shadow update needs one bound recipient ranking tile')
    validate_rows(selected)
    if selected != [context.rows[i] for i in indices]:
        raise ValueError('Shadow rows/context binding differs')
    embedding = net.local(query)
    state = net.prepare_support(*support, cluster_plan=plan)
    output = net.predict_embeddings(embedding, state)
    logits = output['logits'].float()
    targets = torch.tensor([r['target'] for r in selected], device=logits.device)
    score = logits[:, 1] - logits[:, 0]
    difference = score[targets == 1, None] - score[None, targets == 0]
    coefficients = logits.new_tensor([
        context.steps / (2 * context.counts[r['target']] * context.uses[i])
        for i, r in zip(indices, selected)
    ])
    return {
        'ranking': F.softplus(-difference).sum() * context.steps / context.pairs * settings['ranking_weight'],
        'observation_ce': (F.cross_entropy(logits, targets, reduction='none') * coefficients).sum()
                          * settings['observation_auxiliary_weight'],
        'alignment': output['alignment_loss'] * output['alignment_loss_weight'],
    }


def direction(reference, other):
    """Signed projection onto reference; None means its direction is undefined."""
    rr = torch.dot(reference, reference)
    oo = torch.dot(other, other)
    if not bool(rr > 0):
        return dict(cosine=None, signed_projection=None, opposing_projection=None)
    if not bool(oo > 0):
        return dict(cosine=None, signed_projection=0., opposing_projection=0.)
    dot = torch.dot(reference, other)
    projection = float(dot / rr)
    return dict(cosine=float(dot / (rr.sqrt() * oo.sqrt())),
                signed_projection=projection, opposing_projection=max(0., -projection))


def _vectors(named, values):
    # One flat vector is retained per objective. Missing gradients remain None
    # in the update; the zeros here are only coordinates for direction metrics.
    pieces = [(g if g is not None else torch.zeros_like(p)).detach().reshape(-1)
              for (_, p), g in zip(named, values)]
    return torch.cat(pieces)


def _module_indices(named):
    groups = {k: [] for k in MODULES}
    offset = 0
    for name, parameter in named:
        for label, prefixes in MODULES.items():
            if name.startswith(prefixes):
                groups[label].append(torch.arange(offset, offset + parameter.numel(), device=parameter.device))
        offset += parameter.numel()
    return {k: torch.cat(parts) if parts else torch.empty(0, dtype=torch.long, device=named[0][1].device)
            for k, parts in groups.items()} | {'global': torch.arange(offset, device=named[0][1].device)}


def _optimizer(net, saved):
    groups = saved['optimizer']['param_groups']
    if len(groups) != 1 or len(groups[0]['params']) != len(list(net.parameters())):
        raise ValueError('Saved AdamW parameter grouping differs from production model')
    group = groups[0]
    optimizer = torch.optim.AdamW(net.parameters(), lr=group['lr'], weight_decay=group['weight_decay'],
                                  fused=group.get('fused', False))
    # deepcopy is essential: optimizer.step must never mutate saved moments.
    optimizer.load_state_dict(copy.deepcopy(saved['optimizer']))
    return optimizer


def _preserve_parameter_layouts(net, state):
    """State loading copies values, so restore saved strides separately for AdamW.

    The overlap pipeline can change Conv3D weights to channels_last_3d. Fused
    AdamW requires parameters, derivatives and its saved moments to agree.
    """
    restored = []
    with torch.no_grad():
        for name, parameter in net.named_parameters():
            old = state[name]
            if not torch.is_tensor(old) or old.layout != torch.strided or parameter.shape != old.shape:
                raise ValueError(f'Saved trainable parameter layout differs: {name}')
            if parameter.stride() != old.stride():
                parameter.data = torch.empty_strided(old.shape, old.stride(), device=parameter.device,
                                                     dtype=parameter.dtype).copy_(old)
                restored.append(name)
    return restored


def _steps(optimizer):
    return [float(state['step']) for state in optimizer.state.values() if 'step' in state]


def _moment_comparison(optimizer, named, vectors, grouped, clip):
    """Saved first moment and the actual clipped current-gradient contribution."""
    beta1 = optimizer.param_groups[0]['betas'][0]
    averages = []
    uninitialized = []
    for name, parameter in named:
        state = optimizer.state.get(parameter, {})
        average = state.get('exp_avg')
        if average is None:
            # AdamW initializes an absent state to exactly zero on its first
            # connected step; this is not an invented fallback derivative.
            if state:
                raise ValueError(f'Saved AdamW state lacks exp_avg: {name}')
            uninitialized.append(name)
            average = torch.zeros_like(parameter)
        if average.shape != parameter.shape or not bool(torch.isfinite(average).all()):
            raise ValueError(f'Saved AdamW first moment is invalid: {name}')
        averages.append(average.detach().reshape(-1))
    saved_average = torch.cat(averages)
    contributions = {}
    for branch, vector in vectors.items():
        scale = min(1., float(clip / (vector.norm() + 1e-6)))
        contributions[branch] = vector * scale * (1. - beta1)
    reports = {}
    for label, positions in grouped.items():
        old = saved_average[positions]
        old_norm = float(old.norm())
        reports[label] = dict(saved_exp_avg_norm=old_norm,
            beta1_saved_exp_avg_norm=float((beta1 * old).norm()),
            current_gradient_contributions={branch: dict(norm=float(value[positions].norm()),
                ratio_to_saved_exp_avg=(float(value[positions].norm()) / old_norm if old_norm else None),
                vs_saved_exp_avg=direction(old, value[positions]))
                for branch, value in contributions.items()})
    return dict(beta1=beta1, modules=reports, uninitialized_parameter_names=uninitialized,
                gradient_contribution='(1-beta1) * current gradient after production clipping')


def shadow_update(factory, saved, query, support, plan, rows, indices, context, budget,
                  *, include_history_only=False):
    """Real AdamW steps on a fresh clone; restores caller RNG even after failure.

    factory must construct the exact model in retained FP32 execution. query and
    support are the next scheduled physical tile and its original support/plan.
    No forward labels, schedule, observation or candidate counts are changed.
    include_history_only adds a copied AdamW step with a real zero tensor for
    every trainable parameter; None would skip moment/decay and is forbidden.
    """
    caller_rng = rng_state()
    saved_digest = hash_state(saved)
    input_digest = hash_state(dict(support=support, plan=plan))
    result = None
    try:
        budget.check()
        if saved['identity']['precision'] != 'FP32':
            raise ValueError('Shadow update must match saved FP32 production precision')
        settings = saved['identity']['ranking']
        if settings != configuration():
            raise ValueError('Shadow diagnostic supports the exact same-donor live objective only')
        clip = saved['identity']['base']['training']['grad_clip']
        if not isinstance(clip, (int, float)) or clip <= 0:
            raise ValueError('Saved production gradient clipping must be positive')
        net = factory()
        restored_layouts = _preserve_parameter_layouts(net, saved['model'])
        net.load_state_dict(saved['model'], strict=True)
        net.train()
        if getattr(net, 'checkpoint_support', False) or getattr(getattr(net.local, 'cnn', None), 'checkpointing', False):
            raise ValueError('Shadow factory must use retained CNN/L1/L2 activations')
        named = [(name, parameter) for name, parameter in net.named_parameters() if parameter.requires_grad]
        if not named or any(parameter.dtype != torch.float32 for _, parameter in named):
            raise ValueError('Shadow model trainable parameters must retain production FP32')
        parameters = [parameter for _, parameter in named]
        original = [parameter.detach().clone() for parameter in parameters]
        if len(indices) != len(query):
            raise ValueError('Shadow query and original scheduled tile have different physical size')
        if len(indices) > saved['state']['batch']:
            raise ValueError('Shadow tile exceeds saved physical batch')
        restore_rng(saved['rng'])
        with torch.enable_grad(), torch.autocast(parameters[0].device.type, enabled=False):
            terms = objective_terms(net, query, support, plan, rows, indices, context, settings)
            full = sum(terms.values())
            if not bool(torch.isfinite(full)):
                raise FloatingPointError('Nonfinite shadow production objective')
            budget.check()
            per_loss = {key: torch.autograd.grad(value, parameters, retain_graph=True, allow_unused=True)
                        for key, value in terms.items()}
            net.zero_grad(set_to_none=True)
            full.backward()
            from l0_regions.execution_pipeline import gradient_check_batched
            gradient_check_batched(net)
            full_gradients = tuple(parameter.grad.detach().clone() for parameter in parameters)
        all_gradients = per_loss | {'full': full_gradients}
        vectors = {key: _vectors(named, values) for key, values in all_gradients.items()}
        budget.check()
        summed = vectors['ranking'] + vectors['observation_ce'] + vectors['alignment']
        torch.testing.assert_close(summed, vectors['full'], atol=2e-5, rtol=2e-4,
                                   msg='Decomposed gradients differ from actual full backward')
        grouped = _module_indices(named)
        gradients = {}
        for label, positions in grouped.items():
            rank = vectors['ranking'][positions]
            gradients[label] = dict(norms={key: float(vector[positions].norm()) for key, vector in vectors.items()},
                observation_ce_vs_ranking=direction(rank, vectors['observation_ce'][positions]),
                alignment_vs_ranking=direction(rank, vectors['alignment'][positions]),
                auxiliary_sum_vs_ranking=direction(rank, (vectors['observation_ce'] + vectors['alignment'])[positions]))
        losses = {key: float(value.detach()) for key, value in terms.items()} | {'full': float(full.detach())}
        del terms, full, summed
        # Autograd has released the full graph. Only detached gradients and the
        # one clone remain while independent optimizer branches are measured.
        branch_reports = {}
        deltas = {}
        moment_comparison = None
        branch_names = BRANCHES + (('history_only',) if include_history_only else ())
        for branch in branch_names:
            net.load_state_dict(saved['model'], strict=True)
            restore_rng(saved['rng'])
            optimizer = _optimizer(net, saved)
            optimizer.zero_grad(set_to_none=True)
            if include_history_only and moment_comparison is None:
                moment_comparison = _moment_comparison(optimizer, named, vectors, grouped, clip)
            grads = (tuple(torch.zeros_like(parameter) for parameter in parameters)
                     if branch == 'history_only' else all_gradients[branch])
            missing = []
            for (name, parameter), gradient in zip(named, grads):
                if gradient is None:
                    missing.append(name)
                else:
                    # autograd.grad need not return the parameter's layout;
                    # fused AdamW requires the same layout as saved moments.
                    parameter.grad = torch.empty_like(parameter, memory_format=torch.preserve_format).copy_(gradient.detach())
            present = [parameter.grad for parameter in parameters if parameter.grad is not None]
            if not present or not bool(torch.stack([torch.isfinite(g).all() for g in present]).all()):
                raise FloatingPointError(f'Empty or nonfinite gradients in shadow {branch}')
            if branch in ('full', 'history_only'):
                gradient_check_batched(net)
            before_steps = _steps(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(parameters, clip, error_if_nonfinite=True)
            if parameters[0].is_cuda:
                torch.cuda.synchronize(parameters[0].device)
            start = time.perf_counter()
            optimizer.step()
            if parameters[0].is_cuda:
                torch.cuda.synchronize(parameters[0].device)
            seconds = time.perf_counter() - start
            delta = torch.cat([(parameter.detach() - old).flatten() for parameter, old in zip(parameters, original)])
            deltas[branch] = delta
            per_module = {}
            for label, positions in grouped.items():
                d = delta[positions]
                per_module[label] = dict(delta_norm=float(d.norm()),
                    vs_negative_ranking_gradient=direction(-vectors['ranking'][positions], d))
            after_steps = _steps(optimizer)
            branch_reports[branch] = dict(weighted_loss=(0. if branch == 'history_only' else losses[branch]), gradient_norm_before_clip=float(norm),
                clipping_factor=min(1., float(clip / (norm + 1e-6))), modules=per_module,
                unused_gradient_parameters=missing, unused_gradient_policy='None; no AdamW decay/moment update',
                adam_state_entries=len(optimizer.state),
                adam_step_before_min=min(before_steps) if before_steps else None,
                adam_step_before_max=max(before_steps) if before_steps else None,
                adam_step_after_min=min(after_steps) if after_steps else None,
                adam_step_after_max=max(after_steps) if after_steps else None,
                optimizer_seconds=seconds)
            if branch == 'history_only':
                branch_reports[branch]['explicit_zero_gradients'] = True
                branch_reports[branch]['zero_gradient_parameter_count'] = len(parameters)
            budget.check()
            del optimizer
        for branch in [key for key in branch_names if key != 'full']:
            branch_reports[branch]['vs_full_update'] = {
                label: dict(direction(deltas['full'][positions], deltas[branch][positions]),
                            delta_difference_norm=float((deltas[branch] - deltas['full'])[positions].norm()))
                for label, positions in grouped.items()
            }
        history = None
        if include_history_only:
            history = {}
            for label, positions in grouped.items():
                old = deltas['history_only'][positions]
                rank = deltas['ranking'][positions]
                full_delta = deltas['full'][positions]
                full_norm, rank_norm = float(full_delta.norm()), float(rank.norm())
                history[label] = dict(history_delta_norm=float(old.norm()),
                    ratio_to_full_delta_norm=(float(old.norm()) / full_norm if full_norm else None),
                    ratio_to_ranking_delta_norm=(float(old.norm()) / rank_norm if rank_norm else None),
                    vs_full_delta=direction(full_delta, old), vs_ranking_delta=direction(rank, old),
                    delta_full_minus_history=dict(delta_norm=float((full_delta - old).norm()),
                        vs_negative_ranking_gradient=direction(-vectors['ranking'][positions], full_delta - old)),
                    delta_ranking_minus_history=dict(delta_norm=float((rank - old).norm()),
                        vs_negative_ranking_gradient=direction(-vectors['ranking'][positions], rank - old)))
        result = dict(diagnostic_only=True, production_optimizer_updates=0, cloned_optimizer_steps=len(branch_names),
            train_mode=True, dropout_enabled=True, precision='FP32', activation_storage='retained',
            physical_batch=len(indices), configured_physical_batch=saved['state']['batch'],
            saved_step=saved['state']['step'], losses=losses, gradients=gradients,
            branches=branch_reports, decomposition_full_backward_verified=True,
            optimizer_state_reused=True, same_forward_realization_for_all_branches=True,
            saved_parameter_layouts_restored=restored_layouts,
            direction_note='Delta includes saved Adam moments, epsilon and weight decay; gradient conflict alone does not determine its direction')
        if include_history_only:
            result['history_isolation'] = dict(modules=history, saved_moment_comparison=moment_comparison,
                explicit_zero_gradients=True, includes_weight_decay=True,
                note='AdamW uses a nonlinear second-moment denominator. Delta subtraction is a counterfactual contrast, not an additive decomposition or a percentage attribution.')
        budget.check()
    finally:
        restore_rng(caller_rng)
    if hash_state(saved) != saved_digest:
        raise AssertionError('Shadow diagnostic mutated original checkpoint payload')
    if hash_state(dict(support=support, plan=plan)) != input_digest:
        raise AssertionError('Shadow diagnostic mutated original support/plan')
    result['saved_payload_unchanged'] = True
    result['support_plan_unchanged'] = True
    result['caller_rng_restored'] = True
    return result
