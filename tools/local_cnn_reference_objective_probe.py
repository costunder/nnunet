"""Same-forward DEBUG objective derivatives and copied AdamW counterfactuals.

The caller supplies already weighted ranking/CE/alignment/full tensors from its
one native forward, before the real backward/clip/step. No forward is run here.
Each shadow step copies the *current* optimizer state, including later moments;
only one shadow optimizer exists at a time. No model or optimizer is replaced.
"""
import copy
import math

import torch
from torch import nn

from hiercp_v222.v1_execution import rng_state, restore_rng
from l0_regions.training import hash_state


TERMS = ('ranking', 'observation_ce', 'alignment', 'full')
MODULES = {
    'CNN': ('local.cnn.',),
    'readout_fusion': ('local.project.', 'local.fuse.'),
    'L1': ('l1.', 'label_seed'),
    'L2': ('l2.', 'l2_updates.'),
}
BRANCHES = ('rank_only', 'auxiliary', 'full')


def _sum_gradients(*columns):
    """Keep a missing derivative None: AdamW must skip that parameter entirely."""
    result = []
    for values in zip(*columns):
        present = [value for value in values if value is not None]
        result.append(sum(present[1:], present[0]) if present else None)
    return tuple(result)


def _finite(values, label):
    present = [value for value in values if value is not None]
    if any(value.is_sparse for value in present):
        raise ValueError(f'{label}: sparse derivatives are unsupported by AdamW')
    if present and not bool(torch.stack([torch.isfinite(value).all() for value in present]).all()):
        raise FloatingPointError(f'{label}: nonfinite derivative or parameter delta')


def _norm(values, indices):
    present = [values[i] for i in indices if values[i] is not None]
    if not indices:
        return dict(norm=None, reason='module_has_no_parameters', connected_parameters=0)
    if not present:
        return dict(norm=None, reason='no_parameter_dependency', connected_parameters=0)
    square = torch.stack([value.detach().double().square().sum() for value in present]).sum()
    if not bool(torch.isfinite(square)):
        raise FloatingPointError('Nonfinite norm reduction')
    number = float(square.sqrt())
    return dict(norm=number, reason='connected_zero_derivative' if number == 0 else None,
                connected_parameters=len(present))


def _cosine(left, right, indices):
    a, b = _norm(left, indices), _norm(right, indices)
    if a['norm'] is None or b['norm'] is None:
        return dict(cosine=None, reason='left_' + a['reason'] if a['norm'] is None
                    else 'right_' + b['reason'])
    if a['norm'] == 0 or b['norm'] == 0:
        return dict(cosine=None, reason='left_zero_direction' if a['norm'] == 0 else 'right_zero_direction')
    products = [(left[i].detach().double()*right[i].detach().double()).sum()
                for i in indices if left[i] is not None and right[i] is not None]
    # Disjoint connected coordinates have dot product zero, not missing direction.
    dot = float(torch.stack(products).sum()) if products else 0.
    cosine = dot/(a['norm']*b['norm'])
    if not math.isfinite(cosine):
        raise FloatingPointError('Nonfinite cosine reduction')
    return dict(cosine=cosine, reason=None)


def _layout_copy(value, parameter):
    # The overlap pipeline can use channels_last_3d Conv3D weights. Both fused
    # moments and .grad must retain the same strides as each copied parameter.
    return torch.empty_strided(parameter.shape, parameter.stride(), device=parameter.device,
                               dtype=parameter.dtype).copy_(value.detach())


def _shadow_step(named, gradients, optimizer, state, grad_clip):
    parameters = [nn.Parameter(_layout_copy(parameter, parameter)) for _, parameter in named]
    mapping = {id(parameter): shadow for (_, parameter), shadow in zip(named, parameters)}
    groups = [dict(copy.deepcopy({key: value for key, value in group.items() if key != 'params'}),
                   params=[mapping[id(parameter)] for parameter in group['params']])
              for group in optimizer.param_groups]
    # The complete group options control every step. Some PyTorch versions add
    # derived defaults (e.g. decoupled_weight_decay) that are not constructor
    # keywords; passing the groups also preserves those version-specific options.
    shadow = torch.optim.AdamW(groups)
    # Loading without deepcopy can alias exp_avg/exp_avg_sq to the live optimizer.
    shadow.load_state_dict(copy.deepcopy(state))
    for parameter, derivative in zip(parameters, gradients):
        parameter.grad = None if derivative is None else _layout_copy(derivative, parameter)
    norm = torch.nn.utils.clip_grad_norm_(parameters, grad_clip, error_if_nonfinite=True)
    coefficient = min(1., float(grad_clip/(norm+1e-6)))
    clipped_hash = hash_state({name: parameter.grad for (name, _), parameter in zip(named, parameters)})
    shadow.step()
    deltas = tuple(parameter.detach()-original.detach()
                   for (_, original), parameter in zip(named, parameters))
    _finite(deltas, 'shadow AdamW')
    report = dict(gradient_norm_before_clip=float(norm), clipping_factor=coefficient,
                  clipped_gradient_sha256=clipped_hash,
                  delta_sha256=hash_state({name: value for (name, _), value in zip(named, deltas)}),
                  post_parameter_sha256=hash_state({name: parameter.detach()
                        for (name, _), parameter in zip(named, parameters)}),
                  post_optimizer_sha256=hash_state(shadow.state_dict()),
                  parameters_skipped_for_missing_derivative=sum(value is None for value in gradients))
    # Returned tensors are parameter-sized derivatives only, never another model
    # or optimizer. The copied moments and parameters are released before next arm.
    del shadow, groups, mapping, parameters
    return deltas, report


def _admit(net, named, weighted_terms, optimizer, grad_clip):
    if not isinstance(named, (tuple, list)) or not named:
        raise ValueError('Complete ordered named trainable parameters required')
    expected = [(name, parameter) for name, parameter in net.named_parameters() if parameter.requires_grad]
    if (len(named) != len(expected) or any(name != n or parameter is not p
            for (name, parameter), (n, p) in zip(named, expected))):
        raise ValueError('named differs from the original complete trainable parameter binding')
    if type(optimizer) is not torch.optim.AdamW:
        raise ValueError('Original torch.optim.AdamW required; no substitute optimizer')
    bound = [parameter for group in optimizer.param_groups for parameter in group['params']]
    if len(bound) != len(named) or {id(p) for p in bound} != {id(p) for _, p in named}:
        raise ValueError('Original AdamW must contain exactly the supplied trainable parameters')
    if any(group.get('differentiable', False) for group in optimizer.param_groups):
        raise ValueError('Differentiable AdamW is not an admitted native diagnostic execution policy')
    if isinstance(grad_clip, bool) or not math.isfinite(float(grad_clip)) or grad_clip <= 0:
        raise ValueError('Finite positive original global grad_clip required')
    device, dtype = named[0][1].device, named[0][1].dtype
    if any(p.device != device or p.dtype != dtype or p.layout != torch.strided
           for _, p in named) or dtype not in (torch.float32, torch.float64):
        raise ValueError('Single-device FP32/FP64 strided trainable parameters required')
    if not isinstance(weighted_terms, dict) or set(weighted_terms) != set(TERMS):
        raise ValueError(f'Already weighted same-forward terms required: {TERMS}')
    for key, value in weighted_terms.items():
        if not torch.is_tensor(value) or value.ndim != 0 or value.device != device:
            raise ValueError(f'{key}: same-device scalar loss Tensor required')
        if not bool(torch.isfinite(value)):
            raise FloatingPointError(f'{key}: nonfinite weighted loss')
    torch.testing.assert_close(weighted_terms['full'].detach(),
        sum(weighted_terms[key].detach() for key in TERMS[:-1]), rtol=1e-5, atol=1e-7)
    return device


def objective_probe(net, named, weighted_terms, optimizer, grad_clip):
    """Return JSON data from this forward; caller still performs its real full step.

    ``weighted_terms`` keys are ranking, observation_ce, alignment, full. The
    full entry is the original loss, not a recomputed model forward. Norms/cosines
    use the complete coordinates of each module (missing derivatives act as zero
    coordinates only for geometry). Missing .grad remains None in AdamW. Shadow
    derivatives replace copied .grad; the actual comparison step must likewise
    clear old gradient buffers before its full backward (no accumulation).
    """
    device = _admit(net, named, weighted_terms, optimizer, grad_clip)
    saved_rng = rng_state()
    saved_model_hash = hash_state(net.state_dict())
    saved_optimizer_hash = hash_state(optimizer.state_dict())
    saved_grad_hash = hash_state({name: parameter.grad for name, parameter in named})
    saved_grads = [(parameter.grad, None if parameter.grad is None else parameter.grad.detach().clone())
                   for _, parameter in named]
    bn = [(name + '.' + key, value, value.detach().clone())
          for name, module in net.named_modules() if isinstance(module, nn.modules.batchnorm._BatchNorm)
          for key, value in module.named_buffers(recurse=False) if value is not None]
    restored_bn = []
    # This read-only state_dict refers to current moments; each shadow alone
    # deepcopies it. No original moment tensor is handed to shadow.step().
    state = optimizer.state_dict()
    indices = {label: [i for i, (name, _) in enumerate(named) if name.startswith(prefixes)]
               for label, prefixes in MODULES.items()}
    indices['global'] = list(range(len(named)))
    try:
        gradients = {}
        for key in TERMS:
            loss = weighted_terms[key]
            values = (torch.autograd.grad(loss, [p for _, p in named], retain_graph=True,
                                         allow_unused=True) if loss.requires_grad else (None,)*len(named))
            _finite(values, key)
            gradients[key] = values
        component_sum = _sum_gradients(*(gradients[key] for key in TERMS[:-1]))
        residual = tuple(None if a is None and b is None else
                         (-b if a is None else a if b is None else a-b)
                         for a, b in zip(gradients['full'], component_sum))
        _finite(residual, 'full versus component derivative sum')
        for actual, expected in zip(gradients['full'], component_sum):
            if actual is None or expected is None:
                if actual is not expected:
                    raise AssertionError('Original full loss differs from weighted component dependencies')
        # Independent reverse passes accumulate FP32 sums in different orders,
        # especially around BN bias cancellation. Validate the vector residual
        # against roundoff at the component-gradient scale, not elementwise
        # relative error on a nearly-zero cancelled coordinate. Keep every
        # measured residual in the JSON; never replace it with zero.
        all_indices = list(range(len(named)))
        residual_norm = _norm(residual, all_indices)['norm']
        component_scale = sum(_norm(gradients[key], all_indices)['norm'] or 0. for key in TERMS[:-1])
        residual_tolerance = 64*torch.finfo(named[0][1].dtype).eps*max(1., component_scale)
        if residual_norm is not None and residual_norm > residual_tolerance:
            raise AssertionError('Original full derivative differs from weighted component sum: '
                                 f'global residual={residual_norm}, roundoff tolerance={residual_tolerance}')
        module_gradients = {}
        for label, group in indices.items():
            module_gradients[label] = dict(parameters=len(group),
                terms={key: _norm(gradients[key], group) for key in TERMS},
                cosines={f'ranking_vs_{key}': _cosine(gradients['ranking'], gradients[key], group)
                         for key in ('observation_ce', 'alignment', 'full')},
                full_component_sum_residual=_norm(residual, group))
        shadow_deltas, shadow_reports = {}, {}
        branch_gradients = dict(rank_only=gradients['ranking'],
                                auxiliary=_sum_gradients(gradients['observation_ce'], gradients['alignment']),
                                full=gradients['full'])
        for branch in BRANCHES:
            deltas, report = _shadow_step(named, branch_gradients[branch], optimizer, state, grad_clip)
            report['module_delta_norms'] = {label: _norm(deltas, group) for label, group in indices.items()}
            shadow_deltas[branch], shadow_reports[branch] = deltas, report
        delta_cosines = {label: {
            'rank_only_vs_full': _cosine(shadow_deltas['rank_only'], shadow_deltas['full'], group),
            'auxiliary_vs_full': _cosine(shadow_deltas['auxiliary'], shadow_deltas['full'], group),
            'rank_only_vs_auxiliary': _cosine(shadow_deltas['rank_only'], shadow_deltas['auxiliary'], group),
            'ranking_gradient_vs_full_descent': _cosine(gradients['ranking'],
                                                        tuple(-v for v in shadow_deltas['full']), group),
        } for label, group in indices.items()}
        steps = [float(value['step']) for value in optimizer.state.values() if 'step' in value]
        report = dict(format='local_cnn_reference_objective_probe_debug_v1', diagnostic_only=True,
            device=str(device), precision=str(named[0][1].dtype), additional_model_forwards=0,
            derivative_source='one caller forward; original already weighted scalar losses',
            weighted_losses={key: float(value.detach()) for key, value in weighted_terms.items()},
            module_gradients=module_gradients, delta_cosines=delta_cosines,
            full_component_sum_roundoff_tolerance=residual_tolerance,
            shadow_updates=shadow_reports,
            full_delta_sha256=shadow_reports['full']['delta_sha256'],
            full_clipped_gradient_sha256=shadow_reports['full']['clipped_gradient_sha256'],
            full_post_parameter_sha256=shadow_reports['full']['post_parameter_sha256'],
            full_post_optimizer_sha256=shadow_reports['full']['post_optimizer_sha256'],
            optimizer_state_before_sha256=saved_optimizer_hash,
            shadow_gradient_accumulation_steps=1,
            optimizer_history=dict(state_entries=len(optimizer.state),
                minimum_parameter_step=min(steps) if steps else None,
                maximum_parameter_step=max(steps) if steps else None,
                same_current_moments_copied_for_every_shadow=True, moments_reset=False),
            original_grad_clip=float(grad_clip), simultaneous_shadow_optimizers=1,
            memory_policy='parameter/moment/derivative copies only; shadows sequential; no model/activation copy',
            derivative_tensor_bytes=sum(v.numel()*v.element_size() for values in gradients.values()
                                        for v in values if v is not None),
            parameter_tensor_bytes=sum(p.numel()*p.element_size() for _, p in named),
            actual_full_step_verification='caller compares full_delta_sha256 after its original backward/clip/AdamW step',
            production_optimizer_steps=0, production_checkpoints_written=0)
    finally:
        # Activation-checkpoint replay can update BN buffers during a derivative
        # calculation. Restore that diagnostic side effect, never a forward or
        # optimizer modification; reference joint L1 itself is not checkpointed.
        with torch.no_grad():
            for name, value, old in bn:
                if not torch.equal(value, old):
                    restored_bn.append(name)
                    value.copy_(old)
        restore_rng(saved_rng)
        if hash_state({name: p.grad for name, p in named}) != saved_grad_hash:
            with torch.no_grad():
                for (_, parameter), (reference, value) in zip(named, saved_grads):
                    parameter.grad = reference
                    if reference is not None:
                        reference.copy_(value)
            raise AssertionError('Derivative hook modified original gradient buffers; buffers restored')
        if hash_state(net.state_dict()) != saved_model_hash:
            raise AssertionError('Objective diagnostic changed original parameters or non-BN model state')
        if hash_state(optimizer.state_dict()) != saved_optimizer_hash:
            raise AssertionError('Objective diagnostic changed original AdamW state')
        if hash_state(rng_state()) != hash_state(saved_rng):
            raise AssertionError('Objective diagnostic did not preserve original RNG')
    report.update(original_parameters_gradients_bn_rng_optimizer_preserved=True,
                  checkpoint_backward_bn_buffers_restored=restored_bn)
    return report
