"""Explicit DEBUG BN/dropout controls on an isolated reference L1 model.

The exact native physical tile, full-cohort loss coefficients, initial weights
and supplied own-branch frozen teacher are shared by every control. Only module
training modes differ. A separate BN-eval/dropout-off cloned full update uses
the original clipping and fresh AdamW settings; no production state is written.
"""
import copy
import math

import torch
from torch import nn

from hiercp_v222.v1_execution import rng_state, restore_rng
from l0_regions.donor_learning import LiveContext, configuration, forward_loss
from l0_regions.execution_pipeline import gradient_check_batched
from l0_regions.training import hash_state
from tools.diagnose_local_cnn_learning import score_summary
from tools.local_cnn_interaction_updates import _module_norms, _query
from tools.local_cnn_reference_l1 import ReferencePromptGraphModel, ReferenceRelationLayer


MODES = (
    ('eval', False, False),
    ('bn_train_only', True, False),
    ('dropout_train_only', False, True),
    ('current_training', True, True),
)


def _gradients(net):
    return {name: parameter.grad for name, parameter in net.named_parameters()}


def _parameters(net):
    return hash_state(dict(net.named_parameters()))


def _inputs(query, support, plan, targets, indices):
    native = {key: value for key, value in vars(query).items() if key != '_verified_signature'}
    return hash_state(dict(query=native, support=support, plan=plan,
                           targets=targets, indices=indices))


def _set_modes(net, bn_training, dropout_training):
    """Control functional attention, MHA and module dropout via their real modes."""
    net.train(dropout_training)
    batch_norms = {}
    dropouts = {}
    for name, module in net.named_modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            if not module.track_running_stats or module.running_mean is None or module.running_var is None:
                raise ValueError('Reference mode controls require actual tracked BN running statistics')
            module.train(bn_training)
            batch_norms[name] = module
        elif isinstance(module, nn.Dropout):
            dropouts[name] = dict(kind='module', p=module.p, training=module.training)
        elif isinstance(module, nn.MultiheadAttention):
            dropouts[name] = dict(kind='multihead_attention', p=module.dropout, training=module.training)
        elif isinstance(module, ReferenceRelationLayer):
            dropouts[name] = dict(kind='reference_functional_attention_and_residual',
                                  p=module.dropout, training=module.training)
    if not batch_norms:
        raise ValueError('Reference BN modules required; no alternative architecture fallback')
    if any(row['training'] != dropout_training for row in dropouts.values()):
        raise AssertionError('Actual functional/module dropout modes differ from requested control')
    return batch_norms, dropouts


def _bn_summary(modules, calls=None):
    return {name: dict(training=module.training,
        batch_statistics_used=module.training,
        running_statistics_used=not module.training,
        forward_calls=None if calls is None else calls[name],
        num_batches_tracked=int(module.num_batches_tracked),
        momentum=module.momentum, eps=module.eps,
        running_mean_norm=float(module.running_mean.detach().norm()),
        running_var_min=float(module.running_var.detach().min()),
        running_var_mean=float(module.running_var.detach().mean()),
        running_var_max=float(module.running_var.detach().max()),
        buffers_sha256=hash_state(dict(mean=module.running_mean,
            variance=module.running_var, counter=module.num_batches_tracked)))
        for name, module in modules.items()}


def _forward(net, query, support, plan, targets, context, ids):
    """Capture the original loss forward's outputs without a second model pass."""
    captured = {}
    marker = object()
    prior = net.__dict__.get('predict_embeddings', marker)
    original = net.predict_embeddings
    def predict(*args, **kwargs):
        if 'output' in captured:
            raise AssertionError('One exact full-objective forward required per mode measurement')
        captured['embedding'] = args[0]
        captured['output'] = original(*args, **kwargs)
        return captured['output']
    # This wrapper lives only on the diagnostic clone and returns the unchanged
    # implementation's exact output. No query class enters predict_embeddings.
    net.predict_embeddings = predict
    try:
        loss, terms = forward_loss(net, query, support, plan, targets, None,
                                    context, configuration(), indices=ids)
    finally:
        if prior is marker:
            del net.predict_embeddings
        else:
            net.predict_embeddings = prior
    if 'output' not in captured or not bool(torch.isfinite(loss)):
        raise FloatingPointError('Finite original objective and actual reference output required')
    logits = captured['output']['logits'].float()
    if logits.shape != (len(ids), 2) or not bool(torch.isfinite(logits).all()):
        raise FloatingPointError('Finite two-class exact native-tile logits required')
    return loss, terms, logits, captured['embedding']


def _score(logits, targets):
    report = score_summary((logits[:, 1]-logits[:, 0]).detach(), targets)
    report.update(mean_ranking_margin=report['mean_positive_minus_unobserved'],
        observed=int((targets == 1).sum()), unobserved=int((targets == 0).sum()),
        scope='same original physical tile P x U; not full-case validation')
    return report


def _loss_report(loss, terms, alignment_weight):
    return dict(full_loss=float(loss.detach()),
        terms={key: float(value.detach()) for key, value in terms.items()},
        weighted_alignment_loss=float(terms['alignment_loss'].detach()*alignment_weight))


def _query_route(value, embedding, cnn_named):
    gradients = torch.autograd.grad(value, [embedding, *[p for _, p in cnn_named]],
                                    retain_graph=True, allow_unused=True)
    available = [gradient for gradient in gradients if gradient is not None]
    if available and not bool(torch.stack([torch.isfinite(g).all() for g in available]).all()):
        raise FloatingPointError('Nonfinite per-loss query/CNN derivative')
    query_gradient = gradients[0]
    cnn_gradients = [gradient for gradient in gradients[1:] if gradient is not None]
    query_norm = float(query_gradient.detach().norm()) if query_gradient is not None else None
    cnn_norm = (float(torch.stack([g.detach().float().square().sum()
                    for g in cnn_gradients]).sum().sqrt()) if cnn_gradients else None)
    return dict(weighted_loss=float(value.detach()),
        query_embedding_graph_dependency=query_gradient is not None,
        query_embedding_gradient_norm=query_norm,
        CNN_parameter_gradient_norm=cnn_norm,
        CNN_connected_parameter_tensors=len(cnn_gradients),
        CNN_total_parameter_tensors=len(cnn_named),
        query_gradient_route_active=query_norm is not None and query_norm > 0,
        CNN_gradient_route_active=cnn_norm is not None and cnn_norm > 0,
        interpretation='A graph dependency with an exactly zero derivative is not an active query route')


def _check_bn(before, after, training, expected_calls=1):
    for name, row in before.items():
        if (after[name]['forward_calls'] != expected_calls
                or after[name]['num_batches_tracked']-row['num_batches_tracked'] != int(training)*expected_calls
                or (not training and after[name]['buffers_sha256'] != row['buffers_sha256'])):
            raise AssertionError('BN control did not use exactly the requested statistics once per forward')


def _bn_hooks(modules):
    calls = {name: 0 for name in modules}
    handles = []
    for name, module in modules.items():
        def count(_module, _args, _output, name=name):
            calls[name] += 1
        handles.append(module.register_forward_hook(count))
    return calls, handles


def mode_probe(net, query, support, plan, targets, context, indices, training, budget):
    """Run four same-weight controls and one isolated frozen-BN full update."""
    if not isinstance(net, ReferencePromptGraphModel):
        raise ValueError('An explicit reference clone is required; legacy/production fallback forbidden')
    if not isinstance(context, LiveContext):
        raise ValueError('Original full-cohort LiveContext required')
    ids = list(indices)
    if (not ids or len(ids) != context.audit['physical_batch']
            or tuple(ids) not in {tuple(tile) for tile in context.order}):
        raise ValueError('An exact original full physical ranking tile required')
    rows = [context.rows[i] for i in ids]
    if len({row['case_id'] for row in rows}) != 1 or {row['target'] for row in rows} != {0, 1}:
        raise ValueError('One original recipient and both P/U observation classes required')
    device = next(net.parameters()).device
    named = [(name, parameter) for name, parameter in net.named_parameters() if parameter.requires_grad]
    if not named or any(p.dtype != torch.float32 or p.device != device for _, p in named):
        raise ValueError('Single-device original FP32 reference parameters required')
    cnn_named = [(name, p) for name, p in named if name.startswith('local.cnn.')]
    if not cnn_named:
        raise ValueError('Actual query CNN parameter path required; no detached-embedding substitute')
    _query(query, ids, context, device)
    if (targets.shape != (len(ids),) or targets.dtype != torch.long or targets.device != device
            or not torch.equal(targets, torch.tensor([row['target'] for row in rows], device=device))):
        raise ValueError('Original native tile and observation truth binding required')
    if (not isinstance(support, tuple) or len(support) != 3
            or any(value.device != device or value.requires_grad for value in support)
            or not isinstance(plan, dict)
            or plan.get('l1_reference_architecture') != net.reference_transfer['architecture']
            or any(not torch.equal(plan[key], value) for key, value in
                   zip(('support_embeddings', 'owners', 'classes'), support))):
        raise ValueError('Same detached support and its supplied own-reference frozen teacher required')
    for key in ('lr', 'weight_decay', 'grad_clip'):
        if (not math.isfinite(training[key]) or training[key] < 0
                or (key != 'weight_decay' and training[key] == 0)):
            raise ValueError('Original AdamW/clipping settings required')
    if type(training['fused_optimizer']) is not bool:
        raise ValueError('Explicit original fused optimizer policy required')

    saved_rng = rng_state()
    saved_state = hash_state(net.state_dict())
    saved_parameters = _parameters(net)
    saved_gradients = hash_state(_gradients(net))
    saved_modes = tuple(module.training for module in net.modules())
    saved_inputs = _inputs(query, support, plan, targets, ids)
    saved_plan = hash_state(plan)
    results = []
    try:
        with torch.autocast(device.type, enabled=False):
            for name, bn_training, dropout_training in MODES:
                budget.check()
                clone = copy.deepcopy(net)
                restore_rng(saved_rng)
                bns, dropouts = _set_modes(clone, bn_training, dropout_training)
                before = _bn_summary(bns)
                calls, hooks = _bn_hooks(bns)
                try:
                    original_gradients = hash_state(_gradients(clone))
                    loss, terms, logits, embedding = _forward(clone, query, support, plan,
                                                             targets, context, ids)
                    settings = configuration()
                    values = dict(ranking=terms['ranking_loss']*settings['ranking_weight'],
                        observation_ce=terms['observation_auxiliary_loss']*settings['observation_auxiliary_weight'],
                        alignment=terms['alignment_loss']*clone.alignment_loss_weight)
                    cloned_cnn = [(n, p) for n, p in clone.named_parameters()
                                  if p.requires_grad and n.startswith('local.cnn.')]
                    routes = {}
                    for term_name, value in values.items():
                        budget.check()
                        routes[term_name] = _query_route(value, embedding, cloned_cnn)
                    after = _bn_summary(bns, calls)
                    _check_bn(before, after, bn_training)
                    if (_parameters(clone) != saved_parameters
                            or hash_state(_gradients(clone)) != original_gradients
                            or hash_state(plan) != saved_plan):
                        raise AssertionError('Fixed-weight mode derivative control changed parameters/gradients/teacher')
                    results.append(dict(mode=name, bn_training=bn_training, dropout_training=dropout_training,
                        score=_score(logits, targets), **_loss_report(loss, terms, clone.alignment_loss_weight),
                        per_loss_query_CNN_gradients=routes, batch_norm_before=before, batch_norm_after=after,
                        dropout_modules=dropouts, parameter_sha256=saved_parameters,
                        teacher_plan_sha256=saved_plan, input_sha256=saved_inputs,
                        initial_rng_sha256=hash_state(saved_rng), optimizer_updates=0))
                finally:
                    for handle in hooks:
                        handle.remove()
                del clone, loss, terms, logits, embedding, values, cloned_cnn
                budget.check()

            budget.check()
            clone = copy.deepcopy(net)
            restore_rng(saved_rng)
            bns, dropouts = _set_modes(clone, False, False)
            before_bn = _bn_summary(bns)
            calls, hooks = _bn_hooks(bns)
            cloned_named = [(n, p) for n, p in clone.named_parameters() if p.requires_grad]
            initial = [p.detach().clone() for _, p in cloned_named]
            optimizer = torch.optim.AdamW([p for _, p in cloned_named], lr=training['lr'],
                weight_decay=training['weight_decay'], fused=training['fused_optimizer'])
            try:
                optimizer.zero_grad(set_to_none=True)
                loss, terms, logits, embedding = _forward(clone, query, support, plan, targets, context, ids)
                pre = dict(score=_score(logits, targets), **_loss_report(loss, terms, clone.alignment_loss_weight))
                loss.backward()
                gradient_check_batched(clone)
                full_gradients = _module_norms(cloned_named, [p.grad for _, p in cloned_named])
                norm = torch.nn.utils.clip_grad_norm_([p for _, p in cloned_named], training['grad_clip'],
                                                     error_if_nonfinite=True)
                optimizer.step()
                budget.check()
                del loss, terms, logits, embedding
                with torch.no_grad():
                    loss, terms, logits, embedding = _forward(clone, query, support, plan, targets, context, ids)
                post = dict(score=_score(logits, targets), **_loss_report(loss, terms, clone.alignment_loss_weight))
                after_bn = _bn_summary(bns, calls)
                _check_bn(before_bn, after_bn, False, expected_calls=2)
                deltas = _module_norms(cloned_named, [p.detach()-old for (_, p), old in zip(cloned_named, initial)])
                if (any(value is not None and not math.isfinite(value) for value in deltas.values())
                        or any(not deltas[module] for module in ('CNN', 'L1', 'L2'))
                        or hash_state(plan) != saved_plan):
                    raise AssertionError('Frozen-BN full update requires finite CNN/L1/L2 deltas and unchanged teacher')
                frozen_update = dict(mode='eval', bn_training=False, dropout_training=False,
                    before=pre, after=post, batch_norm_before=before_bn, batch_norm_after=after_bn,
                    dropout_modules=dropouts, teacher_plan_sha256=saved_plan,
                    initial_parameter_sha256=saved_parameters, input_sha256=saved_inputs,
                    initial_rng_sha256=hash_state(saved_rng), optimizer_updates=1,
                    optimizer=dict(name='AdamW', history='fresh; not production saved moments',
                        lr=training['lr'], weight_decay=training['weight_decay'],
                        fused=training['fused_optimizer'], grad_clip=training['grad_clip']),
                    gradient_norm_before_clip=float(norm), module_full_gradient_norms=full_gradients,
                    clipping_factor=min(1., training['grad_clip']/(float(norm)+1e-6)),
                    parameter_delta_norms=deltas)
            finally:
                for handle in hooks:
                    handle.remove()
            del clone, optimizer, initial, cloned_named, loss, terms, logits, embedding
    finally:
        restore_rng(saved_rng)
        if (hash_state(net.state_dict()) != saved_state or hash_state(_gradients(net)) != saved_gradients
                or tuple(m.training for m in net.modules()) != saved_modes
                or hash_state(rng_state()) != hash_state(saved_rng)
                or _inputs(query, support, plan, targets, ids) != saved_inputs):
            raise AssertionError('Mode controls changed the supplied reference, gradients, modes, RNG or native inputs')
    return dict(diagnostic_only=True, reference_only=True, modes=results,
        frozen_BN_dropout_off_full_update=frozen_update,
        same_initial_weights_teacher_tile_and_rng=True,
        original_state_gradients_modes_rng_and_inputs_preserved=True,
        physical_batch=len(ids), configured_physical_batch=context.audit['physical_batch'],
        indices=ids, observation_ids=[row['id'] for row in rows], case_id=rows[0]['case_id'],
        observed=int((targets == 1).sum()), unobserved=int((targets == 0).sum()),
        ranking_pairs=int((targets == 1).sum())*int((targets == 0).sum()),
        full_objective=configuration(), full_schedule_audit=copy.deepcopy(context.audit),
        teacher_scope='one supplied own-reference frozen teacher; no fitting or replacement in any mode',
        normalization_scope='unchanged full-cohort LiveContext ranking and CE coefficients',
        production_optimizer_updates=0, production_checkpoint_written=False,
        production_ready=False, full_training=False, full_evaluation=False,
        limitations=['Same physical tile controls and one cloned update are diagnostic, not final accuracy.',
            'BN batch-statistics controls write only cloned BN buffers; no running-stat calibration is claimed.',
            'The separate frozen-BN update changes both BN and dropout modes relative to current training.',
            'Fresh AdamW is not an exact continuation of the saved production optimizer.'])
