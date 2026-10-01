"""Fixed-weight L1 transfer controls on one unchanged native physical tile.

Only cloned diagnostic states are used. The caller supplies the same initial
L0 embeddings and detached support to every branch. No optimizer, checkpoint,
training-ready marker, candidate pruning, or production mode change occurs.
Joint BatchNorm changes dependency routing: a support-only alignment formula
can depend on query features through training statistics. This is reported,
not silently removed or described as target leakage.
"""
import copy

import torch
from torch import nn
from torch.nn import functional as F

from hiercp_v222.clustering import prototype_logits
from hiercp_v222.v1_execution import rng_state, restore_rng
from l0_regions.donor_learning import LiveContext, configuration, validate_rows
from l0_regions.training import hash_state
from tools.local_cnn_interaction_updates import _evaluate
from tools.local_cnn_l1_probe import _relation_stages, candidate_signal
from tools.local_cnn_reference_l1 import ReferencePromptGraphModel
from tools.local_cnn_reference_transfer import POLICIES, clone_reference_control


MODES = ('eval_fresh', 'train_joint_batch_statistics_dropout_disabled',
         'eval_after_one_same_tile_bn_pass')


def _gradients(net):
    return {name: parameter.grad for name, parameter in net.named_parameters()}


def _disable_dropout(net):
    """Remove stochastic dropout only in an explicitly labeled isolated probe."""
    changed = []
    for name, module in net.named_modules():
        if isinstance(module, nn.Dropout):
            changed.append(dict(module=name, attribute='p', original=module.p))
            module.p = 0.
        elif isinstance(module, nn.MultiheadAttention):
            changed.append(dict(module=name, attribute='dropout', original=module.dropout))
            module.dropout = 0.
        elif isinstance(module, type(net.l1[0])) and isinstance(net, ReferencePromptGraphModel):
            changed.append(dict(module=name, attribute='dropout', original=module.dropout))
            module.dropout = 0.
    return changed


def _bn(net):
    return {name: dict(num_batches_tracked=int(module.num_batches_tracked),
                      momentum=module.momentum, eps=module.eps,
                      running_mean_norm=float(module.running_mean.detach().norm()),
                      running_var_min=float(module.running_var.detach().min()),
                      running_var_mean=float(module.running_var.detach().mean()),
                      running_var_max=float(module.running_var.detach().max()),
                      buffers_sha256=hash_state(dict(mean=module.running_mean,
                          variance=module.running_var, counter=module.num_batches_tracked)))
            for name, module in net.l1.named_modules() if isinstance(module, nn.BatchNorm1d)}


def _ratios(before, after):
    return {key+'_output_over_input': after[key]/before[key] if before[key] > 0 else None
            for key in ('centered_energy', 'normalized_centered_energy')}


def _stage_report(values):
    reports = {key: candidate_signal(value) for key, value in values.items()}
    before, after = reports['input'], reports['output']
    return dict(stages=reports, **_ratios(before, after))


def _head(net, embeddings, support, plan):
    """Instrument the actual reference run or exact original query equations."""
    if isinstance(net, ReferencePromptGraphModel):
        state = net.prepare_support(*support, cluster_plan=plan)
        result = net.encode_joint(embeddings, *support, return_trace=True)
        # The immutable teacher is the same branch-specific eval-support plan.
        aligned = net.align_labels(result['local_labels'], state['cluster_plan'])
        logits = prototype_logits(result['query'], aligned['labels'], plan, net.temperature)
        start = result['graph']['query_start']
        layers = [dict(layer=row['layer'], **_stage_report({key: row[key][start:]
            for key in ('input', 'aggregate', 'pre_bn', 'operator_output', 'output')}))
            for row in result['trace']]
        return dict(logits=logits, alignment_loss=aligned['alignment_loss'],
                    alignment_loss_weight=net.alignment_loss_weight), layers, result['graph']
    state = net.prepare_support(*support, cluster_plan=plan)
    query, layers = embeddings, []
    for index, (layer, labels) in enumerate(zip(net.l1, state['histories'])):
        terms = _relation_stages(layer, labels, query, 1.)
        src = torch.arange(len(labels), device=query.device).repeat(len(query))
        dst = torch.arange(len(query), device=query.device).repeat_interleave(len(labels))
        actual = layer.messages(labels, query, src, dst, query.new_zeros(len(src), 2))
        torch.testing.assert_close(terms['final_norm'], actual, atol=2e-6, rtol=2e-6)
        values = dict(terms)
        values['output'] = terms['final_norm']
        layers.append(dict(layer=index+1, exact_original_query_trace_verified=True,
                           **_stage_report(values)))
        query = terms['final_norm']
    logits = prototype_logits(query, state['labels'], plan, net.temperature)
    return dict(logits=logits, alignment_loss=state['alignment_loss'],
                alignment_loss_weight=net.alignment_loss_weight), layers, None


def _score(output, truth):
    from tools.diagnose_local_cnn_learning import score_summary
    logits = output['logits'].detach().float()
    if logits.shape != (len(truth), 2) or not bool(torch.isfinite(logits).all()):
        raise FloatingPointError('Finite two-class bound tile logits required')
    score = logits[:, 1]-logits[:, 0]
    common = dict(score_std=float(score.std(unbiased=False)),
                  score_min=float(score.min()), score_max=float(score.max()),
                  candidates=len(truth), observed=int(truth.sum()),
                  scope='exact physical tile; not full-case ranking or validation')
    if len(truth.unique()) != 2:
        return dict(status='NOT_EVALUABLE', reason='The unchanged tile has only one observation class', **common)
    common.update(score_summary(score, truth))
    return dict(status='MEASURED', **common)


def _degree_bias(net, original, graph, patients):
    if graph is None:
        return dict(operator='legacy', placement='once after aggregation, per node',
                    query_incoming_degree=2*patients,
                    layers=[dict(layer=i+1, output_bias_norm=float(layer.update.out.bias.detach().norm()))
                            for i, layer in enumerate(net.l1)])
    degrees = torch.bincount(graph['edge_index'][1], minlength=len(graph['nodes']))
    ns, start = graph['support_count'], graph['query_start']
    roles = {'support_data': degrees[:ns], 'patient_label': degrees[ns:start], 'query': degrees[start:]}
    rows = []
    for index, (new, old) in enumerate(zip(net.l1, original.l1)):
        bias = new.out_proj.bias.detach()
        old_bias = old.update.out.bias.detach()
        audits = {}
        for role, degree in roles.items():
            unique, count = degree.unique(return_counts=True)
            # One bias per incoming edge, regardless of attention coefficient.
            contribution = degree[:, None]*bias
            shift = contribution-old_bias
            audits[role] = dict(nodes=len(degree), degree_min=int(degree.min()),
                degree_max=int(degree.max()), degree_mean=float(degree.float().mean()),
                degree_histogram=[dict(degree=int(d), nodes=int(c)) for d, c in zip(unique, count)],
                official_bias_component_norm_min=float(contribution.norm(dim=1).min()),
                official_bias_component_norm_max=float(contribution.norm(dim=1).max()),
                shift_relative_old_once_bias_norm_min=float(shift.norm(dim=1).min()),
                shift_relative_old_once_bias_norm_max=float(shift.norm(dim=1).max()))
        rows.append(dict(layer=index+1, roles=audits))
    return dict(operator='official reference', placement='per edge before sum',
                formula='incoming_degree * out_proj.bias',
                total_nodes=len(degrees), total_edges=graph['edge_index'].shape[1],
                query_reverse_edges=graph['query_reverse_edges'], self_loops=graph['self_loops'],
                layers=rows)


def _loss_gradient_probe(net, embeddings, support, plan, truth, context, indices):
    """Original coefficient formulas, with only query 128D input derivatives."""
    probe = copy.deepcopy(net)
    probe.train()
    _disable_dropout(probe)
    query = embeddings.detach().clone().requires_grad_(True)
    output = probe.predict_embeddings(query, probe.prepare_support(*support, cluster_plan=plan))
    logits = output['logits'].float()
    score = logits[:, 1]-logits[:, 0]
    p, u = truth == 1, truth == 0
    settings = configuration()
    rank = F.softplus(score[None, u]-score[p, None]).sum()*context.steps/context.pairs
    coefficients = logits.new_tensor([context.steps/(2*context.counts[context.rows[i]['target']]*context.uses[i])
                                      for i in indices])
    ce = (F.cross_entropy(logits, truth, reduction='none')*coefficients).sum()
    terms = dict(ranking=settings['ranking_weight']*rank,
                 observation_ce=settings['observation_auxiliary_weight']*ce,
                 alignment=output['alignment_loss_weight']*output['alignment_loss'])
    results = {}
    for name, value in terms.items():
        if not bool(torch.isfinite(value)):
            raise FloatingPointError('Nonfinite original weighted loss component')
        gradient, = torch.autograd.grad(value, query, retain_graph=True, allow_unused=True)
        if gradient is not None and not bool(torch.isfinite(gradient).all()):
            raise FloatingPointError('Nonfinite query embedding gradient')
        results[name] = dict(weighted_loss=float(value.detach()),
            query_embedding_dependency=gradient is not None,
            query_embedding_gradient_norm=float(gradient.detach().norm()) if gradient is not None else None)
    return dict(losses=results, scope='query L0-output derivatives; CNN parameters are not recomputed here',
        normalization='unchanged complete LiveContext ranking/CE coefficients and original alignment weight',
        physical_batch=len(indices), ranking_pairs=int(p.sum())*int(u.sum()),
        query_targets_used_only_at_loss=True, separate_clone=True,
        dropout_disabled=True, batch_norm=_bn(probe),
        alignment_interpretation='Joint training BatchNorm can route support alignment through query features; no query target enters the graph')


def probe_fixed_weights(net, embeddings, support, *, policies, truth, budget,
                        loss_context, indices, evaluation_provider=None):
    """Compare frozen L0 → L1 → original L2 across explicit transfer controls."""
    if (not policies or len(set(policies)) != len(policies)
            or any(policy not in POLICIES for policy in policies)):
        raise ValueError('Unique explicit reference transfer policies required')
    if not isinstance(loss_context, LiveContext):
        raise ValueError('Original complete-cohort LiveContext required')
    ids = list(indices)
    if (not ids or len(ids) != len(set(ids)) or any(type(i) is not int or i < 0 or i >= len(loss_context.rows) for i in ids)
            or len(ids) > loss_context.audit['physical_batch']):
        raise ValueError('Unchanged unique physical tile indices required')
    if tuple(ids) not in {tuple(tile) for tile in loss_context.order}:
        raise ValueError('Exact ordered tile from the original complete LiveContext schedule required')
    rows = [loss_context.rows[i] for i in ids]
    validate_rows(rows)
    if len({row['case_id'] for row in rows}) != 1:
        raise ValueError('One original recipient case per physical ranking tile required')
    device = next(net.parameters()).device
    if (embeddings.shape != (len(ids), 128) or embeddings.device != device
            or embeddings.dtype != torch.float32 or embeddings.requires_grad
            or truth.shape != (len(ids),) or truth.device != device or truth.dtype != torch.long
            or not torch.equal(truth, torch.tensor([row['target'] for row in rows], device=device))):
        raise ValueError('Detached original FP32 L0 tile and bound truth required')
    if (len(support) != 3 or support[0].ndim != 2 or support[0].shape[1] != 128
            or not len(support[0]) or support[0].requires_grad or support[0].dtype != torch.float32
            or any(value.device != device for value in support)
            or support[1].shape != support[2].shape or support[1].shape != (len(support[0]),)
            or support[1].dtype != torch.long or support[2].dtype != torch.long
            or bool((support[1] < 0).any()) or bool(((support[2] < 0) | (support[2] > 1)).any())
            or len(support[2].unique()) != 2):
        raise ValueError('Complete detached bound support, ownership and both observed classes required')
    patients = int(support[1].max())+1
    if patients < 2 or bool((torch.bincount(support[1], minlength=patients) == 0).any()):
        raise ValueError('Contiguous nonempty eligible support patients required')
    if not bool(torch.isfinite(embeddings).all() & torch.isfinite(support[0]).all()):
        raise FloatingPointError('Nonfinite original query/support; no fallback')
    initial_rng = rng_state()
    initial_state, initial_gradients = hash_state(net.state_dict()), hash_state(_gradients(net))
    initial_modes = tuple(module.training for module in net.modules())
    input_hash = hash_state(dict(embeddings=embeddings, support=support, truth=truth, indices=ids))
    results = []
    try:
        with torch.autocast(device.type, enabled=False):
            for name in ('legacy', *policies):
                budget.check()
                restore_rng(initial_rng)
                clone, contract = ((copy.deepcopy(net), dict(architecture='unchanged_legacy_L1'))
                                   if name == 'legacy' else clone_reference_control(net, policy=name))
                clone.eval()
                dropout = _disable_dropout(clone)
                # Each branch's teacher is computed independently, once. No
                # plan from a different model/branch is supplied by this probe.
                plan = clone.fit_support_clusters(*support)
                plan_hash = hash_state(plan)
                teacher_model_hash = hash_state(clone.state_dict())
                parameter_hash = hash_state(dict(clone.named_parameters()))
                gradient_base = copy.deepcopy(clone)
                snapshots = []
                bias_audit = None
                for mode in MODES:
                    budget.check()
                    clone.train(mode == MODES[1])
                    before = _bn(clone)
                    with torch.no_grad():
                        output, layers, graph = _head(clone, embeddings, support, plan)
                    after = _bn(clone)
                    expected = 1 if mode == MODES[1] else 0
                    if any(after[key]['num_batches_tracked']-before[key]['num_batches_tracked'] != expected
                           for key in before):
                        raise AssertionError('Fixed tile must write BN exactly once in train mode and never in eval')
                    if mode != MODES[1] and before != after:
                        raise AssertionError('Eval fixed-weight probe altered running BN state')
                    if hash_state(plan) != plan_hash:
                        raise AssertionError('Frozen own-branch teacher plan changed during fixed probe')
                    if hash_state(dict(clone.named_parameters())) != parameter_hash:
                        raise AssertionError('Fixed probe changed branch parameter values')
                    snapshots.append(dict(mode=mode, L0=candidate_signal(embeddings),
                        layers=layers, score=_score(output, truth),
                        batch_norm_before=before, batch_norm_after=after,
                        teacher_plan_sha256=plan_hash,
                        teacher_reused_from='same branch initial eval-support pass'))
                    if bias_audit is None:
                        bias_audit = _degree_bias(clone, net, graph, patients)
                gradients = _loss_gradient_probe(gradient_base, embeddings, support, plan,
                                                 truth, loss_context, ids)
                evaluation = _evaluate(evaluation_provider, gradient_base)
                results.append(dict(branch=name, transfer=contract, modes=snapshots,
                    frozen_teacher=dict(own_branch=True, fit_calls=1, sha256=plan_hash,
                        branch=name, transfer_policy=name if name != 'legacy' else None,
                        scope='same physical tile modes and separate derivative clone; optional full-case evaluation fits its own eligible support teachers',
                        basis='own initial cloned L1 eval-support-only; no query features or query targets',
                        model_state_sha256=teacher_model_hash,
                        frozen_parameter_sha256=parameter_hash),
                    dropout_disabled_for_fixed_comparison=dropout,
                    degree_bias=bias_audit, per_loss_query_gradient=gradients,
                    initial_full_case_evaluation=evaluation,
                    same_initial_L0_embeddings_sha256=hash_state(embeddings)))
                del clone, gradient_base, plan
                budget.check()
    finally:
        restore_rng(initial_rng)
        if (hash_state(net.state_dict()) != initial_state or hash_state(_gradients(net)) != initial_gradients
                or tuple(module.training for module in net.modules()) != initial_modes
                or hash_state(rng_state()) != hash_state(initial_rng)):
            raise AssertionError('Fixed controls altered original model, gradients, modes or caller RNG')
        if hash_state(dict(embeddings=embeddings, support=support, truth=truth, indices=ids)) != input_hash:
            raise AssertionError('Fixed controls changed the original bound query/support tile')
    return dict(diagnostic_only=True, fixed_weight=True, optimizer_created=False,
        production_optimizer_updates=0, production_checkpoint_written=False,
        production_ready=False, full_training=False, full_evaluation=False,
        physical_batch=len(ids), configured_physical_batch=loss_context.audit['physical_batch'],
        indices=ids, input_content_sha256=input_hash, branches=results,
        original_state_gradients_modes_rng_and_inputs_preserved=True,
        frozen_L0_embeddings=True, full_objective=configuration(),
        teacher_scope='fresh own-branch eval-support plan; no exact recreation of saved production teacher or RNG update',
        limitations=['Multiple reference operator and initialization changes remain; no single-factor causal attribution',
            'One same-tile BN pass is neither validated calibration nor trained running statistics',
            'Tile pair-win/spread are diagnostic values, not full-case MRR or CP accuracy',
            'A detached-query gradient probe does not independently measure CNN parameter gradients'])
