"""Explicit diagnostic prefix for cloned L1 interaction candidates.

Both branches reset AdamW and use the same full-cohort loss coefficients, native
query tiles and eligible support records. This is a new short DEBUG experiment,
not an exact resume, a next saved update, or a production checkpoint writer.
"""
from collections import Counter
import math
import os
import random
import time

import numpy as np
import torch
from tqdm.auto import tqdm

from hiercp_v222.v1_execution import rng_state, restore_rng
from l0_local_cnn.model import LocalBatch
from l0_regions.donor_learning import LiveContext, configuration, forward_loss
from l0_regions.execution_pipeline import gradient_check_batched
from l0_regions.resident import check_verified
from l0_regions.training import hash_state
from tools.local_cnn_shadow_probe import MODULES


def _methods(net):
    return tuple((name, type(module), getattr(getattr(module, 'messages', None), '__func__', None))
                 for name, module in net.named_modules())


def _execution_runtime(device):
    return dict(device_type=device.type, autocast_enabled=torch.is_autocast_enabled(device.type),
                configured_autocast_dtype=str(torch.get_autocast_dtype(device.type)),
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
                tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
                tf32_cudnn=torch.backends.cudnn.allow_tf32,
                cudnn_benchmark=torch.backends.cudnn.benchmark,
                cudnn_deterministic=torch.backends.cudnn.deterministic,
                float32_matmul_precision=torch.get_float32_matmul_precision(),
                cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'))


def _module_norms(named, values):
    reports = {}
    for label, prefixes in MODULES.items():
        chosen = [value.detach().float().square().sum() for (name, _), value in zip(named, values)
                  if name.startswith(prefixes)]
        reports[label] = float(torch.stack(chosen).sum().sqrt()) if chosen else None
    reports['global'] = float(torch.stack([value.detach().float().square().sum() for value in values]).sum().sqrt())
    return reports


def _evaluate(provider, net):
    if provider is None:
        return dict(status='NOT_RUN', reason='No explicit matched evaluation provider supplied')
    caller_rng = rng_state()
    modes = [(module, module.training) for module in net.modules()]
    old = hash_state(net.state_dict())
    try:
        net.eval()
        device = next(net.parameters()).device
        with torch.no_grad(), torch.autocast(device.type, enabled=False):
            result = provider(net)
        if not isinstance(result, dict):
            raise ValueError('Candidate evaluation provider must return an explicit metric/scope dictionary')
        if hash_state(net.state_dict()) != old:
            raise AssertionError('Candidate evaluation mutated cloned model weights/buffers')
        return result
    finally:
        for module, training in modes:
            module.training = training
        restore_rng(caller_rng)


def _support(value, ids, context, device):
    if not isinstance(value, (tuple, list)) or len(value) != 3:
        raise ValueError('Explicit support tuple, ordered record IDs and query group required')
    support, record_ids, query_group = value
    if not isinstance(support, tuple) or len(support) != 3:
        raise ValueError('Full support embeddings/owners/classes tuple required')
    x, owners, classes = support
    if x.ndim != 2 or x.shape[1] != 128 or not len(x) or x.requires_grad:
        raise ValueError('Detached complete 128D epoch support memory required')
    if x.dtype != torch.float32 or x.device != device or not bool(torch.isfinite(x).all()):
        raise ValueError('Finite FP32 support on the original model device required')
    if owners.shape != classes.shape or owners.shape != (len(x),):
        raise ValueError('Support identity metadata length differs')
    if owners.dtype != torch.long or classes.dtype != torch.long or owners.device != device or classes.device != device:
        raise ValueError('Support owner/class IDs must be int64 on original device')
    if len(record_ids) != len(x) or len(set(record_ids)) != len(record_ids):
        raise ValueError('Complete unique ordered support record IDs required')
    rows = context.rows
    if {rows[i]['patient_group'] for i in ids} != {query_group}:
        raise ValueError('Support query group differs from the bound query tile')
    lookup = {row['id']: row for row in rows}
    if any(record not in lookup for record in record_ids):
        raise ValueError('Support records are outside the full bound training cohort')
    owner_values, class_values = owners.detach().cpu().tolist(), classes.detach().cpu().tolist()
    owner_groups = {}
    for record, owner, target in zip(record_ids, owner_values, class_values):
        row = lookup[record]
        if target != row['target'] or query_group in (row['patient_group'], row['donor_group']):
            raise ValueError('Support class/patient/donor exclusion binding differs')
        owner_groups.setdefault(owner, set()).add(row['patient_group'])
    if sorted(owner_groups) != list(range(len(owner_groups))) or len(owner_groups) < 2:
        raise ValueError('At least two contiguous nonempty independent support owners required')
    if any(len(group) != 1 for group in owner_groups.values()):
        raise ValueError('Support owner contains different patient groups')
    selected = {next(iter(group)) for group in owner_groups.values()}
    if len(selected) != len(owner_groups):
        raise ValueError('One support patient was duplicated under multiple owners')
    expected = {row['id'] for row in rows if row['patient_group'] in selected
                and query_group not in (row['patient_group'], row['donor_group'])}
    if set(record_ids) != expected:
        raise ValueError('Support dropped eligible observations from a selected patient')
    if set(class_values) != {0, 1}:
        raise ValueError('Both observed classes required in the complete support')
    return support, list(record_ids), query_group


def _query(batch, ids, context, device):
    if not isinstance(batch, LocalBatch):
        raise ValueError('Actual verified native LocalBatch required; no embedding-only update')
    check_verified(batch)
    if len(batch) != len(ids) or batch.indices.detach().cpu().tolist() != ids:
        raise ValueError('Native query batch and scheduled record order differ')
    if batch.images.dtype != torch.float32 or batch.images.device != device:
        raise ValueError('Native FP32 query must use the original model device')
    for offset, index in enumerate(ids):
        row = context.rows[index]
        if (batch.audit[int(batch.recipient[offset])]['case'] != row['case_id'] or
                batch.audit[int(batch.donor[offset])]['case'] != row['donor_case_id']):
            raise ValueError('Native query donor/recipient crop role binding differs')
    return batch


def probe_candidate_updates(net, *, scale, steps, lr, train_tiles, batch_provider, support_provider,
                            loss_context, settings, physical_batch, budget, weight_decay, grad_clip,
                            seed, fused_optimizer=None, evaluation_provider=None, progress=False):
    """Run the entire cloned diagnostic in FP32, preserving caller autocast.

    Disabling only the forward is insufficient when this function is invoked
    inside an ambient autocast context: backward hooks or optimizer internals
    can otherwise observe that context again. The cloned update, evaluation
    callbacks and diagnostics all use the same explicit precision boundary.
    """
    parameter = next(net.parameters(), None)
    if parameter is None:
        raise ValueError('Original trainable FP32 model required')
    with torch.autocast(parameter.device.type, enabled=False):
        return _probe_candidate_updates(net, scale=scale, steps=steps, lr=lr,
            train_tiles=train_tiles, batch_provider=batch_provider, support_provider=support_provider,
            loss_context=loss_context, settings=settings, physical_batch=physical_batch,
            budget=budget, weight_decay=weight_decay, grad_clip=grad_clip, seed=seed,
            fused_optimizer=fused_optimizer, evaluation_provider=evaluation_provider, progress=progress)


def _probe_candidate_updates(net, *, scale, steps, lr, train_tiles, batch_provider, support_provider,
                            loss_context, settings, physical_batch, budget, weight_decay, grad_clip,
                            seed, fused_optimizer, evaluation_provider, progress):
    """Run exactly the caller's short prefix twice, with fresh matched AdamW.

    No original optimizer state is accepted or reused. Each branch fits its own
    teacher plan once per contiguous query-patient episode. A callback can score
    matched held-out query/support data before/after; its RNG use is isolated.
    """
    from tools.local_cnn_interaction_candidate import clone_candidate
    for label, value, minimum in [('steps', steps, 1), ('physical_batch', physical_batch, 2)]:
        if type(value) is not int or value < minimum:
            raise ValueError(f'Explicit {label} >= {minimum} required')
    if type(seed) is not int or seed < 0:
        raise ValueError('Explicit nonnegative diagnostic seed required')
    if type(fused_optimizer) is not bool:
        raise ValueError('Explicit original fused-optimizer policy required')
    if type(progress) is not bool:
        raise ValueError('Diagnostic progress must be an explicit Boolean')
    for label, value, minimum, inclusive in [('scale', scale, 0., True), ('lr', lr, 0., False),
                                          ('weight_decay', weight_decay, 0., True), ('grad_clip', grad_clip, 0., False)]:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or (value < minimum if inclusive else value <= minimum):
            raise ValueError(f'Explicit finite {label} required')
    if not isinstance(loss_context, LiveContext) or settings != configuration():
        raise ValueError('Original full-cohort same-donor loss/context required')
    if not isinstance(train_tiles, list) or len(train_tiles) != loss_context.steps or steps > len(train_tiles):
        raise ValueError('Explicit prefix steps must fit the complete current diagnostic epoch schedule')
    if Counter(tuple(sorted(tile)) for tile in train_tiles) != Counter(tuple(sorted(tile)) for tile in loss_context.order):
        raise ValueError('Diagnostic schedule does not preserve every original ranking tile')
    if any(not tile or len(tile) > physical_batch or len(set(tile)) != len(tile) for tile in train_tiles):
        raise ValueError('Scheduled physical query tile is invalid')
    if loss_context.audit['physical_batch'] != physical_batch:
        raise ValueError('Physical batch differs from full-cohort loss normalization')
    caller_rng = rng_state()
    original_hash, original_methods = hash_state(net.state_dict()), _methods(net)
    original_gradients = hash_state({name: parameter.grad for name, parameter in net.named_parameters()})
    original_modes = tuple(module.training for module in net.modules())
    named_original = [(name, parameter) for name, parameter in net.named_parameters() if parameter.requires_grad]
    if not named_original or any(parameter.dtype != torch.float32 for _, parameter in named_original):
        raise ValueError('Original trainable FP32 model required')
    device = named_original[0][1].device
    if any(parameter.device != device for _, parameter in named_original):
        raise ValueError('One explicit original model device required')
    execution_runtime = _execution_runtime(device)
    matched_inputs, branches, progress_bars = [], [], []
    try:
        for label, factor in [('baseline', 0.), ('candidate', float(scale))]:
            budget.check()
            clone = clone_candidate(net, factor)
            if hash_state(clone.state_dict()) != original_hash:
                raise AssertionError('Candidate clone did not preserve all original weights/buffers')
            named = [(name, parameter) for name, parameter in clone.named_parameters() if parameter.requires_grad]
            if [(name, tuple(p.shape), p.stride()) for name, p in named] != [(name, tuple(p.shape), p.stride()) for name, p in named_original]:
                raise AssertionError('Interaction diagnostic changed trainable structure/layout')
            initial = [parameter.detach().clone() for _, parameter in named]
            optimizer = torch.optim.AdamW((p for _, p in named), lr=lr, weight_decay=weight_decay,
                                          fused=fused_optimizer)
            random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
            before = _evaluate(evaluation_provider, clone)
            clone.train()
            reports, last_group, plan = [], None, None
            prefix_start = time.perf_counter()
            progress_bar = tqdm(enumerate(train_tiles[:steps], start=1), total=steps,
                                desc=f'CLONED L1 {label}', disable=not progress)
            progress_bars.append(progress_bar)
            for step, indices in progress_bar:
                ids = list(indices)
                budget.check()
                query = _query(batch_provider(ids), ids, loss_context, device)
                support, support_ids, group = _support(support_provider(ids), ids, loss_context, device)
                # The verification signature contains object IDs, not data; it
                # is removed for content equality across reconstructed batches.
                query_values = {key: value for key, value in vars(query).items() if key != '_verified_signature'}
                content_hash = hash_state(dict(query=query_values, support=support, support_record_ids=support_ids, group=group))
                if label == 'baseline':
                    matched_inputs.append(content_hash)
                elif content_hash != matched_inputs[step - 1]:
                    raise ValueError('Baseline/candidate native query or eligible support data differ')
                if group != last_group:
                    with torch.no_grad(), torch.autocast(device.type, enabled=False):
                        plan = clone.fit_support_clusters(*support)
                    last_group = group
                elif not torch.equal(plan['support_embeddings'], support[0]):
                    raise ValueError('Support memory changed inside a frozen-plan patient episode')
                targets = torch.tensor([loss_context.rows[i]['target'] for i in ids], dtype=torch.long, device=device)
                weights = torch.tensor([len(loss_context.rows)/(2*loss_context.counts[target]) for target in (0, 1)], device=device)
                optimizer.zero_grad(set_to_none=True)
                if device.type == 'cuda':
                    torch.cuda.synchronize(device)
                start = time.perf_counter()
                with torch.autocast(device.type, enabled=False):
                    loss, terms = forward_loss(clone, query, support, plan, targets, weights,
                                               loss_context, settings, indices=ids)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError('Nonfinite candidate diagnostic production objective')
                loss.backward()
                gradient_check_batched(clone)
                gradients = _module_norms(named, [parameter.grad for _, parameter in named])
                norm = torch.nn.utils.clip_grad_norm_([p for _, p in named], grad_clip, error_if_nonfinite=True)
                optimizer.step()
                if device.type == 'cuda':
                    torch.cuda.synchronize(device)
                seconds = time.perf_counter() - start
                budget.check();check_verified(query)
                if hash_state(dict(query=query_values, support=support, support_record_ids=support_ids, group=group)) != content_hash:
                    raise AssertionError('Candidate update mutated original native query/support inputs')
                weighted_alignment = clone.alignment_loss_weight * terms['alignment_loss']
                torch.testing.assert_close(loss.detach(),
                    terms['ranking_loss']*settings['ranking_weight']+
                    terms['observation_auxiliary_loss']*settings['observation_auxiliary_weight']+weighted_alignment)
                reports.append(dict(step=step, indices=ids, physical_batch=len(ids), query_group=group,
                    ordered_support_record_ids=support_ids, support_records=len(support_ids),
                    support_patient_count=int(support[1].max()) + 1,
                    loss=float(loss.detach()), terms={key: float(value.detach()) for key, value in terms.items()},
                    weighted_alignment_loss=float(weighted_alignment.detach()),
                    gradient_norm_before_clip=float(norm), module_gradient_norms=gradients,
                    clipping_factor=min(1., float(grad_clip/(norm+1e-6))),
                    synchronized_update_seconds=seconds))
                progress_bar.set_postfix(loss=f'{float(loss.detach()):.4g}',
                                         seconds=f'{seconds:.3g}', actual_batch=len(ids))
                del query, support, loss, terms
            progress_bar.close()
            prefix_seconds = time.perf_counter() - prefix_start
            after = _evaluate(evaluation_provider, clone)
            deltas = [parameter.detach() - value for (_, parameter), value in zip(named, initial)]
            if not bool(torch.stack([torch.isfinite(value).all() for value in deltas]).all()):
                raise FloatingPointError('Nonfinite cloned candidate parameter updates')
            branches.append(dict(branch=label, scale=factor, before=before, after=after, updates=reports,
                teacher_plans='Fitted independently from this branch weights; one per contiguous patient episode',
                module_parameter_delta_norms=_module_norms(named, deltas), prefix_seconds=prefix_seconds,
                observed_cuda_peak_bytes=(torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None),
                cuda_peak_scope='Cumulative since caller last reset; this helper does not reset caller peak counters',
                cloned_weight_hash_after=hash_state(clone.state_dict())))
            del clone, optimizer, initial, named, deltas, plan
            budget.check()
        result = dict(diagnostic_only=True, production_optimizer_updates=0, checkpoints_written=0,
            exact_resume=False, next_saved_update=False, explicit_short_prefix=True,
            precision='FP32', autocast_enabled=False,
            execution_runtime=execution_runtime,
            stochastic_comparison='Same explicit seed and dropout stream; without deterministic CUDA algorithms this does not guarantee bitwise repeatability',
            cloned_optimizer_updates_per_branch=steps, physical_batch=physical_batch,
            effective_batch=physical_batch, gradient_accumulation_steps=1, branch_input_content_equal=True,
            optimizer=dict(name='AdamW', history='fresh reset; no saved optimizer moments',
                           lr=lr, weight_decay=weight_decay, grad_clip=grad_clip, seed=seed,
                           fused=fused_optimizer),
            original_weight_hash=original_hash, loss_configuration=settings,
            full_schedule_audit=loss_context.audit,
            prefix_unique_observations=len({index for tile in train_tiles[:steps] for index in tile}),
            full_cohort_observations=len(loss_context.rows), full_schedule_tiles=len(train_tiles),
            branches=branches,
            timing_scope=dict(synchronized_update_seconds='Forward/loss, backward, gradient reporting/check, clipping and AdamW; excludes provider/loading, teacher fit, input hashes and evaluation',
                prefix_seconds='Provider/loading/transfer and verification, support teacher fitting and all updates; excludes cloning and before/after evaluation',
                speedup_claim=False,
                limitation='Cold/warm loader, teacher preparation and GPU cache differ across branches; this short prefix does not establish an epoch speedup'),
            interpretation='Short cloned fresh-optimizer full-objective comparison. Mechanical success is not final accuracy or CP validity; full data, loss weights and network sizes remain original.')
    finally:
        for progress_bar in progress_bars:
            progress_bar.close()
        restore_rng(caller_rng)
        if hash_state(net.state_dict()) != original_hash or _methods(net) != original_methods:
            raise AssertionError('Diagnostic candidate changed original model weights/types/methods')
        if tuple(module.training for module in net.modules()) != original_modes:
            raise AssertionError('Diagnostic candidate changed original module training modes')
        if hash_state({name: parameter.grad for name, parameter in net.named_parameters()}) != original_gradients:
            raise AssertionError('Diagnostic candidate changed original gradient buffers')
    result.update(original_weights_types_methods_modes_preserved=True, caller_rng_restored=True)
    return result
