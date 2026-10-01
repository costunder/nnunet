"""Opt-in CUDA finite-step scoring of the existing copied AdamW shadow arms.

Every arm recomputes the full native query CNN. The supplied saved support and
own teacher remain fixed; all arms use pre-training-forward buffers, eval BN,
and dropout off. This deterministic readout is a different function from the
train-mode derivative source and is not a production loss/model change.
"""
import copy
import time

import torch

from hiercp_v222.v1_execution import rng_state, restore_rng
from l0_regions.donor_learning import LiveContext, configuration, forward_loss
from l0_regions.training import hash_state
from tools.local_cnn_interaction_updates import _query
from tools.local_cnn_reference_causal import capture_prediction, tile_scores


class FiniteShadowScores:
    """One unchanged no-step baseline and sequential exact shadow readouts."""
    def __init__(self, net, query, support, plan, targets, context, indices, budget):
        self.net, self.query, self.support, self.plan = net, query, support, plan
        self.targets, self.context, self.ids, self.budget = targets, context, list(indices), budget
        device = next(net.parameters()).device
        if device.type != 'cuda':
            raise ValueError('Finite shadow scoring requires CUDA; CPU fallback forbidden')
        if (not isinstance(context, LiveContext) or not self.ids
                or len(self.ids) != context.audit['physical_batch']
                or tuple(self.ids) not in {tuple(tile) for tile in context.order}):
            raise ValueError('Original full physical ranking tile required')
        self.rows = [context.rows[i] for i in self.ids]
        if len({row['case_id'] for row in self.rows}) != 1 or {row['target'] for row in self.rows} != {0, 1}:
            raise ValueError('Bound one-case P/U tile required')
        _query(query, self.ids, context, device)
        if (targets.shape != (len(self.ids),) or targets.device != device or targets.dtype != torch.long
                or not torch.equal(targets, torch.tensor([r['target'] for r in self.rows], device=device))):
            raise ValueError('Native query and original observation truth binding required')
        if (not isinstance(support, tuple) or len(support) != 3
                or any(value.device != device or value.requires_grad for value in support)
                or not isinstance(plan, dict)
                or any(not torch.equal(plan[key], value) for key, value in
                       zip(('support_embeddings', 'owners', 'classes'), support))):
            raise ValueError('Same detached episodic support and supplied own frozen teacher required')
        self.named = [(name, p) for name, p in net.named_parameters() if p.requires_grad]
        if not self.named or any(p.dtype != torch.float32 or p.device != device for _, p in self.named):
            raise ValueError('Complete single-device original FP32 parameter path required')
        self.initial_parameters = hash_state(dict(self.named))
        self.buffers = {name: value.detach().clone() for name, value in net.named_buffers()}
        self.initial_buffers = hash_state(self.buffers)
        self.initial_rng = rng_state()
        self.input_hash = self._input_hash()
        self.teacher_hash, self.support_hash = hash_state(plan), hash_state(support)
        self.arms, self.baseline_scores = {}, None
        self.arms['no_change'] = self._evaluate('no_change', None, self.initial_parameters)

    def _input_hash(self):
        native = {key: value for key, value in vars(self.query).items() if key != '_verified_signature'}
        return hash_state(dict(query=native, support=self.support, plan=self.plan,
            targets=self.targets, indices=self.ids,
            context=dict(rows=self.context.rows, order=self.context.order, uses=dict(self.context.uses),
                counts=dict(self.context.counts), steps=self.context.steps, pairs=self.context.pairs,
                audit=self.context.audit)))

    @torch.no_grad()
    def _evaluate(self, branch, parameters, expected_parameter_hash):
        wall_started = time.perf_counter()
        row = None
        caller_rng = rng_state()
        caller_state = hash_state(self.net.state_dict())
        caller_grads = hash_state({n: p.grad for n, p in self.net.named_parameters()})
        caller_modes = tuple(module.training for module in self.net.modules())
        clone = None
        try:
            self.budget.check()
            if self._input_hash() != self.input_hash or hash_state(dict(self.named)) != self.initial_parameters:
                raise AssertionError('Finite scoring changed original native input/teacher or pre-step weights')
            clone = copy.deepcopy(self.net).eval()
            bound = dict(clone.named_buffers())
            if set(bound) != set(self.buffers):
                raise ValueError('Finite readout buffer binding differs')
            for name, value in self.buffers.items():
                bound[name].copy_(value)
            cloned_named = {name: p for name, p in clone.named_parameters() if p.requires_grad}
            if parameters is not None:
                if set(parameters) != set(cloned_named):
                    raise ValueError('Exact complete shadow parameter mapping required')
                for name, p in cloned_named.items():
                    value = parameters[name]
                    if (value.shape != p.shape or value.dtype != p.dtype or value.device != p.device
                            or value.requires_grad or not bool(torch.isfinite(value).all())):
                        raise ValueError('Finite matching detached FP32 CUDA shadow parameters required')
                    p.copy_(value)
            parameter_hash = hash_state(cloned_named)
            if parameter_hash != expected_parameter_hash or hash_state(dict(clone.named_buffers())) != self.initial_buffers:
                raise AssertionError('Finite readout differs from exact parameters or fixed pre-forward buffers')
            restore_rng(self.initial_rng)
            device = next(clone.parameters()).device
            self.budget.check()
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            with torch.autocast('cuda', enabled=False), capture_prediction(clone) as captured:
                loss, terms = forward_loss(clone, self.query, self.support, self.plan, self.targets, None,
                    self.context, configuration(), indices=self.ids)
            torch.cuda.synchronize(device)
            seconds = time.perf_counter()-started
            self.budget.check()
            if len(captured) != 1 or not bool(torch.isfinite(loss)):
                raise FloatingPointError('Exactly one finite original native scoring forward required')
            scores = (captured[0]['logits'][:, 1]-captured[0]['logits'][:, 0]).detach().float()
            row = tile_scores(captured[0], self.targets, self.rows)
            row.update(arm=branch, full_loss=float(loss), terms={k: float(v) for k, v in terms.items()},
                parameter_sha256=parameter_hash, buffer_sha256=self.initial_buffers,
                fixed_teacher_sha256=self.teacher_hash, support_sha256=self.support_hash,
                input_sha256=self.input_hash, initial_rng_sha256=hash_state(self.initial_rng),
                scoring_mode='eval BN; dropout off; all pre-training-forward buffers fixed',
                native_CNN_recomputed=True, cached_pre_step_L0_used=False,
                synchronized_scoring_seconds=seconds,
                execution_policy=dict(deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                    deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
                    cudnn_deterministic=torch.backends.cudnn.deterministic,
                    cudnn_benchmark=torch.backends.cudnn.benchmark,
                    tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
                    tf32_cudnn=torch.backends.cudnn.allow_tf32),
                timing_scope='native CNN + L1/L2 + original loss readout; excludes clone/copy/hash and original optimizer step')
            if self.baseline_scores is None:
                self.baseline_scores = scores.clone()
            difference = scores-self.baseline_scores
            row['score_change_from_no_change'] = dict(mean=float(difference.mean()),
                rms=float(difference.square().mean().sqrt()), maximum_absolute=float(difference.abs().max()),
                positive_minus_unobserved_mean_change=float(difference[self.targets==1].mean()
                    -difference[self.targets==0].mean()))
            if (hash_state(dict(clone.named_buffers())) != self.initial_buffers
                    or hash_state({n:p for n,p in clone.named_parameters() if p.requires_grad}) != parameter_hash):
                raise AssertionError('Eval finite readout advanced buffers or changed its exact parameters')
            del captured, loss, terms, scores, cloned_named, bound
            return row
        finally:
            del clone
            restore_rng(caller_rng)
            if (hash_state(self.net.state_dict()) != caller_state
                    or hash_state({n:p.grad for n,p in self.net.named_parameters()}) != caller_grads
                    or tuple(module.training for module in self.net.modules()) != caller_modes
                    or hash_state(rng_state()) != hash_state(caller_rng)
                    or self._input_hash() != self.input_hash):
                raise AssertionError('Finite scoring mutated original model/gradients/modes/RNG/native input/teacher')
            self.budget.check()
            if row is not None:
                row['total_control_seconds'] = time.perf_counter()-wall_started

    def score_shadow(self, branch, parameters, shadow_report):
        if branch not in ('rank_only', 'full') or branch in self.arms:
            raise ValueError('Exactly one existing rank-only and full shadow readout required')
        row = self._evaluate(branch, parameters, shadow_report['post_parameter_sha256'])
        row['optimizer_shadow'] = dict(delta_sha256=shadow_report['delta_sha256'],
            post_parameter_sha256=shadow_report['post_parameter_sha256'],
            clipped_gradient_sha256=shadow_report['clipped_gradient_sha256'],
            clipping_factor=shadow_report['clipping_factor'])
        self.arms[branch] = row

    def report(self):
        if set(self.arms) != {'no_change', 'rank_only', 'full'}:
            raise ValueError('Finite shadow report requires all three measured arms; missing arms never pass')
        return dict(diagnostic_only=True, arms=self.arms, physical_batch=len(self.ids),
            ranking_pairs=sum(r['target'] for r in self.rows)*sum(1-r['target'] for r in self.rows),
            observation_ids=[r['id'] for r in self.rows], exact_post_step_parameters_used=True,
            fixed_pre_training_forward_buffers=True, same_native_query_support_teacher_rng=True,
            scoring_forwards=3, scoring_seconds=sum(row['synchronized_scoring_seconds'] for row in self.arms.values()),
            total_control_seconds=sum(row['total_control_seconds'] for row in self.arms.values()),
            callback_control_seconds=sum(self.arms[key]['total_control_seconds'] for key in ('rank_only', 'full')),
            derivative_source='existing original train-mode forward; same copied current AdamW moments/global clipping',
            readout_source='isolated deterministic eval-mode clones, same saved episodic support/own teacher',
            CNN_reencoded_for_all_arms=True, original_state_inputs_and_rng_preserved=True,
            production_optimizer_updates=0, production_checkpoint_written=False,
            production_ready=False, full_training=False, full_evaluation=False,
            limitations=['One native P/U tile and fixed teacher are not full-case or segmentation accuracy.',
                'Train-mode derivatives and deterministic eval readout are different functions.',
                'Rank-only shadow retains current AdamW moments, clipping, weight decay and every original parameter.',
                'Full shadow freezes pre-forward BN buffers for comparison; the actual training forward may update its own BN.'])
