"""v2.3 native all-P training with patient-balanced loss and dual validation.

Every candidate local encoder remains differentiable. Candidate chunking only
bounds activation storage; one disjoint-union upper forward retains each full
patient graph. P labels enter the loss and metrics, never the neural inputs.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import copy
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
import time

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from .u_bridge_training import (
    EXPECTED_PARAMETERS, _append, _check_budget, _groups, _probe_after,
    _probe_before, _write_new, atomic_save, capture_rng, cpu_copy, digest,
    gradient_receipt, restore_optimizer_history, restore_rng, retry_amp_overflow,
)


FORMAT = 'v23_native_all_P_patient_balanced_training_v1'
TARGET_POLICIES = ('native_prefix', 'hard_score_top', 'score_stratified_mix')
FIELDS = ('tumor', 'source_context', 'target_context', 'source_relation',
          'target_relation', 'source_c0', 'source_c1', 'source_c2',
          'target_c0', 'target_c1', 'target_c2', 'fused')
CONSISTENCY_FIELDS = ('tumor', 'source_context', 'target_context',
                      'source_relation', 'target_relation', 'fused')


def _positive_indices(plan):
    return tuple(int(index) for index in plan.positive_indices)


def _validate_score_inputs(scores, positive_indices):
    if not isinstance(scores, (tuple, list)) or not scores:
        raise ValueError('Nonempty ordered patient score vectors required')
    if len(scores) != len(positive_indices):
        raise ValueError('Patient score/positive ownership mismatch')
    device = scores[0].device
    for score, positives in zip(scores, positive_indices):
        if not torch.is_tensor(score) or score.ndim != 1 or score.numel() < 2:
            raise ValueError('Complete one-dimensional patient candidate scores required')
        if score.device != device:
            raise ValueError('Patient scores must share a device')
        if (not positives or len(set(positives)) != len(positives)
                or any(type(index) is not int or not 0 <= index < len(score) for index in positives)
                or len(positives) == len(score)):
            raise ValueError('Every training patient requires distinct actual P and U indices')
        torch._assert_async(torch.isfinite(score).all(), 'Nonfinite native patient scores')


def patient_balanced_objective(scores, positive_indices, consistency, *, consistency_weight=.1):
    """Mean U loss per P, mean P per patient, mean patients; no P-P pairs.

    ``consistency`` contains the six-key cosine mean for each actual candidate,
    in concatenated patient/candidate order. It is also averaged within each
    patient before averaging patients, preventing candidate-rich patients from
    dominating either objective component.
    """
    positives = [tuple(indices) for indices in positive_indices]
    _validate_score_inputs(scores, positives)
    if consistency_weight != .1:
        raise ValueError('Original two-view consistency weight0.1 must be preserved')
    counts = [len(score) for score in scores]
    if (not torch.is_tensor(consistency) or consistency.ndim != 1
            or len(consistency) != sum(counts) or consistency.device != scores[0].device):
        raise ValueError('One actual differentiable two-view consistency per candidate required')
    torch._assert_async(torch.isfinite(consistency).all(), 'Nonfinite two-view consistency')
    # Only CPU index metadata is assembled in Python. The complete P x U
    # numerical operation and weighted reduction execute once on the device.
    p_index, u_index, weights = [], [], []
    offset = 0
    for count, positive in zip(counts, positives):
        positive_set = set(positive)
        unobserved = [index for index in range(count) if index not in positive_set]
        weight = 1. / (len(scores) * len(positive) * len(unobserved))
        p_index.extend(offset + p for p in positive for _ in unobserved)
        u_index.extend(offset + u for _ in positive for u in unobserved)
        weights.extend([weight] * (len(positive) * len(unobserved)))
        offset += count
    flat = torch.cat(tuple(score.float() for score in scores))
    device = flat.device
    p = torch.tensor(p_index, device=device, dtype=torch.long)
    u = torch.tensor(u_index, device=device, dtype=torch.long)
    pair_weights = torch.tensor(weights, device=device, dtype=torch.float32)
    ranking = (F.softplus(flat[u] - flat[p]) * pair_weights).sum()
    patient = torch.repeat_interleave(torch.arange(len(counts), device=device),
                                     torch.tensor(counts, device=device))
    means = consistency.float().new_zeros(len(counts)).scatter_add_(0, patient, consistency.float())
    means = means / torch.tensor(counts, device=device, dtype=torch.float32)
    view = means.mean()
    return ranking + consistency_weight * view, dict(ranking=ranking, consistency=view,
        patient_count=len(scores), observed_P=sum(map(len, positives)), pairs=len(p_index))


@dataclass
class V23BatchOutput:
    scores: tuple[torch.Tensor, ...]
    consistency: torch.Tensor
    counts: tuple[int, ...]
    case_ids: tuple[str, ...]
    record_ids: tuple[str, ...]
    audits: tuple[dict, ...]
    workload: dict


class V23Scorer(torch.nn.Module):
    """Original L0/CNN plus a single batched L1/L2/head invocation.

    ``geometry(plan, provider)`` returns the immutable complete CPU patient and
    prototype graphs and its input audit. Providers use original real-CT native
    observations and must certify RNG-free per-observation loading.
    """
    def __init__(self, net, providers, geometry, *, physical_candidate_batch,
                 checkpoint_local_chunks, amp, budget):
        super().__init__()
        if type(physical_candidate_batch) is not int or physical_candidate_batch < 1:
            raise ValueError('An explicitly measured physical candidate batch is required')
        if type(checkpoint_local_chunks) is not bool or type(amp) is not bool:
            raise ValueError('Explicit activation-storage and precision contract required')
        if set(providers) != {'inner_train', 'inner_val'} or not callable(geometry):
            raise ValueError('Separate complete train/validation providers and upper geometry required')
        self.net, self.providers, self.geometry = net, providers, geometry
        self.physical_candidate_batch = physical_candidate_batch
        self.checkpoint_local_chunks, self.amp, self.budget = checkpoint_local_chunks, amp, budget
        self._lookups = {}
        for partition, provider in providers.items():
            rows = provider.ds.rows
            lookup = {row['id']: index for index, row in enumerate(rows)}
            if len(lookup) != len(rows):
                raise ValueError('Native input provider has duplicate record identities')
            self._lookups[partition] = lookup

    def _encode(self, cpu, *, training):
        net = self.net
        device_owner = next(net.parameters())
        def encode(owner):
            device = owner.device
            gpu = cpu.to(device)
            if gpu.graph.num_graphs != 2 * len(gpu):
                raise ValueError('Both actual original sampled local views required')
            with torch.autocast(device.type, enabled=self.amp):
                source, target = net.local_encoder.encode_dense_maps(
                    gpu.source_patches, gpu.source_index, gpu.target_patches)
                owners = gpu.graph_observation_index
                fields = net.local_encoder.forward_graph(gpu.graph,
                    source.index_select(0, owners), target.index_select(0, owners))
                if tuple(fields) != FIELDS:
                    raise ValueError('Original twelve semantic L0 fields changed')
                views = [value.reshape(len(gpu), 2, 128) for value in fields.values()]
                packed = torch.cat([value.mean(1) for value in views], dim=1)
                selected = dict(zip(FIELDS, views))
                consistency = torch.stack([1. - F.cosine_similarity(selected[key][:, 0].float(),
                    selected[key][:, 1].float(), dim=-1) for key in CONSISTENCY_FIELDS]).mean(0)
            torch._assert_async(torch.isfinite(packed).all() & torch.isfinite(consistency).all(),
                                'Nonfinite original local embeddings/view consistency')
            return packed, consistency
        if training and self.checkpoint_local_chunks:
            # Nonreentrant checkpoint supports parameter-only differentiation;
            # the existing real parameter identifies CUDA for RNG capture.
            # Without a Tensor input checkpoint sees no CUDA device and cannot
            # restore dropout draws even when preserve_rng_state is requested.
            return activation_checkpoint(encode, device_owner,
                use_reentrant=False, preserve_rng_state=True)
        return encode(device_owner)

    def forward(self, plans, *, epoch, training):
        from torch_geometric.data import Batch
        if not plans or type(training) is not bool:
            raise ValueError('Nonempty explicit native patient batch required')
        partitions = {plan.partition for plan in plans}
        if len(partitions) != 1:
            raise ValueError('Train and validation patients cannot share a forward batch')
        partition = next(iter(partitions))
        if partition not in self.providers or (training and partition != 'inner_train'):
            raise ValueError('Native partition/training ownership mismatch')
        case_ids = tuple(plan.case_id for plan in plans)
        if len(set(case_ids)) != len(case_ids):
            raise ValueError('Patient batching must not duplicate a patient')
        device = next(self.net.parameters()).device
        if device.type != 'cuda':
            raise RuntimeError('v2.3 scoring requires real CUDA; no CPU fallback')
        provider = self.providers[partition]
        lookup = self._lookups[partition]
        record_ids = tuple(record_id for plan in plans for record_id in plan.record_ids)
        ids = [lookup[record_id] for record_id in record_ids]
        counts = tuple(len(plan.record_ids) for plan in plans)
        graphs, prototypes, audits = [], [], []
        for plan in plans:
            graph, prototype, audit = self.geometry(plan, provider)
            if graph['candidate'].num_nodes != len(plan.record_ids):
                raise ValueError('Joint upper geometry lost complete patient candidates')
            if audit.get('P_U_labels_in_forward') is not False:
                raise ValueError('GT targets cannot enter upper neural geometry')
            graphs.append(graph.clone()); prototypes.append(prototype.clone()); audits.append(copy.deepcopy(audit))
        parts, consistency = [], []
        local_nodes = local_edges = 0
        for start in range(0, len(ids), self.physical_candidate_batch):
            cpu = provider.get(ids[start:start + self.physical_candidate_batch], epoch=epoch)
            local_nodes += sum(store.num_nodes for store in cpu.graph.node_stores)
            local_edges += sum(store.num_edges for store in cpu.graph.edge_stores)
            packed, view = self._encode(cpu, training=training)
            parts.append(packed); consistency.append(view)
            _check_budget(self.budget)
        fields = dict(zip(FIELDS, torch.cat(parts).split(128, dim=1)))
        upper = SimpleNamespace(patient_batch=Batch.from_data_list(graphs).to(device),
            prototype_batch=Batch.from_data_list(prototypes).to(device), counts=counts,
            case_ids=case_ids)
        with torch.autocast(device.type, enabled=self.amp):
            scores = tuple(self.net._score_upper(upper, fields))
        if len(scores) != len(plans) or any(score.shape != (count,) for score, count in zip(scores, counts)):
            raise ValueError('Batched complete upper forward lost patient/candidate mapping')
        _check_budget(self.budget)
        workload = dict(patients=len(plans), candidate_rows=len(ids), local_graph_views=2 * len(ids),
            local_nodes=local_nodes, local_edges=local_edges,
            upper_nodes=sum(store.num_nodes for store in upper.patient_batch.node_stores),
            upper_edges=sum(store.num_edges for store in upper.patient_batch.edge_stores),
            physical_candidate_batch=self.physical_candidate_batch,
            upper_invocations=1, upper_execution='disjoint_union_complete_patient_graphs',
            query_GT_in_forward=False, input_shape=[len(ids), 5, 48, 48, 48],
            candidate_chunking_is_model_or_graph_truncation=False)
        return V23BatchOutput(scores, torch.cat(consistency), counts, case_ids, record_ids,
                              tuple(audits), workload)


def score_patient(score, plan):
    """Actual per-P-vs-U gate metrics plus full-set first-P comparison metrics."""
    value = torch.as_tensor(score).detach().double().cpu()
    if value.shape != (len(plan.record_ids),) or not bool(torch.isfinite(value).all()):
        raise ValueError('One finite actual score per native candidate required')
    positive = torch.tensor(_positive_indices(plan), dtype=torch.long)
    unobserved = torch.tensor(tuple(plan.unobserved_indices), dtype=torch.long)
    if len(unobserved) == 0:
        raise ValueError('Native ranking requires actual U comparisons')
    if len(positive):
        difference = value[positive, None] - value[unobserved][None, :]
        ranks = 1 + (difference <= 0).sum(1)
        per_p = dict(mrr=float(ranks.double().reciprocal().mean()), top1=float((ranks == 1).double().mean()),
            ranks=ranks.tolist(), pair_win=float((difference > 0).double().mean()),
            pair_loss=float(F.softplus(-difference).mean()))
        # Fixed deterministic key order matches historical native evaluation.
        from tools.v22_candidate_order import candidate_order, record_key
        order = candidate_order(value.tolist(), [record_key(row) for row in plan.rows]).tolist()
        observed = set(positive.tolist())
        first = next(index + 1 for index, candidate in enumerate(order) if candidate in observed)
        recalled = {k: sum(candidate in observed for candidate in order[:k]) for k in (1, 5, 10)}
    else:
        per_p, first, recalled = None, None, {k: 0 for k in (1, 5, 10)}
    return dict(case_id=plan.case_id, record_ids=list(plan.record_ids), scores=value.tolist(),
        positive_indices=positive.tolist(), unobserved_indices=unobserved.tolist(),
        observed_P=len(positive), unobserved_U=len(unobserved), per_P=per_p,
        first_P_rank=first, recalled_observed_P=recalled,
        donor_case_id=plan.rows[0]['donor_case_id'], donor_component=plan.rows[0]['donor_component'])


def aggregate_patients(rows):
    if not rows or len({row['case_id'] for row in rows}) != len(rows):
        raise ValueError('Complete unique patient evaluation rows required')
    valid = [row for row in rows if row['observed_P']]
    if not valid:
        raise ValueError('No observed-P patient; ranking metrics are undefined')
    total_p = sum(row['observed_P'] for row in rows)
    pairs = sum(row['observed_P'] * row['unobserved_U'] for row in valid)
    mean = lambda values: math.fsum(values) / len(valid)
    return dict(denominators=dict(patients=len(rows), ranking_patients=len(valid), zero_P_patients=len(rows) - len(valid),
        observed_P=total_p, unobserved_U=sum(row['unobserved_U'] for row in rows), P_U_pairs=pairs),
        metrics=dict(per_P_patient_mrr=mean([row['per_P']['mrr'] for row in valid]),
            per_P_patient_top1=mean([row['per_P']['top1'] for row in valid]),
            patient_balanced_pair_loss=mean([row['per_P']['pair_loss'] for row in valid]),
            patient_balanced_pair_win=mean([row['per_P']['pair_win'] for row in valid]),
            case_first_P_mrr=mean([1. / row['first_P_rank'] for row in valid]),
            case_hit_at_1=mean([float(row['first_P_rank'] == 1) for row in valid]),
            **{f'observed_micro_recall_at_{k}': math.fsum(row['recalled_observed_P'][k] if k in row['recalled_observed_P']
                  else row['recalled_observed_P'][str(k)] for row in rows) / total_p for k in (1, 5, 10)}),
        cases=copy.deepcopy(rows))


def best_selection_key(report):
    metrics = report['metrics']
    return (metrics['per_P_patient_mrr'], metrics['per_P_patient_top1'], -metrics['patient_balanced_pair_loss'])


def new_curriculum(*, initial_u, increment_u, total_u, gate_mrr, gate_top1,
                   minimum_stage_epochs, consecutive_passes):
    for key, value in dict(initial_u=initial_u, increment_u=increment_u, total_u=total_u,
                           minimum_stage_epochs=minimum_stage_epochs, consecutive_passes=consecutive_passes).items():
        if type(value) is not int or value < 1:
            raise ValueError('Explicit positive integer curriculum setting required: ' + key)
    if initial_u > total_u or not 0 <= gate_mrr <= 1 or not 0 <= gate_top1 <= 1:
        raise ValueError('Invalid cumulative native-U gate contract')
    return dict(active_u=initial_u, initial_u=initial_u, increment_u=increment_u, total_u=total_u,
        gate_mrr=gate_mrr, gate_top1=gate_top1, minimum_stage_epochs=minimum_stage_epochs,
        consecutive_passes=consecutive_passes, stage_id=0, stage_completed_epochs=0,
        pass_streak=0, last_completed_epoch=0, transitions=[], all_P_active_from_start=True)


def finish_curriculum_epoch(curriculum, stage_report, *, epoch, successful_train_cases,
                            expected_train_cases, updates_in_epoch):
    """Expand only after real complete epochs and held-out per-P gate passes."""
    current = copy.deepcopy(curriculum)
    if (type(epoch) is not int or epoch != current['last_completed_epoch'] + 1
            or type(updates_in_epoch) is not int or updates_in_epoch < 1
            or len(successful_train_cases) != len(set(successful_train_cases))
            or set(successful_train_cases) != set(expected_train_cases)):
        raise ValueError('Cumulative expansion requires exactly complete actual training coverage')
    if stage_report.get('active_u') != current['active_u'] or stage_report.get('epoch') != epoch:
        raise ValueError('Stage validation was scored under another actual candidate plan/epoch')
    metrics = stage_report['metrics']
    gate = metrics['per_P_patient_mrr'] >= current['gate_mrr'] and metrics['per_P_patient_top1'] >= current['gate_top1']
    current['stage_completed_epochs'] += 1
    current['pass_streak'] = current['pass_streak'] + 1 if gate else 0
    expanded = (current['active_u'] < current['total_u']
        and current['stage_completed_epochs'] >= current['minimum_stage_epochs']
        and current['pass_streak'] >= current['consecutive_passes'])
    transition = dict(epoch=epoch, completed_stage=current['stage_id'], previous_active_u=current['active_u'],
        gate_passed=gate, stage_completed_epochs=current['stage_completed_epochs'], pass_streak=current['pass_streak'], expanded=expanded)
    if expanded:
        current['active_u'] = min(current['total_u'], current['active_u'] + current['increment_u'])
        current['stage_id'] += 1; current['stage_completed_epochs'] = 0; current['pass_streak'] = 0
    transition['next_active_u'] = current['active_u']
    current['last_completed_epoch'] = epoch
    current['transitions'].append(transition)
    return current, transition


def _plans(population, case_ids, active_u, target_selection=None):
    if target_selection is None or active_u == 128:
        return [population.case(case_id, active_u_count=active_u) for case_id in case_ids]
    return [population.case(case_id, active_u_count=active_u,
                           active_u_indices=target_selection[case_id]) for case_id in case_ids]


def new_target_state(population, policy, active_u, *, selections=None, receipts=()):
    """Freeze actual U membership across training, both gate checks and resume."""
    from .v23_targets import target_selection_manifest, validate_target_selection
    if policy not in TARGET_POLICIES:
        raise ValueError('Explicit supported native-U target management policy required')
    cases = list(population.partition_cases('inner_train', ranking_only=True)) + list(
        population.partition_cases('inner_val'))
    if selections is None:
        selections = {case: list(range(active_u)) for case in cases}
    selections = validate_target_selection(population, active_u, selections)
    manifest = target_selection_manifest(population, active_u, selections)
    return dict(policy=policy, active_u=active_u, selections=selections,
                selection_manifest=manifest, receipts=copy.deepcopy(list(receipts)))


def validate_target_state(population, targets, *, policy, active_u):
    expected = new_target_state(population, policy, active_u,
        selections=targets['selections'], receipts=targets['receipts'])
    if digest(targets) != digest(expected):
        raise ValueError('Checkpointed native-U membership/policy/manifest changed')
    return expected


def bank_u_scores(score, plan):
    """Extract only actual U scores in the original full128 bank order."""
    values = torch.as_tensor(score).detach().double().cpu()
    if (plan.active_u_count != 128 or values.shape != (len(plan.record_ids),)
            or len(plan.unobserved_indices) != 128 or not bool(torch.isfinite(values).all())):
        raise ValueError('Actual complete finite full128 forward required for target mining')
    positions = tuple(plan.unobserved_bank_positions)
    if sorted(positions) != list(range(128)):
        raise ValueError('Full native U bank position ownership changed')
    result = [None] * 128
    for local, bank in zip(plan.unobserved_indices, positions):
        result[bank] = float(values[local])
    return result


def advance_target_state(population, targets, active_u, *, epoch, model_sha256, mining_rows=None):
    """Admit new U only at a completed gate; existing U remain unchanged."""
    from .v23_targets import admit_targets
    previous = targets['active_u']
    if not previous < active_u <= 128:
        raise ValueError('Target admission must be a real cumulative expansion')
    policy = targets['policy']
    cases = list(targets['selections'])
    if policy == 'native_prefix':
        selections = {case: list(range(active_u)) for case in cases}
        receipt = dict(policy=policy, epoch=epoch, previous_active_u=previous,
            active_u=active_u, native_bank_order_preserved=True, score_mining=False)
    else:
        if (not mining_rows or [row['case_id'] for row in mining_rows] != cases
                or any(row['model_sha256'] != model_sha256 for row in mining_rows)):
            raise ValueError('Complete fixed-model train/validation U-only mining coverage required')
        selections, admissions = {}, {}
        for row in mining_rows:
            case = row['case_id']
            result = admit_targets(policy, targets['selections'][case], row['u_scores'],
                add_count=active_u-previous, seed=42, case_id=case,
                epoch=epoch, model_sha256=model_sha256)
            selections[case], admissions[case] = result['indices'], result['receipt']
        receipt = dict(policy=policy, epoch=epoch, previous_active_u=previous,
            active_u=active_u, score_mining=True, no_grad=True, U_scores_only=True,
            model_sha256=model_sha256, cases=admissions,
            mining_rows_sha256=digest(mining_rows))
    for case in cases:
        if not set(targets['selections'][case]).issubset(selections[case]):
            raise ValueError('Target management cannot remove a previously admitted U')
    receipt['sha256'] = digest(receipt)
    return new_target_state(population, policy, active_u, selections=selections,
        receipts=[*targets['receipts'], receipt]), receipt


def execution_identity(request, continuation=None):
    identity = dict(request_sha256=request['request_sha256'], source=request['source'],
        initialization='exact original shared random initialization; no trained weights')
    if continuation is not None:
        identity['initialization'] = 'original shared random initialization; explicit checkpoint continuation'
        identity['checkpoint_continuation'] = cpu_copy(continuation)
    return identity


def apply_patient_order(population, state, policy, random_order):
    from .v23_targets import patient_epoch_order
    cases = list(population.partition_cases('inner_train', ranking_only=True))
    previous = state['history'][-1]['train']['cases'] if state['history'] else []
    if not state['history']:
        if state['epoch'] != 1 or state['updates'] != 0 or state['train_position'] != 0:
            raise ValueError('Missing preceding completed TRAIN history outside explicit cold start')
        result = patient_epoch_order(cases, previous, 'native_prefix', random_order)
        if policy != 'native_prefix':
            receipt = dict(format='v23_explicit_first_epoch_seeded_patient_order_bootstrap_v1',
                requested_policy=policy, bootstrap_reason='no completed TRAIN epoch yet',
                original_seeded_order_receipt=result['receipt'], order=result['order'],
                fake_losses_used=False, prior_validation_used=False, full_patient_coverage=True)
            receipt['sha256'] = digest(receipt)
            result['receipt'] = receipt
    else:
        result = patient_epoch_order(cases, previous, policy, random_order)
    if (len(result['order']) != len(cases) or set(result['order']) != set(cases)
            or len(result['order']) != len(set(result['order']))):
        raise ValueError('Patient target scheduling must retain all training patients exactly once')
    return result


def training_binding(config, population, *, identity, physical_patient_batch,
                     physical_candidate_batch, world_size, workers, initial_hash, epochs=40, debug=False):
    bound_config = copy.deepcopy(config)
    for key in ('resume_checkpoint', 'pause_file', 'debug_pause_after_updates'):
        bound_config['v23_runtime'].pop(key, None)
    return dict(format=FORMAT, identity=cpu_copy(identity), config=bound_config,
        population=population.manifest(),
        train_cases=list(population.partition_cases('inner_train', ranking_only=True)),
        val_cases=list(population.partition_cases('inner_val')), epochs=epochs, debug=debug,
        physical_patient_batch=physical_patient_batch,
        physical_candidate_batch=physical_candidate_batch,
        data_parallel_world_size=world_size, workers=workers, initial_state_sha256=initial_hash)


def remaining_training_updates(state, train_cases, physical_patient_batch, world_size, epochs=40):
    """Count only unprocessed batches; a continuation does not restart forty epochs."""
    batches = parallel_patient_batches(state['train_order'] or train_cases, physical_patient_batch, world_size)
    if state['phase'] == 'training':
        cursor = 0; current = 0
        for batch in batches:
            if cursor >= state['train_position']: current += 1
            cursor += len(batch)
    elif state['phase'] == 'initial_full_validation':
        current = len(batches)
    else:
        current = 0
    return current + max(0, epochs-state['epoch']) * len(parallel_patient_batches(
        train_cases, physical_patient_batch, world_size))


def _file_sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(8 * 1024**2), b''):
            value.update(part)
    return value.hexdigest()


def publish_epoch_curve(path, curve):
    """Admit an already-published owned epoch after interruption, exactly once."""
    path = Path(path)
    matches = []
    if path.exists():
        if path.is_symlink(): raise ValueError('Owned curve must be a regular file')
        with path.open(encoding='utf8') as stream:
            for line in stream:
                stored = json.loads(line)
                if stored['epoch'] == curve['epoch']: matches.append(stored)
    if matches:
        if len(matches) != 1 or digest(matches[0]) != digest(curve):
            raise ValueError('Previously published owned epoch curve differs or is duplicated')
        return
    _append(path, curve)


def handoff_single_gpu_checkpoint(source_checkpoint, source_identity, destination_request,
        config, calibration, output, *, source_proof, population,
        physical_patient_batch=4, source_best_checkpoint=None, debug=False):
    """Explicit verified fork, preserving trained state and forty total epochs.

    This performs CPU file verification/copy only. The caller verifies old
    source and geometry/calibration receipts and supplies those immutable
    proofs. No source file, old checkpoint or server process is modified.
    """
    from .contracts import canonical_hash
    source_checkpoint, source_identity, output = map(Path, (source_checkpoint, source_identity, output))
    if any(path.is_symlink() or not path.is_file() for path in (source_checkpoint, source_identity)):
        raise ValueError('Regular immutable source checkpoint/identity required')
    required = ('source_commit', 'source_request', 'source_files_sha256',
        'source_checkpoint_file_sha256', 'source_identity_file_sha256',
        'geometry_receipt_sha256', 'calibration_receipt_sha256',
        'core_equations_preserved', 'original_model_parameters')
    if any(key not in source_proof for key in required):
        raise ValueError('Explicit verified source/geometry/calibration handoff proof required')
    if (source_proof['core_equations_preserved'] is not True
            or (not debug and source_proof['original_model_parameters'] != EXPECTED_PARAMETERS)):
        raise ValueError('Handoff cannot change original model, loss or numerical equations')
    for key in ('source_checkpoint_file_sha256', 'source_identity_file_sha256',
                'geometry_receipt_sha256', 'calibration_receipt_sha256'):
        if (len(source_proof[key]) != 64 or any(c not in '0123456789abcdef' for c in source_proof[key])):
            raise ValueError('Exact handoff SHA256 proof required: ' + key)
    if (_file_sha256(source_checkpoint) != source_proof['source_checkpoint_file_sha256']
            or _file_sha256(source_identity) != source_proof['source_identity_file_sha256']):
        raise ValueError('Source checkpoint/identity file SHA256 differs')
    ownership = json.loads(source_identity.read_text(encoding='utf8'))
    old_binding = ownership['binding']
    saved = torch.load(source_checkpoint, map_location='cpu', weights_only=False)
    checksum = saved.get('content_sha256')
    if (saved.get('format') != FORMAT or checksum != digest({k:v for k,v in saved.items() if k != 'content_sha256'})
            or saved['identity_sha256'] != ownership['identity_sha256']
            or ownership['identity_sha256'] != digest(old_binding)):
        raise ValueError('Source checkpoint content or ownership identity differs')
    old_request = source_proof['source_request']
    if (canonical_hash({k:v for k,v in old_request.items() if k != 'request_sha256'}) != old_request['request_sha256']
            or old_binding['identity']['request_sha256'] != old_request['request_sha256']
            or old_binding['identity']['source'] != old_request['source']
            or source_proof['source_files_sha256'] != old_request['source']):
        raise ValueError('Verified source request/source SHA binding differs')
    if (canonical_hash({k:v for k,v in destination_request.items() if k != 'request_sha256'})
            != destination_request['request_sha256']):
        raise ValueError('Destination singleton request SHA differs')
    if (len(destination_request['gpus']) != 1 or destination_request['gpus'][0] not in (1,5,6)
            or old_binding['data_parallel_world_size'] != 3 or len(saved['rank_rng']) != 3
            or physical_patient_batch != 4 or old_binding['physical_patient_batch'] != 4):
        raise ValueError('Only explicit measured physical4 world3-to-singleton handoff is supported')
    if (old_binding['epochs'] != 40 or old_binding['debug'] is not debug
            or config['v23_runtime'].get('debug') is not debug
            or digest(old_binding['population']) != digest(population.manifest())):
        raise ValueError('Handoff must preserve complete production population and forty total epochs')
    for key in ('model', 'graph', 'training', 'ct_clip', 'runtime'):
        if digest(config[key]) != digest(old_binding['config'][key]):
            raise ValueError('Original model/graph/optimizer/precision setting changed: ' + key)
    if (calibration['world_size'] != 1 or calibration.get('request_sha256') != destination_request['request_sha256']
            or calibration.get('physical_GPUs') != destination_request['gpus']
            or not calibration['measured_full_P_U128_backward']
            or not calibration['original_model_and_RNG_preserved']
            or calibration['initial_state_sha256'] != old_binding['initial_state_sha256']
            or calibration['selected_physical_patient_batch'] != physical_patient_batch
            or calibration['selected_physical_candidate_batch'] != old_binding['physical_candidate_batch']):
        raise ValueError('Explicit verified per-GPU full128 physical4 calibration rebind required')
    state = saved['state']
    if state['phase'] == 'complete' or state['epoch'] > 40:
        raise ValueError('Completed forty-epoch training cannot receive forty additional epochs')
    if state.get('status') == 'FAILED' and not source_proof.get('diagnosed_recovery'):
        raise ValueError('Failed source requires an explicit diagnosed recovery receipt')
    scheduler = SimpleNamespace(last_epoch=saved['scheduler']['last_epoch'])
    _validate_progress(state, old_binding['train_cases'], old_binding['val_cases'], 4, 40, scheduler, 3)
    _validate_progress(state, old_binding['train_cases'], old_binding['val_cases'], 4, 40, scheduler, 1)
    for slot in saved['optimizer']['state'].values():
        step = slot['step']
        if int(step.item() if torch.is_tensor(step) else step) != state['updates']:
            raise ValueError('Source optimizer step history differs from successful updates')
    continuation = dict(source_commit=source_proof['source_commit'],
        source_checkpoint=str(source_checkpoint.resolve()), source_identity=str(source_identity.resolve()),
        source_checkpoint_file_sha256=source_proof['source_checkpoint_file_sha256'],
        source_checkpoint_content_sha256=checksum, source_identity_sha256=saved['identity_sha256'],
        source_request_sha256=old_request['request_sha256'], source_world_size=3,
        source_rank_rng_index=0, destination_world_size=1,
        destination_physical_GPU=destination_request['gpus'][0],
        inherited_epoch=state['epoch'], inherited_phase=state['phase'], inherited_updates=state['updates'],
        geometry_receipt_sha256=source_proof['geometry_receipt_sha256'],
        calibration_receipt_sha256=source_proof['calibration_receipt_sha256'])
    identity = execution_identity(destination_request, continuation)
    config = copy.deepcopy(config); config['v23_runtime']['batch_calibration'] = copy.deepcopy(calibration)
    binding = training_binding(config, population, identity=identity,
        physical_patient_batch=4, physical_candidate_batch=old_binding['physical_candidate_batch'],
        world_size=1, workers=config['v23_runtime']['workers'],
        initial_hash=old_binding['initial_state_sha256'], debug=debug)
    fork = cpu_copy(saved)
    fork['identity_sha256'] = digest(binding)
    fork['rank_rng'] = [cpu_copy(saved['rank_rng'][0])]
    policy = config['v23_runtime'].get('target_selection_policy', 'native_prefix')
    if 'targets' in state:
        selections = state['targets']['selections']
        receipts = state['targets']['receipts']
    else:
        selections, receipts = None, ()
    fork['state']['targets'] = new_target_state(population, policy, state['curriculum']['active_u'],
        selections=selections, receipts=receipts)
    order_receipt = None
    if (fork['state']['phase'] == 'training' and fork['state']['train_position'] == 0
            and fork['state']['train_order'] is not None):
        ordered = apply_patient_order(population, fork['state'], policy, fork['state']['train_order'])
        fork['state']['train_order'] = ordered['order']
        order_receipt = ordered['receipt']
        fork['state'].setdefault('patient_order_receipts', []).append(order_receipt)
    fork['state']['handoff'] = continuation
    fork.pop('content_sha256'); fork['content_sha256'] = digest(fork)
    preserved = ('model', 'optimizer', 'scheduler', 'scaler', 'shuffle_generator')
    if any(digest(fork[key]) != digest(saved[key]) for key in preserved):
        raise ValueError('Handoff changed trained numerical state')
    if (_file_sha256(source_checkpoint) != source_proof['source_checkpoint_file_sha256']
            or _file_sha256(source_identity) != source_proof['source_identity_file_sha256']):
        raise ValueError('Source changed during handoff snapshot')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Handoff requires a fresh training namespace; existing results preserved')
    best_fork, best_proof = None, None
    if state['best'] is not None:
        if source_best_checkpoint is None or 'source_best_checkpoint_file_sha256' not in source_proof:
            raise ValueError('Actual inherited BEST checkpoint and file proof required; latest cannot substitute BEST')
        source_best_checkpoint = Path(source_best_checkpoint)
        if (source_best_checkpoint.is_symlink() or not source_best_checkpoint.is_file()
                or _file_sha256(source_best_checkpoint) != source_proof['source_best_checkpoint_file_sha256']):
            raise ValueError('Source BEST checkpoint file proof differs')
        source_best = torch.load(source_best_checkpoint, map_location='cpu', weights_only=False)
        best_checksum = source_best.get('content_sha256')
        if (source_best.get('format') != FORMAT or source_best['identity_sha256'] != saved['identity_sha256']
                or best_checksum != digest({k:v for k,v in source_best.items() if k != 'content_sha256'})
                or len(source_best['rank_rng']) != 3
                or digest(source_best['state']['best']) != digest(state['best'])):
            raise ValueError('Actual source BEST ownership/metric/content differs')
        best_fork = cpu_copy(source_best)
        best_fork['identity_sha256'] = digest(binding)
        best_fork['rank_rng'] = [cpu_copy(source_best['rank_rng'][0])]
        best_fork['state']['targets'] = new_target_state(population, policy,
            source_best['state']['curriculum']['active_u'])
        best_proof = dict(source_checkpoint=str(source_best_checkpoint.resolve()),
            source_file_sha256=source_proof['source_best_checkpoint_file_sha256'],
            source_content_sha256=best_checksum, model_sha256=digest(source_best['model']),
            best_record=cpu_copy(source_best['state']['best']),
            preserved_state_sha256={key:digest(source_best[key]) for key in preserved},
            actual_best_weights_preserved=True, latest_weights_substituted=False)
        best_fork['state']['handoff'] = dict(**continuation, inherited_best_origin=best_proof)
        best_fork.pop('content_sha256'); best_fork['content_sha256'] = digest(best_fork)
        if (any(digest(best_fork[key]) != digest(source_best[key]) for key in preserved)
                or _file_sha256(source_best_checkpoint) != source_proof['source_best_checkpoint_file_sha256']):
            raise ValueError('Source BEST changed during verified handoff')
    output.mkdir(parents=True, exist_ok=True)
    receipt = dict(format='v23_verified_world3_to_single_gpu_checkpoint_handoff_v1',
        destination_request_sha256=destination_request['request_sha256'], destination_identity=identity,
        destination_binding_sha256=digest(binding), continuation=continuation,
        source_proof=cpu_copy(source_proof), preserved_state_sha256={key:digest(saved[key]) for key in preserved},
        source_rank0_rng_sha256=digest(saved['rank_rng'][0]),
        source_history_sha256=digest(state['history']), source_best_record=cpu_copy(state['best']),
        patient_order_change_at_empty_cursor=order_receipt,
        inherited_successful_train_rows_sha256=digest(state['train_rows']),
        inherited_validation_rows_sha256=digest(state['evaluation_rows']),
        destination_checkpoint_content_sha256=fork['content_sha256'],
        total_target_epochs=40, completed_epochs=len(state['history']),
        remaining_epochs=40-len(state['history']), old_effective_patient_batch=12,
        new_effective_patient_batch=4, physical_patient_batch=4,
        target_policy=policy, no_new_optimizer_update=True, original_files_written=False,
        inherited_best_checkpoint=best_proof,
        destination_best_checkpoint_content_sha256=None if best_fork is None else best_fork['content_sha256'], debug=debug)
    _write_new(output/'training_identity.json', dict(identity_sha256=digest(binding), binding=cpu_copy(binding)))
    atomic_save(output/'checkpoint_latest.pt', fork)
    if best_fork is not None: atomic_save(output/'checkpoint_best.pt', best_fork)
    _write_new(output/'handoff_receipt.json', receipt)
    return receipt


def parallel_patient_batches(case_ids, physical_patient_batch, world_size):
    """Global batches with all patients once and nonempty local tail batches."""
    if (type(physical_patient_batch) is not int or physical_patient_batch < 2
            or type(world_size) is not int or world_size < 1
            or len(case_ids) != len(set(case_ids)) or len(case_ids) < world_size):
        raise ValueError('Unique complete patients and measured local batching required')
    maximum = physical_patient_batch * world_size
    batches = [list(case_ids[start:start+maximum]) for start in range(0, len(case_ids), maximum)]
    if len(batches) > 1 and len(batches[-1]) < world_size:
        # Rebalance the last two batches within the calibrated maximum rather
        # than duplicate a patient or let a rank skip its optimizer collective.
        combined = batches[-2] + batches[-1]
        split = len(combined) // 2
        batches[-2:] = [combined[:split], combined[split:]]
    if any(not world_size <= len(batch) <= maximum for batch in batches):
        raise ValueError('Population cannot form nonempty rank batches within measured capacity')
    return batches


def rank_cases(global_cases, world_size, rank):
    if not 0 <= rank < world_size or len(global_cases) < world_size:
        raise ValueError('Each data-parallel rank must receive real patients')
    quotient, remainder = divmod(len(global_cases), world_size)
    start = rank * quotient + min(rank, remainder)
    count = quotient + (rank < remainder)
    return list(global_cases[start:start+count])


class _Distributed:
    def __init__(self):
        import torch.distributed as dist
        self.dist = dist
        active = dist.is_available() and dist.is_initialized()
        self.world = dist.get_world_size() if active else 1
        self.rank = dist.get_rank() if active else 0

    def gather(self, value):
        if self.world == 1: return [value]
        values = [None] * self.world
        self.dist.all_gather_object(values, value)
        return values

    def rows(self, rows):
        return [row for part in self.gather(rows) for row in part]

    def root_action(self, action):
        # I/O errors are broadcast explicitly; peers cannot silently wait at a
        # checkpoint collective after rank0 fails to write its owned output.
        outcome = [None]
        if self.rank == 0:
            try:
                outcome[0] = dict(ok=True, value=action())
            except Exception as error:
                outcome[0] = dict(ok=False, error=f'{type(error).__name__}: {error}')
        if self.world > 1: self.dist.broadcast_object_list(outcome, src=0)
        if not outcome[0]['ok']:
            raise RuntimeError('Rank0 operation failed; all ranks safely stop: ' + outcome[0]['error'])
        return outcome[0]['value']

    def prepare_geometry(self, geometry, active_u, target_selection=None):
        prepare = getattr(geometry, 'prepare', None)
        if callable(prepare): self.root_action(lambda: prepare(active_u, target_selection=target_selection)
            if target_selection is not None else prepare(active_u))
        if self.world > 1: self.dist.barrier()
        admit = getattr(geometry, 'admit', None)
        if callable(admit):
            admit(active_u, target_selection=target_selection) if target_selection is not None else admit(active_u)
        elif callable(prepare) and self.rank != 0:
            # Existing sealed stage preparation must be a read/validate reuse;
            # the factory may instead expose explicit read-only ``admit``.
            prepare(active_u, target_selection=target_selection) if target_selection is not None else prepare(active_u)
        if self.world > 1: self.dist.barrier()


def _validate_model(net, debug):
    groups = _groups(net)
    count = sum(parameter.numel() for parameter in net.parameters())
    trainable = sum(parameter.numel() for parameter in net.parameters() if parameter.requires_grad)
    if not debug and (count != EXPECTED_PARAMETERS or trainable != EXPECTED_PARAMETERS):
        raise ValueError('Production v2.3 requires all10,434,532 original trainable parameters')
    if next(net.parameters()).device.type != 'cuda':
        raise RuntimeError('Actual CUDA model is required')
    return groups


def calibrate_training_batches(net, scorer, population, config, *, patient_batches,
                               candidate_batches, budget, debug=False):
    """Measure full128-U forward/backward/update on cloned original parameters.

    Calibration cannot alter production weights or RNG. The graph includes all
    P and128U; choosing smaller patient batches cannot change graph contents.
    """
    _validate_model(net, debug)
    if (patient_batches != sorted(set(patient_batches)) or min(patient_batches) < 2
            or candidate_batches != sorted(set(candidate_batches)) or min(candidate_batches) < 1):
        raise ValueError('Explicit multiple-patient and candidate calibration settings required')
    training = config['training']
    train_ids = list(population.partition_cases('inner_train', ranking_only=True))
    if max(patient_batches) > len(train_ids):
        raise ValueError('Calibration may not duplicate training patients')
    canonical = getattr(scorer.providers['inner_train'], '_canonical', None)
    def weight(case):
        plan = population.case(case, active_u_count=128)
        if canonical:
            return sum(canonical[identity]['sampled_two_view_edges'] for identity in plan.record_ids)
        return len(plan.record_ids)
    ordered = sorted(train_ids, key=weight, reverse=True)
    repeats = config['v23_runtime']['calibration_repeats']
    if type(repeats) is not int or repeats < 1:
        raise ValueError('Explicit measured calibration repetitions required')
    original_rng, original_hash = capture_rng(), digest(net.state_dict())
    previous_net, previous_chunk = scorer.net, scorer.physical_candidate_batch
    trials = []
    try:
        for physical in patient_batches:
            plans = _plans(population, ordered[:physical], 128)
            for chunk in candidate_batches:
                print(f'v2.3 calibration START | patients{physical} candidate_chunk{chunk} '
                      f'fullP+128U records{sum(len(plan.record_ids) for plan in plans)} repeats{repeats}', flush=True)
                clone = copy.deepcopy(net)
                clone.train(); scorer.net = clone; scorer.physical_candidate_batch = chunk
                optimizer = torch.optim.AdamW(clone.parameters(), lr=training['lr'],
                    weight_decay=training['weight_decay'], fused=training['fused_optimizer'])
                scaler = torch.amp.GradScaler('cuda', enabled=training['amp'])
                output = loss = probes = None
                try:
                    torch.cuda.reset_peak_memory_stats()
                    measurements, overflows = [], []
                    for repeat in range(repeats):
                        torch.cuda.synchronize(); started = time.perf_counter()
                        print(f'v2.3 calibration repeat START | patients{physical} '
                              f'candidate_chunk{chunk} repeat{repeat+1}/{repeats}', flush=True)
                        while True:
                            optimizer.zero_grad(set_to_none=True)
                            output = scorer(plans, epoch=1, training=True)
                            loss, _ = patient_balanced_objective(output.scores, [_positive_indices(plan) for plan in plans], output.consistency)
                            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
                            receipt = gradient_receipt(clone, _groups(clone))
                            if receipt['missing']:
                                raise RuntimeError('Calibration disconnected genuine original parameter gradients')
                            if receipt['finite']: break
                            before, after = retry_amp_overflow(scaler, optimizer)
                            overflows.append(dict(repeat=repeat, scale_before=before, scale_after=after))
                            print(f'v2.3 calibration AMP overflow | patients{physical} candidate_chunk{chunk} '
                                  f'repeat{repeat+1} no update scale{before:g}->{after:g}; same input retry', flush=True)
                            output = loss = None
                        torch.nn.utils.clip_grad_norm_(clone.parameters(), training['grad_clip'], error_if_nonfinite=True)
                        probes = _probe_before(_groups(clone))
                        scaler.step(optimizer); scaler.update(); torch.cuda.synchronize(); _check_budget(budget)
                        changed = _probe_after(probes)
                        measurements.append(dict(repeat=repeat, seconds=time.perf_counter()-started,
                            gradient=receipt, sampled_parameter_changes=changed))
                        print(f'v2.3 calibration repeat COMPLETE | patients{physical} candidate_chunk{chunk} '
                              f'repeat{repeat+1}/{repeats} seconds{measurements[-1]["seconds"]:.2f} '
                              f'peakVRAM{torch.cuda.max_memory_allocated()/2**30:.2f}GiB', flush=True)
                    seconds = math.fsum(row['seconds'] for row in measurements) / repeats
                    trials.append(dict(physical_patient_batch=physical, physical_candidate_batch=chunk,
                        accepted=True, seconds=seconds, patients_per_second=physical / seconds,
                        candidates_per_second=sum(output.counts) / seconds,
                        peak_cuda_bytes=torch.cuda.max_memory_allocated(), gradient=receipt,
                        case_ids=list(output.case_ids), workload=output.workload, all_P_and_U128=True,
                        measurements=measurements, AMP_overflows=overflows, optimizer_updates_on_clone=repeats))
                except torch.cuda.OutOfMemoryError as error:
                    print(f'v2.3 calibration rejected CUDA_OOM | patients{physical} candidate_chunk{chunk}', flush=True)
                    trials.append(dict(physical_patient_batch=physical, physical_candidate_batch=chunk,
                        accepted=False, error=str(error), failure='CUDA_OOM', graph_model_population_unchanged=True))
                finally:
                    output = loss = probes = None
                    scorer.net = previous_net
                    del clone, optimizer, scaler
                    torch.cuda.empty_cache()
                print(f'v2.3 calibration trial COMPLETE | patients{physical} candidate_chunk{chunk} '
                      f'accepted={trials[-1]["accepted"]}', flush=True)
        accepted = [trial for trial in trials if trial['accepted']]
        if not accepted:
            raise MemoryError('No explicitly measured original-model native128 training batch fits; no fallback')
        selected = max(accepted, key=lambda row: row['patients_per_second'])
    finally:
        scorer.net, scorer.physical_candidate_batch = previous_net, previous_chunk
        restore_rng(original_rng)
    if digest(net.state_dict()) != original_hash:
        raise RuntimeError('Calibration changed original production parameters')
    return dict(format=FORMAT, trials=trials,
        selected_physical_patient_batch=selected['physical_patient_batch'],
        selected_physical_candidate_batch=selected['physical_candidate_batch'],
        initial_state_sha256=original_hash, original_model_and_RNG_preserved=True,
        measured_full_P_U128_backward=True, debug=debug)


def run_training(net, scorer, population, config, *, output, physical_patient_batch,
                 workers, identity, budget, epochs=40, debug=False):
    """Run/resume the complete all-P experiment; never overwrite old experiments."""
    import psutil
    from tqdm import tqdm
    distributed = _Distributed()
    groups = _validate_model(net, debug)
    if not debug and epochs != 40:
        raise ValueError('Production v2.3 preserves forty epochs')
    if type(physical_patient_batch) is not int or physical_patient_batch < 2 or workers < 2:
        raise ValueError('Measured parallel patient batching and CPU workers required')
    runtime, training = copy.deepcopy(config['v23_runtime']), config['training']
    if training['consistency_weight'] != .1 or training['gradient_accumulation_steps'] != 1:
        raise ValueError('Original consistency0.1 and accumulation1 contract required')
    calibration = runtime['batch_calibration']
    initial_hash = digest(net.state_dict())
    if (calibration['selected_physical_patient_batch'] != physical_patient_batch
            or calibration['selected_physical_candidate_batch'] != scorer.physical_candidate_batch
            or not calibration['original_model_and_RNG_preserved']
            or calibration['initial_state_sha256'] != initial_hash
            or calibration.get('world_size') != distributed.world
            or not calibration.get('measured_full_P_U128_backward')
            or not any(row['accepted'] and row['physical_patient_batch'] == physical_patient_batch
                and row['physical_candidate_batch'] == scorer.physical_candidate_batch for row in calibration['trials'])):
        raise ValueError('Production execution requires its actual unchanged-weight full128 backward calibration')
    train_cases = list(population.partition_cases('inner_train', ranking_only=True))
    val_cases = list(population.partition_cases('inner_val', ranking_only=False))
    if not train_cases or not val_cases or set(train_cases) & set(val_cases):
        raise ValueError('Complete disjoint patient populations required')
    if any(not _positive_indices(population.case(case, active_u_count=128)) for case in train_cases):
        raise ValueError('Zero-P train patients must be explicitly listed as undefined rank-loss cases')
    if scorer.net is not net or scorer.amp != bool(training['amp']):
        raise ValueError('Scorer does not own this exact training model/precision')
    if distributed.world > 1:
        from torch.nn.parallel import DistributedDataParallel
        execution = DistributedDataParallel(scorer, device_ids=[torch.cuda.current_device()],
            output_device=torch.cuda.current_device(), broadcast_buffers=False,
            find_unused_parameters=False)
    else:
        execution = scorer
    if not debug and (runtime['initial_u'], runtime['increment_u'], runtime['total_u']) != (7, 7, 128):
        raise ValueError('Production cumulative candidate contract is initial7, add7, full128U')
    curriculum = new_curriculum(**{key: runtime[key] for key in (
        'initial_u', 'increment_u', 'total_u', 'gate_mrr', 'gate_top1',
        'minimum_stage_epochs', 'consecutive_passes')})
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=training['lr'], weight_decay=training['weight_decay'], fused=training['fused_optimizer'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40)
    scaler = torch.amp.GradScaler('cuda', enabled=training['amp'])
    generator = torch.Generator().manual_seed(42 + 2003)
    binding = training_binding(config, population, identity=identity,
        physical_patient_batch=physical_patient_batch, physical_candidate_batch=scorer.physical_candidate_batch,
        world_size=distributed.world, workers=workers, initial_hash=initial_hash, epochs=epochs, debug=debug)
    binding_hash = digest(binding)
    root = Path(output).resolve()
    distributed.root_action(lambda: root.mkdir(parents=True, exist_ok=True))
    ownership = root / 'training_identity.json'
    resume = runtime.get('resume_checkpoint')
    def admit_output():
        if ownership.exists():
            if json.loads(ownership.read_text(encoding='utf8'))['identity_sha256'] != binding_hash or not resume:
                raise FileExistsError('Existing results require explicit exact-identity resume; no overwrite')
        else:
            if any(root.iterdir()):
                raise FileExistsError('Existing experiment files preserved; choose a new training directory')
            _write_new(ownership, dict(identity_sha256=binding_hash, binding=cpu_copy(binding)))
    distributed.root_action(admit_output)
    state = dict(epoch=1, phase='initial_full_validation', train_position=0, train_order=None,
        updates=0, attempts=0, overflows=0, curriculum=curriculum, history=[], best=None,
        train_rows=[], evaluation_position=0, evaluation_rows=[], epoch_start_updates=0,
        connected=[], initial_validation=None, status='RUNNING')
    target_policy = runtime.get('target_selection_policy', 'native_prefix')
    if runtime.get('independent_single_gpu') and distributed.world != 1:
        raise ValueError('Independent v2.3 runs cannot join a data-parallel process group')
    state['targets'] = new_target_state(population, target_policy, curriculum['active_u'])
    if resume:
        saved = torch.load(resume, map_location='cpu', weights_only=False)
        checksum = saved.pop('content_sha256', None)
        if saved.get('format') != FORMAT or saved.get('identity_sha256') != binding_hash or checksum != digest(saved):
            raise ValueError('Checkpoint format, bound model/data contract, or content checksum differs')
        net.load_state_dict(saved['model'], strict=True); optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler']); scaler.load_state_dict(saved['scaler'])
        state = saved['state']; generator.set_state(saved['shuffle_generator'])
        if len(saved['rank_rng']) != distributed.world:
            raise ValueError('Resume requires the same measured data-parallel world size')
        restore_rng(saved['rank_rng'][distributed.rank])
        restore_optimizer_history(optimizer, state['updates'])
        _validate_progress(state, train_cases, val_cases, physical_patient_batch, epochs, scheduler, distributed.world)
        validate_target_state(population, state['targets'], policy=target_policy,
                              active_u=state['curriculum']['active_u'])
        del saved
    torch.set_num_threads(workers)
    pause_after = runtime.get('debug_pause_after_updates')
    if pause_after is not None and (not debug or type(pause_after) is not int or pause_after < 1):
        raise ValueError('Pause-after-update is a separately marked DEBUG control only')
    pause_file = Path(runtime.get('pause_file', root / 'STOP_AFTER_BATCH'))
    def paused():
        return pause_file.exists() or (pause_after is not None and state['updates'] >= pause_after)
    def save(status='RUNNING', best=False):
        state['status'] = status
        rank_rng = distributed.gather(cpu_copy(capture_rng()))
        def publish():
            payload = dict(format=FORMAT, identity_sha256=binding_hash,
                model=cpu_copy(net.state_dict()), optimizer=cpu_copy(optimizer.state_dict()),
                scheduler=cpu_copy(scheduler.state_dict()), scaler=cpu_copy(scaler.state_dict()),
                state=cpu_copy(state), rank_rng=rank_rng, shuffle_generator=generator.get_state().clone())
            payload['content_sha256'] = digest(payload)
            if best: atomic_save(root / 'checkpoint_best.pt', payload)
            atomic_save(root / 'checkpoint_latest.pt', payload)
        distributed.root_action(publish)
    contract = root / 'execution_contract.json'
    def publish_contract():
        if contract.exists(): return
        _write_new(contract, dict(binding=binding, model=type(net).__name__, parameters=sum(p.numel() for p in net.parameters()),
            trainable_parameters=sum(p.numel() for p in params), hidden_dim=128, heads=4,
            layers=dict(L0=3, L1=2, L2=2), input=[5,48,48,48], views=2, margin_mm=10,
            epochs=epochs, updates_per_epoch=len(parallel_patient_batches(train_cases, physical_patient_batch, distributed.world)),
            total_planned_updates=state['updates'] + remaining_training_updates(
                state, train_cases, physical_patient_batch, distributed.world, epochs),
            successful_updates_inherited=state['updates'], completed_epochs_inherited=len(state['history']),
            remaining_planned_updates=remaining_training_updates(
                state, train_cases, physical_patient_batch, distributed.world, epochs),
            remaining_target_epochs=epochs-len(state['history']),
            physical_patient_batch=physical_patient_batch, gradient_accumulation_steps=1,
            data_parallel_world_size=distributed.world,
            effective_patient_batch=physical_patient_batch*distributed.world, physical_candidate_batch=scorer.physical_candidate_batch,
            loss='mean_U softplus(U-P), mean_P within patient, mean_patients +0.1 patient-balanced original six-key two-view consistency',
            all_P_active_from_epoch1=True, other_P_in_negative_set=False, donor_fixed_per_patient=True,
            full_validation='all native P + fixed128U actual joint graph',
            stage_validation='all native P + current cumulative activeU actual joint graph',
            best_rule='full native per-P patient-MRR, then per-P top1, then negative patient-balanced pair loss',
            stage_gate='all-P-vs-U patient-MRR/top1; never selects BEST',
            target_selection_policy=target_policy, target_membership_frozen_between_gate_expansions=True,
            patient_order_uses_previous_completed_training_only=True,
            continued_checkpoint_state=bool(identity.get('checkpoint_continuation')),
            checkpoint_continuation=identity.get('checkpoint_continuation'),
            initial_low_full_validation_causes_early_stop=False,
            gpu=torch.cuda.get_device_name(), gpu_count=torch.cuda.device_count(),
            cpu_affinity_cores=len(psutil.Process().cpu_affinity()), RAM=psutil.virtual_memory()._asdict(),
            precision_AMP=training['amp'], checkpoint_local_chunks=scorer.checkpoint_local_chunks,
            worker_count=workers, model_training_completed=False, debug=debug))
    distributed.root_action(publish_contract)
    def publish_validation(path, report):
        if path.exists():
            stored = json.loads(path.read_text(encoding='utf8'))
            if digest(stored) != digest(report):
                raise ValueError('Owned previously published validation differs from completed checkpoint rows')
        else:
            _write_new(path, report)
    def finish_epoch(pending):
        stage, report = pending['stage'], pending['full']
        if pending['improved']: state['best'] = pending['candidate_best']
        curve = dict(epoch=state['epoch'], update=state['updates'], train=aggregate_patients(state['train_rows']),
            stage_validation=stage, full_validation=report, curriculum_transition=pending['transition'],
            target_selection_sha256=pending['epoch_target_selection_sha256'],
            next_stage_target_selection_sha256=digest(state['targets']['selection_manifest']),
            next_stage_active_u=pending['curriculum']['active_u'],
            target_selection_policy=target_policy)
        state['history'].append(curve)
        distributed.root_action(lambda: publish_epoch_curve(root / 'curve.jsonl', curve))
        scheduler.step(); state.update(epoch=state['epoch']+1, phase='training', train_position=0,
            train_order=None, train_rows=[], curriculum=pending['curriculum'])
        for key in ('pending_epoch_completion', 'target_mining_position', 'target_mining_rows', 'target_mining_model_sha256'):
            state.pop(key, None)
        if state['epoch'] > epochs: state['phase'] = 'complete'
        save(best=pending['improved'])
    save()
    started = time.perf_counter()
    try:
        while state['phase'] != 'complete':
            _check_budget(budget)
            if paused():
                save('PAUSED'); break
            if state['phase'] == 'target_mining':
                pending = state['pending_epoch_completion']
                net.eval(); distributed.prepare_geometry(scorer.geometry, 128)
                model_hash = digest(net.state_dict())
                if model_hash != state['target_mining_model_sha256']:
                    raise ValueError('Target mining resumed under changed model weights')
                mining_cases = list(state['targets']['selections'])
                cursor = 0
                # Providers cannot mix partitions in one forward. Every case
                # is still scored once using complete all-P+128U geometry.
                for partition_cases in (train_cases, val_cases):
                    for global_cases in parallel_patient_batches(partition_cases, physical_patient_batch, distributed.world):
                        start = cursor; cursor += len(global_cases)
                        if start < state['target_mining_position']: continue
                        if paused(): break
                        cases = rank_cases(global_cases, distributed.world, distributed.rank)
                        plans = _plans(population, cases, 128)
                        rng = capture_rng(); began = time.perf_counter()
                        try:
                            with torch.no_grad():
                                result = execution(plans, epoch=training['fixed_validation_epoch'], training=False)
                            rows = distributed.rows([dict(case_id=plan.case_id,
                                u_scores=bank_u_scores(score, plan), model_sha256=model_hash)
                                for score, plan in zip(result.scores, plans)])
                            if [row['case_id'] for row in rows] != global_cases:
                                raise ValueError('U-only mining lost complete patient ownership')
                            state['target_mining_rows'].extend(rows); state['target_mining_position'] = cursor
                            distributed.root_action(lambda: _append(root/'target_mining_timing.jsonl',
                                dict(epoch=state['epoch'], case_ids=global_cases, full_U=128, no_grad=True,
                                     U_scores_only=True, seconds=time.perf_counter()-began, workload=result.workload)))
                            del result
                        finally:
                            restore_rng(rng)
                        save()
                    if paused(): break
                if paused(): continue
                if [row['case_id'] for row in state['target_mining_rows']] != mining_cases:
                    raise ValueError('Target mining must cover all ranking train and validation patients')
                targets, target_receipt = advance_target_state(population, state['targets'],
                    pending['curriculum']['active_u'], epoch=state['epoch'], model_sha256=model_hash,
                    mining_rows=state['target_mining_rows'])
                distributed.root_action(lambda: publish_validation(
                    root/f'target_admission_epoch_{state["epoch"]:03d}.json', target_receipt))
                state['targets'] = targets
                finish_epoch(pending)
                continue
            if state['phase'] == 'training':
                net.train()
                selection = state['targets']['selections']
                distributed.prepare_geometry(scorer.geometry, state['curriculum']['active_u'], selection)
                if state['train_order'] is None:
                    permutation = torch.randperm(len(train_cases), generator=generator).tolist()
                    random_order = [train_cases[index] for index in permutation]
                    ordered = apply_patient_order(population, state, target_policy, random_order)
                    state['train_order'] = ordered['order']
                    state.setdefault('patient_order_receipts', []).append(ordered['receipt'])
                    state['epoch_start_updates'] = state['updates']; save()
                order = state['train_order']
                batches = parallel_patient_batches(order, physical_patient_batch, distributed.world)
                offsets = [sum(map(len, batches[:index])) for index in range(len(batches))]
                for start, global_cases in zip(offsets, batches):
                    if start < state['train_position']: continue
                    if paused(): break
                    cases = rank_cases(global_cases, distributed.world, distributed.rank)
                    plans = _plans(population, cases, state['curriculum']['active_u'], selection)
                    began = time.perf_counter(); torch.cuda.reset_peak_memory_stats()
                    while True:
                        optimizer.zero_grad(set_to_none=True); state['attempts'] += 1
                        result = execution(plans, epoch=state['epoch'], training=True)
                        loss, terms = patient_balanced_objective(result.scores, [_positive_indices(plan) for plan in plans], result.consistency)
                        # DDP averages gradients over ranks. Unequal real tail
                        # sizes require weighting local patient means so the
                        # resulting gradient is the exact global patient mean.
                        loss = loss * (len(cases) * distributed.world / len(global_cases))
                        if not bool(torch.isfinite(loss)):
                            raise FloatingPointError('Nonfinite actual patient-balanced all-P loss')
                        scaler.scale(loss).backward(); scaler.unscale_(optimizer)
                        gradients = gradient_receipt(net, groups)
                        if gradients['finite']: break
                        before, after = retry_amp_overflow(scaler, optimizer); state['overflows'] += 1
                        del result, loss, terms
                        save(); distributed.root_action(lambda: _append(root / 'update_timing.jsonl', dict(status='AMP_OVERFLOW_RETRY_SAME_PATIENTS',
                            epoch=state['epoch'], case_ids=cases, cursor=start, updates=state['updates'],
                            scale_before=before, scale_after=after, gradient=gradients)))
                        if paused(): break
                    if paused() and not gradients['finite']: break
                    if gradients['missing']:
                        raise RuntimeError('Disconnected original trainable parameters: ' + repr(gradients['missing']))
                    clipped = torch.nn.utils.clip_grad_norm_(params, training['grad_clip'], error_if_nonfinite=True)
                    probes = _probe_before(groups); scaler.step(optimizer); scaler.update(); torch.cuda.synchronize()
                    changed = _probe_after(probes)
                    rows = distributed.rows([score_patient(score, plan) for score, plan in zip(result.scores, plans)])
                    if [row['case_id'] for row in rows] != global_cases:
                        raise ValueError('Data-parallel patient ownership lost ordered complete coverage')
                    state['train_rows'].extend(rows); state['train_position'] += len(global_cases); state['updates'] += 1
                    state['connected'] = sorted(set(state['connected']) | {name for name, p in net.named_parameters() if p.requires_grad and p.grad is not None})
                    seconds = time.perf_counter() - began
                    row = dict(status='OPTIMIZER_UPDATED', epoch=state['epoch'], update=state['updates'],
                        active_u=state['curriculum']['active_u'], case_ids=cases,
                        record_ids=list(result.record_ids), observed_P=sum(len(plan.positive_indices) for plan in plans),
                        physical_patients=len(plans), **result.workload, gradient=gradients,
                        sampled_parameter_changes=changed, loss=float(loss.detach()),
                        ranking=float(terms['ranking'].detach()), consistency=float(terms['consistency'].detach()),
                        gradient_before_clip=float(clipped), seconds=seconds, patients_per_second=len(plans)/seconds,
                        candidates_per_second=sum(result.counts)/seconds, peak_cuda_bytes=torch.cuda.max_memory_allocated(),
                        peak_reserved_bytes=torch.cuda.max_memory_reserved(), rss_bytes=psutil.Process().memory_info().rss,
                        cpu_percent=psutil.Process().cpu_percent(), learning_rate=optimizer.param_groups[0]['lr'])
                    rank_rows = distributed.gather(dict(rank=distributed.rank, **row))
                    distributed.root_action(lambda: _append(root / 'update_timing.jsonl', dict(
                        epoch=state['epoch'], update=state['updates'], global_patient_count=len(global_cases),
                        global_case_ids=global_cases, rank_reports=rank_rows)))
                    if distributed.rank == 0: tqdm.write(f'v2.3 epoch{state["epoch"]}/{epochs} U{state["curriculum"]["active_u"]} '
                        f'patients{state["train_position"]}/{len(train_cases)} allP{row["observed_P"]} '
                        f'loss={row["loss"]:.5f} step={seconds:.2f}s VRAM={row["peak_cuda_bytes"]/2**30:.2f}GiB')
                    del result, loss, terms, probes; save()
                if paused(): continue
                if state['train_position'] != len(order) or {row['case_id'] for row in state['train_rows']} != set(train_cases):
                    raise ValueError('Incomplete all-P patient epoch; candidate expansion is forbidden')
                state['phase'] = 'stage_validation'; save(); continue
            phase = state['phase']
            if phase not in ('initial_full_validation', 'stage_validation', 'full_validation'):
                raise ValueError('Invalid checkpointed training phase')
            net.eval(); active = state['curriculum']['active_u'] if phase == 'stage_validation' else 128
            selection = state['targets']['selections'] if phase == 'stage_validation' else None
            distributed.prepare_geometry(scorer.geometry, active, selection)
            label_epoch = 0 if phase == 'initial_full_validation' else state['epoch']
            batches = parallel_patient_batches(val_cases, physical_patient_batch, distributed.world)
            offsets = [sum(map(len, batches[:index])) for index in range(len(batches))]
            for start, global_cases in zip(offsets, batches):
                if start < state['evaluation_position']: continue
                if paused(): break
                cases = rank_cases(global_cases, distributed.world, distributed.rank)
                plans = _plans(population, cases, active, selection)
                rng = capture_rng(); began = time.perf_counter()
                try:
                    with torch.no_grad():
                        result = execution(plans, epoch=training['fixed_validation_epoch'], training=False)
                    rows = distributed.rows([score_patient(score, plan) for score, plan in zip(result.scores, plans)])
                    if [row['case_id'] for row in rows] != global_cases:
                        raise ValueError('Data-parallel validation ownership lost complete patient coverage')
                    torch.cuda.synchronize()
                    state['evaluation_rows'].extend(rows); state['evaluation_position'] += len(global_cases)
                    timing = dict(epoch=label_epoch, phase=phase, active_u=active,
                        case_ids=cases, workload=result.workload, seconds=time.perf_counter()-began,
                        peak_cuda_bytes=torch.cuda.max_memory_allocated(), rss_bytes=psutil.Process().memory_info().rss)
                    rank_times = distributed.gather(dict(rank=distributed.rank, **timing))
                    distributed.root_action(lambda: _append(root / 'validation_timing.jsonl', dict(
                        epoch=label_epoch, phase=phase, active_u=active, global_case_ids=global_cases, rank_reports=rank_times)))
                    del result
                finally:
                    restore_rng(rng)
                save()
            if paused(): continue
            if [row['case_id'] for row in state['evaluation_rows']] != val_cases:
                raise ValueError('Dual validation must score every configured validation patient exactly once')
            report = aggregate_patients(state['evaluation_rows'])
            report.update(epoch=label_epoch, updates=state['updates'], active_u=active,
                phase=phase, all_P_scored=True, actual_joint_upper=True, debug=debug,
                per_P_tie_policy='pessimistic: rank1 + number of U scores >= this P; other P excluded',
                target_selection_policy=target_policy,
                actual_target_membership_sha256=digest(state['targets']['selection_manifest']) if phase == 'stage_validation' else None,
                legacy_first_P_tie_policy='unchanged geometry_sha256 native candidate_order')
            distributed.root_action(lambda: publish_validation(root / f'{phase}_epoch_{label_epoch:03d}.json', report))
            state['evaluation_position'] = 0; state['evaluation_rows'] = []
            if phase == 'initial_full_validation':
                state['initial_validation'] = report; state['phase'] = 'training'; save()
            elif phase == 'stage_validation':
                state['stage_validation'] = report; state['phase'] = 'full_validation'; save()
            else:
                stage = state.pop('stage_validation')
                curriculum, transition = finish_curriculum_epoch(state['curriculum'], stage,
                    epoch=state['epoch'], successful_train_cases=[row['case_id'] for row in state['train_rows']],
                    expected_train_cases=train_cases, updates_in_epoch=state['updates']-state['epoch_start_updates'])
                key = best_selection_key(report)
                improved = state['best'] is None or key > tuple(state['best']['selection_key'])
                pending = dict(stage=stage, full=report, curriculum=curriculum,
                    transition=transition, improved=improved,
                    candidate_best=dict(epoch=state['epoch'], updates=state['updates'], selection_key=list(key),
                        candidate_universe='all native P + fixed128U', selected_by='full_validation_only'),
                    epoch_target_selection_sha256=digest(state['targets']['selection_manifest']))
                if transition['expanded'] and target_policy != 'native_prefix':
                    state.update(phase='target_mining', pending_epoch_completion=pending,
                        target_mining_position=0, target_mining_rows=[],
                        target_mining_model_sha256=digest(net.state_dict()))
                    save(); continue
                if transition['expanded']:
                    state['targets'], target_receipt = advance_target_state(population, state['targets'],
                        curriculum['active_u'], epoch=state['epoch'], model_sha256=digest(net.state_dict()))
                    distributed.root_action(lambda: publish_validation(
                        root/f'target_admission_epoch_{state["epoch"]:03d}.json', target_receipt))
                finish_epoch(pending)
            metrics = report['metrics']
            if distributed.rank == 0: tqdm.write(f'v2.3 {phase} epoch{label_epoch} U{active} | perP-MRR={metrics["per_P_patient_mrr"]:.6f} '
                f'perP-top1={metrics["per_P_patient_top1"]:.6f} firstP-MRR={metrics["case_first_P_mrr"]:.6f} '
                f'Hit1={metrics["case_hit_at_1"]:.6f}')
    except Exception as error:
        # Each rank writes only its own failure file without collective waits;
        # another rank may already be failing inside CUDA/NCCL.
        _append(root / f'failures_rank_{distributed.rank}.jsonl', dict(epoch=state['epoch'], phase=state['phase'],
            updates=state['updates'], error=f'{type(error).__name__}: {error}',
            recovery='Last owned successful checkpoint preserved; exact resume retries unchanged patient cursor'))
        raise
    status = 'DEBUG_COMPLETE' if debug and state['phase'] == 'complete' else 'COMPLETE' if state['phase'] == 'complete' else 'PAUSED'
    save(status)
    receipt = dict(format=FORMAT, status=status, debug=debug, actual_CUDA=True, full_training=state['phase']=='complete' and not debug,
        completed_epochs=len(state['history']), optimizer_updates=state['updates'], backward_attempts=state['attempts'],
        AMP_overflows=state['overflows'], best=state['best'], active_u=state['curriculum']['active_u'],
        identity_sha256=binding_hash, model_sha256=digest(net.state_dict()),
        optimizer_sha256=digest(optimizer.state_dict()), connected_parameter_tensors=len(state['connected']),
        expected_parameter_tensors=len([p for p in params]), invocation_seconds=time.perf_counter()-started,
        checkpoint=str(root/'checkpoint_latest.pt'), quality_verified=False, independent_test_evaluated=False)
    distributed.root_action(lambda: _append(root/'invocations.jsonl', receipt))
    return receipt


def _validate_progress(state, train_cases, val_cases, physical, epochs, scheduler, world_size=1):
    if state['phase'] not in ('initial_full_validation', 'training', 'stage_validation', 'full_validation', 'target_mining', 'complete'):
        raise ValueError('Unknown resumed v2.3 phase')
    if state['epoch'] != len(state['history']) + 1 or not 1 <= state['epoch'] <= epochs+1:
        raise ValueError('Saved epoch/history progress differs')
    if scheduler.last_epoch != len(state['history']):
        raise ValueError('Saved cosine scheduler differs from completed epochs')
    order = state['train_order']
    position = state['train_position']
    if order is not None and (len(order) != len(train_cases) or set(order) != set(train_cases)):
        raise ValueError('Resumed train permutation would lose or repeat patients')
    if type(position) is not int or not 0 <= position <= len(train_cases):
        raise ValueError('Invalid resumed patient cursor')
    train_boundaries = {0}
    if order is not None:
        cursor = 0
        for batch in parallel_patient_batches(order, physical, world_size):
            cursor += len(batch); train_boundaries.add(cursor)
    if position not in train_boundaries:
        raise ValueError('Resumed cursor splits a physical patient batch')
    expected = [] if order is None else order[:position]
    if [row['case_id'] for row in state['train_rows']] != expected:
        raise ValueError('Resumed successful patient coverage differs from cursor')
    vpos = state['evaluation_position']
    val_boundaries = {0}; cursor = 0
    for batch in parallel_patient_batches(val_cases, physical, world_size):
        cursor += len(batch); val_boundaries.add(cursor)
    if type(vpos) is not int or vpos not in val_boundaries:
        raise ValueError('Invalid resumed validation cursor')
    if [row['case_id'] for row in state['evaluation_rows']] != val_cases[:vpos]:
        raise ValueError('Resumed validation coverage differs from cursor')
    if state['curriculum']['last_completed_epoch'] != len(state['history']):
        raise ValueError('Resumed curriculum advancement differs from actual complete epochs')
    if state['phase'] == 'target_mining':
        pending = state['pending_epoch_completion']
        if (not pending['transition']['expanded']
                or pending['curriculum']['active_u'] <= state['curriculum']['active_u']
                or pending['stage']['epoch'] != state['epoch'] or pending['full']['epoch'] != state['epoch']):
            raise ValueError('Resumed target mining must belong to the completed real gate')
        mining_cases = train_cases + val_cases
        boundaries = {0}; cursor = 0
        for partition in (train_cases, val_cases):
            for batch in parallel_patient_batches(partition, physical, world_size):
                cursor += len(batch); boundaries.add(cursor)
        position = state['target_mining_position']
        if position not in boundaries or [row['case_id'] for row in state['target_mining_rows']] != mining_cases[:position]:
            raise ValueError('Resumed mining cursor/coverage splits a real patient batch')
