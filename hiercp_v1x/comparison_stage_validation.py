"""Resumable held-out evaluation using the actual stage candidate graph.

This is additional evaluation; it never selects the global best, steps an
optimizer/scheduler, or substitutes a subset of full-bank scores.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import time

import psutil
import torch

from .comparison_curriculum_data import candidate_plan
from .comparison_progress import PhaseProgress, RunningPatientMetrics
from .u_bridge_training import (aggregate_rows, capture_rng, restore_rng,
    score_row, digest, _write_new, _append, _check_budget)


def _validate_full_validation(state, val_ids):
    if (state.get('validation_position') != len(val_ids) or
            [row['sample_index'] for row in state.get('validation_rows', [])] != list(val_ids)):
        raise ValueError('Stage evaluation requires the completed full129 validation cursor and rows')


def validate_stage_cursor(state, val_ids, physical_batch, keys, *, policy=None, view_epoch=None):
    saved = state.get('stage_validation')
    if saved is None:
        return
    epoch = 0 if state['phase'] == 'initial_validation' else state['epoch']
    if state['phase'] not in ('initial_validation', 'validation'):
        raise ValueError('Unfinished stage evaluation outside validation phase')
    _validate_full_validation(state, val_ids)
    if saved['epoch'] != epoch or saved['update'] != state['updates']:
        raise ValueError('Stage evaluation belongs to different model update/epoch')
    if saved['keys'] != list(keys):
        raise ValueError('Stage candidate pool changed during partial validation')
    position = saved['position']
    if type(position) is not int or not 0 <= position <= len(val_ids):
        raise ValueError('Invalid stage validation cursor')
    if position != len(val_ids) and position % physical_batch:
        raise ValueError('Stage cursor splits a physical source batch')
    if [row['sample_index'] for row in saved['rows']] != list(val_ids[:position]):
        raise ValueError('Stage validation would repeat or skip held-out sources')
    if any(row['candidate_keys'] != list(keys) for row in saved['rows']):
        raise ValueError('Stage rows use another candidate pool')
    if policy is not None and saved['policy_sha256'] != policy['policy_sha256']:
        raise ValueError('Stage evaluation policy/view identity differs')
    if view_epoch is not None and saved['view_epoch'] != view_epoch:
        raise ValueError('Stage evaluation policy/view identity differs')


def evaluate_stage_validation(*, net, provider, state, arm, keys, policy, root,
        val_ids, lookup, physical_batch, epochs, view_epoch, chunk, amp, budget,
        checkpoint, pause_requested, debug):
    # Import locally: runtime also imports this module into a frozen namespace.
    from .comparison_runtime import _prefetch, closing
    root = Path(root)
    label_epoch = 0 if state['phase'] == 'initial_validation' else state['epoch']
    keys = list(keys)
    _validate_full_validation(state, val_ids)
    if 'stage_validation' not in state:
        state['stage_validation'] = dict(epoch=label_epoch, update=state['updates'],
            keys=keys, policy_sha256=policy['policy_sha256'], view_epoch=view_epoch,
            position=0, rows=[], active_seconds=0.)
        checkpoint()
    validate_stage_cursor(state, val_ids, physical_batch, keys, policy=policy, view_epoch=view_epoch)
    saved = state['stage_validation']
    count = len(keys)
    remaining = [val_ids[i:i + physical_batch]
                 for i in range(saved['position'], len(val_ids), physical_batch)]
    running = RunningPatientMetrics(saved['rows'])
    net.eval()
    with PhaseProgress(root=root, arm=arm, phase=f'validation_stage{count}',
            epoch=label_epoch, epochs=epochs, total=len(val_ids),
            initial=saved['position'], physical_batch=physical_batch,
            metrics=running.metrics()) as progress:
        with candidate_plan(provider, arm, policy['policy_sha256'], keys,
                mode='stage_validation', epoch=view_epoch) as stage_provider, closing(
                _prefetch(stage_provider, remaining, arm, view_epoch, training=False,
                    full=False, pin_memory=False, with_timing=True,
                    receipt_path=root / 'stage_prefetch.jsonl')) as staged:
            for ids in remaining:
                if pause_requested():
                    return None
                before_rng = capture_rng()
                began = time.perf_counter()
                progress.update(stage='load')
                try:
                    batch, input_timing = next(staged)
                    _check_budget(budget)
                    if list(batch.counts) != [count] * len(ids) or batch.local_batch_view2 is None:
                        raise ValueError('Stage evaluation requires the complete actual subset graph and two views')
                    if tuple(batch.bridge_indices) != tuple(ids):
                        raise ValueError('Stage validation source binding changed')
                    progress.update(stage='forward', timings=input_timing)
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()
                    forward = time.perf_counter()
                    with torch.no_grad(), torch.autocast('cuda', enabled=amp):
                        scores = net.score_inference_chunked(batch, local_chunk_size=chunk)
                    torch.cuda.synchronize()
                    forward_seconds = time.perf_counter() - forward
                    if len(scores) != len(ids):
                        raise ValueError('Stage joint upper lost source mapping')
                    rows = [score_row(score, lookup[index], stage_provider.candidate_keys(
                        index, arm, view_epoch, full=False), expected_candidates=count)
                        for index, score in zip(ids, scores)]
                    saved['rows'].extend(rows)
                    saved['position'] += len(ids)
                    running.add(rows)
                    _check_budget(budget)
                    timing = dict(epoch=label_epoch, update=state['updates'],
                        source_indices=ids, candidate_keys=keys, candidate_rows=count * len(ids),
                        policy_sha256=policy['policy_sha256'], local_chunk_size=chunk,
                        **input_timing, full_joint_forward_seconds=forward_seconds,
                        peak_cuda_bytes=torch.cuda.max_memory_allocated(),
                        rss_bytes=psutil.Process().memory_info().rss)
                    del scores, batch
                finally:
                    restore_rng(before_rng)
                saved['active_seconds'] += time.perf_counter() - began
                progress.update(stage='save', completed=saved['position'], metrics=running.metrics())
                checkpoint_seconds = checkpoint()
                saved['active_seconds'] += checkpoint_seconds
                elapsed = time.perf_counter() - began
                timing.update(checkpoint_seconds=checkpoint_seconds, batch_wall_seconds=elapsed,
                              samples_per_second_including_wait=len(ids) / elapsed)
                _append(root / 'stage_validation_timing.jsonl', timing)
                progress.update(timings=dict(forward_seconds=forward_seconds,
                                            checkpoint_seconds=checkpoint_seconds))
        validate_stage_cursor(state, val_ids, physical_batch, keys)
        if saved['position'] != len(val_ids):
            raise ValueError('Stage validation did not cover every held-out source')
        report = aggregate_rows(saved['rows'])
        report.update(epoch=label_epoch, update=state['updates'], rows=copy.deepcopy(saved['rows']),
            candidate_keys=keys, candidates_per_problem=count, policy_sha256=policy['policy_sha256'],
            fixed_view_epoch=view_epoch, wall_seconds=saved['active_seconds'],
            wall_scope='active stage validation across resumed invocation segments', debug=debug,
            full_held_out_source_evaluation=True, full_candidate_evaluation=count == 129,
            actual_subset_graph_forward=True, joint_upper_once_per_source_problem=True,
            used_for_global_best=False)
        path = root / f'validation_stage_epoch_{label_epoch:03d}.json'
        if not path.exists():
            _write_new(path, report)
        else:
            stored = json.loads(path.read_text(encoding='utf8'))
            # The last checkpoint excludes its own completed save duration;
            # only timing may differ after restart. All metric/graph/view/policy
            # fields must match, so an obsolete report is never accepted.
            immutable = lambda value: {k: v for k, v in value.items() if k != 'wall_seconds'}
            if digest(immutable(stored)) != digest(immutable(report)):
                raise ValueError('Existing stage validation disagrees with resumed result')
        progress.finish()
        PhaseProgress.write(f'{arm} stage{count} epoch {label_epoch} COMPLETE '
            f'{len(val_ids)}/{len(val_ids)} sources | patient macro '
            f'MRR={report["metrics"]["mrr"]:.6f} top1={report["metrics"]["top1"]:.6f}')
        return report
