"""Print the complete saved reference-L1 diagnosis without rerunning a model.

This module deliberately uses only Python's standard library. It formats stored
measurements; unavailable measurements stay unavailable and are never scored.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


FORMAT = 'local_cnn_reference_comparison_debug_v1'
UNAVAILABLE = 'unavailable'


def _finite_json(value: Any, path: str = 'report') -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f'{path}: non-finite number')
    if isinstance(value, dict):
        for key, item in value.items():
            _finite_json(item, f'{path}.{key}')
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _finite_json(item, f'{path}[{index}]')


def _object(value: Any, path: str) -> dict:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f'{path}: expected object')
    return value


def _list(value: Any, path: str) -> list:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f'{path}: expected list')
    return value


def _show(value: Any) -> str:
    if value is None:
        return UNAVAILABLE
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, allow_nan=False)
    return str(value)


def _number(value: Any, path: str) -> str:
    if value is None:
        return UNAVAILABLE
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{path}: expected finite number or null')
    if not math.isfinite(value):
        raise ValueError(f'{path}: non-finite number')
    return str(value)


def _boolean(value: Any, path: str) -> str:
    if value is None:
        return UNAVAILABLE
    if not isinstance(value, bool):
        raise ValueError(f'{path}: expected boolean or null')
    return str(value)


def _status(obj: dict) -> str | None:
    value = obj.get('status')
    if value is not None and not isinstance(value, str):
        raise ValueError('status: expected string or null')
    if value and value != 'MEASURED':
        return value + (' | ' + _show(obj.get('reason')) if obj.get('reason') else '')
    return None


def _metric_line(obj: dict, path: str, pair_source: dict | None = None) -> str:
    metrics = _object(obj.get('metrics'), f'{path}.metrics')
    pair_source = obj if pair_source is None else pair_source
    fields = [('MRR', metrics.get('ranking_mrr')),
              ('pair-win', pair_source.get('pair_win_rate')),
              ('rank_loss', metrics.get('ranking_pairwise_loss')),
              ('R@1', metrics.get('ranking_recall_at_1')),
              ('R@5', metrics.get('ranking_recall_at_5')),
              ('R@10', metrics.get('ranking_recall_at_10'))]
    return ' | '.join(f'{label}={_number(value, path + "." + label)}' for label, value in fields)


def _evaluation(lines: list[str], value: Any, label: str, path: str) -> None:
    evaluation = _object(value, path)
    if not evaluation:
        lines.append(f'{label}: {UNAVAILABLE}')
        return
    stopped = _status(evaluation)
    if stopped:
        lines.append(f'{label}: {stopped}')
        return
    lines.append(f'{label} | scope={_show(evaluation.get("scope"))} | support={_show(evaluation.get("support_scope"))}')
    for split in ('train', 'validation'):
        group = _object(evaluation.get(split), path + '.' + split)
        if not group:
            lines.append(f'  {split}: {UNAVAILABLE}')
            continue
        stopped = _status(group)
        if stopped:
            lines.append(f'  {split}: {stopped}')
            continue
        lines.append(f'  {split} aggregate | {_metric_line(group, path + "." + split)}')
        cases = _list(group.get('cases'), path + '.' + split + '.cases')
        if 'cases' not in group or group.get('cases') is None:
            lines.append(f'    cases: {UNAVAILABLE}')
        elif not cases:
            lines.append('    cases: [] (stored empty list)')
        for index, item in enumerate(cases):
            case_path = f'{path}.{split}.cases[{index}]'
            case = _object(item, case_path)
            stopped = _status(case)
            if stopped:
                lines.append(f'    case={_show(case.get("case_id"))} | status={stopped}')
                continue
            score = _object(case.get('score'), case_path + '.score')
            metrics = _object(case.get('metrics'), case_path + '.metrics')
            ranks = case.get('observed_ranks')
            if ranks is not None:
                _list(ranks, case_path + '.observed_ranks')
                for rank in ranks:
                    _number(rank, case_path + '.observed_ranks')
            score_status = _status(score)
            lines.append(f'    case={_show(case.get("case_id"))} | records={_number(case.get("records"), case_path + ".records")}'
                         + f' | retained={_boolean(case.get("all_case_candidates_retained"), case_path + ".all_case_candidates_retained")}'
                         + ' | MRR=' + _number(metrics.get('ranking_mrr'), case_path + '.MRR')
                         + ' | pair-win=' + _number(score.get('pair_win_rate'), case_path + '.pair_win_rate')
                         + ' | rank_loss=' + _number(metrics.get('ranking_pairwise_loss'), case_path + '.rank_loss')
                         + ' | R@1=' + _number(metrics.get('ranking_recall_at_1'), case_path + '.R@1')
                         + ' | R@5=' + _number(metrics.get('ranking_recall_at_5'), case_path + '.R@5')
                         + ' | R@10=' + _number(metrics.get('ranking_recall_at_10'), case_path + '.R@10')
                         + ' | observed_ranks=' + _show(ranks)
                         + (' | score_status=' + score_status if score_status else ''))


def _fixed_branch(lines: list[str], branch: dict, path: str) -> None:
    modes = _list(branch.get('modes'), path + '.modes')
    if not modes:
        lines.append(f'  modes: {UNAVAILABLE}')
    for index, mode_value in enumerate(modes):
        mode_path = f'{path}.modes[{index}]'
        mode = _object(mode_value, mode_path)
        stopped = _status(mode)
        if stopped:
            lines.append('  mode=' + _show(mode.get('mode')) + ' | ' + stopped)
            continue
        l0 = _object(mode.get('L0'), mode_path + '.L0')
        signals = ['L0=' + _number(l0.get('normalized_centered_energy'), mode_path + '.L0.normalized_centered_energy')]
        layers = _list(mode.get('layers'), mode_path + '.layers')
        if not layers:
            signals.append('L1 layers=' + UNAVAILABLE)
        for layer_index, layer_value in enumerate(layers):
            layer_path = f'{mode_path}.layers[{layer_index}]'
            layer = _object(layer_value, layer_path)
            stages = _object(layer.get('stages'), layer_path + '.stages')
            output = _object(stages.get('output'), layer_path + '.stages.output')
            signals.append('L1_' + _number(layer.get('layer'), layer_path + '.layer')
                           + '=' + _number(output.get('normalized_centered_energy'), layer_path + '.normalized_centered_energy')
                           + '(ratio=' + _number(layer.get('normalized_centered_energy_output_over_input'), layer_path + '.normalized_ratio') + ')')
        score = _object(mode.get('score'), mode_path + '.score')
        lines.append('  mode=' + _show(mode.get('mode')) + ' | normalized_energy ' + ' -> '.join(signals)
                     + ' | tile_status=' + (_status(score) or _show(score.get('status')))
                     + ' | pair-win=' + _number(score.get('pair_win_rate'), mode_path + '.score.pair_win_rate')
                     + ' | observed=' + _number(score.get('observed'), mode_path + '.score.observed')
                     + ' | candidates=' + _number(score.get('candidates'), mode_path + '.score.candidates'))
    gradients = _object(branch.get('per_loss_query_gradient'), path + '.per_loss_query_gradient')
    stopped = _status(gradients)
    if stopped:
        lines.append('  query gradients | ' + stopped)
    else:
        losses = _object(gradients.get('losses'), path + '.per_loss_query_gradient.losses')
        gradient_parts = []
        for name in ('ranking', 'observation_ce', 'alignment'):
            loss = _object(losses.get(name), path + '.losses.' + name)
            loss_stopped = _status(loss)
            if loss_stopped:
                gradient_parts.append(name + ':' + loss_stopped)
                continue
            dependency = loss.get('query_embedding_dependency')
            norm = _number(loss.get('query_embedding_gradient_norm'), path + '.' + name + '.norm')
            if dependency is False and loss.get('query_embedding_gradient_norm') is None:
                norm = 'none (no query dependency)'
            gradient_parts.append(name + ':dep=' + _boolean(dependency, path + '.' + name + '.dependency')
                                  + ',norm=' + norm
                                  + ',loss=' + _number(loss.get('weighted_loss'), path + '.' + name + '.loss'))
        lines.append('  query gradients | ' + ' | '.join(gradient_parts))
    lines.append('  gradient_scope=' + _show(gradients.get('scope')))
    _evaluation(lines, branch.get('initial_full_case_evaluation'), '  initial_full_case_evaluation (separate from physical tile)', path + '.initial_full_case_evaluation')


def format_summary(report: dict) -> str:
    """Format every stored branch and selected case without model computation."""
    if not isinstance(report, dict):
        raise ValueError('report: expected JSON object')
    _finite_json(report)
    if report.get('format') != FORMAT:
        raise ValueError('Unsupported reference comparison report format')
    comparison = _object(report.get('comparison'), 'comparison')
    if comparison.get('diagnostic_only') is not True:
        raise ValueError('Expected an explicitly diagnostic-only comparison')
    snapshot = _object(report.get('snapshot'), 'snapshot')
    binding = _object(comparison.get('native_tile_binding'), 'native_tile_binding')
    lines = ['SAVED REFERENCE L1 DIAGNOSTIC SUMMARY | No model execution, optimizer update or metric recomputation.',
             'Scope: normalized-energy spread and exact tile scores are diagnostics; full-case selected-case rankings are separate; not full accuracy or CP efficacy.',
             'run=' + _show(report.get('run')) + ' | checkpoint=' + _show(report.get('checkpoint')),
             'snapshot | epoch=' + _number(snapshot.get('epoch'), 'snapshot.epoch')
             + ' | step=' + _number(snapshot.get('step'), 'snapshot.step')
             + ' | phase=' + _show(snapshot.get('phase'))
             + ' | tile saved_next=' + _boolean(binding.get('saved_next_optimization_tile'), 'binding.saved_next')
             + ' | schedule_position=' + _number(binding.get('schedule_position'), 'binding.schedule_position')
             + ' | actual_batch=' + _number(binding.get('actual_batch', comparison.get('physical_batch') if comparison.get('fixed_weight') is True else None), 'binding.actual_batch')
             + ' | configured_batch=' + _number(binding.get('original_physical_batch', comparison.get('configured_physical_batch')), 'binding.configured_batch')
             + ' | P=' + _number(binding.get('positive_count'), 'binding.positive_count')
             + ' | U=' + _number(binding.get('unobserved_count'), 'binding.unobserved_count'),
             'tile selection=' + _show(binding.get('selection'))]
    branches = _list(comparison.get('branches'), 'comparison.branches')
    if not branches:
        lines.append('branches: ' + UNAVAILABLE)
    for index, branch_value in enumerate(branches):
        path = f'comparison.branches[{index}]'
        branch = _object(branch_value, path)
        lines.extend(['', 'BRANCH ' + _show(branch.get('branch'))])
        stopped = _status(branch)
        if stopped:
            lines.append('  status=' + stopped)
            continue
        if comparison.get('fixed_weight') is True or 'modes' in branch:
            _fixed_branch(lines, branch, path)
        else:
            lines.append('  optimizer=' + _show(branch.get('optimizer')))
            updates = _list(branch.get('updates'), path + '.updates')
            count = UNAVAILABLE if branch.get('updates') is None else str(len(updates))
            lines.append('  stored_updates=' + count + ' | requested_steps=' + _number(comparison.get('steps_per_branch'), 'comparison.steps_per_branch'))
            _evaluation(lines, branch.get('before'), '  before', path + '.before')
            _evaluation(lines, branch.get('after'), '  after', path + '.after')
    lines.extend(['', 'Stored flags | preserved=' + _boolean(report.get('original_model_and_memory_preserved'), 'report.original_model_and_memory_preserved')
                  + ' | checkpoint_written=' + _boolean(report.get('production_checkpoint_written'), 'report.production_checkpoint_written')
                  + ' | training_started=' + _boolean(report.get('production_training_started'), 'report.production_training_started')
                  + ' | ready=' + _boolean(report.get('production_ready'), 'report.production_ready')
                  + ' | full_training=' + _boolean(report.get('full_training'), 'report.full_training')
                  + ' | full_evaluation=' + _boolean(report.get('full_evaluation'), 'report.full_evaluation')])
    limitations = _list(comparison.get('limitations'), 'comparison.limitations')
    if limitations:
        lines.append('Limitations: ' + ' | '.join(_show(item) for item in limitations))
    return '\n'.join(lines) + '\n'


def _case_signals(lines: list[str], value: Any, label: str, path: str) -> None:
    evaluation = _object(value, path)
    if not evaluation:
        lines.append(f'  {label}: {UNAVAILABLE}')
        return
    stopped = _status(evaluation)
    if stopped:
        lines.append(f'  {label}: {stopped}')
        return
    for split in ('train', 'validation'):
        group_path = path + '.' + split
        group = _object(evaluation.get(split), group_path)
        stopped = _status(group)
        if not group or stopped:
            lines.append(f'  {label} {split}: {stopped or UNAVAILABLE}')
            continue
        lines.append(f'  {label} {split} | scope={_show(evaluation.get("scope"))}')
        cases = _list(group.get('cases'), group_path + '.cases')
        if group.get('cases') is None:
            lines.append('    cases: ' + UNAVAILABLE)
        elif not cases:
            lines.append('    cases: [] (stored empty list)')
        for index, item in enumerate(cases):
            case_path = f'{group_path}.cases[{index}]'
            case = _object(item, case_path)
            stopped = _status(case)
            if stopped:
                lines.append('    case=' + _show(case.get('case_id')) + ' | ' + stopped)
                continue
            score = _object(case.get('score'), case_path + '.score')
            score_stopped = _status(score)
            fields = ('score_std', 'score_min', 'score_max',
                      'mean_positive_minus_unobserved', 'exact_tie_rate', 'comparisons')
            signals = ' | '.join(key + '=' + _number(score.get(key), case_path + '.score.' + key) for key in fields)
            lines.append('    case=' + _show(case.get('case_id'))
                         + ' | records=' + _number(case.get('records'), case_path + '.records')
                         + (' | score_status=' + score_stopped if score_stopped else '')
                         + ' | ' + signals)
            trace = _object(case.get('trace'), case_path + '.trace')
            trace_stopped = _status(trace)
            if trace_stopped:
                lines.append('      trace=' + trace_stopped)
                continue
            stages = _list(trace.get('stages'), case_path + '.trace.stages')
            stage_signals = []
            for stage_index, stage_item in enumerate(stages):
                stage_path = f'{case_path}.trace.stages[{stage_index}]'
                stage = _object(stage_item, stage_path)
                stage_stopped = _status(stage)
                if stage_stopped:
                    stage_signals.append(_show(stage.get('stage')) + ':' + stage_stopped)
                    continue
                stage_signals.append(_show(stage.get('stage'))
                                     + '(raw=' + _number(stage.get('centered_energy'), stage_path + '.centered_energy')
                                     + ',normalized=' + _number(stage.get('normalized_centered_energy'), stage_path + '.normalized_centered_energy')
                                     + ',mean_norm=' + _number(stage.get('mean_norm'), stage_path + '.mean_norm') + ')')
            if trace.get('stages') is None:
                stage_signals.append(UNAVAILABLE)
            elif not stages:
                stage_signals.append('[] (stored empty list)')
            prototype = '/'.join(_number(trace.get('prototype_cross_class_cosine_' + key), case_path + '.trace.prototype_' + key) for key in ('min', 'max', 'mean'))
            lines.append('      trace ' + ' -> '.join(stage_signals) + ' | prototype_cosine min/max/mean=' + prototype)


def _gradient_signals(branch: dict, path: str) -> str:
    gradient = _object(branch.get('per_loss_query_gradient'), path + '.per_loss_query_gradient')
    stopped = _status(gradient)
    if stopped:
        return '  query gradients | ' + stopped
    parts = ['ranking_pairs=' + _number(gradient.get('ranking_pairs'), path + '.gradient.ranking_pairs')]
    losses = _object(gradient.get('losses'), path + '.gradient.losses')
    for name in ('ranking', 'observation_ce', 'alignment'):
        loss = _object(losses.get(name), path + '.gradient.losses.' + name)
        stopped = _status(loss)
        if stopped:
            parts.append(name + ':' + stopped)
            continue
        dependency = loss.get('query_embedding_dependency')
        norm = _number(loss.get('query_embedding_gradient_norm'), path + '.gradient.' + name + '.norm')
        if dependency is False and loss.get('query_embedding_gradient_norm') is None:
            norm = 'none (no query dependency)'
        parts.append(name + ':status=' + _show(loss.get('status'))
                     + ',loss=' + _number(loss.get('weighted_loss'), path + '.gradient.' + name + '.weighted_loss')
                     + ',dep=' + _boolean(dependency, path + '.gradient.' + name + '.dependency')
                     + ',norm=' + norm)
    return '  query gradients | ' + ' | '.join(parts)


def format_signals(report: dict) -> str:
    """Format stored per-case score/spread signals; never recompute scores."""
    if not isinstance(report, dict):
        raise ValueError('report: expected JSON object')
    _finite_json(report)
    if report.get('format') != FORMAT:
        raise ValueError('Unsupported reference comparison report format')
    comparison = _object(report.get('comparison'), 'comparison')
    if comparison.get('diagnostic_only') is not True:
        raise ValueError('Expected an explicitly diagnostic-only comparison')
    snapshot = _object(report.get('snapshot'), 'snapshot')
    lines = ['SAVED REFERENCE L1 SIGNALS | No model execution, score reranking or metric recomputation.',
             'Scope: original saved selected-case signals; not full accuracy or CP efficacy.',
             'snapshot | epoch=' + _number(snapshot.get('epoch'), 'snapshot.epoch')
             + ' | step=' + _number(snapshot.get('step'), 'snapshot.step')
             + ' | phase=' + _show(snapshot.get('phase')),
             'run=' + _show(report.get('run')) + ' | checkpoint=' + _show(report.get('checkpoint'))]
    branches = _list(comparison.get('branches'), 'comparison.branches')
    if not branches:
        lines.append('branches: ' + UNAVAILABLE)
    for index, item in enumerate(branches):
        path = f'comparison.branches[{index}]'
        branch = _object(item, path)
        lines.append('BRANCH ' + _show(branch.get('branch')))
        stopped = _status(branch)
        if stopped:
            lines.append('  status=' + stopped)
            continue
        if comparison.get('fixed_weight') is True or 'modes' in branch:
            lines.append(_gradient_signals(branch, path))
            _case_signals(lines, branch.get('initial_full_case_evaluation'), 'initial_full_case_evaluation', path + '.initial_full_case_evaluation')
        else:
            _case_signals(lines, branch.get('before'), 'before', path + '.before')
            _case_signals(lines, branch.get('after'), 'after', path + '.after')
            updates = _list(branch.get('updates'), path + '.updates')
            if branch.get('updates') is None:
                lines.append('  updates: ' + UNAVAILABLE)
            elif not updates:
                lines.append('  updates: [] (stored empty list)')
            for update_index, update_value in enumerate(updates):
                update_path = f'{path}.updates[{update_index}]'
                update = _object(update_value, update_path)
                stopped = _status(update)
                prefix = '  update step=' + _number(update.get('step'), update_path + '.step')
                if stopped:
                    lines.append(prefix + ' | ' + stopped)
                    continue
                terms = _object(update.get('terms'), update_path + '.terms')
                stopped = _status(terms)
                if stopped:
                    lines.append(prefix + ' | terms=' + stopped)
                    continue
                keys = ('ranking_pairs', 'ranking_loss', 'observation_auxiliary_loss', 'alignment_loss')
                lines.append(prefix + ' | ' + ' | '.join(key + '=' + _number(terms.get(key), update_path + '.terms.' + key) for key in keys))
    return '\n'.join(lines) + '\n'


def read_summary(path: str | Path, *, signals: bool = False) -> str:
    """Read a saved JSON report without changing its bytes."""
    with Path(path).open('r', encoding='utf-8-sig') as source:
        report = json.load(source)
    return format_signals(report) if signals else format_summary(report)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    parser.add_argument('--output', type=Path, help='Optional NEW text file; existing files are never overwritten')
    parser.add_argument('--signals', action='store_true', help='Print saved per-case score, spread and loss signals instead of the default summary')
    args = parser.parse_args(argv)
    summary = read_summary(args.report, signals=args.signals)
    if args.output is not None:
        with args.output.open('x', encoding='utf-8', newline='\n') as target:
            target.write(summary)
    print(summary, end='', flush=True)


if __name__ == '__main__':
    main()
