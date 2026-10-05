"""Compact terminal reporting of saved transition metadata, never neural work.

All evidence is retained in the returned JSON value. Display shortening affects
only terminal text; it never selects a subset of observations or metric rows.
"""
from __future__ import annotations

from collections import Counter
import csv
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath

FORMAT = 'hiercp_transition_terminal_summary_v1'


def _mapping(value):
    return value if isinstance(value, dict) else {}


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        result = float(value)
    except ValueError:
        return None
    return result if math.isfinite(result) else None


def _metric(value):
    """Actual original/native metadata key names; no zero substitution."""
    value = _mapping(value)
    keys = {'MRR': ('mrr', 'ranking_mrr'), 'top1': ('acc', 'ranking_recall_at_1'),
            'margin': ('margin',), 'loss': ('loss', 'ranking_pairwise_loss')}
    return {name: next((_number(value[k]) for k in names if k in value), None)
            for name, names in keys.items()}


class EvidenceReader:
    def __init__(self, receipt, directory):
        self.directory = Path(directory).resolve()
        self.files = {item['member']: item for item in receipt.get('input_files', [])
                      if isinstance(item, dict) and isinstance(item.get('member'), str)}
        self.failures = []

    def read(self, member):
        value = PurePosixPath(member)
        if (value.is_absolute() or '..' in value.parts or '\\' in member or ':' in member):
            raise ValueError('Unsafe saved metadata member: ' + member)
        path = self.directory.joinpath(*value.parts).resolve()
        if not path.is_relative_to(self.directory):
            raise ValueError('Saved metadata escapes report directory')
        if not path.is_file():
            return None
        raw = path.read_bytes()
        expected = self.files.get(member)
        if expected is not None and (hashlib.sha256(raw).hexdigest() != expected.get('sha256')
                                     or len(raw) != expected.get('bytes')):
            self.failures.append(dict(member=member, reason='saved_evidence_hash_or_size_differs'))
            return None
        return raw

    def json(self, member):
        raw = self.read(member)
        if raw is None:
            return None
        try:
            return json.loads(raw.decode('utf-8-sig'))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            self.failures.append(dict(member=member, reason='invalid_json', error=str(error)))
            return None

    def rows(self, member):
        raw = self.read(member)
        if raw is None:
            return []
        try:
            text = raw.decode('utf-8-sig')
            if member.endswith('.jsonl'):
                return [json.loads(line) for line in text.splitlines() if line.strip()]
            if member.endswith('.csv'):
                return list(csv.DictReader(io.StringIO(text)))
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return parsed
            if isinstance(parsed, dict):
                return [parsed]
            raise ValueError('Expected metadata mapping or row list')
        except (UnicodeDecodeError, json.JSONDecodeError, csv.Error, ValueError) as error:
            self.failures.append(dict(member=member, reason='metadata_parse_failed', error=str(error)))
            return []


def _comparison(reader, role, result_name):
    prefix = 'comparisons/' + role + '/results/' + result_name + '/'
    rows, sources = [], []
    for name in sorted(reader.files):
        if name.startswith(prefix) and name.endswith(('.json', '.jsonl', '.csv')):
            parsed = reader.rows(name)
            rows.extend((name, item) for item in parsed if isinstance(item, dict))
            sources.append(name)
    epoch_rows, conflicts, postruns = {}, [], []
    initial = None
    for member, row in rows:
        if member.endswith(('initial_validation.json', 'validation_initial.json')):
            metrics = _metric(row.get('metrics'))
            if any(value is not None for value in metrics.values()):
                initial = dict(metrics=metrics, member=member, full_validation=row.get('full_validation'))
        event = row.get('event')
        if event == 'PostRun':
            postruns.append(dict(member=member, **row))
        if event != 'EpochPostRun':
            continue
        epoch = row.get('epoch')
        if type(epoch) is not int or epoch < 1:
            reader.failures.append(dict(member=member, reason='invalid_epoch_cursor'))
            continue
        candidate = dict(epoch=epoch, validation=_metric(row.get('validation')),
                         train=_metric(row.get('train')), best_epoch=row.get('best_epoch'),
                         checkpoint_selection=row.get('checkpoint_selection'),
                         epoch_wall_seconds=_mapping(row.get('telemetry')).get('epoch_wall_seconds'),
                         evidence_members=[member])
        existing = epoch_rows.get(epoch)
        if existing is not None:
            if (existing['validation'] != candidate['validation']
                    or existing['best_epoch'] != candidate['best_epoch']):
                conflicts.append(dict(epoch=epoch, previous=existing, new=candidate))
            else:
                existing['evidence_members'].append(member)
        else:
            epoch_rows[epoch] = candidate
    conflict_epochs = {row['epoch'] for row in conflicts}
    last_epoch = max(epoch_rows) if epoch_rows else None
    latest = epoch_rows.get(last_epoch) if last_epoch not in conflict_epochs else None
    # Follow recorded checkpoint selection. Highest MRR alone is not the original
    # tie-breaking rule and is never substituted when selection is unavailable.
    best_epoch = latest.get('best_epoch') if latest is not None else None
    best = epoch_rows.get(best_epoch) if type(best_epoch) is int and best_epoch not in conflict_epochs else None
    full_training = True if any(row.get('full_training_complete') is True for row in postruns) else None
    observed_epochs = sorted(epoch_rows)
    return dict(role=role, result_name=result_name, initial_validation=initial,
                recorded_epochs=observed_epochs, epoch_records=len(epoch_rows),
                all_40_epoch_records_present=observed_epochs == list(range(1, 41)) and not conflicts,
                latest=latest, latest_recorded_epoch=last_epoch, best=best, best_epoch=best_epoch,
                full_training_complete_reported=full_training,
                conflicts=conflicts, postruns=postruns, source_members=sources,
                scope='Saved v1 eight-candidate curriculum metrics; not full native P/U evaluation or CP efficacy')


def _split_counts(audit):
    split, rows = _mapping(audit.get('split')), audit.get('per_case')
    rows = rows if isinstance(rows, list) else []
    result = {}
    for name, cases in split.items():
        if not isinstance(cases, list):
            result[name] = dict(available=False, reason='split_case_list_missing')
            continue
        members = set(cases)
        selected = [row for row in rows if isinstance(row, dict) and row.get('case_id') in members]
        p = [row.get('P') for row in selected]
        u = [row.get('U') for row in selected]
        valid = all(type(count) is int and count >= 0 for count in p + u)
        result[name] = dict(available=valid, declared_cases=len(cases), inventoried_cases=len(selected),
                            P=sum(p) if valid else None, U=sum(u) if valid else None,
                            records=sum(p + u) if valid else None)
    return result


def build_terminal_summary(receipt, output_dir):
    """Read saved evidence only; return the complete summary without writing files."""
    if not isinstance(receipt, dict):
        raise ValueError('Receipt mapping required')
    for flag in ('quality_verified', 'production_ready', 'training_started', 'neural_forward_executed'):
        if receipt.get(flag) is not False:
            raise ValueError('A metadata receipt cannot assert ' + flag)
    reader = EvidenceReader(receipt, output_dir)
    identity = _mapping(receipt.get('checkpoint_identity'))
    state = _mapping(receipt.get('checkpoint_state'))
    experiment = _mapping(reader.json('native/experiment.json'))
    request = _mapping(experiment.get('request'))
    execution = _mapping(reader.json('native/attempt/execution_contract.json'))
    schedule = _mapping(reader.json('native/attempt/learning_schedule.json'))
    checkpoint = _mapping(reader.json('native/checkpoint_metadata.json'))
    saved_payload = _mapping(checkpoint.get('payload'))
    optimizer = _mapping(saved_payload.get('optimizer'))
    groups = optimizer.get('param_groups')
    groups = groups if isinstance(groups, list) else []
    learning_rates = [_number(_mapping(group).get('lr')) for group in groups]
    inventory = _mapping(receipt.get('inventory'))
    missing = list(receipt.get('required_fields_missing', []))
    required = Counter('.'.join(name.split('.')[:2]) for name in missing if isinstance(name, str))
    native = dict(checkpoint=receipt.get('native_checkpoint'), saved_cursor={
        name: state.get(name) for name in ('epoch', 'step', 'phase', 'next_batch', 'batch', 'selected_epoch')},
        checkpoint_identity={name: identity.get(name) for name in (
            'local_cnn', 'config', 'base', 'ranking', 'learning_policy', 'support_training',
            'precision', 'epochs', 'workers', 'candidates', 'resource_limits',
            'resident_budget_bytes', 'activation_storage', 'execution_pipeline')},
        wrapper_request=request, execution_contract=execution, learning_schedule=schedule,
        current_optimizer_learning_rates=learning_rates,
        configured_learning_rate=_mapping(_mapping(identity.get('base')).get('training')).get('lr'),
        optimizer_scheduler_state=saved_payload.get('scheduler'),
        full_inventory=inventory, split_counts=_split_counts(inventory))
    history = state.get('validation_history')
    history = history if isinstance(history, list) else []
    native['saved_validation_history'] = history
    native['metric_semantics'] = dict(
        MRR='case mean reciprocal rank of first observed P; cases without P excluded',
        top1='compatibility key for ranking_recall_at_1: micro recall over observed P, not case top1 accuracy',
        loss='within-case observed-P/unobserved-U pairwise loss')
    valid_history = [row for row in history if isinstance(row, dict) and type(row.get('epoch')) is int]
    native['initial_validation'] = _metric(state.get('initial_validation'))
    native['latest_validation'] = None if not valid_history else dict(
        epoch=valid_history[-1]['epoch'], metrics=_metric(valid_history[-1]))
    comparisons = [_comparison(reader, role, result) for role, result in
                   (('baseline', 'v1.0'), ('half_A', 'half_A'), ('half_B', 'half_B'))]
    return dict(format=FORMAT, mode='read_only_saved_metadata_terminal_report', native=native,
                comparisons=comparisons, source_checks=receipt.get('sources', []),
                source_check_counts=[dict(root=item.get('root'),
                    expected_files=item.get('expected_files'),
                    mismatches=len(item.get('mismatches', [])),
                    required_source_fields_missing=len(item.get('required_source_identity_files_missing', [])),
                    matches_checkpoint_source=item.get('matches_checkpoint_source'))
                    for item in receipt.get('sources', []) if isinstance(item, dict)],
                binding_failed=[name for name, value in _mapping(receipt.get('bindings')).items() if value is not True],
                required_fields_missing=missing, required_field_categories=dict(sorted(required.items())),
                collection_errors=receipt.get('errors', []), parsing_failures=reader.failures,
                exact_native_recipe_bound=receipt.get('exact_native_recipe_bound'),
                training_started=False, optimizer_updates=0, neural_forward_executed=False,
                full_evaluation_executed=False, quality_verified=False, production_ready=False,
                full_summary_path=str(Path(output_dir).resolve().with_name(
                    Path(output_dir).name + '.terminal_summary.json')),
                scope='Saved metadata collection. Original run results are reported evidence, not independently rerun metrics.')


def _brief(value, width=500):
    if value is None:
        return 'UNKNOWN'
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(',', ':'))
    encoded = encoded.replace('\n', ' ').replace('\r', ' ')
    return encoded if len(encoded) <= width else encoded[:width] + ' ... [full JSON preserved]'


def _display_metrics(value, *, native=False):
    metrics = _mapping(value)
    labels = {'MRR': 'MRR(first_P)', 'top1': 'R@1(observed_P)'} if native else {}
    return ' '.join(labels.get(name, name) + '=' + ('UNKNOWN' if metrics.get(name) is None else f"{metrics[name]:.6f}")
                    for name in ('MRR', 'top1', 'margin', 'loss'))


def render_terminal_summary(summary):
    """Render fewer than 80 lines, preserving complete details in summary JSON."""
    if summary.get('format') != FORMAT:
        raise ValueError('Transition terminal summary format differs')
    native = _mapping(summary.get('native'))
    identity = _mapping(native.get('checkpoint_identity'))
    config = _mapping(identity.get('config'))
    inventory = _mapping(native.get('full_inventory'))
    lines = ['=== TRANSITION RECEIPT | SAVED METADATA ONLY ===',
             'Neural forward=NO | optimizer updates=0 | new training=NO | new evaluation=NO',
             'Full server summary: ' + str(summary.get('full_summary_path')),
             'Native checkpoint: ' + str(native.get('checkpoint') or 'UNKNOWN'),
             'Saved epoch/step/phase/cursor/batch: ' + _brief(native.get('saved_cursor')),
             'Exact native recipe bound: ' + _brief(summary.get('exact_native_recipe_bound')),
             'L0: ' + _brief(identity.get('local_cnn')),
             'Upper config: ' + _brief({key: config.get(key) for key in (
                 'task_layers', 'alignment_layers', 'label_count', 'temperature', 'prototype_score', 'alignment_loss_weight')}),
             'GT: P=observed tumor; U=unobserved comparison; neither is donor-specific CP suitability',
             'Recorded label contract: ' + _brief(config.get('label_definition')),
             'Ranking: ' + _brief(identity.get('ranking')),
             'Learning policy: ' + _brief(identity.get('learning_policy')),
             'Support: ' + _brief(identity.get('support_training')),
             'Request resources: ' + _brief(_mapping(native.get('wrapper_request')).get('settings')),
             'Saved resources: ' + _brief(identity.get('resource_limits')),
             'Execution/cache: ' + _brief(identity.get('execution_pipeline')),
             'Precision: ' + _brief(identity.get('precision')) + ' | activation: ' + _brief(identity.get('activation_storage')),
             'Configured LR: ' + _brief(native.get('configured_learning_rate'))
             + ' | saved optimizer LR: ' + _brief(native.get('current_optimizer_learning_rates') or None),
             'Declared batch candidates/workers/epochs: ' + _brief({key: identity.get(key) for key in ('candidates', 'workers', 'epochs')}),
             'Resolved physical/effective/accumulation: ' + _brief({key: _mapping(native.get('execution_contract')).get(key)
                                                                 for key in ('physical_batch', 'effective_batch', 'accumulation')}),
             'Complete inventory: ' + _brief({key: inventory.get(key) for key in ('records', 'unique_ids', 'observed_positive', 'unobserved_comparison')}),
             'Split counts: ' + _brief(native.get('split_counts'), 1000),
             'Native initial validation | ' + _display_metrics(native.get('initial_validation'), native=True),
             'Native last validation epoch=' + str(_mapping(native.get('latest_validation')).get('epoch', 'UNKNOWN'))
             + ' | ' + _display_metrics(_mapping(native.get('latest_validation')).get('metrics'), native=True),
             'Native R@1 denominator=all observed P; v1 top1 denominator=original eight-candidate samples',
             'Source root checks: ' + _brief(summary.get('source_check_counts'), 1000),
             'Missing fields: ' + str(len(summary.get('required_fields_missing', [])))
             + ' | categories: ' + _brief(summary.get('required_field_categories'), 1000),
             'Failed bindings: ' + str(len(summary.get('binding_failed', []))),
             'Collection errors: ' + str(len(summary.get('collection_errors', [])))
             + ' | parsing failures: ' + str(len(summary.get('parsing_failures', []))),
             '--- Existing comparison evidence (v1 eight-candidate metrics) ---']
    for result in summary.get('comparisons', []):
        role = result['role']
        initial = _mapping(result.get('initial_validation'))
        best, latest = _mapping(result.get('best')), _mapping(result.get('latest'))
        lines.extend([
            role + ' | records=' + str(result.get('epoch_records')) + '/40'
            + ' | all40=' + _brief(result.get('all_40_epoch_records_present'))
            + ' | full training reported=' + _brief(result.get('full_training_complete_reported')),
            role + ' initial | ' + _display_metrics(initial.get('metrics')),
            role + ' best epoch=' + str(result.get('best_epoch') if result.get('best_epoch') is not None else 'UNKNOWN')
            + ' | ' + _display_metrics(best.get('validation')),
            role + ' last recorded epoch=' + str(result.get('latest_recorded_epoch') if result.get('latest_recorded_epoch') is not None else 'UNKNOWN')
            + ' | ' + _display_metrics(latest.get('validation')),
            role + ' last wall seconds=' + _brief(latest.get('epoch_wall_seconds'))
            + ' | conflicting epoch records=' + str(len(result.get('conflicts', [])))])
    lines.extend(['Complete missing-field names, source mismatches and parse errors are preserved in the server summary JSON.',
                  'Metadata does not establish new model quality, GPU validity or production readiness.',
                  '=== END TRANSITION RECEIPT ==='])
    if len(lines) >= 80:
        raise ValueError('Terminal summary exceeds display contract')
    return '\n'.join(lines)
