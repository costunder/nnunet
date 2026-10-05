"""Collect completed actual C/D DEBUG evidence; never load tensors or run models.

Only existing small JSON/JSONL reports are copied into a new owned output.
Checkpoint bytes are hashed, not copied/deserialized. Two updates and complete
DEBUG P/128U coverage prove mechanics only, not production quality/readiness.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PureWindowsPath
import shutil


ROOT = Path(__file__).resolve().parents[1]
MAX_REPORT_BYTES = 16 * 1024**2  # Evidence-file limit, never an input/data cap.
C_MODULES = ('native_CNN', 'native_readout', 'native_fusion', 'upper_L1',
             'upper_L2', 'upper_L2_updates', 'upper_total')
D_MODULES = ('L0', 'L1', 'L2', 'L2_updates')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value, name, *, positive=False):
    require(type(value) in (int, float) and math.isfinite(value)
            and (not positive or value > 0), name + ' must be finite' + (' and >0' if positive else ''))
    return value


def read(path):
    path = Path(path)
    require(path.suffix in ('.json', '.jsonl') and path.stat().st_size <= MAX_REPORT_BYTES,
            'Only small JSON/JSONL evidence files are admitted: ' + str(path))
    def invalid(value):
        raise ValueError('Nonfinite JSON constant: ' + value)
    text = path.read_text(encoding='utf8')
    if path.suffix == '.jsonl':
        return [json.loads(line, parse_constant=invalid) for line in text.splitlines() if line.strip()]
    return json.loads(text, parse_constant=invalid)


def member(root, relative):
    root = Path(root).resolve(strict=True)
    require(isinstance(relative, str) and relative and not Path(relative).is_absolute()
            and not PureWindowsPath(relative).is_absolute() and ':' not in relative,
            'Evidence/source member must be relative: ' + str(relative))
    path = root / relative
    require(not path.is_symlink(), 'Symlink evidence/source member refused: ' + relative)
    path = path.resolve(strict=True)
    require(path.is_relative_to(root) and path.is_file(), 'Evidence/source member escapes its root: ' + relative)
    return path


def validate_sources(manifest, workspace):
    """All manifest-bound sources, including frozen config, must match bytes."""
    workspace = Path(workspace).resolve(strict=True)
    sources = manifest.get('sources')
    require(isinstance(sources, dict) and sources, 'Missing execution-bound source inventory')
    for name, digest in sources.items():
        require(isinstance(digest, str) and sha(member(workspace, name)) == digest,
                'Current execution-bound source byte mismatch: ' + name)
    proof = manifest['source']
    snapshot = Path(proof['source']).resolve(strict=True)
    require(snapshot.is_relative_to(workspace), 'Archived snapshot must belong to this workspace')
    verified = proof.get('verified_files')
    require(isinstance(verified, dict) and verified, 'Missing actual original snapshot byte proof')
    for name, digest in verified.items():
        require(sha(member(snapshot, name)) == digest, 'Preserved snapshot source byte mismatch: ' + name)
    archive = member(workspace, 'versions/v1/pipeline_v1_source.zip')
    require(sha(archive) == proof['archive_sha256'], 'Preserved original archive byte mismatch')
    return dict(current_execution_bytes_equal=True, execution_sources=sources,
                execution_source_count=len(sources), original_archive_sha256=proof['archive_sha256'],
                original_snapshot_source_count=len(verified),
                nonexecution_evidence_docs='Not part of the execution source inventory or copied as source proof')


def validate_evaluation(value):
    require(value.get('debug') is True and value.get('quality_verified') is False
            and value.get('full_evaluation') is False and value.get('full_training_complete') is False,
            'Evaluation must explicitly remain DEBUG mechanics, without quality/full-run claims')
    require(value.get('task') == 'native_observed_P_vs_unobserved_U'
            and value.get('GT_is_donor_compatibility') is False
            and value.get('original_eight_candidate_metrics') is False,
            'Observed P/U task must retain its original GT meaning')
    cohort = value['cohort']
    require((cohort['records'], cohort['observed_P'], cohort['unobserved_U']) == (136, 8, 128)
            and len(cohort['case_ids']) == 1 and value['support_records'] == 401
            and value['full_128_U_per_case'] is True, 'Expected whole DEBUG136=P8+128U and full401-row bank')
    require(value.get('C_curriculum_support_reused') is False,
            'Common native401-row observation bank cannot reuse C eight-candidate support labels')
    denominator = value['denominators']
    require((denominator['cases'], denominator['rank_evaluable_cases'], denominator['zero_P_cases'],
             denominator['observed_P'], denominator['unobserved_U'], denominator['P_U_pairs']) == (1, 1, 0, 8, 128, 1024),
            'Actual DEBUG evaluation denominator counts changed')
    scoring = value['scoring_contract']
    require(scoring['upper_chunking'] is False and scoring['upper_execution'] == 'single_joint_case'
            and scoring['upper_invocations'] == 1 and scoring['query_GT_in_forward'] is False,
            'Whole-case single upper invocation without query GT required')
    for name in ('execution_contract_bound', 'scoring_callbacks_executed',
                 'callback_internal_execution_verified', 'raw_CT_execution_verified'):
        require(value.get(name) is True, 'Missing actual execution verification: ' + name)
    require(len(value['cases']) == 1, 'Exactly the DEBUG validation case required')
    case = value['cases'][0]
    scores = case['case_scores']
    ids = [row['record_id'] for row in scores]
    require(case['case_id'] == cohort['case_ids'][0] and len(scores) == len(set(ids)) == 136
            and (case['records'], case['observed_P'], case['unobserved_U']) == (136, 8, 128)
            and set(ids) == set(case['scored_record_ids']) == set(scoring['scored_record_ids'])
            and sum(row['observed'] == 1 for row in scores) == 8
            and sum(row['observed'] == 0 for row in scores) == 128,
            'Actual complete unique P/128U score rows required')
    for row in scores:
        require(type(row['observed']) is int and row['observed'] in (0, 1), 'Exact external P/U labels required')
        finite(row['score'], 'Actual evaluation score')
    for name, metric in value['metrics'].items():
        finite(metric, 'Evaluation metric ' + name)
    return dict(case_ids=cohort['case_ids'], records=136, observed_P=8, unobserved_U=128,
                full_training_bank_rows=401, upper_invocations=1, query_GT_in_forward=False,
                scores_sha256=value['scores_sha256'], model_sha256=value['model_sha256'],
                initial_upper_sha256=value['initial_upper_sha256'], metrics=value['metrics'],
                quality_verified=False, full_evaluation=False)


def _artifact(root, name):
    path = Path(name).resolve(strict=True)
    require(path.is_relative_to(Path(root).resolve(strict=True)), 'Reported artifact escapes its actual arm output')
    return path


def validate_d_gradients(gradient):
    require(set(D_MODULES) <= gradient.keys(), 'Missing D core-module gradients')
    for name in D_MODULES:
        finite(gradient[name], 'D gradient ' + name, positive=True)
    require(type(gradient.get('trainable_parameter_tensors')) is int
            and gradient['trainable_parameter_tensors'] > 0
            and type(gradient.get('parameter_tensors_with_gradient')) is int
            and gradient['parameter_tensors_with_gradient'] == gradient['trainable_parameter_tensors']
            and gradient.get('missing_parameter_gradients') == [],
            'D full trainable-parameter gradient coverage is missing or incomplete')
    return gradient['trainable_parameter_tensors']


def read_arm(root, arm, workspace=ROOT):
    root = Path(root).resolve(strict=True)
    manifest_path = root / 'manifest.json'
    manifest = read(manifest_path)
    require(manifest.get('arm') == arm and manifest.get('debug') is True
            and manifest['settings']['debug_updates'] == 2, 'Explicit matching two-update DEBUG manifest required')
    source_receipt = validate_sources(manifest, workspace)
    constructed_path = root / 'model_execution.jsonl'
    constructed = read(constructed_path)
    require(constructed and all(row == constructed[0] for row in constructed),
            'All actual constructed-model execution reports must have identical configuration')
    constructed_invocations = len(constructed)
    constructed = constructed[0]
    require(constructed['arm'] == arm and constructed['debug'] is True
            and constructed['stage'] == 'constructed_crossed_model'
            and (constructed['upper']['hidden_dim'], constructed['upper']['heads'],
                 constructed['upper']['L1_layers'], constructed['upper']['L2_layers']) == (128, 4, 2, 2)
            and constructed['actual_common_train_observations'] == 401
            and constructed['actual_common_validation_observations'] == 136
            and constructed['worker_count'] == manifest['settings']['workers'],
            'Actual constructed architecture/cohort/workers differ from the full DEBUG contract')
    finite(constructed['total_parameters'], 'Actual total parameters', positive=True)
    finite(constructed['trainable_parameters'], 'Actual trainable parameters', positive=True)
    require(constructed['trainable_parameters'] <= constructed['total_parameters'], 'Invalid constructed parameter counts')
    training = root / 'training'
    if arm == 'C':
        report_paths = sorted(training.glob('c_training_report_*.json'),
                              key=lambda path: (path.stat().st_mtime_ns, path.name))
        require(report_paths, 'Missing completed actual C DEBUG invocation reports')
        invocation_reports = [read(path) for path in report_paths]
        for invocation_report in invocation_reports:
            require(invocation_report['arm'] == 'C' and invocation_report['status'] == 'DEBUG_COMPLETE'
                    and invocation_report['debug'] is True and invocation_report['updates'] == 2
                    and invocation_report['actual_CUDA'] is True and invocation_report['actual_raw_CT'] is True
                    and invocation_report['full_training'] is False and invocation_report['full_evaluation'] is False
                    and invocation_report['quality_verified'] is False and invocation_report['source_preserved'] is True
                    and invocation_report['missing_gradients'] == [], 'Incomplete or inconsistent actual C DEBUG report')
        report_path, report = report_paths[-1], invocation_reports[-1]
        require(all(row['run_identity_sha256'] == report['run_identity_sha256']
                    and row['input_spec'] == report['input_spec'] and row['execution'] == report['execution']
                    for row in invocation_reports), 'C resume changed source/configuration execution identity')
        update_paths = sorted(training.glob('c_updates_*.jsonl'),
                              key=lambda path: (path.stat().st_mtime_ns, path.name))
        require(update_paths, 'Missing actual C optimizer journals')
        attempts = [row for path in update_paths for row in read(path)]
        update_path = update_paths[-1]
        updates = [row for row in attempts if row['status'] == 'OPTIMIZER_UPDATED']
        require([row['update'] for row in updates] == [1, 2], 'Exactly two successful C optimizer updates required')
        changes = {}
        for row in updates:
            finite(row['loss'], 'C successful update loss')
            require(row['gradient']['finite'] is True and not row['gradient']['nonfinite_groups'],
                    'Successful C optimizer update lacks finite gradients')
            require(set(C_MODULES) <= row['parameter_updates'].keys()
                    and set(C_MODULES) <= row['gradient']['norms'].keys(), 'Missing C core-module evidence')
            for name in C_MODULES:
                finite(row['gradient']['norms'][name], 'C gradient ' + name, positive=True)
                change = row['parameter_updates'][name]
                finite(change['L2_change'], 'C module optimizer change ' + name, positive=True)
                require(change['changed_parameter_tensors'] > 0, 'C core parameters did not change: ' + name)
                changes[name] = changes.get(name, 0.) + change['L2_change']
        evaluation_path = _artifact(root, report['final_DEBUG_evaluation']['common']['evaluation_artifact'])
        checkpoint = _artifact(root, report['checkpoint'])
        architecture = dict(input_L0=report['input_spec'], execution=report['execution'])
    else:
        report_path = training / 'DEBUG_neural_execution.jsonl'
        reports = read(report_path)
        require(len(reports) == 1, 'Exactly one final actual D neural DEBUG receipt required')
        report = reports[0]
        require(report['debug'] is True and report['actual_CT'] is True and report['actual_CUDA'] is True
                and report['updates'] == 2 and report['full_training'] is False and report['full_evaluation'] is False
                and report['quality_verified'] is False, 'Incomplete actual D DEBUG report')
        update_path = training / 'updates.jsonl'
        updates = read(update_path)
        require([row['step'] for row in updates] == [1, 2], 'Exactly two successful D optimizer updates required')
        changes = report['optimizer_weight_changes']
        require(set(D_MODULES) <= changes.keys(), 'Missing D core-module optimizer evidence')
        for name, change in changes.items():
            finite(change, 'D module optimizer change ' + name, positive=True)
        for row in updates:
            finite(row['loss'], 'D successful update loss')
            validate_d_gradients(row['gradients'])
            require(row['local_graphs'] == 2 * row['physical_observations'], 'D actual two-view graph coverage changed')
        require(len({row['gradients']['trainable_parameter_tensors'] for row in updates}) == 1,
                'D trainable-parameter count changed between optimizer updates')
        require(report['initial_neural_sha256'] != report['final_neural_sha256'], 'D neural bytes did not change')
        evaluation_path = _artifact(root, report['evaluation_artifact'])
        checkpoint = member(training, 'checkpoint_latest.pt')
        architecture = dict(input_L0='actual archived v1 six-role dense CNN/GAT3/readouts; five-channel48; two views',
                            checkpoint_dense_encoder=False, checkpoint_local_blocks=False,
                            target_erasure=True, views=2, original_view_consistency_weight=.1)
    evaluation = validate_evaluation(read(evaluation_path))
    architecture['constructed_model'] = constructed
    paths = {manifest_path, report_path, update_path, evaluation_path, root / 'resources.jsonl'}
    # Only actual small reports; never recursively traverse canonical graphs/CT.
    paths.update(root.glob('*.jsonl'))
    paths.update(training.glob('*.json'))
    paths.update(training.glob('*.jsonl'))
    for path in paths:
        read(path)
    return dict(arm=arm, debug=True, actual_raw_CT=True, actual_CUDA=True,
                successful_optimizer_updates=2, finite_successful_gradients=True,
                constructed_model_invocations=constructed_invocations,
                core_module_weight_changes=changes, architecture=architecture,
                hardware=manifest['hardware'], settings=manifest['settings'], evaluation=evaluation,
                source=source_receipt, manifest_sha256=sha(manifest_path),
                native_inventory_sha256=manifest['native_inventory_sha256'], scope_contract=manifest['scope'],
                reused_prepared_cache=manifest.get('reused_prepared_cache'),
                checkpoint=dict(file=checkpoint.name, file_sha256=sha(checkpoint), bytes=checkpoint.stat().st_size,
                                copied=False, tensor_deserialized=False),
                full_training=False, full_evaluation=False, quality_verified=False), sorted(paths)


def read_preparation(root):
    root = Path(root).resolve(strict=True)
    index = root if root.is_file() else root / 'canonical_cache/index.json'
    if not index.is_file():
        index = root / 'index.json'
    meta = read(index)
    require(meta['debug'] is True and meta['complete'] is True and meta['prepared_observations'] == 537
            and len(meta['records']) == 537 and meta['views_per_observation'] == 2
            and meta['source_total_assignments'] == 14102 and meta['skipped_observations'] == 0
            and meta['donor_redraws'] == 0 and meta['hidden_subset'] is False,
            'Expected complete537-observation/1074-view DEBUG preparation with native14102 metadata')
    for kind in ('nodes', 'edges'):
        values = meta['sampled_two_view_' + kind]
        require(len(values) == 537 and values == [row['sampled_two_view_' + kind] for row in meta['records']],
                'Preparation sampled graph measurement coverage changed')
        for row, value in zip(meta['records'], values):
            require(type(value) is int and value > 0 and value == sum(row['sampled_view_' + kind])
                    and row['measurement_epoch'] == 0, 'Actual epoch0 two-view measurement required')
        summary = meta['sampled_two_view_summary'][kind]
        require(summary['sum'] == sum(values) and summary['max'] == max(values)
                and summary['min'] == min(values) and summary['actual_local_graphs'] == 1074,
                'Preparation sampled measurement summary changed')
    metric_paths = [member(index.parent, name) for name in meta['preparation_metric_files']]
    events = [row for path in metric_paths for row in read(path)]
    require(events and all(row['debug'] is True for row in events), 'Explicit actual DEBUG preparation telemetry required')
    done = [row for row in events if row['event'] == 'all_observations_completed']
    observations = [row for row in events if row['event'] == 'observation_completed']
    require(done and done[-1]['completed_observations'] == 537 and len({r['observation_id'] for r in observations}) == 537,
            'Actual preparation completion/progress telemetry omitted observations')
    for event in events:
        for name in ('rss_bytes', 'raw_resident_bytes', 'canonical_resident_bytes',
                     'disk_cumulative_canonical_bytes', 'disk_free_bytes', 'seconds'):
            finite(event[name], 'Preparation telemetry ' + name)
    summary = dict(debug=True, prepared_observations=537, actual_local_graphs=1074,
        full_native_assignment_metadata=14102, canonical_index_sha256=sha(index),
        sampled_two_view_summary=meta['sampled_two_view_summary'],
        preparation_seconds=meta['preparation_seconds'], max_recorded_rss_bytes=max(row['rss_bytes'] for row in events),
        max_recorded_raw_resident_bytes=max(row['raw_resident_bytes'] for row in events),
        max_recorded_canonical_resident_bytes=max(row['canonical_resident_bytes'] for row in events),
        disk_cumulative_canonical_bytes=meta['disk_cumulative_canonical_bytes'],
        final_disk_free_bytes=done[-1]['disk_free_bytes'],
        compressed_graph_bytes=sum(row['compressed_graph_bytes'] for row in meta['records']),
        shared_source_files=len({(row['segment'], row['shared_source']['path']) for row in meta['records']}),
        input_L0_contract=meta['local_identity']['contract'], scope_contract=meta['scope_contract'],
        input_inventory_sha256=meta['input_inventory_sha256'], full_training=False, quality_verified=False)
    return summary, [index, index.parent / 'prepare_request.json', *metric_paths]


def collect(c_output, d_output, preparation_output, output, *, workspace=ROOT):
    # Validate everything before creating/publishing any destination result.
    c, c_files = read_arm(c_output, 'C', workspace)
    d, d_files = read_arm(d_output, 'D', workspace)
    preparation, preparation_files = read_preparation(preparation_output)
    require(c['source']['execution_sources'] == d['source']['execution_sources'], 'Final C/D execution source inventories differ')
    require(c['evaluation']['initial_upper_sha256'] == d['evaluation']['initial_upper_sha256'], 'C/D initial common upper bytes differ')
    require(c['native_inventory_sha256'] == d['native_inventory_sha256'] == preparation['input_inventory_sha256']
            and c['scope_contract'] == d['scope_contract'] == preparation['scope_contract'],
            'C/D preparation inventory or physical scope differs')
    if d['reused_prepared_cache'] is not None:
        require(d['reused_prepared_cache']['sha256'] == preparation['canonical_index_sha256']
                and d['reused_prepared_cache']['read_only'] is True, 'D reused another preparation cache')
    # This frozen declaration is itself byte-bound in both actual run manifests.
    declaration = read(member(workspace, 'config/v17_crossed_training.json'))
    common = declaration['shared']
    require((common['hidden_dim'], common['heads'], common['upper_task_layers'], common['upper_alignment_layers'])
            == (128, 4, 2, 2), 'Full common upper width/head/layer contract changed')
    upper = dict(operator='actual HalfBUpper / tracked legacy PromptGraphModel', hidden_dim=128, heads=4,
                 L1_layers=2, L2_layers=2, temperature=common['temperature'],
                 scalar_score='native patient-mass logit1-minus-logit0',
                 initial_tensor_sha256=c['evaluation']['initial_upper_sha256'])
    c['architecture']['common_upper'] = upper
    d['architecture']['common_upper'] = upper
    d['architecture']['input_L0_contract'] = preparation['input_L0_contract']
    for path in [*c_files, *d_files, *preparation_files]:
        read(path)
    output = Path(output).resolve()
    require(not output.exists(), 'Evidence destination already exists; previous results are never overwritten')
    output.mkdir(parents=True, exist_ok=False)
    copied = []
    for label, root, files in (('C', Path(c_output), c_files), ('D', Path(d_output), d_files),
                              ('preparation', Path(preparation_files[0]).parent, preparation_files)):
        root = root.resolve(strict=True)
        for path in files:
            value = read(path)  # Explicitly refuse non-JSON or oversized evidence.
            del value
            relative = path.resolve().relative_to(root)
            target = output / label / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with path.open('rb') as source, target.open('xb') as destination:
                shutil.copyfileobj(source, destination)
            require(sha(target) == sha(path), 'Copied report byte mismatch')
            copied.append(dict(path=target.relative_to(output).as_posix(), sha256=sha(target), bytes=target.stat().st_size))
    receipt = dict(format='v17_crossed_actual_CT_CUDA_DEBUG_evidence_v1',
        created_UTC=datetime.now(timezone.utc).isoformat(), debug=True, actual_CT_CUDA_mechanics_verified=True,
        C=c, D=d, preparation=preparation, copied_reports=copied,
        collector_sha256=sha(Path(__file__)), collector_bound_to_training=False,
        verification_scope='Saved source-bound actual-run receipts; collector does not rerun models or deserialize checkpoints',
        tensors_or_CT_copied=False, checkpoint_tensor_deserialization=False,
        full_training=False, full_evaluation=False, quality_verified=False, production_ready=False)
    with (output / 'verification.json').open('x', encoding='utf8') as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
    return output / 'verification.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('C-output', 'D-output', 'preparation-output', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    print(collect(args.C_output, args.D_output, args.preparation_output, args.output))


if __name__ == '__main__':
    main()
