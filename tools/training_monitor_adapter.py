"""Read-only version adapters for the shared training monitor UI schema 1.

Only observed values are translated. Missing records remain None; this module
does not infer successful training/evaluation, start jobs, or import models.
"""
from __future__ import annotations


STAGES = {'gnn': 'GNN', 'scoring': 'CP 후보 채점', 'native': 'nnU-Net', 'metadata': '표시 기록 대기'}
RESOURCE_SOURCES = {'live owned arm process tree': '현재 GNN 프로세스 트리',
    'last successful train update': '과거 GNN train update 기록'}


def _report(report):
    report = report or {}
    metrics = report.get('metrics') or {}
    return dict(epoch=report.get('epoch'), active_u=report.get('active_u'),
        mrr=metrics.get('per_P_patient_mrr'), top1=metrics.get('per_P_patient_top1'))


def _problem(value):
    problem = value.get('read_problem') or {}
    detail = problem.get('detail')
    path = problem.get('path')
    return ' | '.join(str(item) for item in (path, detail) if item)


def from_pipeline(value, *, experiment_version=None):
    """Adapt verified GNN/CP/native metadata; top read health has precedence."""
    gnn, cp, native = (value.get(key) or {} for key in ('gnn', 'scoring', 'native'))
    reports = gnn.get('reports') or {}
    metrics = native.get('native_metrics') or {}
    stage = value.get('active_stage')
    resource = cp if stage == 'scoring' else native if stage == 'native' else gnn
    source = RESOURCE_SOURCES.get(resource.get('RSS_source'), resource.get('RSS_source'))
    if stage == 'scoring' and resource.get('RSS_GiB') is not None:
        source = '현재 CP 채점 프로세스'
    elif stage == 'native' and resource.get('RSS_GiB') is not None:
        source = '현재 nnU-Net 작업 프로세스'
    warning = None
    if not value.get('snapshot_fresh'):
        warning = ('최근 정상 표시 보존; 수치는 새 기록이 아님' if value.get('last_valid_snapshot_preserved')
            else '완성된 표시 기록 대기; 수치 미확인')
    elif cp.get('partial_console_line_pending'):
        warning = 'console 마지막 줄 기록 중; 완료된 줄만 표시'
    error = _problem(value) or gnn.get('error')
    if not error and cp.get('status') in ('SCORING_FAILED', 'EXITED_WITHOUT_COMPLETION', 'READ_ERROR'):
        error = cp.get('detail') or cp.get('status')
    if not error and native.get('status') in ('FAILED', 'EXITED_WITHOUT_COMPLETION', 'CONTROLLER_EXITED_CHILD_ALIVE'):
        error = native.get('native_pending_reason') or native.get('status')
    return dict(gpu=value.get('GPU'), experiment_version=experiment_version,
        state=dict(status=value.get('status'), stage=STAGES.get(stage, stage),
            fresh=value.get('snapshot_fresh'), last_confirmed_status=value.get('last_confirmed_status')),
        gnn=dict(epoch=gnn.get('epoch'), epochs=gnn.get('epochs'), completed_epochs=gnn.get('completed_epochs'),
            active_u=gnn.get('active_U'), optimizer_updates=gnn.get('optimizer_updates'),
            loss=gnn.get('loss'), loss_update=gnn.get('optimizer_updates'),
            full_completed=gnn.get('full_training_completed'), phase_label=gnn.get('stage_label') or gnn.get('stage'),
            progress=gnn.get('progress'), stage_validation=_report(reports.get('stage_validation')),
            full_validation=_report(reports.get('full_validation') or reports.get('initial_full_validation'))),
        cp=dict(status=cp.get('status'), source_done=cp.get('source_done'), source_total=cp.get('source_total'),
            positions_done=cp.get('CP_done'), positions_total=cp.get('CP_total'),
            last_batch_seconds=cp.get('last_batch_seconds'), log_age_seconds=cp.get('seconds_since_console_write'),
            input_basis=('GPU' + str(value.get('GPU')) + ' GNN BEST') if cp.get('own_BEST_pin_verified') else None),
        native=dict(status=native.get('status'), stage=native.get('stage'), epoch=native.get('epoch'),
            epochs=native.get('epochs'), completed_epochs=native.get('completed_epochs'),
            started=native.get('native_training_started'), full_training=native.get('full_native_training'),
            full_evaluation=native.get('full_native_evaluation'), train_loss=metrics.get('train_loss'),
            val_loss=metrics.get('val_loss'), pseudo_dice=metrics.get('pseudo_dice')),
        resources=dict(rss_gib=resource.get('RSS_GiB'), ram_limit_gib=resource.get('RAM_limit_GiB'),
            source=source, process_status=resource.get('process_status')),
        waiting_reason=value.get('native_pending_reason'), warning=warning, error=error)


def from_training(value, *, experiment_version=None):
    """Adapt a standalone GNN/native viewer without claiming unobserved stages."""
    native_stage = value.get('stage') == 'nnunet_training' or bool(value.get('native_metrics'))
    if native_stage:
        native = dict(value, native_training_started=value.get('status') == 'RUNNING')
        snapshot = dict(GPU=value.get('GPU'), status=value.get('status'), active_stage='native',
            native=native, gnn=None, scoring=None)
    else:
        snapshot = dict(GPU=value.get('GPU'), status=value.get('status'), active_stage='gnn',
            gnn=value, scoring=None, native=None)
    for key in ('snapshot_fresh', 'last_confirmed_status', 'last_valid_snapshot_preserved', 'read_problem'):
        snapshot[key] = value.get(key)
    model = from_pipeline(snapshot, experiment_version=experiment_version)
    if value.get('error'):
        model['error'] = value['error']
    return model


def from_scoring(value, *, experiment_version=None, include_native=True):
    """No hardcoded storage wait: use only the actual recorded pending reason."""
    snapshot = dict(GPU=value.get('GPU'), status=value.get('status'), active_stage='scoring',
        gnn=None, scoring=value, native=None,
        native_pending_reason=value.get('native_pending_reason') if include_native else None)
    for key in ('snapshot_fresh', 'last_confirmed_status', 'last_valid_snapshot_preserved', 'read_problem'):
        snapshot[key] = value.get(key)
    model = from_pipeline(snapshot, experiment_version=experiment_version)
    basis = 'GPU' + str(value.get('GPU', '?')) + ' GNN BEST'
    model['cp']['input_basis'] = basis if value.get('own_BEST_pin_verified') else '입력 계획: ' + basis
    return model
