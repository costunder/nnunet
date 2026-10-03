"""Explicit v1.x suite preparation and single-stage foreground execution."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import argparse
import json
from hiercp_v1x.experiment import (initialize, load_suite, command_plan, execute_stage,
                                 compare_files, preparation_root, execution_reference)
from hiercp_v1x.contracts import STAGES, LOCAL_SAMPLING_MODES, LOCAL_SAMPLING_ROLES


def admission_from_probe(max_voxels, probe_path):
    """Bind completed actual failed-source ROI evidence, without choosing a cap."""
    if max_voxels is None and probe_path is None:
        return None
    if max_voxels is None or probe_path is None:
        raise ValueError('--roi-max-voxels and --roi-budget-probe must be supplied together')
    from hiercp_v1x.contracts import make_preparation_admission, V1_ARCHIVE_SHA256
    from hiercp_v1x.experiment import read, digest
    report = read(probe_path)
    rows = report.get('measurements')
    if (report.get('format') != 'v1_roi_budget_probe_v1'
            or report.get('scope') != 'debug_geometry_only'
            or report.get('archived_source_sha256') != V1_ARCHIVE_SHA256
            or report.get('original_roi_max_voxels') != 8_000_000
            or report.get('candidate_roi_max_voxels') != max_voxels
            or report.get('completed') is not True
            or report.get('originals_preserved') is not True
            or report.get('training_started') is not False
            or report.get('cache_publication_created') is not False
            or not isinstance(rows, list) or not rows
            or len(rows) != report.get('failed_sample_requests')
            or any(not _valid_roi_measurement(row, max_voxels) for row in rows)):
        raise ValueError('ROI admission requires a completed, geometry-preserving failed-ROI cost probe; it is not full-cache admission')
    return make_preparation_admission(max_voxels, digest(probe_path))


def _valid_roi_measurement(row, max_voxels):
    if not isinstance(row, dict):
        return False
    if row.get('status') != 'PASS' or row.get('original_guard_reproduced') is not True:
        return False
    if row.get('replay_contract') != 'original_sample_first_roi_failure_v1':
        return row.get('geometry_equal') is True  # Original source-only evidence.
    import math
    requested, initial, effective = (row.get(k) for k in
                                     ('requested_shape', 'payload_initial_shape', 'effective_shape'))
    if any(not isinstance(s, list) or len(s) != 3 or
           any(type(v) is not int or v <= 0 for v in s) for s in (requested, initial, effective)):
        return False
    phase = row.get('original_failure_phase')
    replayed = row.get('replayed_geometry')
    if (row.get('geometry_preserved') is not True or row.get('full_mask_preserved') is not True
            or phase not in ('source', 'target')
            or row.get('failed_operation') not in ('build_patch_payload', 'transform_footprint_physical')
            or not isinstance(replayed, dict) or replayed.get('requested_shape') != requested
            or row.get('original_requested_shape') != requested
            or any(a < b for a, b in zip(initial, requested))
            or any(a < b for a, b in zip(effective, initial))
            or math.prod(effective) != row.get('roi_voxels') or math.prod(effective) > max_voxels
            or type(row.get('full_mask_voxels')) is not int or row['full_mask_voxels'] <= 0):
        return False
    return phase != 'target' or (isinstance(row.get('target_spec'), dict)
                                and row['target_spec'].get('center') == row.get('actual_center')
                                and row.get('erase_target') is True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest='action', required=True)
    init = commands.add_parser('init')
    init.add_argument('--experiment', required=True)
    init.add_argument('--medical-root', required=True)
    init.add_argument('--split', default=str(ROOT/'config/split_cp80_fold0.json'))
    init.add_argument('--local-sampling', choices=LOCAL_SAMPLING_MODES,
                      help='Explicit matched graph comparison; omitted retains legacy cumulative suite')
    init.add_argument('--role-seeds', type=int, nargs=6,
                      metavar=('TS', 'TI', 'SC', 'SL', 'TC', 'TL'),
                      help='Mandatory strict_nested seed budgets: tumor surface/interior, source context/liver surface, target context/liver surface; no final N/E cap')
    init.add_argument('--reference-experiment',
                      help='Matched explicit-native experiment owning shared canonical preparation and measured batch/worker lock')
    init.add_argument('--record-epochs', action='store_true',
                      help='Bind identical observational epoch timing/logging to both matched arms')
    init.add_argument('--roi-max-voxels', type=int,
                      help='Explicit measured resource-only ROI allocation guard increase; no automatic selection')
    init.add_argument('--roi-budget-probe',
                      help='Completed probe_v1_roi_budget JSON for exactly this proposed guard; no quality promotion')
    init.add_argument('--recover-preparation-from',
                      help='Disjoint failed native suite whose verified successful cache bytes are retained in a fresh output')
    for name in ('plan', 'run'):
        q = commands.add_parser(name)
        q.add_argument('--experiment', required=True)
        q.add_argument('--stage', choices=tuple(STAGES), required=True)
        q.add_argument('--target', choices=('prepare', 'train', 'generate'), required=True)
        if name == 'run': q.add_argument('--gpu', type=int)
    status = commands.add_parser('status')
    status.add_argument('--experiment', required=True)
    compare = commands.add_parser('compare')
    compare.add_argument('--baseline', required=True)
    compare.add_argument('--candidate', required=True)
    compare.add_argument('--predecessor')
    collect = commands.add_parser('collect')
    collect.add_argument('--experiment', required=True)
    collect.add_argument('--stage', choices=tuple(STAGES), required=True)
    collect.add_argument('--output')
    a = p.parse_args()
    if a.action == 'init':
        profile = dict(zip(LOCAL_SAMPLING_ROLES, a.role_seeds)) if a.role_seeds is not None else None
        admission = admission_from_probe(a.roi_max_voxels, a.roi_budget_probe)
        if a.recover_preparation_from is not None and admission is not None:
            from hiercp_v1x.experiment import read, digest
            probe = read(a.roi_budget_probe)
            old = Path(a.recover_preparation_from).resolve()
            if (Path(probe.get('experiment', '')).resolve() != old
                    or probe.get('failed_manifest_sha256') != digest(old/'shared/cache/manifest.csv')):
                raise ValueError('ROI probe does not belong to this exact failed preparation')
        m = initialize(a.experiment, a.medical_root, a.split,
                       local_sampling=a.local_sampling, sampling_profile=profile,
                       reference_experiment=a.reference_experiment, record_epochs=a.record_epochs,
                       preparation_admission=admission, recover_preparation_from=a.recover_preparation_from)
        result = dict(experiment=str(Path(a.experiment).resolve()), stages=list(m['stages']),
                      train_cases=len(m['split']['train']), validation_cases=len(m['split']['val']),
                      excluded_outer_cases=len(m['split']['outer_validation_excluded']),
                      shared_cache=True, preparation_root=str(preparation_root(a.experiment, m)),
                      execution_reference=str(execution_reference(a.experiment, m)),
                      local_sampling=m.get('sampling_contract', {'mode': 'native', 'legacy': True}),
                      epoch_recording=m.get('epoch_recording'),
                      preparation_admission=m.get('preparation_admission'),
                      preparation_recovery=m.get('preparation_recovery'),
                      training_started=False, graph_quality_passed=False)
    elif a.action == 'plan':
        cwd, commands = command_plan(a.experiment, a.stage, a.target)
        m = load_suite(a.experiment)
        result = dict(cwd=str(cwd), commands=commands,
                      local_sampling=m.get('sampling_contract', {'mode': 'native', 'legacy': True}),
                      training_started=False)
    elif a.action == 'run':
        result = execute_stage(a.experiment, a.stage, a.target, gpu=a.gpu)
    elif a.action == 'status':
        m = load_suite(a.experiment)
        result = dict(stages={s:info['spec'] for s,info in m['stages'].items()}, target_contract=m['target_contract'],
                      checkpoint_paths={s:str(Path(a.experiment).resolve()/'results'/s/'checkpoint_best.pt') for s in STAGES},
                      manifest=str(Path(a.experiment).resolve()/'manifest.json'),
                      local_sampling=m.get('sampling_contract', {'mode': 'native', 'legacy': True}),
                      preparation_root=str(preparation_root(a.experiment, m)),
                      execution_reference=str(execution_reference(a.experiment, m)),
                      sampling_contracts={s:c['contract_sha256'] for s,c in m.get('sampling_contracts', {}).items()},
                      historical_accuracy_reproduced=False, nnunet_started=False)
    elif a.action == 'collect':
        from hiercp_v1x.results import collect_result
        result = collect_result(a.experiment, a.stage, a.output)
    else:
        result = compare_files(a.baseline, a.candidate, a.predecessor)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
