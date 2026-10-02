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
        m = initialize(a.experiment, a.medical_root, a.split,
                       local_sampling=a.local_sampling, sampling_profile=profile,
                       reference_experiment=a.reference_experiment, record_epochs=a.record_epochs)
        result = dict(experiment=str(Path(a.experiment).resolve()), stages=list(m['stages']),
                      train_cases=len(m['split']['train']), validation_cases=len(m['split']['val']),
                      excluded_outer_cases=len(m['split']['outer_validation_excluded']),
                      shared_cache=True, preparation_root=str(preparation_root(a.experiment, m)),
                      execution_reference=str(execution_reference(a.experiment, m)),
                      local_sampling=m.get('sampling_contract', {'mode': 'native', 'legacy': True}),
                      epoch_recording=m.get('epoch_recording'),
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
