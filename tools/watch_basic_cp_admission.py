"""Read-only admission adapter over the experiment's pinned UI2/native viewer."""
from __future__ import annotations

import argparse
import ast
import importlib
import json
import math
from pathlib import Path
import sys
from types import FunctionType

ADMISSION_LABELS = {'waiting_GPU': 'GPU 점유 해제 대기',
                    'waiting_RAM': 'RAM 확보 대기',
                    'CUDA_DEBUG': '실제 CUDA B2/B4 검사'}


def reason_mapping(reason):
    """Parse the controller's literal observation without executing its text."""
    if isinstance(reason, dict):
        return reason
    if not isinstance(reason, str):
        return None
    try:
        value = ast.literal_eval(reason)
    except (ValueError, SyntaxError):
        return None
    return value if isinstance(value, dict) else None


def concise_reason(phase, state, gpu):
    """Use only actual observation fields; retain unrecognized reasons verbatim."""
    reason = state.get('reason')
    if phase == 'waiting_GPU':
        observation = reason_mapping(reason)
        actors = observation.get('occupied_GPU_compute_actors') if observation else None
        if (isinstance(actors, list) and actors
                and all(isinstance(actor, dict) and type(actor.get('pid')) is int
                        and actor['pid'] > 0 for actor in actors)):
            pids = ', '.join(str(pid) for pid in dict.fromkeys(actor['pid'] for actor in actors))
            return 'GPU' + str(gpu) + ' 점유 중: PID ' + pids + ' | 점유 해제 대기'
        free = observation.get('GPU_free_MiB') if observation else None
        if actors == [] and type(free) in (int, float) and math.isfinite(free) and free >= 0:
            return 'GPU' + str(gpu) + ' VRAM 여유 ' + format(free / 1024, '.2f') + 'GiB | 필요 40.00GiB'
    if phase == 'waiting_RAM':
        observation = state.get('RAM_admission')
        if not isinstance(observation, dict):
            observation = reason_mapping(reason)
        if not isinstance(observation, dict) and isinstance(reason, str):
            # Typed RAM errors include their exact JSON observation after a prefix.
            start = reason.find('{')
            if start >= 0:
                observation = reason_mapping(reason[start:])
        keys = ('host_available_RAM_bytes', 'group_process_tree_RSS_bytes',
                'required_group_and_cache_capacity_bytes')
        if (isinstance(observation, dict)
                and all(type(observation.get(key)) in (int, float)
                        and math.isfinite(observation[key]) and observation[key] >= 0 for key in keys)):
            available, rss, required = (observation[key] / 2**30 for key in keys)
            return ('RAM 확보 대기: 여유 ' + format(available, '.2f')
                    + 'GiB + 실험 RSS ' + format(rss, '.2f')
                    + 'GiB = ' + format(available + rss, '.2f')
                    + 'GiB | 필요 ' + format(required, '.2f') + 'GiB')
    return reason


def load_pinned_viewers(code_root):
    """Load only the explicitly bound live export, without rewriting its files."""
    code_root = Path(code_root).resolve(strict=True)
    for filename in ('watch_basic_cp_expanded.py', 'watch_basic_cp_ablation.py'):
        if not (code_root / 'tools' / filename).is_file():
            raise ValueError('Pinned viewer export is missing: ' + filename)
    sys.path.insert(0, str(code_root))
    expanded = importlib.import_module('tools.watch_basic_cp_expanded')
    viewer = importlib.import_module('tools.watch_basic_cp_ablation')
    for module in (expanded, viewer):
        if Path(module.__file__).resolve(strict=True).parents[1] != code_root:
            raise ValueError('Viewer module belongs to a different code export: ' + module.__name__)
    return expanded, viewer


def build_entry(expanded, viewer):
    """Use the unchanged native formatter and existing eight-row task layout."""
    def snapshot(root):
        value = expanded.expanded_snapshot(root)
        phase = value['controller_stage']
        if phase in ADMISSION_LABELS:
            state = viewer.read(Path(root) / 'status.json')
            state = state or {}
            value = dict(value, stage='scoring', stage_label=ADMISSION_LABELS[phase],
                         waiting_reason=concise_reason(phase, state, value['physical_GPU']),
                         waiting_reason_detail=state.get('reason'))
        return value

    def preparation_model(value):
        model = expanded.preparation_model(value)
        if value['controller_stage'] in ADMISSION_LABELS:
            model['waiting_reason'] = value.get('waiting_reason')
            model['controller']['stage'] = value['controller_stage']
            model['native'] = dict(model.get('native') or {},
                                   status=value['controller_status'],
                                   stage=value['controller_stage'])
        return model

    scope = dict(viewer.__dict__, snapshot=snapshot, preparation_model=preparation_model)
    return FunctionType(viewer.main.__code__, scope, viewer.main.__name__,
                        viewer.main.__defaults__, viewer.main.__closure__)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--code-root', type=Path, required=True)
    parser.add_argument('--gpu', type=int, choices=range(7), required=True)
    args, rest = parser.parse_known_args(argv)
    manifest = json.loads(args.manifest.read_bytes())
    code_root = args.code_root.resolve(strict=True)
    if code_root != Path(manifest['code']).resolve(strict=True):
        raise ValueError('Explicit code root must equal the manifest pinned export')
    root = Path(manifest['registration'][str(args.gpu)]['root'])
    expanded, viewer = load_pinned_viewers(code_root)
    return build_entry(expanded, viewer)(['--root', str(root), *rest])


if __name__ == '__main__':
    main()
