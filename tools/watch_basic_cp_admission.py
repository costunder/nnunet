"""Read-only admission adapter over the experiment's pinned UI2/native viewer."""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys
from types import FunctionType

ADMISSION_LABELS = {'waiting_GPU': 'GPU 점유 해제 대기',
                    'waiting_RAM': 'RAM 확보 대기',
                    'CUDA_DEBUG': '실제 CUDA B2/B4 검사'}


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
            value = dict(value, stage='scoring', stage_label=ADMISSION_LABELS[phase],
                         waiting_reason=state.get('reason') if state else None)
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
