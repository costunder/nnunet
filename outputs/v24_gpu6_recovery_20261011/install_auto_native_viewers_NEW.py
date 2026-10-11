"""Install only fresh read-only GPU4/5/6 viewers; never contact a server.

Run this standalone helper on the target host with --config FILE --config-sha
SHA256 [--check]. Root supplies a new immutable code_root, canonical python,
fresh viewer_directory, modules_sha256 for the four MODULES below, and arms
{'4': {arm_root, scoring_root, native_root}, '5': {...},
 '6': {scoring_root, native_root[, gnn_root]}}. GPU6 source is read from its
actual scorer request/own BEST pin. Missing future scoring/native records on
GPU4/5 are legitimate registration waits, never an assertion of launched work.
No model, dataset, running job, old viewer or source checkout is modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat


FORMAT = 'v24_auto_native_pipeline_viewer_install_request_v1'
MODULES = ('tools/watch_v24_gpu_pipeline.py', 'tools/watch_v24_direct_gnn_training.py',
    'tools/watch_v24_matched_cp_scoring.py', 'tools/watch_v24_training.py',
    'tools/training_monitor_adapter.py', 'tools/training_monitor_ui.py', 'tools/__init__.py')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def digest(value):
    return isinstance(value, str) and len(value) == 64 and not set(value) - set('0123456789abcdef')


def identity(filename):
    info = filename.stat()
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def path(value, *, directory=False, missing=False):
    if not isinstance(value, str) or not Path(value).is_absolute() or '..' in Path(value).parts:
        raise ValueError('Explicit absolute path required: ' + repr(value))
    result = Path(value)
    for item in (result, *result.parents):
        if item.is_symlink():
            raise ValueError('Symlink input/output refused: ' + str(item))
    if missing and not result.exists():
        return result
    info = result.stat()
    if hasattr(os, 'getuid') and info.st_uid != os.getuid():
        raise ValueError('Another UID owns viewer input/output: ' + str(result))
    if not (stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode)):
        raise ValueError('Owned regular input/output required: ' + str(result))
    return result


def record(filename):
    filename = path(str(filename))
    try:
        value = json.loads(filename.read_bytes())
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError(str(filename) + ': JSON incomplete or malformed: ' + str(error)) from error
    if not isinstance(value, dict):
        raise ValueError(str(filename) + ': JSON object required')
    return value


def validate(config):
    if (config.get('format') != FORMAT or not isinstance(config.get('arms'), dict)
            or set(config['arms']) != {'4', '5', '6'}):
        raise ValueError('Explicit new GPU4/5/6 viewer install contract required')
    code = path(config.get('code_root'), directory=True)
    python = path(config.get('python'))
    target = path(config.get('viewer_directory'), directory=True, missing=True)
    path(str(target.parent), directory=True)
    if target.exists():
        raise FileExistsError('Fresh viewer_directory already exists; preserve it: ' + str(target))
    hashes = config.get('modules_sha256')
    if not isinstance(hashes, dict) or set(hashes) != set(MODULES):
        raise ValueError('All shared UI and immutable viewer module SHA256 bindings required')
    for name in MODULES:
        module = path(str(code / name))
        before = identity(module)
        if not digest(hashes[name]) or sha(module.read_bytes()) != hashes[name] or identity(module) != before:
            raise ValueError('Immutable viewer module SHA/stat differs: ' + str(module))
    prepared, protected = {}, [code]
    for number in ('4', '5', '6'):
        arm = config['arms'][number]
        if not isinstance(arm, dict):
            raise ValueError('Explicit arm mapping required: GPU' + number)
        allowed = {'arm_root', 'scoring_root', 'native_root'} if number != '6' else {'gnn_root', 'scoring_root', 'native_root'}
        if set(arm) - allowed or not {'scoring_root', 'native_root'} <= set(arm):
            raise ValueError('Explicit arm/source/scoring/native paths required: GPU' + number)
        score = path(arm['scoring_root'], directory=True, missing=number != '6')
        native = path(arm['native_root'], directory=True, missing=True)
        scored = record(score / 'request.json') if (score / 'request.json').exists() else None
        if scored is not None and (scored.get('format') != 'v24_full642_depth1_current_BEST_scoring_request_v1'
                or scored.get('gpu') != int(number) or scored.get('output') != str(score / 'bank')):
            raise ValueError(str(score / 'request.json') + ': actual own-arm scorer binding differs')
        if number == '6':
            if scored is None:
                raise ValueError('GPU6 requires its actual sealed completed scorer request')
            pin = record(path(scored.get('pin_path')))
            if (pin.get('format') != 'v24_completed_single_arm_GT_blind_best_v1'
                    or pin.get('physical_GPU') != 6 or pin.get('full_training') is not True
                    or pin.get('debug') is not False or pin.get('completed_epochs') != 40
                    or pin.get('optimizer_updates') != 680):
                raise ValueError('GPU6 requires its actual own full40/680 BEST pin')
            source = path(pin.get('source_output'), directory=True)
            source_code = str(path(pin.get('source_code'), directory=True))
            if arm.get('gnn_root', str(source)) != str(source):
                raise ValueError('GPU6 configured source differs from actual scorer own BEST pin')
            if any(key in scored for key in ('source_output', 'source_code')) and (
                    scored.get('source_output') != str(source) or scored.get('source_code') != source_code):
                raise ValueError('GPU6 scorer source differs from actual own BEST pin')
            fixed = ['--gnn-root', str(source), '--gpu', '6']
        else:
            direct_root = path(arm.get('arm_root'), directory=True)
            direct = record(direct_root / 'direct_training_request.json')
            source = path(direct.get('output'), directory=True)
            source_code = str(path(direct.get('source_code'), directory=True))
            if (direct.get('format') != 'v24_existing_direct_GPU_exact_continuation_request_v1'
                    or direct.get('GPU_arm') != int(number) or source != direct_root / 'gnn'):
                raise ValueError(str(direct_root) + ': actual current direct GNN source differs')
            if scored is not None and (scored.get('source_output') != str(source) or scored.get('source_code') != source_code):
                raise ValueError(str(score) + ': scorer differs from current resumed GNN source')
            fixed = ['--arm-root', str(direct_root)]
            protected.append(direct_root)
        if (native / 'request.json').exists():
            declared = record(native / 'request.json')
            if (declared.get('format') != 'v24_scored_bank_to_native_chain_request_v1'
                    or declared.get('chain_root') != str(native) or declared.get('GPU_arm') != int(number)
                    or declared.get('source_output') != str(source) or declared.get('source_code') != source_code
                    or declared.get('scoring_root') != str(score)):
                raise ValueError(str(native) + ': actual registered native controller source differs')
        fixed.extend(['--renderer', str(code / MODULES[3]), '--renderer-sha', hashes[MODULES[3]],
            '--scoring-root', str(score), '--native-root', str(native), '--experiment-version', 'v2.4'])
        prepared[number] = dict(args=fixed, current_gnn=str(source), source_code=source_code,
            scoring_root=str(score), native_root=str(native))
        protected.extend([source, Path(source_code), score, native])
    for original in protected:
        if target == original or target.is_relative_to(original) or original.is_relative_to(target):
            raise ValueError('Fresh viewer output overlaps preserved input/code/results: ' + str(original))
    if (len({row['native_root'] for row in prepared.values()}) != 3
            or len({Path(row['native_root']).parent for row in prepared.values()}) != 1
            or len({row['scoring_root'] for row in prepared.values()}) != 3
            or len({row['current_gnn'] for row in prepared.values()}) != 3):
        raise ValueError('Exactly three independent sources/scorers and native arms in one fresh group required')
    return dict(code_root=str(code), python=str(python), viewer_directory=str(target),
        modules_sha256=hashes, arms=prepared)


def wrapper(config, arm):
    payload = dict(code_root=config['code_root'], modules_sha256=config['modules_sha256'], args=arm['args'])
    script = '''"""New read-only current GNN / CP / native viewer. Ctrl+C stops only display."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import sys

BOUND = json.loads(PAYLOAD)
code = Path(BOUND['code_root'])
for name, expected in BOUND['modules_sha256'].items():
    filename = code / name
    for part in (filename, *filename.parents):
        if part.is_symlink():
            raise ValueError('Viewer source symlink refused: ' + str(part))
    info = filename.stat()
    before = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
    data = filename.read_bytes()
    current = filename.stat()
    after = (current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns, current.st_ctime_ns)
    if (not stat.S_ISREG(info.st_mode) or hasattr(os, 'getuid') and info.st_uid != os.getuid()
            or hashlib.sha256(data).hexdigest() != expected or after != before):
        raise ValueError('Pinned viewer code SHA/stat/UID differs: ' + str(filename))
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--once', action='store_true')
parser.add_argument('--json', action='store_true')
parser.add_argument('--no-clear', action='store_true')
parser.add_argument('--interval', type=float, default=2.)
options = parser.parse_args()
args = BOUND['args'] + ['--interval', str(options.interval)]
for flag in ('once', 'json', 'no_clear'):
    if getattr(options, flag):
        args.append('--' + flag.replace('_', '-'))
sys.dont_write_bytecode = True
sys.path.insert(0, str(code))
from tools.watch_v24_gpu_pipeline import main
main(args)
'''
    return script.replace('PAYLOAD', repr(json.dumps(payload, ensure_ascii=False)))


def install(prepared):
    """Exclusive creation only; a partial failure is preserved for inspection."""
    target = Path(prepared['viewer_directory'])
    target.mkdir(mode=0o700)
    created = []
    try:
        for number, arm in prepared['arms'].items():
            filename = target / ('view_gpu' + number + '_auto.py')
            data = wrapper(prepared, arm)
            compile(data, str(filename), 'exec')
            with filename.open('x', encoding='utf8', newline='\n') as stream:
                stream.write(data); stream.flush(); os.fsync(stream.fileno())
            created.append(dict(GPU=int(number), path=str(filename), sha256=sha(filename.read_bytes()),
                command=[prepared['python'], '-B', str(filename)], current_gnn=arm['current_gnn'],
                scoring_root=arm['scoring_root'], native_root=arm['native_root']))
        launcher_path = target / 'view_training.py'
        script = wrapper(prepared, prepared['arms']['4'])
        payload = dict(code_root=prepared['code_root'], modules_sha256=prepared['modules_sha256'],
            arms={number: dict(args=arm['args']) for number, arm in prepared['arms'].items()})
        # One stable user command, with an explicit arm rather than inference.
        first = script.index('BOUND = json.loads(')
        end = script.index('\ncode = ', first)
        script = script[:first] + 'BOUND = json.loads(' + repr(json.dumps(payload, ensure_ascii=False)) + ')' + script[end:]
        script = script.replace("parser.add_argument('--once', action='store_true')",
            "parser.add_argument('--gpu', type=int, choices=(4, 5, 6), required=True)\nparser.add_argument('--once', action='store_true')")
        script = script.replace("args = BOUND['args'] +", "args = BOUND['arms'][str(options.gpu)]['args'] +")
        compile(script, str(launcher_path), 'exec')
        with launcher_path.open('x', encoding='utf8', newline='\n') as stream:
            stream.write(script); stream.flush(); os.fsync(stream.fileno())
    except (OSError, ValueError, SyntaxError) as error:
        raise RuntimeError('Fresh viewer installation incomplete; no files overwritten/deleted. '
            'Preserved directory: ' + str(target) + '; created=' + repr(created) + '; cause=' + str(error)) from error
    return dict(format='v24_auto_native_pipeline_viewers_installed_v1', created=created,
        launcher=dict(path=str(launcher_path), sha256=sha(launcher_path.read_bytes())),
        old_files_written=False, jobs_started=False, signals_sent=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--config-sha', required=True)
    parser.add_argument('--check', action='store_true', help='Validate only, create no files')
    args = parser.parse_args(argv)
    config_path = path(args.config)
    raw = config_path.read_bytes()
    if not digest(args.config_sha) or sha(raw) != args.config_sha:
        raise ValueError(str(config_path) + ': exact install config raw SHA256 differs')
    config = json.loads(raw)
    if not isinstance(config, dict):
        raise ValueError(str(config_path) + ': install config object required')
    prepared = validate(config)
    result = dict(validated=True, files_created=False, prepared=prepared) if args.check else install(prepared)
    result['config_sha256'] = args.config_sha
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
