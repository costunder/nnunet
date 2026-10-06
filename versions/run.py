"""Short version entry points; delegate to the original, bound runtime."""
from __future__ import annotations

import json
import os
from pathlib import Path
import runpy
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
INDEX = Path(__file__).with_name('index.json')


def entries():
    return json.loads(INDEX.read_text(encoding='utf8'))['entries']


def resolve(key):
    matched = [row for row in entries() if row['id'] == key]
    if len(matched) != 1:
        raise ValueError(f'Unknown version/variant: {key}. Run python versions/run.py list')
    return matched[0]


def arguments(entry, supplied):
    """Fixed arm/config choices are explicit; other arguments stay untouched."""
    supplied = list(supplied)
    for flag, value in entry.get('fixed_options', {}).items():
        found = [i for i, arg in enumerate(supplied) if arg == flag or arg.startswith(flag + '=')]
        if len(found) > 1:
            raise ValueError(f'Duplicate {flag}')
        if found:
            i = found[0]
            if supplied[i] == flag:
                if i + 1 >= len(supplied):
                    raise ValueError(f'Missing value for {flag}')
                actual = supplied[i + 1]
            else:
                actual = supplied[i].split('=', 1)[1]
            if actual != value:
                raise ValueError(f'{entry["id"]} requires {flag} {value}; select the other entry explicitly')
        else:
            supplied[0:0] = [flag, value]
    return supplied


def launch(key, supplied=None):
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf8')
    entry = resolve(key)
    supplied = list(sys.argv[1:] if supplied is None else supplied)
    if supplied == ['--describe']:
        print(json.dumps(entry, ensure_ascii=False, indent=2))
        return
    runner = entry.get('runner')
    if not runner:
        raise ValueError(f'{key}: {entry["status"]}. See versions/{entry["folder"]}/README.md')
    target = (ROOT / runner).resolve(strict=True)
    if not target.is_relative_to(ROOT):
        raise ValueError('Runner is outside this checkout')
    forwarded = arguments(entry, supplied)
    if entry.get('kind', 'python') == 'shell':
        env = os.environ.copy()
        for name, value in entry.get('fixed_env', {}).items():
            if name in env and env[name] != value:
                raise ValueError(f'{key} requires {name}={value}')
            env[name] = value
        # One existing foreground launcher; no new experiment defaults here.
        subprocess.run(['bash', str(target), *forwarded], cwd=ROOT, env=env, check=True)
        return
    previous_argv, previous_cwd, previous_path = sys.argv, Path.cwd(), list(sys.path)
    sys.argv = [str(target), *forwarded]
    sys.path.insert(0, str(ROOT))
    try:
        os.chdir(ROOT)
        runpy.run_path(str(target), run_name='__main__')
    finally:
        sys.argv = previous_argv
        os.chdir(previous_cwd)
        sys.path[:] = previous_path


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf8')
    args = sys.argv[1:]
    if not args or args in (['list'], ['--help'], ['-h']):
        print('python versions/run.py <entry> [original arguments]\n')
        for row in entries():
            print(f'{row["id"]:24s} {row["title"]} [{row["status"]}]')
        print('\nUse <entry> --describe for source/config/status. Relative paths use the repository root.')
        return
    launch(args[0], args[1:])


if __name__ == '__main__':
    main()
