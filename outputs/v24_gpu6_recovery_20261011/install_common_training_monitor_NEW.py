"""Install a fresh, sealed display-only release; preserve live experiment code.

An exact local Git export is supplied by the calling task as PAYLOAD. There is
no scheduler invocation, GPU/model import, process signal or existing-file write.
"""
import base64
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time

sys.dont_write_bytecode = True
BASE = Path('/var/tmp/aicompetition06_v24_20261010')
AUTO = BASE / 'GPU456_AUTO_NATIVE_20261011_1791730500000000000'
PYTHON = '/home/aicompetition06/.conda/envs/nnunet/bin/python3.10'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def create(filename, raw):
    """Never overwrite even a failed/partial earlier UI install."""
    filename.parent.mkdir(parents=True, exist_ok=True)
    with filename.open('xb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())


def sealed_actors():
    import psutil
    rows=[]
    for gpu in (4,5,6):
        root=AUTO/('gpu'+str(gpu))
        raw=(root/'registration.json').read_bytes()
        record=json.loads(raw); state=json.loads((root/'status.json').read_bytes())
        actor=psutil.Process(record['pid'])
        if (actor.create_time()!=record['create_time'] or actor.cmdline()!=record['command']
                or actor.uids().real!=41833 or actor.status()==psutil.STATUS_ZOMBIE or state['status']=='FAILED'):
            raise ValueError('Existing native controller identity/state differs: GPU'+str(gpu))
        rows.append(dict(GPU=gpu,pid=actor.pid,birth=actor.create_time(),status=state['status'],
            registration_sha256=sha(raw)))
    return rows


def install(payload):
    if os.getuid()!=41833 or socket.gethostname().split('.')[0]!='ece-a6gpu6':
        raise ValueError('Explicit own UID41833/ece-a6gpu6 target required')
    if not re.fullmatch('[0-9a-f]{40}',payload['commit']):
        raise ValueError('Exact local Git commit required')
    name=payload['release_name']
    if not re.fullmatch('UI_COMMON_1_[0-9_]+',name):
        raise ValueError('Explicit fresh UI schema1 release namespace required')
    release=BASE/name
    if release.exists() or (BASE/'view_training.py').exists():
        raise FileExistsError('New UI namespace/launcher already exists; preserve it and inspect outcome')
    old_raw=(AUTO/'viewer_install_request.json').read_bytes()
    old=json.loads(old_raw)
    frozen={str(Path(old['code_root'])/name):sha((Path(old['code_root'])/name).read_bytes())
        for name in old['modules_sha256']}
    if any(digest!=old['modules_sha256'][str(Path(filename).relative_to(old['code_root']))]
            for filename,digest in frozen.items()):
        raise ValueError('Old sealed viewer code differs before installation')
    before=sealed_actors()
    release.mkdir(mode=0o700)
    code=release/'code'; code.mkdir(mode=0o700)
    for relative,row in payload['files'].items():
        location=Path(relative)
        if location.is_absolute() or '..' in location.parts:
            raise ValueError('Invalid Git export path: '+relative)
        raw=base64.b64decode(row['base64'],validate=True)
        if sha(raw)!=row['sha256']:
            raise ValueError('Exact Git export SHA differs: '+relative)
        create(code/location,raw)
    helper_path=code/'outputs/v24_gpu6_recovery_20261011/install_auto_native_viewers_NEW.py'
    spec=importlib.util.spec_from_file_location('_new_common_UI_installer',helper_path)
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    config=dict(old,code_root=str(code),viewer_directory=str(release/'viewers'),
        modules_sha256={name:payload['files'][name]['sha256'] for name in helper.MODULES})
    create(release/'viewer_install_request.json',json.dumps(config,ensure_ascii=False,indent=2).encode())
    tests=payload['test_modules']
    result=subprocess.run([PYTHON,'-B','-m','unittest',*tests,'-q'],cwd=code,
        capture_output=True,text=True,timeout=120)
    create(release/'CPU_tests.log',(result.stdout+'\n'+result.stderr).encode())
    if result.returncode:
        raise RuntimeError('Fresh UI Linux CPU tests failed; existing training untouched: '+result.stderr)
    installed=helper.install(helper.validate(config))
    target=Path(installed['launcher']['path']); expected=installed['launcher']['sha256']
    script=('import hashlib,runpy\nfrom pathlib import Path\nTARGET='+repr(str(target))
        +'\nEXPECTED='+repr(expected)+'\nif hashlib.sha256(Path(TARGET).read_bytes()).hexdigest()!=EXPECTED:'
        +'raise ValueError("Pinned common viewer changed")\nrunpy.run_path(TARGET,run_name="__main__")\n')
    create(BASE/'view_training.py',script.encode())
    views=[]
    for gpu in (4,5,6):
        command=[PYTHON,'-B',str(BASE/'view_training.py'),'--gpu',str(gpu),'--once']
        text=subprocess.run(command,capture_output=True,text=True,timeout=35)
        if text.returncode or len(text.stdout.splitlines())!=16:
            raise RuntimeError('Actual shared sixteen-row viewer failed: '+repr((gpu,text.returncode,text.stdout,text.stderr)))
        metadata=subprocess.run(command+['--json'],capture_output=True,text=True,timeout=35)
        if metadata.returncode:
            raise RuntimeError('Actual metadata verification failed: '+metadata.stderr)
        state=json.loads(metadata.stdout)
        if state.get('GPU')!=gpu or state.get('snapshot_fresh') is not True or state.get('status')=='READ_ERROR':
            raise ValueError('Actual current source snapshot differs: '+repr(state))
        views.append(dict(GPU=gpu,text=text.stdout,snapshot=state))
    if len({tuple(line.split(':',1)[0] for line in row['text'].splitlines()) for row in views})!=1:
        raise ValueError('Actual GPU4/5/6 row labels differ')
    if (AUTO/'viewer_install_request.json').read_bytes()!=old_raw or any(sha(Path(name).read_bytes())!=digest for name,digest in frozen.items()):
        raise ValueError('Old sealed viewer configuration/code changed during install')
    after=sealed_actors()
    if [(row['pid'],row['birth'],row['registration_sha256']) for row in before]!=[(row['pid'],row['birth'],row['registration_sha256']) for row in after]:
        raise ValueError('Existing controller identities changed during display installation')
    receipt=dict(status='COMMON_UI1_ACTUAL_GPU456_INSTALLED_VERIFIED',at=time.time(),commit=payload['commit'],
        release=str(release),command=[PYTHON,'-B',str(BASE/'view_training.py'),'--gpu','4|5|6'],
        CPU_tests_output=result.stdout+result.stderr,views=views,actors_before=before,actors_after=after,
        old_files_written=False,jobs_started=False,signals_sent=False)
    create(release/'installation_verified.json',json.dumps(receipt,ensure_ascii=False,indent=2).encode())
    print(json.dumps(receipt,ensure_ascii=False),flush=True)


if __name__=='__main__':
    raise ValueError('Calling task must supply its exact Git export PAYLOAD; standalone execution is refused')
