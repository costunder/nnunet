"""Stop only this user's exact observations child of a recorded v2.22 runner.

Linux only. Refuses other stages, orphan processes, ambiguous matches, shells,
other users, descendants, and PID reuse. Never kills the wrapper or a session.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import psutil


def argument(command,flag):
    if command.count(flag)!=1:raise ValueError(f'Exactly one {flag} required')
    return command[command.index(flag)+1]


def resolved(value,cwd):
    return (Path(cwd)/value).resolve()


def is_runner(command,cwd,project,output):
    try:
        scripts=[x for x in command[1:] if x.endswith('.py')]
        return (len(scripts)==1 and resolved(scripts[0],cwd)==project/'tools/run_v222_server.py'
                and resolved(argument(command,'--output'),cwd)==output)
    except (ValueError,IndexError):return False


def check_phase(output):
    if any(output.glob('03_*.started.json')) or (output/'02_observations.complete.json').exists():
        raise RuntimeError('Observations already completed or a later stage started; refuse to stop')
    if not (output/'02_observations.started.json').is_file():raise RuntimeError('Recorded observations stage required')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--stop',action='store_true',help='Terminate only the verified observations child; status is read-only by default')
    a=p.parse_args()
    if not sys.platform.startswith('linux'):raise RuntimeError('This controller is for the Linux server only')
    output=a.output.resolve();check_phase(output)
    receipt=json.loads((output/'02_observations.started.json').read_text())
    expected=receipt['command'];script=Path(expected[2]).resolve()
    if script.name!='v1_server.py' or expected[3]!='observations':raise RuntimeError('Unexpected recorded command')
    project=script.parents[1]
    if resolved(argument(expected,'--output'),project)!=output/'observations':raise RuntimeError('Observation output mismatch')
    ancestors={x.pid for x in psutil.Process().parents()}|{os.getpid()}
    parents=[]
    for process in psutil.process_iter():
        try:
            if process.pid in ancestors or process.uids().real!=os.getuid():continue
            if is_runner(process.cmdline(),process.cwd(),project,output):parents.append(process)
        except (psutil.NoSuchProcess,psutil.AccessDenied):continue
    if len(parents)!=1:raise RuntimeError(f'Expected one owned runner, found {len(parents)}; no signals sent')
    parent=parents[0];children=parent.children(recursive=True)
    if len(children)!=1:raise RuntimeError('Expected one direct observations child; no signals sent')
    child=children[0]
    def verify():
        check_phase(output)
        if (child.pid in ancestors or child.uids().real!=os.getuid() or child.ppid()!=parent.pid
                or child.cmdline()!=expected or child.children()):
            raise RuntimeError('Child identity/stage changed; no termination permitted')
        if not is_runner(parent.cmdline(),parent.cwd(),project,output):raise RuntimeError('Runner changed')
    verify();created=child.create_time()
    evidence=dict(pid=child.pid,created_at=created,parent_pid=parent.pid,command=expected,
        reason='Replace redundant 128-case preparation benchmark; optimizer has not started',
        action='terminate_only_verified_observations_child' if a.stop else 'read_only_status')
    print(json.dumps(evidence,indent=2),flush=True)
    if not a.stop:return
    # Briefly freeze this exact child to prevent it completing and advancing the
    # wrapper between identity checks. Always unfreeze, including refusal paths.
    child.suspend()
    try:
        verify()
        if child.create_time()!=created:raise RuntimeError('PID reused')
        child.terminate()
    finally:
        try:child.resume()
        except psutil.NoSuchProcess:
            print('Verified observations child already ended.',flush=True)
    child.wait(timeout=30)
    # Wrapper handles returncode -15 itself and records failure; no parent signal.
    deadline=time.monotonic()+30
    while time.monotonic()<deadline:
        if (output/'02_observations.failed.json').is_file():break
        time.sleep(.2)
    if not (output/'02_observations.failed.json').is_file():
        raise RuntimeError('Child stopped but wrapper failure receipt is unconfirmed; do not restart yet')
    parent.wait(timeout=30)
    evidence.update(child_stopped=True,wrapper_ended_after_child_failure=True,files_preserved=True)
    with (output/f'observations_stop_{time.time_ns()}.json').open('x') as f:json.dump(evidence,f,indent=2)
    print(json.dumps(evidence,indent=2),flush=True)


if __name__=='__main__':main()
