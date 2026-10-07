"""Stop one verified existing comparison job; preserve its last checkpoint."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.dont_write_bytecode=True


def parse(argv=None):
    from tools.run_comparison_arm import ARMS
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm',choices=ARMS,required=True)
    p.add_argument('--experiment',type=Path,required=True)
    return p.parse_args(argv)


def run(a):
    import psutil
    from tools.run_comparison_arm import _active_owner, _path
    root=_path(a.experiment)
    request=dict(arm=a.arm,experiment=root,checkpoint=root/a.arm/'checkpoint_latest.pt')
    owner=_active_owner(request)
    if owner is None:
        report=dict(status='NOT_RUNNING',arm=a.arm,experiment=str(root),
                    checkpoint=str(request['checkpoint']),lock_and_outputs_preserved=True)
        print(json.dumps(report,allow_nan=False),flush=True)
        return report
    process=psutil.Process(owner['pid'])
    ancestors={os.getpid(),*(parent.pid for parent in psutil.Process().parents())}
    if process.pid in ancestors:
        raise RuntimeError('Refusing to signal this terminal or an ancestor')
    if process.create_time()!=owner['create_time'] or process.cmdline()!=owner['command']:
        raise RuntimeError('Exact requested job changed; no process was signaled')
    report=dict(status='STOP_REQUESTED',arm=a.arm,pid=process.pid,command=owner['command'],
                create_time=owner['create_time'],experiment=str(root),
                checkpoint=str(request['checkpoint']),checkpoint_exists=request['checkpoint'].is_file(),
                recovery='last atomic checkpoint; uncommitted work is replayed on resume',
                signal='SIGTERM',parent_shell_or_SSH_signaled=False,lock_and_outputs_preserved=True)
    print(json.dumps(report,allow_nan=False),flush=True)
    # Recheck lock and full job identity immediately before this single PID signal.
    if _active_owner(request)!=owner:
        raise RuntimeError('Requested job ownership changed; no process was signaled')
    process.terminate()
    try:
        process.wait(timeout=10)
    except psutil.TimeoutExpired:
        report['status']='TERMINATION_PENDING'
        print('The exact job has not returned; no additional force or group signal was sent.',flush=True)
    else:
        report['status']='STOPPED_LAST_SAVED'
    print(json.dumps(report,allow_nan=False),flush=True)
    return report


if __name__=='__main__':
    result=run(parse())
    if result['status']=='TERMINATION_PENDING':
        raise RuntimeError('Termination pending; do not start a second writer for this arm')
