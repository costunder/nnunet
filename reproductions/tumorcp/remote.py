"""Explicit Git deployment and independent PBS jobs for the KiTS19 reproduction.

Connection credentials and hosts belong to the caller's existing transport
module, never this source tree. This tool does not change or cancel old jobs.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path, PurePosixPath
import re
import textwrap

OFFICIAL = '0fd9da71712f0b1467f32206e96823cc880256bb'
SOURCE_REPO = 'https://github.com/costunder/nnunet.git'
BRANCH = 'codex/tumorcp-kits19'


def remote_root(value):
    root = PurePosixPath(value)
    if (not root.is_absolute() or '..' in root.parts or len(root.parts) < 5
            or root.name != 'tumorcp' or root.parent.name != 'reproductions'
            or any(c in value for c in ('\n', '\r', ',', '='))):
        raise ValueError('An explicit absolute .../reproductions/tumorcp root is required')
    return str(root)


def transport(path):
    path = Path(path).resolve(strict=True)
    spec = importlib.util.spec_from_file_location('owned_existing_transport', path)
    module = importlib.util.module_from_spec(spec)
    # Existing transport modules may import neighbouring, already configured
    # helpers. Their private contents are never read or printed here.
    import sys
    sys.path.insert(0, str(path.parent))
    spec.loader.exec_module(module)
    if not callable(getattr(module, 'run', None)):
        raise ValueError('Existing transport must expose run(script, timeout)')
    return module


def header(root):
    return 'ROOT = ' + repr(root) + '\n' + textwrap.dedent('''
        import hashlib,json,os,shutil,subprocess,sys,time,uuid
        from pathlib import Path
        root=Path(ROOT)
        if os.getuid()==0:raise RuntimeError('Never deploy as an administrator')
        def emit(value):print(json.dumps(value),flush=True)
        def read(path):return json.loads(path.read_text())
        def write_new(path,value):
            with path.open('x',encoding='utf8') as handle:
                json.dump(value,handle,indent=2);handle.flush();os.fsync(handle.fileno())
        def run(argv):
            result=subprocess.run(argv,capture_output=True,text=True,timeout=120)
            if result.returncode:raise RuntimeError(json.dumps({'command':argv,'returncode':result.returncode,'stdout':result.stdout[-3000:],'stderr':result.stderr[-3000:]}))
            return result.stdout.strip()
    ''')


def inspect_script(root):
    return header(root) + textwrap.dedent('''
        ancestor=root
        while not ancestor.exists():ancestor=ancestor.parent
        out={'read_only':True,'root':str(root),'exists':root.exists(),'disk':shutil.disk_usage(ancestor)._asdict()}
        if root.exists():
            out['entries']=[p.name for p in root.iterdir()]
            if (root/'bootstrap.json').is_file():out['bootstrap']=read(root/'bootstrap.json')
        emit(out)
    ''')


def bootstrap_script(root, commit):
    if not re.fullmatch('[0-9a-f]{40}', commit):
        raise ValueError('A full, published Git commit SHA is required')
    return header(root) + f'COMMIT={commit!r}\nOFFICIAL={OFFICIAL!r}\nREPO={SOURCE_REPO!r}\nBRANCH={BRANCH!r}\n' + textwrap.dedent(r'''
        marker=root/'bootstrap.json'
        if marker.is_file():
            saved=read(marker)
            if saved['commit']!=COMMIT or saved['upstream_commit']!=OFFICIAL or saved['root']!=str(root):
                raise ValueError('Existing reproduction namespace has different source; preserved')
            for directory,sha in [(root/'code',COMMIT),(root/'upstream',OFFICIAL)]:
                if run(['git','-C',str(directory),'rev-parse','HEAD'])!=sha:
                    raise ValueError('Existing pinned Git checkout differs; preserved')
            emit({'bootstrap':'VERIFIED_EXISTING','receipt':saved})
        else:
            intent=root/'bootstrap_intent.json'
            if root.exists():
                if not intent.is_file():raise FileExistsError('Existing unclaimed namespace preserved: '+str(root))
                claim=read(intent)
                if any(claim.get(k)!=v for k,v in [('commit',COMMIT),('upstream_commit',OFFICIAL),('root',str(root))]):
                    raise ValueError('Partial deployment has different source; preserved')
                allowed={'bootstrap_intent.json','code','upstream','queue','downloads','data'}
                if set(p.name for p in root.iterdir())-allowed:
                    raise ValueError('Unknown entries in partial deployment; inspect before recovery')
            else:
                root.mkdir(parents=True,exist_ok=False)
                write_new(intent,{'commit':COMMIT,'upstream_commit':OFFICIAL,'root':str(root),'created_unix':time.time()})
            # Actual small allocation checks current per-user writing, not just
            # filesystem-wide df. It does not reserve future dataset capacity.
            probe=root/('.write_probe_'+uuid.uuid4().hex)
            try:
                with probe.open('xb') as handle:
                    block=bytes(2**20)
                    for _ in range(128):handle.write(block)
                    handle.flush();os.fsync(handle.fileno())
                probe_bytes=probe.stat().st_size
            finally:
                if probe.is_file():probe.unlink()
            def checkout(directory,url,sha,sparse=False):
                if not directory.exists():directory.mkdir()
                if not (directory/'.git').exists():
                    if list(directory.iterdir()):raise ValueError('Unknown checkout entries preserved')
                    run(['git','-C',str(directory),'init'])
                origins=run(['git','-C',str(directory),'remote']).splitlines()
                if not origins:run(['git','-C',str(directory),'remote','add','origin',url])
                elif origins!=['origin'] or run(['git','-C',str(directory),'config','--get','remote.origin.url'])!=url:
                    raise ValueError('Partial checkout remote differs')
                current=subprocess.run(['git','-C',str(directory),'rev-parse','--verify','HEAD'],capture_output=True,text=True,timeout=10)
                if current.returncode==0:
                    if current.stdout.strip()!=sha:raise ValueError('Existing checkout HEAD differs; preserved')
                    if run(['git','-C',str(directory),'status','--porcelain']):raise ValueError('Existing source has edits; preserved')
                    return
                if sparse:
                    run(['git','-C',str(directory),'config','core.sparseCheckout','true'])
                    (directory/'.git/info/sparse-checkout').write_text('/reproductions/tumorcp/\n')
                run(['git','-C',str(directory),'fetch','--depth','1','origin',sha])
                run(['git','-C',str(directory),'checkout','--detach',sha])
            code=root/'code';checkout(code,REPO,COMMIT,True)
            upstream=root/'upstream';checkout(upstream,'https://github.com/YaoZhang93/TumorCP.git',OFFICIAL)
            for directory in (code,upstream):
                if run(['git','-C',str(directory),'status','--porcelain']):
                    raise RuntimeError('New source checkout is not clean')
            for name in ('queue','downloads','data'): (root/name).mkdir(exist_ok=True)
            source=code/'reproductions/tumorcp'
            for name in ('run.py','smoke.py','prepare.pbs','train.pbs','sources.json','requirements.txt'):
                if not (source/name).is_file():raise FileNotFoundError(source/name)
            saved={'format':'tumorcp_kits19_Git_deployment_v1','root':str(root),'commit':COMMIT,
                   'upstream_commit':OFFICIAL,'base_python':sys.executable,
                   'write_probe_bytes':probe_bytes,'future_capacity_reserved':False,
                   'dataset_download_started':False,'environment_installed':False,
                   'new_jobs_submitted':False,'existing_jobs_changed':False,'created_unix':time.time()}
            write_new(marker,saved)
            emit({'bootstrap':'COMPLETE','receipt':saved})
    ''')


def submit_script(root, stage, dependency):
    if stage not in ('prepare', 'train'):
        raise ValueError(stage)
    if dependency and not re.fullmatch(r'[0-9]+\.[A-Za-z0-9_-]+', dependency):
        raise ValueError('Exact owned PBS preparation job ID required')
    if stage == 'prepare' and dependency:
        raise ValueError('Preparation does not depend on an unrelated job')
    if stage == 'train' and not dependency:
        raise ValueError('GPU training requires explicit afterok preparation dependency')
    return header(root) + f'STAGE={stage!r}\nDEPENDENCY={dependency!r}\n' + textwrap.dedent('''
        boot=read(root/'bootstrap.json')
        if boot['root']!=str(root):raise ValueError('Deployment root mismatch')
        for directory,sha in [(root/'code',boot['commit']),(root/'upstream',boot['upstream_commit'])]:
            if run(['git','-C',str(directory),'rev-parse','HEAD'])!=sha or run(['git','-C',str(directory),'status','--porcelain']):
                raise ValueError('Pinned source changed before job submission; no job started')
        if STAGE=='train':
            previous=read(root/'queue/submitted_prepare.json')
            if previous['stage']!='prepare' or previous['job_id']!=DEPENDENCY:
                raise ValueError('GPU dependency must be this reproduction preparation job')
        result_file=root/'queue'/('submitted_'+STAGE+'.json')
        if result_file.is_file():
            saved=read(result_file)
            if saved['dependency']!=DEPENDENCY:raise ValueError('Saved independent submission dependency differs')
            emit({'submission':'ALREADY_RECORDED','receipt':saved})
        else:
            intent=root/'queue'/('submission_intent_'+STAGE+'.json')
            if intent.exists():raise RuntimeError('Submission outcome may be unknown; inspect PBS before retry. Intent: '+str(intent))
            source=root/'code/reproductions/tumorcp'
            if STAGE=='prepare':
                queue='shr_tgpu';select='1:ncpus=8:mem=96gb:ngpus=0:Qlist=tgpu:container_engine=singularity';wall='24:00:00'
            else:
                queue='mig_agpu';select='1:ncpus=8:mem=64gb:ngpus=1:Qlist=mig_agpu:container_engine=singularity';wall='72:00:00'
            argv=['/opt/pbs/bin/qsub','-N','TCPP_'+STAGE,'-q',queue,'-l','select='+select,
                  '-l','walltime='+wall,'-j','oe','-o',str(root/'queue'/(STAGE+'.log')),
                  '-v','CP_ROOT='+str(root)+',CP_BASE_PYTHON='+boot['base_python']]
            if DEPENDENCY:argv+=['-W','depend=afterok:'+DEPENDENCY]
            argv.append(str(source/(STAGE+'.pbs')))
            write_new(intent,{'stage':STAGE,'dependency':DEPENDENCY,'command':argv,'created_unix':time.time()})
            job=run(argv)
            import re
            if not re.fullmatch(r'[0-9]+\.[A-Za-z0-9_-]+',job):
                raise RuntimeError('Unrecognized qsub result; intent preserved; do not resubmit blindly: '+job)
            saved={'stage':STAGE,'job_id':job,'dependency':DEPENDENCY,'queue':queue,
                   'select':select,'walltime':wall,'command':argv,'existing_jobs_changed':False,
                   'submitted_unix':time.time()}
            write_new(result_file,saved)
            emit({'submission':'SUBMITTED','receipt':saved})
    ''')


def status_script(root):
    return header(root) + textwrap.dedent('''
        jobs=[]
        for name in ('prepare','train'):
            path=root/'queue'/('submitted_'+name+'.json')
            if path.is_file():
                value=read(path);jobs.append(value['job_id']);emit({'submission':value})
        if jobs:
            result=subprocess.run(['/opt/pbs/bin/qstat','-f','-F','json',*jobs],capture_output=True,text=True,timeout=25)
            if result.returncode:emit({'qstat_returncode':result.returncode,'stdout':result.stdout[-3000:],'stderr':result.stderr[-3000:]})
            else:
                value=json.loads(result.stdout)
                keys=('Job_Name','job_state','queue','exec_host','exec_vnode','Resource_List','Hold_Types','comment','Exit_status')
                emit({'jobs':{k:{name:v.get(name) for name in keys} for k,v in value.get('Jobs',{}).items()}})
        for name in ('state.json','prepared.json','smoke.json','completion.json'):
            path=root/'run'/name
            if path.is_file():
                value=read(path)
                emit({'path':str(path),'summary':{k:v for k,v in value.items() if k not in ('train_ids','heldout_ids','files','records','metric_per_case','upstream_sha256','runtime_sha256')}})
        emit({'read_only':True,'existing_jobs_changed':False})
    ''')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--transport',type=Path,required=True,help='Existing private connection helper exposing run(script, timeout)')
    parser.add_argument('--root',type=remote_root,required=True)
    command=parser.add_subparsers(dest='action',required=True)
    command.add_parser('inspect');boot=command.add_parser('bootstrap');boot.add_argument('--commit',required=True)
    submit=command.add_parser('submit');submit.add_argument('--stage',choices=('prepare','train'),required=True);submit.add_argument('--dependency')
    command.add_parser('status')
    args=parser.parse_args()
    if args.action=='inspect':script=inspect_script(args.root)
    elif args.action=='bootstrap':script=bootstrap_script(args.root,args.commit)
    elif args.action=='submit':script=submit_script(args.root,args.stage,args.dependency)
    else:script=status_script(args.root)
    module=transport(args.transport)
    print(module.run(script,timeout=300 if args.action=='bootstrap' else 90))


if __name__=='__main__':main()
