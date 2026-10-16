"""Deploy one liver TumorCP paper profile using an existing private transport.

Reuse only the read-only KiTS19 environment and official upstream, never its
data, plans, checkpoint, mutable run directory or existing PBS jobs.
"""
from __future__ import annotations

import argparse
from pathlib import Path, PurePosixPath
import re
import textwrap

from remote import OFFICIAL, SOURCE_REPO, header, status_script, transport

SPLIT_SHA = '7a993854c570b2cabe98dffc5df97550d74da3a688c3c3b75942ce4cbd15bc44'
PINS_CHECK = '''import importlib.metadata as metadata
import json,sys
from pathlib import Path
installed={}
for line in Path(sys.argv[1]).read_text().splitlines():
    line=line.strip()
    if not line or line.startswith('#'):continue
    name,expected=line.split('==')
    actual=metadata.version(name)
    if actual!=expected:raise RuntimeError(f'Private environment {name}: expected {expected}, actual {actual}')
    installed[name]=actual
installed['torch']=metadata.version('torch')
if installed['torch']!='2.6.0+cu118':raise RuntimeError('Shared Torch runtime changed; inspect before reuse')
print(json.dumps(installed))
'''


def server_path(value):
    path=PurePosixPath(value)
    if not path.is_absolute() or '..' in path.parts or any(c in value for c in ('\n','\r',',','=')):
        raise ValueError('Explicit absolute server path without PBS separators required')
    return str(path)


def liver_root(value):
    value=server_path(value);path=PurePosixPath(value)
    if len(path.parts)<5 or path.name!='tumorcp_liver' or path.parent.name!='reproductions':
        raise ValueError('Use a separate absolute .../reproductions/tumorcp_liver namespace')
    return value


def inspect_script(root, shared, data, split):
    return header(root)+f'SHARED={shared!r}\nDATA={data!r}\nSPLIT={split!r}\n'+textwrap.dedent('''
        ancestor=root
        while not ancestor.exists():ancestor=ancestor.parent
        out={'read_only':True,'root':str(root),'root_exists':root.exists(),'disk':shutil.disk_usage(ancestor)._asdict(),
             'shared_private_python_exists':(Path(SHARED)/'.env/bin/python').is_file(),
             'upstream_exists':(Path(SHARED)/'upstream/.git').is_dir()}
        if root.is_dir():out['entries']=[p.name for p in root.iterdir()]
        for label,value in [('data',DATA),('split',SPLIT)]:out[label]={'path':value,'exists':Path(value).exists()}
        emit(out)
    ''')


def bootstrap_script(root, shared, data, split, commit):
    if not re.fullmatch('[0-9a-f]{40}',commit):raise ValueError('Full published Git commit required')
    values=dict(SHARED=shared,DATA=data,SPLIT=split,COMMIT=commit,OFFICIAL=OFFICIAL,REPO=SOURCE_REPO,SPLIT_SHA=SPLIT_SHA,PINS_CHECK=PINS_CHECK)
    prefix=''.join(f'{key}={value!r}\n' for key,value in values.items())
    return header(root)+prefix+textwrap.dedent(r'''
        shared=Path(SHARED);data=Path(DATA);source_split=Path(SPLIT)
        if root==shared or shared in root.parents or root in shared.parents:
            raise ValueError('Liver and KiTS19 mutable namespaces must be separate')
        if hashlib.sha256(source_split.read_bytes()).hexdigest()!=SPLIT_SHA:
            raise ValueError('Historical 105/26 split changed; no replacement partition')
        original=read(source_split)
        if not isinstance(original,list) or len(original)!=1 or set(original[0])!={'train','val'}:
            raise ValueError('Historical source split shape differs')
        train=original[0]['train'];heldout=original[0]['val'];all_ids=[f'liver_{i}' for i in range(131)]
        if (len(train)!=105 or len(set(train))!=105 or len(heldout)!=26 or len(set(heldout))!=26
                or set(train)&set(heldout) or set(train)|set(heldout)!=set(all_ids)):
            raise ValueError('Require all131 patients and exact disjoint105/26 split')
        for label in ('imagesTr','labelsTr'):
            present={p.name[:-7] for p in (data/label).glob('liver_*.nii.gz')}
            if present!=set(all_ids):raise ValueError('Original liver inventory differs: '+label)
        upstream=shared/'upstream';python=shared/'.env/bin/python'
        if not python.is_file():raise FileNotFoundError('Completed shared private environment required')
        if run(['git','-C',str(upstream),'rev-parse','HEAD'])!=OFFICIAL or run(['git','-C',str(upstream),'status','--porcelain']):
            raise ValueError('Shared official source identity or bytes changed')
        profile={'format':'tumorcp_paper_dataset_profile_v1','dataset':'MSD Task03_Liver','task':'Task029_Liver',
                 'raw_layout':'msd','all_ids':all_ids,'train_ids':train,'heldout_ids':heldout,
                 'split_provenance':'Existing v2.4 outer fold0; unchanged105 training and26 held-out patients',
                 'source_split_file':str(source_split),'source_split_sha256':SPLIT_SHA,'allow_empty_intra':True}
        marker=root/'bootstrap.json'
        expected={'format':'tumorcp_liver_single_Git_deployment_v1','root':str(root),'commit':COMMIT,
                  'shared_root':str(shared),'raw_data':str(data),'source_split':str(source_split),'source_split_sha256':SPLIT_SHA,
                  'upstream_commit':OFFICIAL,'upstream':str(upstream),'env_python':str(python)}
        if marker.is_file():
            saved=read(marker)
            if any(saved.get(key)!=value for key,value in expected.items()):raise ValueError('Existing liver namespace differs')
            if read(root/'profile.json')!=profile:raise ValueError('Saved liver profile changed')
        else:
            intent=root/'bootstrap_intent.json'
            if root.exists():
                if not intent.is_file() or read(intent)!=expected:raise FileExistsError('Unclaimed or different namespace preserved')
                allowed={'bootstrap_intent.json','code','logs','queue','profile.json'}
                if set(p.name for p in root.iterdir())-allowed:raise ValueError('Unknown partial deployment entries')
            else:
                root.mkdir(parents=True,exist_ok=False);write_new(intent,expected)
            probe=root/('.write_probe_'+uuid.uuid4().hex)
            try:
                with probe.open('xb') as f:
                    block=bytes(2**20)
                    for _ in range(128):f.write(block)
                    f.flush();os.fsync(f.fileno())
                probe_bytes=probe.stat().st_size
            finally:
                if probe.is_file():probe.unlink()
            code=root/'code'
            if not code.exists():code.mkdir()
            if not (code/'.git').is_dir():
                if list(code.iterdir()):raise ValueError('Unknown source directory preserved')
                run(['git','-C',str(code),'init'])
            origins=run(['git','-C',str(code),'remote']).splitlines()
            if not origins:run(['git','-C',str(code),'remote','add','origin',REPO])
            elif origins!=['origin'] or run(['git','-C',str(code),'config','--get','remote.origin.url'])!=REPO:
                raise ValueError('Source remote differs')
            current=subprocess.run(['git','-C',str(code),'rev-parse','--verify','HEAD'],capture_output=True,text=True,timeout=10)
            if current.returncode==0:
                if current.stdout.strip()!=COMMIT or run(['git','-C',str(code),'status','--porcelain']):
                    raise ValueError('Existing source checkout differs')
            else:
                run(['git','-C',str(code),'config','core.sparseCheckout','true'])
                (code/'.git/info/sparse-checkout').write_text('/reproductions/tumorcp/'+chr(10))
                run(['git','-C',str(code),'fetch','--depth','1','origin',COMMIT])
                run(['git','-C',str(code),'checkout','--detach',COMMIT])
            for name in ('queue','logs'):(root/name).mkdir(exist_ok=True)
            if (root/'profile.json').exists():
                if read(root/'profile.json')!=profile:raise ValueError('Partial profile differs')
            else:write_new(root/'profile.json',profile)
            installed=json.loads(run([str(python),'-B','-c',PINS_CHECK,str(code/'reproductions/tumorcp/requirements.txt')]))
            expected.update(write_probe_bytes=probe_bytes,future_capacity_reserved=False,
                            shared_environment_modified=False,raw_CT_copied=False,new_jobs_submitted=False,
                            existing_jobs_changed=False,installed_packages=installed,
                            config_sha256=hashlib.sha256((code/'reproductions/tumorcp/liver.json').read_bytes()).hexdigest(),
                            requirements_sha256=hashlib.sha256((code/'reproductions/tumorcp/requirements.txt').read_bytes()).hexdigest(),
                            profile_sha256=hashlib.sha256((root/'profile.json').read_bytes()).hexdigest())
            write_new(marker,expected);saved=expected
        if run(['git','-C',str(root/'code'),'rev-parse','HEAD'])!=COMMIT or run(['git','-C',str(root/'code'),'status','--porcelain']):
            raise ValueError('Pinned liver source changed')
        installed=json.loads(run([str(python),'-B','-c',PINS_CHECK,str(root/'code/reproductions/tumorcp/requirements.txt')]))
        if installed!=saved['installed_packages']:raise ValueError('Shared private environment changed')
        for field,name in [('config_sha256','liver.json'),('requirements_sha256','requirements.txt')]:
            if hashlib.sha256((root/'code/reproductions/tumorcp'/name).read_bytes()).hexdigest()!=saved[field]:
                raise ValueError('Pinned '+name+' differs from deployment receipt')
        emit({'bootstrap':'VERIFIED_COMPLETE','receipt':saved,'dataset_counts':{'all':131,'training':105,'heldout':26}})
    ''')


def submit_script(root, stage, dependency):
    if stage not in ('prepare','train'):raise ValueError(stage)
    if dependency and not re.fullmatch(r'[0-9]+\.[A-Za-z0-9_-]+',dependency):raise ValueError('Exact owned PBS ID required')
    if stage=='prepare' and dependency:raise ValueError('Independent liver preparation has no foreign dependency')
    if stage=='train' and not dependency:raise ValueError('GPU requires exact liver preparation dependency')
    return header(root)+f'STAGE={stage!r}\nDEPENDENCY={dependency!r}\n'+textwrap.dedent('''
        boot=read(root/'bootstrap.json');source=root/'code/reproductions/tumorcp'
        if boot['format']!='tumorcp_liver_single_Git_deployment_v1' or boot['root']!=str(root):raise ValueError('Not this independent liver deployment')
        if run(['git','-C',str(root/'code'),'rev-parse','HEAD'])!=boot['commit'] or run(['git','-C',str(root/'code'),'status','--porcelain']):
            raise ValueError('Pinned code changed before submission')
        if hashlib.sha256((root/'profile.json').read_bytes()).hexdigest()!=boot['profile_sha256']:raise ValueError('Fixed cohort profile changed')
        if STAGE=='train':
            previous=read(root/'queue/submitted_prepare.json')
            if previous['stage']!='prepare' or previous['job_id']!=DEPENDENCY:raise ValueError('Not this liver preparation job')
        receipt=root/'queue'/('submitted_'+STAGE+'.json')
        if receipt.is_file():
            saved=read(receipt)
            if saved['dependency']!=DEPENDENCY:raise ValueError('Existing submission differs')
            emit({'submission':'ALREADY_RECORDED','receipt':saved})
        else:
            intent=root/'queue'/('submission_intent_'+STAGE+'.json')
            if intent.exists():raise RuntimeError('Unknown submission outcome; inspect PBS and intent before retry')
            if STAGE=='prepare':
                queue='shr_tgpu';select='1:ncpus=8:mem=96gb:ngpus=0:Qlist=tgpu:container_engine=singularity';wall='24:00:00';name='TCPP_Lprep'
            else:
                queue='mig_agpu';select='1:ncpus=8:mem=64gb:ngpus=1:Qlist=mig_agpu:container_engine=singularity';wall='72:00:00';name='TCPP_Ltrain'
            variables={'CP_ROOT':str(root),'CP_ENV_PYTHON':boot['env_python'],'CP_UPSTREAM':boot['upstream'],'CP_DATA':boot['raw_data']}
            argv=['/opt/pbs/bin/qsub','-N',name,'-q',queue,'-l','select='+select,'-l','walltime='+wall,
                  '-j','oe','-o',str(root/'queue'/(STAGE+'.log')),'-v',','.join(key+'='+value for key,value in variables.items())]
            if DEPENDENCY:argv+=['-W','depend=afterok:'+DEPENDENCY]
            argv.append(str(source/('liver_'+STAGE+'.pbs')))
            write_new(intent,{'stage':STAGE,'dependency':DEPENDENCY,'command':argv,'created_unix':time.time()})
            job=run(argv)
            import re
            if not re.fullmatch(r'[0-9]+\.[A-Za-z0-9_-]+',job):raise RuntimeError('Unknown qsub outcome; do not submit twice: '+job)
            saved={'stage':STAGE,'job_id':job,'dependency':DEPENDENCY,'queue':queue,'select':select,'walltime':wall,
                   'command':argv,'existing_jobs_changed':False,'submitted_unix':time.time()}
            write_new(receipt,saved);emit({'submission':'SUBMITTED','receipt':saved})
    ''')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--transport',type=Path,required=True)
    parser.add_argument('--root',type=liver_root,required=True)
    parser.add_argument('--shared',type=server_path,help='Read-only completed KiTS19 environment and official upstream root')
    parser.add_argument('--data',type=server_path,help='Original Task03_Liver directory')
    parser.add_argument('--split',type=server_path,help='Exact historical Dataset730 splits_final.json')
    actions=parser.add_subparsers(dest='action',required=True)
    actions.add_parser('inspect');boot=actions.add_parser('bootstrap');boot.add_argument('--commit',required=True)
    submit=actions.add_parser('submit');submit.add_argument('--stage',choices=('prepare','train'),required=True);submit.add_argument('--dependency')
    actions.add_parser('status');args=parser.parse_args()
    if args.action in ('inspect','bootstrap') and not all((args.shared,args.data,args.split)):
        raise ValueError('Explicit --shared, --data and --split required')
    if args.action=='inspect':script=inspect_script(args.root,args.shared,args.data,args.split)
    elif args.action=='bootstrap':script=bootstrap_script(args.root,args.shared,args.data,args.split,args.commit)
    elif args.action=='submit':script=submit_script(args.root,args.stage,args.dependency)
    else:script=status_script(args.root)
    print(transport(args.transport).run(script,timeout=300 if args.action=='bootstrap' else 90))


if __name__=='__main__':main()
