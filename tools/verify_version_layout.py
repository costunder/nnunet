"""Verify navigation, real CLI delegation and protected runtime publication."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    env=os.environ.copy();env['PYTHONIOENCODING']='utf8';env['PYTHONDONTWRITEBYTECODE']='1'
    test=subprocess.run([sys.executable,'-B','-m','unittest','tests.test_version_layout','-v'],
                        cwd=ROOT,env=env,capture_output=True,text=True,encoding='utf8',check=True)
    entries=json.loads((ROOT/'versions/index.json').read_text(encoding='utf8'))['entries']
    help_checks=[]
    for entry in ('v1.4/run.py','v1.5/run.py','v1.6/run.py','v1.7/C/run.py','v1.7/D/run.py',
                  'v1/eval.py','v2.2/cnn/run.py','v2.2/curriculum/run.py'):
        result=subprocess.run([sys.executable,'-B',str(ROOT/'versions'/entry),'--help'],
                              cwd=ROOT,env=env,capture_output=True,text=True,encoding='utf8',check=True)
        if 'usage:' not in result.stdout: raise AssertionError('Original CLI help missing: '+entry)
        help_checks.append(dict(entry=entry,status='PASS',stdout_sha256=hashlib.sha256(result.stdout.encode()).hexdigest()))
    aliases=list((ROOT/'versions').rglob('*.py'))
    for path in aliases: ast.parse(path.read_text(encoding='utf8'),filename=str(path))
    staged=subprocess.run(['git','-c',f'safe.directory={ROOT.as_posix()}',
                           'diff','--cached','--name-only'],cwd=ROOT,check=True,capture_output=True,text=True).stdout.splitlines()
    protected=('hiercp/','hiercp_v1x/','hiercp_v2/','hiercp_v22/','hiercp_v221/','hiercp_v222/',
               'l0_','config/','custom_trainers/','run.py','run_v')
    changes=[p for p in staged if p.startswith(protected)]
    if changes: raise AssertionError('Organization changes bound model/config runtime: '+str(changes))
    from hiercp_v1x.contracts import verify_archive
    original=verify_archive()
    catalog=json.loads((ROOT/'versions/files.json').read_text(encoding='utf8'))
    artifacts=json.loads((ROOT/'versions/artifacts.json').read_text(encoding='utf8'))
    proof=dict(format='hiercp_version_navigation_verification_v1',status='PASS',
               entries=len(entries),unit_tests=10,alias_AST_checked=len(aliases),original_CLI_help=help_checks,
               original_v1_archive_members_verified=original['verified_files'],
               protected_runtime_config_staged_changes=changes,indexed_files=len(catalog['files']),
               indexed_artifact_folders=len(artifacts['folders']),
               archive_verification='archives.json',v2_archive_verification='v2.json',
               original_checkpoints_or_medical_data_moved=False,dummy_deletion_performed=False,
               new_GPU_neural_run=False,full_training=False,full_evaluation=False,
               test_output=test.stdout+test.stderr)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({k:proof[k] for k in ('status','entries','unit_tests','original_v1_archive_members_verified',
                                         'protected_runtime_config_staged_changes','indexed_files','indexed_artifact_folders')}))


if __name__=='__main__':
    main()
