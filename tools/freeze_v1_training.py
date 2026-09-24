"""Append-only source snapshot for the requested paired training run."""
from pathlib import Path
import hashlib
import json
import sys
from zipfile import ZipFile,ZIP_DEFLATED
ROOT=Path(__file__).resolve().parents[1]
paths=[ROOT/'run_v222_v1_l0.py',ROOT/'config/prompt_graph_v222_v1_l0.json',ROOT/'config/train.json',
       ROOT/'docs/v222_v1_training_20260924.md']
for folder in ('hiercp','hiercp_v22','hiercp_v222'):
    paths.extend((ROOT/folder).glob('*.py'))
paths.extend([ROOT/'work/v222_v1_training_smoke_20260924_r2/result.json'])
out=ROOT/'versions/v2.22'/(sys.argv[1] if len(sys.argv)>1 else 'v1_full_training_20260924')
if not out.resolve().is_relative_to(ROOT/'versions/v2.22'):raise ValueError('Invalid archive path')
if len(sys.argv)>1:
    paths.extend([ROOT/'work/v222_v1_empty_context_check_20260924/result.json',ROOT/'tools/check_v1_empty_context.py'])
if len(sys.argv)>1 and sys.argv[1]=='v1_execution_r6_20260924':
    paths.extend(ROOT/p for p in (
        'docs/v222_v1_execution_r6_20260924.md','tests/test_v1_execution.py',
        'tools/verify_v1_resume.py','tools/smoke_v1_execution_lifecycle.py',
        'tools/profile_v1_execution.py','tools/migrate_v1_execution_cache.py','tools/supervise_v1_execution.py',
        'work/v222_v1_resume_check_20260924_r6_final/result.json',
        'work/v222_v1_lifecycle_smoke_20260924_r6/result.json',
        'work/v222_v1_execution_comparison_20260924.json'))
out.mkdir(parents=True,exist_ok=False)
manifest={}
with ZipFile(out/'source.zip','x',ZIP_DEFLATED) as archive:
    for p in paths:
        name=p.relative_to(ROOT).as_posix();data=p.read_bytes()
        archive.writestr(name,data);manifest[name]=hashlib.sha256(data).hexdigest()
with (out/'manifest.json').open('x',encoding='utf-8') as f:json.dump(manifest,f,indent=2)
print(out)
