"""Build a complete navigation index without moving bound runtime/data files."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
CATALOG_DIRS = ('hiercp', 'hiercp_v1x', 'hiercp_v2', 'hiercp_v22', 'hiercp_v221',
                'hiercp_v222', 'hiercp_v22_cnn', 'l0_cnn_control', 'l0_ezsp',
                'l0_exploration', 'l0_local_cnn', 'l0_regions', 'l0_sage',
                'l0_sparse_feature', 'config', 'docs', 'tools', 'tests',
                'custom_trainers', 'basic_cp_online', 'feedback')
EXTENSIONS = {'.py', '.md', '.txt', '.json', '.sh', '.html', '.cjs', '.toml'}


def version_for(relative):
    """Human version names differ from immutable historical format/package IDs."""
    top, _, name = relative.partition('/')
    stem = Path(name or top).stem.lower()
    if top == 'hiercp_v1x':
        if any(k in stem for k in ('half_a',)):
            return 'v1.5'
        if any(k in stem for k in ('half_b',)):
            return 'v1.6'
        if any(k in stem for k in ('transition', 'historical', 'native30')):
            return 'v1.7' if 'native30' not in stem else 'v1'
        if any(k in stem for k in ('bounded', 'scope_',)):
            return 'v1.4'
        return 'v1'
    if top == 'hiercp':
        return 'v1'
    if top == 'hiercp_v2':
        return 'v2.1'
    if top in ('hiercp_v22', 'hiercp_v221', 'hiercp_v222', 'hiercp_v22_cnn') or top.startswith('l0_'):
        return 'v2.2'
    if re.search(r'(?:^|_)v1[._]?7(?:_|$)', stem) or any(k in stem for k in ('crossed', 'transition', 'historical_full128', 'full128_competition')):
        return 'v1.7'
    if re.search(r'(?:^|_)v1[._]?6(?:_|$)', stem) or 'half_b' in stem:
        return 'v1.6'
    if re.search(r'(?:^|_)v1[._]?5(?:_|$)', stem) or 'half_a' in stem:
        return 'v1.5'
    if re.search(r'(?:^|_)v1[._]?4(?:_|$)', stem) or 'scope_' in stem or 'bounded_training' in stem:
        return 'v1.4'
    if any(k in stem for k in ('v222', 'v221', 'v22', 'local_cnn', 'region_', 'regions_', 'l0_', 'exploration', 'sparse', 'ssn', 'cnn_v1', 'cnn-v1')):
        return 'v2.2'
    if any(k in stem for k in ('v21', 'pipeline_v2', 'prompt_graph_v2', 'shared_donor', 'prompt_graph_v2')) or stem == 'run_v2':
        return 'v2.1'
    if any(k in stem for k in ('v1_', 'v1x', 'native_v1', 'native30')) or stem in ('run', 'run_v1', 'train'):
        return 'v1'
    return 'common'


def short_title(path):
    stem = Path(path).stem
    stem = re.sub(r'_20\d{6}(?:_.*)?$', '', stem)
    stem = re.sub(r'^(?:test_|verify_|check_)', '', stem)
    return stem.replace('_', ' ')


def collect():
    tracked=set(subprocess.run(['git','-c',f'safe.directory={ROOT.as_posix()}',
                'ls-files'],cwd=ROOT,check=True,capture_output=True,text=True).stdout.splitlines())
    modified=set(subprocess.run(['git','-c',f'safe.directory={ROOT.as_posix()}',
                 'diff','--name-only'],cwd=ROOT,check=True,capture_output=True,text=True).stdout.splitlines())
    paths = set()
    for folder in CATALOG_DIRS:
        paths.update(p for p in (ROOT / folder).rglob('*')
                     if p.is_file() and p.suffix in EXTENSIONS
                     and not {'__pycache__', '.pytest_cache'}.intersection(p.parts))
    paths.update(p for p in ROOT.iterdir() if p.is_file() and p.suffix in EXTENSIONS)
    rows = []
    for path in sorted(paths):
        relative = path.relative_to(ROOT).as_posix()
        raw = path.read_bytes()
        category = 'code'
        if relative.startswith('docs/') or path.suffix == '.md': category = 'notes'
        elif relative.startswith('config/'): category = 'config'
        elif relative.startswith('tests/'): category = 'tests'
        elif relative.startswith('tools/') or relative.startswith('run'): category = 'tools'
        rows.append(dict(path=relative, version=version_for(relative), category=category,
                         title=short_title(relative), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(),
                         in_git=relative in tracked,working_tree_modified=relative in modified))
    return rows


def build():
    rows = collect()
    payload = dict(format='hiercp_version_file_catalog_v1',
                   scope='Repository source/config/docs/tools/tests; data/checkpoints are separate bound artifacts',
                   sha256_scope='Actual local file bytes. working_tree_modified marks preserved unpublished edits; in_git only means the path is tracked.',
                   files=rows)
    (ROOT / 'versions/files.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    for version in sorted({row['version'] for row in rows}):
        folder = ROOT / 'versions' / version
        folder.mkdir(exist_ok=True)
        content = [f'# {version} 파일 안내', '',
                   '원본 경로와 SHA는 유지합니다. 아래는 짧은 이름으로 찾는 실행·코드·설정·기록 목록입니다.',
                   '게시 여부는 Git index 기준입니다. 로컬 파일은 서버 checkout에 없을 수 있습니다.', '']
        for category, label in (('code','구현'), ('config','설정'), ('tools','실행·분석'), ('notes','기록'), ('tests','검사')):
            selected = [r for r in rows if r['version']==version and r['category']==category]
            if not selected: continue
            content += [f'<details><summary>{label} ({len(selected)})</summary>', '', '| 내용 | 원본 | Git |', '| --- | --- | --- |']
            for row in selected:
                publication='로컬 수정' if row['working_tree_modified'] else ('게시 대상' if row['in_git'] else '로컬')
                content.append(f'| {row["title"]} | [열기](../../{row["path"]}) | {publication} |')
            content += ['', '</details>', '']
        (folder / 'files.md').write_text('\n'.join(content), encoding='utf8')
    print(json.dumps(dict(indexed=len(rows), versions=sorted({r['version'] for r in rows}), training_started=False)))


def artifacts():
    """Index existing output directories; never rewrite or hash CT payloads."""
    rows=[]
    for parent in ('work','validation','output','outputs','experiment_results','exports','tmp'):
        location=ROOT/parent
        if not location.is_dir(): continue
        for child in sorted(location.iterdir()):
            if not child.is_dir(): continue
            name=child.name
            is_test=any(token in name.lower() for token in ('unit','test','debug','smoke','probe','valid_signature','tmp'))
            # This is a storage/navigation hint, never a quality/completion claim.
            rows.append(dict(path=child.relative_to(ROOT).as_posix(),version=version_for(name),
                             kind=parent,display=short_title(name),
                             scope_hint='test/debug name; inspect receipt before deletion' if is_test else 'inspect receipt',
                             deletion_authorized=False))
    write= lambda p,s:p.write_text(s.rstrip()+'\n',encoding='utf8')
    write(ROOT/'versions/artifacts.json',json.dumps(dict(format='hiercp_artifact_locations_v1',
        path_policy='Keep exact checkpoint/cache/evidence paths; list all top-level artifact folders',
        deletion_performed=False,folders=rows),ensure_ascii=False,indent=2))
    for version in sorted({r['version'] for r in rows}):
        folder=ROOT/'versions'/version;folder.mkdir(exist_ok=True)
        text=[f'# {version} 실험 파일 위치','','실제 저장 경로는 그대로 유지합니다. 이름만으로 학습 완료·품질·삭제 가능 여부를 판정하지 않습니다.','']
        for kind in ('work','validation','output','outputs','experiment_results','exports','tmp'):
            selected=[r for r in rows if r['version']==version and r['kind']==kind]
            if not selected: continue
            text += [f'<details><summary>{kind} ({len(selected)})</summary>','','| 내용 | 위치 |','| --- | --- |']
            text += [f'| {r["display"]} | [열기](../../{r["path"]}/) |' for r in selected]
            text += ['','</details>','']
        write(folder/'runs.md','\n'.join(text))
    text=['# 실험 저장 위치','','[버전 안내](README.md)에서 실행 방법을 선택하고, 각 버전의 runs.md에서 기존 결과·cache·검증 위치를 찾습니다.','',
          '| 버전 | 저장 폴더 수 | 위치 |','| --- | ---: | --- |']
    for v in sorted({r['version'] for r in rows}):
        text.append(f'| {v} | {sum(r["version"]==v for r in rows)} | [runs.md]({v}/runs.md) |')
    text += ['','테스트·DEBUG 이름의 폴더는 별도로 표시하지만 더미라고 자동 삭제하지 않습니다. 이번 작업에서 기존 학습 결과·CT cache·checkpoint 삭제는 하지 않았습니다.','',
             '새 실험은 버전/방법과 고유 실험 이름을 사용합니다. 기존 서버 실험 경로는 results.json 및 각 버전 README에 보존합니다.']
    write(ROOT/'versions/artifacts.md','\n'.join(text))
    print(json.dumps(dict(artifact_folders=len(rows),deletion_performed=False,training_started=False)))


if __name__ == '__main__':
    build()
    artifacts()
