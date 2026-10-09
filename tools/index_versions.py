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
    relative = str(relative).replace('\\', '/')
    top, _, name = relative.partition('/')
    stem = Path(name or top).stem.lower()
    family = re.sub(r'^(?:test_|verify_)', '', stem)
    if re.search(r'(?:^|_)v1[._]?9(?:_|$)', stem):
        return 'v1.9'
    if re.search(r'(?:^|_)v1[._]?8(?:_|$)', stem) or 'u_bridge' in stem:
        return 'v1.8'
    if (re.search(r'^(?:(?:test|verify)_)?(?:(?:run|resume|report|stop|server)_)?comparison_', stem)
            and not any(k in stem for k in ('comparison_randomness', 'comparison_seed'))):
        return 'v1.9'
    if family.startswith(('arm_process', 'arm_device_', 'arm_launch_',
                          'owned_continuation', 'preparation_reuse')):
        return 'v1.9'
    if family == 'host_memory':
        return 'v1.8'
    if top == 'hiercp_v1x':
        if stem in ('arm_process', 'owned_continuation', 'preparation_reuse'):
            return 'v1.9'
        if stem == 'host_memory':
            return 'v1.8'
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


def navigation_checklist():
    """Documentation scope only; do not claim new neural or hardware checks."""
    return ['', '## 작업 완료 체크리스트', '',
        '이 파일은 탐색 인덱스다. 학습·평가·자원 benchmark와 데이터 이동·삭제는 실행하지 않는다.', '',
        '- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.',
        '- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.',
        '- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.',
        '- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다.',
        '- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다.',
        '- [ ] physical batch size와 병렬화 가능성을 실제로 검토했다. (인덱스 생성에 해당 없음)',
        '- [ ] GPU, CPU, RAM 활용 상태를 측정하거나 확인했다. (인덱스 생성에 해당 없음)',
        '- [ ] OOM 발생 시 모델 축소보다 메모리 및 병목 원인을 먼저 조사했다. (인덱스 생성에 해당 없음)',
        '- [x] 디버그 설정과 최종 설정을 분리해서 안내했다.',
        '- [x] dummy, placeholder, random fallback을 사용하지 않았다.',
        '- [ ] 핵심 모듈이 forward, loss, gradient와 optimizer에 연결되어 있다. (새 실행 검사 없음)',
        '- [x] 실제 실행 설정과 변경 사항을 명확하게 보고했다. 기존 구현·기록으로 연결한다.',
        '- [x] smoke test와 전체 학습 또는 전체 평가를 구분해서 보고했다.']


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
        (folder / 'files.md').write_text('\n'.join(content+navigation_checklist()), encoding='utf8')
    print(json.dumps(dict(indexed=len(rows), versions=sorted({r['version'] for r in rows}), training_started=False)))


def artifact_locations():
    """Read locations and completed relocation receipts, without reading payloads."""
    rows=[]
    receipts=[]
    def add(path, *, kind, version=None, **extra):
        relative=path.relative_to(ROOT).as_posix()
        name=path.name
        is_test=any(token in name.lower() for token in ('unit','test','debug','smoke','probe','valid_signature','tmp'))
        rows.append(dict(path=relative,version=version or version_for(name),
                         kind=kind,display=short_title(name),
                         scope_hint='test/debug name; inspect receipt before deletion' if is_test else 'inspect receipt',
                         deletion_authorized=False,**extra))

    for parent in ('work','validation','output','outputs','experiment_results','exports','tmp'):
        location=ROOT/parent
        if not location.is_dir(): continue
        for child in sorted(location.iterdir()):
            if not child.is_dir(): continue
            if parent=='work' and child.name in ('archive','cache','vessels'): continue
            add(child,kind=parent)

    # A grouping directory is navigation, not a new experiment. Keep its original
    # names in the relocation receipt rather than repeating 32k file records.
    organization=ROOT/'work/archive/organization'
    groups={}
    if organization.is_dir():
        add(organization,kind='organization',category='relocation_receipts')
        for folder in sorted(organization.iterdir()):
            if not folder.is_dir(): continue
            summary_path,plan_path=folder/'summary.json',folder/'plan.json'
            if not summary_path.is_file() or not plan_path.is_file(): continue
            summary=json.loads(summary_path.read_text(encoding='utf-8-sig'))
            if summary.get('sha256_and_size_verified') is not True: continue
            plan=json.loads(plan_path.read_text(encoding='utf-8-sig'))
            if not isinstance(plan,list):
                raise ValueError(f'Relocation plan must be an array: {plan_path}')
            if summary.get('moved_directories')!=len(plan):
                raise ValueError(f'Relocation count differs: {folder}')
            if not (folder/'files.json').is_file():
                raise ValueError(f'Completed relocation file inventory missing: {folder}')
            receipt_path=plan_path.relative_to(ROOT).as_posix()
            receipts.append(dict(path=receipt_path,
                summary=summary_path.relative_to(ROOT).as_posix(),
                directory_count=len(plan),original_current_mapping='source -> destination',
                file_inventory=(folder/'files.json').relative_to(ROOT).as_posix(),
                file_bytes_verified=True))
            for move in plan:
                source,destination=Path(move['source']),Path(move['destination'])
                if not destination.is_relative_to(ROOT/'work/archive'):
                    raise ValueError(f'Relocation destination outside archive: {destination}')
                if not destination.is_dir():
                    raise ValueError(f'Relocated directory missing: {destination}')
                if not destination.is_relative_to(ROOT/'work/archive/tests'):
                    add(destination,kind='archive',version=version_for(source.name),
                        category='preserved_artifacts',original_path=str(source),
                        relocation_receipts=[receipt_path])
                    continue
                key=destination.parent
                entry=groups.setdefault(key,dict(directory_count=0,relocation_receipts=set(),versions=set()))
                entry['directory_count']+=1
                entry['relocation_receipts'].add(receipt_path)
                version=version_for(source.name)
                family=str(move.get('group','')).replace('\\','/').split('/')[0]
                if version=='common':
                    version={'regions':'v2.2','transition':'v1.7','v1':'v1'}.get(family,version)
                entry['versions'].add(version)
        for path,entry in sorted(groups.items()):
            versions=sorted(entry.pop('versions'))
            add(path,kind='archived_tests',version=versions[0] if len(versions)==1 else 'common',
                category='generated_test_fixtures',version_families=versions,
                directory_count=entry['directory_count'],
                relocation_receipts=sorted(entry['relocation_receipts']))

    cache=ROOT/'work/cache'
    if cache.is_dir():
        add(cache,kind='cache',category='cache_index')
        for child in sorted(cache.iterdir()):
            if child.is_dir(): add(child,kind='cache',category='auxiliary_cache')
    vessels=ROOT/'work/vessels'
    if vessels.is_dir():
        add(vessels,kind='vessels',category='derived_vessel_visualizations')
        rows[-1]['display']='혈관 시각화'
    vessel_inputs=ROOT/'datasets/msd_liver/vessels'
    if vessel_inputs.is_dir():
        add(vessel_inputs,kind='vessels',version='common',category='derived_vessel_training_inputs')
        rows[-1]['display']='혈관 학습 입력'
    archive=ROOT/'work/archive'
    if archive.is_dir():
        for child in sorted(archive.iterdir()):
            if not child.is_dir() or child.name=='organization': continue
            if child.name=='tests' and groups: continue
            mapping=child/'relocation.json'
            extra=dict(category='preserved_artifacts')
            if mapping.is_file():
                entries=json.loads(mapping.read_text(encoding='utf-8-sig'))
                if not isinstance(entries,list):
                    raise ValueError(f'Relocation file map must be an array: {mapping}')
                receipt=mapping.relative_to(ROOT).as_posix()
                extra.update(relocation_receipts=[receipt],file_count=len(entries))
                receipts.append(dict(path=receipt,file_count=len(entries),
                    original_current_mapping='original -> current'))
            add(child,kind='archive',**extra)
    return rows,receipts


def artifacts():
    """Write navigation only; completed receipts preserve relocated original paths."""
    rows,receipts=artifact_locations()
    write= lambda p,s:p.write_text(s.rstrip()+'\n',encoding='utf8')
    write(ROOT/'versions/artifacts.json',json.dumps(dict(format='hiercp_artifact_locations_v1',
        path_policy='Bound experiment paths remain unchanged; grouped archives link completed original-to-current relocation receipts',
        deletion_performed=False,relocation_receipts=receipts,folders=rows),ensure_ascii=False,indent=2))
    for version in sorted({r['version'] for r in rows}):
        folder=ROOT/'versions'/version;folder.mkdir(exist_ok=True)
        text=[f'# {version} 실험 파일 위치','','봉인된 실험 경로는 유지합니다. 이동한 보존물은 원래 이름과 현재 위치를 relocation receipt로 추적합니다.',
              '이름만으로 학습 완료·품질·삭제 가능 여부를 판정하지 않습니다. 생성된 UNIT fixture 묶음은 실험 실행 수가 아닙니다.','']
        for kind in ('work','validation','output','outputs','experiment_results','exports','tmp',
                     'cache','vessels','archive','archived_tests','organization'):
            selected=[r for r in rows if r['version']==version and r['kind']==kind]
            if not selected: continue
            text += [f'<details><summary>{kind} ({len(selected)})</summary>','','| 내용 | 위치 |','| --- | --- |']
            for row in selected:
                label=row['display']
                if 'directory_count' in row: label+=f' ({row["directory_count"]}개 fixture)'
                links=f'[열기](../../{row["path"]}/)'
                links+=''.join(f' · [원래→현재](../../{p})' for p in row.get('relocation_receipts',[]))
                text.append(f'| {label} | {links} |')
            text += ['','</details>','']
        write(folder/'runs.md','\n'.join(text+navigation_checklist()))
    text=['# 실험 저장 위치','','[버전 안내](README.md)에서 실행 방법을 선택하고, 각 버전의 runs.md에서 기존 결과·cache·검증 위치를 찾습니다.','',
          '| 버전 | 위치·분류 항목 수 | 위치 |','| --- | ---: | --- |']
    for v in sorted({r['version'] for r in rows}):
        text.append(f'| {v} | {sum(r["version"]==v for r in rows)} | [runs.md]({v}/runs.md) |')
    text += ['','archive/tests는 완료된 SHA·크기 검증 receipt를 통해 묶음별로 표시합니다. 개별 원래 폴더명과 이동 위치는 원래→현재 링크에 보존하며, fixture 수를 학습 실험 수로 세지 않습니다.',
             'cache·혈관 산출물·보존 폴더도 별도 항목입니다. 봉인된 runtime/checkpoint/cache 경로는 이동하지 않습니다.',
             '테스트·DEBUG 이름만으로 더미라고 자동 삭제하지 않습니다. 이 인덱서는 데이터를 이동·삭제하지 않습니다.','',
             '새 실험은 버전/방법과 고유 실험 이름을 사용합니다. 기존 서버 실험 경로는 results.json 및 각 버전 README에 보존합니다.']
    write(ROOT/'versions/artifacts.md','\n'.join(text+navigation_checklist()))
    print(json.dumps(dict(artifact_folders=len(rows),deletion_performed=False,training_started=False)))


if __name__ == '__main__':
    build()
    artifacts()
