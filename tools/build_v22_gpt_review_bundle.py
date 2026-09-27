"""Create a new, verified source-only GPT review bundle without replacing code.txt."""
from __future__ import annotations
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from zipfile import ZipFile, ZIP_DEFLATED

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

REQUEST=(ROOT/'docs/gpt_review_request_v22.txt').read_text(encoding='utf-8')
GUIDE=(ROOT/'docs/gpt_review_guide_v22.md').read_text(encoding='utf-8')
GUIDE+='\n\n---\n\n'+(ROOT/'docs/v22_s01_installed_boundary_20260928.md').read_text(encoding='utf-8')
GUIDE+='\n\n---\n\n'+(ROOT/'docs/v22_segmentation_gpu_review_20260928.md').read_text(encoding='utf-8')
GUIDE+='\n\n---\n\n'+(ROOT/'docs/v22_online_review_fixes_20260927.md').read_text(encoding='utf-8')
GUIDE+='\n\n---\n\n'+(ROOT/'docs/v22_current_model_integrity_20260927.md').read_text(encoding='utf-8')
GUIDE+='\n\n---\n\n'+(ROOT/'docs/v22_resume_integrity_20260927.md').read_text(encoding='utf-8')
GUIDE+='\n\n---\n\n'+(ROOT/'docs/v22_rereview_repairs_20260927.md').read_text(encoding='utf-8')
GUIDE+='\n\n---\n\n'+(ROOT/'docs/v22_review_repairs_20260927.md').read_text(encoding='utf-8')
GUIDE+='\n\n---\n\n'+(ROOT/'docs/v22_observed_ranking_20260927.md').read_text(encoding='utf-8')

ACTIVE=[
 'tools/inspect_installed_epoch_boundary.py','tools/verify_v22_epoch_boundary_debug.py',
 'tools/v22_seg_state.py','tests/test_v22_seg_state.py','tools/verify_v22_segmentation_cuda_debug.py',
 'tools/v22_online_runtime.py','tools/v22_online_native.py','tools/v22_online_storage.py',
 'tools/v22_online_checkpoint.py','tests/test_v22_online_review_fixes.py',
 'tools/verify_v22_online_actual_debug.py',
 'tools/v22_online_selection.py','tools/v22_online_rank_bank.py','tools/v22_online_rank_adapter.py',
 'custom_trainers/nnUNetTrainer_OnlineRankV22.py','tools/train_v22_online_rank.py',
 'tools/v22_cache_storage.py','tests/test_v22_online_rank_bridge.py',
 'tools/verify_v22_current_model_admission_debug.py',
 'tools/v22_resume_integrity.py','tests/test_v22_resume_integrity.py',
 'tools/verify_v22_integrity_process_debug.py','tools/verify_v22_same_episode_debug.py',
 'hiercp_v222/record_binding.py','tools/v22_resume_state.py','tools/v22_calibration.py',
 'tests/test_v22_rereview.py','tools/verify_v22_rereview_continuation_debug.py','tools/verify_v22_calibration_rejection_debug.py',
 'hiercp_v222/placement.py','hiercp_v22/data.py','tools/v22_artifacts.py',
 'tools/v22_candidate_order.py','tools/v22_runtime_receipts.py','tools/v22_pair_audit.py',
 'tools/v22_debug_profile.py','tests/test_v22_review_repairs.py',
 'tools/verify_v22_portable_resume_debug.py','tools/verify_v22_review_selection_debug.py',
 'tools/build_v22_review_debug_cache.py','tools/audit_v22_physical_grids.py',
 'tools/v22_ranking_training.py','tools/v22_ranking_steps.py','tools/v22_rank_objective.py',
 'tools/v22_rank_recommendation.py','tools/v222_review_contracts.py','config/v22_observed_ranking.json',
 'tests/test_v22_observed_ranking.py','tools/verify_v22_ranking_real_debug.py',
 'tools/verify_v22_ranking_cycle_debug.py','tools/verify_v22_recommendation_real_debug.py',
 'tools/run_v222_server.py','tools/run_v222_process_runtime.py','tools/run_v222_optimized.py',
 'run_v222_v1_l0.py','config/prompt_graph_v222_v1_l0.json','config/train.json','config/split_cp80_fold0.json',
 'tools/v1_server.py','tools/v222_prepare_optimized.py','hiercp_v222/v1_cache.py','hiercp_v222/v1_recovery.py',
 'hiercp_v222/data.py','hiercp_v222/v1_local.py','hiercp_v22/schema.py','hiercp_v22/spatial.py',
 'hiercp_v22/geometry.py','hiercp_v22/sample.py','hiercp_v22/local.py','hiercp/model.py',
 'hiercp_v222/deterministic_sampling.py','hiercp_v222/model.py','hiercp_v222/clustering.py',
 'hiercp_v222/v1_execution.py','hiercp_v222/v1_training.py','hiercp_v222/contracts.py','hiercp_v221/contracts.py',
 'tools/v222_runtime_cache.py','tools/v222_process_loader.py','tools/v222_runtime_execution.py',
 'tools/v222_support_snapshot.py','tools/v222_resume_guard.py','tools/watch_v222_server.py',
]

def git(*args):
    return subprocess.check_output(['git','-c',f'safe.directory={ROOT.as_posix()}',*args],cwd=ROOT)

def digest(raw):return hashlib.sha256(raw).hexdigest()

def frame(name,raw):
    text=raw.decode('utf-8-sig').replace('\r\n','\n')
    return f'===== FILE: {name} =====\n{text}\n===== END FILE: {name} =====\n\n'

def main():
    commit=git('rev-parse','HEAD').decode().strip()
    output=ROOT/'exports'/f'v2.2_gpt_review_{commit[:7]}_complete'
    output.mkdir(parents=True,exist_ok=False)
    tracked={p.decode() for p in git('ls-files','-z').split(b'\0') if p}
    selected=set(tracked)-{'code.txt'}
    # Include current untracked research modules explicitly, labelled in the manifest.
    roots=[p.name for p in ROOT.iterdir() if p.is_dir() and p.name.startswith('hiercp')]
    roots+=['basic_cp_online','custom_trainers','tools','tests','docs','config']
    for folder in roots:
        for p in (ROOT/folder).rglob('*'):
            if p.suffix in {'.py','.md','.json','.toml','.txt','.html','.cjs','.sh','.ps1'} and '__pycache__' not in p.parts:
                selected.add(p.relative_to(ROOT).as_posix())
    selected.update(p.name for p in ROOT.glob('*.py'))
    allowed_archives={p for p in tracked if p.endswith('.zip') and p.startswith(('versions/','feedback/'))}
    blobs={};manifest=[];excluded=[]
    for name in sorted(selected):
        p=ROOT/name
        if not p.is_file() or p.is_symlink() or not p.resolve().is_relative_to(ROOT):
            raise ValueError(f'Invalid selected source: {name}')
        if any(part in {'.git','.venv','work','Data','results','node_modules','__pycache__'} for part in p.relative_to(ROOT).parts) or p.name.startswith('.env'):
            excluded.append(dict(path=name,reason='not source'));continue
        raw=p.read_bytes()
        binary=name in allowed_archives
        if not binary:
            try:raw.decode('utf-8-sig')
            except UnicodeDecodeError:
                excluded.append(dict(path=name,reason='non-text outside preservation archives'));continue
            if b'\0' in raw:raise ValueError(f'Unexpected binary source {name}')
        if name.endswith('.ipynb'):
            from tools.export_gpt_handoff import verify_source_only_notebook
            verify_source_only_notebook(raw.decode('utf-8-sig'),name)
        if binary:
            with ZipFile(p) as archive:
                if archive.testzip() is not None:raise ValueError(f'Corrupt source archive: {name}')
        blobs[name]=raw
        head=git('show',f'{commit}:{name}') if name in tracked else None
        manifest.append(dict(path=name,bytes=len(raw),sha256=digest(raw),tracked=name in tracked,
            matches_base_commit=raw==head if head is not None else None,
            active_review_entry=name in ACTIVE,binary_preservation_archive=binary))
    if any(name not in blobs for name in ACTIVE):raise ValueError('Missing active source')
    missing_imports=set();syntax_errors=[]
    for name,raw in blobs.items():
        if not name.endswith('.py'):continue
        try:tree=ast.parse(raw.decode('utf-8-sig'),filename=name)
        except SyntaxError as error:
            syntax_errors.append(dict(path=name,line=error.lineno,message=error.msg));continue
        for node in ast.walk(tree):
            targets=([a.name for a in node.names] if isinstance(node,ast.Import)
                else [node.module] if isinstance(node,ast.ImportFrom) and not node.level and node.module else [])
            for target in targets:
                stem=target.replace('.','/')
                for candidate in (stem+'.py',stem+'/__init__.py'):
                    if (ROOT/candidate).is_file() and candidate not in blobs:missing_imports.add(candidate)
    if syntax_errors:raise ValueError(f'Python syntax errors: {syntax_errors}')
    # Frozen files intentionally use CRLF checkouts via .gitattributes; compare
    # Git's clean representation for dirtiness, while retaining raw-byte hashes.
    dirty={p.decode() for p in git('diff','--name-only','-z',commit,'--').split(b'\0') if p}
    if dirty.intersection(blobs):
        raise ValueError(f'Tracked source differs from HEAD: {sorted(dirty.intersection(blobs))}')
    if missing_imports:raise ValueError(f'Missing existing local import modules: {sorted(missing_imports)}')
    # Source-only archive must retain every file required by the frozen v1 guard.
    preserved=json.loads(blobs['versions/v1/manifest.json'])['files']
    missing=set(preserved)-set(blobs)-{'code.txt'}
    if missing:raise ValueError(f'Incomplete frozen-source dependencies: {sorted(missing)}')
    documents={
        '01_GPT_REVIEW_REQUEST.txt':REQUEST,
        '02_PIPELINE_AND_CODE_MAP.md':GUIDE,
    }
    ordered=ACTIVE+sorted(set(blobs)-set(ACTIVE))
    code_names=[name for name in ordered if not name.endswith(('.md','.zip')) and not name.startswith('validation/')]
    header=f'# CURRENT SOURCE SNAPSHOT\nBase commit: {commit}\nCurrent working tree; untracked sources are marked in manifest.json.\nActive review path: tools/run_v222_server.py --runtime process --training-objective observed_rank_v1 -> tools/v22_ranking_training.py.\nHistorical alternatives included for dependency/review completeness, not all active.\nFiles use original relative paths; bodies have LF line endings.\n\n'
    frames=[frame(name,blobs[name]) for name in code_names]
    documents['03_ALL_CODE.txt']=header+''.join(frames)
    documents['03_ACTIVE_PATH_CODE.txt']=header+''.join(frame(name,blobs[name]) for name in ACTIVE)
    evidence_names=[n for n in ('docs/v22_s01_installed_boundary_20260928.md','validation/v222_r6/installed_epoch_boundary_20260928.json','validation/v222_r6/s01_GPU_boundary_20260928_DEBUG.json',
        'docs/v22_segmentation_gpu_review_20260928.md','validation/v222_r6/segmentation_GPU_review_20260928_DEBUG.json',
        'docs/v22_online_review_fixes_20260927.md','validation/v222_r6/online_review_fixes_20260927_DEBUG.json',
        'docs/v22_online_bridge_storage_20260927.md','docs/v22_current_model_integrity_20260927.md','validation/v222_r6/current_model_integrity_20260927_DEBUG.json',
        'docs/v22_resume_integrity_20260927.md','validation/v222_r6/resume_integrity_20260927_DEBUG.json',
        'docs/v22_rereview_repairs_20260927.md','validation/v222_r6/rereview_repairs_20260927_DEBUG.json',
        'docs/v22_review_repairs_20260927.md',
        'validation/v222_r6/review_repairs_20260927_DEBUG.json','docs/v22_review_intake_20260927.md',
        'docs/v22_observed_ranking_20260927.md','docs/v222_independent_review_response_20260927.md',
        'gpt_handoff.md','PATCH_NOTES.md','SERVER_V222.md','REFERENCES.md',
        'docs/v222_support_snapshot_20260926.md','docs/v222_support_process_20260926.md') if n in blobs]
    evidence_names+=sorted(n for n in blobs if n.startswith('validation/') and n.endswith('.json') and n not in evidence_names)
    documents['04_EVIDENCE_AND_HISTORY.txt']=f'# Evidence and historical context; newest records first\nBase commit: {commit}\nHistorical proposals do not override the current call path. Metrics are DEBUG evidence, not final medical scores.\n\n'+''.join(frame(n,blobs[n]) for n in evidence_names)
    part=[];size=0;parts=[]
    for content in frames:
        if part and size+len(content.encode())>550_000:
            parts.append(header+''.join(part));part=[];size=0
        part.append(content);size+=len(content.encode())
    if part:parts.append(header+''.join(part))
    for number,content in enumerate(parts,1):documents[f'split_code/part_{number:02d}_of_{len(parts):02d}.txt']=content
    index=['# Active source symbols (original file line numbers)\n']
    for name in ACTIVE:
        if not name.endswith('.py'):continue
        parsed=ast.parse(blobs[name].decode('utf-8-sig'),filename=name)
        index.append(f'\n## {name}\n')
        for node in parsed.body:
            if isinstance(node,(ast.ClassDef,ast.FunctionDef,ast.AsyncFunctionDef)):
                index.append(f'- `{name}:{node.lineno}` — `{node.name}`\n')
                if isinstance(node,ast.ClassDef):
                    for method in node.body:
                        if isinstance(method,(ast.FunctionDef,ast.AsyncFunctionDef)):
                            index.append(f'  - `{name}:{method.lineno}` — `{node.name}.{method.name}`\n')
    documents['SYMBOL_INDEX.md']=''.join(index)
    metadata=dict(format='v22_gpt_review_source_snapshot_v1',generated_utc=datetime.now(timezone.utc).isoformat(),
        base_commit=commit,review_release='v2.2 observed_rank_v1; S01 installed-parent source audit and actual CUDA first/middle/final boundary DEBUG; production unchanged from c989ae1',
        source_scope='Tracked source plus explicitly selected current untracked source in named code/docs/config directories',
        excludes=['Current root code.txt (existing user file preserved)','CT/masks','work/cache/checkpoints','environments','Git database'],
        source_count=len(blobs),files=manifest,other_exclusions=excluded,
        tracked_sources_git_clean=True,
        checkout_note='matches_base_commit compares raw bytes; false with git-clean sources can reflect intentional .gitattributes CRLF checkout conversion. ZIP SHA256 always describes original working-tree bytes.',
        source_syntax_errors=syntax_errors,
        archives_note='Immutable previous-source/audit archives retained for verify_v1; historical code.txt inside them is historical, not current')
    documents['manifest.json']=json.dumps(metadata,ensure_ascii=False,indent=2)+'\n'
    documents['00_START_HERE.txt']=f'''GPT 교차 검증용 v2.2 계열 파일 묶음
현재 대상: v2.2 observed_rank_v1 + stride4; paired v1 L0 + L1/L2
최신 검증: S01 조건부 종료 분기는 로컬 설치본에 없음. 실제 상속/함수 원문과 CUDA 첫·중간·마지막 epoch 경계 검사 추가. 서버 설치본은 별도 확인 필요.
production 코드는 c989ae1 그대로이며 F03/U01 보호 유지. 실제 OS 신호 종료를 검증했다고 주장하지 않습니다.
v4 이하 checkpoint에 새 hash를 붙여 exact resume하지 않습니다. 동일 최신 core provenance/content-bound graph는 검증 후 재사용 가능.
관측 종양 순위 학습 → 모든 주석 종양과 paste 겹침 제외 → 실제 온라인 입력 연결. 전체 support/segmentation 공존 G4와 효용 G5는 미완료.
기준 commit: {commit}

가장 간단한 사용법:
1. GPT에 01_GPT_REVIEW_REQUEST.txt, 02_PIPELINE_AND_CODE_MAP.md,
   03_ALL_CODE.txt, 04_EVIDENCE_AND_HISTORY.txt를 첨부하세요.
2. 01의 요청문대로 검토해 달라고 하세요.
3. 전체 파일을 읽지 못한다면 03_ACTIVE_PATH_CODE.txt로 먼저 현재 경로를 보고,
   split_code/part_XX_of_{len(parts):02d}.txt를 차례로 추가하세요. 분할본은 파일 본문을 자르지 않습니다.
4. 특정 함수의 원본 줄 번호는 SYMBOL_INDEX.md로 찾을 수 있습니다.

전체 ZIP에는 source_tree/ 아래 원래 경로와 바이트 그대로인 {len(blobs)}개 파일이 있습니다.
SHA256와 Git 추적 여부, 기준 commit 일치 여부는 manifest.json에 있습니다.
과거 대안 코드도 포함돼 있으므로 현재 구현과 혼동하지 않도록 02를 먼저 읽으세요.
원본 CT·그래프 캐시·가중치는 없으며 학습을 재실행하는 데이터 패키지가 아닙니다.
코드 설명과 로컬 DEBUG 검증은 독립적인 설계 타당성/전체 학습/의료 성능의 증명이 아닙니다.
'''
    for name,text in documents.items():
        p=output/name;p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('x',encoding='utf-8',newline='\n') as stream:stream.write(text)
    # Round-trip exact source bytes in zip; no extraction or execution required.
    archive_path=output/'V2_2_GPT_REVIEW_BUNDLE.zip'
    with ZipFile(archive_path,'x',compression=ZIP_DEFLATED,compresslevel=6) as archive:
        for name,raw in blobs.items():archive.writestr('source_tree/'+name,raw)
        for name in documents:archive.write(output/name,name)
    with ZipFile(archive_path) as archive:
        if archive.testzip() is not None:raise ValueError('Bundle CRC check failed')
        for record in manifest:
            if digest(archive.read('source_tree/'+record['path']))!=record['sha256']:
                raise ValueError('Archived source hash mismatch')
    if any((ROOT/name).read_bytes()!=raw for name,raw in blobs.items()):
        raise ValueError('Source changed during export; do not publish this snapshot')
    # Compare all complete frames, not repeated per-part headers.
    rebuilt=''.join(part[len(header):] for part in parts)
    if rebuilt!=''.join(frames):raise ValueError('Split code coverage mismatch')
    report=dict(base_commit=commit,source_files=len(blobs),active_files=len(ACTIVE),
        code_files=len(code_names),split_parts=len(parts),archive_crc_valid=True,
        all_source_sha256_verified=True,split_code_exact_coverage=True,
        existing_local_absolute_import_modules_included=True,source_syntax_errors=syntax_errors,
        source_mutations=False,model_or_training_run=False,
        archive_bytes=archive_path.stat().st_size,archive_sha256=digest(archive_path.read_bytes()),
        artifacts={name:dict(bytes=(output/name).stat().st_size,sha256=digest((output/name).read_bytes())) for name in documents})
    (output/'BUNDLE_CHECK.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(output=str(output),**{k:v for k,v in report.items() if k!='artifacts'}),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
