"""Collect read-only actual DEBUG evidence; never promote it to model quality."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess
from zipfile import ZipFile

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'validation/v17_recipient_context_20261005'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    mappings={
        'before_actual_geometry.json':'work/v17_empty_context_actual_DEBUG_20261005_before/report.json',
        'after_actual_geometry.json':'work/v17_empty_context_actual_DEBUG_20261005_after/report.json',
        'actual_CUDA_DEBUG.json':'work/v17_empty_context_CUDA_DEBUG_20261005_r1/report.json',
        'actual_nonempty_537_readonly_compatibility.json':'work/v17_actual_nonempty_reuse_FINAL_DEBUG_20261005_160121/report.json',
        'published_checkout_compatibility.json':'work/v17_reuse_checkout_provenance_DEBUG_20261005_155956/report.json'}
    reports={}
    for name,source in mappings.items():
        data=(ROOT/source).read_bytes()
        reports[name]=json.loads(data)
        destination=OUT/name
        if destination.exists():
            if destination.read_bytes()!=data:
                raise ValueError('Existing immutable evidence differs: '+name)
        else:
            with destination.open('xb') as stream:
                stream.write(data)
    old,new=reports['before_actual_geometry.json'],reports['after_actual_geometry.json']
    if (old['constructed'] or not new['constructed'] or old['geometry_measurements']!=new['geometry_measurements']
            or old['observation']!=new['observation'] or old['scope']!=new['scope']):
        raise ValueError('Actual unchanged failed geometry replay evidence differs')
    gpu=reports['actual_CUDA_DEBUG.json']
    for name in ('hiercp_v1x/transition_v1_local.py','hiercp_v1x/transition_v1_empty_context.py'):
        expected=(gpu['local_identity']['module_sha256'] if name.endswith('transition_v1_local.py')
                  else gpu['local_identity']['recipient_absence_adapter']['module_sha256'])
        if sha(ROOT/name)!=expected:
            raise ValueError('Actual CUDA executed adapter/local source changed after measurement: '+name)
    if any(not t['changed'] or t['gradient']['missing_parameter_gradients'] for t in gpu['trials']):
        raise ValueError('Actual CUDA native update/gradient evidence incomplete')
    snapshot=ROOT/'work/v14_matched_learning_DEBUG_20261004_r3/source/v1.0'
    archive=ROOT/'versions/v1/pipeline_v1_source.zip'
    archived={}
    with ZipFile(archive) as zipped:
        for item in zipped.infolist():
            if item.is_dir():continue
            digest=hashlib.sha256(zipped.read(item)).hexdigest()
            if sha(snapshot/item.filename)!=digest:
                raise ValueError('Preserved original archive file differs: '+item.filename)
            archived[item.filename]=digest
    if len(archived)!=202:
        raise ValueError('Complete original202-file snapshot required')
    packages=('hiercp_v1x','hiercp_v22','hiercp_v222','l0_regions','l0_local_cnn','l0_exploration','l0_sage','l0_ezsp')
    tracked=subprocess.run(['git','-c','safe.directory='+ROOT.as_posix(),'ls-files','--',*packages],
        cwd=ROOT,check=True,capture_output=True,text=True).stdout.splitlines()
    paths={ROOT/n for n in tracked if n.endswith('.py')}
    paths.update(ROOT.glob('hiercp_v1x/transition_*.py'))
    paths.update(ROOT/n for n in ('tools/run_v17_crossed_training.py','tools/__init__.py',
        'tools/prepare_v17_transition_debug.py','config/v17_crossed_training.json'))
    pending=[p for p in paths if p.suffix=='.py'];scanned=set()
    while pending:
        path=pending.pop()
        if path in scanned:continue
        scanned.add(path)
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8-sig'))):
            names=[]
            if isinstance(node,ast.ImportFrom) and node.module:
                if node.module.startswith('tools.'):names.append(node.module)
                elif node.module=='tools':names.extend('tools.'+n.name for n in node.names)
            elif isinstance(node,ast.Import):names.extend(n.name for n in node.names if n.name.startswith('tools.'))
            for name in names:
                p=ROOT/'tools'/(name.split('.')[1]+'.py')
                if p.is_file() and p not in paths:paths.add(p);pending.append(p)
    extra=('tools/probe_v17_empty_recipient_context.py','tools/verify_v17_empty_recipient_cuda.py',
        'tools/collect_v17_recipient_context_evidence.py','tests/test_transition_v1_empty_context.py',
        'tests/test_transition_recipient_absence_reuse.py','tests/v17_aa28082_sources_UNIT.json',
        'docs/v17_recipient_context_failure_20261005.md','tools/server_v17_crossed.sh')
    code={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(paths)}
    extra_code={n:sha(ROOT/n) for n in extra}
    unique={}
    for log in (OUT/'unit_checks_initial.log',OUT/'reuse_checks_final.log'):
        contents=log.read_text(encoding='utf8',errors='replace')
        if re.search(r'\nOK(?: \(skipped=\d+\))?\s*$',contents) is None:
            raise ValueError('Final UNIT suite did not pass: '+log.name)
        pending=None
        for line in contents.splitlines():
            if line.startswith('test_') and ' ... ' in line:
                pending,status=line.split(' ... ',1)
                if status=='ok' or status.startswith('skipped'):
                    unique[pending]=status;pending=None
            elif pending is not None and line.strip()=='ok':
                unique[pending]='ok';pending=None
        if pending is not None:
            raise ValueError('Final UNIT case status missing: '+pending)
    if not unique or any(not (value=='ok' or value.startswith('skipped')) for value in unique.values()):
        raise ValueError('Final unique UNIT result inventory incomplete')
    result=dict(format='v17_recipient_context_actual_DEBUG_and_contract_evidence_v1',
        failure_observation=old['observation'],actual_geometry_replayed=True,
        full_mask_margin_coordinates_GT_donor_preserved=True,actual_CUDA_native_loss_updates=3,
        CUDA_mixed_observations=32,CUDA_mixed_graph_views=64,
        CUDA_peak_bytes=gpu['trials'][0]['peak_bytes'],parameters=gpu['parameters'],
        UNIT_unique_executed=len(unique),UNIT_pass=sum(v=='ok' for v in unique.values()),
        UNIT_skipped={k:v for k,v in unique.items() if v.startswith('skipped')},
        archived_original_files=archived,archive_sha256=sha(archive),
        runtime_execution_sources=code,additional_evidence_sources=extra_code,
        artifacts={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(OUT.iterdir()) if p.is_file()},
        production_training_started=False,full14102_preparation_verified=False,
        server_40GiB_batch32_admission_verified=False,full21_evaluation_complete=False,
        quality_verified=False,production_checkpoint_created=False,
        original_checkpoints_modified=False,absence_test_ranking_pairs=0,
        absence_test_objectives='actual observation CE, alignment, input view consistency; recipient has zero observed P',
        P_U_ranking_control_pairs=1,scope='local actual CT/CUDA execution and unchanged-completed-record compatibility; not full training or recommendation quality')
    with (OUT/'verification.json').open('x',encoding='utf8') as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps({k:result[k] for k in ('UNIT_unique_executed','UNIT_pass','CUDA_mixed_observations','CUDA_peak_bytes','quality_verified')}))


if __name__=='__main__':main()
