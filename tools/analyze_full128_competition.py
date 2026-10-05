"""Read saved complete V1/A/B/C scores; exact competitor-count diagnostic.

No CT decode, neural forward, training, sampling or checkpoint modification.
All observed P stay present. Fewer U are an analytical subset of fixed scores,
never a replay of the original curriculum eight or a subset-upper forward.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.dont_write_bytecode=True


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):
            digest.update(block)
    return digest.hexdigest()


def analyze_summary(path, *, debug=False):
    from hiercp_v1x.competition_analysis import analyze_competition
    path=Path(path).resolve(strict=True)
    summary=json.loads(path.read_text(encoding='utf8'))
    if (summary.get('format')!='historical_V1_A_B_C_full128_readonly_evaluation_v1'
            or summary.get('complete') is not True or summary.get('debug') is not debug
            or summary.get('full_evaluation') is not (not debug)
            or summary.get('training_started') is not False
            or summary.get('same_native_query_cohort') is not True
            or set(summary.get('arms',{}))!={'V1','A','B','C'}):
        raise ValueError('Complete same-cohort historical summary and explicit DEBUG identity required')
    request_path=path.with_name('request.json')
    request=json.loads(request_path.read_text(encoding='utf8'))
    request_digest=hashlib.sha256(json.dumps(request,sort_keys=True,
        separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
    if (request.get('format')!=summary['format'] or request.get('debug') is not debug
            or request.get('training_started') is not False):
        raise ValueError('Saved original evaluation request differs')
    old={str(path):sha(path),str(request_path):sha(request_path)};results={};cohorts=set()
    identities=[]
    for arm in ('V1','A','B','C'):
        entry=summary['arms'][arm]
        report_path=Path(entry['report']).resolve(strict=True)
        if report_path!=path.parent/arm/'report.json':
            raise ValueError('Arm report is outside its original summary directory')
        seal_path=report_path.with_name('complete.json')
        seal=json.loads(seal_path.read_text(encoding='utf8'))
        report=json.loads(report_path.read_text(encoding='utf8'))
        if (seal.get('report_sha256')!=sha(report_path) or seal.get('arm')!=arm
                or seal.get('request_sha256')!=request_digest
                or report.get('evaluation_request_sha256')!=request_digest
                or report.get('arm')!=arm or report.get('actual_CUDA') is not True
                or report.get('execution_contract_bound') is not True
                or report.get('old_evidence_preserved') is not True
                or report.get('training_started') is not False
                or report.get('optimizer_updates')!=0
                or entry['metrics']!=report['metrics']
                or entry['denominators']!=report['denominators']
                or entry['checkpoint']!=report['checkpoint']):
            raise ValueError('Saved report bytes/completion/summary/checkpoint receipt differ')
        old.update({str(report_path):sha(report_path),str(seal_path):sha(seal_path)})
        cohorts.add(report['cohort']['cohort_sha256'])
        results[arm]=analyze_competition(report,debug=debug)
        identities.append({case['case_id']:{row['record_id']:(row['candidate_key'],row['observed'])
            for row in case['case_scores']} for case in report['cases']})
    if len(cohorts)!=1:
        raise ValueError('Arms have different saved query cohorts')
    if any(identity!=identities[0] for identity in identities[1:]):
        raise ValueError('Arms have different saved candidate identities or observed GT')
    if any(sha(p)!=digest for p,digest in old.items()):
        raise ValueError('Original evaluation artifacts changed while being read')
    return dict(format='historical_fixed_score_competition_analysis_v1',debug=debug,
        original_summary=str(path),original_files_sha256=old,arms=results,
        neural_forward_executed=False,training_started=False,optimizer_updates=0,
        original_artifacts_written=False,quality_verified=False,
        interpretation='All P retained; exact expectation over uniformly chosen k U with full-case scores/order fixed. Not original curriculum8 replay or subset-upper inference.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--summary',type=Path,required=True)
    p.add_argument('--output',type=Path)
    p.add_argument('--debug',action='store_true')
    a=p.parse_args()
    report=analyze_summary(a.summary,debug=a.debug)
    if a.output is not None:
        path=a.output.resolve()
        original=a.summary.resolve(strict=True).parent
        if path==original or path.is_relative_to(original):
            raise ValueError('Choose a NEW analysis file outside the preserved evaluation directory')
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('x',encoding='utf8') as stream:
            json.dump(report,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
    print('FIXED-SCORE diagnostic | all P kept | no neural forward/training',flush=True)
    print('Original curriculum8 and subset-upper execution are not reproduced.',flush=True)
    arms=report['arms']
    print('Denominators: '+json.dumps(arms['V1']['denominators'],ensure_ascii=False),flush=True)
    print('U kept | V1 MRR / Hit@1    | A MRR / Hit@1     | B MRR / Hit@1     | C MRR / Hit@1',flush=True)
    for i,row in enumerate(arms['V1']['curve']):
        cells=[]
        for arm in ('V1','A','B','C'):
            item=arms[arm]['curve'][i]
            if item['unobserved_U']!=row['unobserved_U']:
                raise ValueError('Diagnostic curves changed the competitor-count grid')
            cells.append(f"{item['expected_case_first_P_mrr']:.6f} / {item['expected_case_hit_at_1']:.6f}")
        print(f"{row['unobserved_U']:>6} | "+' | '.join(cells),flush=True)
    if a.output is not None:print('REPORT: '+str(a.output.resolve()),flush=True)


if __name__=='__main__':main()
