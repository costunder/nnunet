"""Recovery/failure probes on an existing actual-CT DEBUG cache; no training."""
import argparse
import json
import sys
from pathlib import Path
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('cache','prepared','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    import torch
    from l0_regions import training_data as td
    from l0_regions.preparation_reuse import PreparedReuse,verified_origin
    from tools.v22_artifacts import tree_hash
    a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(8)
    opts=dict(batch=8,workers=8,reg1=.02,view_epoch=0,budget=td.Budget(6*2**30,12*2**30),
        debug=True,profile_policy='research-report',reuse_prepared=a.prepared)
    old={p.name:td.sha(p) for p in a.prepared.iterdir() if p.is_file()}
    with patch.object(td,'prepare',side_effect=AssertionError('Completed batch recomputed')), \
         patch.object(td,'reference_from_checkpoint',side_effect=OSError(116,'Stale file handle')):
        index=td.prepare_cache(a.cache,None,a.output/'legacy_chain_reused',**opts)
    meta=json.loads(index.read_text());assert meta['reused_preparation']['reused_records']==10
    for row in sum(meta['partitions'].values(),[]):
        before=torch.load(a.prepared/row['file'],weights_only=True)
        after=torch.load(index.parent/row['file'],weights_only=True)
        assert tree_hash(before)==tree_hash(after),'Reused item contents changed'
    failed=a.output/'injected_save_error'
    with patch.object(td,'save_new',side_effect=OSError('DEBUG injected storage failure')):
        try:td.prepare_cache(a.cache,None,failed,**opts)
        except OSError as exc:assert 'DEBUG injected' in str(exc)
        else:raise AssertionError('Save exception swallowed')
    assert not (failed/'index.json').exists() and not list(failed.glob('*_audit.json'))
    real=PreparedReuse.restore
    def broken(self,part,ids,ds,pool):
        raise ValueError('DEBUG injected input corruption')
    failed_read=a.output/'injected_read_error'
    with patch.object(PreparedReuse,'restore',broken):
        try:td.prepare_cache(a.cache,None,failed_read,**opts)
        except ValueError as exc:assert 'DEBUG injected' in str(exc)
        else:raise AssertionError('Read exception swallowed')
    assert not (failed_read/'index.json').exists() and not list(failed_read.glob('*_audit.json'))
    import copy
    source=json.loads((a.prepared/'request.json').read_text())['source_identity']
    for filename in ('l0_regions/preparation.py','l0_ezsp/partition.py'):
        wrong=copy.deepcopy(source);wrong['preparation'][filename]='unknown'
        try:verified_origin(wrong,td.source_identity(preparation=True))
        except ValueError:pass
        else:raise AssertionError('Unknown source accepted')
    assert old=={p.name:td.sha(p) for p in a.prepared.iterdir() if p.is_file()}
    report=dict(debug=True,actual_CT=True,status='PASS',full_training=False,
        legacy_chained_reused_records=10,original_files_unchanged=True,reused_items_bitwise_equal=True,
        errors_propagated=['writer','prefetch','unknown_packaging_source','changed_partition_source'],
        false_completed_indices=0,failed_batch_audit_markers=0,original_checkpoint_reopened=False)
    (a.output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
