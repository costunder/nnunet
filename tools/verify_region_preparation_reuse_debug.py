"""Actual-CT interrupted preparation recovery; no long training or old-file edits."""
import argparse,copy,json,shutil,sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from l0_regions.training_data import Budget,prepare_cache,dataset,source_identity
from l0_regions.preparation_reuse import PreparedReuse,verified_origin,digest
from l0_regions.data import load
from tools.v22_artifacts import tree_hash


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('old-prepared','cache','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(8);budget=Budget(6*2**30,12*2**30)
    torch.cuda.set_per_process_memory_fraction(budget.cuda_bytes/torch.cuda.get_device_properties(0).total_memory)
    old=a.old_prepared;request=json.loads((old/'request.json').read_text())
    if not request['debug']:raise ValueError('This short verification requires a DEBUG cache')
    baseline={str(f.relative_to(old)):digest(f) for f in old.iterdir() if f.is_file()}
    partial=a.output/'interrupted_copy';partial.mkdir()
    for name in ('request.json','frozen_cnn.pt'):
        shutil.copyfile(old/name,partial/name)
    for f in old.glob('inner_train_*'):shutil.copyfile(f,partial/f.name)
    # A half-written second batch has no audit marker and must be regenerated.
    (partial/'inner_val_000000.pt').write_bytes(b'incomplete DEBUG interrupted output')
    from l0_regions.training_data import prepare
    calls=[]
    def counted(batch,*args,**kwargs):
        calls.append(len(batch));return prepare(batch,*args,**kwargs)
    with patch('l0_regions.training_data.prepare',side_effect=counted), \
         patch('l0_regions.training_data.reference_from_checkpoint',side_effect=OSError(116,'Stale file handle')) as parent:
        index=prepare_cache(a.cache,None,a.output/'recovered',batch=8,workers=8,reg1=.02,view_epoch=0,
            budget=budget,debug=True,profile_policy='research-report',reuse_prepared=partial)
    assert parent.call_count==0 and meta_flag(a.output/'recovered/index.json') is False
    if calls!=[2]:raise AssertionError('Completed train batch was recomputed, or validation was skipped')
    meta=json.loads(index.read_text());proof=meta['reused_preparation']
    assert proof['reused_records']==8 and proof['reused_batches']==1
    assert meta['admission_failures']==10 and sum(map(len,meta['partitions'].values()))==10
    for row in meta['partitions']['inner_train']:
        previous=load(old/row['file'],row['binding']);now=load(index.parent/row['file'],row['binding'])
        assert tree_hash(dict(previous))==tree_hash(dict(now))
    expected=dict(meta);expected['source_identity']=source_identity(preparation=True)
    rejected=[]
    for field,newvalue in [('profile_policy','strict'),('view_epoch',1),('debug',False),('cnn_sha256','changed')]:
        wrong=copy.deepcopy(expected);wrong[field]=newvalue
        try:PreparedReuse(partial,wrong,8)
        except ValueError:rejected.append(field)
        else:raise AssertionError('Accepted changed '+field)
    ds=dataset(a.cache,'inner_train',True)
    with ThreadPoolExecutor(max_workers=8) as pool:
        wrong=PreparedReuse(partial,expected,4)
        try:wrong.restore('inner_train',[0,1,2,3],ds,pool)
        except ValueError:rejected.append('batch_schedule')
        else:raise AssertionError('Accepted partial original batch')
        # Corrupt only the temporary test copy, never the user's original data.
        side=partial/'inner_train_000000.json';payload=json.loads(side.read_text());payload['sha256']='corrupt'
        side.write_text(json.dumps(payload))
        try:PreparedReuse(partial,expected,8).restore('inner_train',list(range(8)),ds,pool)
        except ValueError:rejected.append('file_digest')
        else:raise AssertionError('Accepted corrupt complete record')
    wrong=copy.deepcopy(request['source_identity']);wrong['preparation']['l0_regions/preparation.py']='unknown'
    try:verified_origin(wrong,source_identity(preparation=True))
    except ValueError:rejected.append('unknown_source')
    else:raise AssertionError('Accepted unknown source revision')
    assert baseline=={str(f.relative_to(old)):digest(f) for f in old.iterdir() if f.is_file()}
    report=dict(debug=True,actual_CT=True,status='PASS',gpu=torch.cuda.get_device_name(0),
        peak_cuda_bytes=torch.cuda.max_memory_allocated(),physical_prepare_batch=8,workers=8,
        completed_records_reused=8,missing_records_prepared=2,partition_batch_calls=calls,
        reused_item_content_exact=True,old_files_unchanged=True,rejections=rejected,original_checkpoint_open_calls=parent.call_count,
        original_revision=proof['revision'],all_profile_violations_retained=10,
        source=source_identity(preparation=True),full_training=False,production_ready=False,
        scope='Actual complete DEBUG train8/val2; completed train batch reused without materialize/partition; incomplete val batch regenerated')
    (a.output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report))


def meta_flag(path):return json.loads(path.read_text())['original_checkpoint_reopened']


if __name__=='__main__':main()
