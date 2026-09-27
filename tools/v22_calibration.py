"""Admit only measured calibration from the exact current execution contract."""
import math


def finite_numbers(value):
    if isinstance(value,float) and not math.isfinite(value):raise ValueError('Nonfinite calibration measurement')
    if isinstance(value,dict):
        for item in value.values():finite_numbers(item)
    elif isinstance(value,list):
        for item in value:finite_numbers(item)


def validate_calibration(old,reports,*,identity,parameters,train_samples,val_samples,gpu,runtime):
    finite_numbers(old);finite_numbers(reports)
    keys=('artifact_contract','geometry_contract','source_identity','cache_sha256','ranking_contract',
          'training_objective','feature_coordinates','support_task_contract','config','base','debug')
    if any(old.get(k)!=identity[k] for k in keys):raise ValueError('Calibration semantic/source/cache/config contract differs')
    if old.get('calibration_runtime_sha256')!=runtime:
        raise ValueError('Calibration runtime changed: remeasure; no execution migration is approved')
    if (old.get('parameters'),old.get('train_samples'),old.get('val_samples'),old.get('gpu'))!=(parameters,train_samples,val_samples,gpu):
        raise ValueError('Calibration model/cohort/device differs')
    for name,key in [('memory_batch_calibration.json','memory_physical_batch'),('training_batch_calibration.json','physical_batch')]:
        rows=reports.get(name)
        if not isinstance(rows,list) or not rows:raise ValueError('Nonempty batch calibration list required')
        measured=[];seen=set()
        for row in rows:
            if not isinstance(row,dict):raise ValueError('Malformed calibration row')
            count=row.get('physical_batch')
            if type(count)!=int or count<1 or count in seen or type(row.get('accepted'))!=bool:raise ValueError('Invalid/duplicate batch measurement')
            seen.add(count)
            if row['accepted']:
                if row.get('executed') is not True or row.get('training')!=(key=='physical_batch'):
                    raise ValueError('Accepted batch was not measured in the correct phase')
                for field in ('graphs_per_second','peak_vram_bytes','budget_bytes','nodes','edges'):
                    if not isinstance(row.get(field),(float,int)) or row[field]<=0:raise ValueError('Missing positive calibration measurement')
                if row['peak_vram_bytes']>=row['budget_bytes']:raise ValueError('Accepted batch exceeds measured budget')
                measured.append(row)
        if not measured or type(old.get(key))!=int or old[key]!=max(measured,key=lambda r:r['graphs_per_second'])['physical_batch']:
            raise ValueError('Selected batch is not the measured admissible throughput winner')
    loader=reports.get('loader_calibration.json')
    if not isinstance(loader,dict) or not isinstance(loader.get('reports'),list) or not loader['reports']:
        raise ValueError('Measured loader report required')
    seen=set()
    for row in loader['reports']:
        if type(row.get('workers'))!=int or row['workers']<0 or row['workers'] in seen:raise ValueError('Invalid worker candidate')
        seen.add(row['workers'])
        for field in ('seconds','warm_seconds','graphs_per_batch','batches_per_pass','producers'):
            if not isinstance(row.get(field),(float,int)) or row[field]<=0:raise ValueError('Invalid worker measurement')
    selected=min(loader['reports'],key=lambda r:r['warm_seconds'])['workers']
    if type(old.get('workers'))!=int or old['workers']!=selected or loader.get('selected')!=selected:
        raise ValueError('Selected worker was not the measured winner')
    return reports
