"""Own Basic ablation full250/BEST26 result schema1 and exclusive archive.

No metric math, training, prediction, source result overwrites or deletions run
here. Original Basic BEST linkage remains unverified in every comparison.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import time


ROOT = Path(__file__).resolve().parents[1]
ARMS = {'multi3': 'three lesions', 'candidates256': '256 positions', 'blend': 'boundary blending'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    value = json.loads(Path(path).read_text(encoding='utf8'))
    require(isinstance(value, dict), 'JSON object required: ' + str(path))
    return value


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*2**20), b''):
            digest.update(block)
    return digest.hexdigest()


def publish(path, value):
    with Path(path).open('x', encoding='utf8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def exact(path, root=None, *, directory=False):
    path = Path(path).absolute()
    require(path.resolve(strict=True) == path and not any(p.is_symlink() for p in (path,*path.parents)),
            'Exact nonsymlink result path required: ' + str(path))
    require(path.is_dir() if directory else path.is_file(), 'Result artifact missing: ' + str(path))
    if root is not None:
        require(Path(root).absolute() in path.parents, 'Artifact is outside this own experiment: ' + str(path))
    return path


def stat(path):
    value = Path(path).stat()
    return [value.st_dev,value.st_ino,value.st_size,value.st_mtime_ns,value.st_ctime_ns,value.st_nlink]


def plain(value):
    if hasattr(value, 'tolist'):
        value = value.tolist()
    elif hasattr(value, 'item'):
        value = value.item()
    if isinstance(value,(list,tuple)):
        return [plain(item) for item in value]
    if isinstance(value,dict):
        return {key:plain(item) for key,item in value.items()}
    require(not isinstance(value,float) or math.isfinite(value), 'Nonfinite native training scalar')
    return value


def training_record(saved, best, log):
    require(saved.get('current_epoch') == 250, 'Actual native FINAL completion witness must be250')
    logging = saved.get('logging')
    require(isinstance(logging,dict), 'Actual native checkpoint logger required')
    result = dict(planned_epochs=250,completed_epochs=250,best_current_epoch_raw=best['best_current_epoch'],
        best_selection_rule='original nnU-Net best EMA foreground Dice',
        best_epoch_index_definition='raw current_epoch saved by original nnU-Net; not relabelled',
        measurement_scope='own full250 FINAL checkpoint logger and root/train.log OnlineCP counters')
    for key,target in [('train_losses','train_loss'),('val_losses','val_loss'),
                       ('dice_per_class_or_region','online_pseudo_dice')]:
        values = logging.get(key,[])
        require(len(values) == 250, 'Full250 native logging entries required: ' + key)
        result[target] = plain(values[-1])
    result['latest_loss_epoch'] = 249
    starts,ends = logging.get('epoch_start_timestamps',[]),logging.get('epoch_end_timestamps',[])
    if len(starts) == len(ends) == 250:
        require(all(math.isfinite(float(a)) and math.isfinite(float(b)) and b>=a for a,b in zip(starts,ends)),
                'Native epoch timestamps invalid')
        result.update(training_seconds=sum(float(b)-float(a) for a,b in zip(starts,ends)),
            started_at=plain(starts[0]),ended_at=plain(ends[-1]),
            epoch_time=dict(measured_epochs=250,source='original saved epoch timestamps'))
    counts = {}
    for epoch,applied,attempted in re.findall(r'\[OnlineCP\] epoch=(\d+) applied=(\d+)/(\d+)',
                                             Path(log).read_text(encoding='utf8')):
        epoch,applied,attempted = int(epoch),int(applied),int(attempted)
        require(attempted > 0 and 0 <= applied <= attempted, 'Original OnlineCP sample counters invalid')
        require(epoch not in counts or counts[epoch] == (applied,attempted), 'Conflicting OnlineCP epoch counts')
        counts[epoch] = (applied,attempted)
    require(set(counts) == set(range(250)), 'Own train.log must contain all250 OnlineCP epoch counters0..249')
    applied,attempted = sum(v[0] for v in counts.values()),sum(v[1] for v in counts.values())
    result.update(cp_applied=applied,cp_attempted=attempted,cp_apply_rate=applied/attempted,
        cp_count_scope='original augmented sample counter; multi3 places three lesions per successful sample',
        cp_counter_source=str(log),cp_counter_source_sha256=sha(log))
    pastes = {}
    for epoch,arm,pasted,events,border in re.findall(
            r'\[BasicCPAblation\] epoch=(\d+) arm=(\w+) pasted_lesions=(\d+) events=(\d+) blend_border_voxels=(\d+)',
            Path(log).read_text(encoding='utf8')):
        row = (arm,int(pasted),int(events),int(border))
        epoch = int(epoch)
        require(epoch not in pastes or pastes[epoch] == row,'Conflicting Basic ablation paste telemetry')
        pastes[epoch] = row
    if pastes:
        require(set(pastes) == set(range(250)), 'Partial ablation paste telemetry must not look complete')
        for epoch,(arm,pasted,events,border) in pastes.items():
            require(arm in ARMS and events == counts[epoch][0]
                    and pasted == events*(3 if arm=='multi3' else 1), 'Actual paste count differs from successful augmentation events')
        require(len({row[0] for row in pastes.values()}) == len({row[3] for row in pastes.values()}) == 1,
                'Abalation arm/blend width changed during training')
        result.update(pasted_lesions=sum(row[1] for row in pastes.values()),
            actual_ablation_arm=next(iter(pastes.values()))[0],blend_border_voxels=next(iter(pastes.values()))[3])
    return result


def validate_completed(native, proof):
    from hiercp_v1x import v24_native_best_eval_runtime as guard
    require(native['arm'] in ARMS, 'Unknown Basic one-factor arm')
    root,native_root = exact(native['root'],directory=True),exact(native['native_root'],directory=True)
    expected = native['split']['outer_val']
    require(len(native['split']['outer_train']) == 105 and len(expected) == len(set(expected)) == 26,
            'Original full105train/26validation split required')
    require(proof.get('epochs') == 250 and proof.get('checkpoint') == guard.BEST
            and proof.get('evaluation_checkpoint_is_final') is False,
            'Own full250/BEST26 completion proof required; FINAL is training witness only')
    certificate = exact(root/'training_and_BEST26_complete.json',root)
    require(read(certificate) == proof, 'Caller completion differs from saved own training receipt')
    fold = exact(proof['fold'],native_root,directory=True)
    receipt_path = exact(fold/guard.RECEIPT,native_root)
    best = guard.validate_best_validation_receipt(receipt_path,fold=fold,
        expected_cases=expected,bank_sha256=native['bank_sha256'])
    require(proof['best_sha256'] == best['checkpoint_sha256']
            and sorted(proof['validation_cases']) == sorted(expected), 'Own BEST/cohort proof differs')
    final = exact(fold/'checkpoint_final.pth',native_root)
    require(sha(final) == proof['final_training_witness_sha256'], 'Own FINAL training witness changed')
    model = read(exact(root/'actual_model.json',root))
    require(model.get('physical_batch') == model.get('effective_batch') == 2
            and model.get('gradient_accumulation') == 1 and model.get('epochs') == 250
            and model.get('patch_size') == [128]*3 and model.get('debug') is False
            and model.get('subset') is False and model.get('pretrained_segmentation_used') is False
            and sorted(model.get('actual_train_cases',[])) == sorted(native['split']['outer_train'])
            and sorted(model.get('actual_validation_cases',[])) == sorted(expected),
            'Actual native full-model matched configuration differs')
    return root,native_root,fold,best,model


def completed_report(native, proof):
    import torch
    from hiercp_v1x.v25_downstream import _summary_evidence
    from tools import render_nnunet_result_schema_v1 as renderer
    root,native_root,fold,best,model = validate_completed(native,proof)
    name = 'Basic CP ' + ARMS[native['arm']] + ' (GPU' + str(native['gpu']) + ')'
    log = exact(root/'train.log',root)
    saved = torch.load(fold/'checkpoint_final.pth',map_location='cpu',weights_only=False)
    train = training_record(saved,best,log)
    del saved
    pre = Path(native['private_pre'])
    dataset = read(exact(pre/'dataset.json',native_root))
    bank = read(native['bank'])
    require(sha(native['bank']) == native['bank_sha256'] and bank.get('source_entries') == 642
            and bank.get('eligible_cases') == 81 and bank.get('cp_probability') == .5
            and bank.get('candidate_count') == (256 if native['arm']=='candidates256' else 128),
            'Actual original Basic source/gate/arm candidate population differs')
    evaluation = exact(root/'evaluation',root,directory=True)
    schema = read(renderer.SCHEMA_PATH)
    online = renderer._online({'directory':str(evaluation),'experiment':name,'status':'VERIFIED_COMPLETE'},schema)
    require(online['status'] == 'VERIFIED_COMPLETE', 'Actual full26 historical evaluation incomplete')
    summary = online['summary']
    require(summary.get('validation_cases') == 26 and summary.get('dataset_id') == 730
            and summary.get('outer_fold') == 0 and sorted(summary['evaluation_contract']['cohort']) == sorted(best['validation_cases'])
            and Path(summary['hier_validation']).absolute() == fold/'validation',
            'Historical evaluator cohort/prediction namespace differs from ownBEST26')
    own_inventory = {item['case_id']:item for item in summary['evaluation_contract']['file_inventory'] if item['role']=='hier'}
    require(set(own_inventory) == set(best['validation_cases'])
            and all(row['sha256'] == best['predictions_sha256'][case]
                and Path(row['path']).absolute() == fold/'validation'/(case+'.nii.gz')
                for case,row in own_inventory.items()), 'Historical evaluator input bytes differ from actual own BEST predictions')
    basic = dict(checkpoint_origin_status='HISTORICAL_REFERENCE / original training-BEST linkage unverified',
        native=dict(dataset_labels=dataset['labels'],verification=dict(best_receipt_verified=False,
            source=str(evaluation/'completion.json'),reason='Historical Basic original checkpoint-to-prediction linkage remains unverified')))
    basic_summary = Path(summary['basic_validation'])/'summary.json'
    if basic_summary.is_file():
        basic['native'].update(_summary_evidence(basic_summary))
    native_info = dict(status='VERIFIED_COMPLETE',receipt=best,**_summary_evidence(fold/'validation/summary.json'),
        dataset_labels=dataset['labels'],dataset_labels_provenance=dict(path=str(pre/'dataset.json'),sha256=sha(pre/'dataset.json')),
        verification=dict(best_receipt_verified=True,source=str(fold/'best_validation_complete.json')),
        training_receipt=dict(path=str(fold/'checkpoint_final.pth'),sha256=proof['final_training_witness_sha256']))
    cp = dict(bank= native['bank'],bank_sha256=native['bank_sha256'],eligible_cases=81,source_entries=642,
        candidates_per_source=bank['candidate_count'],cp_probability=.5,source_policy='uniform own-patient source; original eligibility/source arrays',
        location_policy='uniform complete candidate universe; GNN scores unused',
        copied_lesions_per_successful_sample=3 if native['arm']=='multi3' else 1,
        appearance=read(root/'preflight.json').get('appearance'),
        intensity_scale_range=bank['intensity_scale_range'],intensity_shift_range_hu=bank['intensity_shift_range_hu'],
        ablation=native['arm'],only_intervention=ARMS[native['arm']])
    evidence = dict(kind='ACTUAL_COMPLETED_BASIC_CP_ONE_FACTOR_FULL250_OWN_BEST26',
        experiments=[dict(experiment=name,completion=dict(training_status='VERIFIED_COMPLETE',completed_epochs=250,
            planned_epochs=250,best_native_evaluation_status='VERIFIED_COMPLETE',predicted_cases=26,expected_cases=26,
            lesion_metrics_status='VERIFIED_COMPLETE'),training=train,native=native_info,
            settings=dict(experiment=dict(arm=native['arm'],physical_GPU=native['gpu'],one_factor=True),
                gnn=dict(used_for_position_selection=False),cp=cp,nnunet=model))],basic=basic,
        online_evaluation=dict(experiment=name,directory=str(evaluation),status='VERIFIED_COMPLETE'),
        comparison=dict(status='HISTORICAL_REFERENCE',source=str(evaluation/'completion.json'),
            reason='Historical Basic prediction/cohort bytes verified; original Basic BEST-to-prediction linkage unverified'),
        verification=dict(full_training=True,full_best_native_evaluation=True,lesion_metrics=True,
            static_checks='Own admitted code and runtime source hashes preserved',
            unit_smoke_checks='Own separate full-model CUDA B2/B4 calibration; no debug steps count as production epochs'),
        limitations=['Historical Basic original BEST linkage remains unverified; historical arithmetic statistics retained separately from matched comparisons',
            'Outer26 was used for validation/model selection during training; no independent test claim',
            'Raw native undefined/NaN values remain in original summary; only report copies use null with recorded reasons',
            'Unrecorded full-training peak memory stays NA; debug calibration memory is not substituted',
            'New experiment results and archive are in volatile tmpfs; no durable persistence is claimed'],
        artifacts=[dict(path=str(path),sha256=sha(path)) for path in [root/'training_and_BEST26_complete.json',
            fold/'best_validation_complete.json',root/'actual_model.json',root/'preflight.json',root/'calibration.json',log]])
    report,_ = renderer.render_report(evidence)
    require({key:len(report['tables'][key]) for key in ['whole_lesion_detection','size_stratified_detection','basic_cp_comparison']}
            == {'whole_lesion_detection':12,'size_stratified_detection':36,'basic_cp_comparison':44}, 'Fixed Result schema1 row inventory changed')
    require(len(report['historical_reference_comparison']) == 44 and all(row['experiment_minus_basic'] is None
            for row in report['tables']['basic_cp_comparison']), 'Unverified Basic provenance cannot become matched statistics')
    publish(root/'result_evidence.json',evidence)
    text,document = renderer.write_report(evidence,root/'result_schema1')
    return {key:dict(path=str(path),sha256=sha(path)) for key,path in
            [('evidence',root/'result_evidence.json'),('text',text),('json',document)]}


def archive_completed(native, proof, reports):
    root,native_root,fold,_,_ = validate_completed(native,proof)
    destination = Path(native.get('archive_root',str(native_root/'completed_native_archive'))).absolute()
    require(native_root in destination.parents and destination.resolve() == destination
            and not any(path.is_symlink() for path in (destination,*destination.parents)),
            'Archive must remain in this fresh own native namespace')
    require(native.get('archive_storage_kind') == 'tmpfs' and native.get('persistent_archive') is False,
            'Declared tmpfs archive must not claim durable persistence')
    require(not destination.exists(), 'Fresh own archive required; no existing result overwritten')
    sources = {}
    def add(path,relative):
        path = exact(path)
        require(root in path.parents or native_root in path.parents, 'Only this new experiment artifacts may be archived')
        require(relative not in sources, 'Duplicate archive destination')
        sources[relative] = path
    for name in ['checkpoint_best.pth','checkpoint_final.pth','best_validation_complete.json']:
        add(fold/name,'fold_0/'+name)
    for path in sorted((fold/'validation').rglob('*')):
        if path.is_file(): add(path,'fold_0/validation/'+path.relative_to(fold/'validation').as_posix())
    for path in sorted((root/'evaluation').rglob('*')):
        if path.is_file(): add(path,'evaluation/'+path.relative_to(root/'evaluation').as_posix())
    for field in ['evidence','text','json']:
        witness = reports[field]
        path = exact(witness['path'],root)
        require(sha(path) == witness['sha256'], 'Published completed report changed')
        add(path,'reports/'+path.name)
    for name in ['actual_model.json','preflight.json','calibration.json','training_and_BEST26_complete.json','train.log','native.json']:
        add(root/name,'experiment/'+name)
    measured = sum(path.stat().st_size for path in sources.values())
    reserve = int(native['archive_reserve_bytes'])
    required_free = max(int(native['persistent_required_free_bytes']),int(native.get('archive_required_free_bytes',0)))
    require(reserve > 0 and required_free >= 0, 'Explicit persistent archive/storage budget required')
    free = shutil.disk_usage(native_root).free
    require(measured <= reserve, f'Own BEST+FINAL+predictions/results archive requires {measured} bytes; admitted reserve is {reserve}; no files dropped')
    require(free >= required_free+measured, f'Archive copy would consume required persistent reserve: free{free},copy{measured},reserve{required_free}')
    inventory = {name:dict(path=str(path),sha256=sha(path),bytes=path.stat().st_size,stat=stat(path)) for name,path in sources.items()}
    publish(root/'archive_budget_preflight.json',dict(measured_bytes=measured,archive_reserve_bytes=reserve,
        persistent_required_free_bytes=required_free,free_bytes_before=free,source_files=len(sources),
        checkpoint_best_and_final_included=True,original_results_preserved=True))
    destination.mkdir()
    rows = []
    for name,path in sources.items():
        target = destination/name
        target.parent.mkdir(parents=True,exist_ok=True)
        with path.open('rb') as source,target.open('xb') as result:
            shutil.copyfileobj(source,result,length=8*2**20)
            result.flush()
            os.fsync(result.fileno())
        require(stat(path) == inventory[name]['stat'] and sha(path) == inventory[name]['sha256']
                and sha(target) == inventory[name]['sha256'] and stat(target)[:2] != stat(path)[:2],
                'Archive must contain independent byte-exact copies of unchanged own sources')
        rows.append(dict(relative=name,source=str(path),path=str(target),sha256=inventory[name]['sha256'],bytes=inventory[name]['bytes']))
    require(shutil.disk_usage(native_root).free >= required_free, 'Required shared tmpfs reserve was lost during archive copy')
    receipt = dict(format='basic_cp_ablation_completed_native_archive_v1',complete=True,
        archive_root=str(destination),entries=rows,bytes=measured,created_at=time.time(),
        original_results_preserved=True,independent_copies=True,checkpoint_best_and_final_included=True,
        archive_storage_kind='tmpfs',persistent_archive=False,durable_persistence_claimed=False,
        best_evaluation_checkpoint='checkpoint_best.pth',final_role='training completion witness only',
        full_native_training_epochs=250,full_native_BEST_validation_cases=26)
    publish(destination/'archive_complete.json',receipt)
    return dict(path=str(destination/'archive_complete.json'),sha256=sha(destination/'archive_complete.json'),
        archive_root=str(destination),bytes=measured,files=len(rows))
