"""CPU-only rolling durable rescue cache; original experiments remain immutable.

One CURRENT.pth per GPU. A native FINAL250 and actual own BEST26 receipt are
required before CURRENT becomes an evaluated BEST; that BEST is never labeled
as a training resume at epoch250. Only this fresh cache's operational files may
be replaced. Source checkpoint inode is held through validation, hashing/copy.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import socket
import stat
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FORMAT = 'basic_cp_seven_single_checkpoint_durable_cache_v1'
BANK_SHA = '9e5be3e3ab8f04fd23c166b3267bc4efbfb4b321ef974920dfc08fbf5dd275e6'
UID, NODE, GIB = 41833, 'ece-a6gpu6', 2**30
LOGGERS = {'mean_fg_dice', 'ema_fg_dice', 'dice_per_class_or_region', 'train_losses',
           'val_losses', 'lrs', 'epoch_start_timestamps', 'epoch_end_timestamps'}
ARMS = {0:'multi5', 1:'multi7', 2:'multi10', 3:'candidates512',
        4:'candidates1024', 5:'candidates2048', 6:'blend50'}
BEST_VALIDATORS={}


def require(condition, message):
    if not condition: raise ValueError(message)


def exact(path, *, directory=False):
    path = Path(path).absolute()
    require(path.resolve(strict=True) == path and not any(p.is_symlink() for p in (path, *path.parents)),
            'Exact non-symlink evidence path required: '+str(path))
    row = path.stat()
    require(row.st_uid == UID and (stat.S_ISDIR(row.st_mode) if directory else stat.S_ISREG(row.st_mode)),
            'Own UID regular evidence required: '+str(path))
    return path


def read(path):
    value = json.loads(Path(path).read_bytes())
    require(isinstance(value, dict), 'JSON object required: '+str(path))
    return value


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*2**20), b''): h.update(block)
    return h.hexdigest()


def identity(row):
    # Unlinking an atomically superseded pathname changes ctime/link count,
    # while the held inode's bytes and mtime remain unchanged.
    return (row.st_dev, row.st_ino, row.st_size, row.st_mtime_ns)


def fsync_directory(path):
    fd = os.open(path, os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)


def publish(path, value):
    path = Path(path); temporary = path.with_name('.'+path.name+'_'+str(time.time_ns()))
    with temporary.open('x', encoding='utf8') as f:
        json.dump(value, f, ensure_ascii=False, allow_nan=False, indent=2); f.flush(); os.fsync(f.fileno())
    os.replace(temporary, path)
    fsync_directory(path.parent)


def checkpoint_info(value, trainer):
    epoch = value.get('current_epoch'); logging = value.get('logging', {})
    require(type(epoch) is int and 0 < epoch <= 250, 'Native completed checkpoint epoch1..250 required')
    require(set(logging) == LOGGERS and all(isinstance(v, (list, tuple)) and len(v) == epoch for v in logging.values()),
            'All eight native logger arrays must match the whole completed epoch')
    require(all(math.isfinite(float(a)) and math.isfinite(float(b)) and float(b) >= float(a)
                for a,b in zip(logging['epoch_start_timestamps'], logging['epoch_end_timestamps'])),
            'Every completed epoch requires finite ordered start/end timestamps')
    weights, optimizer, scaler = (value.get(k) for k in ('network_weights','optimizer_state','grad_scaler_state'))
    require(isinstance(weights, dict) and weights and all(hasattr(v,'numel') for v in weights.values()),
            'Actual full native tensor network state required')
    require(isinstance(optimizer, dict) and optimizer.get('state') and optimizer.get('param_groups'),
            'Full native optimizer state and parameter groups required')
    scaler_fields={'scale','growth_factor','backoff_factor','growth_interval','_growth_tracker'}
    require(isinstance(scaler, dict) and scaler_fields.issubset(scaler)
            and math.isfinite(float(scaler['scale'])) and float(scaler['scale'])>0
            and all(math.isfinite(float(scaler[k])) and float(scaler[k])>0 for k in ('growth_factor','backoff_factor'))
            and type(scaler['growth_interval']) is int and scaler['growth_interval']>0
            and type(scaler['_growth_tracker']) is int and scaler['_growth_tracker']>=0,
            'Actual native CUDA GradScaler state required')
    require(value.get('trainer_name') == trainer and isinstance(value.get('init_args'),dict)
            and '_best_ema' in value and 'inference_allowed_mirroring_axes' in value,
            'Original native trainer/initialization/inference state required')
    return dict(checkpoint_completed_epoch=epoch, logger_lengths={k:len(v) for k,v in logging.items()},
                optimizer_states=len(optimizer['state']), scaler_present=True,
                network_tensor_elements=sum(v.numel() for v in weights.values()))


class Cache:
    def __init__(self, root, expiry):
        self.root, self.expiry = Path(root), expiry

    def guard(self, incoming=0):
        import psutil
        require(time.time() < self.expiry-180, 'Host lifetime reached its180 second persistence reserve')
        require(psutil.Process().memory_info().rss <= 4*GIB, 'CPU persistence RSS exceeds4GiB')
        used = sum(p.stat().st_size for p in self.root.rglob('*') if p.is_file())
        require(used+incoming <= 8*GIB, 'Durable cache including atomic temporary exceeds8GiB')
        require(shutil.disk_usage(self.root).free >= incoming+GIB, 'Durable cache requires1GiB free after atomic temporary')

    def copy(self, source, target, *, expected=None, trainer=None, role=None, extra=None, append_only=False):
        """Hold one source FD; allow atomic name advancement, reject inode mutation."""
        source, target = Path(source), Path(target); target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name('.'+target.name+'_'+str(time.time_ns()))
        self.guard(source.stat().st_size)
        if self.root/'banks' in target.parents:
            used=sum(p.stat().st_size for p in (self.root/'banks').rglob('*') if p.is_file())
            require(used+source.stat().st_size<=500*2**20, 'Durable banks including atomic temporary exceed500MiB')
        try:
            with source.open('rb') as inp:
                before = os.fstat(inp.fileno()); proof = {}
                if trainer is not None:
                    import torch
                    value = torch.load(inp, map_location='cpu', weights_only=False)
                    try: proof = checkpoint_info(value, trainer)
                    finally: del value; gc.collect()
                    require(not torch.cuda.is_initialized(), 'Persistence must never initialize CUDA')
                    require(not(extra or {}).get('required_epoch') or proof['checkpoint_completed_epoch']==extra['required_epoch'],
                            'FINAL full250 epoch differs; previous CURRENT remains unchanged')
                    require(proof['network_tensor_elements'] >= (extra or {}).get('minimum_network_tensor_elements',0),
                            'Checkpoint network state is smaller than its actual full-model parameter witness')
                self.guard(before.st_size); inp.seek(0); h = hashlib.sha256()
                with temporary.open('xb') as out:
                    remaining=before.st_size
                    while remaining:
                        block=inp.read(min(8*2**20,remaining))
                        require(block, 'Source ended before its admitted snapshot length')
                        self.guard(); h.update(block); out.write(block)
                        remaining-=len(block)
                    out.flush(); os.fsync(out.fileno())
                after=os.fstat(inp.fileno())
                if append_only:
                    require((before.st_dev,before.st_ino)==(after.st_dev,after.st_ino) and after.st_size>=before.st_size,
                            'Append-only log inode/prefix disappeared')
                    inp.seek(0); remaining=before.st_size; prefix=hashlib.sha256()
                    while remaining:
                        block=inp.read(min(8*2**20,remaining)); require(block, 'Log prefix disappeared')
                        prefix.update(block); remaining-=len(block)
                    require(prefix.hexdigest()==h.hexdigest(), 'Append-only log prefix changed')
                else: require(identity(before) == identity(after), 'Source inode changed during snapshot')
                checksum = h.hexdigest()
                require(expected is None or checksum == expected, 'Pinned source checksum differs')
                require(digest(temporary) == checksum, 'Durable copied bytes differ from held source FD')
                proof.update(format=FORMAT, source=str(source), sha256=checksum, bytes=before.st_size,
                             source_inode=list(identity(before)), source_named_path_advanced=identity(source.stat())!=identity(before),
                             copied_at_unix=time.time(), **(extra or {}))
                if trainer is not None and proof.get('checkpoint_is_training_resume',role=='TRAINING_LATEST'):
                    proof['saved_resume_epoch']=proof['checkpoint_completed_epoch']
                if role=='OWN_BEST_TRAINING_STATE':
                    proof['saved_resume_epoch']=proof['checkpoint_completed_epoch']
                    latest=proof['latest_on_tmpfs_N']
                    proof['epochs_may_repeat_after_tmpfs_loss']=None if latest is None else max(0,latest-proof['saved_resume_epoch'])
                if trainer is not None:
                    proof['role'] = role
                    # Recovery validates this receipt against CURRENT bytes if interrupted between replacements.
                    publish(target.parent/'PENDING.json', proof)
                os.replace(temporary, target)
                if trainer is not None: publish(target.parent/'CURRENT.json', proof)
                fsync_directory(target.parent)
                return proof
        finally:
            if temporary.exists(): temporary.unlink()  # Only our exclusive incomplete temporary.

    def immutable(self, source, target, expected=None):
        target = Path(target); checksum = expected or digest(source)
        require(digest(source)==checksum, 'Frozen source artifact changed: '+str(source))
        if target.exists():
            require(digest(target) == checksum, 'Durable frozen artifact differs: '+str(target)); return
        self.copy(source, target, expected=checksum)

    def recover_pending(self, folder):
        folder = Path(folder)
        if (folder/'PENDING.json').exists() and (folder/'CURRENT.pth').exists():
            pending = read(folder/'PENDING.json')
            current=read(folder/'CURRENT.json') if (folder/'CURRENT.json').exists() else {}
            if (digest(folder/'CURRENT.pth')==pending['sha256'] and
                    (current.get('sha256')!=pending['sha256'] or
                     pending.get('copied_at_unix',0)>current.get('copied_at_unix',0))):
                publish(folder/'CURRENT.json',pending)
        if (folder/'CURRENT.json').exists():
            require(digest(folder/'CURRENT.pth') == read(folder/'CURRENT.json')['sha256'],
                    'CURRENT metadata mismatch; retain cache bytes and report interrupted publication')


def fold(native):
    return Path(native['native_root'])/'nnUNet_results'/native['baseline']['dataset_name']/(
        native['trainer']+'__'+native['baseline']['plans_name']+'__3d_fullres')/'fold_0'


def request_code(request):
    command=request['controller_identity']['command']; root=Path(request['root'])
    require(len(command)==6 and command[:3]==['/home/aicompetition06/.conda/envs/nnunet/bin/python3.10','-B','-u']
            and command[4:]==['--request',str(root/'request.json')], 'Exact own controller code command required')
    script=Path(command[3]); code=script.parents[1]
    require(script.name=='run_basic_cp_expanded.py' and script.parent.name=='tools' and code.parent==root.parent,
            'Actual arm code root differs from its pinned controller command')
    return exact(code,directory=True)


def best_validator(code):
    if code is None:
        from hiercp_v1x.v24_native_best_eval_runtime import validate_best_validation_receipt
        return validate_best_validation_receipt
    path=Path(code)/'hiercp_v1x/v24_native_best_eval_runtime.py'
    if str(path) not in BEST_VALIDATORS:
        name='hiercp_v1x._durable_best_'+hashlib.sha256(str(path).encode()).hexdigest()[:16]
        spec=importlib.util.spec_from_file_location(name,path); module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        BEST_VALIDATORS[str(path)]=module.validate_best_validation_receipt
    return BEST_VALIDATORS[str(path)]


def latest_epoch(native):
    """Read the actual latest inode, not an assumed status counter."""
    path=fold(native)/'checkpoint_latest.pth'
    if not path.exists(): return None
    import torch
    with path.open('rb') as f:
        before=os.fstat(f.fileno()); value=torch.load(f,map_location='cpu',weights_only=False)
        try: epoch=checkpoint_info(value,native['trainer'])['checkpoint_completed_epoch']
        finally: del value; gc.collect()
        require(identity(before)==identity(os.fstat(f.fileno())), 'Latest checkpoint inode changed during epoch observation')
    require(not torch.cuda.is_initialized(),'Persistence initialized CUDA')
    return epoch


def select_checkpoint(native, *, unfinished='latest', code=None):
    """BEST classification uses the project's real native full26 receipt validator."""
    goal = fold(native); final = goal/'checkpoint_final.pth'
    proof_path = Path(native['root'])/'training_and_BEST26_complete.json'
    if proof_path.exists():
        proof = read(proof_path)
        require(proof.get('epochs') == 250 and proof.get('checkpoint') == 'checkpoint_best.pth'
                and proof.get('fold') == str(goal) and proof.get('evaluation_checkpoint_is_final') is False,
                'Actual full250 own BEST evaluation completion required')
        # Cache.copy separately validates final's full native logger/model state.
        import torch
        with final.open('rb') as f:
            before = os.fstat(f.fileno()); value = torch.load(f,map_location='cpu',weights_only=False)
            try: require(checkpoint_info(value,native['trainer'])['checkpoint_completed_epoch']==250, 'Native FINAL250 required')
            finally: del value; gc.collect()
            require(identity(before)==identity(os.fstat(f.fileno())), 'FINAL changed during validation')
        require(digest(final) == proof['final_training_witness_sha256'], 'Final250 witness SHA differs')
        receipt = best_validator(code)(goal/'best_validation_complete.json', fold=goal,
            expected_cases=native['split']['outer_val'], bank_sha256=native['bank_sha256'])
        require(receipt['checkpoint_sha256'] == proof['best_sha256'], 'Own evaluated BEST checksum differs')
        return goal/'checkpoint_best.pth', 'BEST_EVALUATED26', dict(training_completed_epochs=250,
            evaluated_cases=26, native_evaluation_complete=True, checkpoint_is_training_resume=False,
            expected_sha256=receipt['checkpoint_sha256'])
    require(unfinished in ('best','latest'),'Explicit unfinished checkpoint policy best/latest required')
    if unfinished=='best':
        return goal/'checkpoint_best.pth','OWN_BEST_TRAINING_STATE',dict(evaluated_cases=0,
            native_evaluation_complete=False,checkpoint_is_training_resume=True,unfinished_checkpoint_policy='best',
            latest_on_tmpfs_N=latest_epoch(native),complete_latest_backup=False,
            may_repeat_epochs_after_saved_BEST=True,live_training_and_evaluation_unchanged=True)
    if final.exists():
        # Native removes latest at train-end. Preserve the preceding durable
        # latest until exact own BEST26 is proved; do not replace it with FINAL.
        return goal/'checkpoint_latest.pth', 'TRAINING_LATEST_PENDING_BEST26', dict(evaluated_cases=0,
            checkpoint_is_training_resume=True,native_evaluation_complete=False,unfinished_checkpoint_policy='latest',
            complete_latest_backup=True,historical_best_not_durable_until_verified_BEST26=True)
    return goal/'checkpoint_latest.pth', 'TRAINING_LATEST', dict(evaluated_cases=0,
        checkpoint_is_training_resume=True,native_evaluation_complete=False,unfinished_checkpoint_policy='latest',
        complete_latest_backup=True,historical_best_not_durable_until_verified_BEST26=True)


def copy_bank(cache, source, count, original, originals, *, binding=None, binding_sha256=None, destination_name=None):
    """Preserve full or published partial banks without generating any candidates."""
    from hiercp_v1x.basic_cp_ablation_bank import read_entry, verify_preservation
    source = Path(source); name=str(count) if destination_name is None else destination_name
    require(isinstance(name,str) and name not in ('.','..') and Path(name).name==name,
            'Single owned bank destination component required')
    destination = cache.root/'banks'/name
    destination.mkdir(parents=True,exist_ok=True)
    complete = source/'complete.json'
    complete_at_start = complete.exists()
    receipts = sorted((source/'entry_receipts').rglob('*.json'))
    rows = []
    for path in receipts:
        cache.guard(); row = read(exact(path)); relative = Path(row['entry'])
        require(not relative.is_absolute() and '..' not in relative.parts and row['entry'] in originals,
                'Published candidate receipt must belong to an original source')
        entry = exact(source/relative)
        require(row['original_entry_sha256'] == originals[row['entry']], 'Published source checksum differs')
        bank_used=sum(p.stat().st_size for p in (cache.root/'banks').rglob('*') if p.is_file())
        incoming=sum(p.stat().st_size for p,target in ((entry,destination/relative),
            (path,destination/'entry_receipts'/path.relative_to(source/'entry_receipts'))) if not target.exists())
        require(bank_used+incoming<=500*2**20, 'Candidate-bank allocation would exceed500MiB; retain existing cache')
        preservation=verify_preservation(read_entry(Path(original).parent/relative), read_entry(entry),
                                         original_count=128, candidate_count=count)
        require(all(row.get('proof',{}).get(k)==v for k,v in preservation.items()),
                'Published source/128-prefix preservation proof differs')
        cache.immutable(entry, destination/relative, row['new_entry_sha256'])
        cache.immutable(path, destination/'entry_receipts'/path.relative_to(source/'entry_receipts'))
        rows.append(row['entry'])
    if complete_at_start:
        require(binding is not None and binding_sha256 is not None, 'Actual completed bank binding witness required')
        witness=dict(index=str(source/'index.json'),index_sha256=digest(source/'index.json'),
            complete_sha256=digest(complete),manifest_sha256=digest(source/'preservation_manifest.json'),
            preparation_started_sha256=digest(source/'preparation_started.json'))
        if count==256:
            from hiercp_v1x.basic_cp_ablation_bank import verify_prepared_bank
            verify_prepared_bank(witness,Path(binding),binding_sha256)
        else:
            from hiercp_v1x.basic_cp_large_bank import verify_prepared_bank
            verify_prepared_bank(witness,Path(binding),binding_sha256,candidate_count=count)
        done = read(exact(complete)); index = read(exact(source/'index.json'))
        require(done.get('complete') is True and done.get('source_entries') == 642 and len(rows)==len(set(rows))==642
                and set(rows)==set(originals) and done.get('candidate_count')==count
                and done.get('original_index_sha256')==BANK_SHA and index.get('candidate_count')==count
                and index.get('total_candidates')==642*count and index['entries_by_case']==read(original)['entries_by_case']
                and done['index_sha256']==digest(source/'index.json')
                and done['preservation_manifest_sha256']==digest(source/'preservation_manifest.json'),
                'Only actual verified full642 candidate bank may become durable complete')
        require(done['entries_sha256']=={read(p)['entry']:read(p)['new_entry_sha256'] for p in receipts},
                'Completed bank checksum population differs')
        for metadata_name in ('index.json','preservation_manifest.json','preparation_started.json','complete.json'):
            cache.immutable(exact(source/metadata_name), destination/metadata_name)
    publish(destination/'durable_receipt.json', dict(format=FORMAT, source=str(source), candidate_count=count,
        source_entries=len(rows), complete=complete_at_start, original_index_sha256=BANK_SHA,
        source_index_sha256=digest(source/'index.json') if (source/'index.json').exists() else None,
        destination_name=name,raw_CT_preprocessing_copied=False, originals_unchanged=True))
    require(sum(p.stat().st_size for p in (cache.root/'banks').rglob('*') if p.is_file()) <= 500*2**20,
            'Durable candidate-bank allocation exceeds500MiB; preserve existing cache and report')


def validate_host_lifetime(reference, value, actual_PBS_text):
    """Validate the admitted same-host job against a genuine running PBS read.

    CPU unit fixtures exercise this parser; they do not establish a running job.
    Production admission obtains actual_PBS_text directly with native qstat.
    """
    from hiercp_v1x import basic_cp_ablation_resources as native_resources
    require(set(reference)=={'path','sha256','expires_at_unix'}
            and value.get('format')=='v25_owned_GPU_allocation_v1'
            and value.get('owner')=='aicompetition06' and value.get('UID')==UID
            and value.get('node')==NODE
            and isinstance(value.get('job_id'),str)
            and re.fullmatch(r'\d+\.ECE-util1',value['job_id']) is not None
            and value.get('job_state')=='R'
            and type(reference['expires_at_unix'])in(int,float)
            and math.isfinite(reference['expires_at_unix'])
            and value.get('expires_at_unix')==reference['expires_at_unix'],
            'Exact admitted own running same-host PBS lifetime witness required')
    native_resources.validate_host_lifetime_GPU(value)
    walltime=value.get('walltime','').split(':')
    require(len(walltime)==3 and all(part.isdigit()for part in walltime),
            'Actual witness start/walltime is required')
    hours,minutes,seconds=map(int,walltime)
    require(minutes<60 and seconds<60 and hours*3600+minutes*60+seconds>0,
            'Actual witness walltime must be a positive valid duration')
    expiration=native_resources._pbs_start(value['stime'])+hours*3600+minutes*60+seconds
    require(abs(expiration-reference['expires_at_unix'])<=1,
            'Witness deadline differs from its actual start/walltime')
    observation=native_resources.BasicLease._validate_pbs_observation(
        SimpleNamespace(lifetime=value),actual_PBS_text)
    require(reference['expires_at_unix']-time.time()>=native_resources.EXPIRY_RESERVE_SECONDS,
            'Actual PBS lifetime has crossed the unchanged180second reserve')
    return observation


def admit(manifest_path):
    require(os.getuid()==UID and socket.gethostname().split('.')[0]==NODE, 'Exact own host/UID required')
    path = exact(manifest_path); manifest = read(path); group = exact(manifest['group'],directory=True)
    require(set(manifest['registration'])=={str(i) for i in range(7)}, 'Exactly seven registered GPU arms required')
    requests = {}; admitted_lifetime = None
    for gpu,row in manifest['registration'].items():
        root = exact(row['root'], directory=True); q = read(exact(root/'request.json'))
        require(Path('/dev/shm/aicompetition06_basic_cp_ablation_20261013') in root.parents
                and q['root']==str(root) and q['gpu']==int(gpu) and q['uid']==UID and q['arm']==ARMS[int(gpu)]
                and q['controller_identity']==row['controller_identity']
                and q.get('shared_resource_guard_version')==3
                and digest(root/'request.json')==row['request_sha256'], 'Fresh V3 arm/request/code registration differs')
        copies={0:5,1:7,2:10}.get(int(gpu),1); candidates={3:512,4:1024,5:2048}.get(int(gpu),128)
        require(q['copies_per_CP_event']==copies and q['candidates_per_source']==candidates,
                'Exact seven authorized one-factor cardinalities required')
        require({'hiercp_v1x/basic_cp_group_resources_v3.py','hiercp_v1x/basic_cp_shared_bank_resources_v3.py',
                 'hiercp_v1x/basic_cp_completed_epoch_checkpoint.py','tools/run_basic_cp_expanded.py'}.issubset(q['code_sha256']),
                'Actual V3 resource and completed-epoch helper code pins required')
        code=request_code(q)
        for relative,checksum in q['code_sha256'].items():
            rel = Path(relative)
            require(not rel.is_absolute() and '..' not in rel.parts
                    and digest(exact(code/rel))==checksum, 'Frozen per-arm code admission differs')
        lifetime = q['host_lifetime']; value = read(exact(lifetime['path']))
        require(digest(lifetime['path'])==lifetime['sha256'], 'Exact admitted lifetime witness SHA required')
        if admitted_lifetime is None:
            from hiercp_v1x import basic_cp_ablation_resources as native_resources
            require(isinstance(value.get('job_id'),str)
                    and re.fullmatch(r'\d+\.ECE-util1',value['job_id']) is not None,
                    'Numeric admitted same-host PBS job identity required')
            actual=native_resources._pbs_query(['qstat','-f','-F','json',value['job_id']],
                                               timeout=native_resources.PBS_QUERY_TIMEOUT_SECONDS)
            validate_host_lifetime(lifetime,value,actual)
            admitted_lifetime=dict(lifetime)
        else:
            require(lifetime==admitted_lifetime, 'All seven arms must retain one exact admitted actual host lifetime')
        require(q['bank_sha256']==BANK_SHA and digest(exact(q['bank']))==BANK_SHA, 'Frozen original128 bank SHA differs')
        requests[gpu] = q
    return manifest, requests


def snapshot(cache, manifest, requests, *, unfinished='latest'):
    cache.guard(); originals = None; original = requests['0']['bank']; outcomes = {}; native_banks=[]
    for gpu,q in requests.items():
        root = Path(q['root']); folder = cache.root/root.name; folder.mkdir(exist_ok=True)
        cache.recover_pending(folder); cache.immutable(root/'request.json', folder/'request.json',
            manifest['registration'][gpu]['request_sha256'])
        cache.immutable(q['host_lifetime']['path'], cache.root/'host_lifetime.json', q['host_lifetime']['sha256'])
        for key in ('readonly_admission','candidate_binding','prepared256_witness'):
            if q.get(key): cache.immutable(exact(q[key]), folder/(key+'.json'), q.get(key+'_sha256'))
        for name in ('native.json','preflight.json','calibration.json','actual_model.json','own_checkpoint_resume.json',
                     'prepared_candidate_bank_admission.json','training_and_BEST26_complete.json'):
            if (root/name).exists(): cache.immutable(exact(root/name), folder/name)
        if originals is None and q['candidates_per_source']==128 and (root/'preflight.json').exists():
            pre = read(root/'preflight.json')
            if pre.get('complete') is True and pre.get('index_sha256')==BANK_SHA: originals=pre['entries_sha256']
        if not(root/'native.json').exists(): outcomes[gpu]='WAITING_FOR_NATIVE'; continue
        native = read(root/'native.json')
        require(all(native[k]==q[k] for k in ('root','arm','gpu','uid','commit','code_sha256')),
                'Actual native identity/code differs from its frozen request')
        if native['candidates_per_source']>128:
            require(digest(exact(native['bank']))==native['bank_sha256'], 'Actual native bank index identity differs')
            native_banks.append((gpu,q,native))
        if (root/'actual_model.json').exists():
            model=read(root/'actual_model.json')
            require(model['epochs']==250 and model['physical_batch']==model['effective_batch']==2
                    and model['patch_size']==[128]*3 and len(model['actual_train_cases'])==105
                    and len(model['actual_validation_cases'])==26 and model['debug'] is False and model['subset'] is False
                    and set(model['actual_train_cases'])==set(native['split']['outer_train'])
                    and set(model['actual_validation_cases'])==set(native['split']['outer_val']),
                    'Full unchanged native training/model/data contract required')
        source,role,extra = select_checkpoint(native,unfinished=unfinished,code=request_code(q))
        if (root/'actual_model.json').exists(): extra['minimum_network_tensor_elements']=model['parameters']
        runtime=exact(native['private_runtime'],directory=True)
        for path in runtime.rglob('*.py'):
            cache.immutable(exact(path),folder/'runtime'/path.relative_to(runtime))
        if source.exists():
            prior = read(folder/'CURRENT.json') if (folder/'CURRENT.json').exists() else {}
            # Checksum binding remains necessary even if source's name/current epoch is unchanged.
            if prior.get('source_inode')!=list(identity(source.stat())) or prior.get('role')!=role:
                expected = extra.pop('expected_sha256',None); required_epoch = extra.pop('required_epoch',None)
                if required_epoch is not None:
                    extra['required_epoch']=required_epoch
                proof = cache.copy(exact(source), folder/'CURRENT.pth', expected=expected,
                    trainer=native['trainer'], role=role, extra=extra)
                require(required_epoch is None or proof['checkpoint_completed_epoch']==required_epoch, 'FINAL full250 epoch differs')
            elif role=='OWN_BEST_TRAINING_STATE':
                prior.update(latest_on_tmpfs_N=extra['latest_on_tmpfs_N'],
                    epochs_may_repeat_after_tmpfs_loss=None if extra['latest_on_tmpfs_N'] is None else
                        max(0,extra['latest_on_tmpfs_N']-prior['saved_resume_epoch']))
                publish(folder/'CURRENT.json',prior)
            outcomes[gpu]=role
        else: outcomes[gpu]='NO_OWN_BEST_YET' if role=='OWN_BEST_TRAINING_STATE' else (
            'KEEP_PREVIOUS_LATEST_PENDING_BEST26' if role=='TRAINING_LATEST_PENDING_BEST26' else 'NO_WHOLE_CHECKPOINT_YET')
        for path in sorted(root.rglob('*')):
            if path.is_file() and path.suffix in ('.json','.csv','.md','.log','.txt') and 'runtime' not in path.parts:
                relative = path.relative_to(root)
                if path.name in ('request.json','native.json') or relative.parts[0].startswith('B'): continue
                # Mutable progress snapshots are stored in this cache only; no source overwrite.
                cache.copy(exact(path), folder/'artifacts'/relative,append_only=path.suffix=='.log')
    require(originals is not None and len(originals)==642, 'Verified full642 original source SHA witness required')
    names = [n for group in read(original)['entries_by_case'].values() for n in group]
    require(len(names)==len(set(names))==642 and set(names)==set(originals), 'Original full source population differs')
    cache.immutable(original, cache.root/'banks/128/index.json',BANK_SHA)
    for relative,checksum in originals.items():
        cache.immutable(exact(Path(original).parent/relative), cache.root/'banks/128'/relative,checksum)
    bankroot = Path(requests['5']['shared_banks'])
    for count in (512,1024,2048):
        if (bankroot/str(count)).exists(): copy_bank(cache,bankroot/str(count),count,original,originals,
            binding=requests['5']['candidate_binding'],binding_sha256=requests['5']['candidate_binding_sha256'])
    bank_mapping={}
    for gpu,q,native in native_banks:
        source=Path(native['bank']); count=native['candidates_per_source']; name=str(count)
        if source.parent!=bankroot/str(count):
            name='native_GPU'+gpu+'_'+str(count)
            require((source.parent/'complete.json').exists(), 'Native candidate bank must retain its actual complete witness')
            copy_bank(cache,source.parent,count,original,originals,binding=q['candidate_binding'],
                      binding_sha256=q['candidate_binding_sha256'],destination_name=name)
        require(digest(cache.root/'banks'/name/'index.json')==native['bank_sha256'],
                'Native bank is not covered by a different same-count index SHA')
        bank_mapping[gpu]=dict(source_index=str(source),source_index_sha256=native['bank_sha256'],
                              durable_index='banks/'+name+'/index.json')
    if requests['5'].get('prepared256_witness'):
        prepared=read(exact(requests['5']['prepared256_witness'])); index=exact(prepared['index'])
        from hiercp_v1x.basic_cp_ablation_bank import verify_prepared_bank
        verify_prepared_bank(prepared,Path(requests['5']['candidate_binding']),requests['5']['candidate_binding_sha256'])
        copy_bank(cache,index.parent,256,original,originals,binding=requests['5']['candidate_binding'],
                  binding_sha256=requests['5']['candidate_binding_sha256'],destination_name='prepared256')
    publish(cache.root/'snapshot.json', dict(format=FORMAT, updated_at_unix=time.time(), outcomes=outcomes,
        expiry_unix=cache.expiry, original_results_overwritten=False, processes_signaled=False,
        CPU_only=True, durable_budget_bytes=8*GIB, required_free_bytes=GIB,
        unfinished_checkpoint_policy=unfinished,complete_latest_backup=unfinished=='latest',
        unfinished_arm_historical_BEST_not_independently_durable=unfinished=='latest',
        native_prediction_volumes_and_raw_CT_not_copied=True,native_bank_backups=bank_mapping))
    return outcomes


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True); parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--once',action='store_true'); parser.add_argument('--interval',type=int,default=60)
    parser.add_argument('--resume-cache',action='store_true',help='Reuse only this exact manifest/helper operational cache')
    parser.add_argument('--unfinished-checkpoint',choices=('best','latest'),default='latest')
    args=parser.parse_args(); os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    manifest,requests=admit(args.manifest); output=args.output.absolute()
    require(output.parent==Path('/var/tmp') and output.name.startswith('aicompetition06_basic_cp_seven_resume_')
            and not output.is_symlink(), 'Own fixed /var/tmp persistence namespace required')
    if args.resume_cache:
        exact(output,directory=True); saved=read(exact(output/'identity.json'))
        require(saved['format']==FORMAT and saved['uid']==UID and saved['node']==NODE
                and saved['commit']==manifest['commit'] and saved['helper_source_sha256']==digest(__file__)
                and saved['unfinished_checkpoint_policy']==args.unfinished_checkpoint
                and digest(exact(output/'source_manifest.json'))==digest(args.manifest),
                'Only this exact own manifest/helper cache may resume')
    else: output.mkdir(mode=0o700)  # Exclusive fresh namespace; never admit arbitrary existing user files.
    import fcntl
    lock=os.open(output/'writer.lock',os.O_RDWR|os.O_CREAT,0o600)
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)  # Held to interpreter completion; no competing cache writer.
    import psutil
    process=psutil.Process()
    cpus=[82,83]
    require(set(cpus).issubset(process.cpu_affinity()), 'Own available shared persistence CPU82/83 required')
    process.cpu_affinity(cpus)  # Shared with candidate pool; no PBS/exclusive CPU ownership claim.
    cache=Cache(output,min(q['host_lifetime']['expires_at_unix'] for q in requests.values()))
    sys.path.insert(0,manifest['code'])  # Actual BEST validator path must equal its native receipt source proof.
    cache.immutable(args.manifest,output/'source_manifest.json')
    cache.immutable(Path(__file__),output/'persistence_helper.py')
    for q in requests.values():
        for relative,checksum in q['code_sha256'].items():
            cache.immutable(request_code(q)/relative,output/'code'/q['commit']/relative,checksum)
    publish(output/'identity.json',dict(format=FORMAT,uid=UID,node=NODE,pid=os.getpid(),birth=process.create_time(),
        command=process.cmdline(),cpu_affinity=cpus,RSS_limit_bytes=4*GIB,commit=manifest['commit'],
        helper_source_sha256=digest(__file__),unfinished_checkpoint_policy=args.unfinished_checkpoint,
        CPU_cores_are_shared=True,PBS_exclusive_CPU_claim=False))
    while time.time()<cache.expiry-180:
        try:
            observed,current=admit(args.manifest)
            require(observed==manifest and current==requests, 'Frozen manifest/request contracts changed during persistence')
            outcomes=snapshot(cache,manifest,requests,unfinished=args.unfinished_checkpoint)
            print(json.dumps(dict(event='DURABLE_SNAPSHOT',outcomes=outcomes),ensure_ascii=False),flush=True)
        except Exception as error:
            publish(output/'stopped.json',dict(format=FORMAT,error=type(error).__name__+': '+str(error),
                previous_valid_cache_preserved=True,processes_signaled=False,original_results_overwritten=False))
            raise
        if args.once or all(v=='BEST_EVALUATED26' for v in outcomes.values()): return
        time.sleep(min(60,max(1,args.interval),max(1,cache.expiry-180-time.time())))
    publish(output/'stopped.json',dict(format=FORMAT,reason='Host lifetime180 second reserve',
        previous_valid_cache_preserved=True,processes_signaled=False,original_results_overwritten=False))


if __name__=='__main__': main()
