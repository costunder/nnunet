"""Watch own GPU5 latest; derive a rolling whole-state resume cache with exact epoch.

The e6 trainer saves its rolling latest after incrementing the epoch and the
base saver increments the header again. Eight complete logger arrays establish
N completed epochs. This auxiliary CPU observer changes only current_epoch toN
in a separately serialized cache; original latest/BEST/final files are untouched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import stat
import time

UID = 41833
NODE = 'ece-a6gpu6'
GPU5_REQUEST_SHA = '7a16e6708ad48dfd8aefd6eb35b0c1ab65fec45d7b561a826b4e85e5da4c6c2f'
SOURCE_COMMIT = 'e6a3e69565966121efe208104a7ad3bd86f4a90a'
SOURCE_BANK_SHA = '9e5be3e3ab8f04fd23c166b3267bc4efbfb4b321ef974920dfc08fbf5dd275e6'
LOGGER_KEYS = {'mean_fg_dice','ema_fg_dice','dice_per_class_or_region','train_losses',
               'val_losses','lrs','epoch_start_timestamps','epoch_end_timestamps'}
FORMAT = 'basic_gpu5_exact_epoch_derived_rolling_resume_cache_v1'
RSS_LIMIT = 4*2**30
DISK_FLOOR = 10*2**30
EXPIRY_RESERVE = 180
_STORAGE_LOCATIONS = {}
_STORAGE_TAGGER_REGISTERED = False


def require(condition,message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(2**20),b''):
            digest.update(block)
    return digest.hexdigest()


def snapshot(path):
    row = Path(path).stat()
    return (row.st_dev,row.st_ino,row.st_size,row.st_mtime_ns,row.st_ctime_ns,row.st_uid,row.st_mode)


def exact(path,*,root=None,directory=False):
    path = Path(path)
    require(path.is_absolute() and '..' not in path.parts and path.resolve(strict=True) == path
            and not any(p.is_symlink() for p in (path,*path.parents)),
            'Exact non-symlink source/cache path required: '+str(path))
    row = path.stat()
    require(row.st_uid == UID and (stat.S_ISDIR(row.st_mode) if directory else stat.S_ISREG(row.st_mode)),
            'Own exact regular evidence required: '+str(path))
    if root is not None:
        require(root in path.parents,'Evidence escapes its exact namespace: '+str(path))
    return path


def read(path):
    value = json.loads(Path(path).read_bytes())
    require(isinstance(value,dict),'Actual JSON object required: '+str(path))
    return value


def _atomic_json(path,value):
    temporary = path.parent/('.'+path.name+'_'+str(os.getpid())+'_'+str(time.time_ns()))
    with temporary.open('x',encoding='utf8') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)


def _exclusive_json(path,value):
    with path.open('x',encoding='utf8') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.flush();os.fsync(stream.fileno())


def _tensor_fingerprint(value,digest):
    import torch
    import numpy as np
    def token(data):
        digest.update(len(data).to_bytes(8,'little'));digest.update(data)
    token((type(value).__module__+'.'+type(value).__qualname__).encode())
    if torch.is_tensor(value):
        require(value.layout == torch.strided,'Unexpected sparse checkpoint state cannot be silently rewritten')
        token(str(value.dtype).encode());token(repr(tuple(value.shape)).encode())
        token(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    elif isinstance(value,np.ndarray):
        require(not value.dtype.hasobject,'Object arrays in full checkpoint state are unsupported')
        token(str(value.dtype).encode());token(repr(tuple(value.shape)).encode())
        token(np.ascontiguousarray(value).tobytes())
    elif isinstance(value,np.generic):
        token(str(value.dtype).encode());token(value.tobytes())
    elif isinstance(value,dict):
        token(str(len(value)).encode())
        for key,item in value.items():
            _tensor_fingerprint(key,digest);_tensor_fingerprint(item,digest)
    elif isinstance(value,(list,tuple)):
        token(str(len(value)).encode())
        for item in value:_tensor_fingerprint(item,digest)
    elif value is None or isinstance(value,(str,int,float,bool,bytes)):
        token(value if isinstance(value,bytes) else repr(value).encode())
    elif isinstance(value,torch.device):
        token(str(value).encode())
    else:
        raise ValueError('Unexpected full-state object cannot be silently rewritten: '+type(value).__qualname__)


def state_fingerprint(saved):
    require(isinstance(saved,dict),'Actual nnU-Net checkpoint dictionary required')
    digest = hashlib.sha256()
    _tensor_fingerprint({key:value for key,value in saved.items() if key!='current_epoch'},digest)
    return digest.hexdigest()


def completed_epoch(saved,*,trainer):
    import torch
    require(saved.get('trainer_name') == trainer,'Own exact native trainer name required')
    logging = saved.get('logging')
    require(isinstance(logging,dict) and set(logging) == LOGGER_KEYS,
            'Exactly the eight actual nnU-Net logger arrays required')
    lengths = {key:len(value) for key,value in logging.items()
               if isinstance(value,(list,tuple))}
    require(set(lengths) == LOGGER_KEYS and len(set(lengths.values())) == 1,
            'Whole completed checkpoint requires all eight logger lengths equal')
    count = next(iter(lengths.values()))
    require(0 < count <= 250,'Actual completed checkpoint epoch must be1..250')
    raw = saved.get('current_epoch')
    require(type(raw) is int and raw in (count,count+1),
            'Only verified e6 one-step latest-header discrepancy may be corrected')
    starts,ends = logging['epoch_start_timestamps'],logging['epoch_end_timestamps']
    require(all(math.isfinite(float(start)) and math.isfinite(float(end)) and end>=start
                for start,end in zip(starts,ends)),
            'Partial epoch timestamps cannot be promoted to a whole checkpoint')
    require(all(starts[index]>=ends[index-1] for index in range(1,count)),
            'Completed epoch timestamps must follow the actual previous completion')
    weights = saved.get('network_weights')
    require(isinstance(weights,dict) and weights and all(isinstance(key,str) and torch.is_tensor(value)
            and value.numel()>0 for key,value in weights.items()),'Actual full network tensor state required')
    optimizer = saved.get('optimizer_state')
    require(isinstance(optimizer,dict) and optimizer.get('state') and isinstance(optimizer.get('state'),dict)
            and isinstance(optimizer.get('param_groups'),list) and optimizer['param_groups']
            and all(group.get('params') for group in optimizer['param_groups']),
            'Actual optimizer state and all parameter groups required')
    scaler = saved.get('grad_scaler_state')
    require(isinstance(scaler,dict) and scaler and math.isfinite(float(scaler.get('scale',float('nan'))))
            and scaler['scale']>0,'Actual native GradScaler state required')
    require(isinstance(saved.get('init_args'),dict) and saved['init_args']
            and math.isfinite(float(saved.get('_best_ema',float('nan')))),
            'Actual initializer and BEST-selection state required')
    return count,dict(raw_current_epoch=raw,completed_epochs=count,logger_lengths=lengths,
                      header_delta=raw-count,trainer=trainer,
                      all_eight_loggers_and_epoch_timestamps_whole=True)


def corrected_state(saved,*,trainer):
    count,evidence = completed_epoch(saved,trainer=trainer)
    before = state_fingerprint(saved)
    corrected = saved.copy()
    corrected['current_epoch'] = count
    require(state_fingerprint(corrected) == before,'Full state changed outside the sole epoch header')
    evidence.update(state_except_epoch_sha256=before,only_current_epoch_changed=True,
                    model_optimizer_scaler_logging_and_best_selection_preserved=True)
    return corrected,evidence


def _register_storage_tagger():
    """Retain original torch storage location tags while reading/saving only CPU."""
    global _STORAGE_TAGGER_REGISTERED
    if _STORAGE_TAGGER_REGISTERED:return
    import torch
    def tagger(storage):return _STORAGE_LOCATIONS.get(storage._cdata)
    def deserializer(storage,location):return None
    torch.serialization.register_package(0,tagger,deserializer)
    _STORAGE_TAGGER_REGISTERED = True


def held_checkpoint(path):
    """Load/hash one held inode; atomic path replacement cannot mix two epochs."""
    import torch
    path = Path(path)
    exact(path)
    fd = os.open(path,os.O_RDONLY|getattr(os,'O_NOFOLLOW',0))
    locations = {}
    try:
        before = os.fstat(fd)
        require(before.st_uid == UID and stat.S_ISREG(before.st_mode),'Held source must be own regular inode')
        digest = hashlib.sha256()
        with os.fdopen(fd,'rb',closefd=False) as stream:
            for block in iter(lambda:stream.read(2**20),b''):digest.update(block)
            stream.seek(0)
            def map_cpu(storage,location):
                locations[storage._cdata] = location
                return storage
            saved = torch.load(stream,map_location=map_cpu,weights_only=False)
        after = os.fstat(fd)
        require((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)
                == (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns),
                'Held checkpoint bytes changed during read; no mixed/partial cache written')
        path_now = path.stat() if path.exists() else None
        record = dict(source_path=str(path),source_sha256=digest.hexdigest(),source_bytes=after.st_size,
            held_device=after.st_dev,held_inode=after.st_ino,held_mtime_ns=after.st_mtime_ns,
            path_replaced_during_read=path_now is None or (path_now.st_dev,path_now.st_ino)!=(after.st_dev,after.st_ino),
            source_storage_location_counts={tag:sum(value==tag for value in locations.values())
                                           for tag in sorted(set(locations.values()))})
        return saved,record,locations
    finally:
        os.close(fd)


def derive_checkpoint(source,output,*,trainer,prior_completed=0,check_resources=None):
    """Produce one owned rolling cache; source files are opened read-only."""
    import torch
    saved,source_record,locations = held_checkpoint(source)
    if check_resources is not None:check_resources()
    corrected,record = corrected_state(saved,trainer=trainer)
    count = record['completed_epochs']
    require(count>=prior_completed,'Source whole checkpoint regressed; prior rolling cache preserved')
    _register_storage_tagger()
    _STORAGE_LOCATIONS.clear();_STORAGE_LOCATIONS.update(locations)
    target = output/'CURRENT.pth'
    temporary = output/('.CURRENT_'+str(os.getpid())+'_'+str(time.time_ns())+'.pth')
    with temporary.open('xb') as stream:
        torch.save(corrected,stream);stream.flush();os.fsync(stream.fileno())
    if check_resources is not None:check_resources()
    # Re-read actual serialized bytes and verify all state, not only in-memory refs.
    actual,cache_record,_ = held_checkpoint(temporary)
    if check_resources is not None:check_resources()
    require(actual.get('current_epoch') == count and state_fingerprint(actual) == record['state_except_epoch_sha256'],
            'Derived serialized checkpoint altered full training state')
    require(cache_record['source_storage_location_counts'] == source_record['source_storage_location_counts'],
            'Original serialized torch storage locations changed')
    record.update(source_record,format=FORMAT,cache_path=str(target),
        cache_sha256=cache_record['source_sha256'],cache_bytes=cache_record['source_bytes'],
        original_latest_BEST_final_written=False,CUDA_initialized=False,
        original_torch_storage_locations_preserved=True,published_at=time.time())
    os.replace(temporary,target)
    _atomic_json(output/'current.json',record)
    receipt = output/'epochs'/f"epoch_{count:03d}_source_{source_record['source_sha256'][:16]}.json"
    if receipt.exists():
        previous = read(receipt)
        require(previous['source_sha256'] == record['source_sha256']
                and previous['state_except_epoch_sha256'] == record['state_except_epoch_sha256'],
                'Existing epoch evidence differs; no receipt overwritten')
    else:_exclusive_json(receipt,record)
    require(not torch.cuda.is_initialized(),'CPU-only corrector unexpectedly initializedCUDA')
    return record


class Watch:
    def __init__(self,manifest,output):
        import psutil
        require(os.getuid()==UID and socket.gethostname().split('.')[0]==NODE,
                'Actual own serverUID/node required')
        self.manifest = exact(manifest)
        self.manifest_sha = sha(self.manifest)
        self.manifest_stat = snapshot(self.manifest)
        document = read(self.manifest)
        row = document.get('registration',{}).get('5')
        require(isinstance(row,dict) and row.get('arm')=='candidates2048'
                and row.get('request_sha256')==GPU5_REQUEST_SHA,'Exact known GPU5 source manifest required')
        self.root = exact(row['root'],directory=True)
        self.request_path = exact(self.root/'request.json',root=self.root)
        require(sha(self.request_path)==GPU5_REQUEST_SHA,'Exact GPU5 e6 requestSHA differs')
        self.request = read(self.request_path)
        require(self.request.get('controller_identity')==row['controller_identity']
                and self.request.get('commit')==SOURCE_COMMIT and self.request.get('gpu')==5
                and self.request.get('arm')=='candidates2048' and self.request.get('copies_per_CP_event')==1
                and self.request.get('candidates_per_source')==2048 and self.request.get('epochs')==250
                and self.request.get('source_bank_sha256',self.request.get('bank_sha256'))==SOURCE_BANK_SHA,
                'Known full GPU5 candidate2048 experiment differs')
        self.controller = row['controller_identity']
        require(self.controller.get('UID')==UID and self.controller.get('pid')==3432151
                and self.controller.get('create_time')==1791926249.89,
                'Exact GPU5 controller birthUID required')
        self.files = {str(self.manifest):self.manifest_stat,str(self.request_path):snapshot(self.request_path)}
        command = self.controller['command']
        require(command[-2:]==['--request',str(self.request_path)],'Exact source controller argv request required')
        tool = Path(next(value for value in command if value.endswith('/tools/run_basic_cp_expanded.py')))
        self.code = exact(tool.parent.parent,directory=True)
        for relative,digest in self.request['code_sha256'].items():
            path = exact(self.code/relative,root=self.code)
            require(sha(path)==digest,'Immutable source codeSHA differs: '+relative)
            self.files[str(path)] = snapshot(path)
        self.source_bank = exact(self.request['bank'])
        require(sha(self.source_bank)==SOURCE_BANK_SHA,'Actual original128 sourcebankSHA differs')
        self.files[str(self.source_bank)] = snapshot(self.source_bank)
        lifetime = self.request['host_lifetime']
        lifetime_path = exact(lifetime['path'])
        require(sha(lifetime_path)==lifetime['sha256'],'Actual head-lifetime proofSHA differs')
        proof = read(lifetime_path)
        require(proof.get('UID')==UID and proof.get('owner')=='aicompetition06'
                and proof.get('node')==NODE and proof.get('expires_at_unix')==lifetime['expires_at_unix'],
                'Actual unchanged own head-lifetime expiry proof required')
        self.expiry = float(lifetime['expires_at_unix'])
        require(math.isfinite(self.expiry),'Finite genuine head expiry required')
        self.monotonic_expiry = time.monotonic()+self.expiry-time.time()
        self.files[str(lifetime_path)] = snapshot(lifetime_path)
        self.output = Path(output).absolute()
        require(self.output.parent.resolve(strict=True)==self.output.parent
                and self.root not in self.output.parents and self.output!=self.root
                and not any(p.is_symlink() for p in (self.output.parent,*self.output.parent.parents)),
                'Fresh separate auxiliary derived cache namespace required')
        self.output.mkdir(mode=0o700,exist_ok=False)
        (self.output/'epochs').mkdir()
        self.native = self.native_stat = self.native_sha = None
        self.last_source_stat = None
        self.completed = 0
        self.last_current = None
        self.process = psutil.Process()
        self.started = time.time()
        self.status = dict(format=FORMAT,status='WAITING_FOR_NATIVE',gpu=5,source_root=str(self.root),
            output=str(self.output),source_request_sha256=GPU5_REQUEST_SHA,source_commit=SOURCE_COMMIT,
            source_controller_identity=self.controller,manifest=str(self.manifest),manifest_sha256=self.manifest_sha,
            cache_current_epoch=None,source_pipeline_complete=False,original_latest_BEST_final_written=False,
            started_at=self.started,RSS_limit_bytes=RSS_LIMIT,head_expires_at_unix=self.expiry)
        _exclusive_json(self.output/'registration.json',dict(self.status,observer_identity={
            'pid':self.process.pid,'birth':self.process.create_time(),'UID':self.process.uids().real,
            'command':self.process.cmdline()}))

    def guard(self):
        import torch,shutil
        require(not torch.cuda.is_initialized(),'Auxiliary observer must remainCPU-only')
        require(self.process.memory_info().rss <= RSS_LIMIT,'ObserverRSS exceeds4GiB; original state preserved')
        require(shutil.disk_usage(self.output).free >= DISK_FLOOR,
                'Derived cache storage has less than10GiB free; original checkpoints preserved')
        require(min(self.expiry-time.time(),self.monotonic_expiry-time.monotonic())>=EXPIRY_RESERVE,
                'Head lifetime reached180-second reserve; rolling derived cache preserved')
        require(all(snapshot(path)==observed for path,observed in self.files.items()),
                'Pinned manifest/request/sourcebank/sourcecode/head proof metadata changed')

    def source_status(self):
        path = self.root/'status.json'
        value = read(path) if path.exists() else {}
        return value.get('status'),value.get('stage'),value.get('error')

    def controller_alive(self):
        import psutil
        try:
            process = psutil.Process(self.controller['pid'])
            require(process.create_time()==self.controller['create_time'] and process.uids().real==UID
                    and process.cmdline()==self.controller['command'],'Pinned GPU5 controller identity changed')
            return process.status() not in (psutil.STATUS_ZOMBIE,psutil.STATUS_DEAD)
        except psutil.NoSuchProcess:return False

    def admit_native(self):
        path = self.root/'native.json'
        actor = self.root/'train_actor.json'
        if not path.exists() or not actor.exists():return False
        native = read(path)
        require(native.get('gpu')==5 and native.get('arm')=='candidates2048'
                and native.get('copies_per_CP_event')==1 and native.get('candidates_per_source')==2048
                and native.get('epochs')==250 and native.get('physical_batch')==2
                and native.get('full_training_cases')==105 and native.get('full_validation_cases')==26
                and native.get('source_preprocessed_arrays_written') is False,
                'Actual unchanged full GPU5 native contract required')
        bank_path = exact(native['bank'])
        require(sha(bank_path)==native['bank_sha256'],'Actual derived2048 bankSHA differs')
        bank = read(bank_path)
        require(bank.get('source_entries')==642 and bank.get('eligible_cases')==81
                and bank.get('candidate_count')==2048 and bank.get('total_candidates')==642*2048,
                'Actual all642 native2048 source bank required')
        expected_root = self.root/'native'
        require(native.get('root')==str(self.root) and native.get('native_root')==str(expected_root),
                'Exact source native namespace required')
        train = read(actor)
        require(train.get('stage')=='train' and train.get('UID')==UID,
                'Exact own registered native training actor required')
        self.native = native
        self.native_stat,self.native_sha = snapshot(path),sha(path)
        self.files[str(path)] = self.native_stat
        self.files[str(bank_path)] = snapshot(bank_path)
        self.files[str(actor)] = snapshot(actor)
        self.fold = expected_root/'nnUNet_results'/native['baseline']['dataset_name']/(
            native['trainer']+'__'+native['baseline']['plans_name']+'__3d_fullres')/'fold_0'
        self.source_latest = self.fold/'checkpoint_latest.pth'
        self.status.update(native_sha256=self.native_sha,source_latest=str(self.source_latest),
                           native_bank_sha256=native['bank_sha256'])
        return True

    def publish(self,status,**fields):
        self.status.update(status=status,updated_at=time.time(),**fields)
        _atomic_json(self.output/'status.json',self.status)
        print(json.dumps(self.status,ensure_ascii=False,allow_nan=False),flush=True)

    def run(self,poll_seconds):
        require(type(poll_seconds) in (int,float) and math.isfinite(poll_seconds) and poll_seconds>0,
                'Finite positive polling interval required')
        self.publish('WAITING_FOR_NATIVE')
        try:
            while True:
                self.guard()
                source_status,stage,error = self.source_status()
                if self.native is None:self.admit_native()
                if self.native is not None and self.source_latest.exists():
                    observed = snapshot(self.source_latest)
                    if observed != self.last_source_stat:
                        record = derive_checkpoint(self.source_latest,self.output,trainer=self.native['trainer'],
                                                   prior_completed=self.completed,check_resources=self.guard)
                        self.guard()
                        self.completed = record['completed_epochs']
                        self.last_current = record
                        # Pin the held original inode, not a newer path observed after loading.
                        self.last_source_stat = observed
                        self.publish('CACHING',cache_current_epoch=self.completed,
                            latest_source_raw_epoch=record['raw_current_epoch'],
                            cache_sha256=record['cache_sha256'],source_sha256=record['source_sha256'],
                            source_pipeline_status=source_status,source_stage=stage,
                            observer_RSS_bytes=self.process.memory_info().rss)
                if source_status=='COMPLETE':
                    self.publish('STOPPED_SOURCE_COMPLETE',source_pipeline_complete=True,
                                 source_pipeline_status=source_status,source_stage=stage)
                    return 0
                if source_status in ('FAILED','EXITED_WITHOUT_COMPLETION','INTERRUPTED','CANCELLED'):
                    self.publish('STOPPED_SOURCE_ERROR',source_pipeline_status=source_status,
                                 source_stage=stage,error=error or 'Source controller terminated without completion')
                    return 1
                if not self.controller_alive():
                    self.publish('STOPPED_CONTROLLER_TERMINAL',source_pipeline_status=source_status,
                                 source_stage=stage,error='Exact source controller has naturally released; completion unconfirmed')
                    return 1
                time.sleep(poll_seconds)
        except Exception as exc:
            self.publish('FAILED',error=f'{type(exc).__name__}: {exc}',source_pipeline_complete=False)
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command',required=True)
    watch = commands.add_parser('watch')
    watch.add_argument('--manifest',required=True,type=Path)
    watch.add_argument('--gpu',type=int,default=5,choices=(5,))
    watch.add_argument('--gpu5',action='store_const',const=5,dest='gpu')
    watch.add_argument('--output',required=True,type=Path)
    watch.add_argument('--poll-seconds',type=float,default=5.)
    args = parser.parse_args(argv)
    os.environ['CUDA_VISIBLE_DEVICES']=''
    return Watch(args.manifest,args.output).run(args.poll_seconds)


if __name__=='__main__':
    raise SystemExit(main())
