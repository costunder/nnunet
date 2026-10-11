"""Explicit GPU4/5 pre-start adoption: existing CP, fixed fresh native code.

No predecessor files or processes are changed. A separate exclusive lease
waits for exact predecessor/scorer release and whole scoring evidence before
the original three full native stages. GPU5 retains its truthful early pin.
"""
from __future__ import annotations
import ast
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from . import v24_scored_native_chain as chain
from . import v24_early_stop_downstream as early
from . import v24_matched_basic_cp_pipeline as pipeline

FORMAT='v24_native_prestart_adoption_request_v1'
STATUS_FORMAT='v24_native_prestart_adoption_status_v1'
FILES=('hiercp_v1x/v24_native_prestart_guard.py','tools/run_v24_native_prestart_guard.py')
ROOT=Path(__file__).resolve().parents[1]
UID=41833


def publish(path,value,*,replace=False,raw=False):
    """Always remove our own temporary file, including serialization failure."""
    path=Path(path)
    temporary=path.parent/('.'+path.name+'.'+str(os.getpid())+'.'+str(time.time_ns())+'.tmp')
    try:
        with temporary.open('xb') as stream:
            payload=value if raw else (json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode('utf8')
            stream.write(payload);stream.flush();os.fsync(stream.fileno())
        if replace:
            chain.owned(path)
            os.replace(temporary,path)
        else:
            os.link(temporary,path)
    finally:
        if temporary.exists():temporary.unlink()


def witness(path):
    return chain.FileProofCache().capture(chain.owned(path))


def actor_identity(record):
    from .v24_direct_gpu_access import _valid_identity
    value=dict(pid=record.get('pid'),create_time=record.get('create_time'),
        command=copy.deepcopy(record.get('command')),UID=record.get('UID',UID))
    if not _valid_identity(value):raise ValueError('Exact own PID/birth/full argv/UID required')
    return value


def actor_alive(actor):
    return chain.process_snapshot(actor['pid'],actor['create_time'],actor['command']) is not None


def stages(request):
    resource=request['resource_contract'];gpu=request['GPU_arm'];code=Path(request['code'])
    if gpu==5:
        return [dict(name=action.replace('-','_'),action=action,command=[request['python'],'-B','-u',
            str(code/'tools/run_v24_early_stop_downstream.py'),action,'--request',request['early_request_path']])
            for action in chain.ACTIONS]
    values=copy.deepcopy(resource['stages'])
    for row in values:row['command'][3]=str(code/'tools/run_v24_matched_basic_cp.py')
    return values


def validate_request(request):
    if (request.get('format')!=FORMAT or request.get('GPU_arm') not in (4,5)
            or request.get('rescore') is not False or request.get('resume') is not False
            or request.get('native_not_started_at_admission') is not True
            or request.get('native_epochs')!=250 or request.get('native_physical_batch')!=2
            or request.get('evaluation_checkpoint')!='checkpoint_best.pth'):
        raise ValueError('Explicit original GPU45 pre-start full250/B2/BEST26 adoption required')
    resource=chain.validate_declaration(request['resource_contract'])
    root=chain.absolute(request['recovery_root']);code=chain.absolute(request['code'])
    old=request['original_declaration']
    if (request['GPU_arm']==4 and old!=resource
            or request['GPU_arm']==5 and early.validate_request(old)['resource_contract']!=resource
            or request['original_request'].get('path')!=str(Path(old.get('controller_root',old.get('chain_root')))/'request.json')):
        raise ValueError('Exact preserved predecessor/resource request required')
    if (request['GPU_arm']!=resource['GPU_arm'] or request['python']!=resource['python']
            or request['source_output']!=resource['source_output'] or root==Path(resource['chain_root'])
            or code==Path(request['original_declaration']['code'])):
        raise ValueError('Adoption must retain own source/Python with distinct root and fixed code')
    commit=request.get('commit');files=request.get('code_files')
    required=set(FILES)|set(chain.FILES)|set(pipeline.FILES)
    if request['GPU_arm']==5:required.update(early.FILES)
    if (not isinstance(commit,str) or len(commit)!=40 or set(commit)-set('0123456789abcdef')
            or not isinstance(files,dict) or not required<=set(files)):
        raise ValueError('Complete fresh Git/source manifest required')
    for name,digest in files.items():
        if Path(name).is_absolute() or '..' in Path(name).parts or not chain.checksum(digest):
            raise ValueError('Exact relative fixed-code source manifest required')
    actor_identity(request['predecessor_actor']);actor_identity(request['scoring_actor'])
    if request.get('stages')!=stages(request):raise ValueError('Exact unchanged three native stage commands required')
    if request['GPU_arm']==5:
        if request.get('early_request_path')!=str(root/'early/request.json'):
            raise ValueError('Separate exact early request namespace required')
        declared=early.validate_request(request['early_declaration'])
        if (declared['resource_contract']!=resource or declared['source_output']!=request['source_output']
                or declared['controller_root']!=str(root/'early') or declared['code']!=str(code)
                or declared['commit']!=commit or declared['code_files']!=files
                or declared['scoring_root']!=request['original_declaration']['scoring_root']
                or declared['authorization']!=request['original_declaration']['authorization']):
            raise ValueError('Exact authorized early pin/source/scoring adoption required')
    return request


def verify_code(request,cache):
    code=chain.owned(request['code'],directory=True)
    if subprocess.check_output(['git','-C',str(code),'rev-parse','HEAD'],text=True).strip()!=request['commit']:
        raise ValueError('Fresh fixed checkout Git HEAD changed')
    tracked=set(subprocess.check_output(['git','-C',str(code),'ls-files','-z'],text=True).split('\0'))
    if not set(request['code_files'])<=tracked:raise ValueError('Fixed-code manifest contains untracked files')
    for name,digest in request['code_files'].items():
        path=chain.owned(code/name)
        cache.verify(dict(path=str(path),sha256=digest,stat=chain.identity(path)))
    if Path(__file__).resolve()!=code/FILES[0]:raise ValueError('Guard must execute exact fixed checkout')
    tree=ast.parse((code/'hiercp_v1x/v24_matched_basic_cp_runtime.py').read_text(encoding='utf8'))
    function=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='install_native_runtime')
    start=next(node for node in ast.walk(function) if isinstance(node,ast.FunctionDef) and node.name=='start')
    imports=[node for node in ast.walk(start) if isinstance(node,ast.ImportFrom) and node.module=='v24_nnunet_cp']
    resources=[keyword.value for node in ast.walk(start) if isinstance(node,ast.Call)
        for keyword in node.keywords if keyword.arg=='resources']
    if (len(imports)!=1 or [item.name for item in imports[0].names]!=['require_project_budget']
            or len(resources)!=1 or not isinstance(resources[0],ast.Call)
            or not isinstance(resources[0].func,ast.Name) or resources[0].func.id!='require_project_budget'):
        raise ValueError('Actual production on_train_start must use fixed project-bounded observer')


def verify_previous_code(request,cache):
    """The running old scorer's whole checkout remains immutable too."""
    old=request['original_declaration'];code=chain.owned(old['code'],directory=True)
    if subprocess.check_output(['git','-C',str(code),'rev-parse','HEAD'],text=True).strip()!=old['commit']:
        raise ValueError('Original predecessor checkout Git HEAD changed')
    for name,digest in old['code_files'].items():
        path=chain.owned(code/name)
        cache.verify(dict(path=str(path),sha256=digest,stat=chain.identity(path)))


def build_request(original_request_path,recovery_root,code,commit,code_files,inspection_arm):
    """Read-only assembly from current predecessor identity; no signals or writes."""
    path=chain.owned(original_request_path);old=chain.read(path);root=Path(recovery_root).absolute()
    gpu=old['GPU_arm'];resource=old if gpu==4 else old['resource_contract']
    chain.validate_declaration(resource)
    if gpu==5:early.validate_request(old)
    registration=chain.read(path.parent/'registration.json');state=chain.read(path.parent/'status.json')
    lease=chain.read(path.parent/'lease.json')
    parent=actor_identity(registration)
    parent_command=[resource['python'],'-B','-u',str(Path(old['code'])/('tools/watch_v24_scored_bank_to_nnunet.py'
        if gpu==4 else 'tools/run_v24_early_stop_downstream.py')),*([] if gpu==4 else ['watch']),'--request',str(path)]
    if (registration.get('request_sha256')!=chain.sha(path) or parent['command']!=parent_command
            or state.get('request_sha256')!=chain.sha(path) or state.get('supervisor_pid')!=parent['pid']
            or state.get('supervisor_create_time')!=parent['create_time'] or state.get('UID')!=UID
            or lease.get('request_sha256')!=chain.sha(path) or lease.get('command')!=parent_command
            or lease.get('pid',lease.get('supervisor_pid'))!=parent['pid']
            or lease.get('create_time',lease.get('supervisor_create_time'))!=parent['create_time']
            or state.get('full_native_training_completed') is not False
            or state.get('full_native_evaluation_completed') is not False or state.get('resource_violation') is not None):
        raise ValueError('Exact original pre-start controller/request/lease identity required')
    if (inspection_arm.get('GPU')!=gpu or inspection_arm.get('request')!=old
            or inspection_arm.get('registration')!=registration or inspection_arm.get('native_exists') is not False):
        raise ValueError('Captured actual pre-start inspection differs')
    if (Path(resource['chain_root'])/'native').exists():raise FileExistsError('Native already exists; no pre-start replay')
    if gpu==4:
        if state.get('child_pid') is not None or state.get('stage') is not None:
            raise ValueError('GPU4 original supervisor must still be childless before native')
        score_registration=chain.read(Path(resource['scoring_root'])/'registration.json')
        scorer=actor_identity(score_registration)
    else:
        if state.get('status')!='RUNNING' or state.get('stage')!='score' or state.get('stages_completed')!=['pin_early_stopped']:
            raise ValueError('Only genuine authorized GPU5 live score stage can be adopted')
        scorer=actor_identity(dict(pid=state['child_pid'],create_time=state['child_create_time'],command=state['command'],UID=UID))
        if scorer['command']!=old['stages'][1]['command']:raise ValueError('Exact existing GPU5 score child required')
    actor_alive(parent);actor_alive(scorer)  # Foreign PID reuse is never treated as release.
    protected=[witness(path),witness(path.parent/'registration.json'),witness(path.parent/'lease.json')]
    for claim in {resource['native_claim_path'],old.get('claim_path',resource['native_claim_path'])}:
        protected.append(witness(claim))
    if gpu==5:
        protected += [witness(old['pin_path']),witness(path.parent/'stage_pin_early_stopped.json')]
        protected += copy.deepcopy(old['protected_files'])
    request=dict(format=FORMAT,GPU_arm=gpu,recovery_root=str(root),code=str(Path(code).absolute()),
        commit=commit,code_files=copy.deepcopy(code_files),python=resource['python'],source_output=resource['source_output'],
        original_request=protected[0],original_declaration=old,resource_contract=resource,
        predecessor_actor=parent,scoring_actor=scorer,protected_files=protected,
        native_not_started_at_admission=True,rescore=False,resume=False,native_epochs=250,native_physical_batch=2,
        evaluation_checkpoint='checkpoint_best.pth',early_request_path=None)
    if gpu==5:
        request['early_request_path']=str(root/'early/request.json')
        request['early_declaration']=early.build_request(old['original_native_request']['path'],
            controller_root=root/'early',scoring_root=old['scoring_root'],code=code,commit=commit,
            code_files=code_files,target_epoch=old['target_epoch'],authorization_path=old['authorization']['path'],
            release_actors=[*old['release_actors'],parent,scorer],protected_files=protected)
    request['stages']=stages(request)
    return validate_request(request)


def seal_request(request):
    validate_request(request);root=chain.owned(request['recovery_root'],missing=True)
    if root.exists() and list(root.iterdir()):raise FileExistsError('Fresh empty adoption namespace required')
    if not root.exists():chain.owned(root.parent,directory=True);root.mkdir(mode=0o700)
    if request['GPU_arm']==5:
        nested=root/'early';nested.mkdir(mode=0o700)
        source=chain.owned(request['original_declaration']['pin_path']);before=witness(source)
        payload=source.read_bytes();pin=early.validate_current_pin(json.loads(payload))
        if witness(source)!=before:raise ValueError('Original early pin changed while copying metadata')
        publish(nested/'own_early_stop_best_pin.json',payload,raw=True)
        publish(nested/'request.json',request['early_declaration'])
        if early.validate_current_pin(chain.read(nested/'own_early_stop_best_pin.json'))!=pin:
            raise ValueError('Fresh early pin metadata differs')
    publish(root/'request.json',request)
    return root/'request.json'


class Controller:
    """Explicit adoption, never replaying the original fresh-only constructors."""
    def __init__(self,path):
        import psutil
        self.path=chain.owned(path);self.request=validate_request(chain.read(self.path))
        self.root=chain.owned(self.request['recovery_root'],directory=True)
        if self.path!=self.root/'request.json':raise ValueError('Exact own adoption request path required')
        self.resource=self.request['resource_contract'];self.old=self.request['original_declaration']
        self.arm=Path(self.resource['chain_root']);self.digest=chain.sha(self.path)
        self.cache=chain.FileProofCache();self.cache.capture(self.path)
        verify_code(self.request,self.cache);verify_previous_code(self.request,self.cache);chain.verify_code(self.resource,self.cache)
        for item in self.request['protected_files']:self.cache.verify(item)
        if chain.read(self.request['original_request']['path'])!=self.old:raise ValueError('Preserved original request changed')
        if (self.arm/'native').exists() or (self.arm/'matched_storage_prepare_native.json').exists():
            raise FileExistsError('Native already started; no adoption replay')
        actor=psutil.Process()
        if actor.uids().real!=UID or actor.cpu_affinity()!=self.resource['CPU_affinity']:
            raise ValueError('Original own UID/four CPUs required')
        command=[self.request['python'],'-B','-u',str(Path(self.request['code'])/FILES[1]),'watch','--request',str(self.path)]
        if actor.cmdline()!=command:raise ValueError('Exact registered adoption supervisor command required')
        self.scoring=None;self.storage=None
        if self.request['GPU_arm']==5:
            self.cache.capture(self.request['early_request_path'])
            if chain.read(self.request['early_request_path'])!=self.request['early_declaration']:
                raise ValueError('Fresh early request differs')
            early.verify_pin_binding(self.request['early_declaration'],self.cache)
        lease=dict(format=FORMAT,pid=actor.pid,create_time=actor.create_time(),command=command,
            UID=UID,request_sha256=self.digest,recovery_root=str(self.root))
        publish(self.arm/'native_prestart_guard.lease.json',lease)
        self.cache.capture(self.arm/'native_prestart_guard.lease.json')
        publish(self.root/'lease.json',lease);self.cache.capture(self.root/'lease.json')
        self.state=dict(format=STATUS_FORMAT,GPU_arm=self.request['GPU_arm'],request_sha256=self.digest,
            source_output=self.request['source_output'],source_code=self.resource['source_code'],
            supervisor_pid=actor.pid,supervisor_create_time=actor.create_time(),UID=UID,
            actual_CPU_affinity=actor.cpu_affinity(),status='WAITING_FOR_EXISTING_SCORING',stage='score',
            child_pid=self.request['scoring_actor']['pid'],child_create_time=self.request['scoring_actor']['create_time'],
            command=self.request['scoring_actor']['command'],stage_log=str(Path(self.request['original_request']['path']).parent/'score.log')
                if self.request['GPU_arm']==5 else str(Path(self.resource['scoring_root'])/'console.log'),
            stages_completed=[],stage_proofs=[],full_native_training_completed=False,
            full_native_evaluation_completed=False,signals_sent=False,rescore=False,
            full_GNN_training=self.request['GPU_arm']==4,adopted_GNN_epochs=40 if self.request['GPU_arm']==4 else self.old['target_epoch'])

    def update(self,**values):
        self.state.update(values,checked_at=time.time());publish(self.root/'status.json',self.state,replace=(self.root/'status.json').exists())

    def adopt_scoring(self):
        self.cache.check()
        parent_live=actor_alive(self.request['predecessor_actor']);score_live=actor_alive(self.request['scoring_actor'])
        if parent_live or score_live:
            self.update(status='WAITING_FOR_EXISTING_SCORING',stage='score',
                waiting_reason='Exact predecessor/scorer release and complete full642 evidence pending; existing CP continues')
            return False
        old_path=Path(self.request['original_request']['path'])
        publish(self.root/'handoff.json',dict(format=FORMAT,predecessor_actor=self.request['predecessor_actor'],
            scoring_actor=self.request['scoring_actor'],predecessor_status=witness(old_path.parent/'status.json'),
            signals_sent=False,original_files_overwritten=False))
        if self.request['GPU_arm']==4:
            chain.preserve_predecessors(self.resource,self.cache)
            completion=chain.source_complete(self.resource)
            if completion is None:raise RuntimeError('Original full40/680 GNN proof or actor release missing')
            self.scoring=chain.scoring_complete(self.resource,self.cache)
            if self.scoring is None:raise RuntimeError('Released scorer lacks complete whole642/82176 proof')
            if (self.arm/'admitted_scoring.json').exists():
                if chain.read(self.arm/'admitted_scoring.json')!=self.scoring:raise ValueError('Original admitted scoring differs')
            else:publish(self.arm/'admitted_scoring.json',self.scoring)
        else:
            receipt_path=old_path.parent/'stage_score.json';receipt=chain.read(receipt_path)
            overlay=Path(self.old['scoring_root'])/'bank/score_overlay.json'
            if (receipt.get('format')!=early.FORMAT or receipt.get('status')!='COMPLETE'
                    or receipt.get('action')!='score' or receipt.get('request_sha256')!=self.request['original_request']['sha256']
                    or receipt.get('full_GNN_training') is not False or receipt.get('completed_GNN_epochs')!=self.old['target_epoch']
                    or receipt.get('actual_CUDA') is not True or receipt.get('completed_sources')!=642
                    or receipt.get('completed_CP_positions')!=82176 or receipt.get('output')!=str(overlay)
                    or receipt.get('output_sha256')!=chain.sha(chain.owned(overlay))):
                raise ValueError('Actual original CUDA642/82176 truthful early score receipt required')
            pin=early.verify_pin_binding(self.request['early_declaration'],self.cache)
            binding=early.private_modules()['EARLY_PIPELINE'].validate_profiled_overlay(overlay,full_hash=True,expected_depth1=True)
            if binding['overlay']['pin']!=pin:raise ValueError('Adopted CP overlay differs from own early BEST')
            admitted=chain.read(self.arm/'admitted_scoring.json')
            if (admitted.get('format')!=early.FORMAT or admitted.get('full_GNN_training') is not False
                    or admitted.get('completed_GNN_epochs')!=self.old['target_epoch']
                    or admitted.get('pin',{}).get('path')!=self.old['pin_path']
                    or admitted.get('overlay',{}).get('path')!=str(overlay)):
                raise ValueError('Original admitted early scoring/source proof differs')
            for key in ('pin','overlay'):self.cache.verify(admitted[key])
            self.scoring=dict(receipt=self.cache.capture(receipt_path),overlay=self.cache.capture(overlay),
                actual_CUDA=True,completed_sources=642,completed_CP_positions=82176,full_GNN_training=False,
                completed_GNN_epochs=self.old['target_epoch'])
        self.cache.capture(self.arm/'admitted_scoring.json')
        publish(self.root/'adopted_scoring.json',self.scoring)
        self.update(status='SCORING_ADOPTED',stage=None,child_pid=None,child_create_time=None,
            waiting_reason=None,adopted_scoring=self.scoring,stages_completed=['adopted_existing_scoring'])
        return True

    def ready(self,action):
        self.cache.check()
        if actor_alive(self.request['predecessor_actor']) or actor_alive(self.request['scoring_actor']):
            raise RuntimeError('Exact predecessor/scorer reappeared; no native work')
        self.storage=chain.storage_ready(self.resource,self.cache)
        resources=chain.resources_ready(self.resource,action)
        self.update(status='READY_FOR_NATIVE_STAGE',stage=None,child_pid=None,waiting_reason=None,
            storage=self.storage,resources=resources)

    def execute(self,stage):
        import psutil
        self.ready(stage['action'])
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',CUDA_DEVICE_ORDER='PCI_BUS_ID',PYTHONDONTWRITEBYTECODE='1',
            OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',V24_ARM_RSS_GIB='48',
            V24_READONLY_STORAGE_ADMISSION_SHA256=self.storage['sha256'])
        if self.request['GPU_arm']==5:
            env[early.ENV_REQUEST]=self.request['early_request_path'];env[early.ENV_SHA]=chain.sha(self.request['early_request_path'])
        else:
            env['V24_CONTEXT_DEPTH1_CHAIN_REQUEST']=self.request['original_request']['path']
            env['V24_CONTEXT_DEPTH1_CHAIN_REQUEST_SHA256']=self.request['original_request']['sha256']
        log=self.root/(stage['name']+'.log');command=stage['command']
        self.update(status='RUNNING',stage=stage['name'],command=command,stage_log=str(log))
        with log.open('xb',buffering=0) as stream:
            child=subprocess.Popen(command,cwd=self.request['code'],env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
            actor=psutil.Process(child.pid);chain.process_snapshot(child.pid,actor.create_time(),command)
            self.update(child_pid=child.pid,child_create_time=actor.create_time(),stage_started=time.time())
            violation=None
            while child.poll() is None:
                try:self.cache.check()
                except (OSError,ValueError,RuntimeError) as error:violation=type(error).__name__+': '+str(error)
                try:
                    rss=actor.memory_info().rss
                    for descendant in actor.children(recursive=True):
                        try:rss+=descendant.memory_info().rss
                        except psutil.NoSuchProcess:continue
                except psutil.NoSuchProcess:
                    if child.poll() is None:raise RuntimeError('Exact native child disappeared without exit record')
                    break
                if rss>48*2**30:violation='Native child tree exceeds original48GiB; completion refused'
                self.update(child_process_tree_RSS_bytes=rss,RAM_budget_bytes=48*2**30,resource_violation=violation)
                time.sleep(5)
            result=child.wait()
        if result or violation:raise RuntimeError('Adopted native stage failed: '+stage['action']+' '+str(result)+' '+str(violation))
        if self.request['GPU_arm']==5:
            receipt=chain.read(Path(self.request['early_request_path']).parent/('stage_'+stage['name']+'.json'))
            if (receipt.get('status')!='COMPLETE' or receipt.get('action')!=stage['action']
                    or receipt.get('request_sha256')!=chain.sha(self.request['early_request_path'])
                    or receipt.get('full_GNN_training') is not False):raise ValueError('Exact adopted early stage receipt required')
            proof=early.native_proof(self.request['early_declaration'],stage['action'])
        else:proof=chain.native_stage_proof(self.resource,stage['action'])
        self.cache.check();self.state['stages_completed'].append(stage['name']);self.state['stage_proofs'].append(proof)
        self.update(child_pid=None,child_create_time=None)
        return proof

    def run(self):
        self.update()
        try:
            while not self.adopt_scoring():time.sleep(5)
            for stage in self.request['stages']:
                while True:
                    try:
                        proof=self.execute(stage);break
                    except (chain.WaitingForStorage,chain.WaitingForGPU,chain.WaitingForAllocation) as error:
                        self.update(status='WAITING_FOR_NATIVE_RESOURCES',stage=None,child_pid=None,waiting_reason=str(error));time.sleep(5)
            self.update(status='COMPLETE',stage=None,child_pid=None,child_create_time=None,
                full_native_training_completed=True,full_native_evaluation_completed=True,completion_proof=proof,
                evaluation_checkpoint='checkpoint_best.pth',completed=time.time())
        except Exception as error:
            self.update(status='FAILED',error=type(error).__name__+': '+str(error),failed=time.time());raise
        return self.state


def watch(path):
    sys.dont_write_bytecode=True
    return Controller(path).run()
