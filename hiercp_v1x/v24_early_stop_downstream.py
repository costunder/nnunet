"""Explicit user-authorized early GNN BEST -> full642 CP -> full250 native.

The original full40/680 admission functions, module globals and source files
are never changed. Private copies redirect only pin-admission imports and the
native child entry point. Every scoring/model/loader/training expression is
retained, and the early pin keeps its own format and full_training=False.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import types

from . import v24_scored_native_chain as chain

ROOT = Path(__file__).resolve().parents[1]
FORMAT = 'v24_user_authorized_early_stop_full642_full250_request_v1'
STATUS_FORMAT = 'v24_user_authorized_early_stop_full642_full250_status_v1'
CLAIM_FORMAT = 'v24_user_authorized_early_stop_downstream_exclusive_claim_v1'
CURRENT_PIN_FORMAT = 'v24_user_authorized_epoch_boundary_early_stop_best_v1'
ACTIONS = ('pin-early-stopped', 'score', 'prepare-native', 'calibrate-native', 'train')
FILES = ('hiercp_v1x/v24_early_stop_gnn.py', 'hiercp_v1x/v24_early_stop_downstream.py',
    'tools/run_v24_early_stop_downstream.py')
ENV_REQUEST = 'V24_EARLY_STOP_DOWNSTREAM_REQUEST'
ENV_SHA = 'V24_EARLY_STOP_DOWNSTREAM_REQUEST_SHA256'
_PRIVATE = {}
_GUARDS = {}


def __getattr__(name):
    if name == 'ORIGINAL_API':
        return sys.modules[__name__]
    if name in ('EARLY_RUNTIME', 'EARLY_PIPELINE'):
        return private_modules()[name]
    from . import v24_nnunet_cp as original
    return getattr(original, name)


def validate_current_pin(pin):
    from .v24_early_stop_gnn import validate_early_stop_pin
    return validate_early_stop_pin(pin)


def admit_pin(path, inventory):
    from .v24_early_stop_gnn import admit_early_stop_pin
    return admit_early_stop_pin(path, inventory)


def _admit_arm_gpu(pin, gpu):
    validate_current_pin(pin)
    if gpu != 5 or pin['physical_GPU'] != gpu:
        raise ValueError('Authorized early-stop downstream belongs only to original GPU5')


def pin_parameters(pin):
    return validate_current_pin(pin)['model_contract']['parameters']


def _inspect_completed_current(source_output, source_code, inventory, *, gpu, input_cache, stunet_checkpoint=None):
    request, _ = environment_request()
    if stunet_checkpoint is not None or gpu != 5:
        raise ValueError('Original GPU5 CNN/GAT cannot receive STU weights')
    from .v24_early_stop_gnn import inspect_early_stop
    return inspect_early_stop(source_output, source_code, inventory, input_cache=input_cache,
        target_epoch=request['target_epoch'], authorization_path=request['authorization']['path'], gpu=gpu)


def _private_clone(module, *, replacements=None, overrides=None):
    """Only explicit dependency imports/child entry constants may be changed."""
    source = inspect.getsource(module)
    tree = ast.parse(source)
    counts = {}
    constants = dict(replacements or {})
    hits = {value: 0 for value in constants}
    class Redirect(ast.NodeTransformer):
        function = None
        def visit_FunctionDef(self, node):
            previous = self.function
            self.function = node.name
            result = self.generic_visit(node)
            self.function = previous
            return result
        def visit_ImportFrom(self, node):
            if node.level != 1:
                return node
            if node.module == 'v24_nnunet_cp':
                counts[node.module] = counts.get(node.module, 0) + 1
                return ast.copy_location(ast.ImportFrom(module='v24_early_stop_downstream', names=node.names, level=1), node)
            if node.module == 'v24_matched_basic_cp_pipeline':
                if any(name.name != 'early_stage_guard' for name in node.names):
                    raise ValueError('Unexpected native pipeline dependency; explicit review required')
                counts[node.module] = counts.get(node.module, 0) + 1
                return ast.copy_location(ast.ImportFrom(module='v24_early_stop_downstream', names=node.names, level=1), node)
            if node.module is None:
                redirected = {'v24_nnunet_cp': 'ORIGINAL_API', 'v24_matched_basic_cp_runtime': 'EARLY_RUNTIME'}
                nodes = []
                for alias in node.names:
                    if alias.name in redirected:
                        counts[alias.name] = counts.get(alias.name, 0) + 1
                        nodes.append(ast.copy_location(ast.ImportFrom(module='v24_early_stop_downstream',
                            names=[ast.alias(name=redirected[alias.name], asname=alias.asname or alias.name)], level=1), node))
                    else:
                        nodes.append(ast.copy_location(ast.ImportFrom(module=None, names=[alias], level=1), node))
                return nodes
            return node
        def visit_Constant(self, node):
            if isinstance(node.value, str) and node.value in constants and self.function in ('calibrate_native','train'):
                hits[node.value] += 1
                return ast.copy_location(ast.Constant(constants[node.value]), node)
            return node
    tree = ast.fix_missing_locations(Redirect().visit(tree))
    if any(value != 1 for value in hits.values()):
        raise ValueError('Exactly one original child entry constant required: ' + str(hits))
    clone = types.ModuleType(module.__name__ + '_private_early_stop')
    clone.__dict__.update(__package__='hiercp_v1x', __file__=module.__file__)
    if clone.__name__ in sys.modules:
        raise RuntimeError('Private early adapter already exists; fresh stage process required')
    sys.modules[clone.__name__] = clone  # Dataclass annotation lookup needs its own module.
    exec(compile(tree, module.__file__ + ':explicit_early_admission', 'exec'), clone.__dict__)
    clone.__dict__.update(overrides or {})
    proof = dict(original_source_sha256=chain.sha(module.__file__),
        original_AST_sha256=hashlib.sha256(ast.dump(ast.parse(source)).encode()).hexdigest(),
        adapted_AST_sha256=hashlib.sha256(ast.dump(tree).encode()).hexdigest(),
        admission_import_redirects=counts, child_entry_redirects=hits,
        model_geometry_loss_optimizer_and_loader_expressions_changed=False,
        original_module_globals_changed=False)
    return clone, proof


def private_modules():
    if not _PRIVATE:
        from . import v24_matched_basic_cp_scoring as scoring
        from . import v24_matched_basic_cp_runtime as runtime
        from . import v24_matched_basic_cp_pipeline as pipeline
        runtime_copy, runtime_proof = _private_clone(runtime)
        pipeline_copy, pipeline_proof = _private_clone(pipeline, replacements={
            'tools/run_v24_matched_basic_cp.py': 'tools/run_v24_early_stop_downstream.py',
            'from hiercp_v1x.v24_matched_basic_cp_runtime import run_training_entry; run_training_entry()':
            'from hiercp_v1x.v24_early_stop_downstream import run_training_entry; run_training_entry()'},
            overrides={'early_stage_guard': early_stage_guard})
        scoring_copy, scoring_proof = _private_clone(scoring)
        _PRIVATE.update(EARLY_RUNTIME=runtime_copy, EARLY_PIPELINE=pipeline_copy,
            EARLY_SCORING=scoring_copy, proofs=dict(runtime=runtime_proof, pipeline=pipeline_proof, scoring=scoring_proof))
    return _PRIVATE


def claim_path(request):
    source = chain.absolute(request['source_output'])
    key = hashlib.sha256(('GPU5:' + str(source) + ':' + chain.PROFILE).encode()).hexdigest()
    return source.parent / 'early_stop_downstream_claims' / (key + '.json')


def build_request(native_request_path, *, controller_root, scoring_root, code, commit,
        code_files, target_epoch, authorization_path, release_actors, protected_files):
    """Read-only declaration assembly; registration/launch belongs to installer."""
    native_path = chain.owned(native_request_path)
    native = copy.deepcopy(chain.validate_declaration(chain.read(native_path)))
    request = dict(format=FORMAT, GPU_arm=5, source_output=native['source_output'],
        source_code=native['source_code'], source_commit=native['source_commit'],
        controller_root=str(chain.absolute(str(controller_root))), scoring_root=str(chain.absolute(str(scoring_root))),
        code=str(chain.absolute(str(code))), commit=commit, code_files=copy.deepcopy(code_files),
        target_epoch=target_epoch, original_planned_epochs=40, original_planned_optimizer_updates=680,
        full_GNN_training=False, source_completion_kind='USER_AUTHORIZED_EPOCH_BOUNDARY_EARLY_STOP',
        context_depth1=True, source_entries=642, CP_positions=82176,
        native_epochs=250, native_physical_batch=2, full_native_evaluation_cases=26,
        debug=False, resource_contract=native,
        original_native_request=dict(path=str(native_path), sha256=chain.sha(native_path), stat=chain.identity(native_path)),
        authorization=dict(path=str(chain.owned(authorization_path)), sha256=chain.sha(authorization_path), stat=chain.identity(authorization_path)),
        release_actors=copy.deepcopy(release_actors), protected_files=copy.deepcopy(protected_files))
    root = Path(request['controller_root'])
    request['pin_path'] = str(root / 'own_early_stop_best_pin.json')
    entry = str(Path(request['code']) / 'tools/run_v24_early_stop_downstream.py')
    request['stages'] = [dict(name=action.replace('-', '_'), action=action,
        command=[native['python'], '-B', '-u', entry, action, '--request', str(root / 'request.json')]) for action in ACTIONS]
    request['claim_path'] = str(claim_path(request))
    return validate_request(request)


def validate_request(request):
    if (request.get('format') != FORMAT or request.get('GPU_arm') != 5
            or type(request.get('target_epoch')) is not int or not 1 <= request['target_epoch'] < 40
            or request.get('original_planned_epochs') != 40 or request.get('original_planned_optimizer_updates') != 680
            or request.get('full_GNN_training') is not False
            or request.get('source_completion_kind') != 'USER_AUTHORIZED_EPOCH_BOUNDARY_EARLY_STOP'
            or request.get('context_depth1') is not True or request.get('source_entries') != 642
            or request.get('CP_positions') != 82176 or request.get('native_epochs') != 250
            or request.get('native_physical_batch') != 2 or request.get('full_native_evaluation_cases') != 26
            or request.get('debug') is not False):
        raise ValueError('Explicit authorized early GNN/full642/full250/B2/BEST26 contract required')
    native = chain.validate_declaration(request['resource_contract'])
    if native['GPU_arm'] != 5 or any(request[key] != native[key] for key in ('source_output', 'source_code', 'source_commit')):
        raise ValueError('Early downstream must keep actual GPU5 source and unchanged native/storage contract')
    paths = {key:chain.absolute(request[key]) for key in ('controller_root','scoring_root','code','pin_path','claim_path')}
    if paths['pin_path'] != paths['controller_root']/'own_early_stop_best_pin.json' or paths['claim_path'] != claim_path(request):
        raise ValueError('Fresh own early pin and globally derived exclusive claim required')
    protected = [Path(native[key]) for key in ('chain_root','scoring_root','source_output','source_code','input_cache','baseline_preprocessed','historical_basic_bank')]
    for fresh in (paths['controller_root'], paths['scoring_root']):
        if any(fresh == old or fresh.is_relative_to(old) or old.is_relative_to(fresh) for old in [*protected, paths['code']]):
            raise ValueError('Fresh controller/scoring namespace overlaps protected original artifacts')
    if paths['controller_root'] == paths['scoring_root'] or paths['controller_root'].is_relative_to(paths['scoring_root']) or paths['scoring_root'].is_relative_to(paths['controller_root']):
        raise ValueError('Independent fresh controller and scoring namespaces required')
    if (not isinstance(request.get('commit'), str) or len(request['commit']) != 40
            or set(request['commit']) - set('0123456789abcdef')
            or not isinstance(request.get('code_files'), dict) or not set(FILES) <= set(request['code_files'])):
        raise ValueError('Fresh tracked early adapter/early pin source commit and manifest required')
    for name, digest in request['code_files'].items():
        if Path(name).is_absolute() or '..' in Path(name).parts or not chain.checksum(digest):
            raise ValueError('Exact relative code SHA required')
    for key in ('authorization', 'original_native_request'):
        witness = request.get(key)
        if not isinstance(witness, dict) or set(witness) != {'path','sha256','stat'} or not chain.checksum(witness['sha256']):
            raise ValueError('Sealed own authorization/original-native request witness required')
        chain.absolute(witness['path'])
    if not isinstance(request.get('release_actors'), list) or not request['release_actors']:
        raise ValueError('Exact original GNN/native/scoring waiter release identities required')
    from .v24_direct_gpu_access import _valid_identity
    if any(not _valid_identity(actor) for actor in request['release_actors']):
        raise ValueError('Original actors need PID/birth/full argv/UID41833')
    if not isinstance(request.get('protected_files'), list):
        raise ValueError('Explicit original claim/controller metadata preservation witnesses required')
    entry = str(paths['code']/'tools/run_v24_early_stop_downstream.py')
    expected = [dict(name=action.replace('-','_'), action=action,
        command=[native['python'],'-B','-u',entry,action,'--request',str(paths['controller_root']/'request.json')]) for action in ACTIONS]
    if request.get('stages') != expected:
        raise ValueError('Exact early pin/full642/native preparation/calibration/full train commands required')
    return request


def verify_deployment(request, cache):
    cache.verify(request['authorization']); cache.verify(request['original_native_request'])
    if chain.read(request['original_native_request']['path']) != request['resource_contract']:
        raise ValueError('Original native resource contract changed')
    chain.verify_code(request['resource_contract'], cache)
    code = chain.owned(request['code'], directory=True)
    if subprocess.check_output(['git','-C',str(code),'rev-parse','HEAD'], text=True).strip() != request['commit']:
        raise ValueError('Fresh early adapter checkout HEAD changed')
    tracked = set(subprocess.check_output(['git','-C',str(code),'ls-files','-z'], text=True).split('\0'))
    if not set(request['code_files']) <= tracked:
        raise ValueError('Early source manifest includes untracked files')
    for name, digest in request['code_files'].items():
        path = chain.owned(code/name)
        cache.verify(dict(path=str(path),sha256=digest,stat=chain.identity(path)))
    for witness in request['protected_files']:
        cache.verify(witness)
    if Path(__file__).resolve() != code/'hiercp_v1x/v24_early_stop_downstream.py':
        raise ValueError('Early adapter must run from its exact sealed fresh checkout')


def released(request):
    for actor in request['release_actors']:
        if chain.process_snapshot(actor['pid'],actor['create_time'],actor['command']) is not None:
            return False
    return True


def verify_pin_binding(request, cache):
    """Bind the truthful early pin and every frozen original source witness."""
    path = chain.owned(request['pin_path']); cache.capture(path)
    pin = validate_current_pin(chain.read(path))
    resource = request['resource_contract']
    if (pin['completed_epochs'] != request['target_epoch']
            or pin['source_output'] != request['source_output'] or pin['source_code'] != request['source_code']
            or pin['inventory'] != resource['inventory'] or pin['input_cache'] != resource['input_cache']
            or pin['early_stop_authorization']['path'] != request['authorization']['path']
            or pin['early_stop_authorization']['sha256'] != request['authorization']['sha256']):
        raise ValueError('Actual own early pin/source/inventory/authorization differs')
    for proof in pin['early_stop_source_file_proofs'].values():
        cache.verify(proof)
    return pin


def environment_request():
    path = chain.owned(os.environ[ENV_REQUEST]); digest = os.environ[ENV_SHA]
    if not chain.checksum(digest) or chain.sha(path) != digest:
        raise ValueError('Immutable early downstream environment request changed')
    return validate_request(chain.read(path)), path


def early_stage_guard(*, stage_action, gpu=None, chain_root=None):
    request, path = environment_request()
    if gpu not in (None,5):
        raise ValueError('Early stage GPU must remain physical GPU5')
    resource = request['resource_contract']
    action = 'prepare-current-bank' if stage_action == 'score' else stage_action
    if action == 'pin-early-stopped':
        action = 'pin-current-gnn'
    from . import v24_direct_gpu_access as direct
    key = (str(path),os.environ[ENV_SHA],action)
    if key not in _GUARDS:
        cache = chain.FileProofCache(); cache.capture(path); verify_deployment(request,cache)
        if Path(request['pin_path']).exists():
            verify_pin_binding(request,cache)
        if not released(request):
            raise RuntimeError('Exact original GPU5 actors are still alive; no early downstream CUDA')
        validated = direct.validate_controller_request(resource)
        access = direct.make_guard(validated['proof_path'],5,resource['CPU_affinity'],stage_action=action)
        os.environ['CUDA_VISIBLE_DEVICES'] = resource['GPU_UUID']
        def guard():
            cache.check()
            if not released(request):
                raise RuntimeError('Original GPU5 actor reappeared; no downstream continuation')
            return access()
        _GUARDS[key] = guard
    _GUARDS[key]()
    return _GUARDS[key]


def native_proof(request, action):
    """Reuse original full250/BEST26 proof with only early pin boundary redirected."""
    from . import v24_scored_native_chain as original
    function_source = inspect.getsource(original.native_stage_proof)
    holder = types.ModuleType('hiercp_v1x._early_native_proof')
    holder.__dict__.update(original.native_stage_proof.__globals__)
    holder.__dict__.update(__package__='hiercp_v1x')
    tree = ast.parse(function_source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module is None:
            for alias in node.names:
                if alias.name == 'v24_nnunet_cp':
                    node.module = 'v24_early_stop_downstream'; alias.name = 'ORIGINAL_API'
                elif alias.name == 'v24_matched_basic_cp_pipeline':
                    node.module = 'v24_early_stop_downstream'; alias.name = 'EARLY_PIPELINE'
    holder.__dict__['external_overlay_for_stage'] = lambda _: Path(request['scoring_root'])/'bank/score_overlay.json'
    exec(compile(ast.fix_missing_locations(tree),__file__+':original_native_proof','exec'),holder.__dict__)
    resource = copy.deepcopy(request['resource_contract'])
    resource['scoring_root'] = request['scoring_root']
    return holder.native_stage_proof(resource,action)


def stage(request_path, action):
    request_path = chain.owned(request_path); request = validate_request(chain.read(request_path))
    os.environ[ENV_REQUEST] = str(request_path); os.environ[ENV_SHA] = chain.sha(request_path)
    cache = chain.FileProofCache(); cache.capture(request_path); verify_deployment(request,cache)
    if not released(request):
        raise RuntimeError('Original exact GPU5 GNN/controller/waiter still active')
    resource = request['resource_contract']; arm = Path(resource['chain_root'])
    storage = chain.storage_ready(resource,cache)
    os.environ['V24_READONLY_STORAGE_ADMISSION_SHA256'] = storage['sha256']
    if action == 'pin-early-stopped':
        from .v24_early_stop_gnn import pin_early_stopped
        result = pin_early_stopped(source_output=request['source_output'],source_code=request['source_code'],
            inventory_path=resource['inventory'],input_cache=resource['input_cache'],
            target_epoch=request['target_epoch'],authorization_path=request['authorization']['path'],
            output=request['pin_path'],gpu=5)
    else:
        pin = verify_pin_binding(request,cache)
        modules = private_modules(); pipeline = modules['EARLY_PIPELINE']
        guard = early_stage_guard(stage_action=action,gpu=5)
        overlay = Path(request['scoring_root'])/'bank/score_overlay.json'
        if action == 'score':
            result = modules['EARLY_SCORING']._score_matched_basic_bank(pin_path=request['pin_path'],
                inventory_path=resource['inventory'],baseline_preprocessed=resource['baseline_preprocessed'],
                input_cache=resource['input_cache'],source_bank=resource['historical_basic_bank'],
                output=overlay.parent,gpu=5,stunet_checkpoint=None,
                depth1_context_profile=True,depth1_debug=False,deadline_guard=guard)
            pipeline.validate_profiled_overlay(result,full_hash=True,expected_depth1=True)
            chain.publish(arm/'admitted_scoring.json',dict(pin=dict(path=request['pin_path'],sha256=chain.sha(request['pin_path']),stat=chain.identity(request['pin_path'])),
                format=FORMAT,source_completion_kind=request['source_completion_kind'],full_GNN_training=False,
                completed_GNN_epochs=pin['completed_epochs'],completed_GNN_optimizer_updates=pin['optimizer_updates'],
                overlay=dict(path=str(result),sha256=chain.sha(result),stat=chain.identity(result))))
        elif action == 'prepare-native':
            result = pipeline.prepare_native(overlay,arm/'native',storage['path'],gpu=5)
        elif action in ('calibrate-native','train'):
            result = (pipeline.calibrate_native(arm/'native/native.json',gpu=5) if action=='calibrate-native'
                else pipeline.train(arm/'native/native.json',gpu=5))
        else:
            raise ValueError('Unsupported early downstream stage: '+action)
        if action in chain.ACTIONS:
            chain.publish(arm/('matched_storage_'+action.replace('-','_')+'.json'),dict(
                format=pipeline.STAGE_FORMAT,status='COMPLETE',action=action,storage_profile=pipeline.PROFILE,
                storage_admission=storage['path'],storage_admission_sha256=storage['sha256'],
                runtime_sources_sha256={name:chain.sha(ROOT/name) for name in pipeline.FILES},
                cached_inputs_written=False,model_data_scale_preserved=True,source_completion_kind=request['source_completion_kind'],
                full_GNN_training=False,completed_GNN_epochs=pin['completed_epochs'],native_epochs=250,
                native_physical_batch=2,cp_probability=.5,source_entries=642,candidates_per_source=128,
                source_payloads_written=False,evaluation_checkpoint='checkpoint_best.pth',completed_at=time.time()))
            native_proof(request,action)
    cache.check()
    receipt = dict(format=FORMAT,status='COMPLETE',action=action,request_sha256=chain.sha(request_path),
        full_GNN_training=False,completed_GNN_epochs=request['target_epoch'],original_planned_epochs=40,
        output=str(result),output_sha256=chain.sha(result),completed_at=time.time())
    if action == 'score':
        import torch
        if not torch.cuda.is_initialized():
            raise RuntimeError('Full642 early scorer returned without actual CUDA initialization')
        receipt.update(actual_CUDA=True,completed_sources=642,completed_CP_positions=82176,
            completed_GNN_optimizer_updates=pin['optimizer_updates'],
            peak_CUDA_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_CUDA_reserved_bytes=torch.cuda.max_memory_reserved())
    if _PRIVATE:
        receipt['private_admission_adapter_proofs'] = _PRIVATE['proofs']
    chain.publish(Path(request['controller_root'])/('stage_'+action.replace('-','_')+'.json'),receipt)
    return result


def calibration_worker(native_path, physical_batch, output):
    return private_modules()['EARLY_PIPELINE'].calibration_worker(native_path,physical_batch,output)


def run_training_entry():
    return private_modules()['EARLY_RUNTIME'].run_training_entry()


class Controller:
    def __init__(self, request_path):
        import psutil
        self.path = chain.owned(request_path); self.request = validate_request(chain.read(self.path))
        self.root = chain.owned(self.request['controller_root'],directory=True)
        if self.path != self.root/'request.json':
            raise ValueError('Fresh controller request must remain in own sealed root')
        self.cache = chain.FileProofCache(); self.cache.capture(self.path)
        verify_deployment(self.request,self.cache)
        actor = psutil.Process()
        if actor.cpu_affinity() != self.request['resource_contract']['CPU_affinity'] or actor.uids().real != 41833:
            raise ValueError('Original own UID41833/four GPU5 CPUs required')
        for fresh in (self.root/'status.json',self.root/'lease.json',Path(self.request['pin_path']),Path(self.request['scoring_root'])/'bank'):
            if fresh.exists():
                raise FileExistsError('Fresh early run required; explicit recovery for existing outputs: '+str(fresh))
        arm = Path(self.request['resource_contract']['chain_root'])
        if (arm/'native').exists() or (arm/'admitted_scoring.json').exists():
            raise FileExistsError('Original GPU5 native arm already has downstream output; no duplicate/overwrite')
        claim = claim_path(self.request); chain.owned(claim.parent,directory=True)
        self.digest = chain.sha(self.path)
        chain.publish(claim,dict(format=CLAIM_FORMAT,request_sha256=self.digest,controller_root=str(self.root),
            source_output=self.request['source_output'],GPU_arm=5,target_epoch=self.request['target_epoch']))
        chain.publish(self.root/'lease.json',dict(pid=actor.pid,create_time=actor.create_time(),command=actor.cmdline(),
            UID=41833,request_sha256=self.digest,claim=str(claim),claim_sha256=chain.sha(claim)))
        self.cache.capture(claim)
        self.state = dict(format=STATUS_FORMAT,GPU_arm=5,request_sha256=self.digest,source_output=self.request['source_output'],
            supervisor_pid=actor.pid,supervisor_create_time=actor.create_time(),UID=41833,
            full_GNN_training=False,original_planned_epochs=40,authorized_stop_epoch=self.request['target_epoch'],
            full_native_training_completed=False,full_native_evaluation_completed=False,
            stages_completed=[],child_pid=None,child_create_time=None,status='WAITING_FOR_EARLY_STOP_RELEASE',
            source_completion_kind=self.request['source_completion_kind'],signals_sent=False)

    def update(self, **values):
        self.state.update(values,checked_at=time.time())
        chain.publish(self.root/'status.json',self.state,replace=True)

    def ready(self, action):
        self.cache.check()
        if not released(self.request):
            self.update(status='WAITING_FOR_EARLY_STOP_RELEASE',stage=None,child_pid=None,
                waiting_reason='Exact original GNN/native/scoring waiter actor release is pending; no CUDA started')
            return False
        storage = chain.storage_ready(self.request['resource_contract'],self.cache)
        resources = chain.resources_ready(self.request['resource_contract'], 'prepare-native' if action in ('pin-early-stopped','score') else action)
        self.update(status='READY',stage=None,child_pid=None,waiting_reason=None,storage=storage,resources=resources)
        return True

    def execute(self, stage):
        import psutil
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES='',CUDA_DEVICE_ORDER='PCI_BUS_ID',PYTHONDONTWRITEBYTECODE='1',
            OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
            V24_ARM_RSS_GIB='64' if stage['action'] in ('pin-early-stopped','score') else '48')
        env[ENV_REQUEST] = str(self.path); env[ENV_SHA] = self.digest
        log = self.root/(stage['name']+'.log')
        self.update(status='RUNNING',stage=stage['name'],command=stage['command'],stage_log=str(log))
        with log.open('xb',buffering=0) as stream:
            child = subprocess.Popen(stage['command'],cwd=self.request['code'],env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
            actor = psutil.Process(child.pid)
            chain.process_snapshot(child.pid,actor.create_time(),stage['command'])
            self.update(child_pid=child.pid,child_create_time=actor.create_time())
            violation = None
            while child.poll() is None:
                try:
                    self.cache.check()
                except (ValueError,OSError,RuntimeError) as error:
                    violation = type(error).__name__+': '+str(error)
                try:
                    rss = actor.memory_info().rss
                    for descendant in actor.children(recursive=True):
                        try:
                            rss += descendant.memory_info().rss
                        except psutil.NoSuchProcess:
                            continue
                except psutil.NoSuchProcess:
                    if child.poll() is None:
                        raise RuntimeError('Exact own early downstream child disappeared without recorded exit')
                    break
                budget = (64 if stage['action'] in ('pin-early-stopped','score') else 48)*2**30
                if rss > budget:
                    violation = 'Actual stage process tree exceeded original RAM budget; no next stage'
                self.update(child_process_tree_RSS_bytes=rss,RAM_budget_bytes=budget,resource_violation=violation)
                time.sleep(5)
            result = child.wait()
        if result or violation:
            raise RuntimeError('Early downstream stage failed: '+stage['action']+' code='+str(result)+' '+str(violation)+'; original logs preserved; no process signaled')
        receipt = chain.read(self.root/('stage_'+stage['name']+'.json'))
        if receipt.get('status') != 'COMPLETE' or receipt.get('request_sha256') != self.digest or receipt.get('action') != stage['action'] or receipt.get('full_GNN_training') is not False:
            raise ValueError('Exact truthful early-stage completion receipt required')
        output = chain.owned(receipt['output'])
        if chain.sha(output) != receipt['output_sha256']:
            raise ValueError('Actual early stage output differs from its sealed receipt')
        if stage['action'] == 'pin-early-stopped':
            verify_pin_binding(self.request,self.cache)
        if stage['action'] == 'score':
            if receipt.get('actual_CUDA') is not True or receipt.get('completed_sources') != 642 or receipt.get('completed_CP_positions') != 82176:
                raise ValueError('Actual CUDA whole642/82176 scoring receipt required')
            private_modules()['EARLY_PIPELINE'].validate_profiled_overlay(output,full_hash=True,expected_depth1=True)
        if stage['action'] in chain.ACTIONS:
            proof = native_proof(self.request,stage['action'])
        else:
            proof = receipt
        self.state['stages_completed'].append(stage['name'])
        self.update(child_pid=None,child_create_time=None)
        return proof

    def run(self):
        self.update()
        try:
            for stage in self.request['stages']:
                while True:
                    try:
                        if self.ready(stage['action']):
                            break
                    except (chain.WaitingForStorage,chain.WaitingForGPU,chain.WaitingForAllocation) as error:
                        label = ('WAITING_FOR_NATIVE_STORAGE' if isinstance(error,chain.WaitingForStorage) else
                            'WAITING_FOR_ASSIGNED_GPU' if isinstance(error,chain.WaitingForGPU) else 'WAITING_ALLOCATION')
                        self.update(status=label,stage=None,child_pid=None,waiting_reason=str(error))
                    time.sleep(5)
                proof = self.execute(stage)
            self.update(status='COMPLETE',stage=None,child_pid=None,completed_at=time.time(),completion_proof=proof,
                full_native_training_completed=True,full_native_evaluation_completed=True,
                completed_GNN_epochs=self.request['target_epoch'],evaluation_checkpoint='checkpoint_best.pth')
        except Exception as error:
            self.update(status='FAILED',error=type(error).__name__+': '+str(error),failed_at=time.time())
            raise


def run(request_path):
    sys.dont_write_bytecode = True
    return Controller(request_path).run()
