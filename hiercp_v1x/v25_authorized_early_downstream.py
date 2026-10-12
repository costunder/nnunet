"""Explicitly authorized B epoch-boundary BEST -> full CP/native pipeline.

Only the full40/680 GNN completion admission and its result description differ.
Owned checkpoint identity, whole-patient/U history, full128 BEST selection,
model tensors, v2.5 context, CP642x128, native250 and native BEST26 stay exact.
Original source modules/globals/checkpoints are never modified. Child processes
enter this adapter so the separate admission also applies in native workers.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import types

FORMAT = 'v25_user_authorized_B_epoch_boundary_early_downstream_v1'
AUTHORIZATION_FORMAT = 'v25_user_authorized_B_epoch_boundary_stop_v1'
PIN_FORMAT = 'v25_user_authorized_B_partial_own_full128_BEST_v1'
ENV_AUTHORIZATION = 'V25_B_EARLY_AUTHORIZATION'
ENV_AUTHORIZATION_SHA = 'V25_B_EARLY_AUTHORIZATION_SHA256'
FILES = ('hiercp_v1x/v25_authorized_early_downstream.py', 'tools/run_v25_authorized_early_downstream.py',
    'tools/run_v25_B_early_downstream_execution.py')
ROOT = Path(__file__).resolve().parents[1]
_PRIVATE = {}
_VERIFIED_PINS = set()
ENV_CP_OUTPUT = 'V25_B_EARLY_CP_OUTPUT'
ENV_SCIENTIFIC_REQUEST = 'V25_B_EARLY_SCIENTIFIC_REQUEST'
ENV_SCIENTIFIC_REQUEST_SHA = ENV_SCIENTIFIC_REQUEST+'_SHA256'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*2**20), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def witness(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError('Exact regular preserved input required: '+str(path))
    path = path.resolve(strict=True)
    stat = path.stat()
    return dict(path=str(path), sha256=sha(path), stat=[stat.st_dev,stat.st_ino,stat.st_size,
        stat.st_mtime_ns,stat.st_ctime_ns,stat.st_nlink])


def verify_witness(value):
    if witness(value['path']) != value:
        raise ValueError('Preserved early-stop source bytes/metadata changed: '+value['path'])
    return value


def released(actors):
    """Reject every still-live exact predecessor; never send a process signal."""
    import psutil
    if not isinstance(actors,list) or not actors:
        raise ValueError('Explicit original source/controller actor identities required')
    for saved in actors:
        if (not isinstance(saved,dict) or type(saved.get('pid')) is not int
                or type(saved.get('create_time')) not in (int,float)
                or saved.get('UID') != 41833 or not isinstance(saved.get('command'),list)
                or not saved['command']):
            raise ValueError('Exact original PID/birth/command/UID release witness required')
        try:
            actor=psutil.Process(saved['pid'])
            if abs(actor.create_time()-saved['create_time']) > .001:
                continue
            if actor.uids().real != saved['UID'] or actor.cmdline() != saved['command']:
                raise ValueError('Original PID identity cannot be safely resolved')
            return False
        except psutil.NoSuchProcess:
            continue
    return True


def authorization(path, *, expected_sha=None, require_release=True):
    path=Path(path).resolve(strict=True);value=read(path)
    if expected_sha is not None and sha(path)!=expected_sha:
        raise ValueError('Immutable user early-stop authorization changed')
    if (value.get('format')!=AUTHORIZATION_FORMAT or value.get('arm')!='B'
            or value.get('physical_GPU')!=5 or value.get('explicit_user_authorization')is not True
            or not isinstance(value.get('user_instruction'),str) or not value['user_instruction'].strip()
            or value.get('stop_after_current_epoch')is not True
            or value.get('use_own_full128_BEST')is not True
            or value.get('original_planned_epochs')!=40 or value.get('original_planned_optimizer_updates')!=680
            or type(value.get('target_completed_epochs'))is not int or not 1<=value['target_completed_epochs']<40
            or value.get('CP_source_entries')!=642 or value.get('CP_candidates_per_source')!=128
            or value.get('native_epochs')!=250 or value.get('native_evaluation_cases')!=26
            or value.get('native_evaluation_checkpoint')!='checkpoint_best.pth'
            or value.get('context_unchanged')is not True or value.get('model_and_loss_unchanged')is not True):
        raise ValueError('Exact explicit B epoch-boundary-only/ownBEST/full642/native250/BEST26 authorization required')
    if require_release and not released(value.get('release_actors')):
        raise RuntimeError('Exact predecessor GNN/controller actor is still alive; no downstream execution')
    return value


def bind_authorization(path):
    value=authorization(path)
    os.environ[ENV_AUTHORIZATION]=str(Path(path).resolve(strict=True))
    os.environ[ENV_AUTHORIZATION_SHA]=sha(path)
    return value


def _authorization_environment():
    return authorization(os.environ[ENV_AUTHORIZATION],expected_sha=os.environ[ENV_AUTHORIZATION_SHA])


def inspect_partial_best(training_root, *, authorization_path, bound_identity=None,
                         actual_model_sha=None, actual_context_sha=None):
    """Read full original history/identity/weights without editing a checkpoint."""
    import torch
    from .v25_training import FORMAT as TRAINING_FORMAT,validate_curriculum_progress,validate_saved_best
    from .u_bridge_training import digest
    root=Path(training_root).resolve(strict=True)
    auth=authorization(authorization_path)
    owner_path=root/'training_identity.json';latest_path=root/'checkpoint_latest.pt';best_path=root/'checkpoint_best.pt'
    guards={key:witness(path) for key,path in (('identity',owner_path),('latest',latest_path),('best',best_path))}
    owner=read(owner_path);binding=owner['binding']
    latest=torch.load(latest_path,map_location='cpu',weights_only=False)
    best=torch.load(best_path,map_location='cpu',weights_only=False)
    if (digest(binding)!=owner['identity_sha256'] or binding.get('debug')is not False
            or binding.get('epochs')!=40 or binding.get('physical_patient_batch')!=4
            or binding.get('workers')<4 or len(binding.get('train_cases',[]))!=65
            or len(binding.get('val_cases',[]))!=21
            or binding['identity']['config'].get('arm')!='B'
            or (bound_identity is not None and binding['identity']!=bound_identity)):
        raise ValueError('Original own B full-model/40epoch/65train/21val/B4 identity required')
    for payload in (latest,best):
        if (payload.get('format')!=TRAINING_FORMAT or payload.get('identity_sha256')!=owner['identity_sha256']
                or payload.get('content_sha256')!=digest({k:v for k,v in payload.items() if k!='content_sha256'})):
            raise ValueError('Original own saved checkpoint content/identity differs')
    state=latest['state'];completed=len(state['history'])
    if (completed!=auth['target_completed_epochs'] or state.get('status')!='PAUSED'
            or state.get('phase')!='training' or state.get('epoch')!=completed+1
            or state.get('evaluation_position')!=0
            or state.get('evaluation_rows')!=[] or 'pending_epoch_completion' in state):
        raise ValueError('Only PAUSED after the authorized full epoch with its truthful cooperative next-epoch cursor is admitted')
    policy=binding['config']['v24_runtime']['curriculum']
    progress=validate_curriculum_progress(state,policy=policy,train_cases=binding['train_cases'],
        val_cases=binding['val_cases'],physical_patient_batch=4)
    validate_saved_best(root,state,owner['identity_sha256'])
    if state.get('best') is None:
        raise ValueError('A genuine saved own full128 BEST is required; no initialized-model fallback')
    model_sha=digest(best['model']);context_sha=binding['model_contract']['context']['contract_sha256']
    if ((actual_model_sha is not None and actual_model_sha!=model_sha)
            or (actual_context_sha is not None and actual_context_sha!=context_sha)):
        raise ValueError('Actual loaded model/context differs from original owned winning tensors')
    initial_path=root/'initial_full_validation_epoch_000.json'
    if digest(read(initial_path))!=digest(state['initial_validation']):
        raise ValueError('Actual original initial full128 report differs')
    reports={str(initial_path):witness(initial_path)}
    for curve in state['history']:
        for role in ('train_probe','stage_validation','full_validation'):
            path=root/f'{role}_epoch_{curve["epoch"]:03d}.json'
            if digest(read(path))!=digest(curve[role]):
                raise ValueError('Original published completed patient/U report differs: '+str(path))
            reports[str(path)]=witness(path)
    winning=best['state']['pending_epoch_completion']['full_validation']
    if digest(winning)!=digest(read(root/f'full_validation_epoch_{state["best"]["epoch"]:03d}.json')):
        raise ValueError('Actual BEST model full128 winning report binding differs')
    for proof in (*guards.values(),*reports.values()):verify_witness(proof)
    result=dict(format=PIN_FORMAT,arm='B',physical_GPU=5,debug=False,full_training=False,
        source_completion_kind='USER_AUTHORIZED_EPOCH_BOUNDARY_EARLY_STOP',
        original_planned_epochs=40,original_planned_optimizer_updates=680,
        completed_epochs=completed,optimizer_updates=state['updates'],optimizer_attempts=state['attempts'],
        optimizer_overflows=state['overflows'],selected_by='fixed_full128_validation_only',
        next_epoch_cursor=dict(epoch=state['epoch'],phase=state['phase'],train_position=state['train_position'],
            partial_patient_count=len(state['train_rows']),status=state['status']),
        checkpoint=str(best_path),checkpoint_sha256=guards['best']['sha256'],
        model_sha256=model_sha,context_contract_sha256=context_sha,
        request_sha256=binding['identity']['request_sha256'],identity_sha256=owner['identity_sha256'],
        best_record=state['best'],model_contract=binding['model_contract'],
        original_curriculum_policy=policy,original_U_history=[curve['curriculum_transition'] for curve in state['history']],
        original_patient_and_full128_history_verified=True,deterministic_curriculum_verification=progress,
        source_file_proofs=guards,full128_and_patient_report_proofs=reports,
        authorization=witness(authorization_path),
        adapter_source_sha256={name:sha(ROOT/name) for name in FILES},
        source_checkpoint_written=False,source_model_optimizer_RNG_rewritten=False,
        native_epochs=250,native_evaluation_checkpoint='checkpoint_best.pth',native_evaluation_cases=26)
    from .contracts import canonical_hash
    _VERIFIED_PINS.add(canonical_hash(result))
    return result


def validate_partial_provenance(provenance, *, model_sha256,context_contract_sha256):
    """Change completion admission only; every original noncompletion gate holds."""
    from . import v25_cp as original
    if (not isinstance(provenance,dict) or provenance.get('format')!=PIN_FORMAT
            or provenance.get('arm')!='B' or provenance.get('physical_GPU')!=5
            or provenance.get('full_training')is not False or provenance.get('debug')is not False
            or provenance.get('source_completion_kind')!='USER_AUTHORIZED_EPOCH_BOUNDARY_EARLY_STOP'
            or provenance.get('original_planned_epochs')!=40 or provenance.get('original_planned_optimizer_updates')!=680
            or type(provenance.get('completed_epochs'))is not int or not 1<=provenance['completed_epochs']<40
            or type(provenance.get('optimizer_updates'))is not int or provenance['optimizer_updates']<=0
            or provenance.get('selected_by')!='fixed_full128_validation_only'
            or provenance.get('original_patient_and_full128_history_verified')is not True
            or provenance.get('source_checkpoint_written')is not False
            or provenance.get('source_model_optimizer_RNG_rewritten')is not False
            or provenance.get('model_sha256')!=model_sha256
            or provenance.get('context_contract_sha256')!=context_contract_sha256
            or any(not original._checksum(provenance.get(key)) for key in
                ('model_sha256','context_contract_sha256','request_sha256','checkpoint_sha256','identity_sha256'))):
        raise ValueError('Truthful authorized partial own B/full128 model/context provenance required')
    authproof=verify_witness(provenance['authorization'])
    auth=authorization(authproof['path'],expected_sha=authproof['sha256'])
    if auth['target_completed_epochs']!=provenance['completed_epochs']:
        raise ValueError('Actual partial epoch differs from explicit authorization')
    for proof in (*provenance['source_file_proofs'].values(),*provenance['full128_and_patient_report_proofs'].values()):
        verify_witness(proof)
    if (provenance['checkpoint']!=provenance['source_file_proofs']['best']['path']
            or provenance['checkpoint_sha256']!=provenance['source_file_proofs']['best']['sha256']
            or provenance['adapter_source_sha256']!={name:sha(ROOT/name) for name in FILES}):
        raise ValueError('Original winning checkpoint/adapter SHA differs')
    key=original.canonical_hash(provenance)
    if key not in _VERIFIED_PINS:
        derived=inspect_partial_best(Path(provenance['source_file_proofs']['identity']['path']).parent,
            authorization_path=authproof['path'],actual_model_sha=model_sha256,actual_context_sha=context_contract_sha256)
        if derived!=provenance:
            raise ValueError('Partial pin fields differ from actual original history/checkpoint admission')
        _VERIFIED_PINS.add(key)
    return copy.deepcopy(provenance)


def load_authorized_own_best(args,bound,net,runtime):
    from .u_bridge_training import digest
    import torch
    _authorization_environment()
    if args.arm!='B' or args.gpu!=5:
        raise ValueError('Authorized early downstream is only own B on physical GPU5')
    pin=inspect_partial_best(args.output/'training',authorization_path=os.environ[ENV_AUTHORIZATION],bound_identity=bound)
    from .v25_training import validate_model
    if net.v24_model_contract!=pin['model_contract']:
        raise ValueError('Actual full model configuration differs from original owned B model contract')
    validate_model(net,pin['model_contract'],debug=False)
    saved=torch.load(pin['checkpoint'],map_location='cpu',weights_only=False)
    net.load_state_dict(saved['model'],strict=True);net.eval()
    validate_partial_provenance(pin,model_sha256=digest(net.state_dict()),
        context_contract_sha256=net.v24_model_contract['context']['contract_sha256'])
    document=read(os.environ[ENV_SCIENTIFIC_REQUEST])
    target=Path(document.get('pin_output',str(Path(os.environ[ENV_SCIENTIFIC_REQUEST]).with_name('own_BEST_pin.json'))))
    from .v25_cp import _publish
    if target.exists():
        if read(target)!=pin:raise ValueError('Existing immutable own BEST pin differs; preserved')
    else:_publish(target,pin)
    return Path(pin['checkpoint']),pin


def _private_clone(module, *, overrides=None, route_native_children=False, score_output=False):
    """Compile a separate namespace; retain every original arithmetic expression."""
    source=inspect.getsource(module);tree=ast.parse(source);counts={}
    routes={'v25_cp':'CP','v25_native_cp':'NATIVE','v25_downstream':'DOWNSTREAM'}
    strings={
        'from hiercp_v1x.v25_native_cp import run_training_entry; run_training_entry()':
        'from hiercp_v1x.v25_authorized_early_downstream import run_training_entry; run_training_entry()',
        'tools/run_v25_downstream.py':'tools/run_v25_authorized_early_downstream.py'}
    class Route(ast.NodeTransformer):
        def visit_BinOp(self,node):
            if score_output:
                for leaf,new in (('cp_bank',"Path(os.environ['V25_B_EARLY_CP_OUTPUT'])"),
                        ('cp_scores.json',"Path(os.environ['V25_B_EARLY_CP_OUTPUT']).parent/'cp_scores.json'")):
                    expected=ast.parse("args.output/'"+leaf+"'",mode='eval').body
                    if ast.dump(node)==ast.dump(expected):
                        counts['fresh_'+leaf]=counts.get('fresh_'+leaf,0)+1
                        return ast.copy_location(ast.parse(new,mode='eval').body,node)
            return self.generic_visit(node)
        def visit_ImportFrom(self,node):
            if node.level==1 and node.module=='v25_server_resources':
                if [(alias.name,alias.asname)for alias in node.names]!=[('AllocationLease',None)]:
                    raise ValueError('Original CP resource import changed')
                counts['B_CP_resource_lease']=counts.get('B_CP_resource_lease',0)+1
                return ast.copy_location(ast.ImportFrom(module='v25_B_early_downstream_resources',
                    level=1,names=[ast.alias(name='BEarlyDownstreamAllocationLease',asname='AllocationLease')]),node)
            if score_output and node.level==0 and node.module=='hiercp_v1x.v25_server_resources':
                if [(alias.name,alias.asname)for alias in node.names]!=[('AllocationLease',None)]:
                    raise ValueError('Original score resource import changed')
                counts['B_resource_lease']=counts.get('B_resource_lease',0)+1
                return ast.copy_location(ast.ImportFrom(module='hiercp_v1x.v25_B_early_downstream_resources',
                    level=0,names=[ast.alias(name='BEarlyDownstreamAllocationLease',asname='AllocationLease')]),node)
            if node.level==1 and node.module is None:
                result=[]
                for alias in node.names:
                    if alias.name in routes:
                        counts[alias.name]=counts.get(alias.name,0)+1
                        result.append(ast.ImportFrom(module='v25_authorized_early_downstream',level=1,
                            names=[ast.alias(name=routes[alias.name],asname=alias.asname or alias.name)]))
                    else:result.append(ast.ImportFrom(module=None,level=1,names=[alias]))
                return [ast.copy_location(item,node) for item in result]
            if node.level==0 and node.module=='hiercp_v1x.v25_cp':
                counts['absolute_v25_cp']=counts.get('absolute_v25_cp',0)+1
                node.module='hiercp_v1x.v25_authorized_early_downstream'
            return node
        def visit_Constant(self,node):
            if route_native_children and isinstance(node.value,str):
                if node.value=='v25_native_cp':
                    counts['dynamic_native_import']=counts.get('dynamic_native_import',0)+1
                    return ast.copy_location(ast.Constant('v25_authorized_early_downstream'),node)
                if node.value in strings:
                    counts[node.value]=counts.get(node.value,0)+1
                    return ast.copy_location(ast.Constant(strings[node.value]),node)
            return node
    transformed=ast.fix_missing_locations(Route().visit(tree))
    if module.__name__=='hiercp_v1x.v25_cp' and counts.get('B_CP_resource_lease')!=1:
        raise ValueError('Original CP inner resource admission must be redirected exactly once')
    if score_output and any(counts.get(key)!=1 for key in ('fresh_cp_bank','fresh_cp_scores.json','B_resource_lease','absolute_v25_cp')):
        raise ValueError('Original score admission/output expressions changed; exact review required')
    clone=types.ModuleType(module.__name__+'_private_authorized_B')
    clone.__dict__.update(__package__=module.__package__,__file__=module.__file__)
    sys.modules[clone.__name__]=clone
    exec(compile(transformed,str(module.__file__)+':authorized_B_admission','exec'),clone.__dict__)
    clone.__dict__.update(overrides or {})
    clone.authorized_B_adaptation_proof=dict(original_source_sha256=sha(module.__file__),
        private_source_AST_sha256=hashlib.sha256(ast.dump(transformed).encode()).hexdigest(),
        admission_and_child_import_counts=counts,original_module_globals_changed=False,
        original_model_paste_context_loss_optimizer_expressions_changed=False)
    return clone


def _truthful_collect(original_collector,request,completion,metrics):
    value=original_collector(request,completion,metrics)
    overlay=read(request['score_overlay']);pin=overlay['best_provenance']
    validate_partial_provenance(pin,model_sha256=overlay['model_sha256'],
        context_contract_sha256=overlay['context_contract']['contract_sha256'])
    value['kind']='ACTUAL_COMPLETED_NATIVE250_BEST26_FROM_AUTHORIZED_PARTIAL_V25_B_GNN'
    for experiment in value['experiments']:
        experiment['settings']['gnn'].update(epochs=pin['completed_epochs'],optimizer_updates=pin['optimizer_updates'],
            original_planned_epochs=40,original_planned_optimizer_updates=680,full_GNN_training=False,
            source_completion_kind=pin['source_completion_kind'],user_authorization=pin['authorization'],
            original_U_history=pin['original_U_history'])
    value['verification']['full_training']=dict(scope='native segmentation',completed_epochs=250,verified=True)
    value['verification']['full_GNN_training']=False
    value['verification']['actual_GNN_completed_epochs']=pin['completed_epochs']
    value['limitations'].append('B GNN explicitly stopped after its authorized completed epoch; original40/680 was not completed. Own actual full128 BEST used unchanged; native250/BEST26 remains full scale.')
    return value


def private_modules():
    if not _PRIVATE:
        from . import v25_cp as original_cp,v25_native_cp as original_native,v25_downstream as original_downstream
        cp=_private_clone(original_cp,overrides=dict(validate_best_provenance=validate_partial_provenance))
        _PRIVATE['CP']=cp
        native=_private_clone(original_native,route_native_children=True)
        native.FILES=tuple(dict.fromkeys((*original_native.FILES,*FILES)))
        _PRIVATE['NATIVE']=native
        downstream=_private_clone(original_downstream)
        original_collector=downstream.collect_evidence
        downstream.collect_evidence=lambda request,completion,metrics:_truthful_collect(original_collector,request,completion,metrics)
        _PRIVATE['DOWNSTREAM']=downstream
    return _PRIVATE


def __getattr__(name):
    if name.startswith('__'):raise AttributeError(name)
    modules=private_modules()
    if name in modules:return modules[name]
    if 'NATIVE'in modules and hasattr(modules['NATIVE'],name):return getattr(modules['NATIVE'],name)
    if 'CP'in modules and hasattr(modules['CP'],name):return getattr(modules['CP'],name)
    raise AttributeError(name)


def run_score(arguments, *, authorization_path, cp_output):
    """Use the original full v2.5 runtime/CP math with separate early admission."""
    bind_authorization(authorization_path)
    from tools import run_v25 as original
    parsed=original.parse(arguments)
    if parsed.mode!='score-cp' or parsed.arm!='B' or parsed.gpu!=5:
        raise ValueError('Only original full score-cp B/GPU5 arguments admitted')
    target=Path(cp_output).absolute()
    if target.is_relative_to(parsed.output.absolute()/'training') or target==parsed.output.absolute()/'cp_bank':
        raise ValueError('Early CP scores require a fresh separate declared namespace')
    os.environ[ENV_CP_OUTPUT]=str(target)
    bridge=_private_clone(original,overrides=dict(load_own_best=load_authorized_own_best),score_output=True)
    from tools.run_v25_execution import prefix_geometry_runtime
    with prefix_geometry_runtime(parsed,bridge):
        return bridge.main(arguments)


def run_downstream(*, authorization_path, **arguments):
    bind_authorization(authorization_path)
    if arguments.get('arm')!='B' or arguments.get('gpu')!=5:
        raise ValueError('Only own B/GPU5 authorized downstream admitted')
    return private_modules()['DOWNSTREAM'].run_downstream(**arguments)


def install_from_environment():
    """Every descendant re-admits the separately sealed science and resources."""
    from . import v25_B_early_downstream_resources as resources
    seal=resources.load_seal()
    proof=seal['scientific_request'];path=Path(proof['path'])
    if sha(path)!=proof['sha256']:raise ValueError('Exact scientific early request SHA differs')
    value=read(path)
    authproof=value.get('authorization_request')
    if value.get('format')==AUTHORIZATION_FORMAT:
        authpath=path
    elif isinstance(authproof,dict) and sha(authproof['path'])==authproof['sha256']:
        authpath=Path(authproof['path'])
    else:raise ValueError('Actual explicit scientific authorization required')
    bind_authorization(authpath)
    os.environ[ENV_SCIENTIFIC_REQUEST]=str(path)
    os.environ[ENV_SCIENTIFIC_REQUEST_SHA]=proof['sha256']
    os.environ[ENV_CP_OUTPUT]=str(Path(value['cp_output']).absolute())
    contract=resources.install_from_environment()
    return dict(format=FORMAT,resources=contract,authorization=witness(authpath),
        scientific_request=witness(path),original_module_globals_changed=False,
        original_model_context_CP_native_math_changed=False,GNN_full40_claimed=False)


def calibration_worker(native_path,physical_batch,output):
    install_from_environment()
    return private_modules()['NATIVE'].calibration_worker(native_path,physical_batch,output)


def run_training_entry():
    install_from_environment()
    return private_modules()['NATIVE'].run_training_entry()


def verify_stage_completion(stage,output,cp_output):
    install_from_environment()
    if stage in ('score','scoring'):
        value=private_modules()['CP'].validate_overlay(Path(cp_output)/'score_overlay.json',full_hash=True)
        if value['overlay']['complete']is not True:raise ValueError('Full actual CP completion required')
        return value
    if stage=='native':
        return private_modules()['DOWNSTREAM'].verify_completion(Path(output)/'downstream/completion.json')
    raise ValueError('Only actual scoring/native completion may be admitted')
