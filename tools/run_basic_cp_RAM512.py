"""SHA-admitted TOTAL512 execution of the unchanged full Basic CP pipeline.

Historical runner and trainers are never patched. Their scientific functions
execute in private globals; only resource dispatch, prospective Popen claims,
the CPU provider, and complete independent archive IO receive new guards.
"""
from __future__ import annotations

import ast
import copy
import contextlib
import inspect
import importlib.util
import os
from pathlib import Path
import subprocess as _subprocess
import sys
import time
from types import FunctionType

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_basic_cp_expanded as historical
from hiercp_v1x.basic_cp_512_claims import ResourceClaimUnavailable, validate_native
from hiercp_v1x import basic_cp_512_resources as resources
from hiercp_v1x.basic_cp_gpu_admission import GPUAdmissionUnavailable

RUNNER_FILE = 'tools/run_basic_cp_RAM512.py'
PROVIDER_FILE = 'hiercp_v1x/basic_cp_large_bank_RAM512.py'
PROVIDER_MODULE = 'hiercp_v1x.basic_cp_large_bank_RAM512'
_LEASES = {}
_ACTIVE_RESERVATION = None
_MIGRATIONS = {}
_PRIOR_VALIDATORS = {}
_SCOPE = dict(vars(historical), __file__=str(Path(__file__).resolve()), __name__=__name__)


def _clone(function, scope):
    result = FunctionType(function.__code__, scope, function.__name__, function.__defaults__, function.__closure__)
    result.__kwdefaults__ = copy.deepcopy(function.__kwdefaults__)
    result.__annotations__ = dict(function.__annotations__)
    result.__doc__ = function.__doc__
    return result


for _name, _value in vars(historical).items():
    if inspect.isfunction(_value) and _value.__globals__ is vars(historical):
        _SCOPE[_name] = _clone(_value, _SCOPE)


def _admit_execution(value):
    validate_native(value)
    required = {RUNNER_FILE, PROVIDER_FILE, resources.GUARD_FILE, resources.CLAIMS_FILE}
    if not required <= set(value['code_sha256']):
        raise ValueError('Explicit TOTAL512 runner/provider/guard/claim SHA pins required')
    resources.validate_request(value)


def _lease_class(value):
    _admit_execution(value)
    return resources.BasicLease


def _lease_for(value):
    _admit_execution(value)
    key = (value['root'], value['bank'], value['bank_sha256'], value['RAM512_claim_ledger'])
    if key not in _LEASES:
        _LEASES[key] = resources.BasicLease(value)
    return _LEASES[key]


def resource_guard(value, *, idle=False):
    if 'GPU_queue_deferred_transfer' in value:
        from hiercp_v1x.basic_cp_GPU_queue_dependencies import reject_native_dispatch
        reject_native_dispatch(value)
    return _lease_for(value).verify(force=idle, idle=idle)


class _ReservedSubprocess:
    """A private facade; the global subprocess module remains unmodified."""
    def __getattr__(self, name):
        return getattr(_subprocess, name)

    def Popen(self, command, **kwargs):
        reservation = _ACTIVE_RESERVATION
        if reservation is None:
            raise RuntimeError('Actual model/source Popen requires a previously published RAM512 claim')
        if list(command) != reservation.claim['command']:
            raise RuntimeError('Popen argv must equal its exact prospective allocation')
        resources.check_initial_GPU_idle(reservation.native)
        child = _subprocess.Popen(command, **kwargs)
        reservation.popen_started = True
        # Child self-binding can occur first. Both writes are exact and
        # idempotent, so there is no unreserved fast-start Torch window.
        import psutil
        from hiercp_v1x.basic_cp_ablation_resources import _actor
        reservation.bind_child(_actor(psutil.Process(child.pid)))
        return child


def _private_actor():
    tree = ast.parse(inspect.getsource(historical.actor))
    native_guard = []
    shared_import = []
    for node in tree.body[0].body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            if isinstance(node.value.func, ast.Name) and node.value.func.id == 'resource_guard':
                native_guard.append(node)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == 'hiercp_v1x.basic_cp_RAM_admission_pause':
            if [(alias.name, alias.asname) for alias in node.names] == [('SharedBankLease', None)]:
                shared_import.append(node)
    if len(native_guard) != 1 or len(shared_import) != 1:
        raise RuntimeError('Exact original native guard and CPU-only shared resource import required')
    # The source has no CUDA workload. Its prospective observer has already
    # performed the new initial GPU5-idle admission; its runtime CPU64 guard
    # validates UUID/noCUDA and permits later foreign arrival without touching it.
    guard = native_guard[0]
    parent = tree.body[0].body
    index = parent.index(guard)
    parent[index] = ast.If(test=ast.Compare(left=ast.Name(id='label',ctx=ast.Load()),
        ops=[ast.NotEq()],comparators=[ast.Constant('candidate_bank')]),body=[guard],orelse=[])
    shared_import[0].module = 'hiercp_v1x.basic_cp_512_resources'
    loops = [node for node in ast.walk(tree) if isinstance(node,ast.While)]
    if len(loops) != 1:
        raise RuntimeError('Exact original actor monitoring loop required')
    loops[0].body.insert(1,ast.Expr(value=ast.Call(
        func=ast.Name(id='_refresh_current_reservation',ctx=ast.Load()),args=[],keywords=[])))
    ast.fix_missing_locations(tree)
    exec(compile(tree, str(Path(__file__).resolve()), 'exec'), _SCOPE)
    return _SCOPE['actor']


_SCOPE.update(_lease_class=_lease_class, _lease_for=_lease_for,
    resource_guard=resource_guard, subprocess=_ReservedSubprocess())
_ACTOR = _private_actor()


def _provider_command(native, command, label):
    command = list(command)
    if label != 'candidate_bank':
        return command
    _admit_execution(native)
    if (native['gpu'] != 5 or native.get('shared_candidate_geometry_cache') is not True
            or native.get('shared_bank_workers') != 24
            or native.get('shared_bank_RSS_limit_GiB') != 384
            or command.count('-m') != 1 or command.count('--rss-limit-gib') != 1
            or command.count('--workers') != 1):
        raise ValueError('Exact shared GPU5 source/24workers/legacy384 request required')
    module_index = command.index('-m') + 1
    if command[module_index] != 'hiercp_v1x.basic_cp_large_bank_recovery':
        raise ValueError('Only the admitted complete recovery provider may receive RAM512 guard dispatch')
    rss_index, worker_index = command.index('--rss-limit-gib')+1, command.index('--workers')+1
    if command[rss_index] != '384' or command[worker_index] != '24':
        raise ValueError('Original producer arguments changed before explicit RAM512 dispatch')
    command[module_index] = PROVIDER_MODULE
    command[rss_index] = '320'
    return command


def actor(native, command, label, env):
    global _ACTIVE_RESERVATION
    if 'GPU_queue_deferred_transfer' in native:
        from hiercp_v1x.basic_cp_GPU_queue_dependencies import reject_native_dispatch
        reject_native_dispatch(native)
    _admit_execution(native)
    command = _provider_command(native, command, label)
    while True:
        reservation = None
        try:
            reservation = resources.allocation_reservation(native, label, command)
            reservation.popen_started = False
            with reservation:
                if _ACTIVE_RESERVATION is not None:
                    raise RuntimeError('One controller cannot create overlapping model/source actors')
                _ACTIVE_RESERVATION = reservation
                try:
                    return _ACTOR(native, command, label, env)
                finally:
                    _ACTIVE_RESERVATION = None
        except (ResourceClaimUnavailable, resources.RAMAdmissionUnavailable) as failure:
            if reservation is not None and reservation.popen_started:
                raise
            # The context has closed its mutex. All source/PBS/GPU/storage
            # inputs are still checked; no narrower science fallback exists.
            historical.status(native, 'waiting_RAM', 'WAITING', reason=str(failure),
                RAM512_reservation=copy.deepcopy(failure.observation))
            _lease_for(native).observe_waiting()
            time.sleep(5)
        except GPUAdmissionUnavailable as failure:
            if (label not in ('candidate_bank','historical_metrics')
                    or (reservation is not None and reservation.popen_started)):
                # Existing native child() owns exit75 cleanup and fresh DEBUG
                # output retries; preserve that scientific witness workflow.
                raise
            historical.status(native,'waiting_GPU','WAITING',reason=str(failure.observation),
                GPU_admission=copy.deepcopy(failure.observation),retry_CPU_stage=label)
            _lease_for(native).observe_waiting()
            time.sleep(5)


_CAPTURE = _SCOPE['_capture_native_tree']


def _capture_native_tree(native, tracked):
    return _CAPTURE(native, tracked)


def _refresh_current_reservation():
    if _ACTIVE_RESERVATION is not None:
        _ACTIVE_RESERVATION.refresh()


@contextlib.contextmanager
def _archive_context(native, byte_count, destination):
    lease=_lease_for(native)
    while True:
        reservation=lease.archive_reservation(byte_count,destination=destination)
        try:
            reservation.__enter__()
        except (ResourceClaimUnavailable, resources.RAMAdmissionUnavailable)as failure:
            historical.status(native,'report','WAITING',reason=str(failure),
                RAM512_archive_reservation=copy.deepcopy(failure.observation),
                archive_copy_started=False,completed_epochs=250,evaluated_cases=26)
            lease.observe_waiting()
            time.sleep(5)
        else:break
    try:
        yield reservation
    except BaseException:
        reservation.__exit__(*sys.exc_info())
        raise
    else:
        reservation.__exit__(None,None,None)


def _report_scope(function, validator=None):
    scope=dict(function.__globals__)
    for name,value in list(scope.items()):
        if inspect.isfunction(value) and value.__globals__ is function.__globals__:
            scope[name]=_clone(value,scope)
    if validator is not None:
        # validate_completed imports the guard locally. Rebinding a global
        # alone would leave that local import on the new export and reject an
        # unchanged old-path BEST certificate. Replace exactly that import.
        original=scope['validate_completed'];tree=ast.parse(inspect.getsource(original))
        imports=[n for n in ast.walk(tree)if isinstance(n,ast.ImportFrom)
            and n.module=='hiercp_v1x'and[(a.name,a.asname)for a in n.names]==[('v24_native_best_eval_runtime','guard')]]
        if len(imports)!=1:raise RuntimeError('Exact completed-result local BEST guard import required')
        body=tree.body[0].body;index=body.index(imports[0])
        body[index]=ast.Assign(targets=[ast.Name(id='guard',ctx=ast.Store())],
            value=ast.Name(id='_RAM512_completed_validator',ctx=ast.Load()))
        scope['_RAM512_completed_validator']=validator;ast.fix_missing_locations(tree)
        exec(compile(tree,str(Path(__file__).resolve()),'exec'),scope)
    return scope


def _guarded_archive_function(validator=None):
    from tools import basic_cp_expanded_result as result
    function = result.archive_completed
    # Its scientific byte/ctime/independence guards remain in the original
    # function scope; only the prospective IO reservation is inserted.
    tree = ast.parse(inspect.getsource(function))
    body = tree.body[0].body
    matches = [index for index,node in enumerate(body) if isinstance(node,ast.Assign)
        and any(isinstance(target,ast.Name) and target.id=='measured' for target in node.targets)]
    if len(matches) != 1:
        raise RuntimeError('Exact independent archive source-byte measurement required')
    split = matches[0]+1
    body[split:] = [ast.With(items=[ast.withitem(context_expr=ast.Call(
        func=ast.Name(id='_archive_context',ctx=ast.Load()),args=[ast.Name(id='native',ctx=ast.Load()),
        ast.Name(id='measured',ctx=ast.Load()),ast.Name(id='destination',ctx=ast.Load())],
        keywords=[]),optional_vars=None)],body=body[split:])]
    ast.fix_missing_locations(tree)
    scope = _report_scope(function,validator)
    scope['_archive_context']=_archive_context
    exec(compile(tree,str(Path(__file__).resolve()),'exec'),scope)
    return scope['archive_completed']


_ARCHIVE = _guarded_archive_function()


def _migration(native):
    reference=native.get('RAM512_migration_state')
    if reference is None:return {'state':'TRAINING','historical_metrics_complete':False}
    from tools import persist_basic_cp_seven as durable
    if (not isinstance(reference,dict) or set(reference)!={'path','sha256'}
            or not isinstance(reference['path'],str) or not Path(reference['path']).is_absolute()):
        raise ValueError('Exact RAM512 migration-state path/SHA required')
    path=durable.exact(reference['path'])
    if durable.digest(path)!=reference['sha256']:
        raise RuntimeError('Actual migration-stage evidence SHA changed')
    key=(native['root'],reference['sha256'])
    if key not in _MIGRATIONS:
        value=durable.read(path)
        if (value.get('format')!='basic_cp_RAM512_migration_state_v1'
                or value.get('gpu')!=native['gpu'] or value.get('arm')!=native['arm']
                or value.get('native_root')!=native['native_root'] or value.get('trainer')!=native['trainer']
                or value.get('state') not in ('TRAINING','FINAL250_PENDING_BEST26','BEST26_READY','FULL_COMPLETE')
                or type(value.get('historical_metrics_complete'))is not bool
                or not isinstance(value.get('source_files_sha256'),dict)):
            raise ValueError('Actual same-experiment full250/BEST26 migration stage required')
        for source,checksum in value['source_files_sha256'].items():
            if durable.digest(durable.exact(source))!=checksum:
                raise RuntimeError('Actual released source stage/checkpoint evidence changed')
        if value['state']!='TRAINING' and value.get('completed_epochs')!=250:
            raise ValueError('Completed training roles require actual full250')
        if value['state']in('BEST26_READY','FULL_COMPLETE') and value.get('evaluated_cases')!=26:
            raise ValueError('Completed BEST role requires all original26 cases')
        _MIGRATIONS[key]=value
    return copy.deepcopy(_MIGRATIONS[key])


def _prior_validator(native):
    descriptor=native.get('RAM512_completed_best_validator')
    if descriptor is None:return None
    from tools import persist_basic_cp_seven as durable
    if not isinstance(descriptor,dict) or set(descriptor)!={'code','export','receipt'}:
        raise ValueError('Exact original-code BEST26 validator descriptor required')
    code=durable.exact(descriptor['code'],directory=True)
    for name in ('export','receipt'):
        ref=descriptor[name]
        if not isinstance(ref,dict) or set(ref)!={'path','sha256'}:
            raise ValueError('Original BEST validator export/receipt path/SHA required')
        if durable.digest(durable.exact(ref['path']))!=ref['sha256']:
            raise RuntimeError('Original BEST validator evidence SHA changed')
    export=durable.read(descriptor['export']['path'])
    pins=export.get('code_sha256')
    if export.get('code')!=str(code) or not isinstance(pins,dict):
        raise ValueError('Original BEST validator code export differs')
    relative='hiercp_v1x/v24_native_best_eval_runtime.py'
    if pins.get(relative)!=native['code_sha256'].get(relative):
        raise ValueError('BEST validator scientific bytes must remain unchanged')
    key=(str(code),descriptor['export']['sha256'],descriptor['receipt']['sha256'])
    if key not in _PRIOR_VALIDATORS:
        for name,checksum in pins.items():
            part=Path(name)
            if part.is_absolute() or '..'in part.parts or durable.digest(durable.exact(code/part))!=checksum:
                raise RuntimeError('Original pinned BEST validator export bytes changed')
        name='hiercp_v1x._RAM512_prior_BEST_'+descriptor['export']['sha256'][:24]
        spec=importlib.util.spec_from_file_location(name,code/relative)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        goal=fold(native)
        if Path(descriptor['receipt']['path'])!=goal/module.RECEIPT:
            raise ValueError('Original BEST receipt must retain its exact actual fold/inode path')
        module.validate_best_validation_receipt(descriptor['receipt']['path'],fold=goal,
            expected_cases=native['split']['outer_val'],bank_sha256=native['bank_sha256'])
        _PRIOR_VALIDATORS[key]=module
    return _PRIOR_VALIDATORS[key]


def _completed_report_for(native):
    from tools import basic_cp_expanded_result as result
    validator=_prior_validator(native)
    if validator is None:return result.completed_report
    return _report_scope(result.completed_report,validator)['completed_report']


def _archive_for(native):
    validator=_prior_validator(native)
    return _ARCHIVE if validator is None else _guarded_archive_function(validator)


def _needs_training_stage(native):
    role=_migration(native)['state']
    if role in ('BEST26_READY','FULL_COMPLETE'):
        if _prior_validator(native)is None:
            raise ValueError('Completed BEST26 requires its original pinned validator/receipt')
        return False
    return True


def _needs_historical_metrics(native):
    return _migration(native)['historical_metrics_complete'] is not True


def _carry_complete_pipeline(request):
    path=Path(request['root'])/'native.json'
    if not path.exists():return False
    native=historical.read(path);migration=_migration(native)
    if migration['state']!='FULL_COMPLETE':return False
    _admit_execution(native);_lease_for(native).observe_waiting()
    if _prior_validator(native)is None:
        raise ValueError('Full completed pipeline requires unchanged original BEST26 validation')
    carried=migration.get('completed_result')
    if (not isinstance(carried,dict) or not isinstance(carried.get('reports'),dict)
            or not isinstance(carried.get('archive'),dict) or migration['historical_metrics_complete']is not True):
        raise ValueError('Actual full completed report/archive evidence required; no metric fallback')
    historical.status(native,'complete','COMPLETE',completed_epochs=250,evaluated_cases=26,
        full_native_training=True,full_native_evaluation=True,historical_metrics_complete=True,
        reports=carried['reports'],archive=carried['archive'],actual_completed_pipeline_reused=True,
        completed_source_root=migration['source_root'],existing_predictions_and_BEST_receipt_preserved=True)
    return True


def _wait_for_queue_transfer(request):
    if 'GPU_queue_deferred_transfer' not in request:
        return
    from hiercp_v1x.basic_cp_GPU_queue_dependencies import wait_for_transfer
    native = historical.read(Path(request['root'])/'native.json')
    migration = _migration(native)
    if migration['state'] != 'TRAINING':
        raise ValueError('Only incomplete actual training may wait for an approved GPU transfer')
    wait_for_transfer(request, _lease_for(request), historical.status,
        completed_epochs=migration.get('completed_epochs'))


def _private_run():
    tree = ast.parse(inspect.getsource(historical.run))
    waits = [n for n in ast.walk(tree) if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call)
        and isinstance(n.value.func,ast.Name) and n.value.func.id=='_wait_for_GPU']
    imports = [n for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)
        and n.module=='tools.basic_cp_expanded_result']
    if len(waits) != 1 or len(imports) != 1:
        raise RuntimeError('Exact original controller GPU wait and full completed archive import required')
    wait=waits[0]; body=tree.body[0].body; index=body.index(wait)
    body[index]=ast.If(test=ast.Compare(left=ast.Subscript(value=ast.Name(id='request',ctx=ast.Load()),
        slice=ast.Constant('candidates_per_source'),ctx=ast.Load()),ops=[ast.NotEq()],
        comparators=[ast.Constant(2048)]),body=[wait],orelse=[])
    imported=imports[0]
    if [a.name for a in imported.names] != ['completed_report','archive_completed']:
        raise RuntimeError('Original report and archive import API changed')
    body.remove(imported)
    report_index=next(i for i,n in enumerate(body)if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call)
        and isinstance(n.value.func,ast.Name) and n.value.func.id=='status'
        and len(n.value.args)>1 and isinstance(n.value.args[1],ast.Constant) and n.value.args[1].value=='report')
    for name,factory in reversed((('completed_report','_completed_report_for'),('archive_completed','_archive_for'))):
        body.insert(report_index,ast.Assign(targets=[ast.Name(id=name,ctx=ast.Store())],
            value=ast.Call(func=ast.Name(id=factory,ctx=ast.Load()),args=[ast.Name(id='native',ctx=ast.Load())],keywords=[])))
    calls=[n for n in body if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call)
        and isinstance(n.value.func,ast.Name) and n.value.func.id=='child'
        and len(n.value.args)>1 and isinstance(n.value.args[1],ast.Constant) and n.value.args[1].value=='train']
    if len(calls)!=1:raise RuntimeError('Exact original full train/validation stage dispatch required')
    index=body.index(calls[0]);training_status=body[index-1]
    body[index-1:index+1]=[ast.If(test=ast.Call(func=ast.Name(id='_needs_training_stage',ctx=ast.Load()),
        args=[ast.Name(id='native',ctx=ast.Load())],keywords=[]),body=[training_status,calls[0]],orelse=[])]
    metrics=[n for n in body if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call)
        and isinstance(n.value.func,ast.Name) and n.value.func.id=='actor']
    if len(metrics)!=1:raise RuntimeError('Exact original full historical metrics stage required')
    index=body.index(metrics[0])
    body[index]=ast.If(test=ast.Call(func=ast.Name(id='_needs_historical_metrics',ctx=ast.Load()),
        args=[ast.Name(id='native',ctx=ast.Load())],keywords=[]),body=[metrics[0]],orelse=[])
    body.insert(0,ast.If(test=ast.Call(func=ast.Name(id='_carry_complete_pipeline',ctx=ast.Load()),
        args=[ast.Name(id='request',ctx=ast.Load())],keywords=[]),body=[ast.Return(value=None)],orelse=[]))
    body.insert(1,ast.Expr(value=ast.Call(func=ast.Name(id='_wait_for_queue_transfer',ctx=ast.Load()),
        args=[ast.Name(id='request',ctx=ast.Load())],keywords=[])))
    ast.fix_missing_locations(tree)
    exec(compile(tree,str(Path(__file__).resolve()),'exec'),_SCOPE)
    return _SCOPE['run']


_SCOPE.update(actor=actor, _capture_native_tree=_capture_native_tree,
    _refresh_current_reservation=_refresh_current_reservation, _ARCHIVE=_ARCHIVE,
    _needs_training_stage=_needs_training_stage,_needs_historical_metrics=_needs_historical_metrics,
    _carry_complete_pipeline=_carry_complete_pipeline,_completed_report_for=_completed_report_for,_archive_for=_archive_for)
_SCOPE['_wait_for_queue_transfer'] = _wait_for_queue_transfer
run = _private_run()
_SCOPE.update(run=run)
main = _SCOPE['main']
_train_native = _SCOPE['_train_native']
train = _SCOPE['train']
preflight = _SCOPE['preflight']
debug_trial = _SCOPE['debug_trial']
child = _SCOPE['child']
prepare_bank = _SCOPE['prepare_bank']
fold = _SCOPE['fold']


if __name__ == '__main__':
    result = main()
    if result is not None:
        raise SystemExit(result)
