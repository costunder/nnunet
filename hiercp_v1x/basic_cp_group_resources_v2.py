"""Fresh group guard for an exact worker exit race and pinned prior ledgers.

The live v1 guard is never edited.  Only a final empty argv with unchanged
PID/birth/UID and an actual Linux PF_EXITING observation retains that task's
actual RSS.  A new group may retain already-running controllers from an older
immutable ledger only when both ledger SHA and the selected row are pinned.
"""
from __future__ import annotations

import ast
import copy
import inspect
import json
from pathlib import Path
import textwrap
import time

from . import basic_cp_group_resources as original
from . import basic_cp_seven_resources as resource

GUARD_FILE = 'hiercp_v1x/basic_cp_group_resources_v2.py'


def _new_constructor():
    tree=ast.parse(textwrap.dedent(inspect.getsource(original.BasicLease.__init__)))
    function=tree.body[0];super_changes=registry_changes=0
    for node in ast.walk(function):
        if isinstance(node,ast.Call)and ast.unparse(node)=='super().__init__(request)':
            node.func=ast.parse('resource.BasicLease.__init__',mode='eval').body
            node.args=[ast.Name(id='self',ctx=ast.Load()),ast.Name(id='request',ctx=ast.Load())]
            super_changes+=1
    expected="'shared_resource_registry' in admitted and admitted['shared_resource_registry'] != str(path)"
    def replace(statements):
        nonlocal registry_changes
        result=[]
        for statement in statements:
            if isinstance(statement,ast.If)and ast.unparse(statement.test)==expected:
                result.append(ast.parse('self._admit_previous_registry(gpu, admitted, path, document)').body[0])
                registry_changes+=1
                continue
            for field in ('body','orelse','finalbody'):
                block=getattr(statement,field,None)
                if isinstance(block,list):setattr(statement,field,replace(block))
            result.append(statement)
        return result
    function.body=replace(function.body)
    if super_changes!=1 or registry_changes!=1:
        raise RuntimeError('Exact immutable v1 constructor and prior-registry boundary required')
    namespace=dict(original.__dict__,GUARD_FILE=GUARD_FILE)
    exec(compile(ast.fix_missing_locations(tree),__file__,'exec'),namespace)
    return namespace['__init__']


def _new_descendant():
    prior=original.BasicLease._descendant
    tree=ast.parse(textwrap.dedent(inspect.getsource(prior)))
    function=tree.body[0];changes=0
    expected="final_metadata != metadata"
    for node in ast.walk(function):
        if not isinstance(node,ast.If)or ast.unparse(node.test)!=expected:
            continue
        # This unique block precedes terminal re-observation inside the final
        # snapshot mismatch branch; no initial/live argv rule is loosened.
        if not any(isinstance(item,ast.Raise)and 'PID/birth/UID changed during live observation'in ast.unparse(item)
                   for item in node.body):
            continue
        parent=None
        for candidate in ast.walk(function):
            if isinstance(candidate,ast.If)and node in candidate.body:
                parent=candidate;break
        if parent is None:raise RuntimeError('Exact final descendant snapshot branch missing')
        position=parent.body.index(node)+1
        addition=ast.parse('if final_metadata == metadata and not final_command and self._final_exit_snapshot(process, metadata):\n    return process').body[0]
        parent.body.insert(position,addition);changes+=1
    if changes!=1:
        raise RuntimeError('Exactly one final descendant PID/argv mismatch boundary required')
    namespace=dict(prior.__globals__)
    exec(compile(ast.fix_missing_locations(tree),__file__,'exec'),namespace)
    return namespace['_descendant']


_constructor=_new_constructor()
_descendant=_new_descendant()


class BasicLease(original.BasicLease):
    """Unchanged full-size group contract with two narrowly pinned repairs."""

    def __init__(self,request):
        if original.GUARD_FILE not in request.get('code_sha256',{}):
            raise ValueError('The immutable original group guard must remain SHA-admitted')
        _constructor(self,request)

    _descendant=_descendant

    def _final_exit_snapshot(self,process,metadata):
        # At most200ms waiting plus the original <=30ms terminal re-observation.
        # A still-live empty argv without an actual exit flag remains a failure.
        for attempt in range(3):
            actual=resource._metadata(process)
            if actual!=metadata:
                raise RuntimeError('Final empty-argv task changed exact PID/birth/UID before exit observation')
            if process.status()in(resource.psutil.STATUS_ZOMBIE,resource.psutil.STATUS_DEAD):
                return False
            exiting=resource._kernel_exiting(process)
            if resource._metadata(process)!=metadata:
                raise RuntimeError('Final kernel-exit task changed exact PID/birth/UID')
            if exiting:
                self._terminal_observations[process.pid]=dict(metadata,status='PF_EXITING',
                    reason='final snapshot empty argv with unchanged PID/birth/UID and actual kernel exit',
                    observed_at=time.time(),RSS_still_accounted=True,argv_invented=False)
                return True
            if attempt<2:time.sleep(.1)
        return False

    def _admit_previous_registry(self,gpu,admitted,current_path,document):
        if not hasattr(self,'_previous_registry_manifest'):
            manifest=document.get('allowed_previous_resource_registries',{})
            if (not isinstance(manifest,dict)or not set(manifest)<={str(i)for i in range(7)}
                    or str(self.request['gpu'])in manifest):
                raise ValueError('Only explicitly retained other-controller prior registry descriptors are allowed')
            self._previous_registry_manifest=copy.deepcopy(manifest)
            self._previous_registry_cache={}
            for key,descriptor in manifest.items():
                if not isinstance(descriptor,dict)or set(descriptor)not in({'path','sha256'},{'path','sha'}):
                    raise ValueError('Prior registry requires an exact path and one unambiguous SHA')
                path=descriptor['path'];digest=descriptor.get('sha256',descriptor.get('sha'))
                if not original._absolute(path)or not resource._checksum(digest)or path==str(current_path):
                    raise ValueError('Distinct absolute immutable prior registry path and SHA required')
                self._admit(path,digest)
                prior=json.loads(Path(path).read_bytes())
                if (prior.get('format')!=original.FORMAT or not isinstance(prior.get('registration'),dict)
                        or set(prior['registration'])!={str(i)for i in range(7)}
                        or prior['registration'][key]!=self.registration[key]):
                    raise ValueError('Pinned prior registry selected controller/request row differs from the fresh registry')
                self._previous_registry_cache[key]=prior
            self._previous_registry_manifest_bytes=json.dumps(manifest,sort_keys=True,separators=(',',':'))
        path=admitted.get('shared_resource_registry')
        if gpu==self.request['gpu']:
            if path!=str(current_path):
                raise ValueError('Fresh own controller must bind its current immutable resource registry')
            return
        if path==str(current_path):
            if str(gpu)in self._previous_registry_manifest:
                raise ValueError('Unused prior-registry permission must not replace the current binding')
            return
        if path is None and gpu==5 and str(gpu)not in self._previous_registry_manifest:
            # This is the separately admitted existing CPU-bank controller.
            # Its exact request, lifetime, source and PID are still v1-pinned.
            return
        descriptor=self._previous_registry_manifest.get(str(gpu))
        if descriptor is None or path!=descriptor['path']:
            raise ValueError('Different live-controller registry path needs its explicitly pinned prior ledger')

    def _inputs(self):
        super()._inputs()
        if hasattr(self,'_previous_registry_manifest_bytes')and json.dumps(
                self._previous_registry_manifest,sort_keys=True,separators=(',',':'))!=self._previous_registry_manifest_bytes:
            raise ValueError('Pinned previous-registry descriptor permissions changed in memory')


SharedGroupLease=BasicLease
FORMAT=original.FORMAT
__all__=['BasicLease','SharedGroupLease','FORMAT','GUARD_FILE']
