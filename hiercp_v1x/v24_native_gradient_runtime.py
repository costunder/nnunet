"""Runtime proof for the actual native decoder's reversed auxiliary outputs.

Only the original helper's boolean head-order expression is replaced privately.
Native loss, parameter identity, missing-gradient and AMP-retry equations stay
unchanged, as does the complete original clone-step code object.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import inspect
from pathlib import Path
import textwrap
import threading
from types import FunctionType

ROOT=Path(__file__).resolve().parents[1]
FORMAT='v24_exact_native_decoder_gradient_runtime_v1'
PIPELINE_FILE='hiercp_v1x/v24_nnunet_cp.py'
HELPER='_native_inactive_auxiliary_gradient_proof'
STEP='_native_clone_step_with_amp_retry'


def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def _same(a,b):return ast.dump(a)==ast.dump(b)
def _statement(text):return ast.parse(text).body[0]


def _source_function(name):
    tree=ast.parse((ROOT/PIPELINE_FILE).read_text(encoding='utf8'))
    functions=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name==name]
    if len(functions)!=1:raise ValueError('Exact original native function missing: '+name)
    return ast.Module(body=functions,type_ignores=[])


def _replace_expression(tree):
    tree=copy.deepcopy(tree)
    assignments=[node for node in ast.walk(tree) if isinstance(node,ast.Assign)
        and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='reversed_outputs']
    if (len(assignments)!=1 or not isinstance(assignments[0].value,ast.Call)
            or not isinstance(assignments[0].value.func,ast.Name) or assignments[0].value.func.id!='any'):
        raise ValueError('Exactly one original reversed_outputs proof expression required')
    original_value=copy.deepcopy(assignments[0].value)
    assignments[0].value=ast.parse('_v24_decoder_head_order(trainer, decoder, source, tree)',mode='eval').body
    return ast.fix_missing_locations(tree),original_value


def gradient_runtime_contract():
    """Pure source binding; actual decoder source is admitted by the live helper."""
    original=_source_function(HELPER);adapted,_=_replace_expression(original)
    return dict(format=FORMAT,source_files_sha256={PIPELINE_FILE:_sha(ROOT/PIPELINE_FILE),
        'hiercp_v1x/v24_native_gradient_runtime.py':_sha(__file__)},
        original_helper_AST_sha256=hashlib.sha256(ast.dump(original).encode()).hexdigest(),
        adapted_helper_AST_sha256=hashlib.sha256(ast.dump(adapted).encode()).hexdigest(),
        replacement='reversed_outputs expression only',original_step_code_preserved=True,
        loss_weights_parameter_identity_and_AMP_equations_unchanged=True,
        decoder_source_admitted_at_actual_step=True,existing_eight_CP_files_changed=False)


def prove_decoder_head_order(decoder,source):
    """Prove complete ascending-head→one-reverse→returned-list dataflow."""
    tree=ast.parse(textwrap.dedent(source))
    functions=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='forward']
    if len(functions)!=1:raise ValueError('Original native seg_outputs[::-1] head ordering is unproved')
    fn=functions[0]
    body=[node for node in fn.body if not(isinstance(node,ast.Expr) and isinstance(node.value,ast.Constant) and isinstance(node.value.value,str))]
    direct=_statement('return seg_outputs[::-1]')
    # Preserve the established direct-return syntax, proving its ordered head
    # comprehension as well. This is the original CPU AMP DEBUG fixture path.
    comprehension=_statement('seg_outputs = [head(x) for head in self.seg_layers]')
    if len(body)==2 and _same(body[0],comprehension) and _same(body[1],direct):
        return dict(output_dataflow='ordered head comprehension -> direct reverse return',
            deep_supervision_branch_required=False,reversal_count=1,complete_ascending_head_order=True,
            head_count=len(decoder.seg_layers),decoder_forward_source_sha256=hashlib.sha256(textwrap.dedent(source).encode()).hexdigest())
    # Full native branch: stage s produces output using precisely head s,
    # every stage executes, and deep supervision returns the entire list.
    if (getattr(decoder,'deep_supervision',None) is not True
            or not getattr(decoder,'seg_layers',None) or not getattr(decoder,'stages',None)
            or len(decoder.seg_layers)!=len(decoder.stages)):
        raise ValueError('Original native seg_outputs[::-1] head ordering is unproved: active complete deep supervision required')
    init=_statement('seg_outputs = []')
    initial=[node for node in body if _same(node,init)]
    loops=[node for node in body if isinstance(node,ast.For) and _same(node.target,ast.Name(id='s',ctx=ast.Store()))
        and _same(node.iter,ast.parse('range(len(self.stages))',mode='eval').body)]
    head_if=_statement('''if self.deep_supervision:
    seg_outputs.append(self.seg_layers[s](x))
elif s == (len(self.stages) - 1):
    seg_outputs.append(self.seg_layers[-1](x))''')
    if len(initial)!=1 or len(loops)!=1 or loops[0].orelse:
        raise ValueError('Original native seg_outputs[::-1] head ordering is unproved: unique complete stage loop required')
    loop=loops[0];branches=[node for node in loop.body if _same(node,head_if)]
    if len(branches)!=1 or body.index(initial[0])>=body.index(loop):
        raise ValueError('Original native seg_outputs[::-1] head ordering is unproved: ascending stage/head branch differs')
    if any(isinstance(node,(ast.Break,ast.Continue,ast.Return,ast.Raise,ast.Yield,ast.YieldFrom)) for node in ast.walk(loop)):
        raise ValueError('Original native seg_outputs[::-1] head ordering is unproved: interrupted stage coverage')
    # s cannot be reassigned; supervision and heads cannot be rewritten.
    if (sum(isinstance(node,ast.Name) and node.id=='s' and isinstance(node.ctx,ast.Store) for node in ast.walk(fn))!=1
            or any(isinstance(node,ast.Attribute) and node.attr in ('seg_layers','deep_supervision')
                and isinstance(node.ctx,(ast.Store,ast.Del)) for node in ast.walk(fn))):
        raise ValueError('Original native seg_outputs[::-1] head ordering is unproved: head index/state changes')
    reverse=_statement('seg_outputs = seg_outputs[::-1]')
    branch=_statement('''if not self.deep_supervision:
    r = seg_outputs[0]
else:
    r = seg_outputs''')
    returned=_statement('return r')
    if len(body)>=3 and _same(body[-3],reverse) and _same(body[-2],branch) and _same(body[-1],returned):
        tail=body[-3:];style='ascending native heads -> one reverse assignment -> deep-supervision list alias return'
    elif _same(body[-1],direct):
        tail=[body[-1]];style='ascending native heads -> direct reverse return'
    else:raise ValueError('Original native seg_outputs[::-1] head ordering is unproved: complete returned-list dataflow differs')
    if body.index(loop)>=len(body)-len(tail):
        raise ValueError('Original native seg_outputs[::-1] head ordering is unproved: reverse must follow all stages')
    approved=[initial[0],branches[0],*tail]
    allowed={id(node) for parent in approved for node in ast.walk(parent)}
    if any(isinstance(node,ast.Name) and node.id in ('seg_outputs','r') and id(node) not in allowed for node in ast.walk(fn)):
        raise ValueError('Original native seg_outputs[::-1] head ordering is unproved: output alias/reassignment/mutation')
    head_nodes={id(node) for node in ast.walk(branches[0])}
    if any(isinstance(node,ast.Attribute) and node.attr=='seg_layers' and id(node) not in head_nodes for node in ast.walk(fn)):
        raise ValueError('Original native seg_outputs[::-1] head ordering is unproved: head alias/mutation outside the ordered branch')
    if sum(isinstance(node,ast.Return) for node in ast.walk(fn))!=1:
        raise ValueError('Original native seg_outputs[::-1] head ordering is unproved: ambiguous return')
    return dict(output_dataflow=style,deep_supervision_branch_required=True,deep_supervision_active=True,
        reversal_count=1,complete_ascending_head_order=True,head_count=len(decoder.seg_layers),
        decoder_forward_source_sha256=hashlib.sha256(textwrap.dedent(source).encode()).hexdigest())


def clone_step(original):
    """Private original step with only its head-order proof helper replaced."""
    source=ast.parse(textwrap.dedent(inspect.getsource(original)))
    if original.__name__!=STEP or not _same(source,_source_function(STEP)):
        raise ValueError('Exact complete original native clone-step source required')
    helper=original.__globals__[HELPER]
    tree=ast.parse(textwrap.dedent(inspect.getsource(helper)))
    if not _same(tree,_source_function(HELPER)):raise ValueError('Exact original native inactive-gradient helper required')
    changed,_=_replace_expression(tree);contract=gradient_runtime_contract();state=threading.local()
    files={ROOT/name:(ROOT/name).stat() for name in contract['source_files_sha256']}
    def guard():
        for path,before in files.items():
            now=path.stat()
            if (now.st_dev,now.st_ino,now.st_size,now.st_mtime_ns,now.st_ctime_ns)!=(before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns):
                raise ValueError('Admitted native gradient runtime source changed')
    def head_order(trainer,decoder,source,tree):
        guard()
        proof=prove_decoder_head_order(decoder,source)
        filename=Path(inspect.getsourcefile(type(decoder).forward)).resolve(strict=True)
        proof.update(decoder_source_file=str(filename),decoder_source_file_sha256=_sha(filename),
            source_contract=copy.deepcopy(contract))
        if proof['deep_supervision_branch_required'] and getattr(trainer,'enable_deep_supervision',None) is not True:
            raise ValueError('Native trainer and decoder deep-supervision flags must both be active')
        state.proof=proof
        return True
    namespace=dict(helper.__globals__,_v24_decoder_head_order=head_order)
    exec(compile(changed,inspect.getsourcefile(helper)+':v24_exact_gradient_proof','exec'),namespace)
    adapted=namespace[HELPER]
    def admitted_helper(trainer,parameters):
        guard();state.proof=None
        proof=adapted(trainer,parameters)
        if state.proof is not None:proof['gradient_runtime']=copy.deepcopy(state.proof)
        return proof
    result=FunctionType(original.__code__,dict(original.__globals__,**{HELPER:admitted_helper}),
        original.__name__,original.__defaults__,original.__closure__)
    result.__kwdefaults__=copy.deepcopy(original.__kwdefaults__)
    return result
