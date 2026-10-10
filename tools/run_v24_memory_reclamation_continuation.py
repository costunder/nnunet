"""Resume GPU6 exact state only after actual memory-only full128 parity.

The cold DEBUG control uses the preserved allocator. Production resumes with
the actual current candidate source and unchanged original numerical CLI.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from types import FunctionType

ROOT=Path(__file__).resolve().parents[1]
sys.dont_write_bytecode=True
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
FORMAT='v24_GPU6_memory_only_exact_state_continuation_v1'
RECEIPT='memory_reclamation_continuation.json'


def _publish(path,value):
    with Path(path).open('x',encoding='utf8') as stream:json.dump(value,stream,indent=2,allow_nan=False)


def _clone(function,**globals_override):
    namespace=dict(function.__globals__);namespace.update(globals_override)
    result=FunctionType(function.__code__,namespace,function.__name__,function.__defaults__,function.__closure__)
    result.__kwdefaults__=copy.deepcopy(function.__kwdefaults__)
    return result


def _args_contract(args):
    if args.gpu!=6 or args.mode!='train':raise ValueError('Original full GPU6 training arm required')


def runtime_identity():
    from tools.v24_memory_reclamation_gate import _guard,UNCHANGED_NAMES
    paths=(ROOT/'tools/run_v24_memory_reclamation_continuation.py',ROOT/'tools/run_v24_memory_reclamation_probe.py',
        ROOT/'tools/v24_memory_reclamation_gate.py',ROOT/'tools/run_v24_performance_probe.py',
        ROOT/'tools/run_v24_performance_continuation.py',ROOT/'hiercp_v1x/v24_training_continuation.py',
        ROOT/'hiercp_v1x/v24_memory_runtime.py',*(ROOT/'hiercp_v1x'/name for name in UNCHANGED_NAMES))
    return dict(files_sha256={str(path):_guard(path)['raw_sha256'] for path in paths},
        candidate_memory_sha256=_guard(ROOT/'hiercp_v1x/v24_memory_runtime.py')['raw_sha256'],
        original_strict_resume=True,memory_only_execution_change=True,
        physical_patient_batch=4,physical_candidate_chunk=64,epochs=40,
        model_sampling_loss_and_batch_unchanged=True,production_numerical_policy_changed=False)


def candidate_continuation_runtime_proof():
    """Inspect real candidate bytes while a DEBUG process selects the old module.

This proof names the prepared continuation's target implementation. The probe
separately records its actually selected old/new module and bound contracts.
The original strict checkpoint inspector and admission code remain identical.
"""
    from tools.v24_memory_reclamation_gate import _guard,_recheck
    from hiercp_v1x import v24_training_continuation as original
    path=ROOT/'hiercp_v1x/v24_memory_runtime.py';proof=_guard(path)
    spec=importlib.util.spec_from_file_location('hiercp_v1x._memory_continuation_candidate_source',path)
    if spec is None or spec.loader is None:raise ValueError('Real candidate continuation memory source required')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    contract=module.memory_runtime_contract();_recheck(proof)
    if contract['runtime_source_sha256']!=proof['raw_sha256']:raise ValueError('Candidate continuation source changed')
    tool=ROOT/'tools/run_v24_training_continuation.py'
    return dict(module_path=str(Path(original.__file__).resolve()),module_sha256=_guard(original.__file__)['raw_sha256'],
        wrapper_CLI_path=str(tool),wrapper_CLI_sha256=_guard(tool)['raw_sha256'],
        memory_module_path=str(path),memory_module_sha256=proof['raw_sha256'],memory_runtime_contract=contract)


def prepare(args,source_output,source_code,source_interruption_proof=None):
    from tools import run_v24_performance_continuation as shared
    from hiercp_v1x import v24_training_continuation as original
    from tools.v24_memory_reclamation_gate import memory_runtime_provenance,_sha
    _args_contract(args);provenance=memory_runtime_provenance(source_code)
    old=original.inspect_continuation(source_output,source_code)
    admission=shared._source_admission(old['latest'],source_output,source_code,args.gpu,source_interruption_proof)
    if set(old['latest']['numerical_state_sha256'])!=set(original.NUMERICAL) or old['latest']['optimizer_parameter_tensors']!=537:
        raise ValueError('Full original GPU6 six-state537-parameter checkpoint required')
    proof=_clone(original.prepare_continuation,_runtime_proof=candidate_continuation_runtime_proof)(
        args,source_output=source_output,source_code=source_code)
    value=json.loads(proof.read_text(encoding='utf8'))
    if value['source']!=old or memory_runtime_provenance(source_code)!=provenance:
        raise ValueError('Preserved exact source or memory-only implementations changed during prepare')
    document=dict(format=FORMAT,original_continuation_receipt_sha256=_sha(proof),runtime=runtime_identity(),
        source=old,source_admission=admission,destination=str(args.output.resolve()),
        memory_only_source_provenance=provenance,CPU_RAM_GPU_limits_changed=False,model_or_dataset_reduced=False,
        physical_or_effective_batch_changed=False,production_optimizer_updates=0,created_at=time.time())
    _publish(args.output/RECEIPT,document)
    return document


def admit(args,source_output,source_code,source_interruption_proof=None):
    from hiercp_v1x import v24_training_continuation as original
    from tools import run_v24_performance_continuation as shared
    from tools.v24_memory_reclamation_gate import memory_runtime_provenance,_sha
    _args_contract(args)
    path,value=_clone(original.admit_continuation,_runtime_proof=candidate_continuation_runtime_proof)(args)
    own=json.loads((args.output/RECEIPT).read_text(encoding='utf8'))
    if (own.get('format')!=FORMAT or own.get('runtime')!=runtime_identity()
            or own.get('original_continuation_receipt_sha256')!=_sha(path)
            or own.get('destination')!=str(args.output.resolve()) or own.get('source')!=value['source']
            or own['source']['source_output']!=str(Path(source_output).resolve(strict=True))
            or own['source']['source_code']!=str(Path(source_code).resolve(strict=True))
            or own.get('memory_only_source_provenance')!=memory_runtime_provenance(source_code)):
        raise ValueError('Actual original six-state continuation and candidate memory-only identity differs')
    stored=own['source_admission'].get('interruption')
    if stored is not None:
        if source_interruption_proof is not None and str(Path(source_interruption_proof).resolve(strict=True))!=stored['path']:
            raise ValueError('Explicit original interruption proof differs')
        source_interruption_proof=Path(stored['path'])
    if shared._source_admission(own['source']['latest'],source_output,source_code,args.gpu,source_interruption_proof)!=own['source_admission']:
        raise ValueError('Original paused/interrupted source admission changed')
    return path,own


def train(args,source_output,source_code,probe_baseline,probe_optimized,source_interruption_proof=None):
    from tools import run_v24_performance_continuation as shared
    from tools.v24_memory_reclamation_gate import compare_probe_files,_sha
    _args_contract(args)
    _,document=admit(args,source_output,source_code,source_interruption_proof)
    if probe_baseline is None or probe_optimized is None:raise ValueError('Both fresh actual memory-only probes required before production')
    expected=document['source']['latest']['raw_sha256']
    result=compare_probe_files(probe_baseline,probe_optimized,expected)
    baseline=json.loads(Path(probe_baseline).read_text(encoding='utf8'));optimized=json.loads(Path(probe_optimized).read_text(encoding='utf8'))
    def comparison(left,right,checkpoint):
        if left!=baseline or right!=optimized or checkpoint!=expected:
            raise ValueError('Original production call attempted a different actual memory probe pair')
        return compare_probe_files(probe_baseline,probe_optimized,checkpoint)
    current=sys.modules.get('hiercp_v1x.v24_memory_runtime')
    if current is not None and _sha(current.__file__)!=document['runtime']['candidate_memory_sha256']:
        raise ValueError('Production may only select the actual current candidate memory module')
    _publish(args.output/'memory_reclamation_actual_probe_admission.json',result)
    clone=_clone(shared.train,admit=admit,compare_probes=comparison,RECEIPT=RECEIPT)
    if clone.__code__ is not shared.train.__code__:raise ValueError('Original production execution code changed')
    return clone(args,source_output,source_code,probe_baseline,probe_optimized,source_interruption_proof)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,add_help=False)
    parser.add_argument('--action',choices=('prepare','train'),required=True)
    parser.add_argument('--source-output',type=Path,required=True);parser.add_argument('--source-code',type=Path,required=True)
    parser.add_argument('--probe-baseline',type=Path);parser.add_argument('--probe-optimized',type=Path)
    parser.add_argument('--source-interruption-proof',type=Path)
    options,remaining=parser.parse_known_args(argv)
    from tools.run_v24_all_p import parse
    args=parse(remaining);_args_contract(args)
    os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID';os.environ['CUDA_VISIBLE_DEVICES']='' if options.action=='prepare' else str(args.gpu)
    result=(prepare(args,options.source_output,options.source_code,options.source_interruption_proof)
        if options.action=='prepare' else train(args,options.source_output,options.source_code,
            options.probe_baseline,options.probe_optimized,options.source_interruption_proof))
    if options.action=='prepare':print(json.dumps(result),flush=True)
    return result


if __name__=='__main__':main()
