"""Fresh official STU-Net-S, all-P+128U from epoch one on assigned GPU4."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import sys
import time
from types import FunctionType, SimpleNamespace
import builtins

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))


def clone_main(original,build_hook,publisher=None):
    """Replace only the private factory import; retain the original CLI code."""
    namespace=dict(original.__globals__)
    if publisher is not None:namespace['publish']=publisher
    original_builtins=namespace.get('__builtins__',builtins)
    private=dict(original_builtins if isinstance(original_builtins,dict) else vars(original_builtins))
    importer=builtins.__import__
    def importing(name,globals=None,locals=None,fromlist=(),level=0):
        module=importer(name,globals,locals,fromlist,level)
        if name=='hiercp_v1x.v24_factory' and level==0 and tuple(fromlist)==('V24NativeInputs','build_runtime'):
            return SimpleNamespace(V24NativeInputs=module.V24NativeInputs,build_runtime=build_hook(module.build_runtime))
        return module
    private['__import__']=importing;namespace['__builtins__']=private
    result=FunctionType(original.__code__,namespace,original.__name__,original.__defaults__,original.__closure__)
    result.__kwdefaults__=copy.deepcopy(original.__kwdefaults__)
    return result


def atomic_new_json(path,value):
    """Publish a complete new record atomically without replacing an old file."""
    path=Path(path)
    temporary=path.with_name(path.name+'.GPU4_new_'+str(os.getpid())+'_'+str(time.time_ns()))
    created=False
    try:
        with temporary.open('x',encoding='utf8') as stream:
            created=True
            json.dump(value,stream,indent=2,allow_nan=False);stream.flush();os.fsync(stream.fileno())
        os.link(temporary,path)
    finally:
        if created:temporary.unlink()


def atomic_publish(path,value):
    path=Path(path)
    if path.exists():
        if json.loads(path.read_text(encoding='utf8'))!=value:
            raise FileExistsError('Existing GPU4 metadata differs; old results preserved: '+str(path))
        return
    atomic_new_json(path,value)


def select_comparison_batch(original):
    """Measure every configured candidate and retain the GPU6 B4/chunk64 contract."""
    def calibrate(*args,**kwargs):
        report=original(*args,**kwargs)
        trials=[row for row in report['trials'] if row['physical_patient_batch']==4
            and row['physical_candidate_batch']==64 and row['accepted']]
        if len(trials)!=1:
            raise MemoryError('Measured full128 GPU4 B4/chunk64 comparison does not fit; no smaller batch or model fallback')
        winner=dict(physical_patient_batch=report['selected_physical_patient_batch'],
            physical_candidate_batch=report['selected_physical_candidate_batch'])
        report.update(selected_physical_patient_batch=4,selected_physical_candidate_batch=64,
            unconstrained_throughput_winner=winner,
            selection_reason='Preserve GPU6 physical/effective B4, chunk64 and680 optimization steps; change only U admission schedule')
        return report
    return calibrate


def main(argv=None):
    from tools import run_v24_all_p as cli
    values=sys.argv[1:] if argv is None else argv
    args=cli.parse(values)
    if args.gpu!=4 or args.resume is not None:
        raise ValueError('Fresh assigned GPU4 experiment required; no GPU6 trained GNN or optimizer state loaded')
    config=cli.validate_config(json.loads(args.config.read_text(encoding='utf8')),4,args.stunet_checkpoint)
    os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES']='' if args.mode=='prepare' else '4'
    sys.dont_write_bytecode=True
    from hiercp_v1x import v24_factory,v24_memory_runtime as memory,v24_hash_runtime as hashing
    from hiercp_v1x import v24_prefetch_runtime as prefetch,v24_input_runtime as inputs
    from hiercp_v1x.v24_preparation_runtime import install_runtime
    from hiercp_v1x import v24_training as training
    from hiercp_v1x.u_bridge_training import capture_rng,digest
    install_runtime(v24_factory)
    installed=dict(memory=memory.install_memory_runtime(v24_factory),hash=hashing.install(),
        prefetch=prefetch.install_runtime(memory),inputs=inputs.install_runtime(pin_final_outputs=True))
    original_calibration=training.calibrate_training_batches
    original_writer=training._write_new
    training._write_new=atomic_new_json
    if args.mode=='calibrate':training.calibrate_training_batches=select_comparison_batch(original_calibration)
    calls=[]
    def hook(original):
        def build(*positional,**keywords):
            result=original(*positional,**keywords)
            net,scorer,population,training_config,contract,close=result
            try:
                before_model,before_rng=digest(net.state_dict()),digest(capture_rng())
                binding=dict(memory=memory.bind_memory_runtime(scorer),prefetch=prefetch.bind(scorer),inputs=inputs.bind(scorer))
                if digest(net.state_dict())!=before_model or digest(capture_rng())!=before_rng:
                    raise ValueError('GPU4 execution adapters changed the official initialized model or RNG')
                request=cli.request(args,config)
                if json.loads((args.output/'request.json').read_text(encoding='utf8'))!=request:
                    raise ValueError('GPU4 immutable execution request changed during build')
                raw=scorer.geometry.memory_guard.__self__
                for name,checksum in request['source'].items():
                    path=ROOT/name
                    if path.is_symlink() or cli.sha(path)!=checksum:
                        raise ValueError('GPU4 initialized execution source changed: '+name)
                    proof=v24_factory._stat(path)
                    raw._input_file_proofs[str(path.resolve())]=proof
                    raw._file_proofs[str(path.resolve())]=proof
                raw.guard_source()
                if (contract['encoder']!='official_pretrained_STU_Net_S' or contract.get('encoder_fine_tuned') is not True
                        or contract['parameters']!=13331792 or contract['trainable_parameters']!=13331792
                        or training_config['v24_runtime']['curriculum']['initial_u']!=128):
                    raise ValueError('Exact complete official six-stage trainable STU-Net-S/full128 contract required')
                record=dict(format='v24_GPU4_full_U_fresh_runtime_build_v1',mode=args.mode,
                    model_contract=contract,model_SHA256=before_model,model_and_RNG_preserved=True,
                    official_L0_initialization=True,trained_GPU6_GNN_weights_loaded=False,
                    physical_patient_batch=4,candidate_chunk=64,epochs=40,optimization_steps=680,
                    installed=installed,bound=binding,source_files=request['source'],
                    all_execution_sources_guarded=True,created=time.time())
                with (args.output/('GPU4_runtime_build_'+str(time.time_ns())+'.json')).open('x',encoding='utf8') as stream:
                    json.dump(record,stream,indent=2,allow_nan=False)
                calls.append(record)
                return result
            except BaseException:
                close()
                raise
        return build
    cloned=clone_main(cli.main,hook,atomic_publish)
    if cloned.__code__ is not cli.main.__code__:raise ValueError('Actual full native CLI code must be retained')
    try:
        result=cloned(values)
        if len(calls)!=(0 if args.mode=='prepare' else 1):
            raise ValueError('Unexpected number of actual model/scorer builds')
        return result
    finally:
        training.calibrate_training_batches=original_calibration
        training._write_new=original_writer


if __name__=='__main__':main()
