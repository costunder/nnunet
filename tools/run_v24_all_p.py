"""Fresh full-native GT-free v2.4 singleton CPU prepare/calibrate/train CLI."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
FILES=('hiercp_v1x/v24_training.py','hiercp_v1x/v24_targets.py','hiercp_v1x/v24_factory.py',
    'hiercp_v1x/v24_provider.py','hiercp_v1x/v24_inputs.py','hiercp_v1x/v24_model.py',
    'hiercp_v1x/v24_geometry.py','hiercp_v1x/v24_stunet.py','tools/run_v24_all_p.py',
    'hiercp_v1x/v23_training.py','hiercp_v1x/u_bridge_training.py',
    'hiercp_v1x/transition_v1_local.py','hiercp_v1x/transition_v1_empty_context.py',
    'hiercp_v1x/v23_data.py','hiercp_v22/storage.py')


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*2**20),b''): digest.update(chunk)
    return digest.hexdigest()


def parse(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('prepare','calibrate','train'),required=True)
    parser.add_argument('--config',type=Path,default=ROOT/'config/v24_gpu5_GT_blind.json')
    parser.add_argument('--native-experiment',type=Path,required=True)
    parser.add_argument('--inventory',type=Path,required=True)
    parser.add_argument('--input-cache',type=Path,required=True,help='Fresh shared GT-free input namespace for GPU5/6')
    parser.add_argument('--output',type=Path,required=True,help='Fresh per-experiment result namespace')
    parser.add_argument('--gpu',type=int,choices=(5,6),required=True)
    parser.add_argument('--stunet-checkpoint',type=Path)
    parser.add_argument('--cpu-affinity',type=lambda text:[int(value) for value in text.split(',')],
        help='Exactly four assigned logical CPU IDs; or launch under an existing four-CPU affinity')
    parser.add_argument('--resume',type=Path,help='Exact owned v24 checkpoint; no legacy trained initialization')
    return parser.parse_args(argv)


def validate_config(config,gpu,stunet_checkpoint=None):
    runtime=config['v24_runtime']
    if (config.get('version')!='v2.4' or config.get('seed')!=42 or config.get('epochs')!=40
            or config.get('margin_mm')!=10 or runtime.get('physical_GPU')!=gpu
            or runtime.get('independent_single_GPU')is not True or runtime.get('debug')is not False
            or runtime.get('hidden_subset')is not False or runtime.get('workers')!=4
            or runtime.get('all_P_from_epoch_one')is not True or runtime.get('other_P_as_negative')is not False
            or runtime.get('fixed_validation_epoch')!=29):
        raise ValueError('Explicit whole-population40epoch/seed42/10mm/singleGPU5or6 contract required')
    if (runtime['rss_gib']!=64 or runtime['resident_gib']!=32 or runtime['raw_resident_gib']!=8
            or runtime['geometry_resident_gib']!=8 or runtime['torch_threads']!=1):
        raise ValueError('Explicit shared-host64GiB/32GiB canonical/8GiB raw/8GiB geometry budgets required')
    if (gpu==5 and config['encoder']!='original_CNN_GAT') or (gpu==6 and config['encoder']!='official_pretrained_STU_Net_S'):
        raise ValueError('GPU5 original CNN/GAT; GPU6 official pretrained STU-Net-S, same other contract')
    if gpu==6 and stunet_checkpoint is None:
        raise ValueError('Official pretrained STU-Net-S checkpoint must be supplied explicitly')
    if runtime['prefetch_CPU_chunks']!=1 or not all(runtime[key]is True for key in
            ('persistent_CPU_workers','pin_CPU_batches','non_blocking_H2D')):
        raise ValueError('Explicit persistent/pinned/prefetched CPU path required')
    return config


def request(args,config):
    # Mode/output/resume do not redefine the scientific request. Same sealed
    # sources and config bind prepare/calibration and exact continuation.
    value=dict(format='v24_full_native_GT_free_execution_request_v1',config=copy.deepcopy(config),
        source={name:sha(ROOT/name) for name in FILES},config_file_sha256=sha(args.config),
        native_experiment=str(args.native_experiment.resolve()),inventory=str(args.inventory.resolve()),
        inventory_sha256=sha(args.inventory),input_cache=str(args.input_cache.resolve()),
        physical_GPU=args.gpu,STU_checkpoint_sha256=None if args.stunet_checkpoint is None else sha(args.stunet_checkpoint))
    value['request_sha256']=hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return value


def publish(path,value):
    path=Path(path)
    if path.exists():
        if json.loads(path.read_text(encoding='utf8'))!=value: raise FileExistsError('Existing actual v24 artifact differs: '+str(path))
        return
    with path.open('x',encoding='utf8') as stream: json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)


def main(argv=None):
    args=parse(argv)
    config=validate_config(json.loads(args.config.read_text(encoding='utf8')),args.gpu,args.stunet_checkpoint)
    # Select physical GPU BEFORE importing Torch or the original archive.
    os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu) if args.mode!='prepare' else ''
    if str(ROOT)not in sys.path: sys.path.insert(0,str(ROOT))
    import psutil
    import torch
    from hiercp_v1x.v24_factory import V24NativeInputs,build_runtime
    from hiercp_v1x.v24_targets import validate_policy
    runtime=config['v24_runtime']; coverage=validate_policy(runtime['curriculum'])
    process=psutil.Process()
    if args.cpu_affinity is not None:
        if len(args.cpu_affinity)!=4 or len(set(args.cpu_affinity))!=4 or not set(args.cpu_affinity)<=set(process.cpu_affinity()):
            raise ValueError('Four explicit assigned CPU IDs within current allocation required')
        process.cpu_affinity(args.cpu_affinity)
    torch.set_num_threads(runtime['torch_threads'])
    if len(process.cpu_affinity())!=4 or runtime['torch_threads']!=1:
        raise RuntimeError('Actual four-CPU affinity and one Torch thread required by shared-host fair budget')
    if os.name=='posix' and process.nice()<10:process.nice(10)
    if psutil.virtual_memory().available<runtime['rss_gib']*2**30:
        raise MemoryError('Actual available RAM below explicit64GiB process budget')
    args.output.mkdir(parents=True,exist_ok=True); args.input_cache.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(args.input_cache).free<runtime['minimum_free_disk_gib']*2**30:
        raise OSError('Fresh full-data cache filesystem below explicit free-space reserve')
    bound=request(args,config)
    publish(args.output/'request.json',bound)
    if args.mode=='prepare':
        if args.resume is not None: raise ValueError('CPU prepare never loads trained model checkpoints')
        inputs=V24NativeInputs(args.native_experiment,args.inventory,args.input_cache,runtime)
        try:
            index=inputs.prepare_local()
            initial=inputs.geometry.prepare(runtime['curriculum']['initial_u'])
            full=inputs.geometry.prepare(128)
            inputs.guard_source()
            publish(args.output/'preparation.json',dict(request_sha256=bound['request_sha256'],
                local_index=str(index),local_index_sha256=sha(index),initial_upper=initial,full_upper=full,
                population=inputs.population.manifest(),curriculum=coverage,
                recipient_GT_used_in_forward=False,actual_CPU_only=True,neural_model_created=False,
                production_optimizer_updates=0,old_canonical_cache_reused=False))
        finally: inputs.close()
        return
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise RuntimeError('Actual singleton CUDA required; no CPU fallback')
    properties=torch.cuda.get_device_properties(0); free,total=torch.cuda.mem_get_info()
    if 'A6000'not in properties.name or min(free,total)<=runtime['CUDA_gib']*2**30:
        raise ValueError('Actual A6000/free VRAM below explicit40GiB calibration budget')
    torch.cuda.set_per_process_memory_fraction(runtime['CUDA_gib']*2**30/total)
    from hiercp_v1x.historical_evaluation import ResourceBudget
    budget=ResourceBudget(runtime['CUDA_gib']*2**30,runtime['rss_gib']*2**30)
    net,scorer,population,training_config,model_contract,close=build_runtime(args,config,budget)
    try:
        from hiercp_v1x.v24_training import calibrate_training_batches,run_training
        publish(args.output/('resources_'+args.mode+'_'+str(time.time_ns())+'.json'),dict(
            request_sha256=bound['request_sha256'],physical_GPU=args.gpu,GPU=properties.name,
            total_VRAM_bytes=total,free_VRAM_bytes=free,CPU_affinity=psutil.Process().cpu_affinity(),
            available_RAM_bytes=psutil.virtual_memory().available,model_contract=model_contract,
            graph_config=training_config['graph'],population=population.manifest(),workers=runtime['workers'],
            precision_AMP=training_config['training']['amp'],seed=42,epochs=40,curriculum=coverage,
            fresh_model_optimizer=True,trained_v23_weights_loaded=False,debug=False))
        if args.mode=='calibrate':
            if args.resume is not None: raise ValueError('Fresh model calibration refuses legacy/resume weights')
            report=calibrate_training_batches(net,scorer,population,training_config,budget=budget,model_contract=model_contract)
            report.update(request_sha256=bound['request_sha256'],physical_GPU=args.gpu)
            publish(args.output/'calibration.json',report)
            print(json.dumps(dict(mode='calibrate',selected_patient_batch=report['selected_physical_patient_batch'],
                selected_candidate_batch=report['selected_physical_candidate_batch'])),flush=True)
        else:
            calibration=json.loads((args.output/'calibration.json').read_text(encoding='utf8'))
            if calibration['request_sha256']!=bound['request_sha256'] or calibration['physical_GPU']!=args.gpu:
                raise ValueError('Actual full128 calibration bound to another model/source/GPU')
            training_config['v24_runtime']['batch_calibration']=calibration
            if args.resume is not None: training_config['v24_runtime']['resume_checkpoint']=str(args.resume.resolve(strict=True))
            scorer.physical_candidate_batch=calibration['selected_physical_candidate_batch']
            receipt=run_training(net,scorer,population,training_config,output=args.output/'training',
                identity=bound,budget=budget,model_contract=model_contract,
                physical_patient_batch=calibration['selected_physical_patient_batch'],workers=runtime['workers'])
            print(json.dumps(receipt),flush=True)
    finally: close()


if __name__=='__main__': main()
