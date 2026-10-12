"""Full v2.5 STU-Net-S training and own-BEST Basic-matched CP entry point."""
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

ROOT = Path(__file__).resolve().parents[1]
FILES = ('hiercp_v1x/v25_context.py', 'hiercp_v1x/v25_targets.py',
    'hiercp_v1x/v25_factory.py', 'hiercp_v1x/v25_preparation.py',
    'hiercp_v1x/v25_training.py', 'hiercp_v1x/v25_server_resources.py', 'tools/run_v25.py')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*2**20), b''):
            digest.update(block)
    return digest.hexdigest()


def parse(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('prepare', 'calibrate', 'train', 'gnn-pipeline', 'score-cp'), required=True)
    parser.add_argument('--arm', choices=('A','B','C','D'), required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--config', type=Path, default=ROOT/'config/v25.json')
    parser.add_argument('--native-experiment', type=Path, required=True)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--stunet-checkpoint', type=Path, required=True)
    parser.add_argument('--input-cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--cpu-affinity', type=lambda value: [int(item) for item in value.split(',')])
    parser.add_argument('--source-bank', type=Path)
    parser.add_argument('--baseline-preprocessed', type=Path)
    parser.add_argument('--storage-admission', type=Path)
    parser.add_argument('--allocation-request', type=Path)
    args = parser.parse_args(argv)
    if args.gpu < 0:
        parser.error('Physical GPU number must be nonnegative')
    if args.mode == 'score-cp' and (args.source_bank is None or args.baseline_preprocessed is None):
        parser.error('Full historical Basic bank and baseline preprocessed data are required')
    return args


def configuration(path, arm):
    config = json.loads(Path(path).read_text(encoding='utf8'))
    if (config.get('version') != 'v2.5' or config.get('encoder') != 'official_pretrained_STU_Net_S'
            or config.get('seed') != 42 or config.get('epochs') != 40 or config.get('margin_mm') != 10
            or 'vessels' in config):
        raise ValueError('Confirmed v2.5 fresh official STU/seed42/full40/10mm contract required')
    runtime = config['runtime']
    for key, expected in dict(workers=4, torch_threads=1, CUDA_gib=40, rss_gib=64,
            resident_gib=32, raw_resident_gib=8, geometry_resident_gib=8,
            calibration_repeats=3, fixed_validation_epoch=29, physical_patient_batch=4,
            all_P_from_epoch_one=True, other_P_as_negative=False,
            hidden_subset=False, debug=False).items():
        if runtime.get(key) != expected:
            raise ValueError('Frozen production contract differs: '+key)
    if (runtime['physical_patient_batch_candidates'] != [4,8]
            or runtime['physical_candidate_batch_candidates'] != [32,64,128]
            or runtime.get('independent_single_GPU') is not True
            or not all(runtime[key] is True for key in ('persistent_CPU_workers','pin_CPU_batches','non_blocking_H2D'))):
        raise ValueError('Measured B4/B8/chunk32/64/128 and genuine parallel input required')
    if config.get('context') != dict(liver_depth_exclusive_mm=1.0,tumor_distance_exclusive_mm=0.0,
            tumor_distance_inclusive_mm=10.0,same_train_validation_CP=True):
        raise ValueError('Identical source/target train/validation/CP context boundaries required')
    from hiercp_v1x.v25_targets import validate_policy
    if set(config['arms']) != {'A','B','C','D'}:
        raise ValueError('All four authorized curricula required')
    for key, initial in (('A',8),('B',16),('C',32),('D',128)):
        policy = config['arms'][key]
        validate_policy(policy)
        if policy['initial_u'] != initial or policy['increment_u'] != (None if key == 'D' else initial):
            raise ValueError('Arm identity and exact +n policy differ')
    selected = copy.deepcopy(config)
    selected['arm'] = arm
    selected['runtime']['curriculum'] = copy.deepcopy(config['arms'][arm])
    return selected


def publish(path, value):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text(encoding='utf8')) != value:
            raise FileExistsError('Existing owned evidence differs; preserved: '+str(path))
        return
    with path.open('x', encoding='utf8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def request(args, config):
    from tools.run_v24_all_p import FILES as BASE_FILES
    files = tuple(dict.fromkeys(BASE_FILES+FILES))
    value = dict(format='v25_complete_execution_request_v1', config=config,
        source_sha256={name:sha(ROOT/name) for name in files},
        native_experiment=str(args.native_experiment.resolve()), inventory=str(args.inventory.resolve()),
        inventory_sha256=sha(args.inventory), input_cache=str(args.input_cache.resolve()),
        STU_checkpoint_sha256=sha(args.stunet_checkpoint))
    value['request_sha256'] = hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return value


def validate_output_paths(args):
    output, cache = args.output.resolve(), args.input_cache.resolve()
    def overlaps(first, second):
        return first.is_relative_to(second) or second.is_relative_to(first)
    if overlaps(output, cache):
        raise ValueError('Arm result and shared input cache must be disjoint')
    inventory = json.loads(args.inventory.read_text(encoding='utf8'))
    protected = {args.native_experiment.resolve(), args.stunet_checkpoint.resolve(), args.inventory.resolve()}
    for row in inventory.get('raw_records', []):
        protected.update(Path(row[key]).resolve().parent for key in ('image','label'))
    for optional in (args.source_bank, args.baseline_preprocessed):
        if optional is not None:
            protected.add(optional.resolve())
    for own in (output, cache):
        if any(overlaps(own, source) for source in protected):
            raise ValueError('Fresh namespace overlaps preserved inputs: '+str(own))


def load_own_best(args, bound, net, runtime):
    import torch
    from hiercp_v1x.v25_training import FORMAT, validate_saved_best, validate_curriculum_progress
    from hiercp_v1x.u_bridge_training import digest
    root = args.output/'training'
    best = root/'checkpoint_best.pt'
    payload = torch.load(best,map_location='cpu',weights_only=False)
    ownership = json.loads((root/'training_identity.json').read_text(encoding='utf8'))
    finished = torch.load(root/'checkpoint_latest.pt',map_location='cpu',weights_only=False)
    if (ownership['binding']['identity'] != bound or digest(ownership['binding']) != ownership['identity_sha256']
            or any(value.get('format') != FORMAT or value.get('identity_sha256') != ownership['identity_sha256']
                or value.get('content_sha256') != digest({k:v for k,v in value.items() if k!='content_sha256'})
                for value in (payload, finished))
            or finished['state']['phase'] != 'complete' or len(finished['state']['history']) != 40
            or finished['state']['updates'] != 680):
        raise ValueError('Own complete full40/680 v2.5 BEST required before CP scoring')
    validate_curriculum_progress(finished['state'],policy=runtime['curriculum'],
        train_cases=ownership['binding']['train_cases'],val_cases=ownership['binding']['val_cases'],
        physical_patient_batch=ownership['binding']['physical_patient_batch'])
    validate_saved_best(root,finished['state'],ownership['identity_sha256'])
    net.load_state_dict(payload['model'],strict=True)
    net.eval()
    return best, dict(debug=False,full_training=True,completed_epochs=40,optimizer_updates=680,
        selected_by='fixed_full128_validation_only',model_sha256=digest(net.state_dict()),
        context_contract_sha256=net.v24_model_contract['context']['contract_sha256'],
        checkpoint=str(best),checkpoint_sha256=sha(best),arm=args.arm,request_sha256=bound['request_sha256'])


def main(argv=None):
    args = parse(argv)
    sys.path.insert(0,str(ROOT))
    config = configuration(args.config,args.arm)
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '' if args.mode == 'prepare' else str(args.gpu)
    import psutil
    import torch
    from hiercp_v1x import v25_factory
    from hiercp_v1x.v25_preparation import install_runtime
    from hiercp_v1x.v25_targets import validate_policy
    from hiercp_v1x.u_bridge_experiment import lock
    runtime = config['runtime']
    policy = validate_policy(runtime['curriculum'])
    if args.cpu_affinity is not None:
        if len(args.cpu_affinity) < runtime['workers'] or not set(args.cpu_affinity) <= set(psutil.Process().cpu_affinity()):
            raise ValueError('Actual allowed production CPU affinity is required')
        psutil.Process().cpu_affinity(args.cpu_affinity)
    torch.set_num_threads(runtime['torch_threads'])
    if len(psutil.Process().cpu_affinity()) < runtime['workers']:
        raise RuntimeError('Allocated CPUs below four-worker full production contract')
    if psutil.virtual_memory().available < runtime['rss_gib']*2**30:
        raise MemoryError('Available RAM below full64GiB RSS contract')
    ancestor = args.input_cache.absolute()
    while not ancestor.exists():
        ancestor = ancestor.parent
    if shutil.disk_usage(ancestor).free < runtime['minimum_free_disk_gib']*2**30:
        raise OSError('Cache filesystem lacks reserve; prior data preserved')
    validate_output_paths(args)
    args.output.mkdir(parents=True,exist_ok=True)
    args.input_cache.mkdir(parents=True,exist_ok=True)
    bound = request(args,config)
    cpu_runtime = install_runtime(v25_factory)
    if args.allocation_request is None:
        raise ValueError('Verified current PBS allocation is required before server execution')
    from hiercp_v1x.v25_server_resources import AllocationLease
    lease = AllocationLease(args.allocation_request)
    if lease.request['gpu'] != args.gpu:
        raise ValueError('Selected physical GPU differs from its allocation proof')
    lease.verify(force=True)
    lease.install_preparation_guard(v25_factory)
    prepared = None
    with lock(args.output/'.v25.lock'):
        publish(args.output/'request.json',bound)
        if args.mode in ('prepare','gnn-pipeline'):
            with lock(args.input_cache/'.prepare.lock'):
                inputs = v25_factory.V25NativeInputs(args.native_experiment,args.inventory,args.input_cache,runtime)
                try:
                    index = inputs.prepare_local()
                    full = inputs.geometry.prepare(128)
                    initial = inputs.geometry.prepare(runtime['curriculum']['initial_u'])
                    publish(args.output/'preparation.json',dict(request_sha256=bound['request_sha256'],
                        local_index_sha256=sha(index),full_upper=full,initial_upper=initial,
                        context=inputs.context_contract,population=inputs.population.manifest(),curriculum=policy,
                        CPU_runtime=cpu_runtime,production_optimizer_updates=0,neural_model_created=False,
                        quality_verified=False))
                finally:
                    inputs.close()
                prepared = inputs
            if args.mode == 'prepare':
                print('v2.5 complete immutable preparation finished; production updates=0.',flush=True)
                return
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError('One selected genuine CUDA device required; no CPU fallback')
        free, total = torch.cuda.mem_get_info()
        properties = torch.cuda.get_device_properties(0)
        if min(free,total) <= runtime['CUDA_gib']*2**30:
            raise MemoryError('Selected GPU below40GiB free production budget; no smaller model')
        torch.cuda.set_per_process_memory_fraction(runtime['CUDA_gib']*2**30/total)
        from hiercp_v1x.historical_evaluation import ResourceBudget
        from hiercp_v1x.v25_training import calibrate_training_batches, run_training
        budget = ResourceBudget(runtime['CUDA_gib']*2**30,runtime['rss_gib']*2**30)
        net, scorer, population, training_config, contract, close = v25_factory.build_runtime(args,config,budget,inputs=prepared)
        restore_lease_guard = lease.install_scorer_guard(scorer)
        try:
            hardware = dict(name=properties.name,total_VRAM_bytes=total)
            publish(args.output/('resources_'+str(time.time_ns())+'.json'),dict(physical_GPU=args.gpu,
                hardware=hardware,free_VRAM_bytes=free,CPU_affinity=psutil.Process().cpu_affinity(),
                available_RAM_bytes=psutil.virtual_memory().available,workers=runtime['workers'],model=contract,
                graph_config=training_config['graph'],population=population.manifest(),
                physical_patient_candidates=[4,8],candidate_chunks=[32,64,128],gradient_accumulation=1,
                physical_patient_batch=4,effective_batch_size=4,precision='AMP',epochs=40,seed=42,
                debug=False,quality_verified=False))
            calibration_path = args.output/'calibration.json'
            if args.mode == 'calibrate' or (args.mode == 'gnn-pipeline' and not calibration_path.exists()):
                report = calibrate_training_batches(net,scorer,population,training_config,budget=budget,model_contract=contract)
                report.update(request_sha256=bound['request_sha256'],hardware=hardware)
                publish(calibration_path,report)
                if args.mode == 'calibrate':
                    print('v2.5 full128 CUDA calibration complete; production updates=0.',flush=True)
                    return
            calibration = json.loads(calibration_path.read_text(encoding='utf8'))
            if calibration.get('request_sha256') != bound['request_sha256'] or calibration.get('hardware') != hardware:
                raise ValueError('Exact own source/model/hardware calibration required')
            training_config['v24_runtime']['batch_calibration'] = calibration
            scorer.physical_candidate_batch = calibration['selected_physical_candidate_batch']
            if args.mode == 'score-cp':
                best, provenance = load_own_best(args,bound,net,runtime)
                from hiercp_v1x.v25_cp import score_v25_cp_bank
                result = score_v25_cp_bank(net=net,inputs=scorer.inputs,population=population,
                    source_bank=args.source_bank,baseline_preprocessed=args.baseline_preprocessed,
                    physical_candidate_batch=scorer.physical_candidate_batch,budget=budget,
                    amp=bool(training_config['training']['amp']),best_provenance=provenance,
                    output=args.output/'cp_bank')
                publish(args.output/'cp_scores.json',result)
                return
            if args.storage_admission is None:
                raise ValueError('Full four-arm persistent checkpoint storage admission is required before training')
            from hiercp_v1x import v25_training
            from hiercp_v1x.v25_native_storage import install_gnn_checkpoint_adapter
            storage_receipt = install_gnn_checkpoint_adapter(v25_training,args.storage_admission,arm=args.arm)
            publish(args.output/'gnn_storage_runtime.json',storage_receipt)
            if args.resume is not None and not args.resume.is_file():
                raise FileNotFoundError('Exact owned resume checkpoint unavailable: '+str(args.resume))
            latest = args.resume or args.output/'training/checkpoint_latest.pt'
            if latest.exists():
                training_config['v24_runtime']['resume_checkpoint'] = str(latest.resolve(strict=True))
            result = run_training(net,scorer,population,training_config,output=args.output/'training',
                identity=bound,budget=budget,model_contract=contract,physical_patient_batch=4,workers=runtime['workers'])
            print(json.dumps(result,ensure_ascii=False),flush=True)
        finally:
            restore_lease_guard()
            close()


if __name__ == '__main__':
    main()
