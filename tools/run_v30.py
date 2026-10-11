"""v3.0 = confirmed v2.5 STU-Net-S/context/curriculum plus complete vessels.

Select one arm and one physical GPU by number. No SSH, host or UUID mapping.
Pipeline performs immutable preparation, measured full128 calibration, then
the complete 40-epoch run. Exact own checkpoints resume automatically.
"""
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
V3_FILES = ('hiercp_v1x/v25_context.py','hiercp_v1x/v25_targets.py',
    'hiercp_v1x/v30_factory.py','hiercp_v1x/v30_preparation.py','hiercp_v1x/v30_scorer.py',
    'hiercp_v1x/v30_model.py','hiercp_v1x/v30_vessels.py','hiercp_v1x/v30_training.py',
    'hiercp_v1x/v30_cp.py','tools/run_v30.py')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*2**20), b''):
            digest.update(block)
    return digest.hexdigest()


def parse(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('prepare','calibrate','train','pipeline','score-cp'), default='pipeline')
    parser.add_argument('--arm', choices=('A','B','C','D'), required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--config', type=Path, default=ROOT/'config/v30.json')
    parser.add_argument('--native-experiment', type=Path, required=True)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--vessels', type=Path, required=True, help='Complete 131-case exported vessel index.json')
    parser.add_argument('--stunet-checkpoint', type=Path, required=True)
    parser.add_argument('--input-cache', type=Path, required=True, help='New immutable v3 cache; shared by all four arms')
    parser.add_argument('--output', type=Path, required=True, help='Independent arm-owned result directory')
    parser.add_argument('--resume', type=Path, help='Exact owned v3 checkpoint; defaults to own latest')
    parser.add_argument('--source-bank', type=Path, help='Existing complete matched BasicCP bank for score-cp')
    parser.add_argument('--baseline-preprocessed', type=Path, help='Original Dataset730 preprocessed directory for matched CP')
    args = parser.parse_args(argv)
    if args.gpu < 0:
        parser.error('--gpu must be a nonnegative physical GPU number')
    if args.mode == 'score-cp' and (args.source_bank is None or args.baseline_preprocessed is None):
        parser.error('--source-bank and --baseline-preprocessed required for score-cp')
    return args


def configuration(path, arm):
    config = json.loads(Path(path).read_text(encoding='utf8'))
    if (config.get('version') != 'v3.0' or config.get('base') != 'v2.5'
            or config.get('encoder') != 'official_pretrained_STU_Net_S'
            or config.get('seed') != 42 or config.get('epochs') != 40 or config.get('margin_mm') != 10
            or config.get('MonoPlace3D') is not False):
        raise ValueError('Complete authorized v3.0/v2.5/seed42/40epoch/10mm/STU contract required')
    runtime = config['runtime']
    for key, expected in dict(workers=4, torch_threads=1, CUDA_gib=40, rss_gib=64,
            resident_gib=32, raw_resident_gib=8, geometry_resident_gib=8,
            calibration_repeats=3, fixed_validation_epoch=29,
            all_P_from_epoch_one=True, other_P_as_negative=False,
            hidden_subset=False, debug=False).items():
        if runtime.get(key) != expected:
            raise ValueError('Frozen production resource/data setting differs: '+key)
    if (runtime['physical_patient_batch_candidates'] != [4,8]
            or runtime['physical_candidate_batch_candidates'] != [32,64,128]
            or runtime.get('independent_single_GPU') is not True
            or not all(runtime[key] is True for key in ('persistent_CPU_workers','pin_CPU_batches','non_blocking_H2D'))):
        raise ValueError('Full B4/B8, chunk32/64/128 measured batching required')
    if (set(config['arms']) != {'A','B','C','D'} or any(config['arms'][key]['initial_u'] != value
            or config['arms'][key]['increment_u'] != (None if key=='D' else value)
            for key,value in (('A',8),('B',16),('C',32),('D',128)))):
        raise ValueError('Confirmed four-arm cumulative native-prefix policies required')
    if (config['vessels'].get('hidden_dim') != 128 or config['vessels'].get('heads') != 4
            or config['vessels'].get('layers') != 2 or config['vessels'].get('preserve_all_nodes_edges_components') is not True
            or config['vessels'].get('recipient_tumor_overlap_as_input') is not False):
        raise ValueError('Actual complete two-layer/128D/four-head vessel branch required')
    selected = copy.deepcopy(config)
    selected['arm'] = arm
    selected['runtime']['curriculum'] = copy.deepcopy(config['arms'][arm])
    return selected


def publish(path, value):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text(encoding='utf8')) != value:
            raise FileExistsError('Existing owned v3 evidence differs; preserved: '+str(path))
        return
    with path.open('x', encoding='utf8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def request(args, config):
    # Physical device numbers are execution choices, never anatomy or model
    # identity. Hardware-specific calibration remains a separate requirement.
    from tools.run_v24_all_p import FILES
    files = tuple(dict.fromkeys(FILES+V3_FILES))
    value = dict(format='v30_complete_execution_request_v1', config=config,
        source_sha256={name:sha(ROOT/name) for name in files},
        native_experiment=str(args.native_experiment.resolve()), inventory=str(args.inventory.resolve()),
        inventory_sha256=sha(args.inventory), input_cache=str(args.input_cache.resolve()),
        vessel_index_sha256=sha(args.vessels), STU_checkpoint_sha256=sha(args.stunet_checkpoint))
    value['request_sha256'] = hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return value


def validate_output_paths(args):
    """Admit fresh owned namespaces before creating locks or metadata."""
    output,args_cache=args.output.resolve(),args.input_cache.resolve()
    def overlaps(first,second):
        return first.is_relative_to(second) or second.is_relative_to(first)
    if overlaps(output,args_cache):
        raise ValueError('Arm output and immutable shared input cache must be disjoint directories')
    inventory=json.loads(args.inventory.read_text(encoding='utf8'))
    protected={args.native_experiment.resolve(),args.vessels.resolve().parent,
        args.stunet_checkpoint.resolve(),args.inventory.resolve()}
    for row in inventory.get('raw_records',[]):
        protected.update(Path(row[key]).resolve().parent for key in ('image','label'))
    for optional in (args.source_bank,args.baseline_preprocessed):
        if optional is not None:
            protected.add(optional.resolve())
    for own in (output,args_cache):
        if any(overlaps(own,source) for source in protected):
            raise ValueError('New v3 output/cache overlaps preserved experiment, CT, vessel, inventory or weights: '+str(own))


def main(argv=None):
    args = parse(argv)
    config = configuration(args.config, args.arm)
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu) if args.mode != 'prepare' else ''
    sys.path.insert(0, str(ROOT))
    import psutil
    import torch
    from hiercp_v1x import v30_factory
    from hiercp_v1x.v30_preparation import install_runtime
    from hiercp_v1x.v25_targets import validate_policy
    from hiercp_v1x.u_bridge_experiment import lock
    runtime = config['runtime']
    policy = validate_policy(runtime['curriculum'])
    torch.set_num_threads(runtime['torch_threads'])
    available_cpus = len(psutil.Process().cpu_affinity())
    if available_cpus < runtime['workers']:
        raise RuntimeError('Actual allocated CPUs below four-worker production contract')
    if psutil.virtual_memory().available < runtime['rss_gib']*2**30:
        raise MemoryError('Actual available RAM below declared 64GiB budget')
    ancestor=args.input_cache.absolute()
    while not ancestor.exists():
        ancestor=ancestor.parent
    if shutil.disk_usage(ancestor).free < runtime['minimum_free_disk_gib']*2**30:
        raise OSError('Cache filesystem lacks declared reserve; no previous files deleted')
    validate_output_paths(args)
    args.output.mkdir(parents=True, exist_ok=True)
    args.input_cache.mkdir(parents=True, exist_ok=True)
    bound = request(args, config)
    cpu_runtime = install_runtime(v30_factory)
    prepared_inputs=None
    prepared_vessels=None
    with lock(args.output/'.v30.lock'):
        publish(args.output/'request.json', bound)
        if args.mode in ('prepare','pipeline'):
            # Shared-cache lock is limited to static preparation. Each arm has
            # its own training lock, Adam, cursor and checkpoints afterwards.
            with lock(args.input_cache/'.prepare.lock'):
                inputs = v30_factory.V30NativeInputs(args.native_experiment,args.inventory,args.input_cache,runtime)
                try:
                    from hiercp_v1x.v30_vessels import VesselStore
                    store = VesselStore(args.vessels,ct_paths={case:row['image'] for case,row in inputs.raw.items()})
                    from concurrent.futures import ThreadPoolExecutor
                    with ThreadPoolExecutor(max_workers=runtime['workers']) as pool:
                        bindings = dict(zip(store.case_ids,pool.map(store.binding_from_source,store.case_ids)))
                    vessel_receipt = store.preload(recipient_bindings=bindings,workers=runtime['workers'])
                    index = inputs.prepare_local()
                    full = inputs.geometry.prepare(128)
                    initial = inputs.geometry.prepare(runtime['curriculum']['initial_u'])
                    publish(args.output/'preparation.json',dict(request_sha256=bound['request_sha256'],
                        local_index_sha256=sha(index), full_upper=full,initial_upper=initial,
                        context=inputs.context_contract,vessels=vessel_receipt,
                        population=inputs.population.manifest(),curriculum=policy,
                        CPU_runtime=cpu_runtime,production_optimizer_updates=0,
                        neural_model_created=False,quality_verified=False))
                finally:
                    inputs.close()
                prepared_inputs=inputs
                prepared_vessels=store
            if args.mode == 'prepare':
                print('v3.0 complete immutable preparation finished; no training.', flush=True)
                return
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError('One actual selected CUDA device required; no CPU fallback')
        free,total = torch.cuda.mem_get_info()
        properties = torch.cuda.get_device_properties(0)
        if min(free,total) <= runtime['CUDA_gib']*2**30:
            raise MemoryError('Selected GPU has less than declared 40GiB free budget; production unchanged')
        torch.cuda.set_per_process_memory_fraction(runtime['CUDA_gib']*2**30/total)
        from hiercp_v1x.historical_evaluation import ResourceBudget
        from hiercp_v1x.v30_training import calibrate_training_batches, run_training
        budget = ResourceBudget(runtime['CUDA_gib']*2**30,runtime['rss_gib']*2**30)
        net,scorer,population,training_config,contract,close = v30_factory.build_runtime(
            args,config,budget,inputs=prepared_inputs,vessel_store=prepared_vessels)
        try:
            hardware = dict(name=properties.name,total_VRAM_bytes=total)
            publish(args.output/('resources_'+str(time.time_ns())+'.json'),dict(
                physical_GPU=args.gpu,hardware=hardware,free_VRAM_bytes=free,CPU_affinity=psutil.Process().cpu_affinity(),
                available_RAM_bytes=psutil.virtual_memory().available,workers=runtime['workers'],
                model=contract,graph_config=training_config['graph'],population=population.manifest(),
                physical_patient_candidates=[4,8],candidate_chunks=[32,64,128],gradient_accumulation=1,
                precision='AMP',epochs=40,seed=42,debug=False,quality_verified=False))
            calibration_path = args.output/'calibration.json'
            if args.mode == 'calibrate' or (args.mode == 'pipeline' and not calibration_path.exists()):
                report = calibrate_training_batches(net,scorer,population,training_config,budget=budget,model_contract=contract)
                report.update(request_sha256=bound['request_sha256'],hardware=hardware)
                publish(calibration_path,report)
                if args.mode == 'calibrate':
                    print('v3.0 measured full128 calibration complete; no production update.',flush=True)
                    return
            calibration = json.loads(calibration_path.read_text(encoding='utf8'))
            if calibration.get('request_sha256') != bound['request_sha256'] or calibration.get('hardware') != hardware:
                raise ValueError('Exact own-source/model/hardware calibration required; existing results preserved')
            training_config['v24_runtime']['batch_calibration'] = calibration
            scorer.physical_candidate_batch = calibration['selected_physical_candidate_batch']
            if args.resume is not None and not args.resume.is_file():
                raise FileNotFoundError('Explicit owned resume checkpoint unavailable: '+str(args.resume))
            latest = args.resume or args.output/'training/checkpoint_latest.pt'
            if latest.exists():
                training_config['v24_runtime']['resume_checkpoint'] = str(latest.resolve(strict=True))
            if args.mode == 'score-cp':
                best = args.output/'training/checkpoint_best.pt'
                payload = torch.load(best,map_location='cpu',weights_only=False)
                from hiercp_v1x.v30_training import FORMAT, validate_saved_best, validate_curriculum_progress
                from hiercp_v1x.u_bridge_training import digest
                ownership=json.loads((args.output/'training/training_identity.json').read_text(encoding='utf8'))
                finished=torch.load(args.output/'training/checkpoint_latest.pt',map_location='cpu',weights_only=False)
                if (ownership['binding']['identity'] != bound
                        or digest(ownership['binding']) != ownership['identity_sha256']
                        or payload.get('format') != FORMAT or finished.get('format') != FORMAT
                        or payload.get('identity_sha256') != ownership['identity_sha256']
                        or finished.get('identity_sha256') != ownership['identity_sha256']
                        or payload.get('content_sha256') != digest({k:v for k,v in payload.items() if k!='content_sha256'})
                        or finished.get('content_sha256') != digest({k:v for k,v in finished.items() if k!='content_sha256'})
                        or finished['state']['phase'] != 'complete' or len(finished['state']['history']) != 40
                        or finished['state']['updates'] != 680):
                    raise ValueError('Exact own completed40/680 v3 BEST required to score CP')
                validate_curriculum_progress(finished['state'],policy=runtime['curriculum'],
                    train_cases=ownership['binding']['train_cases'],val_cases=ownership['binding']['val_cases'],
                    physical_patient_batch=ownership['binding']['physical_patient_batch'])
                validate_saved_best(args.output/'training',finished['state'],ownership['identity_sha256'])
                net.load_state_dict(payload['model'],strict=True)
                net.eval()
                from hiercp_v1x.v30_cp import score_v30_cp_bank
                result = score_v30_cp_bank(net=net,inputs=scorer.inputs,population=population,
                    source_bank=args.source_bank,baseline_preprocessed=args.baseline_preprocessed,vessel_store=scorer.vessel_store,
                    recipient_bindings=scorer.recipient_bindings,
                    physical_candidate_batch=scorer.physical_candidate_batch,budget=budget,
                    amp=bool(training_config['training']['amp']),best_provenance=dict(debug=False,full_training=True,completed_epochs=40,
                        optimizer_updates=680,selected_by='fixed_full128_validation_only',model_sha256=digest(net.state_dict()),
                        context_contract_sha256=contract['context']['contract_sha256'],
                        checkpoint=str(best),checkpoint_sha256=sha(best),arm=args.arm,request_sha256=bound['request_sha256']))
                publish(args.output/'cp_scores.json',result)
                return
            result = run_training(net,scorer,population,training_config,output=args.output/'training',
                identity=bound,budget=budget,model_contract=contract,
                physical_patient_batch=calibration['selected_physical_patient_batch'],workers=runtime['workers'])
            print(json.dumps(result,ensure_ascii=False),flush=True)
        finally:
            close()


if __name__ == '__main__':
    main()
