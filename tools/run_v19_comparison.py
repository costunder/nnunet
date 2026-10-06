"""v1.9: matched coordinate exposure and ranking-objective controls; full129 validation."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import random
import sys
import uuid
from contextlib import ExitStack

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
from hiercp_v1x.comparison_experiment import ARMS


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu', type=int, required=True, help='Physical GPU number; live allocation resolution')
    p.add_argument('--arm', choices=(*ARMS, 'all'), required=True)
    p.add_argument('--experiment', type=Path, required=True, help='Stable matched root; same command resumes')
    p.add_argument('--baseline', type=Path, help='Completed signed original v1 m10 experiment')
    p.add_argument('--inventory', type=Path, required=True, help='Existing native LocalCNN inventory/index.json')
    p.add_argument('--prepared-data', type=Path,
                   help='Optional inactive verified v1.8 data namespace; reuse existing canonical entries')
    p.add_argument('--workers', type=int, required=True)
    p.add_argument('--batch-candidates', type=int, nargs='+', required=True,
                   help='Complete eight-candidate source problems per physical batch, measured on all four arms')
    for name in ('cuda-gib', 'rss-gib', 'resident-gib'):
        p.add_argument('--' + name, type=float, required=True)
    p.add_argument('--validation-local-chunk', type=int, required=True,
                   help='L0-only candidate chunk; all129 upper nodes still scored jointly')
    p.add_argument('--debug', action='store_true')
    p.add_argument('--debug-fixture', type=Path)
    p.add_argument('--debug-config', type=Path)
    p.add_argument('--debug-source', type=Path)
    p.add_argument('--debug-bank', type=Path)
    p.add_argument('--debug-epochs', type=int)
    p.add_argument('--debug-pause-updates', type=int, help='DEBUG-only exact-cursor pause after successful updates')
    a = p.parse_args(argv)
    if a.workers < 2 or min(a.cuda_gib, a.rss_gib, a.resident_gib) <= 0 or a.resident_gib >= a.rss_gib:
        p.error('Parallel workers>=2 and positive limits with resident<RSS required')
    if (not a.batch_candidates or any(n < 1 for n in a.batch_candidates)
            or a.batch_candidates != sorted(set(a.batch_candidates)) or a.validation_local_chunk < 1):
        p.error('Unique explicit positive physical batches and L0 chunk required')
    debug_options = [a.debug_fixture, a.debug_config, a.debug_source, a.debug_bank, a.debug_epochs]
    if a.debug:
        if any(x is None for x in debug_options) or not 1 <= a.debug_epochs <= 2:
            p.error('DEBUG requires signed actual CT fixture/config/source/bank and explicit1..2epochs')
    elif a.baseline is None or any(x is not None for x in debug_options) or a.debug_pause_updates is not None:
        p.error('Production needs --baseline and cannot use DEBUG inputs/epochs')
    if a.debug_pause_updates is not None and a.debug_pause_updates < 1:
        p.error('DEBUG pause requires a positive successful-update count')
    return a


def run(a):
    # Device selection occurs before importing torch or any model package.
    from tools.local_cnn_device import select
    select(a.gpu)
    import numpy as np
    import psutil
    import torch
    from hiercp_v1x.comparison_experiment import (FORMAT, FILES, Budget, digest, read, sha,
                                              write_new, lock, prepare_inputs, joint_calibration,
                                              prepared_data_path)
    from hiercp_v1x.scope_probe_support import activate_original, state_digest
    from hiercp_v1x import bounded_scope
    from hiercp_v1x.comparison_training import (run_arm, calibrate_batches, capture_rng,
                                             restore_rng, atomic_save, cpu_copy, digest as neural_digest)
    if not torch.cuda.is_available():
        raise RuntimeError('Actual CUDA required; no CPU neural fallback')
    if torch.cuda.device_count() != 1:
        raise ValueError('Select exactly one allocated GPU for this matched run')
    capacity = torch.cuda.get_device_properties(0).total_memory
    if a.cuda_gib * 2**30 >= capacity:
        raise ValueError(f'CUDA budget must leave driver/allocator headroom; actual capacity={capacity/2**30:.2f}GiB')
    torch.cuda.set_per_process_memory_fraction(a.cuda_gib * 2**30 / capacity)
    torch.set_num_threads(a.workers)
    root = a.experiment.resolve()
    if a.baseline is not None:
        baseline = a.baseline.resolve(strict=True)
        if root.is_relative_to(baseline) or baseline.is_relative_to(root):
            raise ValueError('New experiment must be disjoint from preserved baseline')
    root.mkdir(parents=True, exist_ok=True)
    # Protect shared init, geometry, and matched batch; arm output has its own lock.
    with lock(root / '.setup.lock'):
        samples, raw, config, source, bank_path, regions, proof = prepare_inputs(a)
        original = activate_original(source)
        scope = bounded_scope.install(10, expected_snapshot_root=source)
        from hiercp.model import HierarchicalPyGPlacementModel
        from hiercp.prototype import PrototypeBank
        from hiercp.tensor import configure_runtime, collect_runtime_resources
        configure_runtime(deterministic=bool(config['runtime'].get('deterministic', True)),
                          allow_tf32=bool(config['runtime'].get('allow_tf32', False)),
                          cudnn_benchmark=bool(config['runtime'].get('cudnn_benchmark', False)))
        resource = collect_runtime_resources('cuda', storage_path=root)
        print(json.dumps(dict(stage='v19_actual_allocation', resources=resource)), flush=True)
        if a.debug:
            fixture = torch.load(bank_path, map_location='cpu', weights_only=False, mmap=True)
            bank = fixture['prototype_bank']
            del fixture
        else:
            bank = PrototypeBank.load(bank_path)
        budget = Budget(int(a.cuda_gib * 2**30), int(a.rss_gib * 2**30))
        from hiercp_v1x.comparison_data import ComparisonData
        data_root = prepared_data_path(root, a.prepared_data, a.baseline)
        provider = ComparisonData(samples, raw, config, bank, data_root, a.workers,
                               int(a.resident_gib * 2**30), budget, regions_dir=regions)
        from hiercp_v1x.u_bridge_calibration import CalibrationProvider
        calibration_provider = CalibrationProvider(provider, samples)
        train, val = provider.examples('train'), provider.examples('val')
        if not train or not val:
            raise ValueError('Complete nonempty original train and validation lists required')
        if max(a.batch_candidates) > len(train):
            raise ValueError('Calibration cannot duplicate sources to fake a requested physical batch')
        if a.debug and bank.training_case_ids != ('liver_5', 'liver_6'):
            raise ValueError('DEBUG fixture population bank identity differs')
        semantic = read(ROOT / 'config/v19_comparison_controls.json')
        content = dict(format=FORMAT, debug=a.debug, epochs=a.debug_epochs if a.debug else 40,
            semantic=semantic, original=original, scope=scope, baseline=proof,
            config=config, samples=provider.examples('train') + provider.examples('val'),
            prototype_fingerprint=bank.fingerprint(), workers=a.workers,
            explicit_batch_candidates=a.batch_candidates, validation_local_chunk=a.validation_local_chunk,
            cuda_gib=a.cuda_gib, rss_gib=a.rss_gib, resident_gib=a.resident_gib,
            prepared_data_root=str(data_root),
            calibration_source_stress_order=calibration_provider.stress_receipts,
            helpers={name: sha(ROOT / name) for name in FILES})
        content = json.loads(json.dumps(content, allow_nan=False))
        contract = dict(content, sha256=digest(content))
        path = root / 'experiment.json'
        if path.exists():
            if read(path) != contract:
                raise ValueError('Existing experiment input/model/code/execution differs; originals preserved')
        else:
            write_new(path, contract)
        def new_model():
            random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
            result = HierarchicalPyGPlacementModel(**config['model']).cuda()
            if sum(p.numel() for p in result.parameters()) != 10434532:
                raise ValueError('Original full10,434,532parameter model changed')
            return result
        net = new_model()
        initial_path = root / 'initial.pt'
        if not initial_path.exists():
            atomic_save(initial_path, dict(model=cpu_copy(net.state_dict()), rng=cpu_copy(capture_rng()),
                                          model_sha256=state_digest(net.state_dict()), contract_sha256=contract['sha256']))
        initial = torch.load(initial_path, map_location='cpu', weights_only=False)
        if (initial['contract_sha256'] != contract['sha256']
                or initial['model_sha256'] != state_digest(initial['model'])):
            raise ValueError('Saved common initial neural state changed')
        net.load_state_dict(initial['model'], strict=True); restore_rng(initial['rng'])
        calibration_path = root / 'calibration.json'
        if calibration_path.exists():
            calibration = read(calibration_path)
            if (calibration['contract_sha256'] != contract['sha256']
                    or calibration['GPU_name'] != torch.cuda.get_device_name(0)):
                raise ValueError('Matched calibration belongs to another contract or GPU architecture')
        else:
            reports = {}
            for arm in ARMS:
                try:
                    _, receipt = calibrate_batches(net, calibration_provider, config, arm=arm,
                        candidates=a.batch_candidates, workers=a.workers, budget=budget, debug=a.debug)
                except Exception as error:
                    if hasattr(error, 'reports'):
                        failed = root / ('calibration_failed_' + arm + '.json')
                        if not failed.exists(): write_new(failed, dict(arm=arm, reports=error.reports))
                    raise
                reports[arm] = receipt
                report_path = root / ('calibration_' + arm + '.json')
                if not report_path.exists(): write_new(report_path, receipt)
            calibration = joint_calibration(reports, a.batch_candidates)
            calibration.update(contract_sha256=contract['sha256'], GPU_name=torch.cuda.get_device_name(0),
                               total_VRAM_bytes=capacity,
                               source_stress_order=calibration_provider.stress_receipts,
                               future_cohort_worst_case_verified=False)
            write_new(calibration_path, calibration)
        chosen = calibration['physical_batch']
        print(json.dumps(dict(stage='v19_matched_execution', version='v1.9', debug=a.debug,
            arms=a.arm, source_samples=dict(train=len(train), val=len(val)),
            actual_cases=dict(train=len({s['case_id'] for s in train}), val=len({s['case_id'] for s in val})),
            configured_but_not_materialized_validation_cases=proof.get('configured_but_not_materialized_validation_cases', []),
            parameters=10434532, model=config['model'], input_dense_shape=[5,48,48,48], margin_mm=10,
            train_candidates_per_source=8, eval_candidates_per_source=129, physical_sample_batch=chosen,
            gradient_accumulation_steps=1, effective_sample_batch=chosen, epochs=content['epochs'],
            worker_count=a.workers, GPU=torch.cuda.get_device_name(), total_VRAM_bytes=capacity,
            CPU_affinity=len(psutil.Process().cpu_affinity()), available_RAM_bytes=psutil.virtual_memory().available,
            initial_neural_sha256=initial['model_sha256'], full_training=False, quality_verified=False)), flush=True)
    arms = ARMS if a.arm == 'all' else (a.arm,)
    results = {}
    for arm in arms:
        output = root / arm
        output.mkdir(exist_ok=True)
        with lock(root / ('.' + arm + '.lock')):
            net.load_state_dict(initial['model'], strict=True); restore_rng(initial['rng'])
            receipt = copy.deepcopy(calibration['reports'][arm])
            receipt['selected_physical_batch'] = chosen
            runtime = dict(batch_calibration=receipt, validation_local_chunk_size=a.validation_local_chunk,
                           expected_parameters=10434532, pause_file=str(output / 'STOP_AFTER_BATCH'))
            if a.debug_pause_updates is not None:
                runtime['debug_pause_after_updates'] = a.debug_pause_updates
            latest = output / 'checkpoint_latest.pt'
            if latest.exists(): runtime['resume_checkpoint'] = str(latest)
            arm_config = copy.deepcopy(config); arm_config['u_bridge_runtime'] = runtime
            result = run_arm(net, provider, arm_config, arm=arm, output=output,
                physical_batch=chosen, workers=a.workers, epochs=content['epochs'],
                identity=dict(contract_sha256=contract['sha256'], initial_neural_sha256=initial['model_sha256'],
                              initial_state_sha256=neural_digest(initial['model'])),
                budget=budget, debug=a.debug)
            results[arm] = result
            write_new(root / 'invocations' / (arm + '_' + uuid.uuid4().hex + '.json'),
                dict(contract_sha256=contract['sha256'], arm=arm, result=result,
                     provider=provider.report(), actual_resources=resource))
            print(f'v1.9 {arm} RESULT: {output}', flush=True)
            if result.get('status') == 'PAUSED':
                break
    return results


if __name__ == '__main__':
    arguments = parse()
    from hiercp_v1x.comparison_experiment import lock, prepared_data_path
    data_namespace = prepared_data_path(arguments.experiment, arguments.prepared_data, arguments.baseline)
    with ExitStack() as locks:
        locks.enter_context(lock(arguments.experiment.resolve() / '.pipeline.lock'))
        locks.enter_context(lock(data_namespace / '.data.lock'))
        run(arguments)
