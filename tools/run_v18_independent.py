"""Resume one preserved v1.8 arm in its own cache/output namespace."""
from __future__ import annotations

import argparse
import copy
from contextlib import ExitStack
import json
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True


def parse(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--arm', choices=('selected', 'native'), required=True)
    parser.add_argument('--source-experiment', type=Path, required=True)
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--debug-bank', type=Path,
                        help='Only for an already sealed actual-CT DEBUG experiment')
    return parser.parse_args(argv)


def canonical(value):
    return json.loads(json.dumps(value, allow_nan=False))


def validate_reference(root):
    """The neural controller/engine and sealed settings must remain byte exact."""
    from hiercp_v1x.u_bridge_experiment import FORMAT, FILES, digest, read, sha
    manifest = read(Path(root) / 'experiment.json')
    content = {key: value for key, value in manifest.items() if key != 'sha256'}
    if manifest.get('format') != FORMAT or manifest.get('sha256') != digest(content):
        raise ValueError('Preserved v1.8 contract digest differs')
    if set(manifest.get('helpers', {})) != set(FILES):
        raise ValueError('Preserved v1.8 helper inventory differs')
    changed = [name for name, expected in manifest['helpers'].items() if sha(ROOT / name) != expected]
    if changed:
        raise ValueError(f'Preserved neural/input controller changed: {changed}')
    if (manifest.get('debug') is not True and manifest.get('epochs') != 40
            or manifest.get('semantic', {}).get('version') != 'v1.8'):
        raise ValueError('Preserved v1.8 forty-epoch production contract required')
    return manifest


def input_paths(manifest, inventory, debug_bank):
    from hiercp_v1x.u_bridge_experiment import read, sha, prepare_inputs
    if sha(inventory) != manifest['baseline']['inventory_sha256']:
        raise ValueError('Original frozen128 inventory bytes differ')
    if manifest['debug']:
        if debug_bank is None or sha(debug_bank) != manifest['baseline']['bank_fixture_sha256']:
            raise ValueError('Sealed actual-CT DEBUG bank bytes required')
        samples = copy.deepcopy(manifest['baseline']['source_samples'])
        config = copy.deepcopy(manifest['config'])
        source = Path(manifest['original']['source'])
        return samples, read(inventory)['raw_records'], config, source, debug_bank, None
    if debug_bank is not None:
        raise ValueError('DEBUG bank cannot be used for production')
    from types import SimpleNamespace
    baseline = Path(manifest['baseline']['baseline'])
    samples, raw, config, source, bank, regions, proof = prepare_inputs(
        SimpleNamespace(debug=False, baseline=baseline, inventory=inventory))
    if proof != manifest['baseline'] or config != manifest['config']:
        raise ValueError('Preserved baseline/data/configuration changed')
    return samples, raw, config, source, bank, regions


def run(arguments):
    # Numeric physical GPU selection remains before any torch import.
    from tools.local_cnn_device import select
    select(arguments.gpu)
    from hiercp_v1x.u_bridge_experiment import lock, read, sha, write_new
    from hiercp_v1x.u_bridge_continuation import prepare_continuation
    from hiercp_v1x.host_memory import PressureBudget, pressure_aware_provider
    manifest = validate_reference(arguments.source_experiment)
    samples, raw, config, source, bank_path, regions = input_paths(
        manifest, arguments.inventory, arguments.debug_bank)
    root = arguments.experiment.resolve()
    with ExitStack() as locks:
        locks.enter_context(lock(root / '.pipeline.lock'))
        continuation = prepare_continuation(arguments.source_experiment, root, arguments.arm)
        data_root = Path(continuation['data_root'])
        locks.enter_context(lock(data_root / '.data.lock'))
        import numpy as np
        import random
        import torch
        from hiercp_v1x.scope_probe_support import activate_original, state_digest
        from hiercp_v1x import bounded_scope
        from hiercp_v1x.u_bridge_data import UBridgeData
        from hiercp_v1x.u_bridge_training import run_arm, restore_rng, digest as neural_digest
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError('Exactly one actual CUDA GPU required; no neural CPU fallback')
        capacity = torch.cuda.get_device_properties(0).total_memory
        cuda_bytes = int(manifest['cuda_gib'] * 2**30)
        if cuda_bytes >= capacity:
            raise ValueError('Original CUDA budget must fit selected GPU with headroom')
        torch.cuda.set_per_process_memory_fraction(cuda_bytes / capacity)
        torch.set_num_threads(manifest['workers'])
        original = activate_original(source)
        scope = bounded_scope.install(10, expected_snapshot_root=source)
        if canonical(original) != manifest['original'] or canonical(scope) != manifest['scope']:
            raise ValueError('Original archived model or10mm graph scope changed')
        from hiercp.model import HierarchicalPyGPlacementModel
        from hiercp.prototype import PrototypeBank
        from hiercp.tensor import configure_runtime, collect_runtime_resources
        configure_runtime(deterministic=bool(config['runtime'].get('deterministic', True)),
                          allow_tf32=bool(config['runtime'].get('allow_tf32', False)),
                          cudnn_benchmark=bool(config['runtime'].get('cudnn_benchmark', False)))
        if manifest['debug']:
            fixture = torch.load(bank_path, map_location='cpu', weights_only=False, mmap=True)
            bank = fixture['prototype_bank']
            del fixture
        else:
            bank = PrototypeBank.load(bank_path)
        if bank.fingerprint() != manifest['prototype_fingerprint']:
            raise ValueError('Original prototype bank changed')
        events = root / 'host_memory.jsonl'
        def log_event(row):
            with events.open('a', encoding='utf8') as stream:
                stream.write(json.dumps(row, allow_nan=False) + '\n')
            print('Host memory | ' + json.dumps(row, allow_nan=False), flush=True)
        budget = PressureBudget(cuda_bytes, int(manifest['rss_gib'] * 2**30),
                                resident_bytes=int(manifest['resident_gib'] * 2**30),
                                event_callback=log_event)
        provider = pressure_aware_provider(UBridgeData)(samples, raw, config, bank, data_root,
            manifest['workers'], int(manifest['resident_gib'] * 2**30), budget, regions_dir=regions)
        if canonical(provider.examples('train') + provider.examples('val')) != manifest['samples']:
            raise ValueError('Original source problems/anchors/candidate positions changed')
        calibration = read(root / 'calibration.json')
        if (calibration['contract_sha256'] != manifest['sha256']
                or calibration['GPU_name'] != torch.cuda.get_device_name(0)):
            raise ValueError('Original measured batch lock belongs to another contract/GPU architecture')
        chosen = calibration['physical_batch']
        random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
        net = HierarchicalPyGPlacementModel(**config['model']).cuda()
        if sum(p.numel() for p in net.parameters()) != 10434532:
            raise ValueError('Original full10,434,532parameter model changed')
        initial = torch.load(root / 'initial.pt', map_location='cpu', weights_only=False)
        if (initial['contract_sha256'] != manifest['sha256']
                or initial['model_sha256'] != state_digest(initial['model'])):
            raise ValueError('Original common initial state changed')
        net.load_state_dict(initial['model'], strict=True); restore_rng(initial['rng'])
        output = root / arguments.arm
        output.mkdir(exist_ok=True)
        receipt = copy.deepcopy(calibration['reports'][arguments.arm])
        receipt['selected_physical_batch'] = chosen
        runtime = dict(batch_calibration=receipt,
                       validation_local_chunk_size=manifest['validation_local_chunk'],
                       expected_parameters=10434532, pause_file=str(output / 'STOP_AFTER_BATCH'))
        checkpoint = output / 'checkpoint_latest.pt'
        if checkpoint.exists():
            runtime['resume_checkpoint'] = str(checkpoint)
        arm_config = copy.deepcopy(config); arm_config['u_bridge_runtime'] = runtime
        policy = dict(format='v18_independent_host_memory_execution_v1',
            historical_contract_sha256=manifest['sha256'], arm=arguments.arm,
            independent_data_root=str(data_root), model_parameters=10434532,
            original_physical_batch=chosen, workers=manifest['workers'],
            cuda_gib=manifest['cuda_gib'], rss_gib=manifest['rss_gib'], resident_gib=manifest['resident_gib'],
            train_candidates=8, evaluation_candidates=129, genuine_local_views=2,
            neural_engine_unchanged=True, full_training=False, quality_verified=False,
            execution_helpers={name: sha(ROOT / name) for name in
                ('tools/run_v18_independent.py', 'hiercp_v1x/host_memory.py',
                 'hiercp_v1x/u_bridge_continuation.py')})
        policy_path = root / 'memory_execution.json'
        if policy_path.exists():
            if read(policy_path) != policy:
                raise ValueError('Independent memory execution policy changed')
        else:
            write_new(policy_path, policy)
        resources = collect_runtime_resources('cuda', storage_path=root)
        print(json.dumps(dict(stage='v18_independent_execution', arm=arguments.arm,
            resume=str(checkpoint) if checkpoint.exists() else 'NEW: preserved common initial weights',
            settings=policy, resources=resources), allow_nan=False), flush=True)
        with lock(root / ('.' + arguments.arm + '.lock')):
            result = run_arm(net, provider, arm_config, arm=arguments.arm, output=output,
                physical_batch=chosen, workers=manifest['workers'], epochs=manifest['epochs'],
                identity=dict(contract_sha256=manifest['sha256'], initial_neural_sha256=initial['model_sha256'],
                              initial_state_sha256=neural_digest(initial['model'])),
                budget=budget, debug=manifest['debug'])
        write_new(root / 'invocations' / (arguments.arm + '_' + uuid.uuid4().hex + '.json'),
                  dict(result=result, provider=provider.report(), execution_policy=policy))
        print(f'v1.8 {arguments.arm} RESULT: {output}', flush=True)
        return result


if __name__ == '__main__':
    run(parse())
