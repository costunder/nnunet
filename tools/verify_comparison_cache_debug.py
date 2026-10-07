"""Actual CT/CUDA DEBUG: borrowed preprocessing and the frozen comparison engine.

This isolated two-epoch fixture verifies mechanics, not ranking quality or a
server runtime improvement. It never writes to a reference experiment/cache.
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True


def canonical(value):
    return json.loads(json.dumps(value, allow_nan=False))


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def cache_inventory(root):
    return {p.relative_to(root).as_posix(): sha(p)
            for p in sorted(Path(root).rglob('*')) if p.is_file()}


def verify(arguments):
    from tools.local_cnn_device import select
    select(arguments.gpu)
    import torch
    from hiercp_v1x.comparison_experiment import FORMAT, FILES, digest, write_new
    from hiercp_v1x.comparison_data import ComparisonData
    from hiercp_v1x.comparison_training import run_arm, restore_rng, digest as neural_digest
    from hiercp_v1x.scope_probe_support import activate_original, state_digest
    from hiercp_v1x import bounded_scope
    from hiercp_v1x.host_memory import PressureBudget, pressure_aware_provider
    from hiercp_v1x.preparation_reuse import preparation_reuse

    reference = arguments.reference.resolve(strict=True)
    output = arguments.output.resolve()
    if output.exists():
        raise FileExistsError('DEBUG output must be new; all old results are preserved')
    if output.is_relative_to(reference) or reference.is_relative_to(output):
        raise ValueError('DEBUG output must be disjoint from the reference')
    manifest = read(reference / 'experiment.json')
    if (manifest.get('format') != FORMAT or manifest.get('debug') is not True
            or manifest.get('epochs') != 2 or manifest.get('workers') != 4
            or manifest.get('explicit_batch_candidates') != [2]
            or manifest.get('sha256') != digest({k:v for k,v in manifest.items() if k != 'sha256'})):
        raise ValueError('The sealed actual-CT two-epoch, batch2, worker4 fixture is required')
    if set(manifest['helpers']) != set(FILES):
        raise ValueError('Frozen17-helper inventory differs')
    helpers_before = {name: sha(ROOT / name) for name in FILES}
    if helpers_before != manifest['helpers']:
        raise ValueError('Frozen input/model/training source bytes changed')
    if sha(arguments.inventory) != manifest['baseline']['inventory_sha256']:
        raise ValueError('Frozen128 candidate inventory changed')
    if sha(arguments.bank) != manifest['baseline']['bank_fixture_sha256']:
        raise ValueError('Actual-CT prototype bank fixture changed')
    source = Path(manifest['original']['source'])
    original = activate_original(source)
    scope = bounded_scope.install(10, expected_snapshot_root=source)
    if canonical(original) != manifest['original'] or canonical(scope) != manifest['scope']:
        raise ValueError('Original neural source or10mm graph scope changed')
    from hiercp.model import HierarchicalPyGPlacementModel
    from hiercp.tensor import configure_runtime, collect_runtime_resources
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Exactly one actual CUDA GPU required; no neural CPU fallback')
    torch.set_num_threads(manifest['workers'])
    capacity = torch.cuda.get_device_properties(0).total_memory
    cuda_bytes = int(manifest['cuda_gib'] * 2**30)
    if cuda_bytes >= capacity:
        raise ValueError('Preserved12GiB CUDA budget must leave actual driver headroom')
    torch.cuda.set_per_process_memory_fraction(cuda_bytes / capacity)
    config = copy.deepcopy(manifest['config'])
    configure_runtime(deterministic=bool(config['runtime'].get('deterministic', True)),
                      allow_tf32=bool(config['runtime'].get('allow_tf32', False)),
                      cudnn_benchmark=bool(config['runtime'].get('cudnn_benchmark', False)))
    source_cache = Path(manifest['prepared_data_root']).resolve(strict=True)
    original_files = [reference/'experiment.json', reference/'initial.pt', reference/'calibration.json',
                      reference/'native_fixed/checkpoint_latest.pt']
    originals_before = {str(path): sha(path) for path in original_files}
    source_cache_before = cache_inventory(source_cache)
    initial = torch.load(reference/'initial.pt', map_location='cpu', weights_only=False)
    if (initial['contract_sha256'] != manifest['sha256']
            or initial['model_sha256'] != state_digest(initial['model'])):
        raise ValueError('Preserved initial neural state changed')
    calibration = read(reference/'calibration.json')
    if calibration['physical_batch'] != 2 or calibration['contract_sha256'] != manifest['sha256']:
        raise ValueError('Actual physicalbatch2 calibration changed')
    fixture = torch.load(arguments.bank, map_location='cpu', weights_only=False, mmap=True)
    bank = fixture['prototype_bank']
    if bank.training_case_ids != ('liver_5', 'liver_6') or bank.fingerprint() != manifest['prototype_fingerprint']:
        raise ValueError('Preserved actual-CT population bank changed')
    del fixture
    output.mkdir(parents=True, exist_ok=False)
    events, event_lock = [], threading.Lock()
    def event(row):
        with event_lock:
            events.append(copy.deepcopy(row))
            with (output/'cache_events.jsonl').open('a', encoding='utf8') as stream:
                stream.write(json.dumps(row, allow_nan=False)+'\n')
    budget = PressureBudget(cuda_bytes, int(manifest['rss_gib'] * 2**30),
                            resident_bytes=int(manifest['resident_gib'] * 2**30), event_callback=event)
    resources = collect_runtime_resources('cuda', storage_path=output)
    print(json.dumps(dict(scope='actual_CT_CUDA_DEBUG_only', resources=resources,
        parameters=10434532, physical_batch=2, workers=4, epochs=2,
        training_candidates=8, validation_candidates=129, actual_train_cases=['liver_5','liver_6'],
        actual_validation_cases=['liver_31'], production_training_started=False,
        quality_verified=False), allow_nan=False), flush=True)
    started = time.perf_counter()
    with preparation_reuse(pressure_aware_provider(ComparisonData), [source_cache], event) as Provider:
        provider = Provider(manifest['baseline']['source_samples'], read(arguments.inventory)['raw_records'],
            config, bank, output/'data', manifest['workers'], int(manifest['resident_gib'] * 2**30), budget,
            regions_dir=source_cache/'regions')
        if canonical(provider.examples('train')+provider.examples('val')) != manifest['samples']:
            raise ValueError('Actual original source problems/anchors changed')
        net = HierarchicalPyGPlacementModel(**config['model']).cuda()
        if sum(p.numel() for p in net.parameters()) != 10434532:
            raise ValueError('Full original10,434,532parameter model changed')
        net.load_state_dict(initial['model'], strict=True)
        restore_rng(initial['rng'])
        receipt = copy.deepcopy(calibration['reports']['native_fixed'])
        receipt['selected_physical_batch'] = 2
        runtime = dict(batch_calibration=receipt, validation_local_chunk_size=manifest['validation_local_chunk'],
                       expected_parameters=10434532, pause_file=str(output/'native_fixed/STOP_AFTER_BATCH'))
        arm_config = copy.deepcopy(config); arm_config['u_bridge_runtime'] = runtime
        identity = dict(contract_sha256=manifest['sha256'], initial_neural_sha256=initial['model_sha256'],
                        initial_state_sha256=neural_digest(initial['model']))
        result = run_arm(net, provider, arm_config, arm='native_fixed', output=output/'native_fixed',
            physical_batch=2, workers=4, epochs=2, identity=identity, budget=budget, debug=True)
        checkpoint = output/'native_fixed/checkpoint_latest.pt'
        checkpoint_before = sha(checkpoint)
        checkpoint_content_before = torch.load(checkpoint, map_location='cpu',
            weights_only=False)['content_sha256']
        rows_before = (output/'native_fixed/update_timing.jsonl').read_text(encoding='utf8').splitlines()
        runtime['resume_checkpoint'] = str(checkpoint)
        # Match the real controller: declare the same fresh initial state;
        # the frozen engine then restores all saved neural/optimizer history.
        net.load_state_dict(initial['model'], strict=True)
        restore_rng(initial['rng'])
        resumed = run_arm(net, provider, arm_config, arm='native_fixed', output=output/'native_fixed',
            physical_batch=2, workers=4, epochs=2, identity=identity, budget=budget, debug=True)
        checkpoint_after = sha(checkpoint)
        checkpoint_content_after = torch.load(checkpoint, map_location='cpu',
            weights_only=False)['content_sha256']
        rows_after = (output/'native_fixed/update_timing.jsonl').read_text(encoding='utf8').splitlines()
        stats = provider.report()
    updates = [json.loads(row) for row in rows_before]
    if len(updates) != 2 or any(row['status'] != 'OPTIMIZER_UPDATED' for row in updates):
        raise AssertionError('Expected two fullphysicalbatch2 CUDA optimizer updates')
    if any(row['gradient']['gradient_present'] != 1085 or row['gradient']['missing']
           or row['gradient']['finite'] is not True for row in updates):
        raise AssertionError('Every original1085 trainable parameter tensor requires finite gradients')
    if (checkpoint_content_before != checkpoint_content_after or rows_before != rows_after
            or resumed['resume']['invocation_optimizer_updates'] != 0
            or resumed['resume'].get('completed_checkpoint_no_repeat_verified') is not True):
        raise AssertionError('Completed exact resume changed checkpoint content or repeated neural work')
    if stats['local_builds'] != 0 or stats['field_builds'] != 0:
        raise AssertionError('Completed canonical/field caches were rebuilt instead of borrowed')
    if stats['static_upper']['original_helper_builds'] != 0:
        raise AssertionError('Completed original static upper helpers were recomputed')
    borrowed = {row.get('kind') for row in events if row.get('status') == 'reused'}
    if not {'canonical_local','whole_case_fields','upper_static'} <= borrowed:
        raise AssertionError(f'Expected fields, upper and canonical cache imports; actual={borrowed}')
    originals_preserved = {str(path):sha(path) for path in original_files} == originals_before
    source_cache_preserved = cache_inventory(source_cache) == source_cache_before
    helpers_preserved = {name:sha(ROOT/name) for name in FILES} == helpers_before
    if not all((originals_preserved, source_cache_preserved, helpers_preserved)):
        raise AssertionError('Reference data/checkpoints/frozen engine bytes changed')
    validations = [read(output/'native_fixed'/f'validation_epoch_{epoch:03d}.json') for epoch in range(3)]
    # Exercise the actual additive wrapper against a copied completed sealed
    # experiment. Its historical prepared-data path remains signed and intact.
    # Only mutable checkpoint/log copies are made; no source file is hardlinked.
    wrapper_root = output/'wrapper_complete'
    wrapper_root.mkdir(exist_ok=False)
    for name in ('experiment.json','initial.pt','calibration.json',
                 'calibration_selected.json','calibration_native.json',
                 'calibration_native_fixed.json','calibration_native_listwise.json'):
        shutil.copyfile(reference/name, wrapper_root/name)
    shutil.copytree(reference/'native_fixed', wrapper_root/'native_fixed')
    wrapper_checkpoint = wrapper_root/'native_fixed/checkpoint_latest.pt'
    wrapper_checkpoint_before = sha(wrapper_checkpoint)
    wrapper_content_before = torch.load(wrapper_checkpoint, map_location='cpu',
        weights_only=False)['content_sha256']
    wrapper_updates_before = (wrapper_root/'native_fixed/update_timing.jsonl').read_bytes()
    debug_fixture = Path(manifest['baseline']['source_samples'][0]['path']).parent
    debug_config = source.parent.parent/'report.json'
    if sha(debug_config) != manifest['baseline']['configuration_sha256']:
        raise ValueError('Explicit original DEBUG configuration path differs')
    del net, provider
    gc.collect(); torch.cuda.empty_cache()
    command = [sys.executable, '-B', '-u', str(ROOT/'tools/resume_comparison_cached.py'),
        '--gpu', str(arguments.gpu), '--arm', 'native_fixed', '--experiment', str(wrapper_root),
        '--cache-sources', str(source_cache), '--inventory', str(arguments.inventory.resolve()),
        '--debug-fixture', str(debug_fixture), '--debug-config', str(debug_config),
        '--debug-source', str(source), '--debug-bank', str(arguments.bank.resolve())]
    with (output/'wrapper_stdout.log').open('w', encoding='utf8') as stdout, \
            (output/'wrapper_stderr.log').open('w', encoding='utf8') as stderr:
        executed = subprocess.run(command, cwd=ROOT, stdout=stdout, stderr=stderr, check=False)
    if executed.returncode:
        raise RuntimeError(f'Actual CUDA wrapper failed({executed.returncode}); '
                           f'outputs preserved in {output}/wrapper_stderr.log')
    wrapper_content_after = torch.load(wrapper_checkpoint, map_location='cpu',
        weights_only=False)['content_sha256']
    if (wrapper_content_before != wrapper_content_after
            or (wrapper_root/'native_fixed/update_timing.jsonl').read_bytes() != wrapper_updates_before):
        raise AssertionError('Wrapper completed resume changed checkpoint content or added optimizer updates')
    if ({str(path):sha(path) for path in original_files} != originals_before
            or cache_inventory(source_cache) != source_cache_before):
        raise AssertionError('Wrapper changed original reference/cache bytes')
    report = dict(scope='actual_CT_CUDA_DEBUG_only', debug=True, actual_CUDA=True,
        quality_verified=False, production_training_started=False, full_production_evaluation=False,
        parameters=10434532, trainable_parameter_tensors=1085, physical_batch=2, workers=4, epochs=2,
        original129_joint_upper_evaluation=True, genuine_graph_views=2,
        result=result, completed_resume=resumed, optimizer_updates=2,
        completed_resume_additional_updates=0, completed_checkpoint_content_identical=True,
        completed_checkpoint_archive_bytes_identical=checkpoint_before == checkpoint_after,
        archive_byte_scope='Frozen engine reserializes completed checkpoints; semantic digest verifies neural/state content',
        original_reference_preserved=originals_preserved, original_cache_preserved=source_cache_preserved,
        frozen17_helpers_preserved=helpers_preserved, initial_neural_sha256=initial['model_sha256'],
        cache_provider=stats, reused_kinds=sorted(borrowed), actual_updates=updates,
        actual_wrapper_completed_resume=dict(returncode=0, additional_updates=0,
            copied_checkpoint_content_identical=True,
            copied_checkpoint_archive_bytes_identical=sha(wrapper_checkpoint) == wrapper_checkpoint_before,
            original_reference_cache_preserved=True,
            stdout='wrapper_stdout.log', stderr='wrapper_stderr.log'),
        validation=[dict(epoch=r['epoch'], candidate_universe=r['candidate_universe'],
                         metrics=r['metrics']) for r in validations],
        resources=resources, wall_seconds=time.perf_counter()-started)
    write_new(output/'report.json', canonical(report))
    print(json.dumps(dict(report=str(output/'report.json'), updates=2,
        completed_resume_additional_updates=0, reused_kinds=sorted(borrowed),
        originals_preserved=True, quality_verified=False), allow_nan=False), flush=True)
    return report


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--reference', type=Path, default=ROOT/'work/v19_checked_DEBUG')
    p.add_argument('--inventory', type=Path, default=ROOT/'work/v22_cumulative_u16_DEBUG_20261006/inventory.json')
    p.add_argument('--bank', type=Path, default=ROOT/'work/v1x_real_CT_DEBUG_20261003_prepare/fixture.pt')
    p.add_argument('--output', type=Path, required=True)
    return p.parse_args(argv)


if __name__ == '__main__':
    verify(parse())
