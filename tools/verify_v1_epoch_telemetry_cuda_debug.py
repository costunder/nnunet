"""Bounded actual-CT CUDA execution and numerical diagnostic of epoch_pass.

Original full model, physical two samples, eight candidates and two views are
retained. This is a one-batch DEBUG per path, never production training.
"""
from __future__ import annotations
import argparse
import ast
import copy
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)

def epoch_function(source, namespace):
    train = ast.parse(source).body[0]
    functions = [n for n in train.body if isinstance(n, ast.FunctionDef) and n.name == 'epoch_pass']
    if len(functions) != 1:
        raise ValueError('Expected exact original epoch_pass closure')
    tree = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0),
                            copy.deepcopy(functions[0])], type_ignores=[])
    ast.fix_missing_locations(tree)
    exec(compile(tree, '<actual-v1-epoch-pass-DEBUG>', 'exec'), namespace)
    return namespace['epoch_pass']

def child(args):
    source = Path(args.source).resolve()
    sys.path.insert(0, str(source))
    import torch
    from torch.utils.data import DataLoader
    from hiercp import pipeline
    from hiercp.data import HierarchicalCacheDataset
    from hiercp.model import HierarchicalPyGPlacementModel
    from hiercp.loss import CurriculumConfig, curriculum_ranking_loss, ranking_metric_sums
    from hiercp.tensor import (capture_rng_state, restore_rng_state, configure_runtime,
                              set_seed, process_memory_snapshot, cuda_memory_snapshot)
    from hiercp_v1x.sampling_runtime import install_from_environment
    from hiercp_v1x.epoch_telemetry import (
        EpochTelemetry, overlay_source, prepare_sampling_capture,
        timed_collate_samples, telemetry_worker_init,
    )
    assert Path(pipeline.__file__).resolve() == source/'hiercp/pipeline.py'
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1, 'Actual CUDA required'
    total = torch.cuda.get_device_properties(0).total_memory
    assert total > 12 * 2**30, 'This explicitly bounded DEBUG needs more than 12 GiB total VRAM'
    torch.cuda.set_per_process_memory_fraction(12 * 2**30 / total)
    torch.set_num_threads(2)
    install_from_environment()
    cfg = json.loads(Path(args.config).read_text())
    configure_runtime(deterministic=cfg['runtime']['deterministic'],
                      allow_tf32=cfg['runtime']['allow_tf32'],
                      cudnn_benchmark=cfg['runtime']['cudnn_benchmark'])
    if args.strict_debug_determinism:
        # Applies only to this DEBUG comparison. Original production runtime
        # deliberately remains byte-identical to the preserved v1 config.
        torch.use_deterministic_algorithms(True)
    set_seed(42, deterministic=cfg['runtime']['deterministic'])
    manifest = json.loads((Path(args.samples)/'fixture_manifest.json').read_text())
    files = [Path(args.samples)/r['name'] for r in manifest['files'] if r['split']=='train']
    assert len(files) == 2
    for row in manifest['files']:
        assert sha(Path(args.samples)/row['name']) == row['sha256']
    prepare_sampling_capture()
    ds = HierarchicalCacheDataset(files, training=True, seed=42, mmap=True)
    ds.set_epoch(29)
    loader = DataLoader(ds, batch_size=2, num_workers=2, multiprocessing_context='spawn',
                        persistent_workers=True, prefetch_factor=2, pin_memory=True,
                        collate_fn=timed_collate_samples, worker_init_fn=telemetry_worker_init,
                        generator=torch.Generator().manual_seed(42), shuffle=False)
    cpu = next(iter(loader))
    assert cpu.sample_count == 2 and cpu.counts == (8, 8) and cpu.local_batch_view2 is not None
    assert cpu.v1x_epoch_telemetry_sampling['sampling_measured_samples'] == 2
    original_source = inspect.getsource(pipeline.run_train)
    observed_source, identity = overlay_source(original_source)
    session = EpochTelemetry(Path(args.report).parent, {'debug': True, 'mode': args.mode}, identity)
    net = HierarchicalPyGPlacementModel(**cfg['model']).cuda()
    assert sum(p.numel() for p in net.parameters()) == 10434532
    initial = {k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
    initial_rng = capture_rng_state()
    training = cfg['training']
    curriculum = CurriculumConfig(**{k:v for k,v in training.items()
                                    if k in CurriculumConfig.__dataclass_fields__})
    reports, states, rngs, connectivity = {}, {}, {}, {}
    try:
        for label, text in (('original', original_source), ('original_repeat', original_source),
                            ('instrumented', observed_source)):
            net.load_state_dict(initial)
            optimizer = torch.optim.AdamW(net.parameters(), lr=training['lr'],
                                         weight_decay=training['weight_decay'],
                                         fused=training['fused_optimizer'])
            scaler = torch.amp.GradScaler('cuda', enabled=training['amp'])
            batch = copy.deepcopy(cpu).to('cuda', non_blocking=False)
            restore_rng_state(initial_rng)
            namespace = dict(pipeline.__dict__, torch=torch,
                             curriculum_ranking_loss=curriculum_ranking_loss,
                             ranking_metric_sums=ranking_metric_sums,
                             process_memory_snapshot=process_memory_snapshot,
                             cuda_memory_snapshot=cuda_memory_snapshot,
                             model=net, optimizer=optimizer, scaler=scaler,
                             device=torch.device('cuda:0'), use_amp=training['amp'],
                             accumulation_steps=1, curriculum=curriculum,
                             consistency_weight=training['consistency_weight'], training=training,
                             trainable_named_parameters=list(net.named_parameters()),
                             trainable_parameters=list(net.parameters()),
                             gradient_connected_parameter_names=set(), resource_report={'cpu_affinity_cores':2},
                             _v1x_epoch_telemetry=session)
            function = epoch_function(text, namespace)
            reports[label] = function([batch], 29, True)
            expected_names = {name for name,p in net.named_parameters() if p.requires_grad}
            connected = namespace['gradient_connected_parameter_names']
            assert connected == expected_names, sorted(expected_names - connected)
            connectivity[label] = dict(expected=len(expected_names), connected=len(connected), verified=True)
            states[label] = {k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
            rngs[label] = capture_rng_state()
            del optimizer, scaler, batch, namespace, function
            torch.cuda.empty_cache()
        metric_keys = ('loss','ranking','consistency','ce','pair','ordinal','mined','acc','mrr',
                       'positive','hardest_negative','margin','samples','batches','optimizer_steps')
        errors = {label:{k:abs(reports['original'][k]-reports[label][k]) for k in metric_keys}
                  for label in ('original_repeat','instrumented')}
        state_errors = {
            label:max((states['original'][k].float()-states[label][k].float()).abs().max().item()
                      for k in initial if initial[k].numel())
            for label in ('original_repeat','instrumented')}
        update_comparison = {}
        names = [name for name,p in net.named_parameters() if p.requires_grad]
        for label in ('original_repeat','instrumented'):
            dot = old_sq = new_sq = state_sq = difference_sq = 0.0
            for name in names:
                old = states['original'][name].double()
                new = states[label][name].double()
                before = initial[name].double()
                old_delta, new_delta = old-before, new-before
                dot += (old_delta*new_delta).sum().item()
                old_sq += old_delta.square().sum().item()
                new_sq += new_delta.square().sum().item()
                state_sq += old.square().sum().item()
                difference_sq += (old-new).square().sum().item()
            assert old_sq > 0 and new_sq > 0, 'Optimizer did not update the model'
            update_comparison[label] = dict(
                update_cosine=dot/(old_sq*new_sq)**0.5,
                relative_state_L2_error=(difference_sq/state_sq)**0.5,
                relative_update_L2_error=(difference_sq/old_sq)**0.5)
        # RNG payload includes Python/NumPy plus CPU/CUDA tensors. Compare values,
        # not pickle bytes whose storage identifiers need not be identical.
        def equal(a,b):
            if torch.is_tensor(a): return torch.equal(a,b)
            if isinstance(a,dict): return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
            if isinstance(a,(list,tuple)): return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
            if hasattr(a,'shape'): return bool((a==b).all())
            return a==b
        rng_equal = {label:equal(rngs['original'], rngs[label])
                     for label in ('original_repeat','instrumented')}
        write_new(Path(args.report).parent/'raw_comparison.json',
                  dict(debug=True, reports=reports, metric_errors=errors,
                       max_parameter_errors=state_errors, RNG_states_equal=rng_equal,
                       update_comparison=update_comparison, gradient_connectivity=connectivity,
                       strict_deterministic_algorithms_DEBUG_only=args.strict_debug_determinism,
                       production_runtime_changed=False, quality_verified=False))
        # Native v1 AMP includes nondeterministic 3D grid sampling/index_add.
        # Bit equality is consequently not asserted. Compare original replay
        # as a control and state an explicit, dtype-derived mechanical bound.
        # This bound is unrelated to ranking-quality noninferiority.
        epsilon = torch.finfo(torch.float16 if training['amp'] else torch.float32).eps
        exact_keys = ('acc','mrr','samples','batches','optimizer_steps')
        assert all(row[k] == 0 for row in errors.values() for k in exact_keys), errors
        assert all(error <= epsilon*max(1.0,abs(reports['original'][key]))
                   for row in errors.values() for key,error in row.items()), errors
        assert all(row['relative_state_L2_error'] <= epsilon
                   for row in update_comparison.values()), update_comparison
        strict_update_similarity = {
            label:row['update_cosine'] >= 1-epsilon
            for label,row in update_comparison.items()}
        assert all(rng_equal.values()), 'Observer/control changed RNG state'
        r = reports['instrumented']
        assert r['gpu_optimization_seconds'] > 0 and r['sampling_worker_seconds_sum'] > 0
        record = dict(status='COMPLETED_DIAGNOSTIC', debug=True, actual_CT=True, actual_CUDA=True,
                      mechanical_execution_verified=True,
                      numerical_parity_status='NOT_ESTABLISHED',
                      strict_update_similarity_passed=strict_update_similarity,
                      mode=args.mode, parameters=10434532, physical_sample_batch=2,
                      candidates_per_sample=8, views=2, workers=2,
                      updates_per_path=1, max_parameter_errors=state_errors, metric_errors=errors,
                      update_comparison=update_comparison, gradient_connectivity=connectivity,
                      bitwise_numerical_parity_verified=False,
                      mechanical_tolerance=dict(dtype_epsilon=epsilon,
                          metric_absolute_or_relative_bound='epsilon * max(1, abs(original))',
                          pooled_state_relative_L2_max=epsilon,
                          diagnostic_update_cosine_min=1-epsilon,
                          ranking_and_count_metrics_exact=True, ranking_quality_tolerance=None),
                      RNG_states_equal=rng_equal, reports=reports, telemetry_identity=identity,
                      strict_deterministic_algorithms_DEBUG_only=args.strict_debug_determinism,
                      production_runtime_changed=False, checkpoint_created=False,
                      telemetry_record=str(session.record_path), gpu=torch.cuda.get_device_name(0),
                      full_training=False, full_evaluation=False, quality_verified=False,
                      scope='Actual CT original/instrumented epoch_pass parity; not long-run ranking quality')
        write_new(args.report, record)
        print(f'{args.mode}: actual CT CUDA execution complete; numerical parity NOT_ESTABLISHED', flush=True)
    finally:
        session.close()

def parent(args):
    sys.path.insert(0, str(ROOT))
    from hiercp_v1x.experiment import initialize
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    old = ROOT/'work/v1_sampling_runtime416_CUDA_DEBUG_20261003'
    medical = json.loads((old/'native/manifest.json').read_text())['medical_root']
    native, nested = root/'native', root/'nested'
    initialize(native, medical, ROOT/'config/split_cp80_fold0.json', local_sampling='native', record_epochs=True)
    roles = ('tumor_surface','tumor_interior','source_context','source_liver_surface','target_context','target_liver_surface')
    initialize(nested, medical, ROOT/'config/split_cp80_fold0.json', local_sampling='strict_nested',
               sampling_profile=dict(zip(roles,(64,32,96,64,96,64))), reference_experiment=native, record_epochs=True)
    branches = {}
    for mode, suite in (('native',native), ('strict_nested',nested)):
        output = root/f'{mode}_DEBUG'
        output.mkdir()
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', HIERCP_V1X_SAMPLING_CONTRACT=str(suite/'sampling/v1.0.json'))
        if args.strict_debug_determinism:
            env['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        env.pop('PYTHONPATH',None)
        report = output/'report.json'
        command = [sys.executable,'-u',str(Path(__file__).resolve()),'--child',
                        '--source',str(suite/'source/v1.0'),'--config',str(suite/'configs/v1.0.json'),
                        '--samples',str(old/'samples'),'--report',str(report),'--mode',mode]
        if args.strict_debug_determinism:
            command.append('--strict-debug-determinism')
        subprocess.run(command, env=env, cwd=suite/'source/v1.0', check=True)
        branches[mode] = json.loads(report.read_text())
    write_new(root/'report.json', dict(status='COMPLETED_DIAGNOSTIC', debug=True, actual_CT=True, actual_CUDA=True,
                                     numerical_parity_status='NOT_ESTABLISHED',
                                     branches=branches, full_training=False, full_evaluation=False,
                                     quality_verified=False))
    print(f'DEBUG report: {root/"report.json"}', flush=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--child', action='store_true')
    parser.add_argument('--strict-debug-determinism', action='store_true',
                        help='Explicit DEBUG-only deterministic CUDA reduction comparison; never a production setting')
    for name in ('source','config','samples','report','mode','output'):
        parser.add_argument('--'+name)
    args = parser.parse_args()
    return child(args) if args.child else parent(args)

if __name__ == '__main__':
    main()
