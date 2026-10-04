"""Actual-CUDA DEBUG learning of a new L0 CNN/role bridge on preserved v1.

The frozen v1.4 replay supplies the 10mm fixture protocol and comparison curve.
Only the first half changes; native GT, full eight candidates, 128-candidate
pool, two training views, L1/L2/score, complete loss and forty production epochs
remain intact. This tool writes audit states and diagnostic evidence only.
"""
from __future__ import annotations

import argparse
import copy
from concurrent.futures import ThreadPoolExecutor
import dataclasses
import hashlib
import json
import math
from pathlib import Path
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

from hiercp_v1x import bounded_scope
from hiercp_v1x.contracts import canonical_hash
from hiercp_v1x.scope_learning_inputs import rebuild_scope, supervision_digest
from hiercp_v1x.scope_probe_support import activate_original, state_digest, _sha
from tools.verify_v14_learning_replay import verify_reference, write_new


def verify_bounded_reference(report, contract, native, native_contract, manifest):
    """Bind the recorded bounded curve to actual native GT/config/fixture."""
    protocol = verify_reference(native, native_contract, manifest)
    if report.get('status') != 'COMPLETE':
        raise ValueError('Completed bounded reference required')
    for name in ('debug', 'actual_CT', 'actual_CUDA', 'finite_complete_updates',
                 'original_fixture_preserved', 'original_reference_preserved',
                 'original_source_preserved'):
        if report.get(name) is not True:
            raise ValueError(f'Bounded reference requires {name}=true')
    for name in ('full_training', 'full_evaluation', 'quality_verified',
                 'production_ready', 'checkpoint_written', 'native_retrained'):
        if report.get(name) is not False:
            raise ValueError(f'Bounded DEBUG reference cannot claim {name}')
    if (report.get('margin_mm') != 10.0 or contract.get('arguments', {}).get('margin_mm') != 10.0
            or report.get('configuration') != manifest['config']
            or report.get('fixture_sha256') != manifest['fixture_sha256']
            or report.get('initial_neural_sha256') != protocol['initial_neural_sha256']
            or report.get('protocol') != protocol or contract.get('protocol') != protocol):
        raise ValueError('Bounded scope, original configuration or initialization differs')
    if (contract.get('debug') is not True or contract.get('quality_verified') is not False
            or canonical_hash({k:v for k,v in contract.items() if k != 'identity_sha256'})
                != contract.get('identity_sha256')):
        raise ValueError('Bounded execution contract identity differs')
    receipt = report.get('bounded_scope', {})
    if (receipt.get('margin_mm') != 10.0 or receipt.get('context_outer_radius_mm') != 10.0
            or receipt.get('CNN_shape') != [48, 48, 48]
            or receipt.get('debug') is not True or receipt.get('quality_verified') is not False):
        raise ValueError('Recorded 10mm/full-footprint/48-cubed scope required')
    updates = report.get('updates', [])
    if (len(updates) != report.get('completed_optimizer_updates')
            or len(updates) != contract['arguments'].get('updates') or len(updates) < 2):
        raise ValueError('Completed bounded update count differs')
    for step, row in enumerate(updates, 1):
        if (row.get('step') != step or not math.isfinite(row['loss'])
                or row.get('gradient_observation', {}).get('finite') is not True):
            raise ValueError('Bounded reference update evidence is incomplete/nonfinite')
    curves = report.get('curve', [])
    steps = [row.get('step') for row in curves]
    if steps != sorted(set(steps)) or not {0, 2, len(updates)}.issubset(steps):
        raise ValueError('Initial/matched/final bounded reference curve required')
    for row in curves:
        for split, count in (('train', len(protocol['train_cases'])),
                             ('validation', len(protocol['validation_cases']))):
            metric = row[split]
            if (len(metric['scores']) != count or any(len(scores) != 8 for scores in metric['scores'])
                    or not all(math.isfinite(value) for scores in metric['scores'] for value in scores)
                    or len(metric['positive_minus_best_other']) != count
                    or not math.isfinite(metric['mean_margin'])
                    or not all(math.isfinite(metric[key]) and 0 <= metric[key] <= 1
                               for key in ('MRR', 'top1'))):
                raise ValueError('Bounded ranking evidence has wrong cohort/nonfinite values')
    return protocol


def compare_curves(curve, baseline):
    """Compare only the same number of successful updates, never final8 vs16."""
    reference = {row['step']: row for row in baseline}
    result = []
    for row in curve:
        if row['step'] not in reference:
            continue
        old = reference[row['step']]
        result.append(dict(step=row['step'], half_a={s:row[s] for s in ('train', 'validation')},
            bounded_v14={s:old[s] for s in ('train', 'validation')},
            difference={split:{key:row[split][key] - old[split][key]
                for key in ('MRR', 'top1', 'mean_margin')} for split in ('train', 'validation')},
            loss_comparison_available=False))
    return result


def module_groups(net):
    return dict(CNN=net.local_encoder.cnn, bridge_projection=net.local_encoder.project,
        bridge_fusion=net.local_encoder.fuse, local_encoder_total=net.local_encoder,
        L1=net.patient_encoder, L1_readout=net.patient_readout,
        L2=net.prototype_encoder, L2_readout=net.population_readout, scalar_score=net.score_head)


def half_a_state_digest(state):
    """Hash tensor bytes and explicit extra-state contracts, without changing v1."""
    import torch
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode())
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().contiguous()
            digest.update(str((str(value.dtype), tuple(value.shape))).encode())
            digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif name.endswith('_extra_state') and isinstance(value, dict):
            digest.update(b'canonical_extra_state_dict')
            digest.update(canonical_hash(value).encode())
        else:
            raise TypeError(f'Unrecognized Half-A state entry {name}: {type(value)}')
    return digest.hexdigest()


def same_rng(left, right):
    import numpy as np
    import torch
    return (left['python'] == right['python']
        and left['numpy'][0] == right['numpy'][0]
        and np.array_equal(left['numpy'][1], right['numpy'][1])
        and left['numpy'][2:] == right['numpy'][2:]
        and torch.equal(left['torch_cpu'], right['torch_cpu'])
        and len(left['torch_cuda']) == len(right['torch_cuda'])
        and all(torch.equal(a, b) for a, b in zip(left['torch_cuda'], right['torch_cuda'])))


def gradient_observation(net, groups):
    """One compact host copy of finite/group gradient diagnostic reductions."""
    import torch
    names, norms = [], []
    for name, module in groups.items():
        values = [p.grad.detach().double().square().sum() for p in module.parameters()
                  if p.requires_grad and p.grad is not None]
        if not values:
            raise AssertionError(f'Complete original loss does not reach {name}')
        names.append(name); norms.append(torch.stack(values).sum().sqrt())
    present = [p.grad for p in net.parameters() if p.grad is not None]
    if not present:
        raise AssertionError('No trainable gradients')
    packed = torch.stack([torch.isfinite(g).all() for g in present] + norms).cpu().tolist()
    finite = all(packed[:len(present)])
    raw = dict(zip(names, packed[len(present):]))
    return dict(finite=finite,
        norms={name:value if math.isfinite(value) else str(value) for name,value in raw.items()},
        zero_groups=[name for name,value in raw.items() if value == 0],
        nonfinite_groups=[name for name,value in raw.items() if not math.isfinite(value)],
        norm_reduction_dtype='float64')


def parameter_changes(groups, before):
    """Measure real optimizer changes without per-parameter host synchronization."""
    import torch
    names, values = [], []
    for name, module in groups.items():
        parameters = [p for p in module.parameters() if p.requires_grad]
        if not parameters:
            raise AssertionError(f'Expected trainable parameter group {name}')
        squared = torch.stack([(p.detach().double() - before[id(p)].double()).square().sum()
                               for p in parameters]).sum().sqrt()
        changed = torch.stack([torch.ne(p.detach(), before[id(p)]).any().double()
                               for p in parameters]).sum()
        names.append(name); values.append(torch.stack([squared, changed]))
    packed = torch.stack(values).cpu().tolist()
    return {name:dict(L2_change=row[0], changed_parameter_tensors=int(row[1]),
                     parameter_tensors=sum(p.requires_grad for p in groups[name].parameters()))
            for name,row in zip(names, packed)}


def evaluate(net, batch, config, objective):
    import torch
    from hiercp.loss import curriculum_ranking_loss, ranking_metrics
    from hiercp.tensor import capture_rng_state, restore_rng_state
    saved_rng, previous_mode = capture_rng_state(), net.training
    try:
        net.eval()
        with torch.no_grad(), torch.autocast('cuda', enabled=config['training']['amp']):
            out = net(batch)
            base, terms = curriculum_ranking_loss(out.scores, batch.difficulty_list(),
                epoch=29, config=objective)
            loss = base + config['training']['consistency_weight'] * out.consistency
            top1, mrr = ranking_metrics(out.scores)
            scores = torch.stack([value.detach().float() for value in out.scores]).cpu().tolist()
            values = torch.stack([loss.detach(), base.detach(), out.consistency.detach(),
                                  *[value.detach() for value in terms.values()]]).double().cpu().tolist()
        if any(len(row) != 8 for row in scores) or not all(math.isfinite(x) for row in scores for x in row):
            raise AssertionError('Nonfinite/wrong-eight-candidate ranking output')
        if not all(math.isfinite(value) for value in values):
            raise AssertionError('Nonfinite complete original evaluation loss')
        margins = [row[0] - max(row[1:]) for row in scores]
        return dict(top1=top1, MRR=mrr, positive_minus_best_other=margins,
            scores=scores, mean_margin=sum(margins) / len(margins),
            loss=values[0], ranking_loss=values[1], consistency=values[2],
            loss_terms=dict(zip(terms, values[3:])), loss_epoch=29,
            candidate_graphs=sum(map(len, scores)))
    finally:
        net.train(previous_mode)
        restore_rng_state(saved_rng)


def execute(a, root, baseline, manifest, protocol, inputs):
    from tools.local_cnn_device import select
    select(a.gpu)
    source = a.source.resolve(strict=True)
    proof = activate_original(source)
    import psutil
    import torch
    from hiercp.data import collate_samples, summarize_hierarchical_batch
    from hiercp.loss import CurriculumConfig, curriculum_ranking_loss
    from hiercp.model import HierarchicalPyGPlacementModel
    from hiercp.sample import materialize_sample_views
    from hiercp.tensor import capture_rng_state, configure_runtime, set_seed
    from hiercp_v1x.half_a_model import MARKER_KEY, half_a_identity, model_contract, install_half_a
    from tqdm import tqdm

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Exactly one actual CUDA GPU required; no CPU training fallback')
    total = torch.cuda.get_device_properties(0).total_memory
    limit = int(a.cuda_gib * 2**30)
    if not 0 < limit < total:
        raise ValueError('Explicit CUDA budget must leave physical device headroom')
    torch.cuda.set_per_process_memory_fraction(limit / total)
    torch.set_num_threads(protocol['workers'])
    started = time.perf_counter()
    process = psutil.Process()
    ram = psutil.virtual_memory()
    gpu_free, _ = torch.cuda.mem_get_info()
    peak_rss = [process.memory_info().rss]
    stop = threading.Event()
    def monitor():
        while not stop.wait(.1):
            peak_rss[0] = max(peak_rss[0], process.memory_info().rss)
    watcher = threading.Thread(target=monitor, daemon=True); watcher.start()
    def budget():
        peak_rss[0] = max(peak_rss[0], process.memory_info().rss)
        if peak_rss[0] > int(a.rss_gib * 2**30):
            raise MemoryError('Explicit RSS budget exceeded; full model/cohort preserved')
        if time.perf_counter() - started > a.wall_seconds:
            raise TimeoutError('Explicit short DEBUG wall budget exceeded; no quality PASS')
    try:
        config = manifest['config']
        fixture = torch.load(a.fixture, map_location='cpu', weights_only=False, mmap=True)
        if (fixture['config'] != config or fixture['train_cases'] != protocol['train_cases']
                or fixture['validation_cases'] != protocol['validation_cases']):
            raise ValueError('Exact real fixture configuration/cohort differs')
        samples = fixture['samples']
        if [(s['case_id'], s['split']) for s in samples] != (
                [(c, 'train') for c in protocol['train_cases']]
                + [(c, 'val') for c in protocol['validation_cases']]):
            raise ValueError('Original sample order/split differs')
        for row in manifest['source_records']:
            for kind in ('image', 'label'):
                if _sha(Path(row[kind])) != row[kind + '_sha256']:
                    raise ValueError('Actual raw CT/annotation differs')
                inputs[str(Path(row[kind]).resolve())] = row[kind + '_sha256']
        before_supervision = supervision_digest(samples)
        if before_supervision != baseline['supervision_sha256']:
            raise ValueError('Candidate/GT/L1/L2 input differs from recorded bounded baseline')
        receipt = bounded_scope.install(10.0, source)
        if receipt != baseline['bounded_scope']:
            raise ValueError('The frozen bounded10mm input adapter differs')
        bounded, preparation = rebuild_scope(samples, manifest, config, bounded_scope,
            10.0, protocol['workers'], budget)
        if supervision_digest(bounded) != before_supervision:
            raise AssertionError('10mm reconstruction altered original candidates/GT/L1/L2')
        def reset():
            runtime = config['runtime']
            set_seed(config['seed'], deterministic=runtime['deterministic'])
            configure_runtime(deterministic=runtime['deterministic'], allow_tf32=runtime['allow_tf32'],
                cudnn_benchmark=runtime['cudnn_benchmark'])
        reset()
        began = time.perf_counter()
        with ThreadPoolExecutor(max_workers=protocol['workers']) as pool:
            views = list(pool.map(lambda sample: materialize_sample_views(copy.deepcopy(sample),
                training=sample['split'] == 'train', epoch=29, global_seed=config['seed']), bounded))
        view_seconds = time.perf_counter() - began
        training = [sample for sample in views if sample['split'] == 'train']
        validation = [sample for sample in views if sample['split'] == 'val']
        reset()
        net = HierarchicalPyGPlacementModel(**config['model'])
        native_identity = state_digest(net.state_dict())
        if native_identity != protocol['initial_neural_sha256']:
            raise AssertionError('Preserved seed42 native initialization differs')
        old_state = {name:value.detach().clone() for name,value in net.state_dict().items()}
        old_rng = capture_rng_state()
        upper_before = {name:value for name,value in old_state.items() if not name.startswith('local_encoder.')}
        upper_identity = state_digest(upper_before)
        net = install_half_a(net)
        installed_rng = capture_rng_state()
        if not same_rng(old_rng, installed_rng):
            raise AssertionError('Installing Half-A changed the preserved caller RNG streams')
        upper_after = {name:value for name,value in net.state_dict().items()
                       if not name.startswith('local_encoder.') and name != MARKER_KEY}
        if state_digest(upper_after) != upper_identity:
            raise AssertionError('Replacing first half altered original upper state')
        half_initial_identity = half_a_state_digest(net.state_dict())
        with (root / 'initialization_audit_DEBUG.pt').open('xb') as stream:
            torch.save(dict(debug=True, full_training=False, production_ready=False,
                purpose='initialization/RNG audit; untrained state; no optimizer or production checkpoint',
                native_initial_state=old_state, native_constructed_rng=old_rng,
                half_a_initial_state=net.state_dict(), half_a_constructed_rng=installed_rng), stream)
        del old_state, old_rng, upper_before, upper_after
        net.cuda()
        groups = module_groups(net)
        cpu = collate_samples(training).pin_memory()
        shape = summarize_hierarchical_batch(cpu)
        if cpu.local_batch_view2 is None:
            raise AssertionError('Original training two-view consistency path is missing')
        batch = cpu.to('cuda', non_blocking=True); del cpu
        held = collate_samples(validation).to('cuda')
        torch.cuda.synchronize()
        fields = {field.name for field in dataclasses.fields(CurriculumConfig)}
        objective = CurriculumConfig(**{key:value for key,value in config['training'].items() if key in fields})
        objective.validate()
        parameters = [p for p in net.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=config['training']['lr'],
            weight_decay=config['training']['weight_decay'], fused=config['training']['fused_optimizer'])
        if {id(p) for group in optimizer.param_groups for p in group['params']} != {id(p) for p in parameters}:
            raise AssertionError('Fresh original optimizer omits trainable parameters')
        scaler = torch.amp.GradScaler('cuda', enabled=config['training']['amp'])
        result = dict(debug=True, smoke_test=True, actual_CT=True, actual_CUDA=True, status='RUNNING',
            full_training=False, full_evaluation=False, quality_verified=False, training_quality_verified=False,
            production_ready=False, production_checkpoint_written=False, diagnostic_initial_state_written=True,
            protocol=protocol, margin_mm=10.0, bounded_scope=receipt,
            original_configuration=config, configuration_sha256=canonical_hash(config),
            experiment='Half-A: CNN L0 and real semantic bridge with preserved v1 upper model',
            half_a=half_a_identity(), half_a_model_contract=model_contract(),
            source_proof=proof, original_neural_sha256=native_identity,
            original_upper_initial_state_sha256=upper_identity, upper_initial_state_preserved=True,
            caller_RNG_preserved_during_install=True,
            half_a_initial_neural_sha256=half_initial_identity, supervision_sha256=before_supervision,
            model_parameters=sum(p.numel() for p in net.parameters()),
            trainable_parameters=sum(p.numel() for p in parameters),
            group_parameters={name:sum(p.numel() for p in module.parameters()) for name,module in groups.items()},
            bridge_pooling='parameter-free actual role/shell means; shared learned projection and fusion',
            bridge_has_independent_trainable_parameters=False,
            input=shape, training_two_views=True, held_out_view_epoch=0,
            physical_sample_batch=len(training), physical_candidate_graph_batch=len(training)*8,
            gradient_accumulation_steps=1, data_parallel_workers=1, effective_sample_batch=len(training),
            full_fixture_samples=len(samples), used_fixture_samples=len(views), used_fixture_ratio=len(views)/len(samples),
            dataset_scope='selected actual-CT DEBUG fixture; not full production cohort',
            candidates_per_sample=8, candidate_pool=128, production_epochs_preserved=40,
            fixed_optimization_curriculum_epoch=29, workers=protocol['workers'], cached_GPU_batch=True,
            GPU=torch.cuda.get_device_name(0), torch_version=torch.__version__, cuda_build=torch.version.cuda,
            device_total_bytes=total, cuda_limit_bytes=limit, precision=dict(amp=config['training']['amp']),
            resources_at_start=dict(cpu_physical=psutil.cpu_count(logical=False), cpu_logical=psutil.cpu_count(),
                ram_total_bytes=ram.total, ram_available_bytes=ram.available, GPU_free_bytes=gpu_free,
                rss_limit_bytes=int(a.rss_gib*2**30), disk_free_bytes=psutil.disk_usage(str(root)).free),
            preparation=preparation, bounded_cache_reused=False,
            bounded_cache_unavailable_reason='The recorded v1.4 run did not serialize its bounded canonical samples.',
            sampled_view_preparation_seconds=view_seconds, attempts=[], updates=[], curve=[],
            input_sha256=inputs, fixture_sha256=_sha(a.fixture), reference_report_sha256=_sha(a.reference),
            current_sources_sha256={name:_sha(ROOT/name) for name in (
                'tools/verify_v1_half_a_learning.py', 'tools/verify_v14_learning_replay.py',
                'hiercp_v1x/half_a_model.py', 'hiercp_v1x/bounded_scope.py',
                'hiercp_v1x/scope_learning_inputs.py', 'hiercp_v1x/scope_probe_support.py')},
            optimizer_fresh=True, exact_resume_claimed=False, native_retrained=False,
            bitwise_native_reproduction_claimed=False, baseline_retrained=False,
            timings_are_shared_GPU_debug_compute_only=True)
        print(json.dumps({key:result[key] for key in ('experiment', 'debug', 'model_parameters',
            'trainable_parameters', 'group_parameters', 'input', 'physical_sample_batch',
            'physical_candidate_graph_batch', 'effective_sample_batch', 'production_epochs_preserved',
            'fixed_optimization_curriculum_epoch', 'workers', 'precision', 'GPU')}, allow_nan=False), flush=True)
        def append(name, row):
            result[name].append(row)
            with (root / f'{name}.jsonl').open('a', encoding='utf8') as stream:
                stream.write(json.dumps(row, allow_nan=False) + '\n')
        def evaluate_step(step):
            budget()
            row = dict(step=step, train=evaluate(net, batch, config, objective),
                       validation=evaluate(net, held, config, objective))
            append('curve', row)
            print(f"Half-A fixed-view DEBUG step{step} | train loss={row['train']['loss']:.5f} "
                  f"MRR={row['train']['MRR']:.4f} top1={row['train']['top1']:.4f} "
                  f"margin={row['train']['mean_margin']:.5f} | held-out loss={row['validation']['loss']:.5f} "
                  f"MRR={row['validation']['MRR']:.4f} top1={row['validation']['top1']:.4f} "
                  f"margin={row['validation']['mean_margin']:.5f}", flush=True)
        initial_parameters = {id(p):p.detach().clone() for p in parameters}
        evaluate_step(0)
        reset()  # Original reference reset dropout streams before update1.
        training_rng_before = capture_rng_state()
        connected = set()
        executed = dict(CNN_calls=0, CNN_ROIs=0, scalar_score_calls=0, scalar_score_rows=0,
                        L1_block_calls=0, L2_block_calls=0)
        def cnn_hook(module, args, output):
            executed['CNN_calls'] += 1; executed['CNN_ROIs'] += int(args[0].shape[0])
        def score_hook(module, args, output):
            executed['scalar_score_calls'] += 1; executed['scalar_score_rows'] += int(output.shape[0])
        def block_hook(group):
            def hook(module, args, output):
                executed[group] += 1
            return hook
        handles = [net.local_encoder.cnn.register_forward_hook(cnn_hook),
                   net.score_head.register_forward_hook(score_hook)]
        handles.extend(block.register_forward_hook(block_hook('L1_block_calls')) for block in net.patient_encoder.blocks)
        handles.extend(block.register_forward_hook(block_hook('L2_block_calls')) for block in net.prototype_encoder.blocks)
        observed = next(net.score_head.parameters())
        def optimizer_counter():
            value = optimizer.state.get(observed, {}).get('step', 0)
            return int(value.item()) if isinstance(value, torch.Tensor) else int(value)
        bar = tqdm(total=a.updates, desc='Half-A actual CUDA DEBUG', unit='update')
        step, attempt = 0, 0
        try:
            while step < a.updates:
                attempt += 1; budget()
                net.train(); optimizer.zero_grad(set_to_none=True)
                torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
                began = time.perf_counter()
                counts_before = dict(executed)
                with torch.autocast('cuda', enabled=config['training']['amp']):
                    out = net(batch)
                    base, terms = curriculum_ranking_loss(out.scores, batch.difficulty_list(), epoch=29, config=objective)
                    loss = base + config['training']['consistency_weight'] * out.consistency
                if not bool(torch.isfinite(loss)):
                    append('attempts', dict(attempt=attempt, completed_updates=step,
                        status='NONFINITE_FORWARD_LOSS', loss=str(float(loss.detach()))))
                    raise AssertionError('Nonfinite original complete loss; optimizer was not called')
                torch.cuda.synchronize(); forward = time.perf_counter() - began
                back = time.perf_counter(); scale_before = scaler.get_scale()
                scaler.scale(loss).backward(); scaler.unscale_(optimizer); torch.cuda.synchronize()
                backward = time.perf_counter() - back
                observation = gradient_observation(net, groups)
                counter_before = optimizer_counter()
                if not observation['finite']:
                    if not scaler.is_enabled():
                        append('attempts', dict(attempt=attempt, completed_updates=step,
                            status='NONFINITE_GRADIENT_AMP_DISABLED', gradient=observation))
                        raise AssertionError('Nonfinite gradient with original AMP disabled')
                    scaler.step(optimizer); scaler.update(); torch.cuda.synchronize()
                    unchanged = optimizer_counter() == counter_before
                    append('attempts', dict(attempt=attempt, completed_updates=step,
                        status='AMP_OVERFLOW_SKIPPED', gradient=observation, loss=float(loss.detach()),
                        scale_before=scale_before, scale_after=scaler.get_scale(), optimizer_step_unchanged=unchanged))
                    if not unchanged or not scaler.get_scale() < scale_before:
                        raise AssertionError('AMP overflow did not skip/update its scale correctly')
                    print(f'AMP overflow attempt{attempt}: completed={step}, optimizer skipped, '
                          f'scale {scale_before:g}->{scaler.get_scale():g}', flush=True)
                    del out, loss, base, terms
                    continue
                if observation['nonfinite_groups']:
                    raise AssertionError('Finite gradients produced nonfinite float64 diagnostic norm')
                clip = torch.nn.utils.clip_grad_norm_(parameters, config['training']['grad_clip'], error_if_nonfinite=True)
                before_update = {id(p):p.detach().clone() for p in parameters}
                opt = time.perf_counter()
                scaler.step(optimizer); scaler.update(); torch.cuda.synchronize()
                optimizer_seconds = time.perf_counter() - opt
                if optimizer_counter() != counter_before + 1:
                    raise AssertionError('Finite attempt did not complete exactly one optimizer update')
                changes = parameter_changes(groups, before_update); del before_update
                connected.update(name for name,p in net.named_parameters() if p.requires_grad and p.grad is not None)
                step += 1
                row = dict(step=step, attempt=attempt, loss=float(loss.detach()),
                    loss_terms={key:float(value.detach()) for key,value in terms.items()},
                    consistency=float(out.consistency.detach()), forward_seconds=forward,
                    backward_seconds=backward, optimizer_seconds=optimizer_seconds,
                    update_seconds=time.perf_counter()-began,
                    gradient_groups=observation['norms'], gradient_observation=observation,
                    actual_parameter_updates=changes, gradient_before_clip=float(clip),
                    amp_scale_before=scale_before, amp_scale_after=scaler.get_scale(),
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(), RSS_bytes=process.memory_info().rss,
                    sample_scores=len(out.scores), candidate_scores=sum(score.numel() for score in out.scores))
                row['neural_execution'] = {name:value-counts_before[name] for name,value in executed.items()}
                row['bridge_embeddings'] = {name:list(value.shape) for name,value in out.local_embeddings.items()}
                row['bridge_view2_embeddings'] = {name:list(value.shape) for name,value in out.local_embeddings_view2.items()}
                row['patient_candidate_nodes'] = int(out.patient_batch['candidate'].num_nodes)
                append('attempts', dict(attempt=attempt, completed_updates=step,
                    status='OPTIMIZER_UPDATED', optimizer_counter=optimizer_counter(),
                    scale_before=scale_before, scale_after=scaler.get_scale(), gradient=observation))
                append('updates', row)
                del out, loss, base, terms, clip
                bar.update(1)
                if step == 2 or step % 4 == 0 or step == a.updates:
                    evaluate_step(step)
                budget()
        finally:
            bar.close()
            for handle in handles:
                handle.remove()
        final_changes = parameter_changes(groups, initial_parameters); del initial_parameters
        expected = {name for name,p in net.named_parameters() if p.requires_grad}
        missing = sorted(expected - connected)
        result.update(status='COMPLETE', completed_optimizer_updates=step,
            attempted_backward_passes=attempt,
            amp_overflow_skipped_attempts=sum(row['status']=='AMP_OVERFLOW_SKIPPED' for row in result['attempts']),
            finite_complete_updates=True, missing_gradients=missing,
            connected_trainable_parameter_tensors=len(connected), expected_trainable_parameter_tensors=len(expected),
            final_parameter_changes=final_changes,
            every_core_group_updated=all(row['changed_parameter_tensors'] > 0 for row in final_changes.values()),
            every_core_group_had_nonzero_gradient=all(any(row['gradient_groups'][name] > 0
                for row in result['updates']) for name in groups),
            finite_zero_gradient_updates=[dict(step=row['step'], groups=row['gradient_observation']['zero_groups'])
                for row in result['updates'] if row['gradient_observation']['zero_groups']],
            final_neural_sha256=half_a_state_digest(net.state_dict()),
            initial_train=result['curve'][0]['train'], initial_held_out=result['curve'][0]['validation'],
            final_train=result['curve'][-1]['train'], final_held_out=result['curve'][-1]['validation'],
            matched_update_comparisons=compare_curves(result['curve'], baseline['curve']),
            reference_full_debug_curve=baseline['curve'],
            baseline_total_updates=len(baseline['updates']), comparison_final_step_matches=a.updates,
            process_peak_rss_bytes=peak_rss[0], total_wall_seconds=time.perf_counter()-started,
            GPU_peak_allocated_bytes=max(row['peak_allocated_bytes'] for row in result['updates']),
            GPU_peak_reserved_bytes=max(row['peak_reserved_bytes'] for row in result['updates']),
            samples_per_update=len(training), candidate_graphs_per_update=len(training)*8,
            update_compute_samples_per_second=len(training)*step/sum(row['update_seconds'] for row in result['updates']),
            original_inputs_preserved=all(_sha(Path(path)) == identity for path,identity in inputs.items()),
            original_source_preserved=all(_sha(source/name) == identity for name,identity in proof['verified_files'].items()),
            original_archive_preserved=_sha(ROOT/'versions/v1/pipeline_v1_source.zip') == proof['archive_sha256'],
            experiment_sources_preserved=all(_sha(ROOT/name) == identity
                for name,identity in result['current_sources_sha256'].items()),
            CNN_dependencies_preserved=all(_sha(ROOT/name) == identity
                for name,identity in result['half_a']['source_sha256'].items()))
        with (root / 'training_rng_audit_DEBUG.pt').open('xb') as stream:
            torch.save(dict(debug=True, production_ready=False, training_rng_before=training_rng_before,
                            training_rng_after=capture_rng_state(), completed_optimizer_updates=step), stream)
        if missing or not all(result[key] for key in ('every_core_group_updated',
                'every_core_group_had_nonzero_gradient', 'original_inputs_preserved', 'original_source_preserved',
                'original_archive_preserved', 'experiment_sources_preserved', 'CNN_dependencies_preserved')):
            result['status'] = 'INCOMPLETE'
            write_new(root / 'incomplete_report.json', result)
            raise AssertionError('Gradient/update connectivity or original input/source preservation failed')
        write_new(root / 'report.json', result)
        return result
    finally:
        stop.set(); watcher.join()


def recorded_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT/path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--reference', type=Path, required=True, help='Completed bounded v1.4 replay report')
    parser.add_argument('--fixture', type=Path, required=True, help='Exact packed native real-CT fixture reused by v1.4')
    parser.add_argument('--source', type=Path, help='Read-only verified original v1 archive snapshot; defaults to reference source/v1.0')
    parser.add_argument('--updates', type=int, required=True, help='Explicit successful DEBUG updates; production epochs remain40')
    parser.add_argument('--cuda-gib', type=float, required=True)
    parser.add_argument('--rss-gib', type=float, required=True)
    parser.add_argument('--wall-seconds', type=float, required=True)
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args()
    if (a.gpu < 0 or a.updates < 2
            or not all(math.isfinite(v) and v > 0 for v in (a.cuda_gib, a.rss_gib, a.wall_seconds))):
        parser.error('Explicit >=2 successful DEBUG updates and finite resource budgets required')
    baseline = json.loads(a.reference.read_text(encoding='utf8'))
    baseline_contract_path = a.reference.with_name('execution_contract.json')
    baseline_contract = json.loads(baseline_contract_path.read_text(encoding='utf8'))
    native_path = recorded_path(baseline_contract['arguments']['reference'])
    native_contract_path = native_path.with_name('execution_contract.json')
    manifest_path = a.fixture.with_name('fixture_manifest.json')
    native = json.loads(native_path.read_text(encoding='utf8'))
    native_contract = json.loads(native_contract_path.read_text(encoding='utf8'))
    manifest = json.loads(manifest_path.read_text(encoding='utf8'))
    protocol = verify_bounded_reference(baseline, baseline_contract, native, native_contract, manifest)
    if (_sha(a.fixture) != manifest['fixture_sha256'] or _sha(native_path) != baseline['reference_report_sha256']
            or a.updates not in {row['step'] for row in baseline['curve']}):
        raise ValueError('Exact preserved fixture/reference and matching recorded DEBUG update point required')
    for name, identity in baseline['current_sources_sha256'].items():
        if _sha(ROOT/name) != identity:
            raise ValueError(f'Frozen input/replay source differs from bounded baseline: {name}')
    paths = (a.reference, baseline_contract_path, native_path, native_contract_path, a.fixture, manifest_path)
    inputs = {str(path.resolve()):_sha(path) for path in paths}
    if any(inputs.get(path) != identity for path,identity in baseline_contract['inputs'].items()):
        raise ValueError('Recorded original reference/fixture metadata changed')
    a.source = a.source or a.reference.parent/'source/v1.0'
    root = a.output.resolve()
    if root.exists():
        raise FileExistsError('Existing evidence is preserved; choose a fresh DEBUG output')
    if any(root.is_relative_to(path.resolve().parent) for path in paths) or root.is_relative_to(a.source.resolve()):
        raise ValueError('New result directory must be disjoint from all preserved inputs/source')
    root.mkdir(parents=True)
    provenance = dict(debug=True, smoke_test=True, full_training=False, full_evaluation=False,
        production_ready=False, quality_verified=False, actual_CT_requested=True, actual_CUDA_requested=True,
        protocol=protocol, arguments={key:str(value) if isinstance(value, Path) else value for key,value in vars(a).items()},
        input_sha256=inputs, original_configuration_sha256=canonical_hash(manifest['config']),
        intended_change='replace only L0 neural first half; preserve original upper model, GT and complete loss')
    provenance['identity_sha256'] = canonical_hash(provenance)
    write_new(root/'execution_contract.json', provenance)
    try:
        result = execute(a, root, baseline, manifest, protocol, inputs)
    except Exception:
        write_new(root/'failed.json', dict(debug=True, smoke_test=True, full_training=False,
            full_evaluation=False, production_ready=False, quality_verified=False, error=traceback.format_exc()))
        raise
    print(f"REPORT: {root/'report.json'}", flush=True)
    print(json.dumps(dict(completed_optimizer_updates=result['completed_optimizer_updates'],
        AMP_skipped_attempts=result['amp_overflow_skipped_attempts'],
        initial_train=result['initial_train']['MRR'], final_train=result['final_train']['MRR'],
        initial_held_out=result['initial_held_out']['MRR'], final_held_out=result['final_held_out']['MRR'],
        quality_verified=False), allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
