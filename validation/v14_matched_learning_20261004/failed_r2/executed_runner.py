"""Short CUDA learning replay of v1.4, reusing a completed native reference.

Only the bounded arm is executed. Its first reference-count updates reproduce
the recorded fixed-view/epoch29 protocol; later explicit DEBUG updates test
continued learning on that same minibatch. No full-cohort quality approval,
production checkpoint, native retraining or nnU-Net launch is produced.
"""
from __future__ import annotations

import argparse
import copy
from concurrent.futures import ThreadPoolExecutor
import dataclasses
import json
import math
from pathlib import Path, PurePosixPath
import sys
import time
import traceback
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
from hiercp_v1x import bounded_scope
from hiercp_v1x.scope_learning_inputs import rebuild_scope, supervision_digest
from hiercp_v1x.scope_probe_support import activate_original, state_digest, _sha
from hiercp_v1x.contracts import canonical_hash, verify_archive


def verify_reference(report, contract, manifest):
    """Accept completed real evidence, preserving its actual diagnostic limits."""
    for name in ('debug', 'smoke_test', 'actual_CT', 'actual_CUDA'):
        if report.get(name) is not True:
            raise ValueError(f'Completed native reference requires {name}=true')
    for name in ('full_training', 'full_evaluation', 'production_ready', 'training_quality_verified', 'graph_size_quality_verified'):
        if report.get(name) is not False:
            raise ValueError(f'Native DEBUG reference cannot claim {name}')
    if not (contract.get('debug') is True and manifest.get('debug') is True
            and contract.get('actual_CT_requested') is True and contract.get('actual_CUDA_requested') is True):
        raise ValueError('Explicit actual-data/CUDA DEBUG contract required')
    for owner, keys in ((contract, ('full_training', 'full_evaluation', 'production_ready')),
                        (manifest, ('full_training', 'production_ready'))):
        if any(owner.get(name) is not False for name in keys):
            raise ValueError('Reference contract/fixture cannot claim production or full training')
    if (contract.get('seed') != 42 or contract.get('optimization_curriculum_epoch') != 29
            or contract.get('fixed_validation_epoch') != 29):
        raise ValueError('Recorded seed42/fixed epoch29 protocol required')
    if (report.get('fixture_sha256') != manifest.get('fixture_sha256')
            or not isinstance(report.get('fixture_sha256'), str) or len(report['fixture_sha256']) != 64
            or any(x not in '0123456789abcdef' for x in report['fixture_sha256'])):
        raise ValueError('Native reference fixture identity differs')
    cfg = manifest['config']
    for name, section in (('original_model', 'model'), ('original_graph', 'graph')):
        if report.get(name) != cfg[section] or contract.get(name) != cfg[section]:
            raise ValueError(f'Native reference {section} differs')
    if (contract.get('original_loss') != cfg['training'] or contract.get('original_runtime') != cfg['runtime']
            or cfg['seed'] != contract['seed'] or cfg['training']['epochs'] != 40
            or cfg['cache']['total_candidates'] != 8 or cfg['cache']['candidate_pool_size'] != 128):
        raise ValueError('Original full model/loss/curriculum configuration differs')
    train, val = manifest['train_cases'], manifest['validation_cases']
    if (contract.get('train_cases') != train or contract.get('held_out_cases') != val
            or manifest.get('prototype_training_cases') != train
            or len(set(train)) != len(train) or len(set(val)) != len(val) or set(train) & set(val)):
        raise ValueError('Native cohort order or training-only population fit differs')
    batch = contract['physical_sample_batch']
    if (type(batch) is not int or batch < 2 or batch != len(train)
            or report.get('physical_sample_batch') != batch
            or report.get('physical_graph_batch') != batch * 8
            or report.get('effective_sample_batch') != batch
            or contract.get('effective_sample_batch') != batch
            or report.get('gradient_accumulation_steps') != 1
            or contract.get('gradient_accumulation_steps') != 1
            or contract.get('physical_candidate_graph_batch') != batch * 8
            or contract.get('candidates_per_sample') != 8 or contract.get('candidate_pool_size') != 128
            or report.get('candidates_per_sample') != 8 or report.get('candidate_pool') != 128):
        raise ValueError('Native physical batch or candidate contract differs')
    expected_runtime = dict(cudnn_deterministic=cfg['runtime']['deterministic'],
        cudnn_benchmark=cfg['runtime']['cudnn_benchmark'],
        matmul_tf32=cfg['runtime']['allow_tf32'], cudnn_tf32=cfg['runtime']['allow_tf32'])
    if report.get('runtime') != expected_runtime:
        raise ValueError('Native GPU backend settings differ')
    branch = report['branches']['original_v1']
    if branch.get('parameters') != 10434532 or branch.get('trainable_parameters') != 10434532:
        raise ValueError('Original neural model dimensions differ')
    updates = branch['updates']
    if not updates or len(updates) != contract.get('updates_per_stage'):
        raise ValueError('Completed native update count differs')
    for number, row in enumerate(updates, 1):
        if row.get('step') != number or not math.isfinite(row['loss']):
            raise ValueError('Native loss/update evidence is incomplete or nonfinite')
        groups = row['gradient_groups']
        if not all(name in groups for name in ('CNN', 'L0', 'L1', 'L2', 'scalar_score')):
            raise ValueError('Native core gradient evidence missing')
        if not all(math.isfinite(value) and value > 0 for value in groups.values()):
            raise ValueError('Native core gradient is zero or nonfinite')
    for key, count in (('initial_train', len(train)), ('final_train', len(train)),
                       ('initial_held_out', len(val)), ('final_held_out', len(val))):
        metric = branch[key]
        if (len(metric['scores']) != count or any(len(row) != 8 for row in metric['scores'])
                or not all(math.isfinite(x) for row in metric['scores'] for x in row)
                or len(metric['positive_minus_best_other']) != count
                or not all(math.isfinite(x) for x in metric['positive_minus_best_other'])
                or not all(math.isfinite(metric[name]) and 0 <= metric[name] <= 1 for name in ('top1', 'MRR'))):
            raise ValueError('Native ranking evidence has wrong cohort or invalid scores')
    initial = report['shared_initial_state_sha256']
    if len(initial) != 64 or any(x not in '0123456789abcdef' for x in initial):
        raise ValueError('Native initial neural identity is invalid')
    return dict(initial_neural_sha256=initial, reference_updates=len(updates),
        train_cases=train, validation_cases=val, physical_sample_batch=batch,
        workers=report['workers'], seed=42, optimization_curriculum_epoch=29,
        train_view_epoch=29, validation_effective_view_epoch=0, scheduler=None,
        baseline_model='v1.0', experiment_version='v1.4')


def write_new(path, value):
    with Path(path).open('x', encoding='utf8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def fresh_source(root):
    proof = verify_archive(ROOT)
    source = root / 'source/v1.0'
    source.mkdir(parents=True)
    with ZipFile(ROOT / 'versions/v1/pipeline_v1_source.zip') as archive:
        for info in archive.infolist():
            path = PurePosixPath(info.filename)
            if path.is_absolute() or '..' in path.parts or '\\' in info.filename or ':' in info.filename:
                raise ValueError('Unsafe original archive member')
            target = source.joinpath(*path.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as stream:
                stream.write(archive.read(info))
    return source, proof


def ranking(net, batch, config):
    import torch
    from hiercp.loss import ranking_metrics
    net.eval()
    with torch.no_grad(), torch.autocast('cuda', enabled=config['training']['amp']):
        output = net(batch)
        top1, mrr = ranking_metrics(output.scores)
        values = [row.detach().float().cpu().tolist() for row in output.scores]
    margins = [row[0] - max(row[1:]) for row in values]
    if not all(math.isfinite(x) for row in values for x in row):
        raise AssertionError('Nonfinite ranking output')
    return dict(top1=top1, MRR=mrr, positive_minus_best_other=margins,
        scores=values, mean_margin=sum(margins) / len(margins))


def gradient_observation(net):
    """Distinguish a finite zero gradient from AMP overflow, without hiding either."""
    import torch
    groups = dict(CNN=net.local_encoder.dense_encoder, L0=net.local_encoder.blocks,
        local_encoder_total=net.local_encoder, role_attention_pool=net.local_encoder.pool,
        shell_attention_pool=net.local_encoder.context_shell_pool,
        local_fusion=net.local_encoder.final_fuse, L1=net.patient_encoder,
        L2=net.prototype_encoder, scalar_score=net.score_head)
    raw = {}
    for name, module in groups.items():
        values = [p.grad.detach().double().square().sum() for p in module.parameters()
                  if p.requires_grad and p.grad is not None]
        if not values:
            raise AssertionError(f'Complete original loss does not reach {name}')
        raw[name] = float(torch.stack(values).sum().sqrt())
    present = [p.grad for p in net.parameters() if p.grad is not None]
    finite = bool(torch.stack([torch.isfinite(g).all() for g in present]).all())
    return dict(finite=finite,
        norms={key:value if math.isfinite(value) else str(value) for key,value in raw.items()},
        zero_groups=[key for key,value in raw.items() if value == 0],
        nonfinite_groups=[key for key,value in raw.items() if not math.isfinite(value)],
        nonfinite_tensors=[name for name,p in net.named_parameters()
            if p.grad is not None and not bool(torch.isfinite(p.grad).all())] if not finite else [],
        norm_reduction_dtype='float64')


def execute(a, root, report, contract, manifest, protocol):
    from tools.local_cnn_device import select
    select(a.gpu)
    source, archive = fresh_source(root)
    proof = activate_original(source)
    import psutil
    import torch
    from hiercp.model import HierarchicalPyGPlacementModel
    from hiercp.data import collate_samples, summarize_hierarchical_batch
    from hiercp.sample import materialize_sample_views
    from hiercp.loss import CurriculumConfig, curriculum_ranking_loss
    from hiercp.tensor import set_seed, configure_runtime
    from tqdm import tqdm

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Exactly one actual CUDA GPU required; no CPU fallback')
    total = torch.cuda.get_device_properties(0).total_memory
    limit = int(a.cuda_gib * 2**30)
    if not 0 < limit < total:
        raise ValueError('Explicit CUDA budget must leave physical headroom')
    torch.cuda.set_per_process_memory_fraction(limit / total)
    torch.set_num_threads(protocol['workers'])
    started = time.perf_counter()
    process = psutil.Process()
    memory_at_start = psutil.virtual_memory()
    gpu_free_at_start, _ = torch.cuda.mem_get_info()
    peak_rss = [0]
    def budget():
        peak_rss[0] = max(peak_rss[0], process.memory_info().rss)
        if peak_rss[0] > int(a.rss_gib * 2**30):
            raise MemoryError('Explicit RSS budget exceeded; original cohort/model preserved')
        if time.perf_counter() - started > a.wall_seconds:
            raise TimeoutError('Explicit short-diagnostic wall budget exceeded; no partial quality PASS')
    fixture = torch.load(a.fixture, map_location='cpu', weights_only=False, mmap=True)
    config = manifest['config']
    if (fixture['config'] != config or fixture['train_cases'] != protocol['train_cases']
            or fixture['validation_cases'] != protocol['validation_cases']):
        raise ValueError('Packed real-data fixture metadata differs')
    samples = fixture['samples']
    if [(s['case_id'], s['split']) for s in samples] != (
            [(c, 'train') for c in protocol['train_cases']] + [(c, 'val') for c in protocol['validation_cases']]):
        raise ValueError('Original sample order/split differs')
    for row in manifest['source_records']:
        for kind in ('image', 'label'):
            if _sha(Path(row[kind])) != row[kind + '_sha256']:
                raise ValueError('Raw CT/annotation differs from original fixture')
    before = supervision_digest(samples)
    receipt = bounded_scope.install(a.margin_mm, source)
    bounded, preparation = rebuild_scope(samples, manifest, config, bounded_scope,
        a.margin_mm, protocol['workers'], budget)
    if supervision_digest(bounded) != before:
        raise AssertionError('Scope change altered candidates/GT/L1/L2 input')
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
    training = [s for s in views if s['case_id'] in protocol['train_cases']]
    validation = [s for s in views if s['case_id'] in protocol['validation_cases']]
    reset()
    net = HierarchicalPyGPlacementModel(**config['model'])
    identity = state_digest(net.state_dict())
    if identity != protocol['initial_neural_sha256']:
        raise AssertionError('Initial neural state differs from preserved native reference')
    if sum(p.numel() for p in net.parameters()) != report['branches']['original_v1']['parameters']:
        raise AssertionError('Original model size changed')
    net.cuda()
    cpu = collate_samples(training).pin_memory()
    shape = summarize_hierarchical_batch(cpu)
    batch = cpu.to('cuda', non_blocking=True)
    del cpu
    held = collate_samples(validation).to('cuda')
    torch.cuda.synchronize()
    fields = {field.name for field in dataclasses.fields(CurriculumConfig)}
    objective = CurriculumConfig(**{k:v for k,v in config['training'].items() if k in fields})
    optimizer = torch.optim.AdamW(net.parameters(), lr=config['training']['lr'],
        weight_decay=config['training']['weight_decay'], fused=config['training']['fused_optimizer'])
    scaler = torch.amp.GradScaler('cuda', enabled=config['training']['amp'])
    result = dict(debug=True, actual_CT=True, actual_CUDA=True, status='RUNNING',
        full_training=False, full_evaluation=False, quality_verified=False,
        production_ready=False, checkpoint_written=False, native_retrained=False,
        protocol=protocol, margin_mm=a.margin_mm, bounded_scope=receipt,
        configuration=config, initial_neural_sha256=identity, supervision_sha256=before,
        model_parameters=sum(p.numel() for p in net.parameters()),
        trainable_parameters=sum(p.numel() for p in net.parameters() if p.requires_grad), input=shape,
        GPU=torch.cuda.get_device_name(0), torch_version=torch.__version__,
        cuda_build=torch.version.cuda, device_total_bytes=total, cuda_limit_bytes=limit,
        original_native_GPU=report['GPU'], preparation=preparation,
        resources_at_start=dict(cpu_physical=psutil.cpu_count(logical=False),
            cpu_logical=psutil.cpu_count(), ram_total_bytes=memory_at_start.total,
            ram_available_bytes=memory_at_start.available, GPU_free_bytes=gpu_free_at_start,
            disk_free_bytes=psutil.disk_usage(root).free, rss_limit_bytes=int(a.rss_gib*2**30)),
        sampled_view_preparation_seconds=view_seconds, updates=[], attempts=[], curve=[],
        timings_are_shared_GPU_debug_compute_only=True,
        reference_runtime_versions_not_recorded=True, bitwise_native_reproduction_claimed=False,
        exact_resume_claimed=False,
        reference_report_sha256=_sha(a.reference), fixture_sha256=_sha(a.fixture),
        current_sources_sha256={name:_sha(ROOT / name) for name in (
            'tools/verify_v14_learning_replay.py', 'hiercp_v1x/bounded_scope.py',
            'hiercp_v1x/scope_learning_inputs.py', 'hiercp_v1x/scope_probe_support.py')})
    print(json.dumps(dict(experiment='v1.4', baseline='v1.0', debug=True,
        parameters=result['model_parameters'], model=config['model'], margin_mm=a.margin_mm,
        physical_sample_batch=protocol['physical_sample_batch'], candidate_graph_batch=shape,
        full_fixture_samples=len(samples), used_samples=len(views), seed=42,
        candidate_pool=128, curriculum_candidates=8, workers=protocol['workers'],
        successful_updates_requested=a.updates, amp=config['training']['amp'],
        cuda_limit_GiB=a.cuda_gib, rss_limit_GiB=a.rss_gib), allow_nan=False), flush=True)
    def evaluate(step):
        row = dict(step=step, train=ranking(net, batch, config), validation=ranking(net, held, config))
        result['curve'].append(row)
        with (root / 'curve.jsonl').open('a', encoding='utf8') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
        print(f"v1.4 fixed-view step{step} | train MRR={row['train']['MRR']:.4f} top1={row['train']['top1']:.4f} margin={row['train']['mean_margin']:.5f} | held-out MRR={row['validation']['MRR']:.4f} margin={row['validation']['mean_margin']:.5f}", flush=True)
    evaluate(0)
    reset()  # Match the native reference's dropout stream immediately before update1.
    connected = set()
    before_head = next(net.score_head.parameters()).detach().clone()
    observed_parameter = next(net.parameters())
    def optimizer_step():
        state = optimizer.state.get(observed_parameter, {})
        counter = state.get('step', 0)
        return int(counter.item()) if isinstance(counter, torch.Tensor) else int(counter)
    def record_attempt(row):
        result['attempts'].append(row)
        with (root / 'attempts.jsonl').open('a', encoding='utf8') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
    bar = tqdm(total=a.updates, desc='v1.4 matched learning DEBUG', unit='update')
    step, attempt = 0, 0
    while step < a.updates:
        attempt += 1
        budget()
        net.train(); optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
        began = time.perf_counter()
        with torch.autocast('cuda', enabled=config['training']['amp']):
            out = net(batch)
            base, terms = curriculum_ranking_loss(out.scores, batch.difficulty_list(), epoch=29, config=objective)
            loss = base + config['training']['consistency_weight'] * out.consistency
        if not bool(torch.isfinite(loss)):
            record_attempt(dict(attempt=attempt, completed_updates=step, status='NONFINITE_FORWARD_LOSS',
                                loss=str(float(loss.detach()))))
            raise AssertionError('Nonfinite original forward loss; no optimizer update')
        torch.cuda.synchronize(); forward = time.perf_counter() - began
        back = time.perf_counter()
        scale_before = scaler.get_scale()
        scaler.scale(loss).backward(); scaler.unscale_(optimizer); torch.cuda.synchronize()
        backward = time.perf_counter() - back
        observation = gradient_observation(net)
        connected.update(name for name,p in net.named_parameters() if p.requires_grad and p.grad is not None)
        counter_before = optimizer_step()
        if not observation['finite']:
            # Preserve the original AMP mechanism. GradScaler consumes its own
            # found_inf result, skips this attempt and reduces the scale. Never
            # count that skipped attempt as a completed optimizer update.
            if not scaler.is_enabled():
                record_attempt(dict(attempt=attempt, completed_updates=step,
                    status='NONFINITE_GRADIENT_AMP_DISABLED', gradient=observation))
                raise AssertionError('Nonfinite gradient with AMP disabled; no optimizer update')
            scaler.step(optimizer); scaler.update(); torch.cuda.synchronize()
            scale_after = scaler.get_scale()
            unchanged = optimizer_step() == counter_before
            record_attempt(dict(attempt=attempt, completed_updates=step,
                status='AMP_OVERFLOW_SKIPPED', gradient=observation,
                loss=float(loss.detach()), scale_before=scale_before, scale_after=scale_after,
                optimizer_step_unchanged=unchanged, elapsed_seconds=time.perf_counter()-began))
            if not unchanged or not scale_after < scale_before:
                raise AssertionError('AMP overflow did not safely skip/reduce its scale')
            if step < protocol['reference_updates']:
                raise AssertionError('Overflow changed the first native-reference-count update protocol')
            print(f'AMP overflow attempt{attempt} | completed updates={step} | scale {scale_before:g}->{scale_after:g} | optimizer skipped; raw gradients preserved', flush=True)
            del out, loss, base, terms
            budget()
            continue
        if observation['nonfinite_groups']:
            record_attempt(dict(attempt=attempt, completed_updates=step,
                status='GRADIENT_NORM_REDUCTION_OVERFLOW', gradient=observation))
            raise AssertionError('Finite tensor gradients produced a nonfinite diagnostic norm')
        if observation['zero_groups'] and step < protocol['reference_updates']:
            record_attempt(dict(attempt=attempt, completed_updates=step,
                status='REFERENCE_CORE_GRADIENT_ZERO', gradient=observation))
            raise AssertionError('Zero core gradient changed the first native-reference-count protocol')
        clip = torch.nn.utils.clip_grad_norm_(net.parameters(), config['training']['grad_clip'], error_if_nonfinite=True)
        opt = time.perf_counter()
        scaler.step(optimizer); scaler.update(); torch.cuda.synchronize()
        if optimizer_step() != counter_before + 1:
            raise AssertionError('Finite attempt did not complete exactly one original optimizer update')
        step += 1
        row = dict(step=step, attempt=attempt, loss=float(loss.detach()), native_terms={k:float(v.detach()) for k,v in terms.items()},
            forward_seconds=forward, backward_seconds=backward,
            optimizer_seconds=time.perf_counter() - opt, update_seconds=time.perf_counter() - began,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(), RSS_bytes=process.memory_info().rss,
            gradient_groups=observation['norms'], gradient_observation=observation,
            gradient_before_clip=float(clip), amp_scale_before=scale_before, amp_scale_after=scaler.get_scale())
        record_attempt(dict(attempt=attempt, completed_updates=step,
            status='OPTIMIZER_UPDATED', gradient=observation, scale_before=scale_before,
            scale_after=scaler.get_scale(), optimizer_counter=optimizer_step()))
        result['updates'].append(row)
        with (root / 'updates.jsonl').open('a', encoding='utf8') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
        del out, loss, base, terms, clip
        bar.update(1)
        if step == protocol['reference_updates'] or step % 4 == 0 or step == a.updates:
            evaluate(step)
        budget()
    bar.close()
    original = report['branches']['original_v1']
    matched = next(row for row in result['curve'] if row['step'] == protocol['reference_updates'])
    result['reference_at_matched_updates'] = dict(initial_train=original['initial_train'],
        final_train=original['final_train'], initial_validation=original['initial_held_out'],
        final_validation=original['final_held_out'], updates=original['updates'])
    result['matched_update_differences'] = {
        split: {metric:matched[split][metric] - original[old_key][metric] for metric in ('MRR', 'top1')}
        for split,old_key in (('train','final_train'),('validation','final_held_out'))}
    expected = {name for name,p in net.named_parameters() if p.requires_grad}
    result.update(status='COMPLETE', finite_complete_updates=True,
        completed_optimizer_updates=len(result['updates']), attempted_backward_passes=len(result['attempts']),
        amp_overflow_skipped_attempts=sum(row['status']=='AMP_OVERFLOW_SKIPPED' for row in result['attempts']),
        core_modules_have_nonzero_gradients=all(not row['gradient_observation']['zero_groups'] for row in result['updates']),
        finite_zero_gradient_updates=[row['step'] for row in result['updates'] if row['gradient_observation']['zero_groups']],
        connected_trainable_parameter_tensors=len(connected),
        expected_trainable_parameter_tensors=len(expected), missing_gradients=sorted(expected - connected),
        scalar_head_changed=not torch.equal(before_head, next(net.score_head.parameters()).detach()),
        final_neural_sha256=state_digest(net.state_dict()),
        process_peak_rss_bytes=peak_rss[0], total_wall_seconds=time.perf_counter() - started,
        original_fixture_preserved=_sha(a.fixture) == manifest['fixture_sha256'],
        original_reference_preserved=_sha(a.reference) == result['reference_report_sha256'],
        original_source_preserved=all(_sha(source / name) == value for name,value in proof['verified_files'].items()))
    if expected - connected:
        raise AssertionError('Original trainable parameters were disconnected from the complete loss')
    if not all(result[key] for key in ('scalar_head_changed', 'original_fixture_preserved',
                                     'original_reference_preserved', 'original_source_preserved')):
        raise AssertionError('Native source/reference/input changed or scalar head did not learn')
    write_new(root / 'report.json', result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--fixture', type=Path, required=True)
    p.add_argument('--margin-mm', type=float, required=True)
    p.add_argument('--updates', type=int, required=True, help='Explicit short DEBUG update count, not production epochs')
    p.add_argument('--cuda-gib', type=float, required=True)
    p.add_argument('--rss-gib', type=float, required=True)
    p.add_argument('--wall-seconds', type=float, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if (a.gpu < 0 or a.margin_mm not in (10., 20.) or a.updates < 2
            or not all(math.isfinite(v) and v > 0 for v in (a.cuda_gib, a.rss_gib, a.wall_seconds))):
        p.error('Explicit bounded margin, >=2 DEBUG updates and finite resource budgets required')
    report = json.loads(a.reference.read_text(encoding='utf8'))
    contract_path = a.reference.with_name('execution_contract.json')
    fixture_manifest_path = a.fixture.with_name('fixture_manifest.json')
    contract = json.loads(contract_path.read_text(encoding='utf8'))
    manifest = json.loads(fixture_manifest_path.read_text(encoding='utf8'))
    protocol = verify_reference(report, contract, manifest)
    if a.updates < protocol['reference_updates'] or _sha(a.fixture) != manifest['fixture_sha256']:
        raise ValueError('Full reference-count updates and exact original fixture bytes required')
    root = a.output.resolve()
    if root.exists():
        raise FileExistsError('Preserve existing evidence; new DEBUG output required')
    if root.is_relative_to(a.fixture.resolve().parent) or root.is_relative_to(a.reference.resolve().parent):
        raise ValueError('New result must be disjoint from original reference/input directories')
    root.mkdir(parents=True)
    provenance = dict(debug=True, quality_verified=False, protocol=protocol, arguments={
        k:str(v) if isinstance(v, Path) else v for k,v in vars(a).items()},
        inputs={str(path.resolve()):_sha(path) for path in (a.reference, contract_path, a.fixture, fixture_manifest_path)})
    provenance['identity_sha256'] = canonical_hash(provenance)
    write_new(root / 'execution_contract.json', provenance)
    try:
        result = execute(a, root, report, contract, manifest, protocol)
    except Exception:
        write_new(root / 'failed.json', dict(debug=True, quality_verified=False,
            production_ready=False, error=traceback.format_exc()))
        raise
    print(f"REPORT: {root / 'report.json'}", flush=True)
    print(json.dumps(result['matched_update_differences'], indent=2), flush=True)


if __name__ == '__main__':
    main()
