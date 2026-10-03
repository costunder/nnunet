"""Original-v1 ranking learning on an explicit diagnostic scope cohort.

This loop does not change GT, the native curriculum or the neural model. Its
selected-patient results are learning evidence, not full-cohort accuracy. It
writes score/metric evidence only, never checkpoints or readiness markers.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import copy
import dataclasses
import json
import math
from pathlib import Path
import time


NATIVE_PARAMETER_COUNT = 10434532
LOSS_NAMES = ("loss", "ranking", "consistency", "ce", "pair", "ordinal", "mined")


def execution_contract(config, *, physical_batch, workers, epochs, smoke):
    """Require explicit diagnostic execution without shortening the curriculum."""
    if type(physical_batch) is not int or physical_batch < 2:
        raise ValueError("Explicit physical sample batch >= 2 required")
    if type(workers) is not int or workers < 2:
        raise ValueError("Explicit parallel preprocessing workers >= 2 required")
    if type(smoke) is not bool or type(epochs) is not int:
        raise ValueError("Explicit bool smoke and integer epoch count required")
    training = config["training"]
    native_epochs = training["epochs"]
    if type(native_epochs) is not int or native_epochs != 40 or config["seed"] != 42:
        raise ValueError("Original 40-epoch seed42 contract required")
    if ((smoke and not 1 <= epochs <= 2)
            or (not smoke and epochs != native_epochs)):
        raise ValueError("Learning uses all original 40 epochs; only explicit smoke may use 1..2")
    if (config["cache"]["total_candidates"] != 8
            or config["cache"]["candidate_pool_size"] != 128
            or training["fixed_validation_epoch"] != 29
            or training["gradient_accumulation_steps"] != 1
            or training["target_effective_batch_size"] is not None):
        raise ValueError("Original candidates, fixed validation and physical update contract required")
    return dict(seed=42, epochs=epochs, native_epochs=native_epochs,
                scheduler_t_max=native_epochs, fixed_validation_epoch=29,
                physical_sample_batch=physical_batch,
                physical_candidate_graph_batch=8 * physical_batch,
                gradient_accumulation_steps=1, data_parallel_workers=1,
                effective_sample_batch=physical_batch, workers=workers,
                candidates_per_sample=8, candidate_pool=128,
                smoke=smoke, debug=True, full_training=False,
                full_evaluation=False, production_ready=False, quality_verified=False,
                target="native source anchor versus existing curriculum candidates",
                cohort="explicit selected diagnostic patients; not full 84/21 accuracy",
                evaluation="fixed native two views and full eight scores; loss epoch29",
                checkpoint_written=False)


def score_metrics(score_rows):
    """Aggregate full8 native rankings; ties conservatively outrank the positive."""
    rows = [list(row) for row in score_rows]
    if not rows or any(len(row) != 8 for row in rows):
        raise ValueError("Metrics require every native sample's eight candidate scores")
    if any(not math.isfinite(float(value)) for row in rows for value in row):
        raise ValueError("Nonfinite candidate score")
    ranks = [1 + sum(value >= row[0] for value in row[1:]) for row in rows]
    count = len(rows)
    return dict(samples=count, top1=sum(rank == 1 for rank in ranks) / count,
                MRR=sum(1 / rank for rank in ranks) / count,
                margin=sum(row[0] - max(row[1:]) for row in rows) / count,
                pair_win=sum(sum(row[0] > value for value in row[1:]) for row in rows) / (7 * count),
                positive_score=sum(row[0] for row in rows) / count,
                best_other_score=sum(max(row[1:]) for row in rows) / count,
                positive_ranks=ranks)


def independent_transfer(batch, device="cuda", *, non_blocking=True):
    """Transfer a copy of mutable PyG stores; leave fixed evaluation on CPU.

    PyG's __copy__ copies each storage mapping but shares immutable CPU tensor
    data. .to replaces mapping values, so it cannot move the cached CPU graph.
    Dense tensors are likewise reassigned only on the copied batch object.
    """
    fresh = copy.copy(batch)
    for key in ("local_batch", "local_batch_view2", "patient_batch", "prototype_batch"):
        graph = getattr(batch, key)
        setattr(fresh, key, None if graph is None else copy.copy(graph))
    return fresh.to(device, non_blocking=non_blocking)


def _validate_samples(train_samples, validation_samples, physical_batch):
    if len(train_samples) < physical_batch or not validation_samples:
        raise ValueError("Diagnostic learning requires a full train batch and held-out CT")
    train_cases = {sample["case_id"] for sample in train_samples}
    validation_cases = {sample["case_id"] for sample in validation_samples}
    if train_cases & validation_cases:
        raise ValueError("Training and held-out diagnostic patients overlap")
    identities = []
    for split, samples in (("train", train_samples), ("val", validation_samples)):
        for sample in samples:
            if sample.get("split") != split or not sample.get("case_id"):
                raise ValueError("Diagnostic sample split/case identity mismatch")
            if "local_graphs" in sample or "local_graphs_view2" in sample:
                raise ValueError("Training input must retain canonical tables for epoch-dependent views")
            if (len(sample["target_locals"]) != 8
                    or sample["difficulties"].numel() != 8
                    or int(sample["difficulties"][0]) != 0
                    or sample["target_patches"].shape[0] != 8):
                raise ValueError("Complete native eight-candidate anchor curriculum required")
            identities.append((sample["case_id"], int(sample["sample_index"])))
    if len(set(identities)) != len(identities):
        raise ValueError("Repeated diagnostic sample identity")


def run_learning(net, train_samples, validation_samples, config, *,
                 physical_batch, workers, epochs, output_dir, smoke=False,
                 rss_budget_check=None):
    """Train the original model and record fixed full8 ranking each epoch.

    Caller activates/verifies original v1 source, installs exactly one margin,
    rebuilds verified canonical samples and seeds/constructs native model first.
    No implicit subset, early stopping, resource fallback or checkpoint exists.
    """
    import torch
    from torch.utils.data import RandomSampler
    from tqdm import tqdm
    from hiercp.data import collate_samples
    from hiercp.loss import CurriculumConfig, curriculum_ranking_loss
    from hiercp.sample import materialize_sample_views
    from hiercp.tensor import capture_rng_state, restore_rng_state, configure_runtime

    contract = execution_contract(config, physical_batch=physical_batch,
                                  workers=workers, epochs=epochs, smoke=smoke)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one actual CUDA device required; no CPU fallback")
    parameters = [p for p in net.parameters() if p.requires_grad]
    if (sum(p.numel() for p in net.parameters()) != NATIVE_PARAMETER_COUNT
            or not parameters or any(p.device.type != "cuda" for p in parameters)):
        raise ValueError("Unmodified native 10434532-parameter CUDA model required")
    native_parameters = {id(p) for p in net.trainable_parameters()}
    if native_parameters != {id(p) for p in parameters}:
        raise ValueError("Native trainable parameter interface differs")
    _validate_samples(train_samples, validation_samples, physical_batch)
    runtime, training = config["runtime"], config["training"]
    configure_runtime(deterministic=runtime["deterministic"],
                      allow_tf32=runtime["allow_tf32"],
                      cudnn_benchmark=runtime["cudnn_benchmark"])
    objective = CurriculumConfig(**{key: value for key, value in training.items()
        if key in {field.name for field in dataclasses.fields(CurriculumConfig)}})
    objective.validate()
    optimizer = torch.optim.AdamW(parameters, lr=training["lr"],
        weight_decay=training["weight_decay"], fused=training["fused_optimizer"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=contract["scheduler_t_max"])
    scaler = torch.amp.GradScaler("cuda", enabled=training["amp"])
    shuffle_generator = torch.Generator().manual_seed(config["seed"] + 2003)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    curve = output_dir / "learning_curve.jsonl"
    report_path = output_dir / "learning_report.json"
    if curve.exists() or report_path.exists():
        raise FileExistsError("Diagnostic learning outputs are preserved; choose a fresh output")
    def budget():
        if rss_budget_check is not None:
            rss_budget_check()

    def prepare(sample, *, training_mode, epoch):
        # materialize pops canonical keys; never mutate cached source samples.
        local = dict(sample)
        local["source_patch"] = sample["source_patch"].to(dtype=torch.float16)
        local["target_patches"] = sample["target_patches"].to(dtype=torch.float16)
        return materialize_sample_views(local, training=training_mode,
            epoch=epoch, global_seed=config["seed"])

    def prepare_batch(samples, indices, *, training_mode, epoch, pool):
        def one(index):
            return prepare(samples[index], training_mode=training_mode, epoch=epoch)
        prepared = list(pool.map(one, indices))
        result = collate_samples(prepared)
        del prepared
        if training["pin_memory"]:
            result.pin_memory()
        budget()
        return result

    def chunks(order):
        return [order[start:start + physical_batch]
                for start in range(0, len(order), physical_batch)]

    def eval_pass(cached, samples, *, description):
        saved_rng, previous_mode = capture_rng_state(), net.training
        rows, losses = [], [0.0] * len(LOSS_NAMES)
        count, gpu_batch, output, base, parts, loss = 0, None, None, None, None, None
        torch.cuda.synchronize()
        began = time.perf_counter()
        try:
            net.eval()
            with torch.no_grad():
                for cpu_batch in tqdm(cached, desc=description, unit="batch"):
                    budget()
                    gpu_batch = independent_transfer(cpu_batch)
                    with torch.autocast("cuda", enabled=training["amp"]):
                        output = net(gpu_batch)
                        base, parts = curriculum_ranking_loss(output.scores,
                            gpu_batch.difficulty_list(), epoch=contract["fixed_validation_epoch"],
                            config=objective)
                        loss = base + training["consistency_weight"] * output.consistency
                    scores = torch.stack([value.detach().float() for value in output.scores])
                    values = torch.stack([loss.detach(), base.detach(), output.consistency.detach(),
                        parts["ce"], parts["pair"], parts["ordinal"], parts["mined"]]).double()
                    batch_count = len(output.scores)
                    rows.extend(scores.cpu().tolist())
                    for index, value in enumerate(values.cpu().tolist()):
                        losses[index] += value * batch_count
                    count += batch_count
                    del scores, values
                    gpu_batch, output, base, parts, loss = None, None, None, None, None
            if count != len(samples):
                raise AssertionError("Evaluation omitted diagnostic samples")
            result = score_metrics(rows)
            result.update({name: value / count for name, value in zip(LOSS_NAMES, losses)})
            result["scores"] = [dict(case_id=sample["case_id"], sample_index=int(sample["sample_index"]),
                values=values) for sample, values in zip(samples, rows)]
            result["loss_epoch"] = contract["fixed_validation_epoch"]
            torch.cuda.synchronize()
            result["seconds"] = time.perf_counter() - began
            return result
        finally:
            gpu_batch, output, base, parts, loss = None, None, None, None, None
            net.train(previous_mode)
            restore_rng_state(saved_rng)

    history, connected = [], set()
    initial_parameter_probe = next(net.score_head.parameters()).detach().clone()
    fixed = {}
    with curve.open("x", encoding="utf-8") as stream, ThreadPoolExecutor(max_workers=workers) as pool:
        # Precompute native training=False views once; eval never resamples them.
        saved_rng = capture_rng_state()
        try:
            fixed_started = time.perf_counter()
            for split, samples in (("train", train_samples), ("validation", validation_samples)):
                fixed[split] = [prepare_batch(samples, indices, training_mode=False,
                    epoch=contract["fixed_validation_epoch"], pool=pool)
                    for indices in tqdm(chunks(list(range(len(samples)))),
                                        desc=f"fixed {split} views", unit="batch")]
            fixed_preparation_seconds = time.perf_counter() - fixed_started
        finally:
            restore_rng_state(saved_rng)
        def record(epoch, optimization=None):
            row = dict(epoch=epoch, train=eval_pass(fixed["train"], train_samples,
                description=f"fixed train epoch{epoch}"),
                validation=eval_pass(fixed["validation"], validation_samples,
                description=f"held-out epoch{epoch}"),
                optimizer_lr=float(optimizer.param_groups[0]["lr"]),
                optimization=optimization, **contract)
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            history.append(row)
            print(f"Learning epoch {epoch}/{epochs} | train MRR={row['train']['MRR']:.4f} "
                  f"top1={row['train']['top1']:.4f} margin={row['train']['margin']:.4f} | "
                  f"held-out MRR={row['validation']['MRR']:.4f} "
                  f"top1={row['validation']['top1']:.4f} "
                  f"pair-win={row['validation']['pair_win']:.4f} "
                  f"margin={row['validation']['margin']:.4f}", flush=True)
        record(0)
        with ThreadPoolExecutor(max_workers=1) as ahead:
            for epoch in range(1, epochs + 1):
                net.train()
                order = list(RandomSampler(train_samples, generator=shuffle_generator))
                groups = chunks(order)
                future = ahead.submit(prepare_batch, train_samples, groups[0],
                    training_mode=True, epoch=epoch, pool=pool)
                totals = torch.zeros(len(LOSS_NAMES), device="cuda", dtype=torch.float64)
                rank_totals = torch.zeros(4, device="cuda", dtype=torch.float64)
                train_scores = []
                wait_seconds, transfer_seconds, update_seconds = 0.0, 0.0, 0.0
                torch.cuda.reset_peak_memory_stats()
                began = time.perf_counter()
                cpu_batch, gpu_batch, output, base, parts, loss = None, None, None, None, None, None
                bar = tqdm(range(len(groups)), desc=f"learning epoch{epoch}/{epochs}", unit="batch")
                try:
                    for batch_index in bar:
                        started = time.perf_counter()
                        cpu_batch = future.result()
                        # A Future owns its result. Release it before .to mutates
                        # that result, or the last GPU graph survives into eval.
                        future = None
                        wait_seconds += time.perf_counter() - started
                        if batch_index + 1 < len(groups):
                            future = ahead.submit(prepare_batch, train_samples, groups[batch_index + 1],
                                training_mode=True, epoch=epoch, pool=pool)
                        budget()
                        torch.cuda.synchronize()
                        started = time.perf_counter()
                        gpu_batch = cpu_batch.to("cuda", non_blocking=True)
                        cpu_batch = None
                        torch.cuda.synchronize()
                        transfer_seconds += time.perf_counter() - started
                        started = time.perf_counter()
                        optimizer.zero_grad(set_to_none=True)
                        with torch.autocast("cuda", enabled=training["amp"]):
                            output = net(gpu_batch)
                            base, parts = curriculum_ranking_loss(output.scores,
                                gpu_batch.difficulty_list(), epoch=epoch, config=objective)
                            loss = base + training["consistency_weight"] * output.consistency
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        connected.update(name for name, p in net.named_parameters()
                                         if p.requires_grad and p.grad is not None)
                        norm = torch.nn.utils.clip_grad_norm_(parameters, training["grad_clip"],
                                                              error_if_nonfinite=True)
                        scaler.step(optimizer)
                        scaler.update()
                        scores = torch.stack([value.detach().float() for value in output.scores])
                        positives, others = scores[:, :1], scores[:, 1:]
                        ranks = 1 + (others >= positives).sum(dim=1)
                        batch_count = len(output.scores)
                        vector = torch.stack([loss.detach(), base.detach(), output.consistency.detach(),
                            parts["ce"], parts["pair"], parts["ordinal"], parts["mined"]]).double()
                        totals += vector * batch_count
                        rank_totals += torch.stack([(ranks == 1).float().sum(),
                            ranks.float().reciprocal().sum(),
                            (positives > others).float().sum() / 7,
                            (positives[:, 0] - others.max(dim=1).values).sum()]).double()
                        torch.cuda.synchronize()
                        update_seconds += time.perf_counter() - started
                        # A single compact display copy; no per-node/per-gradient sync.
                        display = torch.stack([loss.detach().float(), ranks.float().reciprocal().mean(),
                                               norm.detach().float()]).cpu().tolist()
                        train_scores.extend(dict(case_id=train_samples[index]["case_id"],
                            sample_index=int(train_samples[index]["sample_index"]), values=values)
                            for index, values in zip(groups[batch_index], scores.cpu().tolist()))
                        bar.set_postfix(loss=f"{display[0]:.4f}", MRR=f"{display[1]:.3f}",
                                        grad=f"{display[2]:.3g}")
                        del scores, positives, others, ranks, vector, norm, display
                        gpu_batch, output, base, parts, loss = None, None, None, None, None
                        budget()
                finally:
                    cpu_batch, gpu_batch, output, base, parts, loss = None, None, None, None, None, None
                    optimizer.zero_grad(set_to_none=True)
                    bar.close()
                torch.cuda.synchronize()
                count = len(train_samples)
                optimization = {name: value / count for name, value in
                                zip(LOSS_NAMES, totals.cpu().tolist())}
                optimization.update(dict(top1=float(rank_totals[0]) / count,
                    MRR=float(rank_totals[1]) / count, pair_win=float(rank_totals[2]) / count,
                    margin=float(rank_totals[3]) / count, samples=count,
                    optimizer_steps=len(groups), scores=train_scores, curriculum_epoch=epoch,
                    active_max_difficulty=(1.0 if epoch <= objective.easy_epochs else
                        2.0 if epoch <= objective.inter_epochs else 3.0),
                    seconds=time.perf_counter() - began, loader_wait_seconds=wait_seconds,
                    transfer_seconds=transfer_seconds, GPU_update_seconds=update_seconds,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                    optimizer_lr=float(optimizer.param_groups[0]["lr"])))
                del totals, rank_totals
                record(epoch, optimization)
                scheduler.step()
    changed = not torch.equal(initial_parameter_probe, next(net.score_head.parameters()).detach())
    del initial_parameter_probe, fixed
    expected = {name for name, p in net.named_parameters() if p.requires_grad}
    missing = sorted(expected - connected)
    if not changed:
        raise AssertionError("Native scalar head did not change under original objective")
    report = dict(status="PASS", scope="selected-cohort native ranking learning diagnostic",
        **contract, initial=history[0], epochs_completed=epochs,
        history=history, fixed_view_preparation_seconds=fixed_preparation_seconds,
        scalar_head_changed=changed, connected_parameters=len(connected),
        expected_trainable_parameters=len(expected), missing_gradient_parameters=missing,
        parameter_count=NATIVE_PARAMETER_COUNT,
        train_case_ids=sorted({sample["case_id"] for sample in train_samples}),
        validation_case_ids=sorted({sample["case_id"] for sample in validation_samples}),
        original_GT_preserved=True, original_loss_preserved=True,
        final_lr=float(optimizer.param_groups[0]["lr"]))
    with report_path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    return report
