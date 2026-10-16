"""Independent actual-CT CUDA diagnostic for the full KiTS19 paper network.

This is a DEBUG admission measurement, not production training or an accuracy
experiment. It never opens a production checkpoint. The 42 held-out image,
label, property and preprocessed-array files are not read by this diagnostic.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import sys
import time
import traceback
import uuid


FORMAT = "tumorcp_kits19_real_cuda_smoke_v1"
REPEATS = 3
BATCH_CANDIDATES = (2, 4)


class AuditedCPBatchLoader:
    """Observe original worker-local CP calls; never replace their operations."""
    def __init__(self, loader):
        self.loader = loader

    def __iter__(self):
        return self

    def set_thread_id(self, thread_id):
        self.loader.set_thread_id(thread_id)

    def __next__(self):
        import numpy as np
        original_load, original_cp = np.load, self.loader.aug_one_pair
        events, current = [], {}

        def observed_load(path, *args, **kwargs):
            value = original_load(path, *args, **kwargs)
            name = Path(os.fspath(path)).name
            if name.endswith("_tumor.npy"):
                if value.ndim != 2 or value.shape[1] != 2 or value.shape[0] == 0:
                    raise ValueError(f"Extracted tumors require nonempty (lesions,2) object cells: {path}")
                for index in range(len(value)):
                    image, seg = value[index, 0], value[index, 1]
                    if (not isinstance(image, np.ndarray) or not isinstance(seg, np.ndarray)
                            or image.ndim != 3 or image.shape != seg.shape
                            or image.dtype.kind not in "fi" or seg.dtype.kind not in "fiu"
                            or not np.isfinite(image).all() or not np.isfinite(seg).all()):
                        raise ValueError(f"Original real tumor crop is not numeric finite 3D CT/GT: {path}[{index}]")
                current["donor"] = name[:-len("_tumor.npy")]
            elif name.endswith((".npz", ".npy")):
                current["target"] = name[:-4]
            return value

        def observed_cp(*args, **kwargs):
            # The original signature is source_tumors, target_data,
            # target_organ_positions. It modifies only its copy-on-write CT.
            target = kwargs.get("target_data", args[1] if len(args) > 1 else None)
            if target is None:
                raise TypeError("Original CP target array missing")
            before = int(np.count_nonzero(target[1] == 2))
            result = original_cp(*args, **kwargs)
            after = int(np.count_nonzero(result[1] == 2))
            events.append(dict(target_case=current.get("target"), donor_case=current.get("donor"),
                               tumor_voxels_before=before, tumor_voxels_after=after,
                               actual_GT_changed=before != after))
            return result

        np.load, self.loader.aug_one_pair = observed_load, observed_cp
        try:
            batch = self.loader.generate_train_batch()
            batch["tumorcp_smoke_events"] = events
            return batch
        finally:
            np.load, self.loader.aug_one_pair = original_load, original_cp


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def state_sha(value) -> str:
    """Stable state digest, including tensor contents rather than object IDs."""
    import numpy as np
    import torch
    digest = hashlib.sha256()

    def visit(item):
        if torch.is_tensor(item):
            tensor = item.detach().cpu().contiguous()
            digest.update(f"tensor:{tensor.dtype}:{tuple(tensor.shape)}:".encode())
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, np.ndarray):
            digest.update(f"numpy:{item.dtype}:{item.shape}:".encode())
            digest.update(item.tobytes())
        elif isinstance(item, dict):
            digest.update(b"dict{")
            for key in sorted(item, key=lambda key: (type(key).__name__, repr(key))):
                visit(key)
                visit(item[key])
            digest.update(b"}")
        elif isinstance(item, (list, tuple)):
            digest.update(type(item).__name__.encode() + b"[")
            for child in item:
                visit(child)
            digest.update(b"]")
        else:
            digest.update(type(item).__name__.encode() + b":" + repr(item).encode() + b";")

    visit(value)
    return digest.hexdigest()


def rng_state():
    import numpy as np
    import torch
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch_cpu=torch.get_rng_state(), torch_cuda=torch.cuda.get_rng_state_all())


def restore_rng(state):
    import numpy as np
    import torch
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state_all(state["torch_cuda"])


def publish(path: Path, report: dict):
    """Replace only the report created exclusively by this diagnostic call."""
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def validate_prepared(root: Path) -> tuple[dict, dict]:
    prepared_path = root / "prepared.json"
    prepared = json.loads(prepared_path.read_text(encoding="utf-8"))
    if prepared.get("format") != "tumorcp_kits19_prepared_v1" or prepared.get("complete") is not True:
        raise ValueError("Complete, full-cohort KiTS19 preparation receipt required")
    runtime_path = root / "runtime" / "runtime_receipt.json"
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    if sha_file(runtime_path) != prepared["runtime_receipt_sha256"]:
        raise ValueError("Prepared runtime receipt SHA256 differs")
    for relative, digest in runtime["runtime_sha256"].items():
        path = (root / "runtime" / relative).resolve()
        if root / "runtime" not in path.parents:
            raise ValueError("Runtime receipt path escapes its isolated runtime")
        if sha_file(path) != digest:
            raise ValueError(f"Runtime bytes changed: {relative}")
    plans_path = Path(prepared["plans_file"]).resolve()
    if root not in plans_path.parents or sha_file(plans_path) != prepared["plans_sha256"]:
        raise ValueError("Prepared full-resolution plans are missing or changed")
    split_path = Path(prepared["split_file"]).resolve()
    if root not in split_path.parents or sha_file(split_path) != prepared["split_sha256"]:
        raise ValueError("Prepared original-code split is missing or changed")
    expected = {f"case_{index:05d}" for index in range(210)}
    train, heldout = prepared["train_ids"], prepared["heldout_ids"]
    if (len(train) != 168 or len(heldout) != 42 or len(set(train)) != 168
            or len(set(heldout)) != 42 or set(train) & set(heldout)
            or set(train) | set(heldout) != expected):
        raise ValueError("Prepared split must contain all 210 annotated cases, 168/42 disjoint")
    data_receipt = root / "data_verified.json"
    if sha_file(data_receipt) != prepared["data_verified_sha256"]:
        raise ValueError("Original NIfTI/label verification receipt changed")
    data = json.loads(data_receipt.read_text(encoding="utf-8"))
    if (data.get("format") != "tumorcp_kits19_verified_data_v1" or data.get("complete") is not True
            or len(data["records"]) != 210 or {row["case_id"] for row in data["records"]} != expected):
        raise ValueError("Complete original 210-case NIfTI verification receipt required")
    directory = Path(prepared["fullres_directory"]).resolve()
    if root not in directory.parents or not directory.is_dir():
        raise ValueError("Full-resolution data directory must stay inside the isolated root")
    from sklearn.model_selection import KFold
    keys = sorted(expected)
    indices_tr, indices_val = next(KFold(5, shuffle=True, random_state=12345).split(keys))
    if set(train) != {keys[i] for i in indices_tr} or set(heldout) != {keys[i] for i in indices_val}:
        raise ValueError("Prepared split differs from the preserved official-code fold 0")
    return prepared, runtime


def resource_snapshot():
    import psutil
    import torch
    process, memory = psutil.Process(), psutil.virtual_memory()
    properties = torch.cuda.get_device_properties(0)
    free, total = torch.cuda.mem_get_info(0)
    affinity = process.cpu_affinity() if hasattr(process, "cpu_affinity") else None
    return dict(GPU=properties.name, visible_cuda_devices=torch.cuda.device_count(),
                selected_cuda_device=torch.cuda.current_device(), total_VRAM_bytes=int(total),
                free_VRAM_bytes=int(free), allocated_VRAM_bytes=torch.cuda.memory_allocated(),
                reserved_VRAM_bytes=torch.cuda.memory_reserved(), CPU_logical=psutil.cpu_count(),
                CPU_affinity_cores=None if affinity is None else len(affinity),
                torch_intraop_threads=torch.get_num_threads(),
                torch_interop_threads=torch.get_num_interop_threads(),
                RAM_total_bytes=memory.total, RAM_available_bytes=memory.available,
                process_RSS_bytes=process.memory_info().rss,
                process_CPU_seconds=sum(process.cpu_times()[:2]))


def build_inputs(trainer, prepared, workers: int):
    """Use the original CP loader and image-augmentation transform on train only."""
    import numpy as np
    import torch
    from nnunet.training.dataloading.dataset_loading import DataLoader3D
    from nnunet.training.data_augmentation.data_augmentation_moreDA import get_moreDA_augmentation

    # Avoid load_dataset's eager loading of all 210 patients' property pickles.
    directory = Path(prepared["fullres_directory"]).resolve()
    train = sorted(prepared["train_ids"])
    trainer.dataset_tr = OrderedDict((key, OrderedDict(
        data_file=str(directory / f"{key}.npz"),
        properties_file=str(directory / f"{key}.pkl"))) for key in train)
    trainer.dataset_val = trainer.dataset_tr  # train-only monitor; never heldout.
    params = copy.deepcopy(trainer.data_aug_params)
    params["num_threads"] = workers
    # Keep the original cached-per-thread setting; this measures the real path.
    loader_args = dict(oversample_foreground_percent=trainer.oversample_foreground_percent,
                       pad_mode="constant", pad_sides=trainer.pad_all_sides, memmap_mode="c")
    train_loader = DataLoader3D(trainer.dataset_tr, trainer.basic_generator_patch_size,
                               trainer.patch_size, 2, False,
                               cp_configs=copy.deepcopy(trainer.cp_configs), **loader_args)
    # The validation augmenter is constructed but never started or consumed.
    monitor_loader = DataLoader3D(trainer.dataset_tr, trainer.patch_size, trainer.patch_size,
                                 2, False, cp_configs=None, **loader_args)
    augmenter, unused_monitor = get_moreDA_augmentation(
        AuditedCPBatchLoader(train_loader), monitor_loader, params["patch_size_for_spatialtransform"], params,
        deep_supervision_scales=trainer.deep_supervision_scales,
        seeds_train=list(range(12345, 12345 + workers * 2)),
        seeds_val=list(range(22345, 22345 + workers)),
        pin_memory=True, use_nondetMultiThreadedAugmenter=False)
    batches, cp_events = [], []
    start = time.perf_counter()
    try:
        for index in range(REPEATS * 2):
            before = time.perf_counter()
            value = next(augmenter)
            selected = [str(key) for key in value["keys"]]
            if len(selected) != 2 or not set(selected).issubset(train):
                raise ValueError("Smoke loader selected a non-training patient")
            data, target = value["data"], value["target"]
            for event in value["tumorcp_smoke_events"]:
                if event["target_case"] not in train or event["donor_case"] not in train:
                    raise ValueError("Original CP accessed a donor/target outside the training168 cohort")
                cp_events.append(event)
            if not torch.is_tensor(data) or not isinstance(target, (list, tuple)):
                raise TypeError("Original augmented deep-supervision tensor batch required")
            if list(data.shape) != [2, trainer.num_input_channels, *list(map(int, trainer.patch_size))]:
                raise ValueError("Augmenter changed the complete full-resolution network input")
            if not torch.isfinite(data).all() or not all(torch.isfinite(t).all() for t in target):
                raise ValueError("Actual CT/target batch contains nonfinite values")
            batches.append(dict(data=data.contiguous(), target=[t.contiguous() for t in target],
                                cases=selected, preparation_seconds=time.perf_counter() - before))
            print(f"DEBUG actual KiTS19 input {index + 1}/{REPEATS * 2} | "
                  f"cases={','.join(selected)} | shape={list(data.shape)}", flush=True)
    finally:
        # These APIs finish only the augmenters created by this process.
        augmenter._finish()
        unused_monitor._finish()
    cases = sorted({case for batch in batches for case in batch["cases"]})
    b2_cases = {case for index in range(REPEATS) for case in batches[index * 2]["cases"]}
    if len(cases) < 3 or len(b2_cases) < 3:
        raise RuntimeError("Random actual-CT diagnostic did not cover three distinct training patients; no fake replacement")
    if not cp_events or not any(event["actual_GT_changed"] for event in cp_events):
        raise RuntimeError("No observed actual TumorCP GT change in diagnostic inputs; no completion claim")
    raw_receipt = json.loads(Path(prepared["data_verified_file"]).read_text(encoding="utf-8"))
    raw_cases = {row["case_id"]: row for row in raw_receipt["records"]}
    bindings = []
    for case in sorted(set(cases) | {event["donor_case"] for event in cp_events}):
        row = dict(case_id=case, split="train", files={})
        for suffix in (".npz", ".pkl", ".npy", "_tumor.npy"):
            path = directory / f"{case}{suffix}"
            if suffix in (".npz", ".pkl", "_tumor.npy") and not path.is_file():
                raise FileNotFoundError(path)
            if path.is_file():
                row["files"][suffix] = dict(path=str(path), bytes=path.stat().st_size,
                                             sha256=sha_file(path))
        row["original_NIfTI"] = {}
        for name, digest_key in (("imaging.nii.gz", "image_sha256"),
                                 ("segmentation.nii.gz", "label_sha256")):
            path = Path(raw_receipt["data"]) / case / name
            digest = sha_file(path)
            if digest != raw_cases[case][digest_key]:
                raise ValueError(f"Actual training NIfTI changed after preparation: {path}")
            row["original_NIfTI"][name] = dict(path=str(path), sha256=digest)
        bindings.append(row)
    info = dict(representative_cases=cases, case_bindings=bindings,
                loader="official DataLoader3D + official moreDA, paper CP profile",
                loader_training_cohort=168, heldout_content_files_read=0,
                requested_worker_threads=workers, actual_augmentation_processes=workers * 2,
                official_cached_per_thread=params["num_cached_per_thread"] * 10,
                pin_memory=True, mmap="original copy-on-write .npy; otherwise original .npz; no source mutation",
                actual_CP_events=cp_events, actual_CP_GT_changes=sum(e["actual_GT_changed"] for e in cp_events),
                prepared_real_batches=len(batches), production_input_patch=list(map(int, trainer.patch_size)),
                basic_generator_patch=list(map(int, trainer.basic_generator_patch_size)),
                preparation_wall_seconds=time.perf_counter() - start,
                batch_content_sha256=[state_sha((b["data"], b["target"], b["cases"])) for b in batches])
    return batches, info


def expected_inactive_heads(trainer) -> list[str]:
    # Generic_UNet returns seg_outputs in reverse order. The original loss skips
    # its zero-weight lowest-resolution head; neither the loss nor model is edited.
    count = len(trainer.network.seg_outputs)
    prefixes = [f"seg_outputs.{count - 1 - output}." for output, weight
                in enumerate(trainer.ds_loss_weights) if float(weight) == 0.0]
    return [name for name, value in trainer.network.named_parameters()
            if value.requires_grad and any(name.startswith(prefix) for prefix in prefixes)]


def run_trial(trainer, batches, physical_batch: int, inactive: list[str], saved: dict,
              trial: dict) -> dict:
    import torch
    network, optimizer, scaler = trainer.network, trainer.optimizer, trainer.amp_grad_scaler
    trial.update(physical_batch=physical_batch, gradient_accumulation_steps=1,
                 effective_batch=physical_batch, requested_updates=REPEATS,
                 completed_updates=0, status="RUNNING", repeats=[], DEBUG=True)
    restore_trial(trainer, saved)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    network.train()
    torch.cuda.synchronize()
    start = time.perf_counter()
    trainable = {name: p for name, p in network.named_parameters() if p.requires_grad}
    for index in range(REPEATS):
        values = batches[index * 2:index * 2 + physical_batch // 2]
        data = torch.cat([b["data"] for b in values], dim=0).pin_memory()
        target = [torch.cat([b["target"][scale] for b in values], dim=0).pin_memory()
                  for scale in range(len(values[0]["target"]))]
        optimizer.zero_grad(set_to_none=True)
        events = [torch.cuda.Event(enable_timing=True) for _ in range(8)]
        update_start = time.perf_counter()
        events[0].record()
        data = data.cuda(non_blocking=True)
        target = [t.cuda(non_blocking=True) for t in target]
        events[1].record()
        events[2].record()
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=True):
            output = network(data)
            loss = trainer.loss(output, target)
        events[3].record()
        events[4].record()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        events[5].record()
        events[5].synchronize()
        missing = [name for name, p in trainable.items() if p.grad is None]
        unexpected = sorted(set(missing) - set(inactive))
        if unexpected:
            raise RuntimeError(f"Loss-disconnected trainable parameters: {unexpected}")
        active_gradients = [p.grad for p in trainable.values() if p.grad is not None]
        finite = torch.stack([torch.isfinite(g).all() for g in active_gradients]).all()
        if not bool(torch.isfinite(loss).item()) or not bool(finite.item()):
            raise FloatingPointError("Nonfinite actual CT loss or unscaled parameter gradient; no hidden retry")
        events[6].record()
        grad_norm = torch.nn.utils.clip_grad_norm_(network.parameters(), 12)
        scaler.step(optimizer)
        scaler.update()
        events[7].record()
        events[7].synchronize()
        wall = time.perf_counter() - update_start
        timings = {name: float(events[a].elapsed_time(events[b]) / 1000)
                   for name, a, b in (("transfer_seconds", 0, 1), ("forward_and_loss_seconds", 2, 3),
                                      ("backward_and_unscale_seconds", 4, 5),
                                      ("clip_optimizer_scaler_seconds", 6, 7))}
        record = dict(update=index + 1, cases=[case for b in values for case in b["cases"]],
                      input_shape=list(data.shape), target_shapes=[list(t.shape) for t in target],
                      loss=float(loss.detach().item()), unscaled_grad_norm_before_clip=float(grad_norm.item()),
                      active_parameter_gradients_finite=True,
                      expected_missing_gradients=missing, missing_gradients=unexpected,
                      update_wall_seconds=wall,
                      samples_per_compute_second=physical_batch / wall,
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved(), **timings)
        trial["repeats"].append(record)
        trial["completed_updates"] += 1
        print(f"DEBUG actual CUDA B{physical_batch} {index + 1}/{REPEATS} | "
              f"loss={record['loss']:.6f} | wall={wall:.3f}s | "
              f"peak={record['peak_allocated_bytes'] / 2**30:.3f} GiB", flush=True)
        del data, target, output, loss, events
    changed, unchanged = [], []
    for name, parameter in trainable.items():
        destination = changed if not torch.equal(parameter.detach().cpu(), saved["model"][name]) else unchanged
        destination.append(name)
    connected = sorted(set(trainable) - set(inactive))
    modules = sorted({name.split(".")[0] for name in connected})
    module_changes = {module: any(name.startswith(module + ".") for name in changed) for module in modules}
    if not all(module_changes.values()):
        raise RuntimeError(f"Active network module without actual optimizer weight change: {module_changes}")
    if any(name in changed for name in inactive):
        raise RuntimeError("A zero-weight original deep-supervision head unexpectedly changed")
    trial.update(status="PASS", wall_seconds=time.perf_counter() - start,
                 parameter_count_with_weight_change=len(changed), changed_parameters=changed,
                 unchanged_parameters=unchanged, active_module_weight_changes=module_changes,
                 resources_after=resource_snapshot())
    return trial


def restore_trial(trainer, saved):
    trainer.network.load_state_dict(saved["model"], strict=True)
    trainer.optimizer.load_state_dict(copy.deepcopy(saved["optimizer"]))
    trainer.amp_grad_scaler.load_state_dict(copy.deepcopy(saved["AMP"]))
    trainer.optimizer.zero_grad(set_to_none=True)
    trainer.network.train(saved["training_mode"])
    restore_rng(saved["RNG"])


def run(root: Path, workers: int, output: Path) -> dict:
    root, output = root.resolve(), output.resolve()
    if workers < 1:
        raise ValueError("An explicit positive augmentation worker count is required")
    if root not in output.parents:
        raise ValueError("DEBUG report must remain inside this isolated reproduction root")
    output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(format=FORMAT, DEBUG=True, completed=False, actual_CT=False, CUDA=False,
                  batch_candidates=list(BATCH_CANDIDATES), repeats_per_candidate=REPEATS,
                  model="Generic_UNet", physical_batch=2, production_batch=2,
                  production_training_started=False, full_training=False,
                  full_evaluation=False, quality_verified=False, root=str(root),
                  started_UTC=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  original_state_preserved=False, whole_model_preserved=False, trials=[])
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    trainer, saved, outer_rng, original_backends = None, None, None, None
    try:
        prepared, runtime = validate_prepared(root)
        report.update(prepared_sha256=sha_file(root / "prepared.json"),
                      plans_sha256=prepared["plans_sha256"],
                      runtime_receipt_sha256=prepared["runtime_receipt_sha256"],
                      full_training_cases=168, full_heldout_cases=42,
                      preprocessing_workers=int(prepared["workers"]),
                      actual_smoke_DA_setting=workers,
                      data_fraction="12 actual training patches for DEBUG; production cohort unchanged 168/42",
                      production_epochs=500, production_updates_per_epoch=250,
                      production_train_monitor_batches_per_epoch=50)
        os.environ.update(nnUNet_raw_data_base=str(root / "data"),
                          nnUNet_preprocessed=str(root / "preprocessed"),
                          RESULTS_FOLDER=str(root / "results"))
        runtime_directory = root / "runtime"
        sys.path.insert(0, str(runtime_directory))
        import numpy as np
        import torch
        import nnunet
        if Path(nnunet.__file__).resolve().parent != runtime_directory / "nnunet":
            raise RuntimeError("Imported nnunet is not the SHA-verified isolated TumorCP runtime")
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Exactly one allocated visible CUDA device required; no CPU, synthetic or arbitrary-device fallback")
        report["CUDA"] = True
        outer_rng = rng_state()
        original_backends = dict(deterministic=torch.backends.cudnn.deterministic,
                                 benchmark=torch.backends.cudnn.benchmark)
        report["resources_before"] = resource_snapshot()
        from nnunet.training.network_training.nnUNetTrainerV2_TumorCPPaper import nnUNetTrainerV2_TumorCPPaper
        diagnostic_directory = root / "diagnostics" / ("smoke_" + uuid.uuid4().hex)
        with Path(prepared["plans_file"]).open("rb") as stream:
            full_plans = pickle.load(stream)
        # Match upstream get_default_configuration('3d_fullres'): one-stage
        # plans use sample Dice, multi-stage plans use batch Dice.
        batch_dice = len(full_plans["plans_per_stage"]) > 1
        trainer = nnUNetTrainerV2_TumorCPPaper(
            prepared["plans_file"], 0, output_folder=str(diagnostic_directory),
            dataset_directory=str(Path(prepared["plans_file"]).parent),
            batch_dice=batch_dice, stage=int(prepared["fullres_stage"]), unpack_data=False,
            deterministic=False, fp16=True)
        trainer.initialize(training=False)
        trainer._maybe_init_amp()
        if trainer.network.__class__.__name__ != "Generic_UNet":
            raise TypeError("Original full Generic_UNet network required")
        if (trainer.max_num_epochs, trainer.num_batches_per_epoch,
                trainer.num_val_batches_per_epoch, trainer.batch_size) != (500, 250, 50, 2):
            raise ValueError("Production schedule or physical batch changed")
        named = {name: p for name, p in trainer.network.named_parameters() if p.requires_grad}
        optimized = {id(p) for group in trainer.optimizer.param_groups for p in group["params"]}
        if {id(p) for p in named.values()} != optimized:
            raise ValueError("Optimizer does not contain exactly every original trainable parameter")
        report.update(parameters=sum(p.numel() for p in trainer.network.parameters()),
                      trainable_parameters=sum(p.numel() for p in named.values()),
                      all_trainable_parameters_in_optimizer=True,
                      model_config=dict(base_features=int(trainer.base_num_features),
                                        input_channels=int(trainer.num_input_channels),
                                        classes=int(trainer.num_classes),
                                        pool_kernels=np.asarray(trainer.net_num_pool_op_kernel_sizes).tolist(),
                                        conv_kernels=np.asarray(trainer.net_conv_kernel_sizes).tolist(),
                                        convolutions_per_stage=int(trainer.conv_per_stage),
                                        deep_supervision_weights=np.asarray(trainer.ds_loss_weights).tolist()),
                      precision="AMP float16; original FP32 parameters and SGD optimizer; gradient clip12",
                      batch_Dice=batch_dice, cudnn_deterministic=False,
                      augmentation_seed_scope="fixed DEBUG worker seeds only; production policy unchanged",
                      optimizer="original SGD lr0.01 momentum0.99 Nesterov, original weight_decay",
                      production_checkpoint_opened=False, production_checkpoint_written=False,
                      diagnostic_directory=str(diagnostic_directory))
        saved = dict(model={name: value.detach().cpu().clone() for name, value in trainer.network.state_dict().items()},
                     optimizer=copy.deepcopy(trainer.optimizer.state_dict()),
                     AMP=copy.deepcopy(trainer.amp_grad_scaler.state_dict()),
                     RNG=rng_state(), training_mode=trainer.network.training)
        report["initial_state_sha256"] = state_sha(saved)
        inactive = expected_inactive_heads(trainer)
        report["expected_zero_weight_deep_supervision_parameters"] = inactive
        report["zero_weight_head_rule"] = "Original MultipleOutputLoss2 deliberately skips zero-weight DS heads; never connected artificially"
        publish(output, report)
        batches, inputs = build_inputs(trainer, prepared, workers)
        report.update(actual_CT=True, **inputs)
        for batch in BATCH_CANDIDATES:
            trial = dict(physical_batch=batch, status="STARTING", DEBUG=True)
            report["trials"].append(trial)
            try:
                run_trial(trainer, batches, batch, inactive, saved, trial)
            except torch.OutOfMemoryError as error:
                trial.update(status="OOM", error=str(error), production_batch_unchanged=2,
                             peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                             peak_reserved_bytes=torch.cuda.max_memory_reserved())
                publish(output, report)
                if batch == 2:
                    raise RuntimeError("Original physical batch2 OOM; production may not start") from error
            finally:
                restore_trial(trainer, saved)
                restored = dict(model=trainer.network.state_dict(), optimizer=trainer.optimizer.state_dict(),
                                AMP=trainer.amp_grad_scaler.state_dict(), RNG=rng_state(),
                                training_mode=trainer.network.training)
                if state_sha(restored) != report["initial_state_sha256"]:
                    raise RuntimeError("Independent smoke model/optimizer/AMP/RNG failed exact restoration")
                gc.collect()
                torch.cuda.empty_cache()
            publish(output, report)
        report.update(completed=True, whole_model_preserved=True, original_state_preserved=True,
                      missing_gradients=[], state_restoration_sha256=report["initial_state_sha256"],
                      resources_after=resource_snapshot(),
                      timing_definition="CUDA events measure transfer, forward+loss, backward+unscale, clip+SGD+scaler; update wall additionally includes finite checks and synchronization; input preparation and state audits separate",
                      throughput_scope="actual CUDA DEBUG compute after input preparation; not whole-epoch throughput",
                      batch_decision="Production physical batch remains paper-prescribed2; B4 is only measured diagnostic, no automatic scale change")
    except Exception as error:
        report.update(completed=False, error_type=type(error).__name__, error=str(error),
                      traceback=traceback.format_exc())
        raise
    finally:
        if trainer is not None and saved is not None:
            restore_trial(trainer, saved)
        if outer_rng is not None:
            import torch
            restore_rng(outer_rng)
            torch.backends.cudnn.deterministic = original_backends["deterministic"]
            torch.backends.cudnn.benchmark = original_backends["benchmark"]
            report["caller_RNG_restored"] = state_sha(rng_state()) == state_sha(outer_rng)
        report["finished_UTC"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        publish(output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.root, args.workers, args.output)
    print(f"DEBUG complete: actual KiTS19 / full Generic_UNet / production B2 unchanged | {args.output}", flush=True)
    print(json.dumps({key: result[key] for key in ("completed", "actual_CT", "CUDA", "parameters",
                                                 "representative_cases", "original_state_preserved",
                                                 "production_training_started")}), flush=True)


if __name__ == "__main__":
    main()
