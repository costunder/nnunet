"""Read-only collection of completed v1.x ranking results.

Checkpoint tensors are loaded on CPU for metadata inspection only.  This module
never constructs a model, runs forward/backward, prepares CT, or changes weights.
Metrics describe original source-anchor/curriculum ranking, not observed P/U,
segmentation Dice, or ground-truth CP suitability.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

from .contracts import (ContractError, STAGES, TARGET_CONTRACT, V1_ARCHIVE_SHA256,
                        canonical_hash, config_diff, make_run_contract,
                        resolve_execution_config, validate_resume,
                        validate_stage_config)
from .experiment import (digest, freeze_execution, load_suite, read, write_new,
                         preparation_root, execution_reference, sampler_contract,
                         execution_is_resolved)


BASE_ARCHITECTURE = "hiercp_source_content_population_metric_v5"
CANDIDATE_CONTRACT = "v1_original_anchor_curriculum8_from_pool128"
MASK_CONTRACT = "original_full_source_footprint_and_original_eligibility"
SELECTION_KEYS = ("mrr", "acc", "margin", "ranking", "consistency")


def _checkpoint(path: Path) -> dict[str, Any]:
    import torch
    # No mmap: the server may use a shared filesystem with stale mapping handles.
    value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, dict):
        raise ContractError(f"Invalid checkpoint dictionary: {path}")
    return value


def _same(actual: Any, expected: Any, label: str) -> None:
    if config_diff(expected, actual):
        raise ContractError(f"Completed result changed {label}")


def _selection(value: Any) -> dict[str, float]:
    if not isinstance(value, dict) or set(value) != set(SELECTION_KEYS):
        raise ContractError("Best selection lacks the original five finite metrics")
    result = {}
    for name in SELECTION_KEYS:
        metric = value[name]
        if isinstance(metric, bool) or not isinstance(metric, (int, float)) or not math.isfinite(metric):
            raise ContractError(f"Non-finite best metric: {name}")
        if name in ("mrr", "acc") and not 0 <= metric <= 1:
            raise ContractError(f"Best metric outside [0,1]: {name}")
        result[name] = float(metric)
    return result


def _stage_state(payload: Mapping[str, Any], stage: str, sampling=None) -> None:
    if sampling is not None:
        from .sampling_runtime import validate_checkpoint_identity
        try:
            payload = validate_checkpoint_identity(payload, sampling)
        except ValueError as error:
            raise ContractError(f'Completed checkpoint sampler mismatch: {error}') from error
    state = payload.get("state_dict")
    if not isinstance(state, Mapping) or not state:
        raise ContractError("Checkpoint has no model state")
    marker = state.get("_v1x_stage_revision")
    if stage == "v1.0":
        if marker is not None:
            raise ContractError("Original v1.0 checkpoint contains another stage marker")
        expected = BASE_ARCHITECTURE
    else:
        revision = tuple(STAGES).index(stage)
        if marker is None or not hasattr(marker, "numel") or marker.numel() != 1 or int(marker.item()) != revision:
            raise ContractError("Checkpoint model belongs to another v1.x stage")
        expected = BASE_ARCHITECTURE + f"|v1x_{stage}_r1"
    _same(payload.get("architecture_version"), expected, "model architecture/stage")
    connectivity = payload.get("gradient_connectivity", {})
    if (connectivity.get("format") != "hiercp_gradient_connectivity_v1"
            or connectivity.get("verified") is not True
            or connectivity.get("missing_parameters") != []
            or not connectivity.get("connected_parameters")
            or connectivity.get("expected_parameter_count") != len(connectivity["connected_parameters"])
            or connectivity.get("connected_parameter_count") != len(connectivity["connected_parameters"])
            or len(set(connectivity["connected_parameters"])) != len(connectivity["connected_parameters"])):
        raise ContractError("Completed checkpoint lacks verified parameter connectivity")


def collect_result(experiment: str | Path, stage: str, output: str | Path | None = None) -> dict[str, Any]:
    """Collect full 40-epoch results; output is exclusive or identical on re-read."""
    experiment = Path(experiment).resolve()
    if stage not in STAGES:
        raise ContractError(f"Unknown result stage: {stage}")
    manifest = load_suite(experiment)
    shared, results = preparation_root(experiment, manifest), experiment / "results" / stage
    reference = execution_reference(experiment, manifest)
    sampling = sampler_contract(experiment, stage, manifest)
    resolved_execution = execution_is_resolved(manifest, stage)
    if (shared / "prepare.lock").exists() or (results / "run.lock").exists():
        raise ContractError("Collection requires a stable completed run, not an active writer")
    paths = {"best": results / "checkpoint_best.pt", "last": results / "checkpoint_best.last.pt"}
    if not all(path.is_file() for path in paths.values()):
        raise ContractError("Both original best and last-epoch checkpoint files are required")
    immutable_hashes = {key: digest(path) for key, path in paths.items()}
    payloads = {key: _checkpoint(path) for key, path in paths.items()}
    best, last = payloads["best"], payloads["last"]
    if not (shared / "execution_lock.json").is_file():
        raise ContractError("Baseline execution lock must already exist before read-only collection")
    lock = freeze_execution(experiment)  # Existing lock is verified, never created here.
    stage_config = read(experiment / "configs" / f"{stage}.json")
    resolved = resolve_execution_config(stage_config, lock)
    validate_stage_config(resolved, stage, execution_lock=lock,
                          preparation_admission=manifest.get('preparation_admission'))
    if resolved_execution:
        _same(read(results / "resolved_config.json"), resolved, "resolved execution settings")
    launch = read(results / "launch_contract.json")
    plain_launch = {k: v for k, v in launch.items() if k != "contract_sha256"}
    _same(launch.get("contract_sha256"), canonical_hash(plain_launch), "launch hash")
    _same(launch.get("stage"), stage, "launch stage")
    _same(launch.get("source"), dict(stage=stage, files=manifest["stages"][stage]["source_hashes"]), "source snapshot")
    expected_config = resolved if resolved_execution else stage_config
    _same(launch.get("config"), expected_config, "launch configuration")
    expected_execution_lock = lock if resolved_execution else None
    _same(launch.get("execution_lock"), expected_execution_lock, "baseline physical batch lock")
    _same(launch.get('preparation_admission'), manifest.get('preparation_admission'),
          'explicit ROI resource admission')
    if sampling is not None:
        _same(launch.get('sampling_contract'), sampling, 'selected sampler identity')
    elif 'sampling_contract' in launch:
        raise ContractError('Legacy result cannot silently adopt a sampler')
    cache = {name: digest(shared / "cache" / name) for name in ("config.json", "index.json", "complete.json")}
    cache["prototype_bank_sha256"] = digest(shared / "prototype_bank.pt")
    _same(launch.get("cache"), cache, "shared cache/prototype contents")
    evaluation = dict(split_sha256=manifest["split_sha256"], target_contract=TARGET_CONTRACT,
                      candidate_contract=CANDIDATE_CONTRACT, mask_contract=MASK_CONTRACT)
    _same(launch.get("evaluation"), evaluation, "target/split/candidate/mask definitions")
    if resolved_execution:
        binding = make_run_contract(stage, resolved, cache_identity=cache,
                                   source_identity=launch["source"], evaluation_identity=evaluation,
                                   physical_batch_size=lock["selected_batch_size"], debug=False,
                                   execution_lock=lock, sampling_contract=sampling,
                                   preparation_admission=manifest.get('preparation_admission'))
        validate_resume(binding, read(results / "run_contract.json"))
    index = read(shared / "cache/index.json")
    entries = index.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ContractError("Completed cache has no indexed samples")
    for entry in entries:
        if (not isinstance(entry, dict) or entry.get("split") not in ("train", "val")
                or entry.get("case_id") not in manifest["split"][entry["split"]]
                or Path(entry.get("path", "")).name != entry.get("path")
                or not entry.get("path") or not isinstance(entry.get("artifact_sha256"), str)
                or len(entry["artifact_sha256"]) != 64):
            raise ContractError("Invalid cache sample identity or cohort mapping")
    names = [entry["path"] for entry in entries]
    if len(set(names)) != len(names):
        raise ContractError("Duplicate cache sample identities")
    train_entries = sorted((e for e in entries if e["split"] == "train"), key=lambda e: e["path"])
    val_entries = sorted((e for e in entries if e["split"] == "val"), key=lambda e: e["path"])
    if not train_entries or not val_entries:
        raise ContractError("No complete train/validation ranking cohort")
    config = resolved
    expected_policy = dict(format="hiercp_fixed_validation_v1",
                           epoch=config["training"]["fixed_validation_epoch"],
                           checkpoint_order=["mrr", "acc", "margin", "-ranking", "-consistency"],
                           metric_precision=config["training"]["checkpoint_metric_precision"])
    for label, payload in payloads.items():
        if payload.get("training_complete") is not True or payload.get("target_epochs") != 40:
            raise ContractError("Partial or DEBUG training is not a full v1.x result")
        completed = payload.get("completed_epoch") if label == "best" else payload.get("epoch")
        if completed != 40:
            raise ContractError("The authoritative complete 40-epoch state is required")
        if payload.get("method") != "hiercp-full" or payload.get("framework") != "torch_geometric":
            raise ContractError("Unexpected source model/checkpoint framework")
        _stage_state(payload, stage, sampling)
        _same(payload.get("model_kwargs"), config["model"], "model configuration")
        _same(payload.get("graph_config"), config["graph"], "graph configuration")
        _same(payload.get("ct_clip"), tuple(float(v) for v in config["ct_clip"]), "CT intensity window")
        _same(payload.get("geometry_contract"), config["graph"]["geometry_contract"], "geometry contract")
        _same(payload.get("upper_feature_policy"), "source_content_observed_ct_population_v4", "upper feature policy")
        prototype_cases = payload.get("prototype_training_cases")
        if (not isinstance(prototype_cases, list) or len(prototype_cases) != len(set(prototype_cases))
                or set(prototype_cases) != set(manifest["split"]["train"])):
            raise ContractError("Prototype bank is not bound to the complete training-only cohort")
        if not isinstance(payload.get("prototype_fingerprint"), str) or not payload["prototype_fingerprint"]:
            raise ContractError("Missing prototype identity")
        _same(payload.get("cache_publication"), {"config_sha256": cache["config.json"],
              "index_sha256": cache["index.json"], "complete_sha256": cache["complete.json"]}, "cache publication")
        _same(payload.get("validation_policy"), expected_policy, "fixed validation/selection order")
        signature = payload.get("training_signature", {})
        required = dict(format="hiercp_training_signature_v1", run_mode="production", target_epochs=40,
                        seed=42, batch_size=lock["selected_batch_size"], num_workers=lock["selected_num_workers"],
                        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
                        train_cache_files=[e["path"] for e in train_entries],
                        val_cache_files=[e["path"] for e in val_entries],
                        batch_setting=lock["selected_batch_size"] if resolved_execution else stage_config["training"]["batch_size"],
                        worker_setting=lock["selected_num_workers"] if resolved_execution else stage_config["training"]["num_workers"],
                        gradient_accumulation_setting=config["training"]["gradient_accumulation_steps"],
                        target_effective_batch_size=config["training"]["target_effective_batch_size"],
                        resolved_effective_batch_size=lock["selected_batch_size"] * config["training"]["gradient_accumulation_steps"],
                        consistency_weight=float(config["training"]["consistency_weight"]),
                        optimizer=dict(name="AdamW", lr=float(config["training"]["lr"]),
                                       weight_decay=float(config["training"]["weight_decay"]),
                                       fused=config["training"]["fused_optimizer"]),
                        scheduler=dict(name="CosineAnnealingLR", t_max=40),
                        amp=config["training"]["amp"], grad_clip=float(config["training"]["grad_clip"]),
                        deterministic=config["runtime"]["deterministic"],
                        allow_tf32=config["runtime"]["allow_tf32"],
                        curriculum={key: config["training"][key] for key in
                            ("easy_epochs", "inter_epochs", "intra_epochs", "model_mine_start_epoch",
                             "semi_hard_low_percentile", "semi_hard_high_percentile", "cross_entropy_weight",
                             "pairwise_weight", "ordinal_weight", "mined_weight")})
        _same({key: signature.get(key) for key in required}, required, "training/evaluation signature")
        if signature.get("ablation_mode", "full") != "full":
            raise ContractError("Ablation/subset checkpoint is not a complete v1.x result")
        calibration = payload.get("preflight_calibration", {})
        if (calibration.get("format") != "hiercp_preflight_calibration_v2"
                or calibration.get("selected_batch_size", calibration.get("batch_setting")) != lock["selected_batch_size"]
                or calibration.get("selected_num_workers", calibration.get("worker_setting")) != lock["selected_num_workers"]):
            raise ContractError("Saved checkpoint disagrees with shared execution calibration")
        identity = calibration.get("identity", {})
        _same(identity.get("seed"), 42, "calibration seed")
        _same(Path(identity.get("checkpoint_path", "")).resolve(), paths["best"], "calibration checkpoint")
        _same(Path(identity.get("cache_dir", "")).resolve(), shared / "cache", "calibration cache")
    _same(last.get("format"), "hiercp_training_state_v1", "authoritative last-epoch state")
    _same(Path(last.get("best_checkpoint", "")).resolve(), paths["best"], "best checkpoint association")
    selection = _selection(best.get("best_selection"))
    _same(_selection(best.get("selection")), selection, "best/selection metrics")
    _same(_selection(last.get("best_selection")), selection, "last state's best metrics")
    _same(best.get("best_mrr"), selection["mrr"], "best MRR")
    _same(last.get("best_mrr"), selection["mrr"], "last state's best MRR")
    epoch = best.get("best_epoch")
    if type(epoch) is not int or not 1 <= epoch <= 40:
        raise ContractError("Invalid best epoch")
    _same(best.get("epoch"), epoch, "best checkpoint epoch")
    _same(last.get("best_epoch"), epoch, "last state's best epoch")
    _same(last.get("training_signature"), best.get("training_signature"), "best/last training identity")
    _same(last.get("prototype_fingerprint"), best.get("prototype_fingerprint"), "best/last prototype identity")
    _same(best.get("train_files"), len(train_entries), "train denominator")
    _same(best.get("val_files"), len(val_entries), "validation denominator")
    proof_files = manifest["stages"]["v1.0"]["source_hashes"]
    evaluation.update(source_archive_sha256=V1_ARCHIVE_SHA256,
                      evaluation_source_sha256=canonical_hash({name: proof_files[name] for name in
                          ("hiercp/pipeline.py", "hiercp/curriculum.py", "hiercp/data.py", "hiercp/loss.py")}),
                      candidate_ids_sha256=canonical_hash(dict(validation_entries=val_entries,
                          fixed_view_epoch=expected_policy["epoch"], seed=42,
                          total_candidates=config["cache"]["total_candidates"],
                          pool_size=config["cache"]["candidate_pool_size"])),
                      physical_batch_size=lock["selected_batch_size"],
                      effective_batch_size=lock["selected_batch_size"] * config["training"]["gradient_accumulation_steps"],
                      target_epochs=40,
                      denominator=dict(validation_samples=len(val_entries),
                          validation_cases=len({entry["case_id"] for entry in val_entries}),
                          candidates_per_sample=config["cache"]["total_candidates"]))
    after_hashes = {key: digest(path) for key, path in paths.items()}
    _same(after_hashes, immutable_hashes, "checkpoint bytes during read-only collection")
    report = dict(format="hiercp_v1x_completed_result_v1", stage=stage, complete=True, debug=False,
                  target_contract=TARGET_CONTRACT, evaluation=evaluation,
                  metrics=dict(mrr=selection["mrr"], top1=selection["acc"], margin=selection["margin"],
                               rank_loss=selection["ranking"], consistency=selection["consistency"]),
                  best_epoch=epoch, completed_epochs=40, measured_new_evaluation=False,
                  checkpoints={key: dict(path=str(path), sha256=immutable_hashes[key]) for key, path in paths.items()},
                  launch_contract_sha256=launch["contract_sha256"],
                  result_scope="Recorded original fixed-validation source-anchor curriculum ranking; not CP efficacy",
                  nnunet_started=False, training_started=False, model_forward_executed=False,
                  checkpoint_metadata_device="cpu", checkpoint_bytes_preserved=True)
    if sampling is not None:
        report['sampling_contract'] = sampling
        report['local_sampling'] = manifest['sampling_contract']
        report['native_reference_experiment'] = str(reference)
    output_path = Path(output) if output is not None else results / "comparison_report.json"
    if output_path.exists():
        _same(read(output_path), report, "existing collected report; no overwrite permitted")
    else:
        write_new(output_path, report)
    return report
