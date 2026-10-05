"""Read-only C own-best loading and faithful anchor/curriculum support.

This is deliberately distinct from the published C common evaluator, which
rebuilt an observed-P/unobserved-U support bank. No optimizer is constructed,
no checkpoint is written and no saved support tensor is relabeled.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
import uuid


ROOT = Path(__file__).resolve().parents[1]
BEST_FORMAT = "crossed_C_best_own_model_v1"
TRAIN_FORMAT = "crossed_native_C_training_state_v1"


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def validate_metadata(manifest, contract, best, *, debug, completion=None,
                      latest_sha256=None, reports=()):
    """Metadata-only admission; byte hashes and strict tensors are checked later."""
    if type(debug) is not bool or any(not isinstance(x, dict) for x in (manifest, contract, best)):
        raise ValueError("Explicit C manifest, contract, best and DEBUG flag required")
    if (manifest.get("format") != "v17_crossed_training_identity_v1"
            or manifest.get("arm") != "C" or manifest.get("debug") is not debug
            or contract.get("format") != TRAIN_FORMAT or contract.get("arm") != "C"
            or contract.get("execution", {}).get("debug") is not debug):
        raise ValueError("C experiment/contract DEBUG identity differs")
    identity = contract.get("identity_sha256")
    if (not isinstance(identity, str) or len(identity) != 64
            or any(c not in "0123456789abcdef" for c in identity)
            or best.get("format") != BEST_FORMAT or best.get("arm") != "C"
            or best.get("debug") is not debug or best.get("run_identity_sha256") != identity):
        raise ValueError("C own-best is not bound to the signed training contract")
    selection = best.get("selection")
    if (not isinstance(selection, dict) or type(selection.get("epoch")) is not int
            or not 1 <= selection["epoch"] <= 40
            or type(selection.get("update")) is not int or selection["update"] < 1
            or not isinstance(selection.get("metric"), list) or len(selection["metric"]) != 5):
        raise ValueError("Complete own-task best selection receipt required")
    import math
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in selection["metric"]):
        raise ValueError("Own-task best metric must be finite")
    # transition_c_training.own_selection_key publishes the rounded original
    # (MRR, top1, margin, -ranking_loss) order plus a zero marker for the
    # explicitly absent six-view consistency objective. Preserve all five
    # producer fields; accepting a shorter key would lose selection semantics.
    if selection["metric"][4] != 0:
        raise ValueError("C own-best must preserve its absent-consistency zero marker")
    if not isinstance(best.get("state_dict"), dict) or not best["state_dict"]:
        raise ValueError("Actual own-best tensor state required")
    if not debug:
        if (not isinstance(completion, dict) or completion.get("arm") != "C"
                or completion.get("identity") != manifest or completion.get("debug") is not False
                or completion.get("full_training") is not True
                or completion.get("full_evaluation") is not True
                or completion.get("completed_epochs") != 40
                or completion.get("selected_own_epoch") != selection["epoch"]
                or completion.get("checkpoint_sha256") != latest_sha256):
            raise ValueError("Production C requires its bound completed40 own-best publication")
        completed = [r for r in reports if isinstance(r, dict)
                     and r.get("format") == TRAIN_FORMAT and r.get("arm") == "C"
                     and r.get("debug") is False and r.get("status") == "COMPLETE"
                     and r.get("run_identity_sha256") == identity]
        if not completed or any(r.get("best_own") != selection
                               or r.get("completed_epochs") != 40
                               or r.get("full_training") is not True
                               or r.get("actual_CUDA") is not True
                               or r.get("actual_raw_CT") is not True
                               or [x.get("epoch") for x in r.get("history", [])] != list(range(1, 41))
                               for r in completed):
            raise ValueError("Completed C report or full40 epoch history differs")
    return dict(run_identity_sha256=identity, selected_own_epoch=selection["epoch"],
                selected_own_update=selection["update"], selection=selection, debug=debug,
                complete40_verified=not debug, optimizer_updates=0,
                source_checkpoint_written=False,
                support_GT="source anchor index0 versus original curriculum comparison indices1..7",
                published_common_observed_bank_reused=False,
                same_as_published_C_common_evaluation=False)


def _source_path(recorded, baseline):
    """Resolve a recorded execution source under current helpers/frozen archive."""
    parts = PurePosixPath(str(recorded).replace("\\", "/")).parts
    if ".." in parts:
        raise ValueError("C execution source path traversal refused")
    for package in ("hiercp_v1x", "hiercp_v222", "l0_local_cnn", "l0_exploration"):
        if package in parts:
            return ROOT.joinpath(*parts[parts.index(package):])
    if "hiercp" in parts:
        return Path(baseline) / "source/v1.0" / Path(*parts[parts.index("hiercp"):])
    raise ValueError("Unknown C execution source path: " + str(recorded))


def _verify_sources(contract, baseline):
    saved = contract.get("source_sha256")
    if not isinstance(saved, dict) or not saved:
        raise ValueError("C training execution-source byte identities required")
    resolved = {}
    for recorded, digest in saved.items():
        path = _source_path(recorded, baseline).resolve(strict=True)
        if path in resolved and resolved[path] != digest:
            raise ValueError("Conflicting C execution source identities")
        if _sha(path) != digest:
            raise ValueError("C execution source differs: " + str(path))
        resolved[path] = digest
    return {str(p): digest for p, digest in resolved.items()}


@dataclass
class CCheckpointBundle:
    net: object
    train_loader: object
    config: dict
    inventory: dict
    scope: dict
    checkpoint: Path
    receipt: dict
    support_bank: dict
    budget: object

    def score_case(self, query, case_id):
        """One complete-case query on the original, freshly encoded C bank."""
        import torch
        if (self.net.training or torch.is_grad_enabled() or query.ndim != 2
                or query.shape[1] != 128 or not len(query)):
            raise ValueError("eval/no-grad complete [N,128] case required")
        if case_id not in self.inventory["split"]["inner_val"]:
            raise ValueError("Historical C scoring must use held-out native case identities")
        self.budget.check()
        with torch.autocast("cuda", enabled=self.config["training"]["amp"]):
            scores = self.net.upper.score(query, (case_id,), (len(query),))[0]
        if scores.shape != (len(query),) or not bool(torch.isfinite(scores).all()):
            raise ValueError("Historical C lost a finite candidate score")
        self.budget.check()
        return scores.float()


def load_c(experiment, baseline, native_run, budget, workers, resident_bytes, output,
           debug=False, debug_original_fixture=None, debug_support_fixture=None,
           debug_checkpoint=None):
    """Load only own-best model weights, then rebuild all original C support.

    The root evaluator owns CUDA selection, resources and current output files.
    DEBUG fixtures must be explicit. Production requires complete saved40 data.
    The support batch is inherited from measured C support calibration.
    """
    import torch
    from tools.run_v17_crossed_training import prepare_source, inputs
    from .transition_c_training import _digest, build_native_support_bank, validate_signed_cohort
    from .transition_model import CrossedModel
    from .transition_native_local import NativeCurriculumLoader, make_native_local

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Historical C requires one selected actual CUDA device")
    if type(debug) is not bool or type(workers) is not int or workers < 2:
        raise ValueError("Explicit DEBUG flag and parallel workers>=2 required")
    experiment = Path(experiment).resolve(strict=True)
    baseline = Path(baseline).resolve(strict=True)
    native_run = Path(native_run).resolve(strict=True)
    output = Path(output).resolve()
    if output == experiment or output.is_relative_to(experiment):
        raise ValueError("Historical evaluation output must be separate from C training")
    manifest_path = experiment / "manifest.json"
    contract_path = experiment / "training/c_training_contract.json"
    best_path = experiment / "training/checkpoint_best_own.pt"
    if debug_checkpoint is not None:
        if not debug:
            raise ValueError("Production cannot substitute a DEBUG checkpoint")
        best_path = Path(debug_checkpoint).resolve(strict=True)
    elif debug and (debug_original_fixture is None or debug_support_fixture is None):
        raise ValueError("DEBUG requires both actual original and support fixtures")
    manifest, contract = _json(manifest_path), _json(contract_path)
    metadata_hashes = {str(p): _sha(p) for p in (manifest_path, contract_path, best_path)}
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    completion = None; latest_sha = None; reports = []
    if not debug:
        complete_path = experiment / "training/training_complete.json"
        latest = experiment / "training/checkpoint_latest.pt"
        completion = _json(complete_path); latest_sha = _sha(latest)
        metadata_hashes.update({str(complete_path): _sha(complete_path), str(latest): latest_sha})
        for path in sorted((experiment / "training").glob("c_training_report_*.json")):
            reports.append(_json(path)); metadata_hashes[str(path)] = _sha(path)
        artifact = Path(completion.get("evaluation_artifact", "")).resolve(strict=True)
        if not artifact.is_relative_to(experiment / "training") or _sha(artifact) != completion.get("evaluation_sha256"):
            raise ValueError("C completed full evaluation artifact bytes/path changed")
    receipt = validate_metadata(manifest, contract, best, debug=debug, completion=completion,
                                latest_sha256=latest_sha, reports=reports)
    if _digest({k: v for k, v in contract.items() if k != "identity_sha256"}) != contract["identity_sha256"]:
        raise ValueError("C training contract content digest changed")
    inventory_path = native_run / "inventory/index.json"
    inventory = _json(inventory_path)
    if (inventory.get("debug") is not debug
            or inventory.get("local_cnn", {}).get("margin_mm") != 10
            or _sha(inventory_path) != manifest.get("native_inventory_sha256")):
        raise ValueError("Historical C native inventory/10mm/DEBUG bytes differ")
    source = baseline / "source/v1.0"
    _, scope = prepare_source(source)
    if scope["contract_sha256"] != manifest.get("scope"):
        raise ValueError("Historical C bounded10mm original source scope differs")
    source_hashes = _verify_sources(contract, baseline)
    request = SimpleNamespace(debug=debug, baseline=baseline,
                              debug_original_fixture=debug_original_fixture,
                              debug_support_fixture=debug_support_fixture)
    train_files, val_files, base, binding = inputs(request, source)
    if binding != manifest.get("original_curriculum_binding"):
        raise ValueError("Historical C original signed curriculum publication differs")
    config = contract["configuration"]
    normalized_base = json.loads(json.dumps(base, allow_nan=False))
    if {k:v for k,v in config.items() if k != "transition_runtime"} != normalized_base:
        raise ValueError("C source baseline configuration differs from its trained contract")
    runtime = config["transition_runtime"]
    train_cases = {s["case_id"] for s in train_files} if debug else set(binding["actual_signed_train_cases"])
    train_raw = [row for row in inventory["raw_records"] if row["case_id"] in train_cases]
    loader = NativeCurriculumLoader(train_files, train_raw, base, workers=workers,
        resident_bytes=resident_bytes, rss_bytes=budget.rss_bytes, debug=debug, pin_memory=True,
        sample_sha256=binding.get("original_sample_sha256"))
    cohort = validate_signed_cohort(loader, runtime, "train", runtime["training_case_ids"], debug=debug)
    if cohort != contract["signed_materialized_cohorts"]["train"]:
        raise ValueError("Rebuilt faithful C training cohort differs")
    net = CrossedModel(make_native_local(checkpointing=True, budget=budget, dropout=.1),
        arm="C", scope_contract=scope["contract_sha256"], dropout=.1, debug=debug).cuda()
    if net.initial_upper_sha256 != contract["upper_initial_sha256"]:
        raise ValueError("Historical C upper initialization recipe differs")
    net.load_state_dict(best["state_dict"], strict=True)
    net.eval(); budget.check()
    bank = build_native_support_bank(net, loader, batch_size=runtime["support_batch_size"],
        device=torch.device("cuda:0"), use_amp=config["training"]["amp"],
        manifest_sha256=contract["identity_sha256"], generation="historical_C_eval:" + uuid.uuid4().hex,
        epoch=best["selection"]["epoch"], training_case_ids=runtime["training_case_ids"],
        validation_case_ids=runtime["validation_case_ids"], budget=budget, debug=debug)
    net.upper.bind_support(bank)
    receipt.update(checkpoint=str(best_path), checkpoint_sha256=metadata_hashes[str(best_path)],
        strict_state_loaded=True, support_policy=bank["support_policy"],
        support_candidates=len(bank["embeddings"]), support_samples=len(loader),
        support_patient_case_ids=list(bank["patient_case_ids"]), support_batch_samples=runtime["support_batch_size"],
        support_reencoded_at_best_weights=True, query_only_L0_chunking=True,
        scoring="faithful C upper.score using original source-anchor/curriculum support",
        published_C_common_comparison="same query cohort; different support labels and bank",
        source_artifacts_sha256=metadata_hashes, execution_source_sha256=source_hashes,
        source_experiment_preserved=all(_sha(p) == digest for p, digest in metadata_hashes.items()))
    if not receipt["source_experiment_preserved"]:
        raise ValueError("Source C artifacts changed during read-only evaluation setup")
    return CCheckpointBundle(net, loader, config, inventory, scope, best_path, receipt, bank, budget)
