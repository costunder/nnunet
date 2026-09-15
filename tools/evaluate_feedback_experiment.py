"""Checkpoint-bound native inference and paired evaluation in a NEW workspace.

Resume imports only producer-committed predictions into a new generation. Raw
partial native outputs without case receipts are never certified or overwritten.
This does not train models or retrospectively certify historical predictions.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time
import uuid

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from custom_trainers.onlinecp_curriculum_contract import file_sha256, value_sha256, _json

FORMAT = "hiercp_feedback_evaluation_v1"
ARMS = ("full", "basic")


class NativeInferenceFailure(Exception):
    """Not RuntimeError: prevents nnU-Net's implicit GPU→CPU retry branch."""


def _commit_json(path, payload):
    from tools.online_cp_curriculum import _write_exclusive_json, _sync_directory
    path = Path(path)
    stage_root = path.parent / ".record_staging"
    _assert_output_path(path)
    _assert_output_path(stage_root)
    if stage_root.is_symlink():
        raise ValueError("Record staging must not be a symlink")
    stage_root.mkdir(exist_ok=True)
    stage = stage_root / (uuid.uuid4().hex + ".json")
    _write_exclusive_json(stage, payload)
    if value_sha256(_json(stage)) != value_sha256(payload):
        raise ValueError("Staged inference receipt changed")
    os.link(stage, path)  # atomic no-clobber; preserve stage even on failure
    _sync_directory(path.parent)


def _record(path):
    path = Path(path)
    if path.is_symlink() or any(parent.is_symlink() for parent in path.parents) or not path.is_file():
        raise ValueError(f"Missing/nonregular inference input: {path}")
    return {"path": str(path.resolve()), "sha256": file_sha256(path)}


def _input_stat(path):
    stat = path.stat()
    return {"device": stat.st_dev, "inode": stat.st_ino, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _input_alias(path):
    """Read-only input aliases may contain links; preserve their exact target."""
    path = Path(path).expanduser().absolute()
    if not path.is_file():
        raise ValueError(f"Missing inference input: {path}")
    resolved, before = str(path.resolve()), _input_stat(path)
    sha256 = file_sha256(path)
    if _input_stat(path) != before or str(path.resolve()) != resolved:
        raise ValueError(f"Inference input changed while hashing: {path}")
    return {"path": str(path), "resolved_path": resolved, "sha256": sha256, "stat": before}


def _verify_input_alias(record):
    if not isinstance(record, dict) or set(record) != {"path", "resolved_path", "sha256", "stat"}:
        raise ValueError("Malformed inference input alias")
    if not Path(record["path"]).is_absolute() or _input_alias(record["path"]) != record:
        raise ValueError(f"Inference input/alias changed: {record['path']}")
    return Path(record["path"])


def _input_record(path, source):
    """Bind a registered raw input to the original verified source payload."""
    record = _input_alias(path)
    original = dict(record) if Path(path).absolute() == Path(source).absolute() else _input_alias(source)
    mode = "symlink" if Path(path).is_symlink() else "regular"
    if (record["sha256"] != original["sha256"]
            or (mode == "symlink" and record["resolved_path"] != original["resolved_path"])):
        raise ValueError(f"Registered raw input differs from original: {path}")
    return {**record, "source": original, "materialization": mode}


def _verify_record(record):
    if not isinstance(record, dict) or set(record) not in (
            {"path", "sha256"}, {"path", "resolved_path", "sha256", "stat", "source", "materialization"}):
        raise ValueError("Malformed inference file record")
    path = Path(record["path"])
    if "source" in record:
        source = _verify_input_alias(record["source"])
        _verify_input_alias({key: record[key] for key in ("path", "resolved_path", "sha256", "stat")})
        mode = record["materialization"]
        if (mode not in {"regular", "symlink"} or path.is_symlink() != (mode == "symlink")
                or (mode == "symlink" and path.resolve() != source.resolve())
                or record["sha256"] != record["source"]["sha256"]):
            raise ValueError(f"Registered raw input/source changed: {path}")
        return path
    if (not path.is_absolute() or path.is_symlink() or any(parent.is_symlink() for parent in path.parents)
            or not path.is_file() or file_sha256(path) != record["sha256"]):
        raise ValueError(f"Inference input/artifact changed: {path}")
    return path


def _within(root, path):
    path = Path(path)
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Inference artifact escaped output: {path}")
    return path


def _runtime_witness(plan, expected):
    """Preserve the complete native inventory already certified by the launcher.

    Initial MODULES/core-file approval belongs to load_completed_experiment.
    This pure end witness checks identical names and bytes, without re-reading
    the large CP bank or approving a different runtime after training.
    """
    root = Path(plan["package_destination"])
    if not root.is_dir() or root.is_symlink() or any(p.is_symlink() for p in root.parents):
        raise ValueError("Inference runtime root changed or is redirected")
    actual = {}
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if "__pycache__" in relative.parts:
            continue
        if path.is_symlink():
            raise ValueError(f"Inference runtime link changed: {relative}")
        if path.is_file():
            actual[relative.as_posix()] = file_sha256(path)
        elif not path.is_dir():
            raise ValueError(f"Inference runtime contains a nonregular entry: {relative}")
    if actual != expected:
        raise ValueError("Inference runtime inventory changed")


def _project_source_inventory():
    # These are the executing project's modules, not arbitrary saved-plan paths.
    base = Path(__file__).resolve().parents[1]
    result = {}
    for name in ("hiercp", "custom_trainers", "tools"):
        folder = base / name
        if not folder.is_dir() or folder.is_symlink():
            raise ValueError(f"Missing/redirected inference project source: {folder}")
        for path in folder.rglob("*.py"):
            if "__pycache__" not in path.relative_to(folder).parts:
                record = _record(path)
                result[record["path"]] = record["sha256"]
    return result


def _execution_witness(plan, identity):
    _basic_origin_witness(identity)
    _runtime_witness(plan, identity["runtime_inventory"])
    if _project_source_inventory() != identity["project_source_inventory"]:
        raise ValueError("Inference project source changed during execution")


def _assert_output_path(path):
    path = Path(path)
    if path.is_symlink() or any(parent.is_symlink() for parent in path.parents):
        raise ValueError(f"Inference output ancestor redirect is forbidden: {path}")


def _input_witness(inputs):
    for records in inputs.values():
        for record in records.values():
            _verify_record(record)


def _verify_raw_input_contract(proof, inputs):
    contract = proof.get("raw_input_contract")
    rows = contract.get("source_cases") if isinstance(contract, dict) else None
    if not isinstance(rows, list) or not rows:
        raise ValueError("Missing verified raw input contract/source_cases")
    sources = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("case_id"), str):
            raise ValueError("Malformed verified raw input source")
        case_id = row["case_id"]
        if case_id in sources:
            raise ValueError(f"Duplicate verified raw input source: {case_id}")
        for key in ("image_sha256", "label_sha256"):
            digest = row.get(key)
            if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError(f"Malformed verified raw input hash: {case_id}/{key}")
        sources[case_id] = row
    for case_id, records in inputs.items():
        source = sources.get(case_id)
        if source is None or any(records[key]["sha256"] != source[digest_key] for key, digest_key in (
                ("image", "image_sha256"), ("ground_truth", "label_sha256"), ("registered_label", "label_sha256"))):
            raise ValueError(f"Inference input differs from certified raw source: {case_id}")
    return contract


def _basic_origin_witness(identity):
    """Recheck externally certified artifacts without importing historical code."""
    origin = identity.get("basic_origin")
    if origin is not None:
        for record in (origin["checkpoint"], origin["reuse_receipt"],
                       origin["bank_identity"]["files"]["plans"],
                       origin["bank_identity"]["files"]["dataset"],
                       {"path": str(Path(origin["experiment_root"]) / "execution_journal.json"),
                        "sha256": origin["training_journal_sha256"]}):
            _verify_record(record)


def _verified_basic_origin(proof, experiment_root):
    """Consume the launcher's verified reuse proof, never promote an old bank.

    The launcher certifies completed Basic training, Basic-consumed bank
    equivalence and native implementation equivalence. Prediction still uses
    the current verified runtime; this boundary retains the original evidence
    and checks its bytes, not an inferred Python-source compatibility claim.
    """
    if "basic_origin" not in proof:
        return None
    origin = proof["basic_origin"]
    required = {"format", "experiment_root", "checkpoint", "bank_identity",
                "runtime_inventory", "training_journal_sha256", "reuse_receipt"}
    if (not isinstance(origin, dict) or set(origin) != required
            or origin["format"] != "hiercp_verified_basic_origin_v1"):
        raise ValueError("Malformed verified Basic-origin proof")

    def valid_sha(value):
        return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)

    def artifact(record):
        if (not isinstance(record, dict) or set(record) != {"path", "sha256"}
                or not isinstance(record["path"], str) or not Path(record["path"]).is_absolute()
                or not valid_sha(record["sha256"])):
            raise ValueError("Malformed verified Basic-origin artifact")
        return _verify_record(record)

    original_root = origin["experiment_root"]
    if not isinstance(original_root, str) or not Path(original_root).is_absolute():
        raise ValueError("Basic-origin experiment root must be absolute")
    original_root = Path(original_root)
    current_root = Path(experiment_root).resolve()
    if (not original_root.is_dir() or original_root.is_symlink()
            or any(p.is_symlink() for p in original_root.parents)
            or original_root.resolve() == current_root):
        raise ValueError("Basic-origin must be a separate regular experiment root")
    checkpoint = artifact(origin["checkpoint"])
    receipt = artifact(origin["reuse_receipt"])
    if (checkpoint != Path(proof["checkpoints"]["basic"]).absolute()
            or not checkpoint.resolve().is_relative_to(original_root.resolve())
            or checkpoint.name != "checkpoint_final.pth" or checkpoint.parent.name != "fold_0"):
        raise ValueError("Basic-origin checkpoint path differs from the certified final checkpoint")
    if not receipt.resolve().is_relative_to(current_root):
        raise ValueError("Basic reuse receipt must belong to the new experiment root")
    inventory = origin["runtime_inventory"]
    if (not isinstance(inventory, dict) or not inventory
            or any(not isinstance(key, str) or not key or "\\" in key or ":" in key
                   or any(part in {"", ".", ".."} for part in key.split("/"))
                   or not valid_sha(digest) for key, digest in inventory.items())
            or not valid_sha(origin["training_journal_sha256"])):
        raise ValueError("Malformed original Basic runtime/journal identity")
    artifact({"path": str(original_root / "execution_journal.json"),
              "sha256": origin["training_journal_sha256"]})
    bank = origin["bank_identity"]
    if (not isinstance(bank, dict) or not bank or not isinstance(bank.get("files"), dict)
            or not {"plans", "dataset"}.issubset(bank["files"])):
        raise ValueError("Malformed original Basic bank identity")
    artifact(bank["files"]["plans"])
    artifact(bank["files"]["dataset"])
    return origin


def _checkpoint_metadata(path, arm, bank_identity, loader):
    from hiercp.feedback import tensor_state_sha256
    before = _record(path)
    checkpoint = loader(path)
    from tools.train_online_feedback import TRAINERS
    if (checkpoint.get("current_epoch") != 250 or checkpoint.get("trainer_name") != TRAINERS[arm]
            or checkpoint.get("onlinecp_curriculum_resume", {}).get("bank_identity") != bank_identity):
        raise ValueError(f"Final checkpoint epoch/arm/bank identity mismatch: {arm}")
    weights = checkpoint.get("network_weights")
    if not isinstance(weights, dict) or not weights:
        raise ValueError("Final checkpoint lacks actual network weights")
    result = {"checkpoint": before, "weights_sha256": tensor_state_sha256(weights),
              "trainer": checkpoint["trainer_name"],
              "configuration": checkpoint["init_args"]["configuration"],
              "mirroring_axes": (list(checkpoint["inference_allowed_mirroring_axes"])
                                 if checkpoint.get("inference_allowed_mirroring_axes") is not None else None)}
    del checkpoint, weights
    gc.collect()
    _verify_record(before)
    return result


def _native_predictor_factory(plan, device):
    # Select the preserved private runtime before importing any native package.
    runtime = Path(plan["package_destination"]).resolve()
    sys.path.insert(0, str(runtime.parent))
    for key, value in plan["env_updates"].items():
        os.environ[key] = str(value)
    import nnunetv2
    if Path(nnunetv2.__file__).resolve().parent != runtime:
        raise ValueError("A different nnU-Net runtime was already imported; use a new process")
    import torch
    if device == "cuda" and torch.cuda.device_count() != 1:
        raise ValueError("Exactly one allocated visible GPU is required; visibility was not changed")
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    class NoImplicitCPUFallbackPredictor(nnUNetPredictor):
        def _internal_predict_sliding_window_return_logits(self, *args, **kwargs):
            try:
                return super()._internal_predict_sliding_window_return_logits(*args, **kwargs)
            except RuntimeError as exc:
                raise NativeInferenceFailure(
                    "Native prediction failed; CPU retry/model reduction disabled. Partial outputs preserved.") from exc

    return NoImplicitCPUFallbackPredictor(tile_step_size=.5, use_gaussian=True, use_mirroring=True,
        perform_everything_on_device=device == "cuda", device=torch.device(device))


def _prediction_record(path, image_record, gt_record):
    import nibabel as nib
    import numpy as np
    from tools.online_eval_v2 import verify_geometry
    image = nib.load(str(image_record["path"]))
    label = nib.load(str(gt_record["path"]))
    prediction = nib.load(str(path))
    if len(prediction.shape) != 3 or len(image.shape) != 3 or len(label.shape) != 3:
        raise ValueError("Native CT/label/prediction must be 3D")
    verify_geometry(image, label, Path(gt_record["path"]))
    verify_geometry(image, prediction, Path(path))
    values = np.asanyarray(prediction.dataobj)
    if values.dtype.kind not in "ui" or values.min() < 0 or values.max() > 2:
        raise ValueError(f"Prediction is not an integer 0/1/2 segmentation: {path}")
    return _record(path)


def _verify_case(root, path, signature, arm, case_id, inputs, model, *, seen=frozenset()):
    path = _within(root, path)
    resolved = str(path.resolve())
    if resolved in seen:
        raise ValueError("Cyclic prediction import receipts")
    seen = seen | {resolved}
    row = _json(path)
    expected = {"format": "hiercp_prediction_case_v1", "signature": signature,
                "arm": arm, "case_id": case_id, "inputs": inputs}
    if any(row.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Prediction case receipt identity mismatch: {path}")
    loaded_path = _within(root, _verify_record(row["loaded_receipt"]))
    loaded = _json(loaded_path)
    if (loaded.get("format") != "hiercp_prediction_loaded_v1" or loaded.get("signature") != signature
            or loaded.get("arm") != arm or loaded.get("model") != model
            or loaded.get("loaded_weights_sha256") != model["weights_sha256"]):
        raise ValueError("Loaded-checkpoint evidence does not match prediction case")
    prediction = _within(root, _verify_record(row["prediction"]))
    if prediction.name != case_id + ".nii.gz":
        raise ValueError("Prediction basename does not match its case")
    actual = _prediction_record(prediction, inputs["image"], inputs["ground_truth"])
    if actual != row["prediction"]:
        raise ValueError("Prediction changed while validating geometry")
    if row.get("imported_from") is not None:
        origin = _within(root, _verify_record(row["imported_from"]))
        original = _verify_case(root, origin, signature, arm, case_id, inputs, model, seen=seen)
        if original["prediction"]["sha256"] != actual["sha256"]:
            raise ValueError("Imported prediction differs from its committed origin")
    return row


def _committed_cases(root, signature, arms, inputs):
    result = {arm: {} for arm in ARMS}
    for arm in ARMS:
        for path in sorted(root.glob(f"generations/*/{arm}/case_receipts/*.json")):
            if path.stem not in inputs:
                raise ValueError(f"Unexpected committed prediction case: {path}")
            row = _verify_case(root, path, signature, arm, path.stem, inputs[path.stem], arms[arm])
            previous = result[arm].get(path.stem)
            if previous and previous["row"]["prediction"]["sha256"] != row["prediction"]["sha256"]:
                raise ValueError("Conflicting committed predictions for the same inference identity")
            result[arm][path.stem] = {"path": path, "row": row}
    return result


def _verify_eval_completion(folder, expected_cohort, expected_predictions, expected_ground_truth):
    from tools.online_eval_provenance import verify_evaluation_contract
    marker = _json(folder / "completion.json")
    if marker.get("format") != "online_eval_completion_v1" or marker.get("complete") is not True:
        raise ValueError("Paired evaluator did not publish a complete result")
    for name, digest in marker["outputs"].items():
        path = _within(folder, folder / name)
        if file_sha256(path) != digest:
            raise ValueError(f"Paired evaluation output changed: {name}")
    summary = _json(folder / "summary.json")
    contract = summary["evaluation_contract"]
    verify_evaluation_contract(contract)
    if (contract["cohort"] != expected_cohort
            or contract["ground_truth"] != expected_ground_truth
            or marker["evaluation_contract_sha256"] != contract["contract_sha256"]
            or marker["summary_sha256"] != file_sha256(folder / "summary.json")):
        raise ValueError("Paired evaluation cohort or completion identity mismatch")
    for side, arm in (("basic", "basic"), ("hier", "full")):
        if contract["methods"][side]["predictions"] != expected_predictions[arm]:
            raise ValueError("Paired evaluator used different prediction bytes")
    return _record(folder / "completion.json")


def _verify_arm_completion(root, record, signature, arm, cases, inputs, model):
    path = _within(root, _verify_record(record))
    payload = _json(path)
    if (payload.get("format") != "hiercp_prediction_arm_complete_v1"
            or payload.get("signature") != signature or payload.get("arm") != arm
            or payload.get("cases") != cases or set(payload.get("case_receipts", {})) != set(cases)):
        raise ValueError("Producer arm completion identity/cohort mismatch")
    _within(root, _verify_record(payload["loaded_receipt"]))
    predicted, imported = payload.get("predicted_cases"), payload.get("imported_cases")
    if (not isinstance(predicted, list) or not isinstance(imported, list)
            or len(predicted + imported) != len(cases) or set(predicted) & set(imported)
            or set(predicted + imported) != set(cases)):
        raise ValueError("Producer case accounting is incomplete or duplicated")
    hashes = {}
    for case_id in cases:
        receipt = _within(root, _verify_record(payload["case_receipts"][case_id]))
        row = _verify_case(root, receipt, signature, arm, case_id, inputs[case_id], model)
        if row["loaded_receipt"] != payload["loaded_receipt"]:
            raise ValueError("Producer case uses another loaded-checkpoint generation")
        hashes[case_id] = row["prediction"]["sha256"]
    return hashes


def execute(args, *, load_experiment=None, checkpoint_loader=None, predictor_factory=None, evaluate_fn=None):
    root = Path(args.output_dir).expanduser().absolute()
    if root.is_symlink() or any(parent.is_symlink() for parent in root.parents):
        raise ValueError("Evaluation output root/parents must not be symlink redirects")
    if root.exists() and not args.resume:
        raise FileExistsError("Evaluation output exists; use a NEW directory or verified --resume")
    from tools import run_feedback_experiment as launcher
    from tools import online_eval_v2 as evaluator
    load_experiment = launcher.load_completed_experiment if load_experiment is None else load_experiment
    checkpoint_loader = launcher._checkpoint_payload if checkpoint_loader is None else checkpoint_loader
    predictor_factory = _native_predictor_factory if predictor_factory is None else predictor_factory
    evaluate_fn = evaluator.evaluate if evaluate_fn is None else evaluate_fn
    proof = load_experiment(Path(args.experiment_root).resolve())
    plan = proof["plan"]
    cases = list(proof["validation_case_ids"])
    if not cases or len(cases) != len(set(cases)) or any(
            not isinstance(c, str) or not c or c in {".", ".."} or any(x in c for x in "/\\:\x00") for c in cases):
        raise ValueError("Invalid/duplicate validation case IDs")
    config = _json(Path(plan["project_root"]) / "config/nnunet.json")
    workers = {"preprocessing": args.num_processes_preprocessing,
               "segmentation_export": args.num_processes_segmentation_export}
    workers = {key: config["preprocess"]["processes"] if value is None else value for key, value in workers.items()}
    if any(type(value) is not int or value <= 0 for value in workers.values()):
        raise ValueError("Inference worker counts must be positive explicit integers")
    dataset = proof["bank_identity"]["dataset_name"]
    raw = Path(plan["env_updates"]["nnUNet_raw"]) / dataset
    inputs = {}
    for case_id in cases:
        image = _input_record(raw / "imagesTr" / (case_id + "_0000.nii.gz"),
                              Path(plan["medical_root"]) / "Data/image" / (case_id + "_0000.nii.gz"))
        source_label = Path(plan["medical_root"]) / "Data/labels" / (case_id + ".nii.gz")
        label = _input_record(source_label, source_label)
        raw_label = _input_record(raw / "labelsTr" / (case_id + ".nii.gz"), Path(label["path"]))
        if label["sha256"] != raw_label["sha256"]:
            raise ValueError(f"Ground truth differs from the registered raw dataset: {case_id}")
        inputs[case_id] = {"image": image, "ground_truth": label, "registered_label": raw_label}
    raw_input_contract = _verify_raw_input_contract(proof, inputs)
    ground_truth_hashes = {c: inputs[c]["ground_truth"]["sha256"] for c in cases}
    basic_origin = _verified_basic_origin(proof, args.experiment_root)
    models = {arm: _checkpoint_metadata(proof["checkpoints"][arm], arm,
              basic_origin["bank_identity"] if arm == "basic" and basic_origin is not None else proof["bank_identity"], checkpoint_loader)
              for arm in ARMS}
    for arm in ARMS:
        model_folder = Path(proof["checkpoints"][arm]).parent.parent
        if models[arm]["configuration"] != config["dataset"]["configuration"]:
            raise ValueError("Checkpoint configuration differs from the completed experiment")
        models[arm]["plans"] = _record(model_folder / "plans.json")
        models[arm]["dataset"] = _record(model_folder / "dataset.json")
        if (_json(models[arm]["plans"]["path"]) != _json(proof["bank_identity"]["files"]["plans"]["path"])
                or _json(models[arm]["dataset"]["path"]) != _json(proof["bank_identity"]["files"]["dataset"]["path"])):
            raise ValueError("Inference plans/dataset differ from the verified bank preprocessing")
        if arm == "basic" and basic_origin is not None:
            original_files = basic_origin["bank_identity"]["files"]
            if any(_json(models[arm][key]["path"]) != _json(original_files[key]["path"])
                   for key in ("plans", "dataset")):
                raise ValueError("Basic inference plans/dataset differ from its original bank")
    identity = {"format": FORMAT, "experiment_root": str(Path(args.experiment_root).resolve()),
        "training_journal_sha256": proof["journal_sha256"], "runtime_inventory": proof["runtime_inventory"],
        "bank_identity": proof["bank_identity"], "models": models, "cases": cases, "inputs": inputs,
        "inference": {"device": args.device, "tile_step_size": .5, "use_gaussian": True,
            "use_mirroring": True, "checkpoint_name": "checkpoint_final.pth", "use_folds": [0],
            "save_probabilities": False, "num_parts": 1, "part_id": 0, "workers": workers,
            "runtimeerror_policy": "fail_without_implicit_cpu_retry"},
        "evaluator_version": evaluator.VERSION, "producer_sha256": file_sha256(Path(__file__)),
        "project_source_inventory": _project_source_inventory(), "raw_input_contract": raw_input_contract}
    if basic_origin is not None:
        identity["basic_origin"] = basic_origin
    identity = json.loads(json.dumps(identity, sort_keys=True, allow_nan=False))
    models, inputs = identity["models"], identity["inputs"]
    _execution_witness(plan, identity)
    signature = value_sha256(identity)
    root = Path(args.output_dir).expanduser().absolute()
    for folder in (raw, Path(plan["package_destination"]), Path(plan["medical_root"]) / "Data",
                   Path(plan["env_updates"]["nnUNet_results"])):
        if root.resolve().is_relative_to(folder.resolve()):
            raise ValueError("Evaluation output must not be inside source data/runtime/training results")
    if basic_origin is not None and root.resolve().is_relative_to(Path(basic_origin["experiment_root"]).resolve()):
        raise ValueError("Evaluation output must not be inside the original Basic experiment")
    if root.is_symlink():
        raise ValueError("Evaluation output root must not be a symlink")
    if root.exists():
        if not args.resume:
            raise FileExistsError("Evaluation output exists; use a NEW directory or verified --resume")
        if _json(root / "identity.json") != identity:
            raise ValueError("Evaluation identity changed; existing results preserved")
    else:
        if args.resume:
            raise FileNotFoundError("Cannot resume a missing evaluation workspace")
        root.mkdir(parents=True, exist_ok=False)
        _commit_json(root / "identity.json", identity)
    committed = _committed_cases(root, signature, models, inputs)
    if (root / "completion.json").exists():
        completed = _json(root / "completion.json")
        if completed.get("format") != FORMAT or completed.get("signature") != signature or completed.get("complete") is not True:
            raise ValueError("Malformed feedback evaluation completion")
        for arm in ARMS:
            if set(committed[arm]) != set(cases):
                raise ValueError("Completed evaluation lacks committed prediction cases")
        if set(completed.get("producer_completions", {})) != set(ARMS):
            raise ValueError("Feedback evaluation completion lacks both producer proofs")
        expected = {arm: _verify_arm_completion(root, completed["producer_completions"][arm], signature,
                    arm, cases, inputs, models[arm]) for arm in ARMS}
        eval_marker = _verify_record(completed["evaluation_completion"])
        _within(root, eval_marker)
        _verify_eval_completion(eval_marker.parent, cases, expected, ground_truth_hashes)
        _input_witness(inputs)
        _execution_witness(plan, identity)
        return root / "completion.json"
    generation = root / "generations" / uuid.uuid4().hex
    _assert_output_path(generation)
    generation.mkdir(parents=True, exist_ok=False)
    prediction_dirs, prediction_hashes, producer_receipts = {}, {}, {}
    print(f"[INFERENCE] cohort={len(cases)}; Full and Basic; device={args.device}; workers={workers}; CPU={os.cpu_count()}", flush=True)
    for arm in ARMS:
        arm_root = generation / arm
        output = arm_root / "predictions"
        receipts = arm_root / "case_receipts"
        output.mkdir(parents=True)
        receipts.mkdir()
        model = models[arm]
        predictor = predictor_factory(plan, args.device)
        predictor.initialize_from_trained_model_folder(str(Path(proof["checkpoints"][arm]).parent.parent),
            use_folds=(0,), checkpoint_name="checkpoint_final.pth")
        from hiercp.feedback import tensor_state_sha256
        if (len(predictor.list_of_parameters) != 1
                or tensor_state_sha256(predictor.list_of_parameters[0]) != model["weights_sha256"]):
            raise ValueError("Native predictor loaded different checkpoint weights")
        network = getattr(predictor.network, "_orig_mod", predictor.network)
        # Native versions may defer this load until predict_logits. Load the
        # already-hash-verified sole fold explicitly; strict keys/shapes retain
        # native architecture validation before any prediction is certified.
        network.load_state_dict(predictor.list_of_parameters[0], strict=True)
        if tensor_state_sha256(network.state_dict()) != model["weights_sha256"]:
            raise ValueError("Native network state differs from the verified loaded weights")
        if predictor.trainer_name != model["trainer"]:
            raise ValueError("Native predictor restored a different trainer")
        actual_axes = list(predictor.allowed_mirroring_axes) if predictor.allowed_mirroring_axes is not None else None
        if (predictor.tile_step_size != .5 or predictor.use_gaussian is not True or predictor.use_mirroring is not True
                or actual_axes != model["mirroring_axes"]
                or predictor.dataset_json != _json(model["dataset"]["path"])
                or predictor.plans_manager.plans != _json(model["plans"]["path"])):
            raise ValueError("Native predictor inference settings/plans/dataset differ from the recorded contract")
        for field in ("plans", "dataset"):
            _verify_record(model[field])
        _verify_record(model["checkpoint"])
        loaded_path = arm_root / "loaded.json"
        network_summary = {"class": type(network).__qualname__, "representation": str(network),
            "parameters": sum(parameter.numel() for parameter in network.parameters()),
            "trainable_parameters": sum(parameter.numel() for parameter in network.parameters() if parameter.requires_grad)}
        print(f"[MODEL] {arm}: {model['trainer']}; parameters={network_summary['parameters']}", flush=True)
        _commit_json(loaded_path, {"format": "hiercp_prediction_loaded_v1", "signature": signature,
            "arm": arm, "model": model, "loaded_weights_sha256": model["weights_sha256"],
            "inference": identity["inference"], "network": network_summary})
        remaining = [case_id for case_id in cases if case_id not in committed[arm]]
        native_started = time.perf_counter()
        if remaining:
            predictor.predict_from_files([[inputs[c]["image"]["path"]] for c in remaining], str(output),
                save_probabilities=False, overwrite=False,
                num_processes_preprocessing=workers["preprocessing"],
                num_processes_segmentation_export=workers["segmentation_export"],
                folder_with_segs_from_prev_stage=None, num_parts=1, part_id=0)
        native_elapsed = time.perf_counter() - native_started if remaining else None
        _verify_record(model["checkpoint"])
        if (tensor_state_sha256(predictor.list_of_parameters[0]) != model["weights_sha256"]
                or tensor_state_sha256(network.state_dict()) != model["weights_sha256"]):
            raise ValueError("Native predictor weights changed during inference")
        evaluator.verify_prediction_inventory(output, remaining)
        _execution_witness(plan, identity)
        for case_id in cases:
            for record in inputs[case_id].values():
                _verify_record(record)
            prediction = output / (case_id + ".nii.gz")
            origin = None
            if case_id in committed[arm]:
                previous = committed[arm][case_id]
                origin = _record(previous["path"])
                original = _verify_record(previous["row"]["prediction"])
                os.link(original, prediction)
            record = _prediction_record(prediction, inputs[case_id]["image"], inputs[case_id]["ground_truth"])
            _commit_json(receipts / (case_id + ".json"), {"format": "hiercp_prediction_case_v1",
                "signature": signature, "arm": arm, "case_id": case_id, "inputs": inputs[case_id],
                "prediction": record, "loaded_receipt": _record(loaded_path), "imported_from": origin})
        evaluator.verify_prediction_inventory(output, cases)
        prediction_hashes[arm] = {c: file_sha256(output / (c + ".nii.gz")) for c in cases}
        prediction_dirs[arm] = output
        producer = arm_root / "completion.json"
        _commit_json(producer, {"format": "hiercp_prediction_arm_complete_v1", "signature": signature,
            "arm": arm, "cases": cases, "loaded_receipt": _record(loaded_path),
            "case_receipts": {c: _record(receipts / (c + ".json")) for c in cases},
            "predicted_cases": remaining, "imported_cases": [c for c in cases if c not in remaining],
            "native_inference_executed": bool(remaining), "native_prediction_seconds": native_elapsed,
            "native_cases_per_second": len(remaining) / native_elapsed if native_elapsed else None})
        producer_receipts[arm] = _record(producer)
        del predictor, network
        gc.collect()
    eval_args = evaluator.parser().parse_args([])
    eval_args.project = str(plan["project_root"])
    eval_args.medical_root = str(plan["medical_root"])
    eval_args.paired_root = str(plan.get("reuse_paired_root") or Path(plan["run_root"]) / "paired")
    eval_args.online_root = str(Path(plan["run_root"]) / "online")
    eval_args.outer_fold, eval_args.dataset_id = plan["outer_fold"], plan["dataset_id"]
    eval_args.basic_trainer, eval_args.hier_trainer = models["basic"]["trainer"], models["full"]["trainer"]
    eval_args.basic_validation, eval_args.hier_validation = str(prediction_dirs["basic"]), str(prediction_dirs["full"])
    eval_args.output = str(generation / "paired_evaluation")
    _input_witness(inputs)
    evaluated = Path(evaluate_fn(eval_args))
    _within(generation, evaluated)
    eval_completion = _verify_eval_completion(evaluated, cases, prediction_hashes, ground_truth_hashes)
    for arm in ARMS:
        _verify_record(models[arm]["checkpoint"])
        for field in ("plans", "dataset"):
            _verify_record(models[arm][field])
        actual = _verify_arm_completion(root, producer_receipts[arm], signature, arm, cases, inputs, models[arm])
        if actual != prediction_hashes[arm]:
            raise ValueError("Producer prediction hashes changed during paired evaluation")
    _execution_witness(plan, identity)
    _input_witness(inputs)
    _commit_json(root / "completion.json", {"format": FORMAT, "complete": True, "signature": signature,
        "training_executed": False, "checkpoint_linkage": "producer_verified_actual_loaded_weights",
        "producer_completions": producer_receipts, "evaluation_completion": eval_completion})
    return root / "completion.json"


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--experiment-root", required=True)
    result.add_argument("--output-dir", required=True)
    result.add_argument("--resume", action="store_true")
    result.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    result.add_argument("--num-processes-preprocessing", type=int)
    result.add_argument("--num-processes-segmentation-export", type=int)
    return result


def main():
    args = parser().parse_args()
    try:
        output = execute(args)
    except Exception:
        print(f"[EVALUATION FAILED] Preserved inputs, training checkpoints, and partial outputs: {args.output_dir}. "
              "Use --resume only with unchanged verified inputs, or select a NEW output directory. No training was restarted.",
              file=sys.stderr, flush=True)
        raise
    print(f"[EVALUATION COMPLETE] {output}", flush=True)


if __name__ == "__main__":
    main()
