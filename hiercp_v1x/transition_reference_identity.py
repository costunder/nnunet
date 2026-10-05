"""Stdlib identity separation for independently advancing native references.

A reference is never a weight source for the crossed run. Its current saved
cursor and checkpoint bytes are recorded evidence, while the crossed run's
geometry, native inventory, recipe, resources and execution sources are fixed.
Only explicitly named, validated endpoint fields can advance here. Unknown
fields, experiment metadata, and all non-reference manifest fields stay exact.
"""
from __future__ import annotations

import copy
import json
import re

FORMAT = "v17_native_reference_endpoint_identity_comparison_v1"
REFERENCE_FORMAT = "v17_native_reference_admission_v1"
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_TRANSIENT = frozenset((
    "checkpoint_sha256", "checkpoint_bytes", "checkpoint_recorded_content_sha256",
    "saved_cursor", "optimizer_epochs_completed", "validated_epochs_completed",
    "native_training_complete",
))
_PHASE_CHECKS = frozenset((
    "initial_phase_before_updates", "post_optimization_cursor_complete",
    "final_phase_completed_epochs", "final_phase_query_cursor_reset",
    "optimization_epoch_below_target", "post_optimization_epoch_below_target",
))
_REQUIRED = frozenset((
    "format", "admitted", "debug", "inventory_sha256", "checkpoint",
    "checkpoint_sha256", "checkpoint_bytes", "weights_transferred",
    "tensor_values_exported", "neural_execution", "validation_scope",
    "checkpoint_content_hash_recomputed", "checkpoint_recorded_content_sha256",
    "target_epochs", "optimization_steps_per_epoch", "saved_cursor",
    "optimizer_epochs_completed", "validated_epochs_completed",
    "native_training_complete", "recorded_source_files", "checks",
    "required_fields_missing", "failed_bindings", "experiment_sha256", "metadata_files",
))
_PHASES = frozenset(("initial_validation", "initial_memory", "optimization",
                    "refresh_memory", "validation", "final_memory", "complete"))
_PHASE_ORDER = {name: order for order, name in enumerate((
    "initial_validation", "initial_memory", "optimization", "refresh_memory",
    "validation", "final_memory", "complete"))}


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _exact(old, new):
    """JSON identity keeps bool, integer and floating-point values distinct."""
    try:
        return _canonical(old) == _canonical(new)
    except (TypeError, ValueError):
        return False


def structured_differences(old, new, path=""):
    """Return every changed JSON leaf with no truncation or missing-value alias."""
    if _exact(old, new):
        return []
    if isinstance(old, dict) and isinstance(new, dict):
        result = []
        for key in sorted(old.keys() | new.keys()):
            location = f"{path}.{key}" if path else str(key)
            if key in old and key in new:
                result.extend(structured_differences(old[key], new[key], location))
            else:
                result.append(dict(path=location, old_present=key in old, new_present=key in new,
                    old=copy.deepcopy(old.get(key)), new=copy.deepcopy(new.get(key))))
        return result
    if isinstance(old, list) and isinstance(new, list) and len(old) == len(new):
        return [row for index, (a, b) in enumerate(zip(old, new))
                for row in structured_differences(a, b, f"{path}[{index}]")]
    return [dict(path=path or "$", old_present=True, new_present=True,
                 old=copy.deepcopy(old), new=copy.deepcopy(new))]


def _reference_errors(value):
    if not isinstance(value, dict):
        return ["reference must be a JSON mapping"]
    errors = []
    if _REQUIRED - value.keys():
        return ["required reference fields missing: " + ", ".join(sorted(_REQUIRED - value.keys()))]
    if value["format"] != REFERENCE_FORMAT or value["admitted"] is not True or value["debug"] is not False:
        errors.append("validated production reference admission required")
    for key in ("weights_transferred", "tensor_values_exported", "neural_execution", "checkpoint_content_hash_recomputed"):
        if value[key] is not False:
            errors.append(key + " must remain False")
    if value["required_fields_missing"] != [] or value["failed_bindings"] != []:
        errors.append("reference admission has missing or failed bindings")
    checks = value["checks"]
    if not isinstance(checks, dict) or not checks or any(passed is not True for passed in checks.values()):
        errors.append("all recorded reference checks must be strictly True")
    for key in ("inventory_sha256", "checkpoint_sha256", "checkpoint_recorded_content_sha256", "experiment_sha256"):
        if not isinstance(value[key], str) or not _SHA.fullmatch(value[key]):
            errors.append(key + " must be a SHA256")
    for key in ("checkpoint_bytes", "optimization_steps_per_epoch", "recorded_source_files"):
        if type(value[key]) is not int or value[key] <= 0:
            errors.append(key + " must be a positive integer")
    if type(value["target_epochs"]) is not int or value["target_epochs"] != 40:
        errors.append("target_epochs must remain the validated full40 recipe")
    if not isinstance(value["checkpoint"], str) or not value["checkpoint"]:
        errors.append("selected checkpoint path is required")
    metadata = value["metadata_files"]
    if not isinstance(metadata, dict) or set(metadata) != {"experiment", "inventory", "execution_contract", "learning_schedule"}:
        errors.append("all four exact reference metadata files are required")
    else:
        for name, file in metadata.items():
            if (not isinstance(file, dict) or set(file) != {"path", "bytes", "sha256"}
                    or not isinstance(file["path"], str) or not file["path"]
                    or type(file["bytes"]) is not int or file["bytes"] <= 0
                    or not isinstance(file["sha256"], str) or not _SHA.fullmatch(file["sha256"])):
                errors.append("invalid bound metadata file: " + name)
        if isinstance(metadata.get("inventory"), dict) and metadata["inventory"].get("sha256") != value["inventory_sha256"]:
            errors.append("metadata inventory SHA differs from reference inventory")
        if isinstance(metadata.get("experiment"), dict) and metadata["experiment"].get("sha256") != value["experiment_sha256"]:
            errors.append("metadata experiment SHA differs from reference experiment")
    cursor = value["saved_cursor"]
    valid = (isinstance(cursor, dict) and set(cursor) == {"epoch", "step", "phase", "next_batch", "batch"}
        and type(cursor["epoch"]) is int and 0 <= cursor["epoch"] <= 40
        and type(cursor["step"]) is int and cursor["step"] >= 0
        and isinstance(cursor["phase"], str) and cursor["phase"] in _PHASES
        and type(cursor["next_batch"]) is int and type(value["optimization_steps_per_epoch"]) is int
        and 0 <= cursor["next_batch"] <= value["optimization_steps_per_epoch"]
        and type(cursor["batch"]) is int and cursor["batch"] >= 2)
    if not valid:
        errors.append("invalid validated native saved cursor")
    else:
        steps = value["optimization_steps_per_epoch"]
        if cursor["step"] != cursor["epoch"] * steps + cursor["next_batch"]:
            errors.append("saved step does not match epoch and query cursor")
        if cursor["phase"] in ("initial_validation", "initial_memory") and (cursor["epoch"], cursor["step"], cursor["next_batch"]) != (0, 0, 0):
            errors.append("initial reference cursor already contains updates")
        if cursor["phase"] in ("refresh_memory", "validation") and cursor["next_batch"] != steps:
            errors.append("post-optimization reference cursor is incomplete")
        if cursor["phase"] in ("final_memory", "complete") and (cursor["epoch"], cursor["next_batch"]) != (40, 0):
            errors.append("final reference cursor is incomplete")
        if cursor["phase"] in ("optimization", "refresh_memory", "validation") and cursor["epoch"] >= 40:
            errors.append("active native epoch is beyond target")
        optimizer_epochs = cursor["epoch"] + int(cursor["phase"] in ("optimization", "refresh_memory", "validation") and cursor["next_batch"] == steps)
        if type(value["optimizer_epochs_completed"]) is not int or value["optimizer_epochs_completed"] != optimizer_epochs:
            errors.append("optimizer epoch count differs from saved cursor")
        if type(value["validated_epochs_completed"]) is not int or value["validated_epochs_completed"] != cursor["epoch"]:
            errors.append("validated epoch count differs from saved cursor")
        if type(value["native_training_complete"]) is not bool or value["native_training_complete"] != (cursor["phase"] == "complete"):
            errors.append("training completion differs from saved phase")
    return errors


def compare_reference(old, new):
    """Compare reference evidence, accepting only named endpoint progression.

    Historical DEBUG references are compared exactly. Production receipt
    validation is independent of equality, so matching invalid receipts cannot
    authorize a drift. Changed experiment.json metadata remains rejected until
    there is an independently verified immutable experiment-request proof.
    """
    differences = structured_differences(old, new)
    if isinstance(old, dict) and isinstance(new, dict) and (old.get("debug") is True or new.get("debug") is True):
        return dict(format=FORMAT, admissible=not differences, exact=not differences,
                    differences=differences, rejected_fields=[] if not differences else ["DEBUG reference must remain exact"])
    rejected = ["old: " + issue for issue in _reference_errors(old)]
    rejected.extend("new: " + issue for issue in _reference_errors(new))
    if isinstance(old, dict) and isinstance(new, dict):
        for key in sorted(old.keys() | new.keys()):
            if key in _TRANSIENT and key in old and key in new:
                continue
            if key == "checks" and isinstance(old.get(key), dict) and isinstance(new.get(key), dict):
                previous = {name: passed for name, passed in old[key].items() if name not in _PHASE_CHECKS}
                current = {name: passed for name, passed in new[key].items() if name not in _PHASE_CHECKS}
                if not _exact(previous, current):
                    rejected.append("checks: non-phase reference bindings changed")
            elif key not in old or key not in new or not _exact(old[key], new[key]):
                rejected.append(key + ": immutable or unknown reference field changed")
        # The native contract fixes the measured physical batch. Endpoint
        # progression must not hide a batch migration or an older checkpoint.
        if not _reference_errors(old) and not _reference_errors(new):
            previous, current = old["saved_cursor"], new["saved_cursor"]
            if not _exact(previous["batch"], current["batch"]):
                rejected.append("saved_cursor.batch: native physical batch changed")
            if current["epoch"] < previous["epoch"] or current["step"] < previous["step"]:
                rejected.append("saved_cursor: native training progress moved backwards")
            if (current["epoch"] == previous["epoch"]
                    and _PHASE_ORDER[current["phase"]] < _PHASE_ORDER[previous["phase"]]):
                rejected.append("saved_cursor.phase: native phase moved backwards within its epoch")
    return dict(format=FORMAT, admissible=not rejected, exact=not differences,
                differences=differences, rejected_fields=rejected)


def bind_manifest_reference(previous, current):
    """Keep the existing manifest's frozen endpoint proof on a safe continuation.

    The caller must record the new live admission separately as invocation
    evidence. This function changes no manifest file and never mutates inputs.
    """
    if not isinstance(previous, dict) or not isinstance(current, dict):
        raise ValueError("Crossed manifest reference comparison requires two JSON mappings")
    before = {key: value for key, value in previous.items() if key != "native_experiment_binding"}
    after = {key: value for key, value in current.items() if key != "native_experiment_binding"}
    other = structured_differences(before, after)
    reference = compare_reference(previous.get("native_experiment_binding"), current.get("native_experiment_binding"))
    if other or not reference["admissible"]:
        detail = dict(format=FORMAT, manifest_differences=structured_differences(previous, current),
                      nonreference_differences=other, reference=reference)
        raise ValueError("Crossed manifest identity differs: " + _canonical(detail))
    return copy.deepcopy(previous)
