"""Verified patient preparation sharing; legacy metadata is never rewritten."""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
import uuid
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi

from custom_trainers.onlinecp_curriculum_contract import file_sha256, value_sha256
from hiercp.common import (CasePaths, distance_to_mask_mm,
                           verify_loaded_case_source_signatures)
from hiercp.feedback_resources import snapshot_guard
from hiercp.feedback_storage import _entry_lock, _fsync_directory
from hiercp.region import (_region_cache_metadata, graph_config_budget_compatible,
                           load_or_build_patient_regions, load_patient_regions)


@dataclass
class PreparedFeedbackPatient:
    case: object
    components: np.ndarray
    regions: object
    distance: np.ndarray
    preparation: dict


def _source_content(signature):
    required = ("kind", "integrity_format", "size", "sha256")
    if (not isinstance(signature, dict) or any(key not in signature for key in required)
            or signature["kind"] != "file" or signature["integrity_format"] != "sha256_v1"):
        raise ValueError("Patient region alias requires verified file-content provenance")
    return {key: signature[key] for key in required}


def region_alias_compatible(actual, expected):
    """Ignore ONLY path/mtime aliases and the upper-only patient graph contract.

    Region partitions do not consume patient_graph_contract. All other graph
    settings remain exact except the already documented ROI-budget increase.
    """
    physical = {"storage", "crop_start", "crop_stop", "artifact_sha256"}
    if (not isinstance(actual, dict) or not set(expected) <= set(actual)
            or set(actual) - set(expected) not in (set(), physical)):
        return False
    for name in expected:
        if name in {"image", "label"}:
            if _source_content(actual[name]) != _source_content(expected[name]):
                return False
        elif name == "graph_config":
            old, new = dict(actual[name]), dict(expected[name])
            old.pop("patient_graph_contract", None)
            new.pop("patient_graph_contract", None)
            if old != new and not graph_config_budget_compatible(old, new):
                return False
        elif actual[name] != expected[name]:
            return False
    return True


def _alias_receipt(root, case_id, source_root, actual, expected):
    record = {"format": "hiercp_feedback_patient_content_alias_v1", "case_id": case_id,
              "source_region_root": str(source_root.resolve()),
              "original_metadata": actual, "current_metadata": expected,
              "meaning_projection": "exact_except_file_path_mtime_and_upper_patient_graph_contract"}
    root.mkdir(parents=True, exist_ok=True)
    path = root / (value_sha256(record) + ".json")
    if path.is_symlink():
        raise ValueError("Patient alias receipt cannot be a symlink")
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != record:
            raise ValueError("Immutable patient alias receipt changed")
    else:
        staged = root / ("pending." + uuid.uuid4().hex + ".json")
        with staged.open("x", encoding="utf-8") as handle:
            json.dump(record, handle, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(staged, path)
        except FileExistsError:
            if json.loads(path.read_text(encoding="utf-8")) != record:
                raise ValueError("Concurrent patient alias receipt differs")
    return path


def _new_regions(case, expected, *, liver_label, tumor_label, config, ct_clip, seed, root):
    """Publish a native region directory through one immutable pointer.

    A failed native PID-temporary directory remains in its own generation; it
    cannot block a later attempt and is never silently deleted or adopted.
    """
    root.mkdir(parents=True, exist_ok=True)
    key = value_sha256(expected)
    pointer = root / (key + ".json")
    def read():
        if pointer.is_symlink():
            raise ValueError("Shared region pointer cannot be a symlink")
        record = json.loads(pointer.read_text(encoding="utf-8"))
        if set(record) != {"format", "generation", "metadata_sha256"} or record["format"] != "feedback_native_regions_v1":
            raise ValueError("Shared region pointer schema differs")
        relative = Path(record["generation"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Shared region pointer escaped its generation")
        generation = root / relative
        if (generation.is_symlink() or not generation.resolve().is_relative_to(root.resolve())
                or file_sha256(generation / "metadata.json") != record["metadata_sha256"]):
            raise ValueError("Shared region pointer content/path changed")
        regions, actual = load_patient_regions(generation, mmap=True)
        if not region_alias_compatible(actual, expected):
            raise ValueError("Shared native region meaning differs")
        return regions
    with _entry_lock(root / (key + ".lock")):
        if pointer.exists() or pointer.is_symlink():
            return read()
        namespace = root / "generations" / uuid.uuid4().hex
        namespace.mkdir(parents=True, exist_ok=False)
        regions = load_or_build_patient_regions(
            case, liver_label=liver_label, tumor_label=tumor_label, config=config,
            ct_clip=ct_clip, seed=seed, cache_dir=namespace, overwrite=False, mmap=True)
        destination = namespace / case.paths.case_id
        # Native SHA-checked verification before the pointer is made visible.
        verified, actual = load_patient_regions(destination, mmap=True)
        if not region_alias_compatible(actual, expected):
            raise ValueError("New native region publication changed its meaning")
        del verified
        # These files are owned by this fresh generation, never the original
        # shared region cache. Persist native payload and metadata before the
        # commit pointer can make them visible to another reader.
        for artifact in sorted(destination.iterdir()):
            if artifact.is_symlink() or not artifact.is_file():
                raise ValueError("New native region generation contains an unexpected artifact")
            with artifact.open("r+b") as handle:
                handle.flush()
                os.fsync(handle.fileno())
        _fsync_directory(destination)
        record = {"format": "feedback_native_regions_v1", "generation": destination.relative_to(root).as_posix(),
                  "metadata_sha256": file_sha256(destination / "metadata.json")}
        staged = namespace / "commit.json"
        with staged.open("x", encoding="utf-8") as handle:
            json.dump(record, handle, sort_keys=True, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(namespace)
        _fsync_directory(namespace.parent)
        os.link(staged, pointer)
        _fsync_directory(root)
        return regions


def prepare_feedback_patient(case_id, paths, *, liver_label, tumor_label, config,
                             ct_clip, seed, cache_root, region_cache_dir=None):
    from hiercp.common import load_case
    image_path, label_path = paths
    image_header = nib.load(str(image_path))
    label_header = nib.load(str(label_path))
    if image_header.shape != label_header.shape or len(image_header.shape) != 3:
        raise ValueError("Feedback patient raw image/label shapes differ")
    # Known simultaneously resident raw image/label/component/distance/tumor
    # arrays only. Native full-size measured case waves add their actual peak;
    # this lower bound is never advertised as a total graph-memory guarantee.
    lower = int(np.prod(image_header.shape)) * (4 + 2 + 4 + 8 + 1)
    admission = snapshot_guard(lower, phase="feedback_patient_raw_array_lower_bound")
    case = load_case(CasePaths(case_id, image_path, label_path))
    expected = _region_cache_metadata(case, liver_label=liver_label, tumor_label=tumor_label,
                                     config=config, ct_clip=ct_clip, seed=seed)
    alias = None
    if region_cache_dir is not None:
        original = Path(region_cache_dir)
        if not original.is_absolute():
            raise ValueError("A shared patient region cache must be an explicit absolute path")
        source_root = original / case_id
        if source_root.is_symlink():
            raise ValueError("Shared patient region cache cannot be a symlink")
        if source_root.exists():
            regions, actual = load_patient_regions(source_root, mmap=True)
            if not region_alias_compatible(actual, expected):
                raise ValueError(f"Shared patient region cache meaning/content differs for {case_id}; original preserved")
            alias = _alias_receipt(Path(cache_root) / "aliases", case_id, source_root, actual, expected)
        else:
            regions = _new_regions(case, expected, liver_label=liver_label, tumor_label=tumor_label,
                                   config=config, ct_clip=ct_clip, seed=seed, root=Path(cache_root) / "regions")
    else:
        regions = _new_regions(case, expected, liver_label=liver_label, tumor_label=tumor_label,
                               config=config, ct_clip=ct_clip, seed=seed, root=Path(cache_root) / "regions")
    tumor = case.label == tumor_label
    components, _ = ndi.label(tumor, structure=ndi.generate_binary_structure(3, 1))
    distance = distance_to_mask_mm(tumor, case.spacing)
    verify_loaded_case_source_signatures(case)
    return PreparedFeedbackPatient(case, components, regions, distance,
                                   {"raw_admission": admission, "raw_array_lower_bound_only": True,
                                    "region_alias_receipt": None if alias is None else str(alias),
                                    "same_patient_sources_share_preparation": True})
