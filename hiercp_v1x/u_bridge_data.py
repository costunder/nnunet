"""Original V1 source samples with selected-seven or frozen native-U queries.

The native arm rotates seven U across epochs, not nineteen updates per epoch.
Every full evaluation is one own-source positive plus all 128 frozen U.  This
module never calls the CP placement search or changes the preserved model.
"""
from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import copy
from dataclasses import fields, is_dataclass
import hashlib
import importlib
import json
import math
from pathlib import Path, PurePosixPath
import sys
import threading
from types import SimpleNamespace
import uuid
from zipfile import ZipFile

import numpy as np
import torch

FORMAT = "v18_own_source_selected_native_u_bridge_v1"
P_METADATA_CONTRACT = "original_v1_source_positive_metadata_v1"


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2 ** 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def _verify_original_core(source, scope):
    """Verify archived core files and their imports after PyG JIT generation.

    PyG registers generated propagation modules under names such as
    ``hiercp.model_CompatibilityGatedGATv2Conv_propagate``. Their temporary
    generated files are not original core source files. Original activation
    already checks that the snapshot contains no unlisted importable source;
    here every archived core byte and every loaded core module path are checked.
    """
    from . import bounded_scope
    from .contracts import V1_ARCHIVE_SHA256

    source = Path(source).resolve(strict=True)
    archive = bounded_scope.ROOT / "versions/v1/pipeline_v1_source.zip"
    if (_sha(archive) != V1_ARCHIVE_SHA256
            or scope.get("source_archive_sha256") != V1_ARCHIVE_SHA256):
        raise ValueError("Preserved v1 archive SHA256 differs")
    files = {}
    with ZipFile(archive) as bundle:
        for name in bundle.namelist():
            if not (name.startswith("hiercp/") and name.endswith(".py")):
                continue
            relative = PurePosixPath(name)
            if (relative.is_absolute() or ".." in relative.parts
                    or "\\" in name or ":" in name):
                raise ValueError(f"Unsafe original core path: {name}")
            path = source.joinpath(*relative.parts)
            expected = hashlib.sha256(bundle.read(name)).hexdigest()
            if (path.is_symlink() or not path.resolve().is_relative_to(source)
                    or not path.is_file() or _sha(path) != expected):
                raise ValueError(f"Bridge requires byte-exact original core source: {name}")
            files[name] = expected
            module_name = ".".join(relative.with_suffix("").parts)
            if module_name.endswith(".__init__"):
                module_name = module_name[:-len(".__init__")]
            module = sys.modules.get(module_name)
            if module is not None:
                actual_path = getattr(module, "__file__", None)
                if actual_path is None or Path(actual_path).resolve() != path.resolve():
                    raise ValueError(f"Another original core implementation is imported: {module_name}")
    if files != scope.get("original_module_sha256"):
        raise ValueError("Activated original V1 core identity changed")
    return files


def _centers(value, count, name):
    array = np.asarray(value.tolist() if torch.is_tensor(value) else value)
    if (array.shape != (count, 3) or not np.issubdtype(array.dtype, np.integer)
            or np.any(array < 0)):
        raise ValueError(f"{name}: exactly {count} native integer centers required")
    result = tuple(tuple(int(v) for v in row) for row in array)
    if len(set(result)) != count:
        raise ValueError(f"{name}: duplicate candidate centers")
    return result


def _distance_to_other_tumor_mm(label, source_mask, center, spacing, *,
                                slab_voxels=16 * 1024 ** 2, point_chunk=262144):
    """Exact original float32 EDT value at one source anchor, without a full EDT.

    Scratch chunks limit memory only; every other-tumor voxel is inspected.
    Original distance_to_mask_mm converts spacing to float32, computes physical
    squared distances in float64, takes sqrt, and finally converts to float32.
    """
    if (not isinstance(label, np.ndarray) or label.ndim != 3
            or not isinstance(source_mask, np.ndarray) or source_mask.shape != label.shape
            or source_mask.dtype != np.dtype(bool)):
        raise ValueError("Original whole-case labels and boolean source full mask required")
    if type(slab_voxels) is not int or type(point_chunk) is not int or min(slab_voxels, point_chunk) < 1:
        raise ValueError("Positive distance scratch chunk sizes required")
    anchor = np.asarray(center)
    if (anchor.shape != (3,) or not np.issubdtype(anchor.dtype, np.integer)
            or np.any(anchor < 0) or np.any(anchor >= label.shape)):
        raise ValueError("Exact in-volume integer source anchor required")
    pitch = np.asarray(spacing, dtype=np.float32).astype(np.float64)
    if pitch.shape != (3,) or not np.isfinite(pitch).all() or np.any(pitch <= 0):
        raise ValueError("Three positive finite original native spacings required")
    anchor = anchor.astype(np.float64)
    rows = max(1, slab_voxels // math.prod(label.shape[1:]))
    best = np.inf
    for start in range(0, label.shape[0], rows):
        stop = min(start + rows, label.shape[0])
        coordinates = np.nonzero((label[start:stop] == 2) & ~source_mask[start:stop])
        for offset in range(0, len(coordinates[0]), point_chunk):
            end = offset + point_chunk
            points = np.column_stack((coordinates[0][offset:end] + start,
                                      coordinates[1][offset:end], coordinates[2][offset:end])).astype(np.float64)
            differences = (points - anchor) * pitch
            squared = np.square(differences).sum(axis=1, dtype=np.float64)
            best = min(best, float(squared.min()))
    return float(np.float32(np.sqrt(best)))


def _resident_size(value, seen=None):
    """Conservative host storage accounting; aliases within an item count once."""
    seen = set() if seen is None else seen
    if id(value) in seen:
        return 0
    seen.add(id(value))
    if torch.is_tensor(value):
        if value.device.type != "cpu":
            raise ValueError("Bridge input cache must contain CPU tensors only")
        storage = value.untyped_storage()
        key = ("torch", storage.data_ptr())
        if key in seen:
            return 0
        seen.add(key)
        return storage.nbytes()
    if isinstance(value, np.ndarray):
        return int(value.nbytes)
    if isinstance(value, dict):
        return sum(_resident_size(v, seen) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_resident_size(v, seen) for v in value)
    if is_dataclass(value):
        return sum(_resident_size(getattr(value, f.name), seen) for f in fields(value))
    if hasattr(value, "to_dict"):
        return _resident_size(value.to_dict(), seen)
    if hasattr(value, "__dict__"):
        return _resident_size(vars(value), seen)
    return 0


class UBridgeData:
    """CPU provider using the already activated byte-exact V1/m10 operators.

    ``source_samples`` contains original canonical samples, paths, or signed
    ``{path, sha256}`` entries.  All original source identities remain fixed.
    Raw records require image/label hashes and explicit frozen comparison128.
    ``budget`` is the caller's resource guard: callable or an object with check().
    """
    global_rng_free = True

    def __init__(self, source_samples, raw_records, config, bank, root, workers,
                 resident_bytes, budget, regions_dir=None):
        if type(workers) is not int or workers < 2:
            raise ValueError("Explicit parallel graph workers >=2 required")
        if type(resident_bytes) is not int or resident_bytes <= 0:
            raise ValueError("Explicit positive resident cache budget required")
        if not callable(budget) and not callable(getattr(budget, "check", None)):
            raise TypeError("An actual resource budget guard is required")
        self.config = copy.deepcopy(config)
        graph = self.config["graph"]
        if (float(graph["adaptive_roi_margin_mm"]) != 10.
                or float(graph["context_outer_radius_mm"]) != 10.):
            raise ValueError("This bridge preserves the declared m10 scope")
        if (self.config["cache"]["total_candidates"] != 8
                or self.config["labels"] != {"liver": 1, "tumor": 2}):
            raise ValueError("Original eight-candidate V1/annotation contract required")
        self.bank, self.workers, self.resident_limit, self.budget = bank, workers, resident_bytes, budget
        self.bank_fingerprint = self.bank.fingerprint()
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.graph_dir = self.root / "canonical_local"
        self.graph_dir.mkdir(exist_ok=True)
        self.regions_dir = None if regions_dir is None else Path(regions_dir).resolve(strict=True)
        self.raw = {}
        for row in raw_records:
            case = row["case_id"]
            if case in self.raw:
                raise ValueError("Duplicate raw case identity")
            if any(not row.get(k) for k in ("image", "label", "image_sha256", "label_sha256")):
                raise ValueError(f"{case}: signed original raw CT and annotation required")
            comparison = row.get("comparison", {}).get("centers")
            if comparison is None:
                raise ValueError(f"{case}: frozen128 native-U metadata missing; no resampling")
            frozen = _centers(comparison, 128, case + " native U")
            self.raw[case] = dict(copy.deepcopy(row), frozen_centers=frozen)
        self._examples = []
        identities = set()
        self._cache = OrderedDict()
        self._resident_bytes = 0
        self._lock = threading.RLock()
        self._disk_locks = {}
        self.field_receipts = {}
        self._positive_metadata = {}
        from .u_bridge_upper import UpperCache
        self._upper_cache = UpperCache(self.root / "upper_static", self._check_budget)
        self._upper_bindings = {}
        self.stats = dict(local_builds=0, disk_hits=0, resident_hits=0, evictions=0,
                          field_builds=0, field_reopens=0, field_mapping_reuses=0,
                          field_build_seconds=0., field_reopen_seconds=0., field_reuse_seconds=0.,
                          positive_metadata_builds=0,
                          CP_candidate_search_used=False, U_resampled=False)
        for entry in source_samples:
            sample, path_hash = self._read_original(entry)
            if sample.get("prototype_fingerprint") != self.bank_fingerprint:
                raise ValueError("Original source sample prototype fingerprint differs from preserved bank")
            case, number = sample["case_id"], int(sample["sample_index"])
            identity = (case, number)
            if identity in identities:
                raise ValueError("Duplicate original source sample identity")
            identities.add(identity)
            if case not in self.raw:
                raise ValueError(f"{case}: source sample has no original raw record")
            centers = _centers(sample["candidate_centers"], 8, str(identity))
            difficulty = torch.as_tensor(sample["difficulties"])
            if difficulty.shape != (8,) or int(difficulty[0]) != 0:
                raise ValueError("Original positive must be candidate zero")
            split = str(sample["split"])
            if split not in ("train", "val", "inner_train", "inner_val"):
                raise ValueError("Original train/validation sample partition required")
            component = int(sample["source_component"])
            if component < 1 or number < 0:
                raise ValueError("Original positive component and sample index required")
            self._examples.append(dict(index=len(self._examples), id=f"{case}:{number}",
                case_id=case, sample_index=number, source_component=component, split=split,
                partition="train" if split in ("train", "inner_train") else "val",
                positive_center=centers[0], selected_centers=centers[1:],
                native_centers=self.raw[case]["frozen_centers"], original_sample_sha256=path_hash))
            self._check_budget()
        if not self._examples:
            raise ValueError("Complete nonempty original source sample inventory required")
        for example in self._examples:
            if example["positive_center"] in example["native_centers"]:
                raise ValueError("Frozen U overlaps the preserved positive anchor")
        self._runtime_modules = None
        self._scope = None

    def _check_budget(self):
        (self.budget if callable(self.budget) else self.budget.check)()

    @staticmethod
    def _read_original(entry):
        if isinstance(entry, dict) and "candidate_centers" in entry:
            return entry, None
        if isinstance(entry, dict):
            path = Path(entry["path"]).resolve(strict=True)
            expected = entry.get("sha256")
        else:
            path, expected = Path(entry).resolve(strict=True), None
        actual = _sha(path)
        if expected is not None and actual != expected:
            raise ValueError(f"Original source sample SHA256 changed: {path}")
        sample = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        if not isinstance(sample, dict):
            raise ValueError("Original canonical source sample mapping required")
        return sample, actual

    def examples(self, partition):
        if partition in ("inner_train", "train"):
            partition = "train"
        elif partition in ("inner_val", "val", "validation"):
            partition = "val"
        else:
            raise ValueError("Explicit original train or validation partition required")
        return [copy.deepcopy(x) for x in self._examples if x["partition"] == partition]

    def candidate_keys(self, index, arm, epoch, full=False):
        self._example(index)
        if arm not in ("selected", "native"):
            raise ValueError("Bridge arm must be selected or native")
        if type(epoch) is not int or epoch < 0:
            raise ValueError("Explicit nonnegative integer view epoch required")
        if full:
            return ("P", *(f"U:{i}" for i in range(128)))
        if arm == "selected":
            return ("P", *(f"S:{i}" for i in range(7)))
        if epoch < 1:
            raise ValueError("Native training rotation uses one-based epoch numbering")
        start = (epoch - 1) * 7 % 128
        return ("P", *(f"U:{(start + j) % 128}" for j in range(7)))

    def _example(self, index):
        if type(index) is not int or not 0 <= index < len(self._examples):
            raise IndexError("Unknown original source sample index")
        return self._examples[index]

    def _centers_for(self, example, center_keys):
        values = []
        for key in center_keys:
            if isinstance(key, str):
                if key == "P":
                    center = example["positive_center"]
                else:
                    category, separator, token = key.partition(":")
                    if separator != ":" or category not in ("S", "U") or not token.isdecimal():
                        raise ValueError("Candidate keys are P, S:0..6, U:0..127")
                    population = example["selected_centers"] if category == "S" else example["native_centers"]
                    number = int(token)
                    if not 0 <= number < len(population):
                        raise ValueError("Candidate key outside the frozen population")
                    center = population[number]
            else:
                center = _centers([key], 1, "candidate")[0]
                if center not in (example["positive_center"], *example["selected_centers"], *example["native_centers"]):
                    raise ValueError("Candidate outside original/frozen center inventory")
            values.append(center)
        if (len(values) < 2 or values[0] != example["positive_center"]
                or len(set(values)) != len(values)):
            raise ValueError("One preserved positive first and distinct comparison centers required")
        return tuple(values)

    def _runtime(self):
        if self._runtime_modules is None:
            from . import bounded_scope
            scope = bounded_scope._ACTIVE
            if scope is None or float(scope["margin_mm"]) != 10.:
                raise ValueError("Already activated original V1 source plus bounded_scope.install(10) required")
            modules = {name: importlib.import_module("hiercp." + name)
                       for name in ("common", "region", "schema", "cache", "local", "sample", "data", "hierarchy")}
            root = Path(modules["common"].__file__).resolve().parent.parent
            _verify_original_core(root, scope)
            self._scope = copy.deepcopy(scope)
            self._runtime_modules = SimpleNamespace(**modules)
        return self._runtime_modules

    def _get(self, key, factory):
        with self._lock:
            if key in self._cache:
                item, size = self._cache.pop(key)
                self._cache[key] = (item, size)
                self.stats["resident_hits"] += 1
                return item
        item = factory()
        size = _resident_size(item)
        with self._lock:
            if key in self._cache:
                return self._cache[key][0]
            while self._cache and self._resident_bytes + size > self.resident_limit:
                _, (_, removed) = self._cache.popitem(last=False)
                self._resident_bytes -= removed
                self.stats["evictions"] += 1
            if size <= self.resident_limit:
                self._cache[key] = (item, size)
                self._resident_bytes += size
        self._check_budget()
        return item

    def _case(self, case_id):
        def load():
            runtime, row = self._runtime(), self.raw[case_id]
            for name in ("image", "label"):
                if _sha(row[name]) != row[name + "_sha256"]:
                    raise ValueError(f"{case_id}: original raw {name} SHA256 changed")
            case = runtime.common.load_case(runtime.common.CasePaths(case_id, Path(row["image"]), Path(row["label"])))
            if row.get("shape") is not None and tuple(row["shape"]) != case.shape:
                raise ValueError(f"{case_id}: raw shape changed")
            if row.get("spacing") is not None and not np.allclose(row["spacing"], case.spacing):
                raise ValueError(f"{case_id}: native spacing changed")
            for center in row["frozen_centers"]:
                if any(c >= n for c, n in zip(center, case.shape)) or case.label[center] != 1:
                    raise ValueError(f"{case_id}: frozen U is not an actual label1 center: {center}")
            organ = (case.label == 1) | (case.label == 2)
            from .u_bridge_fields import cached_fields
            binding = dict(image_sha256=row["image_sha256"], label_sha256=row["label_sha256"],
                shape=list(case.shape), spacing=[float(v) for v in case.spacing],
                common_sha256=self._scope["original_module_sha256"]["hiercp/common.py"])
            depth, occupied, receipt = cached_fields(self.root / "whole_case_fields", case_id, binding,
                lambda: runtime.common.organ_depth_mm(organ, case.spacing),
                lambda: runtime.common.distance_to_mask_mm(case.label == 2, case.spacing),
                self._check_budget)
            counts = {"built": ("field_builds", "field_build_seconds"),
                      "reopened": ("field_reopens", "field_reopen_seconds"),
                      "resident_mapping": ("field_mapping_reuses", "field_reuse_seconds")}
            count, seconds = counts[receipt["status"]]
            with self._lock:
                self.stats[count] += 1
                self.stats[seconds] += receipt["wall_seconds"]
                self.field_receipts[case_id] = receipt
            print(f"v1.8 whole-case fields case={case_id} status={receipt['status']} "
                  f"array_bytes={receipt['array_bytes']} disk_bytes={receipt['disk_bytes']} "
                  f"wall_seconds={receipt['wall_seconds']:.3f}", flush=True)
            return case, organ, depth, occupied
        return self._get(("case", case_id), load)

    def _regions(self, case):
        def load():
            runtime = self._runtime()
            graph = runtime.schema.graph_config_from_dict(self.config["graph"])
            seed = runtime.common.stable_case_seed(self.config["seed"], case.paths.case_id,
                                                  runtime.region.REGION_CACHE_SEED_SALT)
            arguments = dict(liver_label=1, tumor_label=2, config=graph,
                             ct_clip=tuple(self.config["ct_clip"]), seed=seed)
            if self.regions_dir is None:
                return runtime.region.load_or_build_patient_regions(case, **arguments,
                    cache_dir=self.root / "regions", overwrite=False, mmap=True)
            directory = self.regions_dir / case.paths.case_id
            if directory.is_symlink():
                raise ValueError("Read-only original region cache cannot be a symlink")
            regions, metadata = runtime.region.load_patient_regions(directory, mmap=True)
            expected = runtime.region._region_cache_metadata(case, **arguments)
            actual_graph = copy.deepcopy(metadata.get("graph_config"))
            if not isinstance(actual_graph, dict):
                raise ValueError("Original region graph identity missing")
            # Physical-scope runs intentionally retain the original population
            # partition. Only these two declared scope fields may differ.
            for field in ("adaptive_roi_margin_mm", "context_outer_radius_mm"):
                actual_graph[field] = self.config["graph"][field]
            compared = dict(metadata, graph_config=actual_graph)
            if not runtime.region._metadata_equal(compared, expected):
                raise ValueError(f"{case.paths.case_id}: original region/raw/config identity differs")
            return regions
        return self._get(("regions", case.paths.case_id), load)

    def _source(self, example, case, organ, depth):
        def prepare():
            runtime = self._runtime()
            source, _, _ = runtime.common.choose_source_tumor(case.image, case.label,
                tumor_label=2, selection=self.config["cache"]["source_selection"],
                pad=self.config["cache"]["source_pad"], rng=np.random.default_rng(
                    runtime.common.stable_case_seed(self.config["seed"], example["case_id"],
                        f"sample_{example['sample_index']}")))
            if (int(source.component_id) != example["source_component"]
                    or tuple(source.anchor_center) != example["positive_center"]):
                raise ValueError(f"{example['id']}: actual original source component/anchor changed")
            prepared = runtime.local.prepare_local_source(case, source,
                full_organ_mask=organ, organ_depth=depth,
                config=runtime.schema.graph_config_from_dict(self.config["graph"]),
                rng=np.random.default_rng(self.config["seed"]), ct_clip=tuple(self.config["ct_clip"]))
            return source, prepared
        return self._get(("source", example["id"]), prepare)

    def _candidate(self, center, case, source, organ, depth, occupied):
        runtime = self._runtime()
        mask = source.patch_mask
        crop = runtime.common.extract_centered_patch(case.image, center, mask.shape,
                                                     pad_value=self.config["ct_clip"][0])
        liver = runtime.common.extract_centered_patch(case.label == 1, center, mask.shape, pad_value=False)
        crop_organ = runtime.common.extract_centered_patch(organ, center, mask.shape, pad_value=False)
        mean, std = runtime.common.context_stats_for_local_mask(crop, crop_organ, mask, ring_width=3)
        # Clipped slices describe the real intersection. They are metadata, not
        # an admission test; build_local_graph keeps its original padded ROI.
        start = np.asarray(center) - np.asarray(mask.shape) // 2
        slices = tuple(slice(max(0, int(a)), min(int(a + n), int(limit)))
                       for a, n, limit in zip(start, mask.shape, case.shape))
        return runtime.common.CandidateInfo(center=center, slices=slices,
            liver_coverage=float(np.sum(mask & liver) / source.voxel_count),
            border_distance_mm=float(depth[center]), occupied_distance_mm=float(occupied[center]),
            context_mean_hu=mean, context_std_hu=std)

    def _positive_candidate(self, example, case, source, organ, depth):
        key = example["id"]
        with self._lock:
            if key in self._positive_metadata:
                return self._positive_metadata[key]
        runtime = self._runtime()
        center = tuple(source.anchor_center)
        mean, std = runtime.common.context_stats_for_local_mask(
            source.patch_image, organ[source.patch_slices], source.patch_mask, ring_width=3)
        candidate = runtime.common.CandidateInfo(center=center, slices=source.patch_slices,
            liver_coverage=1.0, border_distance_mm=float(depth[center]),
            occupied_distance_mm=_distance_to_other_tumor_mm(
                case.label, source.full_mask, center, case.spacing),
            context_mean_hu=mean, context_std_hu=std)
        with self._lock:
            if key not in self._positive_metadata:
                self._positive_metadata[key] = candidate
                self.stats["positive_metadata_builds"] += 1
            return self._positive_metadata[key]

    def _upper_context(self, example, case, source, regions):
        """Reuse only exact original candidate-independent upper helper arrays."""
        runtime = self._runtime()
        key = example["id"]
        with self._lock:
            binding = self._upper_bindings.get(key)
        if binding is None:
            graph = runtime.schema.graph_config_from_dict(self.config["graph"])
            region_metadata = runtime.region._region_cache_metadata(case, liver_label=1,
                tumor_label=2, config=graph, ct_clip=tuple(self.config["ct_clip"]),
                seed=runtime.common.stable_case_seed(self.config["seed"], case.paths.case_id,
                                                   runtime.region.REGION_CACHE_SEED_SALT))
            binding = dict(case_id=case.paths.case_id,
                image_sha256=self.raw[case.paths.case_id]["image_sha256"],
                label_sha256=self.raw[case.paths.case_id]["label_sha256"],
                original_core_sha256=_hash(self._scope["original_module_sha256"]),
                region_identity_sha256=_hash(region_metadata),
                source_component=int(source.component_id), source_voxels=int(source.voxel_count),
                shape=list(case.shape), anchor=list(source.anchor_center),
                spacing=[float(v) for v in case.spacing], ct_clip=list(self.config["ct_clip"]),
                tumor_label=2, max_lesions=graph.max_lesions,
                upper_raw_dim=int(runtime.hierarchy.UPPER_RAW_DIM))
            with self._lock:
                self._upper_bindings[key] = binding
        return self._upper_cache.original_helpers(runtime.hierarchy, case, source, regions, binding)

    def _binding(self, example, center):
        self._runtime()
        if self._scope is None:
            raise ValueError("Canonical caching requires verified m10 scope identity")
        if self.bank.fingerprint() != self.bank_fingerprint:
            raise ValueError("Preserved prototype bank identity changed after bridge construction")
        return dict(format=FORMAT, source_id=example["id"], source_component=example["source_component"],
            positive_center=example["positive_center"], center=center,
            image_sha256=self.raw[example["case_id"]]["image_sha256"],
            label_sha256=self.raw[example["case_id"]]["label_sha256"],
            config=self.config, prototype_fingerprint=self.bank_fingerprint,
            scope_contract_sha256=self._scope["contract_sha256"])

    def _local_map(self, example, build_one, specs, **identity):
        if (identity.get("case_id") != example["case_id"]
                or identity.get("source_component") != example["source_component"]):
            raise ValueError("Original hierarchy requested a different source")
        def obtain(spec):
            center = tuple(int(v) for v in spec.center)
            if (int(spec.corruption) != 0 or not np.array_equal(spec.rotation_matrix, np.eye(3))
                    or not np.array_equal(spec.scale_array, np.ones(3))):
                raise ValueError("Both bridge arms require untransformed real centers")
            binding = self._binding(example, center)
            key = _hash(binding)
            with self._lock:
                lock = self._disk_locks.setdefault(key, threading.Lock())
            def load_or_build():
                with lock:
                    path = self.graph_dir / (key + ".pt")
                    if path.exists():
                        if path.is_symlink():
                            raise ValueError("Canonical cache path cannot be a symlink")
                        payload = torch.load(path, map_location="cpu", weights_only=False)
                        if payload.get("binding") != json.loads(json.dumps(binding)):
                            raise ValueError("Canonical candidate cache identity changed")
                        self.stats["disk_hits"] += 1
                        return payload["built"]
                    try:
                        built = build_one(spec)
                    except Exception as error:
                        raise RuntimeError(f"Bridge canonical geometry failed: case={example['case_id']} "
                            f"sample={example['sample_index']} component={example['source_component']} "
                            f"center={center}; no candidate skipped or replaced") from error
                    self._check_budget()
                    payload = dict(binding=json.loads(json.dumps(binding)), built=built)
                    # Publish only our fresh complete file. Existing artifacts
                    # are never overwritten, including after interrupted runs.
                    temporary = self.graph_dir / (key + "." + uuid.uuid4().hex + ".tmp")
                    with temporary.open("xb") as stream:
                        torch.save(payload, stream)
                    if path.exists():
                        raise FileExistsError(f"Concurrent canonical publication: {path}")
                    temporary.rename(path)
                    self.stats["local_builds"] += 1
                    return built
            return self._get(("local", key), load_or_build)
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            return list(pool.map(obtain, specs))

    def sample(self, index, center_keys, epoch, training):
        example = self._example(index)
        if type(epoch) is not int or epoch < 0 or type(training) is not bool:
            raise ValueError("Explicit epoch and training/evaluation view mode required")
        centers = self._centers_for(example, center_keys)
        runtime = self._runtime()
        case, organ, depth, occupied = self._case(example["case_id"])
        regions = self._regions(case)
        source, prepared = self._source(example, case, organ, depth)
        for center in centers:
            if any(c >= n for c, n in zip(center, case.shape)):
                raise ValueError(f"{example['id']}: candidate center outside original CT: {center}")
        candidates = [self._positive_candidate(example, case, source, organ, depth)]
        candidates.extend(self._candidate(center, case, source, organ, depth, occupied)
                          for center in centers[1:])
        with self._upper_context(example, case, source, regions):
            canonical, specs = runtime.cache.build_inference_sample(case, source, candidates,
                self.bank, graph_config=runtime.schema.graph_config_from_dict(self.config["graph"]),
                liver_label=1, tumor_label=2, ct_clip=tuple(self.config["ct_clip"]),
                seed=self.config["seed"], regions=regions, prepared_source=prepared,
                local_graph_map=lambda build, values, **identity:
                    self._local_map(example, build, values, **identity))
        if _centers(canonical["candidate_centers"], len(centers), "built hierarchy") != centers:
            raise ValueError("Original inference builder changed candidate coverage/order")
        canonical.update(case_id=example["case_id"], sample_index=example["sample_index"],
                         split=example["split"], source_component=example["source_component"])
        canonical["candidate_metadata_contract"] = P_METADATA_CONTRACT
        canonical["difficulties"] = torch.tensor([0] + [1] * (len(centers) - 1), dtype=torch.long)
        canonical["corruptions"] = torch.zeros(len(centers), dtype=torch.long)
        materialized = runtime.sample.materialize_sample_views(canonical, training=training,
            epoch=epoch, global_seed=self.config["seed"])
        materialized["bridge_source_id"] = example["id"]
        materialized["bridge_center_keys"] = tuple(center_keys)
        self._check_budget()
        return materialized

    def batch(self, indices, arm, epoch, training, full=False):
        ids = list(indices)
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("A nonempty physical batch of distinct original source samples is required")
        if training and full:
            raise ValueError("Full129 is evaluation only; training updates preserve original eight queries")
        keys = [self.candidate_keys(i, arm, epoch, full=full) for i in ids]
        # Samples share raw/source/region caches; canonical targets inside each
        # sample are actually constructed by the parallel local_graph_map.
        samples = [self.sample(i, k, epoch, training) for i, k in zip(ids, keys)]
        batch = self._runtime().data.collate_samples(samples)
        if tuple(batch.counts) != tuple(len(k) for k in keys):
            raise ValueError("Original hierarchy collate changed physical query coverage")
        batch.bridge_indices = tuple(ids)
        batch.bridge_center_keys = tuple(keys)
        batch.bridge_candidate_keys = tuple(keys)
        batch.bridge_source_ids = tuple(self._example(i)["id"] for i in ids)
        return batch

    def report(self):
        return dict(format=FORMAT, source_samples=len(self._examples), native_U_per_case=128,
            candidate_metadata_contract=P_METADATA_CONTRACT,
            training_query_count=8, full_evaluation_query_count=129, workers=self.workers,
            resident_bytes=self._resident_bytes, resident_limit_bytes=self.resident_limit,
            whole_case_fields=dict(cases=len(self.field_receipts),
                array_bytes=sum(r["array_bytes"] for r in self.field_receipts.values()),
                disk_bytes=sum(r["disk_bytes"] for r in self.field_receipts.values()),
                receipts=copy.deepcopy(self.field_receipts)),
            static_upper=self._upper_cache.report(),
            **self.stats)
