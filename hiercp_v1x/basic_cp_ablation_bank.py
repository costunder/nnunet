"""Build the Basic-only 256-location intervention from the historical 128 bank.

The complete original 642-source population and its source arrays/scores are
retained. Only candidate_centers and candidate_raw_centers gain 128 locations.
The additional locations pass the historical raw, preprocessed and frozen graph
geometry filters; they are not GNN-scored. A dedicated Basic-only loader must
sample all 256 centers while ignoring the original 128 historical scores.
"""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import copy
import gc
import hashlib
import importlib.util
import json
import multiprocessing
import os
from pathlib import Path
import pickle
import sys
import time
import types
from typing import Iterable, Mapping

import numpy as np

BANK_SHA256 = "9e5be3e3ab8f04fd23c166b3267bc4efbfb4b321ef974920dfc08fbf5dd275e6"
TRAIN_SHA256 = "2fe4ae3d59d0346060c1eda9f28c181fadc320e77acf7a1084ac5d5ce2538dd9"
BENCHMARK_SHA256 = "92a7a866282c7a1828ceb44aeb6fe3097a5bfba63a928b61d507ddc28aba5f78"
COMMON_SHA256 = "12363d1f326b5a5e962d169860d548a0c5c38123672f716b49f8e5f6cf437bc2"
FORMAT = "basic_cp_ablation_candidates256_preserved128_v1"
GENERATION = {"source_pad": 4, "max_draws": 50000, "min_liver_coverage": 0.85,
              "occupied_clearance_vox": 2, "min_center_separation_mm": 12.0}
_WORKER_PROVIDER = None
_WORKER_BINDING = None
_PREPARATION_LEASE = None
_PREPARATION_LEASE_BINDING = None


class BankContractError(ValueError):
    """An immutable original or full-scale extension contract was not met."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise BankContractError(message)


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256(str(value.dtype).encode())
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def stat_identity(path: Path) -> list[int]:
    info = path.stat()
    return [info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns,
            info.st_ctime_ns, info.st_nlink]


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def write_json_exclusive(path: Path, value: Mapping) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())


def publish_progress(root: Path, value: Mapping) -> None:
    """Replace only this fresh preparation's progress file, never source data."""
    temporary = root / f".generation_progress_{os.getpid()}_{time.time_ns()}.tmp"
    write_json_exclusive(temporary, value)
    os.replace(temporary, root / "generation_progress.json")


def cpu_affinity() -> list[int]:
    if hasattr(os, "sched_getaffinity"):
        return sorted(os.sched_getaffinity(0))
    import psutil
    return sorted(psutil.Process().cpu_affinity())


def process_tree_rss() -> int:
    import psutil
    process = psutil.Process()
    total = process.memory_info().rss
    for child in process.children(recursive=True):
        try:
            total += child.memory_info().rss
        except psutil.NoSuchProcess:
            # A naturally completed worker is no longer consuming RAM.
            continue
    return int(total)


def preparation_lease_verify() -> None:
    """Cooperative resource/expiry checks when launched by the Basic controller."""
    global _PREPARATION_LEASE, _PREPARATION_LEASE_BINDING
    value = os.environ.get('BASIC_CP_LEASE_REQUEST')
    if not value:
        require(_PREPARATION_LEASE is None, 'Established preparation lease environment disappeared')
        return
    path = Path(value).absolute()
    require(path.resolve(strict=True) == path and not path.is_symlink(), 'Exact preparation lease request path required')
    binding = (str(path),stat_identity(path))
    if _PREPARATION_LEASE is None:
        from .basic_cp_ablation_resources import BasicLease
        request = read_json(path)
        _PREPARATION_LEASE = BasicLease(request)
        _PREPARATION_LEASE_BINDING = binding
    require(binding == _PREPARATION_LEASE_BINDING, 'Preparation lease request changed')
    _PREPARATION_LEASE.verify()


def cached_historical_pool(original, occupied_mask: np.ndarray, spacing: np.ndarray):
    """Memoize only original patient-wide EDT/dilation in a private namespace.

    The historical function's bytecode, candidate RNG, geometry and filters are
    unchanged. Input identity is fixed for this patient, and an unexpected
    invocation fails instead of accepting a stale array. The original module
    and scipy namespace remain untouched.
    """
    require(original.__closure__ is None, "Historical candidate pool must have no closure")
    namespace = dict(original.__globals__)
    original_ndi = namespace["ndi"]
    distance = namespace["distance_to_mask_mm"]
    cache = {"occupied_distance_computations": 0, "forbidden_computations": 0}
    expected_spacing = np.asarray(spacing).copy()

    def cached_distance(mask, voxel_spacing):
        require(mask is occupied_mask and np.array_equal(voxel_spacing, expected_spacing),
                "Patient-wide distance cache input changed")
        if "distance" not in cache:
            cache["distance"] = distance(mask, voxel_spacing)
            cache["distance"].flags.writeable = False
            cache["occupied_distance_computations"] += 1
        return cache["distance"]

    class NDIProxy:
        def __getattr__(self, name):
            return getattr(original_ndi, name)

        def binary_dilation(self, mask, *, structure, iterations):
            require(mask is occupied_mask, "Patient-wide forbidden cache input changed")
            key = (int(iterations), array_sha(np.asarray(structure)))
            if "forbidden" not in cache:
                cache["forbidden_key"] = key
                cache["forbidden"] = original_ndi.binary_dilation(
                    mask, structure=structure, iterations=iterations)
                cache["forbidden"].flags.writeable = False
                cache["forbidden_computations"] += 1
            require(cache["forbidden_key"] == key, "Historical forbidden geometry changed")
            return cache["forbidden"]

    namespace["distance_to_mask_mm"] = cached_distance
    namespace["ndi"] = NDIProxy()
    cloned = types.FunctionType(original.__code__, namespace, original.__name__,
                                original.__defaults__, original.__closure__)
    cloned.__kwdefaults__ = original.__kwdefaults__
    return cloned, cache


def safe_entry(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    require(root.resolve() in path.parents and path.suffix == ".npz", f"Entry escapes bank root: {relative}")
    return path


def read_entry(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        result = {name: np.asarray(archive[name]) for name in archive.files}
    needed = {"source_data", "source_mask", "anchor_offset", "candidate_centers",
              "candidate_raw_centers", "scores", "source_component", "source_diameter_mm"}
    require(needed.issubset(result), f"Historical source fields missing: {path}")
    return result


def preprocessed_valid(center: np.ndarray, source_mask: np.ndarray,
                       anchor_offset: np.ndarray, segmentation: np.ndarray,
                       *, liver_label: int, tumor_label: int,
                       minimum_coverage: float) -> bool:
    """Exact historical preprocessed predicate: full bounds/no tumor/.85 liver."""
    require(segmentation.ndim == 4 and segmentation.shape[0] == 1, "Actual preprocessed segmentation must be [1,D,H,W]")
    mask = np.asarray(source_mask, dtype=bool)
    require(mask.ndim == 3 and mask.any(), "Historical source mask must be nonempty")
    starts = np.asarray(center, dtype=np.int64) - np.asarray(anchor_offset, dtype=np.int64)
    stops = starts + np.asarray(mask.shape, dtype=np.int64)
    if np.any(starts < 0) or np.any(stops > np.asarray(segmentation.shape[1:])):
        return False
    target = segmentation[(0, *(slice(int(a), int(b)) for a, b in zip(starts, stops)))]
    if np.any(mask & (target == int(tumor_label))):
        return False
    return float(np.count_nonzero(mask & (target == int(liver_label))) / int(mask.sum())) >= minimum_coverage


def extend_entry(entry: Mapping[str, np.ndarray], proposals: Iterable[tuple[np.ndarray, np.ndarray]],
                 segmentation: np.ndarray, *, liver_label: int, tumor_label: int,
                 minimum_coverage: float, original_count: int, candidate_count: int) -> tuple[dict, dict]:
    """Append already raw/graph-valid proposals, retaining all original bytes.

    original_count/candidate_count are explicit for isolated DEBUG unit tests.
    The final build entrypoint strictly enforces 128 -> 256 and all 642 sources.
    """
    old_pre = np.asarray(entry["candidate_centers"])
    old_raw = np.asarray(entry["candidate_raw_centers"])
    require(old_pre.shape == old_raw.shape == (original_count, 3), "Original candidate shape differs")
    require(np.issubdtype(old_pre.dtype, np.integer) and np.issubdtype(old_raw.dtype, np.integer), "Original centers must be integral")
    require(np.asarray(entry["scores"]).shape == (original_count,) and np.all(np.isfinite(entry["scores"])), "Only actual original finite scores are retained")
    require(candidate_count > original_count, "Extension must add candidate positions")
    seen_pre = {tuple(map(int, row)) for row in old_pre}
    seen_raw = {tuple(map(int, row)) for row in old_raw}
    require(len(seen_pre) == len(seen_raw) == original_count, "Original raw/preprocessed centers must be unique")
    require(all(preprocessed_valid(row, entry["source_mask"], entry["anchor_offset"], segmentation,
                                  liver_label=liver_label, tumor_label=tumor_label,
                                  minimum_coverage=minimum_coverage) for row in old_pre),
            "Original prefix is incompatible with the actual preprocessed segmentation/filter")
    added_pre, added_raw = [], []
    duplicates_raw = duplicates_pre = rejected_preprocessed = tested = 0
    for raw, pre in proposals:
        raw, pre = np.asarray(raw), np.asarray(pre)
        require(raw.shape == pre.shape == (3,) and np.issubdtype(raw.dtype, np.integer)
                and np.issubdtype(pre.dtype, np.integer), "Provider must supply integral raw/preprocessed centers")
        tested += 1
        rk, pk = tuple(map(int, raw)), tuple(map(int, pre))
        if rk in seen_raw:
            duplicates_raw += 1
            continue
        # Historical code marks raw center tested even if its mapped center fails.
        seen_raw.add(rk)
        if pk in seen_pre:
            duplicates_pre += 1
            continue
        if not preprocessed_valid(pre, entry["source_mask"], entry["anchor_offset"], segmentation,
                                  liver_label=liver_label, tumor_label=tumor_label,
                                  minimum_coverage=minimum_coverage):
            rejected_preprocessed += 1
            continue
        seen_pre.add(pk)
        added_raw.append(raw.astype(old_raw.dtype))
        added_pre.append(pre.astype(old_pre.dtype))
        if len(added_pre) == candidate_count-original_count:
            break
    require(len(added_pre) == candidate_count-original_count,
            f"Full extension failed: appended {len(added_pre)}/{candidate_count-original_count}; no source dropped or smaller bank published")
    result = {key: np.asarray(value).copy() for key, value in entry.items()}
    result["candidate_centers"] = np.concatenate((old_pre, np.stack(added_pre)), axis=0)
    result["candidate_raw_centers"] = np.concatenate((old_raw, np.stack(added_raw)), axis=0)
    proof = verify_preservation(entry, result, original_count=original_count, candidate_count=candidate_count)
    proof.update(tested_raw_graph_valid_proposals=tested, duplicate_raw=duplicates_raw,
                 duplicate_preprocessed=duplicates_pre, rejected_preprocessed=rejected_preprocessed)
    return result, proof


def verify_preservation(original: Mapping[str, np.ndarray], extended: Mapping[str, np.ndarray],
                        *, original_count: int, candidate_count: int) -> dict:
    require(set(original) == set(extended), "No original source field may be removed or invented")
    fields = {}
    for name, old in original.items():
        new = np.asarray(extended[name])
        if name in ("candidate_centers", "candidate_raw_centers"):
            require(new.shape == (candidate_count, 3), "Final center cardinality differs")
            require(np.unique(new, axis=0).shape[0] == candidate_count, "Extended centers are not unique")
            new = new[:original_count]
        require(old.dtype == new.dtype and old.shape == new.shape and old.tobytes() == new.tobytes(),
                f"Original field/prefix bytes changed: {name}")
        fields[name] = array_sha(old)
    return {"original_fields_and_first128_preserved": True, "original_array_sha256": fields,
            "original_candidate_count": original_count, "candidate_count": candidate_count,
            "original_GNN_score_count": int(extended["scores"].shape[0]), "scores_used": False,
            "additional_GNN_scores_generated": False}


def validate_original_index(index: Mapping) -> None:
    require(index.get("format") == "hiercp_online_bank_v2", "Only original historical bank v2 admitted")
    require(index.get("candidate_count") == 128 and index.get("source_entries") == 642
            and index.get("eligible_cases") == 81 and index.get("total_candidates") == 82176,
            "Original full642/81/128 population differs")
    groups = index.get("entries_by_case", {})
    names = [name for group in groups.values() for name in group]
    require(len(groups) == 81 and len(names) == len(set(names)) == 642, "Full source identity list differs")
    require(index.get("minimum_liver_coverage") == 0.85 and index.get("liver_label") == 1
            and index.get("tumor_label") == 2 and index.get("draw_count") == 256
            and index.get("attempts") == 4 and index.get("cp_probability") == 0.5,
            "Original candidate-generation/CP contract differs")


class HistoricalGeometryProvider:
    """CPU-only frozen historical source/regions/local-graph geometry provider."""
    def __init__(self, binding: Mapping):
        self.binding = copy.deepcopy(binding)
        self.project = Path(binding["historical_project"]).resolve()
        self.bank_index = Path(binding["bank_index"]).resolve()
        self.bank = read_json(self.bank_index)
        validate_original_index(self.bank)
        require(file_sha(self.bank_index) == BANK_SHA256, "Original index SHA differs")
        code = binding["historical_source_sha256"]
        require(code.get("tools/online_cp_benchmark.py") == BENCHMARK_SHA256
                and code.get("hiercp/common.py") == COMMON_SHA256, "Exact original source binding required")
        for relative, expected in code.items():
            require(file_sha(self.project/relative) == expected, f"Frozen historical source SHA differs: {relative}")
        require(file_sha(Path(binding["train_config"])) == TRAIN_SHA256, "Actual original generation config SHA differs")
        train = read_json(Path(binding["train_config"]))
        require(all(train["generation"][key] == value for key, value in GENERATION.items()) and train["seed"] == 42,
                "Historical raw generation values differ")
        self.generation = train["generation"]
        self.global_seed = int(train["seed"])+int(self.bank["outer_fold"])
        # Never let a current v2.4/v3 import replace the historical geometry.
        existing = [name for name in sys.modules if name == "hiercp" or name.startswith("hiercp.")]
        require(not existing, f"Historical provider needs a fresh process; conflicting modules: {existing}")
        sys.path.insert(0, str(self.project))
        spec = importlib.util.spec_from_file_location("basic256_frozen_historical_benchmark", self.project/"tools/online_cp_benchmark.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        self.benchmark = module
        from hiercp.common import CasePaths, build_candidate_pool, load_case
        from hiercp.curriculum import build_generation_specs
        from hiercp.local import build_local_graph, prepare_local_source
        from hiercp.prototype import PrototypeBank
        from hiercp.region import load_patient_regions, _region_cache_metadata, _metadata_equal, REGION_CACHE_SEED_SALT
        from hiercp.schema import graph_config_from_dict
        from hiercp.spatial import CanonicalGraphUnavailable
        import scipy.ndimage as ndi
        import torch
        for name, imported in tuple(sys.modules.items()):
            if name == "hiercp" or name.startswith("hiercp."):
                source = getattr(imported, "__file__", None)
                require(source and self.project in Path(source).resolve().parents, f"Geometry module outside historical project: {name}")
                relative = str(Path(source).resolve().relative_to(self.project)).replace("\\", "/")
                require(relative in code and file_sha(Path(source)) == code[relative], f"Imported scientific source not bound: {relative}")
        self.CasePaths, self.load_case, self.build_pool = CasePaths, load_case, build_candidate_pool
        self.specs, self.local, self.prepare = build_generation_specs, build_local_graph, prepare_local_source
        self.load_regions, self.region_metadata, self.metadata_equal = load_patient_regions, _region_cache_metadata, _metadata_equal
        self.region_seed_salt, self.geometry_error, self.ndi = REGION_CACHE_SEED_SALT, CanonicalGraphUnavailable, ndi
        checkpoint, prototype = Path(binding["checkpoint"]), Path(binding["prototype"])
        require(file_sha(checkpoint) == self.bank["checkpoint_sha256"] and file_sha(prototype) == self.bank["prototype_sha256"], "Original geometry checkpoint/prototype bytes differ")
        # Only graph_config/ct_clip/prototype descriptors are read; no model
        # construction, forward scoring, optimizer or CUDA context is created.
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.graph = graph_config_from_dict(payload["graph_config"])
        self.ct_clip = tuple(map(float, payload["ct_clip"]))
        self.prototype = PrototypeBank.load(prototype)
        require(self.prototype.fingerprint() == payload["prototype_fingerprint"], "Original prototype fingerprint differs")
        require(file_sha(Path(binding["plans"])) == binding["plans_sha256"],
                "Exact original raw-to-preprocessed plans bytes changed")
        self.plans = read_json(Path(binding["plans"]))
        self.case_map = {case.case_id: case for case in module.discover_cases(Path(binding["raw_data"]))}
        self.case = self.regions = self.components = self.preseg = self.properties = None

    def load_patient(self, case_id: str) -> np.ndarray:
        import blosc2
        # Release the previous patient before decoding the next one. A process
        # retains immutable historical imports/metadata across its tasks.
        self.case = self.regions = self.components = self.preseg = self.properties = None
        self.patient_pool = self.placement_mask = self.occupied_mask = None
        gc.collect()
        paths = self.case_map[case_id]
        self.patient_inputs = {Path(paths.image): stat_identity(Path(paths.image)),
                               Path(paths.label): stat_identity(Path(paths.label))}
        self.case = self.load_case(self.CasePaths(case_id=case_id, image_path=paths.image, label_path=paths.label))
        self.components, _ = self.ndi.label(self.case.label == self.bank["tumor_label"],
                                            structure=self.ndi.generate_binary_structure(3, 1))
        cache_root = Path(self.binding["regions"])/case_id
        require(cache_root.is_dir(), f"Existing historical region cache missing: {case_id}; no regeneration")
        for path in cache_root.rglob("*"):
            if path.is_file():
                self.patient_inputs[path] = stat_identity(path)
        # Explicit read-only loader: missing/incompatible cache is an input
        # error, never a reason to regenerate into the historical directory.
        self.regions, metadata = self.load_regions(cache_root, mmap=True)
        seed = self.benchmark.stable_seed(self.global_seed, case_id, self.region_seed_salt)
        expected = self.region_metadata(self.case, liver_label=self.bank["liver_label"],
                                       tumor_label=self.bank["tumor_label"], config=self.graph,
                                       ct_clip=self.ct_clip, seed=seed)
        require(self.metadata_equal(metadata, expected), f"Historical region cache contract differs: {case_id}")
        pre = Path(self.binding["preprocessed"])
        seg_path, properties_path = pre/f"{case_id}_seg.b2nd", pre/f"{case_id}.pkl"
        self.patient_inputs.update({seg_path: stat_identity(seg_path), properties_path: stat_identity(properties_path)})
        self.preseg = np.asarray(blosc2.open(str(seg_path), mode="r")[:])
        with properties_path.open("rb") as stream:
            self.properties = pickle.load(stream)
        self.placement_mask = self.case.label == self.bank["liver_label"]
        self.occupied_mask = self.case.label == self.bank["tumor_label"]
        self.placement_mask.flags.writeable = self.occupied_mask.flags.writeable = False
        self.patient_pool, self.patient_cache = cached_historical_pool(
            self.build_pool, self.occupied_mask, self.case.spacing)
        return self.preseg

    def verify_patient_inputs(self) -> None:
        require(all(stat_identity(path) == identity for path, identity in self.patient_inputs.items()),
                "Read-only patient/cache/preprocessed input changed during preparation")

    def proposals(self, case_id: str, entry: Mapping[str, np.ndarray]):
        component = int(np.asarray(entry["source_component"])[0])
        source = self.benchmark._source_from_component(self.case, self.components, component,
                                                       int(self.generation["source_pad"]))
        prepared = self.prepare(self.case, source, full_organ_mask=self.regions.full_organ_mask,
                                organ_depth=self.regions.organ_depth, config=self.graph,
                                rng=np.random.default_rng(self.benchmark.stable_seed(self.global_seed, case_id, component, "source")),
                                ct_clip=self.ct_clip)
        old_raw = {tuple(map(int, row)) for row in entry["candidate_raw_centers"]}
        old_pre = {tuple(map(int, row)) for row in entry["candidate_centers"]}
        self.last_proposal_diagnostics = {"graph_rejected": 0, "raw_duplicate": 0, "preprocessed_rejected": 0, "attempts": 0}
        for attempt in range(int(self.bank["attempts"])):
            preparation_lease_verify()
            self.last_proposal_diagnostics["attempts"] += 1
            rng = np.random.default_rng(self.benchmark.stable_seed(self.global_seed, case_id, component, "online_pool", attempt))
            pool, _ = self.patient_pool(self.case, source, placement_mask=self.placement_mask,
                                      full_organ_mask=self.regions.full_organ_mask,
                                      occupied_mask=self.occupied_mask,
                                      organ_distance=self.regions.organ_depth, rng=rng,
                                      num_candidates=int(self.bank["draw_count"]),
                                      max_draws=max(int(self.generation["max_draws"]), int(self.bank["draw_count"])*500),
                                      min_liver_coverage=float(self.generation["min_liver_coverage"]),
                                      occupied_clearance_vox=int(self.generation["occupied_clearance_vox"]),
                                      min_center_separation_mm=float(self.generation["min_center_separation_mm"]))
            specs = self.specs(pool, self.regions, self.prototype, config=self.graph)
            for candidate, spec in zip(pool, specs):
                preparation_lease_verify()
                raw = tuple(map(int, candidate.center))
                if raw in old_raw:
                    self.last_proposal_diagnostics["raw_duplicate"] += 1
                    continue
                old_raw.add(raw)
                pre = self.benchmark._map_raw_point(raw, self.plans["transpose_forward"], self.properties, self.preseg.shape[1:])
                mapped = tuple(map(int, pre))
                if mapped in old_pre or not preprocessed_valid(pre, entry["source_mask"], entry["anchor_offset"], self.preseg,
                                                              liver_label=self.bank["liver_label"], tumor_label=self.bank["tumor_label"], minimum_coverage=0.85):
                    self.last_proposal_diagnostics["preprocessed_rejected"] += 1
                    continue
                try:
                    self.local(self.case, source, spec, full_organ_mask=self.regions.full_organ_mask,
                               organ_depth=self.regions.organ_depth, config=self.graph,
                               rng=np.random.default_rng(self.benchmark.stable_seed(self.global_seed, case_id, component, raw, "local")),
                               ct_clip=self.ct_clip, prepared_source=prepared)
                except Exception as exc:
                    # Exactly the original known geometry rejection classifier;
                    # every other exception fails preparation visibly.
                    message = str(exc).lower()
                    known = ("context coordinates are invalid", "coordinates are invalid", "zero context nodes",
                             "produced zero", "unrepresentable local geometry", "canonical node")
                    if isinstance(exc, self.geometry_error) or (isinstance(exc, ValueError) and any(token in message for token in known)):
                        self.last_proposal_diagnostics["graph_rejected"] += 1
                        continue
                    raise
                old_pre.add(mapped)
                yield np.asarray(raw, dtype=np.int32), pre.astype(np.int32)


def _initialize_worker(binding: Mapping) -> None:
    global _WORKER_PROVIDER, _WORKER_BINDING
    require(_WORKER_PROVIDER is None, "A preparation worker must initialize exactly once")
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    _WORKER_BINDING = copy.deepcopy(binding)
    preparation_lease_verify()
    _WORKER_PROVIDER = HistoricalGeometryProvider(binding)
    import torch
    torch.set_num_threads(1)
    require(not torch.cuda.is_initialized(), "CPU-only candidate preparation initialized CUDA")


def _build_patient(binding: Mapping, case_id: str, relative_entries: list[str], destination: str) -> list[dict]:
    # Each process handles one patient and all its sources, keeping patient CT,
    # labels, preseg and read-only region arrays cached across those sources.
    require(_WORKER_PROVIDER is not None and binding == _WORKER_BINDING,
            "Exact initialized historical worker binding required")
    provider = _WORKER_PROVIDER
    import torch
    segmentation = provider.load_patient(case_id)
    original_root = Path(binding["bank_index"]).resolve().parent
    destination_root = Path(destination)
    rows = []
    for relative in relative_entries:
        preparation_lease_verify()
        source_started = time.perf_counter()
        original_path = safe_entry(original_root, relative)
        identity, original_sha = stat_identity(original_path), file_sha(original_path)
        original = read_entry(original_path)
        result, proof = extend_entry(original, provider.proposals(case_id, original), segmentation,
                                     liver_label=provider.bank["liver_label"], tumor_label=provider.bank["tumor_label"],
                                     minimum_coverage=0.85, original_count=128, candidate_count=256)
        new_path = safe_entry(destination_root, relative)
        new_path.parent.mkdir(parents=True, exist_ok=True)
        with new_path.open("xb") as stream:
            np.savez(stream, **result)
            stream.flush()
            os.fsync(stream.fileno())
        # Re-read actual published NPZ, not only the in-memory concatenation.
        verify_preservation(original, read_entry(new_path), original_count=128, candidate_count=256)
        require(stat_identity(original_path) == identity and file_sha(original_path) == original_sha, "Original entry metadata/bytes changed")
        require(stat_identity(new_path)[:2] != identity[:2], "New entry must not alias old inode")
        row = {"case_id": case_id, "entry": relative, "original_entry_sha256": original_sha,
               "original_stat": identity, "new_entry_sha256": file_sha(new_path),
               "proof": proof, "proposal_diagnostics": provider.last_proposal_diagnostics,
               "seconds": time.perf_counter()-source_started,
               "patient_static_geometry_cache": {key: value for key, value in provider.patient_cache.items()
                                                  if key.endswith("_computations")}}
        rows.append(row)
        provider.verify_patient_inputs()
        preparation_lease_verify()
        require(not torch.cuda.is_initialized(), "CPU-only candidate preparation initialized CUDA")
        receipt_path = safe_entry(destination_root, relative).relative_to(destination_root)
        receipt_path = destination_root/"entry_receipts"/receipt_path.with_suffix(".json")
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        write_json_exclusive(receipt_path, row)
        print(f"[Basic256] {case_id} {relative} old128 preserved +128 /256", flush=True)
    return rows


def build_bank(binding: Mapping, destination: Path, *, workers: int, rss_limit_gib: float) -> dict:
    preparation_lease_verify()
    original_index = Path(binding["bank_index"]).resolve()
    original = read_json(original_index)
    validate_original_index(original)
    require(file_sha(original_index) == BANK_SHA256, "Original index SHA differs")
    require(isinstance(workers, int) and workers >= 2, "Measured parallel preparation worker count >=2 required")
    affinity = cpu_affinity()
    require(workers <= len(affinity), "Workers exceed actually allowed CPU affinity")
    require(np.isfinite(rss_limit_gib) and rss_limit_gib > 0, "Explicit measured preparation RAM limit required")
    rss_limit = int(rss_limit_gib*2**30)
    index_identity = stat_identity(original_index)
    destination = Path(destination).resolve()
    require(destination != original_index.parent and original_index.parent not in destination.parents,
            "New bank must be separate from the original bank tree")
    destination.mkdir(parents=True, exist_ok=False)
    (destination/"entry_receipts").mkdir()
    write_json_exclusive(destination/"preparation_started.json", {"format": FORMAT, "original_index_sha256": BANK_SHA256,
                                                                 "workers": workers, "CPU_affinity": affinity,
                                                                 "RSS_limit_bytes": rss_limit,
                                                                 "binding": binding, "started_at": time.time(), "full_sources": 642,
                                                                 "original_candidate_count": 128, "candidate_count": 256,
                                                                 "CUDA_model_or_scoring_executed": False})
    rows, completed_cases = [], 0
    started = time.time()
    peak_rss = process_tree_rss()
    progress = {"format": FORMAT, "status": "PREPARING", "source_completed": 0,
                "source_total": 642, "case_completed": 0, "case_total": 81,
                "candidate_count": 256, "workers": workers, "CPU_affinity": affinity,
                "RSS_limit_bytes": rss_limit, "RSS_bytes": peak_rss, "peak_RSS_bytes": peak_rss,
                "started_at": started, "updated_at": started,
                "raw_CT_preprocessing_executed": False, "GNN_scoring_executed": False}
    publish_progress(destination, progress)
    try:
        require(peak_rss <= rss_limit, "Candidate preparation initial RAM exceeds admitted limit")
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                                 initializer=_initialize_worker, initargs=(binding,)) as pool:
            # Largest source populations start first; each patient is loaded
            # once by its worker, and independent patients run in parallel.
            jobs = {pool.submit(_build_patient, binding, case_id, entries, str(destination)): case_id
                    for case_id, entries in sorted(original["entries_by_case"].items(),
                                                   key=lambda item: (-len(item[1]), item[0]))}
            pending = set(jobs)
            try:
                while pending:
                    preparation_lease_verify()
                    done, pending = wait(pending, timeout=1.0, return_when=FIRST_COMPLETED)
                    rss = process_tree_rss()
                    peak_rss = max(peak_rss, rss)
                    require(rss <= rss_limit, f"Preparation RAM {rss/2**30:.2f}GiB exceeds admitted {rss_limit_gib}GiB; no model/source reduction")
                    for future in done:
                        rows.extend(future.result())
                        completed_cases += 1
                    verified_sources = sum(1 for _ in (destination/"entry_receipts").rglob("*.json"))
                    progress.update(source_completed=verified_sources, case_completed=completed_cases,
                                    RSS_bytes=rss, peak_RSS_bytes=peak_rss, updated_at=time.time())
                    publish_progress(destination, progress)
            except Exception:
                for future in pending:
                    future.cancel()
                raise
    except Exception as exc:
        progress.update(status="FAILED", updated_at=time.time(), error=f"{type(exc).__name__}: {exc}")
        publish_progress(destination, progress)
        write_json_exclusive(destination/"preparation_failed.json", {"format": FORMAT, "complete": False,
                                                                   "error": f"{type(exc).__name__}: {exc}",
                                                                   "original_results_preserved": True,
                                                                   "no_source_or_candidate_count_fallback": True,
                                                                   "partial_outputs_preserved": True})
        raise
    require(len(rows) == 642, "All642 extended source receipts required")
    require(stat_identity(original_index) == index_identity and file_sha(original_index) == BANK_SHA256, "Original index changed")
    ordered = sorted(rows, key=lambda row: row["entry"])
    ablation = {"format": FORMAT, "uniform_candidate_count": 256, "scores_used": False,
                "historical_score_count": 128, "original_candidate_count": 128,
                "additional_GNN_scores_generated": False, "original_index": str(original_index),
                "original_index_sha256": BANK_SHA256, "geometry": GENERATION,
                "raw_pool_draw_count": 256, "raw_pool_max_draws": 128000, "raw_pool_attempts": 4,
                "historical_graph_geometry_required": True,
                "only_intervention": "uniform candidate universe128->256; original source/order/patch/RNG/appearance/CP probability retained"}
    index = copy.deepcopy(original)
    index.update(candidate_count=256, total_candidates=642*256, ablation=ablation,
                 manifest=str(destination/"preservation_manifest.json"))
    write_json_exclusive(destination/"preservation_manifest.json", {"format": FORMAT, "entries": ordered,
                                                                  "source_entries": 642, "eligible_cases": 81})
    write_json_exclusive(destination/"index.json", index)
    complete = {"format": FORMAT, "complete": True, "candidate_count": 256, "source_entries": 642,
                "eligible_cases": 81, "total_candidates": 642*256, "index_sha256": file_sha(destination/"index.json"),
                "original_index_sha256": BANK_SHA256, "preservation_manifest_sha256": file_sha(destination/"preservation_manifest.json"),
                "entries_sha256": {row["entry"]: row["new_entry_sha256"] for row in ordered},
                "all_original_arrays_and_first128_preserved": True, "all_256_unique_raw_and_preprocessed": True,
                "all_added128_historical_raw_preprocessed_graph_valid": True, "additional_GNN_scores_generated": False,
                "training_executed": False, "CUDA_initialized": False}
    complete.update(workers=workers, CPU_affinity=affinity, peak_RSS_bytes=peak_rss,
                    RSS_limit_bytes=rss_limit, elapsed_seconds=time.time()-started,
                    raw_CT_preprocessing_executed=False, patient_region_cache_rebuilt=False,
                    shared_preprocessed_arrays_written=False)
    write_json_exclusive(destination/"complete.json", complete)
    progress.update(status="COMPLETE", source_completed=642, case_completed=81,
                    updated_at=time.time(), index_sha256=complete["index_sha256"])
    publish_progress(destination, progress)
    return complete


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding", type=Path, required=True, help="Verified historical code/config/asset/data binding JSON")
    parser.add_argument("--output", type=Path, required=True, help="Fresh separate bank directory; existing paths refused")
    parser.add_argument("--workers", type=int, required=True, help="Measured CPU/RAM-admitted process worker count")
    parser.add_argument("--rss-limit-gib", type=float, required=True, help="Explicit aggregate CPU-preparation RAM budget, including workers")
    args = parser.parse_args(argv)
    receipt = build_bank(read_json(args.binding), args.output, workers=args.workers, rss_limit_gib=args.rss_limit_gib)
    print(json.dumps(receipt, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
