"""One CPU preparation for full642 Basic CP banks with512/1024/2048 positions.

The historical128 and optionally a verified completed256 prefix are byte exact.
Only new center arrays grow; original scores stay128 and are never used by Basic.
All new centers pass the frozen raw, preprocessed and graph geometry predicates.
The sampler alone uses a seeded finite permutation of liver voxels instead of
repeated rejection draws, so exhausted anatomy reports its measured cardinality.
Patient CT, region data and existing preprocessing are loaded once per worker job.
"""
from __future__ import annotations

import argparse
import copy
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import json
import multiprocessing
import os
from pathlib import Path
import time
import types
from typing import Mapping

import numpy as np

from . import basic_cp_ablation_bank as historical

FORMAT = "basic_cp_shared_large_candidates_preserved128_v1"
COUNTS = (512, 1024, 2048)
SEARCH = {
    "policy": "uniform_seeded_finite_liver_voxel_permutation_without_replacement_v1",
    "seed": "stable_seed(global_seed,case_id,component,finite_basic_candidate_extension)",
    "all_liver_center_voxels_searched_if_requested_count_unavailable": True,
    "historical_raw_preprocessed_graph_predicates_unchanged": True,
    "occupied_tumor_distance_mm": 12.0,
    "candidate_pair_spacing_constraint": False,
    "no_source_or_count_fallback": True,
    "additional_GNN_scoring": False,
}
_PROVIDER = None
_BINDING = None
_ROOT = None
_LEASE = None
_LEASE_IDENTITY = None

require = historical.require
read_json = historical.read_json
file_sha = historical.file_sha
stat_identity = historical.stat_identity
safe_entry = historical.safe_entry
read_entry = historical.read_entry
write_json_exclusive = historical.write_json_exclusive
BankContractError = historical.BankContractError


def verify_resources(root: Path | None = None) -> None:
    """Use the seven-experiment resource guard; cooperate with own abort marker."""
    global _LEASE, _LEASE_IDENTITY
    root = _ROOT if root is None else Path(root)
    if root is not None and (root/"preparation_abort.json").exists():
        record = read_json(root/"preparation_abort.json")
        require(record.get("format") == FORMAT and record.get("stop_remaining") is True,
                "Own large preparation abort marker differs")
        raise BankContractError("Large candidate preparation stopped: " + record["error"])
    value = os.environ.get("BASIC_CP_LEASE_REQUEST")
    if not value:
        require(_LEASE is None, "Established preparation lease disappeared")
        return
    path = Path(value).absolute()
    require(path.resolve(strict=True) == path and not path.is_symlink(),
            "Exact large preparation lease path required")
    identity = (str(path), stat_identity(path))
    if _LEASE is None:
        from .basic_cp_seven_resources import BasicLease
        _LEASE = BasicLease(read_json(path))
        _LEASE_IDENTITY = identity
    require(identity == _LEASE_IDENTITY, "Large preparation lease changed")
    _LEASE.verify()


def _failure(root: Path, progress: dict, exc: Exception) -> None:
    error = f"{type(exc).__name__}: {exc}"
    record = {"format": FORMAT, "stop_remaining": True, "error": error}
    marker = root/"preparation_abort.json"
    if not marker.exists():
        write_json_exclusive(marker, record)
    progress.update(status="FAILED", error=error, updated_at=time.time(),
                    existing_inputs_preserved=True, partial_outputs_preserved=True,
                    no_source_or_candidate_count_fallback=True)
    historical.publish_progress(root, progress)
    failed = root/"preparation_failed.json"
    if not failed.exists():
        write_json_exclusive(failed, dict(progress))


class FiniteLiverCenters:
    """Uniform permutation of actual liver voxel indices, with no retry loop.

    The1D integer order is the only per-source search allocation; CT/labels are
    shared read-only. Frozen raw predicates reject out-of-bounds source patches.
    Each yielded voxel is consumed once across successive256-candidate pools.
    """
    def __init__(self, center_mask: np.ndarray, flat_liver: np.ndarray, seed: int):
        require(center_mask.ndim == 3 and flat_liver.ndim == 1,
                "Actual3D liver mask and1D finite voxel catalogue required")
        require(np.issubdtype(flat_liver.dtype, np.integer), "Integer liver voxel catalogue required")
        self.mask = center_mask
        self.order = np.asarray(flat_liver).copy()
        np.random.default_rng(seed).shuffle(self.order)
        self.offset = 0
        self.diagnostics = {"finite_liver_center_count": len(self.order),
                            "finite_raw_centers_examined": 0, "full_liver_domain_exhausted": False,
                            "finite_permutation_bytes": self.order.nbytes}

    @property
    def exhausted(self) -> bool:
        return self.offset == len(self.order)

    def sampler(self, center_mask, patch_shape, *, rng, requested, max_draws, **kwargs):
        require(center_mask is self.mask, "Finite sampler liver identity changed")
        require(int(requested) > 0, "Positive frozen raw proposal pool size required")
        stop = min(len(self.order), self.offset + int(requested))
        while self.offset < stop:
            verify_resources()
            end = min(stop, self.offset + 2048)
            flat = self.order[self.offset:end]
            coordinates = np.column_stack(np.unravel_index(flat, self.mask.shape))
            for row in coordinates:
                self.offset += 1
                self.diagnostics["finite_raw_centers_examined"] = self.offset
                yield tuple(map(int, row))
        self.diagnostics["full_liver_domain_exhausted"] = self.exhausted


def finite_pool(pool, sampler):
    """Keep frozen pool bytecode/predicates, replacing only its voxel supplier."""
    require(pool.__closure__ is None and "sample_candidate_centers" in pool.__globals__,
            "Verified frozen pool must expose its sampler without a closure")
    namespace = dict(pool.__globals__)
    namespace["sample_candidate_centers"] = sampler
    result = types.FunctionType(pool.__code__, namespace, pool.__name__,
                                pool.__defaults__, pool.__closure__)
    result.__kwdefaults__ = pool.__kwdefaults__
    require(result.__code__ is pool.__code__, "Frozen raw predicate bytecode changed")
    return result


class LargeGeometryProvider(historical.HistoricalGeometryProvider):
    def load_patient(self, case_id: str) -> np.ndarray:
        segmentation = super().load_patient(case_id)
        self.flat_liver = np.flatnonzero(self.placement_mask).astype(np.int64, copy=False)
        self.flat_liver.flags.writeable = False
        return segmentation

    def proposals(self, case_id: str, entry: Mapping[str, np.ndarray]):
        component = int(np.asarray(entry["source_component"])[0])
        source = self.benchmark._source_from_component(self.case, self.components, component,
                                                       int(self.generation["source_pad"]))
        prepared = self.prepare(self.case, source, full_organ_mask=self.regions.full_organ_mask,
            organ_depth=self.regions.organ_depth, config=self.graph,
            rng=np.random.default_rng(self.benchmark.stable_seed(self.global_seed, case_id, component, "source")),
            ct_clip=self.ct_clip)
        seed = self.benchmark.stable_seed(self.global_seed, case_id, component,
                                          "finite_basic_candidate_extension")
        catalogue = FiniteLiverCenters(self.placement_mask, self.flat_liver, seed)
        pool_builder = finite_pool(self.patient_pool, catalogue.sampler)
        seen_raw = {tuple(map(int, row)) for row in entry["candidate_raw_centers"]}
        seen_pre = {tuple(map(int, row)) for row in entry["candidate_centers"]}
        diagnostics = {"case_id": case_id, "source_component": component,
            "prefix_candidate_count": len(seen_raw), "raw_graph_pool_batches": 0,
            "raw_pool_accepted": 0, "raw_duplicate_prefix": 0,
            "preprocessed_rejected": 0, "graph_rejected": 0,
            "accepted_added": 0, "search": SEARCH, "finite_search": catalogue.diagnostics}
        self.last_proposal_diagnostics = diagnostics
        while not catalogue.exhausted:
            verify_resources()
            before = catalogue.offset
            pool, _ = pool_builder(self.case, source, placement_mask=self.placement_mask,
                full_organ_mask=self.regions.full_organ_mask, occupied_mask=self.occupied_mask,
                organ_distance=self.regions.organ_depth, rng=np.random.default_rng(seed),
                num_candidates=256, max_draws=128000,
                min_liver_coverage=float(self.generation["min_liver_coverage"]),
                occupied_clearance_vox=int(self.generation["occupied_clearance_vox"]),
                min_center_separation_mm=float(self.generation["min_center_separation_mm"]))
            require(catalogue.offset > before or catalogue.exhausted,
                    "Frozen finite pool made no coordinate progress")
            diagnostics["raw_graph_pool_batches"] += 1
            diagnostics["raw_pool_accepted"] += len(pool)
            specs = self.specs(pool, self.regions, self.prototype, config=self.graph)
            require(len(specs) == len(pool), "Frozen graph specification cardinality differs")
            for candidate, spec in zip(pool, specs):
                verify_resources()
                raw = tuple(map(int, candidate.center))
                if raw in seen_raw:
                    diagnostics["raw_duplicate_prefix"] += 1
                    continue
                seen_raw.add(raw)
                pre = self.benchmark._map_raw_point(raw, self.plans["transpose_forward"],
                                                    self.properties, self.preseg.shape[1:])
                mapped = tuple(map(int, pre))
                if mapped in seen_pre or not historical.preprocessed_valid(pre,
                    entry["source_mask"], entry["anchor_offset"], self.preseg,
                    liver_label=self.bank["liver_label"], tumor_label=self.bank["tumor_label"],
                    minimum_coverage=.85):
                    diagnostics["preprocessed_rejected"] += 1
                    continue
                try:
                    self.local(self.case, source, spec, full_organ_mask=self.regions.full_organ_mask,
                        organ_depth=self.regions.organ_depth, config=self.graph,
                        rng=np.random.default_rng(self.benchmark.stable_seed(
                            self.global_seed, case_id, component, raw, "local")),
                        ct_clip=self.ct_clip, prepared_source=prepared)
                except Exception as exc:
                    message = str(exc).lower()
                    known = ("context coordinates are invalid", "coordinates are invalid", "zero context nodes",
                             "produced zero", "unrepresentable local geometry", "canonical node")
                    if isinstance(exc, self.geometry_error) or (isinstance(exc, ValueError)
                                                               and any(token in message for token in known)):
                        diagnostics["graph_rejected"] += 1
                        continue
                    raise
                seen_pre.add(mapped)
                diagnostics["accepted_added"] += 1
                yield np.asarray(raw, dtype=np.int32), np.asarray(pre, dtype=np.int32)
        diagnostics["finite_search"]["full_liver_domain_exhausted"] = True
        raise BankContractError("Finite candidate domain exhausted " + json.dumps({
            "case_id": case_id, "source_component": component,
            "available_unique_raw_and_preprocessed_graph_valid": len(entry["candidate_centers"])
                + diagnostics["accepted_added"], "required_candidate_count": 2048,
            "all_liver_center_voxels_checked": True, "diagnostics": diagnostics}, ensure_ascii=False))


def collect_prefixes(original: Mapping[str, np.ndarray], prefix: Mapping[str, np.ndarray], proposals,
                     *, targets: tuple[int, ...], publish):
    """Consume one stream, publishing every nested prefix exactly at its count.

    Explicit DEBUG target tuples allow isolated tests; production enforcesCOUNTS.
    Scores retain original cardinality, including when starting at verified256.
    """
    require(targets and targets == tuple(sorted(set(targets))), "Strict ascending prefix targets required")
    original_count, prefix_count = len(original["candidate_centers"]), len(prefix["candidate_centers"])
    require(prefix_count < targets[0], "Requested prefixes must extend admitted source prefix")
    historical.verify_preservation(original, prefix, original_count=original_count,
                                   candidate_count=prefix_count)
    raw = [np.asarray(row) for row in prefix["candidate_raw_centers"]]
    pre = [np.asarray(row) for row in prefix["candidate_centers"]]
    seen_raw, seen_pre = {tuple(map(int,row)) for row in raw}, {tuple(map(int,row)) for row in pre}
    next_target = iter(targets)
    target = next(next_target)
    for raw_row, pre_row in proposals:
        raw_row, pre_row = np.asarray(raw_row), np.asarray(pre_row)
        require(raw_row.shape == pre_row.shape == (3,) and np.issubdtype(raw_row.dtype,np.integer)
                and np.issubdtype(pre_row.dtype,np.integer), "Integral3D proposal required")
        raw_key, pre_key = tuple(map(int,raw_row)), tuple(map(int,pre_row))
        require(raw_key not in seen_raw and pre_key not in seen_pre,
                "Provider returned duplicate admitted raw/preprocessed center")
        raw.append(raw_row.astype(original["candidate_raw_centers"].dtype))
        pre.append(pre_row.astype(original["candidate_centers"].dtype))
        seen_raw.add(raw_key); seen_pre.add(pre_key)
        if len(raw) == target:
            result = {name: np.asarray(value).copy() for name,value in original.items()}
            result["candidate_raw_centers"], result["candidate_centers"] = np.stack(raw), np.stack(pre)
            proof = historical.verify_preservation(original,result,
                original_count=original_count,candidate_count=target)
            # Verified256 is also preserved; scores remain the historical128.
            for name in ("candidate_raw_centers","candidate_centers"):
                require(result[name][:prefix_count].tobytes() == prefix[name].tobytes(),
                        "Admitted prefix changed during nested publication")
            proof.update(admitted_prefix_count=prefix_count,
                         all_admitted_prefix_centers_byte_preserved=True)
            publish(target, result, proof)
            target = next(next_target, None)
            if target is None:
                return
    raise BankContractError(f"Finite proposals ended: measured{len(raw)} candidates, required{target}; "
                            "no smaller bank or missing source published")


def _initialize(binding, root):
    global _PROVIDER, _BINDING, _ROOT
    require(_PROVIDER is None, "Large candidate worker initializes exactly once")
    os.environ.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                      CUDA_VISIBLE_DEVICES="")
    _ROOT = Path(root)
    _BINDING = copy.deepcopy(binding)
    verify_resources()
    _PROVIDER = LargeGeometryProvider(binding)
    import torch
    torch.set_num_threads(1)
    require(not torch.cuda.is_initialized(), "CPU bank preparation initializedCUDA")


def _build_patient(binding, case_id, relative_entries, destination, prefix_root):
    require(_PROVIDER is not None and binding == _BINDING, "Exact worker geometry binding required")
    provider = _PROVIDER
    root = Path(destination)
    old_root = Path(binding["bank_index"]).parent
    segmentation = provider.load_patient(case_id)
    for relative in relative_entries:
        verify_resources()
        started = time.perf_counter()
        original_path = safe_entry(old_root, relative)
        old_identity, old_sha = stat_identity(original_path), file_sha(original_path)
        original = read_entry(original_path)
        prefix_path = original_path if prefix_root is None else safe_entry(Path(prefix_root),relative)
        prefix_identity, prefix_sha = stat_identity(prefix_path), file_sha(prefix_path)
        prefix = original if prefix_root is None else read_entry(prefix_path)
        require(all(historical.preprocessed_valid(center,original["source_mask"],
            original["anchor_offset"],segmentation,liver_label=1,tumor_label=2,minimum_coverage=.85)
            for center in prefix["candidate_centers"]), "Admitted prefix geometry differs from actual patient")

        def publish(count, result, proof):
            verify_resources()
            destination_root = root/str(count)
            target = safe_entry(destination_root,relative)
            target.parent.mkdir(parents=True,exist_ok=True)
            with target.open("xb") as stream:
                np.savez(stream,**result)
                stream.flush();os.fsync(stream.fileno())
            historical.verify_preservation(original,read_entry(target),original_count=128,candidate_count=count)
            require(stat_identity(original_path) == old_identity and file_sha(original_path) == old_sha
                    and stat_identity(prefix_path) == prefix_identity and file_sha(prefix_path) == prefix_sha,
                    "Original/admitted prefix changed during preparation")
            row = {"entry":relative,"case_id":case_id,"original_stat":old_identity,
                "original_entry_sha256":old_sha,"new_entry_sha256":file_sha(target),"proof":proof,
                "admitted_prefix": {"path":str(prefix_path),"sha256":prefix_sha,"stat":prefix_identity},
                "proposal_diagnostics":copy.deepcopy(provider.last_proposal_diagnostics),
                "seconds":time.perf_counter()-started}
            receipt = destination_root/"entry_receipts"/Path(relative).with_suffix(".json")
            receipt.parent.mkdir(parents=True,exist_ok=True)
            write_json_exclusive(receipt,row)
            provider.verify_patient_inputs()
            print(f"[Basic{count}] {case_id} {relative} full{count} historical128 preserved",flush=True)

        collect_prefixes(original,prefix,provider.proposals(case_id,prefix),targets=COUNTS,publish=publish)
    import torch
    require(not torch.cuda.is_initialized(),"CPU preparation initializedCUDA")
    return case_id


def _ablation(original_index: Path, count: int, root: Path):
    return {"format":FORMAT,"uniform_candidate_count":count,"scores_used":False,
        "historical_score_count":128,"original_candidate_count":128,
        "additional_GNN_scores_generated":False,"original_index":str(original_index),
        "original_index_sha256":historical.BANK_SHA256,"geometry":historical.GENERATION,
        "historical_graph_geometry_required":True,"additional_candidate_search":SEARCH,
        "shared_generation_root":str(root),"nested_candidate_counts":list(COUNTS),
        "only_intervention":f"uniform candidate universe128->{count}; original source/patch/appearance/CP probability retained"}


def _publish_completed_prefix(root: Path, count: int, binding: Mapping, started: dict,
                              *, peak_rss: int, workers: int, affinity: list[int], rss_limit: int):
    bank_root = root/str(count)
    if (bank_root/"complete.json").exists():
        return None
    original_index = Path(binding["bank_index"])
    original = read_json(original_index)
    names = {name:case for case,entries in original["entries_by_case"].items() for name in entries}
    paths = list((bank_root/"entry_receipts").rglob("*.json"))
    if len(paths) != 642:
        return None
    rows = sorted((read_json(path) for path in paths),key=lambda row:row["entry"])
    require([row["entry"] for row in rows] == sorted(names),"Every642 original source required")
    for row in rows:
        require(row["case_id"] == names[row["entry"]]
                and file_sha(safe_entry(bank_root,row["entry"])) == row["new_entry_sha256"]
                and stat_identity(safe_entry(original_index.parent,row["entry"])) == row["original_stat"],
                "Completed prefix entry/source proof differs")
    manifest = bank_root/"preservation_manifest.json"
    write_json_exclusive(manifest,{"format":FORMAT,"entries":rows,"source_entries":642,"eligible_cases":81})
    index = copy.deepcopy(original)
    index.update(candidate_count=count,total_candidates=642*count,
        ablation=_ablation(original_index,count,root),manifest=str(manifest))
    write_json_exclusive(bank_root/"index.json",index)
    complete = {"format":FORMAT,"complete":True,"candidate_count":count,"source_entries":642,
        "eligible_cases":81,"total_candidates":642*count,"index_sha256":file_sha(bank_root/"index.json"),
        "original_index_sha256":historical.BANK_SHA256,"preservation_manifest_sha256":file_sha(manifest),
        "entries_sha256":{row["entry"]:row["new_entry_sha256"] for row in rows},
        "all_original_arrays_and_first128_preserved":True,"all_centers_unique_raw_and_preprocessed":True,
        "all_added_historical_raw_preprocessed_graph_valid":True,"additional_GNN_scores_generated":False,
        "training_executed":False,"CUDA_initialized":False,"raw_CT_preprocessing_executed":False,
        "patient_region_cache_rebuilt":False,"shared_preprocessed_arrays_written":False,
        "workers":workers,"CPU_affinity":affinity,"peak_RSS_bytes":peak_rss,"RSS_limit_bytes":rss_limit,
        "elapsed_seconds":time.time()-started["started_at"],"additional_candidate_search":SEARCH,
        "shared_generation_root":str(root),"prepared256_reuse_admission":started["prepared256_reuse_admission"]}
    write_json_exclusive(bank_root/"complete.json",complete)
    return complete


def build_shared_banks(binding: Mapping, destination: Path, *, workers: int, rss_limit_gib: float,
                       prepared256_witness: Mapping | None = None,
                       binding_path: Path | None = None, binding_sha256: str | None = None) -> dict:
    """Prepare3 full642 banks in one patient-parallelCPU pass, publish prefixes early."""
    global _ROOT
    verify_resources()
    original_index = Path(binding["bank_index"]).resolve(strict=True)
    historical.validate_original_index(read_json(original_index))
    require(file_sha(original_index) == historical.BANK_SHA256,"Original128 indexSHA differs")
    require(isinstance(workers,int) and workers >= 2,"Measured parallel worker count>=2 required")
    affinity = historical.cpu_affinity()
    require(workers <= len(affinity),"Workers exceed measuredCPU affinity")
    require(np.isfinite(rss_limit_gib) and rss_limit_gib > 0,"Explicit measured aggregateRAM limit required")
    rss_limit = int(rss_limit_gib*2**30)
    admission = None
    prefix_root = None
    if prepared256_witness is not None:
        require(binding_path is not None and binding_sha256 is not None,
                "Completed256 prefix requires exact bindingfile andSHA")
        require(read_json(binding_path) == binding,"Completed256 binding differs")
        admission = historical.verify_prepared_bank(prepared256_witness,binding_path,binding_sha256)
        prefix_root = str(Path(admission["index"]).parent)
    root = Path(destination).resolve()
    require(root != original_index.parent and original_index.parent not in root.parents,
            "Fresh shared banks must be separate from original bank")
    if prefix_root is not None:
        require(root != Path(prefix_root) and Path(prefix_root) not in root.parents,
                "Fresh shared banks must be separate from admitted256 prefix")
    root.mkdir(parents=True,exist_ok=False)
    _ROOT = root
    started = {"format":FORMAT,"binding":copy.deepcopy(binding),"original_index_sha256":historical.BANK_SHA256,
        "full_sources":642,"eligible_cases":81,"original_candidate_count":128,"candidate_counts":list(COUNTS),
        "candidate_count":2048,"additional_candidate_search":SEARCH,"CUDA_model_or_scoring_executed":False,
        "started_at":time.time(),"prepared256_reuse_admission":admission,
        "workers":workers,"CPU_affinity":affinity,"RSS_limit_bytes":rss_limit}
    write_json_exclusive(root/"preparation_started.json",started)
    for count in COUNTS:
        (root/str(count)/"entry_receipts").mkdir(parents=True)
        write_json_exclusive(root/str(count)/"preparation_started.json",dict(started,candidate_count=count))
    progress = {"format":FORMAT,"status":"PREPARING","source_total":642,"case_total":81,
        "case_completed":0,"source_completed":0,"prefix_completed":{str(count):0 for count in COUNTS},
        "candidate_count":2048,"workers":workers,"started_at":started["started_at"],
        "updated_at":time.time(),"raw_CT_preprocessing_executed":False,"GNN_scoring_executed":False}
    historical.publish_progress(root,progress)
    peak_rss = historical.process_tree_rss()
    completed = {}
    try:
        require(peak_rss <= rss_limit,"Initial bank preparationRAM exceeds admitted budget")
        with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context("spawn"),
                                 initializer=_initialize,initargs=(binding,str(root))) as pool:
            futures = set()
            try:
                for case_id,entries in sorted(read_json(original_index)["entries_by_case"].items(),
                                              key=lambda row:(-len(row[1]),row[0])):
                    futures.add(pool.submit(_build_patient,binding,case_id,entries,str(root),prefix_root))
                while futures:
                    verify_resources()
                    done,futures = wait(futures,timeout=1,return_when=FIRST_COMPLETED)
                    rss = historical.process_tree_rss()
                    peak_rss = max(peak_rss,rss)
                    require(rss <= rss_limit,f"PreparationRAM{rss/2**30:.2f}GiB exceeds{rss_limit_gib}GiB; no source/count reduction")
                    for future in done:
                        future.result()
                        progress["case_completed"] += 1
                    for count in COUNTS:
                        progress["prefix_completed"][str(count)] = sum(1 for _ in (root/str(count)/"entry_receipts").rglob("*.json"))
                        receipt = _publish_completed_prefix(root,count,binding,started,peak_rss=peak_rss,
                            workers=workers,affinity=affinity,rss_limit=rss_limit)
                        if receipt is not None:
                            completed[str(count)] = receipt
                    progress.update(source_completed=progress["prefix_completed"]["2048"],
                                    RSS_bytes=rss,peak_RSS_bytes=peak_rss,updated_at=time.time())
                    historical.publish_progress(root,progress)
            except Exception as exc:
                _failure(root,progress,exc)
                for future in futures:
                    future.cancel()
                raise
        require(set(completed) == set(map(str,COUNTS)),"All3 full642 bank completions required")
        require(file_sha(original_index) == historical.BANK_SHA256,"Original index changed during full preparation")
        result = {"format":FORMAT,"complete":True,"prefixes":completed,"full_sources":642,
                  "additional_candidate_search":SEARCH,"peak_RSS_bytes":peak_rss,
                  "CUDA_initialized":False,"training_executed":False}
        write_json_exclusive(root/"complete.json",result)
        progress.update(status="COMPLETE",updated_at=time.time())
        historical.publish_progress(root,progress)
        return result
    except Exception as exc:
        _failure(root,progress,exc)
        raise


def verify_prepared_bank(witness: Mapping, binding_path: Path, binding_sha256: str,
                         candidate_count: int | None = None) -> dict:
    """Read-only full642 admission; ordinary actual geometry preflight still applies."""
    require(set(witness) == {"index","index_sha256","complete_sha256","manifest_sha256",
                            "preparation_started_sha256"},"Exact completedlarge bank witness required")
    observed = {}

    def admit(path, digest=None):
        path = Path(path)
        require(path.is_absolute() and path.resolve(strict=True) == path and path.is_file()
                and not any(p.is_symlink() for p in (path,*path.parents)),"Exact regular bank evidence path required")
        identity = stat_identity(path)
        actual = file_sha(path)
        require(stat_identity(path) == identity and (digest is None or actual == digest),
                f"Completedlarge evidenceSHA changed:{path}")
        observed[str(path)] = identity
        return path,actual

    binding_file,_ = admit(binding_path,binding_sha256)
    binding = read_json(binding_file)
    index_path,index_sha = admit(witness["index"],witness["index_sha256"])
    root = index_path.parent
    require(index_path.name == "index.json","Actual index.json required")
    complete_path,complete_sha = admit(root/"complete.json",witness["complete_sha256"])
    manifest_path,manifest_sha = admit(root/"preservation_manifest.json",witness["manifest_sha256"])
    started_path,started_sha = admit(root/"preparation_started.json",witness["preparation_started_sha256"])
    complete,manifest,started,index = map(read_json,(complete_path,manifest_path,started_path,index_path))
    count = index.get("candidate_count")
    require(count in COUNTS and (candidate_count is None or candidate_count == count),
            "Completedlarge candidate count differs")
    shared = root.parent
    require(root.name == str(count),"Shared prefix directory must name its exact candidate count")
    require(started.get("format") == FORMAT and started.get("binding") == binding
            and started.get("candidate_count") == count and started.get("candidate_counts") == list(COUNTS)
            and started.get("full_sources") == 642 and started.get("additional_candidate_search") == SEARCH
            and started.get("CUDA_model_or_scoring_executed") is False,
            "Completedlarge actual preparation contract differs")
    original_path,original_sha = admit(binding["bank_index"],historical.BANK_SHA256)
    original = read_json(original_path)
    historical.validate_original_index(original)
    expected = copy.deepcopy(original)
    expected.update(candidate_count=count,total_candidates=642*count,
                    ablation=_ablation(original_path,count,shared),manifest=str(manifest_path))
    require(index == expected,"Completedlarge bank changes beyond declared candidates")
    required = {"format":FORMAT,"complete":True,"candidate_count":count,"source_entries":642,
        "eligible_cases":81,"total_candidates":642*count,"index_sha256":index_sha,
        "original_index_sha256":original_sha,"preservation_manifest_sha256":manifest_sha,
        "all_original_arrays_and_first128_preserved":True,"all_centers_unique_raw_and_preprocessed":True,
        "all_added_historical_raw_preprocessed_graph_valid":True,"additional_GNN_scores_generated":False,
        "training_executed":False,"CUDA_initialized":False,"raw_CT_preprocessing_executed":False,
        "patient_region_cache_rebuilt":False,"shared_preprocessed_arrays_written":False,
        "additional_candidate_search":SEARCH,"shared_generation_root":str(shared),
        "prepared256_reuse_admission":started.get("prepared256_reuse_admission")}
    require(all(complete.get(key) == value for key,value in required.items()),
            "Completedlarge full geometry/preservation receipt differs")
    prepared256 = started.get("prepared256_reuse_admission")
    prefix_root = None
    if prepared256 is not None:
        require(isinstance(prepared256,dict),"Actual completed256 reuse admission required")
        admitted256_witness = {key:prepared256[key] for key in
            ("index","index_sha256","complete_sha256","manifest_sha256","preparation_started_sha256")}
        actual256 = historical.verify_prepared_bank(admitted256_witness,binding_file,binding_sha256)
        require(actual256 == prepared256,"Completed256 prefix admission changed")
        prefix_root = Path(actual256["index"]).parent
    names = {name:case for case,entries in original["entries_by_case"].items() for name in entries}
    require(manifest.get("format") == FORMAT and manifest.get("source_entries") == 642
            and manifest.get("eligible_cases") == 81,"Full642 preservation manifest required")
    rows = manifest.get("entries",[])
    require([row.get("entry") for row in rows] == sorted(names),"Manifest642 source identities differ")
    hashes = complete.get("entries_sha256",{})
    require(set(hashes) == set(names),"All642 outputSHA witnesses required")
    for row in rows:
        relative = row["entry"]
        old_path,old_sha = admit(safe_entry(original_path.parent,relative),row["original_entry_sha256"])
        new_path,new_sha = admit(safe_entry(root,relative),hashes[relative])
        require(row.get("case_id") == names[relative] and row["new_entry_sha256"] == new_sha
                and row["original_stat"] == observed[str(old_path)]
                and observed[str(old_path)][:2] != observed[str(new_path)][:2],
                "Actual source identity/inode differs")
        old,new = read_entry(old_path),read_entry(new_path)
        proof = historical.verify_preservation(old,new,original_count=128,candidate_count=count)
        require(all(row.get("proof",{}).get(key) == value for key,value in proof.items()),
                "Actual original128 array/prefix proof differs")
        prefix_info = row.get("admitted_prefix",{})
        prefix_path,prefix_sha = admit(prefix_info["path"],prefix_info["sha256"])
        require(prefix_info["stat"] == observed[str(prefix_path)],"Admitted prefix metadata changed")
        prefix = read_entry(prefix_path)
        prefix_count = len(prefix["candidate_centers"])
        require(prefix_count in (128,256) and row["proof"].get("admitted_prefix_count") == prefix_count,
                "Admitted original/preserved256 prefix count differs")
        require((prefix_root is None and prefix_count == 128 and prefix_path == old_path)
                or (prefix_root is not None and prefix_count == 256
                    and prefix_path == safe_entry(prefix_root,relative)),
                "Prefix must be the exact original or separately verified completed256 source")
        historical.verify_preservation(old,prefix,original_count=128,candidate_count=prefix_count)
        require(all(new[name][:prefix_count].tobytes() == prefix[name].tobytes()
                    for name in ("candidate_centers","candidate_raw_centers")),
                "Completedlarge original/256 prefix changed")
    require(all(stat_identity(Path(path)) == identity for path,identity in observed.items()),
            "Completedlarge inputs changed during admission")
    return {"format":"basic_cp_completedlarge_bank_reuse_admission_v1","complete":True,
        "index":str(index_path),"index_sha256":index_sha,"source_index":str(original_path),
        "source_index_sha256":original_sha,"complete_sha256":complete_sha,"manifest_sha256":manifest_sha,
        "preparation_started_sha256":started_sha,"binding_sha256":binding_sha256,
        "source_entries":642,"eligible_cases":81,"candidate_count":count,"total_candidates":642*count,
        "all642_entry_sha_verified":True,"original_arrays_and_first128_byte_verified":True,
        "historical_geometry_completion_proof_verified":True,"additional_candidate_search":SEARCH,
        "full_geometry_preflight_still_required":True,"bank_regenerated":False,
        "source_or_prepared_bank_written":False,"training_executed":False,"immutable_files":observed}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding",required=True,type=Path)
    parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--workers",required=True,type=int)
    parser.add_argument("--rss-limit-gib",required=True,type=float)
    parser.add_argument("--prepared256-witness",type=Path)
    args = parser.parse_args(argv)
    receipt = build_shared_banks(read_json(args.binding),args.output,workers=args.workers,
        rss_limit_gib=args.rss_limit_gib,prepared256_witness=None if args.prepared256_witness is None
        else read_json(args.prepared256_witness),binding_path=args.binding,binding_sha256=file_sha(args.binding))
    print(json.dumps(receipt,ensure_ascii=False,allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
