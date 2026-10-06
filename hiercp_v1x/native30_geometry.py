"""Original native ROI30/context28 geometry on the signed common P+128U cohort.

The active hiercp modules belong to the caller's preserved original source.
No bounded scope or newer sampling operator is installed. Only selected complete
inner-validation cases and their assigned donors are decoded. GT validates raw
observation provenance outside graph construction and never enters model inputs.
"""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
import copy
from dataclasses import dataclass, fields, is_dataclass, replace
import hashlib
import importlib
import json
from pathlib import Path
import shutil
import threading
import time

import numpy as np
import psutil
import torch

from .transition_evaluation import QUERY_FIELDS, validate_cohort

FORMAT = "original_native30_context28_common_P128U_geometry_v1"
ROLES = ("tumor_surface", "tumor_interior", "source_context",
         "source_liver_surface", "target_context", "target_liver_surface")


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode()).hexdigest()


def _write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def _save_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(path.parent).free <= resident_bytes_of(value) + 8*1024**2:
        raise OSError("Native30 geometry storage lacks room for the complete record; no input truncated")
    with path.open("xb") as stream:
        torch.save(value, stream)


def _runtime(source):
    modules = {name: importlib.import_module("hiercp." + name)
               for name in ("common", "schema", "local", "sample", "spatial", "curriculum")}
    for name, module in modules.items():
        if Path(module.__file__).resolve() != source / "hiercp" / (name + ".py"):
            raise ValueError("Activate the explicit original source before native30 geometry: " + name)
    if tuple(modules["schema"].LOCAL_NODE_TYPES) != ROLES:
        raise ValueError("Original six semantic local roles required")
    return modules


def resident_bytes_of(value):
    """Whole backing storage union, including shared NumPy/Tensor aliases."""
    intervals, seen = {}, set()
    def visit(item):
        if id(item) in seen:
            return
        seen.add(id(item))
        if isinstance(item, torch.Tensor):
            storage = item.untyped_storage()
            if storage.nbytes():
                intervals.setdefault(str(item.device), []).append(
                    (storage.data_ptr(), storage.data_ptr() + storage.nbytes()))
        elif isinstance(item, np.ndarray):
            root = item
            while isinstance(root.base, np.ndarray):
                root = root.base
            if isinstance(root.base, torch.Tensor):
                visit(root.base)
            elif root.size:
                bounds = (np.byte_bounds(root) if hasattr(np, "byte_bounds")
                          else np.lib.array_utils.byte_bounds(root))
                intervals.setdefault("cpu", []).append(tuple(map(int, bounds)))
        elif isinstance(item, Mapping):
            for child in item.values():
                visit(child)
        elif isinstance(item, (tuple, list)):
            for child in item:
                visit(child)
        elif is_dataclass(item):
            for field in fields(item):
                visit(getattr(item, field.name))
        elif hasattr(item, "to_dict"):
            visit(item.to_dict())
    visit(value)
    total = 0
    for values in intervals.values():
        end = None
        for first, last in sorted(values):
            total += last - first if end is None or first >= end else max(0, last - end)
            end = last if end is None else max(end, last)
    return total


def physical_transport(donor, target, source, center):
    """Audit native-axis identity CP in both patients' actual affine frames."""
    donor_frame = np.asarray(donor.image_affine)[:3, :3] / np.asarray(donor.spacing)[None]
    target_frame = np.asarray(target.image_affine)[:3, :3] / np.asarray(target.spacing)[None]
    matrix = target_frame @ np.linalg.inv(donor_frame)
    from_anchor = (donor.image_affine @ np.r_[source.anchor_center, 1.])[:3]
    to_anchor = (target.image_affine @ np.r_[center, 1.])[:3]
    world = np.eye(4)
    world[:3, :3] = matrix
    world[:3, 3] = to_anchor - matrix @ from_anchor
    if not np.allclose(world @ np.r_[from_anchor, 1.], np.r_[to_anchor, 1.], atol=1e-7):
        raise RuntimeError("Original donor/recipient anchor transport failed")
    return dict(axis_order="native_ijk", donor_native_spacing=list(map(float, donor.spacing)),
        recipient_native_spacing=list(map(float, target.spacing)),
        donor_anchor_native_ijk=list(map(int, source.anchor_center)),
        recipient_anchor_native_ijk=list(map(int, center)),
        donor_anchor_world_mm=from_anchor.tolist(), recipient_anchor_world_mm=to_anchor.tolist(),
        donor_relative_mm_to_recipient_relative_mm=np.eye(3).tolist(),
        donor_world_to_recipient_world_mm=world.tolist(),
        convention="native CP identity in native-axis mm; source nodes remain donor-native; no hidden reorientation")


@dataclass
class Native30Batch:
    graph: object
    source_patches: torch.Tensor
    target_patches: torch.Tensor
    source_index: torch.Tensor
    graph_observation_index: torch.Tensor
    indices: torch.Tensor

    def __len__(self):
        return len(self.target_patches)

    def to(self, device, non_blocking=True):
        from copy import copy as shallow_copy
        return Native30Batch(shallow_copy(self.graph).to(device, non_blocking=non_blocking),
            *[getattr(self, name).to(device, non_blocking=non_blocking) for name in
              ("source_patches", "target_patches", "source_index", "graph_observation_index", "indices")])

    def pin_memory(self):
        from copy import copy as shallow_copy
        return Native30Batch(shallow_copy(self.graph).pin_memory(),
            *[getattr(self, name).pin_memory() for name in
              ("source_patches", "target_patches", "source_index", "graph_observation_index", "indices")])


def collate_native30(items):
    """One disjoint 2N graph batch, with invariant donor dense maps shared."""
    from torch_geometric.data import Batch
    if not isinstance(items, (tuple, list)) or not items:
        raise ValueError("Complete nonempty original observation batch required")
    graphs, sources, targets, source_ids, indices, lookup = [], [], [], [], [], {}
    for (views, source, target), index in items:
        if type(index) is not int or index < 0 or len(views) != 2:
            raise ValueError("Two genuine views and original observation index required")
        first, second = views
        if (first.native30_record_id != second.native30_record_id
                or first.native30_epoch.tolist() != second.native30_epoch.tolist()
                or first.native30_view.tolist() != [0] or second.native30_view.tolist() != [1]):
            raise ValueError("Original paired view identity/order differs")
        for graph in views:
            if tuple(graph.node_types) != ROLES or any(graph[role].num_nodes < 1 for role in ROLES):
                raise ValueError("All six actual nonempty roles required; empty roles are not fabricated")
        for patch in (source, target):
            if not isinstance(patch, torch.Tensor) or patch.shape != (5, 48, 48, 48) or patch.device.type != "cpu":
                raise ValueError("Original CPU five-channel fixed48 patches required")
        key = (source.untyped_storage().data_ptr(), source.storage_offset(), source.dtype, tuple(source.shape))
        if key not in lookup:
            lookup[key] = len(sources)
            sources.append(source)
        source_ids.append(lookup[key])
        graphs.extend(views)
        targets.append(target)
        indices.append(index)
    return Native30Batch(Batch.from_data_list(graphs), torch.stack(sources), torch.stack(targets),
        torch.tensor(source_ids, dtype=torch.long), torch.arange(len(items)).repeat_interleave(2),
        torch.tensor(indices, dtype=torch.long))


class Native30Geometry:
    """Whole evaluation geometry, parallel CPU work, reusable byte-bound cache."""
    def __init__(self, inventory_path, config, source, *, workers, resident_bytes,
                 rss_bytes, output, debug=False, debug_case_ids=None, read_only=False):
        if type(read_only) is not bool:
            raise ValueError("Explicit boolean read_only mode required")
        self.read_only = read_only
        if (type(workers) is not int or workers < 2 or type(resident_bytes) is not int
                or type(rss_bytes) is not int or not 0 < resident_bytes < rss_bytes):
            raise ValueError("Explicit parallel workers>=2 and positive resident/RSS headroom required")
        literal_inventory = Path(inventory_path)
        if literal_inventory.is_symlink():
            raise ValueError("Regular bound native inventory required")
        self.path = literal_inventory.resolve(strict=True)
        self.index_sha256 = _sha(self.path)
        self.meta = json.loads(self.path.read_text(encoding="utf8"))
        self.population = validate_cohort(self.meta, case_ids=debug_case_ids, debug=debug)
        self.rows = self.population["rows"]
        self.by_case = self.population["by_case"]
        self.case_ids = self.population["case_ids"]
        self.assignment_sha256 = self.population["native_assignment"]["assignment_sha256"]
        self.debug = debug
        self.ds = self
        self.base = self.config = copy.deepcopy(config)
        self.source = Path(source).resolve(strict=True)
        self.runtime = _runtime(self.source)
        self.source_sha256 = {name: _sha(module.__file__) for name, module in self.runtime.items()}
        from . import native30_data_contract
        self.data_contract_path = Path(native30_data_contract.__file__).resolve()
        self.data_contract_sha256 = _sha(self.data_contract_path)
        self.data_contract_receipt = native30_data_contract.source_receipt()
        self.graph_config = self.runtime["schema"].graph_config_from_dict(dict(self.config["graph"]))
        self.runtime["local"]._require_full_graph(self.graph_config)
        if (self.config.get("seed") != 42 or self.graph_config.patch_size != 48
                or self.graph_config.adaptive_roi_margin_mm != 30.
                or self.graph_config.context_outer_radius_mm != 28.
                or tuple(self.graph_config.context_shells_mm) != (4., 12., 28.)):
            raise ValueError("Exact original seed42/native ROI30/context28/shells4,12,28/fixed48 required")
        self.clip = tuple(self.config["ct_clip"])
        if self.clip != (-200., 250.):
            raise ValueError("Saved original CT clipping contract required")
        self.workers, self.resident_bytes, self.rss_bytes = workers, resident_bytes, rss_bytes
        literal_output = Path(output)
        if self.read_only and (literal_output.is_symlink() or not literal_output.is_dir()):
            raise ValueError("Read-only reuse requires an existing regular geometry directory")
        self.output = literal_output.resolve(strict=self.read_only)
        for old in (self.source, self.path.parent):
            if self.output == old or self.output.is_relative_to(old) or old.is_relative_to(self.output):
                raise ValueError("New geometry cache must be disjoint from preserved source/inventory")
        self.raw_metadata = {item["case_id"]: item for item in self.meta["raw_records"]}
        self._row_lookup = {row["id"]: row for row in self.rows}
        self._indices = {row["id"]: index for index, row in enumerate(self.rows)}
        self.allowed_raw_cases = {row[key] for row in self.rows for key in ("case_id", "donor_case_id")}
        self._raw, self._donors, self._records, self._shared = (OrderedDict() for _ in range(4))
        self._canonical, self._source_receipts = {}, {}
        self._source_lock = threading.Lock()
        self._active_shared_paths = set()
        self._active_cases = set()
        self._file_signatures = {}
        self.timings = []
        self.hits = self.builds = self.loads = 0
        self.last_admission = None
        self._read_only_metadata = {}
        self._read_only_index = None
        self._read_only_payload_files = 0
        self.request = dict(format=FORMAT, input_inventory_sha256=self.index_sha256,
            cohort_sha256=self.population["cohort_sha256"], assignment_sha256=self.assignment_sha256,
            source_sha256=self.source_sha256, config=self.config, debug=self.debug,
            native_data_contract_sha256=self.data_contract_sha256,
            native_data_contract_receipt=self.data_contract_receipt,
            case_ids=self.case_ids, rows=self.rows, native_scope="ROI30_CONTEXT28",
            bounded_scope_installed=False, original_sampler_used=True,
            no_training_preparation=True, query_GT_in_forward=False,
            views_per_observation=2, dense_input_shape=[5, 48, 48, 48])
        self.request_sha256 = _json_sha(self.request)
        if not self.read_only:
            self.output.mkdir(parents=True, exist_ok=True)
        request_path = self.output / "request.json"
        if self.read_only:
            if _json_sha(self._read_cache_json("request.json")) != self.request_sha256:
                raise ValueError("Read-only native30 geometry request differs; original output preserved")
        elif request_path.exists():
            if request_path.is_symlink() or json.loads(request_path.read_text(encoding="utf8")) != self.request:
                raise ValueError("Existing native30 geometry cache request differs; original output preserved")
        else:
            _write_new(request_path, self.request)
        for row in self.rows:
            receipt = self._receipt_path(row)
            if receipt.exists():
                if receipt.is_symlink():
                    raise ValueError("Regular native30 cache receipt required")
                stored = (self._read_cache_json(receipt.relative_to(self.output).as_posix())
                          if self.read_only else json.loads(receipt.read_text(encoding="utf8")))
                if (stored.get("request_sha256") != self.request_sha256 or stored.get("row") != row
                        or self.read_only and _json_sha(stored.get("row")) != _json_sha(row)):
                    raise ValueError("Cached native30 row/request identity changed: " + row["id"])
                self._canonical[row["id"]] = stored
                key = _json_sha((row["donor_case_id"], row["donor_component"]))
                if key in self._source_receipts and self._source_receipts[key] != stored["shared_source"]:
                    raise ValueError("Shared donor source reference differs across retained records")
                self._source_receipts[key] = stored["shared_source"]
        if self.read_only:
            self._admit_read_only_cache()
        self.guard()

    def __len__(self):
        return len(self.rows)

    def _values(self, extra=()):
        return (list(self._raw.values()), list(self._donors.values()),
                list(self._records.values()), list(self._shared.values()), extra)

    def memory_measurement(self, extra=()):
        return dict(rss_bytes=psutil.Process().memory_info().rss,
            total_resident_bytes=resident_bytes_of(self._values(extra)),
            resident_limit_bytes=self.resident_bytes, rss_limit_bytes=self.rss_bytes)

    def guard(self):
        if psutil.Process().memory_info().rss > self.rss_bytes:
            raise MemoryError("Native30 RSS budget exceeded; no graph/input/cohort reduction")
        if _sha(self.path) != self.index_sha256:
            raise ValueError("Signed native inventory changed after admission")
        for name, module in self.runtime.items():
            if _sha(module.__file__) != self.source_sha256[name]:
                raise ValueError("Original native30 source bytes changed: " + name)
        if _sha(self.data_contract_path) != self.data_contract_sha256:
            raise ValueError("Native CP data contract source bytes changed")
        from .native30_data_contract import source_receipt
        if source_receipt() != self.data_contract_receipt:
            raise ValueError("Native CP assignment/transport AST source or actual dependency changed")
        if self.read_only:
            for relative, (signature, digest) in self._read_only_metadata.items():
                path = self._regular_cache_path(relative)
                if self._file_signature(path) != signature:
                    raise ValueError("Read-only native30 cache metadata changed: " + relative)
            for relative in ("request.json", "index.json"):
                if _sha(self._regular_cache_path(relative)) != self._read_only_metadata[relative][1]:
                    raise ValueError("Read-only native30 cache request/index bytes changed: " + relative)
        for path, signature in self._file_signatures.items():
            stat = Path(path).stat()
            if (stat.st_size, stat.st_mtime_ns) != signature[:2] and _sha(path) != signature[2]:
                raise ValueError("Bound raw CT/annotation changed: " + path)

    _guard = guard

    @staticmethod
    def _file_signature(path):
        stat = path.stat()
        return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)

    def _regular_cache_path(self, relative):
        if not isinstance(relative, str) or not relative:
            raise ValueError("Literal relative native30 cache path required")
        parts = Path(relative).parts
        if Path(relative).is_absolute() or any(part in (".", "..") for part in parts):
            raise ValueError("Native30 cache path must stay within its preserved root")
        literal = self.output / relative
        path = literal.resolve(strict=True)
        if (not path.is_relative_to(self.output) or not path.is_file()
                or any((self.output / Path(*parts[:index])).is_symlink()
                       for index in range(1, len(parts)+1))):
            raise ValueError("Regular native30 cache file identity/path required")
        return path

    def _read_cache_json(self, relative):
        path = self._regular_cache_path(relative)
        before = self._file_signature(path)
        data = path.read_bytes()
        if self._file_signature(path) != before:
            raise ValueError("Native30 cache metadata changed during admission: " + relative)
        value = json.loads(data.decode("utf8"))
        if not isinstance(value, dict):
            raise ValueError("Native30 cache metadata must be an object: " + relative)
        self._read_only_metadata[relative] = (before, hashlib.sha256(data).hexdigest())
        return value

    def _index_summary(self):
        return dict(format=FORMAT, complete=True, request_sha256=self.request_sha256,
            actual_CT=True, no_training_preparation=True, validation_records=len(self.rows),
            records=[self._canonical[row["id"]] for row in self.rows])

    def _admit_read_only_cache(self):
        """Validate the whole published cache without deserializing any tensor."""
        expected = {self._receipt_path(row) for row in self.rows}
        ledger = self.output / "completed_records"
        if (ledger.is_symlink() or not ledger.is_dir() or set(ledger.glob("*.json")) != expected
                or set(self._canonical) != set(self._row_lookup)):
            raise ValueError("Read-only native30 reuse requires every exact completed observation; no rebuild")
        index = self._read_cache_json("index.json")
        if _json_sha(index) != _json_sha(self._index_summary()):
            raise ValueError("Read-only native30 index must publish the complete exact ordered receipts")
        payloads = {}
        for row in self.rows:
            receipt = self._canonical[row["id"]]
            source_key = _json_sha((row["donor_case_id"], row["donor_component"]))
            if (receipt.get("id") != row["id"]
                    or receipt.get("path") != "records/" + _json_sha(row["id"]) + ".pt"
                    or receipt.get("shared_source", {}).get("path") != "shared_sources/" + source_key + ".pt"
                    or type(receipt.get("measurement_epoch")) is not int
                    or receipt["measurement_epoch"] != 0
                    or receipt.get("actual_original_views") is not True):
                raise ValueError("Read-only native30 observation payload/view identity differs: " + row["id"])
            for kind in ("nodes", "edges"):
                values = receipt.get("sampled_view_" + kind)
                if (not isinstance(values, list) or len(values) != 2
                        or any(type(value) is not int or value < (1 if kind == "nodes" else 0) for value in values)
                        or type(receipt.get("sampled_two_view_" + kind)) is not int
                        or receipt["sampled_two_view_" + kind] != sum(values)):
                    raise ValueError("Read-only native30 original two-view measurements differ: " + row["id"])
            for ref in (receipt, receipt["shared_source"]):
                relative, digest = ref.get("path"), ref.get("sha256")
                if (not isinstance(digest, str) or len(digest) != 64
                        or any(character not in "0123456789abcdef" for character in digest)):
                    raise ValueError("Read-only native30 payload requires its literal SHA256")
                if relative in payloads and payloads[relative] != digest:
                    raise ValueError("Read-only native30 shared file has contradictory SHA256")
                payloads[relative] = digest
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            list(pool.map(lambda item: self._cache_file(*item), payloads.items()))
        self._read_only_payload_files = len(payloads)
        self._read_only_index = index

    def _make_room(self, active_cases=(), active_records=()):
        active_cases = set(active_cases)
        active_record_ids = {id(record) for record in active_records}
        def exceeds():
            return resident_bytes_of(self._values(active_records)) > self.resident_bytes
        for cache, protected in ((self._records, lambda key, value: id(value) in active_record_ids),
                (self._shared, lambda key, value: key in self._active_shared_paths),
                (self._donors, lambda key, value: key[0] in active_cases),
                (self._raw, lambda key, value: key in active_cases)):
            for key in list(cache):
                if not exceeds():
                    break
                if not protected(key, cache[key]):
                    del cache[key]
        if exceeds():
            raise MemoryError("Complete active native30 raw/donor/geometry batch exceeds resident budget; no rows skipped")
        self.guard()

    def _load_case(self, case_id):
        info = self.raw_metadata[case_id]
        common = self.runtime["common"]
        signatures = {}
        for kind in ("image", "label"):
            path = Path(info[kind]).resolve(strict=True)
            if not path.is_file() or _sha(path) != info[kind + "_sha256"]:
                raise ValueError("Actual signed raw file identity differs: " + case_id + ":" + kind)
            stat = path.stat()
            signatures[str(path)] = (stat.st_size, stat.st_mtime_ns, info[kind + "_sha256"])
        case = common.load_case(common.CasePaths(case_id, Path(info["image"]), Path(info["label"])))
        if (case.image.ndim != 3 or case.image.shape != case.label.shape or not np.isfinite(case.image).all()
                or not np.isin(case.label, [0, 1, 2]).all()
                or not np.allclose(case.spacing, info["spacing"], rtol=1e-6)
                or not np.allclose(case.image_affine, case.label_affine)
                or not np.allclose(np.linalg.norm(case.image_affine[:3, :3], axis=0), case.spacing, rtol=1e-5)):
            raise ValueError("Actual aligned native CT/label/spacing/affine contract differs: " + case_id)
        organ = np.isin(case.label, [1, 2])
        if not organ.any():
            raise ValueError("Actual recipient/donor organ is empty: " + case_id)
        depth = common.organ_depth_mm(organ, case.spacing)
        for array in (case.image, case.label, case.spacing, case.image_affine, case.label_affine, organ, depth):
            array.flags.writeable = False
        return case_id, dict(case=case, organ=organ, depth=depth, signatures=signatures)

    def _preload(self, cases):
        cases = set(cases)
        if not cases <= self.allowed_raw_cases:
            raise ValueError("Only evaluation recipients and their assigned donors may be prepared")
        self._make_room(cases)
        missing = sorted(cases - set(self._raw))
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            for name, value in pool.map(self._load_case, missing):
                self._raw[name] = value
                self._file_signatures.update(value["signatures"])
        self._make_room(cases)

    def case(self, case_id):
        self._preload(self._active_cases | {case_id})
        self._raw.move_to_end(case_id)
        return self._raw[case_id]["case"]

    def organ_depth(self, case_id):
        self.case(case_id)
        value = self._raw[case_id]
        return value["organ"], value["depth"]

    def _native_row(self, row):
        if not isinstance(row, Mapping) or row.get("id") not in self._row_lookup:
            raise ValueError("Exact retained native observation required")
        actual = self._row_lookup[row["id"]]
        if any(row.get(key) != actual[key] for key in QUERY_FIELDS):
            raise ValueError("Native observation/donor/geometry query changed")
        return actual

    def _donor(self, row):
        key = row["donor_case_id"], row["donor_component"]
        if key not in self._donors:
            from .native30_data_contract import sources
            value = self._raw[key[0]]
            collection = sources(value["case"], self.config["cache"]["source_pad"],
                                 self.meta["config"]["donor_max_diameter_mm"])
            matches = [i for i, (component, _) in enumerate(collection.entries) if component == key[1]]
            if len(matches) != 1:
                raise ValueError("Exact signed donor component missing: " + repr(key))
            source, _ = collection[matches[0]]
            if (int(source.full_mask.sum()) != source.voxel_count
                    or int(source.patch_mask.sum()) != source.voxel_count
                    or not np.array_equal(source.patch_mask, source.full_mask[source.patch_slices])
                    or not np.all(value["case"].label[source.full_mask] == 2)
                    or not np.array_equal(source.patch_image, value["case"].image[source.patch_slices])):
                raise ValueError("Full actual donor CT/component was cropped or changed")
            origin = np.asarray([sl.start for sl in source.patch_slices])
            if not np.array_equal(source.anchor_center, origin + np.asarray(source.patch_mask.shape)//2):
                raise ValueError("Original native CP donor anchor differs from shape//2")
            if self.read_only:
                # Upper inputs need the actual donor component, while its local
                # canonical graph/dense patch already belong to the bound cache.
                self._donors[key] = (source,)
                self._donors.move_to_end(key)
                return self._donors[key]
            prepared = self.runtime["local"].prepare_local_source(value["case"], source,
                full_organ_mask=value["organ"], organ_depth=value["depth"], config=self.graph_config,
                rng=np.random.default_rng(self.config["seed"]), ct_clip=self.clip)
            if int(prepared.source_footprint.sum()) != source.voxel_count:
                raise ValueError("Original donor canonical footprint lost occupied voxels")
            patch = torch.from_numpy(prepared.source_patch.astype(np.float16))
            self._donors[key] = (source, prepared, patch)
        self._donors.move_to_end(key)
        return self._donors[key]

    def donor_source(self, row):
        actual = self._native_row(row)
        self._preload({actual["case_id"], actual["donor_case_id"]})
        source = self._donor(actual)[0]
        from .native30_data_contract import donor_in_target_spacing
        target = self._raw[actual["case_id"]]["case"]
        donor = self._raw[actual["donor_case_id"]]["case"]
        transported, mask = donor_in_target_spacing(source, donor.spacing, target.spacing)
        if not np.array_equal(mask, transported.patch_mask) or not mask.any():
            raise ValueError("Complete native CP transported donor footprint differs")
        return source, transported

    def _validate_observation(self, row):
        case = self._raw[row["case_id"]]["case"]
        center = tuple(row["center"])
        if any(point >= size for point, size in zip(center, case.image.shape)):
            raise ValueError("Original observation center lies outside actual CT")
        if row["target"] == 0 and case.label[center] != 1:
            raise ValueError("Recorded unobserved center must remain annotated liver")
        if row["target"] == 1 and sum(p["component"] == row["component"] and p["center"] == row["center"]
                for p in self.raw_metadata[row["case_id"]]["positives"]) != 1:
            raise ValueError("Recorded P anchor/component differs from actual raw observation")

    def _build(self, row):
        if self.read_only:
            raise RuntimeError("Read-only native30 geometry cannot construct a replacement record")
        self._validate_observation(row)
        original, prepared, source_patch = self._donor(row)
        from .native30_data_contract import donor_in_target_spacing
        target_value = self._raw[row["case_id"]]
        target = target_value["case"]
        donor = self._raw[row["donor_case_id"]]["case"]
        transported, mask = donor_in_target_spacing(original, donor.spacing, target.spacing)
        footprint = self.runtime["spatial"].exact_source_footprint(transported)
        adapted = replace(prepared, source_footprint=footprint)
        spec = self.runtime["curriculum"].CandidateSpec(tuple(row["center"]), 0, 0, -1, -1, 1., 0., 0., 0., 0.)
        transformed, transform = self.runtime["local"]._transform_footprint(footprint, spec)
        if (not np.array_equal(transform, np.eye(3, dtype=np.float32))
                or not np.array_equal(transformed, mask)):
            raise ValueError("Original graph footprint differs from complete native CP identity placement")
        built = self.runtime["local"].build_local_graph(target, transported, spec,
            full_organ_mask=target_value["organ"], organ_depth=target_value["depth"],
            config=self.graph_config, rng=np.random.default_rng(self.config["seed"]),
            ct_clip=self.clip, prepared_source=adapted)
        if (not np.array_equal(built.target_local["transform"].numpy(), transform)
                or built.source_patch.shape != (5, 48, 48, 48)
                or built.target_patch.shape != (5, 48, 48, 48)):
            raise ValueError("Original target transform/fixed48 five-channel input changed")
        provenance = dict(observation_id=row["id"], input_inventory_sha256=self.index_sha256,
            assignment_sha256=self.assignment_sha256, donor_group=row["donor_group"],
            recipient_group=row["patient_group"])
        for branch, name in (("donor", row["donor_case_id"]), ("recipient", row["case_id"])):
            for kind in ("image", "label"):
                provenance[branch + "_" + kind + "_sha256"] = self.raw_metadata[name][kind + "_sha256"]
        return dict(format=FORMAT, input_provenance=provenance,
            case_id=row["case_id"], donor_case_id=row["donor_case_id"], component_id=row["donor_component"],
            center=list(row["center"]), graph_config=self.graph_config.to_dict(), seed=self.config["seed"],
            source_local={**built.source_local, "footprint_voxels": original.voxel_count},
            target_local=built.target_local, source_patch=source_patch,
            target_patch=torch.from_numpy(built.target_patch.astype(np.float16)),
            audit=dict(physical_transport=physical_transport(donor, target, original, row["center"]),
                donor_voxels=original.voxel_count, transported_recipient_voxels=int(mask.sum()),
                source_branch="actual donor native CT/mask/spacing",
                target_branch="actual recipient CT; full transported donor footprint erased",
                observed_GT_used=False, original_build_local_graph_used=True,
                original_graph_config_preserved=True, bounded_scope_installed=False))

    def _sample(self, record, index, epoch):
        views = []
        for view in (0, 1):
            salt = f"{record['input_provenance']['observation_id']}:{record['donor_case_id']}:{record['component_id']}:{view}:{epoch}"
            seed = self.runtime["common"].stable_case_seed(record["seed"], record["case_id"], salt)
            graph = self.runtime["sample"].build_local_view(record["source_local"], record["target_local"],
                                                           self.graph_config, seed=seed)
            graph.native30_record_id = record["input_provenance"]["observation_id"]
            graph.native30_view = torch.tensor([view], dtype=torch.long)
            graph.native30_epoch = torch.tensor([epoch], dtype=torch.long)
            views.append(graph)
        return ((tuple(views), record["source_patch"], record["target_patch"]), index)

    def _receipt_path(self, row):
        return self.output / "completed_records" / (_json_sha(row["id"]) + ".json")

    def _cache_file(self, relative, expected_sha):
        path = self._regular_cache_path(relative)
        before = self._file_signature(path)
        if _sha(path) != expected_sha or self._file_signature(path) != before:
            raise ValueError("Native30 geometry cache file identity/path changed")
        return path

    def _prewarm_shared(self, rows):
        for row in rows:
            if row["id"] not in self._canonical:
                continue
            ref = self._canonical[row["id"]]["shared_source"]
            source_path = self._cache_file(ref["path"], ref["sha256"])
            if ref["path"] not in self._shared:
                self._shared[ref["path"]] = torch.load(source_path, map_location="cpu", weights_only=False, mmap=True)

    def _load_record(self, row):
        receipt = self._canonical[row["id"]]
        path = self._cache_file(receipt["path"], receipt["sha256"])
        record = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        record.update(self._shared[receipt["shared_source"]["path"]])
        if (record.get("format") != FORMAT or record.get("case_id") != row["case_id"]
                or record.get("donor_case_id") != row["donor_case_id"] or record.get("center") != row["center"]
                or record.get("component_id") != row["donor_component"]
                or record["input_provenance"]["observation_id"] != row["id"]
                or record.get("graph_config") != self.graph_config.to_dict()):
            raise ValueError("Native30 cached record/source and literal observation differ")
        return record

    def _persist(self, row, record):
        if self.read_only:
            raise RuntimeError("Read-only native30 geometry cannot persist any record or shared source")
        source_key = _json_sha((row["donor_case_id"], row["donor_component"]))
        source_relative = "shared_sources/" + source_key + ".pt"
        source_path = self.output / source_relative
        shared = {key: record[key] for key in ("source_local", "source_patch")}
        with self._source_lock:
            if source_key not in self._source_receipts:
                if source_path.exists():
                    raise ValueError("Unreceipted shared source exists; preserve partial bytes and use a new output")
                _save_new(source_path, shared)
                self._source_receipts[source_key] = dict(path=source_relative, sha256=_sha(source_path))
            ref = self._source_receipts[source_key]
        relative = "records/" + _json_sha(row["id"]) + ".pt"
        path = self.output / relative
        _save_new(path, {key: value for key, value in record.items() if key not in shared})
        (views, _, _), _ = self._sample(record, self._indices[row["id"]], 0)
        nodes = [sum(int(store.num_nodes) for store in graph.node_stores) for graph in views]
        edges = [sum(int(store.edge_index.shape[1]) for store in graph.edge_stores) for graph in views]
        receipt = dict(request_sha256=self.request_sha256, row=row, id=row["id"], path=relative,
            sha256=_sha(path), shared_source=ref, sampled_view_nodes=nodes, sampled_view_edges=edges,
            sampled_two_view_nodes=sum(nodes), sampled_two_view_edges=sum(edges),
            measurement_epoch=0, actual_original_views=True)
        _write_new(self._receipt_path(row), receipt)
        return receipt

    def _records_for(self, rows):
        if self.read_only and any(row["id"] not in self._canonical for row in rows):
            raise ValueError("Read-only native30 canonical coverage changed; no replacement built")
        self._active_cases = {row[key] for row in rows for key in ("case_id", "donor_case_id")}
        self._active_shared_paths = {self._canonical[row["id"]]["shared_source"]["path"]
                                    for row in rows if row["id"] in self._canonical}
        if not self.read_only:
            self._preload(self._active_cases)
        missing = [row for row in rows if row["id"] not in self._records]
        build_rows = [row for row in missing if row["id"] not in self._canonical]
        for row in build_rows:
            self._donor(row)
        self._prewarm_shared(missing)
        self._make_room(self._active_cases)
        def one(row):
            if row["id"] in self._canonical:
                return self._load_record(row)
            try:
                return self._build(row)
            except (ValueError, RuntimeError, MemoryError, OSError) as error:
                raise RuntimeError(f"Native30 geometry failed for {row['id']} donor={row['donor_case_id']} "
                    f"component={row['donor_component']} center={row['center']}; no row skipped: {error}") from error
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            records = list(pool.map(one, missing))
        self._make_room(self._active_cases, records)
        to_persist = [(row, record) for row, record in zip(missing, records) if row["id"] not in self._canonical]
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            receipts = list(pool.map(lambda item: self._persist(*item), to_persist))
        for (row, _), receipt in zip(to_persist, receipts):
            self._canonical[row["id"]] = receipt
        self.builds += len(to_persist)
        self.loads += len(missing)-len(to_persist)
        for row, record in zip(missing, records):
            self._records[row["id"]] = record
        result = [self._records[row["id"]] for row in rows]
        self.hits += len(rows)-len(missing)
        for row in rows:
            self._records.move_to_end(row["id"])
        self._make_room(self._active_cases, result)
        return result

    def get(self, ids, epoch=0):
        if (not isinstance(ids, (tuple, list)) or not ids or len(set(ids)) != len(ids)
                or any(type(index) is not int or not 0 <= index < len(self.rows) for index in ids)
                or type(epoch) is not int or epoch < 0):
            raise ValueError("Explicit unique retained observation indices and nonnegative epoch required")
        start = time.perf_counter()
        records = self._records_for([self.rows[index] for index in ids])
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            items = list(pool.map(lambda item: self._sample(*item), zip(records, ids, [epoch]*len(ids))))
        batch = collate_native30(items)
        self._make_room(self._active_cases, (records, batch, items))
        measurement = self.memory_measurement((records, batch, items))
        self.timings.append(dict(stage="native30_geometry_and_two_views", records=len(ids),
            seconds=time.perf_counter()-start, **measurement))
        return batch

    def prepare(self):
        """Prepare every selected P/128U record; training cases are donors only."""
        start = time.perf_counter()
        if self.read_only:
            self.guard()
            if (set(self._canonical) != set(self._row_lookup)
                    or self._read_only_index != self._index_summary()):
                raise ValueError("Read-only native30 complete publication changed; no rebuild")
            return self._preparation_receipt(start)
        from tqdm import tqdm
        for name, rows in tqdm(self.by_case.items(), total=len(self.by_case), desc="native30 validation geometry", unit="case"):
            for offset in range(0, len(rows), self.workers):
                records = self._records_for(rows[offset:offset+self.workers])
                del records
        if set(self._canonical) != set(self._row_lookup):
            raise ValueError("Whole native30 evaluation geometry incomplete")
        summary = self._index_summary()
        path = self.output / "index.json"
        if path.exists():
            if path.is_symlink() or json.loads(path.read_text(encoding="utf8")) != summary:
                raise ValueError("Complete native30 geometry cache index changed")
        else:
            _write_new(path, summary)
        self.guard()
        return self._preparation_receipt(start)

    def _preparation_receipt(self, start):
        path = self.output / "index.json"
        measured = {}
        for kind in ("nodes", "edges"):
            values = [self._canonical[row["id"]]["sampled_two_view_" + kind] for row in self.rows]
            measured[kind] = dict(min=min(values), max=max(values), mean=sum(values)/len(values),
                                 sum=sum(values), observations=len(values), actual_local_graphs=2*len(values))
        result = dict(source=str(path), sha256=_sha(path), validation_records=len(self.rows),
            cases=len(self.case_ids), preparation_seconds=time.perf_counter()-start,
            actual_CT=True, no_training_preparation=True, original_two_views=True,
            geometry_builds=self.builds, geometry_loads=self.loads, cache_hits=self.hits,
            raw_cases_decoded=sorted(self._raw), raw_files_verified=sorted(self._file_signatures),
            sampled_two_view_summary=measured, resources=self.memory_measurement())
        if self.read_only:
            metadata_sha256 = {relative: binding[1] for relative, binding in self._read_only_metadata.items()}
            result.update(preparation_reused=True, read_only=True, reuse_provenance=dict(
                cache_root=str(self.output), cache_index_sha256=metadata_sha256["index.json"],
                cache_request_file_sha256=metadata_sha256["request.json"],
                request_sha256=self.request_sha256, input_inventory_sha256=self.index_sha256,
                cohort_sha256=self.population["cohort_sha256"], assignment_sha256=self.assignment_sha256,
                metadata_manifest_sha256=_json_sha(metadata_sha256),
                validated_payload_files=self._read_only_payload_files,
                validated_observations=len(self.rows), validated_cases=len(self.case_ids),
                complete_original_publication=True, all_payload_sha256_verified=True,
                graph_reconstruction=False, raw_CT_redecoded_for_preparation=False,
                cache_tensor_deserialization_for_preparation=False, existing_files_written=False))
        return result
