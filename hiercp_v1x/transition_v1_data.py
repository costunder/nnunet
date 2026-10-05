"""Signed native P/U observations -> complete original-input D records.

Native Dataset/CropStore retain the exact donor assignments and observed GT.
Only the D geometry/input representation is newly prepared. All14102 native
observations, including every observed P and all128U per recipient, are required
in production. A new canonical index is published only after complete coverage.
Training uses that lossless shared-source cache; epochs sample original two
views without rebuilding static radius graphs. DEBUG fixtures are explicit.
"""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import copy
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import importlib
import shutil
import time
import uuid

import numpy as np
import psutil
import torch

from . import transition_v1_local as local


FORMAT = "native_PU_actual_original_v1_full_canonical_cache_v2"
MEASUREMENT_FORMAT = "actual_original_two_view_epoch0_counts_v1"
ASSIGNMENT_KEYS = ("id", "case_id", "patient_group", "component", "center", "target",
                   "donor_case_id", "donor_component", "donor_group")


def _native_helper_namespace():
    """Resolve current native helper names without shadowing archived hiercp.

    The activated archive also owns an older tools package. Native Dataset's
    imports need current tools.v22_* modules absent there. Extend only that
    package search path; no original source file/module is replaced.
    """
    package = importlib.import_module("tools")
    path = str(local.ROOT / "tools")
    if path not in package.__path__:
        package.__path__.append(path)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def _checked_sha(path):
    path = Path(path).resolve(strict=True)
    stat = path.stat()
    return local._file_sha(str(path), stat.st_size, stat.st_mtime_ns)


def _publish_new_json(path, value):
    """Atomic complete publication with no replacement of an existing result."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial-" + uuid.uuid4().hex)
    with temporary.open("x", encoding="utf8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    # Hard-link creation is atomic and refuses an existing destination on both
    # Windows and POSIX. Only our exact newly created temporary file is removed.
    os.link(temporary, path)
    temporary.unlink()


def _storage_root(root, row):
    root = Path(root).resolve(strict=True)
    selected = (root / row.get("segment", ".")).resolve(strict=True)
    if not selected.is_relative_to(root) or selected.is_symlink():
        raise ValueError("D canonical segment escapes its owned preparation root")
    return selected


def _verify_stored_files(root, row):
    selected = _storage_root(root, row)
    path = (selected / row["path"]).resolve(strict=True)
    if not path.is_relative_to(selected) or path.is_symlink() or _checked_sha(path) != row["sha256"]:
        raise ValueError("D canonical file identity/path changed")
    reference = row["shared_source"]
    source = (selected / reference["path"]).resolve(strict=True)
    if (not source.is_relative_to(selected) or source.is_symlink()
            or _checked_sha(source) != reference["sha256"]):
        raise ValueError("D shared original donor canonical source identity/path changed")
    return selected


def _sampled_measurement(payload):
    """Count the actual admitted two original views, never a CNN proxy bound."""
    views, _, _ = payload
    if not isinstance(views, (tuple, list)) or len(views) != 2:
        raise ValueError("Both actual original sampled views required for preparation measurement")
    nodes, edges = [], []
    for graph in views:
        count = sum(int(store.num_nodes) for store in graph.node_stores)
        edge_count = sum(int(store.edge_index.shape[1]) for store in graph.edge_stores)
        if count <= 0 or edge_count < 0:
            raise ValueError("Actual nonempty sampled original graph required for measurement")
        nodes.append(count)
        edges.append(edge_count)
    return dict(sampled_measurement_format=MEASUREMENT_FORMAT, measurement_epoch=0,
                sampled_view_nodes=nodes, sampled_view_edges=edges,
                sampled_two_view_nodes=sum(nodes), sampled_two_view_edges=sum(edges))


def _validate_measurement(row):
    if (row.get("sampled_measurement_format") != MEASUREMENT_FORMAT
            or type(row.get("measurement_epoch")) is not int or row["measurement_epoch"] != 0):
        raise ValueError("Canonical D row lacks actual epoch0 two-view measurements")
    for kind in ("nodes", "edges"):
        values = row.get("sampled_view_" + kind)
        total = row.get("sampled_two_view_" + kind)
        if (not isinstance(values, list) or len(values) != 2
                or any(type(value) is not int or value < (1 if kind == "nodes" else 0) for value in values)
                or type(total) is not int or total != sum(values)):
            raise ValueError("Canonical D sampled two-view counts changed or are missing")


def _sampled_summary(rows):
    result = {}
    for kind in ("nodes", "edges"):
        values = [row["sampled_two_view_" + kind] for row in rows]
        result[kind] = dict(min=min(values), max=max(values), mean=sum(values) / len(values),
                            sum=sum(values), observations=len(values), actual_local_graphs=2 * len(values))
    return result


def assignment_digest(rows):
    return hashlib.sha256(local._json([{key: row[key] for key in ASSIGNMENT_KEYS} for row in rows])).hexdigest()


def validate_population(meta, *, debug):
    """Metadata UNIT-testable full-coverage checks; no label enters the model."""
    from l0_regions.donor_learning import validate_rows
    if not isinstance(meta, Mapping) or type(debug) is not bool or meta.get("debug") is not debug:
        raise ValueError("Explicit matching native DEBUG/production inventory required")
    rows = meta["records"]
    validate_rows(rows)
    raw = {row["case_id"]: row for row in meta["raw_records"]}
    donors = {(row["case_id"], row["component_id"]) for row in meta["donor_pool"]}
    train, validation = set(meta["split"]["inner_train"]), set(meta["split"]["inner_val"])
    outer = set(meta["split"]["outer_train"])
    if (train & validation or train | validation != outer
            or any(row["case_id"] not in outer or row["donor_case_id"] not in train
                   or (row["donor_case_id"], row["donor_component"]) not in donors for row in rows)):
        raise ValueError("Complete native cohort and actual independent inner-train donor pool required")
    if not debug:
        if len(train) != 84 or len(validation) != 21 or len(rows) != 14102:
            raise ValueError("Production D must retain original84/21 cases and all14102 native assignments")
        if {row["case_id"] for row in rows} != outer:
            raise ValueError("Native recipient cohort lost an original case")
        by_case = {}
        for row in rows:
            by_case.setdefault(row["case_id"], []).append(row)
        for case, actual in by_case.items():
            if (sum(row["target"] == 0 for row in actual) != 128
                    or sum(row["target"] == 1 for row in actual) != len(raw[case]["positives"])):
                raise ValueError("Native observed P/all128U coverage changed: " + case)
    return dict(debug=debug, total_assignments=len(rows), assignment_sha256=assignment_digest(rows),
                observed_P=sum(row["target"] == 1 for row in rows),
                unobserved_U=sum(row["target"] == 0 for row in rows),
                full14102=len(rows) == 14102, hidden_subset=False, donor_redraw=False)


class NativeObservationDataset:
    """Exactly the tracked native Dataset rows, with an explicit10mm D base."""
    def __init__(self, index, partition, debug, case_ids=None):
        runtime = local._runtime()
        _native_helper_namespace()
        from l0_local_cnn.data import Dataset
        native = Dataset(index, partition, debug)
        self.native = native
        self.path, self.root, self.meta, self.rows = native.path, native.root, native.meta, native.rows
        self.population = validate_population(self.meta, debug=debug)
        if partition not in ("inner_train", "inner_val", "outer_train") or not self.rows:
            raise ValueError("Actual nonempty native observation train/validation partition required")
        self.partition = partition
        self.debug = debug
        self.case_ids = None
        if case_ids is not None:
            if (not debug or not isinstance(case_ids, (list, tuple)) or not case_ids
                    or any(not isinstance(case, str) or not case for case in case_ids)
                    or len(set(case_ids)) != len(case_ids)
                    or not set(case_ids) <= set(self.meta["split"][partition])):
                raise ValueError("Case selection is allowed only for explicitly labeled DEBUG within the native partition")
            selected = set(case_ids)
            self.rows = [row for row in self.rows if row["case_id"] in selected]
            self.case_ids = tuple(case_ids)
            raw = {row["case_id"]: row for row in self.meta["raw_records"]}
            for case in case_ids:
                actual = [row for row in self.rows if row["case_id"] == case]
                if (sum(row["target"] == 0 for row in actual) != 128
                        or sum(row["target"] == 1 for row in actual) != len(raw[case]["positives"])):
                    raise ValueError("DEBUG case selection must retain every observed P and all128U: " + case)
        self.index_sha256 = _sha(self.path)
        self.assignment_sha256 = self.population["assignment_sha256"]
        self.base = copy.deepcopy(self.meta["base"])
        from .bounded_scope import configure
        self.base["graph"] = configure(self.base["graph"], 10.).to_dict()
        local._config(self.base)
        self.scope_contract = runtime["scope"]["contract_sha256"]
        self.preparation_revision = FORMAT

    def __len__(self):
        return len(self.rows)

    def receipt(self):
        return {**self.population, "input_inventory": str(self.path), "input_inventory_sha256": self.index_sha256,
                "partition": self.partition, "actual_partition_observations": len(self),
                "explicit_DEBUG_case_ids": None if self.case_ids is None else list(self.case_ids),
                "all_selected_P_and128U": self.case_ids is not None or not self.debug,
                "scope_contract": self.scope_contract, "local_contract": local.local_contract()}


class _IntervalNode:
    """One covered address interval; a deterministic treap avoids global scans."""
    __slots__ = ("start", "end", "count", "priority", "left", "right", "covered")

    def __init__(self, start, end, count):
        self.start, self.end, self.count = start, end, count
        # Address mixing never consumes the experiment's Python/NumPy/Torch RNG.
        mixed = (start ^ (end << 1)) & ((1 << 64) - 1)
        mixed = ((mixed ^ (mixed >> 30)) * 0xbf58476d1ce4e5b9) & ((1 << 64) - 1)
        mixed = ((mixed ^ (mixed >> 27)) * 0x94d049bb133111eb) & ((1 << 64) - 1)
        self.priority = mixed ^ (mixed >> 31)
        self.left = self.right = None
        self.covered = end - start


def _interval_update(node):
    if node is not None:
        node.covered = (node.end - node.start
                        + (0 if node.left is None else node.left.covered)
                        + (0 if node.right is None else node.right.covered))
    return node


def _interval_merge(left, right):
    if left is None:
        return right
    if right is None:
        return left
    if left.priority <= right.priority:
        left.right = _interval_merge(left.right, right)
        return _interval_update(left)
    right.left = _interval_merge(left, right.left)
    return _interval_update(right)


def _interval_split(node, position):
    """Split by interval start, after splitting any crossing interval itself."""
    if node is None:
        return None, None
    if node.start < position:
        node.right, right = _interval_split(node.right, position)
        return _interval_update(node), right
    left, node.left = _interval_split(node.left, position)
    return left, _interval_update(node)


def _interval_cut(root, position):
    node, previous = root, None
    while node is not None:
        if node.start < position:
            previous, node = node, node.right
        else:
            node = node.left
    if previous is None or previous.end <= position:
        return root
    before, remaining = _interval_split(root, previous.start)
    _, after = _interval_split(remaining, previous.start + 1)
    first = _IntervalNode(previous.start, position, previous.count)
    second = _IntervalNode(position, previous.end, previous.count)
    return _interval_merge(_interval_merge(before, first), _interval_merge(second, after))


def _interval_rows(root):
    stack, node = [], root
    while stack or node is not None:
        while node is not None:
            stack.append(node)
            node = node.left
        node = stack.pop()
        yield node.start, node.end, node.count
        node = node.right


class _StorageIntervalUnion:
    """Incremental union of live storage ranges, including partial aliases.

    Exact shared storages use a reference-count dictionary. A new partially
    overlapping range touches only its overlapping treap intervals, never all
    previously cached records or all physical address ranges.
    """
    def __init__(self):
        self._roots, self._references = {}, {}
        self.total_bytes = 0

    def _change(self, storage, delta):
        domain, start, end = storage
        root = self._roots.get(domain)
        previous_bytes = 0 if root is None else root.covered
        root = _interval_cut(_interval_cut(root, start), end)
        before, remaining = _interval_split(root, start)
        middle, after = _interval_split(remaining, end)
        replacement, cursor = None, start
        for first, last, count in _interval_rows(middle):
            if cursor < first:
                if delta < 0:
                    raise RuntimeError("Resident storage reference accounting underflow")
                replacement = _interval_merge(replacement, _IntervalNode(cursor, first, 1))
            count += delta
            if count < 0:
                raise RuntimeError("Resident storage reference accounting underflow")
            if count:
                replacement = _interval_merge(replacement, _IntervalNode(first, last, count))
            cursor = last
        if cursor < end:
            if delta < 0:
                raise RuntimeError("Resident storage reference accounting underflow")
            replacement = _interval_merge(replacement, _IntervalNode(cursor, end, 1))
        root = _interval_merge(_interval_merge(before, replacement), after)
        if root is None:
            self._roots.pop(domain, None)
        else:
            self._roots[domain] = root
        self.total_bytes += (0 if root is None else root.covered) - previous_bytes

    def add(self, storage):
        count = self._references.get(storage, 0)
        if not count:
            self._change(storage, 1)
        self._references[storage] = count + 1

    def remove(self, storage):
        count = self._references.get(storage, 0)
        if not count:
            raise RuntimeError("Resident storage owner was removed more than once")
        if count == 1:
            self._change(storage, -1)
            del self._references[storage]
        else:
            self._references[storage] = count - 1


def _storage_footprint(value):
    """Inspect one immutable owner once, retaining whole backing allocations."""
    ranges, visited = set(), set()

    def visit(item):
        identity = id(item)
        if identity in visited:
            return
        visited.add(identity)
        if torch.is_tensor(item):
            storage = item.untyped_storage()
            start, size = storage.data_ptr(), storage.nbytes()
            if size:
                ranges.add((str(item.device), start, start + size))
        elif isinstance(item, np.ndarray):
            root = item
            while isinstance(root.base, np.ndarray):
                root = root.base
            if torch.is_tensor(root.base):
                visit(root.base)  # tensor.numpy() still owns the complete tensor storage.
            elif root.size:
                bounds = (np.byte_bounds(root) if hasattr(np, "byte_bounds")
                          else np.lib.array_utils.byte_bounds(root))
                ranges.add(("cpu", int(bounds[0]), int(bounds[1])))
        elif isinstance(item, Mapping):
            for child in item.values():
                visit(child)
        elif isinstance(item, (tuple, list)):
            for child in item:
                visit(child)
        elif isinstance(item, local.PreparedDonor):
            original = item.original
            for child in (original.source_footprint, original.source_patch, original.canonical_nodes,
                          original.canonical_edges, item.source_patch):
                visit(child)
        elif hasattr(item, "full_mask"):
            for child in (item.full_mask, item.patch_mask, item.patch_image):
                visit(child)

    visit(value)
    return tuple(ranges)


def _bytes(value, seen=None):
    """One-owner storage estimate with NumPy/tensor alias deduplication."""
    union = _StorageIntervalUnion()
    if seen is not None:
        for storage in seen:
            union.add(storage)
    before = union.total_bytes
    ranges = _storage_footprint(value)
    for storage in ranges:
        union.add(storage)
    if seen is not None:
        seen.update(ranges)
    return union.total_bytes - before


class _ResidentStorageLedger:
    """Owner reference counts for raw, record, donor and their exact unions."""
    def __init__(self):
        self._groups = {name: _StorageIntervalUnion()
                        for name in ("raw", "record", "donor", "canonical", "all")}
        self._members, self._objects = {}, {}
        self._temporary_ordinal = 0

    def put(self, token, value, category):
        if token in self._members:
            self.remove(token)
        identity = id(value)
        known = self._objects.get(identity)
        if known is None:
            ranges, owners = _storage_footprint(value), 0
        else:
            _, ranges, owners = known
        # Retain the object while registered: a transient donor tuple's id must
        # never be recycled into another owner's cached footprint.
        self._objects[identity] = value, ranges, owners + 1
        groups = (category, "all") if category == "raw" else (category, "canonical", "all")
        for group in groups:
            for storage in ranges:
                self._groups[group].add(storage)
        self._members[token] = identity, ranges, groups

    def remove(self, token):
        identity, ranges, groups = self._members.pop(token)
        for group in groups:
            for storage in ranges:
                self._groups[group].remove(storage)
        value, ranges, owners = self._objects[identity]
        if owners == 1:
            del self._objects[identity]
        else:
            self._objects[identity] = value, ranges, owners - 1

    def bytes(self, group):
        return self._groups[group].total_bytes

    @contextmanager
    def active(self, records):
        # This ledger is mutated only by the serial provider/cache owner.
        tokens = []
        try:
            for record in records:
                self._temporary_ordinal += 1
                token = "active", self._temporary_ordinal
                self.put(token, record, "record")
                tokens.append(token)
            yield
        finally:
            for token in reversed(tokens):
                self.remove(token)


class _AccountedCache(OrderedDict):
    """Incremental storage registration at cache mutation, never at reporting."""
    def __init__(self, ledger, category, extract, values=()):
        self.ledger, self.category, self.extract = ledger, category, extract
        super().__init__()
        self.update(values)

    def __setitem__(self, key, value):
        self.ledger.put((self.category, key), self.extract(value), self.category)
        super().__setitem__(key, value)

    def __delitem__(self, key):
        super().__delitem__(key)
        self.ledger.remove((self.category, key))

    def pop(self, key, *default):
        if key not in self:
            if default:
                return default[0]
            raise KeyError(key)
        value = self[key]
        del self[key]
        return value

    def popitem(self, last=True):
        if not self:
            raise KeyError("dictionary is empty")
        key = next(reversed(self)) if last else next(iter(self))
        return key, self.pop(key)

    def clear(self):
        while self:
            self.popitem()


class OriginalInputProvider:
    """Measured worker/resource settings, whole-record caching, no data caps."""
    def __init__(self, dataset: NativeObservationDataset, *, workers, resident_bytes, rss_bytes, cache_index=None):
        if (not isinstance(dataset, NativeObservationDataset) or type(workers) is not int or workers < 2
                or type(resident_bytes) is not int or type(rss_bytes) is not int
                or not 0 < resident_bytes < rss_bytes):
            raise ValueError("Native D dataset, parallel readers>=2 and explicit resident/RSS budgets required")
        self.ds = dataset
        self.workers, self.resident_bytes, self.rss_bytes = workers, resident_bytes, rss_bytes
        self._resident = _ResidentStorageLedger()
        self._records = _AccountedCache(self._resident, "record", lambda value: value[0])
        self._donors = _AccountedCache(self._resident, "donor", lambda value: value[:2])
        self._cached_bytes = 0
        self._raw = None
        self._crop_store = None
        self.cache_index = None
        self._canonical = None
        self.last_admission = None
        if cache_index is not None:
            self.bind_cache(cache_index)

    def _guard(self):
        if psutil.Process().memory_info().rss > self.rss_bytes:
            raise MemoryError("D provider RSS budget exceeded; no graph/input/data reduction")
        if _checked_sha(self.ds.path) != self.ds.index_sha256:
            raise ValueError("Signed native input inventory changed after D admission")

    def _memory_measurement(self, active_records=()):
        self._account_raw_cache()
        with self._resident.active(active_records):
            self._cached_bytes = self._resident.bytes("canonical")
            return dict(rss_bytes=psutil.Process().memory_info().rss,
                        raw_resident_bytes=self._resident.bytes("raw"),
                        canonical_record_resident_bytes=self._resident.bytes("record"),
                        prepared_donor_resident_bytes=self._resident.bytes("donor"),
                        canonical_resident_bytes=self._cached_bytes,
                        total_resident_bytes=self._resident.bytes("all"),
                        resident_limit_bytes=self.resident_bytes, rss_limit_bytes=self.rss_bytes)

    def _account_raw_cache(self):
        if self._raw is not None and not isinstance(self._raw.cache, _AccountedCache):
            self._raw.cache = _AccountedCache(self._resident, "raw", lambda value: value,
                                             self._raw.cache.items())

    def _initialize_raw(self):
        if self._raw is not None:
            return
        from l0_exploration.data import RawStore
        from l0_local_cnn.data import CropStore
        class OriginalRawStore(RawStore):
            def load(store, case):
                value = super().load(case)  # Actual native file/hash/spacing/label checks.
                import nibabel as nib
                info = store.raw[case]
                image, label = nib.load(info["image"]), nib.load(info["label"])
                from hiercp.common import CasePaths
                value["case"] = SimpleNamespace(paths=CasePaths(case, Path(info["image"]), Path(info["label"])),
                    image=value["ct"], label=value["lab"], spacing=value["spacing"],
                    image_affine=image.affine, label_affine=label.affine)
                from hiercp.common import organ_depth_mm
                value["depth"] = organ_depth_mm(value["organ"], value["spacing"])
                for array in (value["ct"], value["lab"], value["organ"], value["depth"],
                              value["spacing"], image.affine, label.affine):
                    array.flags.writeable = False
                local._case_geometry(value["case"], value["organ"], value["depth"])
                return value
        self._crop_store = CropStore(self.ds.meta, self.workers, self.resident_bytes, self.rss_bytes)
        self._raw = OriginalRawStore(self.ds.meta, self.workers, self.rss_bytes)
        self._account_raw_cache()
        self._crop_store.raw = self._raw

    def _make_room(self, active_cases=(), active_records=()):
        active = set(active_cases)
        self._account_raw_cache()
        # A returned record can remain live after its cache owner is evicted.
        # Admit that complete active batch as well; shared raw/donor arrays are
        # counted once across every owner, including torch.from_numpy aliases.
        with self._resident.active(active_records):
            while self._records and self._resident.bytes("all") > self.resident_bytes:
                self._records.popitem(last=False)
            while self._donors and self._resident.bytes("all") > self.resident_bytes:
                key = next((key for key in self._donors if key[0] not in active), None)
                if key is None:
                    break
                self._donors.pop(key)
            if self._raw is not None:
                for case in list(self._raw.cache):
                    if case not in active and self._resident.bytes("all") > self.resident_bytes:
                        self._raw.cache.pop(case)
            if self._resident.bytes("all") > self.resident_bytes:
                raise MemoryError("Complete active D donor/recipient batch exceeds resident budget; no cropping or skip")
        self._cached_bytes = self._resident.bytes("canonical")
        self._guard()

    def _provenance(self, row):
        raw = {item["case_id"]: item for item in self.ds.meta["raw_records"]}
        result = dict(observation_id=row["id"], input_inventory_sha256=self.ds.index_sha256,
                      assignment_sha256=self.ds.assignment_sha256, donor_group=row["donor_group"],
                      recipient_group=row["patient_group"], debug=self.ds.debug)
        for branch, case in (("donor", row["donor_case_id"]), ("recipient", row["case_id"])):
            for kind in ("image", "label"):
                result[branch + "_" + kind + "_sha256"] = raw[case][kind + "_sha256"]
        return result

    def _prepare_donor(self, row):
        key = row["donor_case_id"], row["donor_component"]
        if key in self._donors:
            # The serial prewarm stage owns cache insertion/eviction. Parallel
            # recipient builders only read this complete immutable source.
            return self._donors[key][:2]
        volume = self._raw.cache[key[0]]
        from hiercp_v22.data import sources
        collection = sources(volume["case"], self.ds.base["cache"]["source_pad"],
                             self.ds.meta["config"]["donor_max_diameter_mm"])
        matches = [index for index, (component, _) in enumerate(collection.entries) if component == key[1]]
        if len(matches) != 1:
            raise ValueError("The exact signed original donor component is absent: " + repr(key))
        source, _ = collection[matches[0]]
        prepared = local.prepare_donor(volume["case"], source, volume["organ"], volume["depth"], self.ds.base)
        size = _bytes((source, prepared))
        self._donors[key] = source, prepared, size
        self._cached_bytes = self._resident.bytes("canonical")
        return source, prepared

    def _build(self, row):
        self._crop_store.validate_observation(row)  # Native GT admission is external to neural graph input.
        source, prepared = self._prepare_donor(row)
        target = self._raw.cache[row["case_id"]]
        self.last_admission = {key: copy.deepcopy(row[key]) for key in ASSIGNMENT_KEYS}
        try:
            return local.pair_record(target["case"], source, np.asarray(prepared.audit["donor_spacing"]), prepared,
                row["center"], target["organ"], target["depth"], self.ds.base,
                donor_id=row["donor_case_id"], provenance=self._provenance(row))
        except (ValueError, RuntimeError, MemoryError, OSError) as error:
            raise RuntimeError(f"D original geometry admission failed for {row['id']}; "
                f"donor={row['donor_case_id']} component={row['donor_component']} "
                f"recipient_center={row['center']}; no row skipped: {error}") from error

    def bind_cache(self, index):
        path = Path(index).resolve(strict=True)
        meta = json.loads(path.read_text(encoding="utf8"))
        if (meta.get("format") != FORMAT or meta.get("complete") is not True or meta.get("debug") is not self.ds.debug
                or meta.get("input_inventory_sha256") != self.ds.index_sha256
                or meta.get("assignment_sha256") != self.ds.assignment_sha256
                or meta.get("scope_contract") != self.ds.scope_contract
                or meta.get("local_identity") != local.source_identity()
                or meta.get("base") != self.ds.base):
            raise ValueError("D canonical cache has another native assignment/input/scope/source contract")
        originals = {row["id"]: row for row in self.ds.meta["records"]}
        stored = {row["id"]: row for row in meta["records"]}
        if (len(stored) != len(meta["records"]) or not set(stored) <= set(originals)
                or not {row["id"] for row in self.ds.rows} <= set(stored)
                or not self.ds.debug and set(stored) != set(originals)):
            raise ValueError("D canonical cache omitted or duplicated a native observation")
        for key, row in stored.items():
            if any(row[field] != originals[key][field] for field in ASSIGNMENT_KEYS):
                raise ValueError("D canonical cache changed native GT/donor assignment: " + key)
            _validate_measurement(row)
        ordered = meta["records"]
        if (meta.get("sampled_measurement_format") != MEASUREMENT_FORMAT
                or meta.get("sampled_two_view_nodes") != [row["sampled_two_view_nodes"] for row in ordered]
                or meta.get("sampled_two_view_edges") != [row["sampled_two_view_edges"] for row in ordered]
                or meta.get("sampled_two_view_summary") != _sampled_summary(ordered)):
            raise ValueError("D canonical index sampled graph measurements changed or are missing")
        self.cache_index, self._canonical = path, stored
        self._cache_index_sha256 = _sha(path)

    def measured_cost(self, ids):
        """Actual epoch0 sampled union cost for calibration, with no graph load."""
        if (self._canonical is None or self.cache_index is None
                or not isinstance(ids, (list, tuple)) or not ids
                or any(type(index) is not int or not 0 <= index < len(self.ds) for index in ids)):
            raise ValueError("Bound measured canonical cache and exact native observation indices required")
        if _checked_sha(self.cache_index) != self._cache_index_sha256:
            raise ValueError("Complete D canonical index changed after admission")
        rows = [self._canonical[self.ds.rows[index]["id"]] for index in ids]
        for row in rows:
            _validate_measurement(row)
        return dict(observations=len(rows), actual_local_graphs=2 * len(rows), measurement_epoch=0,
                    sampled_two_view_nodes=sum(row["sampled_two_view_nodes"] for row in rows),
                    sampled_two_view_edges=sum(row["sampled_two_view_edges"] for row in rows),
                    canonical_bound_bytes=sum(row["bounds"]["bytes"] for row in rows))

    def _load(self, row):
        stored = self._canonical[row["id"]]
        root = self.cache_index.parent
        selected = _verify_stored_files(root, stored)
        from hiercp_v22.storage import load_record
        record = load_record(selected, stored["path"])
        local.validate_record(record)
        if (record["input_provenance"]["observation_id"] != row["id"] or record["case_id"] != row["case_id"]
                or record["donor_case_id"] != row["donor_case_id"] or record["component_id"] != row["donor_component"]
                or record["center"] != row["center"]):
            raise ValueError("D canonical graph and actual native row identity differ")
        return record

    def _records_for(self, rows):
        self._guard()
        if self.cache_index is not None and _checked_sha(self.cache_index) != self._cache_index_sha256:
            raise ValueError("Complete D canonical index changed after admission")
        missing = [row for row in rows if row["id"] not in self._records]
        if self._canonical is None and missing:
            self._initialize_raw()
            active = {row[key] for row in missing for key in ("case_id", "donor_case_id")}
            self._make_room(active)
            self._raw.preload(missing)
            self._make_room(active)
            # Construct shared donor branches before parallel recipient builds.
            for row in missing:
                self._prepare_donor(row)
            build = self._build
        else:
            build = self._load
        if missing:
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                records = list(pool.map(build, missing))
            for row, record in zip(missing, records):
                size = _bytes(record)
                self._records[row["id"]] = record, size
            self._cached_bytes = self._resident.bytes("canonical")
        result = []
        for row in rows:
            record, size = self._records[row["id"]]
            self._records.move_to_end(row["id"])
            result.append(record)
        # Returned complete records stay owned by the caller even if cache
        # eviction occurs. Eviction changes residency, never data inclusion.
        self._make_room(active_records=result)
        return result

    def get(self, ids, *, epoch=0):
        if (not isinstance(ids, (list, tuple)) or not ids
                or any(type(index) is not int or not 0 <= index < len(self.ds) for index in ids)):
            raise ValueError("Exact actual native observation row indices required")
        rows = [self.ds.rows[index] for index in ids]
        records = self._records_for(rows)
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            payloads = list(pool.map(lambda record: local.materialize_pair(record, epoch=epoch), records))
        batch = local.collate(list(zip(payloads, ids)))
        self._guard()
        return batch

    def release_batches(self):
        # Canonical record cache is distinct from ephemeral collated graph batches.
        self._guard()

    def batches(self, schedule, *, epoch=0):
        """One prefetched complete CPU batch while the current GPU batch runs."""
        with ThreadPoolExecutor(max_workers=1) as producer:
            iterator = iter(schedule)
            ids = next(iterator, None)
            if ids is None:
                return
            future = producer.submit(self.get, ids, epoch=epoch)
            for ids in iterator:
                batch = future.result()
                future = producer.submit(self.get, ids, epoch=epoch)
                yield batch
            yield future.result()

    def preparation_request(self):
        """Exact preparation identity, also usable before launching workers."""
        rows = self.ds.rows if self.ds.debug else self.ds.meta["records"]
        return dict(format=FORMAT, input_inventory=str(self.ds.path),
            input_inventory_sha256=self.ds.index_sha256, assignment_sha256=self.ds.assignment_sha256,
            prepared_assignment_sha256=assignment_digest(rows), prepared_observations=len(rows), debug=self.ds.debug,
            scope_contract=self.ds.scope_contract, local_identity=local.source_identity(), base=self.ds.base)

    def preflight(self, output, *, minimum_free_bytes):
        """Prepare all rows, resume signed completed receipts without overwrite.

        Interrupted/uncommitted files stay in their original segment. A resume
        writes missing observations into a fresh segment and revalidates every
        completed record/hash. Only complete full coverage publishes index.json.
        """
        if type(minimum_free_bytes) is not int or minimum_free_bytes <= 0:
            raise ValueError("Explicit positive disk reserve required for full canonical preparation")
        root = Path(output).resolve()
        if root.is_symlink():
            raise ValueError("D preparation must use an owned real directory")
        if (root / "index.json").exists():
            self.bind_cache(root / "index.json")
            return self.cache_index
        rows = self.ds.rows if self.ds.debug else self.ds.meta["records"]
        expected_digest = assignment_digest(rows)
        request = self.preparation_request()
        request_path = root / "prepare_request.json"
        if root.exists():
            if request_path.exists():
                if request_path.is_symlink() or json.loads(request_path.read_text(encoding="utf8")) != request:
                    raise ValueError("Interrupted D preparation belongs to another exact request/source/assignment")
            elif any(root.iterdir()):
                raise ValueError("Existing directory has no owned D preparation request; no results overwritten")
            else:
                _publish_new_json(request_path, request)
        else:
            root.mkdir(parents=True, exist_ok=False)
            _publish_new_json(request_path, request)
        ledger = root / "completed"
        ledger.mkdir(exist_ok=True)
        expected_rows = {row["id"]: row for row in rows}
        completed = {}
        from tqdm import tqdm
        receipts = sorted(ledger.glob("*.json"))
        for receipt in tqdm(receipts, desc="D resume exact completed file checks", unit="observation"):
            if receipt.is_symlink():
                raise ValueError("D preparation completion receipt cannot be a symlink")
            stored = json.loads(receipt.read_text(encoding="utf8"))
            original = expected_rows.get(stored.get("id"))
            if (original is None or stored["id"] in completed
                    or any(stored[key] != original[key] for key in ASSIGNMENT_KEYS)):
                raise ValueError("D completion receipt changed/duplicated a signed native assignment")
            _verify_stored_files(root, stored)
            _validate_measurement(stored)
            completed[stored["id"]] = stored
        resumed = len(completed)
        started = time.perf_counter()
        invocation = uuid.uuid4().hex
        metrics_path = root / ("preparation_metrics_" + invocation + ".jsonl")
        metrics = metrics_path.open("x", encoding="utf8")
        # Include preserved uncommitted files from previous interrupted segments.
        # A single initial scan avoids walking the complete cache at each row.
        segment_files = list((root / "segments").rglob("*")) if (root / "segments").exists() else []
        cumulative_disk = sum(path.stat().st_size for path in segment_files if path.is_file())
        invocation_disk = 0
        seen_source_files = {path.resolve() for path in segment_files if path.is_file()
                             and path.parent.name == "shared_sources"}
        def report(event, **value):
            entry = dict(format="D_actual_canonical_preparation_measurements_v1", event=event,
                         invocation=invocation, debug=self.ds.debug, total_observations=len(rows),
                         completed_observations=len(completed), seconds=time.perf_counter() - started,
                         disk_cumulative_canonical_bytes=cumulative_disk,
                         disk_written_canonical_bytes=invocation_disk,
                         disk_free_bytes=shutil.disk_usage(root).free, **value)
            metrics.write(json.dumps(entry, allow_nan=False) + "\n")
            metrics.flush()
        report("started", resumed_exact_completed_observations=resumed,
               execution_chunk_observations=self.workers, graph_or_observation_cap=False,
               sampled_measurement_format=MEASUREMENT_FORMAT, **self._memory_measurement())
        print("D actual preparation measurements: " + str(metrics_path), flush=True)
        progress = tqdm(total=len(rows), initial=resumed, desc="D actual original canonical preparation", unit="observation")
        segment = "segments/" + uuid.uuid4().hex
        segment_root = root / segment
        segment_root.mkdir(parents=True, exist_ok=False)
        from .transition_preparation_storage import GraphWriter
        writer = GraphWriter(segment_root, minimum_free_bytes=minimum_free_bytes)
        # All observations from one recipient share the native fixed donor. CPU
        # recipient graph preparation is parallel; each actual row is preserved.
        cases = list(dict.fromkeys(row["case_id"] for row in rows))
        case_sizes = {case: sum(row["case_id"] == case for row in rows) for case in cases}
        completed_counts = {case: sum(row["case_id"] == case for row in completed.values()) for case in cases}
        ordinals = {row["id"]: index for index, row in enumerate(rows)}
        def prepare_and_store(item):
            row, record = item
            sample_started = time.perf_counter()
            payload = local.materialize_pair(record, epoch=0)  # Both genuine views, no neural run.
            measured = _sampled_measurement(payload)
            self._guard()  # Include in-flight sampled views in the process RSS guard.
            sample_seconds = time.perf_counter() - sample_started
            del payload
            canonical_bytes = _bytes(record)
            source_bytes = _bytes((record.get("source_local"), record.get("source_patch")))
            target_bytes = _bytes((record.get("target_local"), record.get("target_patch")))
            ordinal = ordinals[row["id"]]
            write_started = time.perf_counter()
            stored = writer.write(f"graphs/{row['case_id']}/{ordinal:06d}.pt.gz", record,
                                  f"{row['donor_case_id']}:{row['donor_component']}")
            source_path = (segment_root / stored["shared_source"]["path"]).resolve(strict=True)
            graph_compressed_bytes = (segment_root / stored["path"]).stat().st_size
            source_compressed_bytes = source_path.stat().st_size
            storage_seconds = time.perf_counter() - write_started
            self._guard()
            stored = ({key: copy.deepcopy(row[key]) for key in ASSIGNMENT_KEYS} | stored | measured
                      | dict(segment=segment, canonical_tensor_bytes=canonical_bytes,
                             source_canonical_tensor_bytes=source_bytes,
                             target_canonical_tensor_bytes=target_bytes,
                             compressed_graph_bytes=graph_compressed_bytes,
                             compressed_shared_source_bytes=source_compressed_bytes))
            return dict(row=row, ordinal=ordinal, stored=stored, measured=measured,
                        sample_seconds=sample_seconds, storage_seconds=storage_seconds,
                        source_path=source_path, write_started=write_started)
        try:
          with ThreadPoolExecutor(max_workers=self.workers) as post_build_pool:
            for case in cases:
                actual = [row for row in rows if row["case_id"] == case and row["id"] not in completed]
                # Worker-sized execution chunks bound in-flight preparation,
                # while preserving every signed row and every original graph.
                for first in range(0, len(actual), self.workers):
                    requested = actual[first:first + self.workers]
                    progress.set_postfix(case=case, phase="building", execution_rows=len(requested),
                        completed_cases=sum(completed_counts[other] == case_sizes[other] for other in cases))
                    report("chunk_started", case_id=case, execution_observations=len(requested),
                           **self._memory_measurement())
                    build_started = time.perf_counter()
                    records = self._records_for(requested)
                    build_seconds = time.perf_counter() - build_started
                    report("chunk_built", case_id=case, execution_observations=len(requested),
                           build_seconds=build_seconds, **self._memory_measurement(records))
                    post_build_started = time.perf_counter()
                    # Independent two-view sampling, serialization, hashing and
                    # compression run concurrently. Only complete receipts and
                    # aggregate progress are published in signed ordinal order.
                    for prepared in post_build_pool.map(prepare_and_store, zip(requested, records)):
                        row, ordinal, stored = prepared["row"], prepared["ordinal"], prepared["stored"]
                        measured, source_path = prepared["measured"], prepared["source_path"]
                        graph_compressed_bytes = stored["compressed_graph_bytes"]
                        source_compressed_bytes = stored["compressed_shared_source_bytes"]
                        new_source_bytes = 0 if source_path in seen_source_files else source_compressed_bytes
                        seen_source_files.add(source_path)
                        written = graph_compressed_bytes + new_source_bytes
                        cumulative_disk += written
                        invocation_disk += written
                        _publish_new_json(ledger / f"{ordinal:06d}.json", stored)
                        completed[row["id"]] = stored
                        completed_counts[case] += 1
                        self._guard()
                        memory = self._memory_measurement(records)
                        progress.set_postfix(case=case, phase="stored", sampled_nodes=measured["sampled_two_view_nodes"],
                            sampled_edges=measured["sampled_two_view_edges"], rss_GiB=round(memory["rss_bytes"] / 2**30, 3),
                            disk_GiB=round(cumulative_disk / 2**30, 3), refresh=False)
                        progress.update(1)
                        report("observation_completed", observation_id=row["id"], case_id=case,
                               donor_case_id=row["donor_case_id"], donor_component=row["donor_component"],
                               sampled_nodes=measured["sampled_view_nodes"], sampled_edges=measured["sampled_view_edges"],
                               sampled_two_view_nodes=measured["sampled_two_view_nodes"],
                               sampled_two_view_edges=measured["sampled_two_view_edges"],
                               canonical_tensor_bytes=stored["canonical_tensor_bytes"],
                               source_canonical_tensor_bytes=stored["source_canonical_tensor_bytes"],
                               target_canonical_tensor_bytes=stored["target_canonical_tensor_bytes"],
                               compressed_graph_bytes=graph_compressed_bytes,
                               compressed_shared_source_bytes=source_compressed_bytes,
                               newly_written_shared_source_bytes=new_source_bytes,
                               build_chunk_seconds=build_seconds, execution_observations=len(requested),
                               materialize_two_view_seconds=prepared["sample_seconds"],
                               storage_write_seconds=prepared["storage_seconds"],
                               write_publish_seconds=time.perf_counter() - prepared["write_started"],
                               **memory)
                    report("chunk_post_build_completed", case_id=case,
                           execution_observations=len(requested),
                           post_build_wall_seconds=time.perf_counter() - post_build_started,
                           source_encode_calls=getattr(writer, "source_encode_calls", None),
                           source_memo_hits=getattr(writer, "source_memo_hits", None),
                           **self._memory_measurement(records))
                    # The previous execution chunk must not stay alive while
                    # Python evaluates construction of the following chunk.
                    del records, prepared
                progress.set_postfix(case=case, completed_cases=sum(
                    completed_counts[other] == case_sizes[other] for other in cases))
            report("all_observations_completed", **self._memory_measurement())
        finally:
            progress.close()
            metrics.close()
        if len(completed) != len(rows) or set(completed) != set(expected_rows):
            raise ValueError("Full D preparation did not cover every native assignment")
        ordered = [completed[row["id"]] for row in rows]
        if assignment_digest(ordered) != expected_digest:
            raise ValueError("D preparation changed an original native P/U/donor assignment")
        meta = dict(format=FORMAT, complete=True, debug=self.ds.debug, input_inventory=str(self.ds.path),
            input_inventory_sha256=self.ds.index_sha256, assignment_sha256=self.ds.assignment_sha256,
            scope_contract=self.ds.scope_contract, local_identity=local.source_identity(), base=self.ds.base,
            population=self.ds.population, records=ordered, workers=self.workers,
            prepared_assignment_sha256=expected_digest, prepared_observations=len(rows),
            prepared_case_ids=cases, source_total_assignments=len(self.ds.meta["records"]),
            explicit_DEBUG_case_ids=None if self.ds.case_ids is None else list(self.ds.case_ids),
            resumed_exact_completed_observations=resumed,
            resident_bytes=self.resident_bytes, rss_bytes=self.rss_bytes,
            skipped_observations=0, donor_redraws=0, hidden_subset=False, views_per_observation=2)
        meta.update(sampled_measurement_format=MEASUREMENT_FORMAT,
            sampled_two_view_nodes=[row["sampled_two_view_nodes"] for row in ordered],
            sampled_two_view_edges=[row["sampled_two_view_edges"] for row in ordered],
            sampled_two_view_summary=_sampled_summary(ordered),
            execution_chunk_observations=self.workers, execution_chunk_is_sample_cap=False,
            preparation_seconds=time.perf_counter() - started,
            disk_cumulative_canonical_bytes=cumulative_disk, disk_written_canonical_bytes=invocation_disk,
            preparation_metric_files=sorted(path.name for path in root.glob("preparation_metrics_*.jsonl")))
        path = root / "index.json"
        _publish_new_json(path, meta)
        self.bind_cache(path)
        return path
