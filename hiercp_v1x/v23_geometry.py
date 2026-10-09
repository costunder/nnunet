"""Immutable all-P/active-U CPU upper graphs for the new v2.3 experiment.

Original geometry equations stay byte-for-byte represented by their AST. Two
admission clauses bind the v2.3 train/validation partition and signed native
candidate coordinates. A native bbox anchor can lie in background; its exact
coordinate and the original region_at assignment remain unchanged. Frozen
historical helpers are never edited.
The cache contains geometry only, never neural features or predictions.
"""
from __future__ import annotations

import ast
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import sys
import threading
import time
from types import SimpleNamespace
import uuid

from .contracts import canonical_hash
from .historical_evaluation import sha

FORMAT = 'v23_allP_activeU_immutable_upper_geometry_v1'


def _expression(text):
    return ast.dump(ast.parse(text, mode='eval').body, include_attributes=False)


def recipient_admitted(recipient, training, train_cases, validation_cases):
    """Admit only the declared side, with the entire original training bank."""
    fitted = tuple(map(str, training))
    train, validation = tuple(map(str, train_cases)), tuple(map(str, validation_cases))
    if (not train or len(train) != len(set(train))
            or len(validation) != len(set(validation)) or set(train) & set(validation)
            or set(fitted) != set(train) or len(fitted) != len(train)):
        return False
    return str(recipient) in (set(train) | set(validation))


def center_admitted(recipient, center, regions, recorded_centers):
    """Admit exact recorded native coordinates, with no target/class argument.

    The original builder still independently enforces integer CT bounds and
    ``spec.region_id == regions.region_at(center)``. Bbox midpoint anchors can
    occupy a concave component's background hole; region_at already implements
    an exact nearest original region for those positions. No coordinate moves.
    """
    import numpy as np
    value = np.asarray(center)
    if (value.shape != (3,) or not np.issubdtype(value.dtype, np.integer)
            or np.any(value < 0) or np.any(value >= regions.full_organ_mask.shape)):
        return False
    coordinate = tuple(map(int, value))
    if recorded_centers is None:
        # Direct mechanical admission without an inventory keeps the historical
        # restriction. Production always passes its frozen native allowlist.
        return bool(regions.full_organ_mask[coordinate])
    return coordinate in recorded_centers.get(str(recipient), frozenset())


def adapted_builders(train_cases, validation_cases, *, recorded_centers=None):
    """Change exactly two admission expressions; preserve all graph equations."""
    from . import historical_patient_graph, historical_evaluation
    original = historical_patient_graph.build_external_hierarchy
    text = inspect.getsource(original)
    tree = ast.parse(text)
    expected = _expression('str(recipient_case.paths.case_id) in training')
    expected_center = _expression('not recipient_regions.full_organ_mask[tuple(center)]')
    centers = None
    if recorded_centers is not None:
        declared = set(map(str, train_cases)) | set(map(str, validation_cases))
        if not isinstance(recorded_centers, dict) or set(recorded_centers) - declared:
            raise ValueError('Native center allowlist must belong to bound train/validation cases')
        centers = {}
        for case, coordinates in recorded_centers.items():
            checked = []
            for center in coordinates:
                values = tuple(center)
                if len(values) != 3 or any(type(value) is not int or value < 0 for value in values):
                    raise ValueError('Frozen native center allowlist requires exact nonnegative integer coordinates')
                checked.append(values)
            centers[case] = frozenset(checked)

    class Admission(ast.NodeTransformer):
        count = 0
        center_count = 0

        def visit_Compare(self, node):
            if ast.dump(node, include_attributes=False) == expected:
                self.count += 1
                return ast.copy_location(ast.parse(
                    'not _v23_recipient_admitted(str(recipient_case.paths.case_id), training)',
                    mode='eval').body, node)
            return self.generic_visit(node)

        def visit_UnaryOp(self, node):
            if ast.dump(node, include_attributes=False) == expected_center:
                self.center_count += 1
                return ast.copy_location(ast.parse(
                    'not _v23_center_admitted(str(recipient_case.paths.case_id), center, recipient_regions)',
                    mode='eval').body, node)
            return self.generic_visit(node)

    replacement = Admission()
    adapted = replacement.visit(copy.deepcopy(tree))
    if replacement.count != 1:
        raise ValueError('Original recipient admission AST changed; exactly one replacement required')
    if replacement.center_count != 1:
        raise ValueError('Original organ-center admission AST changed; exactly one replacement required')
    ast.fix_missing_locations(adapted)
    namespace = dict(original.__globals__)
    namespace['_v23_recipient_admitted'] = lambda recipient, training: recipient_admitted(
        recipient, training, train_cases, validation_cases)
    namespace['_v23_center_admitted'] = lambda recipient, center, regions: center_admitted(
        recipient, center, regions, centers)
    exec(compile(adapted, original.__code__.co_filename + ':v23_admission', 'exec'), namespace)
    builder = namespace[original.__name__]

    upper = historical_evaluation.upper_graphs
    upper_tree = ast.parse(inspect.getsource(upper))
    imports = [node for node in ast.walk(upper_tree) if isinstance(node, ast.ImportFrom)
               and node.module == 'historical_patient_graph'
               and any(alias.name == 'build_external_hierarchy' for alias in node.names)]
    if len(imports) != 1 or imports[0].level != 1 or len(imports[0].names) != 1:
        raise ValueError('Original upper builder import changed; one exact local import required')
    function = upper_tree.body[0]
    if imports[0] not in function.body:
        raise ValueError('Original upper builder import moved; refusing an implicit equation change')
    function.body.remove(imports[0])
    ast.fix_missing_locations(upper_tree)
    upper_namespace = dict(upper.__globals__)
    upper_namespace['build_external_hierarchy'] = builder
    exec(compile(upper_tree, upper.__code__.co_filename + ':v23_admission', 'exec'), upper_namespace)
    receipt = dict(
        format='v23_original_upper_AST_partition_and_native_center_admission_v2',
        historical_builder_file_sha256=sha(inspect.getsourcefile(original)),
        historical_upper_file_sha256=sha(inspect.getsourcefile(upper)),
        original_builder_function_sha256=hashlib.sha256(text.encode()).hexdigest(),
        original_upper_function_sha256=hashlib.sha256(inspect.getsource(upper).encode()).hexdigest(),
        admission_replacements=1, native_center_admission_replacements=1,
        total_admission_replacements=2, upper_import_removals=1,
        native_coordinate_allowlist_bound=centers is not None,
        native_coordinate_allowlist_sha256=None if centers is None else canonical_hash(
            {case: [list(center) for center in sorted(values)] for case, values in sorted(centers.items())}),
        native_coordinate_allowlist_counts=None if centers is None else {
            case: len(values) for case, values in sorted(centers.items())},
        coordinate_allowlist_contains_no_P_U_targets=True,
        native_bbox_anchor_background_permitted_only_if_recorded=True,
        CT_coordinate_bounds_and_original_region_at_guard_preserved=True,
        original_feature_relation_prototype_equations_preserved=True,
        training_cases=list(train_cases), validation_cases=list(validation_cases),
        train_and_validation_partition_bound=True, original_helpers_modified=False,
        recipient_annotation_exposed=True, P_U_targets_in_graph_inputs=False)
    return upper_namespace[upper.__name__], receipt


def _new_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.partial-' + uuid.uuid4().hex)
    with temporary.open('x', encoding='utf8') as output:
        json.dump(value, output, indent=2, allow_nan=False)
        output.write('\n')
        output.flush()
        os.fsync(output.fileno())
    _publish_new(temporary, path)


def _publish_new(temporary, destination):
    """Atomic publication that refuses an existing destination on both hosts."""
    if os.name == 'nt':
        # Windows rename refuses an existing target; POSIX rename replaces it.
        os.rename(temporary, destination)
    else:
        os.link(temporary, destination)
        Path(temporary).unlink()


def _regions(directory):
    directory = Path(directory)
    if (directory.is_symlink() or not directory.is_dir()
            or not (directory / 'metadata.json').is_file()):
        raise ValueError('Existing complete original region publication required: ' + str(directory))
    paths = sorted(path for path in directory.rglob('*') if path.is_file())
    if any(path.is_symlink() for path in paths):
        raise ValueError('Region symlinks cannot be admitted to the new geometry cache')
    return {path.relative_to(directory).as_posix(): sha(path) for path in paths}


class V23UpperGeometryCache:
    """Build CPU graph cases in parallel, then load private independent copies."""
    def __init__(self, bundle, population, output, *, workers, rss_bytes,
                 full_validation_cache=None, resident_bytes=128 * 2**30,
                 memoize_cpu_geometry=True, cache_stage_admission=True):
        import psutil
        if (type(workers) is not int or workers < 2
                or workers > len(psutil.Process().cpu_affinity())
                or type(rss_bytes) is not int or not 0 < rss_bytes <= 192 * 2**30
                or type(resident_bytes) is not int or not 0 < resident_bytes <= 128 * 2**30
                or type(memoize_cpu_geometry) is not bool or type(cache_stage_admission) is not bool):
            raise ValueError('Explicit parallel CPU workers and positive RSS budget required')
        self.bundle, self.population = bundle, population
        self.output = Path(output).resolve()
        self.workers, self.rss_bytes = workers, rss_bytes
        self.resident_budget = resident_bytes
        self.memoize_cpu_geometry = memoize_cpu_geometry
        self.cache_stage_admission = cache_stage_admission
        self._memo = OrderedDict()
        self._resident_size = 0
        self._memo_lock = threading.RLock()
        self._file_proofs = {}
        self._receipt_proofs = {}
        self._admitted_stages = {}
        self._stage_aliases = {}
        self._historical_proofs = None
        self.train_cases = tuple(population.partition_cases('inner_train'))
        self.validation_cases = tuple(population.partition_cases('inner_val'))
        if set(self.train_cases) & set(self.validation_cases):
            raise ValueError('v2.3 geometry requires disjoint patient partitions')
        recorded_centers = {case: tuple(tuple(row['center']) for row in
            self.population.case(case, 128).query_rows)
            for case in self.train_cases + self.validation_cases}
        self._upper, self.adaptation = adapted_builders(self.train_cases, self.validation_cases,
                                                      recorded_centers=recorded_centers)
        self._entries = {}
        self._stage_signatures = {}
        self._plan_stages = {}
        self._region_bindings = {}
        self._input_stats = {}
        self._raw = {row['case_id']: row for row in population.meta['raw_records']}
        self._full_validation = None
        self._full_root = None
        if full_validation_cache is not None:
            if hasattr(full_validation_cache, 'upper_graphs'):
                self._full_validation = full_validation_cache
                self._full_root = Path(full_validation_cache.root)
            else:
                self._full_root = Path(full_validation_cache).resolve(strict=True)
                from .comparison_native_upper_cache import UpperGeometryCache
                index = json.loads((self._full_root / 'index.json').read_text())
                inventory = index['signature']['native_inventory_path']
                self._full_validation = UpperGeometryCache(self._full_root, bundle, inventory)
        preserved = [Path(bundle.baseline).resolve(), Path(bundle.source).resolve()]
        if self._full_root is not None:
            preserved.append(self._full_root)
        if any(self.output == old or self.output.is_relative_to(old)
               or old.is_relative_to(self.output) for old in preserved):
            raise ValueError('New v2.3 geometry output must be disjoint from preserved inputs')
        self.output.mkdir(parents=True, exist_ok=True)
        self.receipt = dict(format=FORMAT, original_equations=self.adaptation,
            neural_features_cached=False, predictions_cached=False, geometry_only=True,
            CPU_workers=workers, RSS_budget_bytes=rss_bytes, stages=[], cache_hits=0,
            original_helpers_modified=False, full_validation_historical_cache_reused=False)
        self.receipt['CPU_resident_memo'] = dict(enabled=memoize_cpu_geometry,
            stage_admission_cache=cache_stage_admission, budget_bytes=resident_bytes,
            resident_bytes=0, peak_resident_bytes=0, entries=0, evictions=0,
            hits=0, historical_hits=0, historical_loads=0, payload_deserializations=0,
            SHA_file_reads=0, stage_admission_hits=0, oversize_streams=0,
            SHA_seconds=0., deserialize_seconds=0., clone_seconds=0., stat_guard_seconds=0.,
            RSS_bytes=0, peak_RSS_bytes=0,
            identity_guard='device/inode/size/mtime_ns/ctime_ns after SHA256 admission',
            private_output_tensors=True, memo_tensor_storage_CPU_only=True,
            resident_accounting='unique tensor storage plus retained Python metadata')
        from . import v23_data, historical_patient_graph, historical_evaluation
        source_files = {str(Path(__file__).resolve()), str(Path(v23_data.__file__).resolve()),
                        str(Path(historical_patient_graph.__file__).resolve()),
                        str(Path(historical_evaluation.__file__).resolve())}
        source_files.update(str(path) for path in Path(bundle.source).rglob('*.py'))
        self._source_files = {path: self._prove_file(path) for path in sorted(source_files)}
        for path, checksum in getattr(bundle, 'receipt', {}).get('files_preserved', {}).items():
            self._source_files[str(Path(path))] = self._prove_file(path, checksum)
        source_root = Path(bundle.source)
        self._source_dirs = {str(path): self._directory_stat(path) for path in
            [source_root, *sorted(path for path in source_root.rglob('*') if path.is_dir())]}

    @staticmethod
    def _directory_stat(path):
        path = Path(path)
        if path.is_symlink() or not path.is_dir():
            raise ValueError('Regular original source directory required: ' + str(path))
        value = path.stat()
        return (value.st_dev, value.st_ino, value.st_mtime_ns, value.st_ctime_ns)

    def _count(self, key, value=1):
        with self._memo_lock:
            self.receipt['CPU_resident_memo'][key] += value

    def _prove_file(self, path, expected=None):
        """Full SHA admission once, then exact identity guards; never stale reuse."""
        path = str(Path(path))
        before = self._stat(path)
        with self._memo_lock:
            previous = self._file_proofs.get(path)
        if previous is not None:
            if before != previous[0] or (expected is not None and previous[1] != expected):
                raise ValueError('Verified geometry payload/source identity differs: ' + path)
            return previous[1]
        began = time.perf_counter()
        digest = sha(path)
        if self._stat(path) != before or (expected is not None and digest != expected):
            raise ValueError('Immutable geometry payload/query/source identity differs: ' + path)
        with self._memo_lock:
            if path in self._file_proofs and self._file_proofs[path] != (before, digest):
                raise ValueError('Concurrent immutable geometry proof differs')
            self._file_proofs[path] = (before, digest)
        self._count('SHA_file_reads')
        self._count('SHA_seconds', time.perf_counter() - began)
        return digest

    def _guard_sources(self):
        began = time.perf_counter()
        for path, checksum in self._source_files.items():
            self._prove_file(path, checksum)
        if any(self._directory_stat(path) != value for path, value in self._source_dirs.items()):
            raise ValueError('Original archived geometry source directory changed')
        self._count('stat_guard_seconds', time.perf_counter() - began)

    def _prove_case_receipt(self, entry, plan):
        path = self._entry_path(plan.active_u_count, plan.case_id,
            stage_key=self._plan_stages.get(self._plan_key(plan), plan.active_u_count))
        checksum = self._prove_file(path)
        with self._memo_lock:
            prior = self._receipt_proofs.get(str(path))
        signature = canonical_hash(entry)
        if prior is None:
            if json.loads(path.read_text()) != entry:
                raise ValueError('Published v2.3 geometry case receipt differs')
            self._prove_file(path, checksum)
            with self._memo_lock:
                self._receipt_proofs[str(path)] = signature
        elif prior != signature:
            raise ValueError('Admitted v2.3 geometry case binding changed')

    @staticmethod
    def _resident_bytes(answer):
        """Count unique CPU tensor storage plus retained Python metadata."""
        import torch
        seen, stores = set(), set()
        dictionaries = []
        def size(value):
            if id(value) in seen:
                return 0
            seen.add(id(value))
            total = sys.getsizeof(value)
            if torch.is_tensor(value):
                if value.device.type != 'cpu':
                    raise ValueError('Geometry memo must contain CPU tensors only')
                storage = value.untyped_storage()
                identity = (storage.data_ptr(), storage.nbytes())
                if identity not in stores:
                    stores.add(identity)
                    total += storage.nbytes()
            elif hasattr(value, 'to_dict'):
                # Retain these temporary mappings until counting finishes so
                # Python cannot recycle their IDs between graph/prototype.
                mapping = value.to_dict()
                dictionaries.append(mapping)
                total += size(mapping)
            elif isinstance(value, dict):
                total += sum(size(key) + size(item) for key, item in value.items())
            elif isinstance(value, (tuple, list)):
                total += sum(map(size, value))
            return total
        return size(answer)

    def _evict_one(self):
        _, value = self._memo.popitem(last=False)
        self._resident_size -= value['bytes']
        self.receipt['CPU_resident_memo']['evictions'] += 1
        self._memo_stats()

    def _memo_stats(self):
        stats = self.receipt['CPU_resident_memo']
        stats.update(resident_bytes=self._resident_size, entries=len(self._memo))
        stats['peak_resident_bytes'] = max(stats['peak_resident_bytes'], self._resident_size)

    def _remember(self, key, answer, binding, guards):
        if not self.memoize_cpu_geometry:
            return
        owned_binding, owned_guards = copy.deepcopy(binding), dict(guards)
        needed = self._resident_bytes((answer, owned_binding, owned_guards))
        with self._memo_lock:
            if needed > self.resident_budget:
                self.receipt['CPU_resident_memo']['oversize_streams'] += 1
                return  # Stream the unchanged complete graph; no scientific cap.
            old = self._memo.pop(key, None)
            if old is not None:
                self._resident_size -= old['bytes']
            while self._memo and self._resident_size + needed > self.resident_budget:
                self._evict_one()
            self._memo[key] = dict(answer=answer, binding=owned_binding, guards=owned_guards, bytes=needed)
            self._resident_size += needed
            self._memo_stats()
        self._rss()

    def _recalled(self, key, binding):
        if not self.memoize_cpu_geometry:
            return None
        with self._memo_lock:
            value = self._memo.get(key)
            if value is None:
                return None
            if value['binding'] != binding:
                raise ValueError('Resident geometry query/config/source binding changed')
            for path, checksum in value['guards'].items():
                self._prove_file(path, checksum)
            self._memo.move_to_end(key)
            self.receipt['CPU_resident_memo']['hits'] += 1
            return value['answer']

    def _private(self, answer):
        began = time.perf_counter()
        graph, prototype, audit = answer
        result = graph.clone(), prototype.clone(), copy.deepcopy(audit)
        self._count('clone_seconds', time.perf_counter() - began)
        self._rss()
        return result

    def _rss(self):
        import psutil
        with self._memo_lock:
            while self._memo and psutil.Process().memory_info().rss > self.rss_bytes:
                self._evict_one()
        if psutil.Process().memory_info().rss > self.rss_bytes:
            raise MemoryError('v2.3 geometry RSS budget exceeded; no case/graph reduction')
        current = psutil.Process().memory_info().rss
        with self._memo_lock:
            stats = self.receipt['CPU_resident_memo']
            stats['RSS_bytes'] = current
            stats['peak_RSS_bytes'] = max(stats['peak_RSS_bytes'], current)

    def _prepare_regions(self, *, readonly=False):
        cases = set(self.population.partition_cases('inner_train', ranking_only=True)) | set(self.validation_cases)
        cases.update(self.population.case(case).donor_case_id for case in tuple(cases))
        for case in sorted(cases):
            if case in self._region_bindings:
                continue
            original = Path(self.bundle.baseline) / 'shared/regions' / case
            if original.is_dir():
                directory = original
                files = _regions(original)
            else:
                if self._full_root is None:
                    raise ValueError('Original region absent; no silent rebuild: ' + case)
                fallback = self._full_root / 'regions' / case
                files = _regions(fallback)
                directory = self.output / 'regions' / case
                if not directory.exists():
                    if readonly:
                        raise ValueError('Published v2.3 region absent; read-only admission cannot rebuild: ' + case)
                    directory.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copytree(fallback, directory)
                if _regions(directory) != files:
                    raise ValueError('Copied original region SHA256 differs: ' + case)
            self._region_bindings[case] = dict(path=str(directory), files=files)
            paths = [directory / name for name in files]
            paths += [Path(self._raw[case][key]) for key in ('image', 'label')]
            for key in ('image', 'label'):
                if sha(self._raw[case][key]) != self._raw[case][key + '_sha256']:
                    raise ValueError('Original v2.3 raw CT/annotation SHA256 differs: ' + case)
            for path in paths:
                self._input_stats[str(path)] = self._stat(path)

    @staticmethod
    def _stat(path):
        path = Path(path)
        if path.is_symlink() or not path.is_file():
            raise ValueError('Regular original input required: ' + str(path))
        value = path.stat()
        return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)

    def _guard_inputs(self, plan):
        began = time.perf_counter()
        for case in (plan.case_id, plan.donor_case_id):
            region = self._region_bindings[case]
            paths = [str(Path(region['path']) / name) for name in region['files']]
            paths += [self._raw[case][key] for key in ('image', 'label')]
            if any(self._stat(path) != self._input_stats[str(path)] for path in paths):
                raise ValueError('Admitted v2.3 CT/annotation/region input changed: ' + case)
        self._count('stat_guard_seconds', time.perf_counter() - began)

    def _binding(self, plan):
        value = dict(format=FORMAT, population_sha256=self.population.manifest()['sha256'],
            query_rows=plan.query_rows, query_sha256=canonical_hash(plan.query_rows),
            active_U=plan.active_u_count, candidate_count=len(plan.record_ids),
            model_graph_config=self.bundle.config['graph'], ct_clip=self.bundle.config['ct_clip'],
            bank_fingerprint=self.bundle.prototype_bank.fingerprint(),
            bank_training_cases=list(self.bundle.prototype_bank.training_case_ids),
            source_adaptation=self.adaptation,
            raw_inputs={case: {key: self._raw[case][key] for key in
                ('image', 'label', 'image_sha256', 'label_sha256')} for case in
                (plan.case_id, plan.donor_case_id)},
            regions={case: self._region_bindings[case] for case in
                     (plan.case_id, plan.donor_case_id)})
        selected = getattr(plan, 'active_u_indices', tuple(range(plan.active_u_count)))
        if selected != tuple(range(plan.active_u_count)):
            value['frozen_U_bank_indices'] = list(selected)
        return value

    @staticmethod
    def _plan_key(plan):
        selected = getattr(plan, 'active_u_indices', tuple(range(plan.active_u_count)))
        if selected == tuple(range(plan.active_u_count)):
            return (plan.active_u_count, plan.case_id)
        return (plan.active_u_count, plan.case_id, tuple(selected))

    def _stage_path(self, stage_key):
        if isinstance(stage_key, tuple):
            count, selection_sha = stage_key
            return self.output / 'target_selections' / selection_sha / ('active_U_' + str(count))
        return self.output / ('active_U_' + str(stage_key))

    def _plans_for_stage(self, active_u_count, target_selection):
        cases = tuple(self.population.partition_cases('inner_train', ranking_only=True)) + self.validation_cases
        if target_selection is None:
            plans = [self.population.case(case, active_u_count) for case in cases]
            selection = None
            stage_key = active_u_count
        else:
            from .v23_targets import target_selection_manifest
            selection = target_selection_manifest(self.population, active_u_count, target_selection)
            plans = [self.population.case(case, active_u_count,
                active_u_indices=selection['selections'][case]) for case in cases]
            stage_key = active_u_count if selection['prefix_equivalent'] else (active_u_count, selection['sha256'])
            if selection['prefix_equivalent']:
                selection = None
        for plan in plans:
            self._plan_stages[self._plan_key(plan)] = stage_key
        return cases, plans, stage_key, selection

    def _stage_request(self, active_u_count, plans, selection):
        request = dict(format=FORMAT, active_U=active_u_count,
            population_sha256=self.population.manifest()['sha256'],
            query_hashes={plan.case_id: canonical_hash(plan.query_rows) for plan in plans},
            adaptation=self.adaptation, original_bank_fingerprint=self.bundle.prototype_bank.fingerprint(),
            model_graph_config=self.bundle.config['graph'], ct_clip=self.bundle.config['ct_clip'])
        if selection is not None:
            request['target_selection'] = selection
        return request

    def _payload_path(self, count, case, *, stage_key=None):
        return self._stage_path(count if stage_key is None else stage_key) / 'cases' / (
            hashlib.sha256(case.encode()).hexdigest() + '.pt')

    def _entry_path(self, count, case, *, stage_key=None):
        return self._payload_path(count, case, stage_key=stage_key).with_suffix('.json')

    def _validate_plan(self, plan):
        if hasattr(plan, 'active_u_indices'):
            expected = self.population.case(plan.case_id, plan.active_u_count,
                                            active_u_indices=plan.active_u_indices)
        else:
            expected = self.population.case(plan.case_id, plan.active_u_count)
        if plan.manifest() != expected.manifest() or plan.query_rows != expected.query_rows:
            raise ValueError('v2.3 upper query belongs to another population or changed candidate order')

    def _load(self, entry, plan, *, private=True):
        import torch
        self._rss()
        self._guard_sources()
        self._guard_inputs(plan)
        path = self._payload_path(plan.active_u_count, plan.case_id,
                                  stage_key=self._plan_stages.get(self._plan_key(plan), plan.active_u_count))
        binding = self._binding(plan)
        if entry['binding'] != binding:
            raise ValueError('Immutable v2.3 geometry payload/query/source identity differs')
        key = ('local', str(path))
        self._prove_file(path, entry['payload_sha256'])
        answer = self._recalled(key, binding)
        if answer is not None:
            result = self._private(answer) if private else answer
            self._guard_inputs(plan)
            self._guard_sources()
            self._prove_file(path, entry['payload_sha256'])
            return result
        began = time.perf_counter()
        if not self.memoize_cpu_geometry and sha(path) != entry['payload_sha256']:
            raise ValueError('Immutable v2.3 geometry payload identity differs')
        saved = torch.load(path, map_location='cpu', weights_only=False)
        self._count('payload_deserializations')
        self._count('deserialize_seconds', time.perf_counter() - began)
        if saved['format'] != FORMAT or saved['binding'] != entry['binding']:
            raise ValueError('Serialized v2.3 geometry receipt differs')
        graph, prototype, audit = saved['graph'], saved['prototype'], saved['audit']
        if (graph['candidate'].raw_x.shape[0] != len(plan.record_ids)
                or audit['candidate_count'] != len(plan.record_ids)
                or audit['P_U_labels_in_forward'] is not False
                or canonical_hash(audit) != entry['audit_sha256']):
            raise ValueError('v2.3 geometry candidate/annotation receipt differs')
        answer = graph, prototype, audit
        self._prove_file(path, entry['payload_sha256'])
        self._guard_inputs(plan)
        self._guard_sources()
        self._remember(key, answer, binding, {str(path): entry['payload_sha256']})
        # Without a resident owner, torch.load already supplies private tensors.
        return self._private(answer) if private and self.memoize_cpu_geometry else answer

    def _historical_context(self, plan, provider):
        """Recheck all in-memory contracts originally admitted by the sealed cache."""
        from .comparison_native_upper_cache import _query, _raw_bindings
        full = self._full_validation
        if _query(plan.query_rows) != _query(full.cohort['by_case'][plan.case_id]):
            raise ValueError('Historical validation complete query changed')
        return canonical_hash(dict(signature=full.signature, entry=full.entries[plan.case_id],
            query=plan.query_rows, debug=self.bundle.receipt['debug'],
            graph=self.bundle.config['graph'], ct_clip=self.bundle.config['ct_clip'],
            source_pad=self.bundle.config['cache']['source_pad'], scope=self.bundle.scope,
            source=str(Path(self.bundle.source).resolve()), baseline=str(Path(self.bundle.baseline).resolve()),
            bank_fingerprint=self.bundle.prototype_bank.fingerprint(),
            bank_training_cases=list(self.bundle.prototype_bank.training_case_ids),
            baseline_proof_sha256=canonical_hash(self.bundle.receipt['baseline_proof']),
            donor_max_diameter_mm=provider.ds.meta['config']['donor_max_diameter_mm'],
            raw_inputs=_raw_bindings(provider.ds.meta, full.cohort),
            training_cases=provider.ds.meta['split']['inner_train']))

    def _capture_historical_proofs(self):
        from . import comparison_native_upper_cache as historical
        full = self._full_validation
        if self._historical_proofs is not None:
            return
        signature = full.signature
        paths = {str(full.index_path): full.index_sha256,
                 str(full.inventory_path): signature['native_inventory_sha256'],
                 str(full.root / 'request.json'): full.index['request_sha256'],
                 signature['bank_path']: signature['bank_sha256'],
                 signature['native_region_config_path']: signature['native_region_config_sha256']}
        paths.update(full.index.get('preserved_files', {}))
        paths.update({str(historical.ROOT / name): checksum for name, checksum in
                      signature['execution_code_sha256'].items()})
        paths[str(historical.ROOT / 'versions/v1/pipeline_v1_source.zip')] = signature['source_proof']['archive_sha256']
        paths.update({str(Path(signature['source_proof']['source']) / name): checksum
                      for name, checksum in signature['source_proof']['verified_files'].items()})
        for raw in signature['raw_inputs'].values():
            paths.update({raw[name]: raw[name + '_sha256'] for name in ('image', 'label')})
        for region in full.index['region_bindings'].values():
            paths.update({str(Path(region['path']) / name): checksum for name, checksum in region['files'].items()})
        for path, checksum in paths.items():
            self._prove_file(path, checksum)
        full._guard_inputs()
        self._historical_proofs = paths
        self._historical_metadata = copy.deepcopy((full.signature, full.index, full.entries))

    def _guard_historical(self):
        if self._historical_proofs is None:
            return
        self._full_validation._guard_inputs()
        if (self._full_validation.signature, self._full_validation.index,
                self._full_validation.entries) != self._historical_metadata:
            raise ValueError('Admitted historical cache metadata/source binding changed')
        for path, checksum in self._historical_proofs.items():
            self._prove_file(path, checksum)

    def _historical(self, plan, provider, *, private=True):
        self._rss()
        self._guard_sources()
        self._guard_inputs(plan)
        self._guard_historical()
        binding = self._historical_context(plan, provider)
        full = self._full_validation
        from .comparison_native_upper_cache import _relative
        entry = full.entries[plan.case_id]
        path = _relative(full.root, entry['path'])
        key = ('historical', str(path))
        answer = self._recalled(key, binding)
        if answer is None:
            began = time.perf_counter()
            loaded = full.upper_graphs(self.bundle, provider, plan.query_rows,
                                      region_output=self.output / 'regions')
            self._count('historical_loads')
            self._count('deserialize_seconds', time.perf_counter() - began)
            self._capture_historical_proofs()
            self._prove_file(path, entry['payload_sha256'])
            # The historical loader returns mmap tensors. Give the memo an
            # ordinary owned CPU snapshot so file writes cannot alter it.
            answer = self._private(loaded) if self.memoize_cpu_geometry else loaded
            self._remember(key, answer, binding, {str(path): entry['payload_sha256']})
        else:
            self._count('historical_hits')
        result = self._private(answer) if private else answer
        if self._historical_context(plan, provider) != binding:
            raise ValueError('Historical validation input binding changed during read')
        self._prove_file(path, entry['payload_sha256'])
        self._guard_inputs(plan)
        self._guard_sources()
        self._guard_historical()
        return result

    def _stage_recalled(self, stage_key, request, plans):
        if not self.cache_stage_admission or stage_key not in self._admitted_stages:
            return None
        complete, admitted_request, guards = self._admitted_stages[stage_key]
        if request != admitted_request:
            raise ValueError('Admitted geometry phase request/query/bank changed')
        self._rss()
        self._guard_sources()
        self._guard_historical()
        for plan in plans:
            self._guard_inputs(plan)
            entry = self._entries[self._plan_key(plan)]
            if not entry.get('reuse_full_validation') and entry['binding'] != self._binding(plan):
                raise ValueError('Admitted geometry phase case binding changed')
        for path, checksum in guards.items():
            self._prove_file(path, checksum)
        self._count('stage_admission_hits')
        return copy.deepcopy(complete)

    def _stage_remember(self, stage_key, request, complete, plans):
        stage = self._stage_path(stage_key)
        guards = {str(stage / name): self._prove_file(stage / name)
                  for name in ('request.json', 'index.json')}
        for entry, plan in zip(complete['cases'], plans):
            if entry.get('reuse_full_validation'):
                full_entry = self._full_validation.entries[plan.case_id]
                path = self._full_root / full_entry['path']
                guards[str(path)] = self._prove_file(path, full_entry['payload_sha256'])
            else:
                self._prove_case_receipt(entry, plan)
                path = self._payload_path(plan.active_u_count, plan.case_id, stage_key=stage_key)
                guards[str(path)] = self._prove_file(path, entry['payload_sha256'])
                receipt_path = self._entry_path(plan.active_u_count, plan.case_id, stage_key=stage_key)
                guards[str(receipt_path)] = self._prove_file(receipt_path)
        self._stage_signatures[stage_key] = guards[str(stage / 'index.json')]
        if self.cache_stage_admission:
            self._admitted_stages[stage_key] = (copy.deepcopy(complete), copy.deepcopy(request), guards)

    def _build(self, plan, provider):
        import torch
        self._rss()
        self._guard_inputs(plan)
        stage_key = self._plan_stages.get(self._plan_key(plan), plan.active_u_count)
        entry_path = self._entry_path(plan.active_u_count, plan.case_id, stage_key=stage_key)
        if entry_path.exists():
            entry = json.loads(entry_path.read_text())
            self._load(entry, plan, private=False)
            return entry
        path = self._payload_path(plan.active_u_count, plan.case_id, stage_key=stage_key)
        if path.exists():
            # A completed no-overwrite payload may precede its receipt at an interruption.
            # Admit only its complete original query/bank/source binding; never replace it.
            saved = torch.load(path, map_location='cpu', weights_only=False)
            if saved.get('format') != FORMAT or saved.get('binding') != self._binding(plan):
                raise ValueError('Unpublished v2.3 geometry payload differs; previous evidence preserved')
            entry = dict(case_id=plan.case_id, active_U=plan.active_u_count,
                binding=self._binding(plan), payload_sha256=sha(path), payload_bytes=path.stat().st_size,
                audit_sha256=canonical_hash(saved['audit']))
            self._load(entry, plan, private=False)
            _new_json(entry_path, entry)
            return entry
        graph, prototype, audit = self._upper(self.bundle, provider, plan.query_rows,
                                             region_output=self.output / 'regions')
        audit = copy.deepcopy(audit)
        audit.update(v23_admission=self.adaptation,
            v23_partition='inner_train' if plan.case_id in self.train_cases else 'inner_val',
            all_observed_P_in_case=True, active_U=plan.active_u_count,
            native_donor_fixed=True, P_as_negative=False)
        binding = self._binding(plan)
        payload = dict(format=FORMAT, graph=graph, prototype=prototype,
                       audit=audit, binding=binding)
        path.parent.mkdir(parents=True, exist_ok=True)
        # A partial payload is a separate attempt, never a replacement of old evidence.
        temporary = path.with_name(path.name + '.partial-' + uuid.uuid4().hex)
        torch.save(payload, temporary)
        with temporary.open('r+b') as saved:
            os.fsync(saved.fileno())
        _publish_new(temporary, path)
        entry = dict(case_id=plan.case_id, active_U=plan.active_u_count,
            binding=binding, payload_sha256=sha(path), payload_bytes=path.stat().st_size,
            audit_sha256=canonical_hash(audit))
        self._load(entry, plan, private=False)
        _new_json(entry_path, entry)
        self._rss()
        self._guard_inputs(plan)
        return entry

    def prepare(self, active_u_count, *, target_selection=None):
        """Prepare every defined training loss case and every held-out case."""
        self.population.case(self.validation_cases[0], active_u_count)
        self._prepare_regions()
        cases, plans, stage_key, selection = self._plans_for_stage(active_u_count, target_selection)
        request = self._stage_request(active_u_count, plans, selection)
        recalled = self._stage_recalled(stage_key, request, plans)
        if recalled is not None:
            return recalled
        stage = self._stage_path(stage_key)
        request_path = stage / 'request.json'
        if request_path.exists():
            if json.loads(request_path.read_text()) != request:
                raise ValueError('Existing v2.3 geometry stage request changed')
        else:
            _new_json(request_path, request)
        provider = SimpleNamespace(ds=SimpleNamespace(meta=self.population.meta))
        prepared = []
        pending = []
        for plan in plans:
            if active_u_count == 128 and plan.case_id in self.validation_cases and self._full_validation is not None:
                self._historical(plan, provider, private=False)
                self._entries[self._plan_key(plan)] = dict(reuse_full_validation=True)
                self.receipt['full_validation_historical_cache_reused'] = True
                prepared.append(dict(case_id=plan.case_id, reuse_full_validation=True))
            else:
                pending.append(plan)
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = {pool.submit(self._build, plan, provider): plan for plan in pending}
            for future in as_completed(futures):
                plan = futures[future]
                entry = future.result()
                self._entries[self._plan_key(plan)] = entry
                prepared.append(entry)
                self._rss()
        prepared.sort(key=lambda item: cases.index(item['case_id']))
        completed = dict(request_sha256=canonical_hash(request), complete=True,
            active_U=active_u_count, cases=prepared, case_count=len(cases),
            all_P_retained=True, geometry_only=True, hidden_subset=False,
            zero_P_train_cases_no_defined_ranking_loss=self.population.manifest()['zero_P_train_cases'],
            zero_P_validation_cases_scored=True, CPU_workers=self.workers)
        index_path = stage / 'index.json'
        if index_path.exists():
            if json.loads(index_path.read_text()) != completed:
                raise ValueError('Existing complete v2.3 geometry stage index changed')
        else:
            _new_json(index_path, completed)
        self._stage_remember(stage_key, request, completed, plans)
        if active_u_count not in self.receipt['stages']:
            self.receipt['stages'].append(active_u_count)
        if selection is not None:
            targets = self.receipt.setdefault('target_selection_stages', [])
            if selection['sha256'] not in targets:
                targets.append(selection['sha256'])
        return copy.deepcopy(completed)

    def get(self, plan, provider):
        self._validate_plan(plan)
        key = self._plan_key(plan)
        if key not in self._entries:
            raise ValueError('Complete joint v2.3 upper geometry stage must be prepared first')
        stage_key = self._plan_stages.get(key, plan.active_u_count)
        index_path = self._stage_path(stage_key) / 'index.json'
        self._prove_file(index_path, self._stage_signatures[stage_key])
        self._prove_file(self._stage_path(stage_key) / 'request.json')
        entry = self._entries[key]
        if entry.get('reuse_full_validation'):
            answer = self._historical(plan, provider)
        else:
            self._prove_case_receipt(entry, plan)
            answer = self._load(entry, plan)
        self.receipt['cache_hits'] += 1
        self._prove_file(index_path, self._stage_signatures[stage_key])
        self._prove_file(self._stage_path(stage_key) / 'request.json')
        return answer

    __call__ = get

    def admit(self, active_u_count, *, target_selection=None):
        """Read-only admission for other DDP ranks after rank-zero publication."""
        self._prepare_regions(readonly=True)
        cases, plans, stage_key, selection = self._plans_for_stage(active_u_count, target_selection)
        stage = self._stage_path(stage_key)
        expected_request = self._stage_request(active_u_count, plans, selection)
        recalled = self._stage_recalled(stage_key, expected_request, plans)
        if recalled is not None:
            self.receipt['read_only_DDP_stage_admission'] = True
            return recalled
        request_path, index_path = stage / 'request.json', stage / 'index.json'
        if not request_path.is_file() or not index_path.is_file():
            raise ValueError('Complete rank-zero geometry publication required before DDP admission')
        self._prove_file(request_path)
        self._prove_file(index_path)
        request, complete = json.loads(request_path.read_text()), json.loads(index_path.read_text())
        if (request != expected_request
                or complete.get('request_sha256') != canonical_hash(request)
                or complete.get('complete') is not True or complete.get('active_U') != active_u_count
                or [entry.get('case_id') for entry in complete.get('cases', [])] != list(cases)):
            raise ValueError('Published v2.3 DDP geometry request/population/query/bank differs')
        for entry, plan in zip(complete['cases'], plans):
            if entry.get('reuse_full_validation'):
                if active_u_count != 128 or plan.case_id not in self.validation_cases or self._full_validation is None:
                    raise ValueError('Exact historical validation cache missing on this DDP rank')
                provider = SimpleNamespace(ds=SimpleNamespace(meta=self.population.meta))
                self._historical(plan, provider, private=False)
            else:
                self._load(entry, plan, private=False)
            self._entries[self._plan_key(plan)] = copy.deepcopy(entry)
        self._stage_remember(stage_key, request, complete, plans)
        if active_u_count not in self.receipt['stages']:
            self.receipt['stages'].append(active_u_count)
        self.receipt['read_only_DDP_stage_admission'] = True
        return copy.deepcopy(complete)

    def finish(self):
        """Verify all admitted original input bytes before final publication."""
        verified = {}
        self._guard_sources()
        self._guard_historical()
        for path, (_, checksum) in self._file_proofs.items():
            self._prove_file(path, checksum)
        for case, binding in self._region_bindings.items():
            if _regions(binding['path']) != binding['files']:
                raise ValueError('Original v2.3 region content changed: ' + case)
            for key in ('image', 'label'):
                path = self._raw[case][key]
                actual = sha(path)
                if actual != self._raw[case][key + '_sha256']:
                    raise ValueError('Original v2.3 CT/annotation SHA256 changed: ' + case)
                verified[path] = actual
        self.receipt['final_full_input_SHA256_verified'] = True
        self.receipt['final_verified_raw_files'] = len(verified)
        return copy.deepcopy(self.receipt)
