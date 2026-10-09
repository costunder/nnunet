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
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
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
                 full_validation_cache=None):
        import psutil
        if (type(workers) is not int or workers < 2
                or workers > len(psutil.Process().cpu_affinity())
                or type(rss_bytes) is not int or rss_bytes <= 0):
            raise ValueError('Explicit parallel CPU workers and positive RSS budget required')
        self.bundle, self.population = bundle, population
        self.output = Path(output).resolve()
        self.workers, self.rss_bytes = workers, rss_bytes
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

    def _rss(self):
        import psutil
        if psutil.Process().memory_info().rss > self.rss_bytes:
            raise MemoryError('v2.3 geometry RSS budget exceeded; no case/graph reduction')

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
        for case in (plan.case_id, plan.donor_case_id):
            region = self._region_bindings[case]
            paths = [str(Path(region['path']) / name) for name in region['files']]
            paths += [self._raw[case][key] for key in ('image', 'label')]
            if any(self._stat(path) != self._input_stats[str(path)] for path in paths):
                raise ValueError('Admitted v2.3 CT/annotation/region input changed: ' + case)

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

    def _load(self, entry, plan):
        import torch
        self._guard_inputs(plan)
        path = self._payload_path(plan.active_u_count, plan.case_id,
                                  stage_key=self._plan_stages.get(self._plan_key(plan), plan.active_u_count))
        if (path.is_symlink() or sha(path) != entry['payload_sha256']
                or entry['binding'] != self._binding(plan)):
            raise ValueError('Immutable v2.3 geometry payload/query/source identity differs')
        saved = torch.load(path, map_location='cpu', weights_only=False)
        if saved['format'] != FORMAT or saved['binding'] != entry['binding']:
            raise ValueError('Serialized v2.3 geometry receipt differs')
        graph, prototype, audit = saved['graph'], saved['prototype'], saved['audit']
        if (graph['candidate'].raw_x.shape[0] != len(plan.record_ids)
                or audit['candidate_count'] != len(plan.record_ids)
                or audit['P_U_labels_in_forward'] is not False
                or canonical_hash(audit) != entry['audit_sha256']):
            raise ValueError('v2.3 geometry candidate/annotation receipt differs')
        return graph, prototype, audit

    def _build(self, plan, provider):
        import torch
        self._rss()
        self._guard_inputs(plan)
        stage_key = self._plan_stages.get(self._plan_key(plan), plan.active_u_count)
        entry_path = self._entry_path(plan.active_u_count, plan.case_id, stage_key=stage_key)
        if entry_path.exists():
            entry = json.loads(entry_path.read_text())
            self._load(entry, plan)
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
            self._load(entry, plan)
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
        self._load(entry, plan)
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
                self._full_validation.upper_graphs(self.bundle, provider, plan.query_rows,
                                                   region_output=self.output / 'regions')
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
        self._stage_signatures[stage_key] = sha(index_path)
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
        if sha(index_path) != self._stage_signatures[stage_key]:
            raise ValueError('Complete v2.3 geometry stage index changed')
        entry = self._entries[key]
        if entry.get('reuse_full_validation'):
            answer = self._full_validation.upper_graphs(self.bundle, provider, plan.query_rows,
                                                        region_output=self.output / 'regions')
        else:
            answer = self._load(entry, plan)
        self.receipt['cache_hits'] += 1
        return answer

    __call__ = get

    def admit(self, active_u_count, *, target_selection=None):
        """Read-only admission for other DDP ranks after rank-zero publication."""
        self._prepare_regions(readonly=True)
        cases, plans, stage_key, selection = self._plans_for_stage(active_u_count, target_selection)
        stage = self._stage_path(stage_key)
        request_path, index_path = stage / 'request.json', stage / 'index.json'
        if not request_path.is_file() or not index_path.is_file():
            raise ValueError('Complete rank-zero geometry publication required before DDP admission')
        request, complete = json.loads(request_path.read_text()), json.loads(index_path.read_text())
        if (request != self._stage_request(active_u_count, plans, selection)
                or complete.get('request_sha256') != canonical_hash(request)
                or complete.get('complete') is not True or complete.get('active_U') != active_u_count
                or [entry.get('case_id') for entry in complete.get('cases', [])] != list(cases)):
            raise ValueError('Published v2.3 DDP geometry request/population/query/bank differs')
        for entry, plan in zip(complete['cases'], plans):
            if entry.get('reuse_full_validation'):
                if active_u_count != 128 or plan.case_id not in self.validation_cases or self._full_validation is None:
                    raise ValueError('Exact historical validation cache missing on this DDP rank')
            else:
                self._load(entry, plan)
            self._entries[self._plan_key(plan)] = copy.deepcopy(entry)
        self._stage_signatures[stage_key] = sha(index_path)
        if active_u_count not in self.receipt['stages']:
            self.receipt['stages'].append(active_u_count)
        self.receipt['read_only_DDP_stage_admission'] = True
        return copy.deepcopy(complete)

    def finish(self):
        """Verify all admitted original input bytes before final publication."""
        verified = {}
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
