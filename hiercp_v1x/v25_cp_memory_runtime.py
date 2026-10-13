"""Execution-only CP workspace reclamation; sealed scientific files stay intact.

The AST whitelist only releases finished local references and installs a budget
facade that reclaims evictable caches before the unchanged hard-budget check.
It changes no candidate, graph, tensor equation, model, RNG draw or score receipt.
Install once in a fresh B scoring process, after its authorized private CP clone.
"""
from __future__ import annotations

import ast
import copy
import gc
import hashlib
import inspect
from pathlib import Path
import threading
import textwrap
import time
import types

import psutil

from .host_memory import _trim_heap

FORMAT = 'v25_CP_finished_workspace_reference_reclamation_execution_v1'
_SOURCE = Path(__file__).resolve()
_INSTALLED = {}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _dump(tree):
    return ast.dump(tree, include_attributes=False)


def _marked(statement):
    statement._lifetime_added = True
    return statement


def _addition(text):
    return [_marked(node) for node in ast.parse(text).body]


def _normal(tree):
    class Restore(ast.NodeTransformer):
        def visit(self, node):
            if getattr(node, '_lifetime_added', False):
                return None
            if hasattr(node, '_lifetime_original'):
                return copy.deepcopy(node._lifetime_original)
            return super().visit(node)
    return Restore().visit(copy.deepcopy(tree))


def _code_shape(code):
    # Filenames/line tables differ for an execution adapter; instructions,
    # constants, nested code and all argument/local/global names must match.
    return (code.co_code, code.co_names, code.co_varnames, code.co_freevars,
        code.co_cellvars, code.co_argcount, code.co_posonlyargcount,
        code.co_kwonlyargcount, code.co_flags,
        tuple(_code_shape(value) if isinstance(value, types.CodeType)
            else (type(value).__name__, repr(value)) for value in code.co_consts))


def _compile(tree, function, namespace):
    target = dict(namespace)
    exec(compile(ast.fix_missing_locations(copy.deepcopy(tree)),
        inspect.getsourcefile(function)+':CP_memory_lifetime', 'exec'), target)
    return target[function.__name__]


def _transform_cp(function, *, authorized_B=False, memory=True):
    original = ast.parse(textwrap.dedent(inspect.getsource(function)))
    tree = copy.deepcopy(original)
    counts = {}
    def hit(name):
        counts[name] = counts.get(name, 0)+1
    class Lifetimes(ast.NodeTransformer):
        def visit_ImportFrom(self, node):
            if authorized_B and node.level == 1 and node.module == 'v25_server_resources':
                if [(item.name, item.asname) for item in node.names] != [('AllocationLease', None)]:
                    raise ValueError('Frozen CP resource import changed')
                replacement = ast.ImportFrom(module='v25_B_early_downstream_resources', level=1,
                    names=[ast.alias(name='BEarlyDownstreamAllocationLease', asname='AllocationLease')])
                replacement._lifetime_original = copy.deepcopy(node)
                hit('existing_authorized_B_lease')
                return ast.copy_location(replacement, node)
            return node
        def visit_Assign(self, node):
            node = self.generic_visit(node)
            if not memory:
                return node
            target = _dump(node.targets[0]) if len(node.targets) == 1 else ''
            if target == _dump(ast.parse('_, (_, released) = collections.popitem(last=False)').body[0].targets[0]):
                if _dump(node.value) != _dump(ast.parse('collections.popitem(last=False)', mode='eval').body):
                    raise ValueError('Frozen CP collection eviction changed')
                replacement = ast.parse('evicted_case, (evicted_collection, released) = collections.popitem(last=False)').body[0]
                replacement._lifetime_original = copy.deepcopy(node)
                hit('release_evicted_collection')
                return [ast.copy_location(replacement, node), *_addition('del evicted_collection')]
            if target == _dump(ast.parse('collection_resident_bytes = 8 * 2**30').body[0].targets[0]):
                if _dump(node.value) != _dump(ast.parse('8 * 2**30', mode='eval').body):
                    raise ValueError('Frozen CP eight-GiB collection budget changed')
                hit('budget_facade')
                return [node, *_addition('''
def release_collections():
    nonlocal collection_bytes
    with cache_lock:
        released = collection_bytes
        collections.clear()
        collection_bytes = 0
        return released
budget = _CP_LIFETIME_BUDGET(budget, inputs, release_collections)
''')]
            if target == _dump(ast.parse('source, diameter = collection[found[0]]').body[0].targets[0]):
                if _dump(node.value) != _dump(ast.parse('collection[found[0]]', mode='eval').body):
                    raise ValueError('Frozen CP complete selected source changed')
                hit('release_selected_collection_local')
                return [node, *_addition('del collection')]
            return node
        def visit_For(self, node):
            node = self.generic_visit(node)
            if not memory:
                return node
            if _dump(node.target) == _dump(ast.parse('for plan, item, graphs in prepared_batch: pass').body[0].target):
                if _dump(node.iter) != _dump(ast.parse('prepared_batch', mode='eval').body):
                    raise ValueError('Frozen CP source-batch activation changed')
                hit('release_batch_loop_aliases')
                return [node, *_addition('del item, graphs')]
            return node
        def visit_With(self, node):
            node = self.generic_visit(node)
            if not memory:
                return node
            # Only the metadata scan's complete source payload survives its
            # mapping retention assignment. Publication needs its payload until
            # cp_score_row has copied only plain receipt fields.
            return node
        def visit_Expr(self, node):
            node = self.generic_visit(node)
            if memory and _dump(node.value) == _dump(ast.parse('store.save_row(row)', mode='eval').body):
                hit('release_publication_payload')
                return [node, *_addition('del payload, loaded, score')]
            return node
    tree = Lifetimes().visit(tree)
    if memory:
        # The metadata assignment is intentionally outside the model path.
        class MetadataRelease(ast.NodeTransformer):
            def visit_Assign(self, node):
                node = self.generic_visit(node)
                expected = ast.parse("plans[entry], payloads[entry] = plan, {key: payload[key] for key in ('candidate_raw_centers', 'candidate_centers', 'source_diameter_mm')}").body[0]
                if _dump(node) == _dump(expected):
                    hit('release_metadata_payload')
                    return [node, *_addition('del payload, loaded')]
                return node
        tree = MetadataRelease().visit(tree)
    expected = dict(existing_authorized_B_lease=1) if authorized_B else {}
    if memory:
        expected.update(release_evicted_collection=1, budget_facade=1,
            release_selected_collection_local=1, release_batch_loop_aliases=1,
            release_publication_payload=1, release_metadata_payload=1)
    if counts != expected or _dump(_normal(tree)) != _dump(original):
        raise ValueError('Frozen CP lifetime AST whitelist differs: '+repr(counts))
    return tree, counts


def _transform_case(function):
    original = ast.parse(textwrap.dedent(inspect.getsource(function)));tree = copy.deepcopy(original)
    counts = {}
    class Lifetimes(ast.NodeTransformer):
        def visit_Assign(self, node):
            node = self.generic_visit(node)
            matches = {
                'context=recipient_context(case,loaded.image,organ,loaded.spacing,loaded.image_affine)':
                    ('release_snapshot_inputs', 'del organ\nloaded.image = context.image'),
                'old,(unused,bytes_)=self._raw_cache.popitem(last=False)':
                    ('release_evicted_raw_alias', 'del unused')}
            for text, (name, release) in matches.items():
                if _dump(node) == _dump(ast.parse(text).body[0]):
                    counts[name] = counts.get(name, 0)+1
                    return [node, *_addition(release)]
            return node
    tree = Lifetimes().visit(tree)
    if counts != dict(release_snapshot_inputs=1, release_evicted_raw_alias=1) or _dump(_normal(tree)) != _dump(original):
        raise ValueError('Frozen native case lifetime AST whitelist differs')
    return tree, counts


class _LifetimeBudget:
    """Same hard budget, with measured inactive-cache reclamation beforehand."""
    def __init__(self, budget, inputs, release_collections):
        self._budget, self._inputs, self._release_collections = budget, inputs, release_collections
        self._lock = threading.RLock()
        self.profile = dict(checks=0, pressure_events=0, collection_bytes_released=0,
            maximum_observed_RSS_bytes=0, last_reclamation=None)
        if budget.rss_bytes != inputs.rss_bytes or inputs.rss_bytes != 64*2**30:
            raise ValueError('Execution reclamation requires the unchanged 64GiB RSS budget')
        inputs._CP_memory_lifetime_profile = self.profile

    def __getattr__(self, name):
        return getattr(self._budget, name)

    def check(self):
        from .v25_upper_memory_runtime import _pressure_trigger
        with self._lock:
            self.profile['checks'] += 1
            before = int(psutil.Process().memory_info().rss)
            self.profile['maximum_observed_RSS_bytes'] = max(self.profile['maximum_observed_RSS_bytes'], before)
            trigger = _pressure_trigger(self._inputs, getattr(self._inputs, 'memory_coordinator', None))
            if before > trigger:
                began = time.perf_counter()
                self.profile['pressure_events'] += 1
                released = self._release_collections()
                self.profile['collection_bytes_released'] += released
                # The callback frame has returned; popped component maps have
                # no cache/loop reference. Active builders retain their exact
                # private immutable inputs. Existing input/coordinator policy
                # retains its own strict hard limit and protected owners.
                collected = gc.collect();heap = _trim_heap()
                self._inputs._rss()
                after = int(psutil.Process().memory_info().rss)
                self.profile['last_reclamation'] = dict(before_RSS_bytes=before,
                    after_RSS_bytes=after, released_collection_logical_bytes=released,
                    gc_collected=collected, heap_trim=heap,
                    seconds=time.perf_counter()-began, active_input_values_changed=False)
                print('V2.5 CP RSS RECLAIM before_bytes=%d after_bytes=%d limit_bytes=%d '
                    'collection_logical_bytes_released=%d' % (before, after,
                    self.rss_bytes, released), flush=True)
            return self._budget.check()


def install_runtime(*, cp_module, expected_frozen_source_sha256=None):
    """Install only in a new execution process; never mutate scientific files."""
    from . import v25_cp as frozen_cp, v25_factory as frozen_factory, v25_preparation
    paths = {'v25_cp.py': Path(frozen_cp.__file__), 'v25_factory.py': Path(frozen_factory.__file__)}
    sources = {name: _sha(path) for name, path in paths.items()}
    if expected_frozen_source_sha256 is not None and sources != expected_frozen_source_sha256:
        raise ValueError('Exact frozen CP/factory source SHA differs')
    key = id(cp_module)
    if key in _INSTALLED:
        installed, case_installed, proof = _INSTALLED[key]
        if cp_module.score_v25_cp_bank is not installed or v25_preparation.AcceleratedNativeInputs._case is not case_installed:
            raise ValueError('Unknown concurrent CP memory execution override')
        return copy.deepcopy(proof)
    authorized = cp_module.__name__ == frozen_cp.__name__+'_private_authorized_B'
    if cp_module is not frozen_cp and not authorized:
        raise ValueError('Only exact frozen or authorized private B CP namespace admitted')
    original_function = frozen_cp.score_v25_cp_bank
    baseline_tree, _ = _transform_cp(original_function, authorized_B=authorized, memory=False)
    baseline = _compile(baseline_tree, original_function, cp_module.__dict__)
    if _code_shape(cp_module.score_v25_cp_bank.__code__) != _code_shape(baseline.__code__):
        raise ValueError('Unknown existing CP scorer equation/code override')
    tree, counts = _transform_cp(original_function, authorized_B=authorized)
    replacement = _compile(tree, original_function, dict(cp_module.__dict__, _CP_LIFETIME_BUDGET=_LifetimeBudget))
    case_original = frozen_factory.V25NativeInputs._case
    cls = v25_preparation.AcceleratedNativeInputs
    if cls._case is not case_original:
        raise ValueError('Unknown native case execution override')
    case_tree, case_counts = _transform_case(case_original)
    case_replacement = _compile(case_tree, case_original, case_original.__globals__)
    proof = dict(format=FORMAT, runtime_source_sha256=_sha(_SOURCE),
        frozen_source_sha256=sources,
        CP_normalized_original_AST_sha256=hashlib.sha256(_dump(_normal(tree)).encode()).hexdigest(),
        CP_transformed_AST_sha256=hashlib.sha256(_dump(tree).encode()).hexdigest(),
        case_normalized_original_AST_sha256=hashlib.sha256(_dump(_normal(case_tree)).encode()).hexdigest(),
        case_transformed_AST_sha256=hashlib.sha256(_dump(case_tree).encode()).hexdigest(),
        CP_whitelist_counts=counts, case_whitelist_counts=case_counts,
        original_score_declaration_and_source_SHA_unchanged=True,
        active_input_graph_model_RNG_equations_unchanged=True,
        physical_patient_batch=4, physical_candidate_batch_unchanged=True,
        CPU_workers=4, RSS_limit_bytes=64*2**30, collection_resident_limit_bytes=8*2**30,
        original_pressure_trigger_and_hard_budget_check_preserved=True,
        scientific_file_bytes_modified=False, running_processes_hot_swapped=False)
    if sources != {name: _sha(path) for name, path in paths.items()}:
        raise ValueError('Frozen source changed during lifetime admission')
    cp_module.score_v25_cp_bank = replacement
    cls._case = case_replacement
    _INSTALLED[key] = replacement, case_replacement, proof
    return copy.deepcopy(proof)


def receipt(inputs):
    return dict(format=FORMAT, runtime_source_sha256=_sha(_SOURCE),
        profile=copy.deepcopy(getattr(inputs, '_CP_memory_lifetime_profile', None)),
        actual_RSS_bytes=int(psutil.Process().memory_info().rss),
        throughput_or_memory_improvement_verified=False)
