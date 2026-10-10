"""CPU-only exact stage graphs from admitted immutable full-128 geometry.

No neural result is cached. Candidate rows retain the population's original
order, and all candidate-dependent relations are rebuilt by the sealed original
functions. The existing geometry cache owns validation and exclusive publication.
"""
from __future__ import annotations

import ast
import copy
from dataclasses import dataclass
import hashlib
import inspect
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import numpy as np
import torch

from .contracts import canonical_hash
from .v24_inputs import query_inputs, tensor_digest

FORMAT = 'v24_exact_full128_upper_semantic_reuse_v1'
_SCIENCE = ('v24_factory.py', 'v24_geometry.py', 'v24_inputs.py',
            'v24_model.py', 'v24_provider.py')
_NODE_FIELDS = ('raw_x', 'pos', 'region_index', 'meta')


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def runtime_contract():
    """Pure source receipt; importing this module never activates original code."""
    root = Path(__file__).resolve().parent
    return dict(format=FORMAT, runtime_source_sha256=_sha(__file__),
        scientific_source_sha256={name: _sha(root/name) for name in _SCIENCE},
        source_geometry='existing admitted all-P+128U only',
        selection='exact canonical record-ID map; original inventory order',
        candidate_KNN='recomputed by original _knn on the complete active set',
        prototype='original build_prototype_graph with exact stored float32 descriptors',
        full_CT_or_EDT_recomputed=False, neural_features_or_predictions_cached=False,
        old_cold_builder_fallback=False, existing_cache_publication_unchanged=True,
        scientific_source_files_modified=False, running_training_hot_swap=False)


@dataclass(frozen=True)
class _RegionAssignment:
    # The original prototype builder reads exactly this member of each spec.
    region_id: int


def _helper_contract(hierarchy):
    function = hierarchy.build_prototype_graph
    tree = ast.parse(inspect.getsource(function))
    fields = {name: {node.attr for node in ast.walk(tree)
                    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                    and node.value.id == name} for name in ('spec', 'regions')}
    if fields != dict(spec={'region_id'}, regions={'region_features', 'region_positions'}):
        raise ValueError('Original prototype builder requires different semantic inputs')
    names = ('_knn', '_full_bipartite', '_set_patient_relation', 'build_prototype_graph')
    return dict(original_helper_source_sha256={name: hashlib.sha256(
        inspect.getsource(getattr(hierarchy, name)).encode()).hexdigest() for name in names},
        prototype_spec_fields=['region_id'],
        prototype_region_fields=['region_features', 'region_positions'])


class UpperReuseAdapter:
    """Separate CPU-preparation instance; original cache methods stay untouched."""
    def __init__(self, inputs):
        if torch.cuda.is_initialized():
            raise RuntimeError('Upper reuse binding requires a separate CPU-only preparation process')
        from .v24_geometry import V24UpperGeometryCache
        from hiercp import hierarchy, schema
        cache = inputs.geometry
        if (not isinstance(cache, V24UpperGeometryCache)
                or cache.population is not inputs.population
                or inputs.runtime['workers'] != 4 or cache.workers != 4
                or inputs.rss_bytes != 64*2**30 or cache.rss_bytes != inputs.rss_bytes
                or inputs.raw_resident_bytes != 8*2**30
                or not callable(inputs.guard_source)):
            raise ValueError('Original four-worker/64GiB RSS/8GiB raw bound CPU inputs required')
        source = Path(inputs.bundle.source).resolve(strict=True)
        implementation = Path(hierarchy.__file__).resolve(strict=True)
        if not implementation.is_relative_to(source):
            raise ValueError('Original hierarchy is outside the admitted sealed snapshot')
        self.inputs, self.cache, self.h, self.schema = inputs, cache, hierarchy, schema
        self.contract = dict(runtime_contract(), **_helper_contract(hierarchy))
        self._source_guard = {}
        for path in (Path(__file__).resolve(), implementation,
                     Path(schema.__file__).resolve(strict=True)):
            proof = cache._file_identity(path)
            self._source_guard[str(path)] = proof
            inputs._input_file_proofs[str(path)] = proof
            inputs._file_proofs[str(path)] = proof
        self._lock = threading.Lock()
        self._source_proofs = {}
        self._rows = []

    def _guard(self):
        for path, proof in self._source_guard.items():
            if self.cache._file_identity(Path(path)) != proof:
                raise ValueError('Admitted upper reuse implementation changed')
        self.inputs.guard_source()
        self.cache._guard()

    def _selection(self, plan):
        population = self.inputs.population
        full = population.case(plan.case_id, 128)
        selected = population.case(plan.case_id, plan.active_u_count,
            active_u_indices=plan.active_u_indices)
        ids = tuple(plan.record_ids)
        queries = query_inputs(plan.query_rows)
        if (ids != tuple(selected.record_ids) or queries != query_inputs(selected.query_rows)
                or len(ids) != len(set(ids))):
            raise ValueError('Stage plan differs from the complete canonical population')
        full_ids = tuple(full.record_ids)
        if len(full_ids) != len(set(full_ids)):
            raise ValueError('Full128 population contains duplicate observation IDs')
        lookup = {record: index for index, record in enumerate(full_ids)}
        if any(record not in lookup for record in ids):
            raise ValueError('Active observation is missing from the actual full128 population')
        indices = tuple(lookup[record] for record in ids)
        full_queries = query_inputs(full.query_rows)
        if queries != tuple(full_queries[index] for index in indices):
            raise ValueError('Actual candidate center or frozen donor differs from full128')
        return full, indices, queries

    def _load_full(self, full):
        path, key, binding = self.cache._path(full)
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError('Admitted full128 upper geometry is required; no cold fallback: '+str(path))
        graph, prototype, audit, proof = self.cache._load(path, full, key, binding)
        self.cache.guard_inputs(full, binding)
        if self.cache._file_identity(path) != proof:
            raise ValueError('Full128 geometry changed after admission')
        with self._lock:
            old = self._source_proofs.get(str(path))
            if old is not None and old['stat_identity'] != list(proof):
                raise ValueError('Previously admitted full128 source file was replaced')
            if old is None:
                checksum = _sha(path)
                if self.cache._file_identity(path) != proof:
                    raise ValueError('Full128 source changed during file SHA verification')
                old = dict(path=str(path), raw_file_sha256=checksum, stat_identity=list(proof),
                    tensor_sha256=tensor_digest((graph.to_dict(), prototype.to_dict())),
                    query_sha256=key)
                self._source_proofs[str(path)] = old
        if (audit['query_inputs_sha256'] != tensor_digest(query_inputs(full.query_rows))
                or audit['recipient_binding'] != binding['recipient']):
            raise ValueError('Full128 graph audit differs from its actual query/input binding')
        return graph, prototype, audit, copy.deepcopy(old)

    def _rebuild(self, full_graph, full_prototype, indices):
        h, schema = self.h, self.schema
        expected_nodes = ('tumor', 'candidate', 'region', 'liver')
        retained = tuple(edge for edge in schema.PATIENT_EDGE_TYPES
                         if 'lesion' not in (edge[0], edge[2]))
        if (tuple(full_graph.node_types) != expected_nodes
                or tuple(full_graph.edge_types) != retained):
            raise ValueError('Full128 graph node/relation ordering differs from original GT-blind hierarchy')
        for name in expected_nodes:
            if tuple(full_graph[name].keys()) != _NODE_FIELDS:
                raise ValueError('Original raw node fields changed: '+name)
            for value in full_graph[name].values():
                if not isinstance(value, torch.Tensor) or value.device.type != 'cpu':
                    raise ValueError('Admitted full128 node values must be actual CPU tensors')
        graph = full_graph.clone()
        selected = torch.tensor(indices, dtype=torch.long)
        for name in _NODE_FIELDS:
            graph['candidate'][name] = full_graph['candidate'][name].index_select(0, selected)
        for edge in tuple(graph.edge_types):
            del graph[edge]
        candidate_positions = graph['candidate'].pos.numpy()
        candidate_regions = graph['candidate'].region_index.numpy()
        n, r = len(indices), int(graph['region'].num_nodes)
        if candidate_regions.dtype != np.int64 or np.any(candidate_regions < 0) or np.any(candidate_regions >= r):
            raise ValueError('Actual candidate region assignments are invalid')
        regions = SimpleNamespace(region_features=graph['region'].raw_x.numpy(),
                                  region_positions=graph['region'].pos.numpy())
        if (regions.region_features.dtype != np.float32 or regions.region_positions.dtype != np.float32
                or candidate_positions.dtype != np.float32):
            raise ValueError('Original complete float32 descriptors are required without recasting')
        # The bank and every prototype equation are admitted against the existing
        # full prototype before they are reused for a stage-specific graph.
        full_specs = [_RegionAssignment(int(region))
                      for region in full_graph['candidate'].region_index.tolist()]
        verified = h.build_prototype_graph(full_specs, full_graph, regions,
                                          self.inputs.bundle.prototype_bank, config=self.inputs.graph_config)
        verified.v24_recipient_GT_free = True
        if tensor_digest(verified.to_dict()) != tensor_digest(full_prototype.to_dict()):
            raise ValueError('Original full128 prototype equations/bank do not reproduce admitted tensors')
        tc, tr = h._full_bipartite(1, n), h._full_bipartite(1, r)
        cr = np.stack([np.arange(n, dtype=np.int64), candidate_regions])
        rl = h._full_bipartite(r, 1)
        relations = {
            ('tumor','compatible_with','candidate'): tc,
            ('candidate','matched_to','tumor'): tc[[1,0]],
            ('candidate','spatial_neighbor','candidate'): h._knn(candidate_positions, self.inputs.graph_config.candidate_k),
            ('candidate','belongs_to','region'): cr,
            ('region','contains_candidate','candidate'): cr[[1,0]],
            ('tumor','conditions','region'): tr,
            ('region','context_for','tumor'): tr[[1,0]],
            ('region','adjacent_to','region'): full_graph[('region','adjacent_to','region')].edge_index.numpy(),
            ('region','inside','liver'): rl,
            ('liver','contains','region'): rl[[1,0]],
        }
        if set(relations) != set(retained):
            raise ValueError('Every original non-lesion patient relation is required')
        for edge in retained:
            h._set_patient_relation(graph, edge, relations[edge])
        specs = [_RegionAssignment(int(region)) for region in candidate_regions]
        prototype = h.build_prototype_graph(specs, graph, regions,
            self.inputs.bundle.prototype_bank, config=self.inputs.graph_config)
        prototype.v24_recipient_GT_free = True
        return graph, prototype

    def derive(self, plan, provider=None):
        """Read/verify full source and return private exact stage geometry."""
        self._guard(); began = time.perf_counter()
        full, indices, queries = self._selection(plan)
        original, full_prototype, audit, source = self._load_full(full)
        original_digest = tensor_digest((original.to_dict(), full_prototype.to_dict()))
        graph, prototype = self._rebuild(original, full_prototype, indices)
        audit = copy.deepcopy(audit)
        audit['candidate_count'] = len(indices)
        audit['query_inputs_sha256'] = tensor_digest(queries)
        if tensor_digest((original.to_dict(), full_prototype.to_dict())) != original_digest:
            raise ValueError('Derivation mutated the immutable source tensors')
        self.cache._check(graph, prototype, audit, plan)
        _, _, binding = self.cache._path(plan)
        self.cache.guard_inputs(plan, binding)
        if self.cache._file_identity(Path(source['path'])) != tuple(source['stat_identity']):
            raise ValueError('Full128 source changed during stage derivation')
        self._guard()
        row = dict(case_id=plan.case_id, active_u_count=plan.active_u_count,
            candidate_count=len(indices), record_ids=list(plan.record_ids), full128_indices=list(indices),
            source=source, tensor_sha256=tensor_digest((graph.to_dict(), prototype.to_dict())),
            audit_sha256=canonical_hash(audit), seconds=time.perf_counter()-began)
        with self._lock: self._rows.append(row)
        return graph, prototype, audit

    __call__ = derive

    def receipt(self):
        self._guard()
        with self._lock:
            value = dict(contract=copy.deepcopy(self.contract),
                source_files=copy.deepcopy(list(self._source_proofs.values())),
                derivations=copy.deepcopy(self._rows),
                workers=self.cache.workers, RSS_limit_bytes=self.inputs.rss_bytes,
                raw_resident_limit_bytes=self.inputs.raw_resident_bytes,
                no_cold_raw_builder_called=True, source_geometry_not_overwritten=True)
        return dict(value, content_sha256=canonical_hash(value))


def bind(inputs):
    """Replace only this CPU helper instance's builder; never patch cache classes."""
    previous = getattr(inputs, '_v24_upper_reuse_adapter', None)
    if previous is not None:
        if not isinstance(previous, UpperReuseAdapter) or inputs.geometry.builder != previous.derive:
            raise ValueError('Foreign upper reuse binding refused')
        previous._guard()
        return previous
    adapter = UpperReuseAdapter(inputs)
    inputs.geometry.builder = adapter.derive
    inputs._v24_upper_reuse_adapter = adapter
    return adapter
