"""Recipient-GT-independent patient/region/population graph construction.

Every recipient-side descriptor is a function of explicit CT, organ anatomy,
candidate coordinates and the allowed donor mask. No recipient tumor label
exists on the object supplied to any retained original geometry function.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import inspect
import json
import os
import tempfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import numpy as np
import torch

from .v24_inputs import RecipientContext, array_digest, query_inputs, tensor_digest
from .v24_model import PATIENT_NODE_TYPES

FORMAT = 'v24_recipient_GT_blind_upper_graphs_v1'
_REGION_BUILDERS = {}


def build_recipient_regions(recipient, *, config, seed, ct_clip):
    """Original full region equations with the explicit organ input substituted."""
    if not isinstance(recipient, RecipientContext):
        raise TypeError('RecipientContext without tumor labels required')
    from hiercp import region
    original = region.build_patient_regions
    filename = inspect.getsourcefile(original)
    identity = hashlib.sha256(Path(filename).read_bytes()).hexdigest()
    if identity not in _REGION_BUILDERS:
        tree = ast.parse(inspect.getsource(original))
        expected = ast.parse('(case.label == int(liver_label)) | (case.label == int(tumor_label))', mode='eval').body
        count = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and ast.dump(node.value) == ast.dump(expected):
                node.value = ast.parse('case.organ_mask', mode='eval').body
                count += 1
        if count != 1:
            raise ValueError('Exactly one original full-organ input expression required')
        # The original region descriptor is observed CT, not annotation-filled CT.
        ct_source = inspect.getsource(region._observed_context_image)
        if any(isinstance(node, ast.Attribute) and node.attr == 'label' for node in ast.walk(ast.parse(ct_source))):
            raise ValueError('Original observed regional CT unexpectedly reads labels')
        ast.fix_missing_locations(tree)
        namespace = dict(vars(region))
        exec(compile(tree, filename + ':v24_explicit_organ', 'exec'), namespace)
        _REGION_BUILDERS[identity] = namespace[original.__name__]
    return _REGION_BUILDERS[identity](recipient, liver_label=1, tumor_label=2,
        config=config, rng=np.random.default_rng(seed), ct_clip=ct_clip)


def recipient_liver_raw(recipient, regions, *, hierarchy, ct_clip):
    """Full-organ CT/anatomy descriptor; no tumor-count statistic."""
    from scipy import ndimage as ndi
    mask = regions.full_organ_mask
    values = recipient.image[mask]
    depths = regions.organ_depth[mask]
    return hierarchy.upper_geometry_vector(center=ndi.center_of_mass(mask), shape=recipient.shape,
        border_distance_mm=float(depths.mean()), occupied_distance_mm=np.inf,
        context_mean_hu=float(values.mean()), context_std_hu=float(values.std()),
        volume_vox=int(mask.sum()), coverage=1., local_thickness_mm=2.*float(depths.max()),
        # This bookkeeping slot is removed from the learned liver projection.
        surface_alignment=0., scale_mean=1., anisotropy=1., valid=1., ct_clip=ct_clip)


def _validate_regions(recipient, regions):
    from hiercp.common import organ_depth_mm
    if (not np.array_equal(regions.full_organ_mask, recipient.organ_mask)
            or not np.array_equal(regions.organ_depth, organ_depth_mm(recipient.organ_mask, recipient.spacing))
            or regions.region_labels.shape != recipient.shape
            or not np.array_equal(regions.region_labels >= 0, recipient.organ_mask)
            or not np.isfinite(regions.region_features).all()
            or not np.isfinite(regions.region_positions).all()):
        raise ValueError('Regions must preserve exact explicit organ/physical geometry/observed CT')


def build_upper_graphs(recipient, donor_case, donor_source, recipient_regions, donor_regions,
                       rows, bank, *, config, ct_clip, training_case_ids):
    """Complete joint patient graph, with authorized annotation paths absent."""
    from hiercp import hierarchy as h
    from hiercp import schema
    from hiercp.common import extract_centered_patch, context_stats_for_local_mask
    from hiercp.curriculum import CandidateSpec
    from hiercp_v22.data import donor_in_target_spacing
    from .historical_patient_graph import _source
    if not isinstance(recipient, RecipientContext):
        raise TypeError('RecipientContext without tumor annotation required')
    queries = query_inputs(rows)
    _source(donor_case, donor_source, 2)
    training = tuple(training_case_ids)
    if (not training or len(set(training)) != len(training)
            or set(bank.training_case_ids) != set(training)
            or donor_case.paths.case_id not in training
            or recipient.case_id == donor_case.paths.case_id
            or any(row['case_id'] != recipient.case_id or row['donor_case_id'] != donor_case.paths.case_id
                   or row['donor_component'] != donor_source.component_id for row in queries)):
        raise ValueError('Exact train-only prototype bank and independent train donor required')
    _validate_regions(recipient, recipient_regions)
    target_source, _ = donor_in_target_spacing(donor_source, donor_case.spacing, recipient.spacing)
    assignments, _ = bank.assign(recipient_regions.region_features,
        top_k=config.prototype_top_m, temperature=config.prototype_temperature)
    specs = []
    for row in queries:
        center = tuple(row['center'])
        if any(x >= extent for x, extent in zip(center, recipient.shape)):
            raise ValueError('Recorded native candidate outside actual CT grid')
        mask = target_source.patch_mask
        ct = extract_centered_patch(recipient.image, center, mask.shape, pad_value=ct_clip[0])
        organ = extract_centered_patch(recipient.organ_mask, center, mask.shape, pad_value=False)
        mean, std = context_stats_for_local_mask(ct, organ, mask)
        region = int(recipient_regions.region_at(center))
        # Annotation-derived occupied clearance is unavailable, including cache raw fields.
        specs.append(CandidateSpec(center, 1, 0, region, int(assignments[region, 0]),
            float((organ & mask).sum()/mask.sum()), float(recipient_regions.organ_depth[center]),
            np.inf, mean, std))
    axis, anisotropy = h._principal_axis(donor_source.full_mask, donor_case.spacing)
    source_raw = h._source_raw(donor_case, donor_source, donor_regions,
        (donor_case.label == 2) & ~donor_source.full_mask, ct_clip=ct_clip)[None]
    candidate_raw = np.stack([h._candidate_raw(recipient, target_source, spec, recipient_regions,
        source_axis=axis, source_anisotropy=anisotropy, ct_clip=ct_clip) for spec in specs]).astype(np.float32)
    candidate_positions = np.stack([h.normalized_position(spec.center, recipient.shape) for spec in specs]).astype(np.float32)
    candidate_regions = np.asarray([spec.region_id for spec in specs], np.int64)
    graph = h.HeteroData()
    graph.patient_graph_contract = schema.PATIENT_GRAPH_CONTRACT
    graph.v24_recipient_GT_free = True
    graph['tumor'].raw_x = torch.from_numpy(source_raw.astype(np.float32))
    graph['tumor'].pos = torch.from_numpy(h.normalized_position(donor_source.anchor_center, donor_case.shape)[None].astype(np.float32))
    graph['tumor'].region_index = torch.tensor([-1], dtype=torch.long)
    for name, raw, pos, regions in (
            ('candidate', candidate_raw, candidate_positions, candidate_regions),
            ('region', recipient_regions.region_features.astype(np.float32), recipient_regions.region_positions.astype(np.float32),
             np.arange(recipient_regions.num_regions, dtype=np.int64)),
            ('liver', recipient_liver_raw(recipient, recipient_regions, hierarchy=h, ct_clip=ct_clip)[None],
             np.zeros((1,3), np.float32), np.array([-1], np.int64))):
        graph[name].raw_x = torch.from_numpy(raw)
        graph[name].pos = torch.from_numpy(pos)
        graph[name].region_index = torch.from_numpy(regions)
    for name in PATIENT_NODE_TYPES:
        graph[name].meta = torch.from_numpy(h._node_meta(graph[name].raw_x.numpy(), name))
    n, r = len(specs), recipient_regions.num_regions
    tc, tr = h._full_bipartite(1, n), h._full_bipartite(1, r)
    cr = np.stack([np.arange(n, dtype=np.int64), candidate_regions])
    rl = h._full_bipartite(r, 1)
    relations = {
        ('tumor','compatible_with','candidate'):tc,
        ('candidate','matched_to','tumor'):tc[[1,0]],
        ('candidate','spatial_neighbor','candidate'):h._knn(candidate_positions, config.candidate_k),
        ('candidate','belongs_to','region'):cr,
        ('region','contains_candidate','candidate'):cr[[1,0]],
        ('tumor','conditions','region'):tr,
        ('region','context_for','tumor'):tr[[1,0]],
        ('region','adjacent_to','region'):recipient_regions.region_edge_index.astype(np.int64),
        ('region','inside','liver'):rl,
        ('liver','contains','region'):rl[[1,0]],
    }
    retained = tuple(edge for edge in schema.PATIENT_EDGE_TYPES if 'lesion' not in (edge[0], edge[2]))
    if set(relations) != set(retained):
        raise ValueError('GT-blind graph must retain every non-lesion original relation')
    for edge in retained: h._set_patient_relation(graph, edge, relations[edge])
    prototype = h.build_prototype_graph(specs, graph, recipient_regions, bank, config=config)
    prototype.v24_recipient_GT_free = True
    audit = dict(format=FORMAT, candidate_count=n, recipient_GT_used_in_forward=False,
        P_U_labels_in_forward=False, annotation_blind=True, recipient_label_available_to_graph=False,
        recipient_lesion_nodes=0, removed_recipient_lesion_relations=6,
        recipient_tumor_count_input_removed=True, recipient_occupied_clearance_input_removed=True,
        complete_candidates=True, retained_region_nodes=r,
        prototype_training_case_ids=list(bank.training_case_ids),
        recipient_binding=recipient.binding(), query_inputs_sha256=tensor_digest(queries),
        donor_mask_sha256=array_digest(donor_source.full_mask))
    return graph, prototype, audit


class V24UpperGeometryCache:
    """Fresh GT-blind namespace with bounded memoization and exact file admission.

    ``builder(plan, provider)`` must call the explicit GT-blind construction
    above. Old GT-exposed upper cache formats/graphs are rejected, not promoted.
    """
    recipient_GT_used_in_forward = False
    def __init__(self, population, output, builder, *, input_binding, guard_inputs,
                 workers, resident_bytes, rss_bytes):
        if (type(workers) is not int or workers < 2 or type(resident_bytes) is not int
                or not 0 < resident_bytes <= 128*2**30 or type(rss_bytes) is not int
                or not resident_bytes < rss_bytes <= 192*2**30):
            raise ValueError('Explicit parallel workers and bounded128GiB/192GiB budgets required')
        self.population, self.output, self.builder = population, Path(output), builder
        if not callable(input_binding) or not callable(guard_inputs):
            raise ValueError('Actual CT/organ/donor/bank/config/source bindings and file guards required')
        self.input_binding, self.guard_inputs = input_binding, guard_inputs
        self.workers, self.resident_bytes, self.rss_bytes = workers, resident_bytes, rss_bytes
        self.output.mkdir(parents=True, exist_ok=True)
        from collections import OrderedDict
        from threading import RLock
        self._memo, self._bytes, self._proofs, self._lock = OrderedDict(), 0, {}, RLock()
        # The shared CPU-provider coordinator may evict resident input views
        # before the process-wide RSS guard. Eviction never reduces data.
        self.memory_guard = None

    def _plan(self, case, count, selection):
        return self.population.case(case, count, active_u_indices=None if selection is None else selection[case])

    def _path(self, plan):
        # Hash the actual recipient CT/organ and train donor/prototype/source contract.
        # The callback must not bind recipient tumor annotation or P/U targets.
        binding = self.input_binding(plan)
        required = {'recipient', 'donor', 'prototype_bank_sha256', 'config_sha256', 'source_sha256'}
        if not isinstance(binding,dict) or set(binding) != required:
            raise ValueError('Closed recipient-GT-free input binding required')
        if (binding['recipient'].get('format') != 'v24_recipient_CT_organ_no_tumor_annotation_v1'
                or binding['recipient'].get('recipient_tumor_GT_used') is not False
                or 'label_sha256' in binding['recipient']):
            raise ValueError('Recipient cache identity must not depend on tumor annotation')
        self.guard_inputs(plan,binding)
        key = tensor_digest(dict(query=query_inputs(plan.query_rows), inputs=binding))
        return self.output / (key+'.pt'), key, binding

    def _guard(self):
        import psutil
        if self.memory_guard is not None:
            if not callable(self.memory_guard):
                raise TypeError('Shared geometry memory guard must be callable')
            self.memory_guard()
        if psutil.Process().memory_info().rss > self.rss_bytes:
            raise MemoryError('v2.4 geometry RSS budget exceeded; no graph/data reduction')

    @staticmethod
    def _check(graph, prototype, audit, plan):
        if (audit.get('format') != FORMAT or audit.get('recipient_GT_used_in_forward') is not False
                or audit.get('P_U_labels_in_forward') is not False
                or graph.get('v24_recipient_GT_free') is not True
                or prototype.get('v24_recipient_GT_free') is not True
                or 'lesion' in graph.node_types or any('lesion' in (e[0],e[2]) for e in graph.edge_types)
                or int(graph['candidate'].num_nodes) != len(plan.record_ids)):
            raise ValueError('Actual complete recipient-GT-blind graph required; legacy cache rejected')

    @staticmethod
    def _file_identity(path):
        st = path.stat()
        return st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns

    def _load(self, path, plan, key, binding, *, expected_tensor_sha256=None):
        for _ in range(3):
            before = self._file_identity(path)
            value = torch.load(path, map_location='cpu', weights_only=False)
            graph, prototype, audit = value['graph'], value['prototype'], value['audit']
            if (value['format'] != FORMAT or value['query_sha256'] != key
                    or value.get('input_binding') != binding):
                raise ValueError('New GT-blind geometry cache ownership/query binding differs')
            actual = tensor_digest((graph.to_dict(), prototype.to_dict()))
            if value['tensor_sha256'] != actual:
                raise ValueError('GT-blind geometry tensor content changed')
            if expected_tensor_sha256 is not None and actual != expected_tensor_sha256:
                raise ValueError('Concurrent GT-blind geometry builders produced different tensors')
            self._check(graph, prototype, audit, plan)
            if self._file_identity(path) == before:
                return graph, prototype, audit, before
            # Removing a publisher's private temporary hard link can change
            # ctime. Re-read and fully revalidate any changed file; never admit
            # a stale proof or silently ignore a content/binding difference.
        raise ValueError('GT-blind geometry file repeatedly changed during admission')

    def _publish(self, path, value, plan, key, binding):
        """Publish a complete payload without replacing another process's file."""
        # The temporary file and destination share a filesystem. Hard-link
        # publication is atomic and exclusive: readers never observe a partial
        # torch.save, and a competing writer cannot overwrite the winner.
        fd, name = tempfile.mkstemp(prefix='.v24-upper-publish-', suffix='.tmp', dir=self.output)
        temporary = Path(name)
        owned = os.fstat(fd)
        try:
            with os.fdopen(fd, 'wb') as stream:
                torch.save(value, stream)
                stream.flush()
                os.fsync(stream.fileno())
            self.guard_inputs(plan, binding)
            try:
                os.link(temporary, path)
            except FileExistsError:
                self.guard_inputs(plan, binding)  # Admit the complete competing payload below.
        finally:
            st = temporary.stat()
            if (st.st_dev, st.st_ino) != (owned.st_dev, owned.st_ino):
                raise ValueError('Task-owned geometry publication temporary file was replaced')
            temporary.unlink()
        # Admit either winner, including equality with our independently built
        # complete tensors. A competing different payload is an explicit error.
        return self._load(path, plan, key, binding,
            expected_tensor_sha256=value['tensor_sha256'])

    def get(self, plan, provider=None):
        self._guard(); path, key, binding = self._path(plan)
        with self._lock:
            cached = self._memo.get(key)
            if cached is not None:
                proof = self._file_identity(path)
                if proof != self._proofs[key]:
                    raise ValueError('Admitted GT-blind geometry file changed')
                self._memo.move_to_end(key)
                graph, prototype, audit, _ = cached
                self.guard_inputs(plan,binding)
                return graph.clone(), prototype.clone(), copy.deepcopy(audit)
        if path.exists():
            graph, prototype, audit, proof = self._load(path, plan, key, binding)
        else:
            graph, prototype, audit = self.builder(plan, provider)
            self._check(graph, prototype, audit, plan)
            self.guard_inputs(plan,binding)
            value = dict(format=FORMAT, query_sha256=key, input_binding=binding,
                graph=graph, prototype=prototype, audit=audit,
                tensor_sha256=tensor_digest((graph.to_dict(), prototype.to_dict())))
            graph, prototype, audit, proof = self._publish(path, value, plan, key, binding)
        self._check(graph, prototype, audit, plan)
        self.guard_inputs(plan,binding)
        size = sum(t.numel()*t.element_size() for item in (graph, prototype)
                   for store in item.stores for t in store.values() if isinstance(t, torch.Tensor))
        self._guard()
        if self._file_identity(path) != proof:
            raise ValueError('GT-blind geometry file changed before resident admission')
        if size <= self.resident_bytes:
            with self._lock:
                previous = self._memo.pop(key, None)
                if previous is not None:
                    self._bytes -= previous[3]
                while self._memo and self._bytes+size > self.resident_bytes:
                    old_key, (_,_,_,old_size) = self._memo.popitem(last=False)
                    self._bytes -= old_size
                    self._proofs.pop(old_key, None)
                self._memo[key] = graph, prototype, copy.deepcopy(audit), size; self._bytes += size
                self._proofs[key] = proof
        return graph.clone(), prototype.clone(), copy.deepcopy(audit)

    __call__ = get

    def prepare(self, count, target_selection=None):
        cases = self.population.partition_cases('inner_train', ranking_only=True) + self.population.partition_cases('inner_val')
        plans = [self._plan(case,count,target_selection) for case in cases]
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            list(pool.map(self.get,plans))
        return dict(format=FORMAT, active_u=count, cases=len(plans), recipient_GT_used_in_forward=False)

    def admit(self, count, target_selection=None):
        cases = self.population.partition_cases('inner_train', ranking_only=True) + self.population.partition_cases('inner_val')
        for case in cases:
            plan = self._plan(case,count,target_selection)
            if not self._path(plan)[0].is_file():
                raise FileNotFoundError('Complete GT-blind geometry must be prepared: '+case)
            self.get(plan)
        return dict(format=FORMAT, active_u=count, admitted_cases=len(cases), recipient_GT_used_in_forward=False)
