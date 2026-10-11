"""Owned v3.0 BEST scores for the entire unchanged historical Basic CP bank.

Only the score overlay changes. The original same-patient source schedule,
128 legal positions, source arrays, normalized preprocessed hard paste, CP0.5,
20mm diameter and .85 liver coverage gates remain bound to their source bytes.
The actual v3.0 scorer supplies both local views, complete joint P/CP upper
graphs and every recipient's complete native vessel graph. No MDN is present.
"""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
import copy
from dataclasses import asdict
from pathlib import Path
import pickle
from threading import RLock
import time

import numpy as np

from .contracts import canonical_hash
from .v24_matched_basic_cp_scoring import (
    MatchedInputProvider, admit_source_bank, guard_source_bank, matched_builders,
    matched_plan, source_batches, validate_payload,
)


FORMAT = 'v30_owned_BEST_complete_Basic_81case642source128positions_v1'
CP_CONTRACT = dict(
    eligible_patients=81, total_sources=642, candidates_per_source=128,
    total_scored_candidates=82176, donor_policy='same_patient',
    source_selection='same_as_historical_Basic', candidate_payload_preserved=True,
    non_score_payload_preserved=True, paste_contract='historical_Basic_preprocessed_patch',
    cp_probability=.5, maximum_diameter_mm=20., minimum_liver_coverage=.85,
    intensity_scale_range=[.95, 1.05], intensity_shift_range_hu=[-5., 5.],
    selection='gnn_argmax', selection_policy='argmax',
    source_policy='same_patient_uniform_original_sources_per_visit',
    paste_policy='historical_preprocessed_patch_hard_paste',
    train_cases=105, validation_cases=26, physical_batch=2,
    patch=[128, 128, 128], epochs=250,
)


def _sha(path):
    from .v24_nnunet_cp import sha
    return sha(path)


def _stat(path):
    value = Path(path).stat()
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def _checksum(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def validate_best_provenance(provenance, *, model_sha256, context_contract_sha256):
    """Require the caller's independently audited completed own-arm BEST.

    The root runner verifies checkpoint/request bytes and loads the model.
    This second admission binds that completed evidence to the actual model
    tensors and shared train/validation/CP context used by this helper.
    """
    if (not isinstance(provenance, Mapping) or type(provenance.get('completed_epochs')) is not int or
            provenance['completed_epochs'] != 40 or type(provenance.get('optimizer_updates')) is not int or
            provenance['optimizer_updates'] != 680 or
            provenance.get('selected_by') != 'fixed_full128_validation_only' or
            not _checksum(provenance.get('model_sha256')) or provenance['model_sha256'] != model_sha256 or
            not _checksum(provenance.get('context_contract_sha256')) or
            provenance['context_contract_sha256'] != context_contract_sha256 or
            provenance.get('debug') is not False or provenance.get('full_training') is not True):
        raise ValueError('Audited completed full40/680 own-arm full128 BEST and identical v25 context required')
    canonical_hash(provenance)
    return copy.deepcopy(dict(provenance))


def validate_basic_contract(metadata):
    """Original legal-paste gates are admission requirements, never regenerated."""
    required = dict(format='hiercp_online_bank_v2', candidate_count=128,
        cp_probability=.5, maximum_diameter_mm=20., minimum_liver_coverage=.85,
        intensity_scale_range=[.95, 1.05], intensity_shift_range_hu=[-5., 5.])
    if (not isinstance(metadata, Mapping) or 'paste_contract' in metadata or
            any(metadata.get(key) != value for key, value in required.items())):
        raise ValueError('Exact historical Basic128/CP0.5/20mm/coverage.85/HU hard-paste contract required')


def validate_source_context(prepared, source, case_id, profile):
    """Reject depth4, inference-only depth1/2mm, empty-source or unbound inputs."""
    from . import v25_context
    proof = getattr(prepared, 'v25_context_geometry', None)
    if (getattr(prepared, 'v25_context_contract', None) != profile or not isinstance(proof, dict) or
            proof.get('profile_sha256') != profile or proof.get('minimum_liver_depth_exclusive_mm') != 1. or
            proof.get('context_inner_radius_mm') != 0. or proof.get('context_outer_radius_mm') != 10.):
        raise ValueError('Fresh shared v25 source context depth>1 and 0<distance<=10 required for CP')
    actual = v25_context._check_local({'nodes': prepared.canonical_nodes,
        v25_context.PROOF_KEY: proof}, profile)
    identity = actual.get('source_identity', {})
    if (identity.get('case_id') != case_id or identity.get('component_id') != source.component_id or
            identity.get('full_source_mask_sha256') != v25_context._hash_array(source.full_mask) or
            actual['canonical_context_nodes'] <= 0 or
            prepared.canonical_counts.get('source_context') != actual['canonical_context_nodes']):
        raise ValueError('Actual annotated selected source, complete canonical context and mask binding required')
    return copy.deepcopy(actual)


def cp_score_row(plan, complete_scores, payload, *, original_sha256, model_sha256,
                 input_binding, workload):
    """Keep P as joint context and select among all128 original CP positions."""
    import torch
    validate_payload(payload)
    scores = complete_scores.detach().float().cpu().numpy() if isinstance(complete_scores, torch.Tensor) else np.asarray(complete_scores)
    if (scores.shape != (len(plan.record_ids),) or scores.dtype.kind not in 'fi' or
            not np.isfinite(scores).all() or len(plan.query_rows) != len(plan.record_ids) or
            len(plan.record_ids) != plan.observed_P + 128):
        raise ValueError('Complete finite actual all-P plus128 CP score vector required')
    raw = np.asarray([row['center'] for row in plan.query_rows[plan.observed_P:]], dtype=np.int64)
    if raw.shape != (128, 3) or not np.array_equal(raw, payload['candidate_raw_centers']):
        raise ValueError('Actual scored queries differ from the original ordered128 Basic CP centers')
    if (len(set(plan.record_ids)) != len(plan.record_ids) or
            any(set(row) != {'id', 'case_id', 'center', 'donor_case_id', 'donor_component'} or
                row['case_id'] != plan.case_id or row['donor_case_id'] != plan.case_id or
                row['donor_component'] != plan.donor_component for row in plan.query_rows)):
        raise ValueError('Complete GT-blind same-patient selected-source query identities required')
    if not _checksum(original_sha256) or not _checksum(model_sha256):
        raise ValueError('Original source and actual owned BEST model checksums required')
    values = np.asarray(scores[plan.observed_P:], dtype=np.float64)
    return dict(case_id=plan.case_id, source_component=plan.donor_component, entry=plan.entry,
        original_sha256=original_sha256, scores=values.tolist(), selected_index=int(np.argmax(values)),
        candidate_raw_centers=np.asarray(payload['candidate_raw_centers']).tolist(),
        candidate_centers=np.asarray(payload['candidate_centers']).tolist(),
        joint_observed_P=plan.observed_P, joint_candidates=len(plan.record_ids),
        all_original_P_in_context=True, all_original128_CP_scored=True,
        recipient_GT_used_in_forward=False, model_sha256=model_sha256,
        input_binding=copy.deepcopy(input_binding), workload=copy.deepcopy(workload),
        only_CP_field_changed='scores', original_non_score_payloads_preserved=True,
        original_source_schedule_and_paste_contract_preserved=True)


def validate_complete_score_table(rows, source_proof, metadata):
    """Never label an omitted source, candidate, or patient as a complete overlay."""
    expected = source_proof.get('entries', {})
    order = metadata.get('entries_by_case', {})
    if (not isinstance(rows, Mapping) or len(expected) != 642 or len(order) != 81 or
            sum(map(len, order.values())) != 642 or set(rows) != set(expected) or
            {entry for names in order.values() for entry in names} != set(expected)):
        raise ValueError('Complete original81 patient/642 source/82176 CP coverage required; no subset')
    for entry, row in rows.items():
        values = np.asarray(row.get('scores'), dtype=np.float64)
        original = expected[entry]
        if (row.get('entry') != entry or row.get('case_id') != original['case_id'] or
                row.get('source_component') != original['source_component'] or
                row.get('original_sha256') != original['sha256'] or values.shape != (128,) or
                not np.isfinite(values).all() or row.get('selected_index') != int(np.argmax(values)) or
                row.get('all_original_P_in_context') is not True or
                row.get('all_original128_CP_scored') is not True or
                row.get('recipient_GT_used_in_forward') is not False or
                row.get('only_CP_field_changed') != 'scores' or
                row.get('original_non_score_payloads_preserved') is not True or
                row.get('original_source_schedule_and_paste_contract_preserved') is not True):
            raise ValueError('Complete finite original-order128 CP scores and stable argmax required: ' + entry)
    return dict(complete_cases=81, complete_sources=642, complete_positions=82176)


def score_v30_cp_bank(*, net, inputs, population, source_bank, baseline_preprocessed,
                     vessel_store, recipient_bindings, physical_candidate_batch,
                     budget, amp, best_provenance, on_scored_source=None):
    """Actual CUDA scoring of every original source with the caller's loaded BEST.

    ``inputs`` is the fresh v3 factory workspace with the shared v25 profile
    already installed; no original ranking128 upper/canonical cache is used.
    The optional callback can durably publish each accepted source row. Only
    the returned full642 overlay is a completion receipt. This helper performs
    no native training, paste, checkpoint write, or production optimizer update.
    """
    import torch
    from . import v25_context
    from .v30_model import VesselAugmentedScoreHead

    if (not isinstance(net.score_head, VesselAugmentedScoreHead) or net.training is not False or
            next(net.parameters()).device.type != 'cuda'):
        raise RuntimeError('CP requires the actual complete v3 vessel model in eval on CUDA; no CPU/model fallback')
    if (type(physical_candidate_batch) is not int or physical_candidate_batch not in (32, 64, 128) or
            type(amp) is not bool or not callable(getattr(budget, 'check', None)) or
            (on_scored_source is not None and not callable(on_scored_source))):
        raise ValueError('Measured candidate32/64/128, explicit precision/resource budget and valid callback required')
    profile = v25_context.identity()
    if getattr(inputs, 'context_contract', None) != profile or v25_context._INSTALLED is None:
        raise ValueError('Fresh installed identical train/validation/CP v25 context profile required')
    import blosc2
    from .v30_scorer import V30Scorer
    from .v30_vessels import RecipientBinding
    from .v24_inputs import prepare_donor, immutable_array, array_digest
    from .v24_nnunet_cp import read, validate_baseline
    from .u_bridge_training import digest, capture_rng, restore_rng
    from hiercp_v22.data import sources
    from hiercp_v22.bank import map_center
    model_sha = digest(net.state_dict())
    provenance = validate_best_provenance(best_provenance, model_sha256=model_sha,
        context_contract_sha256=profile['contract_sha256'])
    split = inputs.inventory['split']
    metadata, source_proof = admit_source_bank(source_bank, split)
    validate_basic_contract(metadata)
    baseline = validate_baseline(baseline_preprocessed, split)
    if set(inputs.training_cases) != set(split['inner_train']):
        raise ValueError('Original full inner-train84-only prototype scope must be retained')
    vessel_inventory = vessel_store.audit()
    if (vessel_inventory.get('inventory_cases') != 131 or
            set(vessel_inventory.get('case_ids', ())) != set(inputs.raw) or
            vessel_inventory.get('native_sampling_ratio') != 1. or
            vessel_inventory.get('graph_caps') is not False or
            vessel_inventory.get('component_pruning') is not False or
            vessel_inventory.get('artificial_connections') is not False):
        raise ValueError('Complete unchanged native131 vessel inventory required for CP; no graph subset')
    local, upper, admission = matched_builders(split['outer_train'])
    plans, payloads = {}, {}
    for case, names in metadata['entries_by_case'].items():
        for entry in names:
            with np.load(Path(source_proof['root']) / entry, allow_pickle=False) as loaded:
                payload = {key: loaded[key] for key in loaded.files}
            plan = matched_plan(case, entry, payload, population)
            if plan.partition not in ('inner_train', 'inner_val'):
                raise ValueError('Every historical Basic source must retain its original inner105 partition')
            plans[entry], payloads[entry] = plan, {key: payload[key] for key in
                ('candidate_raw_centers', 'candidate_centers', 'source_diameter_mm')}
    # Retain source payload arrays only for validation/publication, never feed
    # pre-existing Basic scores, source_data or GT into neural candidate inputs.
    active, active_graphs, providers = {}, {}, {}
    original_rng = capture_rng()
    began = time.perf_counter()
    rows = {}
    mapping_plans = read(Path(baseline['preprocessed']) / (baseline['plans_name'] + '.json'))
    directory = Path(baseline['preprocessed']) / baseline['data_identifier']
    collections, headers, cache_lock = OrderedDict(), {}, RLock()
    collection_bytes = 0
    # This is static array residency, not a source/sample/graph cap. Every
    # source is still constructed and scored after its cached patient is evicted.
    collection_resident_bytes = 8 * 2**30

    def collection_for(case_id, raw):
        nonlocal collection_bytes
        with cache_lock:
            if case_id in collections:
                collections.move_to_end(case_id)
                return collections[case_id][0]
        collection = sources(raw['donor_case'], inputs.config['cache']['source_pad'], 20.)
        retained = collection.components.nbytes + collection.sizes.nbytes
        retained += raw['donor_case'].image.nbytes + raw['donor_case'].label.nbytes
        with cache_lock:
            while collections and collection_bytes + retained > collection_resident_bytes:
                _, (_, released) = collections.popitem(last=False); collection_bytes -= released
            if retained <= collection_resident_bytes:
                collections[case_id] = collection, retained; collection_bytes += retained
        return collection

    def mapping_header(case_id):
        with cache_lock:
            if case_id in headers:
                properties, shape, proof = headers[case_id]
                if _stat(directory/(case_id+'.pkl')) != proof['properties_stat'] or _stat(directory/(case_id+'.b2nd')) != proof['data_stat']:
                    raise ValueError('Original preprocessed mapping/header bytes changed: ' + case_id)
                return properties, shape, proof
        properties_path, data_path = directory/(case_id+'.pkl'), directory/(case_id+'.b2nd')
        properties_stat, data_stat = _stat(properties_path), _stat(data_path)
        with properties_path.open('rb') as stream:
            properties = pickle.load(stream)
        data = blosc2.open(str(data_path), mode='r'); shape = tuple(data.shape[1:]); del data
        proof = dict(properties_sha256=_sha(properties_path), properties_stat=properties_stat, data_stat=data_stat)
        if _stat(properties_path) != properties_stat or _stat(data_path) != data_stat:
            raise ValueError('Original preprocessed mapping changed during admission: ' + case_id)
        with cache_lock: headers[case_id] = properties, shape, proof
        return properties, shape, proof

    def prepare_source(item):
        budget.check()
        case_id, entry = item
        plan = plans[entry]; raw = inputs._case(case_id)
        recipient = raw['context']
        binding = RecipientBinding.from_mapping(recipient_bindings[case_id])
        if (binding.case_id != case_id or binding.ct_sha256 != inputs.raw[case_id]['image_sha256'] or
                binding.native_shape_xyz != tuple(recipient.shape) or
                not np.allclose(binding.affine_voxel_to_ras_mm, recipient.image_affine, rtol=0, atol=1e-7)):
            raise ValueError('Actual CP recipient native CT/header differs from whole-vessel source: ' + case_id)
        collection = collection_for(case_id, raw)
        found = [index for index, (component, _) in enumerate(collection.entries) if component == plan.donor_component]
        if len(found) != 1:
            raise ValueError('Original Basic selected full annotated source component is absent: ' + entry)
        source, diameter = collection[found[0]]
        if not np.isclose(diameter, payloads[entry]['source_diameter_mm'][0], rtol=1e-6, atol=1e-6):
            raise ValueError('Historical Basic source/raw selected component diameter differs')
        source.full_mask = immutable_array(source.full_mask)
        source.v24_mask_sha256 = array_digest(source.full_mask)
        prepared = prepare_donor(raw['donor_case'], source, config=inputs.graph_config,
            seed=inputs.config['seed'], ct_clip=inputs.ct_clip)
        source_geometry = validate_source_context(prepared, source, case_id, profile['contract_sha256'])
        regions = inputs._region(case_id)
        properties, shape, header = mapping_header(case_id)
        mapped = np.stack([map_center(point, mapping_plans, properties, shape)
            for point in payloads[entry]['candidate_raw_centers']])
        if not np.array_equal(mapped, payloads[entry]['candidate_centers']):
            raise ValueError('Original128 Basic CP raw-to-preprocessed mapping or candidate order differs')
        centers = np.asarray([row['center'] for row in plan.query_rows], dtype=np.int64)
        if np.any(centers < 0) or np.any(centers >= np.asarray(recipient.shape)):
            raise ValueError('Complete original P/128CP query leaves actual native recipient CT extent')
        graphs = upper(recipient, raw['donor_case'], source, regions, regions,
            plan.query_rows, inputs.bundle.prototype_bank, config=inputs.graph_config,
            ct_clip=inputs.ct_clip, training_case_ids=inputs.training_cases)
        if graphs[2].get('recipient_GT_used_in_forward') is not False or graphs[2].get('P_U_labels_in_forward') is not False:
            raise ValueError('Matched CP upper graph did not certify recipient GT blindness')
        input_binding = dict(recipient=recipient.binding(), recipient_native_header=asdict(binding),
            donor_case_id=case_id, donor_component=source.component_id,
            donor_mask_sha256=source.v24_mask_sha256, donor_CT_sha256=inputs.raw[case_id]['image_sha256'],
            donor_label_sha256=inputs.raw[case_id]['label_sha256'],
            source_bank_entry_sha256=source_proof['entries'][entry]['sha256'],
            original_non_score_arrays_sha256=source_proof['entries'][entry]['non_score_arrays_sha256'],
            raw_centers_sha256=array_digest(payloads[entry]['candidate_raw_centers']),
            preprocessed_centers_sha256=array_digest(payloads[entry]['candidate_centers']),
            canonical_source_counts=copy.deepcopy(prepared.canonical_counts), source_context_geometry=source_geometry,
            context_contract=copy.deepcopy(profile), properties_sha256=header['properties_sha256'],
            preprocessed_CT_stat_identity=list(header['data_stat']), original_candidate_order_and_mapping_verified=True)
        budget.check()
        return plan, (raw, source, prepared, regions, input_binding), graphs

    def build_local(row):
        raw, source, prepared, _, _ = active[(row['case_id'], row['donor_component'])]
        record = local(raw['context'], raw['donor_case'], source, prepared, row,
            config=inputs.graph_config, seed=inputs.config['seed'], ct_clip=inputs.ct_clip,
            scope_contract=inputs.bundle.scope['contract_sha256'], donor_mask_sha256=source.v24_mask_sha256)
        for role in ('source', 'target'):
            v25_context._check_local(record[role+'_local'], profile['contract_sha256'])
        return record

    class Geometry:
        recipient_GT_used_in_forward = False
        def __call__(self, plan, provider):
            return active_graphs[plan.entry]

    try:
        for partition in ('inner_train', 'inner_val'):
            providers[partition] = MatchedInputProvider([plan for plan in plans.values() if plan.partition == partition],
                build_local, budget, workers=4)
        scorer = V30Scorer(net, providers, Geometry(), vessel_store=vessel_store,
            recipient_bindings=recipient_bindings, physical_candidate_batch=physical_candidate_batch,
            checkpoint_local_chunks=False, amp=amp, budget=budget, prefetch_cpu_chunks=True, pin_cpu_batches=True)
        with ThreadPoolExecutor(max_workers=4, thread_name_prefix='v30_CP_geometry') as executor:
            for partition in ('inner_train', 'inner_val'):
                for batch in source_batches(metadata['entries_by_case'], split[partition], physical=4):
                    budget.check(); guard_source_bank(source_proof); inputs.guard_source()
                    prepared_batch = list(executor.map(prepare_source, batch))
                    batch_plans = []
                    for plan, item, graphs in prepared_batch:
                        active[(plan.case_id, plan.donor_component)], active_graphs[plan.entry] = item, graphs
                        batch_plans.append(plan)
                    try:
                        with torch.no_grad(): result = scorer(batch_plans, epoch=29, training=False)
                        torch.cuda.synchronize(); budget.check()
                        if result.case_ids != tuple(plan.case_id for plan in batch_plans) or len(result.scores) != len(batch_plans):
                            raise ValueError('Actual v3 CP forward lost disjoint source/patient ownership')
                        if result.workload.get('upper_invocations') != 1 or result.workload.get('local_graph_views') != 2*sum(len(plan.record_ids) for plan in batch_plans):
                            raise ValueError('Every CP source requires both actual local views and one complete batched upper forward')
                        if (result.workload.get('vessel_patient_encoder_invocations') != 1 or
                                result.workload.get('vascular_subsampling') is not False or
                                result.workload.get('vessel_nodes', 0) < len(batch_plans) or
                                result.workload.get('vessel_components', 0) < len(batch_plans)):
                            raise ValueError('Every CP source requires its complete actual patient vessel graph and components')
                        for plan, score in zip(batch_plans, result.scores):
                            with np.load(Path(source_proof['root'])/plan.entry, allow_pickle=False) as loaded:
                                payload = {key: loaded[key] for key in loaded.files}
                            row = cp_score_row(plan, score, payload, original_sha256=source_proof['entries'][plan.entry]['sha256'],
                                model_sha256=model_sha, input_binding=active[(plan.case_id, plan.donor_component)][4],
                                workload=result.workload)
                            if plan.entry in rows:
                                raise ValueError('Duplicate CP source scoring receipt: ' + plan.entry)
                            rows[plan.entry] = row
                            if on_scored_source is not None: on_scored_source(copy.deepcopy(row))
                        del result
                    finally:
                        restore_rng(original_rng)
                        active.clear(); active_graphs.clear(); prepared_batch.clear()
        coverage = validate_complete_score_table(rows, source_proof, metadata)
        guard_source_bank(source_proof); inputs.guard_source()
        if digest(net.state_dict()) != model_sha or digest(capture_rng()) != digest(original_rng):
            raise ValueError('Frozen owned BEST model or scoring RNG changed')
        for name, checksum in baseline['source_files_sha256'].items():
            if _sha(Path(baseline['preprocessed'])/name) != checksum:
                raise ValueError('Historical Basic preprocessing metadata changed during scoring')
        for case_id in headers: mapping_header(case_id)
        overlay = dict(format=FORMAT, complete=True, debug=False, entries=rows,
            entries_by_case=copy.deepcopy(metadata['entries_by_case']), source_bank=source_proof,
            baseline=baseline, split=copy.deepcopy(split), cp_contract=copy.deepcopy(CP_CONTRACT),
            **coverage, best_provenance=provenance, model_sha256=model_sha,
            context_contract=profile, vessel_inventory=vessel_store.audit(),
            only_CP_field_changed='scores', original_non_score_payloads_preserved=True,
            original_source_schedule_and_paste_contract_preserved=True, full_P_context=True,
            recipient_GT_used_in_forward=False, production_optimizer_updates=0,
            model_and_RNG_unchanged=True, actual_CUDA=True, full_CP_scoring_executed=True,
            native_training_executed=False, native_evaluation_executed=False,
            physical_patient_batch=4, physical_candidate_batch=physical_candidate_batch, CPU_workers=4,
            seconds=time.perf_counter()-began, peak_allocated_CUDA_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_CUDA_bytes=torch.cuda.max_memory_reserved(), admission=admission)
        overlay['content_sha256'] = canonical_hash(overlay)
        return overlay
    finally:
        for provider in providers.values(): provider.close()
        restore_rng(original_rng)
