"""Explicit synthetic CPU UNIT checks, not CP/model/CUDA training results.

Exercise source/candidate coverage, shared-context admission, owned BEST and
the actual complete vessel score head. No real CT or full CP bank is scored.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import textwrap
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from hiercp_v1x import v25_context
from hiercp_v1x.v24_matched_basic_cp_scoring import matched_plan
from hiercp_v1x.v30_cp import (
    CP_CONTRACT, cp_score_row, score_v30_cp_bank, validate_basic_contract,
    validate_best_provenance, validate_complete_score_table, validate_source_context,
)
from hiercp_v1x.v30_model import vessel_score_context
from tests.test_v24_matched_basic_cp_scoring import PlanPopulation_DEBUG, payload_DEBUG
from tests.test_v30_model import head_fixture_UNIT, vessel_fixture_UNIT


SHA_SOURCE, SHA_MODEL, SHA_CONTEXT = 'a'*64, 'b'*64, 'c'*64


def best_provenance_UNIT():
    return dict(CPU_UNIT_SYNTHETIC_METADATA=True, completed_epochs=40, optimizer_updates=680,
        selected_by='fixed_full128_validation_only', model_sha256=SHA_MODEL,
        context_contract_sha256=SHA_CONTEXT, full_training=True, debug=False)


def bank_metadata_UNIT():
    return dict(format='hiercp_online_bank_v2', candidate_count=128, cp_probability=.5,
        maximum_diameter_mm=20., minimum_liver_coverage=.85,
        intensity_scale_range=[.95, 1.05], intensity_shift_range_hu=[-5., 5.])


def score_row_UNIT(*, scores=None, count=5):
    payload = payload_DEBUG()
    plan = matched_plan('case', 'CPU_UNIT/source.npz', payload, PlanPopulation_DEBUG(count))
    if scores is None: scores = np.linspace(-2., 3., len(plan.record_ids))
    row = cp_score_row(plan, scores, payload, original_sha256=SHA_SOURCE, model_sha256=SHA_MODEL,
        input_binding=dict(CPU_UNIT_SYNTHETIC_METADATA=True),
        workload=dict(CPU_UNIT_SYNTHETIC_METADATA=True))
    return row, plan, payload


def context_metadata_UNIT():
    # Actual CPU tensor hashing/identity checks, with explicitly synthetic
    # geometry metadata. This is not a real image/EDT/preparation receipt.
    nodes = {'source_context': dict(x=torch.arange(16.).reshape(2, 8)),
             'source_liver_surface': dict(x=torch.arange(8.).reshape(1, 8))}
    source = SimpleNamespace(component_id=3, full_mask=np.ones((3, 3, 3), dtype=bool))
    proof = dict(CPU_UNIT_SYNTHETIC_METADATA=True, format=v25_context.FORMAT,
        profile_sha256=SHA_CONTEXT, minimum_liver_depth_exclusive_mm=1.,
        context_inner_radius_mm=0., context_outer_radius_mm=10., canonical_context_nodes=2)
    proof['geometry_sha256'] = hashlib.sha256(v25_context._json(proof)).hexdigest()
    proof['canonical_nodes_sha256'] = v25_context._node_hash(nodes)
    proof['source_identity'] = dict(case_id='CPU_UNIT_source', component_id=3,
        full_source_mask_sha256=v25_context._hash_array(source.full_mask))
    prepared = SimpleNamespace(canonical_nodes=nodes, canonical_counts={'source_context': 2},
        v25_context_contract=SHA_CONTEXT, v25_context_geometry=proof)
    return prepared, source


class OwnedBESTAndBasicContractCPUUnit(unittest.TestCase):
    def test_owned_completed_full128_BEST_binds_actual_model_and_shared_context(self):
        provenance = best_provenance_UNIT()
        accepted = validate_best_provenance(provenance, model_sha256=SHA_MODEL,
            context_contract_sha256=SHA_CONTEXT)
        self.assertEqual(accepted, provenance)
        accepted['optimizer_updates'] = 0
        self.assertEqual(provenance['optimizer_updates'], 680)
        for field, value in (('completed_epochs', 39), ('optimizer_updates', 679),
                             ('model_sha256', '0'*64), ('context_contract_sha256', '0'*64),
                             ('selected_by', 'stage_validation'), ('debug', True),
                             ('full_training', False), ('completed_epochs', True)):
            bad = dict(provenance, **{field: value})
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_best_provenance(bad, model_sha256=SHA_MODEL, context_contract_sha256=SHA_CONTEXT)
        for field in provenance:
            if field == 'CPU_UNIT_SYNTHETIC_METADATA': continue
            bad = copy.deepcopy(provenance); del bad[field]
            with self.subTest(missing=field), self.assertRaises(ValueError):
                validate_best_provenance(bad, model_sha256=SHA_MODEL, context_contract_sha256=SHA_CONTEXT)

    def test_original_basic_source_selection_and_paste_gates_are_unchanged(self):
        metadata = bank_metadata_UNIT(); validate_basic_contract(metadata)
        self.assertEqual(CP_CONTRACT['source_policy'], 'same_patient_uniform_original_sources_per_visit')
        self.assertEqual(CP_CONTRACT['paste_policy'], 'historical_preprocessed_patch_hard_paste')
        self.assertEqual(CP_CONTRACT['physical_batch'], 2)
        self.assertEqual(CP_CONTRACT['epochs'], 250)
        self.assertEqual(CP_CONTRACT['patch'], [128]*3)
        for field, value in (('candidate_count', 127), ('cp_probability', .6),
                             ('maximum_diameter_mm', 10.), ('minimum_liver_coverage', .8),
                             ('intensity_scale_range', [1., 1.]),
                             ('intensity_shift_range_hu', [0., 0.]),
                             ('paste_contract', 'onlinecp_raw_target_paste_v1')):
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_basic_contract(dict(metadata, **{field: value}))

    def test_old_context_zero_source_and_changed_actual_tensor_are_rejected(self):
        prepared, source = context_metadata_UNIT()
        accepted = validate_source_context(prepared, source, 'CPU_UNIT_source', SHA_CONTEXT)
        self.assertEqual(accepted['canonical_context_nodes'], 2)
        for field, value in (('context_inner_radius_mm', 2.), ('minimum_liver_depth_exclusive_mm', 4.),
                             ('profile_sha256', '0'*64), ('canonical_context_nodes', 0)):
            bad = copy.deepcopy(prepared); bad.v25_context_geometry[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_source_context(bad, source, 'CPU_UNIT_source', SHA_CONTEXT)
        bad = copy.deepcopy(prepared); bad.canonical_nodes['source_context']['x'][0, 0] += 1
        with self.assertRaises(ValueError): validate_source_context(bad, source, 'CPU_UNIT_source', SHA_CONTEXT)
        with self.assertRaises(ValueError): validate_source_context(prepared, source, 'CPU_UNIT_other_source', SHA_CONTEXT)
        bad = copy.deepcopy(source); bad.full_mask[0, 0, 0] = False
        with self.assertRaises(ValueError): validate_source_context(prepared, bad, 'CPU_UNIT_source', SHA_CONTEXT)

    def test_production_scorer_never_uses_cpu_fallback(self):
        model = torch.nn.Module(); model.score_head = head_fixture_UNIT(); model.eval()
        with self.assertRaisesRegex(RuntimeError, 'CUDA; no CPU/model fallback'):
            score_v30_cp_bank(net=model, inputs=None, population=None, source_bank=None,
                baseline_preprocessed=None, vessel_store=None, recipient_bindings={},
                physical_candidate_batch=64, budget=None, amp=False,
                best_provenance=best_provenance_UNIT())


class OriginalOrderedCPScoreCPUUnit(unittest.TestCase):
    def test_every_P_is_scored_but_only_original128_CP_can_be_selected(self):
        row, plan, payload = score_row_UNIT(count=37)
        complete = np.arange(len(plan.record_ids), dtype=np.float64)
        complete[:37] = 1e9  # P cannot become a legal CP placement.
        actual = cp_score_row(plan, complete, payload, original_sha256=SHA_SOURCE,
            model_sha256=SHA_MODEL, input_binding={}, workload={})
        self.assertEqual(actual['joint_observed_P'], 37)
        self.assertEqual(actual['joint_candidates'], 165)
        self.assertEqual(actual['scores'], complete[37:].tolist())
        self.assertEqual(actual['selected_index'], 127)
        self.assertEqual(actual['candidate_raw_centers'], payload['candidate_raw_centers'].tolist())
        json.dumps(row, allow_nan=False)

    def test_original_payload_and_allowed_annotations_are_not_relabelled_or_mutated(self):
        _, plan, payload = score_row_UNIT()
        before = {key: value.copy() for key, value in payload.items()}
        cp_score_row(plan, np.arange(len(plan.record_ids)), payload, original_sha256=SHA_SOURCE,
            model_sha256=SHA_MODEL, input_binding={}, workload={})
        for key, value in before.items(): np.testing.assert_array_equal(payload[key], value)
        self.assertTrue(all(set(row) == {'id', 'case_id', 'center', 'donor_case_id', 'donor_component'}
                            for row in plan.query_rows))

    def test_zero_P_and_equal_scores_retain128_queries_and_first_argmax_tie(self):
        row, _, _ = score_row_UNIT(scores=np.zeros(128), count=0)
        self.assertEqual(row['joint_observed_P'], 0)
        self.assertEqual(row['joint_candidates'], 128)
        self.assertEqual(row['selected_index'], 0)
        self.assertEqual(len(row['scores']), 128)

    def test_missing_nonfinite_subset_reordered_and_gt_contaminated_scores_fail(self):
        _, plan, payload = score_row_UNIT()
        good = np.arange(len(plan.record_ids), dtype=np.float32)
        for mutation in ('partial', 'matrix', 'nan', 'reordered', 'GT', 'wrong_source'):
            altered_plan = copy.deepcopy(plan); scores = good.copy()
            if mutation == 'partial': scores = scores[:-1]
            elif mutation == 'matrix': scores = scores[None]
            elif mutation == 'nan': scores[0] = np.nan
            elif mutation == 'reordered': altered_plan.query_rows[-1]['center'] = altered_plan.query_rows[-2]['center']
            elif mutation == 'GT': altered_plan.query_rows[0]['target'] = 1
            else: altered_plan.query_rows[0]['donor_case_id'] = 'CPU_UNIT_wrong_donor'
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                cp_score_row(altered_plan, scores, payload, original_sha256=SHA_SOURCE,
                    model_sha256=SHA_MODEL, input_binding={}, workload={})

    def test_complete_table_admission_retains_entire81_642_82176_scope(self):
        template, _, _ = score_row_UNIT()
        order, proof, table = {}, {}, {}
        for case in range(81):
            case_id = f'CPU_UNIT_case_{case:03d}'
            names = [f'CPU_UNIT/{case}_{component}.npz' for component in range(8 if case<75 else 7)]
            order[case_id] = names
            for component, name in enumerate(names, 1):
                proof[name] = dict(case_id=case_id, source_component=component, sha256=SHA_SOURCE)
                table[name] = dict(copy.deepcopy(template), entry=name, case_id=case_id, source_component=component)
        source = dict(entries=proof); metadata = dict(entries_by_case=order)
        self.assertEqual(validate_complete_score_table(table, source, metadata),
            dict(complete_cases=81, complete_sources=642, complete_positions=82176))
        for mutation in ('missing_source', 'partial_scores', 'nan_score', 'wrong_argmax', 'wrong_source', 'extra_source'):
            bad = copy.deepcopy(table); name = next(iter(bad))
            if mutation == 'missing_source': del bad[name]
            elif mutation == 'partial_scores': bad[name]['scores'].pop()
            elif mutation == 'nan_score': bad[name]['scores'][0] = float('nan')
            elif mutation == 'wrong_argmax': bad[name]['selected_index'] = 0
            elif mutation == 'wrong_source': bad[name]['original_sha256'] = '0'*64
            else: bad['CPU_UNIT_extra'] = copy.deepcopy(bad[name])
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_complete_score_table(bad, source, metadata)


class ActualVesselHeadToCPSelectionCPUUnit(unittest.TestCase):
    def test_entire128_cp_vector_uses_actual_vessel_path_and_same_patient_ownership(self):
        threads = torch.get_num_threads(); torch.set_num_threads(2)
        try:
            torch.manual_seed(42)
            head = head_fixture_UNIT().eval(); model = SimpleNamespace(score_head=head)
            vessels = vessel_fixture_UNIT()
            _, plan, payload = score_row_UNIT()
            pair = torch.linspace(-1., 1., len(plan.record_ids)*1546).reshape(-1, 1546)
            position = torch.tensor([row['center'] for row in plan.query_rows], dtype=torch.float32)
            owner = torch.zeros(len(plan.record_ids), dtype=torch.long)
            def score(batch):
                with torch.no_grad(), vessel_score_context(model, batch, position, owner):
                    return head(pair).flatten()
            values = score(vessels)
            self.assertEqual(values.shape, (plan.observed_P+128,))
            row = cp_score_row(plan, values, payload, original_sha256=SHA_SOURCE, model_sha256=SHA_MODEL,
                input_binding=dict(CPU_UNIT_ACTUAL_VESSEL_HEAD=True), workload={})
            self.assertEqual(row['selected_index'], int(values[plan.observed_P:].argmax()))
            self.assertEqual(len(row['scores']), 128)
            self.assertIsNone(head._context)
            other = copy.deepcopy(vessels); other.node_features[other.node_patient_index == 1] += .25
            torch.testing.assert_close(score(other), values, rtol=0, atol=0)
            same = copy.deepcopy(vessels); same.node_features[same.node_patient_index == 0] += .25
            self.assertGreater(float((score(same)-values).abs().max()), 0)
            self.assertIsNone(head._context)
        finally:
            torch.set_num_threads(threads)

    def test_cold_actual_archived_selfCP_builders_keep_shared_source_target_context_and_two_views(self):
        # Fresh process is required by the original-source and v25 installation
        # contracts. Real archived operators run on a clearly synthetic32^3
        # CPU UNIT volume; this is not complete642 CP scoring or CUDA evidence.
        code = textwrap.dedent('''
            import copy
            from dataclasses import replace
            from types import SimpleNamespace
            from tests.test_v24_gt_blind import RecipientGTBlindDebug as F
            F.setUpClass()
            from hiercp_v1x.transition_v1_local import _runtime
            from hiercp_v1x import v25_context, v24_input_runtime
            from hiercp_v1x.v24_inputs import prepare_donor, materialize_pair, tensor_digest
            from hiercp_v1x.v24_matched_basic_cp_scoring import matched_builders, MatchedInputProvider
            from hiercp_v1x.v30_cp import validate_source_context
            runtime = _runtime(expected_snapshot_root=F.snapshot, scope_contract=F.scope['contract_sha256'])
            v25_context.install(runtime, approved_profile=True)
            profile = v25_context.identity()['contract_sha256']
            cfg = replace(F.config, context_inner_radius_mm=0.)
            local, upper, proof = matched_builders([F.donor.paths.case_id] +
                ['CPU_UNIT_outer_' + str(index) for index in range(104)])
            assert sum(proof['upper']['same_patient_recipient_other_lesion_input_removals'].values()) == 1
            rows = [dict(row, case_id=F.donor.paths.case_id, donor_case_id=F.donor.paths.case_id)
                    for row in F.rows]
            reference = None
            for other_lesion in (False, True):
                donor = copy.deepcopy(F.donor)
                if other_lesion: donor.label[18:21, 18:21, 18:21] = 2
                prepared = prepare_donor(donor, F.source, config=cfg, seed=42, ct_clip=F.clip)
                source_proof = validate_source_context(prepared, F.source, F.donor.paths.case_id, profile)
                assert source_proof['canonical_context_nodes'] > 0
                assert source_proof['context_inner_radius_mm'] == 0.
                records = [local(F.donor_context, donor, F.source, prepared, row,
                    config=cfg, seed=42, ct_clip=F.clip, scope_contract=F.scope['contract_sha256']) for row in rows]
                graph, proto, audit = upper(F.donor_context, donor, F.source,
                    F.donor_regions, F.donor_regions, rows, F.bank,
                    config=cfg, ct_clip=F.clip, training_case_ids=(F.donor.paths.case_id,))
                assert audit['recipient_GT_used_in_forward'] is False
                assert audit['same_patient_donor_other_lesion_occupancy_used'] is False
                assert graph['candidate'].num_nodes == len(rows)
                for record in records:
                    for role in ('source', 'target'):
                        actual = v25_context._check_local(record[role + '_local'], profile)
                        assert actual['context_inner_radius_mm'] == 0.
                        assert actual['minimum_liver_depth_exclusive_mm'] == 1.
                views = [materialize_pair(record, epoch=29) for record in records]
                signature = (tensor_digest((graph.to_dict(), proto.to_dict())),
                    [record['tensor_sha256'] for record in records],
                    [tensor_digest((tuple(g.to_dict() for g in view[0]), view[1], view[2])) for view in views])
                if reference is None: reference = signature
                assert signature == reference
                lookup = {row['id']: record for row, record in zip(rows, records)}
                # CPU UNIT has no CUDA pinned allocator. This only changes the
                # isolated fixture's collator, never the production pin policy.
                v24_input_runtime._PIN_OUTPUTS = False
                provider = MatchedInputProvider([SimpleNamespace(query_rows=rows)],
                    lambda row: lookup[row['id']], SimpleNamespace(check=lambda: None), workers=4)
                try:
                    batch = provider.get([0, 1, 2], epoch=29)
                    assert batch.graph.num_graphs == 6
                    assert all(value == profile for value in batch.graph['v25_context_contract'])
                    assert provider.profile_rows['observations'] == 3
                finally: provider.close()
            F.tearDownClass()
            print('CPU_UNIT_ACTUAL_V25_MATCHED_SOURCE_TARGET_TWO_VIEWS_PASS')
        ''')
        result = subprocess.run([sys.executable, '-X', 'utf8', '-B', '-c', code],
            cwd=Path(__file__).resolve().parents[1], capture_output=True,
            text=True, encoding='utf8', errors='replace', timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('CPU_UNIT_ACTUAL_V25_MATCHED_SOURCE_TARGET_TWO_VIEWS_PASS', result.stdout)


if __name__ == '__main__':
    unittest.main()
