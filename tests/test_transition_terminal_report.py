"""CPU metadata UNIT fixtures only; no CT, model, GPU or training execution."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from hiercp_v1x.transition_terminal_report import build_terminal_summary, render_terminal_summary


class TransitionTerminalMetadataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='transition_terminal_UNIT_')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.receipt = dict(quality_verified=False, production_ready=False, training_started=False,
            neural_forward_executed=False, exact_native_recipe_bound=False, input_files=[],
            required_fields_missing=['native.source_root_matching_checkpoint'], errors=[], bindings={}, sources=[],
            native_checkpoint='/server/UNIT/checkpoint_latest.pt',
            checkpoint_identity=dict(local_cnn={'margin_mm': 10, 'channels': [12, 24, 32]},
                config={'label_definition': ['U unobserved', 'P observed'], 'task_layers': 2},
                base={'training': {'lr': .0001}}, ranking={'ranking_weight': 1},
                epochs=40, workers=16, candidates=[32], precision='FP32',
                support_training={'patients': 16}, resource_limits={'cuda_bytes': 40 * 2**30}),
            checkpoint_state=dict(epoch=22, step=11872, phase='optimization', next_batch=146, batch=32,
                initial_validation={'ranking_mrr': .1, 'ranking_recall_at_1': 0, 'ranking_pairwise_loss': .69},
                validation_history=[dict(epoch=22, ranking_mrr=.2, ranking_recall_at_1=.01, ranking_pairwise_loss=.68)]),
            inventory=dict(records=14102, unique_ids=14102, observed_positive=662, unobserved_comparison=13440,
                split={'inner_train': ['case_' + str(i) for i in range(84)],
                       'inner_val': ['case_' + str(i) for i in range(84, 105)], 'outer': ['case_outer']},
                per_case=[dict(case_id='case_' + str(i), P=6 + (i < 32), U=128) for i in range(105)]))
        self.write('native/experiment.json', {'request': {'settings': {'workers': 16, 'cuda_gib': 40,
            'rss_gib': 192, 'resident_gib': 128, 'device_cache_gib': 8, 'support_patients': 16}}})
        self.write('native/attempt/execution_contract.json', {'physical_batch': 32, 'effective_batch': 32, 'accumulation': 1})
        self.write('native/attempt/learning_schedule.json', {'optimization_steps': 533, 'actual_batch_sizes': [32] * 533})
        self.write('native/checkpoint_metadata.json', {'payload': {'optimizer': {'param_groups': [{'lr': .00005}]}}})

    def write(self, member, value, *, jsonl=False, listed=True):
        path = self.root / member
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = ('\n'.join(json.dumps(row) for row in value) if jsonl else json.dumps(value)).encode('utf8')
        path.write_bytes(raw)
        if listed:
            self.receipt['input_files'] = [entry for entry in self.receipt['input_files'] if entry['member'] != member]
            self.receipt['input_files'].append(dict(member=member, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()))

    def telemetry(self, role='baseline', result='v1.0', name='epoch_telemetry_UNIT.jsonl', epochs=40):
        rows = [dict(event='EpochPostRun', epoch=i, validation={'mrr': 1., 'acc': 1., 'margin': i, 'loss': .1},
                     train={'mrr': 1., 'acc': 1., 'margin': i, 'loss': .01}, best_epoch=11 if i >= 11 else i,
                     telemetry={'epoch_wall_seconds': 245.}) for i in range(1, epochs + 1)]
        rows.append(dict(event='PostRun', full_training_complete=True, status='complete'))
        self.write(f'comparisons/{role}/results/{result}/{name}', rows, jsonl=True)
        return rows

    def summary(self):
        return build_terminal_summary(self.receipt, self.root)

    def test_complete_cursor_and_exact_native_configuration_are_preserved(self):
        summary = self.summary()
        native = summary['native']
        self.assertEqual(native['saved_cursor']['step'], 11872)
        self.assertEqual(native['saved_cursor']['batch'], 32)
        self.assertEqual(native['checkpoint_identity']['local_cnn']['channels'], [12, 24, 32])
        self.assertEqual(native['wrapper_request']['settings']['device_cache_gib'], 8)
        self.assertEqual(native['configured_learning_rate'], .0001)
        self.assertEqual(native['current_optimizer_learning_rates'], [.00005])
        self.assertEqual(native['latest_validation']['metrics']['MRR'], .2)

    def test_full_inventory_and_all_split_case_counts_are_used(self):
        summary = self.summary()['native']
        self.assertEqual(summary['full_inventory']['records'], 14102)
        self.assertEqual(len(summary['full_inventory']['per_case']), 105)
        self.assertEqual(summary['split_counts']['inner_train']['U'], 84 * 128)
        self.assertEqual(summary['split_counts']['inner_val']['U'], 21 * 128)
        self.assertEqual(sum(summary['split_counts'][key]['records'] for key in ('inner_train', 'inner_val')), 14102)
        self.assertEqual(summary['split_counts']['outer']['inventoried_cases'], 0)

    def test_all_40_epoch_rows_and_reported_checkpoint_selection(self):
        self.telemetry()
        result = self.summary()['comparisons'][0]
        self.assertEqual(result['recorded_epochs'], list(range(1, 41)))
        self.assertTrue(result['all_40_epoch_records_present'])
        self.assertEqual(result['best']['epoch'], 11)
        self.assertEqual(result['best']['validation']['margin'], 11)
        self.assertEqual(result['latest']['epoch'], 40)
        self.assertTrue(result['full_training_complete_reported'])

    def test_each_original_comparison_arm_is_read(self):
        self.telemetry()
        self.telemetry('half_A', 'half_A')
        self.telemetry('half_B', 'half_B')
        for result in self.summary()['comparisons']:
            self.assertEqual(result['epoch_records'], 40)
            self.assertEqual(result['latest']['validation']['MRR'], 1)

    def test_missing_metrics_are_unknown_not_zero(self):
        result = self.summary()['comparisons'][0]
        self.assertIsNone(result['best'])
        self.assertIsNone(result['latest'])
        text = render_terminal_summary(self.summary())
        self.assertIn('baseline initial | MRR=UNKNOWN', text)
        self.assertIn('baseline best epoch=UNKNOWN', text)

    def test_missing_optimizer_lr_is_unknown(self):
        self.write('native/checkpoint_metadata.json', {'payload': {}})
        self.assertIn('saved optimizer LR: UNKNOWN', render_terminal_summary(self.summary()))

    def test_malformed_saved_metadata_has_explicit_failure(self):
        member = 'comparisons/baseline/results/v1.0/epoch_telemetry_BAD.jsonl'
        path = self.root / member
        path.parent.mkdir(parents=True)
        path.write_bytes(b'{not json')
        self.receipt['input_files'].append(dict(member=member, bytes=len(path.read_bytes()),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        summary = self.summary()
        self.assertEqual(len(summary['parsing_failures']), 1)
        self.assertEqual(summary['parsing_failures'][0]['reason'], 'metadata_parse_failed')
        self.assertIsNone(summary['comparisons'][0]['latest'])

    def test_changed_saved_evidence_is_not_reported_as_valid_metric(self):
        self.telemetry()
        member = 'comparisons/baseline/results/v1.0/epoch_telemetry_UNIT.jsonl'
        (self.root / member).write_text('{}', encoding='utf8')
        summary = self.summary()
        self.assertEqual(summary['parsing_failures'][0]['reason'], 'saved_evidence_hash_or_size_differs')
        self.assertIsNone(summary['comparisons'][0]['latest'])

    def test_conflicting_latest_epoch_is_unknown_instead_of_older_fallback(self):
        rows = self.telemetry()
        rows[-2]['validation']['mrr'] = .5
        self.write('comparisons/baseline/results/v1.0/epoch_telemetry_CONFLICT.jsonl', [rows[-2]], jsonl=True)
        result = self.summary()['comparisons'][0]
        self.assertFalse(result['all_40_epoch_records_present'])
        self.assertEqual(result['latest_recorded_epoch'], 40)
        self.assertIsNone(result['latest'])
        self.assertIsNone(result['best'])
        self.assertEqual(len(result['conflicts']), 1)

    def test_duplicate_identical_epoch_metrics_do_not_inflate_epoch_count(self):
        rows = self.telemetry()
        self.write('comparisons/baseline/results/v1.0/epoch_telemetry_RESUME.jsonl', rows, jsonl=True)
        result = self.summary()['comparisons'][0]
        self.assertEqual(result['epoch_records'], 40)
        self.assertFalse(result['conflicts'])
        self.assertEqual(len(result['latest']['evidence_members']), 2)

    def test_reporter_does_not_change_input_or_write_files(self):
        before = {path.relative_to(self.root).as_posix(): path.read_bytes() for path in self.root.rglob('*') if path.is_file()}
        summary = self.summary()
        render_terminal_summary(summary)
        after = {path.relative_to(self.root).as_posix(): path.read_bytes() for path in self.root.rglob('*') if path.is_file()}
        self.assertEqual(before, after)

    def test_many_missing_source_fields_preserved_with_compact_output(self):
        self.receipt['required_fields_missing'] = ['native.source_identity.core/file_' + str(i) for i in range(500)]
        self.receipt['sources'] = [dict(root='/server/root', expected_files=500, mismatches=[{'file': str(i)} for i in range(500)],
            required_source_identity_files_missing=list(range(500)), matches_checkpoint_source=False)]
        summary = self.summary()
        self.assertEqual(len(summary['required_fields_missing']), 500)
        self.assertEqual(len(summary['source_checks'][0]['mismatches']), 500)
        text = render_terminal_summary(summary)
        self.assertLess(len(text.splitlines()), 80)
        self.assertIn('Missing fields: 500', text)
        self.assertNotIn('file_499', text)

    def test_no_neural_or_quality_promotion(self):
        summary = self.summary()
        for flag in ('quality_verified', 'production_ready', 'training_started', 'neural_forward_executed', 'full_evaluation_executed'):
            self.assertFalse(summary[flag])
        self.assertEqual(summary['optimizer_updates'], 0)
        self.receipt['quality_verified'] = True
        with self.assertRaisesRegex(ValueError, 'quality_verified'):
            self.summary()

    def test_unsafe_saved_member_rejected(self):
        self.receipt['input_files'].append({'member': 'comparisons/baseline/results/v1.0/../../../../bad.json'})
        with self.assertRaisesRegex(ValueError, 'Unsafe saved metadata'):
            self.summary()

    def test_original_initial_validation_schema(self):
        self.write('comparisons/half_A/results/half_A/initial_validation.json',
                   {'metrics': {'mrr': .301521, 'acc': .055556, 'margin': -.014954}, 'full_validation': True})
        initial = self.summary()['comparisons'][1]['initial_validation']
        self.assertEqual(initial['metrics']['MRR'], .301521)
        self.assertTrue(initial['full_validation'])

    def test_nonfinite_metric_is_unknown(self):
        rows = self.telemetry()
        rows[-2]['validation']['mrr'] = float('nan')
        self.write('comparisons/baseline/results/v1.0/epoch_telemetry_UNIT.jsonl', rows, jsonl=True)
        latest = self.summary()['comparisons'][0]['latest']
        self.assertIsNone(latest['validation']['MRR'])


if __name__ == '__main__':
    unittest.main()
