"""UNIT telemetry fixtures; no training, checkpoint loading or file mutations."""
import io
import json
import os
from pathlib import Path
import tempfile
import unittest

from tools.report_comparison_runtime import (build_report, parse_jsonl, summarize_cache,
    render_text, summarize_input, summarize_training, inspect_arm, _read_gpu_execution_logs)


class ComparisonRuntimeReportUnitTests(unittest.TestCase):
    @staticmethod
    def _gpu_line(event, time, **extra):
        return json.dumps(dict(event=event, time=time, **extra)) + '\n'

    def test_gpu_legacy_and_invocation_logs_merge_by_event_time_in_archived_arm(self):
        with tempfile.TemporaryDirectory(prefix='UNIT_runtime_archive_') as name:
            arm = Path(name) / 'archived_run' / 'selected'
            events = arm / 'gpu_execution_events'
            events.mkdir(parents=True)
            calibration = dict(baseline_seconds=4., selected_seconds=2., measured_speedup=2.,
                               baseline_peak_cuda_bytes=1024, selected_peak_cuda_bytes=1024)
            (arm / 'gpu_execution.jsonl').write_text(
                self._gpu_line('calibration_completed', 10, calibration=calibration), encoding='utf8')
            (events / 'a_latest.jsonl').write_text(
                self._gpu_line('calibration_completed', 40, calibration=calibration) +
                self._gpu_line('context_bound_policy_cache_loaded', 41, measured_workloads=2), encoding='utf8')
            (events / 'z_earlier.jsonl').write_text(
                self._gpu_line('optimized_execution_OOM_original_retry', 20) +
                self._gpu_line('original_policy_cache_admission_skipped', 21), encoding='utf8')
            (events / 'not_an_event.jsonl.tmp').write_text('unfinished ignored staging file', encoding='utf8')
            before = {str(path): (path.stat().st_mtime_ns, path.read_bytes()) for path in arm.rglob('*') if path.is_file()}
            report = inspect_arm(arm)
            after = {str(path): (path.stat().st_mtime_ns, path.read_bytes()) for path in arm.rglob('*') if path.is_file()}
            self.assertEqual(before, after)
            gpu = report['gpu_execution']
            self.assertEqual(gpu['measured_batches'], 2)
            self.assertEqual(gpu['latest_calibration']['time'], 40)
            self.assertEqual(gpu['latest_event']['time'], 41)
            self.assertEqual(gpu['original_execution_retries'], 1)
            self.assertEqual(gpu['policy_cache_load_events'], 1)
            self.assertEqual(gpu['policy_cache_admission_skips'], 1)
            self.assertEqual(len(gpu['log_sources']), 3)
            self.assertTrue(gpu['complete_read'])
            self.assertIn('reused prior measurements', render_text(report))
            self.assertTrue(report['read_only'])
            self.assertFalse(report['checkpoint_loaded'])

    def test_new_gpu_event_directory_works_without_legacy_log_and_each_file_read_once(self):
        with tempfile.TemporaryDirectory(prefix='UNIT_gpu_event_alias_') as name:
            arm = Path(name)
            events = arm / 'gpu_execution_events'
            events.mkdir()
            source = events / 'original.jsonl'
            source.write_text(self._gpu_line('context_bound_policy_cache_loaded', 7), encoding='utf8')
            os.link(source, events / 'same_file_alias.jsonl')
            warnings = []
            rows, details = _read_gpu_execution_logs(arm, warnings)
            self.assertEqual(len(rows), 1)
            self.assertEqual(len(details['log_sources']), 1)
            self.assertEqual(warnings, [])

    def test_gpu_damage_notices_survive_merge_and_other_valid_events_remain_visible(self):
        with tempfile.TemporaryDirectory(prefix='UNIT_gpu_partial_') as name:
            arm = Path(name)
            events = arm / 'gpu_execution_events'
            events.mkdir()
            (arm / 'gpu_execution.jsonl').write_text(
                self._gpu_line('calibration_started', 1) + '{"event":broken}\n' +
                self._gpu_line('optimized_execution_OOM_original_retry', 2), encoding='utf8')
            (events / 'new.jsonl').write_text(
                self._gpu_line('context_bound_policy_cache_loaded', 3) + '{"event":', encoding='utf8')
            report = inspect_arm(arm)
            gpu = report['gpu_execution']
            self.assertFalse(gpu['complete_read'])
            self.assertEqual(gpu['excluded_invalid_lines'], 2)
            self.assertEqual(gpu['original_execution_retries'], 1)
            self.assertEqual(gpu['policy_cache_load_events'], 1)
            self.assertTrue(any('gpu_execution.jsonl:2: invalid GPU JSONL line excluded' in row for row in report['warnings']))
            self.assertTrue(any('new.jsonl:2: unfinished final append ignored' in row for row in report['warnings']))
            self.assertIn('GPU audit coverage is partial', render_text(report))

    def test_gpu_missing_time_warns_and_does_not_replace_latest_dated_measurement(self):
        with tempfile.TemporaryDirectory(prefix='UNIT_gpu_undated_') as name:
            arm = Path(name)
            (arm / 'gpu_execution.jsonl').write_text(
                self._gpu_line('calibration_started', 4) +
                json.dumps(dict(event='calibration_started', marker='undated')) + '\n', encoding='utf8')
            warnings = []
            rows, details = _read_gpu_execution_logs(arm, warnings)
            self.assertEqual(rows[-1]['time'], 4)
            self.assertEqual(details['records_without_timestamp'], 1)
            self.assertTrue(any('chronological position unknown' in row for row in warnings))

    def test_cold_staging_capacity_is_not_reported_as_measured_utilization(self):
        report = build_report({}, [], [], [], [])
        report['warnings'] = []
        report['cpu_staging'] = dict(latest_mode=dict(mode='cold_shared',
            planned_slots=6, current_admission_slots=2, candidate_workers=16,
            certified_cold_concurrency=True), latest_finished_segment=dict(
                cold_batches=80, warm_batches=2, peak_queued_batches=4))
        text = render_text(report)
        self.assertIn('cold_shared', text)
        self.assertIn('planned slots=6 current admission=2', text)
        self.assertIn('not measured active workers', text)
        self.assertIn('not completed-epoch totals', text)

    def test_gpu_policy_measurement_is_not_reported_as_epoch_speedup(self):
        report = build_report({}, [], [], [], [])
        report['warnings'] = []
        report['gpu_execution'] = dict(available=True, measured_batches=1,
            latest_calibration=dict(calibration=dict(baseline_seconds=4., selected_seconds=2.,
                measured_speedup=2., baseline_peak_cuda_bytes=2 * 2**30,
                selected_peak_cuda_bytes=6 * 2**30)), original_execution_retries=1,
            latest_update_policy=dict(settings=dict(dense_batch_size=16,
                checkpoint_dense_encoder=False, checkpoint_local_blocks=False),
                reason='measured_current_actual_batch'))
        rendered = render_text(report)
        self.assertIn('forward/backward 4.00->2.00s (2.00x)', rendered)
        self.assertIn('not whole-epoch speedup', rendered)
        self.assertIn('Latest actual update execution', rendered)
        self.assertIn('Same-input original-execution OOM retries: 1', rendered)

    def test_update_log_keeps_actual_gpu_policy_and_setup_timing(self):
        row = dict(status='OPTIMIZER_UPDATED', update=3, epoch=1,
            gpu_execution=dict(reason='measured_current_actual_batch', settings=dict(dense_batch_size=16)),
            execution_calibration_seconds=90., other_unneeded_large_field=['UNIT'])
        rows, warnings = parse_jsonl(io.StringIO(json.dumps(row)+'\n'), kind='update')
        self.assertFalse(warnings)
        self.assertEqual(rows[0]['gpu_execution'], row['gpu_execution'])
        self.assertEqual(rows[0]['execution_calibration_seconds'], 90.)
        self.assertNotIn('other_unneeded_large_field', rows[0])

    def test_inclusive_details_and_compact_cache_are_reported_without_double_counting(self):
        row = dict(arm='native_fixed', training=True, full129=False, view_epoch=2,
            source_indices=[3], status='complete', input_seconds=10.,
            regions=dict(patient_graph_seconds=8.), other_assembly_views_collate_seconds=2.,
            inclusive_details=dict(liver_raw_seconds=dict(seconds=7., calls=2),
                                   inference_assembly_inclusive_seconds=dict(seconds=9., calls=2)),
            compact_upper_cache=dict(liver_raw=dict(calls=2, builds=1, reopens=0, resident_hits=1,
                                                   original_seconds=6., lookup_seconds=7.)))
        rows, notices = parse_jsonl(io.StringIO(json.dumps(row)+'\n'), kind='input')
        self.assertFalse(notices)
        mode = summarize_input(rows, 'native_fixed', 2)['training_epoch']
        self.assertEqual(mode['accounted_region_plus_remainder_seconds'], 10.)
        self.assertEqual(mode['inclusive_details']['timings']['liver_raw_seconds']['observed_calls'], 2)
        self.assertEqual(mode['compact_upper_cache']['kinds']['liver_raw']['builds'], 1)
        old = {key: value for key, value in row.items() if key not in ('inclusive_details', 'compact_upper_cache')}
        mixed = summarize_input([old, row], 'native_fixed', 2)['training_epoch']
        self.assertEqual(mixed['inclusive_details']['records_with_value'], 1)
        self.assertEqual(mixed['inclusive_details']['records_total'], 2)
        self.assertFalse(mixed['inclusive_details']['timings']['liver_raw_seconds']['timing']['complete_measurement'])
        report = build_report({'arm': 'native_fixed'}, [], [], [], [], epoch=2,
                              input_rows=rows, input_available=True)
        report['warnings'] = []
        rendered = render_text(report)
        self.assertIn('Inclusive details (overlap; do not add', rendered)
        self.assertIn('Compact upper cache', rendered)

    def test_training_step_excludes_loader_and_contains_checkpoint(self):
        rows = [dict(status="OPTIMIZER_UPDATED", epoch=1, update=1, sample_indices=[1, 2], physical_samples=2,
                     loader_wait_seconds=3., step_seconds=7., checkpoint_seconds=2., forward_seconds=1.,
                     backward_seconds=4.)]
        summary = summarize_training(rows, 2)
        self.assertEqual(summary["observed_loader_plus_step_seconds"], 10.)
        self.assertEqual(summary["timings"]["checkpoint_seconds"]["percent_of_observed_loader_plus_step"], 20.)
        self.assertEqual(summary["timings"]["loader_wait_seconds"]["percent_of_observed_loader_plus_step"], 30.)
        self.assertEqual(summary["observed_sources_per_second"], .2)

    def test_legacy_phase_missing_is_unknown_even_with_training_100_percent(self):
        update = dict(status="OPTIMIZER_UPDATED", epoch=1, update=1, sample_indices=[1, 2], physical_samples=2,
                      loader_wait_seconds=3., step_seconds=7.)
        result = build_report(dict(training_samples=2, validation_samples=1, total_planned_updates=40), [update], [], [], [])
        self.assertEqual(result["phase"], "unknown")
        self.assertEqual(result["training_epoch"]["coverage"]["percent"], 100.)
        self.assertIn("validation/transition expected", result["expected_next_phase_from_legacy_records"])
        self.assertEqual(result["optimizer_history"]["percent"], 2.5)

    def test_partial_live_scores_are_separate_from_completed_full_validation(self):
        progress = dict(phase="validation129", epoch=1, stage="forward", completed_sources=1, total_sources=4,
                        metrics_scope="partial patient macro of observed sources", metrics=dict(mrr=.5, top1=0.))
        complete = dict(epoch=0, metrics=dict(mrr=.1, top1=.05), source_problems=4)
        result = build_report({}, [], [], [], [], progress=progress, latest_validation=complete)
        self.assertEqual(result["phase"], "validation129")
        self.assertEqual(result["live_progress"]["percent"], 25.)
        self.assertEqual(result["live_progress"]["metrics"]["mrr"], .5)
        self.assertEqual(result["latest_completed_validation"]["metrics"]["mrr"], .1)
        self.assertEqual(result["latest_completed_validation"]["epoch"], 0)

    def test_duplicate_validation_rows_do_not_inflate_coverage(self):
        row = dict(epoch=1, source_indices=[3], source_problems=1, loader_seconds=3., full_joint_forward_seconds=7.)
        result = build_report(dict(validation_samples=2), [], [row, row], [], [])
        summary = result["validation_epoch"]
        self.assertEqual(summary["coverage"]["unique_logged_sources"], 1)
        self.assertEqual(summary["coverage"]["percent"], 50.)
        self.assertEqual(summary["coverage"]["repeated_source_observations"], 1)
        self.assertEqual(summary["observed_blocking_load_plus_forward_seconds"], 20.)

    def test_cache_whole_history_distinguishes_build_reopen_mapping(self):
        rows = [dict(stage="whole_case_fields_read", case_id="UNIT_A", status="built", wall_seconds=10.),
                dict(stage="whole_case_fields_read", case_id="UNIT_A", status="resident_mapping", wall_seconds=.1),
                dict(stage="whole_case_fields_read", case_id="UNIT_B", status="reopened", wall_seconds=20.),
                dict(stage="host_cache_pressure", status="released", wall_seconds=4.)]
        result = summarize_cache(rows)
        self.assertEqual(result["field_calls"], 3)
        self.assertEqual(result["unique_field_case_ids"], 2)
        self.assertEqual(result["summed_field_helper_seconds"]["seconds"], 30.1)
        self.assertEqual(result["pressure_events"], 1)
        self.assertEqual(result["field_statuses"]["built"]["calls"], 1)

    def test_only_unfinished_final_jsonl_append_is_ignored(self):
        rows, warnings = parse_jsonl(io.StringIO('{"epoch":1}\n{"epoch":'), kind="update")
        self.assertEqual(rows, [dict(epoch=1)])
        self.assertEqual(len(warnings), 1)
        with self.assertRaisesRegex(ValueError, "invalid complete JSONL"):
            parse_jsonl(io.StringIO('{"epoch":broken}\n{"epoch":2}\n'))

    def test_missing_measurements_remain_unknown_and_nonfinite_is_rejected(self):
        result = build_report({}, [], [], [], [])
        self.assertIsNone(result["training_epoch"]["observed_loader_plus_step_seconds"])
        self.assertIsNone(result["validation_epoch"]["observed_blocking_load_plus_forward_seconds"])
        self.assertIsNone(result["optimizer_history"]["percent"])
        with self.assertRaisesRegex(ValueError, "Nonfinite JSON"):
            parse_jsonl(io.StringIO('{"wall_seconds":NaN}\n'))

    def test_incomplete_legacy_timing_cannot_claim_observed_throughput(self):
        rows = [dict(status="OPTIMIZER_UPDATED", sample_indices=[1], physical_samples=1,
                     loader_wait_seconds=3., step_seconds=7.),
                dict(status="OPTIMIZER_UPDATED", sample_indices=[2], physical_samples=1,
                     step_seconds=7.)]
        summary = summarize_training(rows, 2)
        self.assertIsNone(summary["observed_loader_plus_step_seconds"])
        self.assertIsNone(summary["observed_sources_per_second"])
        self.assertEqual(summary["timings"]["loader_wait_seconds"]["records_with_value"], 1)

    def test_valid_terminal_phase_overrides_older_live_observation(self):
        result = build_report({}, [], [], [], [], progress=dict(phase="validation129", epoch=2),
                              final_report=dict(final_phase="complete"))
        self.assertEqual(result["phase"], "complete")
        self.assertEqual(result["phase_source"], "final report")
        self.assertEqual(result["live_progress"]["phase"], "validation129")

    def test_prefetch_validation_construction_not_added_to_blocking_wall(self):
        row = dict(epoch=1, source_indices=[3], source_problems=1, loader_seconds=20.,
                   loader_wait_seconds=2., full_joint_forward_seconds=8., checkpoint_seconds=1.,
                   batch_wall_seconds=12.)
        result = build_report(dict(validation_samples=1), [], [row], [], [])
        summary = result["validation_epoch"]
        self.assertEqual(summary["observed_blocking_load_plus_forward_seconds"], 10.)
        self.assertEqual(summary["observed_batch_wall_seconds"], 12.)
        self.assertEqual(summary["timings"]["blocking_loader_seconds"]["percent_of_observed_blocking_load_plus_forward"], 20.)
        self.assertIsNone(summary["timings"]["loader_seconds"]["percent_of_observed_blocking_load_plus_forward"])
        self.assertEqual(summary["prefetch_batch_records"], 1)

    def test_unavailable_optional_logs_are_not_zero_work(self):
        result = build_report({}, [], [], [], [], cache_available=False, input_available=False)
        self.assertFalse(result["cache_history"]["available"])
        self.assertNotIn("field_calls", result["cache_history"])
        self.assertFalse(result["input_preparation"]["available"])

    def test_input_regions_disjoint_and_validation_view_epoch_is_history(self):
        def row(training, epoch, arm="native_listwise"):
            return dict(arm=arm, training=training, full129=not training, view_epoch=epoch,
                        source_indices=[3], status="complete", input_seconds=10.,
                        regions=dict(case_fields_seconds=1., regions_seconds=1., source_seconds=1.,
                                     candidate_metadata_seconds=1., local_graphs_seconds=4.),
                        other_assembly_views_collate_seconds=2.)
        result = summarize_input([row(True, 2), row(True, 1), row(False, 1), row(False, 1),
                                  row(False, 1, "selected")], "native_listwise", 2)
        train, validation = result["training_epoch"], result["validation_history"]
        self.assertEqual(train["input_construction"]["seconds"], 10.)
        self.assertEqual(train["accounted_region_plus_remainder_seconds"], 10.)
        self.assertEqual(train["timings"]["local_graphs_seconds"]["percent_of_cpu_input_construction"], 40.)
        self.assertEqual(validation["input_construction"]["seconds"], 20.)
        self.assertEqual(validation["view_epochs"], [1])
        self.assertEqual(validation["source_coverage"]["repeated_source_observations"], 1)
        self.assertIn("not the outer training epoch", validation["scope"])
        bad = row(True, 2)
        bad["other_assembly_views_collate_seconds"] = 3.
        with self.assertRaisesRegex(ValueError, "do not sum"):
            summarize_input([bad], "native_listwise", 2)

    def test_large_graph_payload_not_retained_by_timing_reader(self):
        rows, warnings = parse_jsonl(io.StringIO('{"epoch":1,"update":1,"actual_input":{"huge":"UNIT"}}\n'), kind="update")
        self.assertNotIn("actual_input", rows[0])
        self.assertFalse(warnings)

    def test_sampled_views_region_survives_reader_and_exact_disjoint_audit(self):
        # UNIT timings exercise the actual sampled_views_seconds log key. The
        # earlier format placed this cost inside the assembly remainder.
        row = dict(arm="native_fixed", training=False, full129=True, view_epoch=29,
                   source_indices=[2], status="complete", input_seconds=10.,
                   regions=dict(case_fields_seconds=1., regions_seconds=1., source_seconds=1.,
                                candidate_metadata_seconds=1., local_graphs_seconds=1.,
                                sampled_views_seconds=3.), other_assembly_views_collate_seconds=2.)
        rows, notices = parse_jsonl(io.StringIO(json.dumps(row) + '\n'), kind="input")
        self.assertFalse(notices)
        mode = summarize_input(rows, "native_fixed", 2)["validation_history"]
        self.assertEqual(mode["accounted_region_plus_remainder_seconds"], 10.)
        self.assertEqual(mode["timings"]["sampled_views_seconds"]["seconds"], 3.)
        self.assertEqual(mode["timings"]["sampled_views_seconds"]["percent_of_cpu_input_construction"], 30.)
        older = dict(row, regions={key: value for key, value in row["regions"].items()
                                  if key != "sampled_views_seconds"}, input_seconds=7.)
        mixed = summarize_input([older, row], "native_fixed", 2)["validation_history"]
        self.assertEqual(mixed["accounted_region_plus_remainder_seconds"], 17.)
        self.assertEqual(mixed["timings"]["sampled_views_seconds"]["records_with_value"], 1)
        self.assertIsNone(mixed["timings"]["sampled_views_seconds"]["percent_of_cpu_input_construction"])
        mismatched = dict(row, other_assembly_views_collate_seconds=3.)
        with self.assertRaisesRegex(ValueError, "do not sum"):
            summarize_input([mismatched], "native_fixed", 2)


if __name__ == "__main__":
    unittest.main()
