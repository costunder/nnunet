"""Preparation orchestration only; fixtures do not claim CT/GPU validation."""
import io
import json
from pathlib import Path
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
import unittest
from unittest.mock import Mock, patch
from tools.run_v22_server_prepare import prepare_only, ProgressLog, main


class PreparationTests(unittest.TestCase):
    def test_storage_failure_prevents_graph_preparation(self):
        with tempfile.TemporaryDirectory() as d:
            with patch('tools.v1_server.observations') as obs, \
                 patch('hiercp_v222.v1_cache.configuration',return_value=({'minimum_free_gb':80},{})), \
                 patch('tools.v22_cache_storage.make_plan',return_value={'counts':{}}), \
                 patch('tools.v22_cache_storage.admit',side_effect=OSError('disk full')), \
                 patch('tools.v222_prepare_optimized.prepare') as build:
                with self.assertRaisesRegex(OSError,'disk full'):
                    prepare_only(Path(d),Path(d)/'reference',Path(d),Mock(),Mock())
                obs.assert_called_once()
                build.assert_not_called()
                self.assertFalse((Path(d)/'preparation_complete.json').exists())

    def test_completes_preparation_without_training_and_passes_storage_plan(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);index=root/'prepared.json'
            index.write_text(json.dumps(dict(complete=True,debug=False,records=[{},{}])))
            admission=dict(projected_payload_bytes=1,free_before_bytes=200*1024**3,reserved_free_bytes=80*1024**3)
            with patch('tools.v1_server.observations'), \
                 patch('hiercp_v222.v1_cache.configuration',return_value=({'minimum_free_gb':80},{})), \
                 patch('tools.v22_cache_storage.make_plan',return_value={'counts':{}}), \
                 patch('tools.v22_cache_storage.admit',return_value=admission), \
                 patch('tools.v222_prepare_optimized.prepare',return_value=index) as build:
                result=prepare_only(root,root/'reference',root,Mock(),Mock())
                build.assert_called_once_with(root/'observations/index.json',root/'paired_cache',storage_plan=root/'storage_plan.json')
                self.assertFalse(result['training_started'])
                self.assertFalse(result['G3_passed'])
                self.assertFalse(result['G4_passed'])
                self.assertEqual(result['records'],2)

    def test_existing_output_record_is_never_overwritten(self):
        from tools.run_v22_server_prepare import write_new
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'record.json';write_new(p,{'old':True});before=p.read_bytes()
            with self.assertRaises(FileExistsError):write_new(p,{'new':True})
            self.assertEqual(p.read_bytes(),before)

    def test_progress_preserves_log_and_handles_lifecycle_separately(self):
        from tqdm import tqdm
        log=io.StringIO();bar=tqdm(file=io.StringIO(),disable=False);self.addCleanup(bar.close)
        sink=ProgressLog(log,bar)
        text='human text\n'+json.dumps(dict(event='stage_started',stage='paired_cache'))+'\n'
        sink.write(text[:8]);sink.write(text[8:])
        sink.write(json.dumps(dict(stage='paired_cache',completed=7,total=11,case='case'))+'\n')
        self.assertEqual(bar.n,7);self.assertEqual(bar.total,11)
        self.assertTrue(log.getvalue().startswith(text))

    def progress_sink(self):
        from tqdm import tqdm
        log=io.StringIO();bar=tqdm(file=io.StringIO(),disable=False)
        self.addCleanup(bar.close)
        return ProgressLog(log,bar),log

    def test_two_bodies_before_newlines_reproduces_server_interleaving(self):
        sink,log=self.progress_sink();barrier=threading.Barrier(2)
        def emit(case):
            sink.write(json.dumps(dict(stage='raw_case_started',case=case)))
            barrier.wait(timeout=10)
            sink.write('\n')
            sink.flush()
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(emit,['liver_1','liver_2']))
        rows=[json.loads(line) for line in log.getvalue().splitlines()]
        self.assertEqual({r['case'] for r in rows},{'liver_1','liver_2'})
        self.assertEqual(len(rows),2)
        self.assertEqual(sink.partials,{})

    def test_fragmented_multithread_print_and_log_have_exact_coverage(self):
        sink,log=self.progress_sink();barrier=threading.Barrier(16)
        def emit(worker):
            for step in range(100):
                body=json.dumps(dict(stage='raw_case_started',case=f'{worker}:{step}'))
                # Deliberately interleave JSON fragments, not just complete bodies.
                sink.write(body[:13]);barrier.wait(timeout=15)
                print(body[13:],file=sink,flush=True)
        with ThreadPoolExecutor(max_workers=16) as pool:
            list(pool.map(emit,range(16)))
        rows=[json.loads(line) for line in log.getvalue().splitlines()]
        self.assertEqual(len(rows),1600)
        self.assertEqual({r['case'] for r in rows},{f'{w}:{s}' for w in range(16) for s in range(100)})
        self.assertEqual(sink.partials,{})

    def test_finish_preserves_unterminated_text_without_premature_parsing(self):
        sink,log=self.progress_sink()
        sink.write('partial diagnostic');sink.flush()
        self.assertEqual(log.getvalue(),'')
        sink.finish();sink.finish()
        self.assertEqual(log.getvalue(),'partial diagnostic')
        self.assertEqual(sink.partials,{})

    def test_malformed_complete_event_is_not_silently_swallowed(self):
        sink,log=self.progress_sink()
        with self.assertRaises(json.JSONDecodeError):sink.write('{broken}\n')
        self.assertEqual(log.getvalue(),'{broken}\n')


if __name__=='__main__':unittest.main()
