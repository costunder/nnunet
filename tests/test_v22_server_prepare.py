"""Preparation orchestration only; fixtures do not claim CT/GPU validation."""
import io
import json
from pathlib import Path
import tempfile
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


if __name__=='__main__':unittest.main()
