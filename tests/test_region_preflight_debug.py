"""Dependency failure boundaries and actual tiny CUDA kernel probe, not model quality."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from tools import v22_region_preflight as preflight


class RegionPreflight(unittest.TestCase):
    def test_matching_wheel_pages(self):
        self.assertEqual(preflight.wheel_page('2.6.0+cu118','11.8'),'https://data.pyg.org/whl/torch-2.6.0+cu118.html')
        self.assertEqual(preflight.wheel_page('2.8.0+cu128','12.8'),'https://data.pyg.org/whl/torch-2.8.0+cu128.html')

    def test_missing_scatter_is_actionable(self):
        actual=preflight.importlib.import_module
        def load(name):
            if name=='torch_scatter':raise ModuleNotFoundError(name)
            return actual(name)
        with patch.object(preflight.importlib,'import_module',side_effect=load):
            with self.assertRaisesRegex(RuntimeError,'torch_scatter.*No cache preparation'):
                preflight.dependencies()

    def test_failure_precedes_output_and_cache(self):
        from tools import run_fixed_regions as runner
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)/'untouched'
            argv=['runner','prepare','--cache','missing.json','--output',str(out),'--partition-checkpoint','missing.pt',
                '--prepare-batch','8','--workers','8','--reg-scale1','.02','--reg-scale2','.02','--view-epoch','0',
                '--cuda-gib','6','--rss-gib','12']
            with patch('sys.argv',argv),patch.object(preflight,'check',side_effect=RuntimeError('missing dependency')),patch.object(runner,'prepare_cache') as prepare:
                with self.assertRaisesRegex(RuntimeError,'missing dependency'):runner.main()
                prepare.assert_not_called();self.assertFalse(out.exists())

    def test_actual_cuda_kernel(self):
        self.assertEqual(preflight.check()['status'],'PASS')


if __name__=='__main__':unittest.main(verbosity=2)
