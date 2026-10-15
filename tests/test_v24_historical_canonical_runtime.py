"""CPU DEBUG checks for byte-identical historical metrics in one Git checkout."""
from contextlib import nullcontext
import hashlib
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools import evaluate_v24_historical_metrics as wrapper


class CanonicalHistoricalRuntimeTests(unittest.TestCase):
    root=Path(__file__).absolute().parents[1]

    def resolving(self):
        # Windows sandbox denies GetFinalPathName for otherwise readable files.
        # Linux production path/identity validation is never adapted.
        return patch.object(Path,'resolve',lambda self,strict=False:self.absolute())if os.name=='nt'else nullcontext()

    def test_exact_historical_sources_and_original_statistics_defaults(self):
        for name,checksum in wrapper.EXPECTED.items():
            self.assertEqual(hashlib.sha256((self.root/'tools/historical_v24'/name).read_bytes()).hexdigest(),checksum)
        with self.resolving():evaluator=wrapper.load_evaluator(self.root)
        options=evaluator.parser().parse_args([])
        self.assertEqual(options.bootstrap_iterations,20000)
        self.assertEqual(options.permutation_iterations,50000)
        self.assertEqual(evaluator.DEFAULT_DATASET_ID,730)
        evaluator.self_test()

    def test_historical_import_alias_is_restored_without_changing_current_module(self):
        current=SimpleNamespace(CPU_DEBUG_existing_module=True)
        with patch.dict(sys.modules,{'tools.online_eval_provenance':current}),self.resolving():
            evaluator=wrapper.load_evaluator(self.root)
            self.assertIs(sys.modules['tools.online_eval_provenance'],current)
        actual=evaluator.build_evaluation_contract.__globals__['__file__']
        self.assertEqual(Path(actual).absolute(),self.root/'tools/historical_v24/online_eval_provenance.py')

    def test_changed_historical_source_is_rejected_before_evaluation(self):
        original=Path.read_bytes
        def read(path):
            return b'CPU_DEBUG_changed_source'if path==self.root/'tools/historical_v24/online_eval_v2.py'else original(path)
        with self.resolving(),patch.object(Path,'read_bytes',read),self.assertRaisesRegex(ValueError,'hash mismatch'):
            wrapper.load_evaluator(self.root)


if __name__=='__main__':unittest.main()
