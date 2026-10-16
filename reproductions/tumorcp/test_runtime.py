"""Static/source contract tests; these are not CT/CUDA or quality evidence."""
import ast
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from runtime import CP_MODE_SOURCE, PACK_TUMORS_SOURCE, build_runtime


HERE = Path(__file__).resolve().parent


class RuntimeContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # The managed Windows sandbox may allow creation of its temp root but
        # reject nested writes there; keep these owned test artifacts in-workspace.
        artifact_root = HERE / 'test_artifacts'
        artifact_root.mkdir(exist_ok=True)
        cls.base = Path(tempfile.mkdtemp(prefix='runtime-contract-', dir=artifact_root))
        cls.destination = cls.base / 'runtime'
        cls.upstream = HERE / 'upstream'
        cls.before = {p.relative_to(cls.upstream).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in (cls.upstream / 'nnunet').rglob('*.py')}
        cls.receipt = build_runtime(cls.upstream, cls.destination)

    def test_event_schedule_is_single_and_exactly_balanced(self):
        scope = {}
        exec(compile(CP_MODE_SOURCE, '<CP mode pure function>', 'exec'), scope)
        choose = scope['_paper_tumorcp_mode']
        counts = {None: 0, 'intra': 0, 'inter': 0}
        for gate in range(100):
            for mode in range(100):
                counts[choose((gate + .5) / 100, (mode + .5) / 100, .8)] += 1
        self.assertEqual(counts, {None: 2000, 'intra': 4000, 'inter': 4000})
        for invalid in (-.1, 1., 2.):
            with self.assertRaises(ValueError):
                choose(invalid, .5, .8)

    def test_single_equal_and_ragged_tumors_preserve_numeric_interpolation(self):
        import io
        import numpy as np
        from scipy.ndimage import map_coordinates
        scope = {'np': np}
        exec(compile(PACK_TUMORS_SOURCE, '<tumor packing helper>', 'exec'), scope)
        pack = scope['_paper_pack_tumors']
        a = np.arange(27, dtype=np.float32).reshape(3, 3, 3)
        b = np.arange(64, dtype=np.float32).reshape(4, 4, 4)
        ma, mb = np.full(a.shape, 2, dtype=np.uint8), np.full(b.shape, 2, dtype=np.uint8)
        for components in ([[a, ma]], [[a, ma], [a.copy(), ma.copy()]], [[a, ma], [b, mb]]):
            packed = pack(components)
            self.assertEqual(packed.shape, (len(components), 2))
            stream = io.BytesIO()
            np.save(stream, packed)
            stream.seek(0)
            loaded = np.load(stream, allow_pickle=True)
            for actual, expected in zip(loaded, components):
                data, mask = actual
                self.assertEqual(data.dtype, np.dtype('float32'))
                self.assertEqual(mask.dtype, np.dtype('uint8'))
                np.testing.assert_array_equal(data, expected[0])
                np.testing.assert_array_equal(mask, expected[1])
                self.assertTrue(np.isfinite(map_coordinates(data, np.array([[1.], [1.], [1.]]), order=3)).all())

    def test_upstream_preserved(self):
        after = {p.relative_to(self.upstream).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in (self.upstream / 'nnunet').rglob('*.py')}
        self.assertEqual(self.before, after)

    def test_original_architecture_and_optimizer_are_unchanged(self):
        relative = 'training/network_training/nnUNetTrainerV2_ImgDAObjCPAllInter.py'
        self.assertEqual((self.upstream / 'nnunet' / relative).read_bytes(),
                         (self.destination / 'nnunet' / relative).read_bytes())
        text = (self.destination / 'nnunet' / relative).read_text()
        self.assertIn('self.network = Generic_UNet(', text)
        self.assertIn('torch.optim.SGD(', text)

    def test_loader_exact_paper_arguments_and_copy_on_write(self):
        text = (self.destination / 'nnunet/training/dataloading/dataset_loading.py').read_text()
        self.assertIn("key != i", text)
        self.assertIn("paper_elastic_alpha", text)
        self.assertIn("paper_elastic_sigma", text)
        self.assertIn('memmap_mode="c"', text)
        ast.parse(text)

    def test_holdout_monitor_and_final_contract(self):
        text = (self.destination / 'nnunet/training/network_training/nnUNetTrainerV2_TumorCPPaper.py').read_text()
        tree = ast.parse(text)
        klass = next(n for n in tree.body if isinstance(n, ast.ClassDef))
        methods = {n.name for n in klass.body if isinstance(n, ast.FunctionDef)}
        self.assertTrue({'do_split', 'process_plans', 'enter_evaluation', 'validate'} <= methods)
        self.assertIn('chosen = holdout if self._paper_holdout_evaluation else train', text)
        self.assertIn('self.load_final_checkpoint(train=False)', text)
        self.assertIn("kwargs['run_postprocessing_on_folds'] = False", text)
        self.assertIn('(500, 250, 50)', text)

    def test_modern_checkpoint_loading_is_explicit(self):
        for relative in ('nnunet/training/network_training/network_trainer.py',
                         'nnunet/training/model_restore.py'):
            self.assertIn('weights_only=False', (self.destination / relative).read_text())

    def test_idempotent_verified_reuse(self):
        self.assertEqual(self.receipt, build_runtime(self.upstream, self.destination))
        receipt = json.loads((self.destination / 'runtime_receipt.json').read_text())
        self.assertFalse(receipt['training_started'])
        self.assertFalse(receipt['quality_verified'])

    def test_unrelated_destination_rejected(self):
        unrelated = self.base / 'unrelated'
        unrelated.mkdir()
        with self.assertRaises(FileExistsError):
            build_runtime(self.upstream, unrelated)

    def test_z_changed_runtime_is_rejected(self):
        own_path = self.destination / 'nnunet/cp_configs.py'
        with own_path.open('a', encoding='utf-8') as handle:
            handle.write('\n# Deliberate own-test byte change\n')
        with self.assertRaises(ValueError):
            build_runtime(self.upstream, self.destination)


if __name__ == '__main__':
    unittest.main(verbosity=2)
