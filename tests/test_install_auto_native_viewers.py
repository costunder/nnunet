"""DEBUG CPU fixtures for fresh viewer installation, no server/jobs/models."""
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('install_auto_native_viewers_DEBUG',
    ROOT / 'outputs/v24_gpu6_recovery_20261011/install_auto_native_viewers_NEW.py')
helper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(helper)


def write(filename, value):
    filename.write_text(json.dumps(value), encoding='utf8')


class FreshAutoViewerCPU_DEBUG(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory(prefix='auto_viewer_DEBUG_', dir=ROOT / 'outputs')
        self.addCleanup(self.folder.cleanup)
        self.base = Path(self.folder.name)
        code = self.base / 'NEW_code'; (code / 'tools').mkdir(parents=True)
        hashes = {}
        for name in helper.MODULES:
            filename = code / name
            filename.write_text('"""DEBUG module fixture, never execute training."""\n', encoding='utf8')
            hashes[name] = helper.sha(filename.read_bytes())
        python = self.base / 'DEBUG_python_not_executed'
        python.write_text('DEBUG never execute', encoding='utf8')
        scientific = self.base / 'preserved_science'; scientific.mkdir()
        arms = {}
        for number in ('4', '5', '6'):
            score = self.base / ('score' + number)
            native = self.base / ('future_native' + number)
            arms[number] = dict(scoring_root=str(score), native_root=str(native))
            if number != '6':
                direct = self.base / ('direct' + number); (direct / 'gnn').mkdir(parents=True)
                arms[number]['arm_root'] = str(direct)
                write(direct / 'direct_training_request.json', dict(
                    format='v24_existing_direct_GPU_exact_continuation_request_v1', GPU_arm=int(number),
                    output=str(direct / 'gnn'), source_code=str(scientific)))
            else:
                score.mkdir(); source = self.base / 'actual_GPU6_gnn'; source.mkdir()
                write(score / 'own_best_pin.json', dict(format='v24_completed_single_arm_GT_blind_best_v1',
                    physical_GPU=6, source_output=str(source), source_code=str(scientific),
                    full_training=True, debug=False, completed_epochs=40, optimizer_updates=680))
                write(score / 'request.json', dict(format='v24_full642_depth1_current_BEST_scoring_request_v1',
                    gpu=6, pin_path=str(score / 'own_best_pin.json'), output=str(score / 'bank')))
        self.config = dict(format=helper.FORMAT, code_root=str(code), python=str(python),
            viewer_directory=str(self.base / 'NEW_viewers'), modules_sha256=hashes, arms=arms)

    def test_CPU_check_creates_nothing_and_GPU6_source_is_derived(self):
        filename = self.base / 'config.json'; write(filename, self.config)
        with contextlib.redirect_stdout(io.StringIO()) as stream:
            helper.main(['--config', str(filename), '--config-sha', helper.sha(filename.read_bytes()), '--check'])
        result = json.loads(stream.getvalue())
        self.assertFalse(result['files_created'])
        self.assertFalse(Path(self.config['viewer_directory']).exists())
        self.assertEqual(result['prepared']['arms']['6']['current_gnn'], str(self.base / 'actual_GPU6_gnn'))
        self.assertEqual(result['prepared']['arms']['6']['args'][:4],
            ['--gnn-root', str(self.base / 'actual_GPU6_gnn'), '--gpu', '6'])

    def test_fresh_three_wrappers_compile_and_second_install_preserves_bytes(self):
        prepared = helper.validate(self.config)
        result = helper.install(prepared)
        originals = {row['path']: Path(row['path']).read_bytes() for row in result['created']}
        self.assertEqual({Path(name).name for name in originals},
            {'view_gpu4_auto.py', 'view_gpu5_auto.py', 'view_gpu6_auto.py'})
        for name, data in originals.items():
            compile(data, name, 'exec')
            self.assertIn(b'Pinned viewer code SHA/stat/UID', data)
            self.assertIn(b'--native-root', data)
        with self.assertRaises(FileExistsError):
            helper.validate(self.config)
        with self.assertRaises(FileExistsError):
            helper.install(prepared)
        self.assertEqual(originals, {name: Path(name).read_bytes() for name in originals})
        self.assertFalse(result['jobs_started'])

    def test_module_SHA_change_prevents_any_output(self):
        module = Path(self.config['code_root']) / helper.MODULES[0]
        module.write_text('# DEBUG modified\n', encoding='utf8')
        with self.assertRaisesRegex(ValueError, 'SHA/stat differs'):
            helper.validate(self.config)
        self.assertFalse(Path(self.config['viewer_directory']).exists())

    def test_GPU6_cross_arm_pin_or_configured_source_is_refused(self):
        path = self.base / 'score6/own_best_pin.json'
        pin = helper.record(path); pin['physical_GPU'] = 5; write(path, pin)
        with self.assertRaisesRegex(ValueError, 'full40/680 BEST pin'):
            helper.validate(self.config)
        pin['physical_GPU'] = 6; write(path, pin)
        self.config['arms']['6']['gnn_root'] = str(self.base / 'incorrect_source')
        with self.assertRaisesRegex(ValueError, 'configured source differs'):
            helper.validate(self.config)

    def test_registered_native_must_match_current_GNN_and_scoring(self):
        native = Path(self.config['arms']['4']['native_root']); native.mkdir()
        write(native / 'request.json', dict(format='v24_scored_bank_to_native_chain_request_v1',
            chain_root=str(native), GPU_arm=4, source_output=str(self.base / 'direct4/gnn'),
            source_code=str(self.base / 'preserved_science'), scoring_root=str(self.base / 'score4')))
        self.assertIn('4', helper.validate(self.config)['arms'])
        request = helper.record(native / 'request.json'); request['GPU_arm'] = 5
        write(native / 'request.json', request)
        with self.assertRaisesRegex(ValueError, 'native controller source differs'):
            helper.validate(self.config)

    def test_partial_metadata_and_wrong_config_SHA_create_nothing(self):
        (self.base / 'score6/request.json').write_text('{', encoding='utf8')
        with self.assertRaisesRegex(ValueError, 'JSON incomplete'):
            helper.validate(self.config)
        filename = self.base / 'config.json'; write(filename, self.config)
        with self.assertRaisesRegex(ValueError, 'raw SHA256 differs'):
            helper.main(['--config', str(filename), '--config-sha', '0' * 64])
        self.assertFalse(Path(self.config['viewer_directory']).exists())


if __name__ == '__main__':
    unittest.main()
