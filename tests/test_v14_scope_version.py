"""Version-display regression only; no neural or accuracy validation claim."""
from __future__ import annotations

import copy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import uuid

from tools import run_v14_scope_training as wrapper


class VersionDisplayUnits(unittest.TestCase):
    def setUp(self):
        self.root = wrapper.ROOT / 'work' / ('v14_version_UNIT_' + uuid.uuid4().hex)
        self.root.mkdir(parents=True)
        self.manifest = dict(margin_mm=10., contract_sha256='a' * 64)

    def test_version_is_a_native_branch_and_does_not_mutate_training_contract(self):
        before = copy.deepcopy(self.manifest)
        result = wrapper.version_record(self.manifest)
        self.assertEqual(self.manifest, before)
        self.assertEqual(result['experiment_version'], 'v1.4')
        self.assertEqual(result['forked_from'], 'v1.0')
        self.assertEqual(result['baseline_model_version'], 'v1.0')
        self.assertFalse(result['includes_v11_v12_v13_changes'])
        self.assertEqual(result['results_relative_path'], 'results/v1.0')
        self.assertEqual(result['scope_contract_sha256'], self.manifest['contract_sha256'])

    def test_version_sidecar_preserves_existing_manifest_and_checkpoint_bytes(self):
        manifest = self.root / 'manifest.json'
        checkpoint = self.root / 'results/v1.0/checkpoint_best.last.pt'
        checkpoint.parent.mkdir(parents=True)
        # Explicit UNIT bytes exercise preservation only, never loaded as a model.
        manifest.write_bytes(b'UNIT original manifest bytes')
        checkpoint.write_bytes(b'UNIT original checkpoint bytes')
        before = {p: p.read_bytes() for p in (manifest, checkpoint)}
        with patch.object(wrapper.controller, 'verify_bound_experiment') as verify:
            path = wrapper.record_version(self.root, self.manifest)
            self.assertEqual(wrapper.record_version(self.root, self.manifest), path)
            self.assertEqual(verify.call_count, 2)
        self.assertEqual(json.loads(path.read_text())['experiment_version'], 'v1.4')
        for path, data in before.items():
            self.assertEqual(path.read_bytes(), data)

    def test_different_record_is_rejected_without_overwriting_it(self):
        path = self.root / 'experiment_version.json'
        path.write_text('{"experiment_version":"UNIT another version"}', encoding='utf8')
        before = path.read_bytes()
        with patch.object(wrapper.controller, 'verify_bound_experiment'), self.assertRaises(ValueError):
            wrapper.record_version(self.root, self.manifest)
        self.assertEqual(path.read_bytes(), before)

    def test_failed_contract_verification_never_creates_version_sidecar(self):
        with patch.object(wrapper.controller, 'verify_bound_experiment', side_effect=ValueError('UNIT tamper')):
            with self.assertRaises(ValueError):
                wrapper.record_version(self.root, self.manifest)
        self.assertFalse((self.root / 'experiment_version.json').exists())

    def run_with_mocked_training(self, codes):
        # Request dispatch only; no fake training result or checkpoint is created.
        with patch.object(wrapper.controller, 'verify_bound_experiment'), \
                patch.object(wrapper.controller, 'request_for',
                    side_effect=lambda root, manifest, phase: dict(phase=phase, contract_sha256=manifest['contract_sha256'])), \
                patch.object(wrapper.subprocess, 'Popen') as launch, redirect_stdout(io.StringIO()) as out:
            launch.return_value.wait.side_effect = codes
            if codes[0]:
                with self.assertRaises(RuntimeError):
                    wrapper.run(self.root, self.manifest)
            else:
                wrapper.run(self.root, self.manifest)
        self.assertFalse((self.root / 'run.lock').exists())
        return launch, out.getvalue()

    def test_same_single_prepare_and_train_commands_without_native_or_30mm_run(self):
        launch, output = self.run_with_mocked_training([0, 0])
        phases = []
        for call in launch.call_args_list:
            command = call.args[0]
            self.assertEqual(command[:4], [wrapper.sys.executable, '-u', '-m', 'hiercp_v1x.scope_training_entry'])
            request = json.loads(Path(command[-1]).read_text())
            phases.append(request['phase'])
            self.assertEqual(request['contract_sha256'], self.manifest['contract_sha256'])
            self.assertEqual(call.kwargs['cwd'], wrapper.ROOT)
            self.assertEqual(call.kwargs['env']['PYTHONDONTWRITEBYTECODE'], '1')
            self.assertNotIn('HIERCP_V1X_SAMPLING_CONTRACT', call.kwargs['env'])
        self.assertEqual(phases, ['prepare', 'train'])
        self.assertIn('v1.4 | native v1.0 branch | margin=10mm', output)
        self.assertIn('v1.4 RESULTS:', output)

    def test_failed_prepare_never_launches_training(self):
        launch, _ = self.run_with_mocked_training([1])
        self.assertEqual(launch.call_count, 1)
        self.assertEqual(json.loads(Path(launch.call_args.args[0][-1]).read_text())['phase'], 'prepare')

    def test_frozen_8c7e914_execution_helpers_still_match_preserved_receipt(self):
        receipt = wrapper.controller.read(wrapper.ROOT / 'validation/v1_scope_learning_20261004/verification.json')
        expected = receipt['verified_current_execution_source_sha256']
        for name in wrapper.controller.HELPERS:
            with self.subTest(helper=name):
                self.assertEqual(wrapper.controller.digest(wrapper.ROOT / name), expected[name])


if __name__ == '__main__':
    unittest.main()
