"""UNIT metadata/launcher regression; these are not CT/neural evidence."""
from contextlib import redirect_stderr
from io import StringIO
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from tools.run_v18_independent import input_paths, parse, validate_reference
from hiercp_v1x.u_bridge_experiment import FILES, FORMAT, digest, sha

ROOT = Path(__file__).resolve().parents[1]


class IndependentTests(unittest.TestCase):
    def test_one_arm_cli_and_no_resource_or_candidate_overrides(self):
        args = ['--gpu', '3', '--arm', 'selected', '--source-experiment', 'UNIT_old',
                '--experiment', 'UNIT_selected', '--inventory', 'UNIT_inventory']
        self.assertEqual(parse(args).arm, 'selected')
        self.assertEqual(parse(args[:3] + ['native'] + args[4:]).arm, 'native')
        for extra in (['--arm', 'both'], ['--rss-gib', '256'], ['--batch-candidates', '1']):
            with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                parse(args + extra)

    def test_current_neural_source_bytes_and_contract_are_required(self):
        with TemporaryDirectory(dir=ROOT, prefix='UNIT_v18_independent_') as directory:
            root = Path(directory)
            body = dict(format=FORMAT, debug=False, epochs=40,
                        semantic=dict(version='v1.8'), helpers={name: sha(ROOT/name) for name in FILES})
            body['sha256'] = digest(body)
            path = root/'experiment.json'
            path.write_text(json.dumps(body), encoding='utf8')
            self.assertEqual(validate_reference(root), body)
            broken = copy.deepcopy(body); broken['helpers'][FILES[0]] = '0'*64
            broken['sha256'] = digest({k:v for k,v in broken.items() if k != 'sha256'})
            path.write_text(json.dumps(broken), encoding='utf8')
            with self.assertRaisesRegex(ValueError, 'controller changed'):
                validate_reference(root)
            broken['epochs'] = 1
            path.write_text(json.dumps(broken), encoding='utf8')
            with self.assertRaisesRegex(ValueError, 'digest differs'):
                validate_reference(root)

    def test_inventory_is_exact_and_production_debug_bank_refused(self):
        with TemporaryDirectory(dir=ROOT, prefix='UNIT_v18_inputs_') as directory:
            inventory = Path(directory)/'inventory.json'
            inventory.write_text('{"raw_records":[]}', encoding='utf8')
            manifest = dict(debug=False, baseline=dict(inventory_sha256=sha(inventory)))
            with self.assertRaisesRegex(ValueError, 'DEBUG bank'):
                input_paths(manifest, inventory, inventory)
            inventory.write_text('{"raw_records":[1]}', encoding='utf8')
            with self.assertRaisesRegex(ValueError, 'inventory bytes differ'):
                input_paths(manifest, inventory, None)

    def test_server_script_contains_one_independent_root_without_resource_override(self):
        script = (ROOT/'tools/server_v18_independent.sh').read_text(encoding='utf8')
        self.assertIn('v18_${CP_ARM}_m10_seed42_memory', script)
        self.assertIn('--source-experiment "$CP_SOURCE_EXPERIMENT"', script)
        for forbidden in ('--rss-gib', '--batch-candidates', '--debug', 'CP_ARM:-both'):
            self.assertNotIn(forbidden, script)


if __name__ == '__main__':
    unittest.main()
