"""CPU contract checks; these never substitute for actual full training."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from tools.run_v25 import configuration, parse

ROOT = Path(__file__).resolve().parents[1]


class V25CLIContractCPU(unittest.TestCase):
    def test_four_arms_keep_full_production_budget(self):
        for arm, initial in (('A',8),('B',16),('C',32),('D',128)):
            config = configuration(ROOT/'config/v25.json',arm)
            self.assertEqual(config['version'],'v2.5')
            self.assertNotIn('vessels',config)
            runtime = config['runtime']
            self.assertEqual((runtime['physical_patient_batch'],runtime['workers']),(4,4))
            self.assertEqual(runtime['curriculum']['initial_u'],initial)
            self.assertEqual(runtime['curriculum']['total_epochs'],40)
            self.assertEqual(runtime['curriculum']['minimum_full_bank_epochs'],8)

    def test_contract_changes_are_explicit_errors(self):
        original = json.loads((ROOT/'config/v25.json').read_text(encoding='utf8'))
        variants = []
        for section, key, value in ((None,'version','v3.0'),(None,'epochs',4),
                ('runtime','physical_patient_batch',8),('runtime','hidden_subset',True),
                ('context','tumor_distance_exclusive_mm',2.0)):
            changed = copy.deepcopy(original)
            (changed if section is None else changed[section])[key] = value
            variants.append(changed)
        changed = copy.deepcopy(original);changed['arms'].pop('B');variants.append(changed)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'profile.json'
            for changed in variants:
                path.write_text(json.dumps(changed),encoding='utf8')
                with self.assertRaises(ValueError):configuration(path,'A')

    def test_physical_gpu_is_independent_of_arm(self):
        args = parse(['--mode','prepare','--arm','B','--gpu','6',
            '--native-experiment','native','--inventory','inventory',
            '--stunet-checkpoint','official','--input-cache','fresh_cache','--output','armB'])
        self.assertEqual((args.arm,args.gpu),('B',6))


if __name__ == '__main__':
    unittest.main()
