"""Policy decisions on synthetic manifests, separate from actual CT smoke."""
import copy
import unittest
from l0_regions.profile_policy import allow_profile,validate_cache_policy


def manifest(policy='research-report',debug=False):
    return dict(debug=debug,profile_policy=policy,admission_failures=1,
        full_training_admitted=False,research_training_admitted=not debug and policy=='research-report',
        partitions={'inner_train':[dict(profile_exceeded=True)],'inner_val':[dict(profile_exceeded=False)]})


class ProfilePolicy(unittest.TestCase):
    def test_full_research_keeps_violation(self):
        m=manifest();before=copy.deepcopy(m)
        validate_cache_policy(m,'research-report',False)
        self.assertEqual(m,before)
        self.assertTrue(allow_profile('research-report',False))
        self.assertFalse(m['full_training_admitted'])

    def test_strict_still_rejects(self):
        with self.assertRaisesRegex(ValueError,'strict policy'):
            validate_cache_policy(manifest('strict'),'strict',False)
        self.assertFalse(allow_profile('strict',False))

    def test_train_needs_same_explicit_policy(self):
        with self.assertRaisesRegex(ValueError,'policy mismatch'):
            validate_cache_policy(manifest(),'strict',False)

    def test_counts_cannot_hide_violations(self):
        m=manifest();m['admission_failures']=0
        with self.assertRaisesRegex(ValueError,'count mismatch'):
            validate_cache_policy(m,'research-report',False)

    def test_no_false_strict_admission(self):
        m=manifest();m['full_training_admitted']=True
        with self.assertRaisesRegex(ValueError,'classification mismatch'):
            validate_cache_policy(m,'research-report',False)

    def test_no_debug_promotion(self):
        with self.assertRaisesRegex(ValueError,'DEBUG mode mismatch'):
            validate_cache_policy(manifest(debug=True),'research-report',False)

    def test_unknown_policy_rejected(self):
        with self.assertRaisesRegex(ValueError,'Unknown'):
            allow_profile('ignore-everything',False)

    def test_model_uses_policy_with_debug_false(self):
        from unittest.mock import patch
        from types import SimpleNamespace
        from l0_regions.training import TrainEncoder
        calls=[]
        def construct(obj,reference,**kw):
            calls.append(kw);obj.core=SimpleNamespace(blocks=[])
        # Unit-test constructor wiring, not a simulated model or training result.
        with patch('l0_regions.training.RegionSAGEEncoder.__init__',construct):
            TrainEncoder(None,budget=object(),debug=False,contract=dict(profile={'region_scales':1},profile_policy='research-report'))
        self.assertTrue(calls[0]['allow_unvalidated_profile'])
        self.assertEqual(calls[0]['levels'],1)


if __name__=='__main__':unittest.main(verbosity=2)
