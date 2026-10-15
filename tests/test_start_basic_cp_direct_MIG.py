"""CPU-only tests of direct CUDA identity admission, never a CUDA simulation."""
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from tools import start_basic_cp_direct_MIG as start


class DirectIdentityAdmissionTests(unittest.TestCase):
    def test_real_child_observation_requires_the_exact_selected_UUID(self):
        token='MIG-11111111-1111-1111-1111-111111111111'
        proof=dict(UUID=token,total_bytes=10200547328,context_access=True,DEBUG=True,
                   kernels_started=False,production_training_started=False)
        with patch.object(start.subprocess,'run',return_value=SimpleNamespace(
                returncode=0,stdout=json.dumps(proof),stderr='')) as execute:
            self.assertEqual(start.probe_execution_device(token),proof)
            self.assertEqual(execute.call_args.kwargs['env']['CUDA_VISIBLE_DEVICES'],token)

    def test_wrong_UUID_or_no_context_access_never_becomes_production_admission(self):
        for proof in (dict(UUID='MIG-other',context_access=True),dict(UUID='MIG-selected',context_access=False)):
            with self.subTest(proof=proof), patch.object(start.subprocess,'run',return_value=SimpleNamespace(
                    returncode=0,stdout=json.dumps(proof),stderr='')),self.assertRaises(ValueError):
                start.probe_execution_device('MIG-selected')

    def test_driver_permission_failure_is_reported_without_retry_or_fallback(self):
        with patch.object(start.subprocess,'run',return_value=SimpleNamespace(
                returncode=1,stdout='',stderr='CUDA_ERROR_NOT_PERMITTED')) as execute:
            with self.assertRaisesRegex(ValueError,'CUDA_ERROR_NOT_PERMITTED'):
                start.probe_execution_device('MIG-selected')
            self.assertEqual(execute.call_count,1)


if __name__=='__main__':unittest.main()
