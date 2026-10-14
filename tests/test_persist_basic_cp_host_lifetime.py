"""DEBUG CPU parser fixtures; no running PBS job or final training is claimed.

The old deadline/start match the recorded129776 allocation. A different numeric
ID exercises future admitted-job binding only; real admission always reads qstat.
"""
import copy
import json
import unittest
from unittest.mock import patch

from tools import persist_basic_cp_seven as durable


def DEBUG_documents(job='129776.ECE-util1', start=1791866463.0):
    expires = start + 24 * 3600
    witness = dict(format='v25_owned_GPU_allocation_v1', UID=41833, owner='aicompetition06', node='ece-a6gpu6',
                   gpu=6, job_id=job, expires_at_unix=expires, job_state='R', stime=start, walltime='24:00:00')
    reference = dict(path='/DEBUG_owned_same_host_lifetime.json', sha256='a' * 64, expires_at_unix=expires)
    actual = dict(Jobs={job: dict(job_state='R', Job_Owner='aicompetition06@ECE-util1', exec_host='ece-a6gpu6/0*6',
                                 Resource_List=dict(ngpus=1, walltime='24:00:00'), stime=start)})
    return reference, witness, actual


class HostLifetimeParserTests(unittest.TestCase):
    def validate(self, reference, witness, raw, *, now=None):
        # Fixed fixture time is explicit; it does not turn expired live PBS into R.
        if now is None: now = witness['expires_at_unix'] - 3600
        with patch.object(durable.time, 'time', return_value=now):
            return durable.validate_host_lifetime(reference, witness, json.dumps(raw))

    def test_original_recorded_job_uses_actual_native_PBS_validator(self):
        ref, witness, raw = DEBUG_documents()
        observation = self.validate(ref, witness, raw)
        self.assertEqual(observation['job_id'], '129776.ECE-util1')
        self.assertEqual(observation['expires_at_unix'], 1791952863.0)
        self.assertEqual(observation['node'], 'ece-a6gpu6')

    def test_other_numeric_job_is_bound_to_its_own_running_response_and_deadline(self):
        ref, witness, raw = DEBUG_documents('129999.ECE-util1', 1791952863.0)
        observation = self.validate(ref, witness, raw)
        self.assertEqual(observation['job_id'], '129999.ECE-util1')
        self.assertEqual(observation['expires_at_unix'], 1792039263.0)
        with self.assertRaises(RuntimeError):
            self.validate(ref, witness, DEBUG_documents()[2])

    def test_wrong_owned_witness_contract_is_rejected(self):
        for field, value in [('UID', 1), ('owner', 'another_user'), ('node', 'another_host'),
                             ('gpu', 0), ('job_state', 'Q'), ('job_id', '129999.other_server'),
                             ('job_id', 129999), ('format', 'other_format')]:
            with self.subTest(field=field, value=value):
                ref, witness, raw = DEBUG_documents()
                witness[field] = value
                with self.assertRaises(ValueError): self.validate(ref, witness, raw)

    def test_actual_queued_or_other_user_job_never_counts_as_running_owned_job(self):
        for field, value in [('job_state', 'Q'), ('Job_Owner', 'another_user@ECE-util1'), ('exec_host', 'another_host/0')]:
            with self.subTest(field=field):
                ref, witness, raw = DEBUG_documents()
                raw['Jobs'][witness['job_id']][field] = value
                with self.assertRaises(RuntimeError): self.validate(ref, witness, raw)

    def test_actual_PBS_start_or_walltime_change_invalidates_immutable_deadline(self):
        for change in ('start', 'walltime', 'ngpus'):
            with self.subTest(change=change):
                ref, witness, raw = DEBUG_documents()
                job = raw['Jobs'][witness['job_id']]
                if change == 'start': job['stime'] += 60
                elif change == 'walltime': job['Resource_List']['walltime'] = '23:00:00'
                else: job['Resource_List']['ngpus'] = 2
                with self.assertRaises(RuntimeError): self.validate(ref, witness, raw)

    def test_recorded_witness_start_walltime_and_reference_cannot_be_fabricated(self):
        for change in ('start', 'walltime', 'reference', 'malformed', 'zero'):
            with self.subTest(change=change):
                ref, witness, raw = DEBUG_documents()
                if change == 'start': witness['stime'] += 60
                elif change == 'walltime': witness['walltime'] = '25:00:00'
                elif change == 'reference': ref['expires_at_unix'] += 60
                elif change == 'malformed': witness['walltime'] = '24:70:00'
                else: witness['walltime'] = '00:00:00'
                with self.assertRaises(ValueError): self.validate(ref, witness, raw)

    def test_expired_or180second_boundary_job_cannot_admit_cache_writer(self):
        ref, witness, raw = DEBUG_documents()
        for now in (witness['expires_at_unix'] + 1, witness['expires_at_unix'] - 179):
            with self.subTest(now=now), self.assertRaises(ValueError):
                self.validate(ref, witness, raw, now=now)

    def test_optional_actual_server_timestamp_remains_native_strict(self):
        ref, witness, raw = DEBUG_documents()
        raw['timestamp'] = ''
        self.assertEqual(self.validate(ref, witness, raw)['job_state'], 'R')
        raw['timestamp'] = float('nan')
        with self.assertRaises(RuntimeError): self.validate(ref, witness, raw)


if __name__ == '__main__': unittest.main()
