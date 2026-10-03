"""UNIT byte/metadata evidence reuse; fixture bytes are not clinical CT."""
import copy
import json
from pathlib import Path
import unittest
import uuid
from hiercp_v1x.experiment import ROOT
from hiercp_v1x.contracts import V1_ARCHIVE_SHA256
from hiercp_v1x.roi_probe_reuse import sha, verified_source_reuse


class RoiProbeReuseUnits(unittest.TestCase):
    def setUp(self):
        self.root = ROOT/'work'/('roi_reuse_UNIT_'+uuid.uuid4().hex)
        self.root.mkdir()
        self.experiment = self.root/'experiment'
        self.snapshot = self.experiment/'source/v1.0'
        self.medical = self.root/'medical_UNIT'
        for folder in (self.snapshot, self.medical/'Data/image', self.medical/'Data/labels'):
            folder.mkdir(parents=True, exist_ok=True)
        self.report_path = self.root/'probe.json'
        self.sidecars = self.root/'probe_requests'
        self.sidecars.mkdir()
        self.config = {'graph': {'adaptive_roi_max_voxels': 8000000, 'patch_size':48}}
        self.requests, self.rows = [], []
        for ordinal, case in enumerate(('liver_116', 'liver_129')):
            image = self.medical/'Data/image'/f'{case}_0000.nii.gz'
            label = self.medical/'Data/labels'/f'{case}.nii.gz'
            image.write_bytes(b'UNIT image hash-only fixture')
            label.write_bytes(b'UNIT label hash-only fixture')
            geometry = dict(requested_shape=[219,213,223], requested_voxels=10402281)
            failure = dict(case_id=case, sample_index=0, original_failure='UNIT original guard',
                geometry=geometry, source_image_sha256=sha(image), source_label_sha256=sha(label),
                split_name='train')
            self.requests.append(failure)
            row = dict(case_id=case, sample_index=0, status='PASS' if ordinal == 0 else 'FAILED',
                geometry_equal=True, original_guard_reproduced=True, spatial_sha256='a'*64,
                requested_shape=geometry['requested_shape'], effective_shape=geometry['requested_shape'],
                roi_voxels=10402281, limitation='Source ROI fields only; UNIT evidence',
                source_image_sha256=sha(image), source_label_sha256=sha(label),
                full_mask_voxels=100, source_component=1, model_input_shape=[5,48,48,48],
                peak_rss_bytes=1000, wall_seconds=1.0, geometry_seconds=.5, load_seconds=.25)
            self.rows.append(row)
            if ordinal == 0:
                answer = {k:v for k,v in row.items() if k not in ('peak_rss_bytes','wall_seconds')}
                self.write(self.sidecars/'000_answer.json', answer)
                request = dict(config=self.config, snapshot=str(self.snapshot),
                    medical_root=str(self.medical), shared=str(self.experiment/'shared'),
                    spatial_sha256='a'*64, candidate_roi_max_voxels=12000000,
                    answer_path=str(self.sidecars/'000_answer.json'),
                    failure={k:v for k,v in failure.items() if k != 'split_name'})
                self.write(self.sidecars/'000_request.json', request)
        self.report = dict(format='v1_roi_budget_probe_v1', scope='debug_geometry_only',
            archived_source_sha256=V1_ARCHIVE_SHA256, experiment=str(self.experiment),
            failed_manifest_sha256='b'*64, original_roi_max_voxels=8000000,
            candidate_roi_max_voxels=12000000, originals_preserved=True,
            training_started=False, cache_publication_created=False, completed=False,
            failed_sample_requests=2, measurements=self.rows,
            resource_budget=dict(rss_bytes=16*2**30, case_timeout_seconds=120))
        self.write(self.report_path, self.report)

    def write(self, path, value):
        path.write_text(json.dumps(value), encoding='utf-8')

    def check(self, **changes):
        args = dict(experiment=self.experiment, requests=self.requests, config=self.config,
            snapshot=self.snapshot, medical_root=self.medical, spatial_sha256='a'*64,
            failed_manifest_sha256='b'*64, candidate_voxels=12000000,
            rss_bytes=16*2**30, case_timeout_seconds=120)
        args.update(changes)
        return verified_source_reuse(self.report_path, **args)

    def test_only_verified_pass_reused_from_incomplete_report(self):
        before = self.report_path.read_bytes()
        rows, proof = self.check()
        self.assertEqual(set(rows), {('liver_116',0)})
        self.assertTrue(rows[('liver_116',0)]['reused'])
        self.assertEqual(rows[('liver_116',0)]['wall_seconds'], 1.0)
        self.assertEqual(proof['records'], 1)
        self.assertEqual(before, self.report_path.read_bytes())

    def test_changed_config_or_resource_limit_rejected(self):
        for change in (dict(config={'graph':{'adaptive_roi_max_voxels':1}}),
                       dict(rss_bytes=1), dict(case_timeout_seconds=121)):
            with self.assertRaises(ValueError): self.check(**change)

    def test_changed_raw_ct_rejected(self):
        (self.medical/'Data/image/liver_116_0000.nii.gz').write_bytes(b'UNIT changed raw bytes')
        with self.assertRaisesRegex(ValueError, 'raw CT bytes changed'): self.check()

    def test_changed_answer_rejected(self):
        path = self.sidecars/'000_answer.json'
        answer = json.loads(path.read_text())
        answer['roi_voxels'] = 1
        self.write(path, answer)
        with self.assertRaisesRegex(ValueError, 'request/answer differs'): self.check()

    def test_changed_request_source_or_medical_root_rejected(self):
        path = self.sidecars/'000_request.json'
        original = json.loads(path.read_text())
        for key, value in (('medical_root', str(self.root)), ('spatial_sha256', 'c'*64)):
            changed = copy.deepcopy(original)
            changed[key] = value
            self.write(path, changed)
            with self.assertRaises(ValueError): self.check()

    def test_changed_manifest_binding_rejected(self):
        with self.assertRaises(ValueError): self.check(failed_manifest_sha256='c'*64)

    def test_duplicate_or_changed_cohort_rejected(self):
        for rows in ([self.rows[0],self.rows[0]], list(reversed(self.rows))):
            changed = copy.deepcopy(self.report)
            changed['measurements'] = rows
            self.write(self.report_path, changed)
            with self.assertRaises(ValueError): self.check()

    def test_claimed_pass_without_exact_geometry_rejected(self):
        changed = copy.deepcopy(self.report)
        changed['measurements'][0]['geometry_equal'] = False
        self.write(self.report_path, changed)
        with self.assertRaises(ValueError): self.check()


if __name__ == '__main__':
    unittest.main()
