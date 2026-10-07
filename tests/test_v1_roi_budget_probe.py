"""ROI diagnostic parser UNIT tests; no CT decoding, neural work or training."""
from tests.artifacts import unit_artifact_root
import csv
from pathlib import Path
import unittest
import uuid

from hiercp_v1x.experiment import ROOT
from hiercp_v1x.roi_budget_probe import parse_roi_failure, failure_requests


def message(shape=(219,213,223), footprint=(141,135,137), margin=30):
    import math
    count = math.prod(shape)
    return ('Adaptive ROI request exceeds graph.adaptive_roi_max_voxels; no context reduction was applied. '
        f'footprint_shape={footprint}, spacing_mm=(0.7890625, 0.7890625, 0.699999988079071), '
        f'requested_margin_mm={margin}, requested_shape={shape}, requested_voxels={count}, '
        f'effective_shape={shape}, effective_voxels={count}, voxel_budget=8000000. '
        'Increase graph.adaptive_roi_max_voxels after measuring RAM and processing time.')


class RoiProbeParserUnits(unittest.TestCase):
    def test_exact_logged_geometry(self):
        result = parse_roi_failure(message())
        self.assertEqual(result['requested_voxels'], 10_402_281)
        self.assertEqual(result['requested_shape'], [219,213,223])
        self.assertEqual(result['footprint_shape'], [141,135,137])
        self.assertEqual(result['voxel_budget'], 8_000_000)

    def test_non_voxel_failure_is_not_relabelled(self):
        for invalid in ('CUDA OOM', 'node limit exceeded', 'Including a real liver-surface anchor exceeds adaptive_roi_max_voxels'):
            with self.assertRaises(ValueError):
                parse_roi_failure(invalid)

    def test_reduced_or_inconsistent_geometry_rejected(self):
        valid = message()
        for invalid in (valid.replace('effective_shape=(219, 213, 223)', 'effective_shape=(100, 100, 100)'),
                        valid.replace('requested_voxels=10402281', 'requested_voxels=1'),
                        valid.replace('effective_voxels=10402281', 'effective_voxels=1'),
                        valid.replace('requested_margin_mm=30', 'requested_margin_mm=-1')):
            with self.assertRaises(ValueError):
                parse_roi_failure(invalid)

    def test_all_four_failures_remain_separate_from_case_summary(self):
        directory = unit_artifact_root()/('roi_probe_parser_UNIT_'+uuid.uuid4().hex)
        directory.mkdir()
        path = directory/'manifest.csv'
        rows = [dict(case_id='liver_116', sample_index='', status='sample_failure', message='case summary')]
        for case, index, shape, footprint, margin in (
                ('liver_116',0,(219,213,223),(141,135,137),30),
                ('liver_116',1,(219,213,223),(141,135,137),30),
                ('liver_129',1,(189,219,203),(103,133,117),30),
                ('liver_84',0,(213,225,237),(93,104,107),45.3777)):
            rows.append(dict(case_id=case, sample_index=index, status='resource_budget_error',
                             message=message(shape,footprint,margin)))
        with path.open('x', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)
        result = failure_requests(path)
        self.assertEqual(len(result), 4)
        self.assertEqual({row['geometry']['requested_voxels'] for row in result},
                         {10_402_281,8_402_373,11_358_225})

    def test_duplicate_failure_and_absent_failure_rejected(self):
        directory = unit_artifact_root()/('roi_probe_parser_UNIT_'+uuid.uuid4().hex)
        directory.mkdir()
        for name, rows in (('none', []), ('duplicate', [dict(case_id='liver_116',sample_index=0,
                status='resource_budget_error',message=message())]*2)):
            path = directory/(name+'.csv')
            with path.open('x', encoding='utf-8', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=('case_id','sample_index','status','message'))
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaises(ValueError):
                failure_requests(path)


if __name__ == '__main__':
    unittest.main()
