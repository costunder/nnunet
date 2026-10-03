"""UNIT admission preserves exact target geometry and real mask coverage."""
import copy
import unittest
from tools.run_v1x_experiment import _valid_roi_measurement


class RoiReplayAdmissionUnits(unittest.TestCase):
    def row(self):
        return dict(status='PASS', original_guard_reproduced=True,
            replay_contract='original_sample_first_roi_failure_v1',
            geometry_preserved=True, full_mask_preserved=True, geometry_equal=False,
            original_failure_phase='target', failed_operation='build_patch_payload',
            requested_shape=[201,201,201], payload_initial_shape=[201,201,201],
            effective_shape=[203,201,201], roi_voxels=203*201*201, full_mask_voxels=100,
            replayed_geometry=dict(requested_shape=[201,201,201]),
            original_requested_shape=[201,201,201], target_spec=dict(center=[30,40,50]),
            actual_center=[30,40,50], erase_target=True)

    def test_original_surface_extension_is_recorded_not_called_a_reduction(self):
        self.assertTrue(_valid_roi_measurement(self.row(), 12000000))

    def test_reduced_shapes_and_wrong_voxel_cost_rejected(self):
        for key, value in (('effective_shape',[199,201,201]),
                           ('payload_initial_shape',[199,201,201]), ('roi_voxels',1)):
            row = self.row()
            row[key] = value
            self.assertFalse(_valid_roi_measurement(row, 12000000))

    def test_mask_failure_or_changed_center_rejected(self):
        for key, value in (('full_mask_preserved',False), ('full_mask_voxels',0),
                           ('actual_center',[1,2,3]), ('erase_target',False)):
            row = self.row()
            row[key] = value
            self.assertFalse(_valid_roi_measurement(row, 12000000))

    def test_original_failure_proof_and_resource_ceiling_required(self):
        row = self.row()
        self.assertFalse(_valid_roi_measurement(row, 8000000))
        for key, value in (('original_guard_reproduced',False), ('status','FAILED'),
                           ('original_requested_shape',[1,1,1]),
                           ('replayed_geometry',dict(requested_shape=[1,1,1]))):
            changed = copy.deepcopy(row)
            changed[key] = value
            self.assertFalse(_valid_roi_measurement(changed, 12000000))

    def test_original_source_evidence_keeps_its_exact_shape_requirement(self):
        source = dict(status='PASS', original_guard_reproduced=True, geometry_equal=True)
        self.assertTrue(_valid_roi_measurement(source, 12000000))
        source['geometry_equal'] = False
        self.assertFalse(_valid_roi_measurement(source, 12000000))

    def test_malformed_shape_never_promoted(self):
        self.assertFalse(_valid_roi_measurement(None, 12000000))
        row = self.row()
        row['replayed_geometry'] = None
        self.assertFalse(_valid_roi_measurement(row, 12000000))
        for shape in (None, [201,201], [201,201,float('nan')], [201,True,201]):
            row = self.row()
            row['effective_shape'] = shape
            self.assertFalse(_valid_roi_measurement(row, 12000000))


if __name__ == '__main__':
    unittest.main()
