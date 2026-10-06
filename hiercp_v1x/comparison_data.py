"""Only comparison-coordinate exposure changes; original view epochs advance."""
from __future__ import annotations

from .comparison_experiment import ARMS
from .u_bridge_data import UBridgeData


class ComparisonData(UBridgeData):
    def candidate_keys(self, index, arm, epoch, full=False):
        if arm not in ARMS:
            raise ValueError('Unknown declared v1.9 comparison arm')
        if arm in ('selected', 'native'):
            return super().candidate_keys(index, arm, epoch, full=full)
        if type(epoch) is not int or epoch < 1:
            raise ValueError('One-based actual view epoch required')
        # Only the key-selection epoch is fixed. Inherited batch() still uses
        # its actual epoch for both original sampled views and augmentation.
        key_epoch = 1 if arm == 'native_fixed' and not full else epoch
        return super().candidate_keys(index, 'native', key_epoch, full=full)

    def report(self):
        result = super().report()
        result['comparison_controls'] = dict(
            arms=list(ARMS), actual_view_epoch_unchanged=True,
            fixed_training_keys=['P'] + [f'U:{i}' for i in range(7)],
            full_validation='same original P + every frozen native U:0..127',
            original_P_definition_changed=False,
        )
        return result
