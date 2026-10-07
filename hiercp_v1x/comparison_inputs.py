"""Exact local mask extraction for the preserved comparison provider."""
from __future__ import annotations

import numpy as np


def local_mask_provider(original):
    """Avoid a full-CT label comparison for every candidate patch.

    Original: crop(label == 1, padding=False).
    Equivalent: crop(label, padding=0) == 1.
    ROI, candidate metadata and padding semantics are unchanged.
    """
    class LocalMaskProvider(original):
        def _candidate(self, center, case, source, organ, depth, occupied):
            runtime = self._runtime()
            mask = source.patch_mask
            crop = runtime.common.extract_centered_patch(case.image, center, mask.shape,
                pad_value=self.config['ct_clip'][0])
            liver = runtime.common.extract_centered_patch(case.label, center, mask.shape,
                pad_value=0) == 1
            crop_organ = runtime.common.extract_centered_patch(organ, center, mask.shape, pad_value=False)
            mean, std = runtime.common.context_stats_for_local_mask(crop, crop_organ, mask, ring_width=3)
            start = np.asarray(center) - np.asarray(mask.shape) // 2
            slices = tuple(slice(max(0, int(a)), min(int(a + n), int(limit)))
                           for a, n, limit in zip(start, mask.shape, case.shape))
            return runtime.common.CandidateInfo(center=center, slices=slices,
                liver_coverage=float(np.sum(mask & liver) / source.voxel_count),
                border_distance_mm=float(depth[center]), occupied_distance_mm=float(occupied[center]),
                context_mean_hu=mean, context_std_hu=std)
    LocalMaskProvider.__name__ = 'LocalMask' + original.__name__
    return LocalMaskProvider
