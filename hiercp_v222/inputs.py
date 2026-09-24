"""CT-only inputs: deliberately no target, donor or segmentation argument."""
from functools import lru_cache
import numpy as np
import torch
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

def validate_contract(contract):
    if contract.get('format') != 'v222_raw_ct_full_ball_v1':
        raise ValueError('A fitted, frozen v2.22 physical input contract is required')
    for key in ('outer_radius_mm', 'node_spacing_mm', 'edge_radius_mm'):
        if not np.isfinite(contract[key]) or contract[key] <= 0:
            raise ValueError(f'Invalid physical contract: {key}')
    if contract['patch_size'] != 48 or contract['fit_partition'] != 'inner_train':
        raise ValueError('Full-resolution input and training-only fit required')
    if contract.get('center_masking') is not False or contract.get('node_policy') != 'full_uniform_physical_ball':
        raise ValueError('Raw CT and full uniform topology required; no hidden central exclusion')

def context_patch(image, spacing, center, contract):
    """Sample original CT without central replacement or annotation channels.

    All locations use identical physical geometry. No GT boundary, organ mask,
    donor shape, label or patient ID can affect this function.
    """
    validate_contract(contract)
    image = np.asarray(image)
    spacing = np.asarray(spacing, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    if image.ndim != 3 or spacing.shape != (3,) or center.shape != (3,) or not np.all(spacing > 0):
        raise ValueError('3-D CT, positive spacing and one voxel-space center required')
    if not np.all(np.isfinite(center)) or np.any(center < 0) or np.any(center > np.asarray(image.shape)-1):
        raise ValueError('Context center outside image')
    radius = contract['outer_radius_mm']
    lower = np.floor(center-radius/spacing).astype(int)-1
    upper = np.ceil(center+radius/spacing).astype(int)+2
    shape = upper-lower
    low, high = contract['ct_clip']
    crop = np.full(tuple(shape), low, dtype=np.float32)
    start = np.maximum(lower, 0); stop = np.minimum(upper, image.shape)
    dst = tuple(slice(int(a), int(b)) for a, b in zip(start-lower, stop-lower))
    src = tuple(slice(int(a), int(b)) for a, b in zip(start, stop))
    crop[dst] = image[src]
    axis = np.linspace(-radius, radius, contract['patch_size'], dtype=np.float64)
    grid = np.stack(np.meshgrid(*[axis/spacing[i]+center[i]-lower[i] for i in range(3)], indexing='ij'))
    value = ndi.map_coordinates(crop, grid, order=1, mode='constant', cval=low, prefilter=False)
    value = (np.clip(value, low, high)-low)/(high-low)*2-1
    if not np.isfinite(value).all():
        raise FloatingPointError('Nonfinite visible CT')
    return value.astype(np.float32)[None]

@lru_cache(maxsize=8)
def _topology(radius, step, edge_radius):
    axis = np.arange(-np.floor(radius/step), np.floor(radius/step)+1)*step
    pos = np.stack(np.meshgrid(axis, axis, axis, indexing='ij'), -1).reshape(-1, 3)
    norm = np.linalg.norm(pos, axis=1)
    pos = pos[norm <= radius]
    if not len(pos):
        raise ValueError('Empty full spatial ball')
    pairs = cKDTree(pos).query_pairs(edge_radius, output_type='ndarray')
    edges = np.concatenate((pairs.T, pairs[:, ::-1].T), axis=1).astype(np.int64)
    grid = np.ascontiguousarray((pos[:, ::-1]/radius).astype(np.float32))
    return torch.from_numpy(grid), torch.from_numpy(edges)

def topology(contract):
    validate_contract(contract)
    return _topology(*(float(contract[k]) for k in ('outer_radius_mm', 'node_spacing_mm', 'edge_radius_mm')))
