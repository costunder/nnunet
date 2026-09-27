"""Versioned physical placement shared by graph, eligibility and raw paste."""
from dataclasses import dataclass
import hashlib
import json
import numpy as np
from itertools import product

GEOMETRY_CONTRACT='paired_explicit_anchor_symmetric_padding_v2'
GRID_TOLERANCE_VOXELS=1e-4  # NIfTI float32 header roundoff; never a resampling tolerance.


def grid_error_voxels(image_affine,label_affine,shape):
    a=np.asarray(image_affine,dtype=float);b=np.asarray(label_affine,dtype=float)
    if any(x.shape!=(4,4) or not np.isfinite(x).all() for x in (a,b)) or abs(np.linalg.det(a[:3,:3]))<1e-12:
        raise ValueError('Finite nonsingular physical grids required')
    corners=np.array([list(p)+[1.] for p in product(*[(0,int(s)-1) for s in shape])])
    error=corners@b.T@np.linalg.inv(a).T-corners
    return float(np.abs(error[:,:3]).max())


def array_hash(value):
    value=np.ascontiguousarray(value)
    h=hashlib.sha256(str((value.dtype.str,value.shape)).encode())
    h.update(value.tobytes());return h.hexdigest()


def validate_grid(case):
    spacing=np.asarray(case.spacing,dtype=float)
    a=np.asarray(case.image_affine,dtype=float);b=np.asarray(case.label_affine,dtype=float)
    if spacing.shape!=(3,) or not np.isfinite(spacing).all() or (spacing<=0).any():
        raise ValueError('Nonfinite/nonpositive CT spacing')
    if any(x.shape!=(4,4) or not np.isfinite(x).all() for x in (a,b)):
        raise ValueError('Finite 4x4 CT/label affine required')
    if grid_error_voxels(a,b,case.image.shape)>GRID_TOLERANCE_VOXELS:
        raise ValueError('CT and label must occupy the same nonsingular physical grid')
    if not np.allclose(np.linalg.norm(a[:3,:3],axis=0),spacing,rtol=1e-5,atol=1e-5):
        raise ValueError('CT header spacing and affine disagree')
    if case.image.shape!=case.label.shape or case.image.ndim!=3:
        raise ValueError('Aligned three-dimensional CT and label required')
    return case


@dataclass(frozen=True)
class PlacementSpec:
    recipient: str
    donor: str
    component: int
    center: tuple
    spacing: tuple
    affine: tuple
    image: np.ndarray
    mask: np.ndarray
    anchor: tuple
    transform: tuple

    def __post_init__(self):
        image=np.array(self.image,dtype=np.float32,copy=True)
        mask=np.array(self.mask,copy=True)
        anchor=np.asarray(self.anchor);center=np.asarray(self.center)
        if mask.ndim!=3 or mask.dtype!=bool or not mask.any() or image.shape!=mask.shape or not np.isfinite(image).all():
            raise ValueError('Complete transformed CT/mask required')
        if any(v.shape!=(3,) or not np.issubdtype(v.dtype,np.integer) for v in (anchor,center)):
            raise ValueError('Integer native center/anchor required')
        if ((anchor<0)|(anchor>=mask.shape)).any():raise ValueError('Anchor outside patch')
        spacing=np.asarray(self.spacing);affine=np.asarray(self.affine);transform=np.asarray(self.transform)
        if spacing.shape!=(3,) or not np.isfinite(spacing).all() or (spacing<=0).any():raise ValueError('Invalid spacing')
        if affine.shape!=(4,4) or transform.shape!=(3,3) or not np.isfinite(affine).all() or not np.isfinite(transform).all():
            raise ValueError('Explicit finite physical frame/transform required')
        if abs(np.linalg.det(affine[:3,:3]))<1e-12 or abs(np.linalg.det(transform))<1e-12:
            raise ValueError('Invertible frame/transform required')
        if not self.recipient or not self.donor or self.component<1:raise ValueError('Placement identities required')
        image.flags.writeable=False;mask.flags.writeable=False
        object.__setattr__(self,'image',image);object.__setattr__(self,'mask',mask)

    def metadata(self):
        return dict(geometry_contract=GEOMETRY_CONTRACT,recipient=self.recipient,donor=self.donor,
            component=int(self.component),center=list(self.center),spacing=list(self.spacing),
            affine=[list(r) for r in self.affine],anchor=list(self.anchor),
            transform=[list(r) for r in self.transform],axis_order='native_ijk',
            mask_sha256=array_hash(self.mask),image_sha256=array_hash(self.image))

    def coordinates(self):
        return np.argwhere(self.mask)+np.asarray(self.center)-np.asarray(self.anchor)

    def paste(self,image,label):
        """Exact true-voxel paste; bounding-box padding never clips valid voxels."""
        points=self.coordinates()
        if image.shape!=label.shape or image.ndim!=3:raise ValueError('Aligned native image/label required')
        if ((points<0)|(points>=image.shape)).any():raise ValueError('Paste footprint out of bounds')
        indices=tuple(points.T)
        if np.any(label[indices]==2):raise ValueError('Paste overlaps annotated tumor')
        result=image.copy();seg=label.copy()
        result[indices]=self.image[self.mask];seg[indices]=2
        return result,seg


def placement_spec(target,source,center,donor,*,forward_mm=None,graph_config=None):
    validate_grid(target)
    anchor=np.asarray(source.anchor_center)-np.asarray([s.start for s in source.patch_slices])
    image,mask=source.patch_image,source.patch_mask
    transform=np.eye(3) if forward_mm is None else np.asarray(forward_mm,dtype=np.float64)
    if not np.array_equal(transform,np.eye(3)):
        from scipy import ndimage as ndi
        from hiercp_v22.spatial import transform_footprint_physical
        if graph_config is None:raise ValueError('Graph geometry contract required for physical transform')
        if not np.array_equal(anchor,np.asarray(mask.shape)//2):raise ValueError('Transform requires normalized explicit anchor')
        mask=transform_footprint_physical(mask,transform,target.spacing,graph_config)
        inverse=np.linalg.inv(transform)
        inverse_voxel=(inverse*target.spacing[None,:])/target.spacing[:,None]
        new_anchor=np.asarray(mask.shape)//2
        image=ndi.affine_transform(image,matrix=inverse_voxel,offset=anchor-inverse_voxel@new_anchor,
            output_shape=mask.shape,order=1,mode='nearest',prefilter=False).astype(np.float32)
        anchor=new_anchor
    return PlacementSpec(target.paths.case_id,donor,int(source.component_id),tuple(map(int,center)),
        tuple(map(float,target.spacing)),tuple(map(tuple,target.image_affine.tolist())),
        image,mask,tuple(map(int,anchor)),tuple(map(tuple,transform.tolist())))
