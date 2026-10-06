"""DEBUG mathematical phantoms, not clinical data or final training samples."""
import itertools
import unittest
import numpy as np
from hiercp_v22.features import physical_moments,physical_gradients,node_observations,WIDTH


class PhysicalFeatureDebugTests(unittest.TestCase):
    def test_even_source_box_matches_actual_CT_and_GT(self):
        from pathlib import Path
        from hiercp.common import LoadedCase,CasePaths,organ_depth_mm
        from hiercp.schema import graph_config_from_dict
        from hiercp_v22.contracts import load_config
        from hiercp_v22.data import sources
        from hiercp_v22.geometry import _patch_fields
        from hiercp.spatial import extract_centered
        _,base=load_config();grid=np.indices((64,64,64));organ=((grid-32)**2).sum(0)<28**2
        label=organ.astype(np.uint8);label[23:27,27:33,29:33]=2
        image=(grid[0]*20+grid[1]+grid[2]*.01).astype(np.float32)
        case=LoadedCase(CasePaths('DEBUG_EVEN',Path('unused'),Path('unused')),image,label,np.eye(4),np.eye(4),None,None,np.ones(3),{}, {})
        source,_=sources(case,base['cache']['source_pad'],20)[0]
        self.assertTrue(any(n%2==0 for n in source.patch_mask.shape))
        fields=_patch_fields(case,source.anchor_center,source.patch_mask,organ,organ_depth_mm(organ,case.spacing),
            config=graph_config_from_dict(base['graph']),erase_target=False,ct_clip=tuple(base['ct_clip']))
        actual=extract_centered(source.full_mask,source.anchor_center,fields.footprint.shape,pad_value=False)
        np.testing.assert_array_equal(fields.footprint,actual)
        self.assertTrue(fields.organ[fields.footprint].all())

    def test_fractional_box_against_independent_cell_overlap(self):
        image=np.arange(6*7*8,dtype=np.float64).reshape(6,7,8)/100
        valid=np.ones(image.shape,bool);valid[1:3,2:5,3:6]=False
        spacing=np.array([.7,1.3,2.1]);points=np.array([[0,0,0],[3,4,5],[5,6,7]])
        radii=[1.6,3.3];actual=physical_moments(image,valid,spacing,points,radii)
        cells=np.indices(image.shape).reshape(3,-1).T
        for i,point in enumerate(points):
            for j,r in enumerate(radii):
                lo=point+.5-r/spacing;hi=point+.5+r/spacing
                weight=np.maximum(0,np.minimum(cells+1,hi)-np.maximum(cells,lo)).prod(1)*valid.ravel()
                mean=np.average(image.ravel(),weights=weight)
                std=np.sqrt(np.average((image.ravel()-mean)**2,weights=weight))
                coverage=weight.sum()/np.prod(2*r/spacing)
                np.testing.assert_allclose(actual[i,j*3:j*3+3],[mean,std,coverage],rtol=1e-6,atol=1e-6)

    def test_constant_and_physical_coverage(self):
        image=np.full((11,13,15),.75);valid=np.ones(image.shape,bool)
        result=physical_moments(image,valid,[1,2,3],[[5,6,7]],[4])
        np.testing.assert_allclose(result,[[.75,0,1]],atol=1e-6)

    def test_physical_linear_ramp_and_axis_permutation(self):
        shape=(7,8,9);spacing=np.array([.6,1.2,2.8]);slope=np.array([.2,-.4,.7])
        image=(np.indices(shape)*spacing[:,None,None,None]*slope[:,None,None,None]).sum(0)
        points=np.array([[0,0,0],[3,4,5],[6,7,8]])
        g,a=physical_gradients(image,np.ones(shape,bool),spacing,points)
        np.testing.assert_allclose(g,np.tile(slope,(3,1)),rtol=1e-6);self.assertTrue(a.all())
        perm=[2,0,1];g2,a2=physical_gradients(image.transpose(perm),np.ones(shape,bool).transpose(perm),spacing[perm],points[:,perm])
        np.testing.assert_allclose(g2,g[:,perm]);np.testing.assert_equal(a2,a[:,perm])

    def test_masked_values_do_not_contaminate_statistics_or_gradients(self):
        rng=np.random.default_rng(1);image=rng.random((9,9,9));valid=np.ones(image.shape,bool)
        valid[3:6,3:6,3:6]=False;points=np.array([[2,4,4],[6,4,4],[4,2,4]])
        altered=image.copy();altered[~valid]=99999
        np.testing.assert_array_equal(physical_moments(image,valid,[1,1,2],points),physical_moments(altered,valid,[1,1,2],points))
        for a,b in zip(physical_gradients(image,valid,[1,1,2],points),physical_gradients(altered,valid,[1,1,2],points)):
            np.testing.assert_array_equal(a,b)

    def test_missing_gradient_is_explicit(self):
        image=np.ones((3,3,3));valid=np.zeros(image.shape,bool);valid[1,1,1]=True
        grad,available=physical_gradients(image,valid,[1,1,1],[[1,1,1]])
        self.assertFalse(grad.any());self.assertFalse(available.any())

    def test_invalid_measurements_fail(self):
        image=np.ones((3,3,3));mask=np.ones(image.shape,bool)
        for spacing in ([0,1,1],[np.nan,1,1]):
            with self.assertRaises(ValueError):physical_moments(image,mask,spacing,[[1,1,1]])
        with self.assertRaises(ValueError):physical_moments(image,np.zeros_like(mask),[1,1,1],[[1,1,1]])
        with self.assertRaises(ValueError):physical_moments(image,mask,[1,1,1],[[1,1,1]],[np.nan])
        image[0,0,0]=np.nan
        with self.assertRaises(ValueError):physical_moments(image,mask,[1,1,1],[[1,1,1]])

    def test_node_context_excludes_footprint_at_every_active_column(self):
        import copy
        from types import SimpleNamespace
        shape=(9,9,9);image=np.arange(np.prod(shape),dtype=np.float32).reshape(shape)/1000
        footprint=np.zeros(shape,bool);footprint[3:6,3:6,3:6]=True
        fields=SimpleNamespace(ct_norm=image,footprint=footprint,organ=np.ones(shape,bool),
            liver_normal=np.zeros(shape+(3,)),tumor_normal=np.zeros(shape+(3,)),
            tumor_sdf_mm=np.ones(shape),liver_depth_mm=np.ones(shape))
        specs={'target_context':(np.array([[2,4,4],[6,4,4]]),'tumor',0)}
        cfg=SimpleNamespace(context_shells_mm=(4,12,28),context_radius_mm=28)
        before=node_observations(fields,specs,[1,2,3],cfg)
        fields=copy.deepcopy(fields);fields.ct_norm[footprint]=1e6
        after=node_observations(fields,specs,[1,2,3],cfg)
        np.testing.assert_array_equal(before['target_context'],after['target_context'])
        self.assertEqual(before['target_context'].shape,(2,WIDTH))


if __name__=='__main__':unittest.main()
