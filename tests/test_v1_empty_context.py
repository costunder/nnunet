"""Synthetic coordinate validator fixtures only; never training data."""
import unittest
from types import SimpleNamespace
import numpy as np
from hiercp_v22.spatial import validate_canonical_coordinates,EmptyCanonicalNodeError

class EmptyContextTests(unittest.TestCase):
    def setUp(self):
        mask=np.zeros((3,3,3),dtype=bool);mask[1,1,1]=True
        self.fields=SimpleNamespace(footprint=mask,organ=np.ones_like(mask),outside_tumor_mm=np.full(mask.shape,3.),liver_depth_mm=np.ones(mask.shape))
        self.coordinates=dict(surface=np.array([[1,1,1]]),context=np.empty((0,3),dtype=np.int64),liver_surface=np.array([[0,0,0]]))

    def test_empty_context_requires_explicit_opt_in(self):
        with self.assertRaises(EmptyCanonicalNodeError):
            validate_canonical_coordinates(self.fields,self.coordinates,{})
        validate_canonical_coordinates(self.fields,self.coordinates,{},allow_empty_context=True)

    def test_liver_surface_remains_mandatory(self):
        self.coordinates['liver_surface']=np.empty((0,3),dtype=np.int64)
        with self.assertRaises(EmptyCanonicalNodeError):
            validate_canonical_coordinates(self.fields,self.coordinates,{},allow_empty_context=True)

    def test_opt_in_does_not_relax_existing_geometry(self):
        self.coordinates['context']=np.array([[0,0,1]])
        with self.assertRaisesRegex(ValueError,'liver-surface band'):
            validate_canonical_coordinates(self.fields,self.coordinates,{},allow_empty_context=True)

if __name__=='__main__':unittest.main()
