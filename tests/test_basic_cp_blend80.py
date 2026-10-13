"""CPU DEBUG: exercise the actual paste method on known image/label arrays."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np


class Blend80Tests(unittest.TestCase):
    def test_all_mask_voxels_are_fixed80_after_intensity_jitter_and_labels_are_hard(self):
        source=Path(__file__).resolve().parents[1]/'custom_trainers/nnUNetTrainer_BasicCPBlend80.py'
        tree=ast.parse(source.read_text())
        nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))
               and n.name in ('constant_blend80','Blend80DataLoader')]
        nodes[1].bases=[ast.Name(id='object',ctx=ast.Load())]
        def require(ok,msg):
            if not ok: raise ValueError(msg)
        base=SimpleNamespace(require=require,historical=SimpleNamespace(
            _anchored_slices=lambda *a:tuple(slice(1,6) for _ in range(3))))
        ns={'np':np,'base':base}
        exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(source),'exec'),ns)
        loader=ns['Blend80DataLoader']();loader.arm='blend';loader.online_bank=SimpleNamespace(tumor_label=2)
        data=np.full((1,7,7,7),10,dtype=np.float32);labels=np.ones((1,7,7,7),dtype=np.int16)
        mask=np.zeros((5,5,5),dtype=bool);mask[1:4,1:4,1:4]=True
        entry={'source_mask':mask,'source_data':np.full((1,5,5,5),100,dtype=np.float32),
               'anchor_offset':np.array([2,2,2]),'_basic_ablation_alpha2':np.zeros((5,5,5))}
        loader._apply_paste_to_crop(data,labels,(0,0,0),dict(entry=entry,center=(3,3,3),scale=1.05,normalized_offset=5), 'DEBUG')
        expected=np.full_like(data,10);expected[:,2:5,2:5,2:5]=90
        expected_labels=np.ones_like(labels);expected_labels[:,2:5,2:5,2:5]=2
        np.testing.assert_allclose(data,expected)
        np.testing.assert_array_equal(labels,expected_labels)


if __name__=='__main__':unittest.main()
