"""Read-only header audit of the complete CT/label cohort."""
from pathlib import Path
import json,sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
    import nibabel as nib
    from hiercp.common import discover_cases
    from hiercp_v222.placement import grid_error_voxels,GRID_TOLERANCE_VOXELS
    data=Path(sys.argv[1]);output=Path(sys.argv[2])
    paths=discover_cases(data);split=json.loads((ROOT/'config/split_cp80_fold0.json').read_text())
    if {p.case_id for p in paths}!=set(split['outer_train']+split['outer_val']):raise ValueError('Full cohort required')
    def check(p):
        image=nib.load(p.image_path);label=nib.load(p.label_path)
        a,b=image.affine,label.affine
        spacing=np.asarray(image.header.get_zooms()[:3]);other=np.asarray(label.header.get_zooms()[:3])
        ok=(image.shape==label.shape and np.isfinite(a).all() and np.isfinite(b).all() and
            np.isfinite(spacing).all() and np.isfinite(other).all() and (spacing>0).all() and (other>0).all() and
            grid_error_voxels(a,b,image.shape)<=GRID_TOLERANCE_VOXELS and np.allclose(spacing,other,rtol=1e-5,atol=1e-5) and
            np.allclose(np.linalg.norm(a[:3,:3],axis=0),spacing,rtol=1e-5,atol=1e-5) and abs(np.linalg.det(a[:3,:3]))>=1e-12)
        return dict(case_id=p.case_id,passed=bool(ok),shape=list(image.shape),spacing=spacing.tolist(),max_grid_error_voxels=grid_error_voxels(a,b,image.shape))
    with ThreadPoolExecutor(max_workers=8) as pool:rows=list(pool.map(check,paths))
    value=dict(scope='complete cohort physical-grid header audit; no training or image resampling',cases=len(rows),passed=all(r['passed'] for r in rows),
               tolerance_voxels=GRID_TOLERANCE_VOXELS,max_grid_error_voxels=max(r['max_grid_error_voxels'] for r in rows),rows=rows)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2)
    print(json.dumps({k:v for k,v in value.items() if k!='rows'}))
    if not value['passed']:raise ValueError('Physical-grid audit failed')


if __name__=='__main__':main()
