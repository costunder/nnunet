"""Single-arm v2 full-cohort evaluation using the existing v5 metric definitions."""
import csv
from pathlib import Path
import numpy as np
import nibabel as nib
from tools.online_eval_v2 import CRITERIA,component_pair_matrices,match_components,whole_mask_dice,equivalent_diameter,size_bin,max_dice_match
from .contracts import read_json,write_new,safe_new_root,sha,validate_native
from . import PIPELINE_VERSION

def write_csv(path,rows,fields):
    with Path(path).open('x',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader(); writer.writerows(rows)

def evaluate(native_path,predictions,output,*,debug=False):
    native=validate_native(read_json(native_path)); predictions=Path(predictions); cases=native['split']['outer_val']
    actual={p.name[:-7] for p in predictions.glob('*.nii.gz')}
    if actual!=set(cases): raise ValueError('Predictions must contain exactly all outer-validation cases')
    root=safe_new_root(output); case_rows=[]; lesion_rows=[]; provenance=[]
    for case in cases:
        gt_path=Path(native['raw'])/'labelsTr'/f'{case}.nii.gz'; pred_path=predictions/f'{case}.nii.gz'
        gt_img=nib.load(gt_path); pred_img=nib.load(pred_path)
        if gt_img.header.get_xyzt_units()[0]!='mm':
            raise ValueError(f'Explicit millimetre units required for size metrics: {case}')
        if gt_img.shape!=pred_img.shape or not np.allclose(gt_img.affine,pred_img.affine,rtol=0,atol=1e-5):
            raise ValueError(f'Prediction geometry mismatch: {case}')
        gt_labels=np.asanyarray(gt_img.dataobj); pr_labels=np.asanyarray(pred_img.dataobj)
        if not np.isin(gt_labels,[0,1,2]).all() or not np.isin(pr_labels,[0,1,2]).all():
            raise ValueError('Unsupported segmentation labels')
        gt,pr=gt_labels==2,pr_labels==2
        matrices=component_pair_matrices(gt,pr); dice=whole_mask_dice(gt,pr)
        volume=float(np.prod(gt_img.header.get_zooms()[:3])); quality=max_dice_match(matrices)
        for criterion in CRITERIA:
            match=match_components(matrices,criterion); tp=match.tp
            row={'case_id':case,'criterion':criterion.name,'tumor_dice':dice,'gt_lesions':matrices.num_gt,
                 'predicted_lesions':matrices.num_pred,'tp':tp,'fn':matrices.num_gt-tp,'fp':matrices.num_pred-tp}
            case_rows.append(row)
            for i,size in enumerate(matrices.gt_sizes):
                diameter=equivalent_diameter(float(size)*volume); j=quality.gt_to_pred.get(i)
                lesion_rows.append({'case_id':case,'gt_component':i+1,'diameter_mm':diameter,'size_bin':size_bin(diameter,[10,20]),
                    'criterion':criterion.name,'detected':int(i in match.gt_to_pred),'best_1to1_lesion_dice':float(matrices.dice[i,j]) if j is not None else 0.})
        provenance.append({'case_id':case,'gt_sha256':sha(gt_path),'prediction_sha256':sha(pred_path)})
    summary=[]; sizes=[]
    for criterion in CRITERIA:
        rows=[r for r in case_rows if r['criterion']==criterion.name]
        tp=sum(r['tp'] for r in rows); fn=sum(r['fn'] for r in rows); fp=sum(r['fp'] for r in rows)
        summary.append({'criterion':criterion.name,'cases':len(rows),'mean_tumor_dice':float(np.mean([r['tumor_dice'] for r in rows])),
            'tp':tp,'fn':fn,'fp':fp,'recall':tp/(tp+fn) if tp+fn else None,'precision':tp/(tp+fp) if tp+fp else None,
            'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,'fp_per_case':fp/len(rows)})
        for group in ['le_10mm','gt_10_le_20mm','gt_20mm']:
            selected=[r for r in lesion_rows if r['criterion']==criterion.name and r['size_bin']==group]
            detected=sum(r['detected'] for r in selected)
            sizes.append({'criterion':criterion.name,'size_bin':group,'gt':len(selected),'detected':detected,
                'recall':detected/len(selected) if selected else None,
                'mean_lesion_dice':float(np.mean([r['best_1to1_lesion_dice'] for r in selected])) if selected else None})
    write_csv(root/'case_metrics.csv',case_rows,list(case_rows[0]))
    write_csv(root/'lesion_metrics.csv',lesion_rows,['case_id','gt_component','diameter_mm','size_bin','criterion','detected','best_1to1_lesion_dice'])
    write_csv(root/'summary.csv',summary,list(summary[0])); write_csv(root/'size_metrics.csv',sizes,list(sizes[0]))
    write_new(root/'completion.json',{'format':PIPELINE_VERSION,'metric_definition':'online_basic_hiercp_evaluation_v5',
        'native_sha256':sha(native_path),'cases':provenance,'files':{p.name:sha(p) for p in root.glob('*.csv')},
        'inference_provenance':'See prediction_complete.json; arbitrary input predictions do not prove checkpoint identity',
        'empty_denominators':'unavailable/null; no fabricated zero','debug':debug,
        'data_scope':'synthetic debug fixtures' if debug else 'caller-supplied complete outer-validation cohort'})
    return root
