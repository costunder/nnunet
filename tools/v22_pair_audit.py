"""Measure the existing minibatch estimator without changing its weights."""
import numpy as np


def pair_audit(rows,batches):
    owners={};occurrences={};coefficients={};batch_sizes={}
    by_case={}
    for i,row in enumerate(rows):by_case.setdefault(row['case_id'],[]).append(i)
    for case,ids in by_case.items():
        p=[i for i in ids if rows[i]['target']==1];u=[i for i in ids if rows[i]['target']==0]
        owners[case]=(np.asarray(p,dtype=int),np.asarray(u,dtype=int))
        occurrences[case]=np.zeros((len(p),len(u)),np.int64)
        coefficients[case]=np.zeros((len(p),len(u)),np.float64)
    seen=[]
    for ids in batches:
        seen.extend(ids);selected=set(ids)
        touched={};count=0
        for case in {rows[i]['case_id'] for i in ids}:
            p,u=owners[case]
            mask=np.isin(p,ids)[:,None]|np.isin(u,ids)[None,:]
            touched[case]=mask;count+=int(mask.sum())
        if count:
            for case,mask in touched.items():
                occurrences[case]+=mask;coefficients[case]+=mask/count
        batch_sizes[len(ids)]=batch_sizes.get(len(ids),0)+1
    if sorted(seen)!=list(range(len(rows))):raise ValueError('Estimator audit requires exact query coverage')
    reports=[]
    for case,(p,u) in owners.items():
        weights=coefficients[case];counts=occurrences[case]
        reports.append(dict(case_id=case,pairs=int(weights.size),
            occurrence_min=int(counts.min()) if counts.size else None,occurrence_max=int(counts.max()) if counts.size else None,
            coefficient_min=float(weights.min()) if weights.size else None,coefficient_max=float(weights.max()) if weights.size else None,
            coefficient_sum=float(weights.sum()),
            graph_edges_min=min(rows[i]['bounds']['edges'] for i in by_case[case]),
            graph_edges_max=max(rows[i]['bounds']['edges'] for i in by_case[case])))
    return dict(estimator_unchanged=True,scope='fixed-score scalar coefficients; not full-case L0 gradient equivalence',
                physical_batch_histogram=batch_sizes,cases=reports)
