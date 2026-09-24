"""One shared, training-only donor policy for every augmentation recipient."""
import numpy as np
from .contracts import validate_split

POLICY = 'shared_inner_train_donors_uniform_per_cp_event_v1'
CONTEXT = 'balanced_complete_donor_round_per_partition_v1'

def donor_pool(inventory, split):
    validate_split(split)
    pool = [{'case_id':case, 'component_id':int(component)}
            for case in sorted(split['inner_train']) for component in sorted(inventory[case])]
    if len({p['case_id'] for p in pool}) < 2:
        raise ValueError('At least two independent inner-training donor patients required')
    if len({(p['case_id'],p['component_id']) for p in pool}) != len(pool):
        raise ValueError('Duplicate donor identity')
    return pool

def validate_pool(pool, split):
    validate_split(split)
    identities=[]
    for row in pool:
        if row['case_id'] not in split['inner_train']:
            raise ValueError('Donor leakage: donor is not in inner_train')
        if type(row['component_id']) is not int or row['component_id'] < 1:
            raise ValueError('Invalid donor component')
        identities.append((row['case_id'],row['component_id']))
    if not identities or len(set(identities)) != len(identities):
        raise ValueError('Empty or duplicate donor pool')

def context_round(pool, recipients, seed):
    """Cover every real donor and every recipient without a Cartesian product.

    This is a fixed GNN context training set, not the native CP donor schedule.
    A round has max(donors, recipients) events; both sides are shuffled/cycled.
    Native CP instead draws from the entire same pool at EVERY CP event.
    """
    if not pool or not recipients or len(set(recipients)) != len(recipients):
        raise ValueError('Nonempty donors and unique recipients required')
    rng=np.random.default_rng(seed)
    donors=rng.permutation(len(pool)); patients=rng.permutation(sorted(recipients))
    return [{'recipient':str(patients[i % len(patients)]), 'donor_index':int(donors[i % len(donors)])}
            for i in range(max(len(pool),len(patients)))]

def select_donor(pool, split, recipient, u):
    # The recipient's tumor count is deliberately not an input to this policy.
    if recipient not in split['outer_train']:
        raise ValueError('Augmentation recipient must be outer_train; held-out CP is forbidden')
    if not np.isfinite(u) or not 0 <= u < 1:
        raise ValueError('Uniform donor draw must be in [0,1)')
    return int(u*len(pool))

def validate_training_rows(rows, split, pool):
    validate_pool(pool,split)
    allowed={(p['case_id'],p['component_id']) for p in pool}
    for row in rows:
        if row['case_id'] not in split['outer_train']:
            raise ValueError('Held-out patient in GNN context cache')
        if row['evidence']==1:
            if row['case_id']!=row['donor_case_id']:
                raise ValueError('Foreign donor query cannot become observed T')
        elif row['evidence']==-1:
            if (row['donor_case_id'],row['component_id']) not in allowed:
                raise ValueError('Query donor outside training-only pool')
        else:
            raise ValueError('Prepared observations must be real T or unobserved U')

def reject_cross_split_duplicates(records, split):
    """Identical original images cannot cross patient split boundaries."""
    validate_split(split)
    groups={c:k for k in ('inner_train','inner_val','outer_val') for c in split[k]}
    seen={}
    for row in records:
        case=row['case_id']; group=groups[case]; digest=row['image_sha256']
        if digest in seen and seen[digest][1]!=group:
            raise ValueError(f'Identical-image leakage across partitions: {seen[digest][0]} / {case}')
        seen[digest]=(case,group)
