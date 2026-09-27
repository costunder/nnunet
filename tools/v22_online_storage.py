"""Online/native/checkpoint storage admission, separate from training graphs."""
import json
from pathlib import Path
import shutil
import threading
import numpy as np
import torch

_write_lock=threading.RLock()


def tree_bytes(value):
    """Conservative payload estimate, including array/serialization overhead."""
    if isinstance(value,np.ndarray):return int(value.nbytes)+4096
    if torch.is_tensor(value):return value.numel()*value.element_size()+4096
    if isinstance(value,dict):return 4096+sum(tree_bytes(k)+tree_bytes(v) for k,v in value.items())
    if isinstance(value,(list,tuple)):return 4096+sum(tree_bytes(v) for v in value)
    return len(repr(value).encode())+128


def check_tree_write(root,value,reserve,*,kind):
    root=Path(root)
    while not root.exists():root=root.parent
    required=2*tree_bytes(value)+1024**2  # New temporary + conservative serialization overhead.
    free=shutil.disk_usage(root).free
    if free<=required+reserve:
        raise OSError(f'{kind} disk admission failed: free={free}, write_estimate={required}, reserve={reserve}; not a no-placement event')
    return dict(kind=kind,estimated_write_bytes=required,free_before_bytes=free,reserved_free_bytes=reserve)


def save_online_case(root,relative,value,reserve):
    from custom_trainers.onlinecp_raw_bank import save_case
    with _write_lock:
        report=check_tree_write(root,value,reserve,kind='online_native_case')
        digest=save_case(root,relative,value)
    return digest,report


def save_online_candidate(root,relative,value,reserve):
    from custom_trainers.onlinecp_raw_bank import save_candidate
    with _write_lock:
        report=check_tree_write(root,value,reserve,kind='online_selected_candidate')
        digest=save_candidate(root,relative,value)
    return digest,report


def forecast(root,meta):
    """Empirical per-event estimates, not an upper bound or automatic eviction."""
    root=Path(root);native=[];pairs=[]
    for path in (root/'raw_receipts').glob('*.json'):
        row=json.loads(path.read_text(encoding='utf-8'))
        estimates=[r['estimated_write_bytes'] for r in row.get('storage',[]) if r['kind']=='online_native_combined']
        native.extend(estimates)
    for path in (root/'entries').glob('*.receipt.json'):
        row=json.loads(path.read_text(encoding='utf-8'))
        storage=row.get('transport_audit',{}).get('storage')
        if storage:pairs.append(storage['estimated_write_bytes'])
    recipients=len(meta['split']['outer_train']);slots=recipients*len(meta['donor_pool'])
    return dict(observed_native_cases=len(native),observed_selected_pairs=len(pairs),
        full_recipient_count=recipients,full_pair_slots=slots,
        empirical_native_bytes=max(native)*recipients if native else None,
        empirical_selected_pair_bytes=max(pairs)*slots if pairs else None,
        scope='Observed maximum write estimate times full count; shared-source duplication is conservative. Not a size guarantee. Graph cache, original CT and segmentation checkpoints excluded; no truncation or eviction.')


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bank',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(forecast(args.bank.parent,json.loads(args.bank.read_text(encoding='utf-8'))),indent=2))
