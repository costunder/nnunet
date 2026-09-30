"""Resolve a physical nvidia-smi GPU index before PyTorch initializes CUDA."""
import os
import getpass
import json
from pathlib import Path
import re
import socket
import subprocess

ALLOCATIONS=Path(__file__).resolve().parents[1]/'config/server_gpu_allocations.json'

def recorded_allocation(index,host,user):
    """Explicit project allocation evidence, not an availability-based scheduler."""
    rows=json.loads(ALLOCATIONS.read_text(encoding='utf8'))['allocations']
    matches=[row for row in rows if row['host']==host.split('.')[0]
             and row['user']==user and row['physical_index']==index]
    if len(matches)>1:raise ValueError('Duplicate recorded GPU allocation')
    return matches[0] if matches else None

def resolve(index,listing,visible='',allocation=None):
    devices={};current=None
    for line in listing.splitlines():
        gpu=re.match(r'^GPU (\d+):.*\(UUID:\s*([^)]+)\)',line)
        if gpu:
            current=int(gpu[1]);devices[current]={'uuid':gpu[2].strip(),'mig':[]}
            continue
        mig=re.match(r'^\s+MIG .*Device\s+(\d+):.*\(UUID:\s*([^)]+)\)',line)
        if mig and current is not None:devices[current]['mig'].append(mig[2].strip())
    if index not in devices:raise ValueError(f'Physical GPU {index} is not visible in nvidia-smi -L')
    device=devices[index];slices=device['mig']
    if not slices:return device['uuid'],'GPU'
    allocated={value.strip() for value in visible.split(',')};selected=[value for value in slices if value in allocated]
    if len(selected)==1:return selected[0],'MIG'
    if len(selected)>1:raise ValueError('Current allocation identifies multiple MIG instances; no recorded allocation override')
    if allocation is not None:
        if allocation['physical_index']!=index or allocation['gpu_uuid']!=device['uuid']:
            raise ValueError('Recorded physical GPU identity changed; allocation must be rechecked')
        if allocation['mig_uuid'] not in slices:
            raise ValueError('Recorded MIG allocation no longer exists on this GPU; allocation must be rechecked')
        return allocation['mig_uuid'],'MIG'
    if len(slices)==1:return slices[0],'MIG'
    raise ValueError(f'Physical GPU {index} has {len(slices)} MIG instances; current allocation does not identify exactly one. No arbitrary MIG was selected.')

def select(index):
    if index<0:raise ValueError('GPU index must be nonnegative')
    listing=subprocess.check_output(['nvidia-smi','-L'],text=True)
    allocation=recorded_allocation(index,socket.gethostname(),getpass.getuser())
    visible,kind=resolve(index,listing,os.environ.get('CUDA_VISIBLE_DEVICES',''),allocation)
    os.environ['CUDA_VISIBLE_DEVICES']=visible
    print(f'GPU selection | physical={index} | type={kind} | CUDA_VISIBLE_DEVICES={visible}',flush=True)
    return visible
