"""Resolve a physical nvidia-smi GPU index before PyTorch initializes CUDA."""
import os
import re
import subprocess

def resolve(index,listing,visible=''):
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
    allocated=set(visible.split(','));selected=[value for value in slices if value in allocated]
    if len(selected)==1:return selected[0],'MIG'
    if len(slices)==1:return slices[0],'MIG'
    raise ValueError(f'Physical GPU {index} has {len(slices)} MIG instances; current allocation does not identify exactly one. No arbitrary MIG was selected.')

def select(index):
    if index<0:raise ValueError('GPU index must be nonnegative')
    listing=subprocess.check_output(['nvidia-smi','-L'],text=True)
    visible,kind=resolve(index,listing,os.environ.get('CUDA_VISIBLE_DEVICES',''))
    os.environ['CUDA_VISIBLE_DEVICES']=visible
    print(f'GPU selection | physical={index} | type={kind} | CUDA_VISIBLE_DEVICES={visible}',flush=True)
    return visible
