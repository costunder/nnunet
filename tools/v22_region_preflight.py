"""Read-only dependency/kernel admission before cache IO; never installs packages."""
import importlib
import json
import os
import re
import torch


def wheel_page(version, cuda):
    match=re.match(r'^(\d+)\.(\d+)\.',version)
    if not match or not cuda:raise ValueError('A released CUDA PyTorch build is required')
    # Official PyG installation matrix uses the minor series base wheel page.
    return f'https://data.pyg.org/whl/torch-{match[1]}.{match[2]}.0+cu{cuda.replace(".", "")}.html'


def dependencies():
    modules={}
    for name in ('torch_geometric','torch_scatter'):
        try:modules[name]=importlib.import_module(name)
        except (ImportError,OSError) as exc:
            url=wheel_page(torch.__version__,torch.version.cuda)
            raise RuntimeError(f'Region preparation dependency unavailable: {name}. '
                f'Python env torch={torch.__version__}, CUDA build={torch.version.cuda}. '
                f'Install the matching binary torch-scatter wheel from {url}; '
                'do not replace PyTorch/CUDA. No cache preparation or training has started.') from exc
    return modules


def check():
    modules=dependencies()
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; no CPU fallback')
    from l0_ezsp.backend import official
    merge,_=official()  # Verifies the unchanged official source hashes.
    device=torch.device('cuda',torch.cuda.current_device())
    with torch.no_grad(),torch.random.fork_rng(devices=[device.index]):
        index=torch.tensor([0,0,1],device=device)
        values=torch.tensor([1.,2.,3.],device=device)
        torch.testing.assert_close(modules['torch_scatter'].scatter_sum(values,index),values.new_tensor([3.,3.]))
        minima,_=modules['torch_scatter'].scatter_min(values,index)
        torch.testing.assert_close(minima,values.new_tensor([1.,3.]))
        # Explicit synthetic CUDA kernel probe, never a training/data sample.
        x=torch.tensor([[1.,0.],[1.,0.],[0.,1.],[0.,1.]],device=device)
        e=torch.tensor([[0,1,2],[1,2,3]],device=device)
        parent,_,_=merge(x,torch.ones(4,device=device),e,torch.ones(3,device=device),.02,
            min_size=1,k=-1,max_iterations=32)
        if not bool((parent[0]==parent[1]) & (parent[2]==parent[3]) & (parent[0]!=parent[2])):
            raise RuntimeError('Official CUDA merge kernel probe failed')
        torch.cuda.synchronize()
    result=dict(stage='region_dependency_preflight',status='PASS',torch=torch.__version__,cuda_build=torch.version.cuda,
        torch_scatter=modules['torch_scatter'].__version__,torch_geometric=modules['torch_geometric'].__version__,
        gpu=torch.cuda.get_device_name(device),cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        scope='synthetic scatter/official merge CUDA kernel probe only',training_started=False)
    print(json.dumps(result),flush=True)
    return result


if __name__=='__main__':check()
