"""Input accounting for complete topology and CT-only patches."""
from hiercp.training_resources import _tensor_bytes

def _local_bound(local):
    if local.get('format')!='canonical-full-v22':raise ValueError('Canonical topology required')
    nodes=sum(int(table['grid'].shape[0]) for table in local['nodes'].values())
    edges=sum(int(index.shape[1]) for index in local['edges'].values())
    return nodes,edges,edges*16+_tensor_bytes(local['nodes'])+nodes*64+512
