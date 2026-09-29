"""Load the unchanged, pinned official code; reject modified vendor sources."""
import hashlib
import json
from pathlib import Path
from functools import lru_cache

COMMIT='e3db9f352fae52dff416616742b3c7ff1378451d'
MERGE_BLOB='f37f803137e2cb28d17b75800e239a5b6406f778'

@lru_cache(maxsize=1)
def official():
    root=Path(__file__).parent/'vendor/torch_graph_components'
    pin=json.loads((root/'PIN.json').read_text())
    if pin['commit']!=COMMIT or pin['files']['merge.py']['git_blob']!=MERGE_BLOB:
        raise ValueError('Official EZ-SP pin differs from reconciled handoff')
    for name,entry in pin['files'].items():
        data=(root/name).read_bytes()
        blob=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
        if hashlib.sha256(data).hexdigest()!=entry['sha256'] or blob!=entry['git_blob']:
            raise ValueError('Modified official source: '+name)
    from .vendor.torch_graph_components.merge import merge_components_by_contour_prior
    from .vendor.torch_graph_components.wcc import wcc_by_max_propagation
    return merge_components_by_contour_prior,wcc_by_max_propagation
