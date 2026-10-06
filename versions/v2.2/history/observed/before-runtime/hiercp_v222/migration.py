"""Read-only reuse of verified preprocessing, never reuse older learned scores."""
from pathlib import Path
from .contracts import read_json, write_new, safe_new_root, sha
from . import PIPELINE_VERSION

def reuse_native(path, output):
    from hiercp_v2.contracts import validate_native
    path = Path(path).resolve(); old = validate_native(read_json(path))
    if set(old['planning_patient_ids']) != set(old['split']['outer_train']):
        raise ValueError('Preprocessing planning provenance is not training-only')
    root = safe_new_root(output)
    value = dict(old, format=PIPELINE_VERSION, root=str(root), results_reused=False,
                 preprocessing_origin=dict(path=str(path), sha256=sha(path), release='v2.1'))
    write_new(root/'native.json', value)
    return root/'native.json'
