"""Explicit, audited reuse of data geometry; never migrate learned weights."""
from pathlib import Path
from . import PIPELINE_VERSION
from .contracts import read_json,write_new,sha,safe_new_root,validate_split
from .storage import GraphWriter


def convert_record(record):
    raise ValueError('Raw CT rebuild required for CT-only CNN features; old feature caches are incompatible')


def convert_cache(index_path,output,cfg,base):
    raise ValueError('Raw CT rebuild required: use run_v221.py prepare with the original medical root and complete split; old caches cannot be relabelled')


def reuse_native(native_path,output):
    """Read-only reuse of preprocessing; independent runtime and results root."""
    from hiercp_v2.contracts import validate_native as validate_old
    path=Path(native_path).resolve(); old=validate_old(read_json(path))
    root=safe_new_root(output)
    value=dict(old,format=PIPELINE_VERSION,root=str(root),
               preprocessing_origin={'path':str(path),'sha256':sha(path),'release':'v2.1'},
               results_reused=False)
    write_new(root/'native.json',value)
    return root/'native.json'
