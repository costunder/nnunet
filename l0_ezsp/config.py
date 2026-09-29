import copy
import hashlib
import json
import math
from pathlib import Path

CONTRACT_PATH = Path(__file__).resolve().parents[1]/'config/l0_ezsp_unresolved.json'

def load_profile(*, reg_scale1, reg_scale2, sharding=None):
    """Both reg arguments are required even for a DEBUG forward."""
    for value in (reg_scale1,reg_scale2):
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<0:
            raise ValueError('Both reg values must be explicit finite nonnegative numbers; null has no default')
    if sharding is not None and (type(sharding) is not int or sharding<=1):
        raise ValueError('sharding must be null or an explicit integer edge chunk > 1')
    cfg=json.loads(CONTRACT_PATH.read_text(encoding='utf-8'))
    cfg['partition'].update(reg_scale1=float(reg_scale1),reg_scale2=float(reg_scale2))
    cfg['backend']['options']['sharding']=sharding
    return cfg

def fingerprint(cfg):
    return hashlib.sha256(json.dumps(cfg,sort_keys=True,allow_nan=False,separators=(',',':')).encode()).hexdigest()

def validate(cfg):
    expected=load_profile(reg_scale1=cfg['partition']['reg_scale1'],reg_scale2=cfg['partition']['reg_scale2'],
                          sharding=cfg['backend']['options']['sharding'])
    if cfg!=expected:raise ValueError('Unreviewed EZ-SP profile change; use the explicit diagnostic contract')
    return copy.deepcopy(cfg)
