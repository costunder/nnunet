"""Actual RAM512 limit values over the unchanged UI2 and native console."""
from pathlib import Path
import sys
from types import FunctionType

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools import watch_basic_cp_admission as admission


def build_entry(expanded,viewer):
    entry=admission.build_entry(expanded,viewer)
    original=entry.__globals__['snapshot']
    def snapshot(root):
        value=original(root);request=viewer.read(Path(root)/'request.json')
        required={'memory_budget_revision':512,'aggregate_memory_cap_GiB':512,
            'effective_native_RSS_limit_GiB':32,'effective_shared_bank_RSS_limit_GiB':320}
        if not isinstance(request,dict) or any(type(request.get(k))is not int or request[k]!=v for k,v in required.items()):
            raise ValueError('RAM512 viewer requires its actual approved request limits')
        cpu_source=value['physical_GPU']==5 and value['controller_stage']=='candidate_bank'
        return dict(value,RAM_limit_GiB=request['effective_shared_bank_RSS_limit_GiB']if cpu_source
            else request['effective_native_RSS_limit_GiB'])
    scope=dict(entry.__globals__,snapshot=snapshot)
    return FunctionType(entry.__code__,scope,entry.__name__,entry.__defaults__,entry.__closure__)


def main(argv=None):
    scope=dict(admission.main.__globals__,build_entry=build_entry)
    entry=FunctionType(admission.main.__code__,scope,admission.main.__name__,
        admission.main.__defaults__,admission.main.__closure__)
    return entry(argv)


if __name__=='__main__':main()
