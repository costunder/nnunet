"""Existing UI2/native console; only experiment bindings/counts vary."""
import argparse
import json
from pathlib import Path
import sys
from types import FunctionType

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools import watch_basic_cp_ablation as viewer

def expanded_snapshot(path):
    value=viewer.snapshot(path);request=viewer.read(Path(path)/'request.json')
    count=request['candidates_per_source'];value['candidates_total']=642*count
    if value['controller_stage']=='candidate_bank':
        generation=viewer.read(Path(request['shared_banks'])/'generation_progress.json')or{}
        value['sources_done']=generation.get('prefix_completed',{}).get(str(count))
        value['stage_label']=f'CP 위치 후보 {count}개 준비'
        if request['gpu']==5 and request.get('shared_candidate_geometry_cache'):
            value['RAM_limit_GiB']=request['shared_bank_RSS_limit_GiB']
    if value['sources_done']is not None:value['candidates_done']=value['sources_done']*count
    return value

def preparation_model(value):
    model=viewer.preparation_model(value)
    model['resources']['ram_limit_gib']=value['RAM_limit_GiB']
    return model


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--gpu',type=int,choices=range(7),required=True)
    args,rest=parser.parse_known_args()
    manifest=json.loads(args.manifest.read_bytes());root=Path(manifest['registration'][str(args.gpu)]['root'])
    scope=dict(viewer.__dict__,snapshot=expanded_snapshot,preparation_model=preparation_model)
    entry=FunctionType(viewer.main.__code__,scope,viewer.main.__name__,viewer.main.__defaults__)
    return entry(['--root',str(root),*rest])


if __name__=='__main__':main()
