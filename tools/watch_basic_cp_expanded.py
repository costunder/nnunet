"""Existing UI2/native console; only experiment bindings/counts vary."""
import argparse
import json
from pathlib import Path
import sys
from types import FunctionType

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools import watch_basic_cp_ablation as viewer


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--gpu',type=int,choices=range(7),required=True)
    args,rest=parser.parse_known_args()
    manifest=json.loads(args.manifest.read_bytes());root=Path(manifest['registration'][str(args.gpu)]['root'])
    original=viewer.snapshot
    def snapshot(path):
        value=original(path);request=viewer.read(Path(path)/'request.json')
        count=request['candidates_per_source'];value['candidates_total']=642*count
        if value['sources_done']is not None:value['candidates_done']=value['sources_done']*count
        if value['controller_stage']=='candidate_bank':value['stage_label']=f'CP 위치 후보 {count}개 준비'
        return value
    scope=dict(viewer.__dict__,snapshot=snapshot)
    entry=FunctionType(viewer.main.__code__,scope,viewer.main.__name__,viewer.main.__defaults__)
    return entry(['--root',str(root),*rest])


if __name__=='__main__':main()
