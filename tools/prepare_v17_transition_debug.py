"""Prepare explicit real-CT DEBUG metadata, retaining all14102 GT/assignments.

Neural smoke subsequently selects named whole patients, with all recorded P
and128U intact. The population metadata is never turned into a tiny production
dataset. No image/annotation values, learned weights or output scores are made.
"""
from __future__ import annotations
import argparse,copy,json,hashlib
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-context',type=Path,required=True)
    p.add_argument('--native-debug-template',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    raw=json.loads(a.raw_context.read_text(encoding='utf8'))
    m=json.loads(a.native_debug_template.read_text(encoding='utf8'))
    if len(raw['records'])!=14102 or m.get('debug') is not True:
        raise ValueError('Actual complete raw observation metadata and explicitly DEBUG native template required')
    # This helper is invoked separately before frozen-source neural execution.
    from l0_regions.donor_data import assignment
    m['records']=assignment(raw,42)
    raw_rows={r['id']:r for r in raw['records']}
    for row in m['records']:
        original=raw_rows[row['id']]
        patch=a.raw_context.parent/original['patch']
        # Original context inventory has no fine-graph statistics. These zero
        # graph counts describe its CNN-only source, not the later D graph.
        # The byte key is an actual original tensor file size. D independently
        # reports its newly built graph counts and measured CUDA cost.
        row['bounds']=dict(nodes=0,edges=0,bytes=patch.stat().st_size)
        row['DEBUG_schedule_bounds_scope']='original CNN-only patch file; D graph cost measured independently'
    m['raw_records']=copy.deepcopy(raw['raw_records'])
    m['original_inventory_sha256']=hashlib.sha256(a.raw_context.read_bytes()).hexdigest()
    m['DEBUG_population_metadata_complete']=True
    m['DEBUG_neural_case_selection_requires_explicit_flags']=True
    a.output.mkdir(parents=True,exist_ok=False)
    with (a.output/'index.json').open('x',encoding='utf8') as f:json.dump(m,f,indent=2,allow_nan=False)
    print(json.dumps(dict(debug=True,records=len(m['records']),training_started=False,
                         output=str(a.output/'index.json'),actual_GT_unchanged=True)))

if __name__=='__main__':main()
