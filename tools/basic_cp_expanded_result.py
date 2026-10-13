"""Keep Result schema1 while admitting the seven requested interventions.

Only intervention cardinality and display labels change. Original metric,
BEST proof, file-byte checks and exclusive archive functions remain intact.
"""
import inspect
from pathlib import Path
from types import FunctionType
from tools import basic_cp_ablation_completed_result as base

ARMS={'multi5':'five lesions','multi7':'seven lesions','multi10':'ten lesions',
      'candidates512':'512 positions','candidates1024':'1024 positions',
      'candidates2048':'2048 positions','blend50':'constant blending donor50% recipient50%'}
COPIES={'multi5':5,'multi7':7,'multi10':10,'candidates512':1,'candidates1024':1,'candidates2048':1,'blend50':1}
scope=dict(base.__dict__,ARMS=ARMS,COPIES=COPIES)


def cloned(function):
    result=FunctionType(function.__code__,scope,function.__name__,function.__defaults__,function.__closure__)
    result.__kwdefaults__=function.__kwdefaults__
    return result


for name,value in base.__dict__.items():
    if inspect.isfunction(value):scope[name]=cloned(value)

source=inspect.getsource(base.training_record)
old="events*(3 if arm=='multi3' else 1)"
if source.count(old)!=1:raise RuntimeError('Exact original paste telemetry guard required')
source=source.replace(old,'events*COPIES[arm]')
exec(compile(source,str(Path(__file__).resolve()),'exec'),scope)

source=inspect.getsource(base.completed_report)
edits={
 "'constant blending donor80% recipient20%' if native.get('blend_mode') == 'constant' else ARMS[native['arm']]":"ARMS[native['arm']]",
 "(256 if native['arm']=='candidates256' else 128)":"native['candidates_per_source']",
 "copied_lesions_per_successful_sample=3 if native['arm']=='multi3' else 1":"copied_lesions_per_successful_sample=native['copies_per_CP_event']",
 "location_policy='uniform complete candidate universe; GNN scores unused'":"location_policy=('original uniform primary retained when joint feasible; impossible primary redrawn uniformly within same source original128 joint-feasible locations; GNN scores unused' if native.get('primary_joint_resampling_policy') else 'uniform complete candidate universe; GNN scores unused'), primary_joint_resampling_policy=native.get('primary_joint_resampling_policy')",
}
for old,new in edits.items():
    if source.count(old)!=1:raise RuntimeError('Exact original result cardinality expression required: '+old)
    source=source.replace(old,new)
exec(compile(source,str(Path(__file__).resolve()),'exec'),scope)
completed_report=scope['completed_report']
archive_completed=scope['archive_completed']
training_record=scope['training_record']
