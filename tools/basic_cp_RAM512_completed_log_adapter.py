"""Read-only native completed-log evidence; never edits checkpoints/raw logs."""
import ast
import copy
from datetime import datetime
import hashlib
import inspect
import json
import math
import re
import textwrap
import time

import numpy as np

FIELDS={'mean_fg_dice','ema_fg_dice','dice_per_class_or_region','train_losses','val_losses',
    'lrs','epoch_start_timestamps','epoch_end_timestamps'}
HEADER=re.compile(rb'(?m)^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d(?:\.\d{1,6})?): Epoch (\d+)\s*$')
LINE=re.compile(r'(?m)^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d(?:\.\d{1,6})?): (.*)$')


def require(ok,message):
    if not ok:raise ValueError(message)


def scalar(value,decimals):
    # Exactly the native print path, including NumPy32 tie/str semantics.
    return float(str(np.round(value,decimals=decimals)))


def digest(data):return hashlib.sha256(data).hexdigest()


def _normal(value):
    if isinstance(value,dict):return {str(k):_normal(v)for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [_normal(v)for v in value]
    if isinstance(value,np.ndarray):
        return {'native_numpy_dtype':str(value.dtype),'shape':list(value.shape),'values':_normal(value.tolist())}
    if isinstance(value,np.generic):
        return {'native_numpy_dtype':str(value.dtype),'value':_normal(value.item())}
    if isinstance(value,float)and not math.isfinite(value):return {'native_nonfinite':repr(value)}
    return value


def derive_completed_prefix(data,epochs,logging):
    require(type(data)is bytes and type(epochs)is int and 0<epochs<=250,'Actual raw bytes/full completed epoch count required')
    require(isinstance(logging,dict)and FIELDS<=set(logging)
        and all(isinstance(v,(list,tuple))and len(v)==epochs for v in logging.values()),
        'Every saved native logger array must contain exactly all completed epochs')
    headers=list(HEADER.finditer(data));require(headers,'Actual native timestamped Epoch headers required')
    blocks=[]
    for index,header in enumerate(headers):
        end=headers[index+1].start()if index+1<len(headers)else len(data)
        raw=data[header.start():end];rows=[(datetime.fromisoformat(m.group(1)).timestamp(),m.group(2))for m in LINE.finditer(raw.decode('utf8'))]
        fields={}
        for key,pattern in [('CP',r'\[OnlineCP\] epoch=(\d+) applied=(\d+)/(\d+)'),
                ('train',r'^train_loss (\S+)$'),('val',r'^val_loss (\S+)$'),
                ('dice',r'^Pseudo dice (\[.*\])$'),('LR',r'^Current learning rate: (\S+)$'),
                ('duration',r'^Epoch time: (\S+) s$')]:
            fields[key]=[(stamp,re.search(pattern,text))for stamp,text in rows if re.search(pattern,text)]
        blocks.append(dict(epoch=int(header.group(2)),start=header.start(),end=end,raw=raw,
            header_time=datetime.fromisoformat(header.group(1).decode()).timestamp(),fields=fields))
    selected=[];excluded=[];applied=attempted=0
    for epoch in range(epochs):
        candidates=[block for block in blocks if block['epoch']==epoch];matches=[]
        require(candidates,'Missing native epoch evidence: '+str(epoch))
        for block in candidates:
            f=block['fields'];complete=all(len(f[k])==1 for k in('CP','train','val','dice','LR','duration'))
            if not complete:
                require(not any(len(f[k])>1 for k in f),'Ambiguous duplicated completion fields in a raw epoch block')
                excluded.append(dict(epoch=epoch,byte_start=block['start'],byte_end=block['end'],
                    sha256=digest(block['raw']),reason='Unfinished attempt absent native completion fields',
                    missing_fields=[key for key in f if not f[key]]));continue
            start=float(logging['epoch_start_timestamps'][epoch]);end=float(logging['epoch_end_timestamps'][epoch])
            require(math.isfinite(start)and math.isfinite(end)and end>=start,'Finite ordered checkpoint epoch timestamps required')
            cp_stamp,cp=f['CP'][0];cp_epoch,used,total=map(int,cp.groups())
            train_stamp,train=f['train'][0];val_stamp,val=f['val'][0];dice_stamp,dice=f['dice'][0]
            lr_stamp,lr=f['LR'][0];duration_stamp,duration=f['duration'][0]
            clean=re.sub(r'np\.float(?:16|32|64)\(([^()]*)\)',r'\1',dice.group(1))
            observed_dice=[float(v.strip())for v in clean.strip('[]').split(',')if v.strip()]
            expected_dice=[scalar(value,4)for value in logging['dice_per_class_or_region'][epoch]]
            number_equal=lambda a,b:a==b or(math.isnan(a)and math.isnan(b))
            matched=(start<=block['header_time']<=lr_stamp<=cp_stamp<=end<=train_stamp<=val_stamp<=dice_stamp<=duration_stamp
                and 0<=train_stamp-end<=1 and duration_stamp-end<=1
                and cp_epoch==epoch and total==500 and 0<=used<=500
                and number_equal(float(train.group(1)),scalar(logging['train_losses'][epoch],4))
                and number_equal(float(val.group(1)),scalar(logging['val_losses'][epoch],4))
                and number_equal(float(lr.group(1)),scalar(logging['lrs'][epoch],5))
                and len(observed_dice)==len(expected_dice)and all(number_equal(a,b)for a,b in zip(observed_dice,expected_dice))
                and float(duration.group(1))==scalar(logging['epoch_end_timestamps'][epoch]-logging['epoch_start_timestamps'][epoch],2))
            require(matched,'A fully completed raw epoch disagrees with its exact saved native timestamp/metrics: '+str(epoch))
            matches.append((block,used,total))
        require(len(matches)==1,'Exactly one checkpoint-matched finished raw segment per epoch required: '+str(epoch))
        block,used,total=matches[0];selected.append(block);applied+=used;attempted+=total
    for block in blocks:
        if block['epoch']>=epochs:
            require(not all(len(block['fields'][key])==1 for key in('CP','train','val','dice','LR','duration')),
                'Fully finished raw epoch outside the saved checkpoint cannot be silently discarded')
            excluded.append(dict(epoch=block['epoch'],byte_start=block['start'],byte_end=block['end'],
                sha256=digest(block['raw']),reason='Outside saved completed checkpoint prefix'))
    ranges=[(0,headers[0].start()),*[(block['start'],block['end'])for block in selected]]
    derived=b''.join(data[a:b]for a,b in ranges)
    logger_bytes=json.dumps(_normal(logging),sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf8')
    proof=dict(format='basic_cp_native_checkpoint_matched_completed_log_v1',epochs=epochs,first_epoch=0,last_epoch=epochs-1,
        CP_applied=applied,CP_attempted=attempted,raw_log_sha256=digest(data),derived_log_sha256=digest(derived),
        checkpoint_logger_sha256=digest(logger_bytes),all_native_logger_keys=sorted(logging),native_datetime_timezone=list(time.tzname),
        selected_original_byte_ranges=[dict(byte_start=a,byte_end=b,sha256=digest(data[a:b]))for a,b in ranges],
        excluded_original_byte_ranges=excluded,original_raw_log_preserved=True,checkpoint_or_metric_values_changed=False,
        printed_native_rounding_preserved=True,synthesized_lines=False,renumbered_epochs=False)
    return derived,proof


def private_builder_main(module,*,adapter_path,adapter_sha256):
    """Exactly two operational boundaries; canonical checkpoint maths retained."""
    function=module.main;require(function.__closure__ is None,'Closure-free canonical builder required')
    original=ast.parse(textwrap.dedent(inspect.getsource(function)))
    ast.increment_lineno(original,function.__code__.co_firstlineno-1)
    namespace=dict(function.__globals__);local={}
    exec(compile(ast.fix_missing_locations(copy.deepcopy(original)),function.__code__.co_filename,'exec'),namespace,local)
    rebuilt=local[function.__name__]
    require(rebuilt.__code__.co_code==function.__code__.co_code and rebuilt.__code__.co_consts==function.__code__.co_consts,
        'Inspected canonical builder source must match original actual bytecode')
    tree=copy.deepcopy(original);saved_sites=prefix_sites=import_sites=0
    for node in ast.walk(tree):
        if isinstance(node,ast.ImportFrom)and node.module=='hiercp_v1x.basic_cp_verified_continuation':
            names=[alias for alias in node.names if alias.name!='_completed_prefix'];import_sites+=len(node.names)-len(names);node.names=names
        if isinstance(node,ast.Call)and isinstance(node.func,ast.Name)and node.func.id=='_completed_prefix':
            require(len(node.args)==2 and not node.keywords,'Exact original completed-prefix call required')
            node.args.append(ast.Name(id='_RAM512_completed_logging',ctx=ast.Load()));prefix_sites+=1
    body=tree.body[0].body
    def insert_capture(statements):
        nonlocal saved_sites
        for index,node in enumerate(list(statements)):
            if isinstance(node,ast.Delete)and len(node.targets)==1 and isinstance(node.targets[0],ast.Name)and node.targets[0].id=='saved':
                statements.insert(index,ast.parse("_RAM512_completed_logging=copy.deepcopy(saved['logging'])").body[0]);saved_sites+=1
            for key in('body','orelse','finalbody'):
                value=getattr(node,key,None)
                if isinstance(value,list):insert_capture(value)
            for handler in getattr(node,'handlers',[]):insert_capture(handler.body)
    insert_capture(body)
    require((saved_sites,prefix_sites,import_sites)==(1,1,1),'Exactly one logger capture and one prefix route with its local import binding required')
    def routed(data,epochs,logging):
        prefix,record=derive_completed_prefix(data,epochs,logging)
        record['operational_derivation_adapter']=dict(path=str(adapter_path),sha256=adapter_sha256,
            private_selected_logger_capture_sites=1,private_completed_prefix_route_sites=1,original_builder_function_unchanged=True)
        return prefix,record
    namespace=dict(function.__globals__,_completed_prefix=routed);local={}
    exec(compile(ast.fix_missing_locations(tree),function.__code__.co_filename,'exec'),namespace,local)
    return local[function.__name__]
