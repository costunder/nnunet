"""CPU DEBUG tests execute the actual prepared cache operator's private steps.

No main(), remote process, Torch model, CUDA context, or clinical result runs.
Small real files test publication/cancellation boundaries, not research scale.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest import mock
import psutil


FILE=Path(__file__).resolve().parents[1]/'tools/rebind_basic_cp_RAM512_cache.py'
TREE=ast.parse(FILE.read_text(encoding='utf8'))
MAIN=next(node for node in TREE.body if isinstance(node,ast.FunctionDef)and node.name=='main')
BODY=next(node for node in MAIN.body if isinstance(node,ast.Try)).body
FUNCTIONS={'require','read','exact','inode','digest','pin','own_cache','sync','copy_new',
           'publish','released','old_helper_command','capacity'}


def execute(scope, nodes):
    module=ast.Module(body=nodes,type_ignores=[])
    exec(compile(ast.fix_missing_locations(module),str(FILE),'exec'),scope)


class CacheAdoptionDEBUGTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory(prefix='DEBUG_RAM512_CACHE_ADOPTION_',dir=Path(__file__).parent)
        self.root=Path(self.temporary.name).resolve();self.cache=self.root/'cache';self.cache.mkdir()
        self.scope=dict(P=Path,os=os,stat=stat,hashlib=hashlib,json=json,shutil=shutil,
            time=time,argparse=argparse,psutil=psutil,CACHE=self.cache,OLD=self.root/'old',
            UID=41833 if os.name=='nt'else os.getuid(),GIB=2**30,
            HELPER='tools/persist_basic_cp_seven.py',torch=SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda:False)))
        execute(self.scope,[node for node in TREE.body if isinstance(node,ast.FunctionDef)and node.name in FUNCTIONS])
        actual_stat=Path.stat
        def own_stat(path,*args,**kwargs):
            result=actual_stat(path,*args,**kwargs)
            if os.name!='nt':return result
            values=list(result);values[4]=41833;return os.stat_result(values)
        self.stat_patch=mock.patch.object(Path,'stat',new=own_stat);self.stat_patch.start()
        # Windows DEBUG lacks directory fsync; every file flush/fsync, SHA,
        # exact path, ownership and exclusive-copy check remains exercised.
        if os.name=='nt':self.scope['sync']=lambda path:None

    def tearDown(self):
        self.stat_patch.stop();self.temporary.cleanup()

    def populate(self):
        requests={}
        for gpu in range(7):
            name='GPU'+str(gpu)+'_DEBUG';folder=self.cache/name;folder.mkdir()
            requests[str(gpu)]={'root':str(self.root/'old'/name)}
            (folder/'request.json').write_text(json.dumps({'DEBUG_GPU':gpu}),encoding='utf8')
            artifacts=folder/'artifacts';artifacts.mkdir();(artifacts/'status.json').write_text('{"DEBUG":true}',encoding='utf8')
            (artifacts/'train.log').write_bytes(b'DEBUG preserved prior log\n')
            if gpu!=5:
                (folder/'CURRENT.pth').write_bytes(b'DEBUG_WHOLE_BYTES_'+str(gpu).encode())
                for name in('CURRENT.json','PENDING.json'):(folder/name).write_text('{"DEBUG_CURRENT":true}',encoding='utf8')
                runtime=folder/'runtime';runtime.mkdir();(runtime/'scientific.py').write_bytes(b'# DEBUG exact runtime\n')
        banks=self.cache/'banks/512';(banks/'entries').mkdir(parents=True);(banks/'entry_receipts').mkdir()
        (banks/'entries/DEBUG_source.npz').write_bytes(b'DEBUG_CANDIDATE_ARRAY_BYTES')
        (banks/'entry_receipts/DEBUG_source.json').write_text('{"DEBUG_PROVENANCE":true}',encoding='utf8')
        (banks/'index.json').write_text('{"DEBUG_COMPLETE":false}',encoding='utf8')
        (self.cache/'identity.json').write_text('{"DEBUG_OLD_HELPER":true}',encoding='utf8')
        (self.cache/'persistence_helper.py').write_bytes(b'# DEBUG old admitted helper\n')
        code=self.cache/'code/DEBUG_old_commit';code.mkdir(parents=True);(code/'frozen.json').write_bytes(b'{"DEBUG_CODE":true}')
        self.scope['old_requests']=requests
        return requests

    def select_metadata(self):
        start=next(i for i,node in enumerate(BODY)if isinstance(node,ast.Assign)
            and any(isinstance(target,ast.Name)and target.id=='sources'for target in node.targets))
        end=next(i for i,node in enumerate(BODY[start:],start)if isinstance(node,ast.Assign)
            and any(isinstance(target,ast.Name)and target.id=='incoming'for target in node.targets))
        execute(self.scope,BODY[start:end])
        return self.scope['sources']

    def archive_and_reset(self):
        start=next(i for i,node in enumerate(BODY)if isinstance(node,ast.For)
            and isinstance(node.target,ast.Tuple) and [v.id for v in node.target.elts]==['relative','row']
            and any(isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id=='copy_new'for n in ast.walk(node)))
        self.history=self.cache/'history/DEBUG_RAM512';self.history.mkdir(parents=True)
        self.scope['history']=self.history
        execute(self.scope,BODY[start:start+3])

    def test_all7_metadata_includes_pre_native5_and_bank_provenance_only(self):
        requests=self.populate();sources=self.select_metadata()
        sources={Path(key).as_posix():value for key,value in sources.items()}
        for gpu,request in requests.items():
            name=Path(request['root']).name
            self.assertIn(name+'/request.json',sources)
            self.assertIn(name+'/artifacts/status.json',sources)
            self.assertIn(name+'/artifacts/train.log',sources)
        self.assertIn('banks/512/index.json',sources)
        self.assertIn('banks/512/entry_receipts/DEBUG_source.json',sources)
        self.assertIn('persistence_helper.py',sources)
        self.assertNotIn('banks/512/entries/DEBUG_source.npz',sources)
        self.assertNotIn('GPU6_DEBUG/CURRENT.pth',sources)
        self.assertNotIn('GPU6_DEBUG/runtime/scientific.py',sources)
        self.assertNotIn('code/DEBUG_old_commit/frozen.json',sources)
        self.assertTrue(sources['GPU6_DEBUG/CURRENT.json']['keep'])
        self.assertTrue(sources['GPU6_DEBUG/PENDING.json']['keep'])

    def test_all_original_metadata_archived_before_reset_protected_bytes_unchanged(self):
        self.populate();sources=self.select_metadata()
        protected={p:p.read_bytes()for p in self.cache.rglob('*')if p.is_file()
            and(p.suffix in('.pth','.npz')or'runtime'in p.parts)}
        original={relative:row['path'].read_bytes()for relative,row in sources.items()}
        self.archive_and_reset()
        for relative,row in sources.items():
            self.assertEqual((self.history/'cache_metadata'/relative).read_bytes(),original[relative])
            self.assertEqual(row['path'].exists(),row['keep'])
            if row['keep']:self.assertEqual(row['path'].read_bytes(),original[relative])
        self.assertEqual({p:p.read_bytes()for p in protected},protected)

    def test_failed_history_copy_never_removes_any_old_metadata_or_models(self):
        self.populate();sources=self.select_metadata();before={p:p.read_bytes()for p in self.cache.rglob('*')if p.is_file()}
        actual=self.scope['copy_new'];count=[0]
        def fault(source,target):
            count[0]+=1
            if count[0]==3:raise OSError('DEBUG independent archive failure')
            actual(source,target)
        self.scope['copy_new']=fault
        with self.assertRaisesRegex(OSError,'DEBUG independent archive failure'):self.archive_and_reset()
        self.assertEqual({p:p.read_bytes()for p in before},before)
        self.assertTrue(all(row['path'].exists()for row in sources.values()))

    def test_old_helper_command_and_actual_cache_identity_mismatch_rejected(self):
        old_code=self.root/'old/code';command=['/home/aicompetition06/.conda/envs/nnunet/bin/python3.10',
            '-B','-u',str(old_code/'tools/persist_basic_cp_seven.py'),'--manifest',str(self.scope['OLD']/'launch.json'),
            '--output',str(self.cache),'--unfinished-checkpoint','best','--resume-cache']
        self.scope['old_helper_command'](command,old_code)
        changed=list(command);changed[changed.index('--output')+1]=str(self.root/'different_cache')
        with self.assertRaisesRegex(ValueError,'previous writer contract'):self.scope['old_helper_command'](changed,old_code)
        proof_check=next(node for node in BODY if isinstance(node,ast.Expr)and isinstance(node.value,ast.Call)
            and any(isinstance(n,ast.Constant)and n.value=='Old cache writer/actor relationship differs'for n in ast.walk(node)))
        self.scope.update(actor={'pid':4167493,'birth':1234.5,'command':command},
            identity={'pid':4149407,'birth':1234.5,'command':command})
        with self.assertRaisesRegex(ValueError,'writer/actor relationship'):execute(self.scope,[proof_check])

    def test_reused_actor_birth_releases_only_original_and_never_admits_replacement(self):
        actor={'pid':4167493,'birth':1234.5,'UID':self.scope['UID'],'command':['DEBUG_old_helper']}
        process=mock.Mock();process.create_time.return_value=2345.6
        with mock.patch.object(psutil,'Process',return_value=process):result=self.scope['released'](actor)
        self.assertEqual(result['state'],'gone');self.assertTrue(result['replacement_not_admitted'])
        process.uids.assert_not_called();process.cmdline.assert_not_called()
        process.create_time.return_value=1234.5;process.uids.return_value=SimpleNamespace(real=self.scope['UID'])
        process.status.return_value=psutil.STATUS_RUNNING;process.cmdline.return_value=actor['command']
        with mock.patch.object(psutil,'Process',return_value=process),self.assertRaisesRegex(RuntimeError,'still live'):
            self.scope['released'](actor)

    def test_cache_budget_preserves8GiB_capacity_and1GiB_floor_before_archive(self):
        (self.cache/'DEBUG_payload').write_bytes(b'x'*900);self.scope['GIB']=128
        process=SimpleNamespace(memory_info=lambda:SimpleNamespace(rss=1))
        with mock.patch.object(psutil,'Process',return_value=process), \
                mock.patch.object(shutil,'disk_usage',return_value=SimpleNamespace(free=10000)):
            self.assertEqual(self.scope['capacity'](124),900)
            with self.assertRaisesRegex(ValueError,'retain8GiB'):self.scope['capacity'](125)
        with mock.patch.object(psutil,'Process',return_value=process), \
                mock.patch.object(shutil,'disk_usage',return_value=SimpleNamespace(free=127)):
            with self.assertRaisesRegex(ValueError,'1GiB'):self.scope['capacity']()
        self.assertEqual((self.cache/'DEBUG_payload').stat().st_size,900)

    def test_exclusive_metadata_copy_cannot_overwrite_existing_history(self):
        source=self.cache/'DEBUG_source.json';source.write_bytes(b'DEBUG_OWN_ORIGINAL')
        target=self.cache/'history/DEBUG_existing.json';target.parent.mkdir();target.write_bytes(b'DEBUG_OLD_HISTORY')
        with self.assertRaises(FileExistsError):self.scope['copy_new'](source,target)
        self.assertEqual(target.read_bytes(),b'DEBUG_OLD_HISTORY')
        with self.assertRaisesRegex(ValueError,'escapes own fixed cache'):
            self.scope['copy_new'](source,self.root/'outside.json')


if __name__=='__main__':unittest.main()
