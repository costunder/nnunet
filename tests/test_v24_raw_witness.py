"""CPU DEBUG: actual raw-bank witness APIs; small arrays, no native training."""
import ast
import hashlib
import json
import pickle
import random
import tempfile
import time
import types
from collections import OrderedDict
from pathlib import Path
import unittest
from unittest.mock import patch, Mock

import numpy as np
import torch
from custom_trainers import onlinecp_raw_bank as raw
from hiercp_v1x import v24_nnunet_cp as cp


class RawWitnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Compile the exact production classes without importing unavailable
        # native framework setup. Raw storage and bank logic remain real.
        class NativeFrameworkSetup:
            def __init__(self,*args,**kwargs):
                self.indices=['UNIT_A','UNIT_B']
        ns=dict(np=np,torch=torch,hashlib=hashlib,json=json,OrderedDict=OrderedDict,
            Path=Path,os=__import__('os'),random=random,time=time,types=types,
            nnUNetDataLoader=NativeFrameworkSetup,TRAINER_FORMAT='hiercp_online_trainer_v2',
            BANK_FORMAT='hiercp_online_bank_v2',sha=cp.sha,read=cp.read,
            eligible_argmax=cp.eligible_argmax,CP_SELECTION=cp.CP_SELECTION)
        tree=ast.parse((cp.ROOT/'custom_trainers/nnUNetTrainer_OnlinePairedCP.py').read_text(encoding='utf8'))
        names={'OnlineCPError','OnlineCPBank','nnUNetDataLoaderOnlineCP'}
        nodes=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names]
        future=ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[future,*nodes],type_ignores=[])),
                     'actual_parent_bank_DEBUG','exec'),ns)
        tree=ast.parse((cp.ROOT/'custom_trainers/nnUNetTrainer_FrozenV23CP.py').read_text(encoding='utf8'))
        nodes=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name in ('FrozenV23Bank','FrozenV23Loader')]
        exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),
                     'actual_frozen_bank_DEBUG','exec'),ns)
        cls.namespace=ns
        cls.parent_bank_class=ns['OnlineCPBank']
        cls.frozen_bank_class=ns['FrozenV23Bank']

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='v24_witness_DEBUG_',dir=Path.cwd())
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.entries={};self.metadata=dict(format='hiercp_online_bank_v2',
            paste_contract=raw.PASTE_CONTRACT,candidate_count=128,tumor_label=2,liver_label=1,
            cp_probability=.5,intensity_scale_range=[.95,1.05],intensity_shift_range_hu=[-5,5],
            entries_by_case={},entry_sha256={},split=dict(outer_train=['UNIT_A','UNIT_B']),CP_audits={})
        self.score_manifest=dict(score_files={})
        centers=np.stack([np.arange(128)+2,np.full(128,3),np.full(128,4)],axis=1)
        scores=np.arange(128,dtype=np.float32);mask=np.ones(128,np.uint8);mask[-1]=0
        (self.root/'scores').mkdir()
        for case_id in ('UNIT_A','UNIT_B'):
            case=dict(metadata=dict(case_id=case_id),baseline=np.arange(24,dtype=np.float64).reshape(1,2,3,4),
                segmentation=np.ones((1,2,3,4),np.int16))
            case_path=case_id+'.case.json';case_sha=raw.save_case(self.root,case_path,case)
            candidate=dict(case_id=case_id,source_component=2,case_reference_sha256=case_sha,
                raw_target_center=centers[126],source_ct=np.ones((3,3,3),np.float32),
                source_mask=np.ones((3,3,3),bool),output_bbox=np.asarray([[0,2]]*3))
            selected_path=case_id+'.selected.json'
            selected_sha=raw.save_candidate(self.root,selected_path,candidate)
            score_path=self.root/'scores'/(case_id+'.json')
            score_path.write_text(json.dumps(dict(scores=scores.tolist(),centers=centers.tolist())),encoding='utf8')
            score_sha=cp.sha(score_path)
            self.score_manifest['score_files'][case_id]=dict(path='scores/'+case_id+'.json',sha256=score_sha)
            self.metadata['CP_audits'][case_id]=dict(donor_case_id='UNIT_DONOR',donor_component=2,
                scores_file_sha256=score_sha,selected_candidate=126,eligibility=dict(eligible_mask=mask.tolist()))
            entry=dict(paste_contract=np.asarray([raw.PASTE_CONTRACT]),case_id=np.asarray([case_id]),
                donor_case_id=np.asarray(['UNIT_DONOR']),donor_component_id=np.asarray([2]),
                candidate_centers=centers,candidate_raw_centers=centers,scores=scores,
                source_component=np.asarray([2]),source_diameter_mm=np.asarray([10.]),
                selected_candidate=np.asarray([126]),candidate_eligibility=mask,
                selection_policy=np.asarray([cp.CP_SELECTION]),selected_payload=np.asarray([selected_path]),
                selected_payload_sha256=np.asarray([selected_sha]),raw_case_reference=np.asarray([case_path]),
                raw_case_reference_sha256=np.asarray([case_sha]))
            entry_name=case_id+'.entry.npz';np.savez(self.root/entry_name,**entry)
            self.metadata['entries_by_case'][case_id]=[entry_name]
            self.metadata['entry_sha256'][entry_name]=cp.sha(self.root/entry_name)
            self.entries[case_id]=entry
        self.index=self.root/'index.json'
        self.index.write_text(json.dumps(self.metadata),encoding='utf8')

    def bank(self):
        # The small DEBUG fixture omits the scientific105/pin setup tested by
        # NativeContracts. Actual parent and selected-entry validation run.
        bank=self.frozen_bank_class.__new__(self.frozen_bank_class)
        self.parent_bank_class.__init__(bank,self.index)
        bank.v24_score_manifest=self.score_manifest
        return bank

    def test_preflight_hashes_all_references_without_loading_volume_arrays(self):
        bank=self.bank();expected=set()
        for case,entry in self.entries.items():
            for key in ('raw_case_reference','selected_payload'):
                name=str(entry[key][0]);expected.add(name)
                document=cp.read(self.root/name)
                expected.update(spec['path'] for spec in document['arrays'].values() if 'path' in spec)
                if 'archive' in document:expected.add(document['archive']['path'])
        original=np.load
        def small_entries_only(path,*args,**kwargs):
            self.assertTrue(str(path).endswith('.entry.npz'))
            return original(path,*args,**kwargs)
        backend=bank._get_raw_store()._check.__globals__
        hashes=Mock(wraps=backend['_sha'])
        with patch.dict(backend,{'_sha':hashes}),patch.object(np,'load',side_effect=small_entries_only):
            receipt=bank.prepare_raw_verification()
        self.assertEqual({key[0] for key in receipt['witnesses']},expected)
        self.assertEqual(hashes.call_count,len(expected))
        self.assertFalse(bank._raw_store._cases);self.assertFalse(bank._raw_store._sources)

    def test_actual_runtime_and_spawn_state_reuse_sha_but_recheck_stats(self):
        receipt=self.bank().prepare_raw_verification();bank=self.bank()
        bank.adopt_raw_verification(receipt)
        backend=bank._get_raw_store()._check.__globals__
        with patch.dict(backend,{'_sha':Mock(side_effect=AssertionError('Repeated volume SHA read'))}):
            case,candidate=bank.load_raw_candidate(self.entries['UNIT_A'],126)
            np.testing.assert_array_equal(case['baseline'],np.arange(24).reshape(1,2,3,4))
            self.assertEqual(candidate['case_id'],'UNIT_A')
            restored=pickle.loads(pickle.dumps(bank._raw_store))
            self.assertFalse(restored._cases);self.assertFalse(restored._sources)
            self.assertEqual(restored._witnesses,receipt['witnesses'])
            restored.load_case('UNIT_B.case.json',str(self.entries['UNIT_B']['raw_case_reference_sha256'][0]))
            restored.close()
        bank._raw_store.close()
        bank._raw_store._witnesses.clear()
        self.assertTrue(receipt['witnesses'])

    def test_atomic_array_replacement_is_rejected_by_inherited_witness(self):
        receipt=self.bank().prepare_raw_verification();bank=self.bank();bank.adopt_raw_verification(receipt)
        manifest=cp.read(self.root/'UNIT_A.case.json')
        relative=next(iter(manifest['arrays'].values()))['path']
        path=self.root/relative;temporary=self.root/'replacement.npy'
        temporary.write_bytes(path.read_bytes());temporary.replace(path)
        with self.assertRaisesRegex(ValueError,'changed after verification'):
            bank.load_raw_candidate(self.entries['UNIT_A'],126)
        bank._raw_store.close()

    def test_wrong_hash_or_missing_payload_is_not_admitted(self):
        manifest=cp.read(self.root/'UNIT_A.case.json');path=self.root/next(iter(manifest['arrays'].values()))['path']
        path.write_bytes(path.read_bytes()+b'changed')
        with self.assertRaisesRegex(ValueError,'hash mismatch'):self.bank().prepare_raw_verification()
        path.unlink()
        with self.assertRaisesRegex(ValueError,'Missing raw-bank artifact'):self.bank().prepare_raw_verification()

    def test_index_or_root_change_rejects_real_adoption(self):
        receipt=self.bank().prepare_raw_verification()
        with self.assertRaisesRegex(self.namespace['OnlineCPError'],'exact bank root/index'):
            self.bank().adopt_raw_verification(dict(receipt,root=str(self.root/'other')))
        self.index.write_text(self.index.read_text(encoding='utf8')+'\n',encoding='utf8')
        with self.assertRaisesRegex(self.namespace['OnlineCPError'],'exact bank root/index'):self.bank().adopt_raw_verification(receipt)

    def test_original_factory_adopts_before_workers_and_prepares_only_once(self):
        ns=dict(self.namespace)
        exec('def original_factory(self):\n'
             '    self._online_train_loader=nnUNetDataLoaderOnlineCP(bank_path=self.online_bank_path,policy="hier_argmax",online_seed=42)\n'
             '    return self._make_train_augmenter(0), self.validation_sentinel\n',ns)
        class OriginalFrameworkSetup:
            get_dataloaders=ns['original_factory']
        ns['_nnUNetTrainer_250epochs_OnlineCP']=OriginalFrameworkSetup
        tree=ast.parse((cp.ROOT/'custom_trainers/nnUNetTrainer_FrozenV23CP.py').read_text(encoding='utf8'))
        node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='nnUNetTrainer_250epochs_FrozenV23CP')
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),
                     'actual_frozen_trainer_DEBUG','exec'),ns)
        trainer=ns[node.name].__new__(ns[node.name]);trainer.online_bank_path=str(self.index)
        trainer.print_to_log_file=lambda *args,**kwargs:None;trainer.validation_sentinel=object()
        def before_workers(epoch):
            self.assertEqual(epoch,0)
            self.assertTrue(trainer._online_train_loader.online_bank._raw_store._witnesses)
            self.assertFalse(trainer._online_train_loader.online_bank._raw_store._cases)
            return 'worker_start_DEBUG'
        trainer._make_train_augmenter=before_workers
        # Keep the actual Frozen loader init and actual adopt method. Only the
        # fixture bank constructor and unavailable framework setup are supplied.
        with patch.dict(self.namespace,OnlineCPBank=lambda *a,**kw:self.bank(),FrozenV23Bank=lambda *a,**kw:self.bank()):
            with patch.dict(ns,FrozenV23Bank=lambda *a,**kw:self.bank()):
                before=np.random.get_state();torch_before=torch.get_rng_state().clone();python_before=random.getstate()
                with patch.object(raw,'_sha',wraps=raw._sha) as hashes:
                    first=trainer.get_dataloaders();calls=hashes.call_count
                    second=trainer.get_dataloaders()
                self.assertEqual(hashes.call_count,calls)
                self.assertEqual(first[0],'worker_start_DEBUG');self.assertIs(first[1],trainer.validation_sentinel)
                self.assertIs(second[1],trainer.validation_sentinel)
                after=np.random.get_state();self.assertEqual(before[0],after[0]);np.testing.assert_array_equal(before[1],after[1])
                self.assertEqual(before[2:],after[2:]);self.assertTrue(torch.equal(torch_before,torch.get_rng_state()))
                self.assertEqual(python_before,random.getstate())


if __name__=='__main__':unittest.main()
