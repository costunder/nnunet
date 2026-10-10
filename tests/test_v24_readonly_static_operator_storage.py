"""CPU DEBUG unit fixtures for full105 exact static operator sharing.

Tiny fixture tensors exercise source/publication/reader contracts only. They
are never a final training dataset, a shortened experiment, or a model result.
"""
import copy
import importlib.util
import json
from pathlib import Path
import os
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from hiercp_v1x import v24_readonly_native_storage as core
from hiercp_v1x import v24_readonly_static_operator_storage as extension
from hiercp_v1x import v24_lossless_raw_storage as raw

_fixture_spec=importlib.util.spec_from_file_location('readonly_static_CPU_fixture',Path(__file__).with_name('test_v24_readonly_native_storage.py'))
fixtures=importlib.util.module_from_spec(_fixture_spec);sys.modules[_fixture_spec.name]=fixtures;_fixture_spec.loader.exec_module(fixtures)

CHECKPOINT_SOURCE='''class nnUNetTrainer:
    def save_checkpoint(self, filename):
        checkpoint = {'UNIT': True}
        torch.save(checkpoint, filename)
    def on_train_end(self):
        self.save_checkpoint(join(self.output_folder, "checkpoint_final.pth"))
        if isfile(join(self.output_folder, "checkpoint_latest.pth")):
            os.remove(join(self.output_folder, "checkpoint_latest.pth"))
    def on_epoch_end(self):
        self.save_checkpoint(join(self.output_folder, "checkpoint_latest.pth"))
        self.save_checkpoint(join(self.output_folder, "checkpoint_best.pth"))
'''


def original_prepare(bank_root,relative_manifest,case):
    return save_case(bank_root,relative_manifest,case)


def original_outer(bank_root,relative_manifest,case):
    return _prepare_raw_case(bank_root,relative_manifest,case)


class StaticOperatorStorageCPUUnit(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.ReadonlyNativeStorageUnit(methodName='test_complete_inventories_and_measured_budget')
        self.addCleanup(self.fixture.doCleanups)
        original_make=self.fixture.make_raw
        def make(case_id):
            original_make(case_id)
            manifest_path=self.fixture.bank/'raw_cases'/(case_id+'.json')
            manifest=core.read(manifest_path);values=self.fixture.arrays[case_id]
            values.pop('operator')
            matrix=np.array([[1.,np.nextafter(0.,1.)],[1e-230,-0.]],dtype=np.float64)
            values.update(data_operators=[np.array(matrix,order='C'),np.array(matrix,order='F'),matrix.copy()],
                          seg_operators=[np.eye(2,dtype=np.float64) for _ in range(3)],
                          clip_min=np.array([-1.],dtype=np.float64),clip_max=np.array([1.],dtype=np.float64))
            arrays={};tree=raw.original._encode_tree(values,arrays);specs={}
            old_volume_specs={role:manifest['arrays'][manifest['tree'][role]['$array']] for role in core.ROLES}
            for key,array in arrays.items():
                role=next((role for role in core.ROLES if tree[role]['$array']==key),None)
                name='raw_cases/'+case_id+'.'+key+('.b2nd' if role else '.npy')
                path=self.fixture.bank/name;fixtures.save_array(path,array)
                if role:spec=dict(old_volume_specs[role],path=name,sha256=core.sha(path),stored_bytes=path.stat().st_size)
                else:spec=dict(shape=list(array.shape),dtype=array.dtype.str,path=name,sha256=core.sha(path))
                specs[key]=spec
            lossless=copy.deepcopy(manifest['lossless_storage']);lossless['original_logical_array_bytes']=sum(v.nbytes for v in arrays.values())
            fixtures.write(manifest_path,dict(manifest,tree=tree,arrays=specs,lossless_storage=lossless))
        self.fixture.make_raw=make;self.fixture.setUp()
        self.root=self.fixture.root;self.checkpoint_source=self.fixture.package/extension.CHECKPOINT_SOURCE
        self.checkpoint_source.parent.mkdir(parents=True,exist_ok=True);self.checkpoint_source.write_text(CHECKPOINT_SOURCE,encoding='utf8')
        self.samples=[]
        for name in ('checkpoint_best.pth','checkpoint_final.pth'):
            path=self.root/'UNIT_complete_native'/name;path.parent.mkdir(exist_ok=True);path.write_bytes(b'UNIT_checkpoint_size_only')
            self.samples.append(path)
        self.base=self.fixture.admission();self.base_path=self.root/'UNIT_base_admission.json';core.publish_json(self.base_path,self.base)
        self.arms={arm:str(self.root/'UNIT_group'/arm) for arm in ('gpu4','gpu5','gpu6')}
        self.original_check=raw.LosslessRawBankStore._check
        self.original_codes=extension._reader_codes(raw)
        self.addCleanup(self.restore)

    def restore(self):
        raw.LosslessRawBankStore._check=self.original_check
        extension._ORIGINAL_CHECK=None;extension._READER_ORIGINALS=None;extension._BINDINGS.clear()

    def admission(self):
        with patch.object(extension.shutil,'disk_usage',return_value=SimpleNamespace(free=64*2**30)):
            return extension.build_admission(self.base_path,arm_roots=self.arms,checkpoint_paths=self.samples,
                                             checkpoint_max_bytes=819243768)

    def preparer(self,document):
        inner=core._clone(original_prepare,{})
        outer=core._clone(original_outer,dict(_prepare_raw_case=inner))
        return extension.make_case_preparer(SimpleNamespace(_root_raw_preparation=lambda:SimpleNamespace(
            _prepare_raw_case=inner,prepare_raw_case=outer)),document)

    def publish_bank(self,document,changed_case=None):
        bank=self.root/'UNIT_new_bank';bank.mkdir();preparer=self.preparer(document)
        with patch.object(raw,'_verify_all',side_effect=lambda array,path,spec:spec['roundtrip']):
            for case_id in self.fixture.train:
                case=copy.deepcopy(self.fixture.arrays[case_id])
                if changed_case is not None and case_id==self.fixture.train[0]:changed_case(case)
                preparer(bank,'raw_cases/'+case_id+'.json',case)
        return bank

    def test_all105_eight_roles_exact_dtype_order_budget_and_base_unchanged(self):
        before=core.sha(self.base_path);document=self.admission()
        self.assertEqual(core.sha(self.base_path),before)
        self.assertEqual(document['profile'],core.PROFILE);self.assertEqual(document['storage_extension'],extension.EXTENSION)
        self.assertEqual(sum(len(row['static_operator_arrays']) for row in document['raw_cases'].values()),840)
        self.assertEqual(document['checkpoint_publication_proof']['peak_checkpoint_slots'],3)
        self.assertEqual(document['budget']['native_checkpoint_slots'],3)
        self.assertEqual(document['aggregate_budget']['required_free_bytes'],3*document['budget']['new_writable_bytes_estimate']+10*2**30+512*2**20)
        rows=document['raw_cases'][self.fixture.train[0]]['static_operator_arrays']
        self.assertTrue(any(row['npy_header']['fortran_order'] for row in rows.values()))
        self.assertTrue(all(row['spec']['dtype']==np.dtype('float64').str for row in rows.values()))
        extension.verify_admission(document,full_hash=True)

    def test_missing_operator_role_and_forged_donor_reference_rejected(self):
        document=self.admission();case=document['raw_cases'][self.fixture.train[0]]
        case['static_operator_arrays'].pop(next(iter(case['static_operator_arrays'])))
        with self.assertRaisesRegex(ValueError,'all eight'):extension.verify_admission(document)
        document=self.admission();row=next(iter(document['raw_cases'][self.fixture.train[0]]['static_operator_arrays'].values()))
        row['role']='learned_upper'
        with self.assertRaisesRegex(ValueError,'role/content'):extension.verify_admission(document)

    def test_original_base_content_tamper_and_static_source_mutation_rejected(self):
        document=self.admission();self.base_path.write_text('{}',encoding='utf8')
        with self.assertRaisesRegex(ValueError,'source changed'):extension.verify_admission(document)

    def test_static_file_stat_change_rejected(self):
        document=self.admission();row=next(iter(document['raw_cases'][self.fixture.train[0]]['static_operator_arrays'].values()))
        Path(row['file']['path']).write_bytes(b'UNIT_corruption')
        with self.assertRaisesRegex(ValueError,'source changed'):extension.verify_admission(document)

    def test_fresh_all_bytes_tiny_nonzero_and_signed_zero_must_match(self):
        document=self.admission()
        def alter(case):case['data_operators'][0][0,1]=0.
        with self.assertRaisesRegex(ValueError,'complete original static operator'):self.publish_bank(document,alter)

    def test_fresh_memory_order_mismatch_rejected(self):
        document=self.admission()
        def alter(case):case['data_operators'][1]=np.array(case['data_operators'][1],order='C')
        with self.assertRaisesRegex(ValueError,'complete original static operator'):self.publish_bank(document,alter)

    def test_all105_shared_reader_original_code_and_readonly_mmap(self):
        document=self.admission()
        def reorder(case):
            items=list(case.items());case.clear();case.update(reversed(items))
        bank=self.publish_bank(document,reorder)
        first=core.read(bank/'raw_cases'/(self.fixture.train[0]+'.json'))
        old=core.read(document['raw_cases'][self.fixture.train[0]]['manifest']['path'])
        self.assertNotEqual(first['tree']['clip_min']['$array'],old['tree']['clip_min']['$array'])
        before={row['file']['path']:core.stat(row['file']['path']) for case in document['raw_cases'].values()
                for row in case['static_operator_arrays'].values()}
        proof=extension.install_runtime_adapters(document,bank)
        self.assertEqual(proof['readonly_role_references'],210);self.assertEqual(proof['readonly_static_operator_references'],840)
        self.assertEqual(proof['read_mode'],'r');self.assertEqual(extension._reader_codes(raw),self.original_codes)
        store=raw.LosslessRawBankStore(bank)
        try:
            for case_id in self.fixture.train:
                path=bank/'raw_cases'/(case_id+'.json');case=store.load_case('raw_cases/'+case_id+'.json',core.sha(path))
                for name in ('data_operators','seg_operators'):
                    for axis,array in enumerate(case[name]):
                        expected=self.fixture.arrays[case_id][name][axis]
                        self.assertIsInstance(array,np.memmap);self.assertFalse(array.flags.writeable)
                        self.assertEqual(array.tobytes(),expected.tobytes());self.assertEqual(array.strides,expected.strides)
                for name in ('clip_min','clip_max'):
                    self.assertFalse(case[name].flags.writeable);self.assertEqual(case[name].tobytes(),self.fixture.arrays[case_id][name].tobytes())
            original_code=np.load.__code__
            try:
                np.load.__code__=original_code.replace()
                path=bank/'raw_cases'/(self.fixture.train[0]+'.json')
                with self.assertRaisesRegex(ValueError,'reader code changed'):
                    store.load_case('raw_cases/'+self.fixture.train[0]+'.json',core.sha(path))
            finally:np.load.__code__=original_code
        finally:store.close()
        self.assertEqual(before,{path:core.stat(path) for path in before})
        self.assertEqual(len(list((bank/'raw_cases').glob('*.npy'))),0)

    def test_aggregate_credits_only_new_bytes_and_preserves_growth_floor(self):
        initial=Path(self.arms['gpu4']);initial.mkdir(parents=True);(initial/'UNIT_existing').write_bytes(b'x'*4096)
        document=self.admission();aggregate=document['aggregate_budget'];expected=aggregate['required_free_bytes']
        with patch.object(extension.shutil,'disk_usage',return_value=SimpleNamespace(free=expected)):
            first=extension.check_aggregate_disk(document,initial)
        self.assertEqual(first['required_free_bytes'],expected)
        (initial/'UNIT_fresh').write_bytes(b'y'*8192);new=extension._allocated_bytes(initial)-aggregate['initial_allocated_bytes']['gpu4']
        with patch.object(extension.shutil,'disk_usage',return_value=SimpleNamespace(free=expected-new)):
            after=extension.check_aggregate_disk(document,initial)
        self.assertEqual(after['required_free_bytes'],expected-new)
        with patch.object(extension.shutil,'disk_usage',return_value=SimpleNamespace(free=10*2**30+512*2**20-1)):
            with self.assertRaisesRegex(OSError,'do not fit'):extension.check_aggregate_disk(document,initial,preparation=False)
        with self.assertRaisesRegex(ValueError,'Exact own arm'):extension.check_aggregate_disk(document,self.root/'UNIT_wrong_arm')

    def test_aggregate_budget_or_growth_reduction_rejected(self):
        document=self.admission();document['aggregate_budget']['growth_margin_bytes']=0
        with self.assertRaisesRegex(ValueError,'budget/reserve/growth'):extension.verify_admission(document)

    def test_checkpoint_temporary_filename_rejected(self):
        bad=CHECKPOINT_SOURCE.replace('torch.save(checkpoint, filename)','torch.save(checkpoint, "temporary.pth")')
        self.checkpoint_source.write_text(bad,encoding='utf8');self.base['runtime_files'][extension.CHECKPOINT_SOURCE]=core.proof(self.checkpoint_source)
        with self.assertRaisesRegex(ValueError,'direct torch.save'):
            extension._checkpoint_source_proof(self.base,self.samples,819243768)

    def test_checkpoint_temporary_replacement_operation_rejected(self):
        bad=CHECKPOINT_SOURCE.replace('torch.save(checkpoint, filename)',
                                      'torch.save(checkpoint, filename)\n        os.replace("UNIT_tmp", filename)')
        self.checkpoint_source.write_text(bad,encoding='utf8');self.base['runtime_files'][extension.CHECKPOINT_SOURCE]=core.proof(self.checkpoint_source)
        with self.assertRaisesRegex(ValueError,'temporary/replacement'):
            extension._checkpoint_source_proof(self.base,self.samples,819243768)

    def test_checkpoint_final_before_latest_removal_and_inheritance_required(self):
        bad=CHECKPOINT_SOURCE.replace('        self.save_checkpoint(join(self.output_folder, "checkpoint_final.pth"))\n','')
        bad=bad.replace('    def on_epoch_end(self):','        self.save_checkpoint(join(self.output_folder, "checkpoint_final.pth"))\n    def on_epoch_end(self):')
        self.checkpoint_source.write_text(bad,encoding='utf8');self.base['runtime_files'][extension.CHECKPOINT_SOURCE]=core.proof(self.checkpoint_source)
        with self.assertRaisesRegex(ValueError,'publication ordering'):
            extension._checkpoint_source_proof(self.base,self.samples,819243768)
        self.checkpoint_source.write_text(CHECKPOINT_SOURCE,encoding='utf8');self.base['runtime_files'][extension.CHECKPOINT_SOURCE]=core.proof(self.checkpoint_source)
        custom=self.root/'UNIT_custom_override.py';custom.write_text('class UNITTrainer:\n    def save_checkpoint(self,filename):\n        return None\n',encoding='utf8')
        self.base['custom_trainer_files']=dict(self.base['custom_trainer_files'])
        self.base['custom_trainer_files'][next(iter(self.base['custom_trainer_files']))]=core.proof(custom)
        with self.assertRaisesRegex(ValueError,'inherit'):
            extension._checkpoint_source_proof(self.base,self.samples,819243768)

    def test_actual_private_inherited_checkpoint_compiled_code_verified(self):
        document=self.admission();private=self.root/'UNIT_private_runtime';path=private/'nnunetv2'/extension.CHECKPOINT_SOURCE
        path.parent.mkdir(parents=True);path.write_text(CHECKPOINT_SOURCE,encoding='utf8')
        spec=importlib.util.spec_from_file_location('UNIT_private_checkpoint_class',path)
        module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
        proof=extension.verify_private_checkpoint_publication(document,private,module.nnUNetTrainer)
        self.assertEqual(proof['peak_checkpoint_slots'],3)
        original=module.nnUNetTrainer.save_checkpoint
        module.nnUNetTrainer.save_checkpoint=lambda self,path:None
        with self.assertRaisesRegex(ValueError,'private source'):extension.verify_private_checkpoint_publication(document,private,module.nnUNetTrainer)
        module.nnUNetTrainer.save_checkpoint=original


class AggregateAllocationCPUUnit(unittest.TestCase):
    """Small filesystem-only regressions for concurrent publication accounting."""
    def setUp(self):
        temporary=tempfile.TemporaryDirectory(prefix='static_allocation_CPU_UNIT_',dir=core.ROOT/'outputs')
        self.addCleanup(temporary.cleanup);self.root=Path(temporary.name)

    def test_two_owned_publication_names_count_one_inode(self):
        root=self.root/'UNIT_output';root.mkdir();first=root/'UNIT_pending';first.write_bytes(b'x'*8192)
        os.link(first,root/'UNIT_published');value=first.stat()
        expected=value.st_blocks*512 if hasattr(value,'st_blocks') else value.st_size
        self.assertEqual(extension._allocated_bytes(root),expected)

    def test_external_hardlink_cannot_be_credited(self):
        source=self.root/'UNIT_external';source.write_bytes(b'x'*8192)
        root=self.root/'UNIT_output';root.mkdir();os.link(source,root/'UNIT_incorrect_ref')
        with self.assertRaisesRegex(ValueError,'External hardlinks'):extension._allocated_bytes(root)

    def test_vanished_owned_temp_triggers_bounded_rescan(self):
        root=self.root/'UNIT_output';root.mkdir();real=root/'UNIT_final';real.write_bytes(b'x'*4096)
        missing=root/'UNIT_already_published_temp';value=real.stat()
        expected=value.st_blocks*512 if hasattr(value,'st_blocks') else value.st_size
        with patch.object(Path,'rglob',side_effect=[iter([missing]),iter([real])]) as scan:
            self.assertEqual(extension._allocated_bytes(root),expected)
        self.assertEqual(scan.call_count,2)

    def test_actual_free_space_sampled_after_all_owned_allocations(self):
        arms={arm:self.root/arm for arm in ('gpu4','gpu5','gpu6')}
        for path in arms.values():path.mkdir()
        budget=dict(arms={arm:dict(root=str(path),filesystem_device=path.stat().st_dev) for arm,path in arms.items()},
                    initial_allocated_bytes={arm:0 for arm in arms},per_arm_new_writable_bytes={arm:1024 for arm in arms},
                    minimum_runtime_free_bytes=10*2**30+512*2**20)
        document=dict(aggregate_budget=budget);calls=[]
        def allocation(path):calls.append(str(path));return 0
        def usage(path):
            self.assertEqual(len(calls),3)
            return SimpleNamespace(free=budget['minimum_runtime_free_bytes']+3*1024)
        with patch.object(extension,'verify_admission',return_value=document),patch.object(extension,'_allocated_bytes',side_effect=allocation),patch.object(extension.shutil,'disk_usage',side_effect=usage):
            result=extension.check_aggregate_disk(document,arms['gpu4'])
        self.assertEqual(result['required_free_bytes'],budget['minimum_runtime_free_bytes']+3*1024)


if __name__=='__main__':unittest.main()
