"""CPU UNIT storage fixtures; tiny arrays never stand in for production training.

The complete105/131 identifier inventories exercise admission and read-only
guards. Blosc2 I/O is represented by exact NPY-backed test readers because the
local CPU environment has no Blosc2; actual server decoder admission is separate.
"""
import copy
import inspect
import json
import marshal
from pathlib import Path
import tempfile
import textwrap
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from hiercp_v1x import v24_readonly_native_storage as ro
from hiercp_v1x import v24_lossless_raw_storage as raw
from hiercp_v1x import v24_nnunet_cp as cp


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value),encoding='utf8')


def save_array(path,array):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('wb') as stream:np.save(stream,array,allow_pickle=False)


class UnitBlosc:
    """Exact test-only array decoder; records mode and never writes."""
    calls=[]
    @classmethod
    def open(cls,urlpath,mode,dparams,**kwargs):
        cls.calls.append((str(urlpath),mode))
        if mode!='r':raise AssertionError('UNIT read-only decoder called for write')
        array=np.load(urlpath,allow_pickle=False);array.flags.writeable=False
        return array


blosc2=UnitBlosc


class UnitDatasetBlosc2:
    def __init__(self,folder,identifiers=None,folder_with_segs_from_previous_stage=None):
        self.source_folder=folder;self.identifiers=self.get_identifiers(folder) if identifiers is None else identifiers
        self.folder_with_segs_from_previous_stage=folder_with_segs_from_previous_stage;self.mmap_kwargs={}
    @staticmethod
    def get_identifiers(folder):
        return [path.name[:-5] for path in Path(folder).glob('*.b2nd') if not path.name.endswith('_seg.b2nd')]
    def load_case(self,identifier):
        data=blosc2.open(urlpath=str(Path(self.source_folder)/(identifier+'.b2nd')),mode='r',dparams={'nthreads':1},**self.mmap_kwargs)
        seg=blosc2.open(urlpath=str(Path(self.source_folder)/(identifier+'_seg.b2nd')),mode='r',dparams={'nthreads':1},**self.mmap_kwargs)
        if self.folder_with_segs_from_previous_stage is not None:
            prev=blosc2.open(urlpath=str(Path(self.folder_with_segs_from_previous_stage)/(identifier+'.b2nd')),mode='r',dparams={'nthreads':1},**self.mmap_kwargs)
        else:prev=None
        properties=json.loads((Path(self.source_folder)/(identifier+'.pkl')).read_text())
        return data,seg,prev,properties
    @staticmethod
    def unpack_dataset(folder,overwrite_existing=False,num_processes=1,verify=True):
        pass


class CanonicalCodeFingerprintUnit(unittest.TestCase):
    """Pure CPU regressions for immutable code identity and serialization."""
    def test_reference_topology_and_extra_references_keep_same_fingerprint(self):
        code=(lambda:None).__code__
        shared=''.join(['UNIT non-interned code constant ']*40)
        distinct=shared.encode().decode()
        aliased=code.replace(co_consts=(None,shared,shared))
        separate=code.replace(co_consts=(None,shared,distinct))
        self.assertEqual(aliased.co_consts,separate.co_consts)
        self.assertIsNot(separate.co_consts[1],separate.co_consts[2])
        self.assertNotEqual(marshal.dumps(aliased),marshal.dumps(separate))
        expected=ro.code_fingerprint(aliased)
        self.assertEqual(expected,ro.code_fingerprint(separate))
        extra_references=[separate,separate.co_consts,list(separate.co_consts)]
        self.assertEqual(expected,ro.code_fingerprint(extra_references[0]))
        self.assertEqual(expected,ro.code_fingerprint(separate.replace()))

    def test_constants_bytecode_and_nested_code_changes_are_detected(self):
        add=(lambda value:value+1).__code__;subtract=(lambda value:value-1).__code__
        expected=ro.code_fingerprint(add)
        self.assertNotEqual(expected,ro.code_fingerprint(add.replace(co_consts=(None,2))))
        self.assertNotEqual(expected,ro.code_fingerprint(add.replace(co_code=subtract.co_code)))
        nested=add.replace(co_consts=(None,add))
        changed=nested.replace(co_consts=(None,add.replace(co_consts=(None,2))))
        self.assertNotEqual(ro.code_fingerprint(nested),ro.code_fingerprint(changed))

    def test_constant_type_bits_and_order_are_preserved_without_fallback(self):
        code=(lambda:None).__code__
        negative=code.replace(co_consts=(None,(-0.0,complex(-0.0,1),frozenset(('a','b')),Ellipsis)))
        positive=code.replace(co_consts=(None,(0.0,complex(-0.0,1),frozenset(('b','a')),Ellipsis)))
        self.assertNotEqual(ro.code_fingerprint(negative),ro.code_fingerprint(positive))
        reordered=negative.replace(co_consts=(None,(-0.0,complex(-0.0,1),frozenset(('b','a')),Ellipsis)))
        self.assertEqual(ro.code_fingerprint(negative),ro.code_fingerprint(reordered))
        with self.assertRaisesRegex(TypeError,'Unsupported actual immutable'):
            ro.code_fingerprint(code.replace(co_consts=(None,{})))


class ReadonlyNativeStorageUnit(unittest.TestCase):
    def setUp(self):
        (ro.ROOT/'outputs').mkdir(exist_ok=True)
        temporary=tempfile.TemporaryDirectory(prefix='readonly_native_CPU_UNIT_',dir=ro.ROOT/'outputs')
        self.addCleanup(temporary.cleanup);self.root=Path(temporary.name)
        self.bank=self.root/'UNIT_old_bank';self.bank.mkdir()
        self.pre=self.root/'Dataset730_LiverOnlineCP_OF0';self.pre.mkdir()
        self.package=self.root/'UNIT_nnunetv2';self.package.mkdir()
        write(self.package/'__init__.py',{'UNIT_fixture_only':True})
        dataset_source='from pathlib import Path\nimport json\n'+textwrap.dedent(inspect.getsource(UnitDatasetBlosc2)).replace('UnitDatasetBlosc2','nnUNetDatasetBlosc2')
        dataset_file=self.package/'training/dataloading/nnunet_dataset.py';dataset_file.parent.mkdir(parents=True)
        dataset_file.write_text(dataset_source,encoding='utf8')
        self.train=['UNIT_case_'+str(i) for i in range(105)];self.val=['UNIT_case_'+str(i) for i in range(105,131)]
        self.split=dict(inner_train=self.train[:84],inner_val=self.train[84:],outer_train=self.train,outer_val=self.val)
        write(self.pre/'splits_final.json',[dict(train=self.train,val=self.val)])
        write(self.pre/'dataset.json',{'UNIT':True});write(self.pre/'dataset_fingerprint.json',{'UNIT':True})
        write(self.pre/(cp.PLANS+'.json'),dict(configurations={'3d_fullres':dict(patch_size=[128,128,128],batch_size=2,
            architecture=dict(network_class_name='ResidualEncoderUNet'),data_identifier='UNIT_fullres')}))
        self.data=self.pre/'UNIT_fullres';self.data.mkdir();self.arrays={}
        records=[]
        for case in self.train+self.val:
            image=self.root/'UNIT_raw'/ (case+'.ct');label=self.root/'UNIT_raw'/(case+'.seg')
            image.parent.mkdir(exist_ok=True);image.write_bytes(('UNIT_CT_'+case).encode());label.write_bytes(('UNIT_seg_'+case).encode())
            if case in self.train:
                records.append(dict(case_id=case,image=str(image),label=str(label),image_sha256=ro.sha(image),label_sha256=ro.sha(label)))
            save_array(self.data/(case+'.b2nd'),np.ones((1,2,2,2),dtype=np.float32))
            save_array(self.data/(case+'_seg.b2nd'),np.ones((1,2,2,2),dtype=np.int16))
            write(self.data/(case+'.pkl'),dict(UNIT=True,case_id=case))
            gt=self.pre/'gt_segmentations'/(case+'.nii.gz');gt.parent.mkdir(exist_ok=True);gt.write_bytes(('UNIT_GT_'+case).encode())
            if case in self.train:self.make_raw(case)
        self.inventory=self.root/'UNIT_inventory.json';write(self.inventory,dict(split=self.split,raw_records=records,UNIT=True))
        self.baseline=cp.validate_baseline(self.pre,self.split)
        self.source_bank=dict(split=self.split,baseline=self.baseline,entries_by_case={case:['UNIT_own_entry.npz'] for case in self.train},
            CP_audits={case:dict(lossless_storage=ro.read(self.bank/'raw_cases'/(case+'.json'))['lossless_storage']) for case in self.train},
            source_identity={name:ro.sha(ro.ROOT/name) for name in ro.SCIENCE_FILES},UNIT=True)
        write(self.bank/'index.json',self.source_bank)
        write(self.bank/'request.json',dict(inventory_sha256=ro.sha(self.inventory)))
        self.bank_validator=patch.object(cp,'validate_bank',side_effect=lambda value:value)
        self.bank_validator.start();self.addCleanup(self.bank_validator.stop)
        self.header=patch.object(ro,'_open_header',side_effect=lambda path:dict(shape=list(np.load(path,allow_pickle=False).shape),dtype=np.load(path,allow_pickle=False).dtype.str))
        self.header.start();self.addCleanup(self.header.stop)
        self.original_store=raw.LosslessRawBankStore._check
        self.original_dataset=dict(init=UnitDatasetBlosc2.__init__,load=UnitDatasetBlosc2.load_case,unpack=UnitDatasetBlosc2.unpack_dataset)
        self.addCleanup(self.restore_adapters)

    def restore_adapters(self):
        raw.LosslessRawBankStore._check=self.original_store;ro._STORE_ORIGINAL=None;ro._STORE_BINDINGS.clear()
        UnitDatasetBlosc2.__init__=self.original_dataset['init'];UnitDatasetBlosc2.load_case=self.original_dataset['load']
        UnitDatasetBlosc2.unpack_dataset=staticmethod(self.original_dataset['unpack'])
        ro._DATASET_ORIGINALS=None;ro._DATASET_BINDINGS.clear();UnitBlosc.calls.clear()

    def make_raw(self,case):
        values=dict(baseline_unclipped=np.arange(8,dtype=np.float64).reshape(1,2,2,2),
            baseline_seg=np.ones((1,2,2,2),dtype=np.int16),operator=np.eye(2),metadata=dict(case_id=case,UNIT=True))
        self.arrays[case]=values;tree_arrays={};tree=raw.original._encode_tree(values,tree_arrays)
        volumes={};specs={}
        for key,array in tree_arrays.items():
            role=next((role for role in ro.ROLES if tree[role]['$array']==key),None)
            name='raw_cases/'+case+'.'+key+('.b2nd' if role else '.npy');path=self.bank/name;save_array(path,array)
            spec=dict(shape=list(array.shape),dtype=array.dtype.str,path=name,sha256=ro.sha(path))
            if role:
                checksum=ro.hashlib.sha256(array.tobytes()).hexdigest()
                roundtrip=dict(all_voxels_verified=True,verified_voxels=array.size,verification_tiles=1,
                    source_tile_bytes_sha256=checksum,decoded_tile_bytes_sha256=checksum,
                    traversal='lexicographic full chunk tiles in C order',maximum_verification_tile_bytes=array.nbytes)
                spec.update(storage=raw.FORMAT,chunks=list(array.shape),blocks=list(array.shape),compression=raw.COMPRESSION,
                    roundtrip=roundtrip,uncompressed_bytes=array.nbytes,stored_bytes=path.stat().st_size)
                volumes[role]=dict(dtype=array.dtype.str,shape=list(array.shape),uncompressed_bytes=array.nbytes,
                    stored_bytes=path.stat().st_size,source_tile_bytes_sha256=checksum,decoded_tile_bytes_sha256=checksum,verified_voxels=array.size)
            specs[key]=spec
        lossless=dict(format=raw.FORMAT,volume_count=2,original_logical_array_bytes=sum(v.nbytes for v in tree_arrays.values()),
            original_volume_bytes=sum(v['uncompressed_bytes'] for v in volumes.values()),compressed_volume_file_bytes=sum(v['stored_bytes'] for v in volumes.values()),
            full_precision_volume_roundtrip_verified=True,verified_voxels=sum(v['verified_voxels'] for v in volumes.values()),
            volumes=volumes,candidate_storage_unchanged=True,whole_volume_runtime_decode=False,compression=raw.COMPRESSION)
        write(self.bank/'raw_cases'/(case+'.json'),dict(format=raw.original.STORAGE_FORMAT,kind='case',tree=tree,arrays=specs,
            v24_storage=raw.FORMAT,compression=raw.COMPRESSION,lossless_storage=lossless))

    def admission(self):
        return ro.build_admission(source_bank=self.bank,inventory_path=self.inventory,baseline_preprocessed=self.pre,
            additional_writable_bytes=4*819243768+19*255351*2,runtime_package=self.package)

    def test_complete_inventories_and_measured_budget(self):
        document=self.admission()
        self.assertEqual(len(document['raw_cases']),105);self.assertEqual(len(document['preprocessed']['files']),393)
        self.assertEqual(len(document['ground_truth_files']),131)
        self.assertEqual(document['budget']['required_free_bytes'],document['budget']['new_writable_bytes_estimate']+10*2**30)
        self.assertFalse(document['scores_reused']);self.assertFalse(document['learned_upper_reused'])
        ro.verify_admission(document,full_hash=True)

    def test_missing_raw_case_rejected(self):
        (self.bank/'raw_cases'/(self.train[-1]+'.json')).unlink()
        with self.assertRaisesRegex(ValueError,'full105'):self.admission()

    def test_missing_authoritative_raw_recipient_rejected(self):
        inventory=ro.read(self.inventory);inventory['raw_records'].pop();write(self.inventory,inventory)
        write(self.bank/'request.json',dict(inventory_sha256=ro.sha(self.inventory)))
        with self.assertRaisesRegex(ValueError,'authoritative105 raw recipients'):self.admission()

    def test_outer_validation_raw_record_cannot_enter_recipient_inventory(self):
        inventory=ro.read(self.inventory);case=self.val[0]
        image=self.root/'UNIT_raw'/(case+'.ct');label=self.root/'UNIT_raw'/(case+'.seg')
        inventory['raw_records'].append(dict(case_id=case,image=str(image),label=str(label),
            image_sha256=ro.sha(image),label_sha256=ro.sha(label)))
        write(self.inventory,inventory);write(self.bank/'request.json',dict(inventory_sha256=ro.sha(self.inventory)))
        with self.assertRaisesRegex(ValueError,'authoritative105 raw recipients'):self.admission()

    def test_missing_preprocessed_case_rejected(self):
        (self.data/(self.val[-1]+'.b2nd')).unlink()
        with self.assertRaisesRegex(ValueError,'full131'):self.admission()

    def test_unpacked_or_extra_array_rejected(self):
        (self.data/'UNIT_extra.npy').write_bytes(b'UNIT')
        with self.assertRaisesRegex(ValueError,'full131'):self.admission()

    def test_changed_authoritative_CT_rejected(self):
        (self.root/'UNIT_raw'/(self.train[0]+'.ct')).write_bytes(b'changed original CT')
        with self.assertRaisesRegex(ValueError,'raw source hash'):self.admission()

    def test_changed_cached_volume_rejected(self):
        manifest=ro.read(self.bank/'raw_cases'/(self.train[0]+'.json'))
        path=self.bank/manifest['arrays'][manifest['tree'][ro.ROLES[0]]['$array']]['path']
        save_array(path,np.zeros((1,2,2,2),dtype=np.float64))
        with self.assertRaisesRegex(ValueError,'SHA/header'):self.admission()

    def test_reduced_CT_precision_rejected(self):
        path=self.bank/'raw_cases'/(self.train[0]+'.json');manifest=ro.read(path)
        manifest['arrays'][manifest['tree'][ro.ROLES[0]]['$array']]['dtype']='<f4';write(path,manifest)
        with self.assertRaisesRegex(ValueError,'64-bit'):self.admission()

    def test_missing_whole_volume_proof_rejected(self):
        path=self.bank/'raw_cases'/(self.train[0]+'.json');manifest=ro.read(path)
        manifest['lossless_storage']['full_precision_volume_roundtrip_verified']=False;write(path,manifest)
        with self.assertRaisesRegex(ValueError,'full-precision'):self.admission()

    def test_mutation_after_admission_fails(self):
        document=self.admission();path=Path(document['raw_cases'][self.train[0]]['volumes'][ro.ROLES[0]]['file']['path'])
        path.write_bytes(b'new changed bytes')
        with self.assertRaisesRegex(ValueError,'source changed'):ro.verify_admission(document)

    def test_forbidden_reuse_flags_and_subset_rejected(self):
        document=self.admission();document['scores_reused']=True
        with self.assertRaisesRegex(ValueError,'complete105/131'):ro.verify_admission(document)
        document=self.admission();document['raw_cases'].pop(self.train[0])
        with self.assertRaisesRegex(ValueError,'complete105/131'):ro.verify_admission(document)

    def test_atomic_publication_preserves_existing_file_and_cleans_own_temp(self):
        target=self.root/'UNIT_atomic.json';ro.publish_json(target,dict(value=1));before=target.read_bytes()
        with self.assertRaises(FileExistsError):ro.publish_json(target,dict(value=2))
        self.assertEqual(target.read_bytes(),before);self.assertFalse(list(self.root.glob('*.tmp')))
        with self.assertRaises(ValueError):ro.publish_json(self.root/'UNIT_invalid.json',dict(value=float('nan')))
        self.assertFalse((self.root/'UNIT_invalid.json').exists())

    def test_disk_budget_keeps_full_reserve(self):
        document=self.admission();required=document['budget']['required_free_bytes']
        with patch.object(ro.shutil,'disk_usage',return_value=SimpleNamespace(free=required-1)):
            with self.assertRaises(OSError):ro.check_disk(self.root,document,preparation=True)
        with patch.object(ro.shutil,'disk_usage',return_value=SimpleNamespace(free=10*2**30-1)):
            with self.assertRaises(OSError):ro.check_disk(self.root,document)

    def writer(self,document):
        original=SimpleNamespace()
        def inner(bank_root,relative,case):return save_case(bank_root,relative,case)
        def outer(bank_root,relative,case):return _prepare_raw_case(bank_root,relative,case)
        original._prepare_raw_case=inner;original.prepare_raw_case=outer
        pipeline=SimpleNamespace(_root_raw_preparation=lambda:original)
        return ro.make_case_preparer(pipeline,document)

    def test_fresh_case_writes_private_small_arrays_and_only_static_refs(self):
        document=self.admission();destination=self.root/'UNIT_new_bank';destination.mkdir()
        writer=self.writer(document);case=self.train[0]
        def verify(array,path,spec):
            actual=np.load(path,allow_pickle=False)
            if array.tobytes()!=actual.tobytes():raise ValueError('Every UNIT voxel differs')
            return copy.deepcopy(spec['roundtrip'])
        with patch.object(raw,'_verify_all',side_effect=verify):
            checksum=writer(destination,'raw_cases/'+case+'.json',copy.deepcopy(self.arrays[case]))
        manifest=ro.read(destination/'raw_cases'/(case+'.json'))
        self.assertEqual(ro.sha(destination/'raw_cases'/(case+'.json')),checksum)
        self.assertEqual(len(manifest['readonly_source_volumes']),2)
        self.assertFalse(list(destination.rglob('*.b2nd')));self.assertEqual(len(list(destination.rglob('*.npy'))),1)
        self.assertEqual(manifest['lossless_storage'],document['raw_cases'][case]['lossless_storage'])

    def test_every_voxel_cache_comparison_rejects_changed_fresh_CT(self):
        document=self.admission();destination=self.root/'UNIT_new_bank';destination.mkdir();case=self.train[0]
        value=copy.deepcopy(self.arrays[case]);value[ro.ROLES[0]][0,-1,-1,-1]+=1
        def verify(array,path,spec):
            if array.tobytes()!=np.load(path,allow_pickle=False).tobytes():raise ValueError('Every UNIT voxel differs')
            return spec['roundtrip']
        with patch.object(raw,'_verify_all',side_effect=verify):
            with self.assertRaisesRegex(ValueError,'Every UNIT voxel'):self.writer(document)(destination,'raw_cases/'+case+'.json',value)
        self.assertFalse(list(destination.rglob('*.b2nd')))

    def dataset(self,document):
        native=self.root/'UNIT_native_runtime';private=native/'nnUNet_preprocessed'/self.pre.name/'UNIT_fullres';private.mkdir(parents=True,exist_ok=True)
        target=native/'runtime/nnunetv2/training/dataloading/nnunet_dataset.py';target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes((self.package/'training/dataloading/nnunet_dataset.py').read_bytes())
        module=ro.types.ModuleType('UNIT_private_original_dataset');module.__file__=str(target)
        module.__dict__['blosc2']=UnitBlosc
        exec(compile(target.read_text(),str(target),'exec',dont_inherit=True),module.__dict__)
        module.infer_dataset_class=lambda folder:'UNIT_unmodified_other_profile'
        trainer=SimpleNamespace()
        result=ro.install_dataset_adapter(document,private,dataset_module=module,trainer_module=trainer)
        return private,module,trainer,result

    def test_exact_dataset_class_original_reader_and_read_only_mode(self):
        document=self.admission();private,module,trainer,result=self.dataset(document)
        cls=module.nnUNetDatasetBlosc2
        self.assertIs(module.infer_dataset_class(private),cls)
        self.assertIs(trainer.infer_dataset_class(private),cls)
        data=cls(str(private),self.train)
        self.assertEqual(data.source_folder,str(self.data));ct,seg,prev,properties=data.load_case(self.train[0])
        self.assertFalse(ct.flags.writeable);self.assertFalse(seg.flags.writeable);self.assertIsNone(prev)
        self.assertEqual(properties['case_id'],self.train[0]);self.assertTrue(all(mode=='r' for _,mode in UnitBlosc.calls))
        self.assertIsNone(cls.unpack_dataset(str(private)))
        self.assertEqual(module.infer_dataset_class(self.root/'UNIT_other'),'UNIT_unmodified_other_profile')
        self.assertEqual(result['full_preprocessed_cases'],131)
        self.assertTrue(result['actual_private_class_and_bytecode_verified'])
        self.assertEqual(result['code_fingerprint_format'],ro.CODE_FINGERPRINT)
        self.assertEqual(set(cls.get_identifiers(str(private))),set(self.train+self.val))
        self.assertEqual(set(cls(str(private),None).identifiers),set(self.train+self.val))

    def test_dataset_rejects_foreign_identifier_and_previous_stage(self):
        document=self.admission();private,module,_,_=self.dataset(document);cls=module.nnUNetDatasetBlosc2
        with self.assertRaisesRegex(ValueError,'authoritative full131'):cls(str(private),['foreign_case'])
        with self.assertRaisesRegex(ValueError,'previous-stage'):cls(str(private),self.train,str(self.data))

    def test_dataset_read_checks_cached_source_witness(self):
        document=self.admission();private,module,_,_=self.dataset(document);data=module.nnUNetDatasetBlosc2(str(private),self.train)
        (self.data/(self.train[0]+'.pkl')).write_text('{}')
        with self.assertRaisesRegex(ValueError,'source changed'):data.load_case(self.train[0])

    def test_private_dataset_source_and_original_bytecode_mutation_rejected(self):
        document=self.admission();private,module,_,_=self.dataset(document);data=module.nnUNetDatasetBlosc2(str(private),self.train)
        original=ro._DATASET_ORIGINALS['load'];saved=original.__code__
        equivalent=saved.replace();self.assertEqual(ro.code_fingerprint(saved),ro.code_fingerprint(equivalent))
        original.__code__=equivalent
        try:
            with self.assertRaisesRegex(ValueError,'bytecode changed'):data.load_case(self.train[0])
        finally:original.__code__=saved
        original.__code__=(lambda self,identifier:None).__code__
        try:
            with self.assertRaisesRegex(ValueError,'bytecode changed'):data.load_case(self.train[0])
        finally:original.__code__=saved
        Path(module.__file__).write_text('changed private source')
        with self.assertRaisesRegex(ValueError,'source changed'):data.load_case(self.train[0])

    def test_dataset_private_folder_cannot_contain_duplicate_arrays(self):
        document=self.admission();private=self.root/'UNIT_private_arrays';private.mkdir();(private/'one.npy').write_bytes(b'UNIT')
        with self.assertRaisesRegex(ValueError,'duplicate/unpacked'):ro.install_dataset_adapter(document,private)

    def fresh_manifest_bank(self,document):
        destination=self.root/'UNIT_new_complete_bank';destination.mkdir()
        for case_id,cached in document['raw_cases'].items():
            manifest=ro.read(cached['manifest']['path']);manifest['readonly_storage_profile']=ro.PROFILE
            manifest['readonly_source_volumes']={}
            for role in ro.ROLES:
                key=manifest['tree'][role]['$array'];spec=manifest['arrays'][key]
                relative='raw_cases/'+case_id+'.'+key+'.b2nd';spec['path']=relative
                manifest['readonly_source_volumes'][relative]=dict(case_id=case_id,role=role,source=cached['volumes'][role]['file'])
            write(destination/'raw_cases'/(case_id+'.json'),manifest)
        return destination

    def test_store_keeps_exact_class_and_only_role_scoped_source_resolution(self):
        document=self.admission();destination=self.fresh_manifest_bank(document)
        result=ro.install_raw_store_adapter(document,destination);store=raw.LosslessRawBankStore(destination)
        self.addCleanup(store.close);self.assertIs(type(store),raw.LosslessRawBankStore)
        manifest=ro.read(destination/'raw_cases'/(self.train[0]+'.json'));role=ro.ROLES[0]
        spec=manifest['arrays'][manifest['tree'][role]['$array']]
        self.assertEqual(store._check(spec['path'],spec['sha256']),Path(document['raw_cases'][self.train[0]]['volumes'][role]['file']['path']))
        with self.assertRaisesRegex(ValueError,'hash differs'):store._check(spec['path'],'b'*64)
        with self.assertRaises(ValueError):store._check('scores/old_learned_scores.json','a'*64)
        self.assertEqual(result['readonly_role_references'],210)

    def test_store_rejects_nonvolume_external_path(self):
        document=self.admission();destination=self.fresh_manifest_bank(document)
        path=destination/'raw_cases'/(self.train[0]+'.json');manifest=ro.read(path)
        relative=next(iter(manifest['readonly_source_volumes']));row=manifest['readonly_source_volumes'].pop(relative)
        malicious='scores/borrowed_scores.json';manifest['readonly_source_volumes'][malicious]=row
        manifest['arrays'][manifest['tree'][row['role']]['$array']]['path']=malicious;write(path,manifest)
        with self.assertRaisesRegex(ValueError,'Role-specific'):ro.install_raw_store_adapter(document,destination)

    def test_private_native_metadata_and_fullGT_copied_without_array_links(self):
        document=self.admission();admission=self.root/'UNIT_admission.json';ro.publish_json(admission,document)
        own_bank=self.root/'UNIT_fresh_own_GPU4_bank';own_bank.mkdir()
        bank=copy.deepcopy(self.source_bank);bank.update(storage_profile=ro.PROFILE,storage_admission=str(admission),
            storage_admission_sha256=ro.sha(admission),pin=dict(format=cp.CURRENT_PIN_FORMAT),physical_GPU=4)
        write(own_bank/'index.json',bank);output=self.root/'UNIT_fresh_native'
        with patch.object(cp,'require_project_budget',return_value={'UNIT_only':True}),patch.object(ro.shutil,'disk_usage',return_value=SimpleNamespace(free=100*2**30)):
            path=ro.prepare_native(own_bank/'index.json',output,admission,cp)
        native=ro.read(path);private=output/'nnUNet_preprocessed'/self.pre.name
        self.assertEqual(len(list((private/'gt_segmentations').glob('*.nii.gz'))),131)
        self.assertFalse(list((private/'UNIT_fullres').iterdir()));self.assertFalse(list(output.rglob('*.b2nd')))
        self.assertEqual(native['readonly_native_array_inventory'],document['preprocessed'])
        for name,value in document['private_metadata_files'].items():self.assertEqual(ro.sha(private/name),value['sha256'])
        for name,value in document['custom_trainer_files'].items():
            self.assertEqual(ro.sha(output/'runtime/nnunetv2/training/nnUNetTrainer'/name),value['sha256'])
        self.assertEqual(native['physical_GPU'],4);self.assertFalse(native['readonly_source_files_written'])


if __name__=='__main__':unittest.main()
