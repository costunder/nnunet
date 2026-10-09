"""CPU DEBUG: real Blosc2, all fixture voxels; no patient/GPU performance claim."""
import copy
import json
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import blosc2
import numpy as np
from custom_trainers import onlinecp_raw_bank as raw
from custom_trainers.onlinecp_raw_resampling import apply_candidate,RAW_RESAMPLING_FORMAT
from hiercp_v1x import v24_lossless_raw_storage as storage


class LosslessStorageDebug(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='v24_blosc_DEBUG_',dir=Path.cwd())
        self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)

    def case(self,shape=(1,7,9,11)):
        rng=np.random.default_rng(93)
        return dict(metadata=dict(case_id='DEBUG',format=RAW_RESAMPLING_FORMAT,
            preprocessed_shape=list(shape[1:]),separate_z_axis=None,
            normalization=dict(lower=-500.,upper=600.,mean=100.,std=75.)),
            baseline_unclipped=rng.normal(size=shape).astype(np.float64),
            baseline_seg=rng.integers(-1,3,size=shape,dtype=np.int16),
            data_operators=[np.eye(n,dtype=np.float64) for n in shape[1:]],
            clip_min=np.array([-10.]),clip_max=np.array([10.]),
            preparation=dict(raw_ct=np.zeros((1,3,4,5),np.float32)))

    def publish(self,case=None):
        case=self.case() if case is None else case
        digest=storage.save_case(self.root,'cases/DEBUG.json',case)
        store=storage.LosslessRawBankStore(self.root);self.addCleanup(store.close)
        return case,digest,store,store.load_case('cases/DEBUG.json',digest)

    def test_fresh_activation_ignores_unbound_installed_or_archived_raw_backend(self):
        script='''import hashlib,sys,types
from pathlib import Path
for name in ('custom_trainers.onlinecp_raw_bank','nnunetv2.training.nnUNetTrainer.onlinecp_raw_bank'):
    fake=types.ModuleType(name);fake.__file__='/UNBOUND_ARCHIVE/raw.py';sys.modules[name]=fake
from hiercp_v1x import v24_lossless_raw_storage as storage
expected=Path.cwd()/'custom_trainers/onlinecp_raw_bank.py'
assert Path(storage.original.__file__).resolve()==expected.resolve()
assert storage.ORIGINAL_SOURCE_SHA256==hashlib.sha256(expected.read_bytes()).hexdigest()
print('EXACT_ROOT_BACKEND_DEBUG_PASS')
'''
        result=subprocess.run([sys.executable,'-B','-c',script],cwd=Path(__file__).resolve().parents[1],
                              capture_output=True,text=True,check=True)
        self.assertIn('EXACT_ROOT_BACKEND_DEBUG_PASS',result.stdout)

    def test_every_voxel_full64bit_bits_and_int16_preserved_across_chunk_boundaries(self):
        case=self.case((1,35,133,131))
        bits=case['baseline_unclipped'].view(np.uint64).reshape(-1)
        bits[:8]=[0,0x8000000000000000,1,0x7ff8000000000017,0xfff8000000000018,
                  0x0010000000000000,0x3ff0000000000001,0x7fefffffffffffff]
        labels=case['baseline_seg'].reshape(-1);labels[:4]=[-32768,32767,-1,2]
        case,digest,store,loaded=self.publish(case)
        manifest=json.loads((self.root/'cases/DEBUG.json').read_text())
        proof=manifest['lossless_storage']
        self.assertTrue(proof['full_precision_volume_roundtrip_verified'])
        self.assertEqual(proof['verified_voxels'],2*case['baseline_seg'].size)
        self.assertEqual(proof['original_volume_bytes'],case['baseline_unclipped'].nbytes+case['baseline_seg'].nbytes)
        self.assertGreater(manifest['arrays']['array_0000']['roundtrip']['verification_tiles'],1)
        for name in ('baseline_unclipped','baseline_seg'):
            proxy=loaded[name]
            self.assertEqual(proxy.dtype,case[name].dtype)
            self.assertEqual(proxy[:].tobytes(),case[name].tobytes())
            for index in ((slice(None),slice(31,35),slice(127,133),slice(126,131)),
                          (0,slice(0,2),slice(1,5),slice(2,7))):
                self.assertEqual(proxy[index].tobytes(),case[name][index].tobytes())
        self.assertNotIn('preparation',loaded)
        self.assertEqual(storage.save_case(self.root,'cases/DEBUG.json',case),digest)

    def test_lazy_crop_reads_only_requested_slice_and_never_implicit_whole_volume(self):
        case,digest,store,loaded=self.publish()
        self.assertIsNone(loaded['baseline_unclipped']._decoder)
        crop=(slice(None),slice(2,4),slice(3,6),slice(4,8))
        seen=[];real=storage._open
        class SliceWitness:
            def __init__(self,array):self.array=array
            def __getattr__(self,name):return getattr(self.array,name)
            def __getitem__(self,index):seen.append(index);return self.array[index]
        with patch.object(storage,'_open',side_effect=lambda path:SliceWitness(real(path))):
            output=loaded['baseline_unclipped'][crop]
        self.assertEqual(seen,[crop]);self.assertEqual(output.tobytes(),case['baseline_unclipped'][crop].tobytes())
        self.assertFalse(output.flags.writeable)
        self.assertEqual(loaded['baseline_unclipped']._decoder.dparams.nthreads,1)
        with self.assertRaisesRegex(TypeError,'implicit whole-volume'):np.asarray(loaded['baseline_unclipped'])
        with self.assertRaisesRegex(IndexError,'contiguous'):loaded['baseline_unclipped'][:,::2]

    def test_existing_candidate_payload_and_native_CP_outputs_are_byte_identical(self):
        case,digest,store,loaded=self.publish()
        old_digest=raw.save_case(self.root,'legacy/DEBUG.json',case)
        old=raw.RawBankStore(self.root);self.addCleanup(old.close)
        legacy=old.load_case('legacy/DEBUG.json',old_digest)
        source=np.arange(27,dtype=np.float32).reshape(1,3,3,3)*10-200
        candidate=dict(format=RAW_RESAMPLING_FORMAT,source_ct=source,source_mask=np.ones((3,3,3),bool),
            recipient_ct=np.full((1,3,3,3),120.,np.float32),target_input_origin=np.array([2,3,4]),
            outside_count=np.array([100]),outside_min=np.array([-4.]),outside_max=np.array([4.]),
            output_bbox=[[2,5],[3,6],[4,7]],seg_patch=np.full((1,3,3,3),2,np.int16),
            pasted_support=np.ones((3,3,3),bool),audit=dict(DEBUG=True))
        candidate_digest=raw.save_candidate(self.root,'candidate/selected.json',candidate)
        native_candidate=store.load_candidate('candidate/selected.json',candidate_digest)
        for crop in ([[0,7],[0,9],[0,11]],[[1,5],[2,7],[3,8]]):
            for scale,shift in ((1.,0.),(1.3,350.),(.4,-550.)):
                expected=apply_candidate(legacy,native_candidate,crop,scale=scale,shift_hu=shift)
                actual=apply_candidate(loaded,native_candidate,crop,scale=scale,shift_hu=shift)
                for name in ('data','seg','pasted_support'):
                    self.assertEqual(expected[name].dtype,actual[name].dtype)
                    self.assertEqual(expected[name].tobytes(),actual[name].tobytes())
                self.assertEqual(expected['audit'],actual['audit'])

    def test_complete_original_native_preparation_resampling_and_jitter_fixture_parity(self):
        from tests.test_raw_cp_resampling_debug import RawCPResamplingDebugTests
        from custom_trainers.onlinecp_raw_resampling import prepare_candidate
        # The established actual nnU-Net preprocessing DEBUG fixture retains
        # full cubic operators/global tails, original label stencil and grid.
        fixture=RawCPResamplingDebugTests().fixture()
        case=fixture[-1]
        source=np.arange(125,dtype=np.float32).reshape(5,5,5)*12-650
        mask=np.zeros((5,5,5),bool);mask[1:4,1:4,1:4]=True
        candidate=prepare_candidate(case,source,mask,np.array([2,2,2]),(10,10,10))
        case,digest,store,loaded=self.publish(case)
        old_digest=raw.save_case(self.root,'legacy/full_native.json',case)
        old=raw.RawBankStore(self.root);self.addCleanup(old.close)
        legacy=old.load_case('legacy/full_native.json',old_digest)
        candidate_digest=raw.save_candidate(self.root,'candidate/full_native.json',candidate)
        native_candidate=store.load_candidate('candidate/full_native.json',candidate_digest)
        shape=case['metadata']['preprocessed_shape']
        for crop in ([[0,n] for n in shape],[[1,8],[2,10],[3,11]]):
            for scale,shift in ((1.,0.),(1.3,350.),(.4,-550.)):
                expected=apply_candidate(legacy,native_candidate,crop,scale=scale,shift_hu=shift)
                actual=apply_candidate(loaded,native_candidate,crop,scale=scale,shift_hu=shift)
                for name in ('data','seg','pasted_support'):
                    self.assertEqual(actual[name].dtype,expected[name].dtype)
                    self.assertEqual(actual[name].tobytes(),expected[name].tobytes())
                self.assertEqual(actual['audit'],expected['audit'])

    def test_worker_pickle_keeps_witnesses_but_drops_decoders_and_mmaps(self):
        case,digest,store,loaded=self.publish()
        loaded['baseline_unclipped'][:,1:3,2:4,3:5]
        resumed=pickle.loads(pickle.dumps(store));self.addCleanup(resumed.close)
        self.assertEqual(store._witnesses,resumed._witnesses)
        self.assertFalse(resumed._cases);self.assertFalse(resumed._sources);self.assertIsNone(resumed._proxies)
        with patch.object(storage.original,'_sha',side_effect=AssertionError('Repeated full file SHA')):
            fresh=resumed.load_case('cases/DEBUG.json',digest)
            self.assertEqual(fresh['baseline_seg'][:,1:3,2:4,3:5].tobytes(),case['baseline_seg'][:,1:3,2:4,3:5].tobytes())
        store.close()
        self.assertIsNone(loaded['baseline_unclipped']._decoder)
        with self.assertRaisesRegex(ValueError,'closed'):loaded['baseline_unclipped'][:]

    def test_existing_publication_is_not_replaced_and_tamper_fails_before_cached_crop(self):
        case,digest,store,loaded=self.publish()
        changed=copy.deepcopy(case);changed['baseline_unclipped'][0,0,0,0]+=1
        with self.assertRaisesRegex(ValueError,'roundtrip'):storage.save_case(self.root,'cases/DEBUG.json',changed)
        manifest=json.loads((self.root/'cases/DEBUG.json').read_text())
        spec=manifest['arrays'][manifest['tree']['baseline_unclipped']['$array']]
        path=self.root/spec['path']
        loaded['baseline_unclipped'][:,1:3,2:4,3:5]
        with path.open('r+b') as handle:
            handle.seek(-1,2);value=handle.read(1);handle.seek(-1,2);handle.write(bytes([value[0]^1]))
        with self.assertRaisesRegex(ValueError,'changed after verification'):loaded['baseline_unclipped'][:]

    def test_lossy_filter_or_wrong_dtype_metadata_is_never_admitted(self):
        case,digest,store,loaded=self.publish()
        bad=copy.deepcopy(case);bad['baseline_unclipped']=bad['baseline_unclipped'].astype(np.float32)
        with self.assertRaisesRegex(ValueError,'float64'):storage.save_case(self.root,'wrong.json',bad)
        manifest=json.loads((self.root/'cases/DEBUG.json').read_text())
        manifest['compression']['filter']='TRUNC_PREC'
        path=self.root/'bad.json';path.write_text(json.dumps(manifest),encoding='utf8')
        with self.assertRaisesRegex(ValueError,'Incompatible explicit lossless'):store.load_case('bad.json',raw._sha(path))
        original_file=self.root/manifest['arrays']['array_0000']['path']
        # Test an actual lossy container with a forged lossless spec/hash. The
        # runtime checks the real codec/filter header before decoding the crop.
        lossy=self.root/'lossy.b2nd'
        c=blosc2.asarray(case['baseline_unclipped'],urlpath=str(lossy),mode='w',
            chunks=case['baseline_unclipped'].shape,blocks=case['baseline_unclipped'].shape,
            cparams=dict(codec=blosc2.Codec.ZSTD,nthreads=1,
                         filters=[blosc2.Filter.NOFILTER]*5+[blosc2.Filter.TRUNC_PREC],
                         filters_meta=[0]*5+[12]),dparams=dict(nthreads=1));c=None
        spec=json.loads((self.root/'cases/DEBUG.json').read_text())['arrays']['array_0000']
        spec['path']='lossy.b2nd';spec['sha256']=raw._sha(lossy)
        proxy=storage._CropArray(store,spec)
        with self.assertRaisesRegex(ValueError,'compression contract'):proxy[:]
        self.assertTrue(original_file.exists())


if __name__=='__main__':unittest.main()
