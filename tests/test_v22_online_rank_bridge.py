"""Explicit DEBUG fixtures: native transport is real; scores are test inputs."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from tools.v22_online_selection import CONTRACT, validate_selection, RankedEventLoaderMixin
from tools.v22_online_rank_adapter import RankedBank, RankedLoader
from custom_trainers.nnUNetTrainer_OnlinePairedCP import nnUNetDataLoaderOnlineCP
from tools.v22_online_rank_bank import proposal_centers


def selection():
    return dict(selection_contract=CONTRACT,recipient='r',donor='d',component=1,
        centers=[[5,5,5],[10,10,10]],scores=[9.,8.],eligible=[False,True],
        selected_index=1,keep_original=False,score_override=False,filter_after_model_scoring=True)


class OnlineRankTests(unittest.TestCase):
    def test_rpc_transmits_owner_selection_and_propagates_failure(self):
        import threading
        from tools.v22_online_rank_adapter import RankedMaterializationService,request_entry
        from tools.v22_online_rank_bank import publish_receipt
        with tempfile.TemporaryDirectory(prefix='DEBUG_rank_rpc_') as folder:
            root=Path(folder)
            class FixtureBuilder:
                def __init__(self,index,gpu_lock):self.root=root
                def materialize(self,recipient,donor_index):
                    if donor_index!=0:raise ValueError('DEBUG explicit invalid donor')
                    publish_receipt(root/'event.json',dict(selection=selection()))
                    return dict(relative='event.json')
            with patch('tools.v22_online_rank_adapter.RankedEntryBuilder',FixtureBuilder),patch.dict('os.environ',{}):
                service=RankedMaterializationService(root/'index.json',threading.RLock())
                try:
                    result=request_entry('r',0)
                    self.assertEqual(validate_selection(json.loads((root/result['relative']).read_text())['selection']),1)
                    with self.assertRaisesRegex(RuntimeError,'invalid donor'):request_entry('r',1)
                finally:service.close()

    def test_rank2_and_none_no_score_rewrite(self):
        s=selection();self.assertEqual(validate_selection(s),1);self.assertEqual(s['scores'],[9.,8.])
        s.update(eligible=[False,False],selected_index=None,keep_original=True)
        self.assertIsNone(validate_selection(s))
        s['selected_index']=0
        with self.assertRaises(ValueError):validate_selection(s)

    def test_reject_raw_argmax_and_bad_tie(self):
        s=selection();s['selected_index']=0
        with self.assertRaises(ValueError):validate_selection(s)
        from tools.v22_candidate_order import candidate_key,candidate_order
        s.update(scores=[3.,3.],eligible=[True,True])
        expected=int(candidate_order(s['scores'],[candidate_key('r','d',1,c) for c in s['centers']])[0])
        s['selected_index']=expected;self.assertEqual(validate_selection(s),expected)
        s['selected_index']=1-expected
        with self.assertRaises(ValueError):validate_selection(s)

    def test_five_draws_unchanged_for_no_placement(self):
        class Loader(RankedEventLoaderMixin):pass
        loader=Loader();loader.online_epoch=0
        loader.online_bank=SimpleNamespace(cp_probability=.8,selected_index=lambda entry:None)
        loader._source_entry_names=lambda c:['event'];loader._load_selected_source=lambda c,i:{}
        draws=iter([.1,.2,.3,.4,.5]);consumed=[]
        def draw():
            v=next(draws);consumed.append(v);return v
        loader._rng=lambda:SimpleNamespace(random=draw)
        plan,token=loader._sample_paste_plan('r')
        self.assertIsNone(plan);self.assertEqual(consumed,[.1,.2,.3,.4,.5])
        from custom_trainers.nnUNetTrainer_OnlinePairedCP import _stable_u64,TRAINER_FORMAT
        self.assertEqual(token,_stable_u64(TRAINER_FORMAT,0,'r',1,*[v.hex() for v in consumed[1:]]))

    def test_proposals_do_not_depend_on_tumor_vs_liver_labels(self):
        from hiercp.common import SourceTumor
        from hiercp_v222.v1_cache import configuration
        cfg,base=configuration()
        shape=(25,25,25);image=np.ones(shape,np.float32);label=np.ones(shape,np.int16)
        mask=np.ones((3,3,3),bool)
        source=SourceTumor(1,mask,mask,np.ones(mask.shape,np.float32),tuple(slice(0,3) for _ in range(3)),(1,1,1),(1.,1.,1.),27)
        case=SimpleNamespace(paths=SimpleNamespace(case_id='r'),image=image,label=label,spacing=np.ones(3),shape=shape)
        a,_=proposal_centers(case,source,cfg,base,np.ones(shape),'d')
        label[:]=2
        b,_=proposal_centers(case,source,cfg,base,np.ones(shape),'d')
        self.assertEqual(a,b);self.assertEqual(len(a),128)

    def test_native_rank2_full_input_matches_raw_paste_oracle(self):
        from test_raw_cp_resampling_debug import RawCPResamplingDebugTests,debug_native,debug_plan
        from custom_trainers.onlinecp_raw_bank import save_case,save_candidate
        from custom_trainers.onlinecp_raw_resampling import prepare_candidate
        from hiercp_v222.placement import PlacementSpec
        from hiercp_v222.bank import map_center
        ct,seg,spacing,plans,case=RawCPResamplingDebugTests().fixture(
            plans=debug_plan(transpose=(2,0,1)),cropped=True)
        case['metadata']['case_id']='r'
        image=np.arange(27,dtype=np.float32).reshape(3,3,3)*8+100
        mask=np.ones((3,3,3),bool);center=(10,10,10);anchor=(1,1,1)
        p=PlacementSpec('r','d',1,center,tuple(spacing),tuple(map(tuple,np.diag([*spacing,1.]))),image,mask,anchor,tuple(map(tuple,np.eye(3))))
        data,labels,props=debug_native(ct,seg,spacing,plans)
        with tempfile.TemporaryDirectory(prefix='DEBUG_rank_native_') as folder:
            root=Path(folder);ref='case.json';digest=save_case(root,ref,case)
            candidate=prepare_candidate(case,p.image,p.mask,p.anchor,p.center)
            candidate.update(case_id='r',source_component=1,raw_target_center=list(center),case_reference_sha256=digest)
            payload='candidate.json';pd=save_candidate(root,payload,candidate)
            receipt=dict(status='ranked',selection=selection(),placement=p.metadata(),
                raw_case_reference=ref,raw_case_reference_sha256=digest,selected_payload=payload,selected_payload_sha256=pd)
            entry=dict(_receipt=receipt,case_id=np.asarray(['r']),source_component=np.asarray([1]),
                candidate_centers=np.stack([map_center(c,plans,props,case['metadata']['preprocessed_shape']) for c in selection()['centers']]))
            bank=RankedBank.__new__(RankedBank);bank.root=root;bank._raw_store=None
            bank.cp_probability=.8;bank.paste_contract='onlinecp_raw_target_paste_v1'
            bank.intensity_scale=(.95,1.05);bank.intensity_shift_hu=(-5.,5.)
            loader=RankedLoader.__new__(RankedLoader);loader.online_bank=bank;loader.online_epoch=0
            loader._source_entry_names=lambda c:['event'];loader._load_selected_source=lambda c,i:entry
            draws=iter([.1,.2,.3,.4,.5]);loader._rng=lambda:SimpleNamespace(random=lambda:next(draws))
            loader.batch_size=1;loader.patch_size=np.asarray(data.shape[1:]);loader.need_to_pad=np.zeros(3,dtype=int)
            loader.patch_size_was_2d=False;loader.transforms=None;loader.get_indices=lambda:['r'];loader.get_do_oversample=lambda j:False
            loader._data=SimpleNamespace(load_case=lambda c:(data,labels,None,{'class_locations':{}}))
            loader._raw_candidate_crop_bbox=lambda p,s,c:([0,0,0],list(s))
            try:
                batch=nnUNetDataLoaderOnlineCP.generate_train_batch(loader)
                copied=ct.copy();changed=seg.copy();points=p.coordinates();where=tuple(points.T)
                copied[where]=p.image[p.mask]*.99;changed[where]=2
                expected,target,_=debug_native(copied,changed,spacing,plans)
                np.testing.assert_allclose(np.asarray(batch['data'])[0],expected,rtol=4e-6,atol=4e-6)
                np.testing.assert_array_equal(np.asarray(batch['target'])[0],target)
                self.assertEqual(batch['online_cp_applied'][0],1)
                with self.assertRaises(ValueError):bank.load_raw_candidate(entry,0)
                broken=copy.deepcopy(entry);broken['_receipt']['placement']['anchor']=[0,1,1]
                with self.assertRaisesRegex(ValueError,'anchor'):bank.load_raw_candidate(broken,1)
                # Same native loader, all candidates ineligible: unchanged final
                # nnU-Net input, same five CP draws, no replacement donor.
                receipt['selection'].update(eligible=[False,False],selected_index=None,keep_original=True)
                draws=iter([.1,.2,.3,.4,.5]);loader.get_bbox=lambda s,f,c:([0,0,0],list(s))
                unchanged=nnUNetDataLoaderOnlineCP.generate_train_batch(loader)
                np.testing.assert_array_equal(unchanged['data'][0],data)
                np.testing.assert_array_equal(unchanged['target'][0],labels)
                self.assertEqual(unchanged['online_cp_applied'][0],0)
            finally:bank._get_raw_store().close()


class StorageTests(unittest.TestCase):
    def test_server_plan_generates_storage_report_before_cache(self):
        from tools.run_v222_server import stages
        commands=stages(Path('/data'),Path('/new'),32,9,optimized=True,process_loader=True,
                        storage_reference=Path('/old/index.json'))
        names=[name for name,_ in commands]
        self.assertLess(names.index('storage_plan'),names.index('paired_cache'))
        self.assertIn('--storage-plan',dict(commands)['paired_cache'])

    def test_no_space_or_wrong_index_refused_without_output(self):
        from tools.v22_cache_storage import make_plan,admit
        with tempfile.TemporaryDirectory(prefix='DEBUG_storage_') as folder:
            root=Path(folder)
            for name in ('g','s','d'):(root/name).write_bytes(b'x'*12)
            meta=dict(records=[dict(path='g',shared_source=dict(path='s'))],donor_files={'d':dict(path='d')},donor_pool=[{'case_id':'d','component_id':1}])
            index=root/'index.json';index.write_text(json.dumps(meta))
            output=root/'new';plan=make_plan(index,index,output);counts=plan['counts']
            with patch('tools.v22_cache_storage.shutil.disk_usage',return_value=SimpleNamespace(free=35)):
                with self.assertRaises(OSError):admit(plan,index,output,counts,0)
            self.assertFalse(output.exists())
            index.write_text(json.dumps(meta)+' ')
            with self.assertRaises(ValueError):admit(plan,index,output,counts,0)


if __name__=='__main__':unittest.main()
