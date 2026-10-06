"""DEBUG policy and adversarial split tests. No synthetic fixture is an experiment."""
import copy
import tempfile
import os
import threading
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from hiercp_v22.donors import (POLICY,donor_pool,context_round,select_donor,validate_pool,
                              validate_training_rows,reject_cross_split_duplicates)
from hiercp_v22.contracts import write_new,validate_checkpoint,source_identity,load_config,sha
from hiercp_v22.bank import entry_name,validate_catalog,EntryBuilder

SPLIT=dict(inner_train=['A','B','C'],inner_val=['V'],outer_train=['A','B','C','V'],outer_val=['H'])
POOL=[dict(case_id='A',component_id=1),dict(case_id='C',component_id=2)]

def catalog():
    entries={c:[entry_name(c,i) for i in range(len(POOL))] for c in SPLIT['outer_train']}
    return dict(format='hiercp_online_bank_v2',donor_policy=POLICY,materialization='on_demand_complete_128_pool_v1',
        entry_storage='v21_scores128_selected_raw_target_v1',
        split=copy.deepcopy(SPLIT),donor_pool=copy.deepcopy(POOL),entries_by_case=entries,
        source_slots_by_case={c:[dict(source_component=i+1,status='ok',entry=n) for i,n in enumerate(names)] for c,names in entries.items()},
        eligible_sources_by_case={c:[1,2] for c in entries},no_placement_sources=0,eligible_source_slots=8,
        no_placement_policy='retain_original',source_schedule_format='onlinecp_all_source_slots_v1',
        candidate_count=128,hier_top_k=1,tumor_label=2,liver_label=1,cp_probability=.5,
        intensity_scale_range=[.95,1.05],intensity_shift_range_hu=[-5,5],paste_contract='onlinecp_raw_target_paste_v1')

class SharedDonorLeakageDebugTests(unittest.TestCase):
    def test_validation_tumors_do_not_change_donor_pool_or_training_schedule(self):
        inv={'A':[1],'B':[],'C':[2],'V':[1]}
        before=donor_pool(inv,SPLIT); schedule=context_round(before,SPLIT['inner_train'],42)
        inv['V']=list(range(1,1000)); inv['H']=[999]
        self.assertEqual(donor_pool(inv,SPLIT),before)
        self.assertEqual(context_round(donor_pool(inv,SPLIT),SPLIT['inner_train'],42),schedule)

    def test_every_case_same_complete_donor_pool_and_selection(self):
        validate_catalog(catalog())
        for c in SPLIT['outer_train']:
            self.assertEqual([select_donor(POOL,SPLIT,c,u) for u in (.1,.9)],[0,1])
        with self.assertRaisesRegex(ValueError,'held-out'):select_donor(POOL,SPLIT,'H',.1)

    def test_round_covers_all_donors_and_patients_without_cartesian_product(self):
        pool=[dict(case_id='A',component_id=i+1) for i in range(527)]
        events=context_round(pool,[f'P{i}' for i in range(84)],42)
        self.assertEqual(len(events),527)
        self.assertEqual({e['donor_index'] for e in events},set(range(527)))
        counts=[sum(e['recipient']==f'P{i}' for e in events) for i in range(84)]
        self.assertEqual(max(counts)-min(counts),1)
        few=context_round(POOL,SPLIT['inner_train'],42)
        self.assertEqual({e['recipient'] for e in few},set(SPLIT['inner_train']))

    def test_reject_inner_validation_and_outer_validation_donors(self):
        for case in ['V','H']:
            bad=copy.deepcopy(POOL); bad[0]['case_id']=case
            with self.assertRaisesRegex(ValueError,'leakage'):validate_pool(bad,SPLIT)
            meta=catalog(); meta['donor_pool']=bad
            with self.assertRaises(ValueError):validate_catalog(meta)

    def test_no_missing_zero_tumor_recipient_or_heldout_recipient(self):
        for key in ['entries_by_case','source_slots_by_case','eligible_sources_by_case']:
            bad=catalog(); del bad[key]['B']
            with self.assertRaises(ValueError):validate_catalog(bad)
        bad=catalog(); bad['entries_by_case']['H']=bad['entries_by_case']['B']
        with self.assertRaises(ValueError):validate_catalog(bad)

    def test_external_placement_never_becomes_observed_positive(self):
        row=dict(case_id='B',donor_case_id='A',component_id=1,evidence=-1)
        validate_training_rows([row],SPLIT,POOL)
        row['evidence']=1
        with self.assertRaisesRegex(ValueError,'Foreign'):validate_training_rows([row],SPLIT,POOL)
        row.update(evidence=-1,donor_case_id='V')
        with self.assertRaisesRegex(ValueError,'outside'):validate_training_rows([row],SPLIT,POOL)

    def test_same_image_with_another_id_cannot_cross_split(self):
        for case in ['V','H']:
            rows=[dict(case_id='A',image_sha256='same'),dict(case_id=case,image_sha256='same')]
            with self.assertRaisesRegex(ValueError,'Identical-image'):reject_cross_split_duplicates(rows,SPLIT)

    def test_native_loader_selects_external_donors_for_both_tumor_and_no_tumor_cases(self):
        from hiercp_v22.native_adapter import SharedDonorBank,SharedDonorLoader
        with tempfile.TemporaryDirectory(prefix='shared_donor_DEBUG_') as name:
            file=Path(name)/'index.json'; write_new(file,catalog()); bank=SharedDonorBank(file)
            obj=SharedDonorLoader.__new__(SharedDonorLoader); obj.online_bank=bank; obj.online_epoch=0; obj.online_policy='hier_argmax'
            class FixedDraw:
                def __init__(self,u):self.values=iter([0.,u,.3,.4,.5])
                def random(self):return next(self.values)
            for case in ['A','B','C','V']:
                for u,index in [(.1,0),(.9,1)]:
                    obj._cp_rng=FixedDraw(u)
                    entry=dict(candidate_centers=np.zeros((128,3),np.int32),scores=np.arange(128,dtype=np.float32))
                    with patch.object(bank,'_load',return_value=entry) as load,patch.object(obj,'_make_paste_plan',return_value={'DEBUG_plan':True}) as make:
                        result,_=obj._sample_paste_plan(case)
                        load.assert_called_once_with(entry_name(case,index))
                        self.assertEqual(make.call_args.args[1],127)
                        self.assertEqual(make.call_args.args[-1],case)
                        self.assertTrue(result['DEBUG_plan'])

    def test_service_rejects_heldout_before_loading_any_volume(self):
        builder=EntryBuilder.__new__(EntryBuilder); builder.meta=catalog()
        with self.assertRaisesRegex(ValueError,'held-out'):builder.materialize('H',0)

    def test_contaminated_checkpoint_support_is_rejected(self):
        from hiercp_v22 import PIPELINE_VERSION
        cfg,_=load_config()
        payload=dict(format=PIPELINE_VERSION,complete=True,source_identity=source_identity(),config=cfg,
            split=copy.deepcopy(SPLIT),donor_pool=POOL,patient_ids=['A','B','C'],raw_records=[],
            memory=dict(case_ids=['A','B','C'],embeddings=torch.zeros(3,128),descriptors=torch.zeros(3,2),
                        owners=torch.arange(3),evidence=torch.tensor([1,-1,1]),donor_allowed=torch.ones(3,dtype=torch.bool)))
        validate_checkpoint(payload)
        payload['memory']['case_ids']=['A','V','C']
        with self.assertRaisesRegex(ValueError,'held-out'):validate_checkpoint(payload)

    def test_authenticated_service_errors_propagate_without_fallback(self):
        from hiercp_v22.native_adapter import MaterializationService,request_entry
        with tempfile.TemporaryDirectory(prefix='rpc_DEBUG_') as name,patch.dict(os.environ):
            class DebugBuilder:
                def __init__(self,*args):self.root=Path(name)
                def materialize(self,recipient,donor_index):
                    if recipient=='H':raise ValueError('held-out DEBUG request')
                    return dict(relative=entry_name(recipient,donor_index),sha256='a'*64)
            with patch('hiercp_v22.native_adapter.EntryBuilder',DebugBuilder):service=MaterializationService('DEBUG',threading.RLock())
            try:
                self.assertEqual(request_entry('B',0)['relative'],entry_name('B',0))
                with self.assertRaisesRegex(RuntimeError,'held-out'):request_entry('H',0)
            finally:service.close()
            self.assertEqual(len(list((Path(name)/'failures').glob('*.json'))),1)

    def test_native_validation_factory_is_never_replaced_by_cp_factory(self):
        from hiercp_v22.native_adapter import native_dataloaders,SharedDonorBank,SharedDonorLoader
        from custom_trainers.nnUNetTrainer_OnlinePairedCP import OnlineCPBank,nnUNetDataLoaderOnlineCP,nnUNetDataLoader
        # Exercise the same function rebinding mechanism without constructing a network.
        def factories(_):return OnlineCPBank,nnUNetDataLoaderOnlineCP,nnUNetDataLoader
        # Bind globals (not closure locals), as in the real parent get_dataloaders.
        import types
        exec('def original(_): return OnlineCPBank, nnUNetDataLoaderOnlineCP, nnUNetDataLoader',scope:={
            'OnlineCPBank':OnlineCPBank,'nnUNetDataLoaderOnlineCP':nnUNetDataLoaderOnlineCP,'nnUNetDataLoader':nnUNetDataLoader})
        original=scope['original']; changed=native_dataloaders(None,original)
        self.assertEqual(changed,(SharedDonorBank,SharedDonorLoader,nnUNetDataLoader))
        self.assertEqual(original(None),(OnlineCPBank,nnUNetDataLoaderOnlineCP,nnUNetDataLoader))

    def test_selected_paste_storage_restores_preparation_and_exact_native_labels(self):
        from test_online_raw_bank_preparation_debug import fixture
        from tools.online_raw_bank_preparation import prepare_raw_case,prepare_source_candidates
        from custom_trainers.onlinecp_raw_bank import save_case,RawBankStore
        from hiercp_v22.native_adapter import SharedDonorBank
        image,labels,props,plans,pre,seg=fixture()
        with tempfile.TemporaryDirectory(prefix='selected_raw_DEBUG_') as name:
            root=Path(name);meta=catalog();write_new(root/'index.json',meta)
            case,ref,digest=prepare_raw_case(root,'B',image,labels,props,plans,pre,seg,
                configuration_name='debug',raw_spacing=[.7]*3,raw_spatial_unit='mm',minimum_free_bytes=0)
            store=RawBankStore(root);reader=None
            try:
                prep_ref='raw_preparation/B.json';prep_sha=save_case(root,prep_ref,case['preparation'])
                restored=dict(store.load_case(ref,digest));restored['preparation']=store.load_case(prep_ref,prep_sha)
                source=image[11:16,11:16,11:16];mask=np.zeros(source.shape,dtype=bool)
                mask[1:4,1:4,1:4]=True  # Explicit synthetic external donor, independent of recipient GT
                centers=np.array([[3+i//16,3+(i//4)%4,3+i%4] for i in range(128)],dtype=np.int32)
                selected=37;refs,hashes,_=prepare_source_candidates(root,'B',1,restored,digest,source,mask,[2]*3,
                    centers[selected:selected+1],[12]*3,candidate_map=lambda fn,tasks:[fn(t) for t in tasks])
                scores=np.zeros(128,dtype=np.float32);scores[selected]=1
                entry=dict(paste_contract=np.array(['onlinecp_raw_target_paste_v1']),case_id=np.array(['B']),donor_case_id=np.array(['A']),
                    donor_component_id=np.array([1]),source_component=np.array([1]),source_diameter_mm=np.array([1.]),
                    candidate_centers=centers,candidate_raw_centers=centers,scores=scores,selected_candidate=np.array([selected]),
                    selected_payload=refs,selected_payload_sha256=hashes,raw_case_reference=np.array([ref]),raw_case_reference_sha256=np.array([digest]))
                reader=SharedDonorBank(root/'index.json');reader._validate_raw_entry(entry,'DEBUG')
                runtime,candidate=reader.load_raw_candidate(entry,selected)
                result=reader.raw_apply_function()(runtime,candidate,candidate['output_bbox'],scale=1.,shift_hu=0.)
                self.assertTrue(result['pasted_support'].any());self.assertTrue(np.all(result['seg'][0][result['pasted_support']]==2))
                self.assertEqual(len(list((root/'raw_candidates').rglob('*.json'))),1)
                self.assertEqual(len(entry['scores']),128)
                with self.assertRaises(ValueError):reader.load_raw_candidate(entry,0)
                entry['selected_candidate'][0]=0
                with self.assertRaises(ValueError):reader._validate_raw_entry(entry,'DEBUG')
            finally:
                store.close()
                if reader is not None:reader._get_raw_store().close()

if __name__=='__main__':unittest.main()
