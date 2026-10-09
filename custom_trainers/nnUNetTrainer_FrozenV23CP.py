"""Private native trainer: fixed actual v23 donors, original segmentation val."""
import json
import os
from pathlib import Path
import types
import time
import numpy as np
import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer_OnlinePairedCP import (
    OnlineCPBank, nnUNetDataLoaderOnlineCP, _nnUNetTrainer_250epochs_OnlineCP)
from hiercp_v1x.v24_nnunet_cp import validate_bank, sha, read, FORMAT,require_project_budget,resource_contract,eligible_argmax,CP_SELECTION


class FrozenV23Bank(OnlineCPBank):
    def __init__(self, index_path, cache_entries=64):
        super().__init__(index_path,cache_entries)
        validate_bank(self.metadata)
        manifest_path=self.root/self.metadata['scoring_manifest']
        if sha(manifest_path)!=self.metadata['scoring_manifest_file_sha256']:
            raise ValueError('Complete frozen score manifest changed')
        self.v24_score_manifest=read(manifest_path)
        if (self.v24_score_manifest.get('complete') is not True
                or self.v24_score_manifest.get('pin')!=self.metadata['pin']
                or set(self.v24_score_manifest['score_files'])!=set(self.metadata['split']['outer_train'])):
            raise ValueError('Bank lacks the complete actual105 frozen scores')

    def _get_raw_store(self):
        # This arm owns its lossless decoder. Original/common bank defaults
        # remain unchanged, and large case arrays decode only requested crops.
        if self._raw_store is None:
            from hiercp_v1x.v24_lossless_raw_storage import LosslessRawBankStore
            self._raw_store=LosslessRawBankStore(self.root)
        return self._raw_store

    def _load(self, name):
        if name not in self.metadata['entry_sha256'] or sha(self.root/name)!=self.metadata['entry_sha256'][name]:
            raise ValueError('Actual frozen CP entry changed or is outside the training bank')
        return super()._load(name)

    def _validate_raw_entry(self, entry, path):
        required={'paste_contract','case_id','donor_case_id','donor_component_id','candidate_centers',
            'candidate_raw_centers','scores','source_component','source_diameter_mm','selected_candidate',
            'candidate_eligibility','selection_policy',
            'selected_payload','selected_payload_sha256','raw_case_reference','raw_case_reference_sha256'}
        if not required.issubset(entry):raise ValueError('Incomplete frozen v23 raw paste entry')
        case=str(entry['case_id'][0])
        if case not in self.metadata['split']['outer_train']:raise ValueError('Held-out CP recipient')
        audit=self.metadata['CP_audits'][case]
        score_proof=self.v24_score_manifest['score_files'][case]
        if score_proof['path']!='scores/'+case+'.json' or sha(self.root/score_proof['path'])!=score_proof['sha256']:
            raise ValueError('Original128 frozen scores changed')
        original_scores=read(self.root/score_proof['path'])
        if (str(entry['donor_case_id'][0])!=audit['donor_case_id']
                or int(entry['donor_component_id'][0])!=audit['donor_component']):
            raise ValueError('Actual donor changed')
        for name in ('candidate_centers','candidate_raw_centers'):
            if entry[name].shape!=(128,3) or entry[name].dtype.kind not in 'iu':raise ValueError('Complete128 geometry required')
        if np.unique(entry['candidate_raw_centers'],axis=0).shape[0]!=128:raise ValueError('Duplicate raw CP centers')
        if entry['scores'].shape!=(128,) or not np.isfinite(entry['scores']).all():raise ValueError('Finite128 actual scores required')
        if (not np.array_equal(entry['scores'],np.asarray(original_scores['scores'],dtype=np.float32))
                or not np.array_equal(entry['candidate_raw_centers'],np.asarray(original_scores['centers']))
                or audit['scores_file_sha256']!=score_proof['sha256']):
            raise ValueError('Actual payload score/center arrays differ from frozen inference')
        selected=eligible_argmax(entry['scores'],entry['candidate_eligibility'])
        if (int(entry['selected_candidate'][0])!=selected or selected!=audit['selected_candidate']
                or entry['candidate_eligibility'].tolist()!=audit['eligibility']['eligible_mask']
                or entry['selection_policy'].shape!=(1,) or str(entry['selection_policy'][0])!=CP_SELECTION):
            raise ValueError('Exact eligible original128-bank argmax admission required')
        if float(entry['source_diameter_mm'][0])<=0 or float(entry['source_diameter_mm'][0])>20:raise ValueError('Original small real donor required')
        if str(entry['paste_contract'][0])!=self.paste_contract:raise ValueError('Raw native paste contract changed')
        for name in ('selected_payload','selected_payload_sha256','raw_case_reference','raw_case_reference_sha256'):
            if entry[name].shape!=(1,) or entry[name].dtype.kind!='U' or not str(entry[name][0]):raise ValueError('Exact raw payload reference required')

    def load_raw_candidate(self, entry, candidate_index):
        selected=int(entry['selected_candidate'][0])
        if candidate_index!=selected or selected!=eligible_argmax(entry['scores'],entry['candidate_eligibility']):
            raise ValueError('Only exact eligible original128 GNN argmax raw payload is admitted')
        store=self._get_raw_store()
        candidate=store.load_candidate(str(entry['selected_payload'][0]),str(entry['selected_payload_sha256'][0]))
        case=store.load_case(str(entry['raw_case_reference'][0]),str(entry['raw_case_reference_sha256'][0]))
        recipient=str(entry['case_id'][0])
        if (case['metadata'].get('case_id')!=recipient or candidate.get('case_id')!=recipient
                or candidate.get('source_component')!=int(entry['source_component'][0])
                or candidate.get('case_reference_sha256')!=str(entry['raw_case_reference_sha256'][0])
                or not np.array_equal(candidate.get('raw_target_center'),entry['candidate_raw_centers'][selected])):
            raise ValueError('Native CT/mask paste recipient/source/grid differs')
        return case,candidate

    def prepare_raw_verification(self):
        """Verify referenced bytes once without loading any patient arrays.

        Only this trusted in-memory SHA/stat map is inherited by epoch workers.
        Their normal payload loader still checks every stat and every tensor's
        shape, dtype, recipient, component and selected center.
        """
        index_digest=sha(self.index_path)
        if read(self.index_path)!=self.metadata:
            raise ValueError('Frozen bank index changed before raw verification')
        store=self._get_raw_store()
        if store._cases or store._sources:
            raise ValueError('Raw verification must precede runtime array loading')
        for case_id,names in sorted(self.entries_by_case.items()):
            for name in names:
                entry=self._load(name)
                if str(entry['case_id'][0])!=case_id:
                    raise ValueError('Raw witness recipient differs from bank mapping')
                for reference,digest,kind in (
                        ('raw_case_reference','raw_case_reference_sha256','case'),
                        ('selected_payload','selected_payload_sha256','candidate')):
                    path=store._check(str(entry[reference][0]),str(entry[digest][0]))
                    manifest=read(path)
                    if (manifest.get('format')!='onlinecp_raw_target_payload_v1'
                            or manifest.get('kind')!=kind
                            or not isinstance(manifest.get('arrays'),dict)
                            or not isinstance(manifest.get('tree'),dict)):
                        raise ValueError('Incompatible raw witness manifest')
                    if kind=='candidate':
                        archive=manifest['archive']
                        store._check(archive['path'],archive['sha256'])
                    for spec in manifest['arrays'].values():
                        if kind=='case' or 'path' in spec:
                            store._check(spec['path'],spec['sha256'])
        # Detect files altered during a long full-population streamed audit.
        for relative,digest in tuple(store._witnesses):
            store._check(relative,digest)
        if sha(self.index_path)!=index_digest:
            raise ValueError('Frozen bank index changed during raw verification')
        return dict(root=str(self.root.resolve()),index_sha256=index_digest,
                    witnesses=dict(store._witnesses))


class FrozenV23Loader(nnUNetDataLoaderOnlineCP):
    def __init__(self,*args,bank_path,raw_verification=None,**kwargs):
        super().__init__(*args,bank_path=bank_path,**kwargs)
        self.online_bank=FrozenV23Bank(bank_path)
        if set(self.indices)!=set(self.online_bank.metadata['split']['outer_train']):
            raise ValueError('Native CP loader must retain all105 and exclude outer26')
        if raw_verification is not None:
            self.online_bank.adopt_raw_verification(raw_verification)

    def _make_paste_plan(self,entry,candidate_index,scale,shift_hu,case_id):
        # The parent's five RNG draws, source draw and jitter stay unchanged.
        # Correct its provisional whole-vector argmax at the actual payload
        # boundary, without replacing its sampler or holding mutable state.
        if str(entry['case_id'][0])!=str(case_id):raise ValueError('Admitted CP entry belongs to another recipient')
        selected=eligible_argmax(entry['scores'],entry['candidate_eligibility'])
        if (selected!=int(entry['selected_candidate'][0])
                or str(entry['selection_policy'][0])!=CP_SELECTION):
            raise ValueError('Actual loader selection differs from the admitted eligible raw payload')
        return super()._make_paste_plan(entry,selected,scale,shift_hu,case_id)


class nnUNetTrainer_250epochs_FrozenV23CP(_nnUNetTrainer_250epochs_OnlineCP):
    required_paste_contract='onlinecp_raw_target_paste_v1'
    online_policy='hier_argmax'

    def __init__(self,plans,configuration,fold,dataset_json,device=torch.device('cuda')):
        self.v24_metadata=validate_bank(read(Path(os.environ['ONLINE_CP_BANK'])))
        if configuration!='3d_fullres' or str(fold)!='0':raise ValueError('Exact historical local fold0 required')
        super().__init__(plans,configuration,fold,dataset_json,device)
        if self.configuration_manager.patch_size!=[128,128,128]:
            raise ValueError('Actual full ResEncM128 cube required')
        if self.configuration_manager.batch_size!=self.v24_metadata['baseline']['physical_batch']:
            raise ValueError('Historical physical batch must remain unchanged')

    def do_split(self):
        train,val=super().do_split()
        if set(train)!=set(self.v24_metadata['split']['outer_train']) or set(val)!=set(self.v24_metadata['split']['outer_val']):
            raise ValueError('Native105/26 split differs from frozen GNN CP bank')
        return train,val

    def get_dataloaders(self):
        receipt=getattr(self,'_v24_raw_verification_receipt',None)
        if receipt is None:
            bank=FrozenV23Bank(self.online_bank_path)
            started=time.perf_counter()
            receipt=bank.prepare_raw_verification()
            self._v24_raw_verification_receipt=receipt
            self.print_to_log_file('[FrozenV23CP] startup raw SHA/stat verification '
                f'files={len(receipt["witnesses"])} seconds={time.perf_counter()-started:.3f} '
                'patient_arrays_loaded=False; epoch workers inherit metadata witnesses',
                also_print_to_console=True)
        def verified_loader(*args,bank_path,**kwargs):
            return FrozenV23Loader(*args,bank_path=bank_path,raw_verification=receipt,**kwargs)
        original=_nnUNetTrainer_250epochs_OnlineCP.get_dataloaders
        namespace=dict(original.__globals__,OnlineCPBank=FrozenV23Bank,nnUNetDataLoaderOnlineCP=verified_loader)
        # The original validation nnUNetDataLoader factory is untouched. No
        # scorer, donor annotation, CP service or CP transform enters val.
        method=types.FunctionType(original.__code__,namespace,original.__name__,original.__defaults__,original.__closure__)
        return method(self)

    def initialize(self):
        require_project_budget();torch.set_num_threads(1)
        result=super().initialize()
        torch.set_num_threads(1);require_project_budget()
        from hiercp.preparation_runtime import snapshot
        self.print_to_log_file(json.dumps(dict(pipeline=FORMAT,epochs=self.num_epochs,
            physical_batch=self.batch_size,effective_batch=self.batch_size,gradient_accumulation=1,
            patch_size=self.configuration_manager.patch_size,train105=True,val26=True,
            cp_probability=.5,candidates=128,frozen_GNN_parameters=10434532,
            donor_policy_matches_historical_Basic=False,validation_CP=False,inference_GT=False,
            segmentation_parameters=sum(p.numel() for p in self.network.parameters()),
            resource_snapshot=snapshot(),project_resource_contract=resource_contract(),debug=False)))
        return result

    def train_step(self,batch):
        before=require_project_budget();started=time.perf_counter()
        shape=list(batch['data'].shape)
        output=super().train_step(batch)
        after=require_project_budget()
        count=getattr(self,'_v24_resource_steps',0)+1;self._v24_resource_steps=count
        if count<=3 or count%25==0:
            record=dict(format='v24_native_CP_actual_training_resources_v1',epoch=int(self.current_epoch),
                step=count,input_shape=shape,physical_batch=int(self.batch_size),
                native_step_wall_seconds=time.perf_counter()-started,
                step_includes_original_native_forward_backward_optimizer_and_loss_CPU_transfer=True,
                CUDA_allocated_bytes=torch.cuda.memory_allocated(),
                CUDA_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                CUDA_peak_semantics='cumulative maximum since last explicit allocator reset',
                before=before,after=after,original_model_and_native_loss_unchanged=True)
            with (Path(self.output_folder)/'v24_resource_steps.jsonl').open('a',encoding='utf8') as stream:
                stream.write(json.dumps(record,allow_nan=False)+'\n')
        return output

    def _selection_name(self):return 'frozen-actual-best-v23-P-plus128U-exact-eligible-argmax'
