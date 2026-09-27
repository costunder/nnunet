"""Native workers consume verified ranking selections through the existing RPC."""
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.connection import Listener
from pathlib import Path
import numpy as np
from custom_trainers.nnUNetTrainer_OnlinePairedCP import OnlineCPBank, nnUNetDataLoaderOnlineCP
from comparison_randomness import PairedLoaderMixin, paired_dataloaders
from hiercp_v222.native_adapter import MaterializationService, request_entry
from hiercp_v222.bank import entry_name
from hiercp_v222.contracts import read_json, sha
from tools.v22_online_rank_bank import RankedEntryBuilder, validate_catalog
from tools.v22_online_selection import RankedEventLoaderMixin, validate_selection


class RankedMaterializationService(MaterializationService):
    def __init__(self, index_path, gpu_lock):
        self.builder=RankedEntryBuilder(index_path,gpu_lock)
        self.key=os.urandom(32);self.listener=Listener(('127.0.0.1',0),authkey=self.key)
        self.stopping=threading.Event()
        self.executor=ThreadPoolExecutor(max_workers=os.cpu_count() or 1,thread_name_prefix='rank-cp-event')
        os.environ['V222_CP_RPC_ADDRESS']=json.dumps(self.listener.address)
        os.environ['V222_CP_RPC_AUTH']=self.key.hex()
        self.thread=threading.Thread(target=self._serve,name='rank-cp-dispatch',daemon=True)
        self.thread.start()


class RankedBank(OnlineCPBank):
    def __init__(self,index_path,cache_entries=64):
        super().__init__(index_path,cache_entries)
        validate_catalog(self.metadata)
        self.identities={entry_name(c,i):(c,i) for c in self.metadata['split']['outer_train']
                         for i in range(len(self.metadata['donor_pool']))}

    @staticmethod
    def selected_index(entry):
        if entry['_receipt']['status']=='self_patient_no_placement':
            return None
        return validate_selection(entry['_receipt']['selection'])

    def _load(self,relative_path):
        if relative_path not in self.identities:raise ValueError('Entry outside ranking catalog')
        recipient,index=self.identities[relative_path];donor=self.metadata['donor_pool'][index]
        receipt=self.root/(relative_path+'.receipt.json')
        if not receipt.exists():
            response=request_entry(recipient,index)
            if response['relative']!=relative_path:raise ValueError('RPC returned a different event')
        value=read_json(receipt)
        if (value['catalog_sha256']!=sha(self.index_path) or value['recipient']!=recipient
                or value['donor']!=donor or value['relative']!=relative_path):
            raise ValueError('Event catalog/recipient/donor mismatch')
        identities=self.metadata['identities']['cases']
        same=identities[recipient]['patient_group']==identities[donor['case_id']]['patient_group']
        if value['status']=='self_patient_no_placement':
            if not same or value['selection'] is not None:raise ValueError('Invalid self-patient no-placement')
            return {'_receipt':value}
        if value['status']!='ranked' or same:raise ValueError('Invalid ranking event status')
        selection=value['selection'];selected=validate_selection(selection)
        if (selection['recipient']!=recipient or selection['donor']!=donor['case_id']
                or selection['component']!=donor['component_id'] or len(selection['scores'])!=self.candidate_count):
            raise ValueError('Selection event identity or complete pool mismatch')
        entry={'_receipt':value}
        if selected is None:return entry
        placement=value['placement']
        if (placement['recipient']!=recipient or placement['donor']!=donor['case_id']
                or placement['component']!=donor['component_id'] or placement['center']!=selection['centers'][selected]):
            raise ValueError('Selected placement differs from ranked event')
        mapped=np.asarray(value['native_centers'])
        if mapped.shape!=(self.candidate_count,3) or mapped.dtype.kind not in 'iu':
            raise ValueError('Complete integer native centers required')
        entry.update(case_id=np.asarray([recipient]),source_component=np.asarray([index+1]),
            candidate_centers=mapped,candidate_raw_centers=np.asarray(selection['centers']),
            scores=np.asarray(selection['scores'],dtype=np.float64),selected_candidate=np.asarray([selected]))
        return entry

    def load_raw_candidate(self,entry,candidate_index):
        selected=self.selected_index(entry)
        if (isinstance(candidate_index,(bool,np.bool_)) or not isinstance(candidate_index,(int,np.integer))
                or selected is None or candidate_index!=selected):
            raise ValueError('Only the verified eligible selection may be pasted')
        value=entry['_receipt'];store=self._get_raw_store()
        case=store.load_case(value['raw_case_reference'],value['raw_case_reference_sha256'])
        candidate=store.load_candidate(value['selected_payload'],value['selected_payload_sha256'])
        p=value['placement'];meta=case['metadata']
        if (meta['case_id']!=p['recipient'] or candidate['case_id']!=p['recipient']
                or candidate['source_component']!=int(entry['source_component'][0])
                or candidate['case_reference_sha256']!=value['raw_case_reference_sha256']
                or candidate['raw_target_center']!=p['center']):
            raise ValueError('Selected native payload identity mismatch')
        # Reverse the actual native axis permutation, then compare complete CT,
        # mask and origin against the PlacementSpec used to score and filter.
        from hiercp_v222.placement import array_hash
        permutation=np.asarray(meta['transpose_forward'])
        inverse=np.argsort(permutation)
        mask=candidate['source_mask'].transpose(tuple(inverse)).transpose(2,1,0)
        image=candidate['source_ct'][0].transpose(tuple(inverse)).transpose(2,1,0)
        expected_origin=(np.asarray(p['center'])-np.asarray(p['anchor']))[::-1][permutation]-np.asarray(meta['crop_bbox'])[:,0]
        if (array_hash(mask)!=p['mask_sha256'] or array_hash(image)!=p['image_sha256']
                or not np.array_equal(candidate['target_input_origin'],expected_origin)):
            raise ValueError('Native paste changed ranked CT/mask/anchor geometry')
        return case,candidate


class RankedLoader(RankedEventLoaderMixin,PairedLoaderMixin,nnUNetDataLoaderOnlineCP):
    def __init__(self,*args,bank_path,**kwargs):
        super().__init__(*args,bank_path=bank_path,**kwargs)
        self.online_bank=RankedBank(bank_path)
        if set(self.indices)!=set(self.online_bank.metadata['split']['outer_train']):
            raise ValueError('Native loader must cover exactly outer training')


def native_dataloaders(trainer):
    return paired_dataloaders(trainer,RankedLoader,dict(bank_path=trainer.online_bank_path,
        policy=trainer.online_policy,online_seed=trainer.online_seed))
