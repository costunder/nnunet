"""On-demand CP bank access for parallel native augmentation workers.

Workers request identities over authenticated loopback IPC; raw volumes never
cross IPC. The training process owns the frozen scorer. A shared lock keeps its
CUDA inference out of concurrent segmentation forward/backward/validation.
"""
import json
import os
import threading
import traceback
import types
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.connection import Listener,Client
from pathlib import Path
import numpy as np
from custom_trainers.nnUNetTrainer_OnlinePairedCP import OnlineCPBank,nnUNetDataLoaderOnlineCP
from .bank import EntryBuilder,validate_catalog,entry_name
from .contracts import read_json,sha,write_new

class MaterializationService:
    def __init__(self,index_path,gpu_lock):
        self.builder=EntryBuilder(index_path,gpu_lock)
        self.key=os.urandom(32); self.listener=Listener(('127.0.0.1',0),authkey=self.key)
        self.stopping=threading.Event()
        self.executor=ThreadPoolExecutor(max_workers=os.cpu_count() or 1,thread_name_prefix='v21-cp-event')
        os.environ['V21_CP_RPC_ADDRESS']=json.dumps(self.listener.address)
        os.environ['V21_CP_RPC_AUTH']=self.key.hex()
        self.thread=threading.Thread(target=self._serve,name='v21-cp-dispatch',daemon=True); self.thread.start()

    def _serve(self):
        while not self.stopping.is_set():
            connection=self.listener.accept()
            if self.stopping.is_set():connection.close();break
            self.executor.submit(self._respond,connection)

    def _respond(self,connection):
        try:
            request=connection.recv()
            if not isinstance(request,dict) or set(request)!={'recipient','donor_index'}:raise ValueError('Invalid CP request')
            response=self.builder.materialize(**request)
            connection.send(dict(ok=True,**response))
        except Exception:
            error=traceback.format_exc()
            import uuid
            write_new(self.builder.root/f'failures/{uuid.uuid4().hex}.json',{'error':error})
            connection.send({'ok':False,'error':error})
        finally:connection.close()

    def close(self):
        self.stopping.set()
        with Client(self.listener.address,authkey=self.key):pass
        self.thread.join(); self.executor.shutdown(wait=True); self.listener.close()

def request_entry(recipient,donor_index):
    address=tuple(json.loads(os.environ['V21_CP_RPC_ADDRESS']))
    with Client(address,authkey=bytes.fromhex(os.environ['V21_CP_RPC_AUTH'])) as connection:
        connection.send(dict(recipient=recipient,donor_index=donor_index)); response=connection.recv()
    if not response['ok']:raise RuntimeError('CP materialization failed; no fallback: '+response['error'])
    return response

class SharedDonorBank(OnlineCPBank):
    def __init__(self,index_path,cache_entries=64):
        super().__init__(index_path,cache_entries)
        validate_catalog(self.metadata)
        self.identities={entry_name(c,i):(c,i) for c in self.metadata['split']['outer_train']
                         for i in range(len(self.metadata['donor_pool']))}

    def _validate_raw_entry(self,entry,path):
        required={'paste_contract','case_id','donor_case_id','donor_component_id','candidate_centers',
            'candidate_raw_centers','scores','source_component','source_diameter_mm','selected_candidate',
            'selected_payload','selected_payload_sha256','raw_case_reference','raw_case_reference_sha256'}
        if not required.issubset(entry):raise ValueError('Incomplete selected-paste entry')
        for key in ('candidate_centers','candidate_raw_centers'):
            if entry[key].shape!=(128,3) or entry[key].dtype.kind not in 'iu':raise ValueError('Candidate geometry changed')
        if np.unique(entry['candidate_raw_centers'],axis=0).shape[0]!=128:raise ValueError('Duplicate candidate centers')
        if entry['scores'].shape!=(128,) or not np.isfinite(entry['scores']).all():raise ValueError('Invalid full-pool scores')
        for key in ('selected_candidate','source_component','donor_component_id'):
            if entry[key].shape!=(1,) or entry[key].dtype.kind not in 'iu':raise ValueError('Invalid selected-paste identity')
        if (int(entry['selected_candidate'][0])!=int(np.argmax(entry['scores']))
                or int(entry['source_component'][0])<1 or int(entry['donor_component_id'][0])<1):
            raise ValueError('Selected raw payload must be exact argmax of all 128 scores')
        if entry['source_diameter_mm'].shape!=(1,) or not 0<float(entry['source_diameter_mm'][0])<=20:
            raise ValueError('Invalid small donor diameter')
        for key in ('paste_contract','case_id','donor_case_id','selected_payload','selected_payload_sha256',
                    'raw_case_reference','raw_case_reference_sha256'):
            if entry[key].shape!=(1,) or entry[key].dtype.kind!='U' or not str(entry[key][0]):raise ValueError('Invalid paste reference')
            if key.endswith('sha256') and (len(str(entry[key][0]))!=64 or any(c not in '0123456789abcdef' for c in str(entry[key][0]))):
                raise ValueError('Invalid paste reference hash')
        if str(entry['paste_contract'][0])!=self.paste_contract:raise ValueError('Paste contract changed')

    def load_raw_candidate(self,entry,candidate_index):
        if isinstance(candidate_index,(bool,np.bool_)) or not isinstance(candidate_index,(int,np.integer)):
            raise ValueError('Invalid candidate index')
        selected=int(entry['selected_candidate'][0])
        if candidate_index!=selected or selected!=int(np.argmax(entry['scores'])):
            raise ValueError('Only the fully scored exact argmax raw paste is materialized')
        store=self._get_raw_store()
        candidate=store.load_candidate(str(entry['selected_payload'][0]),str(entry['selected_payload_sha256'][0]))
        case=store.load_case(str(entry['raw_case_reference'][0]),str(entry['raw_case_reference_sha256'][0]))
        recipient=str(entry['case_id'][0])
        if (case['metadata'].get('case_id')!=recipient or candidate.get('case_id')!=recipient
                or candidate.get('source_component')!=int(entry['source_component'][0])
                or candidate.get('case_reference_sha256')!=str(entry['raw_case_reference_sha256'][0])
                or not np.array_equal(candidate.get('raw_target_center'),entry['candidate_raw_centers'][selected])):
            raise ValueError('Selected raw payload recipient/source/geometry provenance mismatch')
        return case,candidate

    def _load(self,relative_path):
        if relative_path not in self.identities:raise ValueError('Entry outside training recipient/donor catalog')
        recipient,index=self.identities[relative_path]
        receipt=self.root/(relative_path+'.receipt.json')
        if not receipt.exists():
            result=request_entry(recipient,index)
            if result['relative']!=relative_path:raise ValueError('CP service returned another pair')
        ready=read_json(receipt)
        donor=self.metadata['donor_pool'][index]
        if (ready['catalog_sha256']!=sha(self.index_path) or ready['recipient']!=recipient or ready['donor']!=donor
                or sha(self.root/relative_path)!=ready['entry_sha256']):
            raise ValueError('CP donor/recipient provenance or entry SHA changed')
        entry=super()._load(relative_path)
        if (str(entry['case_id'][0])!=recipient or str(entry['donor_case_id'][0])!=donor['case_id']
                or int(entry['donor_component_id'][0])!=donor['component_id']):
            raise ValueError('CP payload donor/recipient identity differs from catalog')
        return entry

class SharedDonorLoader(nnUNetDataLoaderOnlineCP):
    def __init__(self,*args,bank_path,**kwargs):
        super().__init__(*args,bank_path=bank_path,**kwargs)
        self.online_bank=SharedDonorBank(bank_path)
        if set(self.indices)!=set(self.online_bank.metadata['split']['outer_train']):
            raise ValueError('Native training loader contains held-out/missing recipients')

def native_dataloaders(trainer,parent_method):
    """Reuse the verified raw-paste loader wiring without mutating v1 globals.

    Only the training loader/bank factories change. The validation loader stays
    the original nnUNetDataLoader with no CP bank, request service or donor RNG.
    """
    namespace=dict(parent_method.__globals__,OnlineCPBank=SharedDonorBank,nnUNetDataLoaderOnlineCP=SharedDonorLoader)
    method=types.FunctionType(parent_method.__code__,namespace,parent_method.__name__,
                              parent_method.__defaults__,parent_method.__closure__)
    return method(trainer)
