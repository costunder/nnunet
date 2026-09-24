"""Cross-patient observed-positive compatibility; no query pseudo-GT."""
from pathlib import Path
import numpy as np
import torch
from .contracts import validate_checkpoint
from .model import PromptGraphModel,context_descriptor
from .training import require_device,loader,calibrate,calibrate_workers,attach_context
from .data import LocalDataset,MemoryLocalDataset

class Scorer:
    def __init__(self,checkpoint,device='cuda'):
        self.device=require_device(device)
        self.payload=validate_checkpoint(torch.load(checkpoint,map_location='cpu',weights_only=False))
        self.cfg=self.payload['config']; self.base=self.payload['base_config']
        self.model=PromptGraphModel(self.cfg,self.base,patient_ids=self.payload['patient_ids']).to(self.device)
        self.model.load_state_dict(self.payload['state_dict'],strict=True); self.model.eval()
        self.memory={k:v.to(self.device) if torch.is_tensor(v) else v for k,v in self.payload['memory'].items()}

    @torch.no_grad()
    def score(self,root,source_row,candidate_rows,case_id):
        if len(candidate_rows)!=self.cfg['candidate_count']: raise ValueError('Complete 128-candidate pool required')
        if case_id not in self.payload['split']['outer_train']:raise ValueError('Held-out recipients cannot enter augmentation scoring')
        rows=([source_row] if source_row is not None else [])+list(candidate_rows); dataset=LocalDataset(root,rows)
        return self._score_dataset(dataset,case_id,1 if source_row is not None else 0)

    @torch.no_grad()
    def score_records(self,records,case_id):
        if len(records)!=self.cfg['candidate_count']:raise ValueError('Complete 128-candidate pool required')
        if case_id not in self.payload['split']['outer_train']:raise ValueError('Held-out augmentation recipient')
        return self._score_dataset(MemoryLocalDataset(records,case_id),case_id,0)

    @torch.no_grad()
    def _score_dataset(self,dataset,case_id,offset):
        batch,encoder_report=calibrate(self.model,dataset,self.device,self.cfg,inference_only=True)
        workers,worker_report=calibrate_workers(dataset,batch,self.cfg)
        encoded=[]; descriptors=[]
        for item in loader(dataset,batch,workers):
            item=item.to(self.device)
            with torch.autocast('cuda',dtype=torch.bfloat16): encoded.append(self.model.encode_local(item).float())
            descriptors.append(context_descriptor(item))
        del item
        encoded=torch.cat(encoded); descriptors=torch.cat(descriptors)
        # Complete target context is fixed across query batches; none becomes T.
        memory,owner=attach_context(self.memory,encoded,descriptors,case_id)
        def objective(query,indices):
            result=self.model.forward_tasks(memory,query,indices.new_full(indices.shape,owner))
            if not bool(result['available'].all()): raise ValueError('No independent observed-positive donor evidence')
            return result['scores']
        batch,report=calibrate(self.model,dataset,self.device,self.cfg,objective,inference_only=True)
        with torch.autocast('cuda',dtype=torch.bfloat16): state=self.model.prepare_task_state(memory)
        values=[]
        for start in range(0,len(encoded),batch):
            q=encoded[start:start+batch]
            with torch.autocast('cuda',dtype=torch.bfloat16):
                result=self.model.forward_tasks(memory,q,torch.full((len(q),),owner,device=self.device,dtype=torch.long),state=state)
                if not bool(result['available'].all()): raise ValueError('No independent observed-positive donor evidence')
                values.append(result['scores'].float().cpu())
        scores=torch.cat(values)[offset:].numpy()
        if scores.shape!=(128,) or not np.isfinite(scores).all(): raise ValueError('Unavailable or nonfinite compatibility score')
        report.update(encoder_calibration=encoder_report,worker_calibration=worker_report,
            selected_workers=workers,task=case_id,target_evidence='all U; own T withheld',
            external_evidence_patients=[c for c,a in zip(memory['case_ids'],memory['donor_allowed'].cpu().tolist()) if a],
            semantic='placement compatibility, not malignancy probability')
        return scores,report
