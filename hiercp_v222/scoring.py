"""Frozen context scoring for online CP; no target annotation reaches the model."""
import time
import numpy as np
import torch
from .contracts import validate_checkpoint
from .model import PromptGraphModel
from .data import support_for_query
from .training import require_device,batch_admission,execution_budget,release_probe_memory
from .runtime import frozen_scoring_runtime

class Scorer:
    def __init__(self, checkpoint):
        require_device('cuda')
        self.payload = validate_checkpoint(torch.load(checkpoint, map_location='cpu', weights_only=False))
        self.cfg = self.payload['config']; self.base = self.payload['base']
        with frozen_scoring_runtime(self.base,self.cfg['seed']):
            self.model = PromptGraphModel(self.cfg, self.base, self.payload['input_contract']).cuda().eval()
            self.model.load_state_dict(self.payload['state_dict'])
            self.model.requires_grad_(False)
        self.memory = {k: v.cuda() if torch.is_tensor(v) else v for k, v in self.payload['memory'].items()}
        self.batch = None
        self._support_states = {}

    @torch.inference_mode()
    def score_records(self, records, recipient):
        with frozen_scoring_runtime(self.base,self.cfg['seed']):
            return self._score_records(records,recipient)

    def _score_records(self, records, recipient):
        if recipient not in self.payload['split']['outer_train']:
            raise ValueError('CP recipient must be a segmentation-training patient')
        if len(records) != self.cfg['candidate_count'] or any(set(r) != {'patch'} for r in records):
            raise ValueError('128 context-only records required; labels/geometry cannot enter scorer')
        group = self.payload['identities']['cases'][recipient]['patient_group']
        if group not in self._support_states:
            support = support_for_query(self.memory, group)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                state = self.model.prepare_support(*support)
            # The frozen scorer needs label histories/prototype memberships, not
            # a second complete L0 memory for every cached recipient group.
            state['cluster_plan'] = {k:v for k,v in state['cluster_plan'].items()
                                     if k not in ('support_embeddings','owners','classes')}
            self._support_states[group] = state
        state = self._support_states[group]
        patches = torch.from_numpy(np.stack([r['patch'] for r in records]))
        reports = []
        if self.batch is None:
            for count in self.base['generation']['scoring_batch_size_candidates']:
                release_probe_memory(self.model)
                baseline,budget=execution_budget(self.cfg)
                admission=batch_admission(count,reports,baseline,budget,nonlinear_sampler=True)
                if not admission['admitted']:
                    reports.append(dict(batch=count,physical_batch=count,accepted=False,executed=False,
                        reason='predicted_peak_exceeds_VRAM_reserve',**admission))
                    break
                try:
                    torch.cuda.reset_peak_memory_stats(); start = time.perf_counter()
                    for _ in range(self.cfg['batch_calibration_repeats']):
                        with torch.autocast('cuda', dtype=torch.bfloat16):
                            result = self.model.predict_embeddings(self.model.local(patches[:count].cuda()), state)
                        torch.cuda.synchronize()
                    elapsed = time.perf_counter()-start
                    peak = torch.cuda.max_memory_allocated()
                    free, total = torch.cuda.mem_get_info()
                    reports.append(dict(batch=count,physical_batch=count, graphs_per_second=count*self.cfg['batch_calibration_repeats']/elapsed,
                                        peak_vram_bytes=peak, accepted=peak < budget,executed=True,budget_bytes=budget))
                    del result
                    release_probe_memory(self.model)
                    if not reports[-1]['accepted']:break
                except torch.cuda.OutOfMemoryError as error:
                    reports.append(dict(batch=count, accepted=False, error=str(error)))
                    torch.cuda.empty_cache(); break
            valid = [r for r in reports if r['accepted']]
            if not valid: raise MemoryError('No full-context scoring batch fits alongside nnU-Net')
            self.batch = max(valid, key=lambda r: r['graphs_per_second'])['batch']
        scores = []
        for chunk in patches.split(self.batch):
            with torch.autocast('cuda', dtype=torch.bfloat16):
                result = self.model.predict_embeddings(self.model.local(chunk.cuda()), state)
            scores.append(result['ranking_score'].float().cpu())
        scores = torch.cat(scores).numpy()
        if scores.shape != (128,) or not np.isfinite(scores).all(): raise FloatingPointError('Invalid context score')
        return scores, dict(physical_batch=self.batch, calibration=reports, all_128_candidates_scored=True,
                            query_group_excluded_from_support=group, clusters=state['cluster_plan']['audit'],
                            support_patient_groups=sorted(set(self.memory['patient_groups'])-{group}),
                            score_is_calibrated_incidence_probability=False)
