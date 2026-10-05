"""Fresh crossed models; original B's neural upper, explicit new input contracts.

No old checkpoint is converted into an exact C/D resume. C's single native
view bank has a separate admission policy; it is never called two v1 views.
"""
from __future__ import annotations

from collections import Counter
import copy
import hashlib
from pathlib import Path
import re

import torch
from torch import nn

from .half_b_model import HalfBUpper, _tensor_signature, _require

C_SUPPORT = 'native_single_view_full_original_curriculum_training_cache'


def tensor_hash(state):
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        if isinstance(value, torch.Tensor):
            h.update(key.encode()); h.update(str(value.dtype).encode())
            h.update(str(tuple(value.shape)).encode())
            h.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


class CrossedUpper(HalfBUpper):
    """Reuse B's actual batched upper with a truthful C-only bank contract."""
    def get_extra_state(self):
        return dict(format='crossed_upper_v1', arm=self.crossed_arm, scope_contract=self.scope_contract,
                    debug=self.debug_support, support_policy=(C_SUPPORT if self.crossed_arm=='C' else
                    'native_full_observation_memory_16_patient_episodes_train_full_bank_eval'),
                    checkpoint_support=self.core.checkpoint_support,
                    neural_operator='unchanged_half_B_legacy_prompt_upper')

    def bind_support(self, memory):
        required = {'embeddings','owners','classes','patient_case_ids','sample_ids',
                    'candidate_indices','expected_samples','training_case_ids',
                    'validation_case_ids','manifest_sha256','generation','epoch',
                    'fixed_view_epoch','debug','full_signed_training_cache',
                    'support_policy','native_single_view','original_two_views'}
        if not isinstance(memory, dict) or required - memory.keys():
            raise ValueError('Complete C single-view support provenance required')
        if (memory['support_policy'] != C_SUPPORT or memory['native_single_view'] is not True
                or memory['original_two_views'] is not False or memory['fixed_view_epoch'] != 0
                or memory['debug'] != self.debug_support
                or memory['full_signed_training_cache'] != (not self.debug_support)):
            raise ValueError('C support is genuine native single view, never an original two-view bank')
        patients, train, val = (tuple(memory[k]) for k in
                                ('patient_case_ids','training_case_ids','validation_case_ids'))
        if (len(patients) < 3 or any(len(set(x)) != len(x) for x in (patients,train,val))
                or not set(patients) <= set(train) or set(train) & set(val)
                or not self.debug_support and (len(train) != 84 or len(val) != 21)):
            raise ValueError('Complete disjoint original train/validation support cohort required')
        values = tuple(memory[k] for k in ('embeddings','owners','classes'))
        x, owners, classes = values; n = len(memory['sample_ids'])
        device = next(self.parameters()).device
        if (x.shape != (n,128) or x.dtype != torch.float32 or x.requires_grad
                or any(t.device != device for t in values)
                or any(t.dtype != torch.long or t.shape != (n,) for t in values[1:])):
            raise ValueError('Detached FP32 full C embeddings and exact long owner/class vectors required')
        expected = memory['expected_samples']; indices = memory['candidate_indices']
        if (not isinstance(expected,dict) or set(expected.values()) != set(patients)
                or n != len(expected)*8 or len(indices) != n
                or not re.fullmatch(r'[0-9a-f]{64}',memory['manifest_sha256'])
                or not isinstance(memory['epoch'],int) or not 0 <= memory['epoch'] <= 40
                or not isinstance(memory['generation'],str) or not memory['generation']
                or memory['generation'] in self._seen_generations):
            raise ValueError('C bank cache coverage, generation or digest changed')
        counts = Counter()
        for sample, candidate, (owner, cls) in zip(memory['sample_ids'],indices,
                                                    torch.stack((owners,classes),1).cpu().tolist()):
            if (sample not in expected or type(candidate) is not int or not 0 <= candidate < 8
                    or not 0 <= owner < len(patients) or expected[sample] != patients[owner]
                    or cls != int(candidate == 0)):
                raise ValueError('C source anchor/curriculum GT provenance changed')
            counts[sample,candidate] += 1
        if counts != Counter({(s,i):1 for s in expected for i in range(8)}):
            raise ValueError('Missing or duplicated C support candidate')
        _require(torch.isfinite(x).all(),'Nonfinite C support')
        self._memory = {**copy.deepcopy({k:v for k,v in memory.items() if k not in
                                      ('embeddings','owners','classes')}),
                        'embeddings':x,'owners':owners,'classes':classes,
                        'patient_case_ids':patients,'training_case_ids':train,
                        'validation_case_ids':val}
        self._memory_signatures = tuple(_tensor_signature(v) for v in values)
        self._seen_generations.add(memory['generation']); self._plans = {}; self._episodes = {}
        return self.support_receipt()

    def support_receipt(self):
        memory = self._require_memory()
        return {k:memory[k] for k in ('generation','epoch','manifest_sha256',
                 'support_policy','native_single_view','original_two_views','debug')}


class CrossedModel(nn.Module):
    def __init__(self, local, *, arm, scope_contract, dropout, debug=False):
        super().__init__()
        if arm not in ('C','D'):
            raise ValueError('Explicit crossed arm C or D required')
        self.arm = arm; self.local = local; self.debug = bool(debug)
        # Exactly the CPU initialization used in install_half_b. It leaves
        # query/data/CUDA generators unchanged, and fixes common upper bytes.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(42)
            with torch.device('cpu'):
                upper = CrossedUpper(dropout=dropout,scope_contract=scope_contract,
                                     debug_support=debug)
        self.upper = upper
        self.upper.crossed_arm=arm
        self.upper.core.checkpoint_support=(arm=='C')
        self.initial_upper_sha256 = tensor_hash(upper.core.state_dict())
        self.scope_contract = scope_contract
        self._last_C_generation = None

    @property
    def core(self):
        return self.upper.core

    def get_extra_state(self):
        return dict(format='crossed_C_D_model_v1',arm=self.arm,debug=self.debug,
                    scope_contract=self.scope_contract,initial_upper_sha256=self.initial_upper_sha256)

    def set_extra_state(self,state):
        if state != self.get_extra_state():
            raise ValueError('Checkpoint arm/scope/initial upper mismatch; no architecture resume')
        self._last_C_generation = None; self.upper.clear_support()

    def prepare_support(self,*args,**kwargs):
        return self.core.prepare_support(*args,**kwargs)

    def fit_support_clusters(self,*args,**kwargs):
        return self.core.fit_support_clusters(*args,**kwargs)

    def predict_embeddings(self,*args,**kwargs):
        return self.core.predict_embeddings(*args,**kwargs)

    def score_old(self,embeddings,case_ids,counts,bank):
        if self.arm != 'C':
            raise ValueError('Original 8-candidate score belongs to C')
        if self._last_C_generation != bank['generation']:
            self.upper.bind_support(bank); self._last_C_generation = bank['generation']
        return self.upper.score(embeddings,case_ids,counts)

    def original_ranking_loss(self,scores,difficulties,*,epoch,config):
        from hiercp.loss import CurriculumConfig,curriculum_ranking_loss
        fields = CurriculumConfig.__dataclass_fields__
        training = config.get('training',config)
        # Unspecified original margins come from the preserved original
        # dataclass, exactly as its native runner constructs the objective.
        cfg = CurriculumConfig(**{k:training[k] for k in fields if k in training})
        return curriculum_ranking_loss(scores,difficulties,epoch=epoch,config=cfg)


def full_support_for_case(memory,query_group):
    """Full observation support; exclude query on recipient AND donor side."""
    keep = torch.tensor([query_group not in (r,d) for r,d in
                         zip(memory['row_groups'],memory['donor_groups'])],
                        device=memory['embeddings'].device,dtype=torch.bool)
    ids = keep.nonzero().flatten()
    if not len(ids):
        raise ValueError('Query exclusion leaves no support observations')
    old = memory['owners'][ids]; names = torch.unique(old,sorted=True)
    owners = torch.searchsorted(names,old)
    classes = memory['classes'][ids]
    if len(names)<2 or set(classes.cpu().tolist()) != {0,1}:
        raise ValueError('Excluded support must retain two patients and both observed classes')
    return memory['embeddings'][ids],owners,classes
