"""Opt-in cost experiment; never a production admission or checkpoint path."""
from dataclasses import dataclass
from time import monotonic
import psutil
import torch
from .encoder import EZSPEncoder

class DiagnosticResourceLimit(RuntimeError):pass

@dataclass
class ResourceBudget:
    cuda_bytes:int
    rss_bytes:int
    seconds:float

    def __post_init__(self):
        if min(self.cuda_bytes,self.rss_bytes,self.seconds)<=0:raise ValueError('Explicit positive diagnostic resource limits required')
        self.started=monotonic()

    def check(self):
        if monotonic()-self.started>self.seconds:raise DiagnosticResourceLimit('Diagnostic wall budget reached at phase boundary')
        if psutil.Process().memory_info().rss>self.rss_bytes:raise DiagnosticResourceLimit('Diagnostic RSS budget reached')
        if torch.cuda.memory_allocated()>self.cuda_bytes:raise DiagnosticResourceLimit('Diagnostic CUDA allocation budget reached')

class DiagnosticEZSPEncoder(EZSPEncoder):
    """Only initial profile limits may be exceeded; all integrity checks remain."""
    def __init__(self,base,profile,*,resource_budget):
        if not isinstance(resource_budget,ResourceBudget):raise TypeError('Explicit resource budget required')
        super().__init__(base,profile);self.resource_budget=resource_budget

    def get_extra_state(self):
        raise RuntimeError('Diagnostic-only state must not be exported as a checkpoint')

    def _admission(self,stats,budget,violations):
        self.resource_budget.check()
        stats['diagnostic_only']=True
        stats['production_admitted']=False
        stats['profile_exceeded']=bool(budget or violations)
        # Structural errors and coverage failures are raised by the shared parent
        # before this hook. This hook changes neither partition nor limits.

    def forward(self,batch):
        self.resource_budget.check()
        output=super().forward(batch)
        if not bool(torch.isfinite(output).all()):raise FloatingPointError('Nonfinite diagnostic output')
        self.resource_budget.check()
        return output
