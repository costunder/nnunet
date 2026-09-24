"""Production checkpoint scorer awaits a complete training objective; L2 forward scoring is restored in model.py."""
from .contracts import require_training_objective

class Scorer:
    def __init__(self,*args,**kwargs):
        require_training_objective()
