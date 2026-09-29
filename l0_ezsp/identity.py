"""New experiment identity; legacy exact resume is explicitly incompatible."""
from . import ENCODER_ID
from .config import fingerprint

def identity(profile):
    return {'encoder_id':ENCODER_ID,'profile_sha256':fingerprint(profile),'feature_coordinates':'stride4',
        'training_ready':False,'support_memory_policy':'rebuild_with_new_encoder','cluster_plan_policy':'refit_with_new_encoder'}

def require_same_experiment(saved,profile):
    if saved.get('l0_ezsp_identity')!=identity(profile):
        raise ValueError('Not an exact EZ-SP resume: old L0 checkpoints/memory/teacher plans cannot be relabelled')

def load_cnn_only(encoder,state):
    """Explicit CNN-only initialization; never loads GAT/L1/L2/optimizer/memory."""
    prefix='local.dense_encoder.'
    selected={k[len(prefix):]:v for k,v in state.items() if k.startswith(prefix)}
    if not selected:raise ValueError('No explicit local.dense_encoder weights')
    encoder.dense_encoder.load_state_dict(selected,strict=True)
