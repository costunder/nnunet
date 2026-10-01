"""Explicit warm-start controls for the diagnostic official-operator L1.

The official operator, graph topology, model dimensions, loss, and data are
unchanged.  Only parameter initialization varies.  The historical
``clone_reference`` and its recorded artifacts remain unmodified.

An affine change of relation coordinates preserves the old *linear projected*
T/F/U embeddings, not the full layer function: ReLU attention, key scaling,
self-loops, per-edge projection, fresh BatchNorm, and GELU still differ.
"""
import copy

import torch
from torch.nn import functional as F

from hiercp_v222.v1_execution import rng_state, restore_rng
from l0_regions.training import hash_state
from tools.local_cnn_reference_l1 import clone_reference


POLICIES = ('raw_columns', 'affine_relations',
            'affine_relations_zero_out_bias')


def _gradients(net):
    return {name: parameter.grad for name, parameter in net.named_parameters()}


@torch.no_grad()
def _relation_audit(old, new, policy, index):
    old_codes = old.edge.weight.new_tensor([[1, 1], [1, 0], [0, 0]])
    new_codes = new.lin_edge.weight.new_tensor([[0, 1], [0, -1], [1, 0]])
    expected = F.linear(old_codes, old.edge.weight, old.edge.bias)
    actual = F.linear(new_codes, new.lin_edge.weight, new.lin_edge.bias)
    error = (actual - expected).abs().amax(dim=1)
    scale = max(1., float(expected.detach().abs().max().cpu()))
    tolerance = 16 * torch.finfo(expected.dtype).eps * scale
    errors = dict(zip(('T', 'F', 'U'), error.detach().cpu().tolist()))
    preserves = policy != 'raw_columns'
    if (not bool(torch.isfinite(actual).all())
            or (preserves and float(error.max().cpu()) > tolerance)):
        raise AssertionError('Relation-coordinate transfer did not preserve finite T/F/U projections')
    if preserves:
        equations = dict(
            old='a = W_old[:,0]; b = W_old[:,1]',
            new_column_0='-a - b/2', new_column_1='b/2',
            new_bias='a + b/2',
            T='W_new @ [0,1] + bias_new = a+b = W_old @ [1,1]',
            F='W_new @ [0,-1] + bias_new = a = W_old @ [1,0]',
            U='W_new @ [1,0] + bias_new = 0 = W_old @ [0,0]')
    else:
        equations = dict(new_weight='W_old', new_bias='0',
                         T='b', F='-b', U='a',
                         warning='Raw column copying changes the T/F/U projected meaning')
    self_projection = F.linear(new.lin_edge.weight.new_zeros(1, 2),
                               new.lin_edge.weight, new.lin_edge.bias)[0]
    return dict(layer=index+1,
                preserved_projected_relations=preserves,
                equations=equations, projected_TFU_max_abs_error=errors,
                floating_absolute_tolerance=tolerance,
                comparison_dtype=str(expected.dtype),
                self_projection=dict(code=[0, 0],
                    equation='bias_new', norm=float(self_projection.detach().norm().cpu()),
                    old_counterpart=False,
                    interpretation='A new self-loop; preservation claims apply only to old T/F/U edges'),
                full_layer_function_preserved=False)


def clone_reference_control(net, *, policy):
    """Return an isolated official-operator candidate and explicit transfer audit.

``policy`` is mandatory; there is no silent preferred initialization.
``raw_columns`` exactly retains the historical candidate initialization.
``affine_relations`` preserves old projected T/F/U, with old output bias still
copied to the official per-edge location.  The third policy additionally zeros
that bias and explicitly loses the old once-per-node output-bias contribution.
Fresh BN is left fresh in every policy; this function never calibrates it.
"""
    if policy not in POLICIES:
        raise ValueError(f'Explicit reference transfer policy required; choose from {POLICIES}')
    caller_rng = rng_state()
    original_state = hash_state(net.state_dict())
    original_gradients = hash_state(_gradients(net))
    original_modes = tuple(module.training for module in net.modules())
    try:
        candidate, base_metadata = clone_reference(net)
        metadata = copy.deepcopy(base_metadata)
        audits, bias_audits = [], []
        for index, (old, new) in enumerate(zip(net.l1, candidate.l1)):
            if old.edge.bias is not None:
                raise ValueError('Current old relation projection must be bias-free; no inferred affine fallback')
            with torch.no_grad():
                if policy != 'raw_columns':
                    a, b = old.edge.weight[:, 0], old.edge.weight[:, 1]
                    new.lin_edge.weight[:, 0].copy_(-a - b/2)
                    new.lin_edge.weight[:, 1].copy_(b/2)
                    new.lin_edge.bias.copy_(a + b/2)
                if policy == 'affine_relations_zero_out_bias':
                    new.out_proj.bias.zero_()
            audits.append(_relation_audit(old, new, policy, index))
            zero = policy == 'affine_relations_zero_out_bias'
            bias_audits.append(dict(layer=index+1,
                old_output_bias_norm=float(old.update.out.bias.detach().norm().cpu()),
                new_output_bias_norm=float(new.out_proj.bias.detach().norm().cpu()),
                old_placement='once after aggregation, per node',
                new_placement='once per incoming edge, before aggregation',
                policy='explicit_zero' if zero else 'copy_old_bias',
                official_bias_component='0' if zero else 'incoming_degree * b_old',
                shift_relative_old_once_per_node_bias='-b_old' if zero else '(incoming_degree - 1) * b_old',
                old_bias_parameter_copied=not zero,
                old_once_per_node_bias_effect_preserved=False,
                full_node_bias_effect_preserved=False,
                full_layer_function_preserved=False))
            prefix = f'l1.{index}.'
            transfer = metadata['transfers'][index]
            if policy != 'raw_columns':
                transfer['copied'].pop(prefix+'lin_edge.weight')
                transfer['new_zero_bias'].remove(prefix+'lin_edge.bias')
                transfer['affine_mapped'] = {
                    prefix+'lin_edge.weight': prefix+'edge.weight [explicit affine T/F/U remap]',
                    prefix+'lin_edge.bias': prefix+'edge.weight[:,0] + edge.weight[:,1]/2'}
            if zero:
                transfer['copied'].pop(prefix+'out_proj.bias')
                transfer['new_zero_bias'].append(prefix+'out_proj.bias')
        metadata.update(transfer_policy=policy, relation_transfer=audits,
                        out_bias_transfer=bias_audits,
                        operator_equations_unchanged_from_reference=True,
                        historical_clone_reference_unchanged=True,
                        full_layer_function_preserved=False,
                        batch_norm_calibrated=False,
                        remaining_operator_confounds=[
                            'Key divided by sqrt(head_dim); copied attention key weights are not rescaled',
                            'ReLU instead of old SiLU/LeakyReLU attention transformations',
                            'New self-loop and value-only messages',
                            'Per-edge output projection and fresh joint BatchNorm',
                            'Old LayerNorm/FFN removed, inter-layer GELU added',
                            'Joint train BN can route support alignment gradients to query features'],
                        L0_L2_initial_weights_and_loss_formulas_preserved=True,
                        gradient_dependency_equivalence=False)
        candidate.reference_transfer = metadata
        if (hash_state(net.state_dict()) != original_state
                or hash_state(_gradients(net)) != original_gradients
                or tuple(module.training for module in net.modules()) != original_modes):
            raise AssertionError('Reference transfer control modified the original state, gradients, or modes')
        return candidate, metadata
    finally:
        restore_rng(caller_rng)
