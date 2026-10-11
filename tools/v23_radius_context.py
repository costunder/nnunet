"""User-corrected <=10mm real-tissue context for frozen BEST CP inference.

The archived model configuration is retained as checkpoint history. Effective
context eligibility has no inner-radius or organ-depth exclusion. Liver depth
remains an actual input feature, and the separate liver-surface role remains.
Nodes in different semantic roles may share physical coordinates. No learned
operator, parameter, tensor channel, sampling budget, or archived file changes.
This is corrected-input inference; it does not retrain the completed GNN.
"""
from __future__ import annotations
import ast
import functools
import hashlib
import inspect
from pathlib import Path

FORMAT = 'v23_matched_context_upper_radius_only_v1'
_INSTALLED = None


def identity():
    return dict(format=FORMAT, module_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='source and target; every historical source and candidate',
        eligibility='actual organ & outside complete donor footprint & tumor distance <=10mm',
        effective_context_inner_radius_mm=0., effective_context_outer_radius_mm=10.,
        organ_depth_exclusion=False, cross_role_coordinate_overlap=True,
        same_role_duplicates_allowed=False, original_config_inner_radius_is_checkpoint_history=True,
        organ_depth_feature='unchanged actual depth',
        empty_context_fallback=False, GNN_weights='unchanged frozen BEST',
        limitation='neural input geometry differs from historical GNN training; GNN not retrained')


def _context_builder(node):
    count = 0
    for child in ast.walk(node):
        if (isinstance(child, ast.Assign) and len(child.targets) == 1
                and isinstance(child.targets[0], ast.Name) and child.targets[0].id == 'context_mask'):
            expected = ast.parse('organ & ~footprint & (outside >= inner) & (outside <= outer) & (liver_depth > boundary_depth + separation)',mode='eval').body
            if ast.dump(child.value,include_attributes=False) != ast.dump(expected,include_attributes=False):
                raise ValueError('Archived context selection expression differs')
            child.value = ast.parse('organ & ~footprint & (outside <= outer)',mode='eval').body
            count += 1
    return count


def _context_validator(node):
    counts = dict(inner=0,depth=0,overlap=0)
    class Correct(ast.NodeTransformer):
        def visit_Compare(self,child):
            expected = ast.parse('distance < inner - 1e-4',mode='eval').body
            if ast.dump(child,include_attributes=False) == ast.dump(expected,include_attributes=False):
                counts['inner'] += 1
                return ast.copy_location(ast.parse('distance < -1e-4',mode='eval').body,child)
            return self.generic_visit(child)
        def visit_If(self,child):
            depth = ast.parse('bool(np.any(liver_depth[tuple(context.T)] <= boundary + separation - 1e-4))',mode='eval').body
            overlap = ast.parse('any(tuple(int(v) for v in row) in context_rows for row in liver)',mode='eval').body
            for key,expected in [('depth',depth),('overlap',overlap)]:
                if ast.dump(child.test,include_attributes=False) == ast.dump(expected,include_attributes=False):
                    counts[key] += 1
                    return None
            return self.generic_visit(child)
    Correct().visit(node)
    if counts != dict(inner=1,depth=1,overlap=1):
        raise ValueError('Archived radius/depth/overlap validator changed: '+str(counts))
    return sum(counts.values())


def install(runtime):
    """Bind only the exact isolated original runtime; production opts in explicitly."""
    global _INSTALLED
    from hiercp_v1x import bounded_scope as scope
    spatial = runtime['spatial']
    contract = runtime['scope']['contract_sha256']
    if float(runtime['scope']['margin_mm']) != 10.:
        raise ValueError('Corrected context preserves the explicit10mm maximum scope')
    if _INSTALLED is not None:
        if _INSTALLED != contract:
            raise RuntimeError('Corrected context cannot change archive or scope in one process')
        return identity()
    previous_validator = spatial.validate_canonical_coordinates
    def validator_policy(node):
        count=_context_validator(node)
        optional=scope._allow_empty_coordinates(node)
        if optional!=1:
            raise ValueError('Original optional liver-surface coordinate guard changed')
        return count+optional
    corrected_validator = scope._patched(inspect.unwrap(previous_validator),validator_policy,4)
    previous_builder = spatial.canonical_coordinate_sets
    corrected_builder = scope._patched(inspect.unwrap(previous_builder),_context_builder,1)

    @functools.wraps(previous_validator)
    def validate(fields,points,config):
        scope._scope_config(config,10.)
        # Genuine zero-tissue inputs keep the original explicit absence/error
        # path. No empty token is substituted for the real tissue excluded by
        # the old depth/inner filters.
        if len(points['context']) == 0:
            return previous_validator(fields,points,config)
        corrected_validator(fields,points,config)
        if len(points['liver_surface'])==0:
            import numpy as np
            organ=np.asarray(spatial._field(fields,'organ_mask','organ'),dtype=bool)
            footprint=np.asarray(spatial._field(fields,'footprint'),dtype=bool)
            depth=np.asarray(spatial._field(fields,'organ_depth','organ_boundary_mm','liver_depth_mm'))
            if np.any(organ&~footprint&(depth>0.)&(depth<=float(config.boundary_depth_mm))):
                raise ValueError('A present real liver surface cannot be silently omitted')

    scope._rebind(previous_validator,validate)
    scope._rebind(previous_builder,corrected_builder)
    _INSTALLED = contract
    return identity()
