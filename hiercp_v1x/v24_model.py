"""GT-blind L1/L2 with unchanged original L0 and learned operator equations.

The explicit authorized changes remove recipient-annotation lesion ownership
and the liver tumor-count input. Removed inputs allocate no unused parameters.
All retained GAT blocks, hidden widths, heads, readouts and L2 are unchanged.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import inspect
import types
from pathlib import Path

FORMAT = 'v24_original_hierarchy_without_recipient_tumor_annotation_v1'
PATIENT_NODE_TYPES = ('tumor', 'candidate', 'region', 'liver')
LIVER_CONTENT_COLUMNS = tuple(i for i in range(14) if i != 10)
_CLASS_CACHE = {}


def model_classes(*, original_snapshot_root=None):
    from hiercp_v1x.transition_v1_local import _runtime
    runtime = _runtime(expected_snapshot_root=original_snapshot_root)
    original = runtime['model']
    identity = runtime['scope']['contract_sha256']
    key = (str(runtime['snapshot']), identity)
    if key in _CLASS_CACHE:
        return _CLASS_CACHE[key]
    namespace = dict(vars(original))
    namespace['PATIENT_NODE_TYPES'] = PATIENT_NODE_TYPES
    namespace['PATIENT_EDGE_TYPES'] = tuple(edge for edge in original.PATIENT_EDGE_TYPES
                                          if 'lesion' not in (edge[0], edge[2]))
    namespace['LIVER_CONTENT_COLUMNS'] = LIVER_CONTENT_COLUMNS
    namespace['_require_content_graph'] = types.FunctionType(original._require_content_graph.__code__,
        namespace, original._require_content_graph.__name__, original._require_content_graph.__defaults__,
        original._require_content_graph.__closure__)
    namespace['_require_content_graph'].__kwdefaults__ = original._require_content_graph.__kwdefaults__
    removed = dict(project=0, projected_node=0, liver_width=0, liver_input=0, context=0)

    class RemoveRecipientGT(ast.NodeTransformer):
        def visit_Assign(self, node):
            if (len(node.targets) == 1 and isinstance(node.targets[0], ast.Attribute)
                    and isinstance(node.targets[0].value, ast.Name)
                    and node.targets[0].value.id == 'self' and node.targets[0].attr == 'lesion_project'):
                removed['project'] += 1
                return None
            if (len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id == 'context_types'
                    and ast.dump(node.value) == ast.dump(ast.parse("('region','lesion','liver')", mode='eval').body)):
                removed['context'] += 1
                node.value = ast.parse("('region','liver')", mode='eval').body
            if (len(node.targets) == 1 and isinstance(node.targets[0], ast.Attribute)
                    and node.targets[0].attr == 'liver_project'):
                calls = [call for call in ast.walk(node.value) if isinstance(call, ast.Call)
                         and isinstance(call.func, ast.Attribute) and call.func.attr == 'Linear']
                if len(calls) != 1 or not isinstance(calls[0].args[0], ast.Name) or calls[0].args[0].id != 'UPPER_RAW_DIM':
                    raise ValueError('Original liver projection changed')
                removed['liver_width'] += 1
                calls[0].args[0] = ast.Constant(len(LIVER_CONTENT_COLUMNS))
            return self.generic_visit(node)

        def visit_Dict(self, node):
            keep = []
            for key, value in zip(node.keys, node.values):
                if isinstance(key, ast.Constant) and key.value == 'lesion':
                    removed['projected_node'] += 1
                else: keep.append((key, value))
            node.keys, node.values = [x[0] for x in keep], [x[1] for x in keep]
            return self.generic_visit(node)

        def visit_Call(self, node):
            if isinstance(node.func, ast.Attribute) and node.func.attr == 'liver_project':
                expected = ast.parse("raw_batch['liver'].raw_x", mode='eval').body
                if len(node.args) != 1 or ast.dump(node.args[0]) != ast.dump(expected):
                    raise ValueError('Original liver forward input changed')
                removed['liver_input'] += 1
                node.args[0] = ast.parse("raw_batch['liver'].raw_x[:, LIVER_CONTENT_COLUMNS]", mode='eval').body
            return self.generic_visit(node)

    source = '\n\n'.join(inspect.getsource(getattr(original, name)) for name in
        ('PatientRegionPyGEncoder', 'HierarchicalPyGPlacementModel'))
    tree = RemoveRecipientGT().visit(ast.parse(source))
    if removed != dict(project=1, projected_node=1, liver_width=1, liver_input=1, context=1):
        raise ValueError('Exactly the authorized five recipient-GT model removals required: ' + str(removed))
    ast.fix_missing_locations(tree)
    exec(compile(tree, original.__file__ + ':v24_recipient_GT_removal', 'exec'), namespace)
    encoder, model = namespace['PatientRegionPyGEncoder'], namespace['HierarchicalPyGPlacementModel']
    model.architecture_version = original.HierarchicalPyGPlacementModel.architecture_version + '|v24_recipient_GT_free'
    receipt = dict(format=FORMAT, original_model_file_sha256=hashlib.sha256(Path(original.__file__).read_bytes()).hexdigest(),
        original_class_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        scope_contract=identity, authorized_removals=removed,
        patient_node_types=list(PATIENT_NODE_TYPES), patient_edge_types=[list(x) for x in namespace['PATIENT_EDGE_TYPES']],
        liver_content_columns=list(LIVER_CONTENT_COLUMNS), recipient_GT_used_in_forward=False,
        original_retained_operator_equations=True)
    _CLASS_CACHE[key] = encoder, model, receipt
    return _CLASS_CACHE[key]


def build_gt_free_model(model_config, *, original_snapshot_root=None):
    _, cls, receipt = model_classes(original_snapshot_root=original_snapshot_root)
    config = copy.deepcopy(model_config)
    if config.get('ablation_mode', 'full') != 'full':
        raise ValueError('v2.4 final comparison retains all L0/L1/L2 levels')
    model = cls(**config)
    if any('lesion' in name for name, _ in model.named_parameters()):
        raise ValueError('Removed recipient lesion path still owns trainable parameters')
    model.v24_model_contract = dict(receipt, model_config=config,
        parameters=sum(parameter.numel() for parameter in model.parameters()),
        trainable_parameters=sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))
    return model
