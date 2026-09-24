"""v2.2 active spatial schema; retired tumor-interior nodes cannot re-enter.

These are spatial node roles, separate from L1 data/label semantics.
Legacy v1/v2.1 schemas are preserved in their original modules.
"""
LOCAL_NODE_TYPES = (
    'tumor_surface', 'source_context', 'source_liver_surface',
    'target_context', 'target_liver_surface',
)
SOURCE_LOCAL_NODE_TYPES = frozenset(LOCAL_NODE_TYPES[:3])
LOCAL_EDGE_TYPES = (
    ('tumor_surface', 'surface_neighbor', 'tumor_surface'),
    ('source_context', 'context_neighbor', 'source_context'),
    ('source_liver_surface', 'surface_neighbor', 'source_liver_surface'),
    ('target_context', 'context_neighbor', 'target_context'),
    ('target_liver_surface', 'surface_neighbor', 'target_liver_surface'),
    ('tumor_surface', 'interfaces_source', 'source_context'),
    ('source_context', 'interfaces_tumor', 'tumor_surface'),
    ('tumor_surface', 'interfaces_target', 'target_context'),
    ('source_context', 'corresponds_to', 'target_context'),
    ('source_context', 'near_liver_surface', 'source_liver_surface'),
    ('source_liver_surface', 'anchors_context', 'source_context'),
    ('target_context', 'near_liver_surface', 'target_liver_surface'),
    ('target_liver_surface', 'anchors_context', 'target_context'),
)
TOPOLOGY_CONTRACT = 'surface_context_no_interior_r5'
