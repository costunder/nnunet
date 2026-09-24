"""v2.22 r4: CNN-conditioned PPR/A* sampling and cross-patient alignment."""
PIPELINE_VERSION = 'hiercp_v222_ppr_astar_cluster_r4'
TRAINING_READY = False
DESIGN_BLOCK_REASON = (
    'r4 rejected: fixed 26 radial destinations and path halo dominate selection. '
    'Algorithm DEBUG success does not establish the requested context graph. '
    'Do not train or deploy this graph design.'
)
