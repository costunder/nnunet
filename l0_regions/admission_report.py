"""Report an admission rejection without changing partition or acceptance rules."""
import copy


def rejection_report(runner, level, bindings, profile):
    audit = copy.deepcopy(runner.last_audit)
    limits = copy.deepcopy(profile['diagnostic_profile_from_original_NOT_VALIDATED'])
    for scale in (1, 2):
        if scale==2 and profile.get('region_scales')==1:
            audit['scale2']={'status':'REMOVED_BY_DESIGN'}
            continue
        audit.setdefault(f'scale{scale}', {'status': 'NOT_RUN'})
        if scale < level:
            audit[f'scale{scale}']['status'] = 'PASS'
        elif scale == level:
            audit[f'scale{scale}']['status'] = 'REJECTED'
    stats = audit[f'scale{level}']
    pairs = []
    for owner, identity in enumerate(bindings):
        violations = []
        nodes = stats.get('nodes_per_pair', [])
        edges = stats.get('edges_per_pair', [])
        for key, values, bound in (
            ('nodes', nodes, limits['nodes_total'][level-1]),
            ('directed_edges', edges, limits['directed_edges_total'][level-1]),
        ):
            if len(values) > owner and values[owner] > bound:
                violations.append(dict(kind=key, actual=values[owner], limit=bound))
        for role, role_stats in stats.get('roles', {}).items():
            key = role+'_per_shell' if 'context' in role else role
            cap = limits['caps_by_role'][key][level-1]
            bbox = limits['bbox_context_diagonal_mm' if 'context' in role else 'bbox_surface_diagonal_mm'][level-1]
            for group in role_stats.get('group_diagnostics', []):
                if group['owner'] != owner:
                    continue
                common = dict(role=role, shell=group['shell'])
                if group['clusters'] > cap:
                    violations.append(dict(common, kind='role_shell_nodes', actual=group['clusters'], limit=cap))
                for kind, field, bound in (
                    ('bbox', 'bbox_excess', bbox),
                    ('variance', 'variance_excess', limits['normalized_feature_sse_per_mass'][level-1]),
                ):
                    excess = group[field]
                    if excess['clusters']:
                        violations.append(dict(common, kind=kind, limit=bound, **excess))
        pairs.append(dict(record_id=identity['record_id'], pair_index=owner,
            nodes=nodes[owner] if len(nodes)>owner else None,
            directed_edges=edges[owner] if len(edges)>owner else None,
            violations=violations,
            status='PROFILE_REJECTED' if violations else 'NOT_ADMITTED_BATCH_STOPPED'))
    return dict(stage='fixed_partition_admission', failed_scale=level,
        reason='unvalidated_initial_profile_rejected', limits=limits, pairs=pairs,
        audit=audit, training_started=False, production_ready=False,
        limits_changed=False, records_skipped=False)


def print_rejection(report):
    print(f"PARTITION REJECTED | scale {report['failed_scale']} | unvalidated initial profile; not a CUDA dependency failure", flush=True)
    for pair in report['pairs']:
        if not pair['violations']:
            continue
        details=[]
        for v in pair['violations']:
            group=f" {v['role']}/shell{v['shell']}" if 'role' in v else ''
            value=(f"{v['actual']}>{v['limit']}" if 'actual' in v else
                   f"{v['clusters']} clusters above {v['limit']}, fine-node fraction={v['fine_node_fraction']:.3f}")
            details.append(f"{v['kind']}{group}: {value}")
        print(f"  {pair['record_id']} | N={pair['nodes']} E={pair['directed_edges']} | "+'; '.join(details), flush=True)
    if report['audit']['scale2'].get('status')=='REMOVED_BY_DESIGN':
        print('Scale 2: REMOVED_BY_DESIGN. No training sample was admitted.', flush=True)
    elif report['failed_scale']==1:
        print('Scale 2: NOT_RUN. No training sample was admitted.', flush=True)
