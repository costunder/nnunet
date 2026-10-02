"""CPU JSON-only UNIT assertions; no CPU model, CT pixels or accuracy claim."""

import copy
import unittest

from tools.audit_relay_connectivity_debug import (
    RELATIONS, audit_record, scene_geometry, validate_polyline,
    validate_seed_preservation,
)


def node(side, role, native, relative, slot):
    return dict(side=side, role=role, native_voxel=list(native), xyz=list(native),
        relative_xyz_mm=list(relative), model_slot=slot, local_index=slot,
        retention='query' if role == 'query' else 'seed', abstract=role == 'query')


def synthetic_record():
    donor = [node('donor', 'query', (10,20,30), (0,0,0), 0),
             node('donor', 'near', (11,20,30), (1,0,0), 1)]
    target = [node('recipient', 'query', (20,20,30), (0,0,0), 0),
              node('recipient', 'near', (21,20,30), (1,0,0), 1)]
    edges = [[0,1,RELATIONS[2]], [1,0,RELATIONS[3]],
             [2,3,RELATIONS[4]], [3,2,RELATIONS[5]], [0,3,RELATIONS[7]]]
    record = dict(id='explicit UNIT record', kind='P',
        donor=dict(case='UNIT donor', nodes=donor, center=[10,20,30],
            box=[[9.5,19.5,29.5],[30.5,40.5,50.5]]),
        scene=dict(case='UNIT recipient', nodes=target, center=[20,20,30],
            box=[[19.5,19.5,29.5],[40.5,40.5,50.5]]),
        joint=dict(nodes=donor+target, edges=edges, edge_path_mm=[None]*5,
            edge_distance_mm=[1.]*5, edge_orientation='source,target,relation_name',
            stats=dict(components=1,node_count=4,directed_typed_edges=5)))
    scene = dict(id=record['id'],kind='P',components=1,isolated_nodes=0,
        pair_node_count=4,donor_node_count=2,recipient_node_count=2,
        donor_directed_edges=2,recipient_directed_edges=2,cross_edges=1,
        relation_counts={name:sum(edge[2]==name for edge in edges) for name in RELATIONS},
        seed_context_counts=[1,1],relay_counts=[0,0],
        native6_substrate_unreachable_counts=[0,0])
    return record, dict(scenes=[scene])


class JSONOnlyAuditUnit(unittest.TestCase):
    def test_saved_edge_bfs_and_original_shape_match_known_graph(self):
        record, arm = synthetic_record()
        result = audit_record(record, arm, relayed=True)
        self.assertEqual(result['weak_components'],1)
        self.assertEqual(result['isolated_count'],0)
        self.assertEqual(result['outside_both_query_components_count'],0)
        self.assertEqual(result['retained_by_branch']['donor']['original_native_shape'],[21,21,21])
        # donor context -> donor query -> recipient context -> recipient query.
        self.assertEqual(result['query_paths']['recipient_query']['ancestors_3_layers_count'],4)
        self.assertFalse(result['verified']['organ_support_independently_rechecked'])

    def test_wrong_saved_cuda_components_are_rejected(self):
        record, arm = synthetic_record()
        arm['scenes'][0]['components']=2
        with self.assertRaisesRegex(ValueError,'component count'):
            audit_record(record,arm,relayed=True)
        record, arm = synthetic_record()
        record['kind']='U'
        with self.assertRaisesRegex(ValueError,'P/U observation changed'):
            audit_record(record,arm,relayed=True)

    def test_native_parent_polyline_checks_direction_axis_and_cumulative_length(self):
        record,_=synthetic_record()
        geometry=scene_geometry(record['donor'])
        source=dict(xyz=[10,20,30],native_voxel=[10,20,30])
        target=dict(xyz=[16,20,30],native_voxel=[16,20,30])
        path=[[x,20,30] for x in range(10,17)]
        valid=validate_polyline(path,source,target,geometry)
        self.assertEqual(valid,dict(native_steps=6,cumulative_mm=6.))
        with self.assertRaisesRegex(ValueError,'real native parent polyline'):
            validate_polyline(1.0,source,target,geometry)
        with self.assertRaisesRegex(ValueError,'direction'):
            validate_polyline(path[::-1],source,target,geometry)
        with self.assertRaisesRegex(ValueError,'skips a voxel'):
            validate_polyline([path[0],path[2],*path[3:]],source,target,geometry)
        # Close endpoints cannot conceal seven genuine native steps.
        excessive=[[10,20,30],[10,21,30],[10,22,30],[10,23,30],
                   [11,23,30],[11,22,30],[11,21,30],[11,20,30]]
        with self.assertRaisesRegex(ValueError,'cumulative 6 mm'):
            validate_polyline(excessive,source,dict(xyz=[11,20,30],native_voxel=[11,20,30]),geometry)
        # A false endpoint/path perturbation cannot hide in serialization.
        forged=copy.deepcopy(path);forged[0][0]+=2e-5
        with self.assertRaisesRegex(ValueError,'direction'):
            validate_polyline(forged,source,target,geometry)

    def test_original_seed_role_or_retention_cannot_be_changed(self):
        old,_=synthetic_record();new=copy.deepcopy(old)
        summary=validate_seed_preservation(old,new)
        self.assertEqual(summary['original_context_seeds_preserved'],2)
        self.assertEqual(summary['original_abstract_queries_preserved'],2)
        for field,value in (('role','mid'),('retention','relay')):
            changed=copy.deepcopy(new)
            changed['joint']['nodes'][1][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):
                validate_seed_preservation(old,changed)

    def test_crop_shape_shift_or_nonfinite_point_is_rejected(self):
        old,_=synthetic_record();new=copy.deepcopy(old)
        new['scene']['box'][1][0]+=1
        with self.assertRaisesRegex(ValueError,'crop shape'):
            validate_seed_preservation(old,new)
        broken=copy.deepcopy(old['donor'])
        broken['nodes'][1]['xyz'][0]=float('nan')
        with self.assertRaisesRegex(ValueError,'finite'):
            scene_geometry(broken)


if __name__=='__main__':
    unittest.main()
