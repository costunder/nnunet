"""Mechanical CPU UNIT coverage/cost/storage tests; no CT or neural outputs.

The seven fabricated metadata rows and mocked canonical files are explicitly
UNIT artifacts. These tests do not claim clinical data preparation or quality.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import uuid

import numpy as np
import torch
from torch_geometric.data import HeteroData

from hiercp_v1x import transition_v1_data as data


ROOT=Path(__file__).resolve().parents[1]


def unit_graph(ordinal,view):
    graph=HeteroData()
    source,target=ordinal+2+view,3+view
    graph['source_fine'].x=torch.zeros(source,16)
    graph['target_fine'].x=torch.zeros(target,16)
    edges=ordinal+view+1
    graph['source_fine','UNIT_contact','target_fine'].edge_index=torch.stack(
        (torch.arange(edges)%source,torch.arange(edges)%target))
    graph['target_fine','UNIT_near','target_fine'].edge_index=torch.tensor([[0,1],[1,2]])
    return graph


def unit_payload(record,*,epoch):
    if epoch!=0:raise AssertionError('Preparation must measure actual fixed epoch0 views')
    index=record['UNIT_ordinal']
    return ((unit_graph(index,0),unit_graph(index,1)),None,None)


class UnitWriter:
    """Storage-only mock: no fabricated canonical graph is used by a model."""
    def __init__(self,root,minimum_free_bytes):self.root=Path(root)
    def write(self,relative,record,source_key):
        path=self.root/relative;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(('UNIT_storage_only_'+str(record['UNIT_ordinal'])).encode())
        source=self.root/'UNIT_shared_source_metadata.pt'
        if not source.exists():source.write_bytes(b'UNIT_source_metadata_only')
        index=record['UNIT_ordinal']
        # Deliberately different canonical bounds, so they cannot impersonate
        # the actually sampled two-view node/edge measurements.
        return dict(path=relative,sha256=data._sha(path),
            bounds=dict(nodes=1000+index,edges=2000+index,bytes=4096+128*index),
            shared_source=dict(path=source.name,sha256=data._sha(source),content_sha256='b'*64))


class UnitProvider(data.OriginalInputProvider):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.requested_chunks=[]
        self._raw=SimpleNamespace(cache={'UNIT_RAW_metadata':dict(
            image=np.arange(6,dtype=np.float32),organ=np.ones(6,dtype=bool))})
    def _guard(self):
        # UNIT has no raw CT or resident model. The signed inventory is real
        # tiny UNIT storage; independent production guards are unchanged.
        if data._checked_sha(self.ds.path)!=self.ds.index_sha256:
            raise ValueError('UNIT inventory changed')
    def _records_for(self,rows):
        self.requested_chunks.append([row['id'] for row in rows])
        result=[]
        for row in rows:
            ordinal=self.ds.rows.index(row)
            record=dict(UNIT_ordinal=ordinal,UNIT_metadata_tensor=torch.arange(ordinal+1,dtype=torch.float32))
            self._records[row['id']]=(record,data._bytes(record));result.append(record)
        self._cached_bytes=data._bytes([record for record,size in self._records.values()])
        return result


def unit_dataset(root):
    rows=[]
    for index in range(7):
        case='UNIT-recipient-a' if index<4 else 'UNIT-recipient-b'
        rows.append(dict(id=f'{case}:{index}',case_id=case,patient_group='UNIT-group:'+case,
            component=1 if index in (0,4) else None,center=[index,0,0],target=int(index in (0,4)),
            donor_case_id='UNIT-training-donor',donor_component=1,donor_group='UNIT-donor-group'))
    path=root/'UNIT_native_inventory.json';path.write_text(json.dumps(rows),encoding='utf8')
    ds=object.__new__(data.NativeObservationDataset)
    ds.rows=rows;ds.meta=dict(records=copy.deepcopy(rows));ds.path=path
    ds.index_sha256=data._sha(path);ds.assignment_sha256=data.assignment_digest(rows)
    ds.debug=True;ds.base={};ds.scope_contract='a'*64;ds.case_ids=None
    ds.population=dict(debug=True,total_assignments=len(rows),UNIT_metadata_only=True)
    return ds


class CompletePreparationMechanicalUNIT(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Keep own tiny artifacts; never recursively delete results under
        # Windows file ACLs and never replace another experiment directory.
        cls.root=ROOT/'work'/('transition_D_preparation_UNIT_'+uuid.uuid4().hex)
        cls.root.mkdir()
        cls.ds=unit_dataset(cls.root)
        cls.provider=UnitProvider(cls.ds,workers=2,resident_bytes=1024**3,rss_bytes=2*1024**3)
        cls.output=cls.root/'canonical'
        with patch('hiercp_v1x.transition_preparation_storage.GraphWriter',UnitWriter),\
                patch.object(data.local,'materialize_pair',side_effect=unit_payload):
            cls.path=cls.provider.preflight(cls.output,minimum_free_bytes=1)
        cls.index=json.loads(cls.path.read_text(encoding='utf8'))

    def test_whole_population_streamed_in_worker_bounded_chunks(self):
        chunks=self.provider.requested_chunks
        self.assertTrue(chunks)
        self.assertTrue(all(1<=len(chunk)<=self.provider.workers for chunk in chunks),chunks)
        self.assertEqual([row for chunk in chunks for row in chunk],[row['id'] for row in self.ds.rows])
        self.assertEqual([row['id'] for row in self.index['records']],[row['id'] for row in self.ds.rows])
        self.assertTrue(self.index['complete']);self.assertEqual(self.index['prepared_observations'],7)
        self.assertEqual(self.index['skipped_observations'],0)
        self.assertFalse(self.index['hidden_subset'])

    def test_actual_heterogeneous_two_views_define_measurement_arrays(self):
        nodes=[];edges=[]
        for ordinal,row in enumerate(self.index['records']):
            views=[unit_graph(ordinal,view) for view in (0,1)]
            expected_nodes=[graph.num_nodes for graph in views]
            expected_edges=[graph.num_edges for graph in views]
            self.assertEqual(row['sampled_view_nodes'],expected_nodes)
            self.assertEqual(row['sampled_view_edges'],expected_edges)
            self.assertEqual(row['sampled_two_view_nodes'],sum(expected_nodes))
            self.assertEqual(row['sampled_two_view_edges'],sum(expected_edges))
            self.assertEqual(row['measurement_epoch'],0)
            self.assertNotEqual(row['sampled_two_view_nodes'],row['bounds']['nodes'])
            nodes.append(sum(expected_nodes));edges.append(sum(expected_edges))
        self.assertEqual(self.index['sampled_two_view_nodes'],nodes)
        self.assertEqual(self.index['sampled_two_view_edges'],edges)
        for kind,values in (('nodes',nodes),('edges',edges)):
            summary=self.index['sampled_two_view_summary'][kind]
            self.assertEqual(summary['sum'],sum(values));self.assertEqual(summary['min'],min(values))
            self.assertEqual(summary['max'],max(values));self.assertEqual(summary['observations'],7)
            self.assertEqual(summary['actual_local_graphs'],14)

    def test_measured_cost_sums_exact_selected_canonical_rows_without_loading(self):
        ids=[6,2,0];rows=[self.index['records'][index] for index in ids]
        with patch.object(self.provider,'_records_for',side_effect=AssertionError('Cost inspection must not build/load graphs')):
            actual=self.provider.measured_cost(ids)
        self.assertEqual(actual,dict(observations=3,actual_local_graphs=6,measurement_epoch=0,
            sampled_two_view_nodes=sum(row['sampled_two_view_nodes'] for row in rows),
            sampled_two_view_edges=sum(row['sampled_two_view_edges'] for row in rows),
            canonical_bound_bytes=sum(row['bounds']['bytes'] for row in rows)))
        for bad_ids in ([],[7],[-1],[True]):
            with self.subTest(ids=bad_ids),self.assertRaises(ValueError):self.provider.measured_cost(bad_ids)

    def test_missing_wrong_or_boolean_measured_stats_are_refused(self):
        original=self.index['records'][0]
        changes=[{'sampled_view_nodes':None},{'sampled_two_view_edges':original['sampled_two_view_edges']+1},
            {'measurement_epoch':1},{'sampled_view_nodes':[True,5]},{'sampled_measurement_format':'UNIT_WRONG'},
            {'sampled_view_edges':[-1,2]}]
        for change in changes:
            with self.subTest(change=change),self.assertRaises(ValueError):
                data._validate_measurement({**original,**change})
        for payload in (([unit_graph(0,0)],None,None),((HeteroData(),HeteroData()),None,None)):
            with self.assertRaises(ValueError):data._sampled_measurement(payload)
        fresh=UnitProvider(self.ds,workers=2,resident_bytes=1024**3,rss_bytes=2*1024**3)
        with self.assertRaises(ValueError):fresh.measured_cost([0])
        fresh.bind_cache(self.path)
        fresh._canonical[self.ds.rows[0]['id']].pop('sampled_view_nodes')
        with self.assertRaises(ValueError):fresh.measured_cost([0])

    def test_index_array_measurement_corruption_refused_on_admission(self):
        corrupt=copy.deepcopy(self.index);corrupt['sampled_two_view_nodes'][0]+=1
        path=self.root/'UNIT_wrong_measurement_index.json';path.write_text(json.dumps(corrupt),encoding='utf8')
        fresh=UnitProvider(self.ds,workers=2,resident_bytes=1024**3,rss_bytes=2*1024**3)
        with self.assertRaisesRegex(ValueError,'sampled graph measurements'):
            fresh.bind_cache(path)

    def test_telemetry_records_actual_UNIT_bytes_RSS_disk_and_phase_timings(self):
        names=self.index['preparation_metric_files']
        self.assertEqual(len(names),1)
        events=[json.loads(line) for line in (self.output/names[0]).read_text(encoding='utf8').splitlines()]
        self.assertEqual(events[0]['event'],'started')
        self.assertEqual(events[-1]['event'],'all_observations_completed')
        observation=[row for row in events if row['event']=='observation_completed']
        self.assertEqual([row['observation_id'] for row in observation],[row['id'] for row in self.ds.rows])
        raw_bytes=data._bytes(self.provider._raw.cache)
        for row in events:
            self.assertGreater(row['rss_bytes'],0)
            self.assertEqual(row['raw_resident_bytes'],raw_bytes)
            self.assertGreaterEqual(row['canonical_record_resident_bytes'],0)
            self.assertGreaterEqual(row['prepared_donor_resident_bytes'],0)
            self.assertGreaterEqual(row['canonical_resident_bytes'],0)
            self.assertEqual(row['resident_limit_bytes'],self.provider.resident_bytes)
            self.assertEqual(row['rss_limit_bytes'],self.provider.rss_bytes)
            self.assertGreater(row['disk_free_bytes'],0)
            self.assertGreaterEqual(row['disk_cumulative_canonical_bytes'],0)
            self.assertGreaterEqual(row['disk_written_canonical_bytes'],0)
            self.assertGreaterEqual(row['seconds'],0)
        for ordinal,(event,stored) in enumerate(zip(observation,self.index['records'])):
            self.assertEqual(event['canonical_tensor_bytes'],4*(ordinal+1))
            self.assertEqual(event['sampled_nodes'],stored['sampled_view_nodes'])
            self.assertEqual(event['sampled_edges'],stored['sampled_view_edges'])
            segment=self.output/stored['segment']
            self.assertEqual(event['compressed_graph_bytes'],(segment/stored['path']).stat().st_size)
            self.assertEqual(event['compressed_shared_source_bytes'],(segment/stored['shared_source']['path']).stat().st_size)
            self.assertTrue(1<=event['execution_observations']<=self.provider.workers)
            for key in ('materialize_two_view_seconds','write_publish_seconds','build_chunk_seconds'):
                self.assertGreaterEqual(event[key],0)
        actual_disk=sum(path.stat().st_size for path in (self.output/'segments').rglob('*') if path.is_file())
        self.assertEqual(events[-1]['disk_cumulative_canonical_bytes'],actual_disk)
        self.assertEqual(events[-1]['disk_written_canonical_bytes'],actual_disk)
        self.assertEqual(self.index['disk_cumulative_canonical_bytes'],actual_disk)
        self.assertEqual(sum(row['newly_written_shared_source_bytes'] for row in observation),
            observation[0]['compressed_shared_source_bytes'])


if __name__=='__main__':unittest.main()
