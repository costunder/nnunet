"""Synthetic log fixtures only; not server measurements."""
import unittest
from types import SimpleNamespace
from tools.summarize_v22_cost_readonly import schedule, summarize
from hiercp_v222.v1_training import groups, contiguous


class CostBoundaries(unittest.TestCase):
    def test_metadata_schedule_matches_actual_grouping_not_global_ceil(self):
        rows=[dict(id=str(i),case_id='a' if i<3 else 'b',patient_group='A' if i<3 else 'B',
                   bounds=dict(edges=20-i)) for i in range(6)]
        meta=dict(records=rows,split=dict(inner_train=['a','b'],inner_val=[]),debug=True)
        result=schedule(meta,2,4)['inner_train']
        ds=SimpleNamespace(rows=rows)
        self.assertEqual(result['grouped_query_batches'],len(list(groups(ds,2))))
        self.assertEqual(result['grouped_query_batches'],4)
        self.assertEqual(result['contiguous_memory_batches'],len(list(contiguous(rows,4))))

    def test_partial_log_cannot_become_complete_epoch(self):
        rows=[dict(stage='optimization',epoch=1,step=29,step_seconds=22.104)]
        result=summarize(rows,[dict(stage='epoch_complete',epoch=1,seconds=999)],epoch=1)
        self.assertIsNone(result['complete_epoch_seconds'])
        self.assertEqual(result['measured_step_seconds'],22.104)
        self.assertIsNone(result['means']['forward_seconds']['mean'])
        with self.assertRaises(ValueError):summarize(rows+rows,[],epoch=1)

    def test_phase_scope_and_invalid_sum(self):
        event=dict(stage='epoch_complete',epoch=1,seconds=10,
            runtime_phase_seconds=dict(optimization=6,refresh_memory=3,validation=1))
        result=summarize(None,[event],epoch=1)
        self.assertEqual(result['complete_epoch_seconds'],10)
        self.assertEqual(result['measured_updates'],0)
        event['seconds']=11
        with self.assertRaises(ValueError):summarize(None,[event],epoch=1)


if __name__=='__main__':unittest.main(verbosity=2)
