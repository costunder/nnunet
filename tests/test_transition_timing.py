"""Metadata-only UNIT: checkpoint-tail accounting, not neural/GPU evidence."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from hiercp_v1x.contracts import canonical_hash
from hiercp_v1x.transition_timing import WallCheckpointPipeline,restore_D_wall


class DCheckpointWallContracts(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir='work',prefix='D_wall_metadata_UNIT_')
        self.root=Path(self.temp.name)
        self.reference=dict(checkpoint_id='a'*32,journal_file='d_checkpoint_wall_'+'b'*32+'.jsonl',snapshot_monotonic=100.)
        self.wall=dict(format='crossed_active_epoch_wall_v1',epoch=1,elapsed_seconds=2.,
                       phase_wall_seconds={'optimization':1.},scope='pre-write')
        self.payload=dict(identity={'arm':'D','metadata_UNIT':True},state=dict(epoch=0,step=2,phase='optimization',
                          epoch_wall=self.wall,epoch_wall_checkpoint=self.reference))

    def tearDown(self):self.temp.cleanup()

    def write(self,**changes):
        row=dict(format='crossed_D_checkpoint_wall_receipt_v1',checkpoint_id='a'*32,epoch=0,step=2,
                 phase='optimization',identity_sha256=canonical_hash(self.payload['identity']),
                 active_epoch_wall=copy.deepcopy(self.wall))
        row['active_epoch_wall']['elapsed_seconds']=4.
        row.update(changes);row['content_sha256']=canonical_hash(row)
        (self.root/self.reference['journal_file']).write_text(json.dumps(row)+'\n',encoding='utf8')
        return row

    def test_exact_bound_write_tail_recovers_without_process_downtime(self):
        self.write()
        restored=restore_D_wall(self.payload,self.root)
        self.assertEqual(restored['epoch_seconds'],4.)
        self.assertEqual(self.payload['state']['epoch_wall']['elapsed_seconds'],2.)
        self.assertEqual(restored['wall_resume_scope'],'pre-write')

    def test_missing_or_interrupted_receipt_retains_explicit_lower_bound(self):
        restored=restore_D_wall(self.payload,self.root)
        self.assertEqual(restored['epoch_wall']['elapsed_seconds'],2.)
        self.assertIn('lower_bound',restored['wall_resume_scope'])
        (self.root/self.reference['journal_file']).write_text('{"incomplete":',encoding='utf8')
        self.assertIn('lower_bound',restore_D_wall(self.payload,self.root)['wall_resume_scope'])

    def test_another_cursor_or_changed_receipt_is_rejected(self):
        self.write(step=3)
        with self.assertRaisesRegex(ValueError,'another saved'):restore_D_wall(self.payload,self.root)
        row=self.write();row['active_epoch_wall']['elapsed_seconds']=99.
        (self.root/self.reference['journal_file']).write_text(json.dumps(row)+'\n',encoding='utf8')
        with self.assertRaisesRegex(ValueError,'content changed'):restore_D_wall(self.payload,self.root)

    def test_nonlocal_journal_or_backwards_elapsed_is_rejected(self):
        changed=copy.deepcopy(self.payload);changed['state']['epoch_wall_checkpoint']['journal_file']='../outside.jsonl'
        with self.assertRaisesRegex(ValueError,'reference changed'):restore_D_wall(changed,self.root)
        wall=copy.deepcopy(self.wall);wall['elapsed_seconds']=1.
        self.write(active_epoch_wall=wall)
        with self.assertRaisesRegex(ValueError,'backwards'):restore_D_wall(self.payload,self.root)

    def test_ordered_writer_publishes_receipt_after_successful_checkpoint(self):
        writer=WallCheckpointPipeline(self.root,'overlapped',None,lambda _:None,'metadata_UNIT')
        with patch('l0_regions.execution_pipeline.CheckpointPipeline._write',return_value={'saved':True}),\
             patch('hiercp_v1x.transition_timing.time.perf_counter',return_value=103.):
            self.assertEqual(writer._write(self.payload),{'saved':True})
        restored=restore_D_wall(self.payload,self.root)
        self.assertEqual(restored['epoch_seconds'],5.)
        self.assertEqual(restored['epoch_wall']['phase_wall_seconds']['asynchronous_checkpoint_write'],3.)


if __name__=='__main__':unittest.main()
