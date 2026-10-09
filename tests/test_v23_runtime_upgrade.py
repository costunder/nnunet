"""UNIT: exact paused cursors and true BEST across a performance-only upgrade."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import torch

from tests.test_v23_training import _handoff_unit_fixture
from hiercp_v1x.contracts import canonical_hash
from hiercp_v1x.u_bridge_training import digest
from hiercp_v1x.v23_training import handoff_single_gpu_checkpoint, _file_sha256
from hiercp_v1x.v23_runtime_upgrade import upgrade_single_gpu_checkpoint, NUMERICAL_KEYS


def fixture(directory, phase='training', cursor=4):
    population, checkpoint, identity, best, proof, request, calibration, config, _, _ = _handoff_unit_fixture(directory)
    source = directory / 'singleton'
    handoff_single_gpu_checkpoint(checkpoint, identity, request, config, calibration, source,
        source_proof=proof, population=population, source_best_checkpoint=best, debug=True)
    checkpoint, identity, best = (source / name for name in
        ('checkpoint_latest.pt', 'training_identity.json', 'checkpoint_best.pt'))
    saved = torch.load(checkpoint, weights_only=False)
    state = saved['state']; state['status'] = 'PAUSED'; state['phase'] = phase
    if phase == 'training':
        state['train_position'] = cursor
        state['train_rows'] = [dict(case_id=case) for case in state['train_order'][:cursor]]
        state['updates'] += 1; state['attempts'] += 1
        saved['optimizer']['state'][0]['step'] += 1
    else:
        state['train_position'] = len(population.train)
        state['train_rows'] = [dict(case_id=case) for case in state['train_order']]
        state['evaluation_position'] = cursor
        state['evaluation_rows'] = [dict(case_id=case) for case in population.val[:cursor]]
        state['updates'] += 17; state['attempts'] += 17
        saved['optimizer']['state'][0]['step'] += 17
    saved.pop('content_sha256'); saved['content_sha256'] = digest(saved); torch.save(saved, checkpoint)
    new = dict(format='UNIT', source={'unit_source':'7'*64}, gpus=[5])
    new['request_sha256'] = canonical_hash(new)
    calibration = dict(calibration, request_sha256=new['request_sha256'])
    config = copy.deepcopy(config)
    config['v23_runtime'].update(prefetch_cpu_chunks=True, pin_cpu_batches=True,
        memoize_cpu_geometry=True, cache_stage_admission=True, geometry_resident_gib_per_rank=128,
        memoize_fixed_views=True, persistent_cpu_workers=True)
    proof = dict(proof, source_commit='b'*40, source_request=request,
        source_files_sha256=request['source'], source_checkpoint_file_sha256=_file_sha256(checkpoint),
        source_identity_file_sha256=_file_sha256(identity), source_best_checkpoint_file_sha256=_file_sha256(best),
        clean_pause_verified=True, new_runtime_full128_backward_verified=True,
        numerical_equivalence_receipt_sha256='8'*64)
    return population, checkpoint, identity, best, proof, new, calibration, config, saved


class RuntimeUpgradeUnit(unittest.TestCase):
    def upgrade(self, directory, values, **changes):
        population, checkpoint, identity, best, proof, request, calibration, config, saved = values
        args = dict(source_proof=proof, population=population, source_best_checkpoint=best, debug=True)
        args.update(changes)
        receipt = upgrade_single_gpu_checkpoint(checkpoint, identity, request, config, calibration,
            directory/'upgraded', **args)
        return receipt, torch.load(directory/'upgraded/checkpoint_latest.pt', weights_only=False)

    def test_partial_training_and_both_pending_validation_cursors_are_preserved(self):
        for phase, cursor in (('training',4), ('full_validation',20), ('full_validation',21)):
            with self.subTest(phase=phase, cursor=cursor), tempfile.TemporaryDirectory(prefix='v23_upgrade_UNIT_', dir=Path.cwd()) as name:
                directory=Path(name); values=fixture(directory,phase,cursor)
                receipt, result=self.upgrade(directory, values); source=values[-1]
                for key in NUMERICAL_KEYS: self.assertEqual(digest(source[key]), digest(result[key]))
                progress=lambda state: digest({k:v for k,v in state.items() if k!='handoff'})
                self.assertEqual(progress(source['state']),progress(result['state']))
                self.assertTrue(receipt['no_new_optimizer_update'])
                old_best=torch.load(values[3],weights_only=False)
                new_best=torch.load(directory/'upgraded/checkpoint_best.pt',weights_only=False)
                for key in NUMERICAL_KEYS: self.assertEqual(digest(old_best[key]),digest(new_best[key]))
                self.assertNotEqual(digest(result['model']),digest(new_best['model']))
                self.assertEqual(source['state']['updates']+receipt['remaining_planned_updates'],603)
                with self.assertRaises(FileExistsError): self.upgrade(directory,values)

    def test_core_policy_or_GPU_change_and_missing_best_are_rejected(self):
        for change in ('model','policy','gpu','best'):
            with self.subTest(change=change), tempfile.TemporaryDirectory(prefix='v23_upgrade_reject_UNIT_',dir=Path.cwd()) as name:
                directory=Path(name); values=list(fixture(directory))
                if change=='model': values[7]['model']['UNIT']=False
                if change=='policy': values[7]['v23_runtime']['target_selection_policy']='native_prefix'
                if change=='gpu':
                    values[5]['gpus']=[6]
                    values[5]['request_sha256']=canonical_hash({k:v for k,v in values[5].items() if k!='request_sha256'})
                with self.assertRaises(ValueError):
                    self.upgrade(directory,values,**({'source_best_checkpoint':None} if change=='best' else {}))
                self.assertFalse((directory/'upgraded').exists())

    def test_source_file_tamper_and_running_checkpoint_are_rejected(self):
        for running in (False,True):
            with self.subTest(running=running), tempfile.TemporaryDirectory(prefix='v23_upgrade_source_UNIT_',dir=Path.cwd()) as name:
                directory=Path(name); values=list(fixture(directory)); checkpoint=values[1]
                if running:
                    saved=values[-1]; saved['state']['status']='RUNNING'
                    saved.pop('content_sha256');saved['content_sha256']=digest(saved);torch.save(saved,checkpoint)
                    values[4]['source_checkpoint_file_sha256']=_file_sha256(checkpoint)
                else:
                    with checkpoint.open('ab') as stream: stream.write(b'changed')
                with self.assertRaises(ValueError): self.upgrade(directory,values)
                self.assertFalse((directory/'upgraded').exists())


if __name__=='__main__': unittest.main()
