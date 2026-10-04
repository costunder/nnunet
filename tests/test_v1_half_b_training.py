"""Explicit CPU UNIT metadata/mechanical checks; no CT, GPU or training claim.

Completed-baseline proof is mocked here. Tiny synthetic signed rows exercise
orchestration only; real archived constructor/gradient and real-CT CUDA checks
belong to separate tests/tools. Existing A files and baseline helpers are read.
"""
from __future__ import annotations

import ast
from collections import Counter
from copy import deepcopy
import json
import inspect
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from zipfile import ZipFile

import torch
from torch import nn

from hiercp_v1x import half_b_support as support, half_b_training as training
from hiercp_v1x.contracts import canonical_hash
from hiercp_v1x.experiment import digest
from tests.test_v1_half_a_training import ROOT, SCOPE, synthetic_unit_baseline, write_json
from tools import run_v1_half_b as runner


def unit_baseline(directory):
    baseline, native, execution, publication, last, best = synthetic_unit_baseline(directory)
    names = [f'UNIT_train_{i}__000.pt' for i in range(3)]
    index = dict(UNIT_metadata_only=True, entries=[
        dict(path=name, case_id=f'UNIT_train_{i}', sample_index=0, split='train')
        for i, name in enumerate(names)] + [
        dict(path='UNIT_val_0__000.pt', case_id='UNIT_val_0', sample_index=0, split='val')])
    write_json(baseline/'shared/cache/index.json', index)
    signature = deepcopy(last['training_signature'])
    signature['train_cache_files'] = names
    proof = dict(files={str(path): digest(path) for path in baseline.rglob('*') if path.is_file()},
        execution=execution, neural_baseline=dict(completed_epoch=40, best_epoch=11,
            best_mrr=1., selection=last['best_selection'], scope_digest=SCOPE,
            training_signature=signature, architecture_version=last['architecture_version']),
        initial_validation=json.loads((baseline/'results/v1.0/initial_validation.json').read_text()),
        cache_reused_without_preparation=True)
    return baseline, native, proof, index


def reseal(value):
    value = deepcopy(value)
    value['contract_sha256'] = canonical_hash({k:v for k,v in value.items() if k != 'contract_sha256'})
    return value


class UnitFixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='half_B_metadata_UNIT_', dir=ROOT/'work')
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.baseline, self.native, self.proof, self.index = unit_baseline(self.directory)
        self.proof_patch = patch.object(training, 'baseline_proof',
            side_effect=lambda path:(deepcopy(self.native), deepcopy(self.proof)))
        self.proof_patch.start()
        self.addCleanup(self.proof_patch.stop)

    def initialized(self):
        return training.initialize(self.baseline, self.directory/'UNIT_B', 3, 40., 192.)

    def manager(self):
        root, receipt = self.initialized()
        pipeline = SimpleNamespace(_loader_kwargs=Mock())
        manager = support.SupportManager(pipeline, receipt, dict(contract_sha256=SCOPE),
                                        root/'results/half_B', Mock())
        self.addCleanup(manager.close)
        return manager


class HalfBRecipeUnits(UnitFixture):
    def test_recipe_only_resolves_measured_batch_workers_and_preserves_v1_target(self):
        before = {str(p):digest(p) for p in self.baseline.rglob('*') if p.is_file()}
        root, receipt = self.initialized()
        cfg = deepcopy(receipt['config'])
        for key in ('batch_size','num_workers'):
            cfg['training'][key] = self.native['config']['training'][key]
        self.assertEqual(cfg, self.native['config'])
        self.assertEqual(receipt['config']['training']['epochs'], 40)
        self.assertEqual(receipt['support_training_samples'], 3)
        self.assertEqual(receipt['support_materialized_patients'], 3)
        self.assertEqual(receipt['support_label_definition'], 'original_v1_candidate_index0_class1_other7_class0')
        self.assertIs(receipt['native_v22_objective_equivalence'], False)
        self.assertIs(receipt['target_supervision_changed'], False)
        self.assertEqual(training.verify(root), receipt)
        self.assertEqual(before, {str(p):digest(p) for p in self.baseline.rglob('*') if p.is_file()})

    def test_exact_repeat_preserves_config_manifest_bytes(self):
        root, receipt = self.initialized()
        before = {p.name:p.read_bytes() for p in (root/'config.json',root/'manifest.json')}
        again, value = self.initialized()
        self.assertEqual((again,value),(root,receipt))
        self.assertEqual(before, {p.name:p.read_bytes() for p in (root/'config.json',root/'manifest.json')})

    def test_resealed_target_support_flags_or_source_cannot_migrate_existing_B(self):
        root, receipt = self.initialized()
        for key,value in [('learning_target','UNIT_observation_task'),
                ('target_supervision_changed',True),('native_v22_objective_equivalence',True),
                ('support_label_definition','UNIT_query_GT_edges'),('support_training_samples',2),
                ('support_materialized_patients',2),('source',str(self.directory/'UNIT_other_source')),
                ('A_automatic_start',True),('combined_automatic_start',True),('epochs',1)]:
            with self.subTest(key=key):
                changed = deepcopy(receipt); changed[key] = value
                write_json(root/'manifest.json', reseal(changed))
                with self.assertRaises(ValueError):training.verify(root)
        write_json(root/'manifest.json', receipt)

    def test_resealed_model_loss_or_scope_config_change_rejected(self):
        root, receipt = self.initialized()
        for category,key,value in [('model','hidden_dim',64),('training','consistency_weight',0.),
                ('training','epochs',1),('graph','adaptive_roi_margin_mm',5.)]:
            with self.subTest(key=key):
                changed = deepcopy(receipt); changed['config'][category][key] = value
                write_json(root/'manifest.json', reseal(changed))
                write_json(root/'config.json', changed['config'])
                with self.assertRaises(ValueError):training.verify(root)
        write_json(root/'manifest.json', receipt); write_json(root/'config.json', receipt['config'])

    def test_resource_change_overlap_and_unbound_output_fail(self):
        root, receipt = self.initialized()
        for gpu,cuda,rss in [(2,40.,192.),(3,39.,192.),(3,40.,191.)]:
            with self.subTest(resources=(gpu,cuda,rss)):
                with self.assertRaises(ValueError):training.initialize(self.baseline,root,gpu,cuda,rss)
        with self.assertRaises(ValueError):training.initialize(self.baseline,self.baseline/'UNIT_B',3,40.,192.)
        unbound = self.directory/'UNIT_unbound'; unbound.mkdir()
        (unbound/'keep.txt').write_text('UNIT retained')
        with self.assertRaises(FileExistsError):training.initialize(self.baseline,unbound,3,40.,192.)
        self.assertEqual((unbound/'keep.txt').read_text(),'UNIT retained')

    def test_implementation_inventory_or_changed_source_digest_rejected(self):
        root, receipt = self.initialized()
        changed = deepcopy(receipt)
        changed['implementation'].pop('hiercp_v1x/half_b_support.py')
        write_json(root/'manifest.json', reseal(changed))
        with self.assertRaises(ValueError):training.verify(root)
        write_json(root/'manifest.json', receipt)
        original = training.digest
        with patch.object(training,'digest',side_effect=lambda path:
                'aa'*32 if Path(path).name=='half_b_support.py' else original(path)):
            with self.assertRaises(ValueError):training.verify(root)

    def test_only_one_B_child_and_no_prepare_A_combined_paths(self):
        root, receipt = self.initialized()
        child = Mock(); child.wait.return_value = 0
        with patch.object(training,'verify',return_value=receipt), patch.object(runner.subprocess,'Popen',return_value=child) as popen:
            runner.run(root,receipt)
        popen.assert_called_once()
        command = popen.call_args.args[0]
        self.assertEqual(command[1:4],['-u','-m','hiercp_v1x.half_b_entry'])
        self.assertEqual(len(command),6)
        request = json.loads(Path(command[-1]).read_text())
        self.assertEqual(request,dict(experiment=str(root),contract_sha256=receipt['contract_sha256']))
        self.assertNotIn('PYTHONPATH',popen.call_args.kwargs['env'])
        self.assertNotIn('HIERCP_V1X_SAMPLING_CONTRACT',popen.call_args.kwargs['env'])
        child.wait.assert_called_once()
        self.assertFalse((root/'run.lock').exists())
        self.assertFalse((root/'results/half_A').exists())


class HalfBSupportMetadataUnits(UnitFixture):
    def test_real_materialized_cohort_differs_from_configured_full84(self):
        manager = self.manager()
        self.assertEqual(manager.cohort['configured_cases'],84)
        self.assertEqual(manager.cohort['materialized_cases'],3)
        self.assertEqual(manager.cohort['samples'],3)
        self.assertEqual(len(manager.cohort['configured_but_not_materialized_cases']),81)
        self.assertEqual(manager.cohort['expected_samples'],
            {f'UNIT_train_{i}__000.pt':f'UNIT_train_{i}' for i in range(3)})

    def test_signed_cohort_rejects_missing_duplicate_foreign_or_wrong_sample_rows(self):
        files = self.proof['neural_baseline']['training_signature']['train_cache_files']
        cases, val = self.native['split']['train'],self.native['split']['val']
        changes = []
        missing = deepcopy(self.index); missing['entries'].pop(0); changes.append(missing)
        duplicate = deepcopy(self.index); duplicate['entries'].append(deepcopy(duplicate['entries'][0])); changes.append(duplicate)
        wrong_case = deepcopy(self.index); wrong_case['entries'][0]['case_id'] = val[0]; changes.append(wrong_case)
        wrong_index = deepcopy(self.index); wrong_index['entries'][0]['sample_index'] = 1; changes.append(wrong_index)
        wrong_split = deepcopy(self.index); wrong_split['entries'][0]['split'] = 'val'; changes.append(wrong_split)
        for value in changes:
            with self.subTest(index=value):
                with self.assertRaises(ValueError):support.signed_training_cohort(files,cases,val,value)
        with self.assertRaises(ValueError):support.signed_training_cohort(files[:2],cases,val,
            {'entries':self.index['entries'][1:]})
        with self.assertRaises(ValueError):support.signed_training_cohort(files+files[:1],cases,val,self.index)

    def test_manager_rejects_changed_bound_index_or_scope(self):
        root, receipt = self.initialized()
        with self.assertRaises(ValueError):support.SupportManager(SimpleNamespace(),receipt,
            dict(contract_sha256='aa'*32),root/'results/half_B',Mock())
        write_json(self.baseline/'shared/cache/index.json',{'UNIT_changed':True})
        with self.assertRaises(ValueError):support.SupportManager(SimpleNamespace(),receipt,
            dict(contract_sha256=SCOPE),root/'results/half_B',Mock())

    def test_refresh_schedule_initial_each_eval_and_resume_only_when_missing(self):
        manager = self.manager(); model = object(); session = SimpleNamespace(_epoch=None)
        args = dict(train_files=manager.files,batch_size=16,workers=8,seed=42,use_amp=False,device='cpu')
        with patch.object(manager,'refresh') as refresh:
            manager.before_pass(model,False,session,**args)
            self.assertEqual(refresh.call_args.kwargs['epoch'],0)
            manager._bound_model, manager._generation = model,'UNIT_bound_generation'
            session._epoch = 1
            manager.before_pass(model,True,session,**args)
            self.assertEqual(refresh.call_count,1)
            manager.before_pass(model,False,session,**args)
            self.assertEqual(refresh.call_args.kwargs['epoch'],1)
            manager._bound_model = None; session._epoch = 12
            manager.before_pass(model,True,session,**args)
            self.assertEqual(refresh.call_args.kwargs['epoch'],11)
            self.assertEqual(refresh.call_args.kwargs['trigger'],'missing_bank_before_resumed_train')
        manager._bound_model = None

    def test_runtime_batch_workers_files_seed_or_precision_cannot_silently_change(self):
        manager = self.manager()
        args = dict(train_files=manager.files,batch_size=16,workers=8,seed=42,use_amp=False,device='cpu')
        for key,value in [('train_files',manager.files[:-1]),('batch_size',8),('workers',0),
                          ('seed',41),('use_amp',True)]:
            with self.subTest(key=key):
                changed = dict(args); changed[key] = value
                with patch.object(manager,'refresh') as refresh:
                    with self.assertRaises(ValueError):manager.before_pass(object(),False,SimpleNamespace(_epoch=1),**changed)
                    refresh.assert_not_called()


class UnitGraph:
    """Mechanical CPU view identity; never a medical/production graph."""
    def __init__(self,offset):self.offset=offset
    def to(self,device):return self


class UnitLocal(nn.Module):
    def __init__(self):
        super().__init__(); self.project=nn.Linear(1,128,bias=False)
        with torch.no_grad():self.project.weight.fill_(1.)
    def encode_dense_maps(self,source,owners,target):return source[owners],target
    def forward_graph(self,graph,source,target):
        # Deliberately advance a RNG stream in this UNIT fixture to prove that
        # the manager restores it even if a future local implementation does so.
        torch.rand(1)
        return {'fused':self.project(target.flatten(1).mean(1,keepdim=True)+graph.offset)}


class UnitBank(nn.Module):
    def __init__(self):super().__init__(); self.memory=None
    def bind_support(self,memory):self.memory=memory
    def clear_support(self):self.memory=None
    def forward(self,*args):raise AssertionError('UNIT upper must not run during L0 refresh')


class UnitModel(nn.Module):
    def __init__(self):super().__init__(); self.local_encoder=UnitLocal(); self.half_b=UnitBank()
    def forward(self,*args):raise AssertionError('UNIT full model must not run during support refresh')
    @staticmethod
    def _mean_embeddings(first,second):return {key:(first[key]+second[key])*.5 for key in first}


def unit_batch(cases):
    count=len(cases)
    return SimpleNamespace(UNIT=True,counts=(8,)*count,case_ids=tuple(cases),
        difficulties=torch.tensor([0,1,2,3,1,2,3,1]*count,dtype=torch.long),
        source_patches=torch.zeros(count,1,1,1,1),
        target_patches=torch.arange(count*8,dtype=torch.float32).reshape(-1,1,1,1,1),
        local_batch=UnitGraph(1.),local_batch_view2=UnitGraph(3.))


class HalfBSupportMechanicalUnits(UnitFixture):
    def test_refresh_binds_all_candidates_two_views_preserves_modes_rng_and_global_query_generators(self):
        manager=self.manager(); model=UnitModel().train()
        model.local_encoder.project.eval()
        batch=unit_batch(manager.cohort['case_ids'])
        manager._worker_generator=torch.Generator().manual_seed(999)
        query_generator=torch.Generator().manual_seed(1234)
        before_query=query_generator.get_state().clone()
        before_rng=torch.get_rng_state().clone()
        modes=[m.training for m in model.modules()]
        with patch.object(manager,'_ensure_loader',return_value=[batch]), patch.object(torch.cuda,'is_available',return_value=False):
            manager.refresh(model,epoch=0,device='cpu',use_amp=False,trigger='UNIT_initial')
        memory=model.half_b.memory
        self.assertEqual(memory['embeddings'].shape,(24,128))
        self.assertEqual(memory['embeddings'].dtype,torch.float32)
        self.assertFalse(memory['embeddings'].requires_grad)
        self.assertTrue(torch.equal(memory['embeddings'][:,0],torch.arange(24)+2.))
        self.assertEqual(memory['owners'].tolist(),[0]*8+[1]*8+[2]*8)
        self.assertEqual(memory['classes'].tolist(),[1,0,0,0,0,0,0,0]*3)
        self.assertEqual(memory['candidate_indices'],tuple(range(8))*3)
        self.assertEqual(Counter(memory['sample_ids']),Counter({name:8 for name in manager.cohort['expected_samples']}))
        self.assertEqual(len(memory['training_case_ids']),84)
        self.assertEqual(len(memory['validation_case_ids']),21)
        self.assertEqual(memory['support_policy'],support.SUPPORT_POLICY)
        self.assertTrue(torch.equal(torch.get_rng_state(),before_rng))
        self.assertTrue(torch.equal(query_generator.get_state(),before_query))
        self.assertEqual([m.training for m in model.modules()],modes)
        records=list(manager.output.glob('support_refresh_*.json'))
        self.assertEqual(len(records),1)
        record=json.loads(records[0].read_text())
        self.assertEqual(record['samples'],3)
        self.assertEqual(record['candidate_embeddings'],24)
        self.assertEqual(record['optimizer_steps'],0)
        self.assertEqual(record['validation_records_used'],0)
        self.assertEqual(record['training_queries_consumed'],0)
        manager.close(); self.assertIsNone(model.half_b.memory)

    def test_partial_or_wrong_case_support_fails_without_binding_or_complete_receipt(self):
        for cases in [['UNIT_train_0'],['UNIT_val_0','UNIT_train_1','UNIT_train_2']]:
            with self.subTest(cases=cases):
                manager=self.manager(); model=UnitModel()
                manager._worker_generator=torch.Generator()
                before=torch.get_rng_state().clone()
                with patch.object(manager,'_ensure_loader',return_value=[unit_batch(cases)]), patch.object(torch.cuda,'is_available',return_value=False):
                    with self.assertRaises(ValueError):manager.refresh(model,epoch=0,device='cpu',use_amp=False,trigger='UNIT_bad')
                self.assertIsNone(model.half_b.memory)
                self.assertTrue(torch.equal(torch.get_rng_state(),before))
                self.assertEqual(list(manager.output.glob('support_refresh_*.json')),[])

    def test_failure_restores_rng_and_mixed_modes(self):
        manager=self.manager(); model=UnitModel().train(); model.half_b.eval()
        manager._worker_generator=torch.Generator()
        before=torch.get_rng_state().clone(); modes=[m.training for m in model.modules()]
        def failed(*args,**kwargs):torch.rand(1);raise RuntimeError('UNIT local failure')
        with patch.object(manager,'_ensure_loader',return_value=[unit_batch(manager.cohort['case_ids'])]), patch.object(torch.cuda,'is_available',return_value=False), patch.object(support,'encode_local_support',side_effect=failed):
            with self.assertRaisesRegex(RuntimeError,'UNIT local failure'):
                manager.refresh(model,epoch=1,device='cpu',use_amp=False,trigger='UNIT_bad')
        self.assertTrue(torch.equal(torch.get_rng_state(),before))
        self.assertEqual([m.training for m in model.modules()],modes)
        self.assertIsNone(model.half_b.memory)


class HalfBObservationOverlayUnits(unittest.TestCase):
    def source(self):
        with ZipFile(ROOT/'versions/v1/pipeline_v1_source.zip') as archive:
            text=archive.read('hiercp/pipeline.py').decode('utf8').replace('\r\n','\n')
        tree=ast.parse(text)
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run_train')
        # Use the same block extraction as inspect.getsource in production,
        # including original trailing comment/blank lines outside the AST span.
        return ''.join(inspect.getblock(text.splitlines(keepends=True)[node.lineno-1:]))

    def test_checked_overlay_keeps_original_full_loop_curriculum_and_loss(self):
        original=self.source(); rewritten,identity=support.observation_source(original)
        tree=ast.parse(rewritten)
        epoch_pass=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='epoch_pass')
        self.assertEqual(ast.unparse(epoch_pass.body[0].value.func),'_v1x_half_b_support_manager.before_pass')
        self.assertEqual(ast.unparse(epoch_pass.body[1].value.func),'model.train')
        for marker in ['for epoch in range(start_epoch, epochs + 1):',
                       'train_dataset.set_epoch(epoch)', 'loss = ranking_loss + consistency_weight * output.consistency',
                       'optimizer.step()', 'scheduler.step()']:
            self.assertEqual(rewritten.count(marker),original.count(marker))
        self.assertEqual(rewritten.count('_v1x_half_b_support_manager.before_pass('),1)
        self.assertEqual(rewritten.count('curriculum_ranking_loss('),original.count('curriculum_ranking_loss('))
        self.assertNotIn('supervised_loss',rewritten)
        self.assertIn('val_worker_generator.set_state(_v1x_initial_worker_rng)',rewritten)
        self.assertEqual(identity['half_B_support_policy'],support.SUPPORT_POLICY)
        self.assertTrue(identity['initial_full_validation_inserted'])

    def test_changed_original_epoch_or_pass_boundary_rejected(self):
        original=self.source()
        for old,new in [('model.train(training_mode)','model.eval()'),
                        ('for epoch in range(start_epoch, epochs + 1):','for epoch in range(start_epoch, 2):')]:
            with self.subTest(old=old):
                with self.assertRaises(ValueError):support.observation_source(original.replace(old,new))


if __name__=='__main__':unittest.main()
