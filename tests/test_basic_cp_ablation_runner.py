"""Full stage delegation/ordering and own-BEST guard integration without CUDA."""
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from tools import run_basic_cp_ablation as runner
from hiercp_v1x import v24_direct_gpu_access as direct
from hiercp_v1x import v24_native_best_eval_runtime as best


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name)
        self.request=dict(root=str(self.root), native_root=str(self.root), arm='blend', gpu=6,
            bank='/exact/bank/index.json', bank_sha256='b'*64,
            readonly_admission='/exact/all131.json', readonly_admission_sha256='c'*64)
        self.native=dict(self.request, trainer='ExactBasicBlend', private_runtime='/exact/private/runtime',
            private_pre='/exact/preprocessed', private_data_folder='/exact/empty/private/folder',
            baseline=dict(dataset_name='Dataset730_LiverOnlineCP_OF0', plans_name='nnUNetResEncUNetMPlans'),
            split=dict(outer_train=['train'+str(i) for i in range(105)],
                outer_val=['val'+str(i) for i in range(26)]),
            evaluator_root='/exact/frozen/evaluator', project='/exact/project')
        self.events=[]
        self.proof=dict(epochs=250, checkpoint='checkpoint_best.pth', best_sha256='d'*64,
            validation_cases=self.native['split']['outer_val'], fold=str(self.root/'own_fold'))
        self.results=ModuleType('tools.basic_cp_ablation_completed_result')
        self.results.completed_report=Mock(side_effect=lambda *a:self.record('report', {'schema':1}))
        self.results.archive_completed=Mock(side_effect=lambda *a:self.record('archive', {'complete':True}))
        self.start(patch.dict(sys.modules, {'tools.basic_cp_ablation_completed_result':self.results}))
        self.guard=self.start(patch.object(runner, 'resource_guard', side_effect=lambda *a,**kw:self.record('guard_idle' if kw.get('idle') else 'guard', {})))
        self.prepare_bank=self.start(patch.object(runner, 'prepare_bank', side_effect=lambda value:self.record('prepare_bank',value)))
        self.prepare=self.start(patch.object(runner, 'prepare', side_effect=lambda value:self.record('prepare',self.native)))
        self.states=[]
        self.start(patch.object(runner, 'status', side_effect=lambda req,phase,state,**values:self.states.append((phase,state,values))))
        self.start(patch.object(runner, 'publish', side_effect=lambda *a:self.record('publish',None)))
        self.start(patch.object(runner, 'read', side_effect=self.read))
        self.child=self.start(patch.object(runner, 'child', side_effect=self.run_child))
        self.actor=self.start(patch.object(runner, 'run_actor', side_effect=lambda *a:self.record('historical_metrics',None)))
        self.environment=self.start(patch.object(runner, 'environment', return_value={'CUDA_VISIBLE_DEVICES':'6'}))

    def start(self, patcher):
        value=patcher.start(); self.addCleanup(patcher.stop); return value

    def record(self, name, value):
        self.events.append(name); return value

    def read(self, path):
        if Path(path).name=='trial.json':return dict(accepted=True, physical_batch=2 if 'B2' in str(path) else 4)
        if Path(path).name=='training_and_BEST26_complete.json':return copy.deepcopy(self.proof)
        raise AssertionError('Unexpected unverified pipeline input: '+str(path))

    def run_child(self, native, stage, **kwargs):
        label=stage+str(kwargs.get('batch',''))
        return self.record(label,None)

    def test_full_preflight_debug_train_best26_metrics_report_order(self):
        runner.run(self.request)
        ordered=[x for x in self.events if x not in ('guard','guard_idle','publish')]
        self.assertEqual(ordered, ['prepare_bank','prepare','preflight','debug2','debug4','train',
            'historical_metrics','report','archive'])
        final=self.states[-1]
        self.assertEqual(final[:2],('complete','COMPLETE'))
        self.assertEqual(final[2]['completed_epochs'],250)
        self.assertEqual(final[2]['evaluated_cases'],26)
        command=self.actor.call_args.args[1]
        self.assertEqual(command[command.index('--checkpoint')+1],'BEST')
        self.assertEqual(command[command.index('--prediction')+1],str(Path(self.proof['fold'])/'validation'))

    def test_first_preflight_failure_never_starts_debug_training_or_evaluation(self):
        self.child.side_effect=RuntimeError('actual CP geometry failed')
        with self.assertRaisesRegex(RuntimeError,'geometry'):
            runner.run(self.request)
        self.assertEqual(self.child.call_count,1)
        self.actor.assert_not_called()
        self.results.completed_report.assert_not_called()
        self.assertNotIn(('complete','COMPLETE'),[(p,s) for p,s,_ in self.states])

    def test_failed_original_batch2_debug_is_not_skipped_or_replaced(self):
        previous=self.read
        def read(path):
            value=previous(path)
            if Path(path).name=='trial.json' and 'B2' in str(path):value['accepted']=False
            return value
        with patch.object(runner,'read',side_effect=read), self.assertRaisesRegex(RuntimeError,'batch2'):
            runner.run(self.request)
        self.assertNotIn('train',self.events)
        self.actor.assert_not_called()
        self.results.completed_report.assert_not_called()

    def test_native_training_failure_never_generates_complete_or_metric_outputs(self):
        def child(native,stage,**kwargs):
            if stage=='train':raise RuntimeError('own training failed')
            return self.run_child(native,stage,**kwargs)
        self.child.side_effect=child
        with self.assertRaisesRegex(RuntimeError,'training failed'):
            runner.run(self.request)
        self.actor.assert_not_called()
        self.results.completed_report.assert_not_called()
        self.assertFalse(any(s=='COMPLETE' for _,s,_ in self.states))

    def test_archival_failure_is_not_reported_as_pipeline_complete(self):
        self.results.archive_completed.side_effect=OSError('persistent archive unavailable')
        with self.assertRaisesRegex(OSError,'archive unavailable'):
            runner.run(self.request)
        self.assertFalse(any(s=='COMPLETE' for _,s,_ in self.states))

    def test_resource_rejection_prevents_first_subprocess_and_train(self):
        self.guard.side_effect=RuntimeError('host lifetime boundary')
        with self.assertRaisesRegex(RuntimeError,'lifetime'):
            runner.run(self.request)
        self.prepare_bank.assert_not_called();self.prepare.assert_not_called()
        self.child.assert_not_called();self.actor.assert_not_called()


class NativeGuardIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name)
        self.native=dict(root=str(self.root),native_root=str(self.root),trainer='ExactBasic',
            bank_sha256='a'*64,split={'outer_val':['val'+str(i) for i in range(26)]},
            baseline={'dataset_name':'Dataset730','plans_name':'OriginalPlans'})
        self.calls=[]
        calls=self.calls
        class Trainer:
            def initialize(self):return None
            def on_epoch_end(self):return None
            def train_step(self,batch,*,marker):
                calls.append(('train_math',batch,marker));return {'loss':batch['value']+1}
            def validation_step(self,batch,*,marker):
                calls.append(('validation_math',batch,marker));return {'val':marker}
            def perform_actual_validation(self,*,save_probabilities=False):
                calls.append(('inference',save_probabilities));return 26
            def load_checkpoint(self,path):return path
        self.trainer=Trainer
        self.module=SimpleNamespace(ExactBasic=Trainer)
        self.start(patch.object(runner,'initialize_native',return_value=(self.module,{'full_preprocessed_cases':131})))
        self.start(patch.object(runner,'resource_guard',side_effect=lambda *a,**kw:self.calls.append(('guard',)) or {}))
        torch=ModuleType('torch');torch.load=Mock(return_value={'current_epoch':250})
        torch.cuda=SimpleNamespace(max_memory_allocated=lambda:0)
        self.torch=torch
        runmodule=ModuleType('nnunetv2.run.run_training')
        runmodule.run_training_entry=Mock()
        self.entry=runmodule.run_training_entry
        self.start(patch.dict(sys.modules,{'torch':torch,'nnunetv2.run.run_training':runmodule}))
        self.start(patch.object(runner,'publish'))
        self.start(patch.object(runner,'sha',return_value='d'*64))
        self.receipt=self.start(patch.object(best,'validate_best_validation_receipt',return_value={
            'checkpoint_sha256':'e'*64,'validation_cases':self.native['split']['outer_val']}))
        self.start(patch.object(best,'_install',side_effect=self.install_best_wrapper))
        self.original_argv=list(sys.argv);self.addCleanup(lambda:setattr(sys,'argv',self.original_argv))
        self.addCleanup(lambda:direct._NATIVE_STEP_BINDINGS.pop(Trainer,None))

    def start(self,patcher):
        value=patcher.start();self.addCleanup(patcher.stop);return value

    def install_best_wrapper(self,cls,cases,*,bank_sha256):
        original=cls.perform_actual_validation
        def best_validation(self,*args,**kwargs):
            self._best_used=True
            return original(self,*args,**kwargs)
        cls.perform_actual_validation=best_validation
        self.calls.append(('BEST_guard_installed',))
        return {'checkpoint':'checkpoint_best.pth','cases':cases,'bank_sha256':bank_sha256}

    def test_best_and_resource_guards_compose_without_invalidating_math_methods(self):
        def actual_entry():
            trainer=self.trainer()
            self.assertEqual(trainer.train_step({'value':2},marker=7),{'loss':3})
            self.assertEqual(trainer.validation_step({'value':4},marker=8),{'val':8})
            self.assertEqual(trainer.perform_actual_validation(save_probabilities=True),26)
            self.assertTrue(trainer._best_used)
        self.entry.side_effect=actual_entry
        runner.train(self.native)
        self.assertIn(('train_math',{'value':2},7),self.calls)
        self.assertIn(('validation_math',{'value':4},8),self.calls)
        self.assertIn(('inference',True),self.calls)
        self.assertEqual(sum(c[0]=='guard' for c in self.calls),5)
        self.assertIn('--val_best',sys.argv)

    def test_own_best_receipt_is_required_and_receives_exact_outer26_bank(self):
        self.receipt.side_effect=ValueError('own BEST receipt missing')
        with self.assertRaisesRegex(ValueError,'own BEST'):
            runner.train(self.native)
        self.torch.load.assert_not_called()
        self.assertEqual(self.receipt.call_args.kwargs['expected_cases'],self.native['split']['outer_val'])
        self.assertEqual(self.receipt.call_args.kwargs['bank_sha256'],'a'*64)
        self.assertEqual(Path(self.receipt.call_args.kwargs['fold']).parent.name,
            'ExactBasic__OriginalPlans__3d_fullres')

    def test_short_final_training_witness_is_not_complete(self):
        self.torch.load.return_value={'current_epoch':249}
        with self.assertRaisesRegex(RuntimeError,'full250'):
            runner.train(self.native)


class CandidateBankPreparationTests(unittest.TestCase):
    def test_256_only_derivative_pins_original_bank_and_rejects_partial_sources(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temporary:
            root=Path(temporary)
            binding=root/'binding.json';binding.write_text('{}')
            request=dict(root=str(root),arm='candidates256',bank='/original/index.json',
                bank_sha256='a'*64,candidate_binding=str(binding),candidate_binding_sha256=runner.sha(binding))
            def actor(*args):
                folder=root/'bank256';folder.mkdir();(folder/'index.json').write_text(json.dumps(
                    {'source_entries':642,'candidate_count':256}))
            with patch.object(runner,'status'),patch.object(runner,'run_actor',side_effect=actor):
                result=runner.prepare_bank(request)
            self.assertEqual(result['source_bank'],request['bank'])
            self.assertEqual(result['source_bank_sha256'],request['bank_sha256'])
            self.assertNotEqual(result['bank'],request['bank'])
            self.assertEqual(result['bank_sha256'],runner.sha(result['bank']))
            (root/'bank256/index.json').write_text(json.dumps({'source_entries':641,'candidate_count':256}))
            with patch.object(runner,'status'),patch.object(runner,'run_actor'),self.assertRaisesRegex(RuntimeError,'Full642'):
                runner.prepare_bank(request)


if __name__=='__main__':unittest.main()
