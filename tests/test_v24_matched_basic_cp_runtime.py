"""CPU DEBUG score-overlay identity tests; no medical training/evaluation."""
import copy
from contextlib import ExitStack
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np

from hiercp_v1x import v24_matched_basic_cp_runtime as runtime


# This independent DEBUG implementation follows the installed historical
# source/RNG/paste-plan API. It is never admitted by the production installer.
DEBUG_HISTORICAL_SOURCE = '''import json
from pathlib import Path
from collections import OrderedDict
import numpy as np
class OnlineCPBank:
    def __init__(self, index_path, cache_entries=64):
        self.index_path=Path(index_path);self.root=self.index_path.parent
        self.metadata=json.loads(self.index_path.read_text())
        self.entries_by_case={case:tuple(names) for case,names in self.metadata['entries_by_case'].items()}
        self.cp_probability=self.metadata['cp_probability']
        self._cache=OrderedDict();self._limit=cache_entries
    def entry_names(self, case):
        return self.entries_by_case.get(case,())
    def _load(self, name):
        if name not in self._cache:
            with np.load(self.root/name,allow_pickle=False) as loaded:
                self._cache[name]={key:loaded[key] for key in loaded.files}
            while len(self._cache)>self._limit:self._cache.popitem(last=False)
        return self._cache[name]
    def load_for_case(self, case, entry_index):
        return self._load(self.entry_names(case)[entry_index])
class nnUNetDataLoaderOnlineCP:
    def __init__(self, bank_path, policy, seed=42):
        self.online_bank=OnlineCPBank(bank_path);self.policy=policy
        self.rng=np.random.default_rng(seed)
    def _select_candidate(self,scores,u):
        return int(np.argmax(scores)) if self.policy=='hier_argmax' else min(127,int(np.floor(u*128)))
    def _sample_paste_plan(self,case):
        names=self.online_bank.entry_names(case)
        apply_cp=float(self.rng.random())<self.online_bank.cp_probability
        source_u,candidate_u,scale_u,shift_u=map(float,self.rng.random(4))
        token=(apply_cp and bool(names),source_u.hex(),candidate_u.hex(),scale_u.hex(),shift_u.hex())
        if not apply_cp or not names:return None,token
        source=min(len(names)-1,int(np.floor(source_u*len(names))))
        entry=self.online_bank.load_for_case(case,source)
        index=self._select_candidate(entry['scores'],candidate_u)
        return dict(entry=entry,center=entry['candidate_centers'][index],scale=.95+scale_u*.1,shift=-5+shift_u*10),token
    def generate_train_batch(self):
        return self._sample_paste_plan('DEBUG_train_000')
class _nnUNetTrainer_250epochs_OnlineCP:
    def initialize(self):
        return 'DEBUG original initialization'
    def get_dataloaders(self):
        return 'DEBUG original loader'
'''


def _json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding='utf8')


class MatchedBasicOverlayCpuDebugTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='matched_Basic_CPU_DEBUG_', dir=Path.cwd())
        cls.root = Path(cls.temp.name).resolve()
        cls.bank = cls.root/'original_bank_DEBUG';cls.bank.mkdir()
        (cls.bank/'entries').mkdir()
        cls.train = ['DEBUG_train_%03d' % i for i in range(105)]
        cls.val = ['DEBUG_val_%03d' % i for i in range(26)]
        groups, witnesses, rows = {}, {}, {}
        centers = np.stack((np.arange(128), np.zeros(128), np.zeros(128)), axis=1).astype(np.int32)
        for ci, case in enumerate(cls.train[:81]):
            groups[case] = []
            for component in range(1, (8 if ci < 80 else 2)+1):
                name = 'entries/%s__%03d.npz' % (case, component)
                groups[case].append(name)
                path = cls.bank/name
                np.savez(path, source_data=np.full((1,2,2,2), component, dtype=np.float32),
                    source_mask=np.ones((2,2,2), dtype=np.uint8), anchor_offset=np.array([1,1,1], dtype=np.int16),
                    candidate_centers=centers, candidate_raw_centers=centers+10,
                    scores=np.arange(128, dtype=np.float32),
                    source_component=np.array([component], dtype=np.int16), source_diameter_mm=np.array([10], dtype=np.float32))
                checksum=runtime._sha(path)
                witnesses[name]=dict(case_id=case,source_component=component,sha256=checksum,stat_identity=runtime._stat(path))
                scores=np.zeros(128,dtype=np.float32);scores[component]=3.
                rows[name]=dict(case_id=case,source_component=component,original_sha256=checksum,
                    scores=scores.tolist(),selected_index=component)
        original=dict(format='hiercp_online_bank_v2',entries_by_case=groups,candidate_count=128,
            cp_probability=.5,source_entries=642,eligible_cases=81,total_candidates=82176,
            network_patch_size=[128]*3,tumor_label=2,liver_label=1,maximum_diameter_mm=20.,
            minimum_liver_coverage=.85,intensity_scale_range=[.95,1.05],intensity_shift_range_hu=[-5.,5.])
        _json(cls.bank/'index.json',original)
        cls.document=dict(format=runtime.FORMAT,complete=True,debug=False,
            complete_cases=81,complete_sources=642,complete_positions=82176,only_CP_field_changed='scores',
            original_non_score_payloads_preserved=True,original_source_schedule_and_paste_contract_preserved=True,
            full_P_context=True,recipient_GT_used_in_forward=False,production_optimizer_updates=0,
            model_and_RNG_unchanged=True,split=dict(outer_train=cls.train,outer_val=cls.val),
            pin={'DEBUG':True},model_sha256='a'*64,
            entries_by_case=groups,entries=rows,source_bank=dict(root=str(cls.bank),
                index_sha256=runtime._sha(cls.bank/'index.json'),index_stat_identity=runtime._stat(cls.bank/'index.json'),
                entries=witnesses,complete_cases=81,complete_sources=642,complete_positions=82176))
        cls.counter=0

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def setUp(self):
        type(self).counter+=1
        self.path=self.root/('overlay_DEBUG_%d.json'%self.counter)
        self.document=copy.deepcopy(type(self).document)
        self.seal()

    def seal(self):
        self.document['content_sha256']=runtime._canonical({k:v for k,v in self.document.items() if k!='content_sha256'})
        _json(self.path,self.document)

    def admit(self, **kwargs):
        return runtime._validate_overlay(self.path,expected_index_sha256=None,validate_pin=False,**kwargs)

    def module(self):
        path=self.root/('historical_preprocessed_DEBUG_%d.py'%self.counter)
        path.write_text(DEBUG_HISTORICAL_SOURCE,encoding='utf8')
        name='historical_CP_DEBUG_%d'%self.counter
        spec=importlib.util.spec_from_file_location(name,path)
        module=importlib.util.module_from_spec(spec);sys.modules[name]=module
        self.addCleanup(sys.modules.pop,name,None);spec.loader.exec_module(module)
        self.addCleanup(runtime._BINDINGS.pop,module,None)
        return module

    def install(self,module=None):
        module=self.module() if module is None else module
        binding=self.admit(full_hash=True)
        proof=runtime._install_overlay(module,binding,expected_source_sha256=runtime._sha(module.__file__))
        return module,proof

    def test_complete642_82176_fixture_admitted_without_data_copy(self):
        binding=self.admit(full_hash=True)
        self.assertEqual(len(binding['scores']),642)
        self.assertEqual(sum(len(x) for x in binding['scores'].values()),82176)
        self.assertEqual(len(set(self.train)-set(binding['original']['entries_by_case'])),24)

    def test_production_installer_refuses_debug_index_and_pin(self):
        with self.assertRaises(ValueError):runtime.validate_overlay(self.path)

    def test_overlay_replaces_only_scores_preserves_array_object_identity_and_cache(self):
        module,_=self.install()
        bank=module.OnlineCPBank(self.path)
        name=self.document['entries_by_case'][self.train[0]][0]
        before_sha=runtime._sha(self.bank/name)
        result=bank._load(name);cached=bank._cache[name]
        self.assertNotEqual(result['scores'].tolist(),cached['scores'].tolist())
        for key in cached:
            if key!='scores':self.assertIs(result[key],cached[key])
        self.assertEqual(int(np.argmax(result['scores'])),1)
        self.assertEqual(int(np.argmax(cached['scores'])),127)
        self.assertEqual(runtime._sha(self.bank/name),before_sha)
        self.assertFalse(result['scores'].flags.writeable)

    def test_same_five_rng_draws_source_sequence_and_intensity_with_changed_positions(self):
        basic=self.module()
        baseline=basic.nnUNetDataLoaderOnlineCP(self.bank/'index.json','basic')
        matched,proof=self.install(self.module())
        compared=matched.nnUNetDataLoaderOnlineCP(self.path,'hier_argmax')
        sources=set();locations=set();events=0
        for _ in range(200):
            first,token1=baseline._sample_paste_plan(self.train[0])
            second,token2=compared._sample_paste_plan(self.train[0])
            self.assertEqual(token1,token2)
            self.assertEqual(first is None,second is None)
            if first is not None:
                component=int(first['entry']['source_component'][0])
                self.assertEqual(component,int(second['entry']['source_component'][0]))
                self.assertEqual(first['scale'],second['scale']);self.assertEqual(first['shift'],second['shift'])
                self.assertEqual(int(second['center'][0]),component)
                sources.add(component);locations.add(tuple(second['center']));events+=1
        self.assertEqual(sources,set(range(1,9)));self.assertGreater(len(locations),1)
        self.assertGreater(events,0)
        self.assertEqual(proof['random_draws_per_visit'],5)

    def test_all24_ineligible_cases_return_original_visit_and_keep_rng_alignment(self):
        original=self.module();basic=original.nnUNetDataLoaderOnlineCP(self.bank/'index.json','basic')
        matched,_=self.install(self.module());loader=matched.nnUNetDataLoaderOnlineCP(self.path,'hier_argmax')
        for case in self.train[81:]:
            first,a=basic._sample_paste_plan(case);second,b=loader._sample_paste_plan(case)
            self.assertIsNone(first);self.assertIsNone(second);self.assertEqual(a,b)

    def test_original_loader_callables_and_bytecode_not_changed(self):
        module=self.module();loader=module.nnUNetDataLoaderOnlineCP
        methods={name:getattr(loader,name) for name in ('__init__','_sample_paste_plan','_select_candidate','generate_train_batch')}
        codes={name:method.__code__ for name,method in methods.items()}
        self.install(module)
        for name,method in methods.items():
            self.assertIs(getattr(loader,name),method);self.assertIs(method.__code__,codes[name])

    def test_counters_observe_original_source_loads_without_rng_draws(self):
        module,_=self.install();bank=module.OnlineCPBank(self.path)
        names=self.document['entries_by_case'][self.train[0]]
        bank.load_for_case(self.train[0],0);bank.load_for_case(self.train[0],1)
        counts=runtime.runtime_observation(trainer_module=module)['counters']
        self.assertEqual(counts['source_loads'],2);self.assertEqual(counts['overlay_score_loads'],2)
        self.assertEqual(counts['original_source_visits'],{names[0]:1,names[1]:1})

    def test_drop_source_cannot_turn_into_completed_smaller_bank(self):
        name=next(iter(self.document['entries']));del self.document['entries'][name];self.seal()
        with self.assertRaisesRegex(ValueError,'All642'):self.admit()

    def test_source_order_change_rejected_even_if_json_resealed(self):
        self.document['entries_by_case'][self.train[0]].reverse();self.seal()
        with self.assertRaisesRegex(ValueError,'ordered'):self.admit()

    def test_heldout_source_and_candidate_truncation_are_rejected(self):
        first=next(iter(self.document['entries']));self.document['entries'][first]['case_id']=self.val[0];self.seal()
        with self.assertRaisesRegex(ValueError,'Own-patient'):self.admit()
        self.document=copy.deepcopy(type(self).document)
        self.document['entries'][first]['scores']=self.document['entries'][first]['scores'][:-1];self.seal()
        with self.assertRaisesRegex(ValueError,'Exactly128'):self.admit()

    def test_argmax_substitution_rejected(self):
        first=next(iter(self.document['entries']));self.document['entries'][first]['selected_index']=127;self.seal()
        with self.assertRaisesRegex(ValueError,'argmax'):self.admit()

    def test_neural_score_only_claim_and_recipient_blind_contract_required(self):
        for name,value in [('only_CP_field_changed','source_data'),('recipient_GT_used_in_forward',True),
                ('model_and_RNG_unchanged',False),('production_optimizer_updates',1)]:
            self.document=copy.deepcopy(type(self).document);self.document[name]=value;self.seal()
            with self.assertRaisesRegex(ValueError,'score-only'):self.admit()

    def test_source_witness_substitution_rejected(self):
        name=next(iter(self.document['source_bank']['entries']))
        self.document['source_bank']['entries'][name]['sha256']='b'*64;self.seal()
        with self.assertRaisesRegex(ValueError,'identity'):self.admit(full_hash=True)

    def test_changed_overlay_and_wrong_process_bank_are_rejected(self):
        module,_=self.install();bank=module.OnlineCPBank(self.path)
        with self.assertRaisesRegex(ValueError,'own-arm'):module.OnlineCPBank(self.bank/'index.json')
        self.path.write_text('{}')
        with self.assertRaisesRegex(ValueError,'source changed'):module.OnlineCPBank(self.path)
        self.assertEqual(bank._v24_matched_overlay,runtime._BINDINGS[module]['binding']['sha256'])

    def test_actual_original_function_bytecode_tamper_detected(self):
        module=self.module();self.install(module)
        def replaced(self):return None
        module.nnUNetDataLoaderOnlineCP.generate_train_batch.__code__=replaced.__code__
        with self.assertRaisesRegex(ValueError,'bytecode'):runtime.runtime_observation(trainer_module=module)

    def test_installation_cannot_switch_own_best_overlay(self):
        module,_=self.install()
        name=next(iter(self.document['entries']));self.document['entries'][name]['scores'][1]=4.;self.seal()
        with self.assertRaisesRegex(ValueError,'cannot switch'):
            runtime._install_overlay(module,self.admit(),expected_source_sha256=runtime._sha(module.__file__))

    def test_best_required_before_native_training_entry(self):
        original=sys.argv;sys.argv=['DEBUG_native']
        try:
            with self.assertRaisesRegex(ValueError,'BEST'):runtime.run_training_entry()
        finally:sys.argv=original


class MatchedNativeTrainingStartCpuDebugTests(unittest.TestCase):
    """Execute the real training-only hooks with explicit CPU DEBUG dependencies."""
    def setUp(self):
        from hiercp_v1x import v24_matched_basic_cp_pipeline as pipeline
        from hiercp_v1x import v24_nnunet_cp as resource
        from hiercp_v1x import v24_readonly_native_storage as readonly
        from hiercp_v1x import v24_readonly_static_operator_storage as static
        from hiercp_v1x import v24_native_best_eval_runtime as best
        from hiercp_v1x import v24_direct_gpu_access as direct
        self.temp = tempfile.TemporaryDirectory(prefix='native_training_start_CPU_DEBUG_', dir=Path.cwd())
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        private = self.root/'private_DEBUG'
        trainer_path = private/'nnunetv2/training/nnUNetTrainer/nnUNetTrainer_OnlinePairedCP.py'
        trainer_path.parent.mkdir(parents=True)
        trainer_path.write_text('# CPU DEBUG historical runtime dependency\n', encoding='utf8')
        alias_path = trainer_path.with_name('nnUNetTrainer_FrozenV23CP.py')
        alias_path.write_text('# CPU DEBUG alias dependency\n', encoding='utf8')
        overlay_path, admission_path = self.root/'overlay_DEBUG.json', self.root/'admission_DEBUG.json'
        _json(overlay_path, {'CPU_DEBUG_dependency': True})
        _json(admission_path, {'CPU_DEBUG_dependency': True})
        self.split = dict(outer_train=['DEBUG_train_%03d'%i for i in range(105)],
                          outer_val=['DEBUG_val_%03d'%i for i in range(26)])
        native = dict(root=str(self.root), overlay=str(overlay_path), overlay_sha256=runtime._sha(overlay_path),
            bank=str(overlay_path), bank_sha256=runtime._sha(overlay_path), trainer=runtime.TRAINER,
            physical_GPU=6, split=self.split, private_runtime=str(private),
            baseline=dict(physical_batch=2, patch_size=[128]*3, epochs=250, cp_probability=.5, split=self.split),
            storage_admission=str(admission_path), storage_admission_sha256=runtime._sha(admission_path),
            private_data_folder=str(self.root/'readonly_data_DEBUG'),
            private_trainer_sha256={path.name:runtime._sha(path) for path in (trainer_path,alias_path)})
        self.native_path = self.root/'native_DEBUG.json'
        _json(self.native_path, native)
        self.binding = dict(overlay=dict(split=self.split, pin=dict(physical_GPU=6)))
        self.original_start, self.original_epoch, self.initialize = Mock(return_value='DEBUG native started'), Mock(), Mock()
        owner = self
        class DebugOriginalTrainer:
            def initialize(self):
                owner.initialize();self.was_initialized=True
            def on_train_start(self):
                return owner.original_start(self)
            def on_train_epoch_start(self):
                return owner.original_epoch(self)
            def do_split(self):
                return owner.split['outer_train'], owner.split['outer_val']
        class DebugMatchedTrainer(DebugOriginalTrainer):
            pass
        self.cls = DebugMatchedTrainer
        self.module = types.ModuleType(runtime.MODULE)
        self.module.__file__ = str(trainer_path)
        self.module.nnUNetTrainer_250epochs_OnlineHierCPExactArgmax = DebugOriginalTrainer
        alias = types.ModuleType('nnunetv2.training.nnUNetTrainer.nnUNetTrainer_FrozenV23CP')
        setattr(alias, runtime.TRAINER, self.cls)
        dataset = types.ModuleType('nnunetv2.training.dataloading.nnunet_dataset')
        dataset.infer_dataset_class = Mock()
        dataloading = types.ModuleType('nnunetv2.training.dataloading')
        dataloading.nnunet_dataset = dataset
        self.stack = ExitStack();self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.dict(sys.modules,
            {'nnunetv2.training.dataloading':dataloading,
             'nnunetv2.training.dataloading.nnunet_dataset':dataset}))
        self.stack.enter_context(patch.dict(os.environ, nnUNet_n_proc_DA='4'))
        self.access = Mock()
        self.stage = self.stack.enter_context(patch.object(pipeline, 'early_stage_guard', return_value=self.access))
        self.stack.enter_context(patch.object(runtime, 'validate_overlay', return_value=self.binding))
        self.stack.enter_context(patch.object(runtime.importlib, 'import_module',
            side_effect=lambda name:self.module if name==runtime.MODULE else alias))
        self.dataset_adapter = self.stack.enter_context(patch.object(readonly, 'install_dataset_adapter', return_value={'DEBUG':True}))
        self.overlay_adapter = self.stack.enter_context(patch.object(runtime, 'install_matched_overlay',
            return_value=dict(source_entries=642, scored_candidates=82176, cp_probability=.5)))
        self.best_adapter = self.stack.enter_context(patch.object(best, 'install_best_evaluation_guard', return_value={'BEST_required':True}))
        self.direct_adapter = self.stack.enter_context(patch.object(direct, 'install_native_step_guards', return_value={'DEBUG':True}))
        self.checkpoint_adapter = self.stack.enter_context(patch.object(static, 'install_checkpoint_adapters', return_value={'DEBUG':True}))
        self.binding_guard = self.stack.enter_context(patch.object(runtime, '_guard_binding'))
        self.stack.enter_context(patch.object(runtime, 'runtime_observation', return_value={'DEBUG_original_CP_observation':True}))
        self.pre_resources = dict(actual_affinity=[32,33,34,35], actual_process_and_children_RSS_bytes=1234,
            logical_CPU_cores=4, process_and_children_RSS_bytes=48*2**30,
            host_available_resources_are_not_project_allocation=True)
        self.post_resources = dict(self.pre_resources, actual_process_and_children_RSS_bytes=5678)
        self.budget = self.stack.enter_context(patch.object(resource, 'require_project_budget',
            side_effect=[self.pre_resources,self.post_resources,self.post_resources]))
        self.publish = self.stack.enter_context(patch.object(readonly, 'publish_json'))
        self.trainer = self.cls()
        self.trainer.was_initialized=False
        self.trainer.batch_size=2;self.trainer.num_epochs=250
        self.trainer.configuration_manager=types.SimpleNamespace(patch_size=[128]*3)
        self.trainer.online_policy='hier_argmax'
        self.trainer.num_iterations_per_epoch=250;self.trainer.num_val_iterations_per_epoch=50
        parameters = [types.SimpleNamespace(numel=lambda:16,requires_grad=True),
                      types.SimpleNamespace(numel=lambda:32,requires_grad=False)]
        self.trainer.network = types.SimpleNamespace(parameters=lambda:iter(parameters))

    def test_full_training_start_executes_project_bounded_observation_and_preserves_contract(self):
        proofs = runtime.install_native_runtime(self.native_path, training=True)
        def original_start_after_budget(trainer):
            self.budget.assert_called_once_with()
            self.assertIs(trainer,self.trainer)
            return 'DEBUG native started'
        self.original_start.side_effect=original_start_after_budget
        self.assertEqual(self.trainer.on_train_start(), 'DEBUG native started')
        self.stage.assert_called_once_with(stage_action='train')
        self.access.assert_called_once_with();self.initialize.assert_called_once_with()
        self.original_start.assert_called_once_with(self.trainer)
        self.best_adapter.assert_called_once_with()
        self.direct_adapter.assert_called_once_with(self.cls,self.access)
        self.checkpoint_adapter.assert_called_once()
        self.binding_guard.assert_called_once_with(self.binding)
        self.assertEqual(self.budget.call_count,2)
        self.publish.assert_called_once()
        path, report = self.publish.call_args.args
        self.assertEqual(path,self.root/'actual_nnunet_configuration.json')
        self.assertEqual(report['resources'],self.post_resources)
        expected = dict(train_cases=105, validation_cases=26, epochs=250,
            physical_batch=2, gradient_accumulation=1, effective_batch=2,
            patch_size=[128]*3, CPU_workers=4, cp_probability=.5,
            source_policy='same_patient_uniform_original_sources_per_visit',
            candidate_count=128, source_entries=642, eligible_cases=81,
            validation_checkpoint='checkpoint_best.pth', debug=False, full_training_complete=False,
            parameters=48, trainable_parameters=16)
        for key,value in expected.items():
            with self.subTest(key=key):self.assertEqual(report[key],value)
        self.assertEqual(report['runtime_proofs'],proofs)
        self.trainer.on_train_epoch_start()
        self.original_epoch.assert_called_once_with(self.trainer)
        self.assertEqual(self.budget.call_count,3)

    def test_project_budget_failure_after_loader_start_is_visible_and_not_published(self):
        runtime.install_native_runtime(self.native_path,training=True)
        self.budget.side_effect=[self.pre_resources,MemoryError('DEBUG project+children exceed48GiB')]
        with self.assertRaisesRegex(MemoryError,'project\\+children'):
            self.trainer.on_train_start()
        self.original_start.assert_called_once_with(self.trainer)
        self.publish.assert_not_called()


if __name__=='__main__':unittest.main()
