"""DEBUG counterexamples for F01/F02/O01/D02; no production scale reduction."""
import copy
import os
from pathlib import Path
import random
import sys
import tempfile
import threading
from types import SimpleNamespace,ModuleType
import unittest
from unittest.mock import patch
import numpy as np
import torch
from tools.v22_online_runtime import admit_cuda_workspace,scoring_runtime,backend_state


class OnlineReviewTests(unittest.TestCase):
    def test_launcher_resume_loads_selected_checkpoint_and_missing_refuses(self):
        import json
        from unittest.mock import Mock
        import nnunetv2.run.run_training as runner
        from tools.train_v22_online_rank import main
        from tools.v22_online_checkpoint import run_contract
        with tempfile.TemporaryDirectory() as folder,patch.dict(os.environ,{},clear=False):
            root=Path(folder);plans=root/'plans.json';plans.write_text(json.dumps({'configurations':{'3d_fullres':{'batch_size':2}}}))
            pre=root/'pre';pre.mkdir();(pre/'dataset.json').write_text('{}')
            native=dict(dataset_name='Dataset999_DEBUG',plans=str(plans),raw=str(root/'raw'),preprocessed=str(pre))
            native_path=root/'native.json';native_path.write_text(json.dumps(native))
            meta=dict(native_preparation=str(native_path),checkpoint_sha256='gnn',native_sha256='native',split={'outer_train':['a']},selection='s',online_identity={})
            bank=root/'bank.json';bank.write_text(json.dumps(meta));results=root/'results'
            name='nnUNetTrainer_250epochs_OnlineRankV22'
            fold=results/native['dataset_name']/f'{name}__plans__3d_fullres'/'fold_0';fold.mkdir(parents=True)
            contract=run_contract(bank,meta,native,8,json.loads(plans.read_text()))
            value=dict(online_run_contract=contract,trainer_name=name,current_epoch=10,network_weights={},optimizer_state={},logging={},grad_scaler_state=None,
                _best_ema=None,init_args={},inference_allowed_mirroring_axes=None,
                online_rng=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=[]))
            selected=fold/'checkpoint_latest.pth';torch.save(value,selected)
            trainer=SimpleNamespace(load_checkpoint=Mock())
            def execute(*a,**kw):
                self.assertTrue(kw['continue_training']);runner.maybe_load_checkpoint(trainer,True,False,None)
            args=['train','--bank',str(bank),'--results',str(results),'--workers','8','--resume']
            with patch.object(sys,'argv',args),patch('tools.v22_online_rank_bank.validate_catalog',side_effect=lambda x:x),patch('hiercp_v222.contracts.validate_native',side_effect=lambda x:x),patch('torch.cuda.is_available',return_value=True),patch('torch.cuda.device_count',return_value=1),patch.object(runner,'run_training',side_effect=execute):
                main();trainer.load_checkpoint.assert_called_once_with(selected.resolve())
                selected.unlink()  # Only this test's own temporary fixture.
                with self.assertRaises(FileNotFoundError):main()
            with patch.object(sys,'argv',args[:-1]),patch('tools.v22_online_rank_bank.validate_catalog',side_effect=lambda x:x),patch('hiercp_v222.contracts.validate_native',side_effect=lambda x:x):
                with self.assertRaises(FileExistsError):main()

    def test_local_engine_ignores_absent_same_and_wrong_installed_module(self):
        from tools.v22_online_rank_adapter import RankedBank
        from custom_trainers.onlinecp_raw_bank import RawBankStore
        from custom_trainers.onlinecp_raw_resampling import apply_candidate
        store_name='nnunetv2.training.nnUNetTrainer.onlinecp_raw_bank'
        apply_name='nnunetv2.training.nnUNetTrainer.onlinecp_raw_resampling'
        for state in ('absent','same','different'):
            with self.subTest(state=state),tempfile.TemporaryDirectory() as folder:
                def wrong(*a,**k):raise AssertionError('Unreviewed installed engine used')
                a=ModuleType(apply_name);a.apply_candidate=apply_candidate if state=='same' else wrong
                s=ModuleType(store_name);s.RawBankStore=RawBankStore if state=='same' else wrong
                with patch.dict(sys.modules,{apply_name:None if state=='absent' else a,store_name:None if state=='absent' else s}):
                    bank=RankedBank.__new__(RankedBank);bank.root=Path(folder);bank._raw_store=None
                    self.assertIs(bank.raw_apply_function(),apply_candidate)
                    self.assertIs(type(bank._get_raw_store()),RawBankStore)
                    for row in bank.paste_engine_identity().values():
                        self.assertIn('custom_trainers',row['path']);self.assertEqual(len(row['sha256']),64)
                    bank._get_raw_store().close()

    def test_cuda_workspace_refuses_late_or_conflicting_admission(self):
        with patch.dict(os.environ,{},clear=False),patch('torch.cuda.is_initialized',return_value=True):
            os.environ.pop('CUBLAS_WORKSPACE_CONFIG',None)
            with self.assertRaisesRegex(RuntimeError,'already initialized'):admit_cuda_workspace()
        with patch.dict(os.environ,{'CUBLAS_WORKSPACE_CONFIG':':16:8'}),patch('torch.cuda.is_initialized',return_value=False):
            with self.assertRaisesRegex(RuntimeError,'Unexpected'):admit_cuda_workspace()

    def test_runtime_restores_rng_backend_on_success_and_error(self):
        from hiercp_v222.v1_cache import configuration
        _,base=configuration();admit_cuda_workspace()
        for failure in (False,True):
            before=backend_state();cpu=torch.get_rng_state();py=random.getstate();npstate=np.random.get_state()
            gpu=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
            try:
                with scoring_runtime(base,42):
                    state=backend_state();self.assertTrue(state['deterministic']);self.assertFalse(state['matmul_tf32'])
                    self.assertFalse(state['benchmark']);self.assertFalse(state['cudnn_tf32'])
                    random.random();np.random.random();torch.rand(3)
                    if gpu:torch.rand(3,device='cuda')
                    if failure:raise LookupError('DEBUG scoring failure')
            except LookupError:
                self.assertTrue(failure)
            self.assertEqual(before,backend_state());self.assertTrue(torch.equal(cpu,torch.get_rng_state()))
            self.assertEqual(py,random.getstate());np.testing.assert_equal(npstate,np.random.get_state())
            for a,b in zip(gpu,torch.cuda.get_rng_state_all() if gpu else []):self.assertTrue(torch.equal(a,b))

    def test_storage_refusal_happens_before_writer(self):
        from tools.v22_online_storage import save_online_case,save_online_candidate
        with tempfile.TemporaryDirectory() as folder,patch('tools.v22_online_storage.shutil.disk_usage',return_value=SimpleNamespace(free=1)):
            for function,writer in ((save_online_case,'save_case'),(save_online_candidate,'save_candidate')):
                with patch('custom_trainers.onlinecp_raw_bank.'+writer) as save:
                    with self.assertRaisesRegex(OSError,'not a no-placement'):function(folder,'x.json',{'x':np.ones(8)},0)
                    save.assert_not_called()
            self.assertEqual(list(Path(folder).iterdir()),[])

    def test_resume_no_missing_checkpoint_fallback_and_contract_mutation(self):
        from tools.v22_online_checkpoint import choose_checkpoint,validate_checkpoint
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(FileNotFoundError):choose_checkpoint(folder)
            with self.assertRaises(FileNotFoundError):choose_checkpoint(folder,Path(folder)/'missing.pth')
        expected=dict(epochs=250,seed=42,workers=8,plans={'a':1},split={'train':['a']},gnn_sha256='x',paste_engine={'a':'x'})
        value=dict(online_run_contract=expected,trainer_name='nnUNetTrainer_250epochs_OnlineRankV22',current_epoch=10,
            network_weights={},optimizer_state={},logging={},grad_scaler_state=None,
            _best_ema=None,init_args={},inference_allowed_mirroring_axes=None,
            online_rng=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=[]))
        validate_checkpoint(value,expected)
        for key in expected:
            bad=copy.deepcopy(value);bad['online_run_contract'][key]='changed'
            with self.assertRaises(ValueError):validate_checkpoint(bad,expected)

    def test_checkpoint_restores_actual_weights_optimizer_epoch_rng(self):
        from tools.v22_online_checkpoint import OnlineCheckpointMixin
        # CPU DEBUG optimization fixture, not a segmentation performance result.
        class nnUNetTrainer_250epochs_OnlineRankV22(OnlineCheckpointMixin):
            def _online_contract(self):return dict(epochs=250,seed=42)
        owner=nnUNetTrainer_250epochs_OnlineRankV22()
        owner.local_rank=0;owner.disable_checkpointing=False;owner.is_ddp=False;owner.was_initialized=True
        owner.network=torch.nn.Linear(4,2);owner.optimizer=torch.optim.Adam(owner.network.parameters())
        owner.network(torch.ones(3,4)).sum().backward();owner.optimizer.step();owner.optimizer.zero_grad()
        owner.grad_scaler=None;owner.current_epoch=9;owner._best_ema=.4;owner.my_init_kwargs={}
        owner.inference_allowed_mirroring_axes=(0,1,2);owner.ranking_gpu_lock=threading.RLock()
        log={'restored':None};owner.logger=SimpleNamespace(get_checkpoint=lambda:{'epoch':9},load_checkpoint=lambda x:log.update(restored=x))
        weights=copy.deepcopy(owner.network.state_dict());optim=copy.deepcopy(owner.optimizer.state_dict())
        with tempfile.TemporaryDirectory() as folder,patch('torch.cuda.is_available',return_value=False),patch('torch.cuda.device_count',return_value=0):
            path=Path(folder)/'checkpoint_latest.pth';owner.save_checkpoint(path)
            expected=torch.rand(6)
            with torch.no_grad():
                for p in owner.network.parameters():p.add_(10)
            owner.optimizer.state.clear();owner.current_epoch=0
            owner.load_checkpoint(path)
            self.assertTrue(torch.equal(expected,torch.rand(6)));self.assertEqual(owner.current_epoch,10)
            for k,v in weights.items():self.assertTrue(torch.equal(v,owner.network.state_dict()[k]))
            for k,values in optim['state'].items():
                for name,v in values.items():self.assertTrue(torch.equal(v,owner.optimizer.state_dict()['state'][k][name]))
            self.assertEqual(log['restored'],{'epoch':9})


if __name__=='__main__':unittest.main()
