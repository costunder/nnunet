"""CPU DEBUG safeguards for prepared common-ledger recovery; no main() executed."""
from pathlib import Path
import ast
import copy
import importlib.util
import types
import unittest
from unittest.mock import patch

FILE=Path(__file__).resolve().parents[1]/'tools/recover_basic_cp_common_cohort.py'
spec=importlib.util.spec_from_file_location('common_prepared',FILE)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class CommonG5Tests(unittest.TestCase):
    def request(self,gpu=5):
        return dict(root='/old/5',native_root='/old/private_native',archive_root='/old/archive',gpu=gpu,arm=module.ARMS[gpu],
            gpu_uuid='actual_UUID',copies_per_CP_event=1,candidates_per_source=2048,bank='/original/index.json',bank_sha256='a'*64,
            cpu_affinity=list(range(gpu*8,gpu*8+8)),RSS_limit_bytes=96*2**30,runtime_reserve_bytes=10*2**30,
            shared_bank_workers=24,shared_bank_CPU_affinity=list(range(64)),shared_bank_RSS_limit_GiB=384,
            previous_partial_banks='/old/banks',previous_producer_request='/old/5/request.json',previous_producer_request_sha256='b'*64,
            interrupted_candidate_recovery_snapshot='/old/snapshot.json',interrupted_candidate_recovery_snapshot_sha256='c'*64,
            interrupted_candidate_recovery_revision=1,PBS_shared_cache_root='/stale/shared_cache',shared_banks='/old/banks',producer_root='/old/5')

    def updates(self):
        values=self.request()
        return dict(format='basic_cp_current_candidate_source_recovery_updates_v1',
            request_updates={key:values[key]for key in module.SOURCE_UPDATE_KEYS})

    def test_source_updates_change_only_five_reviewed_provenance_paths(self):
        original=self.request();before=copy.deepcopy(original);updates=self.updates()
        updates['request_updates']['previous_partial_banks']='/current/banks'
        result=module.apply_source_updates(original,updates)
        self.assertEqual(result['previous_partial_banks'],'/current/banks');self.assertEqual(original,before)
        for key in module.SCIENCE_KEYS:self.assertEqual(original.get(key),result.get(key))

    def test_source_update_cannot_shrink_counts_workers_or_memory(self):
        for key,value in(('candidates_per_source',512),('shared_bank_workers',1),('shared_bank_RSS_limit_GiB',96)):
            updates=self.updates();updates['request_updates'][key]=value
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'Exactly five'):
                module.apply_source_updates(self.request(),updates)

    def test_source_update_missing_provenance_or_format_is_not_inferred(self):
        updates=self.updates();updates['request_updates'].pop('previous_producer_request_sha256')
        with self.assertRaises(ValueError):module.apply_source_updates(self.request(),updates)
        updates=self.updates();updates['format']='unknown'
        with self.assertRaises(ValueError):module.apply_source_updates(self.request(),updates)

    def test_retarget_preserves_native_runtime_paths_and_clears_only_stale_PBS_cache(self):
        original=self.request();before=copy.deepcopy(original)
        result=module.retarget(original,5,Path('/new/5'),Path('/new/registry'),dict(code_sha256={'same':'sha'},commit='real'),Path('/new/banks'),Path('/new/5'))
        self.assertEqual(result['native_root'],original['native_root']);self.assertEqual(result['archive_root'],original['archive_root'])
        self.assertNotIn('PBS_shared_cache_root',result);self.assertEqual(original,before)
        self.assertEqual(result['shared_banks'],str(Path('/new/banks')))

    def test_exact_gpu_arm_remains_required(self):
        with self.assertRaisesRegex(ValueError,'seven arm'):
            module.retarget(self.request(5),6,Path('/new/6'),Path('/new/registry'),dict(code_sha256={},commit='x'),Path('/new/banks'),Path('/new/5'))

    def test_live_old_actor_is_blocking_and_never_signaled(self):
        process=types.SimpleNamespace(create_time=lambda:7.,uids=lambda:types.SimpleNamespace(real=41833),status=lambda:module.psutil.STATUS_RUNNING)
        with patch.object(module.psutil,'Process',return_value=process):
            with self.assertRaisesRegex(ValueError,'still live'):module.released(dict(pid=9,create_time=7.,UID=41833,command=['old']))

    def test_exact_terminal_controller_with_live_worker_remains_blocking(self):
        child=types.SimpleNamespace(status=lambda:module.psutil.STATUS_RUNNING)
        process=types.SimpleNamespace(create_time=lambda:7.,uids=lambda:types.SimpleNamespace(real=41833),
            status=lambda:module.psutil.STATUS_ZOMBIE,children=lambda recursive:[child])
        with patch.object(module.psutil,'Process',return_value=process):
            with self.assertRaisesRegex(ValueError,'live descendant'):module.released(dict(pid=9,create_time=7.,UID=41833,command=['old']))

    def test_nested_release_proof_keeps_exact_controller_stage_and_DA_identity(self):
        value={'controller':dict(pid=9,create_time=7.,UID=41833,command=['ctrl']),
            'workers':[dict(pid=10,birth=8.,UID=41833,command=['native'])]}
        self.assertEqual(module.extract_identities(value),[dict(pid=9,create_time=7.,UID=41833,command=['ctrl']),
            dict(pid=10,create_time=8.,UID=41833,command=['native'])])

    def test_full_model_case_identities_and_batch_never_shrink(self):
        model=dict(trainer='Actual',epochs=250,physical_batch=2,effective_batch=2,gradient_accumulation=1,patch_size=[128]*3,
            workers=4,debug=False,subset=False,pretrained_segmentation_used=False,
            actual_train_cases=list(range(105)),actual_validation_cases=list(range(26)))
        native=dict(trainer='Actual',split=dict(outer_train=list(range(105)),outer_val=list(range(26))))
        module.validate_model(model,native)
        model['physical_batch']=1
        with self.assertRaisesRegex(ValueError,'Actual full model'):module.validate_model(model,native)

    def test_script_has_no_signal_checkpoint_copy_or_precreated_bank(self):
        tree=ast.parse(FILE.read_text());calls=[ast.unparse(n.func)for n in ast.walk(tree)if isinstance(n,ast.Call)]
        self.assertNotIn('torch.save',calls)
        for name in calls:self.assertFalse(name.endswith(('.kill','.terminate','.send_signal')))
        self.assertNotIn('banks.mkdir',calls)
        text=FILE.read_text();self.assertIn('allowed_previous_resource_registries={}',text)
        self.assertIn('fixed_aggregate_capacity_bytes=800*2**30',text)
        self.assertIn("map_location='cpu'",text);self.assertIn('_state(saved,epoch',text)
        self.assertNotIn('EPOCH=199',text)


if __name__=='__main__':unittest.main()
