"""CPU DEBUG production migration-state/provenance tests; no remote launch."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from tools import recover_basic_cp_RAM512_cohort as builder
SCRIPT=Path(builder.__file__)


class BuilderTests(unittest.TestCase):
    def test_incomplete_latest_remains_full_training_resume(self):
        self.assertEqual(builder.classification(latest_epoch=247,final_epoch=None,best26=False,metrics=False,complete_archive=False),'TRAINING')

    def test_final250_withoutbest_is_validation_pending_not_training_complete(self):
        self.assertEqual(builder.classification(latest_epoch=249,final_epoch=250,best26=False,metrics=False,complete_archive=False),'FINAL250_PENDING_BEST26')

    def test_best26_complete_goes_to_metrics_without_duplicate_validation(self):
        self.assertEqual(builder.classification(latest_epoch=250,final_epoch=250,best26=True,metrics=False,complete_archive=False),'BEST26_READY')

    def test_real_complete_pipeline_retains_original_archived_results(self):
        self.assertEqual(builder.classification(latest_epoch=250,final_epoch=250,best26=True,metrics=True,complete_archive=True),'FULL_COMPLETE')

    def test_false_completed_roles_withoutfull250_orbest_fail(self):
        for values in ((249,None,True,False,False),(250,249,True,True,True),(250,250,False,True,False),(250,250,True,False,True)):
            with self.subTest(values=values),self.assertRaises(ValueError):
                builder.classification(latest_epoch=values[0],final_epoch=values[1],best26=values[2],metrics=values[3],complete_archive=values[4])

    def test_retarget_adds_only512_operational_fields_keeps_science_paths(self):
        old=dict(gpu=6,arm='blend50',native_root='/DEBUG/native',archive_root='/DEBUG/native/archive',
            bank='/DEBUG/bank128/index.json',bank_sha256='a'*64,copies_per_CP_event=1,candidates_per_source=128,
            shared_bank_workers=24,shared_bank_RSS_limit_GiB=384,RSS_limit_bytes=96*2**30,
            baseline={'epochs':250,'physical_batch':2,'patch':[128]*3},PBS_shared_cache_root='/DEBUG/oldcache')
        original=copy.deepcopy(old);new=builder.retarget(old,6,builder.GROUP/'GPU6_blend50',builder.GROUP/'registry',
            {'commit':'b'*40,'code_sha256':{}},builder.GROUP/'banks',builder.GROUP/'GPU5_candidates2048',{'path':'/DEBUG/pending','sha256':'c'*64})
        for key in('native_root','archive_root','bank','bank_sha256','copies_per_CP_event','candidates_per_source','shared_bank_workers','shared_bank_RSS_limit_GiB','RSS_limit_bytes','baseline'):
            self.assertEqual(new[key],old[key])
        self.assertEqual(old,original);self.assertNotIn('PBS_shared_cache_root',new)
        self.assertEqual(new['aggregate_memory_cap_GiB'],512);self.assertEqual(new['effective_native_RSS_limit_GiB'],32)

    def test_actual_original_code_requires_named_export_not_assumedg3(self):
        with tempfile.TemporaryDirectory(prefix='DEBUG_G6_CODE_',dir=SCRIPT.parent)as name:
            folder=Path(name);code=folder/'oldcode';code.mkdir();root=folder/'oldGPU6';root.mkdir()
            proof=folder/'oldexport.json';pins={'DEBUG.py':'a'*64}
            proof.write_text(json.dumps({'code':str(code),'commit':'b'*40,'code_sha256':pins}))
            request={'root':str(root),'commit':'b'*40,'code_sha256':pins,'controller_identity':{'command':['NNpython','-B','-u',str(code/'tools/run_basic_cp_expanded.py'),'--request',str(root/'request.json')]}}
            reference={'path':str(proof),'sha256':builder.sha(proof)}
            manifest={'controller_code_exports':{str(code):reference}}
            self.assertEqual(builder.original_code(request,manifest),(code,reference))
            with self.assertRaises(ValueError):builder.original_code(request,{})
            proof.write_text('{}')
            with self.assertRaises(ValueError):builder.original_code(request,manifest)

    def test_new_runtime_delta_does_not_admit_changed_model_math(self):
        with tempfile.TemporaryDirectory(prefix='DEBUG_G6_DELTA_',dir=SCRIPT.parent)as name:
            folder=Path(name);old=folder/'old';new=folder/'new';old.mkdir();new.mkdir()
            relative='model.py';(old/relative).write_bytes(b'WIDTH = 128\n');(new/relative).write_bytes(b'WIDTH = 64\n')
            oldpins={relative:builder.sha(old/relative)};newpins={relative:builder.sha(new/relative)}
            for path in builder.NEW_RUNTIME_FILES:
                file=new/path;file.parent.mkdir(parents=True,exist_ok=True);file.write_bytes(b'# DEBUG explicit operational addition\n');newpins[path]=builder.sha(file)
            with self.assertRaisesRegex(ValueError,'Scientific/model/data'):
                builder.audit_code_delta({'code_sha256':oldpins},old,{'code':str(new),'code_sha256':newpins})
            (new/relative).write_bytes(b'WIDTH = 128\r\n');newpins[relative]=builder.sha(new/relative)
            report=builder.audit_code_delta({'code_sha256':oldpins},old,{'code':str(new),'code_sha256':newpins})
            self.assertTrue(report[relative]['normalized_bytes_and_AST_equal'])

    def test_no_signal_or_checkpoint_copy_site_in_prepared_builder(self):
        tree=ast.parse(SCRIPT.read_text(encoding='utf8'))
        names=[n.func.attr for n in ast.walk(tree)if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)]
        self.assertNotIn('kill',names);self.assertNotIn('terminate',names);self.assertNotIn('send_signal',names)
        self.assertNotIn('torch.save',SCRIPT.read_text(encoding='utf8'))
        self.assertIn('checkpoint_bytes_copied=False',SCRIPT.read_text(encoding='utf8'))


if __name__=='__main__':unittest.main()
