"""DEBUG fixtures test real export bytes and sealed provenance, never training."""
from __future__ import annotations
import copy,json,os
from pathlib import Path
import tempfile,unittest
from unittest.mock import patch
from tools import persist_basic_cp_seven as durable


class ControllerCodeExportDEBUGTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory(prefix='DEBUG_CODE_EXPORT_',dir=Path(__file__).parent)
        self.root=Path(self.temporary.name).resolve()
        self.export=self.root/'preceding_group/code'; self.arm=self.root/'new_group/GPU6_blend50'
        self.arm.mkdir(parents=True);(self.export/'tools').mkdir(parents=True)
        (self.export/'hiercp_v1x').mkdir()
        (self.export/'tools/run_basic_cp_expanded.py').write_bytes(b'# DEBUG actual file bytes\n')
        (self.export/'hiercp_v1x/DEBUG_state.py').write_bytes(b'DEBUG_ONLY = True\n')
        pins={str(p.relative_to(self.export).as_posix()):durable.digest(p) for p in self.export.rglob('*.py')}
        self.request=dict(root=str(self.arm),commit='a'*40,code_sha256=pins,controller_identity={'command':[
            '/home/aicompetition06/.conda/envs/nnunet/bin/python3.10','-B','-u',
            str(self.export/'tools/run_basic_cp_expanded.py'),'--request',str(self.arm/'request.json')]})
        self.proof=self.root/'preceding_group/export.json'
        self.proof.write_text(json.dumps(dict(code=str(self.export),commit=self.request['commit'],code_sha256=pins)),encoding='utf8')
        self.manifest={'controller_code_exports':{str(self.export):dict(path=str(self.proof),sha256=durable.digest(self.proof))}}
        # Windows DEBUG adapter supplies the documented Linux UID for actual
        # pathlib stat results. All path resolution/symlink/file/SHA checks run.
        actual_stat=Path.stat
        def own_stat(path,*args,**kwargs):
            result=actual_stat(path,*args,**kwargs)
            if os.name!='nt':return result
            values=list(result);values[4]=41833;return os.stat_result(values)
        self.stat=patch.object(Path,'stat',new=own_stat);self.stat.start()
        self.uid=patch.object(durable,'UID',41833 if os.name=='nt'else os.getuid());self.uid.start()

    def tearDown(self):
        self.uid.stop();self.stat.stop();self.temporary.cleanup()

    def test_named_external_export_full_proof_and_actual_bytes_admitted(self):
        self.assertEqual(durable.request_code(self.request,self.manifest),self.export)

    def test_external_without_exact_manifest_entry_rejected(self):
        with self.assertRaisesRegex(ValueError,'explicitly named'):durable.request_code(self.request)
        wrong=copy.deepcopy(self.manifest);wrong['controller_code_exports']['unnamed']=wrong['controller_code_exports'].pop(str(self.export))
        with self.assertRaisesRegex(ValueError,'explicitly named'):durable.request_code(self.request,wrong)

    def test_proof_SHA_mismatch_rejected(self):
        self.proof.write_bytes(b'{"DEBUG_CHANGED":true}')
        with self.assertRaisesRegex(ValueError,'proof SHA'):durable.request_code(self.request,self.manifest)

    def test_same_bytes_proof_wrong_commit_or_code_rejected(self):
        original=durable.read(self.proof)
        for key,value in(('commit','b'*40),('code',str(self.arm)),('code_sha256',{'tools/run_basic_cp_expanded.py':self.request['code_sha256']['tools/run_basic_cp_expanded.py']})):
            bad=dict(original,**{key:value});self.proof.write_text(json.dumps(bad),encoding='utf8')
            self.manifest['controller_code_exports'][str(self.export)]['sha256']=durable.digest(self.proof)
            with self.assertRaisesRegex(ValueError,'code/commit/full pins'):durable.request_code(self.request,self.manifest)

    def test_changed_secondary_source_rejected_despite_valid_export_metadata(self):
        (self.export/'hiercp_v1x/DEBUG_state.py').write_bytes(b'DEBUG_ONLY = False\n')
        with self.assertRaisesRegex(ValueError,'Frozen per-arm'):durable.request_code(self.request,self.manifest)

    def test_original_same_group_path_keeps_existing_api_and_full_SHA_check(self):
        request=copy.deepcopy(self.request);request['root']=str(self.export.parent/'GPU6_blend50')
        request['controller_identity']['command'][-1]=str(Path(request['root'])/'request.json')
        self.assertEqual(durable.request_code(request),self.export)
        (self.export/'tools/run_basic_cp_expanded.py').write_bytes(b'# DEBUG mutation\n')
        with self.assertRaisesRegex(ValueError,'Frozen per-arm'):durable.request_code(request)

    def test_parent_traversal_source_pin_rejected(self):
        self.request['code_sha256']['../export.json']=durable.digest(self.proof)
        self.proof.write_text(json.dumps(dict(code=str(self.export),commit=self.request['commit'],code_sha256=self.request['code_sha256'])),encoding='utf8')
        self.manifest['controller_code_exports'][str(self.export)]['sha256']=durable.digest(self.proof)
        with self.assertRaisesRegex(ValueError,'Frozen per-arm'):durable.request_code(self.request,self.manifest)

    def test_wrong_controller_command_target_rejected(self):
        self.request['controller_identity']['command'][-1]=str(self.root/'foreign/request.json')
        with self.assertRaisesRegex(ValueError,'controller code command'):durable.request_code(self.request,self.manifest)

    def test_export_descriptor_must_be_exact_two_fields(self):
        self.manifest['controller_code_exports'][str(self.export)]['unverified_fallback']=True
        with self.assertRaisesRegex(ValueError,'explicitly named'):durable.request_code(self.request,self.manifest)

    def test_actual_wrong_UID_is_rejected(self):
        with patch.object(durable,'UID',123456):
            with self.assertRaisesRegex(ValueError,'Own UID'):durable.request_code(self.request,self.manifest)

    def test_symlink_evidence_is_rejected(self):
        # DEBUG Windows filesystem API adapter exercises the actual symlink
        # rejection without requiring a system symlink-creation privilege.
        original=Path.is_symlink
        def is_symlink(path):return path==self.proof or original(path)
        with patch.object(Path,'is_symlink',new=is_symlink):
            with self.assertRaisesRegex(ValueError,'non-symlink'):durable.request_code(self.request,self.manifest)

    def test_unpublished_secondary_source_file_is_rejected(self):
        (self.export/'hiercp_v1x/DEBUG_state.py').unlink()
        with self.assertRaises(FileNotFoundError):durable.request_code(self.request,self.manifest)


if __name__=='__main__':unittest.main()
