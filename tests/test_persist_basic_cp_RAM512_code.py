"""CPU DEBUG actual byte/provenance admission for the explicit new runner."""
import copy
import json
from pathlib import Path
import unittest
from tools import persist_basic_cp_seven as durable
from hiercp_v1x.basic_cp_512_claims import FLAGS
from tests.test_persist_basic_cp_code_exports import ControllerCodeExportDEBUGTests as Fixture


class RAM512ControllerCodeTests(unittest.TestCase):
    def setUp(self):
        Fixture.setUp(self)
        for relative in ('tools/run_basic_cp_RAM512.py','hiercp_v1x/basic_cp_512_claims.py',
                'hiercp_v1x/basic_cp_512_resources.py','hiercp_v1x/basic_cp_large_bank_RAM512.py'):
            path=self.export/relative;path.write_bytes(b'# DEBUG explicit RAM512 file bytes\n')
            self.request['code_sha256'][relative]=durable.digest(path)
        self.request.update(FLAGS)
        self.request['controller_identity']['command'][3]=str(self.export/'tools/run_basic_cp_RAM512.py')
        self.seal()

    def tearDown(self):Fixture.tearDown(self)

    def seal(self):
        self.proof.write_text(json.dumps(dict(code=str(self.export),commit=self.request['commit'],
            code_sha256=self.request['code_sha256'])),encoding='utf8')
        self.manifest['controller_code_exports'][str(self.export)]['sha256']=durable.digest(self.proof)

    def test_explicit_named_full_SHA_export_and512_fields_accept_actual_bytes(self):
        self.assertEqual(durable.request_code(self.request,self.manifest),self.export)

    def test_new_runner_requires512_marker_even_with_real_SHA(self):
        self.request.pop('memory_budget_revision')
        with self.assertRaisesRegex(ValueError,'controller command'):durable.request_code(self.request,self.manifest)

    def test_512_marker_cannot_run_historical800_dispatcher(self):
        self.request['controller_identity']['command'][3]=str(self.export/'tools/run_basic_cp_expanded.py')
        with self.assertRaisesRegex(ValueError,'controller command'):durable.request_code(self.request,self.manifest)

    def test_all_effective_fields_strict_not_boolean_or_old_values(self):
        for key,value in (('memory_budget_revision',True),('aggregate_memory_cap_GiB',800),
                ('effective_native_RSS_limit_GiB',96),('effective_shared_bank_RSS_limit_GiB',384)):
            original=self.request[key];self.request[key]=value
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'TOTAL512'):durable.request_code(self.request,self.manifest)
            self.request[key]=original

    def test_same_group_dynamic_export_still_requires_named_provenance(self):
        self.request['root']=str(self.export.parent/'GPU6_blend50')
        self.request['controller_identity']['command'][-1]=str(Path(self.request['root'])/'request.json')
        with self.assertRaisesRegex(ValueError,'explicitly named'):durable.request_code(self.request)
        self.assertEqual(durable.request_code(self.request,self.manifest),self.export)

    def test_missing_claim_or_provider_pin_rejected_before_admission(self):
        for key in ('hiercp_v1x/basic_cp_512_claims.py','hiercp_v1x/basic_cp_large_bank_RAM512.py'):
            changed=copy.deepcopy(self.request);changed['code_sha256'].pop(key)
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'SHA pins'):durable.request_code(changed,self.manifest)

    def test_mutated_new_runner_rejected_even_with_valid_frozen_export(self):
        (self.export/'tools/run_basic_cp_RAM512.py').write_bytes(b'# DEBUG corruption\n')
        with self.assertRaisesRegex(ValueError,'Frozen per-arm'):durable.request_code(self.request,self.manifest)


if __name__=='__main__':unittest.main()
