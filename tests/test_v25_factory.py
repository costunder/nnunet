"""CPU UNIT namespace/startup admission; mocked bundles are not native CT evidence."""
from contextlib import ExitStack
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import uuid

from hiercp_v1x import v25_factory as factory, v25_context

ROOT = Path(__file__).resolve().parents[1]


class V25FactoryAdmissionCPUUnit(unittest.TestCase):
    def setUp(self):
        self.root = ROOT/'tmp'/('UNIT_v25_factory_'+uuid.uuid4().hex)
        self.root.mkdir(parents=True, exist_ok=False)
        self.addCleanup(self.cleanup_owned)
        self.bank = self.root/'UNIT_bank.pt'
        self.bank.write_bytes(b'CPU UNIT bank witness, not a production bank')
        self.bundle = SimpleNamespace(source=ROOT, config=dict(
            graph=dict(context_inner_radius_mm=2., context_outer_radius_mm=10.),
            ct_clip=[-200.,250.], cache=dict(source_pad=2)))
        self.signature = dict(source_proof=dict(verified_files={}),
            bank_path=str(self.bank), bank_sha256='1'*64)
        self.inventory = dict(raw_records=[], records=[], split=dict(inner_train=[], inner_val=[]))
        self.runtime = dict(workers=4, raw_resident_gib=8, rss_gib=64,
            geometry_resident_gib=8, minimum_free_disk_gib=10)
        self.events = []

    def cleanup_owned(self):
        if self.root.parent != ROOT/'tmp' or not self.root.name.startswith('UNIT_v25_factory_'):
            raise ValueError('CPU UNIT cleanup escaped the fresh owned folder')
        for path in sorted(self.root.rglob('*'), key=lambda path:len(path.parts), reverse=True):
            if not path.is_relative_to(self.root) or path.is_symlink():
                raise ValueError('Unexpected CPU UNIT cleanup path')
            path.rmdir() if path.is_dir() else path.unlink()
        self.root.rmdir()

    def construct(self):
        def install(runtime, *, approved_profile):
            self.assertTrue(approved_profile)
            self.events.append('shared_context_installed')
        def graph(values):
            self.assertEqual(self.events, ['shared_context_installed'])
            self.assertEqual(values['context_inner_radius_mm'], 0.)
            self.assertEqual(values['context_outer_radius_mm'], 10.)
            self.events.append('graph_config_created')
            return SimpleNamespace(**values)
        population=SimpleNamespace(partition_cases=lambda *args,**kwargs:[])
        with ExitStack() as stack:
            stack.enter_context(patch('hiercp_v1x.comparison_native_upper_cache._load_geometry_inputs',
                return_value=(self.bundle,self.inventory,{},self.signature,{})))
            stack.enter_context(patch('hiercp_v1x.v23_data.V23Population', return_value=population))
            stack.enter_context(patch('hiercp_v1x.transition_v1_local._runtime', return_value={'CPU_UNIT':True}))
            stack.enter_context(patch('hiercp_v1x.v25_context.install', side_effect=install))
            stack.enter_context(patch('hiercp.schema.graph_config_from_dict', side_effect=graph))
            stack.enter_context(patch('hiercp_v1x.v24_geometry.V24UpperGeometryCache', return_value=object()))
            return factory.V25NativeInputs('CPU_UNIT_experiment','CPU_UNIT_inventory',self.root/'cache',self.runtime)

    def test_context_is_installed_before_first_geometry_and_old_bundle_stays_unchanged(self):
        original=copy.deepcopy(self.bundle.config)
        inputs=self.construct()
        self.assertEqual(self.bundle.config, original)
        self.assertEqual(inputs.context_contract, v25_context.identity())
        self.assertEqual(self.events, ['shared_context_installed','graph_config_created'])
        header=inputs._binding_header()
        self.assertEqual(header['format'], factory.INPUT_BINDINGS_FORMAT)
        self.assertEqual(header['context_contract_sha256'],v25_context.identity()['contract_sha256'])
        self.assertNotEqual(header['format'],'v24_immutable_CT_organ_fixed_donor_bindings_v1')

    def test_foreign_old_binding_receipt_refused_before_any_geometry_cache_admission(self):
        cache=self.root/'cache';cache.mkdir()
        path=cache/'input_bindings.json'
        path.write_text(json.dumps(dict(format='v24_immutable_CT_organ_fixed_donor_bindings_v1',complete=True)))
        original=path.read_bytes()
        with self.assertRaisesRegex(ValueError,'binding/source/query proof differs'):
            self.construct()
        self.assertEqual(path.read_bytes(),original)

    def test_unbound_legacy_canonical_index_refused_without_overwrite(self):
        local=self.root/'cache/local';local.mkdir(parents=True)
        path=local/'index.json';path.write_text('{"complete":true,"format":"old"}')
        original=path.read_bytes()
        with self.assertRaisesRegex(ValueError,'immutable input binding proof'):
            self.construct()
        self.assertEqual(path.read_bytes(),original)

    def test_owned_binding_header_rejects_changed_context_profile_even_when_resealed(self):
        from hiercp_v1x.contracts import canonical_hash
        inputs=self.construct()
        inputs._publish_input_bindings()
        original=inputs._bindings_path.read_bytes()
        changed=json.loads(original)
        changed['context_contract_sha256']='0'*64
        changed['content_sha256']=canonical_hash({k:v for k,v in changed.items() if k!='content_sha256'})
        # This is an explicitly fresh CPU UNIT witness, never a user result.
        inputs._bindings_path.write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError,'binding/source/query proof differs'):
            inputs._admit_input_bindings()
        self.assertEqual(inputs._bindings_path.read_text(),json.dumps(changed))


if __name__ == '__main__':
    unittest.main()
