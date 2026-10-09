"""CPU UNIT parity against the uncached original full-input equations."""
import copy
import unittest
from unittest.mock import patch

import numpy as np


class ImmutablePreparationDebug(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import tests.test_v24_gt_blind as fixtures
        fixtures.RecipientGTBlindDebug.setUpClass()
        cls.fixture=fixtures.RecipientGTBlindDebug

    @classmethod
    def tearDownClass(cls):
        cls.fixture.tearDownClass()

    def test_cached_depth_binding_and_donor_proof_match_uncached_tensors_and_both_views(self):
        from hiercp.common import organ_depth_mm
        from hiercp_v1x import v24_inputs as inputs
        f=self.fixture
        def context():
            return inputs.recipient_context(f.recipient.case_id,f.CT,f.organ,f.spacing,f.recipient.image_affine)
        def build(recipient,source,**extra):
            return inputs.build_local_record(recipient,f.donor,source,f.prepared,f.rows[0],
                config=f.config,seed=42,ct_clip=f.clip,scope_contract=f.scope['contract_sha256'],**extra)
        # Exact pre-optimization equations: full EDT and CT/organ/donor hashes
        # were recomputed for every original candidate. No sampling is changed.
        def uncached_binding(recipient):
            return dict(format=inputs.FORMAT,case_id=recipient.case_id,
                CT_sha256=inputs.array_digest(recipient.image),organ_sha256=inputs.array_digest(recipient.organ_mask),
                spacing=recipient.spacing.tolist(),image_affine=recipient.image_affine.tolist(),
                recipient_tumor_GT_used=False)
        with patch.object(inputs.RecipientContext,'organ_depth',property(
                lambda recipient:organ_depth_mm(recipient.organ_mask,recipient.spacing))), \
             patch.object(inputs.RecipientContext,'_input_binding',property(uncached_binding)):
            reference=build(context(),f.source)
        cached=context(); source=copy.deepcopy(f.source)
        source.full_mask=inputs.immutable_array(source.full_mask)
        source.v24_mask_sha256=inputs.array_digest(source.full_mask)
        with patch('hiercp.common.organ_depth_mm',wraps=organ_depth_mm) as edt, \
             patch.object(inputs,'array_digest',wraps=inputs.array_digest) as hashes:
            actual=build(cached,source,donor_mask_sha256=source.v24_mask_sha256)
            repeated=build(cached,source,donor_mask_sha256=source.v24_mask_sha256)
        self.assertEqual(edt.call_count,1)
        self.assertEqual(sum(call.args[0] is cached.image for call in hashes.call_args_list),1)
        self.assertEqual(sum(call.args[0] is cached.organ_mask for call in hashes.call_args_list),1)
        self.assertEqual(sum(call.args[0] is source.full_mask for call in hashes.call_args_list),0)
        self.assertEqual(inputs.tensor_digest(reference),inputs.tensor_digest(actual))
        self.assertEqual(inputs.tensor_digest(reference),inputs.tensor_digest(repeated))
        self.assertEqual(cached.organ_depth.dtype,np.float32)
        np.testing.assert_array_equal(cached.organ_depth,organ_depth_mm(f.organ,f.spacing))
        for epoch in (0,29):
            def view_digest(record):
                graphs,src,target=inputs.materialize_pair(record,epoch=epoch)
                return inputs.tensor_digest((tuple(graph.to_dict() for graph in graphs),src,target))
            self.assertEqual(view_digest(reference),view_digest(actual))

    def test_input_arrays_cannot_be_reenabled_and_binding_is_defensive(self):
        from hiercp_v1x.v24_inputs import recipient_context
        f=self.fixture
        context=recipient_context(f.recipient.case_id,f.CT,f.organ,f.spacing,f.recipient.image_affine)
        original=context.binding()
        changed=context.binding(); changed['CT_sha256']='altered'; changed['spacing'][0]=999
        self.assertEqual(context.binding(),original)
        for value in (context.image,context.organ_mask,context.spacing,context.image_affine,context.organ_depth):
            self.assertFalse(value.flags.writeable)
            with self.assertRaises(ValueError):value.flags.writeable=True

    def test_unadmitted_or_writable_donor_cached_proof_is_rejected(self):
        from hiercp_v1x.v24_inputs import build_local_record,array_digest
        f=self.fixture; source=copy.deepcopy(f.source)
        source.v24_mask_sha256=array_digest(source.full_mask)
        arguments=dict(config=f.config,seed=42,ct_clip=f.clip,scope_contract=f.scope['contract_sha256'])
        with self.assertRaisesRegex(ValueError,'admitted immutable'):
            build_local_record(f.recipient,f.donor,source,f.prepared,f.rows[0],
                donor_mask_sha256=source.v24_mask_sha256,**arguments)
        source.full_mask.flags.writeable=False
        with self.assertRaisesRegex(ValueError,'admitted immutable'):
            build_local_record(f.recipient,f.donor,source,f.prepared,f.rows[0],
                donor_mask_sha256='0'*64,**arguments)


if __name__=='__main__':unittest.main()
