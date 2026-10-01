"""Admission/preservation regression checks; no model accuracy claims."""
import unittest
from unittest.mock import Mock

from tools.diagnose_local_cnn_reference import evaluation_cases, select_cases
from tools.local_cnn_reference_causal import capture_prediction, fitted_case_ids
from tools.local_cnn_reference_runtime import probe_updates
from tests.test_reference_rankable_control import context_fixture


class CausalIntegrationChecks(unittest.TestCase):
    def test_fitted_case_union_preserves_old_eval_cases(self):
        rows=[dict(case_id=name,target=t) for name in ('liver_1','liver_109','liver_117','liver_3','liver_49') for t in (0,1)]
        prior=select_cases(rows,2)
        added=evaluation_cases(rows,2,['liver_117','liver_49'])
        self.assertEqual(added[:2],prior)
        self.assertEqual(set(added),set(prior)|{'liver_117','liver_49'})
        self.assertEqual(len(added),len(set(added)))
        with self.assertRaises(ValueError):evaluation_cases(rows,2,['missing'])
        with self.assertRaises(ValueError):evaluation_cases(rows,2,['liver_117','liver_117'])

    def test_exact_prediction_observer_restores_methods_even_on_error(self):
        class Net:
            def predict_embeddings(self,x):return {'value':x}
        net=Net();original=net.predict_embeddings
        with capture_prediction(net) as outputs:
            value=net.predict_embeddings(7)
            self.assertIs(outputs[0],value)
        self.assertNotIn('predict_embeddings',net.__dict__)
        self.assertEqual(net.predict_embeddings,original)
        net.predict_embeddings=lambda x:x+1
        customized=net.predict_embeddings
        with self.assertRaises(RuntimeError),capture_prediction(net):
            net.predict_embeddings(1)
            raise RuntimeError('fixture failure')
        self.assertIs(net.predict_embeddings,customized)

    def test_actual_fitted_cases_follow_existing_tile_order(self):
        names=fitted_case_ids({'selected_tiles':[{'case_id':'a'},{'case_id':'a'},{'case_id':'b'}]})
        self.assertEqual(names,['a','b'])

    def test_causal_request_is_rejected_before_gpu_providers(self):
        context=context_fixture();provider=Mock()
        for selection,evaluator in [('complete_prefix',provider),('rankable_full_batch_prefix',None)]:
            with self.assertRaises(ValueError):
                probe_updates(object(),steps=1,train_tiles=context.order,batch_provider=provider,
                    support_provider=provider,loss_context=context,physical_batch=32,budget=Mock(),
                    training={},seed=42,evaluation_provider=provider,selection_policy=selection,
                    transfer_policy='affine_relations_zero_out_bias',causal_probe=True,
                    fitted_evaluation_provider=evaluator)
        provider.assert_not_called()


if __name__=='__main__':unittest.main()
