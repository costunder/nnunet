"""UNIT metadata checks; no neural execution or real CT claim."""
import copy
import unittest
from pathlib import Path

from hiercp_v1x.historical_c_checkpoint import validate_metadata, _source_path


def fixture():
    manifest = dict(format="v17_crossed_training_identity_v1", arm="C", debug=False)
    contract = dict(format="crossed_native_C_training_state_v1", arm="C",
                    execution=dict(debug=False), identity_sha256="a" * 64)
    selection = dict(epoch=20, update=1520, metric=[.986, .972, 6.795793, -0.424519, 0.])
    best = dict(format="crossed_C_best_own_model_v1", arm="C", debug=False,
                run_identity_sha256="a" * 64, selection=selection,
                state_dict={"UNIT_metadata_only": "actual tensors checked by strict runtime loader"})
    completion = dict(arm="C", identity=manifest, debug=False, full_training=True,
                      full_evaluation=True, completed_epochs=40, selected_own_epoch=20,
                      checkpoint_sha256="b" * 64)
    report = dict(format="crossed_native_C_training_state_v1", arm="C", debug=False,
                  status="COMPLETE", run_identity_sha256="a" * 64, best_own=selection,
                  completed_epochs=40, full_training=True, actual_CUDA=True, actual_raw_CT=True,
                  history=[dict(epoch=i) for i in range(1, 41)])
    return manifest, contract, best, completion, report


class HistoricalCMetadata(unittest.TestCase):
    def run_admission(self, values=None):
        m, c, b, done, r = fixture() if values is None else values
        return validate_metadata(m, c, b, debug=False, completion=done,
                                 latest_sha256="b" * 64, reports=[r])

    def test_full40_own_best_receipt(self):
        receipt = self.run_admission()
        self.assertEqual(receipt["selected_own_epoch"], 20)
        self.assertTrue(receipt["complete40_verified"])
        self.assertFalse(receipt["same_as_published_C_common_evaluation"])
        self.assertFalse(receipt["published_common_observed_bank_reused"])
        self.assertEqual(receipt["optimizer_updates"], 0)

    def test_another_arm(self):
        data = fixture(); data[2]["arm"] = "D"
        with self.assertRaises(ValueError): self.run_admission(data)

    def test_debug_state_cannot_be_production(self):
        data = fixture(); data[2]["debug"] = True
        with self.assertRaises(ValueError): self.run_admission(data)

    def test_other_identity(self):
        data = fixture(); data[2]["run_identity_sha256"] = "c" * 64
        with self.assertRaises(ValueError): self.run_admission(data)

    def test_missing_completion(self):
        m, c, b, _, r = fixture()
        with self.assertRaises(ValueError):
            validate_metadata(m, c, b, debug=False, reports=[r])

    def test_latest_bytes_changed(self):
        data = fixture(); data[3]["checkpoint_sha256"] = "c" * 64
        with self.assertRaises(ValueError): self.run_admission(data)

    def test_best_epoch_changed(self):
        data = fixture(); data[3]["selected_own_epoch"] = 21
        with self.assertRaises(ValueError): self.run_admission(data)

    def test_report_selection_changed(self):
        data = fixture(); data[4]["best_own"] = copy.deepcopy(data[4]["best_own"])
        data[4]["best_own"]["epoch"] = 21
        with self.assertRaises(ValueError): self.run_admission(data)

    def test_incomplete_epoch_history(self):
        data = fixture(); data[4]["history"].pop(13)
        with self.assertRaises(ValueError): self.run_admission(data)

    def test_no_completed_report(self):
        data = fixture(); data[4]["status"] = "PAUSED"
        with self.assertRaises(ValueError): self.run_admission(data)

    def test_nonfinite_best_metric(self):
        data = fixture(); data[2]["selection"]["metric"][0] = float("nan")
        with self.assertRaises(ValueError): self.run_admission(data)

    def test_exact_current_producer_selection_key(self):
        from hiercp_v1x.transition_c_training import own_selection_key
        data = fixture()
        own = dict(MRR=.986, top1=.972, mean_margin=6.795793, loss=.424519)
        self.assertEqual(data[2]["selection"]["metric"], list(own_selection_key(own)))
        self.run_admission(data)

    def test_short_or_extended_selection_key_rejected(self):
        for width in (3, 4, 6):
            with self.subTest(width=width):
                data = fixture(); metric = data[2]["selection"]["metric"]
                data[2]["selection"]["metric"] = (metric + [0.])[:width]
                with self.assertRaises(ValueError): self.run_admission(data)

    def test_added_consistency_selection_marker_rejected(self):
        data = fixture(); data[2]["selection"]["metric"][4] = .125
        with self.assertRaises(ValueError): self.run_admission(data)

    def test_boolean_metric_rejected(self):
        data = fixture(); data[2]["selection"]["metric"][4] = False
        with self.assertRaises(ValueError): self.run_admission(data)

    def test_missing_tensor_state(self):
        data = fixture(); data[2]["state_dict"] = {}
        with self.assertRaises(ValueError): self.run_admission(data)

    def test_debug_explicitly_not_complete40(self):
        m, c, b, _, _ = fixture()
        m["debug"] = b["debug"] = c["execution"]["debug"] = True
        receipt = validate_metadata(m, c, b, debug=True)
        self.assertFalse(receipt["complete40_verified"])
        self.assertTrue(receipt["debug"])

    def test_source_remapping_is_package_specific(self):
        from hiercp_v1x.historical_c_checkpoint import ROOT
        self.assertEqual(_source_path("/server/old/hiercp_v1x/transition_model.py", "/baseline"),
                         ROOT / "hiercp_v1x/transition_model.py")
        self.assertEqual(_source_path("C:\\old\\hiercp_v222\\model.py", "/baseline"),
                         ROOT / "hiercp_v222/model.py")
        self.assertEqual(_source_path("/old/source/v1.0/hiercp/loss.py", "/baseline"),
                         Path("/baseline/source/v1.0/hiercp/loss.py"))
        with self.assertRaises(ValueError): _source_path("/arbitrary/tools.py", "/baseline")
        with self.assertRaises(ValueError):
            _source_path("/old/hiercp_v1x/../../other.py", "/baseline")


if __name__ == "__main__":
    unittest.main()
