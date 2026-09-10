"""DEBUG boundary tests: typed raw banks must not enter legacy launchers.

Minimal metadata and explicit verifier doubles test rejection timing only;
they are not medical artifacts, trained models or performance evidence.
"""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from custom_trainers.onlinecp_raw_bank import PASTE_CONTRACT
from custom_trainers.nnUNetTrainer_OnlinePairedCPArgmaxV3 import OnlineCPBank as LegacyV3Bank
from custom_trainers.nnUNetTrainer_OnlinePairedCPArgmaxV3 import OnlineCPError as LegacyV3Error
from tools import downstream_level_ablation as ablation
from tools import online_cp_benchmark as online
from tools import online_cp_curriculum as curriculum


class RawBankConsumerGuardsDebugTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="DEBUG_raw_consumer_")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.bank = self.root / "bank"
        self.bank.mkdir()
        self.index = self.bank / "index.json"
        self.index.write_text(json.dumps({"format": "hiercp_online_bank_v2",
                                         "paste_contract": PASTE_CONTRACT}), encoding="utf-8")
        self.train = self.root / "train.json"
        self.nn = self.root / "nn.json"
        self.policy = self.root / "policy.json"
        for path in (self.train, self.nn, self.policy):
            path.write_text("{}", encoding="utf-8")
        self.layout = SimpleNamespace(bank=lambda fold: self.bank,
                                      train_config=self.train, nnunet_config=self.nn)

    def test_argmax_v3_rejects_before_reading_legacy_pool_arrays(self):
        with self.assertRaisesRegex(LegacyV3Error, "[Rr]aw"):
            LegacyV3Bank(str(self.index))
        self.assertEqual(sorted(path.name for path in self.bank.iterdir()), ["index.json"])

    def test_legacy_pair_rejects_before_overwrite_or_native_verification(self):
        sentinel = self.root / "existing_result"
        sentinel.write_bytes(b"DEBUG existing result must survive")
        with mock.patch.object(online, "_verified_bank_identity") as verify, \
                mock.patch.object(online, "_remove_exact_artifact") as remove:
            with self.assertRaisesRegex(online.OnlineBenchmarkError, "Raw-target"):
                online.train_online_pair(self.layout, 0, {}, {}, 760, "cuda", False, True)
            verify.assert_not_called()
            remove.assert_not_called()
        self.assertEqual(sentinel.read_bytes(), b"DEBUG existing result must survive")

    def test_legacy_all_rejects_before_layout_or_preparation(self):
        with mock.patch("sys.argv", ["online_cp_benchmark", "all", "--overwrite"]), \
                mock.patch.object(online, "make_layout") as layout:
            with self.assertRaisesRegex(online.OnlineBenchmarkError, "raw-target"):
                online.main()
            layout.assert_not_called()

    def test_legacy_ablation_assets_reject_before_gnn_or_checkpoint_work(self):
        run = SimpleNamespace(base=SimpleNamespace(source_bank=self.bank))
        with self.assertRaisesRegex(ablation.DownstreamAblationError, "Raw-target"):
            ablation._check_required_assets(run)

    def test_legacy_ablation_rescore_rejects_before_reference_training_check(self):
        run = SimpleNamespace(base=SimpleNamespace(source_bank=self.bank))
        with mock.patch.object(ablation, "_reference_training_contract") as reference:
            with self.assertRaisesRegex(ablation.DownstreamAblationError, "[Rr]aw"):
                ablation._rescore_banks(run, SimpleNamespace(dry_run=False))
            reference.assert_not_called()
        self.assertEqual(sorted(path.name for path in self.bank.iterdir()), ["index.json"])

    def test_rank_only_sidecar_rejects_before_full_bank_verification(self):
        with mock.patch.object(curriculum, "validate_curriculum_config", return_value={}), \
                mock.patch.object(curriculum, "curriculum_config_sha256", return_value="DEBUG"), \
                mock.patch.object(online, "_verified_bank_identity") as verify:
            with self.assertRaisesRegex(ValueError, "[Rr]aw"):
                curriculum.publish(self.layout, 0, 760, self.policy)
            verify.assert_not_called()
        self.assertFalse((self.bank / "curriculum_contract.json").exists())

    def test_feedback_sidecar_retains_the_real_verification_path(self):
        self.policy.write_text(json.dumps({"format": "onlinecp_segmentation_feedback_v1"}),
                               encoding="utf-8")
        with mock.patch("custom_trainers.onlinecp_feedback_policy.validate_feedback_config", return_value={}), \
                mock.patch("custom_trainers.onlinecp_feedback_policy.feedback_config_sha256", return_value="DEBUG"), \
                mock.patch.object(online, "_verified_bank_identity", side_effect=RuntimeError("DEBUG verifier reached")) as verify:
            with self.assertRaisesRegex(RuntimeError, "DEBUG verifier reached"):
                curriculum.publish(self.layout, 0, 760, self.policy)
            verify.assert_called_once()
        self.assertFalse((self.bank / "feedback_contract.json").exists())


if __name__ == "__main__":
    unittest.main()
