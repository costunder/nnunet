"""UNIT v1.9 delivery contract checks; never start a server or neural model."""
import ast
import json
from pathlib import Path
import re
import shutil
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ComparisonDeliveryTests(unittest.TestCase):
    def test_four_arms_share_contract_and_isolate_two_new_controls(self):
        config = json.loads((ROOT / "config/v19_comparison_controls.json").read_text(encoding="utf8"))
        self.assertEqual(config["arms"], ["selected", "native", "native_fixed", "native_listwise"])
        self.assertEqual(set(config["arm_conditions"]), set(config["arms"]))
        self.assertEqual((config["seed"], config["epochs"], config["margin_mm"]), (42, 40, 10))
        conditions = config["arm_conditions"]
        self.assertEqual(conditions["selected"]["objective"], conditions["native"]["objective"])
        self.assertEqual(conditions["native_fixed"]["objective"], conditions["native"]["objective"])
        self.assertIn("U:0..6", conditions["native_fixed"]["comparators"])
        self.assertIn("exact same rotating seven", conditions["native_listwise"]["comparators"])
        self.assertIn("target=0", conditions["native_listwise"]["objective"])
        self.assertIn("not CP-unsuitable", config["U_semantics"])
        self.assertEqual(config["validation_patients"], {"configured": 21, "materialized": 18})

    def test_server_targets_new_root_and_passes_physical_batch_as_array(self):
        script = (ROOT / "tools/server_v19_comparison.sh").read_text(encoding="utf8")
        self.assertIn('CP_GPU="${CP_GPU:-3}"', script)
        self.assertIn('CP_ARM="${CP_ARM:-all}"', script)
        self.assertIn('/experiments/v19_comparison_m10_seed42}', script)
        self.assertIn('python -B -u tools/run_v19_comparison.py', script)
        self.assertNotIn('run_v18_u_bridge.py', script)
        self.assertNotIn('/experiments/v18_u_bridge_m10_seed42', script)
        self.assertIn('read -r -a comparison_candidates', script)
        self.assertIn('--batch-candidates "${comparison_candidates[@]}"', script)
        for flag, value in (("gpu", "CP_GPU"), ("arm", "CP_ARM"), ("experiment", "CP_EXPERIMENT"),
                            ("baseline", "CP_BASELINE"), ("inventory", "CP_INVENTORY"),
                            ("workers", "CP_WORKERS"), ("cuda-gib", "CP_CUDA_GIB"),
                            ("rss-gib", "CP_RSS_GIB"), ("resident-gib", "CP_RESIDENT_GIB"),
                            ("validation-local-chunk", "CP_VALIDATION_LOCAL_CHUNK")):
            self.assertIn(f'--{flag} "${value}"', script)
        self.assertEqual(script.count('python -B -u tools/'), 1)
        self.assertNotRegex(script, r'(?m)^\s*(exit|logout|shutdown|reboot|kill|pkill|rm)\b')

    def test_docs_state_pending_scope_and_include_all13_checklist_items(self):
        result = json.loads((ROOT / "versions/v1.9/results.json").read_text(encoding="utf8"))
        self.assertFalse(result["full_training"])
        self.assertFalse(result["quality_verified"])
        self.assertEqual(result["production_data"]["actual_validation_patients"], 18)
        self.assertEqual(result["production_data"]["signed_validation_source_samples"], 36)
        self.assertEqual(result["matched_arms"], ["selected", "native", "native_fixed", "native_listwise"])
        for name in ("README.md", "config.md", "code.md"):
            text = (ROOT / "versions/v1.9" / name).read_text(encoding="utf8")
            self.assertIn("## 작업 완료 체크리스트", text)
            self.assertEqual(len(re.findall(r"(?m)^- \[[ x]\] ", text)), 13)
            self.assertIn("18명", text)
        ast.parse((ROOT / "versions/v1.9/run.py").read_text(encoding="utf8"))

    @unittest.skipUnless(shutil.which("bash"), "Bash is unavailable on this Windows host; server shell syntax unverified here")
    def test_server_shell_syntax_without_execution(self):
        completed = subprocess.run([shutil.which("bash"), "-n", str(ROOT / "tools/server_v19_comparison.sh")],
                                   capture_output=True, text=True, check=False)
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
