"""UNIT scheduler/NVML boundary fixtures; no actual PBS, CUDA, or training."""
from __future__ import annotations

from contextlib import redirect_stdout
import hashlib
from io import StringIO
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from hiercp_v1x.comparison_allocation import (
    AllocationError, check_current_allocation, declared_requirements,
    parse_qstat, scheduler_environment, validate_allocation,
)
from tools import run_allocated_comparison_arm as wrapper


def qstat(*, ncpus=16, mem="192gb", ngpus=1, state="R", owner="unituser",
          host="unitnode/0*16", extra=""):
    return ("Job Id: 12345.unitserver\n"
            f"    Job_Owner = {owner}@submit\n    job_state = {state}\n"
            f"    Resource_List.ncpus = {ncpus}\n    Resource_List.mem = {mem}\n"
            f"    Resource_List.ngpus = {ngpus}\n    exec_host = {host}\n"
            "    Variable_List = PRIVATE_TOKEN=never_log_this,PBS_O_PATH=/private\n"
            "        ANOTHER_SECRET=do_not_emit\n" + extra)


class AllocationUnitTests(unittest.TestCase):
    def setUp(self):
        self.env = {"PBS_JOBID": "12345.unitserver", "CUDA_VISIBLE_DEVICES": "GPU-unit-3"}
        self.manifest = {"workers": 16, "rss_gib": 192}
        self.inventory = Mock(return_value={"physical_index": 3, "physical_uuid": "GPU-unit-3",
                                            "mig_enabled": False, "mig": []})

    def validate(self, *, environ=None, text=None, manifest=None, gpu=3):
        return validate_allocation(self.manifest if manifest is None else manifest, gpu,
                                   environ=self.env if environ is None else environ,
                                   qstat_text=qstat() if text is None else text,
                                   inventory_reader=self.inventory, user="unituser", hostname="unitnode")

    def test_missing_pbs_stops_before_nvml(self):
        with self.assertRaisesRegex(AllocationError, "PBS_JOBID"):
            self.validate(environ={"CUDA_VISIBLE_DEVICES": "GPU-unit-3"})
        self.inventory.assert_not_called()

    def test_missing_visibility_stops_before_nvml(self):
        with self.assertRaisesRegex(AllocationError, "CUDA_VISIBLE_DEVICES"):
            self.validate(environ={"PBS_JOBID": "12345.unitserver"})
        self.inventory.assert_not_called()

    def test_cpu_allocation_cannot_silently_reduce_workers(self):
        with self.assertRaisesRegex(AllocationError, "workers=16 exceeds PBS ncpus=6"):
            self.validate(text=qstat(ncpus=6))
        self.assertEqual(self.manifest["workers"], 16)
        self.inventory.assert_not_called()

    def test_rss_budget_cannot_silently_shrink(self):
        with self.assertRaisesRegex(AllocationError, "RSS budget"):
            self.validate(text=qstat(mem="128gb"))
        self.assertEqual(self.manifest["rss_gib"], 192)
        self.inventory.assert_not_called()

    def test_outside_physical_gpu_is_rejected(self):
        self.env["CUDA_VISIBLE_DEVICES"] = "GPU-unit-6"
        with self.assertRaisesRegex(AllocationError, "outside scheduler"):
            self.validate()

    def test_multiple_visible_gpu_uuids_admit_one_without_mutating_visibility(self):
        self.env["CUDA_VISIBLE_DEVICES"] = "GPU-unit-6,GPU-unit-3"
        before = dict(self.env)
        result = self.validate(text=qstat(ngpus=2))
        self.assertEqual(result["selected_uuid"], "GPU-unit-3")
        self.assertEqual(result["allocation"]["memory_bytes"], 192 * 2**30)
        self.assertEqual(self.env, before)
        self.assertIn("one arm only", result["scope"])
        self.assertNotIn("never_log_this", json.dumps(result))
        self.assertNotIn("do_not_emit", json.dumps(result))

    def test_visibility_cannot_exceed_scheduler_gpu_count(self):
        self.env["CUDA_VISIBLE_DEVICES"] = "GPU-unit-6,GPU-unit-3"
        with self.assertRaisesRegex(AllocationError, "more devices"):
            self.validate(text=qstat(ngpus=1))

    def test_malformed_nvml_identity_has_no_full_gpu_fallback(self):
        for observed in ({"physical_index": 3, "physical_uuid": "GPU-unit-3"},
                         {"physical_index": 4, "physical_uuid": "GPU-unit-3", "mig_enabled": False},
                         {"physical_index": 3, "physical_uuid": "GPU-unit-3", "mig_enabled": True, "mig": None}):
            self.inventory.return_value = observed
            with self.subTest(observed=observed), self.assertRaises(AllocationError):
                self.validate()

    def test_uuid_tokens_and_job_ids_are_not_shell_arguments(self):
        for bad in ("0", "-1", "GPU-unit-3,", "GPU-unit-3,GPU-unit-3", "GPU-a;cmd"):
            with self.subTest(visible=bad), self.assertRaises(AllocationError):
                scheduler_environment(dict(self.env, CUDA_VISIBLE_DEVICES=bad))
        with self.assertRaises(AllocationError):
            scheduler_environment(dict(self.env, PBS_JOBID="12345;anything"))

    def test_malformed_or_missing_scheduler_resources_are_rejected(self):
        for text in (qstat(ncpus="six"), qstat(mem="unlimited"), qstat(mem="-1gb"),
                     qstat(ngpus=0), qstat().replace("    Resource_List.mem = 192gb\n", ""),
                     qstat(extra="    Resource_List.ncpus = 64\n")):
            with self.subTest(text=text[:70]), self.assertRaises(AllocationError):
                self.validate(text=text)

    def test_job_identity_owner_state_and_host_are_verified(self):
        for text in (qstat().replace("12345.unitserver", "55555.unitserver"),
                     qstat(owner="someoneelse"), qstat(state="Q"), qstat(host="othernode/0*16"),
                     qstat(host="unitnode/0*8+othernode/0*8"), qstat()+qstat()):
            with self.subTest(text=text[:65]), self.assertRaises(AllocationError):
                self.validate(text=text)

    def test_folded_same_host_slots_are_accepted(self):
        result = self.validate(text=qstat(host="unitnode/0*8+\n        unitnode/8*8"))
        self.assertEqual(result["allocation"]["host"], "unitnode")

    def test_mig_requires_exactly_one_assigned_instance_on_requested_gpu(self):
        self.inventory.return_value.update(mig_enabled=True,
                                           mig=[{"uuid": "MIG-unit-a"}, {"uuid": "MIG-unit-b"}])
        self.env["CUDA_VISIBLE_DEVICES"] = "MIG-unit-b"
        result = self.validate()
        self.assertEqual(result["selected_uuid"], "MIG-unit-b")
        self.assertEqual(result["selected_type"], "MIG")
        for visible, ngpus in (("GPU-unit-3", 1), ("MIG-other", 1), ("MIG-unit-a,MIG-unit-b", 2)):
            with self.subTest(visible=visible), self.assertRaisesRegex(AllocationError, "exactly one"):
                self.validate(environ=dict(self.env, CUDA_VISIBLE_DEVICES=visible), text=qstat(ngpus=ngpus))

    def test_missing_conflicting_or_invalid_contract_is_not_invented(self):
        for manifest in ({}, {"workers": 0, "rss_gib": 192}, {"workers": True, "rss_gib": 192},
                         {"workers": 16, "rss_gib": float("nan")},
                         dict(self.manifest, runtime={"workers": 6})):
            with self.subTest(manifest=manifest), self.assertRaises(AllocationError):
                declared_requirements(manifest)
        self.assertEqual(declared_requirements({"runtime": self.manifest}),
                         {"workers": 16, "rss_budget_bytes": 192*2**30})

    def test_current_query_uses_argument_vector_and_no_raw_environment_output(self):
        runner = Mock(return_value=SimpleNamespace(returncode=0, stdout=qstat(), stderr=""))
        with patch("getpass.getuser", return_value="unituser"), patch("socket.gethostname", return_value="unitnode"):
            result = check_current_allocation(self.manifest, 3, environ=self.env,
                                              qstat_runner=runner, inventory_reader=self.inventory)
        self.assertEqual(runner.call_args.args[0], ["qstat", "-f", "12345.unitserver"])
        self.assertEqual(result["requested_physical_gpu"], 3)
        runner.return_value = SimpleNamespace(returncode=1, stdout="PRIVATE", stderr="PRIVATE")
        with self.assertRaisesRegex(AllocationError, "status 1") as caught:
            check_current_allocation(self.manifest, 3, environ=self.env,
                                     qstat_runner=runner, inventory_reader=self.inventory)
        self.assertNotIn("PRIVATE", str(caught.exception))

    def test_wrapper_rejects_before_original_parent_or_owned_child_runs(self):
        for owned_child in (False, True):
            args = SimpleNamespace(gpu=3, owned_child=owned_child)
            with patch.object(wrapper.run_comparison_arm, "resolve_request", return_value={}), \
                 patch.object(wrapper, "_manifest", return_value=self.manifest), \
                 patch.object(wrapper, "check_current_allocation", side_effect=AllocationError("denied")), \
                 patch.object(wrapper.run_comparison_arm, "run") as original:
                with self.assertRaisesRegex(AllocationError, "denied"):
                    wrapper.run(args)
                original.assert_not_called()

    def test_wrapper_delegates_original_arguments_after_success(self):
        args = SimpleNamespace(gpu=3, owned_child=False)
        with patch.object(wrapper.run_comparison_arm, "resolve_request", return_value={}), \
             patch.object(wrapper, "_manifest", return_value=self.manifest), \
             patch.object(wrapper, "check_current_allocation", return_value={"verified": True}), \
             patch.object(wrapper.run_comparison_arm, "run", return_value="original_result") as original, \
             redirect_stdout(StringIO()):
            self.assertEqual(wrapper.run(args), "original_result")
            original.assert_called_once_with(args)

    def test_manifest_digest_is_checked_before_scheduler_or_training(self):
        with tempfile.TemporaryDirectory(prefix="UNIT_allocation_") as name:
            root = Path(name)
            manifest = dict(self.manifest)
            manifest["sha256"] = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            path = root / "experiment.json"
            path.write_text(json.dumps(manifest), encoding="utf8")
            request = {"experiment": root, "requires_clone": False}
            self.assertEqual(wrapper._manifest(request), manifest)
            manifest["workers"] = 1
            path.write_text(json.dumps(manifest), encoding="utf8")
            with self.assertRaisesRegex(AllocationError, "digest changed"):
                wrapper._manifest(request)

    def test_wrapper_import_does_not_import_torch(self):
        result = subprocess.run([sys.executable, "-B", "-c",
                                 "import sys; import tools.run_allocated_comparison_arm; assert 'torch' not in sys.modules"],
                                cwd=wrapper.ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
