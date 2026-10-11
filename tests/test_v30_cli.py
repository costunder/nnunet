"""UNIT CLI/provenance/ownership contracts; no production or CT execution."""
from contextlib import contextmanager, redirect_stderr, redirect_stdout, ExitStack
import copy
from dataclasses import asdict
import io
import itertools
from collections import OrderedDict
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import uuid
import threading

import numpy as np
import torch

from tools import run_v30
from hiercp_v1x.v25_targets import validate_policy, UPDATES_PER_EPOCH
from hiercp_v1x.v30_scorer import V30Scorer
from hiercp_v1x.v30_training import patient_batches
from hiercp_v1x import v30_training
from hiercp_v1x.v30_factory import V30NativeInputs
from hiercp_v1x.v30_vessels import RecipientBinding, VesselBatch


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "v30.json"


@contextmanager
def unit_paths():
    root = ROOT / "tmp" / ("UNIT_v30_cli_" + uuid.uuid4().hex)
    root.mkdir(parents=True)
    try:
        (root / "native").mkdir()
        for name in ("inventory.json", "vessels.json", "STU_request_identity.bin"):
            (root / name).write_bytes(("UNIT request identity fixture only: " + name).encode())
        yield root
    finally:
        if root.parent.resolve() != (ROOT / "tmp").resolve() or not root.name.startswith("UNIT_v30_cli_"):
            raise RuntimeError("UNIT cleanup escaped its disposable fixture directory")
        for path in sorted(root.rglob("*"), key=lambda value: len(value.parts), reverse=True):
            if not path.resolve().is_relative_to(root.resolve()):
                raise RuntimeError("UNIT cleanup escaped its fixture")
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
        root.rmdir()


def arguments(root, *, arm="A", gpu=0, mode="prepare"):
    return ["--mode", mode, "--arm", arm, "--gpu", str(gpu),
        "--native-experiment", str(root / "native"), "--inventory", str(root / "inventory.json"),
        "--vessels", str(root / "vessels.json"), "--stunet-checkpoint", str(root / "STU_request_identity.bin"),
        "--input-cache", str(root / "shared_inputs"), "--output", str(root / ("arm_" + arm))]


def scorer_UNIT(bindings):
    provider = SimpleNamespace(recipient_GT_used_in_forward=False, ds=SimpleNamespace(rows=[]))
    def geometry(plan, provider=None):
        raise AssertionError("UNIT constructor contract must not execute geometry")
    geometry.recipient_GT_used_in_forward = False
    return V30Scorer(torch.nn.Linear(2, 1), {"inner_train": provider, "inner_val": provider}, geometry,
        vessel_store=object(), recipient_bindings=bindings, physical_candidate_batch=32,
        checkpoint_local_chunks=True, amp=False, budget=object(),
        prefetch_cpu_chunks=True, pin_cpu_batches=True)


class V30CliUnitTests(unittest.TestCase):
    def test_progress_for_zero_positive_patients_never_reports_fake_zero_metrics(self):
        value=v30_training.progress_line('full_validation',1,128,[dict(observed_P=0)],21)
        self.assertIn('1/21 5%',value)
        self.assertIn('undefined',value)
        self.assertNotIn('MRR*=0',value)

    def test_new_namespaces_cannot_publish_into_preserved_sources(self):
        with unit_paths() as root:
            args=run_v30.parse(arguments(root))
            args.inventory=root/'inventory'/'index.json'
            args.inventory.parent.mkdir()
            args.inventory.write_text(json.dumps({'raw_records':[{'image':str(root/'CT'/'images'/'UNIT.nii'),
                'label':str(root/'CT'/'labels'/'UNIT.nii')}]}))
            args.vessels=root/'vessels'/'index.json'
            args.stunet_checkpoint=root/'weights'/'small_ep4k.model'
            run_v30.validate_output_paths(args)
            for target in (args.native_experiment/'new',args.vessels.parent/'new',
                    root/'CT'/'images'/'new',args.inventory,args.stunet_checkpoint):
                with self.subTest(target=target),self.assertRaisesRegex(ValueError,'overlaps preserved'):
                    changed=copy.copy(args);changed.output=target
                    run_v30.validate_output_paths(changed)
            with self.assertRaisesRegex(ValueError,'disjoint'):
                changed=copy.copy(args);changed.output=args.input_cache/'A'
                run_v30.validate_output_paths(changed)

    def test_prepared_input_reuse_resets_evicted_byte_accounting(self):
        inputs=object.__new__(V30NativeInputs)
        inputs._lock=threading.RLock()
        inputs._raw_cache=OrderedDict([('UNIT_patient',object())])
        inputs._raw_bytes=8*2**30
        inputs._donors=OrderedDict([(('UNIT_patient',0),object())])
        inputs._regions=OrderedDict([('UNIT_patient',object())])
        inputs.close()
        self.assertEqual(inputs._raw_bytes,0)
        self.assertFalse(inputs._raw_cache or inputs._donors or inputs._regions)

    def test_arbitrary_nonnegative_physical_gpu_numbers_and_four_arms(self):
        with unit_paths() as root:
            for arm in "ABCD":
                for gpu in (0, 1, 4, 7, 31):
                    with self.subTest(arm=arm, gpu=gpu):
                        parsed = run_v30.parse(arguments(root, arm=arm, gpu=gpu))
                        self.assertEqual((parsed.arm, parsed.gpu), (arm, gpu))
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                run_v30.parse(arguments(root, gpu=-1))
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                run_v30.parse(arguments(root, arm="other"))

    def test_four_arm_policies_are_independent_and_keep_the_confirmed_gates(self):
        document = json.loads(CONFIG.read_text())
        for arm, expected in zip("ABCD", ((8, 8), (16, 16), (32, 32), (128, None))):
            selected = run_v30.configuration(CONFIG, arm)
            policy = selected["runtime"]["curriculum"]
            self.assertEqual((policy["initial_u"], policy["increment_u"]), expected)
            receipt = validate_policy(policy)
            self.assertGreaterEqual(40 - receipt["latest_first_full_bank_epoch"] + 1, 8)
            self.assertEqual(policy["total_epochs"], 40)
            policy["initial_u"] = -1
            self.assertEqual(document["arms"][arm]["initial_u"], expected[0])
            self.assertEqual(run_v30.configuration(CONFIG, arm)["runtime"]["curriculum"]["initial_u"], expected[0])

    def test_request_identity_ignores_physical_gpu_mode_and_output_but_binds_sources(self):
        with unit_paths() as root:
            first_args = run_v30.parse(arguments(root, gpu=0))
            second_args = run_v30.parse(arguments(root, gpu=31, mode="train"))
            second_args.output = root / "other_own_output"
            config = run_v30.configuration(CONFIG, "A")
            first = run_v30.request(first_args, config)
            second = run_v30.request(second_args, config)
            self.assertEqual(first, second)
            self.assertFalse(any(key in first for key in ("physical_GPU", "hostname", "host", "GPU_UUID", "uuid")))
            for name in ("hiercp_v1x/v30_model.py", "hiercp_v1x/v30_vessels.py",
                         "hiercp_v1x/v30_scorer.py", "hiercp_v1x/v30_factory.py",
                         "hiercp_v1x/v25_context.py", "hiercp_v1x/v25_targets.py"):
                self.assertEqual(first["source_sha256"][name], run_v30.sha(ROOT / name))
            original = first["request_sha256"]
            (root / "inventory.json").write_bytes(b"UNIT changed source inventory bytes")
            self.assertNotEqual(run_v30.request(first_args, config)["request_sha256"], original)

    def test_shared_input_cache_does_not_merge_arm_identity(self):
        with unit_paths() as root:
            requests = []
            for arm in "ABCD":
                args = run_v30.parse(arguments(root, arm=arm, gpu=12))
                requests.append(run_v30.request(args, run_v30.configuration(CONFIG, arm)))
            self.assertEqual(len({row["input_cache"] for row in requests}), 1)
            self.assertEqual(len({row["request_sha256"] for row in requests}), 4)
            self.assertEqual([row["config"]["arm"] for row in requests], list("ABCD"))

    def test_existing_evidence_never_overwritten_by_a_different_request(self):
        with unit_paths() as root:
            path = root / "request.json"
            value = dict(profile="UNIT", arm="A", request_sha256="a" * 64)
            run_v30.publish(path, value)
            before = path.read_bytes()
            run_v30.publish(path, copy.deepcopy(value))
            with self.assertRaisesRegex(FileExistsError, "preserved"):
                run_v30.publish(path, dict(value, arm="B"))
            self.assertEqual(path.read_bytes(), before)

    def test_frozen_full_production_model_and_data_settings_are_required(self):
        with unit_paths() as root:
            original = json.loads(CONFIG.read_text())
            alterations = (
                ("epochs", 2), ("MonoPlace3D", True), ("encoder", "original_CNN_GAT"),
                ("runtime.workers", 1), ("runtime.hidden_subset", True),
                ("runtime.physical_patient_batch_candidates", [1, 4]),
                ("runtime.physical_candidate_batch_candidates", [8, 16]),
                ("runtime.pin_CPU_batches", False),
                ("runtime.independent_single_GPU", False),
                ("vessels.hidden_dim", 32), ("vessels.layers", 1),
                ("vessels.heads", 1), ("vessels.preserve_all_nodes_edges_components", False),
                ("vessels.recipient_tumor_overlap_as_input", True),
            )
            for dotted, value in alterations:
                with self.subTest(setting=dotted):
                    changed = copy.deepcopy(original)
                    target = changed
                    parts = dotted.split(".")
                    for key in parts[:-1]:
                        target = target[key]
                    target[parts[-1]] = value
                    path = root / "config_UNIT.json"
                    path.write_text(json.dumps(changed))
                    with self.assertRaises(ValueError):
                        run_v30.configuration(path, "A")

    def test_production_B4_has_680_updates_without_singleton_or_lost_patients(self):
        cases = [f"UNIT_ranking_patient_{number}" for number in range(65)]
        batches = patient_batches(cases, 4)
        self.assertEqual(len(batches), UPDATES_PER_EPOCH)
        self.assertEqual(40 * len(batches), 680)
        self.assertEqual([case for batch in batches for case in batch], cases)
        self.assertTrue(all(2 <= len(batch) <= 4 for batch in batches))
        self.assertNotEqual(40 * len(patient_batches(cases, 8)), 680)

    def test_faster_B8_diagnostic_never_changes_production_B4_selection(self):
        # CPU UNIT controller test: real differentiable fixture clone updates,
        # deterministic timing and mocked CUDA counters. This is not calibration
        # evidence for the actual STU/vessel model or any patient data.
        class UnitHead(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.vessel_encoder = torch.nn.Linear(2, 2)
                self.vessel_readout = torch.nn.Linear(2, 2)
                self.vessel_fusion = torch.nn.Linear(2, 2)
                self.original = torch.nn.Linear(2, 1)
            def forward(self, x):
                return self.original(self.vessel_fusion(self.vessel_readout(self.vessel_encoder(x))))
        class UnitNet(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.local_encoder = torch.nn.Module()
                self.local_encoder.dense_encoder = torch.nn.Linear(2, 2)
                self.patient_encoder = torch.nn.Linear(2, 2)
                self.prototype_encoder = torch.nn.Linear(2, 2)
                self.score_head = UnitHead()
            def forward(self, x):
                return self.score_head(self.prototype_encoder(self.patient_encoder(
                    self.local_encoder.dense_encoder(x)))).reshape(-1)
        class UnitPopulation:
            def partition_cases(self, partition, ranking_only=True):
                return [f"UNIT_{number}" for number in range(8)]
            def case(self, case, active_u=None, **kwargs):
                return SimpleNamespace(case_id=case, record_ids=(case,), positive_indices=(0,))
        class UnitScorer:
            def __init__(self, net):
                self.net, self.physical_candidate_batch = net, 32
                self.providers = {"inner_train": SimpleNamespace(records={
                    f"UNIT_{number}": dict(sampled_two_view_edges=number + 1) for number in range(8)})}
            def __call__(self, plans, **kwargs):
                size = 129 * len(plans)
                x = torch.linspace(-1, 1, 2 * size).reshape(size, 2)
                output = self.net(x)
                return SimpleNamespace(scores=tuple(output.reshape(len(plans), 129)),
                    consistency=output.square(), case_ids=tuple(plan.case_id for plan in plans),
                    workload=dict(patients=len(plans), upper_invocations=1))
        net = UnitNet()
        scorer = UnitScorer(net)
        before = v30_training.digest(net.state_dict())
        config = dict(v24_runtime=dict(physical_patient_batch_candidates=[4, 8],
            physical_candidate_batch_candidates=[32, 64, 128], calibration_repeats=3),
            training=dict(lr=1e-4, weight_decay=.01, fused_optimizer=False, amp=False, grad_clip=1.))
        parameters = sum(parameter.numel() for parameter in net.parameters())
        contract = dict(profile="UNIT", recipient_GT_used_in_forward=False,
            trained_v23_weights_loaded=False, parameters=parameters, trainable_parameters=parameters)
        with ExitStack() as stack, redirect_stdout(io.StringIO()):
            for name in ("reset_peak_memory_stats", "synchronize", "empty_cache"):
                stack.enter_context(patch.object(torch.cuda, name))
            stack.enter_context(patch.object(torch.cuda, "max_memory_allocated", return_value=0))
            stack.enter_context(patch.object(v30_training, "_check_budget"))
            stack.enter_context(patch.object(v30_training, "capture_rng", return_value={"profile": "UNIT"}))
            stack.enter_context(patch.object(v30_training, "restore_rng"))
            stack.enter_context(patch.object(v30_training.time, "perf_counter", side_effect=itertools.count()))
            receipt = v30_training.calibrate_training_batches(net, scorer, UnitPopulation(), config,
                budget=SimpleNamespace(cuda_bytes=2**30), model_contract=contract, debug=True)
        self.assertTrue(all(row["accepted"] for row in receipt["trials"]))
        self.assertGreater(max(row["patients_per_second"] for row in receipt["trials"] if row["physical_patient_batch"] == 8),
                           max(row["patients_per_second"] for row in receipt["trials"] if row["physical_patient_batch"] == 4))
        self.assertEqual(receipt["selected_physical_patient_batch"], 4)
        self.assertEqual(len(receipt["trials"]), 6)
        self.assertEqual(v30_training.digest(net.state_dict()), before)
        self.assertIs(scorer.net, net)

    def test_score_cp_requires_complete_matched_bank_and_baseline(self):
        with unit_paths() as root:
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                run_v30.parse(arguments(root, mode="score-cp"))
            valid = arguments(root, mode="score-cp") + ["--source-bank", str(root / "matched_bank"),
                "--baseline-preprocessed", str(root / "Dataset730")]
            parsed = run_v30.parse(valid)
            self.assertEqual(parsed.source_bank, root / "matched_bank")
            self.assertEqual(parsed.baseline_preprocessed, root / "Dataset730")

    def test_scorer_normalizes_mapping_bindings_and_rejects_wrong_case_keys(self):
        binding = RecipientBinding("UNIT_patient", "a" * 64, np.eye(4), (5, 6, 7))
        scorer = scorer_UNIT({"UNIT_patient": asdict(binding)})
        self.assertIsInstance(scorer.recipient_bindings["UNIT_patient"], RecipientBinding)
        self.assertEqual(scorer.recipient_bindings["UNIT_patient"], binding)
        with self.assertRaisesRegex(ValueError, "case identity"):
            scorer_UNIT({"different_UNIT_patient": asdict(binding)})

    def test_pin_memory_covers_every_tensor_and_preserves_audit_and_owners(self):
        # Explicit UNIT pinning dispatch test, independent of CUDA availability.
        batch = VesselBatch(torch.ones(2, 12), torch.ones(2, 3), torch.tensor([[0, 1], [1, 0]]),
            torch.ones(2, 8), torch.zeros(2, dtype=torch.long), torch.zeros(2, dtype=torch.long),
            torch.zeros(1, dtype=torch.long), ("UNIT_patient",), torch.tensor([0, 2]), torch.tensor([0, 1]),
            dict(profile="UNIT", recipient_tumor_overlap_used=False))
        pinned_calls = []
        def unit_pin(tensor):
            pinned_calls.append(tensor)
            return tensor.clone()
        with patch.object(torch.Tensor, "pin_memory", unit_pin):
            pinned = batch.pin_memory()
        self.assertEqual(len(pinned_calls), 9)
        self.assertEqual(pinned.case_ids, batch.case_ids)
        self.assertEqual(pinned.audit, batch.audit)
        self.assertIsNot(pinned.audit, batch.audit)
        torch.testing.assert_close(pinned.component_patient_index, batch.component_patient_index)
        for name in ("node_features", "pos_mm", "edge_index", "edge_attr", "node_patient_index",
                     "component_index", "component_patient_index", "case_ptr", "component_ptr"):
            self.assertNotEqual(getattr(pinned, name).data_ptr(), getattr(batch, name).data_ptr())


if __name__ == "__main__":
    unittest.main()
