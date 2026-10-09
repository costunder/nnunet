"""UNIT checks for independent GPU admission and isolation; no CT training."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from tools import run_v23_all_p as runner


class SingleGPUAdmissionUnit(unittest.TestCase):
    def argv(self, gpu, **extra):
        values = ["--native-experiment", "UNIT/native", "--inventory", "UNIT/inventory.json",
            "--prepared-cache", "UNIT/cache.json", "--full-validation-upper-cache", "UNIT/upper",
            "--output", "UNIT/output", "--gpus", *map(str, gpu)]
        for key, value in extra.items():
            values.extend(["--" + key.replace("_", "-"), str(value)])
        return values

    def test_each_authorized_gpu_is_a_single_independent_request(self):
        for gpu in (1, 5, 6):
            with self.subTest(gpu=gpu):
                args = runner.parse(self.argv([gpu]))
                self.assertEqual(args.gpus, [gpu])
                self.assertIsNone(args.worker_rank)

    def test_multi_gpu_request_and_other_gpu_are_rejected(self):
        for gpus in ([1, 5, 6], [1, 5], [0], [7], [1, 1]):
            with self.subTest(gpus=gpus), self.assertRaises(ValueError):
                runner.parse(self.argv(gpus))

    def test_single_worker_has_exact_gpu_and_owned_output(self):
        args = runner.parse(self.argv([5], mode="train", worker_rank=0,
            invocation="a" * 32))
        args.resume = True
        command = runner.child_command(args, "train", 0)
        position = command.index("--gpus")
        self.assertEqual(command[position + 1:position + 3], ["5", "--worker-rank"])
        self.assertEqual(command[command.index("--output") + 1], str(args.output))
        self.assertIn("--resume", command)
        with self.assertRaises(ValueError):
            runner.parse(self.argv([5], worker_rank=1, invocation="a" * 32))

    def calibration(self):
        return dict(world_size=1, request_sha256="UNIT_REQUEST", physical_GPUs=[5],
            debug=False, measured_full_P_U128_backward=True,
            original_model_and_RNG_preserved=True, initial_state_sha256="UNIT_INITIAL",
            selected_physical_candidate_batch=32, selected_physical_patient_batch=4,
            GPU_reports=[dict(physical_GPU=5, initial_state_sha256="UNIT_INITIAL")])

    def test_calibration_cannot_be_shared_as_an_unbound_distributed_receipt(self):
        request = dict(request_sha256="UNIT_REQUEST", gpus=[5])
        runner.validate_single_gpu_calibration(self.calibration(), request)
        changes = [dict(world_size=3), dict(physical_GPUs=[1]),
            dict(request_sha256="OTHER"), dict(measured_full_P_U128_backward=False),
            dict(GPU_reports=[dict(physical_GPU=1, initial_state_sha256="UNIT_INITIAL")])]
        for change in changes:
            candidate = {**self.calibration(), **change}
            with self.subTest(change=change), self.assertRaises(ValueError):
                runner.validate_single_gpu_calibration(candidate, request)

    def test_training_worker_does_not_initialize_process_group(self):
        # Mocked orchestration fixture: verifies execution topology only.
        with tempfile.TemporaryDirectory(prefix="v23_single_UNIT_") as name:
            output = Path(name)
            args = runner.parse(self.argv([5], mode="train", worker_rank=0,
                invocation="b" * 32))
            args.output = output
            calibration = self.calibration()
            (output / "calibration.json").write_text(json.dumps(calibration), encoding="utf-8")
            config = dict(v23_runtime=dict(workers=12))
            request = dict(request_sha256="UNIT_REQUEST", gpus=[5], source={},
                config=copy.deepcopy(config))
            scorer = Mock()
            scorer.providers = {'inner_train': Mock(), 'inner_val': Mock()}
            scorer.geometry.finish.return_value = {"UNIT": "complete"}
            with patch.object(runner, "gpu_setup", return_value=(Mock(), scorer, Mock(), config, Mock())), \
                    patch("hiercp_v1x.v23_training.run_training", return_value={"UNIT": "complete"}) as train, \
                    patch("torch.distributed.init_process_group") as initialize:
                runner.worker(args, request)
            initialize.assert_not_called()
            train.assert_called_once()
            self.assertEqual(train.call_args.kwargs["physical_patient_batch"], 4)
            self.assertEqual(train.call_args.kwargs["output"], output / "training")
            self.assertEqual(scorer.physical_candidate_batch, 32)
            for provider in scorer.providers.values(): provider.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
