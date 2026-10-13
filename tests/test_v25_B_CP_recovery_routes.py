"""External subprocess boundaries stay native BEST/full250 and fail closed."""
import sys
import unittest

from hiercp_v1x import v25_B_CP_recovery_runtime as runtime
from tools import run_v25_B_CP_recovery_controller as controller


class RecoveryRoutes(unittest.TestCase):
    def test_original_native_best_cli_survives_routing(self):
        command = [sys.executable, '-B', '-u', '-c', runtime.TRAIN_PROGRAM,
                   '730', '3d_fullres', '0', '-tr', 'OriginalFullTrainer',
                   '-p', 'OriginalResEncM', '--val_best', '--c']
        routed = runtime.route_native_command(command)
        self.assertEqual(routed[5:], command[5:])
        self.assertEqual(routed[:4], command[:4])
        self.assertEqual(routed[4], runtime.RECOVERY_TRAIN_PROGRAM)
        self.assertEqual(command[4], runtime.TRAIN_PROGRAM)

    def test_actual_clone_worker_arguments_are_kept(self):
        original = str(runtime.ROOT/'tools/run_v25_authorized_early_downstream.py')
        command = [sys.executable, '-B', '-u', original,
                   'clone-debug-worker', '--native', '/original/native.json',
                   '--physical-batch', '4', '--output', '/original/debugB4.json']
        routed = runtime.route_native_command(command)
        self.assertEqual(routed[4:], command[4:])
        self.assertEqual(routed[3], str(runtime.ROOT/runtime.NATIVE_ENTRY))

    def test_unknown_child_cannot_be_rerouted(self):
        for command in ([sys.executable, '-B', '-u', '-c', 'print(42)'],
                        [sys.executable, '-B', '-u', '/other/native.py'],
                        [sys.executable, '-u', '-c', runtime.TRAIN_PROGRAM]):
            with self.subTest(command=command), self.assertRaises(ValueError):
                runtime.route_native_command(command)

    def test_historical_metrics_entry_keeps_BEST_and_all_original_arguments(self):
        command = [sys.executable, '-B', '-u', str(runtime.ROOT/runtime.ORIGINAL_METRICS_ENTRY),
                   '--prediction', '/original/BEST26', '--checkpoint', 'BEST',
                   '--evaluator-root', '/frozen/historical', '--output', '/fresh/metrics']
        routed = runtime.route_native_command(command)
        self.assertEqual(routed[4:], command[4:])
        self.assertEqual(routed[3], str(runtime.ROOT/runtime.METRICS_ENTRY))
        command[command.index('BEST')] = 'FINAL'
        with self.assertRaises(ValueError):
            runtime.route_native_command(command)

    def test_stage_binding_only_changes_entry(self):
        command = [sys.executable, '-B', '-u',
                   str(controller.ROOT/'tools/run_v25_B_early_downstream_execution.py'),
                   '--controller-seal', '/fresh/seal.json', '--mode', 'score-cp',
                   '--arm', 'B', '--gpu', '5', '--cp-output', '/preserved/cp_bank']
        routed = controller.replace_execution_entry(command)
        self.assertEqual(routed[4:], command[4:])
        self.assertEqual(routed[3], str(controller.ROOT/runtime.EXECUTION_ENTRY))
        with self.assertRaises(ValueError):
            controller.replace_execution_entry([*command[:3], '/unregistered.py', *command[4:]])


if __name__ == '__main__':
    unittest.main()
