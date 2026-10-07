"""UNIT routing/ownership fixtures; no CT, training, process stop or GPU work."""
from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import hashlib
from io import StringIO
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
import queue
import subprocess
import sys
import threading
import time
from unittest.mock import Mock, patch

import psutil

from hiercp_v1x import comparison_experiment, u_bridge_continuation, u_bridge_experiment
from tools import run_comparison_arm as launch


ARMS = ("selected", "native", "native_fixed", "native_listwise")
REPOSITORY = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf8")


def sealed(root, *, family="u_bridge", arm=None, data_root=None):
    root.mkdir(parents=True)
    baseline = root.parent / "UNIT_preserved_baseline"
    baseline.mkdir(exist_ok=True)
    data_root = root / "data" if data_root is None else data_root
    data_root.mkdir(parents=True, exist_ok=True)
    inventory = root.parent / "UNIT_inventory.json"
    if not inventory.exists():
        write_json(inventory, dict(fixture="UNIT metadata only; not actual CT"))
    module = u_bridge_experiment if family == "u_bridge" else comparison_experiment
    manifest = dict(format=module.FORMAT, debug=False, epochs=40,
        semantic=dict(version="v1.8" if family == "u_bridge" else "v1.9"),
        helpers={name: sha(REPOSITORY / name) for name in module.FILES},
        baseline=dict(baseline=str(baseline), inventory_sha256=sha(inventory)),
        prepared_data_root=str(data_root), workers=16, explicit_batch_candidates=[1, 2, 4, 8, 16, 32],
        cuda_gib=40., rss_gib=192., resident_gib=128., validation_local_chunk=8)
    manifest["sha256"] = digest(manifest)
    write_json(root / "experiment.json", manifest)
    (root / "initial.pt").write_bytes(b"UNIT opaque initial state; not a neural checkpoint")
    report_arms = ("selected", "native") if family == "u_bridge" else ARMS
    write_json(root / "calibration.json", dict(contract_sha256=manifest["sha256"], physical_batch=1,
        reports={name: {} for name in report_arms}, GPU_name="UNIT metadata fixture"))
    if arm is not None:
        output = root / arm; output.mkdir()
        write_json(output / "training_identity.json", dict(fixture="UNIT metadata", identity_sha256="e" * 64))
        (output / "checkpoint_latest.pt").write_bytes(b"UNIT independently progressing opaque checkpoint: " + arm.encode())
        (output / "curve.jsonl").write_text('{"UNIT_epoch":3,"UNIT_step":7}\n')
    return inventory


def inventory(root):
    return {p.relative_to(root).as_posix(): sha(p) for p in root.rglob("*") if p.is_file()}


class ArmLaunchTests(unittest.TestCase):
    def directory(self):
        return TemporaryDirectory(prefix="UNIT_arm_launch_", dir=REPOSITORY)

    def args(self, base, arm, gpu=3, extra=()):
        return launch.parse(["--gpu", str(gpu), "--arm", arm, "--experiments-dir", str(base),
                             "--inventory", str(base / "UNIT_inventory.json"), *extra])

    def four_roots(self, base):
        source = base / "v18_u_bridge_m10_seed42"
        sealed(source, arm="selected")
        # Different saved progress in each historical arm remains independent.
        native = source / "native"; native.mkdir()
        write_json(native / "training_identity.json", dict(fixture="UNIT native", identity_sha256="f" * 64))
        (native / "checkpoint_latest.pt").write_bytes(b"UNIT historical native checkpoint")
        for arm in ("selected", "native"):
            root = base / ("v18_" + arm + "_m10_seed42_memory")
            u_bridge_continuation.prepare_continuation(source, root, arm)
        for arm in ("native_fixed", "native_listwise"):
            sealed(base / ("v19_" + arm + "_m10_seed42"), family="comparison", arm=arm)
        return source

    def test_four_default_routes_have_distinct_outputs_data_locks_and_checkpoints(self):
        with self.directory() as directory:
            base = Path(directory); self.four_roots(base); before = inventory(base)
            requests = [launch.resolve_request(self.args(base, arm), environ={}) for arm in ARMS]
            self.assertEqual(len({str(r["experiment"]) for r in requests}), 4)
            self.assertEqual(len({str(r["data_root"]) for r in requests}), 4)
            self.assertEqual(len({str(r["checkpoint"]) for r in requests}), 4)
            self.assertEqual(len({str(r["experiment"] / ".pipeline.lock") for r in requests}), 4)
            self.assertEqual(len({str(r["data_root"] / ".data.lock") for r in requests}), 4)
            for arm, request in zip(ARMS, requests):
                self.assertEqual(request["arm"], arm)
                self.assertEqual(request["gpu"], 3)
                self.assertFalse(request["requires_clone"])
                self.assertEqual(request["data_root"], request["experiment"] / "data")
                self.assertEqual(request["checkpoint"], request["experiment"] / arm / "checkpoint_latest.pt")
            self.assertEqual(before, inventory(base), "Resolving a launch must not change saved experiment files")

    def test_inherited_wrong_CP_EXPERIMENT_does_not_override_explicit_arm_routing(self):
        with self.directory() as directory:
            base = Path(directory); source = self.four_roots(base)
            env = {"CP_EXPERIMENT": str(source), "CP_GPU": "99", "CP_ARM": "native"}
            request = launch.resolve_request(self.args(base, "selected", gpu=5), environ=env)
            self.assertEqual(request["experiment"], base / "v18_selected_m10_seed42_memory")
            self.assertEqual(request["arm"], "selected")
            self.assertEqual(request["gpu"], 5)
            self.assertEqual(env["CP_EXPERIMENT"], str(source))

    def test_absent_v18_destination_requires_clone_instead_of_shared_root_fallback(self):
        with self.directory() as directory:
            base = Path(directory); source = base / "v18_u_bridge_m10_seed42"; sealed(source, arm="selected")
            before = inventory(base)
            for arm in ("selected", "native"):
                request = launch.resolve_request(self.args(base, arm), environ={})
                self.assertTrue(request["requires_clone"])
                self.assertEqual(request["source_experiment"], source)
                self.assertNotEqual(request["experiment"], source)
                self.assertEqual(request["data_root"], request["experiment"] / "data")
                self.assertFalse(request["experiment"].exists())
            self.assertEqual(before, inventory(base))

    def test_absent_v19_existing_experiment_is_rejected_without_initial_restart(self):
        with self.directory() as directory:
            base = Path(directory); sealed(base / "v18_u_bridge_m10_seed42", arm="selected")
            before = inventory(base)
            for arm in ("native_fixed", "native_listwise"):
                with self.subTest(arm=arm), self.assertRaises((ValueError, FileNotFoundError, RuntimeError)):
                    launch.resolve_request(self.args(base, arm), environ={})
            self.assertEqual(before, inventory(base))

    def test_explicit_shared_v18_destination_rejected_for_selected_and_native(self):
        with self.directory() as directory:
            base = Path(directory); source = base / "v18_u_bridge_m10_seed42"; sealed(source, arm="selected")
            before = inventory(base)
            for arm in ("selected", "native"):
                with self.subTest(arm=arm), self.assertRaises((ValueError, RuntimeError)):
                    launch.resolve_request(self.args(base, arm, extra=("--experiment", str(source))), environ={})
            self.assertEqual(before, inventory(base))

    def test_explicit_destination_cannot_use_other_arm_continuation(self):
        with self.directory() as directory:
            base = Path(directory); self.four_roots(base)
            wrong = base / "v18_selected_m10_seed42_memory"; before = inventory(base)
            with self.assertRaises((ValueError, RuntimeError)):
                launch.resolve_request(self.args(base, "native", extra=("--experiment", str(wrong))), environ={})
            self.assertEqual(before, inventory(base))

    def test_explicit_v19_destination_rejects_other_arm_checkpoint_ownership(self):
        with self.directory() as directory:
            base = Path(directory); self.four_roots(base)
            wrong = base / "v19_native_fixed_m10_seed42"; before = inventory(base)
            with self.assertRaises((ValueError, RuntimeError)):
                launch.resolve_request(self.args(base, "native_listwise", extra=("--experiment", str(wrong))), environ={})
            self.assertEqual(before, inventory(base))

    def test_advanced_owned_checkpoint_and_logs_never_replaced_from_historical_source(self):
        with self.directory() as directory:
            base = Path(directory); source = self.four_roots(base)
            own = base / "v18_selected_m10_seed42_memory"
            latest = own / "selected/checkpoint_latest.pt"
            latest.write_bytes(b"UNIT advanced independent step999")
            (own / "selected/curve.jsonl").write_text('{"UNIT_step":999}\n')
            before = inventory(base)
            request = launch.resolve_request(self.args(base, "selected"), environ={})
            self.assertEqual(request["checkpoint"], latest)
            self.assertFalse(request["requires_clone"])
            self.assertEqual(before, inventory(base))
            self.assertNotEqual(latest.read_bytes(), (source / "selected/checkpoint_latest.pt").read_bytes())

    def test_numeric_gpu_is_required_and_forwarded_without_arbitrary_assignment(self):
        with self.directory() as directory:
            base = Path(directory); self.four_roots(base)
            for gpu in (0, 2, 3, 6):
                request = launch.resolve_request(self.args(base, "native_fixed", gpu=gpu), environ={})
                self.assertEqual(request["gpu"], gpu)
            for argv in (["--arm", "selected"], ["--gpu", "UUID_not_number", "--arm", "selected"]):
                with self.subTest(argv=argv), redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                    launch.parse(argv)
            with self.assertRaisesRegex(ValueError, "nonnegative"):
                launch.resolve_request(self.args(base, "selected", gpu=-1), environ={})

    def test_unknown_arm_and_multiple_arms_are_rejected(self):
        for arm in ("both", "all", "UNIT_unknown"):
            with self.subTest(arm=arm), redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                launch.parse(["--gpu", "3", "--arm", arm])

    def test_custom_cache_sources_are_readonly_paths_and_do_not_change_checkpoint_route(self):
        with self.directory() as directory:
            base = Path(directory); self.four_roots(base)
            one, two = base / "UNIT_other_one/data", base / "UNIT_other_two/data"
            one.mkdir(parents=True); two.mkdir(parents=True)
            (one / "UNIT_file.txt").write_text("UNIT immutable source")
            before = inventory(base)
            request = launch.resolve_request(self.args(base, "native", extra=("--cache-sources", str(one), str(two))), environ={})
            self.assertEqual(request["cache_sources"], [one, two])
            self.assertEqual(request["checkpoint"], base / "v18_native_m10_seed42_memory/native/checkpoint_latest.pt")
            self.assertEqual(before, inventory(base))

    def test_v19_shared_historical_data_namespace_is_rejected_before_execution(self):
        with self.directory() as directory:
            base = Path(directory); root = base / "v19_native_fixed_m10_seed42"
            sealed(root, family="comparison", arm="native_fixed", data_root=base / "UNIT_shared_data")
            before = inventory(base)
            with self.assertRaisesRegex(ValueError, "own data namespace"):
                launch.resolve_request(self.args(base, "native_fixed"), environ={})
            self.assertEqual(before, inventory(base))

    def active_fixture(self, base):
        self.four_roots(base)
        request = launch.resolve_request(self.args(base, "selected"), environ={})
        lock = request["experiment"] / ".pipeline.lock"
        write_json(lock, dict(host=launch.socket.gethostname(), pid=713, token="UNIT_owner"))
        process = SimpleNamespace(
            create_time=Mock(return_value=123.5),
            cmdline=Mock(return_value=["python", "/UNIT/tools/run_comparison_arm.py", "--owned-child",
                "--arm", "selected", "--experiment", str(request["experiment"])]),
            exe=Mock(return_value="/UNIT/python"),
            uids=Mock(return_value=SimpleNamespace(real=17)),
            username=Mock(return_value="UNIT_same_owner"),
            terminate=Mock(side_effect=AssertionError("Ownership reporting must never terminate a process")))
        return request, lock, process

    def test_existing_exact_live_owner_is_reported_without_duplicate_or_signals(self):
        with self.directory() as directory:
            request, lock, process = self.active_fixture(Path(directory)); before = lock.read_bytes()
            with patch.object(psutil, "Process", return_value=process), \
                    patch.object(launch.os, "getuid", return_value=17, create=True):
                result = launch._active_owner(request)
            self.assertEqual(result["status"], "ALREADY_RUNNING")
            self.assertFalse(result["duplicate_started"])
            self.assertEqual(result["pid"], 713)
            process.terminate.assert_not_called()
            self.assertEqual(lock.read_bytes(), before)

    def test_live_owner_with_mismatched_arm_command_user_or_pid_creation_is_rejected(self):
        for difference in ("arm", "experiment", "executable", "entry", "user", "create_time"):
            with self.subTest(difference=difference), self.directory() as directory:
                request, lock, process = self.active_fixture(Path(directory)); before = lock.read_bytes()
                args = process.cmdline.return_value
                if difference == "arm": args[args.index("--arm")+1] = "native"
                elif difference == "experiment": args[args.index("--experiment")+1] = "/UNIT/another"
                elif difference == "executable": process.exe.return_value = "/UNIT/bash"
                elif difference == "entry": args[1] = "/UNIT/unrecognized.py"
                elif difference == "user": process.uids.return_value.real = 18
                else: process.create_time.side_effect = [123.5, 124.5]
                with patch.object(psutil, "Process", return_value=process), \
                        patch.object(launch.os, "getuid", return_value=17, create=True):
                    with self.assertRaisesRegex(RuntimeError, "not the verified requested arm"):
                        launch._active_owner(request)
                process.terminate.assert_not_called()
                self.assertEqual(lock.read_bytes(), before)

    def test_stale_owner_is_left_for_existing_lock_recovery_without_removal(self):
        with self.directory() as directory:
            request, lock, _ = self.active_fixture(Path(directory)); before = lock.read_bytes()
            with patch.object(psutil, "Process", side_effect=psutil.NoSuchProcess(713)):
                self.assertIsNone(launch._active_owner(request))
            self.assertEqual(lock.read_bytes(), before)

    def test_child_reuses_owned_receipt_context_and_does_not_copy_advanced_progress(self):
        from hiercp_v1x import owned_continuation
        from tools import resume_comparison_cached
        with self.directory() as directory:
            base = Path(directory); self.four_roots(base)
            args = self.args(base, "selected", gpu=6)
            request = launch.resolve_request(args, environ={})
            before = inventory(base); seen = []
            original_prepare = u_bridge_continuation.prepare_continuation
            def mocked_run(supplied):
                self.assertIsNot(u_bridge_continuation.prepare_continuation, original_prepare)
                seen.append(supplied)
                return dict(status="UNIT_no_neural_run")
            with patch.object(resume_comparison_cached, "run", side_effect=mocked_run), \
                    patch.object(owned_continuation, "verify_owned_continuation", wraps=owned_continuation.verify_owned_continuation):
                result = launch._child(args, request)
            self.assertEqual(result["status"], "UNIT_no_neural_run")
            self.assertEqual(len(seen), 1)
            self.assertEqual(seen[0].gpu, 6)
            self.assertEqual(seen[0].arm, "selected")
            self.assertEqual(seen[0].experiment, request["experiment"])
            self.assertEqual(seen[0].cache_sources, request["cache_sources"])
            self.assertIs(u_bridge_continuation.prepare_continuation, original_prepare)
            self.assertEqual(before, inventory(base))

    def test_child_initial_clone_copies_saved_progress_then_keeps_advanced_destination_on_retry(self):
        from tools import resume_comparison_cached
        with self.directory() as directory:
            base = Path(directory); source = base / "v18_u_bridge_m10_seed42"
            sealed(source, arm="selected"); source_before = inventory(source)
            args = self.args(base, "selected")
            request = launch.resolve_request(args, environ={})
            self.assertTrue(request["requires_clone"])
            with patch.object(resume_comparison_cached, "run", return_value=dict(status="UNIT_no_neural_run")):
                launch._child(args, request)
            self.assertEqual(request["checkpoint"].read_bytes(), (source / "selected/checkpoint_latest.pt").read_bytes())
            self.assertEqual(source_before, inventory(source))
            request["checkpoint"].write_bytes(b"UNIT advanced private checkpoint")
            repeated = launch.resolve_request(args, environ={})
            self.assertFalse(repeated["requires_clone"])
            with patch.object(resume_comparison_cached, "run", return_value=dict(status="UNIT_no_neural_run")):
                launch._child(args, repeated)
            self.assertEqual(request["checkpoint"].read_bytes(), b"UNIT advanced private checkpoint")
            self.assertEqual(source_before, inventory(source))

    def test_child_failure_restores_owned_continuation_alias_and_preserves_checkpoints(self):
        from tools import resume_comparison_cached
        with self.directory() as directory:
            base = Path(directory); self.four_roots(base)
            args = self.args(base, "native")
            request = launch.resolve_request(args, environ={})
            before = inventory(base); original = u_bridge_continuation.prepare_continuation
            with patch.object(resume_comparison_cached, "run", side_effect=RuntimeError("UNIT failed before neural work")):
                with self.assertRaisesRegex(RuntimeError, "UNIT failed"):
                    launch._child(args, request)
            self.assertIs(u_bridge_continuation.prepare_continuation, original)
            self.assertEqual(before, inventory(base))

    def test_supervised_child_command_preserves_arm_gpu_inventory_and_owned_routes(self):
        from hiercp_v1x import arm_process
        with self.directory() as directory:
            base = Path(directory); self.four_roots(base)
            args = self.args(base, "native_listwise", gpu=6)
            before = inventory(base)
            result = dict(child_returncode=0, paused=False, status="UNIT_no_process_spawned")
            with patch.object(launch, "_active_owner", return_value=None), \
                    patch.object(arm_process, "run_owned", return_value=result) as supervise, \
                    redirect_stdout(StringIO()):
                self.assertEqual(launch.run(args), result)
            command = supervise.call_args.args[0]
            self.assertEqual(command[0], launch.sys.executable)
            self.assertEqual(command[1:3], ["-B", "-u"])
            child = launch.parse(command[4:])
            self.assertTrue(child.owned_child)
            self.assertEqual((child.arm, child.gpu), ("native_listwise", 6))
            self.assertEqual(child.inventory, base / "UNIT_inventory.json")
            request = launch.resolve_request(child, environ={})
            self.assertEqual(request["experiment"], base / "v19_native_listwise_m10_seed42")
            self.assertEqual(supervise.call_args.kwargs["checkpoint"], request["checkpoint"])
            self.assertEqual(supervise.call_args.kwargs["grace_seconds"], 10)
            self.assertEqual(before, inventory(base))

    def test_existing_live_arm_parent_never_spawns_duplicate_supervisor(self):
        from hiercp_v1x import arm_process
        with self.directory() as directory:
            base = Path(directory); self.four_roots(base)
            args = self.args(base, "selected")
            active = dict(status="ALREADY_RUNNING", arm="selected", pid=713, duplicate_started=False)
            with patch.object(launch, "_active_owner", return_value=active), \
                    patch.object(arm_process, "run_owned", side_effect=AssertionError("Do not start duplicate")) as supervise, \
                    redirect_stdout(StringIO()):
                self.assertEqual(launch.run(args), active)
            supervise.assert_not_called()

    def test_owned_child_dispatch_does_not_spawn_another_parent_or_supervisor(self):
        from hiercp_v1x import arm_process
        with self.directory() as directory:
            base = Path(directory); self.four_roots(base)
            args = self.args(base, "native", extra=("--owned-child",))
            result = dict(status="UNIT_direct_owned_child")
            with patch.object(launch, "_child", return_value=result) as child, \
                    patch.object(launch, "_active_owner", side_effect=AssertionError("Owned child delegates to its locks")), \
                    patch.object(arm_process, "run_owned", side_effect=AssertionError("Do not recurse supervisor")):
                self.assertEqual(launch.run(args), result)
            child.assert_called_once()
            self.assertEqual(child.call_args.args[1]["data_root"], base / "v18_native_m10_seed42_memory/data")

    def test_real_four_processes_hold_eight_independent_locks_and_release_on_stdin_EOF(self):
        """REAL CPU/file lock isolation only; no model, GPU or neural progress."""
        code = """import json, os, sys
from pathlib import Path
from hiercp_v1x.u_bridge_experiment import lock
root = Path(sys.argv[1])
with lock(root / '.pipeline.lock'), lock(root / 'data/.data.lock'):
    print(json.dumps(dict(stage='UNIT_READY_NO_NEURAL_RUN', pid=os.getpid(), root=str(root))), flush=True)
    sys.stdin.read()
"""
        with self.directory() as directory:
            base = Path(directory)
            historical = base / "UNIT_historical/.pipeline.lock"
            write_json(historical, dict(host=launch.socket.gethostname(), pid=os.getpid(), token="UNIT_active_historical"))
            roots = [base / arm for arm in ARMS]
            for root in roots: (root / "data").mkdir(parents=True)
            before = inventory(base); children = []; ready = queue.Queue()
            def read_ready(child):
                ready.put((child.pid, child.stdout.readline()))
            try:
                for root in roots:
                    # CPython's Windows venv executable is a redirector. Use
                    # the actual interpreter for these stdlib-only children
                    # so Popen PID and the lock's owner PID are identical.
                    child = subprocess.Popen([getattr(sys, "_base_executable", sys.executable), "-B", "-u", "-c", code, str(root)],
                        cwd=REPOSITORY, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                        text=True, encoding="utf8", shell=False)
                    children.append(child)
                    threading.Thread(target=read_ready, args=(child,), daemon=True).start()
                deadline = time.monotonic() + 20
                records = {}
                for _ in children:
                    try:
                        pid, line = ready.get(timeout=max(.01, deadline-time.monotonic()))
                    except queue.Empty:
                        self.fail("Bounded UNIT READY wait expired; owned children receive stdin EOF in cleanup")
                    self.assertTrue(line, f"UNIT child {pid} closed stdout before READY")
                    row = json.loads(line)
                    self.assertEqual(row["stage"], "UNIT_READY_NO_NEURAL_RUN")
                    self.assertEqual(row["pid"], pid); records[pid] = row
                self.assertEqual(len(records), 4)
                for child, root in zip(children, roots):
                    self.assertIsNone(child.poll(), "All four independent owners must be alive together")
                    for path in (root / ".pipeline.lock", root / "data/.data.lock"):
                        self.assertEqual(json.loads(path.read_text())["pid"], child.pid)
                self.assertEqual(historical.read_bytes(), (json.dumps(dict(host=launch.socket.gethostname(),
                    pid=os.getpid(), token="UNIT_active_historical"), sort_keys=True)).encode())
            finally:
                # These exact children were created here. Cooperative EOF only:
                # no terminate, kill, session/group signal or parent-shell action.
                for child in children:
                    if child.stdin is not None:
                        child.stdin.close(); child.stdin = None
                for child in children:
                    try:
                        _, stderr = child.communicate(timeout=20)
                    except subprocess.TimeoutExpired as error:
                        raise RuntimeError(f"Owned UNIT child {child.pid} pending after stdin EOF; no signal sent") from error
                    self.assertEqual(child.returncode, 0, stderr)
            for root in roots:
                self.assertFalse((root / ".pipeline.lock").exists())
                self.assertFalse((root / "data/.data.lock").exists())
            self.assertEqual(before, inventory(base))

    def test_pause_does_not_hide_real_child_failure_and_SIGTERM_pause_remains_expected(self):
        from hiercp_v1x import arm_process
        with self.directory() as directory:
            base = Path(directory); self.four_roots(base)
            args = self.args(base, "native_fixed"); before = inventory(base)
            with patch.object(launch, "_active_owner", return_value=None), redirect_stdout(StringIO()):
                with patch.object(arm_process, "run_owned", return_value=dict(child_returncode=1, paused=True)):
                    with self.assertRaisesRegex(RuntimeError, "child failed"):
                        launch.run(args)
                expected = dict(child_returncode=-15, paused=True)
                with patch.object(arm_process, "run_owned", return_value=expected):
                    self.assertEqual(launch.run(args), expected)
            self.assertEqual(before, inventory(base))

    def test_child_uses_current_environment_selector_and_records_owned_policy(self):
        from tools import current_gpu, local_cnn_device, resume_comparison_cached
        with self.directory() as directory:
            base = Path(directory); self.four_roots(base)
            args = self.args(base, "native_listwise", gpu=6)
            request = launch.resolve_request(args)
            frozen_before = {name:sha(REPOSITORY/name) for name in comparison_experiment.FILES}
            before_checkpoint = sha(request["checkpoint"])
            original_select = local_cnn_device.select
            selected = dict(physical_index=6, backend="NVML", visible="GPU-UNIT-six", kind="GPU")
            from contextlib import contextmanager
            @contextmanager
            def device_context(*, on_select):
                def selector(index):
                    self.assertEqual(index, 6)
                    on_select(selected)
                    return selected["visible"]
                local_cnn_device.select = selector
                try:
                    yield
                finally:
                    local_cnn_device.select = original_select
            def controller(supplied):
                from tools.local_cnn_device import select
                self.assertEqual(select(supplied.gpu), "GPU-UNIT-six")
                return dict(status="UNIT_metadata_only")
            with patch.object(current_gpu, "current_device_selection", device_context), \
                    patch.object(resume_comparison_cached, "run", side_effect=controller):
                self.assertEqual(launch._child(args, request)["status"], "UNIT_metadata_only")
            receipt_path, = (request["experiment"]/"execution_overrides").glob("device_*.json")
            receipt = json.loads(receipt_path.read_text())
            self.assertEqual(receipt["requested_physical_GPU"], 6)
            self.assertEqual(receipt["selections"], [selected])
            self.assertFalse(receipt["nvidia_smi_invoked"])
            self.assertFalse(receipt["remote_connection_invoked"])
            self.assertEqual(receipt["helpers"], {name:sha(REPOSITORY/name) for name in
                             ("tools/current_gpu.py", "tools/run_comparison_arm.py")})
            self.assertEqual(frozen_before, {name:sha(REPOSITORY/name) for name in comparison_experiment.FILES})
            self.assertEqual(before_checkpoint, sha(request["checkpoint"]))
            self.assertIs(local_cnn_device.select, original_select)


if __name__ == "__main__":
    unittest.main()
