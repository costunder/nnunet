"""UNIT ctypes/NVML device selection boundaries; no CUDA/model execution."""
from __future__ import annotations

import ctypes
from contextlib import redirect_stdout
from io import StringIO
import os
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools import current_gpu, local_cnn_device


def value(item):
    return item.value if hasattr(item, "value") else item


def put(pointer, kind, item):
    ctypes.cast(pointer, ctypes.POINTER(kind))[0] = item


class Function:
    def __init__(self, function):
        self.function, self.calls = function, []
    def __call__(self, *arguments):
        self.calls.append(arguments)
        return self.function(*arguments)


class FakeNvml:
    """Synthetic C-call boundary fixture, never an actual GPU inventory."""
    _name = "UNIT_injected_NVML"
    def __init__(self, *, count=7, mode=0, children=None, statuses=None, index_override=None):
        self.gpu_count, self.mode = count, mode
        self.children = children or {}
        self.statuses, self.index_override = statuses or {}, index_override
        self.nvmlInit_v2 = Function(lambda: self.status("init"))
        self.nvmlShutdown = Function(lambda: self.status("shutdown"))
        self.nvmlErrorString = Function(lambda code: f"UNIT NVML error {value(code)}".encode())
        self.nvmlDeviceGetCount_v2 = Function(self.count)
        self.nvmlDeviceGetHandleByIndex_v2 = Function(self.handle)
        self.nvmlDeviceGetIndex = Function(self.index)
        self.nvmlDeviceGetUUID = Function(self.uuid)
        self.nvmlDeviceGetName = Function(self.name)
        self.nvmlDeviceGetMigMode = Function(self.mig_mode)
        self.nvmlDeviceGetMaxMigDeviceCount = Function(self.max_mig)
        self.nvmlDeviceGetMigDeviceHandleByIndex = Function(self.mig_handle)
    def status(self, operation):
        return self.statuses.get(operation, 0)
    def count(self, pointer):
        if not self.status("count"):
            put(pointer, ctypes.c_uint, self.gpu_count)
        return self.status("count")
    def handle(self, index, pointer):
        if not self.status("handle"):
            put(pointer, ctypes.c_void_p, 1000 + value(index))
        return self.status("handle")
    def index(self, handle, pointer):
        if not self.status("index"):
            put(pointer, ctypes.c_uint, self.index_override if self.index_override is not None else value(handle) - 1000)
        return self.status("index")
    def uuid(self, handle, buffer, length):
        if not self.status("uuid"):
            identifier = self.children[value(handle) - 2000] if value(handle) >= 2000 else f"GPU-UNIT-{value(handle) - 1000}"
            buffer.value = identifier.encode()
        return self.status("uuid")
    def name(self, handle, buffer, length):
        if not self.status("name"):
            buffer.value = b"NVIDIA UNIT MIG instance" if value(handle) >= 2000 else b"NVIDIA UNIT physical GPU"
        return self.status("name")
    def mig_mode(self, handle, current, pending):
        if not self.status("mig_mode"):
            put(current, ctypes.c_uint, self.mode)
            put(pending, ctypes.c_uint, self.mode)
        return self.status("mig_mode")
    def max_mig(self, handle, pointer):
        if not self.status("max_mig"):
            put(pointer, ctypes.c_uint, 7)
        return self.status("max_mig")
    def mig_handle(self, handle, slot, pointer):
        slot = value(slot)
        code = self.statuses.get(("mig_handle", slot), 0 if slot in self.children else 6)
        if not code:
            put(pointer, ctypes.c_void_p, 2000 + slot)
        return code


class NvmlInventoryTests(unittest.TestCase):
    def test_exact_physical_index_uuid_and_name_without_cuda_ordinal_mapping(self):
        library = FakeNvml()
        result = current_gpu.inventory(3, library=library)
        self.assertEqual(result["physical_index"], 3)
        self.assertEqual(result["physical_uuid"], "GPU-UNIT-3")
        self.assertEqual(result["backend"], "NVML")
        self.assertEqual(result["library"], "UNIT_injected_NVML")
        self.assertEqual(local_cnn_device.resolve(3, result["listing"]), ("GPU-UNIT-3", "GPU"))
        self.assertEqual(len(library.nvmlDeviceGetHandleByIndex_v2.calls), 1)
        self.assertEqual(library.nvmlDeviceGetHandleByIndex_v2.calls[0][0], 3)
        self.assertEqual(len(library.nvmlShutdown.calls), 1)

    def test_ctypes_signatures_match_nvml_boundary(self):
        library = FakeNvml()
        current_gpu.inventory(2, library=library)
        self.assertEqual(library.nvmlDeviceGetCount_v2.argtypes, [ctypes.POINTER(ctypes.c_uint)])
        self.assertEqual(library.nvmlDeviceGetHandleByIndex_v2.argtypes,
                         [ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)])
        self.assertEqual(library.nvmlDeviceGetUUID.argtypes,
                         [ctypes.c_void_p, ctypes.POINTER(ctypes.c_char), ctypes.c_uint])
        self.assertEqual(library.nvmlDeviceGetMigDeviceHandleByIndex.argtypes,
                         [ctypes.c_void_p, ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)])
        self.assertIs(library.nvmlDeviceGetName.restype, ctypes.c_int)
        self.assertIs(library.nvmlErrorString.restype, ctypes.c_char_p)

    def test_non_mig_gpu_not_supported_status_is_expected_only_for_mode(self):
        library = FakeNvml(statuses={"mig_mode": 3})
        result = current_gpu.inventory(2, library=library)
        self.assertFalse(result["mig_supported"])
        self.assertFalse(result["mig_enabled"])
        self.assertFalse(library.nvmlDeviceGetMaxMigDeviceCount.calls)
        with self.assertRaises(current_gpu.NvmlError) as error:
            current_gpu.inventory(2, library=FakeNvml(statuses={"uuid": 3}))
        self.assertEqual(error.exception.status, 3)

    def test_mig_holes_preserve_actual_slot_and_unique_uuid(self):
        library = FakeNvml(mode=1, children={1: "MIG-UNIT-a", 5: "MIG-UNIT-b"})
        result = current_gpu.inventory(6, library=library)
        self.assertEqual([child["slot"] for child in result["mig"]], [1, 5])
        self.assertEqual(local_cnn_device.resolve(6, result["listing"], "MIG-UNIT-b"), ("MIG-UNIT-b", "MIG"))
        self.assertEqual(len(library.nvmlDeviceGetMigDeviceHandleByIndex.calls), 7)
        self.assertEqual(len(library.nvmlShutdown.calls), 1)

    def test_enabled_mig_empty_or_duplicate_inventory_never_falls_back(self):
        with self.assertRaisesRegex(ValueError, "no full-GPU fallback"):
            current_gpu.inventory(6, library=FakeNvml(mode=1))
        with self.assertRaisesRegex(ValueError, "duplicate MIG"):
            current_gpu.inventory(6, library=FakeNvml(mode=1, children={0: "MIG-UNIT-a", 1: "MIG-UNIT-a"}))

    def test_driver_permissions_and_mig_errors_are_never_swallowed(self):
        for operation in ("count", "handle", "index", "uuid", "name", "mig_mode", "max_mig"):
            library = FakeNvml(mode=1, children={0: "MIG-UNIT-a"}, statuses={operation: 4})
            with self.subTest(operation=operation), self.assertRaises(current_gpu.NvmlError) as error:
                current_gpu.inventory(6, library=library)
            self.assertEqual(error.exception.status, 4)
            self.assertEqual(len(library.nvmlShutdown.calls), 1)
        for code in (3, 4, 9):
            library = FakeNvml(mode=1, children={0: "MIG-UNIT-a"}, statuses={("mig_handle", 0): code})
            with self.subTest(slot_status=code), self.assertRaises(current_gpu.NvmlError) as error:
                current_gpu.inventory(6, library=library)
            self.assertEqual(error.exception.status, code)

    def test_unavailable_driver_init_is_reported_without_shutdown_uninitialized_library(self):
        library = FakeNvml(statuses={"init": 9})
        with self.assertRaises(current_gpu.NvmlError) as error:
            current_gpu.inventory(2, library=library)
        self.assertEqual(error.exception.operation, "nvmlInit_v2")
        self.assertFalse(library.nvmlShutdown.calls)
        self.assertFalse(library.nvmlDeviceGetCount_v2.calls)

    def test_out_of_range_or_remapped_physical_index_is_rejected(self):
        library = FakeNvml(count=1)
        with self.assertRaisesRegex(ValueError, "actual GPU count=1"):
            current_gpu.inventory(3, library=library)
        self.assertFalse(library.nvmlDeviceGetHandleByIndex_v2.calls)
        with self.assertRaisesRegex(ValueError, "no ordinal remapping"):
            current_gpu.inventory(3, library=FakeNvml(index_override=0))
        for wrong in (-1, True, "3", 3.):
            with self.subTest(wrong=wrong), self.assertRaises(ValueError):
                current_gpu.inventory(wrong, library=FakeNvml())

    def test_missing_mig_query_api_is_explicit_and_cleanup_still_occurs(self):
        library = FakeNvml()
        del library.nvmlDeviceGetMigMode
        with self.assertRaisesRegex(RuntimeError, "cannot query MIG mode"):
            current_gpu.inventory(2, library=library)
        self.assertEqual(len(library.nvmlShutdown.calls), 1)

    def test_library_cleanup_failure_reports_primary_operation_too(self):
        with self.assertRaises(current_gpu.NvmlError) as error:
            current_gpu.inventory(2, library=FakeNvml(statuses={"shutdown": 999}))
        self.assertEqual(error.exception.operation, "nvmlShutdown")
        with self.assertRaisesRegex(RuntimeError, "operation failed.*cleanup also failed"):
            current_gpu.inventory(2, library=FakeNvml(statuses={"count": 4, "shutdown": 999}))

    def test_library_discovery_uses_current_environment_without_cli_or_system_path(self):
        library = FakeNvml()
        with patch.object(current_gpu.ctypes.util, "find_library", return_value="UNIT_discovered_nvml"), \
                patch.object(current_gpu.ctypes, "CDLL", return_value=library) as loader, \
                patch("subprocess.check_output", side_effect=AssertionError("No CLI is allowed")):
            result = current_gpu.inventory(2)
        loader.assert_called_once_with("UNIT_discovered_nvml")
        self.assertEqual(result["library"], "UNIT_discovered_nvml")
        for platform, soname in (("linux", "libnvidia-ml.so.1"), ("win32", "nvml.dll")):
            with self.subTest(platform=platform), \
                    patch.object(current_gpu.ctypes.util, "find_library", return_value=None), \
                    patch.object(current_gpu.sys, "platform", platform), \
                    patch.object(current_gpu.ctypes, "CDLL", return_value=FakeNvml()) as loader:
                current_gpu.inventory(2)
                loader.assert_called_once_with(soname)
        with patch.object(current_gpu.ctypes.util, "find_library", return_value=None), \
                patch.object(current_gpu.ctypes, "CDLL", side_effect=OSError("UNIT library unavailable")), \
                self.assertRaisesRegex(RuntimeError, "no CLI, CPU or CUDA-ordinal fallback"):
            current_gpu.inventory(2)


class ScopedSelectionTests(unittest.TestCase):
    def test_current_mig_allocation_wins_over_stale_recorded_row(self):
        library = FakeNvml(mode=1, children={0: "MIG-UNIT-a", 1: "MIG-UNIT-b"})
        stale = dict(physical_index=6, gpu_uuid="GPU-UNIT-6", mig_uuid="MIG-UNIT-removed")
        with patch.object(current_gpu, "_load_library", return_value=(library, "UNIT_NVML")), \
                patch.object(local_cnn_device, "recorded_allocation", return_value=stale), \
                patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "MIG-UNIT-b"}), redirect_stdout(StringIO()):
            record = current_gpu.select_record(6)
            self.assertEqual(record["visible"], "MIG-UNIT-b")
            self.assertEqual(record["type"], "MIG")

    def test_mig_ambiguity_or_stale_record_never_sets_environment_or_calls_callback(self):
        for visible, allocation in (("6", None), ("MIG-UNIT-a,MIG-UNIT-b", None),
                ("6", dict(physical_index=6, gpu_uuid="GPU-UNIT-6", mig_uuid="MIG-UNIT-removed"))):
            callbacks = []
            library = FakeNvml(mode=1, children={0: "MIG-UNIT-a", 1: "MIG-UNIT-b"})
            with self.subTest(visible=visible, allocation=allocation), \
                    patch.object(current_gpu, "_load_library", return_value=(library, "UNIT_NVML")), \
                    patch.object(local_cnn_device, "recorded_allocation", return_value=allocation), \
                    patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": visible}), current_gpu.current_device_selection(callbacks.append):
                with self.assertRaises(ValueError):
                    local_cnn_device.select(6)
                self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], visible)
                self.assertFalse(callbacks)

    def test_nested_selection_is_idempotent_after_cuda_initializes_and_context_restores(self):
        library = FakeNvml()
        original = local_cnn_device.select
        callbacks = []
        def callback(record):
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], record["visible"])
            callbacks.append(record)
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(_initialized=False))
        with patch.object(current_gpu, "_load_library", return_value=(library, "UNIT_NVML")) as loader, \
                patch.object(local_cnn_device, "recorded_allocation", return_value=None), \
                patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "3"}), \
                patch.dict(sys.modules, {"torch": fake_torch}), redirect_stdout(StringIO()):
            with current_gpu.current_device_selection(callback):
                self.assertEqual(local_cnn_device.select(3), "GPU-UNIT-3")
                fake_torch.cuda._initialized = True
                self.assertEqual(local_cnn_device.select(3), "GPU-UNIT-3")
                with self.assertRaisesRegex(ValueError, "cannot change"):
                    local_cnn_device.select(2)
                os.environ["CUDA_VISIBLE_DEVICES"] = "GPU-UNIT-altered"
                with self.assertRaisesRegex(RuntimeError, "changed within"):
                    local_cnn_device.select(3)
            self.assertEqual(loader.call_count, 1)
        self.assertIs(local_cnn_device.select, original)
        self.assertEqual(len(callbacks), 1)
        self.assertEqual(callbacks[0]["physical_index"], 3)
        self.assertEqual(callbacks[0]["backend"], "NVML")

    def test_selection_cache_does_not_survive_into_next_context(self):
        with patch.object(current_gpu, "_load_library", return_value=(FakeNvml(), "UNIT_NVML")) as loader, \
                patch.object(local_cnn_device, "recorded_allocation", return_value=None), \
                patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}), redirect_stdout(StringIO()):
            with current_gpu.current_device_selection():
                self.assertEqual(local_cnn_device.select(2), "GPU-UNIT-2")
            with current_gpu.current_device_selection():
                self.assertEqual(local_cnn_device.select(3), "GPU-UNIT-3")
            self.assertEqual(loader.call_count, 2)

    def test_callback_exception_restores_selector_and_never_hides_error(self):
        original = local_cnn_device.select
        def failing_callback(record):
            raise RuntimeError("UNIT receipt callback failure")
        with patch.object(current_gpu, "_load_library", return_value=(FakeNvml(), "UNIT_NVML")), \
                patch.object(local_cnn_device, "recorded_allocation", return_value=None), \
                patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}), redirect_stdout(StringIO()):
            with self.assertRaisesRegex(RuntimeError, "UNIT receipt callback failure"):
                with current_gpu.current_device_selection(failing_callback):
                    local_cnn_device.select(3)
        self.assertIs(local_cnn_device.select, original)

    def test_already_initialized_different_cuda_selection_is_not_remapped(self):
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(_initialized=True))
        with patch.object(current_gpu, "_load_library", return_value=(FakeNvml(), "UNIT_NVML")), \
                patch.object(local_cnn_device, "recorded_allocation", return_value=None), \
                patch.dict(sys.modules, {"torch": fake_torch}), \
                patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "GPU-UNIT-2"}):
            with self.assertRaisesRegex(RuntimeError, "CUDA is already initialized"):
                current_gpu.select(3)
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "GPU-UNIT-2")


if __name__ == "__main__":
    unittest.main()
