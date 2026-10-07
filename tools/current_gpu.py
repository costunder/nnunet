"""Select an explicit physical GPU with the currently available NVML library.

This execution adapter requires no nvidia-smi executable or CUDA ordinal guess.
The preserved project resolver still enforces current/recorded MIG allocation.
It imports neither PyTorch nor CUDA before completing device selection.
"""
from __future__ import annotations

from contextlib import contextmanager
import ctypes
import ctypes.util
import getpass
import os
import socket
import sys
import threading

NVML_SUCCESS = 0
NVML_ERROR_NOT_SUPPORTED = 3
NVML_ERROR_NOT_FOUND = 6
_CONTEXT = threading.RLock()


class NvmlError(RuntimeError):
    def __init__(self, operation, status, detail):
        self.operation, self.status = operation, int(status)
        super().__init__(f"{operation} failed: NVML status={self.status} ({detail})")


def _load_library():
    discovered = ctypes.util.find_library("nvidia-ml")
    name = discovered or ("nvml.dll" if sys.platform == "win32" else "libnvidia-ml.so.1")
    try:
        return ctypes.CDLL(name), name
    except OSError as error:
        raise RuntimeError(f"Current environment cannot load NVML library {name}: {error}; "
                           "no CLI, CPU or CUDA-ordinal fallback was attempted") from error


def _bind(library, name, arguments, result=ctypes.c_int, *, required=True):
    try:
        function = getattr(library, name)
    except AttributeError as error:
        if not required:
            return None
        raise RuntimeError(f"Current NVML library does not provide {name}") from error
    function.argtypes, function.restype = arguments, result
    return function


class _Driver:
    def __init__(self, library):
        handle = ctypes.c_void_p
        unsigned = ctypes.c_uint
        pointer = ctypes.POINTER
        self.init = _bind(library, "nvmlInit_v2", [])
        self.shutdown = _bind(library, "nvmlShutdown", [])
        self.count = _bind(library, "nvmlDeviceGetCount_v2", [pointer(unsigned)])
        self.by_index = _bind(library, "nvmlDeviceGetHandleByIndex_v2", [unsigned, pointer(handle)])
        self.index = _bind(library, "nvmlDeviceGetIndex", [handle, pointer(unsigned)])
        self.uuid = _bind(library, "nvmlDeviceGetUUID", [handle, pointer(ctypes.c_char), unsigned])
        self.name = _bind(library, "nvmlDeviceGetName", [handle, pointer(ctypes.c_char), unsigned])
        self.mig_mode = _bind(library, "nvmlDeviceGetMigMode", [handle, pointer(unsigned), pointer(unsigned)], required=False)
        self.max_mig = _bind(library, "nvmlDeviceGetMaxMigDeviceCount", [handle, pointer(unsigned)], required=False)
        self.mig_handle = _bind(library, "nvmlDeviceGetMigDeviceHandleByIndex", [handle, unsigned, pointer(handle)], required=False)
        self.error = _bind(library, "nvmlErrorString", [ctypes.c_int], ctypes.c_char_p, required=False)

    def check(self, operation, status):
        status = int(status)
        if status != NVML_SUCCESS:
            detail = self.error(status) if self.error is not None else None
            detail = detail.decode("utf8", errors="replace") if isinstance(detail, bytes) else str(detail or "error text unavailable")
            raise NvmlError(operation, status, detail)

    def text(self, getter, operation, handle):
        buffer = ctypes.create_string_buffer(256)
        self.check(operation, getter(handle, buffer, len(buffer)))
        text = buffer.value.decode("utf8", errors="strict")
        if not text or any(character in text for character in ("\n", "\r")):
            raise ValueError(f"{operation} returned an invalid empty/multiline device identity")
        return text


def inventory(index, *, library=None):
    """Return actual NVML identity for exactly the requested physical index.

    ``library`` is a ctypes-compatible injection boundary for UNIT tests. It is
    never a CPU or inventory fallback. Missing MIG slots are expected only at
    nvmlDeviceGetMigDeviceHandleByIndex; other NVML failures remain errors.
    """
    if type(index) is not int or index < 0:
        raise ValueError("An explicit nonnegative physical GPU integer is required")
    if library is None:
        library, library_name = _load_library()
    else:
        library_name = str(getattr(library, "_name", "UNIT_injected_NVML"))
    driver = _Driver(library)
    driver.check("nvmlInit_v2", driver.init())
    try:
        count = ctypes.c_uint()
        driver.check("nvmlDeviceGetCount_v2", driver.count(ctypes.byref(count)))
        if index >= count.value:
            raise ValueError(f"Physical GPU {index} is unavailable in current NVML inventory; actual GPU count={count.value}")
        handle = ctypes.c_void_p()
        driver.check("nvmlDeviceGetHandleByIndex_v2", driver.by_index(index, ctypes.byref(handle)))
        actual = ctypes.c_uint()
        driver.check("nvmlDeviceGetIndex", driver.index(handle, ctypes.byref(actual)))
        if actual.value != index:
            raise ValueError(f"NVML physical GPU identity differs: requested={index}, returned={actual.value}; no ordinal remapping")
        uuid = driver.text(driver.uuid, "nvmlDeviceGetUUID", handle)
        name = driver.text(driver.name, "nvmlDeviceGetName", handle)
        if not uuid.startswith("GPU-") or ")" in uuid:
            raise ValueError("NVML returned an invalid physical GPU UUID")
        current, pending = ctypes.c_uint(), ctypes.c_uint()
        if driver.mig_mode is None:
            raise RuntimeError("Current NVML library cannot query MIG mode; device allocation cannot be established")
        status = int(driver.mig_mode(handle, ctypes.byref(current), ctypes.byref(pending)))
        supported = status != NVML_ERROR_NOT_SUPPORTED
        if supported:
            driver.check("nvmlDeviceGetMigMode", status)
            if current.value not in (0, 1) or pending.value not in (0, 1):
                raise ValueError("NVML returned an unknown MIG mode")
        mig = []
        if supported and current.value == 1:
            if driver.max_mig is None or driver.mig_handle is None:
                raise RuntimeError("MIG is enabled but current NVML library cannot enumerate its instances")
            maximum = ctypes.c_uint()
            driver.check("nvmlDeviceGetMaxMigDeviceCount", driver.max_mig(handle, ctypes.byref(maximum)))
            for slot in range(maximum.value):
                child = ctypes.c_void_p()
                status = int(driver.mig_handle(handle, slot, ctypes.byref(child)))
                if status == NVML_ERROR_NOT_FOUND:
                    continue
                driver.check("nvmlDeviceGetMigDeviceHandleByIndex", status)
                child_uuid = driver.text(driver.uuid, "nvmlDeviceGetUUID(MIG)", child)
                child_name = driver.text(driver.name, "nvmlDeviceGetName(MIG)", child)
                if not child_uuid.startswith("MIG-") or ")" in child_uuid:
                    raise ValueError("NVML returned an invalid MIG UUID")
                mig.append(dict(slot=slot, uuid=child_uuid, name=child_name))
            if not mig:
                raise ValueError("MIG is enabled but no compute instance is available; no full-GPU fallback")
            if len({child["uuid"] for child in mig}) != len(mig):
                raise ValueError("NVML returned duplicate MIG instance identities")
        lines = [f"GPU {index}: {name} (UUID: {uuid})"]
        lines.extend(f"  MIG {child['name']} Device {child['slot']}: (UUID: {child['uuid']})" for child in mig)
        return dict(physical_index=index, physical_uuid=uuid, name=name, mig=mig,
                    mig_supported=supported, mig_enabled=supported and current.value == 1,
                    mig_pending=pending.value if supported else None,
                    listing="\n".join(lines), backend="NVML", library=library_name,
                    current_host_observed=socket.gethostname())
    finally:
        active = sys.exc_info()[1]
        try:
            driver.check("nvmlShutdown", driver.shutdown())
        except NvmlError as error:
            if active is not None:
                raise RuntimeError(f"NVML operation failed ({active}); NVML reference cleanup also failed ({error})") from active
            raise


def listing(index, *, library=None):
    return inventory(index, library=library)["listing"]


def select_record(index):
    from tools import local_cnn_device
    observed = inventory(index)
    allocation = local_cnn_device.recorded_allocation(index, socket.gethostname(), getpass.getuser())
    visible, kind = local_cnn_device.resolve(index, observed["listing"],
        os.environ.get("CUDA_VISIBLE_DEVICES", ""), allocation)
    # Inspect an already imported module only; this adapter never imports torch.
    imported = sys.modules.get("torch")
    initialized = getattr(getattr(imported, "cuda", None), "_initialized", False)
    if initialized and os.environ.get("CUDA_VISIBLE_DEVICES", "") != visible:
        raise RuntimeError("CUDA is already initialized with a different device selection; no device remapping")
    os.environ["CUDA_VISIBLE_DEVICES"] = visible
    print(f"GPU selection | backend=NVML | physical={index} | type={kind} | CUDA_VISIBLE_DEVICES={visible}", flush=True)
    return dict(observed, visible=visible, type=kind)


def select(index):
    return select_record(index)["visible"]


@contextmanager
def current_device_selection(on_select=None):
    """Patch nested frozen selectors for one process; restore on all exits.

    Once selected, repeated calls reuse the exact same UUID after CUDA starts.
    A different physical index or changed environment is rejected. Selection
    state never escapes this context. The optional callback receives successful
    observed identity only, after environment selection is complete.
    """
    if on_select is not None and not callable(on_select):
        raise TypeError("Selection callback must be callable")
    from tools import local_cnn_device
    with _CONTEXT:
        original = local_cnn_device.select
        selected = None
        def scoped(index):
            nonlocal selected
            if selected is not None:
                if type(index) is not int or index != selected["physical_index"]:
                    raise ValueError("One execution context cannot change its physical GPU")
                if os.environ.get("CUDA_VISIBLE_DEVICES", "") != selected["visible"]:
                    raise RuntimeError("Selected CUDA_VISIBLE_DEVICES changed within the execution context")
                return selected["visible"]
            selected = select_record(index)
            if on_select is not None:
                on_select(dict(selected))
            return selected["visible"]
        local_cnn_device.select = scoped
        try:
            yield
        finally:
            local_cnn_device.select = original
