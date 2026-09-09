"""Read-only resource telemetry for a bank build; never changes its workload."""
from __future__ import annotations

import json
from pathlib import Path
import threading
import time
import uuid

from hiercp.preparation_runtime import snapshot


class BankProgress:
    """One owned JSONL per invocation, with visible progress during long stages."""

    def __init__(self, bank_root, *, heartbeat_seconds=30.0):
        if heartbeat_seconds <= 0:
            raise ValueError("Bank heartbeat interval must be positive")
        self.path = Path(bank_root) / f"preparation_progress.{uuid.uuid4().hex}.jsonl"
        self.interval = float(heartbeat_seconds)
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.error = None
        self.handle = None
        self.thread = None
        self.fields = {}
        self.phase = "initializing"

    def __enter__(self):
        self.started = self.phase_started = time.perf_counter()
        self.handle = self.path.open("x", encoding="utf-8")
        try:
            self._emit("start")
            self.thread = threading.Thread(target=self._monitor, name="bank-progress", daemon=True)
            self.thread.start()
        except BaseException:
            self.handle.close()
            raise
        return self

    def _check(self):
        if self.error is not None:
            raise RuntimeError("Bank progress resource measurement failed") from self.error

    def _emit_locked(self, event, **extra):
        now = time.perf_counter()
        row = {"format": "hiercp_bank_progress_v1", "event": event,
               "phase": self.phase, "elapsed_seconds": now - self.started,
               "phase_seconds": now - self.phase_started, **self.fields,
               "resources": snapshot(), **extra}
        encoded = json.dumps(row, allow_nan=False)
        self.handle.write(encoded + "\n")
        self.handle.flush()
        print("[BankProgress] " + encoded, flush=True)

    def _emit(self, event, **extra):
        with self.lock:
            self._emit_locked(event, **extra)

    def update(self, phase, **fields):
        self._check()
        with self.lock:
            self._emit_locked("phase_end")
            self.phase, self.fields = str(phase), dict(fields)
            self.phase_started = time.perf_counter()
            self._emit_locked("phase_start")

    def counters(self, **fields):
        self._check()
        with self.lock:
            self.fields.update(fields)

    def _monitor(self):
        try:
            while not self.stop.wait(self.interval):
                self._emit("heartbeat")
        except Exception as exc:
            self.error = exc
            self.stop.set()

    def __exit__(self, exc_type, exc, traceback):
        self.stop.set()
        if self.thread is not None:
            self.thread.join()
        try:
            self._check()
            self._emit("failed" if exc is not None else "preparation_finished",
                       error=None if exc is None else f"{type(exc).__name__}: {exc}")
        except Exception as telemetry_error:
            if exc is None:
                raise
            # Keep the original domain failure while making telemetry failure visible.
            print(f"[BankProgressError] {type(telemetry_error).__name__}: {telemetry_error}", flush=True)
        finally:
            self.handle.close()
        return False
