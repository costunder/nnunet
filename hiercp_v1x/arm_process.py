"""Supervise one explicitly spawned Linux arm without signaling its SSH shell.

The frozen engine publishes checkpoints atomically. An interrupted unsaved
batch can be replayed from the last published checkpoint; this supervisor
never claims that an interrupted batch was newly saved and never edits it.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

FORMAT = 'single_owned_arm_supervision_v1'


def process_identity(pid):
    """Read identity, including the exact command, before any child signal."""
    import psutil
    try:
        process = psutil.Process(pid)
        with process.oneshot():
            return dict(pid=process.pid, parent_pid=process.ppid(),
                        create_time=process.create_time(), argv=process.cmdline(),
                        uid=process.uids().real)
    except psutil.NoSuchProcess:
        return None


def _print(record):
    print('Owned arm | ' + json.dumps(record, allow_nan=False), flush=True)


class OwnedArm:
    """One Popen child and one non-resetting cooperative-stop deadline."""
    def __init__(self, command, *, checkpoint, grace_seconds=10, cwd=None,
                 popen_factory=None, identity_reader=None, clock=None, emit=None):
        if (not isinstance(command, (tuple, list)) or not command
                or any(not isinstance(part, str) or not part or '\0' in part for part in command)):
            raise ValueError('An explicit nonempty argument vector is required; shell commands are unsupported')
        if (isinstance(grace_seconds, bool) or not isinstance(grace_seconds, (int, float))
                or not math.isfinite(grace_seconds) or grace_seconds <= 0):
            raise ValueError('A positive bounded cooperative-stop grace period is required')
        self.command = list(command)
        self.checkpoint = Path(checkpoint)
        self.grace_seconds = float(grace_seconds)
        self.clock = time.monotonic if clock is None else clock
        self.identity_reader = process_identity if identity_reader is None else identity_reader
        self.emit = _print if emit is None else emit
        factory = subprocess.Popen if popen_factory is None else popen_factory
        # A fresh child session prevents terminal Ctrl+C from also reaching
        # it automatically. Only its exact PID is signaled below, never a group.
        self.child = factory(self.command, cwd=cwd, shell=False, start_new_session=True)
        self.owner = self.identity_reader(self.child.pid)
        if self.owner is None:
            # A very short child can finish before identity inspection. Its
            # Popen return value is still authoritative; no signal is needed.
            if self.child.poll() is None:
                raise RuntimeError('Spawned child identity unavailable; no process was signaled')
        elif (self.owner.get('pid') != self.child.pid
                or self.owner.get('parent_pid') != os.getpid()
                or self.owner.get('argv') != self.command
                or self.owner.get('uid') != os.getuid()):
            raise RuntimeError('Spawned child identity differs; no process was signaled')
        self.pause_started = None
        self.forwarded_sigint = False
        self.forced_stop = False
        self.emit(dict(format=FORMAT, stage='owned_child_started', command=self.command,
                       child_pid=self.child.pid, checkpoint=str(self.checkpoint),
                       grace_seconds=self.grace_seconds, identity=self.owner))

    def _verified_alive(self):
        if self.child.poll() is not None:
            return False
        actual = self.identity_reader(self.child.pid)
        if actual is None:
            return False
        if self.owner is None or actual != self.owner:
            raise RuntimeError(f'Owned child identity changed for PID={self.child.pid}; no signal was sent')
        return True

    def request_pause(self):
        # Repeated Ctrl+C neither floods the terminal nor restarts the timer.
        if self.pause_started is not None:
            return
        self.pause_started = self.clock()
        if self._verified_alive():
            self.emit(dict(format=FORMAT, stage='cooperative_pause_requested',
                child_pid=self.child.pid, signal='SIGINT', command=self.command,
                grace_seconds=self.grace_seconds, checkpoint=str(self.checkpoint),
                checkpoint_scope='last atomic publication; the active batch is not yet guaranteed saved'))
            self.child.send_signal(signal.SIGINT)
            self.forwarded_sigint = True

    def check(self):
        result = self.child.poll()
        if result is not None:
            return result
        if (self.pause_started is not None and not self.forced_stop
                and self.clock() - self.pause_started >= self.grace_seconds):
            if self._verified_alive():
                self.emit(dict(format=FORMAT, stage='owned_child_termination',
                    child_pid=self.child.pid, signal='SIGTERM', command=self.command,
                    reason='Requested pause exceeded grace while waiting for this owned arm',
                    checkpoint=str(self.checkpoint), checkpoint_exists=self.checkpoint.is_file(),
                    checkpoint_scope='last atomic publication; uncommitted work is replayed on resume',
                    parent_shell_or_SSH_signaled=False))
                # Popen.terminate targets exactly this verified child PID.
                self.child.terminate()
                self.forced_stop = True
        return self.child.poll()

    def report(self, returncode):
        exists = self.checkpoint.is_file() and not self.checkpoint.is_symlink()
        status = ('PAUSED_LAST_SAVED' if exists else 'PAUSED_NO_OWN_CHECKPOINT') if self.pause_started is not None else 'CHILD_RETURNED'
        if self.pause_started is not None and returncode not in (0, -signal.SIGINT, -signal.SIGTERM):
            status = 'FAILED_DURING_PAUSE'
        return dict(format=FORMAT, status=status,
            child_returncode=returncode, child_pid=self.child.pid, command=self.command,
            paused=self.pause_started is not None, forwarded_sigint=self.forwarded_sigint,
            forced_stop=self.forced_stop, checkpoint=str(self.checkpoint),
            checkpoint_exists=exists,
            checkpoint_scope='last atomic publication; no claim that the interrupted batch was newly saved',
            full_training_claimed=False, parent_shell_or_SSH_signaled=False)


def run_owned(command, *, checkpoint, grace_seconds=10, cwd=None):
    """Run one Linux arm, with SIGINT followed by owned-PID SIGTERM if needed.

SIGKILL and process-group/session signals are deliberately absent. This waits
for the terminated child to be reaped. A kernel-uninterruptible process can
remain pending; its identity and last checkpoint are printed for diagnosis.
"""
    if not sys.platform.startswith('linux'):
        raise RuntimeError('Owned arm signal supervision is for Linux servers; no platform signal fallback')
    previous = signal.getsignal(signal.SIGINT)
    arm = None
    pending = {'requested': False}
    def request(signum, frame):
        # Keep the Python signal callback free of process inspection and I/O.
        # It can run between instructions in psutil or stdout internals.
        pending['requested'] = True
    signal.signal(signal.SIGINT, request)
    try:
        arm = OwnedArm(command, checkpoint=checkpoint, grace_seconds=grace_seconds, cwd=cwd)
        while True:
            if pending['requested']:
                pending['requested'] = False
                arm.request_pause()
            result = arm.check()
            if result is not None:
                report = arm.report(result)
                arm.emit(report)
                return report
            time.sleep(.1)
    finally:
        signal.signal(signal.SIGINT, previous)
