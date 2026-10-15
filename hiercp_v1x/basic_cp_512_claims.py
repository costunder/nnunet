"""Atomic prospective allocations for the explicitly approved TOTAL512GiB policy.

This module performs no process signalling and never waits while holding its
file lock. Observers are the pinned one-shot resource guards; a reservation is
published before Popen and remains until all exactly captured actors are gone.
"""
from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import time
import uuid

FORMAT = 'basic_cp_RAM512_allocation_ledger_v1'
FILE = 'hiercp_v1x/basic_cp_512_claims.py'
GIB = 2**30
UID = 41833
FLAGS = dict(memory_budget_revision=512, aggregate_memory_cap_GiB=512,
    effective_native_RSS_limit_GiB=32, effective_shared_bank_RSS_limit_GiB=320,
    owned_tmpfs_envelope_GiB=144, post_source_owned_tmpfs_envelope_GiB=272,
    auxiliary_RAM_envelope_GiB=16)


class ResourceClaimUnavailable(MemoryError):
    """A prospective allocation has to queue without reducing the experiment."""
    def __init__(self, reason, observation):
        self.observation = copy.deepcopy(observation)
        super().__init__(reason)


def identity(value):
    if not isinstance(value, dict):
        raise ValueError('Exact process identity mapping required')
    birth = value.get('create_time', value.get('birth'))
    command = value.get('command')
    if (type(value.get('pid')) is not int or value['pid'] <= 0
            or type(birth) not in (int, float) or not math.isfinite(birth) or birth <= 0
            or value.get('UID') != UID or type(value.get('UID')) is not int
            or not isinstance(command, list) or not command
            or any(not isinstance(part, str) or not part or '\x00' in part for part in command)):
        raise ValueError('Exact positive PID/birth/UID41833/nonempty argv required')
    return dict(pid=value['pid'], create_time=birth, UID=UID, command=list(command))


def _key(value):
    item = identity(value)
    return (item['pid'], item['create_time'], item['UID'])


def _checksum(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def _absolute(value):
    return isinstance(value, str) and Path(value).is_absolute() and '..' not in Path(value).parts


def _regular(path):
    path = Path(path)
    if path.parent.resolve(strict=True) != path.parent or any(p.is_symlink() for p in path.parents):
        raise ValueError('Allocation file parents must remain exact nonsymlink paths')
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or path.is_symlink():
        raise ValueError('Regular nonsymlink owned allocation file required')
    if hasattr(os, 'getuid') and info.st_uid != UID:
        raise ValueError('Allocation file must belong to exact UID41833')
    return info


def _receipt(reference):
    if (not isinstance(reference, dict) or set(reference) != {'path', 'sha256'}
            or not _absolute(reference['path']) or not _checksum(reference['sha256'])):
        raise ValueError('An immutable receipt absolute path and exact SHA are required')
    path = Path(reference['path']); _regular(path)
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != reference['sha256']:
        raise RuntimeError('Allocation phase receipt SHA changed')
    result = json.loads(raw)
    if not isinstance(result, dict):
        raise ValueError('Allocation phase receipt must be a mapping')
    return result


def validate_native(native):
    for name, expected in FLAGS.items():
        if type(native.get(name)) is not int or native[name] != expected:
            raise ValueError('Explicit approved TOTAL512 flag required: ' + name)
    if (type(native.get('gpu')) is not int or native['gpu'] not in range(7)
            or not _absolute(native.get('RAM512_claim_ledger'))
            or not _absolute(native.get('owned_tmpfs_root'))
            or not _checksum(native.get('code_sha256', {}).get(FILE))):
        raise ValueError('Pinned RAM512 claims, absolute common ledger/root, and actual GPU required')
    identity(native['controller_identity'])


def reservation_bytes(snapshot):
    """Conservative envelopes include future allocations and owned tmpfs."""
    natives = {c['gpu'] for c in snapshot['claims'].values() if c['kind'] == 'native'}
    pending = snapshot['source_pending']
    if pending and (len(natives) > 1 or (natives and 6 not in natives and not snapshot['priority_complete'])):
        raise ResourceClaimUnavailable('Source320 requires sole prioritized native32 slot', snapshot)
    store = (144 if pending else 272) * GIB
    source = 320 * GIB if pending else 0
    return dict(owned_tmpfs_reserved_bytes=store, source_reserved_bytes=source,
        native_reserved_bytes=32 * GIB * len(natives), auxiliary_reserved_bytes=16 * GIB,
        required_capacity_bytes=store + source + 32 * GIB * len(natives) + 16 * GIB,
        aggregate_cap_bytes=512 * GIB, reserved_native_GPUs=sorted(natives),
        source_pending=pending, scientific_scale_changed=False)


class Ledger:
    def __init__(self, native, binding, resolver):
        validate_native(native)
        required = {'registry_path', 'registry_sha256', 'request_sha256',
            'host_lifetime_sha256', 'expires_at_unix', 'owned_tmpfs_root', 'source_initially_pending'}
        if (not isinstance(binding, dict) or set(binding) != required
                or not _absolute(binding['registry_path']) or not _checksum(binding['registry_sha256'])
                or not _checksum(binding['host_lifetime_sha256'])
                or not isinstance(binding['request_sha256'], dict)
                or set(binding['request_sha256']) != {str(i) for i in range(7)}
                or any(not _checksum(v) for v in binding['request_sha256'].values())
                or type(binding['expires_at_unix']) not in (int, float)
                or not math.isfinite(binding['expires_at_unix'])
                or type(binding['source_initially_pending']) is not bool
                or binding['owned_tmpfs_root'] != native['owned_tmpfs_root']):
            raise ValueError('All seven immutable registry/request/lifetime/root bindings required')
        self.native, self.binding, self.resolver = copy.deepcopy(native), copy.deepcopy(binding), resolver
        self.path = Path(native['RAM512_claim_ledger'])
        if (not self.path.is_relative_to(Path(native['owned_tmpfs_root']))
                or self.path.parent.resolve(strict=True) != self.path.parent
                or any(parent.is_symlink() for parent in self.path.parents)):
            raise ValueError('Common allocation ledger must stay in the exact owned tmpfs namespace')
        self.lock_path = self.path.with_suffix(self.path.suffix + '.lock')

    def _deadline(self):
        if self.binding['expires_at_unix'] - time.time() < 180:
            raise RuntimeError('Actual owned host lifetime has less than180 seconds')

    @contextlib.contextmanager
    def _locked(self):
        self._deadline()
        if self.lock_path.is_symlink():
            raise ValueError('Allocation lock cannot be a symlink')
        fd = os.open(self.lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            _regular(self.lock_path)
            if os.fstat(fd).st_size == 0:
                os.write(fd, b'0'); os.fsync(fd)
            if os.name == 'nt':
                import msvcrt
                os.lseek(fd, 0, os.SEEK_SET); msvcrt.locking(fd, msvcrt.LK_LOCK, 1)
            else:
                import fcntl
                fcntl.flock(fd, fcntl.LOCK_EX)
            try:
                self._deadline()
                yield
            finally:
                if os.name == 'nt':
                    os.lseek(fd, 0, os.SEEK_SET); msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)

    def _read(self):
        _regular(self.path)
        return self._validate_state(json.loads(self.path.read_bytes()))

    def _validate_state(self, state):
        if not isinstance(state, dict):
            raise RuntimeError('Common RAM512 ledger must remain a mapping')
        if (state.get('format') != FORMAT or state.get('binding') != self.binding
                or not isinstance(state.get('claims'), dict)
                or not isinstance(state.get('storage_reservations'), dict)
                or any(type(state.get(k)) is not bool for k in ('source_pending', 'source_released', 'priority_complete'))
                or state['source_released'] == state['source_pending']
                or type(state.get('revision')) is not int or state['revision'] < 0):
            raise RuntimeError('Common immutable RAM512 ledger binding or phase changed')
        for token, claim in state['claims'].items():
            if (claim.get('token') != token or claim.get('kind') not in ('native', 'cpu_source')
                    or type(claim.get('gpu')) is not int or claim['gpu'] not in range(7)
                    or claim.get('request_sha256') != self.binding['request_sha256'][str(claim['gpu'])]
                    or not isinstance(claim.get('stage'), str) or not claim['stage']
                    or not isinstance(claim.get('captured'), list)):
                raise RuntimeError('Prospective actor claim provenance changed')
            identity(claim['controller_identity'])
            identity(dict(claim['controller_identity'], command=claim['command']))
            if claim['child_identity'] is not None: identity(claim['child_identity'])
            for actor in claim['captured']: identity(actor)
        for token, row in state['storage_reservations'].items():
            if (row.get('token') != token or type(row.get('bytes')) is not int or row['bytes'] <= 0
                    or type(row.get('gpu')) is not int or row['gpu'] not in range(7)
                    or row.get('request_sha256') != self.binding['request_sha256'][str(row['gpu'])]):
                raise RuntimeError('Projected archive storage allocation provenance changed')
            identity(row['controller_identity'])
            if row.get('destination') is not None and (
                    not _absolute(row['destination'])
                    or not Path(row['destination']).is_relative_to(Path(self.native['owned_tmpfs_root']))
                    or not isinstance(row.get('destination_parent_identity'),list)
                    or len(row['destination_parent_identity'])!=3):
                raise RuntimeError('Projected archive destination left the owned tmpfs namespace')
        reservation_bytes(state)
        if 'GPU56_sharing_receipt' in self.native:
            from .basic_cp_GPU56_sharing import verify_claim_capacity
            verify_claim_capacity(state, self.native)
        return state

    def _write(self, state):
        raw = (json.dumps(state, sort_keys=True, indent=2) + '\n').encode()
        temporary = self.path.with_name(self.path.name + '.OWN_' + uuid.uuid4().hex)
        try:
            with temporary.open('xb') as stream:
                stream.write(raw); stream.flush(); os.fsync(stream.fileno())
            os.replace(temporary, self.path)
        finally:
            if temporary.exists(): temporary.unlink()  # Only this private unpublished temp.

    def bootstrap(self, *, source_pending_receipt):
        witness = _receipt(source_pending_receipt)
        if (witness.get('source_initially_pending') is not self.binding['source_initially_pending']
                or not isinstance(witness.get('candidate_counts'), dict)
                or set(witness['candidate_counts']) != {'512', '1024', '2048'}
                or any(type(v) is not int or not 0 <= v <= 642 for v in witness['candidate_counts'].values())):
            raise ValueError('Actual sealed source-prefix counts and pending bootstrap receipt required')
        pending = self.binding['source_initially_pending']
        if pending != any(v != 642 for v in witness['candidate_counts'].values()):
            raise ValueError('Pending-source reservation must match actual full642 completeness')
        with self._locked():
            if self.path.exists():
                return copy.deepcopy(self._read())
            state = dict(format=FORMAT, binding=self.binding, claims={}, storage_reservations={},
                source_pending=pending, source_released=not pending, priority_complete=False,
                source_bootstrap_proof=copy.deepcopy(source_pending_receipt),
                source_release_proof=None if pending else copy.deepcopy(source_pending_receipt),
                priority_completion_proof=None, source_actor_identity=None, source_captured=[], revision=0)
            self._write(state)
            return copy.deepcopy(state)

    def snapshot(self):
        with self._locked(): return copy.deepcopy(self._read())

    def snapshot_atomic_readonly(self):
        """Read one complete published ledger without taking the writer mutex.

        Writers exclusively publish through fsync + atomic replace. A native
        child which is already bound may inspect that immutable version while
        the parent measures resources under the mutation lock. No reservation,
        binding, refresh or release is permitted through this read-only path.
        """
        self._deadline()
        fields = ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns',
                  'st_uid', 'st_gid', 'st_mode')
        signature = lambda row: tuple(getattr(row, name) for name in fields)
        for _ in range(5):
            before = _regular(self.path)
            parents = {parent: parent.lstat() for parent in self.path.parents}
            flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_CLOEXEC', 0)
            fd = os.open(self.path, flags)
            try:
                opened = os.fstat(fd)
                if (not stat.S_ISREG(opened.st_mode)
                        or hasattr(os, 'getuid') and opened.st_uid != UID):
                    raise ValueError('Atomic allocation FD must be an exact owned regular file')
                if signature(opened) != signature(before):
                    continue  # A genuine atomic publication crossed open().
                with os.fdopen(fd, 'rb', closefd=False) as stream:
                    raw = stream.read()
                after = os.fstat(fd)
                if signature(after) != signature(opened):
                    # Atomic replacement may unlink the held old inode. An
                    # in-place write to the still-published inode is forbidden.
                    current = _regular(self.path)
                    if (current.st_dev, current.st_ino) == (opened.st_dev, opened.st_ino):
                        raise RuntimeError('Published allocation inode changed during read')
                    continue
            finally:
                os.close(fd)
            current = _regular(self.path)
            for parent, observed in parents.items():
                value = parent.lstat()
                if (stat.S_ISLNK(value.st_mode) or not stat.S_ISDIR(value.st_mode)
                        or (value.st_dev, value.st_ino, value.st_uid, value.st_gid, value.st_mode)
                        != (observed.st_dev, observed.st_ino, observed.st_uid,
                            observed.st_gid, observed.st_mode)):
                    raise RuntimeError('Atomic allocation ancestor changed during read')
            if signature(current) != signature(after):
                continue
            state = self._validate_state(json.loads(raw))
            revision = state['revision']; digest = hashlib.sha256(raw).hexdigest()
            previous = getattr(self, '_atomic_read_version', None)
            if previous is not None and (revision < previous[0]
                    or revision == previous[0] and digest != previous[1]):
                raise RuntimeError('Published allocation revision regressed or changed without a revision')
            self._deadline()
            self._atomic_read_version = (revision, digest)
            return state
        raise RuntimeError('Atomic allocation publication did not stabilize during bounded read')

    def _actual(self, expected):
        expected = identity(expected)
        observation = self.resolver.inspect(expected)
        if not isinstance(observation, dict) or observation.get('state') not in ('live', 'gone', 'terminal'):
            raise RuntimeError('Resolver must provide actual live/gone/terminal evidence')
        if observation['state'] != 'gone' and identity(observation.get('identity')) != expected:
            raise RuntimeError('Captured PID/birth/UID/argv changed')
        return observation['state']

    def _capture(self, claim):
        known = {_key(actor): actor for actor in claim['captured']}
        roots = [claim['child_identity']] if claim['child_identity'] else []
        if not roots:
            # A controller can die after Popen but before its bind. Exact argv
            # scanning retains the orphan instead of freeing its allocation.
            roots = list(self.resolver.scan(claim['command']))
        for root in roots:
            root = identity(root)
            if root['command'] != claim['command']:
                raise RuntimeError('Unbound prospective child argv mismatch')
            known[_key(root)] = root
            if self._actual(root) == 'live':
                for actor in self.resolver.descendants(root):
                    actor = identity(actor); known[_key(actor)] = actor
        claim['captured'] = list(known.values())
        return claim

    def _gone(self, claim):
        self._capture(claim)
        if claim['child_identity'] is None and not claim['captured']:
            return self._actual(claim['controller_identity']) != 'live'
        if any(self._actual(actor) == 'live' for actor in claim['captured']): return False
        if self.resolver.has_cuda(claim['captured']): return False
        return True

    def _refresh(self, state):
        for token, claim in list(state['claims'].items()):
            self._capture(claim)
            if claim['kind'] == 'cpu_source':
                known = {_key(actor): actor for actor in state['source_captured']}
                known.update({_key(actor): actor for actor in claim['captured']})
                state['source_captured'] = list(known.values())
            if self._gone(claim): del state['claims'][token]
        for token, row in list(state['storage_reservations'].items()):
            # Independent archive copies run synchronously inside this exact
            # controller; no background writer is introduced by this policy.
            if self._actual(row['controller_identity']) != 'live':
                del state['storage_reservations'][token]

    def reserve_storage(self, byte_count, observer, *, destination=None):
        return StorageReservation(self, byte_count, observer, destination=destination)

    def _phases(self, state, observation):
        if not isinstance(observation, dict): raise ValueError('Actual raw resource observation required')
        if observation.get('source_complete') is True and state['source_pending']:
            reference = observation.get('source_completion_proof'); receipt = _receipt(reference)
            if (receipt.get('candidate_counts') != {'512': 642, '1024': 642, '2048': 642}
                    or receipt.get('producer_fully_released') is not True
                    or not isinstance(receipt.get('captured'), list)
                    or state.get('source_actor_identity') is None
                    or _key(state['source_actor_identity']) not in {_key(a) for a in state['source_captured']}
                    or {_key(a) for a in receipt['captured']} != {_key(a) for a in state['source_captured']}
                    or any(c['kind'] == 'cpu_source' for c in state['claims'].values())
                    or any(self._actual(a) == 'live' for a in state['source_captured'])
                    or self.resolver.has_cuda(state['source_captured'])):
                raise RuntimeError('Full642 source proof and all captured producer/worker/tracker release required')
            state.update(source_pending=False, source_released=True, source_release_proof=copy.deepcopy(reference))
        if observation.get('priority_complete') is True and not state['priority_complete']:
            reference = observation.get('priority_completion_proof'); receipt = _receipt(reference)
            if (receipt.get('completed_epochs') != 250 or receipt.get('evaluated_cases') != 26
                    or receipt.get('checkpoint') != 'BEST' or receipt.get('historical_metrics_complete') is not True):
                raise RuntimeError('GPU6 priority release requires actual250/BEST26/full metrics proof')
            state.update(priority_complete=True, priority_completion_proof=copy.deepcopy(reference))

    def assert_current_child(self, stage, command, self_identity, parent_identity):
        child, parent = identity(self_identity), identity(parent_identity)
        if child['command'] != command: raise RuntimeError('Current native/source argv changed')
        with self._locked():
            state = self._read()
            matches = [c for c in state['claims'].values() if c['gpu'] == self.native['gpu']
                and c['stage'] == stage and c['command'] == command and c['controller_identity'] == parent]
            if len(matches) != 1: raise RuntimeError('Exact prospective parent/child allocation claim required')
            claim = matches[0]
            if claim['child_identity'] is not None and claim['child_identity'] != child:
                raise RuntimeError('Allocation child identity already differs')
            if self._actual(child) != 'live' or self._actual(parent) != 'live':
                raise RuntimeError('Allocation self-bind requires actual exact live child and parent')
            claim['child_identity'] = child; self._capture(claim)
            if claim['kind'] == 'cpu_source':
                state['source_actor_identity'] = copy.deepcopy(child)
                known = {_key(a): a for a in state['source_captured']}
                known.update({_key(a): a for a in claim['captured']})
                state['source_captured'] = list(known.values())
            state['revision'] += 1; self._write(state)
            return copy.deepcopy(claim)


class AllocationReservation:
    def __init__(self, ledger, native, stage, command, observer):
        validate_native(native)
        if native['RAM512_claim_ledger'] != str(ledger.path) or native['gpu'] != ledger.native['gpu']:
            raise ValueError('Reservation must use its exact admitted common ledger/native')
        if not isinstance(stage, str) or not stage:
            raise ValueError('Actual actor stage required')
        self.ledger, self.native, self.observer = ledger, native, observer
        self.claim = dict(token=uuid.uuid4().hex, kind='cpu_source' if stage == 'candidate_bank' else 'native',
            gpu=native['gpu'], stage=stage, controller_identity=identity(native['controller_identity']),
            request_sha256=ledger.binding['request_sha256'][str(native['gpu'])], command=list(command),
            child_identity=None, captured=[], claimed_at_unix=time.time())
        if 'GPU56_sharing_receipt' in native:
            from .basic_cp_GPU56_sharing import claim_fields
            self.claim.update(claim_fields(native, stage))
        identity(dict(self.claim['controller_identity'], command=command))
        if self.claim['kind'] == 'cpu_source' and native['gpu'] != 5:
            raise ValueError('Only exact GPU5 producer may reserve shared CPU source320')
        self.entered = False

    def __enter__(self):
        with self.ledger._locked():
            state = self.ledger._read(); self.ledger._refresh(state)
            # Reconcile a naturally ended producer before testing a candidate
            # native claim. Otherwise its stale pending flag could reject the
            # very observation needed to publish the verified released phase.
            phase_observation = self.observer(copy.deepcopy(state), None)
            self.ledger._phases(state, phase_observation)
            observation = self.observer(copy.deepcopy(state), copy.deepcopy(self.claim))
            self.ledger._phases(state, observation)
            if any(c['gpu'] == self.claim['gpu'] for c in state['claims'].values()):
                raise ResourceClaimUnavailable('Exact GPU/stage slot remains allocated until full tree release', state)
            if self.claim['kind'] == 'native' and state['source_pending'] and self.claim['gpu'] in (3, 4, 5):
                raise ResourceClaimUnavailable('Candidate native waits for complete CPU producer release', state)
            if self.claim['kind'] == 'cpu_source' and not state['source_pending']:
                raise RuntimeError('Complete released source cannot be allocated again')
            state['claims'][self.claim['token']] = copy.deepcopy(self.claim)
            if 'GPU56_sharing_receipt' in self.native:
                from .basic_cp_GPU56_sharing import verify_claim_capacity
                verify_claim_capacity(state, self.native)
            plan = reservation_bytes(state)
            if plan['required_capacity_bytes'] > plan['aggregate_cap_bytes']:
                raise ResourceClaimUnavailable('Actual prospective TOTAL512 envelope exceeded', plan)
            # Refresh guard after phase transition and including this claim.
            self.observer(copy.deepcopy(state), None)
            state['revision'] += 1; self.ledger._write(state)
            self.entered = True
        return self

    def bind_child(self, child):
        if not self.entered: raise RuntimeError('Popen must follow published allocation reservation')
        return self.ledger.assert_current_child(self.claim['stage'], self.claim['command'], child,
            self.claim['controller_identity'])

    def refresh(self):
        with self.ledger._locked():
            state = self.ledger._read(); self.ledger._refresh(state)
            observation = self.observer(copy.deepcopy(state), None)
            self.ledger._phases(state, observation)
            state['revision'] += 1; self.ledger._write(state)
            return copy.deepcopy(state)

    def release(self):
        if not self.entered: return
        with self.ledger._locked():
            state = self.ledger._read(); claim = state['claims'].get(self.claim['token'])
            if claim is not None:
                self.ledger._capture(claim)
                if claim['kind'] == 'cpu_source':
                    known = {_key(a): a for a in state['source_captured']}
                    known.update({_key(a): a for a in claim['captured']})
                    state['source_captured'] = list(known.values())
                # No child after an unsuccessful Popen is a voluntary release;
                # an orphan with the exact argv is retained by _capture.
                unstarted = claim['child_identity'] is None and not claim['captured']
                if unstarted or self.ledger._gone(claim): del state['claims'][self.claim['token']]
            observation = self.observer(copy.deepcopy(state), None)
            self.ledger._phases(state, observation)
            state['revision'] += 1; self.ledger._write(state)
            self.entered = False

    def __exit__(self, exc_type, exc, tb):
        self.release()
        return False


class StorageReservation:
    """Publish a complete independent archive's prospective bytes before IO."""
    def __init__(self, ledger, byte_count, observer, *, destination=None):
        if type(byte_count) is not int or byte_count <= 0:
            raise ValueError('Exact positive full archive byte count required')
        self.ledger, self.observer = ledger, observer
        self.row = dict(token=uuid.uuid4().hex, gpu=ledger.native['gpu'], bytes=byte_count,
            controller_identity=identity(ledger.native['controller_identity']),
            request_sha256=ledger.binding['request_sha256'][str(ledger.native['gpu'])],
            purpose='independent_completed_archive', claimed_at_unix=time.time())
        if destination is not None:
            destination=Path(destination)
            if (not destination.is_absolute() or '..'in destination.parts
                    or not destination.is_relative_to(Path(ledger.native['owned_tmpfs_root']))
                    or destination.parent.resolve(strict=True)!=destination.parent
                    or any(p.is_symlink()for p in destination.parents)):
                raise ValueError('Fresh archive destination must stay inside exact owned tmpfs')
            self.row['destination']=str(destination)
            metadata=destination.parent.stat()
            if hasattr(os,'getuid')and metadata.st_uid!=UID:
                raise ValueError('Archive parent must retain exact UID41833')
            self.row['destination_parent_identity']=[metadata.st_dev,metadata.st_ino,metadata.st_uid]
        self.entered = False

    def __enter__(self):
        with self.ledger._locked():
            state=self.ledger._read(); self.ledger._refresh(state)
            if self.ledger._actual(self.row['controller_identity']) != 'live':
                raise RuntimeError('Only exact live own controller may create archive bytes')
            if any(r['gpu']==self.row['gpu'] for r in state['storage_reservations'].values()):
                raise ResourceClaimUnavailable('Same controller archive copy remains allocated',state)
            destination=self.row.get('destination')
            if destination is not None:
                parent=Path(destination).parent;metadata=parent.stat()
                if ([metadata.st_dev,metadata.st_ino,metadata.st_uid]!=self.row['destination_parent_identity']
                        or parent.resolve(strict=True)!=parent or any(p.is_symlink()for p in parent.parents)):
                    raise RuntimeError('Archive destination parent identity changed before allocation')
                if Path(destination).exists():
                    raise RuntimeError('Existing archive destination cannot be overwritten or reused')
                if any(r.get('destination')==destination for r in state['storage_reservations'].values()):
                    raise ResourceClaimUnavailable('Exact independent archive destination remains reserved',state)
            self.ledger._phases(state,self.observer(copy.deepcopy(state),None))
            state['storage_reservations'][self.row['token']]=copy.deepcopy(self.row)
            observation=self.observer(copy.deepcopy(state),None)
            self.ledger._phases(state,observation)
            state['revision']+=1; self.ledger._write(state)
            self.entered=True
        return self

    def __exit__(self, exc_type, exc, tb):
        if not self.entered:return False
        with self.ledger._locked():
            state=self.ledger._read()
            state['storage_reservations'].pop(self.row['token'],None)
            # The synchronous copy has finished (or raised). All actual blocks
            # remain charged; only its separate prospective projection ends.
            state['revision']+=1; self.ledger._write(state)
            self.entered=False
            observation=self.observer(copy.deepcopy(state),None)
            self.ledger._phases(state,observation)
            state['revision']+=1; self.ledger._write(state)
        return False
