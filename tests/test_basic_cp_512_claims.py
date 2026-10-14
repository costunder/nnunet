"""CPU DEBUG coordination tests; tiny receipts are not research results."""
import concurrent.futures
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest import mock

from hiercp_v1x import basic_cp_512_claims as claims


def actor(pid, command):
    return dict(pid=pid, create_time=1000.0+pid, UID=41833, command=command)


class Resolver:
    def __init__(self):
        self.live = {}; self.children = {}; self.cuda = set()

    def inspect(self, expected):
        actual = self.live.get(expected['pid'])
        if actual is None: return dict(state='gone')
        return dict(state='live', identity=copy.deepcopy(actual))

    def descendants(self, expected):
        return copy.deepcopy(self.children.get(expected['pid'], []))

    def scan(self, command):
        return [copy.deepcopy(a) for a in self.live.values() if a['command'] == command]

    def has_cuda(self, identities):
        return any(a['pid'] in self.cuda for a in identities)


class AllocationTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory(prefix='DEBUG_RAM512_claims_',
            dir=Path(__file__).resolve().parents[1]/'tests')
        self.root = Path(self.folder.name)
        # Only file ownership is a DEBUG adapter on non-Windows test hosts;
        # live production identity validation remains UID41833.
        real_regular = claims._regular
        if hasattr(claims.os, 'getuid'):
            def regular(path):
                with mock.patch.object(claims.os, 'getuid', create=True, return_value=41833):
                    info = Path(path).lstat()
                    with mock.patch.object(claims.stat, 'S_ISREG', wraps=claims.stat.S_ISREG):
                        if info.st_uid != 41833:
                            if not claims.stat.S_ISREG(info.st_mode) or Path(path).is_symlink():
                                raise ValueError('DEBUG regular file required')
                            return info
                        return real_regular(path)
            self.regular = mock.patch.object(claims, '_regular', side_effect=regular)
            self.regular.start(); self.addCleanup(self.regular.stop)
        self.addCleanup(self.folder.cleanup)
        self.resolver = Resolver()
        self.binding = dict(registry_path=str(self.root/'registry.json'), registry_sha256='1'*64,
            request_sha256={str(i):str(i+1)*64 for i in range(7)}, host_lifetime_sha256='a'*64,
            expires_at_unix=time.time()+3600, owned_tmpfs_root=str(self.root), source_initially_pending=True)
        self.native = {}
        for gpu in range(7):
            control = actor(100+gpu, ['DEBUG_controller', str(gpu)])
            self.resolver.live[control['pid']] = control
            self.native[gpu] = dict(claims.FLAGS, gpu=gpu,
                RAM512_claim_ledger=str(self.root/'allocations.json'), owned_tmpfs_root=str(self.root),
                controller_identity=control, code_sha256={claims.FILE:'b'*64})
        self.ledger = {gpu:claims.Ledger(self.native[gpu], self.binding, self.resolver) for gpu in range(7)}
        self.bootstrap = self.receipt('DEBUG_pending.json', dict(source_initially_pending=True,
            candidate_counts={'512':642,'1024':642,'2048':89}))
        self.ledger[6].bootstrap(source_pending_receipt=self.bootstrap)
        self.observation = {}

    def receipt(self, name, payload):
        path = self.root/name; path.write_text(json.dumps(payload), encoding='utf8')
        return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())

    def observer(self, snapshot, prospective):
        return copy.deepcopy(self.observation)

    def reservation(self, gpu, stage='train'):
        return claims.AllocationReservation(self.ledger[gpu], self.native[gpu], stage,
            ['DEBUG_full_science_actor',str(gpu),stage], self.observer)

    def bind(self, reservation, pid):
        child = actor(pid,reservation.claim['command']); self.resolver.live[pid] = child
        reservation.bind_child(child)
        return child

    def test_pending320_published_before_any_native_and_gpu6_priority(self):
        state = self.ledger[0].snapshot()
        self.assertTrue(state['source_pending'])
        self.assertEqual(claims.reservation_bytes(state)['required_capacity_bytes'],480*claims.GIB)
        for gpu in range(6):
            with self.subTest(gpu=gpu), self.assertRaises(claims.ResourceClaimUnavailable):
                with self.reservation(gpu): pass
        with self.reservation(6):
            self.assertEqual(claims.reservation_bytes(self.ledger[6].snapshot())['required_capacity_bytes'],512*claims.GIB)

    def test_source_and_gpu6_reservations_fit_without_cuda_creation(self):
        with self.reservation(5,'candidate_bank'), self.reservation(6):
            state = self.ledger[6].snapshot()
            self.assertEqual({c['kind'] for c in state['claims'].values()}, {'native','cpu_source'})
            self.assertEqual(claims.reservation_bytes(state)['required_capacity_bytes'],512*claims.GIB)

    def test_simultaneous_controllers_cannot_claim_duplicate_gpu6_slot(self):
        barrier = threading.Barrier(2); entered = threading.Event(); leave = threading.Event()
        def attempt():
            barrier.wait()
            try:
                with self.reservation(6):
                    entered.set(); leave.wait(3)
                    return 'admitted'
            except claims.ResourceClaimUnavailable:
                leave.set(); return 'queued'
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda _:attempt(),range(2)))
        self.assertCountEqual(results,['admitted','queued'])
        self.assertEqual(self.ledger[6].snapshot()['claims'],{})

    def test_exact_child_selfbind_before_parent_bind_is_idempotent(self):
        with self.reservation(6) as reservation:
            child = actor(300,reservation.claim['command']); self.resolver.live[300] = child
            self.ledger[6].assert_current_child('train',child['command'],child,self.native[6]['controller_identity'])
            reservation.bind_child(child)
            self.assertEqual(self.ledger[6].snapshot()['claims'][reservation.claim['token']]['child_identity'],child)
            self.resolver.live.pop(300)

    def test_child_with_wrong_parent_or_argv_cannot_selfbind(self):
        with self.reservation(6) as reservation:
            child=actor(301,reservation.claim['command']); self.resolver.live[301]=child
            for parent,command in ((self.native[0]['controller_identity'],child['command']),
                    (self.native[6]['controller_identity'],['wrong'])):
                with self.assertRaises(RuntimeError):self.ledger[6].assert_current_child('train',command,child,parent)
            self.resolver.live.pop(301)

    def test_live_da_descendant_retains_slot_after_native_exit(self):
        reservation=self.reservation(6); reservation.__enter__(); child=self.bind(reservation,302)
        da=actor(303,['DEBUG_DA']); self.resolver.live[303]=da; self.resolver.children[302]=[da]
        reservation.refresh(); self.resolver.live.pop(302); reservation.release()
        self.assertIn(reservation.claim['token'],self.ledger[6].snapshot()['claims'])
        with self.assertRaises(claims.ResourceClaimUnavailable):self.reservation(6).__enter__()
        self.resolver.live.pop(303)
        with self.reservation(6):pass

    def test_cuda_residency_retains_claim_even_after_process_is_gone(self):
        reservation=self.reservation(6); reservation.__enter__(); self.bind(reservation,304)
        self.resolver.live.pop(304); self.resolver.cuda.add(304); reservation.release()
        self.assertIn(reservation.claim['token'],self.ledger[6].snapshot()['claims'])
        self.resolver.cuda.clear()
        with self.reservation(6):pass

    def test_orphan_popen_before_bind_remains_reserved(self):
        reservation=self.reservation(6); reservation.__enter__()
        orphan=actor(305,reservation.claim['command']); self.resolver.live[305]=orphan
        self.resolver.live.pop(106); reservation.release()
        self.assertIn(reservation.claim['token'],self.ledger[6].snapshot()['claims'])

    def test_pid_reuse_or_changed_live_argv_is_fatal_not_silent_release(self):
        reservation=self.reservation(6); reservation.__enter__(); child=self.bind(reservation,306)
        self.resolver.live[306]=dict(child,create_time=99999)
        with self.assertRaisesRegex(RuntimeError,'changed'):reservation.release()
        self.assertIn(reservation.claim['token'],self.ledger[6].snapshot()['claims'])

    def test_source_failure_releases_actor_but_preserves_pending320(self):
        reservation=self.reservation(5,'candidate_bank'); reservation.__enter__(); self.bind(reservation,307)
        self.resolver.live.pop(307); reservation.release()
        state=self.ledger[5].snapshot()
        self.assertTrue(state['source_pending']); self.assertFalse(state['source_released'])
        with self.assertRaises(claims.ResourceClaimUnavailable):self.reservation(3).__enter__()

    def finish_source(self, captured):
        reference=self.receipt('DEBUG_full_source.json',dict(candidate_counts={'512':642,'1024':642,'2048':642},
            producer_fully_released=True,captured=captured))
        self.observation=dict(source_complete=True,source_completion_proof=reference)

    def test_completed_bank_alone_cannot_release_live_cpu_workers(self):
        reservation=self.reservation(5,'candidate_bank'); reservation.__enter__(); child=self.bind(reservation,308)
        worker=actor(309,['DEBUG_source_worker']); self.resolver.live[309]=worker; self.resolver.children[308]=[worker]
        reservation.refresh(); captured=self.ledger[5].snapshot()['source_captured']
        self.finish_source(captured); self.resolver.live.pop(308)
        with self.assertRaisesRegex(RuntimeError,'all captured'):reservation.release()

    def test_full_source_release_switches_store272_and_all7_parallel(self):
        reservation=self.reservation(5,'candidate_bank'); reservation.__enter__(); self.bind(reservation,310)
        reservation.refresh(); captured=self.ledger[5].snapshot()['source_captured']; self.resolver.live.pop(310)
        self.finish_source(captured); reservation.release()
        with context_stack() as stack:
            for gpu in range(7):stack.enter_context(self.reservation(gpu))
            state=self.ledger[6].snapshot(); plan=claims.reservation_bytes(state)
            self.assertTrue(state['source_released']); self.assertEqual(plan['owned_tmpfs_reserved_bytes'],272*claims.GIB)
            self.assertEqual(plan['required_capacity_bytes'],512*claims.GIB)

    def test_first_candidate_claim_reconciles_naturally_dead_source(self):
        reservation=self.reservation(5,'candidate_bank');reservation.__enter__();self.bind(reservation,311)
        reservation.refresh();captured=self.ledger[5].snapshot()['source_captured']
        self.resolver.live.pop(311);self.finish_source(captured)
        # Simulate producer-controller disappearance before context __exit__.
        with self.reservation(3):
            state=self.ledger[3].snapshot()
            self.assertTrue(state['source_released'])
            self.assertEqual({c['gpu'] for c in state['claims'].values()},{3})

    def test_phase_receipt_sha_change_fails_without_claim_publication(self):
        self.finish_source([]); Path(self.observation['source_completion_proof']['path']).write_text('{}')
        with self.assertRaisesRegex(RuntimeError,'SHA changed'):self.reservation(6).__enter__()
        self.assertEqual(self.ledger[6].snapshot()['claims'],{})

    def test_source_empty_capture_cannot_claim_actual_producer_release(self):
        self.finish_source([])
        with self.assertRaisesRegex(RuntimeError,'Full642 source'):self.reservation(6).__enter__()
        self.assertTrue(self.ledger[6].snapshot()['source_pending'])

    def test_projected_archives_accumulate_under_same_mutex(self):
        seen=[]
        def observer(snapshot,prospective):
            projected=sum(r['bytes'] for r in snapshot['storage_reservations'].values())
            seen.append(projected)
            if projected>100:raise claims.ResourceClaimUnavailable('DEBUG exact projected floor',snapshot)
            return {}
        with self.ledger[6].reserve_storage(60,observer):
            with self.assertRaises(claims.ResourceClaimUnavailable):
                with self.ledger[0].reserve_storage(50,observer):pass
            with self.ledger[0].reserve_storage(40,observer):pass
        self.assertIn(110,seen); self.assertIn(100,seen)
        self.assertEqual(self.ledger[6].snapshot()['storage_reservations'],{})

    def test_projected_archive_context_unlocked_during_real_copy(self):
        with self.ledger[6].reserve_storage(20,self.observer):
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                state=pool.submit(self.ledger[0].snapshot).result(timeout=3)
            self.assertEqual(sum(r['bytes'] for r in state['storage_reservations'].values()),20)

    def test_failed_archive_preserves_actual_files_and_releases_projection(self):
        artifact=self.root/'DEBUG_independent_copy'; artifact.write_bytes(b'actual DEBUG content')
        with self.assertRaises(OSError):
            with self.ledger[6].reserve_storage(100,self.observer):raise OSError('DEBUG copy IO failure')
        self.assertEqual(artifact.read_bytes(),b'actual DEBUG content')
        self.assertEqual(self.ledger[6].snapshot()['storage_reservations'],{})

    def test_raw_guard_failure_releases_mutex_and_never_publishes_claim(self):
        def failure(snapshot,prospective):raise OSError('DEBUG storage guard failure')
        reservation=claims.AllocationReservation(self.ledger[6],self.native[6],'train',['DEBUG_actor'],failure)
        with self.assertRaises(OSError):reservation.__enter__()
        with self.reservation(6):pass

    def test_all_seven_binding_mismatch_rejected(self):
        wrong=copy.deepcopy(self.binding); wrong['request_sha256']['0']='f'*64
        other=claims.Ledger(self.native[6],wrong,self.resolver)
        with self.assertRaisesRegex(RuntimeError,'binding'):other.snapshot()

    def test_flags_do_not_allow_boolean_or_unapproved_old800(self):
        for key,value in [('memory_budget_revision',True),('aggregate_memory_cap_GiB',800),
                ('post_source_owned_tmpfs_envelope_GiB',144)]:
            changed=dict(self.native[6],**{key:value})
            with self.subTest(key=key),self.assertRaises(ValueError):claims.validate_native(changed)

    def test_unstarted_popen_failure_releases_only_its_own_claim(self):
        with self.assertRaisesRegex(OSError,'Popen'):
            with self.reservation(6):raise OSError('DEBUG Popen failed')
        self.assertEqual(self.ledger[6].snapshot()['claims'],{})

    def test_source_bootstrap_false_complete_cannot_remove_pending_reservation(self):
        bad=self.receipt('DEBUG_bad_pending.json',dict(source_initially_pending=True,
            candidate_counts={'512':642,'1024':642,'2048':642}))
        with self.assertRaises(ValueError):self.ledger[6].bootstrap(source_pending_receipt=bad)


def context_stack():
    import contextlib
    return contextlib.ExitStack()


if __name__=='__main__':unittest.main()
