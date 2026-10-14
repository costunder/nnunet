"""CPU DEBUG proofs for real512 accounting and the actual inherited guards."""
import copy
import json
import stat
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from hiercp_v1x import basic_cp_512_resources as resource
from hiercp_v1x.basic_cp_512_claims import ResourceClaimUnavailable
from tests import test_basic_cp_shared_bank_resources_v6 as fixture
from tests import test_basic_cp_resources_v5 as native_fixture
from tests import test_basic_cp_seven_resources as actors

GIB = resource.GIB


class AccountingTests(unittest.TestCase):
    def measure(self, **changes):
        values = dict(store_bytes=144*GIB, native_rss={6:32*GIB}, source_rss=320*GIB,
            auxiliary_rss=16*GIB, source_pending=True, host_total=1008*GIB,
            host_available=400*GIB, registered_rss=352*GIB)
        values.update(changes)
        return resource.enforce_accounting(**values)

    def test_full_source_and_full_native_exact512_preserve_effective_caps(self):
        value = self.measure()
        self.assertEqual(value['accounted_total_RAM_bytes'],512*GIB)
        self.assertEqual(value['effective_shared_bank_RSS_limit_bytes'],320*GIB)
        self.assertEqual(value['effective_native_RSS_limit_bytes'],32*GIB)
        self.assertTrue(value['scientific_scale_preserved'])

    def test_all_seven_full_native_and_archives_fit_exact512_after_source_release(self):
        value = self.measure(store_bytes=272*GIB,source_rss=0,source_pending=False,
            native_rss={gpu:32*GIB for gpu in range(7)},registered_rss=224*GIB)
        self.assertEqual(value['accounted_total_RAM_bytes'],512*GIB)
        self.assertEqual(value['owned_tmpfs_limit_bytes'],272*GIB)

    def test_actual_caps_are_hard_errors_never_silent_smaller_work(self):
        for values, reason in [({'store_bytes':145*GIB},'144GiB'),
                ({'source_rss':321*GIB},'320GiB'), ({'native_rss':{6:33*GIB}},'32GiB'),
                ({'auxiliary_rss':17*GIB},'16GiB')]:
            with self.subTest(values=values), self.assertRaisesRegex(MemoryError,reason):
                self.measure(**values)

    def test_store272_is_only_after_actual_source_release(self):
        with self.assertRaisesRegex(MemoryError,'144GiB'):
            self.measure(store_bytes=200*GIB)
        with self.assertRaisesRegex(MemoryError,'272GiB'):
            self.measure(source_pending=False,source_rss=0,store_bytes=273*GIB)
        with self.assertRaisesRegex(RuntimeError,'live source'):
            self.measure(source_pending=False)

    def test_source_only_one_native_and_candidate_overlap_is_rejected(self):
        with self.assertRaisesRegex(MemoryError,'only one'):
            self.measure(native_rss={0:GIB,6:GIB})
        with self.assertRaisesRegex(MemoryError,'Candidate native'):
            self.measure(native_rss={3:GIB})

    def test_real_shortage_typed_but_impossible_total_is_fatal(self):
        with self.assertRaises(resource.RAMAdmissionUnavailable) as failure:
            self.measure(store_bytes=120*GIB,source_rss=20*GIB,native_rss={6:10*GIB},
                registered_rss=30*GIB,auxiliary_rss=GIB,host_available=350*GIB)
        self.assertEqual(failure.exception.observation['required_group_and_cache_capacity_bytes'],512*GIB)
        with self.assertRaisesRegex(MemoryError,'total host'):
            self.measure(host_total=511*GIB)

    def test_projected_archives_do_not_get_added_back_as_actual_available_capacity(self):
        with self.assertRaises(resource.RAMAdmissionUnavailable):
            self.measure(store_bytes=120*GIB,projected_storage_bytes=24*GIB,
                source_rss=20*GIB,native_rss={6:10*GIB},registered_rss=30*GIB,
                auxiliary_rss=GIB,host_available=350*GIB)
        with self.assertRaisesRegex(MemoryError,'144GiB'):
            self.measure(store_bytes=140*GIB,projected_storage_bytes=5*GIB)

    def test_unknown_or_nonfinite_observation_is_not_zero_capacity(self):
        for key,value in [('store_bytes',None),('source_rss',float('nan')),
                ('auxiliary_rss',True),('source_pending',1)]:
            with self.subTest(key=key), self.assertRaises(ValueError):self.measure(**{key:value})


class RequestTests(unittest.TestCase):
    def request(self):
        root = Path(__file__).resolve().parent/'DEBUG_ram512_request'
        return dict(resource.FIELDS,root=str(root/'GPU6'),native_root=str(root/'GPU6'),owned_tmpfs_root=str(root),
            RAM512_claim_ledger=str(root/'cohort'/'claims.json'),code_sha256={key:'a'*64 for key in
                (resource.GUARD_FILE,resource.CLAIMS_FILE,resource.group.GUARD_FILE,
                 resource.shared.SHARED_GUARD_FILE,resource.pause.GUARD_FILE)},
            RAM512_source_pending_receipt=dict(path=str(root/'bootstrap.json'),sha256='b'*64))

    def test_every_explicit_field_and_new_code_pin_is_mandatory(self):
        for key in resource.FIELDS:
            for replacement in (True,float(resource.FIELDS[key]),None,resource.FIELDS[key]+1):
                value=self.request();value[key]=replacement
                with self.subTest(key=key,replacement=replacement), self.assertRaises(ValueError):
                    resource.validate_request(value)
        for filename in self.request()['code_sha256']:
            value=self.request();value['code_sha256'].pop(filename)
            with self.assertRaisesRegex(ValueError,'SHA'):resource.validate_request(value)

    def test_only_same_owned_accounting_root_claim_ledger(self):
        value=self.request();self.assertIs(resource.validate_request(value),value)
        value['RAM512_claim_ledger']=str(Path(value['owned_tmpfs_root']).parent/'foreign_claim.json')
        with self.assertRaisesRegex(ValueError,'same owned'):resource.validate_request(value)

    def test_old_resource_modules_are_not_changed_by_new_guard(self):
        self.assertEqual(resource.group.AGGREGATE_CAPACITY,800*GIB)
        self.assertEqual(resource.shared.RSS_LIMIT,384*GIB)
        self.assertEqual(resource.common.RSS_LIMIT,96*GIB)


class CoherentStoreTests(unittest.TestCase):
    def metadata(self,inode,byte_count,*,mode=stat.S_IFREG,device=7,uid=41833):
        return SimpleNamespace(st_dev=device,st_ino=inode,st_uid=uid,st_blocks=byte_count//512,
            st_mode=mode|0o700,st_size=byte_count,st_mtime_ns=1,st_ctime_ns=1)

    def context(self,entries):
        value=SimpleNamespace()
        from unittest.mock import MagicMock
        value=MagicMock();value.__enter__.return_value=iter(entries);return value

    def test_O_PATH_alias_of_BASE_is_not_subtracted_and_hidden_blocks_remain_charged(self):
        from unittest.mock import Mock
        root=Path(__file__).resolve().parent/'DEBUG_DEVICE'/'BASE'
        device_root=root.parent
        foreign=self.metadata(2,15*GIB,uid=19535)
        base_alias=self.metadata(3,5*GIB)
        entries=[]
        for name,value in [('outside',foreign),('BASE_alias',base_alias)]:
            item=Mock(path=str(device_root/name));item.stat.return_value=value;entries.append(item)
        def metadata(path):
            return self.metadata(1,0,mode=stat.S_IFDIR,device=7 if path in(root,device_root)else 8)
        with (patch.object(Path,'stat',autospec=True,side_effect=metadata),
              patch.object(resource.os,'scandir',return_value=self.context(entries)),
              patch.object(resource.os,'open',side_effect=[51,52]),
              patch.object(resource.os,'fstat',side_effect=lambda fd:foreign if fd==51 else base_alias),
              patch.object(resource.os,'statvfs',return_value=SimpleNamespace(f_blocks=150*GIB//4096,f_bfree=0,f_frsize=4096),create=True),
              patch.object(resource.os,'O_PATH',2097152,create=True),
              patch.object(resource.os,'O_NOFOLLOW',0,create=True),
              patch.object(resource.os,'O_CLOEXEC',0,create=True),
              patch.object(resource.os,'close') as close):
            value=resource.outside_inode_store_bound(root,excluded_inode_keys={(7,3)})
        self.assertEqual(value['held_stable_outside_allocated_bytes'],15*GIB)
        self.assertEqual(value['store_upper_bound_bytes'],135*GIB)
        self.assertEqual(close.call_count,2)
        self.assertFalse(value['foreign_files_read_for_content'])

    def test_denied_own_FD_has_explicit_conservative_file_charge_not_zero_or_aux(self):
        from unittest.mock import Mock
        root=Path(__file__).resolve().parent/'DEBUG_BASE';process=Mock(pid=123)
        process.cmdline.return_value=['sshd:','own@notty']
        metadata=dict(pid=123,create_time=1.,UID=41833)
        item=Mock(path=str(root/'checkpoint'));item.stat.return_value=self.metadata(2,125*GIB)
        with (patch.object(Path,'lstat',return_value=self.metadata(1,0,mode=stat.S_IFDIR)),
              patch.object(resource.os,'scandir',return_value=self.context([item])),
              patch.object(Path,'iterdir',side_effect=PermissionError('DEBUG own nondumpable')), 
              patch.object(resource.common,'_metadata',return_value=metadata),
              patch.object(resource,'outside_inode_store_bound',return_value=dict(store_upper_bound_bytes=135*GIB)) as bounds):
            value=resource.allocated_tmpfs(root,{123:process},uid=41833)
        self.assertEqual(value['owned_tmpfs_allocated_bytes'],135*GIB)
        self.assertEqual(value['owned_tmpfs_unknown_file_upper_bound_bytes'],10*GIB)
        self.assertEqual(value['unreadable_own_FD_actors'][0]['pid'],123)
        self.assertFalse(value['denied_FD_zero_fallback_used'])
        self.assertFalse(value['unknown_file_charge_is_actual_owned_attribution'])
        self.assertEqual(bounds.call_args.kwargs['excluded_inode_keys'],{(7,1),(7,2)})

    def test_unknown_bound_acquisition_failure_never_becomes_zero(self):
        from unittest.mock import Mock
        root=Path(__file__).resolve().parent/'DEBUG_BASE';process=Mock(pid=123)
        with (patch.object(Path,'lstat',return_value=self.metadata(1,0,mode=stat.S_IFDIR)),
              patch.object(resource.os,'scandir',return_value=self.context([])),
              patch.object(Path,'iterdir',side_effect=PermissionError),
              patch.object(resource.common,'_metadata',return_value=dict(pid=123,create_time=1.,UID=41833)),
              patch.object(resource,'outside_inode_store_bound',side_effect=OSError('DEBUG failed actual FS observation')),
              self.assertRaises(OSError)):
            resource.allocated_tmpfs(root,{123:process},uid=41833)

    def test_archive_remaining_projection_uses_only_verified_own_destination_growth(self):
        root=Path(__file__).resolve().parent/'DEBUG_ARCHIVE'
        row=dict(destination=str(root/'copy'),destination_parent_identity=[7,1,41833])
        with (patch.object(Path,'is_symlink',return_value=False),
              patch.object(Path,'stat',return_value=self.metadata(1,0,mode=stat.S_IFDIR)),
              patch.object(Path,'exists',return_value=False)):
            self.assertEqual(resource.archive_allocated_bytes(row,str(root)),0)
        row['destination_parent_identity']=[7,2,41833]
        with (patch.object(Path,'is_symlink',return_value=False),
              patch.object(Path,'stat',return_value=self.metadata(1,0,mode=stat.S_IFDIR)),
              self.assertRaisesRegex(RuntimeError,'parent inode')):
            resource.archive_allocated_bytes(row,str(root))


class FullGuardTests(unittest.TestCase):
    start=fixture.SharedV6Tests.start
    process=fixture.SharedV6Tests.process
    query=fixture.SharedV6Tests.query

    def setUp(self):
        fixture.SharedV6Tests.setUp(self)
        bootstrap=self.root/'bootstrap.json'
        bootstrap.write_text(json.dumps(dict(source_initially_pending=True,candidate_counts={'512':0,'1024':0,'2048':0})))
        for filename in (resource.GUARD_FILE,resource.CLAIMS_FILE,resource.pause.GUARD_FILE):
            path=self.root/filename;path.write_text('exact RAM512 CPU DEBUG dependency\n')
            for peer in self.requests.values():peer['code_sha256'][filename]=resource.common.sha(path)
        for gpu,peer in self.requests.items():
            peer.update(resource.FIELDS,owned_tmpfs_root=str(self.root),
                RAM512_claim_ledger=str(self.root/'RAM512_claims.json'),
                RAM512_source_pending_receipt=dict(path=str(bootstrap),sha256=resource.common.sha(bootstrap)),
                RAM_admission_pause_revision=1,shared_banks=str(self.root/'banks'))
            if gpu==5:peer['shared_bank_workers']=24
            path=Path(peer['root'])/'request.json';path.write_text(json.dumps(peer))
            self.document['registration'][str(gpu)]['request_sha256']=resource.common.sha(path)
        self.registry_path.write_text(json.dumps(self.document))
        self.start(patch.object(resource.common,'capture_cgroup_admission',side_effect=lambda pid:copy.deepcopy(self.admission)))
        self.start(patch.object(resource.common,'_query',side_effect=self.query))
        self.state=dict(claims={},source_pending=True,source_released=False,source_captured=[],
            storage_reservations={},priority_complete=False)
        self.state['claims']['source']=dict(kind='cpu_source',gpu=5,stage='candidate_bank',
            controller_identity=self.value['controller_identity'],command=self.stage.command,
            child_identity=self.ident(self.stage),captured=[self.ident(self.stage),self.ident(self.worker)])
        self.lease=resource.SharedBankLease(self.value)
        self.start(patch.object(self.lease,'_claim_state',side_effect=lambda:copy.deepcopy(self.state)))
        self.start(patch.object(self.lease,'_bind_self'))
        self.start(patch.object(self.lease,'source_complete',return_value=False))
        self.start(patch.object(self.lease,'priority_complete',return_value=False))
        self.start(patch.object(resource,'allocated_tmpfs',return_value=dict(owned_tmpfs_allocated_bytes=122*GIB,
            inode_count=10,deleted_open_inode_observations=[])))
        self.start(patch.object(resource,'own_process_inventory',side_effect=self.inventory))
        for actor in self.actors.values():
            actor.parents=lambda:[]

    @staticmethod
    def ident(actor):
        return dict(pid=actor.pid,create_time=actor.birth,UID=actor.uid,command=actor.command)

    def inventory(self):
        values={pid:actor for pid,actor in self.actors.items() if actor.uid==41833}
        return values,{pid:p.rss for pid,p in values.items()},{pid:self.ident(p) for pid,p in values.items()}

    def test_actual_full_shared_guard320_and512_keeps_native_declarations(self):
        self.worker.rss=294*GIB
        value=self.lease.verify(force=True)
        self.assertGreater(value['source_process_tree_RSS_bytes'],294*GIB)
        self.assertEqual(value['aggregate_native_and_cache_capacity_bytes'],512*GIB)
        self.assertEqual(value['own_process_tree_RSS_limit_bytes'],320*GIB)
        self.assertEqual(self.lease.native_request['RSS_limit_bytes'],96*GIB)
        self.assertEqual(self.lease.native_request['shared_bank_RSS_limit_GiB'],384)
        self.assertEqual(len(value['own_CPU_affinity']),64)
        self.assertTrue(value['shared_bank_CPU_only_CUDA_verified'])

    def test_real_inventory_counts_controllers_auxiliary_once_in_available_formula(self):
        value=self.lease.verify(force=True)
        all_rss=sum(actor.rss for actor in self.actors.values())
        self.assertEqual(value['measured_available_plus_owned_charge_bytes'],self.host.available+122*GIB+all_rss)
        self.assertEqual(value['accounted_total_RAM_bytes'],122*GIB+all_rss)

    def test_source_no_claim_never_starts_torch_or_pool(self):
        self.state['claims']={}
        with self.assertRaisesRegex(RuntimeError,'no atomic'):
            self.lease.verify(force=True)

    def test_source_hard320_failure_does_not_pause_or_reduce_workers(self):
        self.worker.rss=321*GIB
        with self.assertRaisesRegex(MemoryError,'320GiB'):self.lease.verify(force=True)
        self.assertFalse(any(call.args==(5,) for call in self.sleep.call_args_list))

    def test_cgroup512_accepted_but511_fails_despite_unlimited_host(self):
        self.admission['memory_limits'][0]['limit_bytes']=512*GIB
        self.lease.verify(force=True)
        self.lease._controller_cgroup=None
        self.admission['memory_limits'][0]['limit_bytes']=511*GIB
        with self.assertRaisesRegex(MemoryError,'512GiB'):self.lease.verify(force=True)

    def test_code_and_source_identity_failures_still_fatal(self):
        self.bank.write_text('source changed\n')
        with self.assertRaisesRegex(ValueError,'metadata changed'):self.lease.verify(force=True)

    def test_foreign_uid_pbs_expiry_cpu_and_cuda_remain_strict(self):
        self.worker.uid=999
        with self.assertRaisesRegex(RuntimeError,'foreign'):self.lease.verify(force=True)
        self.worker.uid=41833
        self.worker.cpus=[40]
        with self.assertRaisesRegex(RuntimeError,'Persistent exact shared'):self.lease.verify(force=True)
        self.worker.cpus=resource.SHARED_CPU_AFFINITY
        self.apps=str(self.worker.pid)+', GPU-other'
        with self.assertRaisesRegex(RuntimeError,'forbidden CUDA'):self.lease.verify(force=True)
        self.apps='';self.wall=4900
        with self.assertRaisesRegex(RuntimeError,'180'):self.lease.verify(force=True)

    def test_foreign_gpu5_may_stay_occupied_during_cpu_only_prep_without_signal(self):
        self.apps='999, GPU-exact';self.free_gpu=1000
        value=self.lease.verify(force=True)
        self.assertTrue(value['CUDA_ownership_not_granted'])
        self.assertTrue(value['shared_bank_CPU_only_CUDA_verified'])
        self.assertFalse(value['processes_signaled'])

    def test_source_quick_keeps_extra512_contract_and_periodic_full_inventory(self):
        self.lease.verify(force=True)
        with patch.object(resource,'own_process_inventory',wraps=resource.own_process_inventory) as observe:
            self.lease.quick_verify();self.assertEqual(observe.call_count,0)
            self.monotonic+=5
            self.lease.quick_verify();self.assertEqual(observe.call_count,1)
        self.lease.request['effective_shared_bank_RSS_limit_GiB']=384
        with self.assertRaisesRegex(ValueError,'contract'):self.lease.quick_verify()

    def test_no_retained_previous_registry_or_implicit_peer512_fields(self):
        peer=self.requests[0];peer.pop('memory_budget_revision')
        path=Path(peer['root'])/'request.json';path.write_text(json.dumps(peer))
        self.document['registration']['0']['request_sha256']=resource.common.sha(path)
        self.registry_path.write_text(json.dumps(self.document))
        with self.assertRaisesRegex(ValueError,'RAM512'):resource.SharedBankLease(self.value)

    def test_raw_reservation_observer_none_never_inserts_fake_claim(self):
        self.lease._shared_observer_lease=self.lease
        with patch.object(resource.shared.SharedBankLease,'verify',return_value={'actual':True}) as observer:
            result=self.lease.reservation_observer(self.state,None)
        self.assertEqual(result,{'actual':True})
        observer.assert_called_once_with(self.lease,force=True)
        self.assertIsNone(self.lease._claim_override)

    def test_running_source_reservation_refresh_observes_actual64_without_pause_or_ledger_lock(self):
        value=self.lease.reservation_observer(self.state,None)
        self.assertEqual(value['own_CPU_affinity'],resource.SHARED_CPU_AFFINITY)
        self.assertTrue(value['group_controller_observations']['5']['shared_CPU_bank_stage'])
        self.assertIsNone(self.lease._claim_override)
        self.assertIsNone(self.lease._shared_observer_lease._claim_override)

    def test_second_native_prospective_queues_without_publishing_or_pausing(self):
        self.state['claims']['native6']=dict(kind='native',gpu=6,stage='train',command=['python','train'])
        prospective=dict(kind='native',gpu=0,stage='train',command=['python','train0'])
        with self.assertRaises(ResourceClaimUnavailable):
            self.lease.reservation_observer(self.state,prospective)
        self.assertNotIn('__prospective__',self.state['claims'])

    def test_resolver_actual_terminal_and_living_argv_change_are_distinguished(self):
        resolver=resource.ProcessResolver(self.lease);identity=self.ident(self.worker)
        self.worker.state='zombie';self.worker.command=[]
        value=resolver.inspect(identity)
        self.assertEqual(value['state'],'terminal')
        self.assertEqual(value['actual_identity']['command'],[])
        self.worker.state='sleeping';self.worker.command=['replaced']
        with self.assertRaisesRegex(RuntimeError,'argv changed'):resolver.inspect(identity)


class NativeGuardTests(unittest.TestCase):
    start=native_fixture.GroupV5Tests.start
    process=native_fixture.GroupV5Tests.process
    query=native_fixture.GroupV5Tests.query
    ident=staticmethod(FullGuardTests.ident)
    inventory=FullGuardTests.inventory

    def setUp(self):
        native_fixture.GroupV5Tests.setUp(self)
        bootstrap=self.root/'bootstrap.json'
        bootstrap.write_text(json.dumps(dict(source_initially_pending=True,candidate_counts={'512':0,'1024':0,'2048':0})))
        for filename in (resource.GUARD_FILE,resource.CLAIMS_FILE,resource.pause.GUARD_FILE,
                         resource.shared.SHARED_GUARD_FILE,resource.group.GUARD_FILE):
            path=self.root/filename;path.write_text('exact native RAM512 CPU DEBUG dependency\n')
            for peer in self.requests.values():peer['code_sha256'][filename]=resource.common.sha(path)
        for gpu,peer in self.requests.items():
            peer.update(resource.FIELDS,owned_tmpfs_root=str(self.root),
                RAM512_claim_ledger=str(self.root/'RAM512_claims.json'),
                RAM512_source_pending_receipt=dict(path=str(bootstrap),sha256=resource.common.sha(bootstrap)),
                RAM_admission_pause_revision=1,shared_banks=str(self.root/'banks'),
                GPU_admission_retry_version=1)
            if gpu==5:peer['shared_bank_workers']=24
            path=Path(peer['root'])/'request.json';path.write_text(json.dumps(peer))
            self.document['registration'][str(gpu)]['request_sha256']=resource.common.sha(path)
        self.registry_path.write_text(json.dumps(self.document))
        self.start(patch.object(resource.common,'capture_cgroup_admission',side_effect=lambda pid:copy.deepcopy(self.admission)))
        self.start(patch.object(resource.common,'_query',side_effect=self.query))
        self.state=dict(claims={},source_pending=True,source_released=False,source_captured=[],
            storage_reservations={},priority_complete=True)
        for gpu,kind,child,captured in [(0,'native',self.stage,[self.stage,self.worker]),
                                       (5,'cpu_source',self.actors[305],[self.actors[305]])]:
            self.state['claims'][str(gpu)]=dict(kind=kind,gpu=gpu,stage='train' if gpu==0 else 'candidate_bank',
                controller_identity=self.requests[gpu]['controller_identity'],command=child.command,
                child_identity=self.ident(child),captured=[self.ident(p) for p in captured])
        self.lease=resource.BasicLease(self.value)
        self.start(patch.object(self.lease,'_claim_state',side_effect=lambda:copy.deepcopy(self.state)))
        self.start(patch.object(self.lease,'_bind_self'))
        self.start(patch.object(self.lease,'source_complete',return_value=False))
        self.start(patch.object(self.lease,'priority_complete',return_value=False))
        self.start(patch.object(resource,'allocated_tmpfs',return_value=dict(owned_tmpfs_allocated_bytes=122*GIB,
            inode_count=10,deleted_open_inode_observations=[])))
        self.start(patch.object(resource,'own_process_inventory',side_effect=self.inventory))
        for actor in self.actors.values():actor.parents=lambda:[]

    def test_native_complete32_and_source320_guard_admits_with_cgroup512(self):
        self.stage.rss=12*GIB;self.actors[305].rss=294*GIB
        self.admission['memory_limits'][0]['limit_bytes']=512*GIB
        value=self.lease.verify(force=True)
        self.assertEqual(value['native_RSS_by_GPU_bytes']['0'],12*GIB+self.worker.rss)
        self.assertEqual(value['source_process_tree_RSS_bytes'],294*GIB)
        self.assertEqual(value['effective_RAM_cap_bytes'],512*GIB)
        self.assertEqual(self.lease.request['RSS_limit_bytes'],96*GIB)

    def test_native33_real_failure_does_not_shrink_model_or_physical_batch(self):
        self.stage.rss=33*GIB
        with self.assertRaisesRegex(MemoryError,'32GiB'):self.lease.verify(force=True)

    def test_native_child_without_claim_is_fatal_before_any_torch(self):
        self.state['claims'].pop('0')
        with self.assertRaisesRegex(RuntimeError,'bind its exact prospective'):
            self.lease.verify(force=True)

    def test_live_foreign_cuda_remains_typed_queue_and_never_admitted(self):
        self.apps='777, GPU-exact'
        self.actors[777]=actors.Actor(777,['foreign','model'],uid=999)
        with self.assertRaises(resource.common.GPUAdmissionUnavailable):self.lease.verify(force=True)

    def test_new_model_cpu_mask_cannot_bypass_original_eight_core_contract(self):
        self.stage.cpus=[0]
        with self.assertRaisesRegex(RuntimeError,'affinity'):self.lease.verify(force=True)

    def test_one_same_group_registration_cannot_silently_use_rolling_prior_ledger(self):
        self.document['allowed_previous_resource_registries']={'1':{'path':str(self.root/'other.json'),'sha256':'a'*64}}
        self.registry_path.write_text(json.dumps(self.document))
        with self.assertRaises((ValueError,FileNotFoundError)):
            resource.BasicLease(self.value)


if __name__=='__main__':unittest.main()
