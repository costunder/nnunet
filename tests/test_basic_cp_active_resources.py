"""CPU-only operational subset/actor-isolation regressions; no remote work."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from hiercp_v1x import basic_cp_single_parent_resources as resource
from hiercp_v1x import basic_cp_direct_MIG_access as direct


class ActiveResourceTests(unittest.TestCase):
    def fixture(self,arms=('multi7','candidates512','candidates2048'),*,shared=False):
        parent='GPU-11111111-1111-1111-1111-111111111111'
        assigned={arm:parent if shared else 'MIG-'+str(index+1) for index,arm in enumerate(arms)}
        holder_id=dict(pid=500,create_time=5,UID=resource.UID,command=['DEBUG_launcher'])
        host_id=dict(pid=600,create_time=6,UID=resource.UID,command=['/bin/bash'])
        memory=dict(path='/system.slice/ssh.service',limit=2048*resource.GIB,usage=1000)
        receipt=dict(format=resource.FORMAT,UID=resource.UID,owner='aicompetition06',node='test-node',
            mode='full_GPU_shared6'if shared else resource.DIRECT_MODE,active_arms=list(arms),
            physical_parent_GPU_UUID=parent,execution_UUID_by_arm=assigned,job_id='123.ECE-util1',
            PBS_lifetime=dict(node_boot_id='boot-DEBUG',query_started_monotonic=100,
                walltime_seconds=86400,elapsed_seconds=0,remaining_seconds=86400,raw_PBS_response_sha256='a'*64),
            CPU_affinity=list(range(8*len(arms))),holder_identity=holder_id,host_allocation_process=host_id,
            PBS_GPU_owned_reservation=False,PBS_CPU_owned_reservation=False,PBS_job_role='host_lifetime_only',
            explicit_existing_direct_execution_requested=True,direct_launcher_cgroup=resource.cgroup_identity(memory),
            VRAM_reservation_bytes_by_arm={arm:9*resource.GIB for arm in arms},VRAM_floor_bytes=256*2**20)
        holder=dict(identity=holder_id,CPU_affinity=receipt['CPU_affinity'],memory_limit_bytes=memory['limit'],
            memory_cgroup=resource.cgroup_identity(memory),environment={},
            host_allocation_holder=dict(identity=host_id,environment={'PBS_JOBID':receipt['job_id']}))
        job=dict(job_id=receipt['job_id'],job_state='R',Job_Owner='aicompetition06@host',exec_host='test-node/0',
            resources_used={'walltime':'00:15:00'},Resource_List=dict(ngpus=1,ncpus=8*len(arms),walltime='24:00:00'))
        if shared:holder['environment'].update(PBS_JOBID=receipt['job_id'],CUDA_VISIBLE_DEVICES=parent)
        devices={token:dict(parent_UUID=parent,index=0,MIG=not shared,GI=None if shared else index+7,
            CI=None if shared else 0,MIG_enabled=not shared,total_bytes=(80 if shared else 10)*resource.GIB,
            used_bytes=0,compute_actors=[])for index,token in enumerate(assigned.values())}
        receipt['execution_device_observations']=copy.deepcopy(devices)
        return receipt,job,holder,devices,memory

    def validate(self,receipt,job,holder,devices):
        return resource.validate_allocation(receipt,job,holder,devices,now_monotonic=1000,
            node_boot_id='boot-DEBUG',query_started_monotonic=1000)

    def test_original_six_default_and_explicit_three_arm_resume(self):
        receipt,*_=self.fixture(resource.ARMS);receipt.pop('active_arms')
        self.assertEqual(resource.execution_arms(receipt),resource.ARMS)
        receipt,job,holder,devices,_=self.fixture()
        self.assertEqual(resource.execution_arms(receipt),('multi7','candidates512','candidates2048'))
        self.assertEqual(self.validate(receipt,job,holder,devices),set(devices))
        holder['memory_limit_bytes']=112*resource.GIB
        self.validate(receipt,job,holder,devices)
        holder['memory_limit_bytes']-=1
        with self.assertRaisesRegex(ValueError,'reserve selected'):
            self.validate(receipt,job,holder,devices)

    def test_empty_duplicate_unknown_unordered_or_inexact_UUID_subset_is_rejected(self):
        for arms in ([],['multi5','multi5'],['not_an_original_arm'],['candidates512','multi7'],'multi5'):
            receipt,*_=self.fixture();receipt['active_arms']=arms
            with self.subTest(arms=arms),self.assertRaisesRegex(ValueError,'active_arms'):
                resource.execution_arms(receipt)
        receipt,*_=self.fixture();receipt['execution_UUID_by_arm']['multi5']='MIG-extra'
        with self.assertRaisesRegex(ValueError,'UUID keys'):resource.execution_arms(receipt)
        receipt,*_=self.fixture();receipt.pop('active_arms')
        with self.assertRaisesRegex(ValueError,'UUID keys'):resource.execution_arms(receipt)

    def test_three_native_envelopes_preserve_individual_and_aggregate_limits(self):
        receipt,_,_,devices,_=self.fixture();arms=resource.execution_arms(receipt)
        rss={arm:32*resource.GIB for arm in arms}
        result=resource.validate_capacity(receipt,devices,native_RSS=rss,auxiliary_RSS=16*resource.GIB,
            cgroup_usage_bytes=512*resource.GIB)
        self.assertEqual(result['native_RSS_bytes'],96*resource.GIB)
        for kind in ('missing_RSS','unselected_peak','individual_RSS','aggregate_RAM'):
            current=copy.deepcopy(receipt);values=dict(rss);usage=512*resource.GIB
            if kind=='missing_RSS':values.pop(arms[-1])
            if kind=='unselected_peak':current['VRAM_reservation_bytes_by_arm']['multi5']=9*resource.GIB
            if kind=='individual_RSS':values[arms[0]]+=1
            if kind=='aggregate_RAM':usage+=1
            with self.subTest(kind=kind),self.assertRaises(ValueError):
                resource.validate_capacity(current,devices,native_RSS=values,auxiliary_RSS=0,cgroup_usage_bytes=usage)

    def test_three_compute_instances_cannot_double_charge_shared_MIG_memory(self):
        receipt,job,holder,devices,_=self.fixture();tokens=list(devices)
        devices[tokens[1]]['GI']=devices[tokens[0]]['GI'];devices[tokens[1]]['CI']=1
        receipt['execution_device_observations']=copy.deepcopy(devices)
        with self.assertRaisesRegex(ValueError,'independent MIG memory partitions'):
            self.validate(receipt,job,holder,devices)

    def test_other_MIG_dynamic_usage_is_independent_but_static_fit_and_RAM_are_global(self):
        receipt,_,_,devices,_=self.fixture();arms=resource.execution_arms(receipt)
        other=receipt['execution_UUID_by_arm'][arms[-1]]
        devices[other]['used_bytes']=devices[other]['total_bytes']
        arguments=dict(native_RSS={arm:4096 for arm in arms},auxiliary_RSS=0,cgroup_usage_bytes=512*resource.GIB)
        resource.validate_capacity(receipt,devices,arm=arms[0],**arguments)
        with self.assertRaisesRegex(ValueError,'CUDA memory floor'):
            resource.validate_capacity(receipt,devices,**arguments)
        receipt['VRAM_reservation_bytes_by_arm'][arms[-1]]=11*resource.GIB
        with self.assertRaisesRegex(ValueError,'Reserved native stages'):
            resource.validate_capacity(receipt,devices,arm=arms[0],**arguments)

    def unknown(self,pid=900):
        actor=Mock(pid=pid);actor.uids.return_value=SimpleNamespace(real=resource.UID)
        actor.create_time.return_value=90;actor.cmdline.return_value=['DEBUG_other_job']
        actor.ppid.return_value=1;actor.status.return_value=resource.psutil.STATUS_RUNNING
        return actor

    def test_other_MIG_actor_is_independent_but_preflight_still_checks_all(self):
        receipt,_,_,devices,_=self.fixture();arms=resource.execution_arms(receipt)
        token=receipt['execution_UUID_by_arm'][arms[-1]];devices[token]['compute_actors']=[{'pid':900}]
        with patch.object(resource.psutil,'Process',return_value=self.unknown())as inspect:
            resource.validate_execution_actors(receipt,devices,{},arm=arms[0])
            inspect.assert_not_called()
            with self.assertRaisesRegex(ValueError,'Unknown own or foreign CUDA actor'):
                resource.validate_execution_actors(receipt,devices,{})

    def test_same_MIG_unknown_same_UID_actor_keeps_precise_process_and_device_evidence(self):
        receipt,_,_,devices,_=self.fixture();arm=resource.execution_arms(receipt)[0]
        token=receipt['execution_UUID_by_arm'][arm];devices[token]['compute_actors']=[{'pid':900}]
        observations=[];actor=self.unknown()
        with patch.object(resource.psutil,'Process',return_value=actor),self.assertRaises(ValueError)as error:
            resource.validate_execution_actors(receipt,devices,{},arm=arm,observations=observations)
        proof=observations[0]
        self.assertEqual(proof['observed_token'],token);self.assertEqual(proof['observed_GPU_instance'],7)
        self.assertEqual(proof['observed_compute_instance'],0)
        self.assertEqual(proof['unregistered_process'],dict(pid=900,UID=resource.UID,create_time=90,
            command=['DEBUG_other_job'],PPID=1))
        self.assertIn(token,str(error.exception));actor.environ.assert_not_called()

    def test_shared_GPU_scope_cannot_hide_an_unknown_actor_or_wrong_MIG_owner(self):
        receipt,_,_,devices,_=self.fixture(shared=True);arm=resource.execution_arms(receipt)[0]
        token=receipt['execution_UUID_by_arm'][arm];devices[token]['compute_actors']=[{'pid':900}]
        with patch.object(resource.psutil,'Process',return_value=self.unknown()),self.assertRaises(ValueError):
            resource.validate_execution_actors(receipt,devices,{},arm=arm)
        receipt,_,_,devices,_=self.fixture();arm=resource.execution_arms(receipt)[0]
        token=receipt['execution_UUID_by_arm'][arm];devices[token]['compute_actors']=[{'pid':900}]
        with self.assertRaisesRegex(ValueError,'another execution device'):
            resource.validate_execution_actors(receipt,devices,{900:dict(UID=resource.UID,arm='candidates512')},arm=arm)

    def test_unknown_actor_disappearance_preserves_partial_evidence_in_guard_receipt(self):
        receipt,_,_,devices,_=self.fixture();arm=resource.execution_arms(receipt)[0]
        token=receipt['execution_UUID_by_arm'][arm];devices[token]['compute_actors']=[{'pid':900}]
        actor=self.unknown();actor.cmdline.side_effect=resource.psutil.NoSuchProcess(900)
        with tempfile.TemporaryDirectory(prefix='CPU_DEBUG_actor_evidence_',dir=Path(__file__).resolve().parent)as folder:
            lease=resource.Lease.__new__(resource.Lease);lease.arm=arm
            lease.cohort={'experiments':{arm:{'output_root':folder}}};lease.last_probe={}
            def fail(**kwargs):
                resource.validate_execution_actors(receipt,devices,{},arm=arm,
                    observations=lease.last_probe.setdefault('execution_actor_observations',[]))
            lease._verify=fail
            with patch.object(resource.psutil,'Process',return_value=actor),self.assertRaises(ValueError):lease.verify()
            proof=json.loads((Path(folder)/'last_resource_guard.json').read_text())
        process=proof['execution_actor_observations'][0]['unregistered_process']
        self.assertEqual(proof['status'],'FAILED');self.assertEqual(process['create_time'],90)
        self.assertEqual(process['UID'],resource.UID);self.assertEqual(process['PPID'],1)
        self.assertIn('NoSuchProcess',process['command_read_error'])

    def test_three_arm_lease_uses_dense_CPU_offsets_all_selected_RSS_and_own_MIG_conflicts(self):
        receipt,job,holder,devices,memory=self.fixture();arms=resource.execution_arms(receipt)
        actors={arm:dict(pid=1000+i,create_time=10+i,UID=resource.UID,command=['CPU_DEBUG',arm])
            for i,arm in enumerate(arms)}
        observations={value['pid']:dict(identity=value,PPID=500,CPU_affinity=list(range(i*8,i*8+8)),
            memory_cgroup=resource.cgroup_identity(memory),RSS_bytes=4096)
            for i,value in enumerate(actors.values())}
        processes={pid:SimpleNamespace(pid=pid,children=lambda recursive:[])for pid in observations}
        lease=resource.Lease.__new__(resource.Lease);lease.receipt=receipt;lease.actors=actors
        lease.own=actors[arms[0]];lease.arm=arms[0];lease.cohort={};lease.files={};lease.previous_trees={};lease.last=0
        pbs=dict(raw=json.dumps({'Jobs':{receipt['job_id']:job}}),query_started_monotonic=1000)
        other_token=receipt['execution_UUID_by_arm'][arms[-1]]
        devices[other_token]['compute_actors']=[{'pid':900}]
        devices[other_token]['used_bytes']=devices[other_token]['total_bytes']
        measured=dict(native_RSS_by_arm={arm:4096 for arm in arms},current_uid_RSS_bytes=90*resource.GIB,
            own_tmpfs_allocated_bytes=20*resource.GIB,cohort_auxiliary_RSS_bytes=4096)
        def snapshot(process,*,validator=None,**kwargs):
            value=observations[process.pid]
            if validator is not None:validator(value)
            return value
        with patch.object(resource.os,'getuid',return_value=resource.UID,create=True), \
                patch.object(resource.socket,'gethostname',return_value='test-node'), \
                patch.object(resource.time,'monotonic',return_value=1000), \
                patch.object(resource,'observe_holder',return_value=(SimpleNamespace(pid=500,children=lambda recursive:[]),memory,holder)), \
                patch.object(resource,'_memory_cgroup',return_value=memory), \
                patch.object(resource,'actual_PBS',return_value=pbs), \
                patch.object(resource,'nvml_devices',return_value=devices)as nvml, \
                patch.object(resource.Path,'read_text',return_value='boot-DEBUG'), \
                patch.dict(resource.os.environ,{'CUDA_VISIBLE_DEVICES':receipt['execution_UUID_by_arm'][arms[0]]}), \
                patch.object(resource,'cuda_execution_UUID'), \
                patch.object(resource.psutil,'Process',side_effect=lambda pid:processes[pid]), \
                patch.object(resource,'process_snapshot',side_effect=snapshot), \
                patch.object(direct,'observe_direct_memory',return_value=measured)as rss:
            result=lease._verify(force=True)
        self.assertEqual(set(nvml.call_args.args[0]),set(receipt['execution_UUID_by_arm'].values()))
        self.assertEqual(set(rss.call_args.args[0]),set(arms));self.assertEqual(rss.call_args.kwargs['expected_arms'],3)
        self.assertEqual(set(rss.call_args.kwargs['native_PIDs_by_arm']),set(arms))
        self.assertEqual(result['actual_global_own_UID_and_tmpfs_RAM_bytes'],110*resource.GIB)
        self.assertEqual([lease.last_probe['native_children'][arm]['expected_CPU_affinity']for arm in arms],
            [list(range(0,8)),list(range(8,16)),list(range(16,24))])


if __name__=='__main__':unittest.main()
