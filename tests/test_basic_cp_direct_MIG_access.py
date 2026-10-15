import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

from hiercp_v1x import basic_cp_direct_MIG_access as access


def proc_stat(pid=123, *, state='R', flags=0, start=900):
    fields = [str(pid), '(worker name ) with spaces)', state] + ['0']*21
    fields[8] = str(flags)
    fields[21] = str(start)
    return ' '.join(fields)+'\n'


def proc_status(pid=123, *, uid=41833, rss='12 kB', state='R'):
    return ('Name:\tworker\nState:\t'+state+'\nPid:\t'+str(pid)+'\nUid:\t'+str(uid)+'\t'+str(uid)+'\t'+str(uid)+'\t'+str(uid)+'\n'
            + ('' if rss is None else 'VmRSS:\t'+rss+'\n'))


class ProcRSSLifecycleTests(unittest.TestCase):
    def sample(self, stats, status, *, status_error=None):
        stats = iter(stats)
        def read(path, *args, **kwargs):
            if path.name == 'stat':
                value = next(stats)
                if isinstance(value, Exception):
                    raise value
                return value
            if status_error is not None:
                raise status_error
            return status
        with mock.patch.object(Path, 'read_text', read):
            return access._process_RSS_sample(Path('/proc/123'))

    def test_stat_comm_parentheses_and_raw_identity(self):
        self.assertEqual(access._proc_stat(proc_stat(flags=4), 123),
                         {'PID':123,'state':'R','flags':4,'starttime_ticks':900})

    def test_running_PF_EXITING_without_mm_is_explicit_release_not_zero(self):
        row = self.sample([proc_stat(),proc_stat(flags=4)], proc_status(rss=None))
        self.assertEqual(row['disposition'],'kernel_exit_without_mm')
        self.assertEqual(row['exit_evidence'],'PF_EXITING')
        self.assertNotIn('RSS_bytes',row)

    def test_live_missing_RSS_still_fails(self):
        with self.assertRaisesRegex(RuntimeError,'unavailable for live PID'):
            self.sample([proc_stat()]*6,proc_status(rss=None))

    def test_transient_exec_missing_mm_is_reread_without_fabricated_RSS(self):
        statuses=iter([proc_status(rss=None),proc_status(rss='23 kB')])
        def read(path,*args,**kwargs):
            return proc_stat() if path.name=='stat' else next(statuses)
        with mock.patch.object(Path,'read_text',read):
            row=access._process_RSS_sample(Path('/proc/123'))
        self.assertEqual(row['RSS_bytes'],23*1024)
        self.assertEqual(row['disposition'],'RSS_observed')

    def test_exiting_with_actual_RSS_remains_charged(self):
        row = self.sample([proc_stat(),proc_stat(flags=4)],proc_status(rss='17 kB'))
        self.assertEqual(row['RSS_bytes'],17*1024)
        self.assertEqual(row['disposition'],'RSS_observed')

    def test_zombie_without_mm_and_reap_between_reads(self):
        row = self.sample([proc_stat(state='Z'),proc_stat(state='Z')],proc_status(rss=None,state='Z'))
        self.assertEqual(row['disposition'],'kernel_exit_without_mm')
        row = self.sample([proc_stat(),FileNotFoundError()],proc_status())
        self.assertEqual(row['disposition'],'reaped_during_sample')
        self.assertNotIn('RSS_bytes',row)

    def test_reused_PID_resamples_new_identity_before_attributing_RSS(self):
        row = self.sample([proc_stat(start=900),proc_stat(start=901),proc_stat(start=901),proc_stat(start=901)],proc_status())
        self.assertEqual(row['starttime_ticks'],901)
        self.assertEqual(row['RSS_bytes'],12*1024)

    def test_continuously_changing_identity_refuses(self):
        with self.assertRaisesRegex(RuntimeError,'identity changed throughout'):
            self.sample([proc_stat(start=i)for i in range(6)],proc_status())

    def test_malformed_RSS_and_unreadable_live_status_refuse(self):
        for rss in ['-1 kB','1 MB','not-observed kB']:
            with self.subTest(rss=rss),self.assertRaisesRegex(RuntimeError,'units/value'):
                self.sample([proc_stat(),proc_stat()],proc_status(rss=rss))
        with self.assertRaises(PermissionError):
            self.sample([proc_stat()],proc_status(),status_error=PermissionError('denied'))

    def test_registered_PID_replacement_charged_globally_not_to_old_actor(self):
        replacement=dict(PID=123,UID=41833,RSS_bytes=700,starttime_ticks=901,disposition='RSS_observed')
        rows,active,released,reuse=access._cohort_RSS_members([replacement],{123},{},41833,
            {123:{'starttime_ticks':900}})
        self.assertEqual(sum(row['RSS_bytes']for row in rows),700)
        self.assertFalse(active)
        self.assertFalse(released)
        self.assertEqual(reuse[0]['observed_starttime_ticks'],901)
        proof=access._memory_accounting(rows,{'a':0},active,expected_arms=1,native_reserve_bytes=1000,
            auxiliary_reserve_bytes=100,max_user_RAM_bytes=2000,own_tmpfs_bytes=0,host_available_bytes=2000)
        self.assertEqual(proof['prospective_user_RAM_bytes'],1800)

    def test_registered_same_birth_UID_change_fails(self):
        changed=dict(PID=123,UID=999,RSS_bytes=700,starttime_ticks=900,disposition='RSS_observed')
        with self.assertRaisesRegex(RuntimeError,'UID changed'):
            access._cohort_RSS_members([],{123},{123:changed},41833,{123:{'starttime_ticks':900}})

    def test_registered_terminal_and_reaped_processes_need_no_exists_guess(self):
        terminal=dict(PID=123,UID=41833,starttime_ticks=900,state='R',flags=4,
                      disposition='kernel_exit_without_mm',exit_evidence='PF_EXITING')
        rows,active,released,reuse=access._cohort_RSS_members([],{123},{123:terminal},41833,
            {123:{'starttime_ticks':900}})
        self.assertEqual(released,[terminal]);self.assertFalse(rows);self.assertFalse(active);self.assertFalse(reuse)
        with mock.patch.object(access,'_process_RSS_sample',return_value={'PID':123,'disposition':'reaped_during_sample'}):
            rows,active,released,reuse=access._cohort_RSS_members([],{123},{},41833,
                {123:{'starttime_ticks':900}})
        self.assertEqual(released[0]['disposition'],'reaped_during_sample')

    def test_missing_enumerated_PID_reuse_is_measured_and_charged(self):
        replacement=dict(PID=123,UID=41833,RSS_bytes=800,starttime_ticks=901,disposition='RSS_observed')
        with mock.patch.object(access,'_process_RSS_sample',return_value=replacement):
            rows,active,released,reuse=access._cohort_RSS_members([],{123},{},41833,
                {123:{'starttime_ticks':900}})
        self.assertEqual(rows,[replacement]);self.assertFalse(active);self.assertEqual(len(reuse),1)

    def test_own_collector_excludes_only_proven_exit_and_preserves_live_RSS(self):
        workspace=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(prefix='DEBUG_proc_RSS_',dir=str(workspace))as folder:
            root=Path(folder).resolve();self.assertIn(workspace,root.parents)
            uid=root.stat().st_uid
            for pid,flags,rss in [(123,4,None),(124,4,'9 kB')]:
                path=root/str(pid);path.mkdir()
                (path/'stat').write_text(proc_stat(pid,flags=flags))
                (path/'status').write_text(proc_status(pid,uid=uid,rss=rss))
            lifecycle={}
            rows,released=access._own_process_RSS(uid,proc_root=root,lifecycle=lifecycle)
        self.assertEqual([row['PID']for row in rows],[124])
        self.assertEqual(rows[0]['RSS_bytes'],9*1024)
        self.assertEqual(released,[123]);self.assertEqual(lifecycle[123]['exit_evidence'],'PF_EXITING')

    @unittest.skipUnless(sys.platform=='linux' and hasattr(os,'getuid') and os.getuid()==access.UID,
                         'DEBUG requires actual Linux UID41833 procfs; no fake production resource evidence')
    def test_DEBUG_100_natural_workers_exercise_actual_memory_observer(self):
        # CPU-only lifecycle DEBUG: production model/data settings are untouched.
        names=('multi5','multi7','multi10','candidates512','candidates1024','candidates2048')
        parent=os.getpid()
        parent_identity=access._proc_stat(Path('/proc',str(parent),'stat').read_text(),parent)
        ended=snapshots=0
        for group in range(10):
            children=[]
            identities={parent:parent_identity}
            try:
                for index in range(10):
                    child=subprocess.Popen([sys.executable,'-B','-c',
                        'import time; payload=bytearray(1024*1024); time.sleep('+str(.2+.03*(index%5))+')'])
                    children.append(child)
                    identities[child.pid]=access._proc_stat(Path('/proc',str(child.pid),'stat').read_text(),child.pid)
                registered=set(identities)
                native={arm:set()for arm in names};native[names[0]]=registered-{parent}
                while True:
                    proof=access.observe_direct_memory({arm:0 for arm in names},registered,
                        native_PIDs_by_arm=native,cohort_identities=identities)
                    snapshots+=1
                    self.assertTrue(proof['native_RSS_from_same_process_snapshot'])
                    self.assertLessEqual(proof['prospective_user_RAM_bytes'],512*access.GIB)
                    self.assertFalse(proof['registered_PID_reuse_observations'])
                    if all(child.poll()is not None for child in children):
                        break
                    time.sleep(.01)
                # A second full observer pass explicitly includes all reaped IDs.
                proof=access.observe_direct_memory({arm:0 for arm in names},registered,
                    native_PIDs_by_arm=native,cohort_identities=identities)
                snapshots+=1
                self.assertTrue(set(child.pid for child in children)<=set(proof['registered_PIDs_proven_released']))
                ended+=len(children)
            finally:
                for child in children:
                    self.assertEqual(child.wait(timeout=30),0)
        self.assertEqual(ended,100)
        print('DEBUG actual direct-memory observer: naturally ended workers='+str(ended)+' observer snapshots='+str(snapshots))


class DirectMIGTests(unittest.TestCase):
    def capture_CPUs(self,actual,utilization,*,allowed=None):
        workspace=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(prefix='DEBUG_shared_CPU_',dir=str(workspace))as folder:
            root=Path(folder).resolve();self.assertIn(workspace,root.parents)
            job=root/'PBS_DEBUG_fully_assigned';job.mkdir();(job/'cpuset.cpus').write_text('0-127')
            real_Path=Path
            def path(value):
                return root if str(value)=='/sys/fs/cgroup/cpuset/pbs_jobs.service/jobid'else real_Path(value)
            with mock.patch.object(access,'observe_direct_memory',return_value={'DEBUG':True})as memory, \
                    mock.patch.object(access,'Path',side_effect=path), \
                    mock.patch.object(access.os,'sched_getaffinity',return_value=set(actual),create=True), \
                    mock.patch.object(access.os,'sched_setaffinity',create=True)as set_affinity, \
                    mock.patch('psutil.cpu_percent',return_value=utilization)as measured:
                result=access.capture_direct_resources({'DEBUG_native':0},{123},allowed_CPU_affinity=allowed)
                set_affinity.assert_not_called();memory.assert_called_once_with({'DEBUG_native':0},{123})
                measured.assert_called_once_with(interval=.1,percpu=True)
        return result

    def test_shared_direct_CPU_selection_uses_actual_allowed128_despite_full_PBS_assignment(self):
        utilization=[(127-cpu)/2 for cpu in range(128)]
        proof=self.capture_CPUs(range(128),utilization)
        selected=[cpu for group in proof['selected_CPU_groups']for cpu in group]
        self.assertEqual(selected,list(range(127,79,-1)))
        self.assertEqual([len(group)for group in proof['selected_CPU_groups']],[8]*6)
        self.assertEqual(len(set(selected)),48)
        self.assertEqual(proof['observed_PBS_assigned_CPUs'],list(range(128)))
        self.assertEqual(proof['selected_CPUs_shared_with_observed_PBS_jobs'],list(range(80,128)))
        self.assertEqual(proof['excluded_PBS_CPUs'],[])
        self.assertFalse(proof['PBS_CPU_owned_reservation']);self.assertFalse(proof['CPU_exclusive_reservation_claimed'])
        self.assertEqual(proof['CPU_affinity'],list(range(128)))
        self.assertIn('shared, not reserved',proof['CPU_selection'])

    def test_shared_direct_CPU_selection_never_exceeds_actual_or_explicit_allowed_cpuset(self):
        for actual,allowed in((list(range(47)),None),(list(range(128)),list(range(47)))):
            with self.subTest(actual=len(actual),allowed=allowed),self.assertRaisesRegex(RuntimeError,'allowed SSH affinity lacks 48'):
                self.capture_CPUs(actual,[0]*128,allowed=allowed)
        with self.assertRaisesRegex(RuntimeError,'exceed actual SSH process affinity'):
            self.capture_CPUs(range(48),[0]*128,allowed=range(49))
        proof=self.capture_CPUs(range(128),[0]*128,allowed=range(64,128))
        self.assertEqual([cpu for group in proof['selected_CPU_groups']for cpu in group],list(range(64,112)))

    def test_shared_direct_CPU_selection_still_requires_measured_utilization_for_allowed_IDs(self):
        with self.assertRaisesRegex(RuntimeError,'observations do not cover'):
            self.capture_CPUs(range(128),[0]*127)

    def accounting(self, **updates):
        values = dict(rows=[{'PID':1,'RSS_bytes':2},{'PID':2,'RSS_bytes':10},{'PID':3,'RSS_bytes':5}],
                      native_RSS_by_arm={'a':8,'b':2}, cohort_PIDs={1,2}, expected_arms=2,
                      native_reserve_bytes=20, auxiliary_reserve_bytes=10,
                      max_user_RAM_bytes=100, own_tmpfs_bytes=7, host_available_bytes=100)
        values.update(updates)
        return access._memory_accounting(**values)

    def mig(self, index, **changes):
        row = dict(MIG_UUID='MIG-'+str(index),parent_UUID='GPU-parent',GPU_instance_id=index,
                   compute_instance_id=0,compute_process_count=0,graphics_process_count=0,
                   total_bytes=10*access.GIB,free_bytes=10*access.GIB,used_bytes=0)
        row.update(changes)
        return row

    def test_cpu_ranges_are_exact(self):
        self.assertEqual(access.parse_CPU_list('0-3,7,11-12'), [0,1,2,3,7,11,12])
        with self.assertRaises(ValueError):access.parse_CPU_list('8-2')

    def test_current_native_and_auxiliary_are_not_double_reserved(self):
        proof = self.accounting()
        self.assertEqual(proof['current_uid_RSS_bytes'],17)
        self.assertEqual(proof['cohort_auxiliary_RSS_bytes'],2)
        self.assertEqual(proof['additional_reserved_bytes'],38)
        self.assertEqual(proof['prospective_user_RAM_bytes'],62)

    def test_actual_inconsistency_or_capacity_refuses(self):
        cases = [{'cohort_PIDs':{99}}, {'native_RSS_by_arm':{'a':15,'b':2}},
                 {'native_RSS_by_arm':{'a':30,'b':2}}, {'auxiliary_reserve_bytes':1},
                 {'host_available_bytes':37}, {'max_user_RAM_bytes':61}]
        for changes in cases:
            with self.subTest(changes=changes), self.assertRaises(RuntimeError):self.accounting(**changes)

    def test_same_parent_empty_exactly_six_and_explicit_exclusion(self):
        rows = [self.mig(i)for i in range(7)]+[self.mig(100,parent_UUID='GPU-other')]
        selected=access.choose_empty_MIGs(rows,'GPU-parent',exclude_UUIDs=['MIG-3'])
        self.assertEqual([x['GPU_instance_id']for x in selected],[0,1,2,4,5,6])

    def test_busy_foreign_or_undersized_is_not_admitted(self):
        cases = [{'parent_UUID':'GPU-other'},{'compute_process_count':1},{'graphics_process_count':1},
                 {'used_bytes':65*2**20},{'total_bytes':8*access.GIB},{'free_bytes':8*access.GIB}]
        for changes in cases:
            with self.subTest(changes=changes), self.assertRaises(RuntimeError):
                access.choose_empty_MIGs([self.mig(i)for i in range(5)]+[self.mig(9,**changes)],'GPU-parent')

    def test_overlapping_MIG_identity_is_rejected(self):
        with self.assertRaises(RuntimeError):
            access.choose_empty_MIGs([self.mig(i)for i in range(5)]+[self.mig(5,MIG_UUID='MIG-0')],'GPU-parent')

    @unittest.skipUnless(os.name=='posix','Linux tmpfs allocation accounting uses actual st_blocks')
    def test_tmpfs_hardlinks_are_charged_once(self):
        with tempfile.TemporaryDirectory()as name:
            root=Path(name);first=root/'checkpoint';first.write_bytes(b'checkpoint-data')
            (root/'same-checkpoint').hardlink_to(first);(root/'not-followed').symlink_to(first)
            proof=access._owned_tmpfs(first.stat().st_uid,[root])
            self.assertEqual(proof['regular_files'],1)
            self.assertEqual(proof['allocated_bytes'],first.stat().st_blocks*512)


if __name__=='__main__':unittest.main()
