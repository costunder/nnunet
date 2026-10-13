"""CPU-only DEBUG regression for final worker exit and immutable prior ledgers."""
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from hiercp_v1x import basic_cp_group_resources as original
from hiercp_v1x import basic_cp_group_resources_v2 as resource
from hiercp_v1x import basic_cp_seven_resources as seven
from tests import test_basic_cp_group_resources as fixture
from tests import test_basic_cp_seven_resources as actors


class GroupV2Tests(unittest.TestCase):
    start=fixture.GroupLeaseTests.start
    query=fixture.GroupLeaseTests.query

    def process(self,pid=None):
        selected=getattr(self,'_default_stage_pid',200)if pid is None else pid
        if selected not in self.actors:raise seven.psutil.NoSuchProcess(selected)
        return self.actors[selected]

    def setUp(self):
        fixture.GroupLeaseTests.setUp(self)
        self.prior_path=self.registry_path
        self.prior_sha=seven.sha(self.prior_path)
        source=self.root/resource.GUARD_FILE;source.write_text('exact v2 guard CPU fixture\n')
        fresh=self.root/'GPU0_fresh';fresh.mkdir()
        controller=actors.Actor(110,['python','fresh_GPU0_controller'])
        stage=actors.Actor(400,['python','fresh_GPU0_training'])
        worker=actors.Actor(401,list(stage.command));worker.parent=lambda:stage
        controller.descendants=[stage,worker]
        self.actors.update({p.pid:p for p in(controller,stage,worker)})
        self.actors.pop(100);self.actors.pop(200);self.actors.pop(201)
        self.controller,self.stage,self.worker=controller,stage,worker
        self._default_stage_pid=400
        self.registry_path=self.root/'fresh_registry.json'
        self.value=copy.deepcopy(self.requests[0])
        self.value.update(root=str(fresh),native_root=str(fresh),shared_resource_registry=str(self.registry_path),
            controller_identity=dict(pid=110,create_time=700,UID=41833,command=controller.command))
        self.value['code_sha256'][resource.GUARD_FILE]=seven.sha(source)
        request_path=fresh/'request.json';request_path.write_text(json.dumps(self.value))
        self.document=copy.deepcopy(self.document)
        self.document['registration']['0'].update(root=str(fresh),controller_identity=self.value['controller_identity'],
                                                  request_sha256=seven.sha(request_path))
        self.document['allowed_previous_resource_registries']={str(gpu):dict(path=str(self.prior_path),sha256=self.prior_sha)
                                                               for gpu in range(1,7)}
        self.rewrite()
        self.lease=resource.BasicLease(self.value)
        # The private compiled method deliberately has independent globals;
        # mock those Linux observations as well as the original live module.
        self.start(patch.dict(resource.BasicLease._descendant.__globals__,{
            '_fork_no_exec':lambda p:getattr(p,'fork_no_exec',False),
            '_kernel_exiting':lambda p:getattr(p,'kernel_exiting',False)}))

    def rewrite(self):self.registry_path.write_text(json.dumps(self.document))

    def final_empty(self,*,exiting):
        self.worker.fork_no_exec=True
        return (patch.object(seven,'_fork_no_exec',side_effect=lambda p:getattr(p,'fork_no_exec',False)),
                patch.object(seven,'_kernel_exiting',return_value=exiting),
                patch.object(self.worker,'cmdline',side_effect=[list(self.stage.command),[]]))

    def test_final_same_birth_empty_argv_actual_kernel_exit_retains_real_rss(self):
        fork,exit_flag,argv=self.final_empty(exiting=True)
        with fork,exit_flag,argv:
            result=self.lease._descendant(self.worker)
        self.assertIs(result,self.worker)
        observation=self.lease._terminal_observations[401]
        self.assertEqual(observation['create_time'],700)
        self.assertEqual(observation['status'],'PF_EXITING')
        self.assertTrue(observation['RSS_still_accounted'])
        self.assertFalse(observation['argv_invented'])
        self.assertNotIn(401,self.lease._managed,'No fabricated final argv is admitted')
        self.assertEqual(self.lease._rss({401:result}),self.worker.rss)

    def test_exact_recorded_race_still_fails_original_live_guard(self):
        fork,exit_flag,argv=self.final_empty(exiting=True)
        with fork,exit_flag,argv,self.assertRaisesRegex(RuntimeError,'changed during live observation'):
            original.BasicLease._descendant(self.lease,self.worker)

    def test_final_empty_without_real_exit_flag_stays_failure_after_bounded_observation(self):
        fork,exit_flag,argv=self.final_empty(exiting=False)
        with fork,exit_flag,argv,self.assertRaisesRegex(RuntimeError,'changed during live observation'):
            self.lease._descendant(self.worker)
        self.assertNotIn(401,self.lease._terminal_observations)

    def test_kernel_exit_observation_cannot_admit_reused_pid_or_changed_uid(self):
        fork,_,argv=self.final_empty(exiting=True)
        def changed(process):self.worker.birth=999;return True
        with fork,patch.object(seven,'_kernel_exiting',side_effect=changed),argv,self.assertRaisesRegex(
                RuntimeError,'kernel-exit task changed exact PID/birth/UID'):
            self.lease._descendant(self.worker)

    def test_live_changed_nonempty_argv_is_not_the_new_terminal_case(self):
        with patch.object(self.worker,'cmdline',side_effect=[list(self.worker.command),['python','replacement']]),self.assertRaisesRegex(
                RuntimeError,'changed during live observation'):
            self.lease._descendant(self.worker)

    def test_existing_other_controllers_keep_prior_registry_and_full_resource_contract(self):
        result=self.lease.verify(force=True)
        self.assertEqual(result['aggregate_native_and_cache_capacity_bytes'],704*2**30)
        self.assertEqual(result['own_process_tree_RSS_limit_bytes'],96*2**30)
        self.assertEqual(result['candidate_subpool_RSS_limit_bytes'],288*2**30)
        self.assertEqual(self.requests[1]['shared_resource_registry'],str(self.prior_path))
        self.assertFalse(result['processes_signaled'])
        self.assertIn(str(self.prior_path),self.lease._files)

    def test_live_v1_constructor_still_rejects_unadmitted_different_registry(self):
        with self.assertRaisesRegex(ValueError,'different shared resource registry'):
            original.BasicLease(self.value)

    def test_previous_registry_sha_selected_row_and_request_are_exactly_pinned(self):
        self.document['registration']['1']['controller_identity']['create_time']+=1;self.rewrite()
        with self.assertRaisesRegex(ValueError,'selected controller/request row differs'):
            resource.BasicLease(self.value)

    def test_prior_ledger_byte_change_fails_initial_sha_and_runtime_metadata(self):
        self.prior_path.write_text(self.prior_path.read_text()+' ')
        with self.assertRaisesRegex(ValueError,'metadata changed'):self.lease.verify(force=True)
        with self.assertRaisesRegex(ValueError,'admitted Basic source'):resource.BasicLease(self.value)

    def test_prior_permission_is_required_and_cannot_be_granted_to_own_fresh_controller(self):
        self.document['allowed_previous_resource_registries'].pop('1');self.rewrite()
        with self.assertRaisesRegex(ValueError,'explicitly pinned prior ledger'):resource.BasicLease(self.value)
        self.document['allowed_previous_resource_registries']['0']=dict(path=str(self.prior_path),sha256=self.prior_sha)
        self.rewrite()
        with self.assertRaisesRegex(ValueError,'other-controller'):resource.BasicLease(self.value)

    def test_descriptor_cannot_use_ambiguous_sha_or_mutate_runtime_permissions(self):
        self.lease._previous_registry_manifest['1']['sha256']='f'*64
        with self.assertRaisesRegex(ValueError,'permissions changed in memory'):self.lease.verify(force=True)
        self.document['allowed_previous_resource_registries']['1']['sha']=self.prior_sha;self.rewrite()
        with self.assertRaisesRegex(ValueError,'unambiguous SHA'):resource.BasicLease(self.value)

    def test_short_sha_alias_is_explicitly_bound_to_identical_ledger_bytes(self):
        descriptor=self.document['allowed_previous_resource_registries']['1']
        descriptor['sha']=descriptor.pop('sha256');self.rewrite()
        lease=resource.BasicLease(self.value)
        self.assertIn(str(self.prior_path),lease._files)

    def test_both_v1_and_v2_guard_shas_are_required(self):
        for guard in(original.GUARD_FILE,resource.GUARD_FILE):
            value=copy.deepcopy(self.value);value['code_sha256'].pop(guard)
            with self.subTest(guard=guard),self.assertRaisesRegex(ValueError,'guard'):
                resource.BasicLease(value)


if __name__=='__main__':unittest.main()
