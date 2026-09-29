"""Synthetic report fixtures; these do not establish CT partition quality."""
import copy
import io
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from l0_ezsp.config import load_profile
from l0_regions.admission_report import rejection_report, print_rejection


class AdmissionReport(unittest.TestCase):
    def fixture(self, level=1):
        group=dict(owner=0,shell=0,clusters=200,
            bbox_excess=dict(clusters=2,fine_nodes=30,fine_node_fraction=.3),
            variance_excess=dict(clusters=0,fine_nodes=0,fine_node_fraction=0.))
        stats=dict(nodes_per_pair=[1500,10],edges_per_pair=[40000,12],
            roles={'target_context':dict(group_diagnostics=[group])})
        runner=SimpleNamespace(last_audit={f'scale{level}':stats})
        if level==2:runner.last_audit['scale1']=dict(nodes_per_pair=[900,10])
        return runner,load_profile(reg_scale1=.02,reg_scale2=.02),[dict(record_id='a'),dict(record_id='b')]

    def test_first_rejection_retains_counts_and_not_run(self):
        runner,profile,bindings=self.fixture()
        before=copy.deepcopy(runner.last_audit)
        r=rejection_report(runner,1,bindings,profile)
        self.assertEqual(r['audit']['scale2'],{'status':'NOT_RUN'})
        self.assertEqual({v['kind'] for v in r['pairs'][0]['violations']},
                         {'nodes','directed_edges','role_shell_nodes','bbox'})
        self.assertEqual(r['pairs'][1]['status'],'NOT_ADMITTED_BATCH_STOPPED')
        self.assertFalse(r['production_ready']);self.assertFalse(r['training_started'])
        self.assertEqual(before,runner.last_audit)

    def test_second_rejection_preserves_first(self):
        runner,profile,bindings=self.fixture(2)
        r=rejection_report(runner,2,bindings,profile)
        self.assertEqual(r['audit']['scale1']['nodes_per_pair'],[900,10])
        self.assertEqual(r['audit']['scale2']['status'],'REJECTED')
        self.assertEqual(r['pairs'][0]['violations'][0]['limit'],256)

    def test_terminal_prints_actual_limits_and_record(self):
        runner,profile,bindings=self.fixture()
        out=io.StringIO()
        with redirect_stdout(out):print_rejection(rejection_report(runner,1,bindings,profile))
        self.assertIn('a | N=1500 E=40000',out.getvalue())
        self.assertIn('1500>1024',out.getvalue())
        self.assertIn('Scale 2: NOT_RUN',out.getvalue())

    def test_admission_still_raises(self):
        from l0_regions.preparation import OfflinePartition, fixed_profile
        from l0_ezsp.partition import CoarseningConstraintError
        runner=OfflinePartition(fixed_profile(load_profile(reg_scale1=.02,reg_scale2=.02)),
            SimpleNamespace(check=lambda:None),allow_unvalidated_profile=False)
        with self.assertRaises(CoarseningConstraintError):
            runner._admission({},True,{})


if __name__=='__main__':unittest.main(verbosity=2)
