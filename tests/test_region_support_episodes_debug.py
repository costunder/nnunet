import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import torch
from hiercp_v222.v1_training import groups
from hiercp_v222.v1_local import support_for_recipient
from l0_regions.support_episodes import PatientEpisodes, contract, verify_migration


def fixture(n=6):
    rows=[dict(id=f'{g}-{i}',patient_group=str(g),donor_group=str(g),target=i%2,
               bounds=dict(edges=i+1)) for g in range(n) for i in range(6)]
    memory=dict(record_ids=[r['id'] for r in rows],patient_groups=list(map(str,range(n))),
                donor_groups=[r['donor_group'] for r in rows],
                owners=torch.tensor([int(r['patient_group']) for r in rows]),
                classes=torch.tensor([r['target'] for r in rows]),
                embeddings=torch.arange(len(rows)*128).reshape(-1,128).float())
    order=list(groups(SimpleNamespace(rows=rows),4,42,0))
    return rows,memory,order


class SupportEpisodesTests(unittest.TestCase):
    def test_complete_selected_patients_and_leakage(self):
        rows,memory,order=fixture()
        rows[7]['donor_group']='0';memory['donor_groups'][7]='0'
        ep=PatientEpisodes(rows,order,2,42,0).bind(memory)
        for query,s in ep.selections.items():
            expected=[i for i,r in enumerate(rows) if r['patient_group'] in s['patients'] and r['donor_group']!=query]
            self.assertEqual(s['indices'],expected)
            self.assertNotIn(query,s['patients'])
            self.assertEqual(len(s['patients']),2)
            x,o,c=ep.support(query)
            self.assertEqual(len(x),len(expected))
            self.assertEqual(set(c.tolist()),{0,1})
            self.assertEqual(set(o.tolist()),{0,1})
        self.assertEqual(ep.audit['query_coverage'],1)
        self.assertEqual(ep.audit['support_observation_coverage'],1)

    def test_deterministic_epoch_schedule_and_resume(self):
        rows,memory,order=fixture()
        left=PatientEpisodes(rows,order,2,42,0).bind(memory)
        right=PatientEpisodes(copy.deepcopy(rows),order,2,42,0).bind(memory)
        self.assertEqual(left.selections,right.selections)
        for query in list(left.selections)[2:]:
            for a,b in zip(left.support(query),right.support(query)):self.assertTrue(torch.equal(a,b))
        different=PatientEpisodes(rows,order,2,42,1)
        self.assertNotEqual(left.selections,different.selections)

    def test_all_patients_matches_original_support_and_gradient(self):
        rows,memory,order=fixture()
        ep=PatientEpisodes(rows,order,5,42,0).bind(memory)
        for query in ep.selections:
            a=ep.support(query);b=support_for_recipient(memory,query)
            for x,y in zip(a,b):self.assertTrue(torch.equal(x,y))
            w=torch.ones(128,3,requires_grad=True)
            grad1=torch.autograd.grad((a[0]@w).square().mean(),w)[0]
            grad2=torch.autograd.grad((b[0]@w).square().mean(),w)[0]
            self.assertTrue(torch.equal(grad1,grad2))

    def test_sparse_positive_support_class_coverage(self):
        rows,memory,order=fixture()
        for r in rows:r['target']=int(r['patient_group'] in ('1','2') and r['target']==1)
        ep=PatientEpisodes(rows,order,2,42,0)
        for s in ep.selections.values():self.assertEqual({rows[i]['target'] for i in s['indices']},{0,1})

    def test_invalid_counts_no_silent_clamp(self):
        rows,_,order=fixture()
        for n in (None,True,1,0,7):
            with self.assertRaises(ValueError):PatientEpisodes(rows,order,n,42,0)

    def test_missing_support_class_fails(self):
        rows,_,order=fixture()
        for r in rows:r['target']=0
        with self.assertRaisesRegex(ValueError,'lacks observed class'):PatientEpisodes(rows,order,2,42,0)

    def test_query_drop_duplicate_and_mixed_rejected(self):
        rows,_,order=fixture()
        for bad in (order[:-1],order+[order[0]],[[0,6]]+order):
            with self.assertRaises(ValueError):PatientEpisodes(rows,bad,2,42,0)

    def test_memory_binding_rejected(self):
        rows,memory,order=fixture()
        for key in ('record_ids','owners','classes','donor_groups'):
            bad=copy.deepcopy(memory)
            if key=='record_ids':bad[key][0]='wrong'
            elif key=='donor_groups':bad[key][0]='wrong'
            else:bad[key][0]=1
            with self.assertRaises(ValueError):PatientEpisodes(rows,order,2,42,0).bind(bad)

    def test_unreviewed_migration_and_other_changes_rejected(self):
        old=dict(batch=32,source=dict(core={},runtime={'l0_regions/training.py':'old'}))
        new=dict(old,support_training=contract(2),source=dict(core={},runtime={'l0_regions/training.py':'new'}))
        with patch('l0_regions.execution_upgrade.blob_hash',return_value='old'):
            self.assertEqual(verify_migration(old,new),'d05e4e3')
            with self.assertRaises(ValueError):verify_migration(old,dict(new,batch=16))
            with self.assertRaises(ValueError):verify_migration(dict(old,support_training=contract(2)),new)
            bad=copy.deepcopy(new);bad['source']['runtime']['tools/v22_rank_objective.py']='change'
            with self.assertRaises(ValueError):verify_migration(old,bad)

if __name__=='__main__':unittest.main()
