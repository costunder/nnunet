"""Explicit UNIT support/evaluation fixtures, not trained CT or CP evidence."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from hiercp_v222.model import PromptGraphModel
from hiercp_v222.v1_local import support_for_recipient
from l0_local_cnn.model import LocalBatch, LocalCNN, MODE
from l0_regions.training import hash_state
from tools.diagnose_local_cnn_learning import validate_interaction_options
from tools.local_cnn_interaction_runtime import make_evaluator, support_binding


class Budget:
    def __init__(self, fail=None):
        self.calls = 0
        self.fail = fail
    def check(self):
        self.calls += 1
        if self.calls == self.fail:
            raise MemoryError('UNIT explicit resource limit')


class Reader:
    def __init__(self, rows, images):
        self.ds = SimpleNamespace(rows=rows)
        self.images = images
        self.calls = []
    def get(self, ids):
        self.calls.append(list(ids))
        images = self.images.index_select(0, torch.tensor([0]+[i+1 for i in ids]))
        audit = [dict(case='UNIT D', origin=[0,0,0], shape=[6,6,6], spacing=[1.,1.,1.], anchor_in_organ=True)]
        audit += [dict(case=self.ds.rows[i]['case_id'], origin=[0,0,0], shape=[6,6,6],
                       spacing=[1.,1.,1.], anchor_in_organ=True) for i in ids]
        return LocalBatch(images, torch.ones_like(images, dtype=torch.bool),
            torch.zeros(len(ids), dtype=torch.long), torch.arange(1,len(ids)+1),
            torch.tensor(ids), audit).validate()


def fixture(device='cpu'):
    torch.set_num_threads(4); torch.manual_seed(951)
    rows = []
    for group in ('UNIT Q', 'UNIT A', 'UNIT B', 'UNIT C'):
        for i in range(4):
            donor = 'UNIT Q' if (group == 'UNIT A' and i == 3) or (group == 'UNIT B' and i == 0) else 'UNIT D'
            rows.append(dict(id=f'{group}:{i}', case_id=group, patient_group=group,
                donor_group=donor, donor_case_id=donor, donor_component=1,
                target=int(i in (0,3)), center=[1,1,i]))
    val = [dict(id=f'UNIT V:{i}', case_id='UNIT V', patient_group='UNIT V',
        donor_group='UNIT D', donor_case_id='UNIT D', donor_component=1,
        target=int(i in (0,3)), center=[1,1,i]) for i in range(4)]
    groups = sorted({row['patient_group'] for row in rows})
    memory = dict(record_ids=[row['id'] for row in rows], donor_groups=[row['donor_group'] for row in rows],
        patient_groups=groups, embeddings=torch.randn(len(rows),128,device=device),
        owners=torch.tensor([groups.index(row['patient_group']) for row in rows],device=device),
        classes=torch.tensor([row['target'] for row in rows],device=device))
    config = dict(architecture=MODE, channels=[12,24,32], convolutions=[2,3,3], hidden_dim=128,
        margin_mm=10., input='native_spacing_organ_only', readout='organ_masked_mean_each_scale',
        fusion='donor_target_difference_product', initialization='fresh_seed42', learning_policy='same_donor_live_v1')
    base = json.loads(Path('config/train.json').read_text())
    cfg = json.loads(Path('config/prompt_graph_v222_v1_l0.json').read_text())
    net = PromptGraphModel(cfg,base,{},local_encoder=LocalCNN(config)).to(device).eval()
    readers = [Reader(rows,torch.randn(len(rows)+1,1,6,6,6)), Reader(val,torch.randn(len(val)+1,1,6,6,6))]
    cases = []
    for split, reader, ids in [('train',readers[0],list(range(4))),('validation',readers[1],list(range(4)))]:
        selected = [reader.ds.rows[i] for i in ids]
        with torch.no_grad():
            embeddings = net.local(reader.get(ids).to(device)).detach()
        cases.append(dict(split=split,rows=selected,indices=ids,reader=reader,embeddings=embeddings,
                          truth=torch.tensor([row['target'] for row in selected],device=device)))
        reader.calls.clear()
    return net, rows, memory, cases, readers


class Episodes:
    def __init__(self, memory, rows, query, selected):
        indices = [i for i,row in enumerate(rows) if row['patient_group'] in selected
                   and query not in (row['patient_group'],row['donor_group'])]
        self.selections = {query:dict(indices=indices,patients=selected)}
        ids = torch.tensor(indices,device=memory['embeddings'].device)
        owners = torch.tensor([selected.index(rows[i]['patient_group']) for i in indices],device=ids.device)
        self.value = (memory['embeddings'][ids],owners,memory['classes'][ids])
    def support(self, query):
        return self.value


class InteractionRuntimeChecks(unittest.TestCase):
    def test_full_and_episode_support_keep_all_eligible_observations_cpu_cuda(self):
        for device in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
            with self.subTest(device=device):
                _, rows, memory, _, _ = fixture(device)
                support, ids, query = support_binding(memory,rows,'UNIT Q')
                expected = [i for i,row in enumerate(rows) if 'UNIT Q' not in (row['patient_group'],row['donor_group'])]
                self.assertEqual(ids,[rows[i]['id'] for i in expected]); self.assertEqual(len(ids),10)
                self.assertEqual(query,'UNIT Q')
                reference = support_for_recipient(memory,'UNIT Q')
                for actual,want in zip(support,reference):torch.testing.assert_close(actual,want,atol=0,rtol=0)
                episodes = Episodes(memory,rows,'UNIT Q',['UNIT A','UNIT C'])
                support,ids,_ = support_binding(memory,rows,'UNIT Q',episodes)
                self.assertEqual(len(ids),7)
                self.assertEqual(ids,[rows[i]['id'] for i in episodes.selections['UNIT Q']['indices']])
                self.assertEqual(support[1].tolist(),[0,0,0,1,1,1,1])

    def test_original_memory_binding_corruptions_rejected(self):
        _, rows, original, _, _ = fixture()
        def reject(mutate):
            memory = copy.deepcopy(original); mutate(memory)
            with self.assertRaises(ValueError):support_binding(memory,rows,'UNIT Q')
        reject(lambda value:value['record_ids'].reverse())
        reject(lambda value:value['donor_groups'].__setitem__(4,'UNIT wrong'))
        reject(lambda value:value['owners'].__setitem__(4,2))
        reject(lambda value:value['classes'].__setitem__(4,0))
        reject(lambda value:value.__setitem__('embeddings',value['embeddings'][:-1]))
        reject(lambda value:value['embeddings'].__setitem__((4,0),float('nan')))
        reject(lambda value:value.__setitem__('owners',value['owners'].reshape(-1,1)))
        reject(lambda value:value.__setitem__('classes',value['classes'].reshape(-1,1)))
        reject(lambda value:value.__setitem__('owners',value['owners'].to(torch.int32)))
        reject(lambda value:value.__setitem__('classes',value['classes'].to(torch.float32)))
        reject(lambda value:value.__setitem__('embeddings',value['embeddings'].to(torch.float64)))
        if torch.cuda.is_available():
            reject(lambda value:value.__setitem__('owners',value['owners'].to('cuda')))
            reject(lambda value:value.__setitem__('classes',value['classes'].to('cuda')))

    def test_episode_drop_substitution_and_compact_owner_corruption_rejected(self):
        _, rows, memory, _, _ = fixture()
        for defect in ('drop','substitution','owner','class','features'):
            with self.subTest(defect=defect):
                episode = Episodes(memory,rows,'UNIT Q',['UNIT A','UNIT C'])
                if defect == 'drop':episode.selections['UNIT Q']['indices'].pop()
                elif defect == 'substitution':episode.selections['UNIT Q']['indices'][0]=0
                elif defect == 'owner':episode.value[1][0]=1
                elif defect == 'class':episode.value[2][0]=1-episode.value[2][0]
                else:episode.value[0][0,0]+=1
                with self.assertRaises(ValueError):support_binding(memory,rows,'UNIT Q',episode)

    def test_evaluator_reuses_only_unchanged_local_state_and_restores_modes_cpu_cuda(self):
        for device in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
            with self.subTest(device=device):
                net, rows, memory, cases, readers = fixture(device)
                budget=Budget(); local_hash=hash_state(net.local.state_dict())
                evaluator=make_evaluator(cases,memory,rows,3,budget,local_hash)
                net.train();net.local.cnn.eval()
                modes=[module.training for module in net.modules()]
                preserved=hash_state(net.state_dict())
                first=evaluator(net)
                self.assertEqual([module.training for module in net.modules()],modes)
                self.assertEqual(hash_state(net.state_dict()),preserved)
                self.assertTrue(all(row['cached_query_basis_reused'] for split in ('train','validation') for row in first[split]['cases']))
                self.assertTrue(all(not reader.calls for reader in readers))
                with torch.no_grad():next(net.local.parameters()).add_(.001)
                changed=hash_state(net.state_dict())
                second=evaluator(net)
                self.assertEqual(hash_state(net.state_dict()),changed)
                self.assertEqual([module.training for module in net.modules()],modes)
                self.assertTrue(all(not row['cached_query_basis_reused'] for split in ('train','validation') for row in second[split]['cases']))
                self.assertEqual([reader.calls for reader in readers],[[[0,1,2],[3]],[[0,1,2],[3]]])
                self.assertEqual(second['train']['cases'][0]['records'],4)
                self.assertEqual(second['validation']['cases'][0]['records'],4)

    def test_evaluator_rejects_partial_reordered_or_substituted_case_and_bad_truth(self):
        for defect in ('drop','reorder','substitution','truth'):
            with self.subTest(defect=defect):
                net, rows, memory, cases, _ = fixture()
                case=cases[0]
                if defect=='drop':
                    for key in ('indices','rows'):case[key]=case[key][:-1]
                    for key in ('truth','embeddings'):case[key]=case[key][:-1]
                elif defect=='reorder':
                    for key in ('indices','rows'):case[key]=list(reversed(case[key]))
                    for key in ('truth','embeddings'):case[key]=case[key].flip(0)
                elif defect=='substitution':case['rows'][0]=dict(case['rows'][0],id='UNIT substituted')
                else:case['truth'][0]=1-case['truth'][0]
                evaluator=make_evaluator(cases,memory,rows,3,Budget(),hash_state(net.local.state_dict()))
                with self.assertRaises(ValueError):evaluator(net)

    def test_evaluator_resource_and_local_state_failure_restore_mixed_modes(self):
        net, rows, memory, cases, _ = fixture()
        net.train();net.local.cnn.eval();modes=[module.training for module in net.modules()]
        evaluator=make_evaluator(cases,memory,rows,3,Budget(fail=1),hash_state(net.local.state_dict()))
        with self.assertRaisesRegex(MemoryError,'resource'):evaluator(net)
        self.assertEqual([module.training for module in net.modules()],modes)
        evaluator=make_evaluator(cases,memory,rows,3,Budget(),hash_state(net.local.state_dict()))
        with patch('tools.local_cnn_interaction_runtime.hash_state',side_effect=RuntimeError('UNIT state hashing failure')):
            with self.assertRaisesRegex(RuntimeError,'hashing'):evaluator(net)
        self.assertEqual([module.training for module in net.modules()],modes)

    def test_incomplete_cli_options_rejected_and_explicit_full_option_set_accepted(self):
        defaults=dict(interaction_scales=None,interaction_update_steps=None,interaction_lr=None,
                      interaction_training_scale=None,interaction_only=False,deep=False)
        valid=dict(interaction_scales=[0.,.25,1.],interaction_update_steps=2,
                   interaction_lr=1e-4,interaction_training_scale=1.,interaction_only=True)
        validate_interaction_options(SimpleNamespace(**(defaults|valid)))
        for field in ('interaction_scales','interaction_update_steps','interaction_lr','interaction_training_scale'):
            with self.subTest(missing=field):
                values=defaults|valid;values[field]=None
                with self.assertRaises(ValueError):validate_interaction_options(SimpleNamespace(**values))
        bad=[dict(interaction_scales=[]),dict(interaction_scales=[1]),dict(interaction_scales=[0,0]),
             dict(interaction_scales=[0,float('nan')]),dict(interaction_update_steps=0),
             dict(interaction_lr=0),dict(interaction_lr=float('inf')),
             dict(interaction_training_scale=0),dict(interaction_training_scale=.5),dict(deep=True)]
        for fields in bad:
            with self.subTest(invalid=fields):
                with self.assertRaises(ValueError):validate_interaction_options(SimpleNamespace(**(defaults|valid|fields)))


if __name__=='__main__':unittest.main()
