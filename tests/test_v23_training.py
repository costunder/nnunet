"""UNIT checks for all-P objectives, real differentiation, and candidate policy.

Small tensors here validate mathematics and state transitions only. They are
not a smoke test of real CT, the full original model, or completed training.
"""
from types import SimpleNamespace

import copy
import inspect
import unittest
from unittest.mock import patch
import torch
from torch.nn import functional as F

from hiercp_v1x.v23_training import (
    CONSISTENCY_FIELDS, FIELDS, V23Scorer, _validate_progress,
    aggregate_patients, best_selection_key, finish_curriculum_epoch,
    new_curriculum, parallel_patient_batches, patient_balanced_objective,
    rank_cases, score_patient,
)


def _raises(error, match=None):
    case = unittest.TestCase()
    return case.assertRaisesRegex(error, match) if match else case.assertRaises(error)


def test_patient_balance_matches_mean_u_then_p_then_patient_and_gradient():
    first = torch.tensor([2., 0., 1., -1.], requires_grad=True)
    second = torch.tensor([-.5, 1., 3., 0., -.2], requires_grad=True)
    consistency = torch.tensor([.1, .2, .3, .4, .5, .6, .7, .8, .9], requires_grad=True)
    positives = [(0, 2), (1,)]
    loss, terms = patient_balanced_objective((first, second), positives, consistency)
    reference = ((F.softplus(first[torch.tensor([1, 3])][None] - first[torch.tensor([0, 2]), None]).mean(1).mean()
        + F.softplus(second[torch.tensor([0, 2, 3, 4])] - second[1]).mean()) / 2
        + .1 * (consistency[:4].mean() + consistency[4:].mean()) / 2)
    torch.testing.assert_close(loss, reference)
    actual_grad = torch.autograd.grad(loss, (first, second, consistency), retain_graph=True)
    reference_grad = torch.autograd.grad(reference, (first, second, consistency))
    for actual, expected in zip(actual_grad, reference_grad):
        torch.testing.assert_close(actual, expected)
    assert terms['pairs'] == 8
    assert terms['observed_P'] == 3
    unittest.TestCase().assertAlmostEqual(float(actual_grad[2][:4].sum()), .05, places=7)
    unittest.TestCase().assertAlmostEqual(float(actual_grad[2][4:].sum()), .05, places=7)


def test_every_positive_receives_push_up_and_only_u_receives_push_down():
    scores = torch.tensor([.2, -.3, .4, .1], requires_grad=True)
    loss, terms = patient_balanced_objective((scores,), [(0, 2)], torch.zeros(4))
    loss.backward()
    assert bool((scores.grad[torch.tensor([0, 2])] < 0).all())
    assert bool((scores.grad[torch.tensor([1, 3])] > 0).all())
    assert terms['pairs'] == 4  # P x U; neither P appears as a negative.


def test_unequal_rank_tail_weight_matches_global_patient_mean():
    weight = torch.tensor(.5, requires_grad=True)
    patients = [torch.stack([weight * factor, weight * -.2]) for factor in (1., 2., 3.)]
    global_loss, _ = patient_balanced_objective(patients, [(0,)] * 3, torch.zeros(6))
    left, _ = patient_balanced_objective(patients[:2], [(0,)] * 2, torch.zeros(4))
    right, _ = patient_balanced_objective(patients[2:], [(0,)], torch.zeros(2))
    simulated_ddp_mean = (left * (2 * 2 / 3) + right * (1 * 2 / 3)) / 2
    torch.testing.assert_close(global_loss, simulated_ddp_mean)
    first_grad = torch.autograd.grad(global_loss, weight, retain_graph=True)[0]
    second_grad = torch.autograd.grad(simulated_ddp_mean, weight)[0]
    torch.testing.assert_close(first_grad, second_grad)


def test_undefined_or_malformed_positive_ownership_is_explicit_error(positives):
    with _raises(ValueError):
        patient_balanced_objective((torch.ones(3),), [positives], torch.zeros(3))


def test_consistency_requires_per_candidate_actual_values():
    with _raises(ValueError, match='per candidate'):
        patient_balanced_objective((torch.ones(3),), [(0,)], torch.tensor(0.))
    with _raises(ValueError, match='weight0.1'):
        patient_balanced_objective((torch.ones(3),), [(0,)], torch.zeros(3), consistency_weight=.2)


def _policy():
    return new_curriculum(initial_u=7, increment_u=7, total_u=128,
        gate_mrr=.8, gate_top1=.7, minimum_stage_epochs=2, consecutive_passes=2)


def _stage(epoch, active=7, mrr=.85, top1=.75):
    return dict(epoch=epoch, active_u=active,
        metrics=dict(per_P_patient_mrr=mrr, per_P_patient_top1=top1))


def _finish(state, epoch, **kwargs):
    return finish_curriculum_epoch(state, _stage(epoch, active=state['active_u'], **kwargs),
        epoch=epoch, successful_train_cases=['a', 'b'], expected_train_cases=['a', 'b'], updates_in_epoch=1)


def test_expansion_needs_two_complete_consecutive_heldout_passes_retains_previous_u():
    state = _policy()
    state, first = _finish(state, 1)
    assert state['active_u'] == 7 and not first['expanded']
    state, second = _finish(state, 2)
    assert state['active_u'] == 14 and second['expanded']
    assert second['previous_active_u'] == 7
    assert state['all_P_active_from_start'] is True
    assert state['pass_streak'] == 0
    assert state['stage_completed_epochs'] == 0


def test_failed_stage_pass_resets_streak_and_initial_full_score_cannot_expand():
    state, _ = _finish(_policy(), 1)
    state, _ = _finish(state, 2, mrr=.2)
    state, report = _finish(state, 3)
    assert state['active_u'] == 7 and not report['expanded']
    state, report = _finish(state, 4)
    assert state['active_u'] == 14 and report['expanded']
    with _raises(ValueError):
        finish_curriculum_epoch(_policy(), _stage(0, active=128), epoch=0,
            successful_train_cases=['a', 'b'], expected_train_cases=['a', 'b'], updates_in_epoch=0)


def test_gate_requires_real_optimizer_history_and_exact_patient_coverage(successful, updates):
    with _raises(ValueError, match='coverage'):
        finish_curriculum_epoch(_policy(), _stage(1), epoch=1,
            successful_train_cases=successful, expected_train_cases=['a', 'b'], updates_in_epoch=updates)


def test_mismatched_stage_graph_count_rejected():
    with _raises(ValueError, match='another actual candidate'):
        finish_curriculum_epoch(_policy(), _stage(1, active=128), epoch=1,
            successful_train_cases=['a', 'b'], expected_train_cases=['a', 'b'], updates_in_epoch=1)


def test_parallel_batches_have_no_duplicated_dropped_or_empty_rank_patients(count, physical, world):
    cases = [str(index) for index in range(count)]
    batches = parallel_patient_batches(cases, physical, world)
    assert [case for batch in batches for case in batch] == cases
    for batch in batches:
        local = [rank_cases(batch, world, rank) for rank in range(world)]
        assert [case for part in local for case in part] == batch
        assert all(1 <= len(part) <= physical for part in local)


def test_best_uses_full_per_p_score_not_first_positive_hiding_missing_p():
    first = dict(metrics=dict(per_P_patient_mrr=.4, per_P_patient_top1=.2,
        patient_balanced_pair_loss=.2, case_first_P_mrr=1., case_hit_at_1=1.))
    second = dict(metrics=dict(per_P_patient_mrr=.6, per_P_patient_top1=.3,
        patient_balanced_pair_loss=.3, case_first_P_mrr=.2, case_hit_at_1=0.))
    assert best_selection_key(second) > best_selection_key(first)


def _plan(positive_indices=(0, 2), case='a'):
    rows = [dict(id=f'{case}{index}', case_id=case, donor_case_id='donor', donor_component=1,
        center=[index, 0, 0], geometry_sha256=str(index).zfill(64)) for index in range(4)]
    return SimpleNamespace(record_ids=tuple(row['id'] for row in rows), rows=rows,
        positive_indices=positive_indices, unobserved_indices=tuple(index for index in range(4) if index not in positive_indices), case_id=case)


def test_per_p_metrics_expose_unlearned_p_and_never_compare_one_p_to_another():
    row = score_patient(torch.tensor([3., 2., 0., 1.]), _plan())
    assert row['first_P_rank'] == 1
    assert row['per_P']['ranks'] == [1, 3]
    unittest.TestCase().assertAlmostEqual(row['per_P']['mrr'], (1. + 1./3) / 2)
    assert row['per_P']['top1'] == .5
    assert row['observed_P'] == 2 and row['unobserved_U'] == 2


def test_zero_p_validation_patients_scored_and_only_undefined_rank_denominator_excluded():
    known = score_patient(torch.tensor([3., 2., 0., 1.]), _plan(case='p'))
    unobserved = score_patient(torch.tensor([.2, .1, -.2, -.3]), _plan((), 'zero'))
    assert len(unobserved['scores']) == 4 and unobserved['per_P'] is None
    report = aggregate_patients([known, unobserved])
    assert report['denominators']['patients'] == 2
    assert report['denominators']['ranking_patients'] == 1
    assert report['denominators']['zero_P_patients'] == 1
    assert report['metrics']['per_P_patient_mrr'] == known['per_P']['mrr']


class _UnitLocal(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dense = torch.nn.Parameter(torch.tensor(.8))
        self.semantic = torch.nn.ParameterDict({key: torch.nn.Parameter(torch.tensor(.4 + index*.01))
                                                for index, key in enumerate(FIELDS)})

    def encode_dense_maps(self, source, source_index, target):
        return source * self.dense, target * self.dense

    def forward_graph(self, graph, source, target):
        base = (source.flatten(1).mean(1) + target.flatten(1).mean(1))[:, None]
        pattern = torch.arange(1, 129, device=base.device).float()[None]
        return {key: base * self.semantic[key] + pattern * .001 for key in FIELDS}


class _UnitNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.local_encoder = _UnitLocal()


class _UnitLocalBatch:
    def __init__(self):
        self.graph = SimpleNamespace(num_graphs=4)
        self.source_patches = torch.ones(2, 5, 1, 1, 1)
        self.target_patches = torch.ones(2, 5, 1, 1, 1) * 2
        self.source_index = torch.tensor([0, 1])
        self.graph_observation_index = torch.tensor([0, 0, 1, 1])

    def __len__(self): return 2
    def to(self, device):
        result = copy.copy(self)
        for key in ('source_patches', 'target_patches', 'source_index', 'graph_observation_index'):
            setattr(result, key, getattr(self, key).to(device))
        return result


def _unit_scorer(net, checkpoint_chunks):
    rows = [dict(id='a')]
    providers = {partition: SimpleNamespace(ds=SimpleNamespace(rows=rows)) for partition in ('inner_train', 'inner_val')}
    return V23Scorer(net, providers, lambda *args: None, physical_candidate_batch=32,
        checkpoint_local_chunks=checkpoint_chunks, amp=False, budget=lambda: None)


def test_all_twelve_original_fields_and_dense_semantic_gradients_survive_checkpointing():
    net = _UnitNet()
    optimizer = torch.optim.SGD(net.parameters(), lr=.1)
    before = {key: parameter.detach().clone() for key, parameter in net.named_parameters()}
    scorer = _unit_scorer(net, True)
    fields, consistency = scorer._encode(_UnitLocalBatch(), training=True)
    assert fields.shape == (2, len(FIELDS)*128)
    assert consistency.shape == (2,)
    (fields.square().mean() + .1*consistency.mean()).backward()
    assert all(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()) for parameter in net.parameters())
    optimizer.step()
    assert all(not torch.equal(parameter, before[key]) for key, parameter in net.named_parameters())


def test_consistency_matches_original_six_key_cosine_mean():
    net = _UnitNet(); scorer = _unit_scorer(net, False); batch = _UnitLocalBatch()
    packed, consistency = scorer._encode(batch, training=False)
    source, target = net.local_encoder.encode_dense_maps(batch.source_patches, batch.source_index, batch.target_patches)
    fields = net.local_encoder.forward_graph(batch.graph, source[batch.graph_observation_index], target[batch.graph_observation_index])
    reference = torch.stack([1. - F.cosine_similarity(fields[key].reshape(2,2,128)[:,0],
        fields[key].reshape(2,2,128)[:,1], dim=-1) for key in CONSISTENCY_FIELDS]).mean(0)
    torch.testing.assert_close(consistency, reference)
    assert packed.shape[-1] == 1536


def test_checkpoint_receives_existing_real_parameter_for_device_and_rng_capture():
    net = _UnitNet(); scorer = _unit_scorer(net, True)
    parameter = next(net.parameters())
    from torch.utils.checkpoint import checkpoint
    with patch('hiercp_v1x.v23_training.activation_checkpoint', wraps=checkpoint) as recorded:
        fields, consistency = scorer._encode(_UnitLocalBatch(), training=True)
        (fields.square().mean() + consistency.mean()).backward()
    args, kwargs = recorded.call_args
    assert len(args) == 2
    assert args[1] is parameter
    assert kwargs['use_reentrant'] is False
    assert kwargs['preserve_rng_state'] is True
    assert len(list(net.parameters())) == 13  # Existing dense + twelve semantics; no new parameter.


class _CudaUnitLocal(_UnitLocal):
    """CUDA UNIT dropout fixture; no clinical data or model quality evidence."""
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(.4)

    def forward_graph(self, graph, source, target):
        return {key: self.dropout(value) for key, value in super().forward_graph(graph, source, target).items()}


def test_cuda_unit_checkpoint_preserves_dropout_forward_every_gradient_and_next_rng():
    if not torch.cuda.is_available():
        raise unittest.SkipTest('CUDA UNIT requires an actual CUDA device; CPU tests do not claim CUDA verification')
    # This small analytic fixture tests device RNG semantics only. It is not an
    # original-model/CT smoke test, calibration, or final training result.
    net = _UnitNet(); net.local_encoder = _CudaUnitLocal()
    net = net.to('cuda').train()
    direct, checked = copy.deepcopy(net), copy.deepcopy(net)
    cpu_rng = torch.random.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state().clone()
    results = []
    try:
        for model, checkpoint_chunks in ((direct, False), (checked, True)):
            torch.random.set_rng_state(cpu_rng); torch.cuda.set_rng_state(cuda_rng)
            scorer = _unit_scorer(model, checkpoint_chunks)
            fields, consistency = scorer._encode(_UnitLocalBatch(), training=True)
            (fields.square().mean() + .1*consistency.mean()).backward()
            gradients = {name: parameter.grad.detach().clone() for name, parameter in model.named_parameters()}
            results.append((fields.detach().clone(), consistency.detach().clone(), gradients,
                            torch.rand(32, device='cuda'), torch.cuda.get_rng_state().clone()))
        first, second = results
        torch.testing.assert_close(first[0], second[0], rtol=0, atol=0)
        torch.testing.assert_close(first[1], second[1], rtol=0, atol=0)
        assert set(first[2]) == set(second[2])
        for name in first[2]:
            torch.testing.assert_close(first[2][name], second[2][name], rtol=0, atol=0)
        torch.testing.assert_close(first[3], second[3], rtol=0, atol=0)
        assert torch.equal(first[4], second[4])
    finally:
        torch.random.set_rng_state(cpu_rng); torch.cuda.set_rng_state(cuda_rng)


def test_resume_cursor_must_align_with_rebalanced_distributed_batches():
    cases = [str(index) for index in range(61)]
    state = dict(phase='training', epoch=1, history=[], train_order=cases,
        train_position=54, train_rows=[dict(case_id=case) for case in cases[:54]],
        evaluation_position=0, evaluation_rows=[], curriculum=dict(last_completed_epoch=0))
    _validate_progress(state, cases, [str(index) for index in range(21)], 2, 40, SimpleNamespace(last_epoch=0), 3)
    state['train_position'] = 60
    state['train_rows'] = [dict(case_id=case) for case in cases[:60]]
    with _raises(ValueError, match='physical patient batch'):
        _validate_progress(state, cases, [str(index) for index in range(21)], 2, 40, SimpleNamespace(last_epoch=0), 3)


def load_tests(loader, tests, pattern):
    """Run standard-library UNIT functions, including every explicit case."""
    parameter_sets = {
        'test_undefined_or_malformed_positive_ownership_is_explicit_error':
            [((),), ((0, 0),), ((3,),), ((0, 1, 2),)],
        'test_gate_requires_real_optimizer_history_and_exact_patient_coverage':
            [(['a'], 1), (['a', 'b', 'b'], 1), (['a', 'b'], 0)],
        'test_parallel_batches_have_no_duplicated_dropped_or_empty_rank_patients':
            [(65, 2, 3), (21, 2, 3), (61, 2, 3), (17, 4, 3), (5, 2, 1)],
    }
    suite = unittest.TestSuite()
    for name, function in sorted(globals().items()):
        if name.startswith('test_') and inspect.isfunction(function):
            for index, arguments in enumerate(parameter_sets.get(name, [()])):
                def run(function=function, arguments=arguments):
                    function(*arguments)
                run.__name__ = name + ('_' + str(index) if name in parameter_sets else '')
                suite.addTest(unittest.FunctionTestCase(run))
    return suite


if __name__ == '__main__':
    unittest.main()
