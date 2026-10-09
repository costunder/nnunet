"""UNIT checks for all-P objectives, real differentiation, and candidate policy.

Small tensors here validate mathematics and state transitions only. They are
not a smoke test of real CT, the full original model, or completed training.
"""
from types import SimpleNamespace

import copy
import inspect
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from torch.nn import functional as F

from hiercp_v1x.v23_training import (
    CONSISTENCY_FIELDS, FIELDS, FORMAT, V23Scorer, _Distributed, _validate_progress,
    advance_target_state, apply_patient_order, bank_u_scores, execution_identity,
    handoff_single_gpu_checkpoint, new_target_state, publish_epoch_curve, remaining_training_updates,
    training_binding, validate_target_state,
    aggregate_patients, best_selection_key, finish_curriculum_epoch,
    new_curriculum, parallel_patient_batches, patient_balanced_objective,
    rank_cases, score_patient,
)
from hiercp_v1x.u_bridge_training import digest
from hiercp_v1x.contracts import canonical_hash


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


class _TargetUnitPopulation:
    """Metadata-only UNIT population, never a clinical production input."""
    def __init__(self):
        self.train = [f'train{i:02d}' for i in range(65)]
        self.val = [f'val{i:02d}' for i in range(21)]
    def partition_cases(self, partition, ranking_only=False):
        return self.train if partition == 'inner_train' else self.val
    def manifest(self):
        return {'sha256':'1'*64, 'UNIT':True, 'DEBUG':True}
    def case(self, case, active_u_count=128, active_u_indices=None):
        indices = list(range(active_u_count)) if active_u_indices is None else sorted(active_u_indices)
        assert case in self.train+self.val
        assert len(indices) == active_u_count and len(set(indices)) == active_u_count
        return SimpleNamespace(case_id=case, active_u_count=active_u_count,
            active_u_indices=tuple(indices), positive_indices=(0,),
            record_ids=tuple([case+'P']+[case+str(i) for i in indices]),
            unobserved_indices=tuple(range(1,active_u_count+1)),
            unobserved_bank_positions=tuple(indices))


def test_singleton_cli_and_calibration_are_bound_to_one_authorized_gpu():
    from tools.run_v23_all_p import parse, validate_single_gpu_calibration
    base = ['--native-experiment','UNIT','--inventory','UNIT','--prepared-cache','UNIT',
            '--full-validation-upper-cache','UNIT','--output','UNIT','--gpus']
    for gpu in (1,5,6):
        assert parse(base+[str(gpu)]).gpus == [gpu]
    for gpus in (['0'],['2'],['1','5','6'],['1','1']):
        with _raises(ValueError, match='exactly one'):
            parse(base+gpus)
    request = dict(request_sha256='a'*64,gpus=[5])
    calibration = dict(world_size=1, request_sha256='a'*64, physical_GPUs=[5], debug=False,
        measured_full_P_U128_backward=True, original_model_and_RNG_preserved=True,
        initial_state_sha256='b'*64, GPU_reports=[dict(physical_GPU=5,initial_state_sha256='b'*64)])
    validate_single_gpu_calibration(calibration,request)
    for key,value in [('world_size',3),('physical_GPUs',[1]),('request_sha256','c'*64)]:
        bad=copy.deepcopy(calibration);bad[key]=value
        with _raises(ValueError):validate_single_gpu_calibration(bad,request)


def test_world_one_never_calls_collectives_or_constructs_ddp():
    calls=[]
    with patch('torch.distributed.is_initialized',return_value=False), \
         patch('torch.distributed.all_gather_object',side_effect=AssertionError('collective')), \
         patch('torch.distributed.broadcast_object_list',side_effect=AssertionError('collective')), \
         patch('torch.distributed.barrier',side_effect=AssertionError('collective')):
        execution = _Distributed()
        assert execution.world == 1 and execution.rank == 0
        assert execution.rows([{'case_id':'UNIT'}]) == [{'case_id':'UNIT'}]
        assert execution.root_action(lambda: 7) == 7
        geometry=SimpleNamespace(prepare=lambda count,**kw:calls.append(('prepare',count,kw)),
                                 admit=lambda count,**kw:calls.append(('admit',count,kw)))
        execution.prepare_geometry(geometry,7,{'UNIT':list(range(7))})
    assert [call[0] for call in calls] == ['prepare','admit']


def test_U_only_mining_maps_actual_full_scores_to_bank_and_freezes_cumulative_membership():
    population=_TargetUnitPopulation();plan=population.case(population.train[0],128)
    score=torch.arange(129,dtype=torch.float64);score[0]=-99999  # P score must be absent from selection.
    u=bank_u_scores(score,plan)
    assert u == list(map(float,range(1,129)))
    state=new_target_state(population,'hard_score_top',7)
    cases=population.train+population.val
    mined=[dict(case_id=case,u_scores=u,model_sha256='a'*64) for case in cases]
    expanded,receipt=advance_target_state(population,state,14,epoch=2,model_sha256='a'*64,mining_rows=mined)
    assert expanded['selections'][cases[0]] == list(range(7))+list(range(121,128))
    assert state['active_u'] == 7 and expanded['active_u'] == 14
    assert receipt['U_scores_only'] and receipt['no_grad']
    assert validate_target_state(population,expanded,policy='hard_score_top',active_u=14) == expanded
    bad=copy.deepcopy(expanded);bad['selections'][cases[0]][0]=20
    with _raises(ValueError):validate_target_state(population,bad,policy='hard_score_top',active_u=14)
    with _raises(ValueError,match='coverage'):
        advance_target_state(population,state,14,epoch=2,model_sha256='a'*64,mining_rows=mined[:-1])


def test_explicit_cold_start_uses_seeded_order_then_requires_real_previous_train_losses():
    population=_TargetUnitPopulation();random_order=population.train[::-1]
    state=dict(epoch=1,updates=0,train_position=0,history=[])
    first=apply_patient_order(population,state,'hard_score_top',random_order)
    assert first['order'] == random_order
    assert first['receipt']['bootstrap_reason'] == 'no completed TRAIN epoch yet'
    assert first['receipt']['fake_losses_used'] is False
    state['epoch']=2
    with _raises(ValueError,match='Missing preceding'):
        apply_patient_order(population,state,'hard_score_top',random_order)
    prior=[dict(case_id=case,observed_P=1,per_P=dict(pair_loss=float(i+1))) for i,case in enumerate(population.train)]
    state['history']=[dict(train=dict(cases=prior))]
    second=apply_patient_order(population,state,'hard_score_top',random_order)
    assert second['order'] == population.train[::-1]
    mix=apply_patient_order(population,state,'score_stratified_mix',random_order)
    assert set(mix['order']) == set(population.train) and len(mix['order']) == 65
    assert mix['order'] != second['order']


def _handoff_unit_fixture(directory):
    population=_TargetUnitPopulation()
    runtime=dict(debug=True,initial_u=7,increment_u=7,total_u=128,workers=12)
    config=dict(model={'UNIT':True},graph={'UNIT':True},training={'UNIT':True},
                ct_clip=[-200,300],runtime={'UNIT':True},v23_runtime=runtime)
    old_request=dict(format='UNIT',source={'unit_source':'2'*64},gpus=[1,5,6])
    old_request['request_sha256']=canonical_hash(old_request)
    old_binding=training_binding(config,population,identity=execution_identity(old_request),
        physical_patient_batch=4,physical_candidate_batch=32,world_size=3,workers=12,
        initial_hash='3'*64,debug=True)
    rows=[dict(case_id=case,observed_P=1,per_P=dict(pair_loss=float(i+1))) for i,case in enumerate(population.train)]
    history=[dict(train=dict(cases=rows)) for _ in range(7)]
    curriculum=_policy();curriculum['last_completed_epoch']=7
    best=dict(epoch=1,updates=6,selection_key=[.4,.2,-.5],selected_by='full_validation_only')
    state=dict(epoch=8,phase='training',train_position=0,train_order=population.train,
        updates=42,attempts=42,overflows=0,history=history,best=best,curriculum=curriculum,
        train_rows=[],evaluation_position=0,evaluation_rows=[],epoch_start_updates=42,
        connected=['UNIT'],initial_validation=None,status='RUNNING')
    saved=dict(format=FORMAT,identity_sha256=digest(old_binding),model={'UNIT':torch.tensor([2.,3.])},
        optimizer={'state':{0:{'step':torch.tensor(42.),'exp_avg':torch.tensor([.2,.3])}}},
        scheduler={'last_epoch':7},scaler={'scale':128.},state=state,
        rank_rng=[{'UNIT_rank':i,'torch':torch.tensor([i],dtype=torch.uint8)} for i in range(3)],
        shuffle_generator=torch.Generator().manual_seed(42).get_state())
    saved['content_sha256']=digest(saved)
    checkpoint=directory/'old_latest.pt';torch.save(saved,checkpoint)
    ownership=directory/'old_identity.json'
    ownership.write_text(json.dumps(dict(identity_sha256=digest(old_binding),binding=old_binding)),'utf8')
    actual_best=copy.deepcopy(saved);actual_best.pop('content_sha256')
    actual_best['model']['UNIT']=torch.tensor([1.,1.])
    actual_best['optimizer']['state'][0]['step']=torch.tensor(6.)
    actual_best['scheduler']['last_epoch']=1
    actual_best['state'].update(epoch=2,history=history[:1],updates=6,train_order=None)
    actual_best['state']['curriculum']['last_completed_epoch']=1
    actual_best['content_sha256']=digest(actual_best)
    best_path=directory/'old_best.pt';torch.save(actual_best,best_path)
    from hiercp_v1x.v23_training import _file_sha256
    proof=dict(source_commit='a'*40,source_request=old_request,
        source_files_sha256=old_request['source'],source_checkpoint_file_sha256=_file_sha256(checkpoint),
        source_identity_file_sha256=_file_sha256(ownership),geometry_receipt_sha256='4'*64,
        calibration_receipt_sha256='5'*64,core_equations_preserved=True,original_model_parameters=2,
        source_best_checkpoint_file_sha256=_file_sha256(best_path))
    destination_request=dict(format='UNIT',source={'unit_source':'6'*64},gpus=[5])
    destination_request['request_sha256']=canonical_hash(destination_request)
    calibration=dict(world_size=1,request_sha256=destination_request['request_sha256'],physical_GPUs=[5],
        measured_full_P_U128_backward=True,original_model_and_RNG_preserved=True,
        initial_state_sha256='3'*64,selected_physical_patient_batch=4,selected_physical_candidate_batch=32)
    config=copy.deepcopy(config);config['v23_runtime']['target_selection_policy']='hard_score_top'
    return population,checkpoint,ownership,best_path,proof,destination_request,calibration,config,saved,actual_best


def test_verified_handoff_preserves_latest_and_actual_best_and_continues_remaining_epochs():
    with tempfile.TemporaryDirectory(prefix='v23_handoff_UNIT_',dir=Path.cwd()) as name:
        directory=Path(name)
        population,checkpoint,ownership,best_path,proof,request,calibration,config,source,best=_handoff_unit_fixture(directory)
        output=directory/'new_training'
        receipt=handoff_single_gpu_checkpoint(checkpoint,ownership,request,config,calibration,output,
            source_proof=proof,population=population,source_best_checkpoint=best_path,debug=True)
        fork=torch.load(output/'checkpoint_latest.pt',weights_only=False)
        fork_best=torch.load(output/'checkpoint_best.pt',weights_only=False)
        for key in ('model','optimizer','scheduler','scaler','shuffle_generator'):
            assert digest(fork[key]) == digest(source[key])
            assert digest(fork_best[key]) == digest(best[key])
        assert digest(fork['rank_rng'][0]) == digest(source['rank_rng'][0]) and len(fork['rank_rng']) == 1
        assert fork['state']['history'] == source['state']['history'] and fork['state']['updates'] == 42
        assert fork['state']['train_order'] == population.train[::-1]
        assert source['state']['train_order'] == population.train  # Source was not mutated.
        assert receipt['remaining_epochs'] == 33 and receipt['total_target_epochs'] == 40
        assert receipt['inherited_best_checkpoint']['latest_weights_substituted'] is False
        assert digest(fork_best['model']) != digest(fork['model'])
        assert remaining_training_updates(fork['state'],population.train,4,1) == 561
        assert fork['state']['updates']+remaining_training_updates(fork['state'],population.train,4,1) == 603
        with _raises(FileExistsError):
            handoff_single_gpu_checkpoint(checkpoint,ownership,request,config,calibration,output,
                source_proof=proof,population=population,source_best_checkpoint=best_path,debug=True)


def test_handoff_rejects_changed_core_or_missing_best_and_completed_training():
    with tempfile.TemporaryDirectory(prefix='v23_handoff_reject_UNIT_',dir=Path.cwd()) as name:
        directory=Path(name)
        population,checkpoint,ownership,best_path,proof,request,calibration,config,source,best=_handoff_unit_fixture(directory)
        bad=copy.deepcopy(config);bad['training']['UNIT']=False
        with _raises(ValueError,match='setting changed'):
            handoff_single_gpu_checkpoint(checkpoint,ownership,request,bad,calibration,directory/'badcore',
                source_proof=proof,population=population,source_best_checkpoint=best_path,debug=True)


        with _raises(ValueError,match='BEST checkpoint'):
            handoff_single_gpu_checkpoint(checkpoint,ownership,request,config,calibration,directory/'nobest',
                source_proof=proof,population=population,debug=True)
        from hiercp_v1x.v23_training import _file_sha256
        source['state']['phase']='complete';source['state']['epoch']=41
        source.pop('content_sha256');source['content_sha256']=digest(source);torch.save(source,checkpoint)
        proof['source_checkpoint_file_sha256']=_file_sha256(checkpoint)
        with _raises(ValueError,match='additional epochs'):
            handoff_single_gpu_checkpoint(checkpoint,ownership,request,config,calibration,directory/'complete',
                source_proof=proof,population=population,source_best_checkpoint=best_path,debug=True)


def test_completed_epoch_curve_publication_is_exactly_once_after_resume():
    with tempfile.TemporaryDirectory(prefix='v23_curve_UNIT_',dir=Path.cwd()) as name:
        path=Path(name)/'curve.jsonl';curve=dict(epoch=8,update=59,UNIT=True,metric=.5)
        publish_epoch_curve(path,curve);publish_epoch_curve(path,copy.deepcopy(curve))
        assert len(path.read_text('utf8').splitlines()) == 1
        with _raises(ValueError,match='differs'):
            publish_epoch_curve(path,dict(curve,metric=.4))


def load_tests(loader, tests, pattern):
    """Run standard-library UNIT functions, including every explicit case."""
    parameter_sets = {
        'test_undefined_or_malformed_positive_ownership_is_explicit_error':
            [((),), ((0, 0),), ((3,),), ((0, 1, 2),)],
        'test_gate_requires_real_optimizer_history_and_exact_patient_coverage':
            [(['a'], 1), (['a', 'b', 'b'], 1), (['a', 'b'], 0)],
        'test_parallel_batches_have_no_duplicated_dropped_or_empty_rank_patients':
            [(65, 2, 3), (21, 2, 3), (61, 2, 3), (17, 4, 3), (5, 2, 1), (65,4,1), (21,4,1)],
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
