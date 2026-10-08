"""Actual-CT CUDA DEBUG: explicit curriculum inputs and stage pause/resume.

All187 signed source identities and the complete10434532-parameter model remain
loaded. Two original training sources and two original held-out sources form an
explicit diagnostic, never a replacement for full-cohort training/evaluation.
Stage validation builds its own real P+63U hierarchy, without slicing129 scores.
"""
from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
import gc
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True,
                   help='Immutable same-arm backup checkpoint with sibling training_identity.json')
    p.add_argument('--inventory', type=Path, required=True)
    p.add_argument('--cache-sources', type=Path, nargs='+', required=True)
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--output', type=Path, required=True, help='New /tmp/*DEBUG* directory')
    p.add_argument('--stage-u-count', type=int, choices=(56, 63), default=63)
    p.add_argument('--full-validation129', action='store_true', default=True,
                   help='Compatibility flag: actual full129 always runs before stage validation')
    return p.parse_args(argv)


@contextmanager
def setup(a):
    from tools.current_gpu import select_record
    selected = select_record(a.gpu)
    import torch
    from tools.resume_comparison_cached import sealed_arguments
    from tools.verify_comparison_cache_debug import read, sha, canonical
    from tools.verify_comparison_empty_context_debug import readonly_inputs
    from hiercp_v1x import bounded_scope, u_bridge_training as engine
    from hiercp_v1x.scope_probe_support import activate_original
    from hiercp_v1x.comparison_geometry_execution import comparison_geometry_execution
    from hiercp_v1x.comparison_data import ComparisonData
    from hiercp_v1x.comparison_preparation import prepared_provider
    from hiercp_v1x.host_memory import PressureBudget, pressure_aware_provider
    from hiercp_v1x.preparation_reuse import preparation_reuse

    a.experiment = a.experiment.resolve(strict=True)
    a.inventory = a.inventory.resolve(strict=True)
    a.cache_sources = list(dict.fromkeys(p.resolve(strict=True) for p in a.cache_sources))
    a.checkpoint = a.checkpoint.resolve(strict=True)
    a.output = a.output.resolve()
    if a.output.exists() or not a.output.is_relative_to(Path('/tmp')) or 'DEBUG' not in a.output.name:
        raise ValueError('Use a new explicitly named /tmp/*DEBUG* output directory')
    manifest, family, _, namespace = sealed_arguments(SimpleNamespace(gpu=a.gpu,
        arm='native_listwise', experiment=a.experiment, inventory=a.inventory,
        debug_fixture=None, debug_config=None, debug_source=None, debug_bank=None))
    if family != 'comparison' or manifest['debug'] or len(manifest['baseline']['source_samples']) != 187:
        raise ValueError('The exact complete187-source production comparison manifest is required')
    baseline = Path(manifest['baseline']['baseline']).resolve(strict=True)
    protected = [a.experiment, baseline, namespace, a.inventory, a.checkpoint.parent, *a.cache_sources]
    if any(a.output.is_relative_to(p) or p.is_relative_to(a.output) for p in protected):
        raise ValueError('DEBUG output must be disjoint from every preserved input')
    a.output.mkdir(parents=True, exist_ok=False)
    a.created_debug_output = True
    with readonly_inputs(protected), comparison_geometry_execution():
        original = Path(manifest['original']['source'])
        if canonical(activate_original(original)) != manifest['original']:
            raise ValueError('Original neural source differs')
        if canonical(bounded_scope.install(10, expected_snapshot_root=original)) != manifest['scope']:
            raise ValueError('Exact original10mm scope differs')
        from hiercp.prototype import PrototypeBank
        from hiercp.tensor import configure_runtime, collect_runtime_resources
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError('Exactly one explicitly selected real CUDA device required')
        config = copy.deepcopy(manifest['config'])
        configure_runtime(deterministic=config['runtime']['deterministic'],
            allow_tf32=config['runtime']['allow_tf32'], cudnn_benchmark=config['runtime']['cudnn_benchmark'])
        torch.set_num_threads(manifest['workers'])
        cuda_bytes = int(manifest['cuda_gib']*2**30)
        capacity = torch.cuda.get_device_properties(0).total_memory
        if not 0 < cuda_bytes < capacity:
            raise ValueError('Original CUDA budget must leave actual device headroom')
        torch.cuda.set_per_process_memory_fraction(cuda_bytes/capacity)
        budget = PressureBudget(cuda_bytes, int(manifest['rss_gib']*2**30),
                                resident_bytes=int(manifest['resident_gib']*2**30))
        bank_path = baseline/'shared/prototype_bank.pt'
        if sha(bank_path) != manifest['baseline']['bank_sha256']:
            raise ValueError('Original prototype bank file changed')
        bank = PrototypeBank.load(bank_path)
        if bank.fingerprint() != manifest['prototype_fingerprint']:
            raise ValueError('Original prototype bank fingerprint changed')
        checkpoint = a.checkpoint
        checkpoint_sha = sha(checkpoint)
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        content_sha = saved.pop('content_sha256', None)
        ownership = read(checkpoint.parent/'training_identity.json')
        if (content_sha != engine.digest(saved) or saved['identity_sha256'] != ownership['identity_sha256']
                or ownership['binding']['arm'] != 'native_listwise'
                or saved.get('comparison_policy', {}).get('arm') != 'native_listwise'):
            raise ValueError('Preserved native_listwise checkpoint content or ownership differs')
        core = pressure_aware_provider(ComparisonData)
        with preparation_reuse(core, a.cache_sources) as reused:
            provider_type = prepared_provider(reused)
            kwargs = dict(source_samples=manifest['baseline']['source_samples'],
                raw_records=read(a.inventory)['raw_records'], config=config, bank=bank,
                root=a.output/'data', workers=manifest['workers'],
                resident_bytes=int(manifest['resident_gib']*2**30), budget=budget,
                regions_dir=baseline/'shared/regions')
            provider = provider_type(**kwargs)
            if canonical(provider.examples('train')+provider.examples('val')) != manifest['samples']:
                raise ValueError('The complete source inventory or original indices changed')
            yield SimpleNamespace(provider=provider, provider_type=provider_type, provider_kwargs=kwargs,
                manifest=manifest, config=config, budget=budget, saved=saved,
                checkpoint=checkpoint, checkpoint_sha=checkpoint_sha,
                resources=collect_runtime_resources('cuda', storage_path=a.output), selected=selected)
        if sha(checkpoint) != checkpoint_sha:
            raise AssertionError('Explicit immutable input checkpoint changed during the independent DEBUG run')


def replay_stage_inputs(context, val_ids, stage_keys, policy):
    from tools.verify_comparison_preparation_debug import fingerprint, compare
    from hiercp_v1x.comparison_curriculum_data import candidate_plan
    from hiercp_v1x import u_bridge_training as engine
    from hiercp_v1x.host_memory import _evict_provider
    epoch = context.config['training']['fixed_validation_epoch']
    fingerprints = {}
    before_rng = engine.digest(engine.capture_rng())
    for index in val_ids:
        with candidate_plan(context.provider, 'native_listwise', policy['policy_sha256'],
                stage_keys, mode='stage_validation', epoch=epoch) as phase:
            batch = phase.batch([index], 'native_listwise', epoch, False, full=False)
            if tuple(batch.counts) != (len(stage_keys),):
                raise AssertionError('Actual stage graph candidate count differs')
            fingerprints[index] = fingerprint(vars(batch))
            del batch
    _evict_provider(context.provider)
    context.provider = context.provider_type(**context.provider_kwargs)
    context.provider._runtime()
    comparisons = {}

    def denied(*args, **kwargs):
        raise AssertionError('Completed curriculum sample-layout replay attempted a cold rebuild')

    with candidate_plan(context.provider, 'native_listwise', policy['policy_sha256'],
            stage_keys, mode='stage_validation', epoch=epoch) as phase:
        for index in val_ids:
            if not phase.cached_batch_ready([index], 'native_listwise', epoch, training=False):
                raise AssertionError('Stage-specific layout was not published')
            with patch.object(context.provider, '_case', denied), patch.object(context.provider, '_source', denied), \
                    patch.object(context.provider, '_upper_context', denied), patch.object(context.provider, '_layout_publish', denied):
                batch = phase.batch([index], 'native_listwise', epoch, False, full=False)
            comparison = compare(fingerprints[index], fingerprint(vars(batch)))
            if not comparison['exact_values_layout_metadata_equal']:
                raise AssertionError('Original/reopened stage graph values or metadata differ')
            comparisons[str(index)] = comparison
            del batch
    if before_rng != engine.digest(engine.capture_rng()):
        raise AssertionError('Stage input preparation changed global model RNG')
    _evict_provider(context.provider)
    gc.collect()
    return dict(candidates_per_source=len(stage_keys), actual_view_epoch=epoch,
                source_indices=val_ids, exact_replay=comparisons, global_rng_unchanged=True)


def stage_pause_resume(context, val_ids, stage_keys, policy, full_report, output):
    import torch
    from hiercp.model import HierarchicalPyGPlacementModel
    from hiercp_v1x import u_bridge_training as engine
    from hiercp_v1x.comparison_checkpoint import ComparisonCheckpointWriter
    from hiercp_v1x.comparison_stage_validation import evaluate_stage_validation
    from hiercp_v1x.comparison_curriculum_data import FULL_KEYS

    saved = context.saved
    full_rows = full_report['rows']
    if (full_report.get('actual_joint129_graph') is not True
            or [row['sample_index'] for row in full_rows] != val_ids
            or any(row['candidate_count'] != 129 or len(row['scores']) != 129
                   or tuple(row['candidate_keys']) != FULL_KEYS for row in full_rows)):
        raise AssertionError('Stage DEBUG requires actual completed full129 scores for these exact held-out sources')
    full_rows_sha = engine.digest(full_rows)
    net = HierarchicalPyGPlacementModel(**context.config['model']).cuda()
    engine._model_contract(net, True)
    net.load_state_dict(saved['model'], strict=True)
    engine.restore_rng(saved['rng'])
    state = dict(phase='validation', epoch=saved['state']['epoch'], updates=saved['state']['updates'],
                 validation_position=len(val_ids), validation_rows=copy.deepcopy(full_rows))
    writer = ComparisonCheckpointWriter()
    paths = []
    score_calls = []
    lookup = {row['index']:row for row in context.provider.examples('val')}
    checkpoint_path = output/'stage_checkpoint_DEBUG.pt'
    output.mkdir(exist_ok=False)
    before = dict(model=engine.digest(net.state_dict()), optimizer=engine.digest(saved['optimizer']),
        scheduler=engine.digest(saved['scheduler']), scaler=engine.digest(saved['scaler']),
        shuffle_generator=engine.digest(saved['shuffle_generator']), rng=engine.digest(engine.capture_rng()))

    def checkpoint():
        payload = dict(format='curriculum_stage_continuation_DEBUG_v1', debug=True,
            model=net.state_dict(), optimizer=saved['optimizer'], scheduler=saved['scheduler'],
            scaler=saved['scaler'], rng=engine.capture_rng(), shuffle_generator=saved['shuffle_generator'],
            state=state, original_training_state=saved['state'], original_identity_sha256=saved['identity_sha256'],
            production_resume_compatible=False, declared_debug_validation_source_indices=val_ids,
            curriculum_policy=policy)
        receipt = writer.save([checkpoint_path],payload,
                              static_generation=('DEBUG_stage',state['epoch'],state['updates']))
        paths.append(dict(position=state.get('stage_validation',{}).get('position',0), **receipt))
        return receipt['total_seconds']

    class CountingNetwork:
        def eval(self):
            net.eval()
            return self

        def score_inference_chunked(self, batch, *, local_chunk_size):
            if tuple(batch.counts) != (len(stage_keys),):
                raise AssertionError('Stage evaluator changed the real single-source candidate graph')
            score_calls.append(list(batch.bridge_indices))
            return net.score_inference_chunked(batch, local_chunk_size=local_chunk_size)

    def evaluate(pause):
        return evaluate_stage_validation(net=CountingNetwork(), provider=context.provider,
            state=state, arm='native_listwise', keys=stage_keys, policy=policy, root=output,
            val_ids=val_ids, lookup=lookup, physical_batch=1, epochs=40,
            view_epoch=context.config['training']['fixed_validation_epoch'],
            chunk=context.manifest['validation_local_chunk'], amp=context.config['training']['amp'],
            budget=context.budget, checkpoint=checkpoint, pause_requested=pause, debug=True)

    paused = evaluate(lambda: state.get('stage_validation',{}).get('position',0) >= 1)
    if paused is not None or state['stage_validation']['position'] != 1 or score_calls != [[val_ids[0]]]:
        raise AssertionError('Stage diagnostic did not pause after exactly the first held-out source')
    reopened = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    checksum = reopened.pop('content_sha256',None)
    if checksum != engine.digest(reopened) or reopened['state']['stage_validation']['position'] != 1:
        raise AssertionError('Stage checkpoint content or saved cursor changed')
    if (reopened['state']['validation_position'] != len(val_ids)
            or engine.digest(reopened['state']['validation_rows']) != full_rows_sha):
        raise AssertionError('Stage checkpoint changed the actual preceding full129 evaluation')
    for name in ('model','optimizer','scheduler','scaler','shuffle_generator','rng'):
        if engine.digest(reopened[name]) != before[name]:
            raise AssertionError('Stage evaluation unexpectedly changed '+name)
    net.load_state_dict(reopened['model'],strict=True)
    state = reopened['state']
    engine.restore_rng(reopened['rng'])
    completed = evaluate(lambda: False)
    if score_calls != [[val_ids[0]],[val_ids[1]]] or state['stage_validation']['position'] != 2:
        raise AssertionError('Resumed stage evaluator repeated or skipped a held-out source')
    completed_again = evaluate(lambda: False)
    if score_calls != [[val_ids[0]],[val_ids[1]]] or engine.digest(completed_again['rows']) != engine.digest(completed['rows']):
        raise AssertionError('Completed stage resume repeated work or changed scores')
    if before['model'] != engine.digest(net.state_dict()) or before['rng'] != engine.digest(engine.capture_rng()):
        raise AssertionError('Stage evaluation did not preserve model/RNG')
    if engine.digest(state['validation_rows']) != full_rows_sha:
        raise AssertionError('Stage resume changed the separate full129 validation rows')
    report = dict(actual_CUDA=True, paused_position=1, resumed_position=2,
        complete_resume_extra_forwards=0, observed_score_calls=score_calls,
        actual_full129_completed_first=True, preceding_full129_rows_sha256=full_rows_sha,
        saved_model_optimizer_scheduler_scaler_shuffle_rng_unchanged=True,
        paused_checkpoint_content_sha256=checksum,
        completed_checkpoint_content_sha256=paths[-1]['content_sha256'],
        checkpoint=str(checkpoint_path), checkpoint_writes=paths,
        stage_report=completed, production_validation_source_count=len(lookup),
        declared_debug_validation_source_count=2, full_cohort_evaluation=False,
        scope='Exactly the declared two held-out sources; stage report full coverage refers only to this DEBUG inventory')
    del net, reopened, writer
    gc.collect()
    torch.cuda.empty_cache()
    return report


def full_validation(context, val_ids):
    import torch
    from hiercp.model import HierarchicalPyGPlacementModel
    from hiercp_v1x import u_bridge_training as engine
    from hiercp_v1x.host_memory import _evict_provider
    from hiercp_v1x.comparison_curriculum_data import FULL_KEYS
    net = HierarchicalPyGPlacementModel(**context.config['model']).cuda()
    engine._model_contract(net, True)
    net.load_state_dict(context.saved['model'],strict=True)
    net.eval()
    rows = []
    rng = engine.capture_rng()
    try:
        for index in val_ids:
            batch = context.provider.batch([index],'native_listwise',
                context.config['training']['fixed_validation_epoch'],False,full=True)
            if tuple(batch.counts) != (129,) or tuple(batch.bridge_candidate_keys[0]) != FULL_KEYS:
                raise AssertionError('Full validation must retain actual P+128 graph')
            with torch.no_grad(),torch.autocast('cuda',enabled=context.config['training']['amp']):
                scores = net.score_inference_chunked(batch,local_chunk_size=context.manifest['validation_local_chunk'])
            rows.append(engine.score_row(scores[0],context.provider._example(index),FULL_KEYS,expected_candidates=129))
            del scores,batch
            context.budget.check()
    finally:
        engine.restore_rng(rng)
    report = dict(engine.aggregate_rows(rows), rows=rows, actual_joint129_graph=True,
                  declared_debug_validation_sources=2, full_cohort_evaluation=False)
    del net
    _evict_provider(context.provider)
    gc.collect()
    torch.cuda.empty_cache()
    return report


def verify(a):
    from tools.verify_comparison_empty_context_debug import write_new
    result = dict(debug=True,status='RUNNING',full_training=False,full_evaluation=False,
        quality_verified=False,production_configuration_changed=False,
        complete_source_inventory=187,debug_training_sources=2,debug_validation_sources=2,
        training_candidates=8,stage_U_count=a.stage_u_count,stage_candidates=a.stage_u_count+1,
        training_physical_batch=2,validation_physical_batch=1,production_physical_batch_unchanged=True,
        full_validation129_requested=a.full_validation129)
    try:
        with setup(a) as context:
            import torch
            from hiercp_v1x import comparison_curriculum as curriculum
            from hiercp_v1x.comparison_curriculum_data import candidate_plan
            from hiercp_v1x.host_memory import _evict_provider
            from tools.verify_comparison_empty_context_debug import INDICES, KEYS, empty_receipt, resumed_update
            from tools.verify_comparison_cache_debug import sha
            from tools.verify_comparison_preparation_debug import fingerprint, compare
            policy = curriculum.policy('native_listwise')
            val = context.provider.examples('val')
            if len(val) < 2:
                raise ValueError('Two distinct genuine held-out sources required for declared DEBUG')
            val_ids = [row['index'] for row in val[:2]]
            if set(val_ids)&set(INDICES):
                raise AssertionError('DEBUG train/validation source leakage')
            stage_keys = ('P',*(f'U:{i}' for i in range(a.stage_u_count)))
            result.update(resources=context.resources,selected_device=context.selected,
                source_checkpoint=str(context.checkpoint),source_checkpoint_sha256=context.checkpoint_sha,
                checkpoint_input_kind='explicit_immutable_same_arm_backup',
                policy=policy,training_indices=list(INDICES),validation_indices=val_ids,
                validation_source_ids=[row['id'] for row in val[:2]],
                production_validation_source_count=len(val),harness_sha256=sha(Path(__file__)))
            write_new(a.output/'request.json',result)
            print('DEBUG actual CUDA full129 evaluation for both held-out sources before stage validation',flush=True)
            result['full129'] = full_validation(context,val_ids)
            write_new(a.output/'full129_DEBUG.json',result['full129'])
            print('DEBUG building actual stage hierarchy and verifying exact reopened layouts',flush=True)
            result['stage_input_replay'] = replay_stage_inputs(context,val_ids,stage_keys,policy)
            write_new(a.output/'stage_input_replay.json',result['stage_input_replay'])
            print('DEBUG actual CUDA stage evaluation: first source, checkpoint, resume second source',flush=True)
            result['stage_pause_resume'] = stage_pause_resume(context,val_ids,stage_keys,policy,
                result['full129'],a.output/'stage_validation_DEBUG')
            write_new(a.output/'stage_resume_receipt.json',result['stage_pause_resume'])
            # These same real U49..55 keys deliberately differ from legacy
            # epoch9 U56..62. The actual view epoch remains9 throughout.
            print('DEBUG actual adaptive keys at view epoch9: P+U49..55, full-model backward/update',flush=True)
            with candidate_plan(context.provider,'native_listwise',policy['policy_sha256'],KEYS,
                    mode='training',epoch=9) as phase:
                batch = phase.batch(INDICES,'native_listwise',9,True,full=False)
                result['training_plan'] = phase.candidate_plan_receipt
                result['training_empty_evidence'] = empty_receipt(batch)
                training_fingerprint = fingerprint(vars(batch))
                if not phase.cached_batch_ready(INDICES,'native_listwise',9,training=True):
                    raise AssertionError('Adaptive train8 publication is not available for replay')
                del batch
                _evict_provider(context.provider)

                def denied(*args, **kwargs):
                    raise AssertionError('Adaptive training layout replay attempted a cold rebuild')

                with patch.object(context.provider,'_case',denied), patch.object(context.provider,'_source',denied), \
                        patch.object(context.provider,'_upper_context',denied), patch.object(context.provider,'_layout_publish',denied):
                    batch = phase.batch(INDICES,'native_listwise',9,True,full=False)
                result['training_input_replay'] = compare(training_fingerprint,fingerprint(vars(batch)))
                if not result['training_input_replay']['exact_values_layout_metadata_equal']:
                    raise AssertionError('Adaptive train8 cache replay changed full graphs, views or candidate order')
            _evict_provider(context.provider)
            gc.collect()
            batch = batch.pin_memory().to('cuda',non_blocking=True)
            torch.cuda.synchronize()
            training_output = a.output/'training_updates_DEBUG'
            training_output.mkdir(exist_ok=False)
            result['updates'] = []
            for arm in ('native','native_listwise'):
                update = resumed_update(batch,context.saved,context.config,context.budget,arm,training_output)
                update['diagnostic_candidate_plan'] = result['training_plan']
                write_new(training_output/(arm+'_update.json'),update)
                result['updates'].append(update)
            result['source_checkpoint_unchanged'] = sha(context.checkpoint)==context.checkpoint_sha
            if not result['source_checkpoint_unchanged']:
                raise AssertionError('Explicit immutable input checkpoint changed')
            result['status'] = 'DEBUG_PASS'
        write_new(a.output/'result.json',result)
        print('DEBUG_PASS '+str(a.output),flush=True)
    except Exception as error:
        result.update(status='DEBUG_FAILED',error_type=type(error).__name__,error=str(error),
                      traceback=traceback.format_exc())
        if getattr(a,'created_debug_output',False) and a.output.is_dir() and not (a.output/'failure.json').exists():
            write_new(a.output/'failure.json',result)
        raise
    return result


if __name__ == '__main__':
    verify(parse())
