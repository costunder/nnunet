"""Archived40-epoch v1 loop with one independently bound CNN L0 replacement."""
from __future__ import annotations

import argparse
import functools
import importlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def install_identity(receipt, scope):
    import torch
    from hiercp import model, contracts
    from .half_a_model import install_half_a, model_contract, MARKER_KEY, require_half_a_state, half_a_spec
    from .half_a_training import MARKER
    constructor = model.HierarchicalPyGPlacementModel.__init__
    load = model.HierarchicalPyGPlacementModel.load_state_dict
    identity = receipt['contract_sha256']
    scope_marker = torch.tensor(list(bytes.fromhex(scope['contract_sha256'])), dtype=torch.uint8)
    marker = torch.tensor(list(bytes.fromhex(identity)), dtype=torch.uint8)
    model.HierarchicalPyGPlacementModel.architecture_version += '|half_A_' + identity
    expected_architecture = model.HierarchicalPyGPlacementModel.architecture_version

    @functools.wraps(constructor)
    def initialized(self, *args, **kwargs):
        constructor(self, *args, **kwargs)
        install_half_a(self)
        self.architecture_version = expected_architecture
        self.register_buffer('v1x_bounded_scope_digest', scope_marker.clone())
        self.register_buffer(MARKER, marker.clone())
        # Computed only once, before calibration/training: unchanged upper
        # weights initialized by the original constructor and its seed.
        from .scope_probe_support import state_digest
        upper = {name:value for name,value in self.state_dict().items()
                 if not name.startswith('local_encoder.') and name not in (MARKER, MARKER_KEY, 'v1x_bounded_scope_digest')}
        print(json.dumps(dict(stage='half_A_initialized', architecture=expected_architecture,
            model_contract=model_contract(), common_upper_initial_sha256=state_digest(upper),
            total_parameters=sum(p.numel() for p in self.parameters()),
            trainable_parameters=sum(p.numel() for p in self.parameters() if p.requires_grad),
            trained_baseline_weights_loaded=False, fresh_optimizer=True), allow_nan=False), flush=True)

    def check_state(state):
        if not isinstance(state, dict):
            raise ValueError('Half-A actual checkpoint state must be a dictionary')
        require_half_a_state(state)
        if state.get('local_encoder._extra_state') != dict(half_a_spec(), bounded_scope_contract=scope['contract_sha256']):
            raise ValueError('Half-A checkpoint local recipe is different')
        for name, expected in ((MARKER, marker), ('v1x_bounded_scope_digest', scope_marker)):
            actual = state.get(name) if isinstance(state, dict) else None
            if (not isinstance(actual, torch.Tensor) or actual.dtype != torch.uint8
                    or actual.shape != expected.shape or not torch.equal(actual.detach().cpu(), expected)):
                raise ValueError('Actual weights belong to another half/scope/experiment')

    @functools.wraps(load)
    def loaded(self, state, *args, **kwargs):
        check_state(state)
        return load(self, state, *args, **kwargs)

    model.HierarchicalPyGPlacementModel.__init__ = initialized
    model.HierarchicalPyGPlacementModel.load_state_dict = loaded
    original_require = contracts.require_current_checkpoint
    original_version = contracts.ARCHITECTURE_VERSION
    @functools.wraps(original_require)
    def required(payload):
        if not isinstance(payload, dict) or payload.get('architecture_version') != expected_architecture:
            raise ValueError('Checkpoint is not this half-A architecture')
        check_state(payload.get('state_dict'))
        checked = dict(payload)
        checked['architecture_version'] = original_version
        original_require(checked)
    for module in tuple(sys.modules.values()):
        if module is not None and getattr(module, '__name__', '').startswith('hiercp.'):
            if getattr(module, 'require_current_checkpoint', None) is original_require:
                module.require_current_checkpoint = required
    contracts.require_current_checkpoint = required
    # The archived completed-run skip occurs before load_state_dict. Validate
    # markers at checkpoint deserialization too, including the best artifact.
    original_load = importlib.import_module('hiercp.tensor').torch_load_compat

    @functools.wraps(original_load)
    def checked_load(path, *args, **kwargs):
        result = original_load(path, *args, **kwargs)
        if Path(path).resolve().parent == Path(receipt['experiment']).resolve() / 'results/half_A':
            if not isinstance(result, dict) or result.get('architecture_version') != expected_architecture:
                raise ValueError('Half-A checkpoint metadata belongs to another experiment')
            check_state(result.get('state_dict'))
            if (result.get('format') == 'hiercp_training_state_v1'
                    and (bool(result.get('training_complete')) or int(result.get('epoch', 0)) >= 40)):
                if (result.get('training_complete') is not True
                        or type(result.get('epoch')) is not int or result['epoch'] != 40
                        or type(result.get('target_epochs')) is not int or result['target_epochs'] != 40):
                    raise ValueError('Completed half-A last state does not contain the full40epoch run')
                # The original completed-run branch only tests is_file(), so
                # validate the actual best artifact before allowing that skip.
                best = original_load(Path(path).resolve().parent / 'checkpoint_best.pt',
                                     *args, **kwargs)
                if (not isinstance(best, dict)
                        or best.get('architecture_version') != expected_architecture
                        or best.get('training_complete') is not True
                        or best.get('completed_epoch') != 40
                        or best.get('target_epochs') != 40
                        or best.get('epoch') != result.get('best_epoch')):
                    raise ValueError('Completed half-A best does not belong to the same full run')
                check_state(best.get('state_dict'))
                for key in ('model_kwargs', 'graph_config', 'geometry_contract', 'ct_clip',
                            'cache_publication', 'training_signature', 'validation_policy',
                            'preflight_calibration'):
                    if key not in result or key not in best or best[key] != result[key]:
                        raise ValueError(f'Completed half-A best/last differ: {key}')
        return result
    importlib.import_module('hiercp.tensor').torch_load_compat = checked_load


def report_actual_encoder(pipeline, receipt):
    from .half_a_model import model_contract
    original = pipeline._print_report

    @functools.wraps(original)
    def report(name, payload):
        if name == 'PreRun':
            import copy
            payload = copy.deepcopy(payload)
            payload['model']['experiment_version'] = 'v1.5_half_A'
            payload['model']['local_encoder'] = model_contract()
            payload['model']['local_layers'] = 0
            payload['model']['configuration']['local_layers_argument'] = payload['model']['configuration'].pop('local_layers')
            payload['model']['configuration']['local_graph_layers_executed'] = 0
            payload['model']['configuration']['checkpoint_local_blocks_argument'] = payload['model']['configuration'].pop('checkpoint_local_blocks')
            payload['model']['configuration']['checkpoint_local_graph_executed'] = False
            payload['model']['dense_feature_dimension'] = 68
            payload['model']['dense_feature_dimension_meaning'] = 'concatenated12/24/32 masked pyramid readout before shared128D projection'
            payload['model']['configuration']['dense_feature_dim_argument'] = payload['model']['configuration'].pop('dense_feature_dim')
            payload['graph_and_input']['local_message_passing'] = False
            payload['graph_and_input']['local_graph_use'] = 'original sampled role/shell ownership for CNN readout; no L0 messages'
            payload['graph_and_input']['edge_execution']['scope'] = 'unchanged L1/L2 only; cached L0 edges are not executed'
            payload['experiment'] = dict(replaced_group='A_L0', original_GT_loss=True,
                measured_execution=receipt['baseline_proof']['execution'],
                cache_preparation=False, baseline_weights_resume=False,
                B_started=False, combined_started=False)
        return original(name, payload)
    pipeline._print_report = report


def bind_execution_resources(pipeline, receipt):
    """Reuse measured comparison controls only under the same allocation."""
    from .experiment import write_new
    original = pipeline._calibration_resource_fingerprint
    expected = receipt['baseline_proof']['execution']['resource_fingerprint']
    @functools.wraps(original)
    def checked(report, *, device):
        actual = original(report, device=device)
        if actual != expected:
            raise ValueError('Current allocation differs from baseline batch/worker measurement; no silent migration')
        lock = dict(format='hiercp_half_A_inherited_execution_lock_v1',
            half_A_contract_sha256=receipt['contract_sha256'],
            origin='verified_completed_baseline_preflight', newly_calibrated=False,
            baseline_experiment=receipt['baseline_experiment'],
            physical_batch=receipt['baseline_proof']['execution']['physical_batch'],
            workers=receipt['baseline_proof']['execution']['workers'],
            gradient_accumulation_steps=receipt['config']['training']['gradient_accumulation_steps'],
            data_parallel_workers=1, resource_fingerprint=actual)
        lock['effective_batch'] = lock['physical_batch'] * lock['gradient_accumulation_steps']
        path = Path(receipt['experiment'])/'execution_lock.json'
        if path.exists():
            if json.loads(path.read_text(encoding='utf8')) != lock:
                raise ValueError('Existing half-A execution lock changed')
        else:
            write_new(path, lock)
        return actual
    pipeline._calibration_resource_fingerprint = checked


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--request', required=True)
    a = p.parse_args(argv)
    from .half_a_training import verify
    request = json.loads(Path(a.request).read_text(encoding='utf8'))
    if set(request) != {'experiment', 'contract_sha256'}:
        raise ValueError('Explicit half-A request fields differ')
    receipt = verify(request['experiment'])
    if request['contract_sha256'] != receipt['contract_sha256']:
        raise ValueError('Half-A worker request differs from frozen manifest')
    from .scope_probe_support import activate_original
    from . import bounded_scope
    from .scope_training_entry import (ResourceBudget, install_model_budget,
        ScopeWorkerInitializer, install_loader_hook,
        _install_training_observation)
    from .sampling_entry import preserve_worker_calibration_rng
    from tools.local_cnn_device import select
    select(receipt['gpu'])
    activate_original(Path(receipt['source']))
    scope = bounded_scope.install(10, expected_snapshot_root=receipt['source'])
    if scope['contract_sha256'] != receipt['baseline_proof']['neural_baseline']['scope_digest']:
        raise ValueError('Actual10mm adapter does not reproduce the baseline scope')
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Actual CUDA required; no CPU training fallback')
    total = torch.cuda.get_device_properties(0).total_memory
    if receipt['cuda_gib'] * 2**30 > total:
        raise ValueError('Explicit half-A CUDA budget exceeds selected GPU')
    torch.cuda.set_per_process_memory_fraction(receipt['cuda_gib'] * 2**30 / total, 0)
    budget = ResourceBudget(receipt['cuda_gib'], receipt['rss_gib'])
    install_model_budget(budget)
    install_identity(receipt, scope)
    baseline = Path(receipt['baseline_experiment'])
    pipeline = importlib.import_module('hiercp.pipeline')
    preserve_worker_calibration_rng(pipeline)
    bind_execution_resources(pipeline, receipt)
    output = Path(receipt['experiment']) / 'results/half_A'
    session = _install_training_observation(pipeline, output, receipt, dict(scope, half_A_contract=receipt['contract_sha256']))
    recorder = None
    previous = sys.argv
    try:
        install_loader_hook(pipeline, ScopeWorkerInitializer(receipt['source'], 10, scope['contract_sha256'], True))
        report_actual_encoder(pipeline, receipt)
        from .half_a_training import validation_cohort
        from .half_a_case_metrics import install_case_score_recorder
        native = json.loads((baseline/'manifest.json').read_text(encoding='utf8'))
        cohort = validation_cohort(receipt['baseline_proof']['neural_baseline']['training_signature']['val_cache_files'],
                                   native['split']['val'],
                                   json.loads((baseline/'shared/cache/index.json').read_text(encoding='utf8')))
        from hiercp.model import HierarchicalPyGPlacementModel
        recorder = install_case_score_recorder(HierarchicalPyGPlacementModel, session, output,
            expected_case_ids=cohort['case_ids'], expected_samples=cohort['samples'])
        from .telemetry_entry import invocation_logs
        with invocation_logs(output):
            print('FIRST HALF A ONLY | fresh CNN L0 | preserved v1 GT/curriculum/loss/L1/L2/scorer | no prepare', flush=True)
            print(json.dumps(dict(stage='half_A_validation_cohort', **cohort,
                                  raw_scores_path=str(recorder.path)), allow_nan=False), flush=True)
            sys.argv = ['hiercp.pipeline', 'train', '--config', str(Path(receipt['experiment'])/'config.json'),
                '--cache-dir', str(baseline/'shared/cache'), '--prototype-bank', str(baseline/'shared/prototype_bank.pt'),
                '--checkpoint', str(output/'checkpoint_best.pt'), '--device', 'cuda']
            budget()
            result = pipeline.main()
            budget()
            return result
    finally:
        sys.argv = previous
        from .experiment import write_new
        try:
            if recorder is not None:
                try:
                    write_new(recorder.path.with_suffix('.coverage.json'), recorder.totals())
                finally:
                    try:
                        recorder.restore()
                    finally:
                        recorder.close()
        finally:
            session.close()


if __name__ == '__main__':
    main()
