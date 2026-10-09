"""Read intermediate comparison weights for the existing native full-P/U evaluator.

This is not the completed historical-BEST loader.  It verifies the preserved
comparison experiment, initial state and selected checkpoint, constructs the
same archived GAT, and imports model tensors only.  No experiment is written,
no optimizer/RNG state is restored, and no training controller is invoked.
"""
from __future__ import annotations

import copy
import importlib
from pathlib import Path

from .historical_checkpoint import HistoricalModel, _activate
from .u_bridge_experiment import digest as json_digest, read, sha

ROOT = Path(__file__).resolve().parents[1]
ARMS = ('selected', 'native', 'native_fixed', 'native_listwise')
PARAMETERS = 10434532
FORMAT = 'comparison_intermediate_native_evaluation_checkpoint_v1'


def _regular(path):
    path = Path(path).absolute()
    if any(item.is_symlink() for item in (path, *path.parents)) or not path.is_file():
        raise ValueError(f'Regular preserved checkpoint/evidence required: {path}')
    return path.resolve(strict=True)


def _manifest(root, arm):
    manifest = read(_regular(root / 'experiment.json'))
    expected = ('u_bridge' if arm in ('selected', 'native') else 'comparison')
    module = importlib.import_module('hiercp_v1x.' + expected + '_experiment')
    if (manifest.get('format') != module.FORMAT
            or manifest.get('sha256') != json_digest({k: v for k, v in manifest.items() if k != 'sha256'})
            or manifest.get('debug') is not False or manifest.get('epochs') != 40
            or set(manifest.get('helpers', {})) != set(module.FILES)):
        raise ValueError('Exact sealed production comparison contract required')
    changed = [name for name, checksum in manifest['helpers'].items()
               if sha(_regular(ROOT / name)) != checksum]
    if changed:
        raise ValueError(f'Frozen comparison helper bytes changed: {changed}')
    if expected == 'u_bridge':
        # This is the controller's original read-only v1.8 admission check.
        from tools.run_v18_independent import validate_reference
        if validate_reference(root) != manifest:
            raise ValueError('Original independent-run reference differs')
        continuation = read(_regular(root / 'continuation.json'))
        request = continuation.get('request', {})
        if (continuation.get('receipt_sha256') != json_digest(
                {k: v for k, v in continuation.items() if k != 'receipt_sha256'})
                or request.get('format') != 'v18_independent_arm_continuation_v1'
                or request.get('arm') != arm or request.get('destination_root') != str(root)
                or request.get('data_root') != str(root / 'data')
                or request.get('contract_sha256') != manifest['sha256']):
            raise ValueError('Independent continuation provenance differs')
    return manifest, expected


def _validate_binding(arm, family, manifest, ownership, calibration, initial):
    """Reconstruct the original signed training binding without a data provider."""
    from . import u_bridge_training as engine
    from .scope_probe_support import state_digest
    binding = ownership.get('binding')
    if not isinstance(binding, dict):
        raise ValueError('Comparison training identity content changed')
    initial_model = initial.get('model')
    if (not isinstance(initial_model, dict)
            or initial.get('contract_sha256') != manifest['sha256']
            or initial.get('model_sha256') != state_digest(initial_model)
            or calibration.get('contract_sha256') != manifest['sha256']):
        raise ValueError('Original common initialization or calibration binding differs')
    initial_hash = engine.digest(initial_model)
    chosen = calibration.get('physical_batch')
    if type(chosen) is not int or chosen < 1:
        raise ValueError('Actual original measured physical batch required')
    actual_calibration = copy.deepcopy(calibration['reports'][arm])
    actual_calibration['selected_physical_batch'] = chosen
    if (actual_calibration.get('arm') != arm
            or actual_calibration.get('initial_state_sha256') != initial_hash
            or actual_calibration.get('original_model_and_RNG_preserved') is not True
            or not any(row.get('accepted') is True and row.get('physical_batch') == chosen
                       for row in actual_calibration.get('reports', []))):
        raise ValueError('Original arm calibration does not bind its exact initialization')
    config = copy.deepcopy(manifest['config'])
    config['u_bridge_runtime'] = dict(batch_calibration=actual_calibration,
        validation_local_chunk_size=manifest['validation_local_chunk'], expected_parameters=PARAMETERS)
    identity = dict(contract_sha256=manifest['sha256'],
        initial_neural_sha256=initial['model_sha256'], initial_state_sha256=initial_hash)
    training_format = engine.FORMAT
    policy = None
    if family == 'comparison':
        from .comparison_training import arm_policy, FORMAT as training_format
        policy = arm_policy(arm)
        config['comparison_policy'] = copy.deepcopy(policy)
        identity['comparison_policy'] = copy.deepcopy(policy)
    # UBridgeData._centers constructs a tuple of coordinate tuples. Its
    # examples() and engine._examples() preserve those tuples inside an outer
    # list. Both experiment.json and training_identity.json serialize them as
    # JSON arrays, but engine.digest deliberately distinguishes tuples/lists.
    # Recover only these known provider fields; never normalize the digest or
    # accept a hash of the lossy JSON binding as the original training identity.
    from .u_bridge_data import _centers
    samples = copy.deepcopy(manifest['samples'])
    for row in samples:
        row['positive_center'] = _centers([row['positive_center']], 1, row['id'] + ' P')[0]
        row['selected_centers'] = _centers(row['selected_centers'], 7, row['id'] + ' selected')
        row['native_centers'] = _centers(row['native_centers'], 128, row['id'] + ' native U')
    # Preserve every sealed sample and its order without loading input caches.
    train = [row for row in samples if row['partition'] == 'train']
    validation = [row for row in samples if row['partition'] == 'val']
    if not train or not validation or len(train) + len(validation) != len(samples):
        raise ValueError('Complete signed comparison train/validation source lists required')
    expected = dict(format=training_format, identity=identity, config=config, arm=arm,
        debug=False, epochs=40, physical_batch=chosen, workers=manifest['workers'],
        train_examples=train, val_examples=validation, initial_state_sha256=initial_hash)
    # Match every persisted JSON field separately from proving the original
    # type-sensitive training hash. A JSON round trip is not a new identity.
    if (json_digest(binding) != json_digest(expected)
            or ownership.get('identity_sha256') != engine.digest(expected)):
        raise ValueError('Checkpoint ownership is not the original complete comparison binding')
    return training_format, policy


def _checkpoint_metadata(saved, *, ownership, training_format, arm, policy, selection):
    from . import u_bridge_training as engine
    if (saved.get('format') != training_format
            or saved.get('identity_sha256') != ownership['identity_sha256']
            or saved.get('content_sha256') != engine.digest(
                {k: v for k, v in saved.items() if k != 'content_sha256'})
            or saved.get('comparison_policy') != policy):
        raise ValueError('Selected comparison checkpoint content/arm/policy identity differs')
    geometry = saved.get('recipient_context_policy')
    if geometry is not None:
        from .comparison_empty_context import identity
        if geometry != identity():
            raise ValueError('Selected checkpoint recipient-context policy differs')
    curriculum = saved.get('curriculum_policy')
    if curriculum is not None:
        from .comparison_curriculum import policy as curriculum_policy
        if curriculum != curriculum_policy(arm):
            raise ValueError('Selected checkpoint continuation curriculum differs')
    state = saved.get('state')
    if not isinstance(state, dict):
        raise ValueError('Real saved training cursor required')
    history = state.get('history')
    if (not isinstance(history, list) or len(history) > 40
            or [row.get('epoch') for row in history] != list(range(1, len(history) + 1))
            or type(state.get('epoch')) is not int or not 1 <= state['epoch'] <= 41
            or type(state.get('updates')) is not int or state['updates'] < 1
            or state.get('phase') not in ('training', 'validation', 'complete')
            or state['epoch'] != len(history) + 1):
        raise ValueError('Trained intermediate checkpoint cursor/history differs')
    best = state.get('best')
    if (not isinstance(best, dict) or type(best.get('epoch')) is not int
            or not 1 <= best['epoch'] <= len(history)
            or best.get('update') != history[best['epoch'] - 1].get('update')
            or best.get('selection_key') != list(engine.selection_key(
                history[best['epoch'] - 1]['validation129']))
            or tuple(best['selection_key']) != max(engine.selection_key(row['validation129'])
                                                  for row in history)):
        raise ValueError('Stored full129 BEST selection history differs')
    if selection == 'best' and (best['epoch'] != len(history)
            or state['updates'] != best['update'] or state.get('position') != 0
            or state['phase'] not in ('training', 'complete')):
        raise ValueError('BEST file is not the model at its recorded full129 selection')
    complete = state['phase'] == 'complete' and len(history) == 40
    if (state['phase'] == 'complete') != complete:
        raise ValueError('Completed training phase does not contain forty complete epochs')
    return dict(format=FORMAT, comparison_arm=arm, selection_name=selection,
        selected_epoch=best['epoch'] if selection == 'best' else state['epoch'],
        selected_epoch_interpretation=('completed full129-selected epoch' if selection == 'best'
            else 'saved training cursor epoch; may include a partial epoch'),
        completed_epochs=len(history), target_epochs=40, full_training_complete=complete,
        training_complete=complete, original_training_updates=state['updates'],
        cursor={key: copy.deepcopy(state.get(key)) for key in
                ('epoch', 'phase', 'position', 'validation_position', 'updates', 'invocation_status')},
        best=copy.deepcopy(best), selection=copy.deepcopy(best) if selection == 'best' else None,
        selection_task='original comparison full129 patient-macro validation',
        identity_sha256=saved['identity_sha256'], content_sha256=saved['content_sha256'],
        recipient_context_policy=copy.deepcopy(geometry), curriculum_policy=copy.deepcopy(curriculum),
        optimizer_imported=False, scheduler_imported=False, scaler_imported=False,
        training_rng_restored=False, last_used_as_best=False,
        training_started=False, optimizer_updates=0)


def load_comparison(arm, experiment, selection, *, budget, activated_scope=None):
    """Return the unchanged original-model branch of HistoricalModel for scoring.

    Select ``best`` or ``latest`` explicitly.  Existing best metadata is never
    interpreted as completion of forty epochs.  Each call belongs to a fresh
    worker because archived model and geometry namespaces are process-global.
    """
    import torch
    from . import u_bridge_training as engine
    from .half_a_training import baseline_proof
    if arm not in ARMS or selection not in ('best', 'latest') or not callable(budget):
        raise ValueError('Explicit comparison arm, best/latest selection and resource guard required')
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Native comparison evaluation requires one selected actual CUDA GPU')
    root = Path(experiment).resolve(strict=True)
    manifest, family = _manifest(root, arm)
    baseline = Path(manifest['baseline']['baseline']).resolve(strict=True)
    source = (baseline / 'source/v1.0').resolve(strict=True)
    bank_path = _regular(baseline / 'shared/prototype_bank.pt')
    checkpoint_path = _regular(root / arm / ('checkpoint_' + selection + '.pt'))
    paths = [root / 'experiment.json', root / 'initial.pt', root / 'calibration.json',
             root / arm / 'training_identity.json', checkpoint_path, bank_path]
    if family == 'u_bridge':
        paths.append(root / 'continuation.json')
    before = {str(_regular(path)): sha(path) for path in paths}
    budget()
    baseline_manifest, proof = baseline_proof(baseline)
    config = copy.deepcopy(baseline_manifest['config'])
    config['graph'].update(adaptive_roi_margin_mm=10., context_outer_radius_mm=10.)
    if (json_digest(proof) != manifest['baseline']['baseline_proof_sha256']
            or config != manifest['config'] or sha(bank_path) != manifest['baseline']['bank_sha256']
            or source != Path(manifest['original']['source']).resolve(strict=True)):
        raise ValueError('Preserved completed baseline configuration/prototype/source differs')
    if (config['seed'] != 42 or config['training']['epochs'] != 40
            or config['cache']['total_candidates'] != 8 or config['cache']['candidate_pool_size'] != 128
            or config['model']['hidden_dim'] != 128 or config['model']['heads'] != 4
            or [config['model'][key] for key in ('local_layers', 'patient_layers', 'prototype_layers')] != [3, 2, 2]):
        raise ValueError('Original full-size GAT model and research contract required')
    ownership = read(root / arm / 'training_identity.json')
    calibration = read(root / 'calibration.json')
    initial = torch.load(root / 'initial.pt', map_location='cpu', weights_only=False, mmap=True)
    training_format, policy = _validate_binding(arm, family, manifest, ownership, calibration, initial)
    initial_keys = set(initial['model'])
    initial_shapes = {key: (value.dtype, tuple(value.shape)) for key, value in initial['model'].items()}
    del initial
    budget()
    saved = torch.load(checkpoint_path, map_location='cpu', weights_only=False, mmap=True)
    checkpoint = _checkpoint_metadata(saved, ownership=ownership, training_format=training_format,
        arm=arm, policy=policy, selection=selection)
    weights = saved.get('model')
    if (not isinstance(weights, dict) or set(weights) != initial_keys
            or any(not isinstance(value, torch.Tensor)
                   or (value.dtype, tuple(value.shape)) != initial_shapes[key]
                   for key, value in weights.items())):
        raise ValueError('Selected actual model keys/dtypes/shapes differ from its original initialization')
    for value in weights.values():
        if (value.is_floating_point() or value.is_complex()) and not bool(torch.isfinite(value).all()):
            raise ValueError('Selected model contains nonfinite tensor values')
    source_proof, scope = _activate(source, activated_scope)
    if (scope != manifest['scope']
            or scope['contract_sha256'] != proof['neural_baseline']['scope_digest']
            or (activated_scope is None and source_proof != manifest['original'])):
        raise ValueError('Actual archived geometry does not reproduce the trained scope')
    marker = weights.get('v1x_bounded_scope_digest')
    if marker is not None:
        expected = torch.tensor(list(bytes.fromhex(scope['contract_sha256'])), dtype=torch.uint8)
        if marker.dtype != torch.uint8 or marker.shape != (32,) or not torch.equal(marker.cpu(), expected):
            raise ValueError('Existing model scope marker differs')
        from .scope_training_entry import install_checkpoint_binding
        install_checkpoint_binding(scope)
    # The actual comparison controller did not install a scope-state buffer.
    # Do not fabricate one: its source/scope are bound by the sealed experiment.
    from hiercp.model import HierarchicalPyGPlacementModel
    from hiercp.prototype import PrototypeBank
    model = HierarchicalPyGPlacementModel(**config['model'])
    if (model.architecture_version != proof['neural_baseline']['architecture_version']
            or sum(p.numel() for p in model.parameters()) != PARAMETERS
            or sum(p.numel() for p in model.parameters() if p.requires_grad) != PARAMETERS):
        raise ValueError('Constructed original architecture or full parameter count differs')
    model.load_state_dict(weights, strict=True)
    expected_model_digest = engine.digest(weights)
    if engine.digest(model.state_dict()) != expected_model_digest:
        raise ValueError('Strict model import did not preserve every saved tensor exactly')
    model.to('cuda').eval()
    if engine.digest(model.state_dict()) != expected_model_digest:
        raise ValueError('CUDA transfer changed actual checkpoint tensor values')
    bank = PrototypeBank.load(bank_path)
    if (bank.fingerprint() != manifest['prototype_fingerprint']
            or set(bank.training_case_ids) != set(baseline_manifest['split']['train'])):
        raise ValueError('Actual prototype bank differs from the original training-only population')
    preserved = {**proof['files'], **before}
    preserved.update({str(ROOT / name): checksum for name, checksum in manifest['helpers'].items()})
    preserved.update({str(source / name): checksum for name, checksum in
                      manifest['original']['verified_files'].items()})
    if any(sha(path) != checksum for path, checksum in preserved.items()):
        raise ValueError('Original model/prototype/source evidence changed during read-only loading')
    checkpoint.update(path=str(checkpoint_path), sha256=before[str(checkpoint_path)],
        model_tensor_sha256=expected_model_digest, architecture=model.architecture_version,
        total_parameters=PARAMETERS, trainable_parameters=PARAMETERS,
        prototype_bank_sha256=before[str(bank_path)], source_proof=source_proof,
        files_preserved=preserved, exact_model_tensor_restore_verified=True,
        scope_marker_in_original_state=marker is not None)
    receipt = dict(format=FORMAT, debug=False, comparison_arm=arm,
        comparison_experiment_sha256=manifest['sha256'], baseline_experiment=str(baseline),
        baseline_proof=proof, source=str(source), config=copy.deepcopy(config),
        full_training_complete=checkpoint['full_training_complete'], checkpoint_written=False,
        native_inventory_sha256=manifest['baseline']['inventory_sha256'])
    del saved, weights
    budget()
    return HistoricalModel('V1', model, config, bank, checkpoint, receipt, scope, source, baseline, root)
