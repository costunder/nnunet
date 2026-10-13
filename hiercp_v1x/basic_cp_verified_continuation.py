"""Exact whole-epoch continuation of an unchanged full Basic CP experiment.

The caller chooses a completed checkpoint and pins its historical evidence.
This module never discovers a partial checkpoint, modifies an old namespace,
recomputes geometry, signals a process, or changes the full training horizon.
"""
from __future__ import annotations

import copy
import os
from pathlib import Path
import re
import shutil

from . import basic_cp_ablation_resume as base

FORMAT = 'basic_cp_verified_whole_native_continuation_v1'
PREFLIGHT_FORMAT = 'basic_cp_one_factor_full250_v1'
WITNESS_KEYS = {'root', 'request_sha256', 'native_sha256', 'preflight_sha256',
                'controller_identity', 'checkpoint_path', 'checkpoint_sha256',
                'best_sha256', 'completed_epochs'}
ARMS = {
    'multi3': (1, 'nnUNetTrainer_250epochs_BasicCPMulti3', 128, 3),
    'candidates256': (4, 'nnUNetTrainer_250epochs_BasicCPCandidates256', 256, 1),
    'blend': (6, 'nnUNetTrainer_250epochs_BasicCPBlend80', 128, 1),
}


class _Evidence:
    def __init__(self):
        self.files = {}

    def admit(self, path, expected=None, *, root=None):
        path = base.exact(path, root=root)
        before = base.snapshot(path)
        digest = base.sha(path)
        base.require(base.snapshot(path) == before and
                     (expected is None or digest == expected),
                     'Continuation source SHA/stat changed: ' + str(path))
        if str(path) in self.files:
            base.require(self.files[str(path)] == before, 'Continuation evidence changed between reads')
        self.files[str(path)] = before
        return path, digest

    def unchanged(self):
        base.require(all(base.snapshot(path) == value for path, value in self.files.items()),
                     'Original continuation evidence changed during read/copy')


def _common(native):
    witness = native.get('verified_native_continuation')
    base.require(isinstance(witness, dict) and set(witness) == WITNESS_KEYS and
                 type(witness['completed_epochs']) is int and 0 < witness['completed_epochs'] < 250,
                 'Exact whole completed native continuation witness required')
    arm = native.get('arm')
    base.require(arm in ARMS, 'Only original Basic multi3, candidates256 and fixed80 may continue')
    gpu, trainer, count, copies = ARMS[arm]
    contract = dict(gpu=gpu, arm=arm, trainer=trainer, sources=642,
                    candidates_per_source=count, copies_per_CP_event=copies,
                    CP_probability=.5, seed=42, epochs=250, train_cases=105,
                    validation_cases=26, physical_batch=2, checkpoint='checkpoint_best.pth')
    if arm == 'blend':
        contract.update(blend_mode='constant', blend_alpha=.8,
                        blend_border_preprocessed_voxels=0)
    else:
        contract['blend_border_preprocessed_voxels'] = 0
    for key, value in contract.items():
        base.require(native.get(key) == value, 'Current full native contract differs: ' + key)
    root = base.exact(native['root'], directory=True)
    new_native = base.exact(native['native_root'], root=root, directory=True)
    oldroot = base.exact(witness['root'], directory=True)
    base.require(oldroot.is_relative_to(base.SOURCE_NAMESPACE) and root != oldroot and
                 root not in oldroot.parents and oldroot not in root.parents,
                 'Separate own original and fresh continuation namespaces required')
    evidence = _Evidence()
    request_path, _ = evidence.admit(oldroot/'request.json', witness['request_sha256'], root=oldroot)
    oldpath, _ = evidence.admit(oldroot/'native.json', witness['native_sha256'], root=oldroot)
    prepath, _ = evidence.admit(oldroot/'preflight.json', witness['preflight_sha256'], root=oldroot)
    request, old, preflight = map(base.read, (request_path, oldpath, prepath))
    base.require(request.get('uid') == base.UID and
                 request.get('controller_identity') == witness['controller_identity'],
                 'Exact original own request/controller identity differs')
    for key, value in contract.items():
        if key != 'trainer':  # The original request acquires its installed trainer in prepare().
            base.require(request.get(key) == value, 'Original request scientific contract differs: ' + key)
    base.released(witness['controller_identity'])
    actorpath, _ = evidence.admit(oldroot/'train_actor.json', root=oldroot)
    actor = base.read(actorpath)
    actor_identity = dict(pid=actor['pid'], create_time=actor['birth'],
                          UID=actor['UID'], command=actor['command'])
    base.require(actor.get('stage') == 'train', 'Exact original production train actor required')
    base.released(actor_identity)
    identities = ('gpu', 'arm', 'trainer', 'bank', 'bank_sha256', 'baseline', 'split',
                  'historical_trainer_sha256', 'readonly_admission', 'readonly_admission_sha256',
                  'shared_preprocessed_cases', 'source_preprocessed_arrays_written',
                  'full_training_cases', 'full_validation_cases', 'CP_probability', 'seed',
                  'sources', 'candidates_per_source', 'copies_per_CP_event',
                  'blend_border_preprocessed_voxels')
    if arm == 'blend':
        identities += ('blend_mode', 'blend_alpha')
    for key in identities:
        base.require(old.get(key) == native.get(key),
                     'Original matched scientific identity differs: ' + key)
    base.require(old.get('root') == str(oldroot) and
                 old.get('native_root') == str(oldroot/'native') and
                 native['historical_trainer_sha256'] == base.HISTORICAL_SHA and
                 native['shared_preprocessed_cases'] == 131 and
                 native['source_preprocessed_arrays_written'] is False and
                 native['full_training_cases'] == 105 and native['full_validation_cases'] == 26,
                 'Original full105/26/shared131/historical native contract required')
    train, val = native['split']['outer_train'], native['split']['outer_val']
    base.require(len(train) == len(set(train)) == 105 and len(val) == len(set(val)) == 26 and
                 set(train).isdisjoint(val), 'Original unique disjoint105/26 case identities required')
    admission_path, _ = evidence.admit(native['readonly_admission'], native['readonly_admission_sha256'])
    admission = base.read(admission_path)
    base.require(admission.get('split') == native['split'] and
                 admission.get('baseline') == native['baseline'] and
                 set(admission.get('preprocessed', {}).get('cases', ())) == set(train) | set(val) and
                 len(admission['preprocessed']['cases']) == 131 and
                 admission.get('source_arrays_written') is False and
                 admission.get('model_data_scale_preserved') is True and admission.get('debug') is False,
                 'Exact full131 read-only preprocessing admission differs')
    if count == 128:
        base.require(old['bank_sha256'] == base.BANK_SHA and
                     request.get('bank') == old['bank'] and request.get('bank_sha256') == base.BANK_SHA,
                     'Original full128 Basic source index differs')
    else:
        base.require(old.get('source_bank_sha256') == base.BANK_SHA and
                     request.get('bank_sha256') in (base.BANK_SHA, old['bank_sha256']),
                     'Original128 source binding of complete256 bank differs')
    model_path, _ = evidence.admit(oldroot/'actual_model.json', root=oldroot)
    model = base.read(model_path)
    expected_model = dict(trainer=trainer, epochs=250, physical_batch=2, effective_batch=2,
                          gradient_accumulation=1, workers=4, patch_size=[128]*3,
                          debug=False, subset=False, pretrained_segmentation_used=False)
    for key, value in expected_model.items():
        base.require(model.get(key) == value, 'Original full model250/B2/workers4 differs: ' + key)
    base.require(sorted(model.get('actual_train_cases', ())) == sorted(train) and
                 sorted(model.get('actual_validation_cases', ())) == sorted(val),
                 'Original actual full training/validation model population differs')
    names = ['nnUNetTrainer_OnlinePairedCP.py', 'nnUNetTrainer_BasicCPAblations.py']
    if arm == 'blend':
        names.append('nnUNetTrainer_BasicCPBlend80.py')
    trainer_hashes = {}
    for name in names:
        relative = Path('nnunetv2/training/nnUNetTrainer')/name
        expected = base.HISTORICAL_SHA if name == names[0] else request.get('code_sha256', {}).get('custom_trainers/'+name)
        _, digest = evidence.admit(Path(old['private_runtime'])/relative, expected, root=oldroot)
        _, current = evidence.admit(Path(native['private_runtime'])/relative, root=new_native)
        base.require(current == digest, 'Installed scientific trainer bytes changed: ' + name)
        trainer_hashes[name] = digest
    evidence.unchanged()
    return witness, root, new_native, oldroot, old, preflight, actor_identity, evidence, trainer_hashes


def _verify_preflight(native, old, result, evidence):
    arm = native['arm']
    _, _, count, copies = ARMS[arm]
    bank_path, _ = evidence.admit(native['bank'], native['bank_sha256'])
    bank = base.read(bank_path)
    entries = bank.get('entries_by_case')
    base.require(isinstance(entries, dict) and len(entries) == 81 and
                 set(entries).issubset(native['split']['outer_train']),
                 'All81 own-patient source cases required')
    names = [name for values in entries.values() for name in values]
    base.require(len(names) == len(set(names)) == 642 and bank.get('candidate_count') == count and
                 bank.get('total_candidates') == 642*count,
                 'All642 full source/candidate entries required')
    values = dict(format=PREFLIGHT_FORMAT, arm=arm, complete=True,
                  index_sha256=native['bank_sha256'], historical_trainer_sha256=base.HISTORICAL_SHA,
                  historical_bank_sha256=base.BANK_SHA, source_entries=642, eligible_cases=81,
                  full_training_cases=105, full_validation_cases=26, candidate_count=count,
                  validated_positions=642*count, three_paste_positions=82176 if arm == 'multi3' else 0,
                  CP_probability=.5, copies_per_event=copies,
                  blend_border_preprocessed_voxels=0, original_gate_primary_RNG_schedule_preserved=True,
                  GNN_scores_used=False, physical_batch_size=2, seed=42, epochs=250,
                  network_patch_size=[128]*3, training_executed=False)
    if arm == 'blend':
        values.update(blend_mode='constant', blend_alpha=.8, blend_recipient_weight=.2, label_blending=False)
    for key, value in values.items():
        base.require(result.get(key) == value, 'Complete original preflight differs: ' + key)
    actual = result.get('actual_training_identifiers', ())
    base.require(len(actual) == len(set(actual)) == 105 and
                 set(actual) == set(native['split']['outer_train']),
                 'All105 preflight training identifiers required')
    checksums = result.get('entries_sha256', {})
    base.require(set(checksums) == set(names), 'Exactly all642 preflight source SHA witnesses required')
    receipts = result.get('case_receipts', ())
    base.require(len(receipts) == 81 and {row['case_id'] for row in receipts} == set(entries),
                 'Exactly all81 complete case receipts required')
    for row in receipts:
        case = row['case_id']; expected = set(entries[case])
        base.require(row.get('source_entries') == len(expected) and
                     row.get('validated_positions') == len(expected)*count and
                     row.get('three_paste_positions') == (len(expected)*count if arm == 'multi3' else 0) and
                     set(row.get('entries_sha256', {})) == expected and
                     all(row['entries_sha256'][name] == checksums[name] for name in expected),
                     'Partial/mismatched original patient preflight: ' + case)
    for name, digest in checksums.items():
        evidence.admit(bank_path.parent/name, digest, root=bank_path.parent)
    if count == 256:
        complete_path, _ = evidence.admit(bank_path.parent/'complete.json', root=bank_path.parent)
        complete = base.read(complete_path)
        base.require(complete.get('complete') is True and
                     complete.get('index_sha256') == native['bank_sha256'] and
                     complete.get('original_index_sha256') == base.BANK_SHA and
                     complete.get('all_original_arrays_and_first128_preserved') is True and
                     complete.get('entries_sha256') == checksums and
                     result.get('extension_complete') == complete,
                     'Exact complete256 preservation/source SHA proof required')
    evidence.unchanged()


def reuse_preflight(native):
    """Return the complete original geometry receipt after full read-only SHA checks."""
    witness, _, _, oldroot, old, result, actor, evidence, trainer_hashes = _common(native)
    _verify_preflight(native, old, result, evidence)
    reused = copy.deepcopy(result)
    # Keep any previous constructor-repair reuse proof as original provenance.
    reused['continuation_reuse'] = dict(format=FORMAT, original_root=str(oldroot),
        original_request_sha256=witness['request_sha256'], original_native_sha256=witness['native_sha256'],
        original_preflight_sha256=witness['preflight_sha256'], full_source_SHA_reverified=642,
        all_complete_case_receipts_verified=81, actual_training_identifiers_verified=105,
        scientific_trainer_sha256=trainer_hashes, geometry_recomputed=False,
        original_reuse_proof_preserved='reuse_proof' in result, source_files_written=False,
        processes_signaled=False)
    evidence.unchanged(); base.released(witness['controller_identity']); base.released(actor)
    return reused


def _completed_prefix(data, epochs):
    text = data.decode('utf8')
    headers = list(re.finditer(r'(?m)^.*?\bEpoch\s+(\d+)\s*$', text))
    boundary = next((header.start() for header in headers if int(header.group(1)) >= epochs), len(text))
    prefix = text[:boundary]
    completed = list(re.finditer(r'(?m)^.*?\bEpoch\s+(\d+)\s*$', prefix))
    base.require([int(header.group(1)) for header in completed] == list(range(epochs)),
                 'Exact unique complete training log headers0..completed-1 required')
    applied = attempted = 0
    for index, header in enumerate(completed):
        segment = prefix[header.end():completed[index+1].start() if index+1 < len(completed) else len(prefix)]
        rows = re.findall(r'\[OnlineCP\] epoch=(\d+) applied=(\d+)/(\d+)', segment)
        base.require(len(rows) == 1, 'Exactly one original OnlineCP counter per completed epoch required')
        epoch, used, total = map(int, rows[0])
        base.require(epoch == index and total == 500 and 0 <= used <= total and
                     re.search(r'\bEpoch time:\s*\d', segment) is not None,
                     'Whole original epoch loss/OnlineCP/completion evidence required')
        applied += used; attempted += total
    return prefix.encode('utf8'), dict(epochs=epochs, first_epoch=0, last_epoch=epochs-1,
                                      CP_applied=applied, CP_attempted=attempted)


def _state(saved, epoch, trainer):
    result = base.validate_checkpoint(saved, epoch=epoch, trainer=trainer)
    for key, values in saved['logging'].items():
        base.require(isinstance(values, (list, tuple)) and len(values) == epoch,
                     'Every native checkpoint logger array must contain exactly completed epochs: ' + key)
    result['all_logger_keys'] = sorted(saved['logging'])
    return result


def prepare_resume(native):
    """Copy selected whole state plus separate original BEST into a fresh fold."""
    import torch
    witness, root, new_native, oldroot, old, result, actor, evidence, trainer_hashes = _common(native)
    _verify_preflight(native, old, result, evidence)
    epoch = witness['completed_epochs']; trainer = native['trainer']
    oldfold = base.exact(base.fold(old), root=oldroot, directory=True)
    selected, selected_sha = evidence.admit(witness['checkpoint_path'], witness['checkpoint_sha256'], root=oldfold)
    base.require(selected.name in ('checkpoint_latest.pth', 'checkpoint_best.pth') and selected.parent == oldfold,
                 'Only explicit original whole latest/BEST checkpoints are eligible; partial backups forbidden')
    best, best_sha = evidence.admit(oldfold/'checkpoint_best.pth', witness['best_sha256'], root=oldfold)
    saved = torch.load(selected, map_location='cpu', weights_only=False)
    state = _state(saved, epoch, trainer)
    best_saved = torch.load(best, map_location='cpu', weights_only=False)
    best_epoch = best_saved.get('current_epoch')
    base.require(type(best_epoch) is int and 0 < best_epoch <= epoch,
                 'Original BEST must be a whole checkpoint within selected completed prefix')
    best_state = _state(best_saved, best_epoch, trainer)
    base.require(set(saved['network_weights']) == set(best_saved['network_weights']) and
                 all(saved['network_weights'][key].shape == best_saved['network_weights'][key].shape and
                     saved['network_weights'][key].dtype == best_saved['network_weights'][key].dtype
                     for key in saved['network_weights']), 'Original selected/BEST full architecture differs')
    del saved, best_saved
    log_path, log_sha = evidence.admit(oldroot/'train.log', root=oldroot)
    prefix, prefix_record = _completed_prefix(log_path.read_bytes(), epoch)
    destination = base.fold(native)
    base.require(destination.is_relative_to(new_native) and not destination.exists() and
                 not (root/'resumed_training_prefix.log').exists() and not (root/'resume.json').exists(),
                 'Fresh untouched continuation fold, prefix and receipt required')
    evidence.unchanged(); base.released(witness['controller_identity']); base.released(actor)
    destination.mkdir(parents=True, exist_ok=False)
    copies = {}
    for source, digest, name in ((selected, selected_sha, 'checkpoint_latest.pth'),
                                (best, best_sha, 'checkpoint_best.pth')):
        target = destination/name
        with source.open('rb') as stream, target.open('xb') as output:
            shutil.copyfileobj(stream, output, 8*2**20); output.flush(); os.fsync(output.fileno())
        base.require(base.sha(target) == digest and base.snapshot(target)[:2] != evidence.files[str(source)][:2],
                     'Byte-exact separate-inode continuation checkpoint copy required')
        copies[name] = dict(path=str(target), sha256=digest, bytes=target.stat().st_size)
    prefix_path = root/'resumed_training_prefix.log'
    with prefix_path.open('xb') as stream:
        stream.write(prefix); stream.flush(); os.fsync(stream.fileno())
    evidence.unchanged(); base.released(witness['controller_identity']); base.released(actor)
    prefix_sha = base.sha(prefix_path)
    receipt = dict(format=FORMAT, completed_epochs=epoch, native_goal_epochs=250, planned_epochs=250,
        remaining_epochs=250-epoch, checkpoint=copies['checkpoint_latest.pth'],
        BEST=copies['checkpoint_best.pth'], checkpoint_current_epoch=epoch, BEST_current_epoch_raw=best_epoch,
        latest_state=state, BEST_state=best_state, source_root=str(oldroot),
        source_checkpoint_path=str(selected), source_request_sha256=witness['request_sha256'],
        source_native_sha256=witness['native_sha256'], source_preflight_sha256=witness['preflight_sha256'],
        source_controller_identity=witness['controller_identity'], source_train_actor_identity=actor,
        source_train_log_sha256=log_sha, training_prefix=dict(prefix_record, path=str(prefix_path), sha256=prefix_sha),
        training_prefix_sha256=prefix_sha, scientific_trainer_sha256=trainer_hashes,
        source_files=evidence.files, original_scientific_trainer_and_bank_preserved=True,
        optimizer_and_scaler_preserved=True, new_copies_independent_inodes=True, old_results_written=False,
        partial_recovery_checkpoint_used=False, full_training_complete=False)
    base.publish(root/'resume.json', receipt)
    return receipt
