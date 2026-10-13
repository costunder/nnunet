"""CPU DEBUG continuation evidence fixtures; no CUDA or training-scale claim."""
import copy
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import torch

from hiercp_v1x import basic_cp_ablation_resume as base
from hiercp_v1x import basic_cp_verified_continuation as continuation
from tests import test_basic_cp_ablation_resume as fixtures


class WholeContinuationTests(unittest.TestCase):
    saved = fixtures.NativeResumeTests.saved
    write_json = staticmethod(fixtures.NativeResumeTests.write_json)

    def setUp(self):
        fixtures.NativeResumeTests.setUp(self)
        self.configure('multi3')

    def configure(self, arm):
        gpu, trainer, count, copies = continuation.ARMS[arm]
        original_bank = self.root/'original128'
        original_bank.mkdir(exist_ok=True)
        bankdir = self.root/('bank_'+arm)
        bankdir.mkdir(exist_ok=True)
        entries = {case: [] for case in self.split['outer_train'][:81]}
        checksums = {}
        for index in range(642):
            name = f'entry_{index}.npz'
            path = bankdir/name
            path.write_bytes(f'CPU_DEBUG real-byte-entry{index}'.encode())
            checksums[name] = base.sha(path)
            entries[self.split['outer_train'][index % 81]].append(name)
        original = original_bank/'index.json'
        self.write_json(original, dict(entries_by_case=entries, candidate_count=128, total_candidates=642*128))
        bank = bankdir/'index.json'
        self.write_json(bank, dict(entries_by_case=entries, candidate_count=count, total_candidates=642*count))
        if hasattr(self, 'bank_patch'):
            self.bank_patch.stop()
        self.bank_patch = patch.object(base, 'BANK_SHA', base.sha(original))
        self.bank_patch.start(); self.addCleanup(self.bank_patch.stop)
        if count == 128:
            # Same index bytes and SHA, independent protected directory.
            self.assertEqual(base.sha(bank), base.BANK_SHA)
        admission_path = self.root/'admission.json'
        self.write_json(admission_path, dict(split=self.split, baseline=self.baseline,
            preprocessed=dict(cases=self.split['outer_train']+self.split['outer_val']),
            source_arrays_written=False, model_data_scale_preserved=True, debug=False))
        common = dict(gpu=gpu, arm=arm, trainer=trainer, bank=str(bank), bank_sha256=base.sha(bank),
            sources=642, candidates_per_source=count, copies_per_CP_event=copies, CP_probability=.5,
            seed=42, epochs=250, train_cases=105, validation_cases=26, physical_batch=2,
            checkpoint='checkpoint_best.pth', blend_border_preprocessed_voxels=0,
            readonly_admission=str(admission_path), readonly_admission_sha256=base.sha(admission_path))
        if arm == 'blend':
            common.update(blend_mode='constant', blend_alpha=.8)
        if count == 256:
            common.update(source_bank=str(original), source_bank_sha256=base.BANK_SHA)
        for value in (self.old, self.native):
            value.update(common)
            if arm == 'blend':
                path = Path(value['private_runtime'])/'nnunetv2/training/nnUNetTrainer/nnUNetTrainer_BasicCPBlend80.py'
                path.write_bytes(b'CPU_DEBUG identical explicit constant80 source')
        self.write_json(self.oldroot/'native.json', self.old)
        request = dict(common, uid=self.uid, controller_identity=self.actor)
        if count == 256:
            request.update(bank=str(original), bank_sha256=base.BANK_SHA)
        request['code_sha256'] = {}
        for filename in ('nnUNetTrainer_BasicCPAblations.py', 'nnUNetTrainer_BasicCPBlend80.py'):
            path = Path(self.old['private_runtime'])/'nnunetv2/training/nnUNetTrainer'/filename
            if path.exists():
                request['code_sha256']['custom_trainers/'+filename] = base.sha(path)
        self.write_json(self.oldroot/'request.json', request)
        self.train_actor = dict(pid=987655, birth=123., UID=self.uid,
            command=['python', 'CPU_DEBUG_train'], stage='train')
        self.write_json(self.oldroot/'train_actor.json', self.train_actor)
        model = base.read(self.oldroot/'actual_model.json'); model['trainer'] = trainer
        self.write_json(self.oldroot/'actual_model.json', model)
        receipts = [dict(case_id=case, source_entries=len(names), validated_positions=len(names)*count,
            three_paste_positions=len(names)*count if arm == 'multi3' else 0,
            entries_sha256={name: checksums[name] for name in names}) for case, names in entries.items()]
        preflight = dict(format=continuation.PREFLIGHT_FORMAT, arm=arm, complete=True,
            index_sha256=base.sha(bank), historical_trainer_sha256=base.HISTORICAL_SHA,
            historical_bank_sha256=base.BANK_SHA, source_entries=642, eligible_cases=81,
            full_training_cases=105, full_validation_cases=26, candidate_count=count,
            validated_positions=642*count, three_paste_positions=82176 if arm == 'multi3' else 0,
            CP_probability=.5, copies_per_event=copies, blend_border_preprocessed_voxels=0,
            original_gate_primary_RNG_schedule_preserved=True, GNN_scores_used=False,
            physical_batch_size=2, seed=42, epochs=250, network_patch_size=[128]*3,
            training_executed=False, actual_training_identifiers=self.split['outer_train'],
            entries_sha256=checksums, case_receipts=receipts,
            reuse_proof=dict(CPU_DEBUG='previous original provenance remains intact'))
        if arm == 'blend':
            preflight.update(blend_mode='constant', blend_alpha=.8, blend_recipient_weight=.2, label_blending=False)
        if count == 256:
            complete = dict(complete=True, index_sha256=base.sha(bank), original_index_sha256=base.BANK_SHA,
                all_original_arrays_and_first128_preserved=True, entries_sha256=checksums)
            self.write_json(bankdir/'complete.json', complete)
            preflight['extension_complete'] = complete
        self.write_json(self.oldroot/'preflight.json', preflight)
        self.oldfold = base.fold(self.old); self.oldfold.mkdir(parents=True, exist_ok=True)
        self.latest = self.oldfold/'checkpoint_latest.pth'; self.best = self.oldfold/'checkpoint_best.pth'
        for path, epoch in ((self.latest, 13), (self.best, 11)):
            value = self.saved(epoch); value['trainer_name'] = trainer
            value['logging']['ema_fg_dice'] = [.7]*epoch
            torch.save(value, path)
        self.log = b''.join((f'2026-10-13: Epoch {index}\n'
            f'[OnlineCP] epoch={index} applied=210/500\nEpoch time: 80 s\n').encode() for index in range(14))
        (self.oldroot/'train.log').write_bytes(self.log)
        self.native.pop('completed_native_resume', None)
        self.native['verified_native_continuation'] = dict(root=str(self.oldroot),
            request_sha256=base.sha(self.oldroot/'request.json'), native_sha256=base.sha(self.oldroot/'native.json'),
            preflight_sha256=base.sha(self.oldroot/'preflight.json'), controller_identity=self.actor,
            checkpoint_path=str(self.latest), checkpoint_sha256=base.sha(self.latest),
            best_sha256=base.sha(self.best), completed_epochs=13)

    def update_witness(self):
        witness = self.native['verified_native_continuation']
        witness.update(request_sha256=base.sha(self.oldroot/'request.json'),
            native_sha256=base.sha(self.oldroot/'native.json'),
            preflight_sha256=base.sha(self.oldroot/'preflight.json'),
            checkpoint_sha256=base.sha(witness['checkpoint_path']), best_sha256=base.sha(self.best))

    def test_reuse_complete_geometry_for_all_three_arms_keeps_original_provenance(self):
        for arm in continuation.ARMS:
            with self.subTest(arm=arm):
                self.configure(arm)
                before = {str(path): base.snapshot(path) for path in self.oldroot.rglob('*') if path.is_file()}
                result = continuation.reuse_preflight(self.native)
                self.assertFalse(result['continuation_reuse']['geometry_recomputed'])
                self.assertEqual(result['continuation_reuse']['full_source_SHA_reverified'], 642)
                self.assertEqual(result['reuse_proof'], base.read(self.oldroot/'preflight.json')['reuse_proof'])
                self.assertEqual(before, {str(path): base.snapshot(path) for path in self.oldroot.rglob('*') if path.is_file()})

    def test_whole13_copies_latest_and_earlier_best_independently_with_exact_log_prefix(self):
        before = {str(path): base.snapshot(path) for path in self.oldroot.rglob('*') if path.is_file()}
        result = continuation.prepare_resume(self.native)
        self.assertEqual(result['completed_epochs'], 13)
        self.assertEqual(result['native_goal_epochs'], 250)
        self.assertEqual(result['BEST_current_epoch_raw'], 11)
        self.assertEqual(result['remaining_epochs'], 237)
        self.assertFalse(result['partial_recovery_checkpoint_used'])
        for original, name in ((self.latest, 'checkpoint_latest.pth'), (self.best, 'checkpoint_best.pth')):
            copied = base.fold(self.native)/name
            self.assertEqual(copied.read_bytes(), original.read_bytes())
            self.assertNotEqual(base.snapshot(copied)[:2], base.snapshot(original)[:2])
        prefix = self.newroot/'resumed_training_prefix.log'
        self.assertEqual(base.sha(prefix), result['training_prefix_sha256'])
        self.assertTrue(self.log.startswith(prefix.read_bytes()))
        self.assertNotIn(b'Epoch 13\n', prefix.read_bytes())
        self.assertEqual(before, {str(path): base.snapshot(path) for path in self.oldroot.rglob('*') if path.is_file()})

    def test_original_best_can_be_selected_as_whole_latest_without_mutating_it(self):
        witness = self.native['verified_native_continuation']
        witness.update(checkpoint_path=str(self.best), checkpoint_sha256=base.sha(self.best), completed_epochs=11)
        result = continuation.prepare_resume(self.native)
        self.assertEqual(result['completed_epochs'], 11)
        self.assertEqual(base.sha(base.fold(self.native)/'checkpoint_latest.pth'), base.sha(self.best))

    def test_fixed80_and256_checkpoint_resume_preserve_full_contract(self):
        for arm in ('blend', 'candidates256'):
            with self.subTest(arm=arm):
                self.configure(arm)
                # Separate fresh namespaces for each successful copy.
                fresh = self.root/('new_'+arm); fresh.mkdir()
                self.native.update(root=str(fresh), native_root=str(fresh/'native'), private_runtime=str(fresh/'native/runtime'))
                source = Path(self.old['private_runtime'])
                import shutil
                shutil.copytree(source, Path(self.native['private_runtime']))
                result = continuation.prepare_resume(self.native)
                self.assertEqual(result['native_goal_epochs'], 250)
                self.assertTrue(result['optimizer_and_scaler_preserved'])

    def test_any_missing_source_or_changed_source_bytes_prevents_geometry_reuse(self):
        bank = Path(self.native['bank']).parent
        (bank/'entry_641.npz').write_bytes(b'CPU_DEBUG changed source bytes')
        with self.assertRaisesRegex(ValueError, 'SHA/stat changed'):
            continuation.reuse_preflight(self.native)

    def test_partial_case_receipt_or_missing_preflight_source_is_rejected(self):
        path = self.oldroot/'preflight.json'; result = base.read(path)
        result['case_receipts'][0]['validated_positions'] -= 128
        self.write_json(path, result); self.update_witness()
        with self.assertRaisesRegex(ValueError, 'Partial/mismatched'):
            continuation.reuse_preflight(self.native)

    def test_partial_checkpoint_logger_optimizer_scaler_and_ema_are_rejected(self):
        for missing in ('epoch_end_timestamps', 'ema_fg_dice', 'optimizer_state', 'grad_scaler_state'):
            with self.subTest(missing=missing):
                saved = torch.load(self.latest, map_location='cpu', weights_only=False)
                changed = copy.deepcopy(saved)
                if missing in changed['logging']:
                    changed['logging'][missing].pop()
                else:
                    changed[missing] = {}
                torch.save(changed, self.latest); self.update_witness()
                with self.assertRaises(ValueError):
                    continuation.prepare_resume(self.native)
                torch.save(saved, self.latest); self.update_witness()
        self.assertFalse((self.newroot/'resume.json').exists())

    def test_future_best_partial_backup_or_scientific_source_change_are_rejected(self):
        value = self.saved(14); value['trainer_name'] = self.native['trainer']
        torch.save(value, self.best); self.update_witness()
        with self.assertRaisesRegex(ValueError, 'BEST must'):
            continuation.prepare_resume(self.native)
        witness = self.native['verified_native_continuation']
        backup = Path(self.old['native_root'])/'recovery_partial_debug.pth'
        torch.save(value, backup)
        witness.update(checkpoint_path=str(backup), checkpoint_sha256=base.sha(backup))
        with self.assertRaisesRegex(ValueError, 'escapes own namespace'):
            continuation.prepare_resume(self.native)
        witness.update(checkpoint_path=str(self.latest), checkpoint_sha256=base.sha(self.latest))
        source = Path(self.native['private_runtime'])/'nnunetv2/training/nnUNetTrainer/nnUNetTrainer_BasicCPAblations.py'
        source.write_bytes(b'CPU_DEBUG changed paste semantics')
        with self.assertRaisesRegex(ValueError, 'scientific trainer bytes'):
            continuation.reuse_preflight(self.native)

    def test_live_original_actor_refuses_without_signals(self):
        process = Mock(); process.create_time.return_value = self.train_actor['birth']
        process.uids.return_value = type('uids', (), {'real': self.uid})()
        from psutil import NoSuchProcess
        def observed(pid):
            if pid == self.train_actor['pid']:
                return process
            raise NoSuchProcess(pid)
        with patch.object(base.psutil, 'Process', side_effect=observed):
            with self.assertRaisesRegex(RuntimeError, 'still present'):
                continuation.reuse_preflight(self.native)
        process.terminate.assert_not_called(); process.kill.assert_not_called()

    def test_changed_science_or_witness_sha_existing_result_and_incomplete_log_fail(self):
        changed = copy.deepcopy(self.native); changed['seed'] = 43
        with self.assertRaisesRegex(ValueError, 'seed'):
            continuation.prepare_resume(changed)
        changed = copy.deepcopy(self.native); changed['verified_native_continuation']['native_sha256'] = '0'*64
        with self.assertRaisesRegex(ValueError, 'SHA/stat changed'):
            continuation.prepare_resume(changed)
        (self.oldroot/'train.log').write_bytes(self.log.replace(b'Epoch time: 80 s\n', b'', 1))
        with self.assertRaisesRegex(ValueError, 'completion evidence'):
            continuation.prepare_resume(self.native)
        (self.oldroot/'train.log').write_bytes(self.log)
        destination = base.fold(self.native); destination.mkdir(parents=True)
        sentinel = destination/'old_result.txt'; sentinel.write_bytes(b'CPU_DEBUG existing result')
        with self.assertRaisesRegex(ValueError, 'Fresh untouched'):
            continuation.prepare_resume(self.native)
        self.assertEqual(sentinel.read_bytes(), b'CPU_DEBUG existing result')


if __name__ == '__main__':
    unittest.main()
