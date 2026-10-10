"""Overlay only own BEST scores onto the unchanged historical Basic CP bank.

The installed historical loader keeps its five random draws, uniform patient
source draw, normalized preprocessed hard paste and ordinary validation loader.
No raw-target payload, alternate donor, duplicate source patch or altered RNG
is introduced. CPU DEBUG fixtures are confined to private test entry points.
"""
from __future__ import annotations

import copy
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path, PurePosixPath
import sys
import types

import numpy as np

FORMAT = 'v24_own_BEST_scores_on_unchanged_Basic_81case642source128positions_v1'
RUNTIME_FORMAT = 'v24_historical_Basic_preprocessed_CP_scores_only_runtime_v1'
HISTORICAL_TRAINER_SHA256 = 'a3df56435fedff46fbc42a502596fdade2632d0211ea412630dba1555f32567d'
HISTORICAL_BANK_INDEX_SHA256 = '9e5be3e3ab8f04fd23c166b3267bc4efbfb4b321ef974920dfc08fbf5dd275e6'
TRAINER = 'nnUNetTrainer_250epochs_FrozenV23CP'
MODULE = 'nnunetv2.training.nnUNetTrainer.nnUNetTrainer_OnlinePairedCP'
_BINDINGS = {}


def _canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=False, allow_nan=False).encode('utf8')).hexdigest()


def _regular(path):
    path = Path(path).absolute()
    if path.resolve(strict=True) != path or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('Exact regular immutable Basic artifact required: ' + str(path))
    if not path.is_file():
        raise ValueError('Missing immutable Basic artifact: ' + str(path))
    return path


def _stat(path):
    value = Path(path).stat()
    return [value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns]


def _sha(path):
    result = hashlib.sha256()
    with _regular(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 2**20), b''):
            result.update(block)
    return result.hexdigest()


def _relative(root, name):
    relative = PurePosixPath(name)
    if (not isinstance(name, str) or not name or relative.is_absolute()
            or '..' in relative.parts or '\\' in name or relative.as_posix() != name):
        raise ValueError('Exact bank-relative source name required')
    path = _regular(Path(root).joinpath(*relative.parts))
    if not path.is_relative_to(Path(root)):
        raise ValueError('Basic source escapes admitted bank')
    return path


def _witness(path, checksum, identity, *, full_hash):
    path = _regular(path)
    before = _stat(path)
    if before != identity or (full_hash and _sha(path) != checksum):
        raise ValueError('Immutable Basic source changed: ' + str(path))
    if _stat(path) != before:
        raise ValueError('Basic source changed during verification')
    return path


def _split(split):
    if not isinstance(split, dict):
        raise ValueError('Full native split is required')
    train, val = split.get('outer_train'), split.get('outer_val')
    if (not isinstance(train, list) or not isinstance(val, list) or len(train) != 105
            or len(val) != 26 or len(set(train)) != 105 or len(set(val)) != 26
            or set(train) & set(val) or any(not isinstance(x, str) or not x or Path(x).name != x for x in train + val)):
        raise ValueError('Unchanged full105 train/full26 validation split required')
    return set(train), set(val)


def _scores(row):
    values = row.get('scores')
    if (not isinstance(values, list) or len(values) != 128
            or any(type(x) not in (float, int) for x in values)):
        raise ValueError('Exactly128 actual neural scores required')
    scores = np.asarray(values, dtype=np.float32)
    if not np.isfinite(scores).all() or row.get('selected_index') != int(np.argmax(scores)):
        raise ValueError('Finite128 scores and stable exact argmax required')
    scores.flags.writeable = False
    return scores


def _validate_overlay(path, *, full_hash=False, expected_index_sha256=HISTORICAL_BANK_INDEX_SHA256,
                      validate_pin=True):
    """Internal DEBUG tests may supply their explicit fixture identity."""
    path = _regular(path)
    before = _stat(path)
    overlay = json.loads(path.read_text(encoding='utf8'))
    if (overlay.get('format') != FORMAT or overlay.get('complete') is not True
            or overlay.get('debug') is not False or overlay.get('complete_cases') != 81
            or overlay.get('complete_sources') != 642 or overlay.get('complete_positions') != 82176
            or overlay.get('only_CP_field_changed') != 'scores'
            or overlay.get('original_non_score_payloads_preserved') is not True
            or overlay.get('original_source_schedule_and_paste_contract_preserved') is not True
            or overlay.get('full_P_context') is not True
            or overlay.get('recipient_GT_used_in_forward') is not False
            or overlay.get('production_optimizer_updates') != 0
            or overlay.get('model_and_RNG_unchanged') is not True
            or overlay.get('content_sha256') != _canonical({k:v for k,v in overlay.items() if k != 'content_sha256'})):
        raise ValueError('Complete unchanged Basic bank/own neural score-only proof required')
    if validate_pin:
        from .v24_nnunet_cp import validate_current_pin
        pin = validate_current_pin(overlay['pin'])
        if overlay.get('model_sha256') != pin['selected']['best_model_sha256']:
            raise ValueError('Overlay neural model must be this completed arm BEST')
    train, val = _split(overlay.get('split'))
    source = overlay['source_bank']
    root = Path(source['root']).absolute()
    if root.resolve(strict=True) != root or any(p.is_symlink() for p in (root, *root.parents)):
        raise ValueError('Exact immutable original Basic bank directory required')
    index = _witness(root / 'index.json', source['index_sha256'], source['index_stat_identity'], full_hash=True)
    if expected_index_sha256 is not None and source['index_sha256'] != expected_index_sha256:
        raise ValueError('Authoritative historical Basic index SHA required')
    original = json.loads(index.read_text(encoding='utf8'))
    if (original.get('format') != 'hiercp_online_bank_v2' or original.get('paste_contract') is not None
            or original.get('candidate_count') != 128 or original.get('cp_probability') != .5
            or original.get('source_entries') != 642 or original.get('eligible_cases') != 81
            or original.get('total_candidates') != 82176 or original.get('network_patch_size') != [128]*3
            or original.get('tumor_label') != 2 or original.get('liver_label') != 1
            or original.get('maximum_diameter_mm') != 20. or original.get('minimum_liver_coverage') != .85
            or original.get('intensity_scale_range') != [.95, 1.05]
            or original.get('intensity_shift_range_hu') != [-5., 5.]):
        raise ValueError('Original Basic preprocessed patch/CP contract required')
    groups = original['entries_by_case']
    if (overlay.get('entries_by_case') != groups or len(groups) != 81
            or not set(groups) <= train or set(groups) & val
            or any(not isinstance(names, list) or not names for names in groups.values())):
        raise ValueError('Original81 eligible cases and ordered same-patient source lists required')
    names = [name for items in groups.values() for name in items]
    if (len(names) != 642 or len(set(names)) != 642
            or set(overlay.get('entries', {})) != set(names)
            or set(source.get('entries', {})) != set(names)
            or source.get('complete_cases') != 81 or source.get('complete_sources') != 642
            or source.get('complete_positions') != 82176):
        raise ValueError('All642 unchanged source entries/82176 positions required')
    score_arrays = {}
    for case, items in groups.items():
        for name in items:
            row, witness = overlay['entries'][name], source['entries'][name]
            if (row.get('case_id') != case or witness.get('case_id') != case
                    or row.get('original_sha256') != witness.get('sha256')
                    or row.get('source_component') != witness.get('source_component')
                    or type(row.get('source_component')) is not int or row['source_component'] < 1):
                raise ValueError('Own-patient original source component/identity changed')
            _witness(_relative(root, name), witness['sha256'], witness['stat_identity'], full_hash=full_hash)
            score_arrays[name] = _scores(row)
    if _stat(path) != before:
        raise ValueError('Sealed score overlay changed during admission')
    return dict(path=str(path), sha256=_sha(path), stat=before, overlay=overlay,
        original=original, source_root=str(root), index=str(index), scores=score_arrays)


def validate_overlay(path, *, full_hash=False):
    return _validate_overlay(path, full_hash=full_hash)


def _guard_binding(binding, *, full_hash=False):
    _witness(binding['path'], binding['sha256'], binding['stat'], full_hash=full_hash)
    source = binding['overlay']['source_bank']
    _witness(binding['index'], source['index_sha256'], source['index_stat_identity'], full_hash=full_hash)
    for name, witness in source['entries'].items():
        _witness(_relative(binding['source_root'], name), witness['sha256'], witness['stat_identity'], full_hash=full_hash)


def _function_proof(function):
    from .v24_native_best_eval_runtime import _function_source_proof
    return _function_source_proof(function)


def _install_overlay(module, binding, *, expected_source_sha256):
    """The original loader's global bank is replaced; loader code is untouched."""
    module_path = _regular(module.__file__)
    if _sha(module_path) != expected_source_sha256:
        raise ValueError('Exact original installed historical preprocessed trainer source required')
    if 'onlinecp_raw_target_paste_v1' in module_path.read_text(encoding='utf8'):
        raise ValueError('Raw-target trainer cannot establish historical Basic paste parity')
    existing = _BINDINGS.get(module)
    if existing is not None:
        if existing['binding']['sha256'] != binding['sha256'] or module.OnlineCPBank is not existing['bank_class']:
            raise ValueError('A native process cannot switch its admitted own-arm score overlay')
        existing['guard']()
        return copy.deepcopy(existing['proof'])
    original = module.OnlineCPBank
    loader = module.nnUNetDataLoaderOnlineCP
    trainer = module._nnUNetTrainer_250epochs_OnlineCP
    functions = [original.__init__, original._load, original.load_for_case,
        loader.__init__, loader._sample_paste_plan, loader._select_candidate,
        loader.generate_train_batch, trainer.initialize, trainer.get_dataloaders]
    source_proofs = [_function_proof(function) for function in functions]
    code_objects = tuple(function.__code__ for function in functions)
    counters = dict(source_loads=0, original_source_visits={}, overlay_score_loads=0)

    def source_guard():
        _witness(module_path, expected_source_sha256, source_proofs[0]['stat'], full_hash=False)
        from .v24_readonly_native_storage import code_fingerprint
        for function, code, proof in zip(functions, code_objects, source_proofs):
            if function.__code__ is not code or code_fingerprint(function.__code__) != proof['code_sha256']:
                raise ValueError('Original Basic loader/RNG/paste bytecode changed')

    class MatchedOverlayBank(original):
        def __init__(self, index_path, cache_entries=64):
            requested = _regular(index_path)
            if requested != Path(binding['path']):
                raise ValueError('Historical trainer must receive this own-arm score overlay')
            source_guard()
            _witness(requested, binding['sha256'], binding['stat'], full_hash=False)
            original.__init__(self, binding['index'], cache_entries)
            if self.metadata != binding['original']:
                raise ValueError('Original Basic bank index changed during original construction')
            self._v24_matched_overlay = binding['sha256']

        def _load(self, name):
            source_guard()
            if name not in binding['scores']:
                raise ValueError('CP source is outside this complete matched Basic bank')
            source = binding['overlay']['source_bank']['entries'][name]
            path = _relative(binding['source_root'], name)
            _witness(path, source['sha256'], source['stat_identity'], full_hash=False)
            entry = original._load(self, name)
            _witness(path, source['sha256'], source['stat_identity'], full_hash=False)
            if ('paste_contract' in entry or entry['scores'].shape != (128,)
                    or entry['candidate_centers'].shape != (128, 3)
                    or entry.get('candidate_raw_centers', np.empty(0)).shape != (128, 3)
                    or int(entry['source_component'][0]) != source['source_component']):
                raise ValueError('Original preprocessed source/128 positions required')
            # Do not mutate the original cache; all patch/mask/geometry arrays
            # retain their identity and bytes. The only replaced array is scores.
            result = dict(entry)
            result['scores'] = binding['scores'][name]
            counters['source_loads'] += 1
            counters['overlay_score_loads'] += 1
            counters['original_source_visits'][name] = counters['original_source_visits'].get(name, 0) + 1
            return result

    module.OnlineCPBank = MatchedOverlayBank
    proof = dict(format=RUNTIME_FORMAT, overlay=binding['path'], overlay_sha256=binding['sha256'],
        original_bank_index=binding['index'], original_bank_index_sha256=binding['overlay']['source_bank']['index_sha256'],
        original_trainer_sha256=expected_source_sha256, original_method_sources=source_proofs,
        original_loader_code_preserved=True, original_preprocessed_paste_preserved=True,
        original_source_rng_preserved=True, random_draws_per_visit=5, cp_probability=.5,
        source_policy='same_patient_uniform_original_sources_per_visit', location_policy='GNN_argmax_of_original128',
        eligible_cases=81, source_entries=642, scored_candidates=82176,
        training_cases=105, validation_cases=26, unchanged_no_CP_training_cases=24,
        original_arrays_written=False, duplicate_source_arrays_created=False, only_CP_field_changed='scores')
    _BINDINGS[module] = dict(binding=binding, bank_class=MatchedOverlayBank,
        proof=proof, guard=source_guard, counters=counters)
    source_guard()
    return copy.deepcopy(proof)


def install_matched_overlay(overlay_path, *, trainer_module=None):
    binding = validate_overlay(overlay_path, full_hash=True)
    module = importlib.import_module(MODULE) if trainer_module is None else trainer_module
    return _install_overlay(module, binding, expected_source_sha256=HISTORICAL_TRAINER_SHA256)


def runtime_observation(*, trainer_module=None):
    module = importlib.import_module(MODULE) if trainer_module is None else trainer_module
    if module not in _BINDINGS:
        raise ValueError('Matched historical Basic score overlay is not installed')
    installed = _BINDINGS[module]
    installed['guard']()
    return dict(proof=copy.deepcopy(installed['proof']), counters=copy.deepcopy(installed['counters']))


def install_native_runtime(native_path, *, training=False):
    """Install immutable full131 reader and historical CP; calibration is separate."""
    path = _regular(native_path)
    native = json.loads(path.read_text(encoding='utf8'))
    overlay_path = _regular(native['overlay'])
    if (_sha(overlay_path) != native['overlay_sha256'] or native.get('bank') != str(overlay_path)
            or native.get('bank_sha256') != native['overlay_sha256'] or native.get('trainer') != TRAINER):
        raise ValueError('Own native output must bind its sealed matched score overlay')
    binding = validate_overlay(overlay_path, full_hash=False)
    if native.get('split', native['baseline']['split']) != binding['overlay']['split']:
        raise ValueError('Native matched full105/26 split changed')
    if native['physical_GPU'] != binding['overlay']['pin']['physical_GPU']:
        raise ValueError('Native GPU differs from this own-arm BEST')
    baseline = native['baseline']
    if (baseline.get('physical_batch') != 2 or baseline.get('patch_size') != [128]*3
            or baseline.get('epochs') != 250 or baseline.get('cp_probability') != .5):
        raise ValueError('Unchanged full250/B2/128cube/CP.5 native configuration required')
    original_admission = native.get('original_storage_admission', native['storage_admission'])
    expected_admission = native.get('original_storage_admission_sha256', native['storage_admission_sha256'])
    if _sha(original_admission) != expected_admission:
        raise ValueError('Original full131 read-only admission changed')
    module = importlib.import_module(MODULE)
    private = Path(native['private_runtime']).absolute()
    expected_module = private / 'nnunetv2/training/nnUNetTrainer/nnUNetTrainer_OnlinePairedCP.py'
    if _regular(module.__file__) != _regular(expected_module):
        raise ValueError('Native must use its exact private original historical CP runtime')
    for name, checksum in native['private_trainer_sha256'].items():
        relative = name if '/' in name else 'training/nnUNetTrainer/' + name
        if _sha(private / 'nnunetv2' / relative) != checksum:
            raise ValueError('Private historical/alias trainer source changed')
    from .v24_readonly_native_storage import install_dataset_adapter
    readonly = install_dataset_adapter(original_admission, native['private_data_folder'])
    from nnunetv2.training.dataloading import nnunet_dataset
    module.infer_dataset_class = nnunet_dataset.infer_dataset_class
    overlay_proof = install_matched_overlay(overlay_path, trainer_module=module)
    alias = importlib.import_module('nnunetv2.training.nnUNetTrainer.nnUNetTrainer_FrozenV23CP')
    cls = getattr(alias, TRAINER)
    if not issubclass(cls, module.nnUNetTrainer_250epochs_OnlineHierCPExactArgmax):
        raise ValueError('Matched trainer alias must inherit original historical exact argmax trainer')
    result = dict(overlay=overlay_proof, preprocessed=readonly, training=bool(training))
    if training:
        from . import v24_readonly_static_operator_storage as static
        result['checkpoint_coordination'] = static.install_checkpoint_adapters(original_admission, private, cls)
        from .v24_native_best_eval_runtime import install_best_evaluation_guard
        result['BEST_evaluation'] = install_best_evaluation_guard()
        original_start, original_epoch = cls.on_train_start, cls.on_train_epoch_start
        def start(self):
            from .v24_nnunet_cp import require_project_budget, snapshot
            if not self.was_initialized:
                self.initialize()
            train, val = self.do_split()
            if (set(train) != set(binding['overlay']['split']['outer_train'])
                    or set(val) != set(binding['overlay']['split']['outer_val'])
                    or self.batch_size != 2 or self.num_epochs != 250
                    or list(self.configuration_manager.patch_size) != [128]*3
                    or self.online_policy != 'hier_argmax'):
                raise ValueError('Actual original full105/26/B2/250/128cube matched CP required')
            require_project_budget()
            _guard_binding(binding)
            started = original_start(self)
            from .v24_readonly_native_storage import publish_json
            workers = int(os.environ['nnUNet_n_proc_DA'])
            if workers != 4:
                raise ValueError('Measured four native data workers required')
            publish_json(Path(native['root']) / 'actual_nnunet_configuration.json', dict(
                format=RUNTIME_FORMAT, model=str(self.network),
                parameters=sum(p.numel() for p in self.network.parameters()),
                trainable_parameters=sum(p.numel() for p in self.network.parameters() if p.requires_grad),
                train_cases=len(train), validation_cases=len(val), epochs=self.num_epochs,
                physical_batch=self.batch_size, gradient_accumulation=1, effective_batch=self.batch_size,
                patch_size=list(self.configuration_manager.patch_size),
                train_steps_per_epoch=self.num_iterations_per_epoch,
                validation_steps_per_epoch=self.num_val_iterations_per_epoch, CPU_workers=workers,
                cp_probability=.5, source_policy='same_patient_uniform_original_sources_per_visit',
                candidate_count=128, source_entries=642, eligible_cases=81,
                validation_checkpoint='checkpoint_best.pth', runtime_proofs=result,
                observed=runtime_observation(trainer_module=module), resources=snapshot(),
                model_weights_fresh=True, debug=False, full_training_complete=False))
            return started
        def epoch(self):
            from .v24_nnunet_cp import require_project_budget
            require_project_budget()
            _guard_binding(binding)
            return original_epoch(self)
        cls.on_train_start = start
        cls.on_train_epoch_start = epoch
    return result


def run_training_entry():
    if sys.argv[1:].count('--val_best') != 1:
        raise ValueError('Matched CP nnU-Net evaluation requires exact BEST; no FINAL fallback')
    native_path = os.environ['V24_MATCHED_NATIVE']
    native = json.loads(_regular(native_path).read_text(encoding='utf8'))
    if _regular(os.environ['ONLINE_CP_BANK']) != _regular(native['overlay']):
        raise ValueError('Native training environment changed the admitted own score overlay')
    install_native_runtime(native_path, training=True)
    from nnunetv2.run.run_training import run_training_entry as original_entry
    return original_entry()
