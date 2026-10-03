"""Materialize exact v1 sources and execute one explicitly requested stage.

No current v1/v2 module is patched. Shared preparation runs once; model results
and checkpoints belong to separate stages. This module never starts nnU-Net.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile

from .contracts import (STAGES, TARGET_CONTRACT, ContractError, canonical_hash,
                        compare_reports, get_stage, make_stage_config,
                        make_run_contract, validate_resume,
                        local_sampling_spec,
                        resolve_execution_config, validate_stage_config,
                        verify_archive)

ROOT = Path(__file__).resolve().parents[1]
FORMAT = "hiercp_v1x_suite_v1"
SAMPLING_FORMAT = "hiercp_v1x_sampling_suite_v1"
SAMPLING_HELPERS = (
    "hiercp_v1x/sampling_runtime.py", "hiercp_v1x/sampling_entry.py",
    "hiercp_v1x/graph_size.py", "hiercp_v1x/nested_graph_size.py",
)
EPOCH_RECORDING_HELPERS = (
    "hiercp_v1x/epoch_telemetry.py", "hiercp_v1x/telemetry_entry.py",
)
ROI_RECOVERY_HELPERS = (
    "hiercp_v1x/cache_budget_recovery.py", "hiercp_v1x/budget_recovery_entry.py",
)
EPOCH_RECORDING = {
    "format": "hiercp_v1_epoch_recording_v1",
    "scope": "identical observational instrumentation; original model, sampler, loss and RNG calls retained",
}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    import hashlib
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(data)
    return h.hexdigest()


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)


def overlay(stage, name='hiercp/model.py'):
    if stage == "v1.0":
        return b""
    if name == 'hiercp/contracts.py':
        return ("\n\n# Stage-specific serialization adapter; original geometry/patient validation remains.\n"
                "_V1RequireCurrentCheckpoint = require_current_checkpoint\n"
                "def require_current_checkpoint(payload):\n"
                f"    expected = ARCHITECTURE_VERSION + '|v1x_{stage}_r1'\n"
                "    if not isinstance(payload, Mapping) or payload.get('architecture_version') != expected:\n"
                "        raise ValueError('Checkpoint belongs to another v1.x stage')\n"
                "    state = payload.get('state_dict', {})\n"
                "    marker = state.get('_v1x_stage_revision') if isinstance(state, Mapping) else None\n"
                f"    if marker is None or not hasattr(marker, 'numel') or marker.numel() != 1 or int(marker.item()) != {tuple(STAGES).index(stage)}:\n"
                "        raise ValueError('Checkpoint has no matching v1.x stage state')\n"
                "    original_metadata = dict(payload)\n"
                "    original_metadata['architecture_version'] = ARCHITECTURE_VERSION\n"
                "    _V1RequireCurrentCheckpoint(original_metadata)\n").encode('utf-8')
    if name != 'hiercp/model.py':
        return b''
    return ("\n\n# Explicit isolated v1.x experiment overlay; original source above is unchanged.\n"
            "_V1ExactPlacementModel = HierarchicalPyGPlacementModel\n"
            "class HierarchicalPyGPlacementModel(_V1ExactPlacementModel):\n"
            "    def __init__(self, **kwargs):\n"
            "        super().__init__(**kwargs)\n"
            "        from hiercp_v1x.models import apply_variant\n"
            f"        apply_variant(self, {stage!r})\n").encode("utf-8")


def sampling_overlay(name):
    """Only append the declared strict sampler hooks to isolated snapshots."""
    from .sampling_runtime import SAMPLING_IMPORT_HOOK
    if name == 'hiercp/sample.py':
        return SAMPLING_IMPORT_HOOK
    if name == 'hiercp/model.py':
        return ("\n\n# Explicit strict-nested checkpoint identity; model mathematics stay unchanged.\n"
                "_V1SamplingPlacementModel = HierarchicalPyGPlacementModel\n"
                "class HierarchicalPyGPlacementModel(_V1SamplingPlacementModel):\n"
                "    def __init__(self, **kwargs):\n"
                "        super().__init__(**kwargs)\n"
                "        from hiercp_v1x.sampling_runtime import apply_sampling_identity, load_environment_contract\n"
                "        apply_sampling_identity(self, load_environment_contract())\n").encode('utf-8')
    if name == 'hiercp/contracts.py':
        return ("\n\n# Strict-nested payload guard delegates all original geometry/stage checks.\n"
                "_V1SamplingRequireCurrentCheckpoint = require_current_checkpoint\n"
                "def require_current_checkpoint(payload):\n"
                "    from hiercp_v1x.sampling_runtime import validate_checkpoint_identity, load_environment_contract\n"
                "    contract = load_environment_contract()\n"
                "    validate_checkpoint_identity(payload, contract)\n"
                "    original_metadata = dict(payload)\n"
                "    suffix = '|sampling_' + contract['contract_sha256']\n"
                "    original_metadata['architecture_version'] = payload['architecture_version'][:-len(suffix)]\n"
                "    _V1SamplingRequireCurrentCheckpoint(original_metadata)\n").encode('utf-8')
    return b''


def _reference(manifest):
    return manifest.get('reference_experiment')


def preparation_root(experiment, manifest=None):
    """Canonical preparation belongs only to the matched native experiment."""
    experiment = Path(experiment).resolve()
    manifest = load_suite(experiment) if manifest is None else manifest
    reference = _reference(manifest)
    return (Path(reference['path']) if reference else experiment) / 'shared'


def execution_reference(experiment, manifest=None):
    experiment = Path(experiment).resolve()
    manifest = load_suite(experiment) if manifest is None else manifest
    reference = _reference(manifest)
    return Path(reference['path']) if reference else experiment


def sampler_contract(experiment, stage, manifest=None):
    manifest = load_suite(experiment) if manifest is None else manifest
    return manifest.get('sampling_contracts', {}).get(stage)


def execution_is_resolved(manifest, stage):
    return stage != 'v1.0' or _reference(manifest) is not None


def normalize_split(payload):
    if 'seed' in payload and payload['seed'] != 42:
        raise ContractError('The input split declares a seed different from42')
    if "inner_train" in payload:
        train, val = payload["inner_train"], payload["inner_val"]
        excluded = payload["outer_val"]
        if set(train) | set(val) != set(payload["outer_train"]):
            raise ContractError("Inner split does not cover the complete outer training cohort")
    else:
        train, val = payload["train"], payload["val"]
        excluded = payload.get("outer_validation_excluded", [])
        if payload.get("seed") != 42:
            raise ContractError("The common split must use seed42")
    for name, ids in (("train", train), ("val", val), ("excluded", excluded)):
        if not isinstance(ids, list) or len(ids) != len(set(ids)) or any(not isinstance(x, str) for x in ids):
            raise ContractError(f"Invalid {name} case IDs")
    if not train or not val or (set(train) & set(val)) or ((set(train) | set(val)) & set(excluded)):
        raise ContractError("Empty or overlapping train/validation/outer-test split")
    return dict(format="hiercp_case_split_v1", seed=42, val_fraction=len(val)/(len(train)+len(val)),
                train=list(train), val=list(val), outer_validation_excluded=list(excluded))


def initialize(experiment, medical, split_file, *, repo=ROOT, local_sampling=None,
               sampling_profile=None, reference_experiment=None, record_epochs=False,
               preparation_admission=None, recover_preparation_from=None):
    experiment, medical = Path(experiment).resolve(), Path(medical).resolve()
    if experiment == Path(repo).resolve() or experiment == medical:
        raise ContractError("Experiment must be a separate output directory")
    for protected in (medical / "Data", medical / "Task03_Liver", Path(repo) / "versions"):
        if experiment == protected or protected in experiment.parents:
            raise ContractError(f"Protected experiment destination: {protected}")
    proof = verify_archive(repo)
    if preparation_admission is not None:
        from .contracts import validate_preparation_admission
        validate_preparation_admission(preparation_admission, root=repo)
    incoming = Path(split_file).resolve()
    split = normalize_split(read(incoming))
    sampling = local_sampling_spec(local_sampling, sampling_profile) if local_sampling is not None else None
    if type(record_epochs) is not bool or (record_epochs and sampling is None):
        raise ContractError('Epoch recording requires explicit native/strict_nested sampling')
    recording = dict(EPOCH_RECORDING) if record_epochs else None
    if sampling is None and (sampling_profile is not None or reference_experiment is not None):
        raise ContractError('Sampling profile/reference requires an explicit local sampling mode')
    reference = None
    if sampling is not None and sampling['mode'] == 'strict_nested':
        if reference_experiment is None:
            raise ContractError('Strict nested comparison requires an explicit matched native reference experiment')
        reference_path = Path(reference_experiment).resolve()
        if reference_path == experiment:
            raise ContractError('A nested experiment cannot reference itself')
        reference_manifest = load_suite(reference_path, repo=repo)
        if (reference_manifest.get('sampling_contract', {}).get('mode') != 'native'
                or _reference(reference_manifest) is not None):
            raise ContractError('Reference must be an explicit native sampling experiment with the matched RNG boundary')
        if reference_manifest['medical_root'] != str(medical) or reference_manifest['split'] != split:
            raise ContractError('Native reference has a different data/split contract')
        if reference_manifest.get('epoch_recording') != recording:
            raise ContractError('Matched arms must use the same epoch recording policy')
        if reference_manifest.get('preparation_admission') != preparation_admission:
            raise ContractError('Matched arms must use the same explicit ROI resource admission')
        reference = dict(path=str(reference_path), manifest_sha256=reference_manifest['manifest_sha256'])
    elif reference_experiment is not None:
        raise ContractError('Only strict nested comparisons can reuse a native reference experiment')
    recovery = None
    if recover_preparation_from is not None:
        if preparation_admission is None or sampling is None or sampling['mode'] != 'native':
            raise ContractError('Cache recovery requires explicit native sampling and measured ROI admission')
        old_root = Path(recover_preparation_from).resolve()
        old = load_suite(old_root, repo=repo)
        if (old_root == experiment or old_root in experiment.parents or experiment in old_root.parents
                or old.get('sampling_contract', {}).get('mode') != 'native'
                or old['medical_root'] != str(medical) or old['split'] != split):
            raise ContractError('Recovery must use a disjoint, matched failed native experiment')
        old_shared = preparation_root(old_root, old)
        if (old_shared/'prepare.lock').exists() or any((old_root/'results').glob('*/run.lock')):
            raise ContractError('Recovery source is active; no concurrent source mutation permitted')
        cache = old_shared/'cache'
        if read(cache/'config.json').get('state') != 'failed' or any((cache/name).exists() for name in ('index.json','complete.json')):
            raise ContractError('Recovery source must be failed and unpublished')
        recovery = dict(path=str(old_root), manifest_sha256=old['manifest_sha256'],
                        cache_config_sha256=digest(cache/'config.json'),
                        progress_sha256=digest(cache/'manifest.csv'))
    data = medical / "Data"
    images, labels, unsupported = {}, {}, []
    for kind, table in (("image", images), ("labels", labels)):
        for p in (data/kind).glob('*.nii*'):
            case = p.name.removesuffix('.nii.gz').removesuffix('.nii')
            if kind == 'image': case = case.removesuffix('_0000')
            if case in table:
                raise ContractError(f'Ambiguous native {kind} files for {case}')
            table[case] = p
            suffix = '_0000.nii.gz' if kind == 'image' else '.nii.gz'
            if not p.name.endswith(suffix):
                unsupported.append(str(p))
    if unsupported:
        raise ContractError(f'Original v1 requires native _0000.nii.gz images and .nii.gz labels: {unsupported}')
    expected = set(split["train"]) | set(split["val"]) | set(split["outer_validation_excluded"])
    if set(images) != expected or set(labels) != expected:
        raise ContractError("The split must account for every image and label; no hidden subset")
    if experiment.exists():
        existing = load_suite(experiment, repo=repo)
        if (existing["medical_root"] != str(medical) or existing["split"] != split
                or existing.get('sampling_contract') != sampling
                or _reference(existing) != reference
                or existing.get('epoch_recording') != recording
                or existing.get('preparation_admission') != preparation_admission
                or existing.get('preparation_recovery') != recovery):
            raise ContractError("Existing experiment has a different data/split contract")
        return existing
    experiment.mkdir(parents=True, exist_ok=False)
    manifests = {}
    helper = Path(repo) / "hiercp_v1x" / "models.py"
    if not helper.is_file():
        raise FileNotFoundError(helper)
    sampling_contracts = {}
    if sampling is not None:
        from .sampling_runtime import create_contract
        for name in SAMPLING_HELPERS:
            if not (Path(repo) / name).is_file():
                raise FileNotFoundError(Path(repo) / name)
    if record_epochs:
        for name in EPOCH_RECORDING_HELPERS:
            if not (Path(repo) / name).is_file():
                raise FileNotFoundError(Path(repo) / name)
    with ZipFile(Path(repo) / "versions/v1/pipeline_v1_source.zip") as zipped:
        for stage in STAGES:
            source = experiment / "source" / stage
            source.mkdir(parents=True, exist_ok=False)
            hashes = {}
            for name in zipped.namelist():
                payload = zipped.read(name)
                payload += overlay(stage, name)
                if sampling is not None and sampling['mode'] == 'strict_nested':
                    payload += sampling_overlay(name)
                target = source / name
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("xb") as out:
                    out.write(payload)
                hashes[name] = digest(target)
            helpers = [("hiercp_v1x/__init__.py", b""), ("hiercp_v1x/models.py", helper.read_bytes())]
            if sampling is not None:
                helpers.extend((name, (Path(repo) / name).read_bytes()) for name in SAMPLING_HELPERS)
            if record_epochs:
                helpers.extend((name, (Path(repo) / name).read_bytes()) for name in EPOCH_RECORDING_HELPERS)
            if preparation_admission is not None:
                helpers.extend((name, (Path(repo) / name).read_bytes()) for name in ROI_RECOVERY_HELPERS)
            for name, payload in helpers:
                target = source / name
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("xb") as out:
                    out.write(payload)
                hashes[name] = digest(target)
            config = make_stage_config(proof["base_config"], stage, root=repo,
                                       preparation_admission=preparation_admission)
            write_new(experiment / "configs" / f"{stage}.json", config)
            if sampling is not None:
                contract = create_contract(sampling['mode'], config['graph'],
                    source_identity=dict(files=hashes,
                                         native_sample_sha256=proof['file_hashes']['hiercp/sample.py']),
                    profile=sampling['profile'])
                sampling_contracts[stage] = contract
                write_new(experiment / 'sampling' / f'{stage}.json', contract)
            manifests[stage] = dict(spec=STAGES[stage].to_dict(), source_hashes=hashes,
                                    config_sha256=canonical_hash(config),
                                    source_original_exact=stage == "v1.0" and (sampling is None or sampling['mode'] == 'native'),
                                    predecessor=STAGES[stage].predecessor)
    write_new(experiment / "shared/split.json", split)
    manifest = dict(format=SAMPLING_FORMAT if sampling is not None else FORMAT, archive_sha256=proof["archive_sha256"],
                    revision=proof["revision"], medical_root=str(medical),
                    split=split, input_split_sha256=digest(incoming),
                    split_sha256=canonical_hash(split), target_contract=TARGET_CONTRACT,
                    stages=manifests, shared_preparation=True, long_training_started=False,
                    historical_v1_accuracy_reproduced=False, production_ready=False,
                    future_v2_observed_P_U_contract="unchanged; separate direction, not relabelled here")
    if sampling is not None:
        manifest.update(sampling_contract=sampling, sampling_contracts=sampling_contracts,
                        reference_experiment=reference,
                        shared_preparation=reference is None,
                        graph_quality_passed=False, matched_graph_comparison_only=True)
    if recording is not None:
        manifest['epoch_recording'] = recording
    if preparation_admission is not None:
        manifest['preparation_admission'] = preparation_admission
    if recovery is not None:
        manifest['preparation_recovery'] = recovery
    manifest["manifest_sha256"] = canonical_hash(manifest)
    write_new(experiment / "manifest.json", manifest)
    return manifest


def load_suite(experiment, *, repo=ROOT, _seen=None):
    experiment = Path(experiment).resolve()
    seen = set() if _seen is None else set(_seen)
    if experiment in seen:
        raise ContractError('Experiment reference cycle is not permitted')
    seen.add(experiment)
    manifest = read(experiment / "manifest.json")
    plain = dict(manifest)
    expected = plain.pop("manifest_sha256", None)
    if manifest.get("format") not in (FORMAT, SAMPLING_FORMAT) or canonical_hash(plain) != expected:
        raise ContractError("Suite manifest identity mismatch")
    proof = verify_archive(repo)
    if manifest["archive_sha256"] != proof["archive_sha256"] or manifest["target_contract"] != TARGET_CONTRACT:
        raise ContractError("Suite changed original source or target contract")
    if set(manifest['stages']) != set(STAGES):
        raise ContractError('Suite does not contain every declared stage')
    if read(experiment / "shared/split.json") != manifest["split"]:
        raise ContractError("Common split changed")
    sampling = manifest.get('sampling_contract')
    recording = manifest.get('epoch_recording')
    admission = manifest.get('preparation_admission')
    if admission is not None:
        from .contracts import validate_preparation_admission
        validate_preparation_admission(admission, root=repo)
    recovery = manifest.get('preparation_recovery')
    if recovery is not None:
        if admission is None or sampling is None or sampling.get('mode') != 'native':
            raise ContractError('Recovery identity lacks explicit native ROI admission')
        old = load_suite(recovery['path'], repo=repo, _seen=seen)
        old_shared = preparation_root(recovery['path'], old)
        if (old['manifest_sha256'] != recovery['manifest_sha256']
                or old['medical_root'] != manifest['medical_root'] or old['split'] != manifest['split']
                or digest(old_shared/'cache/config.json') != recovery['cache_config_sha256']
                or digest(old_shared/'cache/manifest.csv') != recovery['progress_sha256']):
            raise ContractError('Failed native recovery source changed')
    if recording is not None and (recording != EPOCH_RECORDING or sampling is None):
        raise ContractError('Epoch recording policy changed or lacks explicit sampling')
    if manifest['format'] == SAMPLING_FORMAT:
        if (not isinstance(sampling, dict)
                or sampling != local_sampling_spec(sampling.get('mode'), sampling.get('profile'))
                or set(manifest.get('sampling_contracts', {})) != set(STAGES)):
            raise ContractError('Sampling suite contract is incomplete or changed')
        reference = _reference(manifest)
        if sampling['mode'] == 'strict_nested':
            if not isinstance(reference, dict) or set(reference) != {'path', 'manifest_sha256'}:
                raise ContractError('Strict nested suite requires its immutable native reference')
            parent = load_suite(reference['path'], repo=repo, _seen=seen)
            if (parent.get('sampling_contract', {}).get('mode') != 'native'
                    or _reference(parent) is not None
                    or parent['manifest_sha256'] != reference['manifest_sha256']
                    or parent['medical_root'] != manifest['medical_root']
                    or parent['split'] != manifest['split']
                    or parent.get('epoch_recording') != recording
                    or parent.get('preparation_admission') != admission):
                raise ContractError('Native reference identity, data or split changed')
        elif reference is not None:
            raise ContractError('Native sampler must own its preparation and calibration')
    elif sampling is not None or _reference(manifest) is not None:
        raise ContractError('Legacy suite cannot silently adopt a sampling contract')
    for stage, item in manifest["stages"].items():
        expected_names = set(proof['file_hashes']) | {'hiercp_v1x/__init__.py', 'hiercp_v1x/models.py'}
        if sampling is not None:
            expected_names.update(SAMPLING_HELPERS)
        if recording is not None:
            expected_names.update(EPOCH_RECORDING_HELPERS)
        if admission is not None:
            expected_names.update(ROI_RECOVERY_HELPERS)
        if set(item['source_hashes']) != expected_names:
            raise ContractError('Snapshot source inventory changed')
        source = experiment / 'source' / stage
        from .snapshot_inventory import checked_snapshot_inventory
        checked_snapshot_inventory(source, expected_names)
        if item["spec"] != get_stage(stage).to_dict():
            raise ContractError("Stage registry changed")
        config = read(experiment / "configs" / f"{stage}.json")
        validate_stage_config(config, stage, root=repo, preparation_admission=admission)
        if canonical_hash(config) != item["config_sha256"]:
            raise ContractError("Stage config changed")
        for name, sha in item["source_hashes"].items():
            path = experiment / "source" / stage / name
            if digest(path) != sha:
                raise ContractError(f"Snapshot changed: {stage}/{name}")
            allowed = {'hiercp/model.py', 'hiercp/contracts.py'}
            if sampling is not None and sampling['mode'] == 'strict_nested':
                allowed.add('hiercp/sample.py')
            if name in proof['file_hashes'] and name not in allowed and sha != proof['file_hashes'][name]:
                raise ContractError(f'Original source changed: {stage}/{name}')
        with ZipFile(Path(repo)/'versions/v1/pipeline_v1_source.zip') as archive:
            for name in ('hiercp/model.py','hiercp/contracts.py', 'hiercp/sample.py'):
                exact = archive.read(name) + overlay(stage, name)
                if sampling is not None and sampling['mode'] == 'strict_nested':
                    exact += sampling_overlay(name)
                if (experiment/'source'/stage/name).read_bytes() != exact:
                    raise ContractError(f'Unspecified source overlay: {stage}/{name}')
        if sampling is not None:
            from .sampling_runtime import validate_contract
            contract = manifest['sampling_contracts'][stage]
            validate_contract(contract)
            if (read(experiment / 'sampling' / f'{stage}.json') != contract
                    or contract.get('mode') != sampling['mode']
                    or contract.get('profile') != sampling['profile']
                    or contract.get('graph_config') != config['graph']
                    or contract.get('source_identity') != dict(files=item['source_hashes'],
                        native_sample_sha256=proof['file_hashes']['hiercp/sample.py'])):
                raise ContractError('Sampling contract changed source/profile/config binding')
    return manifest


@contextmanager
def owned_lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(dict(pid=os.getpid(), command=sys.argv), stream)
    try:
        yield
    finally:
        path.unlink()  # Only the lock created by this invocation.


def command_plan(experiment, stage, target, *, python=sys.executable):
    experiment = Path(experiment).resolve()
    manifest = load_suite(experiment)
    get_stage(stage)
    shared = preparation_root(experiment, manifest)
    config = experiment / "configs" / f"{stage}.json"
    source = experiment / "source" / stage
    output = experiment / "results" / stage
    if target == "prepare":
        native = execution_reference(experiment, manifest)
        source = native / "source/v1.0"
        config = native / "configs/v1.0.json"
        common = ["--config", str(config), "--data-dir", str(Path(manifest["medical_root"])/"Data"),
                  "--split-file", str(shared/"split.json"), "--region-cache-dir", str(shared/"regions")]
        args = [["prepare-prototypes", *common, "--output", str(shared/"prototype_bank.pt")],
                ["prepare", *common, "--prototype-bank", str(shared/"prototype_bank.pt"),
                 "--cache-dir", str(shared/"cache")]]
        if manifest.get('preparation_recovery') is not None:
            return source, [[str(python), '-u', '-m', 'hiercp_v1x.budget_recovery_entry',
                             '--experiment', str(native), '--', *args[1]]]
    elif target == "train":
        resolved = output / "resolved_config.json"
        args = [["train", "--config", str(resolved if execution_is_resolved(manifest, stage) else config),
                 "--cache-dir", str(shared/"cache"), "--prototype-bank", str(shared/"prototype_bank.pt"),
                 "--checkpoint", str(output/"checkpoint_best.pt"), "--device", "cuda"]]
    elif target == "generate":
        args = [["generate", "--config", str(config), "--data-dir", str(shared/"generation_data"),
                 "--region-cache-dir", str(shared/"regions"), "--prototype-bank", str(shared/"prototype_bank.pt"),
                 "--checkpoint", str(output/"checkpoint_best.pt"), "--out-dir", str(output/"generated"),
                 "--device", "cuda"]]
    else:
        raise ContractError(f"Unsupported target {target}")
    if target in ('train', 'generate') and manifest.get('sampling_contract') is not None:
        if target == 'train' and manifest.get('epoch_recording') is not None:
            return source, [[str(python), '-u', '-m', 'hiercp_v1x.telemetry_entry',
                             '--contract', str(experiment / 'sampling' / f'{stage}.json'),
                             '--request', str(output / 'telemetry_request.json'), '--', *a] for a in args]
        return source, [[str(python), '-u', '-m', 'hiercp_v1x.sampling_entry',
                         '--contract', str(experiment / 'sampling' / f'{stage}.json'), '--', *a] for a in args]
    return source, [[str(python), "-u", "-m", "hiercp.pipeline", *a] for a in args]


def freeze_execution(experiment):
    experiment = Path(experiment).resolve()
    manifest = load_suite(experiment) if (experiment / 'manifest.json').is_file() else None
    native = execution_reference(experiment, manifest) if manifest is not None else experiment
    shared = native / 'shared'
    calibration_path = native/"results/v1.0/checkpoint_best.pt.preflight.json"
    calibration = read(calibration_path)
    if calibration.get("format") != "hiercp_preflight_calibration_v2":
        raise ContractError("Baseline has no measured calibration")
    identity = calibration.get('identity', {})
    if (identity.get('seed') != 42
            or Path(identity.get('cache_dir', '')).resolve() != (shared/'cache').resolve()
            or Path(identity.get('checkpoint_path', '')).resolve() != (native/'results/v1.0/checkpoint_best.pt').resolve()):
        raise ContractError('Calibration belongs to another dataset or experiment')
    resolve_execution_config(read(experiment/"configs/v1.0.json"), calibration)
    lock = dict(selected_batch_size=calibration["selected_batch_size"],
                selected_num_workers=calibration["selected_num_workers"],
                baseline_calibration_sha256=digest(calibration_path),
                resource_fingerprint=calibration["resource_fingerprint"])
    if manifest is not None and manifest.get('sampling_contract') is not None:
        baseline = load_suite(native)
        lock['baseline_sampling_contract_sha256'] = baseline['sampling_contracts']['v1.0']['contract_sha256']
        lock['baseline_manifest_sha256'] = baseline['manifest_sha256']
    path = shared/"execution_lock.json"
    if path.exists():
        if read(path) != lock:
            raise ContractError("Baseline execution calibration changed")
    else:
        write_new(path, lock)
    return lock


def bind_launch(experiment, stage, manifest, config, execution_lock):
    """Immutable sidecar before loading any optimizer/checkpoint state."""
    experiment = Path(experiment)
    shared, output = preparation_root(experiment, manifest), experiment/'results'/stage
    cache = {name:digest(shared/'cache'/name) for name in ('config.json','index.json','complete.json')}
    cache['prototype_bank_sha256'] = digest(shared/'prototype_bank.pt')
    source = dict(stage=stage, files=manifest['stages'][stage]['source_hashes'])
    evaluation = dict(split_sha256=manifest['split_sha256'], target_contract=TARGET_CONTRACT,
                      candidate_contract='v1_original_anchor_curriculum8_from_pool128',
                      mask_contract='original_full_source_footprint_and_original_eligibility')
    launch = dict(stage=stage, source=source, cache=cache, evaluation=evaluation,
                  config=config, execution_lock=execution_lock)
    if manifest.get('preparation_admission') is not None:
        launch['preparation_admission'] = manifest['preparation_admission']
    sampling = sampler_contract(experiment, stage, manifest)
    if sampling is not None:
        launch['sampling_contract'] = sampling
    launch['contract_sha256'] = canonical_hash(launch)
    path = output/'launch_contract.json'
    if path.exists():
        if read(path) != launch:
            raise ContractError('Existing stage source/cache/config contract differs; no exact resume')
    else:
        if any(output.glob('*.pt')):
            raise ContractError('Unbound pre-existing weights are not adopted into this experiment')
        write_new(path, launch)
    if execution_lock is not None:
        binding = make_run_contract(stage, config, cache_identity=cache,
                    source_identity=source, evaluation_identity=evaluation,
                    physical_batch_size=execution_lock['selected_batch_size'], debug=False,
                    execution_lock=execution_lock, sampling_contract=sampling,
                    preparation_admission=manifest.get('preparation_admission'))
        full = output/'run_contract.json'
        if full.exists(): validate_resume(binding, read(full))
        else: write_new(full, binding)
    return launch


def prepare_generation_inputs(experiment, manifest):
    directory = preparation_root(experiment, manifest)/"generation_data"
    data = Path(manifest["medical_root"])/"Data"
    for case in manifest["split"]["train"] + manifest["split"]["val"]:
        for kind in ("image", "labels"):
            matches = list((data/kind).glob(case+('_0000' if kind == 'image' else '')+'.nii*'))
            if len(matches) != 1:
                raise ContractError(f"Cannot identify native input: {case}/{kind}")
            original = matches[0]
            dest = directory/kind/original.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                if not os.path.samefile(original, dest):
                    raise ContractError(f"Existing generation input differs: {dest}")
            else:
                # No CT copies/cache duplication; no source files are modified.
                if os.name == 'nt':
                    os.link(original, dest)
                else:
                    dest.symlink_to(original)


def execute_stage(experiment, stage, target, *, gpu=None):
    experiment = Path(experiment).resolve()
    manifest = load_suite(experiment)
    shared = preparation_root(experiment, manifest)
    output = experiment/"results"/stage
    generation_proof = None
    if target == 'generate':
        from .results import collect_result
        generation_proof = collect_result(experiment, stage)
    with owned_lock(shared / 'prepare.lock' if target == 'prepare' else experiment/f'results/{stage}/run.lock'):
        if target in ('train','generate') and (shared/'prepare.lock').exists():
            raise ContractError('Shared preparation is still active; do not read a partial publication')
        lock = None
        if target == 'train' and execution_is_resolved(manifest, stage):
            lock = freeze_execution(experiment)
            resolved = resolve_execution_config(read(experiment/"configs"/f"{stage}.json"), lock)
            path = output/"resolved_config.json"
            if path.exists():
                if read(path) != resolved:
                    raise ContractError("Exact resume execution settings changed")
            else:
                write_new(path, resolved)
        if target == 'train':
            config = read(output/'resolved_config.json' if execution_is_resolved(manifest, stage) else experiment/'configs/v1.0.json')
            bind_launch(experiment, stage, manifest, config, lock)
            if manifest.get('epoch_recording') is not None:
                native = execution_reference(experiment, manifest)
                baseline = load_suite(native)
                request = dict(format='hiercp_v1_epoch_telemetry_request_v1',
                               output_dir=str(output), stage=stage,
                               mode=manifest['sampling_contract']['mode'],
                               native_reference=str(native),
                               baseline_manifest_sha256=baseline['manifest_sha256'],
                               baseline_sampling_contract_sha256=baseline['sampling_contracts']['v1.0']['contract_sha256'])
                destination = output / 'telemetry_request.json'
                if destination.exists():
                    if read(destination) != request:
                        raise ContractError('Existing epoch telemetry request changed; no silent resume')
                else:
                    write_new(destination, request)
        if target == 'generate':
            for checkpoint in generation_proof['checkpoints'].values():
                if digest(checkpoint['path']) != checkpoint['sha256']:
                    raise ContractError('Verified generation checkpoint changed before launch')
            prepare_generation_inputs(experiment, manifest)
        if gpu is not None and target in ('train', 'generate'):
            from tools.local_cnn_device import select
            select(gpu)
        cwd, commands = command_plan(experiment, stage, target)
        run_id = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8]
        write_new(experiment/"invocations"/(run_id+'.json'), dict(stage=stage, target=target,
                  cwd=str(cwd), commands=commands, actual_dataset_subset=False,
                  physical_batch="baseline measured and shared", target_contract=TARGET_CONTRACT,
                  source_snapshot=manifest["stages"][stage], nnunet_started=False))
        print(f"Foreground {stage} | {target} | {output}", flush=True)
        print("Ctrl+C interrupts this foreground job. Original v1 resumes at its last saved complete epoch; partial epochs are not claimed saved.", flush=True)
        from .snapshot_inventory import isolated_snapshot_bytecode_env
        env = isolated_snapshot_bytecode_env(experiment/'invocations'/(run_id+'_bytecode_lookup'))
        env.pop('PYTHONPATH', None)  # Snapshot imports must not resolve current v2 packages.
        env.pop('HIERCP_V1X_SAMPLING_CONTRACT', None)  # Inherited settings cannot contaminate native preparation.
        for command in commands:
            child = subprocess.Popen(command, cwd=cwd, env=env)
            try:
                code = child.wait()
            except KeyboardInterrupt:
                print(f"Interrupt received | directly launched PID {child.pid} | waiting for foreground child to stop; no session/process-group termination.", flush=True)
                child.wait()
                raise
            if code:
                raise RuntimeError(f"{stage}/{target} failed ({code}); outputs preserved, later stages not started")
        if target == 'train' and stage == 'v1.0' and _reference(manifest) is None:
            freeze_execution(experiment)
    result = dict(stage=stage, target=target, completed=True, nnunet_started=False,
                  quality_claim="execution completion is not evidence of recommendation quality")
    if target == 'train':
        from .results import collect_result
        report = collect_result(experiment, stage)
        result['comparison_report'] = str(output/'comparison_report.json')
        result['metrics'] = report['metrics']
    return result


def compare_files(baseline, candidate, predecessor=None):
    return compare_reports(read(baseline), read(candidate), read(predecessor) if predecessor else None)
