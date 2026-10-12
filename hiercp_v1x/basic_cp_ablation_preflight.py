"""Read-only reuse of a complete preflight after the isolated constructor API fix.

No image/mask geometry is recomputed, file is written, or process is signaled.
All source NPZ bytes are rehashed. Only the wrapper constructor's argument
signature and matching super-call may differ; its scientific body must match.
"""
from __future__ import annotations
import ast
import copy
import hashlib
import json
from pathlib import Path

FORMAT = 'basic_cp_one_factor_full250_v1'
REUSE_FORMAT = 'basic_cp_complete_preflight_ctor_API_reuse_v1'
HISTORICAL_SHA = 'a3df56435fedff46fbc42a502596fdade2632d0211ea412630dba1555f32567d'
BANK_SHA = '9e5be3e3ab8f04fd23c166b3267bc4efbfb4b321ef974920dfc08fbf5dd275e6'
TRAINER_RELATIVE = 'custom_trainers/nnUNetTrainer_BasicCPAblations.py'
NATIVE_TRAINER_RELATIVE = 'training/nnUNetTrainer/nnUNetTrainer_BasicCPAblations.py'


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 2**20), b''):
            digest.update(block)
    return digest.hexdigest()


def constructor_normalized_AST(source, *, current):
    tree = ast.parse(source)
    owners = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == '_BasicCPAblation']
    require(len(owners) == 1, 'Exactly one scientific wrapper class required')
    constructors = [node for node in owners[0].body if isinstance(node, ast.FunctionDef) and node.name == '__init__']
    require(len(constructors) == 1, 'Exactly one scientific wrapper constructor required')
    function = constructors[0]
    args = function.args
    explicit = [arg.arg for arg in args.args] == ['self', 'plans', 'configuration', 'fold', 'dataset_json', 'device']
    varargs = ([arg.arg for arg in args.args] == ['self'] and args.vararg is not None
               and args.vararg.arg == 'args' and args.kwarg is not None and args.kwarg.arg == 'kwargs')
    require(not args.posonlyargs and not args.kwonlyargs, 'Unexpected constructor argument shape')
    if explicit:
        require(args.vararg is None and args.kwarg is None and len(args.defaults) == 1
                and ast.unparse(args.defaults[0]) == "historical.torch.device('cuda')",
                'Exact original five arguments and CUDA device default required')
    else:
        require(not current and varargs and not args.defaults, 'Only the known historical varargs defect is reusable')
    calls = [node for node in ast.walk(function) if isinstance(node, ast.Call)
             and ast.unparse(node.func) == 'super().__init__']
    require(len(calls) == 1, 'Exactly one original-parent constructor call required')
    call = calls[0]
    if explicit:
        require([ast.unparse(arg) for arg in call.args] == ['plans', 'configuration', 'fold', 'dataset_json', 'device']
                and not call.keywords, 'Original five values must be forwarded unchanged')
    else:
        require([ast.unparse(arg) for arg in call.args] == ['*args']
                and len(call.keywords) == 1 and call.keywords[0].arg is None
                and ast.unparse(call.keywords[0].value) == 'kwargs', 'Known old varargs forwarding differs')
    # Erase only the API repair, retaining every statement of the constructor
    # (including all SHA/population/horizon guards) and the rest of the module.
    function.args = ast.arguments(posonlyargs=[], args=[ast.arg(arg='self')], vararg=None,
                                 kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[])
    call.args = []; call.keywords = []
    return ast.dump(tree, include_attributes=False)


def _entry_path(bank_root, name):
    path = (bank_root / name).resolve()
    require(bank_root.resolve() in path.parents, 'Source entry path escapes bank root')
    return path


def verify_preflight_reuse(oldroot, oldrequestSHA, currenttrainerpath, bankpath, bankSHA, arm,
                           expectedtrainids105, *, readonly_admission=None,
                           readonly_admission_sha256=None):
    root = Path(oldroot).resolve()
    requestpath, resultpath, nativepath = root / 'request.json', root / 'preflight.json', root / 'native.json'
    require(sha(requestpath) == oldrequestSHA, 'Exact old immutable request SHA differs')
    request, native, result = read(requestpath), read(nativepath), read(resultpath)
    require(arm in ('multi3', 'blend', 'candidates256') and request['arm'] == native['arm'] == arm,
            'Own experiment arm changed')
    expected_gpu = {'multi3': 1, 'candidates256': 4, 'blend': 6}[arm]
    count = 256 if arm == 'candidates256' else 128
    for key, value in {'gpu': expected_gpu, 'epochs': 250, 'train_cases': 105, 'validation_cases': 26,
                       'physical_batch': 2, 'checkpoint': 'checkpoint_best.pth', 'sources': 642,
                       'candidates_per_source': count, 'copies_per_CP_event': 3 if arm == 'multi3' else 1,
                       'blend_border_preprocessed_voxels': 2 if arm == 'blend' else 0}.items():
        require(request.get(key) == value, 'Original immutable request contract differs: ' + key)
    expected_ids = tuple(expectedtrainids105)
    require(len(expected_ids) == 105 and len(set(expected_ids)) == 105, 'All105 unique current training IDs required')
    require(set(native['split']['outer_train']) == set(expected_ids) and len(native['split']['outer_train']) == 105
            and len(native['split']['outer_val']) == len(set(native['split']['outer_val'])) == 26,
            'Full original105/26 split binding differs')
    require(set(native['split']['outer_train']).isdisjoint(native['split']['outer_val']), 'Original train/test overlap')
    admission = Path(request['readonly_admission'])
    require(sha(admission) == request['readonly_admission_sha256'], 'Original read-only data admission changed')
    if readonly_admission is not None or readonly_admission_sha256 is not None:
        require(readonly_admission is not None and readonly_admission_sha256 is not None,
                'Current admission path and SHA must be supplied together')
        require(sha(readonly_admission) == readonly_admission_sha256 == request['readonly_admission_sha256'],
                'Current and original read-only dataset admissions differ')
    package = Path(native['private_runtime']) / 'nnunetv2'
    oldtrainer = package / NATIVE_TRAINER_RELATIVE
    olddigest = sha(oldtrainer)
    require(olddigest == request['code_sha256'][TRAINER_RELATIVE], 'Actual old installed wrapper SHA differs from old request')
    historical = package / 'training/nnUNetTrainer/nnUNetTrainer_OnlinePairedCP.py'
    require(sha(historical) == HISTORICAL_SHA, 'Actual original historical trainer changed')
    require(constructor_normalized_AST(oldtrainer.read_text(encoding='utf8'), current=False)
            == constructor_normalized_AST(Path(currenttrainerpath).read_text(encoding='utf8'), current=True),
            'Scientific module changed outside the exact constructor argument/API repair')
    bankpath = Path(bankpath).resolve()
    require(sha(bankpath) == bankSHA == native['bank_sha256']
            and Path(native['bank']).resolve() == bankpath, 'Original complete bank path/SHA changed')
    require(arm == 'candidates256' or bankSHA == BANK_SHA, 'Actual historical Basic index SHA differs')
    bank = read(bankpath)
    source_cases = bank['entries_by_case']
    names = [name for values in source_cases.values() for name in values]
    require(len(source_cases) == 81 and len(names) == len(set(names)) == 642
            and bank['candidate_count'] == count and bank['total_candidates'] == 642 * count,
            'Whole source/candidate population differs')
    require(set(source_cases).issubset(expected_ids), 'Bank contains a foreign source patient')
    for key, value in {'format': FORMAT, 'arm': arm, 'complete': True, 'index_sha256': bankSHA,
                       'historical_trainer_sha256': HISTORICAL_SHA, 'historical_bank_sha256': BANK_SHA,
                       'source_entries': 642, 'eligible_cases': 81, 'full_training_cases': 105,
                       'full_validation_cases': 26, 'candidate_count': count,
                       'validated_positions': 642 * count,
                       'three_paste_positions': 82176 if arm == 'multi3' else 0,
                       'CP_probability': .5, 'copies_per_event': 3 if arm == 'multi3' else 1,
                       'blend_border_preprocessed_voxels': 2 if arm == 'blend' else 0,
                       'original_gate_primary_RNG_schedule_preserved': True, 'GNN_scores_used': False,
                       'physical_batch_size': 2, 'seed': 42, 'epochs': 250,
                       'network_patch_size': [128, 128, 128], 'training_executed': False}.items():
        require(result.get(key) == value, 'Complete preflight witness differs: ' + key)
    actual_ids = result.get('actual_training_identifiers', [])
    require(len(actual_ids) == len(set(actual_ids)) == 105 and set(actual_ids) == set(expected_ids),
            'Actual old/current105 identifiers differ')
    checksums = result['entries_sha256']
    require(set(checksums) == set(names), 'Every actual642 source SHA must be present exactly once')
    receipts = result['case_receipts']
    require(len(receipts) == 81 and len({row['case_id'] for row in receipts}) == 81,
            'All81 unique complete patient receipts required')
    for row in receipts:
        case = row['case_id']; expected_names = set(source_cases.get(case, ()))
        require(expected_names and row['source_entries'] == len(expected_names)
                and row['validated_positions'] == len(expected_names) * count
                and row['three_paste_positions'] == (len(expected_names) * count if arm == 'multi3' else 0)
                and set(row['entries_sha256']) == expected_names
                and all(checksums[name] == digest for name, digest in row['entries_sha256'].items()),
                'Patient receipt is partial or disagrees with full source mapping: ' + case)
    for name, digest in checksums.items():
        require(sha(_entry_path(bankpath.parent, name)) == digest, 'Actual source bytes changed after preflight: ' + name)
    proof = {'format': REUSE_FORMAT, 'original_root': str(root), 'original_request_sha256': oldrequestSHA,
             'original_preflight_sha256': sha(resultpath), 'original_native_sha256': sha(nativepath),
             'original_installed_wrapper_sha256': olddigest, 'current_wrapper_sha256': sha(currenttrainerpath),
             'only_change': 'exact constructor parameter signature and parent forwarding API repair',
             'original_historical_trainer_sha256': HISTORICAL_SHA,
             'readonly_admission': {'path': str(admission), 'sha256': request['readonly_admission_sha256']},
             'full_source_SHA_reverified': 642, 'all_complete_case_receipts_verified': 81,
             'actual_training_identifiers_verified': 105, 'geometry_recomputed': False,
             'scientific_model_loss_RNG_paste_geometry_unchanged': True,
             'source_files_written': False, 'processes_signaled': False}
    reused = copy.deepcopy(result)
    require('reuse_proof' not in reused, 'Chained preflight reuse requires explicit original evidence; original full receipt expected')
    reused['reuse_proof'] = proof
    return reused
