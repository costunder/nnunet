"""Archive a completed own BEST and all26 predictions; preserve original inputs."""
from __future__ import annotations
from pathlib import Path
import time

from . import v24_readonly_native_storage as core


def archive_completed(request):
    from . import v24_native_best_eval_runtime as best
    native_root = Path(request['execution_root']) / 'native'
    native_path = core.regular(native_root / 'native.json')
    native = core.read(native_path)
    fold = native_root / 'nnUNet_results' / native['baseline']['dataset_name'] / (
        native['trainer'] + '__' + native['baseline']['plans_name'] + '__'
        + native['baseline']['configuration']) / 'fold_0'
    receipt_path = fold / best.RECEIPT
    receipt = best.validate_best_validation_receipt(receipt_path, fold=fold,
        expected_cases=native['split']['outer_val'], bank_sha256=native['bank_sha256'])
    completed = list(native_root.glob('training_complete_*.json'))
    if len(completed) != 1:
        raise ValueError('Exactly one actual full250 completion required before archive')
    complete = core.read(completed[0])
    final_witness = core.proof(fold / 'checkpoint_final.pth')
    if (complete.get('epochs') != 250 or complete.get('evaluation_checkpoint_name') != best.BEST
            or complete.get('evaluation_checkpoint_sha256') != receipt['checkpoint_sha256']
            or complete.get('checkpoint') != final_witness['path']
            or complete.get('checkpoint_sha256') != final_witness['sha256']
            or native.get('physical_GPU') != 4 or native.get('epochs') != 250):
        raise ValueError('Own GPU4 full250/BEST26 completion is required')
    target = Path(request['metadata_root']) / 'completed_native_archive'
    if target.exists():
        raise FileExistsError('Completed native archive is immutable')
    sources = {best.BEST: core.proof(fold / best.BEST), best.RECEIPT: core.proof(receipt_path),
        'native.json': core.proof(native_path), completed[0].name: core.proof(completed[0])}
    for path in sorted((fold / 'validation').iterdir()):
        if path.is_file():
            sources['validation/' + path.name] = core.proof(path)
        else:
            raise ValueError('Unexpected subdirectory in completed validation; archive explicitly')
    for name in ('actual_nnunet_configuration.json', 'calibration.json'):
        sources[name] = core.proof(native_root / name)
    for path in sorted(native_root.glob('train_*.log')):
        sources[path.name] = core.proof(path)
    required = sum(row['bytes'] for row in sources.values())
    from . import v24_gpu0_native_storage as storage
    storage.check_persistent_archive(request, required)
    target.mkdir()
    for name, witness in sources.items():
        core.copy_private(witness, target / name)
    proof = dict(format='v24_gpu4_GPU0_actual_full250_BEST26_persistent_archive_v1',
        complete=True, source_GPU_arm=4, execution_physical_GPU=0,
        evaluation_checkpoint=best.BEST, full_training_epochs=250,
        validation_cases=receipt['validation_cases'], archive_bytes=required,
        archive_root=str(target), original_native_root=str(native_root),
        source_files=sources, archived_files={name: core.proof(target / name) for name in sources},
        source_final_completion_witness=final_witness,
        source_scores_recomputed=False, source_weights_written=False,
        original_native_outputs_preserved=True, archived_at=time.time())
    core.publish_json(Path(request['metadata_root']) / 'completed_native_archive.json', proof)
    return proof
